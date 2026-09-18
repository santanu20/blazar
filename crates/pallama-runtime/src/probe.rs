//! Hardware assembly: sysinfo (CPU/RAM) + engine manifest devices.

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Retry budget for spawn-class transients. 4 attempts / 250 ms flat
/// backoff is the measured-cure bar from the original vulkan-flake fix
/// (3ce05e1); flat rather than jittered because a single sync caller
/// retries alone — no in-process herd to de-synchronize, and parallel
/// test binaries are bounded by the same 4-attempt cap.
const SPAWN_RETRY_ATTEMPTS: u32 = 4;
const SPAWN_RETRY_BACKOFF: Duration = Duration::from_millis(250);

/// Is this spawn error environment-transient (worth a bounded retry)?
/// The classes, all reproduced live under parallel load:
///
/// - EAGAIN/EWOULDBLOCK (fork/thread pressure — `ErrorKind::WouldBlock`),
///   EINTR (`Interrupted`), ENOMEM;
/// - EMFILE/ENFILE (fd pressure — `ulimit -n` repro: the probe's pipe
///   pair loses the race for the last descriptors);
/// - ETXTBSY (write->close->exec writeback race on a JUST-INSTALLED
///   binary — os error 26, captured live: `fs::copy`/tar-extract closes
///   the fd, execve still briefly sees the inode write-open while pages
///   flush; hits the real install lane, not only tests).
///
/// Everything else is permanent — ENOENT (missing binary), EACCES,
/// ENOEXEC (garbage/foreign-arch asset — the register-or-clean fast-fail
/// pin depends on that one NOT being retried) — and must fail on
/// attempt one so the loud error is not delayed.
fn is_transient_spawn_err(e: &std::io::Error) -> bool {
    use std::io::ErrorKind;
    if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted) {
        return true;
    }
    #[cfg(unix)]
    {
        matches!(
            e.raw_os_error(),
            Some(libc::ENOMEM | libc::EMFILE | libc::ENFILE | libc::ETXTBSY)
        )
    }
    #[cfg(not(unix))]
    {
        false
    }
}

/// Run a spawn-class operation under the bounded transient-retry policy
/// (H16: explicit attempts, backoff, abort on permanent class). Covers
/// `Command::spawn` and `Command::output` alike; the final error is
/// returned as-is — retry never masks, it only rides out pressure.
pub(crate) fn with_spawn_retry<T>(
    mut op: impl FnMut() -> std::io::Result<T>,
) -> std::io::Result<T> {
    let mut attempt = 1;
    loop {
        match op() {
            Ok(v) => return Ok(v),
            Err(e) => {
                if !is_transient_spawn_err(&e) || attempt >= SPAWN_RETRY_ATTEMPTS {
                    return Err(e);
                }
                tracing::warn!(
                    "transient spawn failure ({e}), retry {attempt}/{} in {:?}",
                    SPAWN_RETRY_ATTEMPTS - 1,
                    SPAWN_RETRY_BACKOFF
                );
                std::thread::sleep(SPAWN_RETRY_BACKOFF);
                attempt += 1;
            }
        }
    }
}

/// Run a short-lived probe command (`--version`/`--help`/census class)
/// under a hard deadline. A hung probe binary must fail fast instead of
/// wedging the caller forever (F85); on timeout the child is killed and
/// `None` is returned — callers decide whether that is an error (fatal
/// probes) or an empty census (best-effort probes). Long-running tool
/// invocations (quantize, builds) must NOT route through this.
///
/// Both pipes are drained on reader threads WHILE the child runs: a child
/// that writes more than the OS pipe buffer (~64 KiB) would otherwise
/// block on write, never exit, and turn into a guaranteed deadline kill —
/// a silent 30 s stall masquerading as a hung binary. Spawn failures are
/// logged with their `io::Error` (a transient EAGAIN under fork pressure
/// is retried by [`with_spawn_retry`]; what survives is logged with the
/// error so it stays distinguishable from a missing binary).
pub fn probe_output(cmd: &mut Command, secs: u64) -> Option<std::process::Output> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = match with_spawn_retry(|| cmd.spawn()) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("probe spawn failed ({e}): {:?}", cmd.get_program());
            return None;
        }
    };
    // Drain both pipes concurrently with the poll loop (see doc comment).
    let stdout_handle = child
        .stdout
        .take()
        .map(|p| std::thread::spawn(move || drain_pipe(p)));
    let stderr_handle = child
        .stderr
        .take()
        .map(|p| std::thread::spawn(move || drain_pipe(p)));
    let deadline = Instant::now() + Duration::from_secs(secs);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if Instant::now() >= deadline {
                    tracing::warn!(
                        "probe {:?} exceeded {secs}s deadline — killing",
                        cmd.get_program()
                    );
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(_) => return None,
        }
    };
    let stdout = stdout_handle.map_or_else(Vec::new, |h| h.join().unwrap_or_default());
    let stderr = stderr_handle.map_or_else(Vec::new, |h| h.join().unwrap_or_default());
    Some(std::process::Output {
        status,
        stdout,
        stderr,
    })
}

/// Reader-thread body: drain a pipe to EOF into a buffer. A panicked
/// reader (torn pipe) yields an empty buffer — the caller's status check
/// already fails the probe.
fn drain_pipe(mut pipe: impl Read) -> Vec<u8> {
    let mut buf = Vec::new();
    let _ = pipe.read_to_end(&mut buf);
    buf
}

/// Live `MemAvailable` (MiB). The spawn-time memory guard: below a hard
/// floor the mmap streaming engine will thrash swap for minutes — fail
/// the load with a named error instead (the validate.py heuristic,
/// tightened to a zero-false-positive floor).
#[must_use]
pub fn mem_available_mib() -> u64 {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    sys.available_memory() / (1024 * 1024)
}

use pallama_core::{GpuInfo, Hardware};

use crate::engine::manifest::Manifest;

#[must_use]
pub fn probe_hardware(manifest: Option<&Manifest>) -> Hardware {
    let mut gpus = manifest
        .map(|m| {
            m.devices
                .iter()
                .map(|d| GpuInfo {
                    name: d.name.clone(),
                    description: d.description.clone(),
                    total_mib: d.total_mib,
                    free_mib: d.free_mib,
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if gpus.is_empty() {
        // Engines without a --list-devices census (mistral.rs) leave the
        // manifest's device list empty too — a GPU the daemon cannot see
        // silently starves every capacity decision on that lane (measured:
        // "0 GPUs" banner on a 4070 box, paged-attn auto-fallback blind,
        // 502 loads). Fall back to a system-side NVIDIA census; non-NVIDIA
        // boxes without a census keep the empty list, same as before.
        gpus = nvidia_smi_gpus();
    }
    hardware_with(gpus)
}

/// Best-effort `nvidia-smi` device census. None/skip on any failure —
/// never a guessed entry.
fn nvidia_smi_gpus() -> Vec<GpuInfo> {
    // F85: best-effort census — a hung nvidia-smi yields an empty list,
    // never a wedged caller.
    let Some(out) = probe_output(
        std::process::Command::new("nvidia-smi").args([
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]),
        10,
    ) else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    parse_nvidia_csv(&String::from_utf8_lossy(&out.stdout))
}

/// `name, total MiB, free MiB` per line (nvidia-smi csv,noheader,nounits).
fn parse_nvidia_csv(text: &str) -> Vec<GpuInfo> {
    text.lines()
        .filter_map(|ln| {
            let mut parts = ln.split(',').map(str::trim);
            let name = parts.next()?.to_string();
            if name.is_empty() {
                return None;
            }
            let total_mib: u64 = parts.next()?.parse().ok()?;
            let free_mib: u64 = parts.next()?.parse().ok()?;
            Some(GpuInfo {
                description: format!("NVIDIA {name}"),
                name,
                total_mib,
                free_mib,
            })
        })
        .collect()
}

/// One external process currently holding GPU memory.
#[derive(Debug)]
pub struct GpuTenant {
    pub pid: u32,
    pub process_name: String,
    pub used_mib: u64,
}

/// Best-effort census of processes holding GPU compute memory
/// (`nvidia-smi --query-compute-apps`). None when the tool is absent or
/// fails — the caller reports nothing rather than a guessed tenant list.
/// Pure parser: `pid, process_name, used_mib` per csv line.
#[must_use]
pub fn parse_gpu_tenants(csv: &str) -> Vec<GpuTenant> {
    csv.lines()
        .filter_map(|ln| {
            let mut parts = ln.split(',').map(str::trim);
            let pid: u32 = parts.next()?.parse().ok()?;
            let process_name = parts.next()?.to_string();
            if process_name.is_empty() {
                return None;
            }
            // `used_mib` may carry a unit suffix on some driver versions.
            let used_mib: u64 = parts
                .next()?
                .trim_end_matches(|c: char| !c.is_ascii_digit())
                .parse()
                .ok()?;
            Some(GpuTenant {
                pid,
                process_name,
                used_mib,
            })
        })
        .collect()
}

/// Live GPU compute tenants at boot: whoever is holding VRAM before
/// pallama has spawned any child of its own. Advisory only — pallama
/// never kills another product's process.
#[must_use]
pub fn gpu_compute_tenants() -> Option<Vec<GpuTenant>> {
    let out = probe_output(
        std::process::Command::new("nvidia-smi").args([
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]),
        10,
    )?;
    if !out.status.success() {
        return None;
    }
    Some(parse_gpu_tenants(&String::from_utf8_lossy(&out.stdout)))
}

/// sysinfo half + caller-supplied GPU list: the composition point for a
/// LIVE `--list-devices` census (see `engine::manifest::run_list_devices`).
#[must_use]
pub fn hardware_with(gpus: Vec<GpuInfo>) -> Hardware {
    let mut sys = sysinfo::System::new();
    sys.refresh_cpu_usage();
    sys.refresh_memory();
    let physical_cores = sys
        .physical_core_count()
        .map_or(1, |c| u32::try_from(c).unwrap_or(1))
        .max(1);
    let total_ram_mib = sys.total_memory() / (1024 * 1024); // sysinfo returns bytes
    Hardware {
        physical_cores,
        total_ram_mib,
        gpus,
    }
}

/// PCI vendor ids of display-class hardware (VGA 0300 + 3D 0302), parsed
/// from `lspci -n -d ::0300`-style output. Works WITHOUT any driver
/// installed — this is the doctor's hardware-vs-driver oracle for the
/// "GPU present but its driver userspace is missing" warning. Sorted,
/// deduplicated, lowercase. Known ids: `10de` NVIDIA, `1002` AMD, `8086`
/// Intel; unknown ids pass through verbatim (never guessed into a brand).
#[must_use]
pub fn parse_pci_vendors(lspci_text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in lspci_text.lines() {
        // `0000:01:00.0 0300: 10de:28a0 (rev a1)` — take the token
        // following the class field (` 0300: `), then its vendor half.
        let mut fields = line.split_whitespace();
        let _addr = fields.next();
        let Some(class) = fields.next() else { continue };
        if !class.ends_with(':') {
            continue;
        }
        let Some(id) = fields.next() else { continue };
        let Some((vendor, _device)) = id.split_once(':') else {
            continue;
        };
        if vendor.len() == 4 && vendor.chars().all(|c| c.is_ascii_hexdigit()) {
            let vendor = vendor.to_ascii_lowercase();
            if !out.contains(&vendor) {
                out.push(vendor);
            }
        }
    }
    out
}

/// Live PCI display-hardware census via `lspci -n` (driver-independent).
/// Empty when lspci is absent or the box has no PCI bus (containers,
/// WSL1) — callers treat empty as "no oracle", never as "no GPU".
#[must_use]
pub fn pci_gpu_vendors() -> Vec<String> {
    let Some(out) = probe_output(
        std::process::Command::new("lspci")
            .arg("-n")
            .arg("-d")
            .arg("::0300"),
        10,
    ) else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let mut vendors = parse_pci_vendors(&text);
    // 3D controllers (0302) are a separate class: discrete NVIDIA on
    // hybrid laptops routinely lands here instead of 0300.
    if let Some(out2) = probe_output(
        std::process::Command::new("lspci")
            .arg("-n")
            .arg("-d")
            .arg("::0302"),
        10,
    ) {
        if out2.status.success() {
            let text2 = String::from_utf8_lossy(&out2.stdout).into_owned();
            for v in parse_pci_vendors(&text2) {
                if !vendors.contains(&v) {
                    vendors.push(v);
                }
            }
        }
    }
    vendors.sort();
    vendors.dedup();
    vendors
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn unit__probe_hardware__merges_sysinfo_and_devices() {
        let m = Manifest {
            tag: "t".into(),
            build_number: 1,
            version_raw: "version: 1".into(),
            devices: vec![crate::engine::manifest::DeviceDesc {
                name: "RTX".into(),
                description: "NVIDIA CUDA".into(),
                total_mib: 8_188,
                free_mib: 7_000,
            }],
            flags: std::collections::BTreeSet::default(),
            spec_types: vec![],
            server_path: "/x".into(),
        };
        let hw = probe_hardware(Some(&m));
        assert!(hw.physical_cores >= 1);
        assert!(hw.total_ram_mib > 0);
        assert_eq!(hw.gpus.len(), 1);
        assert_eq!(hw.total_vram_mib(), 8_188);
        // The None-manifest path may legitimately find GPUs via the
        // nvidia-smi fallback (environment-dependent) — portability means
        // asserting the sysinfo-only merge at the composition point.
        let cpu_only = hardware_with(Vec::new());
        assert!(cpu_only.gpus.is_empty());
    }

    #[test]
    fn unit__probe_hardware__nvidia_csv_parser() {
        let gpus = parse_nvidia_csv(
            "NVIDIA GeForce RTX 4070 Laptop GPU, 8188, 5100\n\
             NVIDIA GeForce RTX 3090, 24576, 24000\n",
        );
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].total_mib, 8_188);
        assert_eq!(gpus[0].free_mib, 5_100);
        assert!(gpus[0].description.contains("NVIDIA"));
        assert!(!gpus[0].is_integrated(), "nvidia entries read discrete");
        // Garbage lines skip; partial lines skip — never a guessed entry.
        assert!(parse_nvidia_csv("nope\n\nRTX, only-two\n").is_empty());
    }

    #[test]
    fn unit__parse_gpu_tenants__csv_shapes_and_unit_suffix() {
        let t = parse_gpu_tenants(
            "1588925, /usr/local/bin/pallama, 2314\n\
             1637391, ollama, 1858\n",
        );
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].pid, 1_588_925);
        assert_eq!(t[0].process_name, "/usr/local/bin/pallama");
        assert_eq!(t[0].used_mib, 2_314);
        // Some driver versions append " MiB" to the memory column.
        let s = parse_gpu_tenants("42, ollama, 972 MiB\n");
        assert_eq!(s[0].used_mib, 972, "unit suffix tolerated");
        // Non-numeric pid / empty name / short lines skip silently.
        assert!(parse_gpu_tenants("NotFound, , \nnope\n").is_empty());
    }

    #[test]
    fn unit__parse_pci_vendors__vga_and_3d_lines() {
        let out = parse_pci_vendors(
            "0000:00:02.0 0300: 8086:46a6 (rev 0c)\n\
             0000:01:00.0 0300: 10de:28a0 (rev a1)\n\
             0000:01:00.1 0403: 10de:28ba (rev a1)\n",
        );
        assert_eq!(out, vec!["8086".to_string(), "10de".to_string()]);
    }

    #[test]
    fn unit__parse_pci_vendors__dedup_unknown_and_garbage() {
        let out = parse_pci_vendors(
            "0000:01:00.0 0300: 10de:28a0 (rev a1)\n\
             0000:02:00.0 0300: 10de:2684 (rev a1)\n\
             0000:03:00.0 0300: 1a03:1150 (rev 10)\n\
             garbage line entirely\n\
             0000:04:00.0 0300 no-colon-class\n",
        );
        assert_eq!(out, vec!["10de".to_string(), "1a03".to_string()]);
    }

    #[test]
    fn unit__parse_pci_vendors__uppercase_hex_normalized() {
        let out = parse_pci_vendors("0000:01:00.0 0300: 10DE:28A0 (rev a1)\n");
        assert_eq!(out, vec!["10de".to_string()]);
    }

    #[cfg(unix)]
    #[test]
    fn unit__probe_output__survives_pipe_buffer_flood() {
        // 4 MiB of output — far past the ~64 KiB OS pipe buffer. The
        // pre-fix poll-then-read shape deadlocked exactly here: the child
        // blocked on write, never exited, and the probe surfaced as a
        // bogus "timed out or failed to spawn".
        let mut cmd = std::process::Command::new("/bin/sh");
        cmd.args(["-c", "head -c 4194304 /dev/zero | base64"]);
        let start = std::time::Instant::now();
        let out = probe_output(&mut cmd, 30).expect("flood child must complete");
        assert!(out.status.success());
        assert!(
            out.stdout.len() > 1_000_000,
            "flood output lost: {} bytes",
            out.stdout.len()
        );
        assert!(
            start.elapsed().as_secs() < 25,
            "flood probe took {:?}",
            start.elapsed()
        );
    }

    #[test]
    fn unit__probe_output__missing_binary_is_none_not_panic() {
        let mut cmd = std::process::Command::new("/nonexistent/pallama-probe-binary");
        assert!(probe_output(&mut cmd, 5).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn unit__is_transient_spawn_err__classifies_pressure_vs_permanent() {
        use std::io::Error;
        let transient = [
            Error::from_raw_os_error(libc::EAGAIN),
            Error::from_raw_os_error(libc::EWOULDBLOCK),
            Error::from_raw_os_error(libc::EINTR),
            Error::from_raw_os_error(libc::ENOMEM),
            Error::from_raw_os_error(libc::EMFILE),
            Error::from_raw_os_error(libc::ENFILE),
            Error::from_raw_os_error(libc::ETXTBSY),
            Error::new(std::io::ErrorKind::Interrupted, "interrupted"),
        ];
        for e in &transient {
            assert!(is_transient_spawn_err(e), "{e} must classify transient");
        }
        // Permanent classes fail attempt one — ENOEXEC is the register
        // fast-fail pin's mechanism (garbage asset), it must never retry.
        let permanent = [
            Error::from_raw_os_error(libc::ENOENT),
            Error::from_raw_os_error(libc::EACCES),
            Error::from_raw_os_error(libc::ENOEXEC),
            Error::new(std::io::ErrorKind::PermissionDenied, "denied"),
        ];
        for e in &permanent {
            assert!(!is_transient_spawn_err(e), "{e} must classify permanent");
        }
    }

    #[test]
    fn unit__with_spawn_retry__transient_retries_then_succeeds() {
        let attempts = std::cell::Cell::new(0u32);
        let r = with_spawn_retry(|| {
            attempts.set(attempts.get() + 1);
            match attempts.get() {
                n if n < 3 => Err(std::io::Error::from_raw_os_error(libc::EAGAIN)),
                _ => Ok(42),
            }
        });
        assert_eq!(r.unwrap(), 42);
        assert_eq!(
            attempts.get(),
            3,
            "two transients then success = 3 attempts"
        );
    }

    #[test]
    fn unit__with_spawn_retry__transient_exhausts_after_max_attempts() {
        let attempts = std::cell::Cell::new(0u32);
        let r = with_spawn_retry(|| {
            attempts.set(attempts.get() + 1);
            Err::<(), _>(std::io::Error::from_raw_os_error(libc::EAGAIN))
        });
        assert!(r.is_err(), "exhausted transients surface the final error");
        assert_eq!(attempts.get(), SPAWN_RETRY_ATTEMPTS);
    }

    #[test]
    fn unit__with_spawn_retry__permanent_fails_fast_attempt_one() {
        let attempts = std::cell::Cell::new(0u32);
        let r = with_spawn_retry(|| {
            attempts.set(attempts.get() + 1);
            Err::<(), _>(std::io::Error::from_raw_os_error(libc::ENOENT))
        });
        assert!(r.is_err());
        assert_eq!(attempts.get(), 1, "permanent errors are never retried");
    }
}
