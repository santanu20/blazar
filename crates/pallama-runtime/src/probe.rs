//! Hardware assembly: sysinfo (CPU/RAM) + engine manifest devices.

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Run a short-lived probe command (`--version`/`--help`/census class)
/// under a hard deadline. A hung probe binary must fail fast instead of
/// wedging the caller forever (F85); on timeout the child is killed and
/// `None` is returned — callers decide whether that is an error (fatal
/// probes) or an empty census (best-effort probes). Long-running tool
/// invocations (quantize, builds) must NOT route through this.
pub fn probe_output(cmd: &mut Command, secs: u64) -> Option<std::process::Output> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().ok()?;
    let deadline = Instant::now() + Duration::from_secs(secs);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(_) => return None,
        }
    };
    let mut stdout = Vec::new();
    if let Some(mut s) = child.stdout.take() {
        let _ = s.read_to_end(&mut stdout);
    }
    let mut stderr = Vec::new();
    if let Some(mut s) = child.stderr.take() {
        let _ = s.read_to_end(&mut stderr);
    }
    Some(std::process::Output {
        status,
        stdout,
        stderr,
    })
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
}
