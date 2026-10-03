//! sglang engine install lane: a pip-managed venv per engine tag plus an
//! executable shim that turns `launch_server` into a llama-server-shaped
//! single binary path. Layout (probed by `probe_sglang`, found by
//! `find_engine_binary`):
//!
//! ```text
//! engines/sglang-0.5.19/
//! ├── venv/                 # uv-created, sglang==<version> installed
//! │   └── bin/python
//! └── sglang-server         # shim: exec venv/bin/python -m sglang.launch_server "$@"
//! ```
//!
//! Why venv-per-tag (not one shared venv): Blazar's engine lifecycle —
//! install/use/rollback/prune — requires each tag to be independently
//! deletable (`prune` does `remove_dir_all`), and a version pin must be
//! exact. A shared venv would make rollback a mutation, not a switch.
//!
//! sglang upstream supports Linux CUDA/ROCm first-class; Windows and
//! macOS are not supported upstream, so this lane refuses there with a
//! teaching error instead of installing an engine that cannot spawn.

use anyhow::{Context, Result, anyhow};
use std::path::{Path, PathBuf};

use super::net_probe::Attempt;

/// Version Blazar installs when the user names none. Pinned, not
/// "latest": the profile compiler's flag surface and the fit ladder are
/// verified against this release (`server_args.py` of the tag); an
/// unpinned auto-latest would silently drift the contract. Bump this
/// pin deliberately, after re-verifying the flags in the new release.
pub const SGLANG_DEFAULT_VERSION: &str = "0.5.19";

/// The venv (torch + flashinfer wheels) lands between 5 and 8 GiB on
/// disk. Refuse installs with less than 10 GiB free instead of dying
/// mid-extract with a half-populated site-packages.
const SGLANG_MIN_DISK_BYTES: u64 = 10 * 1024 * 1024 * 1024;

/// pip resolving + downloading multi-GB torch wheels can be slow on
/// modest links; the cap exists to catch a wedged download, not to rush
/// a legitimate one.
const SGLANG_INSTALL_TIMEOUT_SECS: u64 = 1800;

/// First import of the boot-critical chain can legitimately JIT-compile
/// (`deep_ep`'s `init_jit` builds its extension at import when no prebuilt
/// matches, torch import alone is tens of seconds) — the cap catches a
/// wedged compile, not a slow one.
const SGLANG_BOOT_SMOKE_TIMEOUT_SECS: u64 = 600;

/// Free bytes under `path`'s filesystem (`df -B1`). Linux-only lane —
/// see the OS gate in [`install_into`].
pub(crate) fn disk_avail_bytes(path: &Path) -> Result<u64> {
    let out = std::process::Command::new("df")
        .arg("-B1")
        .arg("--output=avail")
        .arg(path)
        .output()
        .context("run df for disk preflight")?;
    if !out.status.success() {
        anyhow::bail!(
            "df exited {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    // Header line ("Available"), then the number.
    let text = String::from_utf8_lossy(&out.stdout);
    let avail = text
        .lines()
        .nth(1)
        .and_then(|l| l.trim().parse::<u64>().ok())
        .context("parse df output")?;
    Ok(avail)
}

/// Run a command to success, streaming each output line into tracing
async fn pump_lines<R>(r: R)
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    use tokio::io::AsyncBufReadExt;
    let mut reader = tokio::io::BufReader::new(r);
    let mut raw = Vec::new();
    loop {
        raw.clear();
        match reader.read_until(b'\n', &mut raw).await {
            Ok(0) | Err(_) => break,
            Ok(_) => tracing::info!(
                target: "blazar::engine",
                "{}",
                String::from_utf8_lossy(&raw).trim_end()
            ),
        }
    }
}

/// (pip/uv progress visibility); stderr carries the real errors.
pub(crate) async fn run_streaming(
    cmd: &mut tokio::process::Command,
    what: &str,
    timeout_secs: u64,
) -> Result<()> {
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    let mut child = cmd.spawn().with_context(|| format!("spawn {what}"))?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let mut tasks = Vec::new();
    if let Some(out) = stdout {
        tasks.push(tokio::spawn(pump_lines(out)));
    }
    if let Some(err) = stderr {
        tasks.push(tokio::spawn(pump_lines(err)));
    }
    for t in tasks {
        let _ = t.await;
    }
    let status = tokio::time::timeout(
        std::time::Duration::from_secs(SGLANG_INSTALL_TIMEOUT_SECS),
        child.wait(),
    )
    .await
    .with_context(|| format!("{what} exceeded {timeout_secs}s"))?
    .with_context(|| format!("wait {what}"))?;
    if !status.success() {
        anyhow::bail!("{what} exited {status}");
    }
    Ok(())
}

/// Is `uv` on PATH? (Fast venv + pip; plain python3 is the fallback.)
pub(crate) fn uv_available() -> bool {
    std::process::Command::new("uv")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Create the venv + install sglang + write the shim inside `dir`.
/// `dir` must exist and be empty; on failure the caller removes `dir`
/// (F88 discipline lives with the `EngineManager` entry point).
#[allow(clippy::too_many_lines)] // linear lane build: preflight, venv, wheels, shim, boot-smoke
pub async fn install_into(dir: &Path, version: &str) -> Result<PathBuf> {
    if !cfg!(target_os = "linux") {
        anyhow::bail!(
            "sglang upstream supports Linux (CUDA/ROCm); {} installs are not \
             supported by the engine itself",
            std::env::consts::OS
        );
    }
    let avail = disk_avail_bytes(dir).context("disk preflight")?;
    if avail < SGLANG_MIN_DISK_BYTES {
        // Compare AND display in the same unit (GiB = 1024^3): an
        // earlier revision divided bytes by 1e9 for the message, so a
        // 9.4-GiB-free disk printed "10.1 GiB avail" while being
        // refused — an apparent contradiction (live-reported).
        #[allow(clippy::cast_precision_loss)]
        let avail_gib = avail as f64 / f64::from(1024 * 1024 * 1024);
        anyhow::bail!(
            "sglang needs >= {} GiB free for its venv (torch + flashinfer); \
             {} has only {avail_gib:.1} GiB avail",
            SGLANG_MIN_DISK_BYTES / (1024 * 1024 * 1024),
            dir.display(),
        );
    }

    let venv = dir.join("venv");
    let venv_python = venv.join("bin").join("python");
    if uv_available() {
        run_streaming(
            tokio::process::Command::new("uv")
                .arg("venv")
                .arg("--python")
                .arg("3.12")
                .arg(&venv),
            "uv venv",
            SGLANG_INSTALL_TIMEOUT_SECS,
        )
        .await?;
        run_streaming(
            tokio::process::Command::new("uv")
                .arg("pip")
                .arg("install")
                // sglang pins pre-release deps (e.g. cuda-tile==1.6.0rc5);
                // uv refuses transitive pre-releases unless allowed, while
                // pip accepts exact pins from dependents.
                .arg("--prerelease=allow")
                .arg("--python")
                .arg(&venv_python)
                .arg(format!("sglang=={version}"))
                .arg("ninja"),
            "uv pip install sglang",
            SGLANG_INSTALL_TIMEOUT_SECS,
        )
        .await?;
    } else {
        run_streaming(
            tokio::process::Command::new("python3")
                .arg("-m")
                .arg("venv")
                .arg(&venv),
            "python3 -m venv",
            SGLANG_INSTALL_TIMEOUT_SECS,
        )
        .await?;
        run_streaming(
            tokio::process::Command::new(&venv_python)
                .arg("-m")
                .arg("pip")
                .arg("install")
                .arg("--upgrade")
                .arg("pip"),
            "pip self-upgrade",
            SGLANG_INSTALL_TIMEOUT_SECS,
        )
        .await?;
        run_streaming(
            tokio::process::Command::new(&venv_python)
                .arg("-m")
                .arg("pip")
                .arg("install")
                .arg(format!("sglang=={version}"))
                .arg("ninja"),
            "pip install sglang",
            SGLANG_INSTALL_TIMEOUT_SECS,
        )
        .await?;
    }

    // Shim: absolute venv python so the script works from any cwd; argv
    // passthrough keeps probe (--) and spawn argv identical in shape.
    // venv/bin goes first on PATH so JIT build tools shipped inside the
    // venv (ninja — flashinfer compiles kernels at runtime) resolve.
    // The sglang wheels pull a `nvidia-cuda-nvcc` wheel whose nvcc MATCHES
    // the bundled CUDA torch: put its bin dir ahead of the system PATH so
    // flashinfer JIT self-hosts instead of dying on a system nvcc that
    // mismatches the wheels (e.g. system 12.0 vs cu13 torch).
    // SGLANG_CACHE_DIR scopes sglang's third-party JIT caches (triton,
    // inductor, nv, flashinfer — all derived from it since v0.5.19,
    // setdefault semantics) to the engine dir: compiled kernels survive
    // restarts AND `engine rm` reclaims them; `:-` keeps an explicit
    // user override in charge.
    let cache = dir.join("cache");
    let shim = dir.join("sglang-server");
    let toolkit = bundled_cuda_toolkit_root(&venv);
    if let Some(root) = &toolkit {
        ensure_dev_toolkit_layout(root);
    }
    let mut path_prefix = venv.join("bin").display().to_string();
    let mut cuda_home_export = String::new();
    if let Some(root) = &toolkit {
        // Hard-set, not ${CUDA_HOME:-default}: the bundled toolkit is the
        // only nvcc matching the venv's torch wheels; a system CUDA_HOME
        // would poison JIT compiles with mismatched headers (the same
        // reasoning as the PATH prefix below).
        cuda_home_export = format!("export CUDA_HOME=\"{}\"\n", root.display());
        path_prefix = format!("{}:{}", root.join("bin").display(), path_prefix);
    }
    let script = format!(
        "#!/bin/sh\nexport PATH=\"{}:$PATH\"\n{}export \
         SGLANG_CACHE_DIR=\"${{SGLANG_CACHE_DIR:-{}}}\"\nexec \"{}\" -m \
         sglang.launch_server \"$@\"\n",
        path_prefix,
        cuda_home_export,
        cache.display(),
        venv_python.display()
    );
    std::fs::create_dir_all(&cache).context("create engine cache dir")?;
    std::fs::write(&shim, script).context("write sglang-server shim")?;
    make_executable(&shim)?;

    // Boot smoke: the `--version` probes (install register, doctor,
    // supervisor rollback gate) exit before the deep imports, so a venv
    // whose boot-critical import chain dies still installed and activated
    // cleanly (live 2026-09-27: deep_ep's find_cuda_home assert + a rope
    // JIT death both reached serve-time as 502s). Import the real entry
    // chain here with the shim's exact env; failure bubbles out of
    // install_into and F88 rolls the dir back — an unbootable venv must
    // never become an activatable row.
    let inherited_path = std::env::var("PATH").unwrap_or_default();
    let mut smoke = tokio::process::Command::new(&venv_python);
    smoke
        .arg("-c")
        .arg("import sglang.srt.entrypoints.engine")
        .env("PATH", format!("{path_prefix}:{inherited_path}"));
    if let Some(root) = &toolkit {
        smoke.env("CUDA_HOME", root);
    }
    if std::env::var_os("SGLANG_CACHE_DIR").is_none() {
        smoke.env("SGLANG_CACHE_DIR", &cache);
    }
    run_streaming(
        &mut smoke,
        "boot-smoke import sglang.srt.entrypoints.engine",
        SGLANG_BOOT_SMOKE_TIMEOUT_SECS,
    )
    .await
    .context("boot-smoke failed: the venv cannot import the sglang server entry chain — rerun manually to inspect: <engine>/venv/bin/python -c 'import sglang.srt.entrypoints.engine'")?;
    Ok(shim)
}

/// The venv's bundled CUDA toolkit ROOT when the sglang wheels pulled a
/// compiler dist (`nvidia-cuda-nvcc`), so the shim can self-host the JIT
/// toolchain: nvcc's bin on PATH plus `CUDA_HOME` for the consumers that
/// demand it by name (`deep_ep`'s `find_cuda_home`, torch `cpp_extension`,
/// flashinfer).
///
/// Two wheel generations, two layouts: CUDA 12-era dists land the
/// toolkit at `nvidia/cuda_nvcc`, CUDA 13-era at `nvidia/cu13`. Probe
/// both instead of hardcoding one — a torch float across reinstall
/// generations silently moved the toolkit and dead-ended every JIT
/// import while the compiler sat one directory over (live incident
/// 2026-09-27: sglang-0.5.19 + torch 2.13 cu13 wheels, spawn 502s from
/// `cuda_home is None` and a rope JIT death).
fn bundled_cuda_toolkit_root(venv: &Path) -> Option<PathBuf> {
    let lib = std::fs::read_dir(venv.join("lib")).ok()?;
    let python_dir = lib
        .filter_map(std::result::Result::ok)
        .find(|e| {
            e.path()
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("python"))
        })?
        .path();
    let nvidia = python_dir.join("site-packages").join("nvidia");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&nvidia)
        .ok()?
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            let Some(name) = p.file_name().and_then(|n| n.to_str()) else {
                return false;
            };
            // cu13-style generation dirs (digits after "cu") and the
            // legacy cuda_nvcc dir carry the compiler; everything else
            // under nvidia/ is runtime libs without a bin/.
            let generation = name.len() > 2
                && name.starts_with("cu")
                && name[2..].chars().all(|c| c.is_ascii_digit());
            generation || name == "cuda_nvcc"
        })
        .collect();
    // Deterministic order; a cu<N> generation wins over the legacy name
    // if both somehow coexist — it matches the wheel set the venv was
    // just resolved against.
    candidates.sort_by(|a, b| {
        let rank = |p: &Path| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            (!name.starts_with("cu"), name.to_string())
        };
        rank(a).cmp(&rank(b))
    });
    let nvcc = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
    candidates
        .into_iter()
        .find(|root| root.join("bin").join(nvcc).is_file())
}

/// Bridge the pip wheels' runtime layout to the developer layout sglang's
/// JIT linker specs assume: `toolchain.py` links `-L$CUDA_HOME/lib64
/// -lcudart`, but the cu13-generation wheels ship `lib/` with versioned
/// sonames only (`libcudart.so.13`, no unversioned dev link) — ld dies
/// with `cannot find -lcudart` right after nvcc compiles cleanly (live
/// 2026-09-28: rope JIT on `sm_89`). Create the two entries a system dev
/// toolkit carries: `lib64 -> lib` and `libcudart.so -> libcudart.so.N`.
/// Best-effort with a warn: failure leaves the deterministic JIT link
/// error visible at spawn instead of masking it here.
fn ensure_dev_toolkit_layout(root: &Path) {
    #[cfg(unix)]
    {
        let lib = root.join("lib");
        if !lib.is_dir() {
            return;
        }
        let lib64 = root.join("lib64");
        if !lib64.exists()
            && let Err(e) = std::os::unix::fs::symlink("lib", &lib64)
        {
            tracing::warn!(
                "cannot link {} -> lib in the bundled toolkit: {e}",
                lib64.display()
            );
        }
        // Highest versioned soname wins if a wheel ever ships several.
        let soname = std::fs::read_dir(&lib).ok().and_then(|entries| {
            entries
                .filter_map(std::result::Result::ok)
                .filter_map(|e| e.file_name().into_string().ok())
                .filter(|n| {
                    n.starts_with("libcudart.so.")
                        && n["libcudart.so.".len()..]
                            .chars()
                            .all(|c| c.is_ascii_digit())
                })
                .max()
        });
        if let Some(soname) = soname {
            let dev_link = lib.join("libcudart.so");
            if !dev_link.exists()
                && let Err(e) = std::os::unix::fs::symlink(&soname, &dev_link)
            {
                tracing::warn!("cannot link libcudart.so -> {soname} in the bundled toolkit: {e}");
            }
        }
    }
    #[cfg(not(unix))]
    let _ = root;
}

#[cfg_attr(not(unix), allow(clippy::unnecessary_wraps, unused_variables))] // chmod is unix-only
pub(crate) fn make_executable(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(path, perms)?;
    }
    Ok(())
}

/// Parse an `X.Y.Z` version into a comparable tuple (`0.5.19` ->
/// `(0, 5, 19)`). Pre-release suffixes (`0.5.20rc1`) compare by their
/// numeric head, which is all the currency hint needs.
#[must_use]
pub fn version_tuple(v: &str) -> Option<(u64, u64, u64)> {
    // Keep only the leading `X.Y.Z` core; anything after the first
    // non-[0-9.] char is a pre-release suffix (`0.5.20rc1` -> `0.5.20`).
    let core_end = v
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(v.len());
    let mut it = v[..core_end]
        .trim_end_matches('.')
        .split('.')
        .filter_map(|p| p.parse::<u64>().ok());
    let maj = it.next()?;
    let min = it.next()?;
    let patch = it.next().unwrap_or(0);
    Some((maj, min, patch))
}

pub async fn pypi_latest_sglang() -> Result<String> {
    #[derive(serde::Deserialize)]
    struct PypiInfo {
        version: String,
    }
    #[derive(serde::Deserialize)]
    struct PypiResp {
        info: PypiInfo,
    }
    // The client-level timeout is the per-attempt cap: each retry issues
    // a fresh request on a clone of this client.
    blazar_core::tls::ensure_tls_provider();
    let http = reqwest::Client::builder()
        .timeout(crate::engine::net_probe::PROBE_ATTEMPT_CAP)
        .user_agent(concat!("blazar/", env!("CARGO_PKG_VERSION")))
        .build()
        .context("pypi http client")?;
    crate::engine::net_probe::retry_probe(move || {
        let http = http.clone();
        async move {
            let resp = match http.get("https://pypi.org/pypi/sglang/json").send().await {
                Ok(resp) => resp,
                Err(e) => return Attempt::Retry(anyhow!("pypi sglang query: {e}")),
            };
            match resp.status() {
                reqwest::StatusCode::OK => {}
                other if other.is_server_error() => {
                    return Attempt::Retry(anyhow!("pypi sglang status {other}"));
                }
                other => {
                    return Attempt::Done(Err(anyhow!("pypi sglang status {other}")));
                }
            }
            match resp.json::<PypiResp>().await {
                Ok(resp) => Attempt::Done(Ok(resp.info.version)),
                // A connection reset mid-body truncates the 200 —
                // decode failures are transport-class, worth one more
                // attempt.
                Err(e) => Attempt::Retry(anyhow!("pypi sglang json: {e}")),
            }
        }
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(non_snake_case)]
    #[cfg(target_os = "linux")] // disk_avail_bytes shells linux df -B1 --output
    fn unit__disk_preflight__parses_df_output() {
        // The real df runs on the box; this pins the PARSER against the
        // exact shape `df -B1 --output=avail <path>` prints.
        let dir = tempfile::tempdir().unwrap();
        let avail = disk_avail_bytes(dir.path()).unwrap();
        assert!(avail > 0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__version_tuple__numeric_and_prerelease_heads() {
        assert_eq!(version_tuple("0.5.19"), Some((0, 5, 19)));
        assert_eq!(version_tuple("0.5.20rc1"), Some((0, 5, 20)));
        assert_eq!(version_tuple("1.2"), Some((1, 2, 0)));
        assert_eq!(version_tuple("latest"), None);
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__bundled_cuda_toolkit_root__detects_both_wheel_generations_and_stays_silent_without() {
        // The wheel lands nvcc.exe on Windows, bare nvcc elsewhere — the
        // probe checks the platform-correct name.
        let nvcc = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
        let site_packages = |venv: &tempfile::TempDir, generation: &str| {
            venv.path()
                .join("lib")
                .join("python3.12")
                .join("site-packages")
                .join("nvidia")
                .join(generation)
        };

        let venv = tempfile::tempdir().unwrap();
        assert_eq!(bundled_cuda_toolkit_root(venv.path()), None);

        // CUDA 13-era layout: the nvidia-cuda-nvcc wheel moved the whole
        // toolkit to nvidia/cu13 (bin + include + lib + nvvm) — a torch
        // float across reinstall generations, not a layout blazar picks.
        let cu13 = site_packages(&venv, "cu13");
        std::fs::create_dir_all(cu13.join("bin")).unwrap();
        // Dir alone (no nvcc binary) still yields None.
        assert_eq!(bundled_cuda_toolkit_root(venv.path()), None);
        std::fs::write(cu13.join("bin").join(nvcc), "#!/bin/sh\n").unwrap();
        let found = bundled_cuda_toolkit_root(venv.path()).expect("cu13 toolkit detected");
        assert!(found.ends_with("cu13"), "{found:?}");

        // Legacy CUDA 12-era layout keeps working: nvidia/cuda_nvcc.
        let legacy = tempfile::tempdir().unwrap();
        let nvcc_dir = site_packages(&legacy, "cuda_nvcc").join("bin");
        std::fs::create_dir_all(&nvcc_dir).unwrap();
        std::fs::write(nvcc_dir.join(nvcc), "#!/bin/sh\n").unwrap();
        let found = bundled_cuda_toolkit_root(legacy.path()).expect("legacy toolkit detected");
        assert!(found.ends_with("cuda_nvcc"), "{found:?}");

        // cu-prefixed dirs without digits (cuda_runtime et al) are runtime
        // libs, never candidates: build the tree and prove the miss.
        let libs = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(site_packages(&libs, "cuda_runtime").join("bin")).unwrap();
        std::fs::write(
            site_packages(&libs, "cuda_runtime").join("bin").join(nvcc),
            "#!/bin/sh\n",
        )
        .unwrap();
        assert_eq!(bundled_cuda_toolkit_root(libs.path()), None);
    }

    #[cfg(unix)]
    #[test]
    #[allow(non_snake_case)]
    fn unit__ensure_dev_toolkit_layout__bridges_runtime_wheels_to_dev_layout() {
        // The cu13-generation wheels ship lib/libcudart.so.13 with no
        // lib64 and no unversioned dev link — sglang's JIT linker specs
        // (`-L$CUDA_HOME/lib64 -lcudart`) need both (live 2026-09-28:
        // `ld: cannot find -lcudart` right after a clean nvcc compile).
        let toolkit = tempfile::tempdir().unwrap();
        let lib = toolkit.path().join("lib");
        std::fs::create_dir_all(&lib).unwrap();
        std::fs::write(lib.join("libcudart.so.13"), b"so").unwrap();
        std::fs::write(lib.join("libcudart_static.a"), b"ar").unwrap();

        ensure_dev_toolkit_layout(toolkit.path());

        let lib64 = std::fs::read_link(toolkit.path().join("lib64")).unwrap();
        assert_eq!(lib64, std::path::Path::new("lib"));
        let dev = std::fs::read_link(lib.join("libcudart.so")).unwrap();
        assert_eq!(dev, std::path::Path::new("libcudart.so.13"));

        // Idempotent: a second pass (reinstall over an existing tree)
        // changes nothing and never errors.
        ensure_dev_toolkit_layout(toolkit.path());

        // Runtime-only root (no lib/) is a silent no-op, not a failure.
        let bare = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(bare.path().join("bin")).unwrap();
        ensure_dev_toolkit_layout(bare.path());
        assert!(!bare.path().join("lib64").exists());
    }
}
