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
//! Why venv-per-tag (not one shared venv): Pallama's engine lifecycle —
//! install/use/rollback/prune — requires each tag to be independently
//! deletable (`prune` does `remove_dir_all`), and a version pin must be
//! exact. A shared venv would make rollback a mutation, not a switch.
//!
//! sglang upstream supports Linux CUDA/ROCm first-class; Windows and
//! macOS are not supported upstream, so this lane refuses there with a
//! teaching error instead of installing an engine that cannot spawn.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Version Pallama installs when the user names none. Pinned, not
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

/// Free bytes under `path`'s filesystem (`df -B1`). Linux-only lane —
/// see the OS gate in [`install_into`].
fn disk_avail_bytes(path: &Path) -> Result<u64> {
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
                target: "pallama::engine",
                "{}",
                String::from_utf8_lossy(&raw).trim_end()
            ),
        }
    }
}

/// (pip/uv progress visibility); stderr carries the real errors.
async fn run_streaming(cmd: &mut tokio::process::Command, what: &str) -> Result<()> {
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
    .with_context(|| format!("{what} exceeded {SGLANG_INSTALL_TIMEOUT_SECS}s"))?
    .with_context(|| format!("wait {what}"))?;
    if !status.success() {
        anyhow::bail!("{what} exited {status}");
    }
    Ok(())
}

/// Is `uv` on PATH? (Fast venv + pip; plain python3 is the fallback.)
fn uv_available() -> bool {
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
        #[allow(clippy::cast_precision_loss)]
        let avail_gib = avail as f64 / 1e9;
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
        )
        .await?;
    } else {
        run_streaming(
            tokio::process::Command::new("python3")
                .arg("-m")
                .arg("venv")
                .arg(&venv),
            "python3 -m venv",
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
        )
        .await?;
    }

    // Shim: absolute venv python so the script works from any cwd; argv
    // passthrough keeps probe (--) and spawn argv identical in shape.
    // venv/bin goes first on PATH so JIT build tools shipped inside the
    // venv (ninja — flashinfer compiles kernels at runtime) resolve.
    // SGLANG_CACHE_DIR scopes sglang's third-party JIT caches (triton,
    // inductor, nv, flashinfer — all derived from it since v0.5.19,
    // setdefault semantics) to the engine dir: compiled kernels survive
    // restarts AND `engine rm` reclaims them; `:-` keeps an explicit
    // user override in charge.
    let cache = dir.join("cache");
    let shim = dir.join("sglang-server");
    let script = format!(
        "#!/bin/sh\nexport PATH=\"{}:$PATH\"\nexport \
         SGLANG_CACHE_DIR=\"${{SGLANG_CACHE_DIR:-{}}}\"\nexec \"{}\" -m \
         sglang.launch_server \"$@\"\n",
        venv.join("bin").display(),
        cache.display(),
        venv_python.display()
    );
    std::fs::create_dir_all(&cache).context("create engine cache dir")?;
    std::fs::write(&shim, script).context("write sglang-server shim")?;
    make_executable(&shim)?;
    Ok(shim)
}

fn make_executable(path: &Path) -> Result<()> {
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

/// Latest `sglang` version on `PyPI` (`/pypi/sglang/json` endpoint).
/// Currency checks only — never a gate; failures surface as errors the CLI
/// turns into a "cannot check" note, not a failed update.
pub async fn pypi_latest_sglang(timeout: std::time::Duration) -> Result<String> {
    #[derive(serde::Deserialize)]
    struct PypiInfo {
        version: String,
    }
    #[derive(serde::Deserialize)]
    struct PypiResp {
        info: PypiInfo,
    }
    let http = reqwest::Client::builder()
        .timeout(timeout)
        .user_agent(concat!("pallama/", env!("CARGO_PKG_VERSION")))
        .build()
        .context("pypi http client")?;
    let resp: PypiResp = http
        .get("https://pypi.org/pypi/sglang/json")
        .send()
        .await
        .context("pypi sglang query")?
        .error_for_status()
        .context("pypi sglang status")?
        .json()
        .await
        .context("pypi sglang json")?;
    Ok(resp.info.version)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(non_snake_case)]
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
}
