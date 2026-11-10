//! openaihost engine install lane (LiteRT-LM): a pip-managed venv per
//! engine tag plus an executable shim that turns the `litert-lm` console
//! script into a llama-server-shaped single binary path. Layout (probed
//! by `probe_openaihost`, found by `find_engine_binary`):
//!
//! ```text
//! engines/litert-0.19.1/
//! ├── venv/                 # uv-created, litert-lm installed
//! │   └── bin/litert-lm
//! └── litert-lm-server      # shim: exec venv/bin/litert-lm "$@"
//! ```
//!
//! Why venv-per-tag (not one shared venv): Blazar's engine lifecycle -
//! install/use/rollback/prune - requires each tag to be independently
//! deletable (`prune` does `remove_dir_all`), and a version pin must be
//! exact. A shared venv would make rollback a mutation, not a switch.
//!
//! Platform matrix: the wheel is pure-python and upstream supports
//! Linux/macOS/Windows (plus Android/RPi at the runtime level) - no
//! backend pinning is needed; the runtime picks its delegates (CPU/GPU/
//! NPU) internally at serve time. The shim takes the platform's
//! executable convention: a `#!/bin/sh` exec wrapper on unix, a
//! `litert-lm-server.cmd` batch forwarder on Windows (venv entry points
//! live under `venv/Scripts` with `.exe` extensions there). Windows
//! live-validation is hardware-blocked from the Linux dev box; the
//! script and layout contracts are unit-pinned on both platforms.

use anyhow::{Context, Result, anyhow};
use std::path::{Path, PathBuf};

use blazar_core::BlazarDirs;

use super::net_probe::Attempt;
use super::sglang_install::{disk_avail_bytes, make_executable, run_streaming, uv_available};

/// litert-lm version Blazar installs when the user names none. Pinned,
/// not "latest": the shim contract, the profile flag surface (served by
/// `serve --help` mining) and the health probe are verified against this
/// release; an unpinned auto-latest would silently drift the contract.
/// Bump deliberately, after re-verifying the serve flags in the new
/// release.
pub const LITERT_LM_DEFAULT_VERSION: &str = "0.19.1";

/// The wheel is pure-python with a modest dependency set (no torch), so
/// the venv lands well under 1 GiB - but refuse installs with less than
/// 1 GiB free instead of dying mid-extract with a half-populated
/// site-packages.
const LITERT_MIN_DISK_BYTES: u64 = 1024 * 1024 * 1024;

/// pip resolving the dependency set can be slow on modest links; the cap
/// exists to catch a wedged download, not to rush a legitimate one.
const LITERT_INSTALL_TIMEOUT_SECS: u64 = 600;

/// First import initializes the runtime's extension modules - the cap
/// catches a wedged import, not a slow one.
const LITERT_BOOT_SMOKE_TIMEOUT_SECS: u64 = 120;

/// Shim file name - the single contract `find_engine_binary`,
/// `register_engine_inner`, and `find_litert_bin` all match on. The
/// extension IS the platform's executable convention: unix sh script
/// vs Windows batch wrapper (`.cmd` needs no chmod; execution rights
/// follow the file's content type on Windows).
#[cfg(windows)]
pub const SHIM_NAME: &str = "litert-lm-server.cmd";
#[cfg(not(windows))]
pub const SHIM_NAME: &str = "litert-lm-server";

/// Venv module interpreter for the no-uv fallback. Windows installs
/// `python.exe` (no `python3` alias by default); unix uses `python3`.
#[cfg(windows)]
const VENV_PY: &str = "python";
#[cfg(not(windows))]
const VENV_PY: &str = "python3";

/// Executable/script directory inside a venv (`bin` on unix, `Scripts`
/// on Windows) - the cpython layout difference every venv-path consumer
/// must share.
#[must_use]
pub fn venv_bin_dir(venv: &Path) -> PathBuf {
    if cfg!(windows) {
        venv.join("Scripts")
    } else {
        venv.join("bin")
    }
}

/// venv python interpreter (`bin/python` unix, `Scripts\python.exe`
/// Windows) - the interpreter `probe_openaihost` and the boot smoke run.
#[must_use]
pub fn venv_python_path(venv: &Path) -> PathBuf {
    if cfg!(windows) {
        venv_bin_dir(venv).join("python.exe")
    } else {
        venv_bin_dir(venv).join("python")
    }
}

/// The venv's `litert-lm` console script the shim execs (`.exe` wrapper
/// on Windows - pip entry points are exe launchers there, not sh
/// scripts).
#[must_use]
pub fn venv_entry_point(venv: &Path) -> PathBuf {
    if cfg!(windows) {
        venv_bin_dir(venv).join("litert-lm.exe")
    } else {
        venv_bin_dir(venv).join("litert-lm")
    }
}

/// Create the venv + install litert-lm + write the shim inside `dir`.
/// `dir` must exist and be empty; on failure the caller removes `dir`
/// (F88 discipline lives with the `EngineManager` entry point).
pub async fn install_into(dir: &Path, version: &str) -> Result<PathBuf> {
    let avail = disk_avail_bytes(dir).context("disk preflight")?;
    if avail < LITERT_MIN_DISK_BYTES {
        #[allow(clippy::cast_precision_loss)]
        let avail_gib = avail as f64 / f64::from(1024 * 1024 * 1024);
        anyhow::bail!(
            "litert-lm needs >= {} GiB free for its venv; {} has only {avail_gib:.1} GiB avail",
            LITERT_MIN_DISK_BYTES / (1024 * 1024 * 1024),
            dir.display(),
        );
    }

    let venv = dir.join("venv");
    let venv_python = venv_python_path(&venv);
    let pin = format!("litert-lm=={version}");
    if uv_available() {
        run_streaming(
            tokio::process::Command::new("uv")
                .arg("venv")
                .arg("--python")
                .arg("3.12")
                .arg(&venv),
            "uv venv",
            LITERT_INSTALL_TIMEOUT_SECS,
        )
        .await?;
        run_streaming(
            tokio::process::Command::new("uv")
                .arg("pip")
                .arg("install")
                .arg("--python")
                .arg(&venv_python)
                .arg(&pin),
            "uv pip install litert-lm",
            LITERT_INSTALL_TIMEOUT_SECS,
        )
        .await?;
    } else {
        run_streaming(
            tokio::process::Command::new(VENV_PY)
                .arg("-m")
                .arg("venv")
                .arg(&venv),
            "python3 -m venv",
            LITERT_INSTALL_TIMEOUT_SECS,
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
            LITERT_INSTALL_TIMEOUT_SECS,
        )
        .await?;
        run_streaming(
            tokio::process::Command::new(&venv_python)
                .arg("-m")
                .arg("pip")
                .arg("install")
                .arg(&pin),
            "pip install litert-lm",
            LITERT_INSTALL_TIMEOUT_SECS,
        )
        .await?;
    }

    // Shim: exec the venv's own console script so argv passthrough keeps
    // probe (`serve --help`) and spawn argv identical in shape.
    // venv's exec dir goes first on PATH so the venv's entry points win
    // over anything ambient. Script body follows the platform's
    // executable convention (SHIM_NAME).
    let entry = venv_entry_point(&venv);
    let shim = dir.join(SHIM_NAME);
    // Runtime cfg pick keeps BOTH script bodies compiled (and therefore
    // unit-pinned) on every platform; the wrong-platform branch is
    // const-folded away.
    let script = if cfg!(windows) {
        shim_script_windows(&entry)
    } else {
        shim_script(&venv_bin_dir(&venv), &entry)
    };
    std::fs::write(&shim, script).with_context(|| format!("write {SHIM_NAME} shim"))?;
    make_executable(&shim)?;

    // Boot smoke: the version/--help probes exit before the deep
    // imports, so a venv whose boot-critical import chain dies still
    // installed and activated cleanly. Import the real package here
    // with the venv python; failure bubbles out of install_into and
    // F88 rolls the dir back - an unbootable venv must never become an
    // activatable row.
    let mut smoke = tokio::process::Command::new(&venv_python);
    smoke.arg("-c").arg("import litert_lm");
    run_streaming(
        &mut smoke,
        "boot-smoke import litert_lm",
        LITERT_BOOT_SMOKE_TIMEOUT_SECS,
    )
    .await
    .with_context(|| {
        format!(
            "boot-smoke failed: the venv cannot import litert-lm - rerun manually to inspect: \
             {} -c 'import litert_lm'",
            venv_python.display()
        )
    })?;
    Ok(shim)
}

/// Latest litert-lm release on `PyPI`, for the doctor currency row.
pub async fn pypi_latest_litert_lm() -> Result<String> {
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
            let resp = match http
                .get("https://pypi.org/pypi/litert-lm/json")
                .send()
                .await
            {
                Ok(resp) => resp,
                Err(e) => return Attempt::Retry(anyhow!("pypi litert-lm query: {e}")),
            };
            match resp.status() {
                reqwest::StatusCode::OK => {}
                other if other.is_server_error() => {
                    return Attempt::Retry(anyhow!("pypi litert-lm status {other}"));
                }
                other => {
                    return Attempt::Done(Err(anyhow!("pypi litert-lm status {other}")));
                }
            }
            match resp.json::<PypiResp>().await {
                Ok(resp) => Attempt::Done(Ok(resp.info.version)),
                // A connection reset mid-body truncates the 200 -
                // decode failures are transport-class, worth one more
                // attempt.
                Err(e) => Attempt::Retry(anyhow!("pypi litert-lm json: {e}")),
            }
        }
    })
    .await
}

/// Same dotted-3 parse as the sglang/mlx lanes, plus glued pre-release
/// tolerance: tags share the `<prefix>-X.Y.Z` shape, `PyPI` versions are
/// plain `X.Y.Z`, and pre-release suffixes (separated `-rc1`/`+dev` or
/// glued `0rc1`) fall back to the leading digits of each component.
#[must_use]
pub fn version_tuple(v: &str) -> Option<(u64, u64, u64)> {
    let digits = |part: &str| {
        let head: String = part.chars().take_while(char::is_ascii_digit).collect();
        head.parse().ok()
    };
    let core = v.split(['-', '+']).next()?;
    let mut it = core.split('.');
    let maj = digits(it.next()?)?;
    let min = digits(it.next().unwrap_or("0"))?;
    let patch = digits(it.next().unwrap_or("0"))?;
    Some((maj, min, patch))
}

// ---------------------------------------------------------------------------
// Bundle lane helpers (pull / rm). The vendor registry is the model
// surface the serve child reads; these fns keep the store row, the
// bundle file, and the registry alias a single lifecycle unit.
// ---------------------------------------------------------------------------

/// Vendor registry alias dir for `alias` (`~/.litert-lm/models/<alias>/`
/// on the default registry; upstream ships no config override by
/// default, and per-child HOME/--config games were rejected as hidden
/// state in the phase-B design).
#[must_use]
pub fn litert_registry_alias_dir(alias: &str) -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".litert-lm").join("models").join(alias))
}

/// Locate the `litert-lm` CLI (the shim) in an installed openaihost
/// engine, active tag first - the same discovery order the quantize
/// tooling uses for llama binaries.
pub fn find_litert_bin(dirs: &BlazarDirs) -> Result<PathBuf> {
    let store = blazar_core::Store::open(dirs).context("open store")?;
    let engines = store.list_engines().context("list engines")?;
    let ordered: Vec<_> = engines
        .iter()
        .filter(|e| e.kind == blazar_core::engine_kind::EngineKind::OpenAiHost)
        .collect();
    let ordered: Vec<_> = ordered
        .iter()
        .filter(|e| e.active)
        .chain(ordered.iter().filter(|e| !e.active))
        .collect();
    for e in ordered {
        let engine_dir = dirs.engines_dir().join(&e.tag);
        if let Ok(bin) = crate::engine::find_engine_binary(&engine_dir, &[SHIM_NAME]) {
            return Ok(bin);
        }
    }
    Err(anyhow!(
        "no openaihost engine installed; run `blazar engine install --kind openaihost` first"
    ))
}

/// Import a downloaded bundle into the vendor registry under `alias`.
/// `litert-lm import` hardlinks on the same filesystem (verified), so
/// the models-dir copy stays the store's artifact and the registry
/// entry costs nothing. The alias dir's existence is re-checked after
/// the CLI reports success - a "successful" import that wrote nothing
/// must fail the pull, not the first chat request (H1).
pub fn litert_import(bin: &Path, bundle: &Path, alias: &str) -> Result<()> {
    let out = std::process::Command::new(bin)
        .arg("import")
        .arg(bundle)
        .arg(alias)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .with_context(|| format!("run {} import", bin.display()))?;
    if !out.status.success() {
        return Err(anyhow!(
            "litert-lm import failed ({}): {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let dir = litert_registry_alias_dir(alias).ok_or_else(|| {
        anyhow!("home directory unavailable: cannot verify the litert-lm registry alias")
    })?;
    if !dir.is_dir() {
        return Err(anyhow!(
            "litert-lm import reported success but {} is missing",
            dir.display()
        ));
    }
    Ok(())
}

/// Shim body for the `litert-lm-server` launcher: PATH export first,
/// exec the venv's console script with argv passthrough.
fn shim_script(venv_bin: &Path, entry: &Path) -> String {
    format!(
        "#!/bin/sh\nexport PATH=\"{venv}:$PATH\"\nexec \"{entry}\" \"$@\"\n",
        venv = venv_bin.display(),
        entry = entry.display(),
    )
}

/// Windows shim body: batch forwarder to the venv's console script.
/// `%*` passes argv through unchanged, and cmd's implicit
/// last-command exit-code propagation keeps probe/spawn status
/// semantics identical to the unix exec shim. std's `Command` routes
/// `.cmd` programs through `cmd.exe /c` with argument escaping (since
/// the `BatBadBut` hardening), so both the manifest probe and the serve
/// spawn can name the shim directly, like any other engine binary.
fn shim_script_windows(entry: &Path) -> String {
    use std::fmt::Write as _;
    let mut script = String::from("@echo off\r\n");
    let _ = write!(script, "\"{}\" %*\r\n", entry.display());
    script
}

/// Remove `alias` from the vendor registry. Prefers the engine's own
/// `litert-lm delete` (authoritative for any internal state); falls
/// back to removing the alias dir directly when no engine is installed
/// (identical effect on the default registry layout). A missing alias
/// dir is a clean no-op: the store row and bundle file are the source
/// of truth, the alias is derived.
pub fn litert_delete(dirs: &BlazarDirs, alias: &str) -> Result<()> {
    let Some(registry_dir) = litert_registry_alias_dir(alias) else {
        return Ok(());
    };
    if !registry_dir.exists() {
        return Ok(());
    }
    if let Ok(bin) = find_litert_bin(dirs) {
        let out = std::process::Command::new(&bin)
            .arg("delete")
            .arg(alias)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output()
            .with_context(|| format!("run {} delete", bin.display()))?;
        if !out.status.success() {
            return Err(anyhow!(
                "litert-lm delete failed ({}): {}",
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        return Ok(());
    }
    // No engine installed: the alias dir IS the registry's entire
    // footprint for this alias on the default layout.
    std::fs::remove_dir_all(&registry_dir)
        .with_context(|| format!("delete {}", registry_dir.display()))?;
    Ok(())
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn unit__version_tuple__dotted3_and_suffix_fallback() {
        assert_eq!(version_tuple("0.19.1"), Some((0, 19, 1)));
        assert_eq!(version_tuple("1.2"), Some((1, 2, 0)));
        // Pre-release suffixes fall back to the leading core.
        assert_eq!(version_tuple("0.20.0rc1"), Some((0, 20, 0)));
        assert_eq!(version_tuple("junk"), None);
    }

    #[test]
    fn unit__shim_script__execs_the_venv_entry_point() {
        // The shim contract probe_openaihost and find_engine_binary
        // rely on: PATH export first, exec the venv's litert-lm
        // console script with argv passthrough.
        let script = shim_script(Path::new("/e/venv/bin"), Path::new("/e/venv/bin/litert-lm"));
        assert!(script.contains("export PATH=\"/e/venv/bin:$PATH\""));
        assert!(script.contains("exec \"/e/venv/bin/litert-lm\" \"$@\""));
    }

    #[test]
    fn unit__shim_script_windows__cmd_forwarder_shape() {
        // The Windows contract: @echo off, quoted entry point, `%*`
        // passthrough, CRLF endings - and no unix-isms.
        let script = shim_script_windows(Path::new(r"C:\e\venv\Scripts\litert-lm.exe"));
        let lines: Vec<&str> = script.split("\r\n").collect();
        assert_eq!(lines[0], "@echo off");
        assert_eq!(lines[1], r#""C:\e\venv\Scripts\litert-lm.exe" %*"#);
        assert_eq!(lines[2], "", "trailing CRLF terminates the batch");
        assert!(!script.contains("exec"), "no unix exec in a batch file");
        assert!(!script.contains('#'), "no shell shebang in a batch file");
    }

    #[test]
    fn unit__shim_and_venv_layout__platform_executable_convention() {
        // One source of truth for the platform contract: shim name,
        // venv exec dir, interpreter, and entry point all agree with
        // cpython's layout on the compiling platform. The Windows arms
        // compile (and run) on Windows CI; the unix arms everywhere.
        if cfg!(windows) {
            assert_eq!(SHIM_NAME, "litert-lm-server.cmd");
            assert_eq!(venv_bin_dir(Path::new("v")), Path::new("v").join("Scripts"));
            assert_eq!(
                venv_python_path(Path::new("v")),
                Path::new("v").join("Scripts").join("python.exe")
            );
            assert_eq!(
                venv_entry_point(Path::new("v")),
                Path::new("v").join("Scripts").join("litert-lm.exe")
            );
        } else {
            assert_eq!(SHIM_NAME, "litert-lm-server");
            assert_eq!(venv_bin_dir(Path::new("v")), Path::new("v").join("bin"));
            assert_eq!(
                venv_python_path(Path::new("v")),
                Path::new("v").join("bin").join("python")
            );
            assert_eq!(
                venv_entry_point(Path::new("v")),
                Path::new("v").join("bin").join("litert-lm")
            );
        }
    }
}
