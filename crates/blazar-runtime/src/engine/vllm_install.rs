//! vLLM lane install: pip venv + `vllm-server` shim, mirroring the sglang
//! lane layout so register/serve/rollback treat both venv lanes identically.
//!
//! ```text
//! engines/vllm-<version>/
//! ├── venv/            python 3.12 + vllm==<version> (torch + cuda wheels)
//! ├── cache/           engine-scoped VLLM_CACHE_ROOT (compile caches)
//! └── vllm-server      sh shim: PATH + cache exports, exec vllm serve "$@"
//! ```
//!
//! Shares `sglang_install`'s serve-time JIT toolchain discipline: vLLM's
//! flashinfer kernels (sampling top-p/top-k at minimum) compile with nvcc
//! inside `EngineCore` init on FIRST SERVE, so the venv must carry a CUDA
//! compiler coherent with its torch wheels and the shim must export it
//! (PATH + `CUDA_HOME`) or the JIT falls back to any system nvcc - which
//! mismatches and dies (live 2026-11-18: system nvcc 12.0 vs cu13 torch,
//! `nvcc fatal : Unknown option '--compress-mode=size'`).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use tokio::process::Command as TokioCommand;

// Shared venv-lane plumbing (df preflight, streamed pip/uv runs, uv probe,
// chmod, and the serve-time JIT toolchain discipline) lives in the sglang
// module rather than being duplicated per lane.
use super::sglang_install::{
    bundled_cuda_toolkit_root, disk_avail_bytes, ensure_dev_toolkit_layout, make_executable,
    pin_jit_compiler_to_torch, run_streaming, uv_available,
};
use crate::engine::net_probe::Attempt;

/// Currency floor for offline installs; bumped per lane release wave.
pub const VLLM_FALLBACK_VERSION: &str = "0.34.0";

/// The venv lands ~6-8 GiB (vllm wheel ~0.5 GiB + torch + cuda dists);
/// refuse to start a download that cannot finish.
const VLLM_MIN_DISK_BYTES: u64 = 10 * 1024 * 1024 * 1024;

const VLLM_INSTALL_TIMEOUT_SECS: u64 = 1800;

/// First `import vllm` pulls the full engine stack through CUDA init -
/// nothing like the fast `--version` probes - so it gets its own budget.
const VLLM_BOOT_SMOKE_TIMEOUT_SECS: u64 = 600;

/// Create the venv + install vllm + write the shim inside `dir`.
/// `dir` must exist and be empty; on failure the caller removes `dir`
/// (F88 discipline lives with the `EngineManager` entry point).
/// `extras` are verbatim pip specs layered onto the venv AFTER the base
/// install and compiler pin, under the shim's exact toolkit env so any
/// kernel-building extra compiles against the coherent nvcc.
#[allow(clippy::too_many_lines)] // linear lane build: preflight, venv, wheels, pin, shim, extras, boot-smoke
pub async fn install_into(dir: &Path, version: &str, extras: &[String]) -> Result<PathBuf> {
    if !cfg!(target_os = "linux") {
        bail!(
            "vllm upstream supports Linux (CUDA); {} installs are not \
             supported by the engine itself",
            std::env::consts::OS
        );
    }
    let extras: Vec<&str> = extras
        .iter()
        .map(std::string::String::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    if extras.len()
        != extras
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
    {
        bail!("duplicate --with package specs: {extras:?}");
    }
    let avail = disk_avail_bytes(dir).context("disk preflight")?;
    if avail < VLLM_MIN_DISK_BYTES {
        // Compare AND display in the same unit (GiB = 1024^3): dividing
        // bytes by 1e9 for the message makes a refused 9.4-GiB-free disk
        // print "10.1 GiB avail" - an apparent contradiction.
        #[allow(clippy::cast_precision_loss)]
        let avail_gib = avail as f64 / f64::from(1024 * 1024 * 1024);
        bail!(
            "vllm needs >= {} GiB free for its venv (torch + cuda wheels); \
             {} has only {avail_gib:.1} GiB avail",
            VLLM_MIN_DISK_BYTES / (1024 * 1024 * 1024),
            dir.display(),
        );
    }

    let venv = dir.join("venv");
    let venv_python = venv.join("bin").join("python");
    let use_uv = uv_available();
    if use_uv {
        run_streaming(
            TokioCommand::new("uv")
                .arg("venv")
                .arg("--python")
                .arg("3.12")
                .arg(&venv),
            "uv venv",
            VLLM_INSTALL_TIMEOUT_SECS,
        )
        .await?;
        run_streaming(
            TokioCommand::new("uv")
                .arg("pip")
                .arg("install")
                // vllm pins stable releases only (no cuda-tile-class
                // pre-release transitives like sglang), so uv's default
                // resolution is enough here.
                .arg("--python")
                .arg(&venv_python)
                .arg(format!("vllm=={version}")),
            "uv pip install vllm",
            VLLM_INSTALL_TIMEOUT_SECS,
        )
        .await?;
    } else {
        run_streaming(
            TokioCommand::new("python3")
                .arg("-m")
                .arg("venv")
                .arg(&venv),
            "python3 -m venv",
            VLLM_INSTALL_TIMEOUT_SECS,
        )
        .await?;
        run_streaming(
            TokioCommand::new(&venv_python)
                .arg("-m")
                .arg("pip")
                .arg("install")
                .arg("--upgrade")
                .arg("pip"),
            "pip self-upgrade",
            VLLM_INSTALL_TIMEOUT_SECS,
        )
        .await?;
        run_streaming(
            TokioCommand::new(&venv_python)
                .arg("-m")
                .arg("pip")
                .arg("install")
                .arg(format!("vllm=={version}")),
            "pip install vllm",
            VLLM_INSTALL_TIMEOUT_SECS,
        )
        .await?;
    }

    // vllm's wheels resolve their own CUDA dists, but the compiler trio
    // FLOATS to the newest line while torch fixes headers to one exact
    // line (live 2026-11-18: vllm 0.34.0 venv carried nvcc 13.5 against
    // torch 2.14+cu130) - the same drift pin_jit_compiler_to_torch exists
    // to close for sglang. Re-resolve onto torch's line so serve-time
    // flashinfer JIT compiles are coherent end to end.
    pin_jit_compiler_to_torch(&venv_python, use_uv).await?;

    // Shim: absolute venv console script so it works from any cwd; argv
    // passthrough keeps probe (`--help=all`, version) and spawn argv
    // identical in shape. PATH puts the toolkit's bin FIRST when the venv
    // carries one - the only nvcc matching its torch wheels; a system
    // nvcc mismatches and flashinfer's build dies on its flags (see the
    // module doc). venv/bin follows so JIT build tools shipped in the
    // venv (ninja) resolve ahead of system ones.
    // CUDA_HOME is hard-set, not ${CUDA_HOME:-default}, for the same
    // reason as the PATH order: consumers that probe it by name (torch
    // cpp_extension, flashinfer) must not find a system toolkit.
    // VLLM_CACHE_ROOT scopes vllm's own caches (torch compile / startup
    // plan artifacts - vllm/envs.py defaults it to ~/.cache/vllm) into
    // the engine dir so `engine rm` reclaims them. FLASHINFER_WORKSPACE_
    // BASE scopes flashinfer's JIT workspace: it resolves at IMPORT time,
    // so a missing export freezes kernel builds at ~/.cache/flashinfer
    // where engine rm cannot reclaim them; `:-` on both keeps an explicit
    // user override in charge.
    let cache = dir.join("cache");
    let shim = dir.join("vllm-server");
    let toolkit = bundled_cuda_toolkit_root(&venv);
    if let Some(root) = &toolkit {
        ensure_dev_toolkit_layout(root);
    } else {
        tracing::warn!(
            target: "blazar::engine",
            "vllm venv carries no bundled CUDA toolkit - flashinfer JIT will build against \
             whatever nvcc the system provides, which may mismatch the torch wheels"
        );
    }
    let mut path_prefix = venv.join("bin").display().to_string();
    let mut cuda_home_export = String::new();
    if let Some(root) = &toolkit {
        cuda_home_export = format!("export CUDA_HOME=\"{}\"\n", root.display());
        path_prefix = format!("{}:{}", root.join("bin").display(), path_prefix);
    }
    let script = wrapper_script(&path_prefix, &cuda_home_export, &cache, &venv);
    std::fs::create_dir_all(&cache).context("create engine cache dir")?;
    std::fs::write(&shim, script).context("write vllm-server shim")?;
    make_executable(&shim)?;

    // Extra pip specs: installed AFTER the compiler pin under the shim's
    // exact env (toolkit bin + CUDA_HOME) so any kernel-building extra
    // compiles against the coherent nvcc - resolution may still re-pin
    // deps per the extra's own contract, and the boot smoke below gates
    // the result.
    if !extras.is_empty() {
        let inherited_path = std::env::var("PATH").unwrap_or_default();
        let mut extra_cmd = if use_uv {
            let mut c = TokioCommand::new("uv");
            c.arg("pip")
                .arg("install")
                .arg("--python")
                .arg(&venv_python);
            c
        } else {
            let mut c = TokioCommand::new(&venv_python);
            c.arg("-m").arg("pip").arg("install");
            c
        };
        extra_cmd
            .args(&extras)
            .env("PATH", format!("{path_prefix}:{inherited_path}"));
        if let Some(root) = &toolkit {
            extra_cmd.env("CUDA_HOME", root);
        }
        let specs = extras.join(" ");
        run_streaming(
            &mut extra_cmd,
            &format!("install lane extras ({specs})"),
            VLLM_INSTALL_TIMEOUT_SECS,
        )
        .await?;
    }

    // Boot smoke: the `--help`/version probes exit before the deep
    // imports, so a venv whose boot-critical import chain dies still
    // installed cleanly. Import the real package here with the shim's
    // PATH; failure bubbles out of install_into and F88 rolls the dir
    // back - an unbootable venv must never become an activatable row.
    let inherited_path = std::env::var("PATH").unwrap_or_default();
    let mut smoke = TokioCommand::new(&venv_python);
    smoke
        .arg("-c")
        .arg("import vllm")
        .env("PATH", format!("{path_prefix}:{inherited_path}"));
    if let Some(root) = &toolkit {
        smoke.env("CUDA_HOME", root);
    }
    if std::env::var_os("VLLM_CACHE_ROOT").is_none() {
        smoke.env("VLLM_CACHE_ROOT", &cache);
    }
    run_streaming(
        &mut smoke,
        "boot-smoke import vllm",
        VLLM_BOOT_SMOKE_TIMEOUT_SECS,
    )
    .await
    .context(
        "boot-smoke failed: the venv cannot import vllm - rerun manually to inspect: \
             <engine>/venv/bin/python -c 'import vllm'",
    )?;
    Ok(shim)
}

/// The `vllm-server` shim body: PATH = toolkit bin : venv bin : system
/// (toolkit segment only when the venv carries one), optional hard-set
/// `CUDA_HOME`, engine-scoped cache exports, then exec the venv's own
/// `vllm` console script (its shebang already points at the venv python)
/// with the `serve` subcommand.
fn wrapper_script(path_prefix: &str, cuda_home_export: &str, cache: &Path, venv: &Path) -> String {
    let cache = cache.display();
    format!(
        "#!/bin/sh\nexport PATH=\"{path_prefix}:$PATH\"\n{cuda_home_export}export \
         VLLM_CACHE_ROOT=\"${{VLLM_CACHE_ROOT:-{cache}}}\"\nexport \
         FLASHINFER_WORKSPACE_BASE=\"${{FLASHINFER_WORKSPACE_BASE:-{cache}}}\"\nexec \
         \"{}\" serve \"$@\"\n",
        venv.join("bin").join("vllm").display(),
    )
}

pub async fn pypi_latest_vllm() -> Result<String> {
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
            let resp = match http.get("https://pypi.org/pypi/vllm/json").send().await {
                Ok(resp) => resp,
                Err(e) => return Attempt::Retry(anyhow!("pypi vllm query: {e}")),
            };
            match resp.status() {
                reqwest::StatusCode::OK => {}
                other if other.is_server_error() => {
                    return Attempt::Retry(anyhow!("pypi vllm status {other}"));
                }
                other => {
                    return Attempt::Done(Err(anyhow!("pypi vllm status {other}")));
                }
            }
            match resp.json::<PypiResp>().await {
                Ok(resp) => Attempt::Done(Ok(resp.info.version)),
                // A connection reset mid-body truncates the 200 -
                // decode failures are transport-class, worth one more
                // attempt.
                Err(e) => Attempt::Retry(anyhow!("pypi vllm json: {e}")),
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
    fn unit__wrapper_script__execs_console_script_and_scopes_cache_to_the_engine_dir() {
        // Bare venv (no bundled toolkit): PATH leads with venv/bin, no
        // CUDA_HOME export at all.
        let script = wrapper_script(
            "/eng/venv/bin",
            "",
            Path::new("/eng/cache"),
            Path::new("/eng/venv"),
        );
        assert!(script.starts_with("#!/bin/sh\n"));
        assert!(script.contains("export PATH=\"/eng/venv/bin:$PATH\""));
        assert!(!script.contains("CUDA_HOME"));
        // The cache roots must be engine-scoped with an explicit user
        // override still in charge; pin both so a regression to
        // ~/.cache/vllm or ~/.cache/flashinfer cannot land silently.
        assert!(script.contains("export VLLM_CACHE_ROOT=\"${VLLM_CACHE_ROOT:-/eng/cache}\""));
        assert!(script.contains(
            "export FLASHINFER_WORKSPACE_BASE=\"${FLASHINFER_WORKSPACE_BASE:-/eng/cache}\""
        ));
        // exec targets the venv's own console script (not `python -m`,
        // which is a different import path) with the serve subcommand.
        assert!(script.contains("exec \"/eng/venv/bin/vllm\" serve \"$@\""));
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__wrapper_script__toolkit_env_self_hosts_the_flashinfer_jit() {
        // Bundled toolkit present: its bin goes AHEAD of the venv (the
        // only nvcc matching the torch wheels) and CUDA_HOME is hard-set,
        // not ${CUDA_HOME:-soft} - a system CUDA_HOME must not win.
        let script = wrapper_script(
            "/eng/venv/lib/python3.12/site-packages/nvidia/cu13/bin:/eng/venv/bin",
            "export CUDA_HOME=\"/eng/venv/lib/python3.12/site-packages/nvidia/cu13\"\n",
            Path::new("/eng/cache"),
            Path::new("/eng/venv"),
        );
        assert!(script.contains(
            "export PATH=\"/eng/venv/lib/python3.12/site-packages/nvidia/cu13/bin:/eng/venv/bin:$PATH\""
        ));
        assert!(
            script.contains(
                "export CUDA_HOME=\"/eng/venv/lib/python3.12/site-packages/nvidia/cu13\"\n"
            )
        );
        assert!(!script.contains("${CUDA_HOME"));
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__fallback_version__is_a_pinned_stable_triple() {
        // The fallback installs offline; it must stay an exact stable
        // release (pre-release heads would need uv's prerelease allow).
        assert_eq!(
            crate::engine::sglang_install::version_tuple(VLLM_FALLBACK_VERSION),
            Some((0, 34, 0))
        );
    }
}
