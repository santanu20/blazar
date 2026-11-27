//! mlx engine install lane: a pip-managed venv per engine tag plus an
//! executable shim that turns `mlx_lm.server` into a llama-server-shaped
//! single binary path. Layout (probed by `probe_mlx`, found by
//! `find_engine_binary`):
//!
//! ```text
//! engines/mlx-0.32.0/
//! ├── venv/                 # uv-created, mlx-lm + backend installed
//! │   └── bin/python
//! └── mlx-server            # shim: exec venv/bin/python -m mlx_lm.server "$@"
//! ```
//!
//! Why venv-per-tag (not one shared venv): Blazar's engine lifecycle -
//! install/use/rollback/prune - requires each tag to be independently
//! deletable (`prune` does `remove_dir_all`), and a version pin must be
//! exact. A shared venv would make rollback a mutation, not a switch.
//!
//! Backend matrix (mirrors how mlx-lm 0.32 itself distributes, `PyPI`
//! `requires_dist`): Linux/NVIDIA installs `mlx[cuda12]` - the CUDA
//! backend rides the standard package as an extra (the separate
//! `mlx-cuda` wheel was the pre-0.32 distribution and is no longer the
//! path mlx-lm resolves); macOS installs plain `mlx-lm` whose Darwin
//! marker pulls Metal `mlx` natively; CPU-only Linux refuses with
//! teaching - a CPU mlx lane would silently serve at toy speed next to
//! the CUDA lanes. `install_plan` decides all of it from the host's OS,
//! CPU, libc and glibc before anything is downloaded.

use anyhow::{Context, Result, anyhow};
use std::path::{Path, PathBuf};

use super::net_probe::Attempt;
use super::sglang_install::{disk_avail_bytes, make_executable, run_streaming, uv_available};

/// mlx-lm version Blazar installs when the user names none. Pinned, not
/// mlx-lm version used only as the offline fallback (installs default
/// to `PyPI` latest; the boot smoke verifies the resolved release, and
/// the manifest help-probe carries the per-release flag surface).
pub const MLX_LM_FALLBACK_VERSION: &str = "0.32.0";

/// The CUDA backend rides `mlx` as a pip extra with its own version -
/// pinned independently of mlx-lm so a backend float can never move
/// under a verified mlx-lm pin (dual-pin discipline).
/// cuda12, not cuda13: the bundled CUDA 12 userspace rides every driver
/// since the 525 series (broadest install base), while cuda13 requires
/// the newest drivers only - one tested surface.
pub const MLX_CUDA_PIN: &str = "mlx[cuda12]==0.32.2";

/// Kernel-compile surface for the CUDA backend: mlx JIT-compiles its
/// kernels at generation time through NVRTC and needs a `CUDA_HOME` with
/// modern headers - the fp8 types mlx 0.32's bundled CCCL requires postdate
/// CUDA 12.0, and system toolkits are routinely that old. These two wheels
/// carry the headers (runtime + crt) and libs the engine-local cuda-home
/// farm links; live-proven 2026-10-02 on a CUDA 12.0 apt box.
pub const MLX_CUDA_RUNTIME_PIN: &str = "nvidia-cuda-runtime-cu12";
pub const MLX_CUDA_NVCC_PIN: &str = "nvidia-cuda-nvcc-cu12";

/// The venv is lighter than sglang's (no torch wheel): the CUDA libs
/// (cublas/cudnn/nccl via the extra) plus transformers land between 2
/// and 3 GiB. Refuse installs with less than 4 GiB free instead of
/// dying mid-extract with a half-populated site-packages.
const MLX_MIN_DISK_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// pip resolving the CUDA wheels can be slow on modest links; the cap
/// exists to catch a wedged download, not to rush a legitimate one.
const MLX_INSTALL_TIMEOUT_SECS: u64 = 1200;

/// First import of the mlx backend chain initializes CUDA context and
/// loads extension modules - the cap catches a wedged import, not a
/// slow one.
const MLX_BOOT_SMOKE_TIMEOUT_SECS: u64 = 300;

/// Where mlx runs its kernels on this host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlxBackend {
    /// Apple Silicon: the Metal backend rides the plain `mlx` wheel.
    Metal,
    /// Linux: the CUDA backend rides `mlx` as the `cuda12` extra.
    Cuda,
}

/// What an install has to pin for one host. It is decided before anything
/// is downloaded, so every refusal names its reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallPlan {
    pub backend: MlxBackend,
    /// pip requirements added next to `mlx-lm==<version>`.
    pub extra_pins: Vec<String>,
}

/// The Linux wheels are `manylinux_2_35`, which means glibc 2.35.
const MLX_WHEEL_GLIBC_FLOOR: (u32, u32) = (2, 35);

const GLIBC_QUERY_TIMEOUT_SECS: u64 = 10;

const HOST_LIBC: &str = if cfg!(target_env = "musl") {
    "musl"
} else {
    "gnu"
};

/// The install for one host: Apple Silicon gets the Metal backend, Linux
/// with an NVIDIA card the CUDA backend. Every other host is refused with
/// its reason, because each would fail later inside pip with a message
/// that names none of it: an Intel Mac has no wheel at all, and a musl or
/// old glibc host cannot load the Linux wheels.
pub fn install_plan(
    os: &str,
    arch: &str,
    libc_env: &str,
    host_glibc: Option<(u32, u32)>,
) -> Result<InstallPlan> {
    match (os, arch) {
        ("macos", "aarch64") => Ok(InstallPlan {
            backend: MlxBackend::Metal,
            extra_pins: Vec::new(),
        }),
        ("macos", other) => anyhow::bail!(
            "mlx publishes Apple Silicon wheels only (arm64, macOS 14 or newer); \
             this Mac runs {other}. Use the llama.cpp lane instead: \
             blazar engine install --kind llamacpp"
        ),
        ("linux", "x86_64" | "aarch64") => {
            if libc_env == "musl" {
                anyhow::bail!(
                    "mlx ships glibc (manylinux) wheels for Linux; a musl host such as \
                     Alpine has no wheel to install"
                );
            }
            if let Some((major, minor)) = host_glibc
                && (major, minor) < MLX_WHEEL_GLIBC_FLOOR
            {
                anyhow::bail!(
                    "mlx Linux wheels need glibc {}.{} or newer; this host runs {major}.{minor}",
                    MLX_WHEEL_GLIBC_FLOOR.0,
                    MLX_WHEEL_GLIBC_FLOOR.1
                );
            }
            Ok(cuda_plan())
        }
        ("linux", other) => anyhow::bail!(
            "mlx publishes Linux wheels for x86_64 and aarch64 only; this host runs {other}"
        ),
        (other, _) => anyhow::bail!(
            "the mlx lane ships a POSIX `mlx-server` shim, so Blazar runs it on macOS \
             and Linux only; on {other} use WSL2 or another engine kind"
        ),
    }
}

fn cuda_plan() -> InstallPlan {
    InstallPlan {
        backend: MlxBackend::Cuda,
        extra_pins: [MLX_CUDA_PIN, MLX_CUDA_RUNTIME_PIN, MLX_CUDA_NVCC_PIN]
            .iter()
            .map(|pin| (*pin).to_string())
            .collect(),
    }
}

/// The glibc this host runs, read from `getconf`. `None` when the tool is
/// missing or prints something else. `None` does not refuse an install:
/// the floor is a courtesy check and pip stays the judge.
fn glibc_version() -> Option<(u32, u32)> {
    let output = crate::probe::probe_output(
        std::process::Command::new("getconf").arg("GNU_LIBC_VERSION"),
        GLIBC_QUERY_TIMEOUT_SECS,
    )?;
    parse_glibc_version(&String::from_utf8_lossy(&output.stdout))
}

/// Parses the `getconf GNU_LIBC_VERSION` line, `glibc 2.39`.
fn parse_glibc_version(text: &str) -> Option<(u32, u32)> {
    let version = text.trim().strip_prefix("glibc ")?.trim();
    let (major, minor) = version.split_once('.')?;
    Some((major.trim().parse().ok()?, minor.trim().parse().ok()?))
}

/// Two independent signals, because either can be missing on a healthy
/// box: the `nvidia-smi` tool and the driver library it is a client of.
fn nvidia_gpu_visible() -> bool {
    !crate::probe::nvidia_smi_gpus().is_empty() || crate::gpu_census::nvml_total_mib().is_some()
}

/// Create the venv + install mlx-lm + backend + write the shim inside
/// `dir`. `dir` must exist and be empty; on failure the caller removes
/// `dir` (F88 discipline lives with the `EngineManager` entry point).
pub async fn install_into(dir: &Path, version: &str) -> Result<PathBuf> {
    // The glibc floor is only readable where glibc is the host libc; every
    // other host passes `None` and the plan decides without it.
    let host_glibc = if cfg!(all(target_os = "linux", target_env = "gnu")) {
        glibc_version()
    } else {
        None
    };
    let plan = install_plan(
        std::env::consts::OS,
        std::env::consts::ARCH,
        HOST_LIBC,
        host_glibc,
    )?;
    // Linux requires an NVIDIA GPU: the mlx CUDA backend is the only
    // supported accelerator on this OS, and a CPU-only mlx install would
    // silently serve at toy speed next to the CUDA lanes.
    if plan.backend == MlxBackend::Cuda && !nvidia_gpu_visible() {
        anyhow::bail!(
            "the mlx lane on Linux uses the CUDA backend and needs an NVIDIA GPU; \
             none is visible through the NVIDIA driver library (NVML) or nvidia-smi. \
             On Apple Silicon machines the lane installs natively; on this host \
             pick another engine kind"
        );
    }
    let venv = install_platform(dir, version, &plan.extra_pins).await?;
    // The farm must exist before the row can ever serve: JIT kernel
    // compiles happen at generation time, so a missing farm is an
    // instant broken lane, not a degraded one. The helper itself exists
    // on Linux only (CUDA farm), so the call is gated to match; the
    // backend is chosen at runtime.
    #[cfg(target_os = "linux")]
    if plan.backend == MlxBackend::Cuda {
        link_cuda_home(dir)?;
    }
    Ok(venv)
}

/// Shared venv build for both supported platforms: `extra_pins` carries
/// the backend pin(s) - empty on macOS (platform marker resolves), the
/// CUDA extra on Linux.
async fn install_platform(dir: &Path, version: &str, extra_pins: &[String]) -> Result<PathBuf> {
    let avail = disk_avail_bytes(dir).context("disk preflight")?;
    if avail < MLX_MIN_DISK_BYTES {
        #[allow(clippy::cast_precision_loss)]
        let avail_gib = avail as f64 / f64::from(1024 * 1024 * 1024);
        anyhow::bail!(
            "mlx needs >= {} GiB free for its venv (mlx backend + transformers); \
             {} has only {avail_gib:.1} GiB avail",
            MLX_MIN_DISK_BYTES / (1024 * 1024 * 1024),
            dir.display(),
        );
    }

    let venv = dir.join("venv");
    let venv_python = venv.join("bin").join("python");
    let mut pins = vec![format!("mlx-lm=={version}")];
    pins.extend(extra_pins.iter().cloned());
    if uv_available() {
        run_streaming(
            tokio::process::Command::new("uv")
                .arg("venv")
                .arg("--python")
                .arg("3.12")
                .arg(&venv),
            "uv venv",
            MLX_INSTALL_TIMEOUT_SECS,
        )
        .await?;
        let mut cmd = tokio::process::Command::new("uv");
        cmd.arg("pip")
            .arg("install")
            .arg("--python")
            .arg(&venv_python);
        for pin in &pins {
            cmd.arg(pin);
        }
        run_streaming(&mut cmd, "uv pip install mlx-lm", MLX_INSTALL_TIMEOUT_SECS).await?;
    } else {
        run_streaming(
            tokio::process::Command::new("python3")
                .arg("-m")
                .arg("venv")
                .arg(&venv),
            "python3 -m venv",
            MLX_INSTALL_TIMEOUT_SECS,
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
            MLX_INSTALL_TIMEOUT_SECS,
        )
        .await?;
        let mut cmd = tokio::process::Command::new(&venv_python);
        cmd.arg("-m").arg("pip").arg("install");
        for pin in &pins {
            cmd.arg(pin);
        }
        run_streaming(&mut cmd, "pip install mlx-lm", MLX_INSTALL_TIMEOUT_SECS).await?;
    }

    // Shim: absolute venv python so the script works from any cwd; argv
    // passthrough keeps probe (--) and spawn argv identical in shape.
    // venv/bin goes first on PATH so the venv's own console scripts win
    // over anything ambient.
    let shim = dir.join("mlx-server");
    let script = format!(
        "#!/bin/sh\nexport PATH=\"{}:$PATH\"\nexec \"{}\" -m mlx_lm.server \"$@\"\n",
        venv.join("bin").display(),
        venv_python.display()
    );
    std::fs::write(&shim, script).context("write mlx-server shim")?;
    make_executable(&shim)?;

    // Boot smoke: the `--version` probes (install register, doctor,
    // supervisor rollback gate) exit before the deep imports, so a venv
    // whose boot-critical import chain dies (CUDA driver mismatch,
    // broken extension) still installed and activated cleanly. Import
    // the real server module here with the venv python; failure bubbles
    // out of install_into and F88 rolls the dir back - an unbootable
    // venv must never become an activatable row.
    let mut smoke = tokio::process::Command::new(&venv_python);
    smoke.arg("-c").arg("import mlx_lm.server");
    run_streaming(
        &mut smoke,
        "boot-smoke import mlx_lm.server",
        MLX_BOOT_SMOKE_TIMEOUT_SECS,
    )
    .await
    .context("boot-smoke failed: the venv cannot import the mlx-lm server - rerun manually to inspect: <engine>/venv/bin/python -c 'import mlx_lm.server'")?;
    Ok(shim)
}

/// Latest mlx-lm release on `PyPI`, for the doctor currency row.
pub async fn pypi_latest_mlx_lm() -> Result<String> {
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
            let resp = match http.get("https://pypi.org/pypi/mlx-lm/json").send().await {
                Ok(resp) => resp,
                Err(e) => return Attempt::Retry(anyhow!("pypi mlx-lm query: {e}")),
            };
            match resp.status() {
                reqwest::StatusCode::OK => {}
                other if other.is_server_error() => {
                    return Attempt::Retry(anyhow!("pypi mlx-lm status {other}"));
                }
                other => {
                    return Attempt::Done(Err(anyhow!("pypi mlx-lm status {other}")));
                }
            }
            match resp.json::<PypiResp>().await {
                Ok(resp) => Attempt::Done(Ok(resp.info.version)),
                // A connection reset mid-body truncates the 200 -
                // decode failures are transport-class, worth one more
                // attempt.
                Err(e) => Attempt::Retry(anyhow!("pypi mlx-lm json: {e}")),
            }
        }
    })
    .await
}

/// Same dotted-3 parse as the sglang lane: tags share the `<prefix>-X.Y.Z`
/// shape and `PyPI` versions are plain `X.Y.Z` (pre-release suffixes fall
/// back to the leading core).
#[must_use]
pub fn version_tuple(v: &str) -> Option<(u64, u64, u64)> {
    super::sglang_install::version_tuple(v)
}

/// Build the engine-local `cuda-home` the mlx CUDA backend compiles
/// against: `include/` merges the `cuda_runtime` and `cuda_nvcc` (crt)
/// headers, `lib64/` merges the `cuda_runtime` and `cuda_nvrtc` libs, and
/// `bin`/`nvvm` symlink the nvcc wheel's ptxas + nvvm tree. No nvcc
/// binary ships in these wheels - mlx JIT-compiles through NVRTC, which
/// the farm's lib64 satisfies. Keeps the lane self-contained on hosts
/// whose system CUDA toolkit is older than the headers mlx requires.
/// Idempotent: an existing farm is removed and rebuilt.
#[cfg(target_os = "linux")]
pub fn link_cuda_home(engine_dir: &Path) -> Result<PathBuf> {
    let venv_lib = engine_dir.join("venv").join("lib");
    let python_dir = std::fs::read_dir(&venv_lib)
        .map_err(|e| anyhow::anyhow!("mlx venv missing lib/ ({}): {e}", venv_lib.display()))?
        .filter_map(std::result::Result::ok)
        .find(|e| e.file_name().to_string_lossy().starts_with("python3"))
        .map(|e| e.path())
        .ok_or_else(|| {
            anyhow::anyhow!("mlx venv has no python3.* dir under {}", venv_lib.display())
        })?;
    let nvidia = python_dir.join("site-packages").join("nvidia");
    let farm = engine_dir.join("cuda-home");

    // rm-first: a stale farm from an older pin set would shadow the
    // headers pip just laid down.
    if farm.exists() {
        std::fs::remove_dir_all(&farm)?;
    }
    let include = farm.join("include");
    let lib64 = farm.join("lib64");
    std::fs::create_dir_all(&include)?;
    std::fs::create_dir_all(&lib64)?;

    let link_in = |dst: &Path, src_dir: &Path| -> Result<()> {
        for entry in std::fs::read_dir(src_dir)? {
            let entry = entry?;
            let src = entry.path();
            let name = entry.file_name();
            let dst_entry = dst.join(&name);
            if dst_entry.exists() {
                continue;
            }
            std::os::unix::fs::symlink(&src, &dst_entry)?;
        }
        Ok(())
    };

    link_in(&include, &nvidia.join("cuda_runtime").join("include"))?;
    link_in(&include, &nvidia.join("cuda_nvcc").join("include"))?;
    link_in(&lib64, &nvidia.join("cuda_runtime").join("lib"))?;
    link_in(&lib64, &nvidia.join("cuda_nvrtc").join("lib"))?;
    std::os::unix::fs::symlink(nvidia.join("cuda_nvcc").join("bin"), farm.join("bin"))?;
    std::os::unix::fs::symlink(nvidia.join("cuda_nvcc").join("nvvm"), farm.join("nvvm"))?;
    Ok(farm)
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    #![allow(non_snake_case)]
    use super::*;

    fn touch(p: &Path) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, b"x").unwrap();
    }

    fn fake_venv(root: &Path) {
        let nvidia = root.join("venv/lib/python3.12/site-packages/nvidia");
        touch(&nvidia.join("cuda_runtime/include/cuda_fp8.h"));
        touch(&nvidia.join("cuda_runtime/lib/libcudart.so.12"));
        touch(&nvidia.join("cuda_nvcc/include/crt/host_defines.h"));
        touch(&nvidia.join("cuda_nvcc/bin/ptxas"));
        touch(&nvidia.join("cuda_nvcc/nvvm/lib/libnvvm.so"));
        touch(&nvidia.join("cuda_nvrtc/lib/libnvrtc.so.12"));
    }

    #[tokio::test]
    async fn unit__link_cuda_home__builds_the_merged_farm_and_is_idempotent() {
        let root = std::env::temp_dir().join(format!("mlx-farm-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        fake_venv(&root);
        let farm = link_cuda_home(&root).unwrap();
        assert!(farm.join("include/cuda_fp8.h").exists());
        assert!(farm.join("include/crt/host_defines.h").exists());
        assert!(farm.join("lib64/libcudart.so.12").exists());
        assert!(farm.join("lib64/libnvrtc.so.12").exists());
        assert!(farm.join("bin/ptxas").exists());
        assert!(farm.join("nvvm/lib/libnvvm.so").exists());
        // idempotent rebuild keeps the same shape
        link_cuda_home(&root).unwrap();
        assert!(farm.join("include/cuda_fp8.h").exists());
        let _ = std::fs::remove_dir_all(&root);
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod plan_tests {
    use anyhow::Result;

    use super::{
        HOST_LIBC, InstallPlan, MLX_CUDA_NVCC_PIN, MLX_CUDA_PIN, MLX_CUDA_RUNTIME_PIN, MlxBackend,
        install_plan, parse_glibc_version,
    };

    fn refusal(plan: Result<InstallPlan>) -> String {
        plan.expect_err("this host is refused").to_string()
    }

    #[test]
    fn unit__install_plan__apple_silicon_installs_metal_with_no_extra_pin() {
        assert_eq!(
            install_plan("macos", "aarch64", HOST_LIBC, None).unwrap(),
            InstallPlan {
                backend: MlxBackend::Metal,
                extra_pins: Vec::new(),
            }
        );
    }

    #[test]
    fn unit__install_plan__intel_mac_is_refused_and_pointed_at_llama_cpp() {
        let message = refusal(install_plan("macos", "x86_64", HOST_LIBC, None));
        assert!(message.contains("Apple Silicon"), "{message}");
        assert!(
            message.contains("blazar engine install --kind llamacpp"),
            "{message}"
        );
    }

    #[test]
    fn unit__install_plan__linux_installs_cuda_with_the_three_backend_pins() {
        assert_eq!(
            install_plan("linux", "x86_64", "gnu", None).unwrap(),
            InstallPlan {
                backend: MlxBackend::Cuda,
                extra_pins: vec![
                    MLX_CUDA_PIN.to_string(),
                    MLX_CUDA_RUNTIME_PIN.to_string(),
                    MLX_CUDA_NVCC_PIN.to_string(),
                ],
            }
        );
        // The floor is inclusive and the aarch64 wheels share it.
        assert_eq!(
            install_plan("linux", "aarch64", "gnu", Some((2, 35)))
                .unwrap()
                .backend,
            MlxBackend::Cuda
        );
    }

    #[test]
    fn unit__install_plan__old_glibc_is_refused_with_both_versions_named() {
        let message = refusal(install_plan("linux", "x86_64", "gnu", Some((2, 31))));
        assert!(message.contains("glibc 2.35"), "{message}");
        assert!(message.contains("2.31"), "{message}");
    }

    #[test]
    fn unit__install_plan__musl_host_is_refused_whatever_version_it_reports() {
        let message = refusal(install_plan("linux", "x86_64", "musl", Some((2, 35))));
        assert!(message.contains("musl"), "{message}");
    }

    #[test]
    fn unit__install_plan__other_linux_cpu_is_refused() {
        let message = refusal(install_plan("linux", "riscv64", "gnu", None));
        assert!(message.contains("x86_64 and aarch64"), "{message}");
    }

    #[test]
    fn unit__install_plan__windows_is_refused_for_the_shim_not_for_upstream() {
        let message = refusal(install_plan("windows", "x86_64", "gnu", None));
        assert!(message.contains("WSL2"), "{message}");
        assert!(message.contains("shim"), "{message}");
    }

    #[test]
    fn unit__parse_glibc_version__reads_the_getconf_line() {
        assert_eq!(parse_glibc_version("glibc 2.39\n"), Some((2, 39)));
        assert_eq!(parse_glibc_version("glibc 2.31"), Some((2, 31)));
        assert_eq!(parse_glibc_version("not a version"), None);
        assert_eq!(parse_glibc_version(""), None);
    }

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    #[test]
    fn unit__glibc_version__reads_this_host() {
        let (major, minor) =
            super::glibc_version().expect("getconf GNU_LIBC_VERSION answers on a glibc host");
        assert!(major >= 2, "read glibc {major}.{minor}");
    }
}
