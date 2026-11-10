//! Kokoro TTS lane install: a pip-managed venv per engine tag plus a
//! stdlib HTTP server wrapping `kokoro_onnx.Kokoro`, launched through
//! an executable shim. Layout (probed by `probe_kokoro`, found by
//! `register_engine_inner`):
//!
//! ```text
//! engines/kokoro-1.0/
//! ├── venv/                 # uv-created, kokoro-onnx==0.6.1 installed
//! │   └── bin/python
//! ├── kokoro.onnx           # model weights (from thewh1teagle/kokoro-onnx releases)
//! ├── voices.bin            # voice style packs
//! ├── kokoro_server.py      # stdlib HTTP server: /health, /voices, POST /tts
//! └── kokoro-server         # shim: exec venv/bin/python kokoro_server.py "$@"
//! ```
//!
//! Why a resident HTTP child (unlike piper's one-shot CLI): the 325 MB
//! ONNX model reloads per request as a CLI; resident, the gateway's
//! `/v1/audio/speech` pays the load once and the idle reaper folds the
//! child when TTS traffic stops (the whisper lane's exact lifecycle).
//!
//! Cross-platform by construction: kokoro-onnx is pure pip (CPU
//! onnxruntime, espeak-ng data bundled via espeakng-loader) - no CUDA
//! gate, no system espeak dependency, so no OS refusal here.
//!
//! Venv-per-tag for the same lifecycle reason as the sglang lane:
//! `prune` is `remove_dir_all`, and a version pin must be exact.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

use super::sglang_install::{disk_avail_bytes, run_streaming, uv_available};

/// Shim name: the install lane writes it, the register finder and the
/// lazy runtime match it - the shared contract (litert precedent).
pub const SHIM_NAME: &str = "kokoro-server";

/// Lane-carried server script, single-sourced from this file. Spawn-time
/// refresh compares the lane copy against this payload so binary upgrades
/// retrofit logic fixes into existing lanes (lanes are self-contained,
/// deletable units - they must carry their own script, but never a stale one).
pub const KOKORO_SERVER_PY: &str = include_str!("kokoro_server.py");

/// Rewrite the lane's `kokoro_server.py` when it drifted from the embedded
/// payload. Returns `Ok(true)` when a refresh happened, `Ok(false)` when the
/// lane copy already matched. Errors surface to the caller (spawn path
/// warns and proceeds on the stale-but-functional script).
pub fn refresh_lane_script(lane_dir: &std::path::Path) -> std::io::Result<bool> {
    let server_py = lane_dir.join("kokoro_server.py");
    let current = std::fs::read_to_string(&server_py).unwrap_or_default();
    if current == KOKORO_SERVER_PY {
        return Ok(false);
    }
    std::fs::write(&server_py, KOKORO_SERVER_PY)?;
    Ok(true)
}

/// pip pin for the inference package (verified against the 0.6.1 sdist:
/// `Kokoro(model, voices)`, `create(text, voice, speed, lang) -> (samples,
/// 24000)`). Bump deliberately - the server script's API surface is this
/// version's.
pub const KOKORO_ONNX_VERSION: &str = "0.6.1";

/// Model line installed when the user names none. v1.0 is the canonical
/// English line the upstream README links (release `model-files-v1.1`).
pub const KOKORO_DEFAULT_MODEL: &str = "1.0";

/// Lane-local asset names (the install contract; release-side names
/// carry the version, see [`model_asset_names`]).
pub const MODEL_FILE: &str = "kokoro.onnx";
pub const VOICES_FILE: &str = "voices.bin";

/// Model + voices + venv (onnxruntime ~15 MB wheel, phonemizer, numpy)
/// land under 1 GiB; 2 GiB free catches the wedged-disk case, not the
/// legitimate one.
const KOKORO_MIN_DISK_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// pip wheels are small; the cap catches a wedged download.
const KOKORO_INSTALL_TIMEOUT_SECS: u64 = 600;

/// Boot smoke budget: onnxruntime init + the 325 MB model load +
/// espeak-ng first phonemize can take a minute on slow disks - the cap
/// catches a wedge, not a slow load.
pub const KOKORO_BOOT_SMOKE_TIMEOUT_SECS: u64 = 240;

/// Release-side asset names for a model line: `kokoro-v1.0.onnx` /
/// `voices-v1.0.bin` in the thewh1teagle/kokoro-onnx releases.
#[must_use]
pub fn model_asset_names(model_version: &str) -> (String, String) {
    (
        format!("kokoro-v{model_version}.onnx"),
        format!("voices-v{model_version}.bin"),
    )
}

/// Create the venv + write the server script + shim inside `dir`. The
/// model assets (`kokoro.onnx`, `voices.bin`) must already be
/// downloaded there - the `EngineManager` entry point owns GH asset
/// fetch + rollback (F88 discipline). On failure the caller removes
/// `dir`.
pub async fn install_into(dir: &Path) -> Result<PathBuf> {
    let model = dir.join(MODEL_FILE);
    let voices = dir.join(VOICES_FILE);
    if !model.is_file() || !voices.is_file() {
        anyhow::bail!(
            "kokoro lane preflight: {} and {} must already be downloaded into {} \
             (the install entry point owns the asset fetch)",
            MODEL_FILE,
            VOICES_FILE,
            dir.display()
        );
    }
    let avail = disk_avail_bytes(dir).context("disk preflight")?;
    if avail < KOKORO_MIN_DISK_BYTES {
        #[allow(clippy::cast_precision_loss)]
        let avail_gib = avail as f64 / f64::from(1024 * 1024 * 1024);
        anyhow::bail!(
            "kokoro needs >= {} GiB free for its venv; {} has only {avail_gib:.1} GiB avail",
            KOKORO_MIN_DISK_BYTES / (1024 * 1024 * 1024),
            dir.display(),
        );
    }

    let venv = dir.join("venv");
    let venv_python = venv.join("bin").join("python");
    let use_uv = uv_available();
    if use_uv {
        run_streaming(
            tokio::process::Command::new("uv")
                .arg("venv")
                .arg("--python")
                .arg("3.12")
                .arg(&venv),
            "uv venv",
            KOKORO_INSTALL_TIMEOUT_SECS,
        )
        .await?;
        run_streaming(
            tokio::process::Command::new("uv")
                .arg("pip")
                .arg("install")
                .arg("--python")
                .arg(&venv_python)
                .arg(format!("kokoro-onnx=={KOKORO_ONNX_VERSION}")),
            "uv pip install kokoro-onnx",
            KOKORO_INSTALL_TIMEOUT_SECS,
        )
        .await?;
    } else {
        run_streaming(
            tokio::process::Command::new("python3")
                .arg("-m")
                .arg("venv")
                .arg(&venv),
            "python3 -m venv",
            KOKORO_INSTALL_TIMEOUT_SECS,
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
            KOKORO_INSTALL_TIMEOUT_SECS,
        )
        .await?;
        run_streaming(
            tokio::process::Command::new(&venv_python)
                .arg("-m")
                .arg("pip")
                .arg("install")
                .arg(format!("kokoro-onnx=={KOKORO_ONNX_VERSION}")),
            "pip install kokoro-onnx",
            KOKORO_INSTALL_TIMEOUT_SECS,
        )
        .await?;
    }

    let server_py = dir.join("kokoro_server.py");
    std::fs::write(&server_py, KOKORO_SERVER_PY)
        .with_context(|| format!("write {}", server_py.display()))?;

    // Shim: resolve the server script next to the shim itself so the
    // lane works from any cwd; absolute venv python per the sglang
    // shim contract.
    let shim = dir.join(SHIM_NAME);
    let script = format!(
        "#!/bin/sh\nSCRIPT_DIR=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\nexec \
         \"{}\" \"$SCRIPT_DIR/kokoro_server.py\" \"$@\"\n",
        venv_python.display()
    );
    std::fs::write(&shim, script).with_context(|| format!("write {}", shim.display()))?;
    super::make_executable(&shim);

    boot_smoke(&shim).await?;
    Ok(shim)
}

/// Reserve a free localhost port (bind :0, read it, drop the listener -
/// the whisper lane's smoke pattern; the race window is one spawn away).
fn reserve_port() -> Result<u16> {
    let l = std::net::TcpListener::bind("127.0.0.1:0").context("bind :0 for smoke port")?;
    Ok(l.local_addr().context("smoke port addr")?.port())
}

/// Boot the shim on an ephemeral port, wait for `/health` (model
/// resident), synthesize one utterance, insist on a RIFF/WAV body, and
/// tear the child down. An unbootable or unsynthesizing lane must never
/// become an activatable row.
async fn boot_smoke(shim: &Path) -> Result<()> {
    let port = reserve_port()?;
    let mut child = tokio::process::Command::new(shim)
        .arg("--port")
        .arg(port.to_string())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("spawn {} --port {port} (boot smoke)", shim.display()))?;

    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(
            KOKORO_BOOT_SMOKE_TIMEOUT_SECS,
        ))
        .build()
        .context("smoke http client")?;
    let base = format!("http://127.0.0.1:{port}");

    // Health poll: model load is the slow part; catch early child death
    // so a crashed spawn fails in milliseconds, not at the deadline.
    let deadline =
        std::time::Instant::now() + std::time::Duration::from_secs(KOKORO_BOOT_SMOKE_TIMEOUT_SECS);
    loop {
        if let Ok(Some(status)) = child.try_wait() {
            anyhow::bail!("kokoro boot smoke: server exited early with {status}");
        }
        if std::time::Instant::now() >= deadline {
            anyhow::bail!(
                "kokoro boot smoke: /health not ready within \
                 {KOKORO_BOOT_SMOKE_TIMEOUT_SECS}s (model load wedged?)"
            );
        }
        if let Ok(resp) = http.get(format!("{base}/health")).send().await
            && resp.status().is_success()
        {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }

    let tts = http
        .post(format!("{base}/tts"))
        .json(&serde_json::json!({
            "input": "Blazar boot smoke.",
            "voice": "af_heart",
        }))
        .send()
        .await
        .context("boot smoke POST /tts")?;
    if !tts.status().is_success() {
        anyhow::bail!("boot smoke POST /tts returned {}", tts.status());
    }
    let wav = tts.bytes().await.context("boot smoke read WAV body")?;
    if !wav.starts_with(b"RIFF") {
        anyhow::bail!(
            "boot smoke /tts body is not RIFF/WAV (first bytes: {:?})",
            &wav[..wav.len().min(8)]
        );
    }

    let _ = child.kill().await;
    Ok(())
}

#[cfg(test)]
mod refresh_tests {
    use super::*;

    #[test]
    #[allow(non_snake_case)]
    fn unit__refresh_lane_script__rewrites_drift_and_skips_match() {
        let dir = std::env::temp_dir().join(format!("blz-kokoro-refresh-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Drifted (stale install-time) copy -> rewritten to the payload.
        std::fs::write(dir.join("kokoro_server.py"), "# stale draft").unwrap();
        assert!(refresh_lane_script(&dir).unwrap());
        assert_eq!(
            std::fs::read_to_string(dir.join("kokoro_server.py")).unwrap(),
            KOKORO_SERVER_PY
        );
        // Matching copy -> no rewrite.
        assert!(!refresh_lane_script(&dir).unwrap());
        std::fs::remove_dir_all(&dir).ok();
    }
}
