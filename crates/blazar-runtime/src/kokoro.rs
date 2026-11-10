//! Kokoro TTS lane runtime: lane resolution from the engines table plus
//! a lazily-spawned resident HTTP child (the whisper lane's lifecycle,
//! minus model swaps - one lane row is one model). The gateway's
//! `/v1/audio/speech` handler drives this; the idle reaper folds the
//! child when TTS traffic stops.
//!
//! Security posture mirrors whisper: the lane server has no auth - it
//! binds loopback only, on an OS-assigned ephemeral port, and dies with
//! its parent (PDEATHSIG).

use anyhow::{Context, Result, anyhow};
use std::path::{Path, PathBuf};

use blazar_core::BlazarDirs;

// One shared ceiling for every TTS lane's input guard (the piper lane
// owns the constant; long documents split identically on both lanes).
pub use crate::piper::MAX_INPUT_CHARS;

/// Per-synthesis budget: CPU synthesis of a max-length input can take
/// tens of seconds; the cap catches a wedged child, not a slow one.
const SYNTH_TIMEOUT_SECS: u64 = 120;

/// Model-load budget at spawn (onnxruntime init + 325 MB weights).
pub const READY_TIMEOUT_SECS: u64 = 240;

/// The engines-table lane row: the active kokoro row first, else the
/// newest installed (the whisper pick without the pin concept - lane
/// choice rides `blazar engine install <ver> --kind kokoro`). `None`
/// when no kokoro row exists or the store cannot be read (warn, never
/// mask) - callers teach the install command.
pub fn lane_row(dirs: &BlazarDirs) -> Option<blazar_core::store::EngineRow> {
    let store = match blazar_core::Store::open(dirs) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("kokoro lane lookup could not open the store: {e:#}");
            return None;
        }
    };
    lane_row_at(&store)
}

/// Store-bound lane lookup - the gateway's cached `with_store` handle
/// pays no fresh SQLite open per speech request (the `dirs` variant is
/// for CLI callers that own no long-lived handle).
#[must_use]
pub fn lane_row_at(store: &blazar_core::Store) -> Option<blazar_core::store::EngineRow> {
    let rows = match store.list_engines() {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("kokoro lane listing failed: {e:#}");
            return None;
        }
    };
    // `list_engines` orders newest-installed first; the active flag
    // still wins, `installed_at` breaks ties toward the NEWEST row.
    rows.into_iter()
        .filter(|r| r.kind == blazar_core::engine_kind::EngineKind::Kokoro)
        .max_by_key(|r| (i64::from(r.active), r.installed_at))
}

/// Shim path for a known lane row (the manifest's probed path when it
/// still exists, else re-derived from the tag dir).
#[must_use]
pub fn shim_for_row(row: &blazar_core::store::EngineRow, dirs: &BlazarDirs) -> Option<PathBuf> {
    let dir = dirs.engines_dir().join(&row.tag);
    if let Ok(mut m) = serde_json::from_str::<crate::engine::manifest::Manifest>(&row.manifest) {
        m.anchor_server_path(&dirs.data_dir);
        let probed = PathBuf::from(&m.server_path);
        if probed.is_file() {
            return Some(probed);
        }
    }
    crate::engine::find_engine_binary(&dir, &[crate::engine::kokoro_install::SHIM_NAME]).ok()
}

/// The lane's shim: the manifest's probed path when it still exists,
/// else re-derived from the tag dir (a relocated data dir must not
/// brick the lane - register's recovery shape).
#[must_use]
pub fn lane_shim(dirs: &BlazarDirs) -> Option<PathBuf> {
    lane_row(dirs).and_then(|row| shim_for_row(&row, dirs))
}

/// Whether a kokoro lane is installed (drives install teaching + the
/// voices endpoint's lane section).
#[must_use]
pub fn lane_installed(dirs: &BlazarDirs) -> bool {
    lane_shim(dirs).is_some()
}

/// Kokoro voice-name grammar: `af_sarah`, `am_michael`,
/// `bf_nimble`-shaped - two lowercase letters, `m`/`f`, underscore,
/// then lowercase alphanumerics (underscore-joined). Piper ids
/// (`en_US-amy-medium`: uppercase + dashes) never match, so the
/// dispatch can route by shape alone.
#[must_use]
pub fn looks_like_voice(voice: &str) -> bool {
    // Kokoro voice ids are `<lang><gender>_<name>` (e.g. `af_sarah`,
    // `am_santa_claus`, `bf_nimble`): two lowercase language chars,
    // `m`/`f` gender, underscore separator, then a lowercase
    // alphanumeric name (underscores allowed inside). Piper's
    // `en_US-amy-medium` shape (uppercase + dash) never matches.
    let bytes = voice.as_bytes();
    bytes.len() >= 4
        && bytes[0].is_ascii_lowercase()
        && (bytes[1] == b'm' || bytes[1] == b'f')
        && bytes[2] == b'_'
        && bytes[3..]
            .iter()
            .all(|&b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
}

/// Validate + synthesize one utterance to WAV bytes. Input guarding
/// mirrors the piper lane (trim, char ceiling) so both TTS lanes
/// reject the same garbage identically.
pub fn validate_input(input: &str) -> Result<&str> {
    let text = input.trim();
    if text.is_empty() {
        anyhow::bail!("input must be non-empty text");
    }
    if text.chars().count() > MAX_INPUT_CHARS {
        anyhow::bail!(
            "input is {} chars (max {MAX_INPUT_CHARS}) - split long documents",
            text.chars().count()
        );
    }
    Ok(text)
}

/// One lazily-spawned kokoro-server child.
struct KokoroChild {
    child: tokio::process::Child,
    port: u16,
    /// Last instant the lane served (or spawned) - the idle reaper's
    /// clock, refreshed on every hit.
    last_used: tokio::time::Instant,
}

/// The single lane slot; teardown is `shutdown()` (H19).
pub struct KokoroRuntime {
    child: tokio::sync::Mutex<Option<KokoroChild>>,
    /// One client for every lane call - per-request construction pays
    /// TLS-provider init + a fresh pool on the TTS hot path.
    http: reqwest::Client,
}

/// Pull the voice list out of the lane's `/voices` body (shared by the
/// spawning and resident-only variants).
fn parse_voices(list: &serde_json::Value) -> Vec<String> {
    list.get("voices")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

impl KokoroRuntime {
    #[must_use]
    pub fn new() -> Self {
        blazar_core::tls::ensure_tls_provider();
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(SYNTH_TIMEOUT_SECS))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            child: tokio::sync::Mutex::new(None),
            http,
        }
    }

    /// Port of a live lane server; spawns the shim on first use. Held
    /// across the spawn so concurrent requests share one child.
    async fn ensure(&self, shim: &Path) -> Result<u16> {
        let mut slot = self.child.lock().await;
        if let Some(live) = slot.as_mut() {
            if tcp_alive(live.port).await {
                live.last_used = tokio::time::Instant::now();
                return Ok(live.port);
            }
            // Dead child: reap before respawning.
            let _ = slot.take();
        }
        let port = ephemeral_port()?;
        let mut std_cmd = std::process::Command::new(shim);
        std_cmd
            .arg("--port")
            .arg(port.to_string())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        #[cfg(unix)]
        {
            // Kernel lifetime tie: the lane server must not outlive the
            // gateway that spawned it.
            crate::probe::parent_death_tie(&mut std_cmd);
        }
        let mut cmd = tokio::process::Command::from(std_cmd);
        cmd.kill_on_drop(true);
        let mut child = cmd
            .spawn()
            .with_context(|| format!("spawn {} --port {port}", shim.display()))?;
        let deadline =
            tokio::time::Instant::now() + std::time::Duration::from_secs(READY_TIMEOUT_SECS);
        loop {
            if tcp_alive(port).await {
                break;
            }
            // A server that died mid-boot fails fast instead of burning
            // the whole ready timeout against a dead port.
            if let Ok(Some(status)) = child.try_wait() {
                return Err(anyhow!(
                    "kokoro-server exited before serving on :{port}: {status}"
                ));
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(anyhow!(
                    "kokoro-server did not come up on :{port} within {READY_TIMEOUT_SECS}s \
                     (model load wedged?)"
                ));
            }
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        }
        tracing::info!(port, "kokoro-server spawned (model resident)");
        *slot = Some(KokoroChild {
            child,
            port,
            last_used: tokio::time::Instant::now(),
        });
        Ok(port)
    }

    /// Synthesize `input` in `voice` at `speed` to WAV bytes. Unknown
    /// voices surface the lane's own `400` message (`ValueError` text)
    /// instead of an opaque error.
    pub async fn synthesize(
        &self,
        shim: &Path,
        input: &str,
        voice: &str,
        speed: f32,
    ) -> Result<Vec<u8>> {
        let text = validate_input(input)?;
        let port = self.ensure(shim).await?;
        let resp = self
            .http
            .post(format!("http://127.0.0.1:{port}/tts"))
            .json(&serde_json::json!({
                "input": text,
                "voice": voice,
                "speed": speed,
            }))
            .send()
            .await
            .context("kokoro POST /tts")?;
        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("kokoro synthesis failed: {body}"));
        }
        let wav = resp.bytes().await.context("kokoro read WAV body")?;
        if !wav.starts_with(b"RIFF") {
            return Err(anyhow!(
                "kokoro /tts body is not RIFF/WAV (first bytes: {:?})",
                &wav[..wav.len().min(8)]
            ));
        }
        // A completed synthesis refreshes the reaper clock: with
        // `whisper_idle_secs` shorter than a synthesis's wall time the
        // reaper would otherwise kill a child that is actively serving.
        if let Ok(mut slot) = self.child.try_lock()
            && let Some(live) = slot.as_mut()
        {
            live.last_used = tokio::time::Instant::now();
        }
        Ok(wav.to_vec())
    }

    /// The lane's voice packs (for `/v1/audio/voices`). Spawns the
    /// child when cold - the listing is authoritative from the loaded
    /// voices file, never a stale embedded list.
    pub async fn voices(&self, shim: &Path) -> Result<Vec<String>> {
        let port = self.ensure(shim).await?;
        let resp = self
            .http
            .get(format!("http://127.0.0.1:{port}/voices"))
            .send()
            .await
            .context("kokoro GET /voices")?;
        if !resp.status().is_success() {
            return Err(anyhow!("kokoro /voices returned {}", resp.status()));
        }
        let list = resp
            .json::<serde_json::Value>()
            .await
            .context("kokoro /voices body")?;
        Ok(parse_voices(&list))
    }

    /// The lane's voice packs ONLY when a child is already resident -
    /// never spawns (the voices endpoint's boots-nothing contract).
    /// `None` = no live child.
    pub async fn voices_if_resident(&self) -> Option<Vec<String>> {
        let slot = self.child.lock().await;
        let live = slot.as_ref()?;
        if !tcp_alive(live.port).await {
            return None;
        }
        let resp = self
            .http
            .get(format!("http://127.0.0.1:{}/voices", live.port))
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let list = resp.json::<serde_json::Value>().await.ok()?;
        Some(parse_voices(&list))
    }

    /// Teardown: kill + reap. Never panics; a failed kill logs only.
    pub async fn shutdown(&self) {
        let mut slot = self.child.lock().await;
        if let Some(mut live) = slot.take() {
            let _ = live.child.kill().await;
            let _ = live.child.wait().await;
            tracing::info!(port = live.port, "kokoro-server stopped");
        }
    }

    /// Idle reaper: the whisper contract (mutex-serialized against
    /// ensure; a racing request simply respawns).
    pub async fn reap_idle(&self, max_idle: std::time::Duration) {
        let mut slot = self.child.lock().await;
        let Some(live) = slot.as_ref() else { return };
        let dead = !tcp_alive(live.port).await;
        let idle = live.last_used.elapsed() >= max_idle;
        if !dead && !idle {
            return;
        }
        let mut live = slot.take().expect("checked Some above");
        let idle_secs = live.last_used.elapsed().as_secs();
        let _ = live.child.kill().await;
        let _ = live.child.wait().await;
        if dead {
            tracing::info!(port = live.port, "kokoro-server had died; reaped the slot");
        } else {
            tracing::info!(
                port = live.port,
                idle_secs,
                "kokoro-server idle-reaped (respawns on next request)"
            );
        }
    }
}

impl Default for KokoroRuntime {
    fn default() -> Self {
        Self::new()
    }
}

async fn tcp_alive(port: u16) -> bool {
    tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .is_ok()
}

/// OS-assigned free port (bind :0, read, drop).
fn ephemeral_port() -> Result<u16> {
    let l = std::net::TcpListener::bind("127.0.0.1:0").context("bind :0 for kokoro port")?;
    Ok(l.local_addr().context("kokoro port addr")?.port())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__kokoro_looks_like_voice__grammar_pins() {
        // Kokoro shapes: language pair + m/f + name parts.
        assert!(looks_like_voice("af_sarah"));
        assert!(looks_like_voice("am_michael"));
        assert!(looks_like_voice("bf_nimble"));
        assert!(looks_like_voice("af_heart"));
        // Multi-part names stay one voice.
        assert!(looks_like_voice("am_santa_claus"));
        // Piper ids (uppercase + dashes) and non-voices never match.
        assert!(!looks_like_voice("en_US-amy-medium"));
        assert!(!looks_like_voice("kokoro"));
        assert!(!looks_like_voice(""));
        assert!(!looks_like_voice("af_"));
        assert!(!looks_like_voice("af"));
        assert!(!looks_like_voice("af_sarah-x"));
        assert!(!looks_like_voice("xx_sarah"));
        assert!(!looks_like_voice("AF_sarah"));
    }

    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__kokoro_validate_input__mirrors_piper_guards() {
        assert!(validate_input("  hello world  ").is_ok());
        let msg = validate_input("   ").unwrap_err().to_string();
        assert!(msg.contains("non-empty"), "{msg}");
        let long = "x".repeat(MAX_INPUT_CHARS + 1);
        let msg = validate_input(&long).unwrap_err().to_string();
        assert!(msg.contains("split long documents"), "{msg}");
    }
}
