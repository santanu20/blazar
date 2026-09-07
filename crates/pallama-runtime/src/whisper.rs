//! Whisper.cpp lane (H8): STT without leaving the machine.
//!
//! `whisper-server` (ggml-org/whisper.cpp releases) is installed under
//! `data/whisper/bin/<tag>/` and ggml models under `data/whisper/models/`.
//! State is plain files — no DB rows — so the lane stays inspectable and
//! trivially removable. The gateway lazily spawns ONE server child on the
//! first `/v1/audio/transcriptions` request and hot-swaps models via the
//! upstream `/load` endpoint instead of restarting per request.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

use crate::engine::gh::GhClient;
use crate::hf::{FilePlan, HfClient};
use pallama_core::PallamaDirs;

/// Verified live 2026-09-07: release ships `whisper-bin-*` assets and no
/// macOS server binary (only an xcframework library zip).
pub const WHISPER_REPO: &str = "ggml-org/whisper.cpp";
/// Verified live 2026-09-07: anonymous-accessible, legacy `ggml-*.bin`
/// files (the ggml-org/whisper-* repos 401 anonymously).
pub const WHISPER_MODEL_REPO: &str = "ggerganov/whisper.cpp";

/// Deterministic default-model preference when the request does not name
/// a pulled size: accuracy-per-second sweet spot first.
const SIZE_PREFERENCE: &[&str] = &[
    "base",
    "small",
    "tiny",
    "medium",
    "large-v3-turbo",
    "large-v3",
];

#[must_use]
pub fn bin_root(dirs: &PallamaDirs) -> PathBuf {
    dirs.data_dir.join("whisper").join("bin")
}

#[must_use]
pub fn models_dir(dirs: &PallamaDirs) -> PathBuf {
    dirs.data_dir.join("whisper").join("models")
}

/// Release asset for the running platform, or `None` where upstream
/// ships no server binary (macOS: build from source).
#[must_use]
pub fn asset_name(os: &str, arch: &str) -> Option<&'static str> {
    match (os, arch) {
        ("linux", "x86_64") => Some("whisper-bin-ubuntu-x64.tar.gz"),
        ("linux", "aarch64") => Some("whisper-bin-ubuntu-arm64.tar.gz"),
        ("windows", "x86_64") => Some("whisper-bin-x64.zip"),
        ("windows", "x86") => Some("whisper-bin-Win32.zip"),
        _ => None,
    }
}

/// Download + extract the latest whisper.cpp release. Returns its tag.
pub async fn install(gh: &GhClient, dirs: &PallamaDirs) -> Result<String> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let Some(asset_name) = asset_name(os, arch) else {
        return Err(anyhow!(
            "whisper.cpp releases ship no {os}/{arch} server binary \
             (macOS: build from source — \
             https://github.com/ggml-org/whisper.cpp/blob/master/docs/build.md)"
        ));
    };
    let release = gh.release_by(WHISPER_REPO, None).await?;
    let asset = release
        .assets
        .iter()
        .find(|a| a.name == asset_name)
        .ok_or_else(|| anyhow!("release {} has no asset {asset_name}", release.tag_name))?;
    let bytes = gh.download_asset_bytes(asset).await?;
    let dir = bin_root(dirs).join(&release.tag_name);
    std::fs::create_dir_all(&dir).with_context(|| format!("mkdir {}", dir.display()))?;
    crate::engine::extract_archive(&bytes, &dir, &asset.name)?;
    server_bin_in(&dir).ok_or_else(|| {
        anyhow!(
            "extracted {} but no whisper-server binary found under {}",
            asset.name,
            dir.display()
        )
    })?;
    Ok(release.tag_name)
}

/// Newest installed `whisper-server` binary plus its directory (needed
/// as `LD_LIBRARY_PATH` on Linux: the binary dlopens sibling libggml*.so).
#[must_use]
pub fn server_bin(dirs: &PallamaDirs) -> Option<(PathBuf, PathBuf)> {
    let root = bin_root(dirs);
    let mut tags: Vec<PathBuf> = std::fs::read_dir(&root)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    tags.sort();
    tags.reverse(); // newest tag last alphabetically for bNNNN
    for tag_dir in tags {
        if let Some(bin) = server_bin_in(&tag_dir) {
            return Some((bin, tag_dir));
        }
    }
    None
}

fn server_bin_in(dir: &Path) -> Option<PathBuf> {
    let name = if cfg!(windows) {
        "whisper-server.exe"
    } else {
        "whisper-server"
    };
    walk_for_file(dir, name)
}

fn walk_for_file(dir: &Path, name: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for e in entries.filter_map(std::result::Result::ok) {
        let p = e.path();
        if p.is_dir() {
            if let Some(hit) = walk_for_file(&p, name) {
                return Some(hit);
            }
        } else if p.file_name().is_some_and(|f| f == name) {
            return Some(p);
        }
    }
    None
}

/// Pulled model sizes (stems of `ggml-*.bin`), sorted.
pub fn list_models(dirs: &PallamaDirs) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(models_dir(dirs))
        .into_iter()
        .flatten()
        .filter_map(std::result::Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| {
            n.starts_with("ggml-")
                && std::path::Path::new(n)
                    .extension()
                    .is_some_and(|e| e == "bin")
        })
        .map(|n| {
            n.trim_start_matches("ggml-")
                .trim_end_matches(".bin")
                .to_string()
        })
        .collect();
    out.sort();
    out
}

#[must_use]
pub fn model_file(dirs: &PallamaDirs, size: &str) -> Option<PathBuf> {
    let f = models_dir(dirs).join(format!("ggml-{size}.bin"));
    f.is_file().then_some(f)
}

/// Map the request's `model` field to a pulled size. `whisper-1` /
/// `whisper-1-latest` / absent → preference order; `whisper-<size>` or a
/// bare size → exact match; unknown → None (caller 400s with the list).
#[must_use]
pub fn resolve_model(requested: Option<&str>, available: &[String]) -> Option<String> {
    if available.is_empty() {
        return None;
    }
    let norm = requested.map(|m| {
        m.trim()
            .trim_start_matches("whisper-")
            .trim_end_matches("-latest")
            .to_string()
    });
    match norm.as_deref() {
        // Unnamed, or OpenAI's "whisper-1" alias: pick by preference.
        None | Some("" | "1") => {}
        Some(size) if available.iter().any(|a| a == size) => return Some(size.to_string()),
        Some(_) => return None,
    }
    let pref_match = |a: &String, p: &str| {
        a == p || a.starts_with(&format!("{p}.")) || a.starts_with(&format!("{p}-"))
    };
    SIZE_PREFERENCE
        .iter()
        .find_map(|p| available.iter().find(|a| pref_match(a, p)).cloned())
        .or_else(|| available.first().cloned())
}

/// Download a ggml model into `data/whisper/models/`. Fails fast on a
/// non-ggml payload (magic check) instead of installing a corrupt model.
pub async fn pull(
    hf: &HfClient,
    dirs: &PallamaDirs,
    size: &str,
    on_progress: impl FnMut(u64, u64),
) -> Result<PathBuf> {
    let valid = size
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '.' || c == '-' || c == '_');
    if size.is_empty() || !valid {
        return Err(anyhow!(
            "invalid model size {size:?}: expected a release size like tiny, base, small, medium, large-v3-turbo"
        ));
    }
    let filename = format!("ggml-{size}.bin");
    let display_name = filename.clone();
    let dir = models_dir(dirs);
    std::fs::create_dir_all(&dir)?;
    let dest = dir.join(&filename);
    let plan = FilePlan {
        filename,
        bytes: 0,
        sha256: None,
    };
    hf.download_file(WHISPER_MODEL_REPO, &plan, &dest, on_progress)
        .await
        .with_context(|| format!("download {WHISPER_MODEL_REPO}/{display_name}"))?;
    let head = std::fs::read(&dest)
        .map(|b| b[..4.min(b.len())].to_vec())
        .unwrap_or_default();
    if !matches!(head.as_slice(), b"ggml" | b"lmgg") {
        // whisper writes its magic 0x67676d6c little-endian: on disk
        // the first four bytes read "lmgg", not "ggml".
        let _ = std::fs::remove_file(&dest);
        return Err(anyhow!(
            "{WHISPER_MODEL_REPO}/{size}: payload is not a ggml whisper model (bad magic) — removed"
        ));
    }
    Ok(dest)
}

/// One lazy whisper-server child, owned by the gateway `AppState` and
/// killed at serve teardown (H19: acquire has a named release).
pub struct WhisperRuntime {
    child: tokio::sync::Mutex<Option<WhisperChild>>,
}

struct WhisperChild {
    child: tokio::process::Child,
    port: u16,
    loaded: String,
}

impl WhisperRuntime {
    #[must_use]
    pub fn new() -> Self {
        Self {
            child: tokio::sync::Mutex::new(None),
        }
    }

    /// Port of a live server loaded with `size`; spawns on first use and
    /// hot-swaps models via POST /load (no restart) when the request
    /// names a different pulled size.
    pub async fn ensure(
        &self,
        size: &str,
        model_path: &Path,
        bin: &Path,
        lib_dir: &Path,
        ready_timeout: std::time::Duration,
    ) -> Result<u16> {
        let mut slot = self.child.lock().await;
        if let Some(live) = slot.as_mut() {
            if tcp_alive(live.port).await {
                if live.loaded != size {
                    // Upstream /load: hot model swap without respawn.
                    let url = format!("http://127.0.0.1:{}/load", live.port);
                    let part = reqwest::multipart::Part::text(model_path.display().to_string())
                        .mime_str("text/plain")
                        .context("mime")?;
                    let form = reqwest::multipart::Form::new().part("model", part);
                    let http = reqwest::Client::new();
                    let resp = http
                        .post(&url)
                        .multipart(form)
                        .send()
                        .await
                        .with_context(|| format!("POST {url}"))?;
                    if !resp.status().is_success() {
                        return Err(anyhow!("whisper /load {size}: HTTP {}", resp.status()));
                    }
                    tracing::info!(
                        port = live.port,
                        from = %live.loaded,
                        to = size,
                        "whisper-server hot-swapped model"
                    );
                    live.loaded = size.to_string();
                }
                return Ok(live.port);
            }
            // Dead child: reap before respawning.
            let _ = slot.take();
        }
        let port = ephemeral_port()?;
        let mut cmd = tokio::process::Command::new(bin);
        cmd.args([
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--model",
            &model_path.display().to_string(),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
        if cfg!(unix) {
            // The binary dlopens sibling libggml*.so; the loader does not
            // search the executable's own directory.
            let libs = lib_dir.display().to_string();
            let existing = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
            cmd.env("LD_LIBRARY_PATH", format!("{libs}:{existing}"));
        }
        let child = cmd
            .spawn()
            .with_context(|| format!("spawn {}", bin.display()))?;
        let deadline = tokio::time::Instant::now() + ready_timeout;
        loop {
            if tcp_alive(port).await {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(anyhow!(
                    "whisper-server did not come up on :{port} within {} s",
                    ready_timeout.as_secs()
                ));
            }
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        }
        *slot = Some(WhisperChild {
            child,
            port,
            loaded: size.to_string(),
        });
        Ok(port)
    }

    /// Teardown: kill + reap. Never panics; a failed kill logs only.
    pub async fn shutdown(&self) {
        let mut slot = self.child.lock().await;
        if let Some(mut live) = slot.take() {
            let _ = live.child.kill().await;
            let _ = live.child.wait().await;
            tracing::info!(port = live.port, model = %live.loaded, "whisper-server stopped");
        }
    }

    /// Currently loaded size + port (for `/api/whisper` status).
    pub async fn status(&self) -> Option<(u16, String)> {
        self.child
            .lock()
            .await
            .as_ref()
            .map(|c| (c.port, c.loaded.clone()))
    }
}

impl Default for WhisperRuntime {
    fn default() -> Self {
        Self::new()
    }
}

async fn tcp_alive(port: u16) -> bool {
    tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .is_ok()
}

fn ephemeral_port() -> Result<u16> {
    let l = std::net::TcpListener::bind(("127.0.0.1", 0)).context("bind ephemeral")?;
    Ok(l.local_addr().context("local addr")?.port())
}

#[cfg(test)]
#[allow(non_snake_case)] // suite convention: unit__scenario__expected (§6b)
mod tests {
    use super::*;

    #[test]
    fn unit__asset_name__platform_matrix() {
        assert_eq!(
            asset_name("linux", "x86_64"),
            Some("whisper-bin-ubuntu-x64.tar.gz")
        );
        assert_eq!(
            asset_name("linux", "aarch64"),
            Some("whisper-bin-ubuntu-arm64.tar.gz")
        );
        assert_eq!(asset_name("windows", "x86_64"), Some("whisper-bin-x64.zip"));
        assert_eq!(asset_name("windows", "x86"), Some("whisper-bin-Win32.zip"));
        assert_eq!(asset_name("macos", "x86_64"), None);
        assert_eq!(asset_name("linux", "riscv64"), None);
    }

    #[test]
    fn unit__resolve_model__exact_preference_unknown() {
        let avail: Vec<String> = ["base.en", "small", "tiny", "large-v3-turbo"]
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        // exact match wins
        assert_eq!(resolve_model(Some("small"), &avail), Some("small".into()));
        // normalized: whisper- prefix and -latest suffix stripped
        assert_eq!(
            resolve_model(Some("whisper-small-latest"), &avail),
            Some("small".into())
        );
        // unnamed: preference order picks base* first (variant suffix ok)
        assert_eq!(resolve_model(None, &avail), Some("base.en".into()));
        // OpenAI alias whisper-1 = unnamed
        assert_eq!(
            resolve_model(Some("whisper-1"), &avail),
            Some("base.en".into())
        );
        // preference beats availability order (small preferred over tiny)
        let both: Vec<String> = ["tiny", "small"].iter().map(|s| (*s).to_string()).collect();
        assert_eq!(resolve_model(None, &both), Some("small".into()));
        // nothing in preference: first available
        let odd: Vec<String> = ["custom", "other"]
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        assert_eq!(resolve_model(None, &odd), Some("custom".into()));
        // unknown name: None (caller 400s)
        assert_eq!(resolve_model(Some("mega"), &avail), None);
        // empty catalog: None
        assert_eq!(resolve_model(None, &[]), None);
    }

    #[test]
    fn unit__list_models_and_model_file__ggml_bins_only() {
        let tmp = tempfile::tempdir().expect("tmp");
        let models = PallamaDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        let dir = models_dir(&models);
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(dir.join("ggml-base.bin"), b"ggml").expect("w1");
        std::fs::write(dir.join("ggml-tiny.en.bin"), b"ggml").expect("w2");
        std::fs::write(dir.join("notes.txt"), b"x").expect("w3");
        let listed = list_models(&models);
        assert_eq!(listed, vec!["base".to_string(), "tiny.en".to_string()]);
        assert_eq!(model_file(&models, "base"), Some(dir.join("ggml-base.bin")));
        assert_eq!(model_file(&models, "medium"), None);
    }

    #[tokio::test]
    async fn lifecycle__whisper_runtime__shutdown_kills_child() {
        if !cfg!(unix) {
            return;
        }
        let rt = WhisperRuntime::new();
        let child = tokio::process::Command::new("sleep")
            .arg("60")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn sleep");
        let pid = child.id().expect("pid");
        *rt.child.lock().await = Some(WhisperChild {
            child,
            port: 1,
            loaded: "base".into(),
        });
        rt.shutdown().await;
        assert!(rt.status().await.is_none());
        // Reaped = wait resolved; process truly gone.
        let alive = std::path::Path::new(&format!("/proc/{pid}")).exists();
        assert!(!alive, "whisper child {pid} leaked past shutdown");
    }
}
