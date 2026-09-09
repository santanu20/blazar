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

/// Pin marker: names the tag `--tag` installs locked the runtime to.
/// Absent (or a plain `--install`) means "track the newest tag".
fn pin_path(dirs: &PallamaDirs) -> PathBuf {
    bin_root(dirs).join("pin")
}

/// Pinned tag from `bin/pin`, trimmed. `None` when unset, empty, or when
/// the content could escape the bin dir (separators / `..`) — a tag is
/// external input, never join it unvalidated.
#[must_use]
pub fn pinned_tag(dirs: &PallamaDirs) -> Option<String> {
    let raw = std::fs::read_to_string(pin_path(dirs)).ok()?;
    let tag = raw.trim();
    valid_tag(tag).then(|| tag.to_string())
}

/// A tag is safe to join into `bin/` only if it is non-empty and carries
/// no path separators or `..` traversal. Shared by the read side
/// (`pinned_tag`) and the write side (`set_pin`) so a written pin can
/// never be unreadable by its own sanitizer.
fn valid_tag(tag: &str) -> bool {
    !tag.is_empty() && !tag.contains(['/', '\\']) && !tag.contains("..")
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

/// Download + extract a whisper.cpp release. `Some(tag)` installs that
/// release; `pin` decides whether it becomes the runtime pin (explicit
/// `--tag` = pin, channel-resolution = no pin). `None` installs latest
/// and returns to tracking the newest tag (clears any pin). Old tags are
/// pruned to `KEEP_TAGS` (pinned always kept). Returns the installed tag.
pub async fn install(
    gh: &GhClient,
    dirs: &PallamaDirs,
    tag: Option<&str>,
    pin: bool,
) -> Result<String> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let Some(asset_name) = asset_name(os, arch) else {
        return Err(anyhow!(
            "whisper.cpp releases ship no {os}/{arch} server binary \
             (macOS: build from source — \
             https://github.com/ggml-org/whisper.cpp/blob/master/docs/build.md)"
        ));
    };
    let release = gh.release_by(WHISPER_REPO, tag).await?;
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
    // Selection policy: pinned installs pin it; unpinned installs track newest.
    match (tag, pin) {
        (Some(_), true) => {
            std::fs::write(pin_path(dirs), format!("{}\n", release.tag_name))
                .with_context(|| format!("write pin {}", pin_path(dirs).display()))?;
        }
        (Some(_), false) => {}
        (None, _) => {
            let _ = std::fs::remove_file(pin_path(dirs));
        }
    }
    prune(dirs)?;
    Ok(release.tag_name)
}

/// Active whisper-server binary plus its directory (needed as
/// `LD_LIBRARY_PATH` on Linux: the binary dlopens sibling libggml*.so).
/// A valid pin selects its tag; otherwise the newest installed tag wins.
#[must_use]
pub fn server_bin(dirs: &PallamaDirs) -> Option<(PathBuf, PathBuf)> {
    if let Some(tag) = pinned_tag(dirs) {
        let dir = bin_root(dirs).join(&tag);
        if let Some(bin) = server_bin_in(&dir) {
            return Some((bin, dir));
        }
        // Dangling pin (dir pruned or deleted): fall through to newest.
        tracing::warn!("whisper pin {tag} has no binary; using newest installed tag");
    }
    for tag_dir in sorted_tag_dirs(dirs) {
        if let Some(bin) = server_bin_in(&tag_dir) {
            return Some((bin, tag_dir));
        }
    }
    None
}

/// Sort key for a whisper.cpp tag: `vX.Y.Z` numeric components (missing
/// minor/patch = 0). Single-component tags (the old date shape
/// `v20250101`) and 4+ component tags do not parse — they order after
/// semver tags, alphabetically within their group.
type TagKey = Option<(u64, u64, u64)>;

fn tag_key(tag: &str) -> TagKey {
    let rest = tag.strip_prefix('v')?;
    if !rest.contains('.') {
        return None;
    }
    let mut nums = [0u64; 3];
    let mut it = rest.split('.');
    for slot in &mut nums {
        match it.next() {
            Some(t) => *slot = t.parse().ok()?,
            None => break,
        }
    }
    if it.next().is_some() {
        return None;
    }
    Some((nums[0], nums[1], nums[2]))
}

/// Installed tag dirs, newest first: semver tags by (major, minor, patch)
/// descending, then unparseable tags alphabetically descending (date
/// shapes compare correctly as strings).
fn sorted_tag_dirs(dirs: &PallamaDirs) -> Vec<PathBuf> {
    let mut tags: Vec<(TagKey, String)> = std::fs::read_dir(bin_root(dirs))
        .into_iter()
        .flatten()
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .filter_map(|p| {
            Some((
                tag_key(p.file_name()?.to_str()?),
                p.file_name()?.to_str()?.to_string(),
            ))
        })
        .collect();
    tags.sort_by(|a, b| match (a.0, b.0) {
        (Some(x), Some(y)) => y.cmp(&x).then_with(|| b.1.cmp(&a.1)),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => b.1.cmp(&a.1),
    });
    tags.into_iter()
        .map(|(_, name)| bin_root(dirs).join(name))
        .collect()
}

/// All installed tags, newest first (for `whisper --list`).
#[must_use]
pub fn installed_tags(dirs: &PallamaDirs) -> Vec<String> {
    sorted_tag_dirs(dirs)
        .into_iter()
        .filter_map(|p| p.file_name()?.to_str().map(str::to_string))
        .collect()
}

/// Pin the active whisper server to an already-installed tag, or unpin
/// (`None`) to track the newest installed tag. The tag must pass the
/// shared sanitizer AND exist on disk — pinning something uninstalled
/// would just dangle at selection time. The prune pass keeps the pinned
/// dir regardless of retention, so re-run it after unpinning to drop a
/// formerly-protected old tag.
pub fn set_pin(dirs: &PallamaDirs, tag: Option<&str>) -> Result<()> {
    match tag {
        Some(t) => {
            let t = t.trim();
            if !valid_tag(t) {
                anyhow::bail!("invalid tag {t:?}: must be a plain tag name (no path separators)");
            }
            if !installed_tags(dirs).iter().any(|installed| installed == t) {
                anyhow::bail!(
                    "tag {t} is not installed (installed: {}) — run: pallama whisper --install --tag {t}",
                    installed_tags(dirs).join(", ")
                );
            }
            std::fs::write(pin_path(dirs), format!("{t}\n"))
                .with_context(|| format!("write pin {}", pin_path(dirs).display()))?;
        }
        None => {
            let _ = std::fs::remove_file(pin_path(dirs));
        }
    }
    prune(dirs)?;
    Ok(())
}

/// Keep the newest `KEEP_TAGS` server dirs; the pinned dir (if any) is
/// never pruned — mirrors the llama engine lane's retention policy.
fn prune(dirs: &PallamaDirs) -> Result<()> {
    let pin = pinned_tag(dirs);
    for dir in sorted_tag_dirs(dirs)
        .into_iter()
        .skip(crate::engine::KEEP_TAGS)
    {
        let name = dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if Some(&name) == pin.as_ref() {
            continue;
        }
        std::fs::remove_dir_all(&dir)
            .with_context(|| format!("prune whisper dir {}", dir.display()))?;
        tracing::info!("pruned old whisper server {name}");
    }
    Ok(())
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
///
/// Security posture: upstream whisper-server has NO auth flag (no
/// `--api-key`; verified against ggml-org/whisper.cpp master
/// examples/server/README.md, 2026-09-08) — unlike llama.cpp children,
/// a per-child secret cannot be minted. Mitigation is loopback-only
/// bind + OS-assigned ephemeral port, pinned by unit test below.
/// Revisit if upstream grows an auth option.
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
        cmd.args(server_args(port, model_path))
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

/// Child argv: loopback bind is a security invariant (upstream has no
/// auth flag — see `WhisperRuntime` doc); pinned by unit test.
fn server_args(port: u16, model_path: &Path) -> Vec<String> {
    vec![
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        port.to_string(),
        "--model".into(),
        model_path.display().to_string(),
    ]
}

#[cfg(test)]
#[allow(non_snake_case)] // suite convention: unit__scenario__expected (§6b)
mod tests {
    use super::*;
    use sha2::Digest;

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
    fn unit__server_args__loopback_bind_is_pinned() {
        let args = server_args(49199, Path::new("/data/whisper/models/ggml-base.bin"));
        assert_eq!(
            args,
            vec![
                "--host".to_string(),
                "127.0.0.1".to_string(),
                "--port".to_string(),
                "49199".to_string(),
                "--model".to_string(),
                "/data/whisper/models/ggml-base.bin".to_string(),
            ],
            "whisper-server has no auth flag upstream; loopback bind is the isolation boundary — never widen to 0.0.0.0"
        );
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

    /// Stage a fake installed tag dir containing a whisper-server file.
    fn stage_server(dirs: &PallamaDirs, tag: &str) -> std::path::PathBuf {
        let dir = bin_root(dirs).join(tag);
        std::fs::create_dir_all(&dir).expect("mkdir");
        let bin = if cfg!(windows) {
            "whisper-server.exe"
        } else {
            "whisper-server"
        };
        std::fs::write(dir.join(bin), b"stub").expect("bin");
        dir
    }

    #[test]
    fn unit__tag_key__shapes() {
        assert_eq!(tag_key("v1.8.3"), Some((1, 8, 3)));
        assert_eq!(tag_key("v1.8"), Some((1, 8, 0)));
        assert_eq!(tag_key("v1.10.0"), Some((1, 10, 0)));
        // Date-era single-component and 4+ component tags do not parse.
        assert_eq!(tag_key("v20250101"), None);
        assert_eq!(tag_key("v1.2.3.4"), None);
        assert_eq!(tag_key("b4242"), None);
    }

    #[test]
    fn unit__server_bin__pinned_tag_beats_newest() {
        let tmp = tempfile::tempdir().expect("tmp");
        let dirs = PallamaDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        stage_server(&dirs, "v2.0.0");
        let old = stage_server(&dirs, "v1.0.0");
        std::fs::create_dir_all(bin_root(&dirs)).expect("root");
        std::fs::write(pin_path(&dirs), "v1.0.0\n").expect("pin");
        let (bin, dir) = server_bin(&dirs).expect("server");
        assert_eq!(dir, old);
        assert!(bin.starts_with(old));
    }

    #[test]
    fn unit__server_bin__dangling_or_hostile_pin_ignored() {
        let tmp = tempfile::tempdir().expect("tmp");
        let dirs = PallamaDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        let newest = stage_server(&dirs, "v2.0.0");
        stage_server(&dirs, "v1.0.0");
        std::fs::create_dir_all(bin_root(&dirs)).expect("root");
        // Dangling: pin names a tag that has no dir.
        std::fs::write(pin_path(&dirs), "v9.9.9\n").expect("pin");
        let (_, dir) = server_bin(&dirs).expect("server");
        assert_eq!(dir, newest);
        // Hostile: path-escaping content must never be joined.
        std::fs::write(pin_path(&dirs), "../../etc\n").expect("pin");
        let (_, dir) = server_bin(&dirs).expect("server");
        assert_eq!(dir, newest);
        // Empty pin = no pin.
        std::fs::write(pin_path(&dirs), "  \n").expect("pin");
        let (_, dir) = server_bin(&dirs).expect("server");
        assert_eq!(dir, newest);
    }

    #[test]
    fn unit__set_pin__round_trip_selects_and_unpins() {
        let tmp = tempfile::tempdir().expect("tmp");
        let dirs = PallamaDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        stage_server(&dirs, "v1.8.0");
        stage_server(&dirs, "v1.9.0");
        set_pin(&dirs, Some("v1.8.0")).expect("pin");
        assert_eq!(pinned_tag(&dirs), Some("v1.8.0".to_string()));
        let (bin, dir) = server_bin(&dirs).expect("server");
        assert!(bin.starts_with(&dir));
        assert!(dir.ends_with("v1.8.0"));
        set_pin(&dirs, None).expect("unpin");
        assert_eq!(pinned_tag(&dirs), None);
        let (_, dir) = server_bin(&dirs).expect("server");
        assert!(dir.ends_with("v1.9.0"));
    }

    #[test]
    fn unit__set_pin__unknown_or_hostile_tag_rejected() {
        let tmp = tempfile::tempdir().expect("tmp");
        let dirs = PallamaDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        stage_server(&dirs, "v1.9.0");
        let err = set_pin(&dirs, Some("v1.8.0")).expect_err("not installed");
        assert!(err.to_string().contains("not installed"));
        assert!(err.to_string().contains("v1.9.0"));
        let err = set_pin(&dirs, Some("../escape")).expect_err("hostile");
        assert!(err.to_string().contains("invalid tag"));
        assert_eq!(pinned_tag(&dirs), None);
    }

    #[test]
    fn unit__set_pin__unpin_prunes_formerly_protected_old_tag() {
        let tmp = tempfile::tempdir().expect("tmp");
        let dirs = PallamaDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        for t in ["v1.1.0", "v1.2.0", "v1.3.0", "v1.4.0"] {
            stage_server(&dirs, t);
        }
        set_pin(&dirs, Some("v1.1.0")).expect("pin oldest");
        assert!(installed_tags(&dirs).contains(&"v1.1.0".to_string()));
        set_pin(&dirs, None).expect("unpin -> prune");
        assert!(!installed_tags(&dirs).contains(&"v1.1.0".to_string()));
        assert_eq!(installed_tags(&dirs).len(), 3);
    }

    #[test]
    fn unit__prune__keeps_newest_three_and_pinned() {
        let tmp = tempfile::tempdir().expect("tmp");
        let dirs = PallamaDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        for t in ["v1.1.0", "v1.2.0", "v1.3.0", "v1.4.0", "v1.5.0"] {
            stage_server(&dirs, t);
        }
        std::fs::create_dir_all(bin_root(&dirs)).expect("root");
        std::fs::write(pin_path(&dirs), "v1.1.0\n").expect("pin");
        prune(&dirs).expect("prune");
        let left = installed_tags(&dirs);
        assert_eq!(left, vec!["v1.5.0", "v1.4.0", "v1.3.0", "v1.1.0"]);
    }

    #[test]
    fn unit__prune__semver_order_not_alphabetical() {
        let tmp = tempfile::tempdir().expect("tmp");
        let dirs = PallamaDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        // Alphabetical order would call v1.9.0 "newest" and prune v1.10.0.
        for t in ["v1.7.0", "v1.8.0", "v1.9.0", "v1.10.0"] {
            stage_server(&dirs, t);
        }
        prune(&dirs).expect("prune");
        let left = installed_tags(&dirs);
        assert_eq!(left, vec!["v1.10.0", "v1.9.0", "v1.8.0"]);
    }

    /// Full wiremock cycle: `--tag` install pins, plain install unpins,
    /// prune keeps the newest `KEEP_TAGS`.
    #[tokio::test]
    async fn install__tag_pins_latest_unpins_prunes() {
        async fn mount(
            api: &wiremock::MockServer,
            endpoint: &str,
            tag: &str,
            asset: &str,
            bytes: &[u8],
        ) {
            use wiremock::matchers::{method, path};
            let body = serde_json::json!({
                "tag_name": tag,
                "prerelease": false,
                "assets": [{
                    "name": asset,
                    "digest": format!("sha256:{:x}", sha2::Sha256::digest(bytes)),
                    "size": bytes.len(),
                    "browser_download_url": format!("{}/download/{}/{}", api.uri(), tag, asset),
                }]
            });
            wiremock::Mock::given(method("GET"))
                .and(path(endpoint))
                .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(body))
                .mount(api)
                .await;
            wiremock::Mock::given(method("GET"))
                .and(path(format!("/download/{tag}/{asset}")))
                .respond_with(wiremock::ResponseTemplate::new(200).set_body_bytes(bytes.to_vec()))
                .mount(api)
                .await;
        }

        let Some(asset) = asset_name(std::env::consts::OS, std::env::consts::ARCH) else {
            return; // platform without an upstream server binary
        };
        let tmp = tempfile::tempdir().expect("tmp");
        let dirs = PallamaDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        let api = wiremock::MockServer::start().await;

        let tarball = |tag: &str| -> Vec<u8> {
            use std::io::Write as _;
            let mut tarbuf = Vec::new();
            {
                let mut builder = tar::Builder::new(&mut tarbuf);
                let root = format!("whisper-{tag}");
                let mut header = tar::Header::new_gnu();
                header.set_size(0);
                header.set_entry_type(tar::EntryType::Directory);
                header.set_mode(0o755);
                header.set_cksum();
                builder
                    .append_data(&mut header, &root, std::io::empty())
                    .unwrap();
                let mut header = tar::Header::new_gnu();
                header.set_size(4);
                header.set_entry_type(tar::EntryType::Regular);
                header.set_mode(0o755);
                header.set_cksum();
                builder
                    .append_data(&mut header, format!("{root}/whisper-server"), &b"stub"[..])
                    .unwrap();
                builder.finish().unwrap();
            }
            let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
            gz.write_all(&tarbuf).unwrap();
            gz.finish().unwrap()
        };

        let v181 = tarball("v1.8.1");
        mount(
            &api,
            "/repos/ggml-org/whisper.cpp/releases/tags/v1.8.1",
            "v1.8.1",
            asset,
            &v181,
        )
        .await;
        let v190 = tarball("v1.9.0");
        mount(
            &api,
            "/repos/ggml-org/whisper.cpp/releases/latest",
            "v1.9.0",
            asset,
            &v190,
        )
        .await;

        let gh = GhClient::with_base(&api.uri(), None).unwrap();

        // Tag install: pins.
        let tag = install(&gh, &dirs, Some("v1.8.1"), true)
            .await
            .expect("tag install");
        assert_eq!(tag, "v1.8.1");
        assert_eq!(pinned_tag(&dirs), Some("v1.8.1".into()));
        assert!(server_bin(&dirs).is_some_and(|(_, d)| d.ends_with("v1.8.1")));

        // Older staged dirs + a latest install: unpins, prunes to 3 newest.
        for t in ["v1.7.0", "v1.6.0", "v1.5.0"] {
            stage_server(&dirs, t);
        }
        let tag = install(&gh, &dirs, None, false)
            .await
            .expect("latest install");
        assert_eq!(tag, "v1.9.0");
        assert_eq!(pinned_tag(&dirs), None);
        assert_eq!(installed_tags(&dirs), vec!["v1.9.0", "v1.8.1", "v1.7.0"]);
    }
}
