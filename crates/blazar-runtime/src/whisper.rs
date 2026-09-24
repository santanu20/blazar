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

use crate::engine::gh::{GhClient, GhRelease};
use crate::hf::{FilePlan, HfClient};
use blazar_core::config::UpdateChannel;
use blazar_core::BlazarDirs;

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
pub fn bin_root(dirs: &BlazarDirs) -> PathBuf {
    dirs.data_dir.join("whisper").join("bin")
}

#[must_use]
pub fn models_dir(dirs: &BlazarDirs) -> PathBuf {
    dirs.data_dir.join("whisper").join("models")
}

/// Pin marker: names the tag `--tag` installs locked the runtime to.
/// Absent (or a plain `--install`) means "track the newest tag".
fn pin_path(dirs: &BlazarDirs) -> PathBuf {
    bin_root(dirs).join("pin")
}

/// Pinned tag from `bin/pin`, trimmed. `None` when unset, empty, or when
/// the content could escape the bin dir (separators / `..`) — a tag is
/// external input, never join it unvalidated.
#[must_use]
pub fn pinned_tag(dirs: &BlazarDirs) -> Option<String> {
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

/// Newest release that actually ships `asset`. GitHub serves the list
/// newest-first. `stable_only` skips prereleases (the `/releases/latest`
/// contract) — whisper.cpp tags releases (v1.9.4, 2026-09-11) that carry
/// no assets while the prerelease b-tags carry them, so an asset-blind
/// "latest" can point installs at a tag that can never install.
fn newest_with_asset<'a>(
    releases: &'a [GhRelease],
    asset: &str,
    stable_only: bool,
) -> Option<&'a GhRelease> {
    releases
        .iter()
        .filter(|r| !stable_only || !r.prerelease)
        .find(|r| r.assets.iter().any(|a| a.name == asset))
}

/// Asset-aware channel resolution: the newest release this platform can
/// actually install, as a full release (one API list call). `Latest`
/// walks all releases (prerelease firehose, mirroring the engine lane);
/// `Stable` walks non-prereleases only and, when none carries the
/// server binary, names the newest prerelease that does as the escape
/// hatch.
async fn release_for_channel(gh: &GhClient, channel: UpdateChannel) -> Result<GhRelease> {
    let asset_name = asset_name(std::env::consts::OS, std::env::consts::ARCH).ok_or_else(|| {
        anyhow!(
            "whisper.cpp releases ship no {}/{} server binary (macOS: build from source — \
             https://github.com/ggml-org/whisper.cpp/blob/master/docs/build.md)",
            std::env::consts::OS,
            std::env::consts::ARCH
        )
    })?;
    let releases = gh.list_releases_repo(WHISPER_REPO).await?;
    let stable_only = matches!(channel, UpdateChannel::Stable);
    if let Some(rel) = newest_with_asset(&releases, asset_name, stable_only) {
        return Ok(rel.clone());
    }
    match newest_with_asset(&releases, asset_name, false) {
        Some(rel) => Err(anyhow!(
            "no stable whisper.cpp release ships the {asset_name} server binary; \
             newest installable is prerelease {} — set update_channel = latest \
             or install it explicitly: blazar whisper --install --tag {}",
            rel.tag_name,
            rel.tag_name
        )),
        None => Err(anyhow!(
            "no recent whisper.cpp release ({} checked) ships the {asset_name} \
             server binary — upstream may have renamed assets",
            releases.len()
        )),
    }
}

/// Asset-aware channel target for the whisper server: the newest
/// release this platform can actually install, by tag. See
/// `release_for_channel` for the channel semantics.
pub async fn channel_target(gh: &GhClient, channel: UpdateChannel) -> Result<String> {
    Ok(release_for_channel(gh, channel).await?.tag_name)
}

/// Download + extract a whisper.cpp release. `Some(tag)` installs that
/// release; `pin` decides whether it becomes the runtime pin (explicit
/// `--tag` = pin, channel-resolution = no pin). `None` installs the
/// newest release that ships this platform's server binary and returns
/// to tracking the newest tag (clears any pin). Old tags are
/// pruned to `KEEP_TAGS` (pinned always kept). Returns the installed tag.
pub async fn install(
    gh: &GhClient,
    dirs: &BlazarDirs,
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
    // Asset-aware latest: an assetless newest tag (v1.9.4 shape) must
    // fall through to the newest release that actually installs.
    let release = match tag {
        Some(t) => gh.release_by(WHISPER_REPO, Some(t)).await?,
        None => release_for_channel(gh, UpdateChannel::Latest).await?,
    };
    let asset = release
        .assets
        .iter()
        .find(|a| a.name == asset_name)
        .ok_or_else(|| anyhow!("release {} has no asset {asset_name}", release.tag_name))?;
    let bytes = gh.download_asset_bytes(asset).await?;
    let dir = bin_root(dirs).join(&release.tag_name);
    // F96: replace, don't merge — a re-install over an existing tag dir
    // must not leave stale binaries from the old extract behind.
    if dir.exists() {
        std::fs::remove_dir_all(&dir).with_context(|| format!("replace {}", dir.display()))?;
    }
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
/// Resolution order: the engines-table lane (`blazar engine install
/// --kind whisper`) first, then the legacy `data/whisper/bin` tree —
/// both stay working installs.
#[must_use]
pub fn server_bin(dirs: &BlazarDirs) -> Option<(PathBuf, PathBuf)> {
    if let Some(hit) = engines_lane_bin(dirs) {
        return Some(hit);
    }
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

/// Whether `bin` was served by the legacy `data/whisper/bin` tree (used
/// to scope the legacy pin display to the lane that honors it).
#[must_use]
pub fn is_legacy_bin(dirs: &BlazarDirs, bin: &Path) -> bool {
    bin.starts_with(bin_root(dirs))
}

/// The engines-table lane's whisper row: the active row when one is
/// flagged, else the newest installed (mirror of the serving pick in
/// [`engines_lane_bin`]). `None` when no whisper row exists (the legacy
/// tree decides) or the store cannot be read (warn, never mask).
fn engines_lane_row(dirs: &BlazarDirs) -> Option<blazar_core::store::EngineRow> {
    let store = match blazar_core::Store::open(dirs) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("whisper engines-lane lookup could not open the store: {e:#}");
            return None;
        }
    };
    let rows = match store.list_engines() {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("whisper engines-lane listing failed: {e:#}");
            return None;
        }
    };
    // `list_engines` orders newest-installed first; the active row (if
    // any) still wins so a pinned older lane is honored.
    rows.into_iter()
        .filter(|r| r.kind == blazar_core::engine_kind::EngineKind::Whisper)
        .max_by_key(|r| i64::from(r.active))
}

/// The engines-table lane: resolve the whisper row's binary the same way
/// register did. `None` when no whisper row exists (legacy tree decides)
/// or the store cannot be read (warn, never mask).
fn engines_lane_bin(dirs: &BlazarDirs) -> Option<(PathBuf, PathBuf)> {
    let row = engines_lane_row(dirs)?;
    let bin = engines_lane_server_bin(dirs, &row)?;
    let lib = bin.parent()?.to_path_buf();
    Some((bin, lib))
}

/// Whether the serving whisper lane is engines-table backed — update
/// hints must name the lane that can actually update the serving binary
/// (`blazar whisper --install` writes the legacy tree, which
/// [`server_bin`] never picks while a row exists).
#[must_use]
pub fn engines_lane_installed(dirs: &BlazarDirs) -> bool {
    engines_lane_row(dirs).is_some_and(|row| engines_lane_server_bin(dirs, &row).is_some())
}

/// The tag currency verdicts must compare against upstream: the engines
/// row's tag when that lane is serving, else the legacy tree's effective
/// tag (pin, else newest). The binary's LIB DIRECTORY is not a tag
/// source — the extract subdir name leaked into the verdict once and
/// produced an update warning that no update could clear (live:
/// "update available: b5130 (running: whisper-bin-ubuntu-x64)").
#[must_use]
pub fn installed_tag(dirs: &BlazarDirs) -> Option<String> {
    if let Some(row) = engines_lane_row(dirs) {
        if engines_lane_server_bin(dirs, &row).is_some() {
            return Some(row.tag);
        }
    }
    // Legacy tree, mirroring server_bin's resolution: a pin whose dir
    // still holds a binary wins (dangling pins fall through to newest).
    if let Some(tag) = pinned_tag(dirs) {
        if server_bin_in(&bin_root(dirs).join(&tag)).is_some() {
            return Some(tag);
        }
    }
    sorted_tag_dirs(dirs)
        .into_iter()
        .next()
        .and_then(|d| d.file_name().map(|n| n.to_string_lossy().into_owned()))
}

/// Row's binary: the install-time probed path when it still exists,
/// otherwise re-derived from the tag dir (a relocated data dir must not
/// brick the lane — same recovery register performs).
fn engines_lane_server_bin(
    dirs: &BlazarDirs,
    row: &blazar_core::store::EngineRow,
) -> Option<PathBuf> {
    if let Ok(m) = serde_json::from_str::<crate::engine::manifest::Manifest>(&row.manifest) {
        let probed = PathBuf::from(&m.server_path);
        if probed.is_file() {
            return Some(probed);
        }
    }
    crate::engine::find_engine_binary(
        &dirs.engines_dir().join(&row.tag),
        &["whisper-server", "whisper-server.exe"],
    )
    .ok()
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
fn sorted_tag_dirs(dirs: &BlazarDirs) -> Vec<PathBuf> {
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
pub fn installed_tags(dirs: &BlazarDirs) -> Vec<String> {
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
pub fn set_pin(dirs: &BlazarDirs, tag: Option<&str>) -> Result<()> {
    match tag {
        Some(t) => {
            let t = t.trim();
            if !valid_tag(t) {
                anyhow::bail!("invalid tag {t:?}: must be a plain tag name (no path separators)");
            }
            if !installed_tags(dirs).iter().any(|installed| installed == t) {
                anyhow::bail!(
                    "tag {t} is not installed (installed: {}) — run: blazar whisper --install --tag {t}",
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
fn prune(dirs: &BlazarDirs) -> Result<()> {
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

pub(crate) fn walk_for_file(dir: &Path, name: &str) -> Option<PathBuf> {
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

/// Size stem of a `ggml-<size>.bin` filename, or `None` for anything
/// else (the repo also ships `ggml-*-encoder.mlmodelc.zip` `CoreML`
/// sidecars that must not surface as pullable models).
#[must_use]
pub fn ggml_size_from_path(filename: &str) -> Option<String> {
    let base = filename.rsplit('/').next().unwrap_or(filename);
    base.strip_prefix("ggml-")?
        .strip_suffix(".bin")
        .map(str::to_string)
}

/// Pulled model sizes (stems of `ggml-*.bin`), sorted.
pub fn list_models(dirs: &BlazarDirs) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(models_dir(dirs))
        .into_iter()
        .flatten()
        .filter_map(std::result::Result::ok)
        .filter_map(|e| ggml_size_from_path(&e.file_name().to_string_lossy()))
        .collect();
    out.sort();
    out
}

/// One ggml model in the upstream index (`ggerganov/whisper.cpp`),
/// with its on-disk byte size — the searchable remote catalog behind
/// `whisper --search`.
#[derive(Debug, Clone)]
pub struct RemoteModel {
    pub size: String,
    pub bytes: u64,
}

/// Every ggml model the upstream repo currently ships, sorted by size
/// stem. One non-recursive tree listing (the models live at the repo
/// root); verified live 2026-09-23: 33 models spanning tiny..large-v3
/// including `.en` variants and q5/q8 quantized builds.
pub async fn remote_models(hf: &HfClient) -> Result<Vec<RemoteModel>> {
    let entries = hf.list_tree(WHISPER_MODEL_REPO, "", false).await?;
    let mut out: Vec<RemoteModel> = entries
        .into_iter()
        .filter(crate::hf::HfTreeEntry::is_file)
        .filter_map(|e| {
            Some(RemoteModel {
                size: ggml_size_from_path(&e.path)?,
                bytes: e.size.unwrap_or(0),
            })
        })
        .collect();
    out.sort_by(|a, b| a.size.cmp(&b.size));
    Ok(out)
}

#[must_use]
pub fn model_file(dirs: &BlazarDirs, size: &str) -> Option<PathBuf> {
    let f = models_dir(dirs).join(format!("ggml-{size}.bin"));
    f.is_file().then_some(f)
}

/// Map the request's `model` field to a pulled size. `whisper-1` /
/// `whisper-1-latest` / absent → preference order; `whisper-<size>`,
/// `ggml-<size>` (the HF file naming) or a bare size → exact match;
/// unknown → None (caller 400s with the list).
#[must_use]
pub fn resolve_model(requested: Option<&str>, available: &[String]) -> Option<String> {
    if available.is_empty() {
        return None;
    }
    let norm = requested.map(|m| {
        m.trim()
            .trim_start_matches("whisper-")
            .trim_start_matches("ggml-")
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
    dirs: &BlazarDirs,
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
    // F93: read EXACTLY 4 bytes — `fs::read` buffered the whole multi-GB
    // ggml into RAM just to inspect the magic (2.9 GB spike for large-v3).
    let mut head = [0u8; 4];
    let magic_ok = std::fs::File::open(&dest)
        .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut head))
        .is_ok();
    if !magic_ok || !matches!(&head, b"ggml" | b"lmgg") {
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
    /// Last instant the lane served (or swapped) a request — the idle
    /// reaper's clock. Refreshed on every `ensure` hit, never elsewhere.
    last_used: tokio::time::Instant,
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
                live.last_used = tokio::time::Instant::now();
                if live.loaded != size {
                    // Upstream /load: hot model swap without respawn.
                    let url = format!("http://127.0.0.1:{}/load", live.port);
                    let part = reqwest::multipart::Part::text(model_path.display().to_string())
                        .mime_str("text/plain")
                        .context("mime")?;
                    let form = reqwest::multipart::Form::new().part("model", part);
                    // F94: bounded client — this POST runs under the
                    // instance mutex; a hung whisper-server would pin
                    // every later transcription behind it.
                    let http = reqwest::Client::builder()
                        .timeout(std::time::Duration::from_mins(2))
                        .build()?;
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
        let mut std_cmd = std::process::Command::new(bin);
        std_cmd
            .args(server_args(port, model_path))
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        #[cfg(unix)]
        {
            // Kernel lifetime tie: the whisper-server must not outlive
            // the process that spawned it (crash/SIGKILL/terminal
            // close would otherwise leak a VRAM-holding child).
            crate::probe::parent_death_tie(&mut std_cmd);
        }
        let mut cmd = tokio::process::Command::from(std_cmd);
        cmd.kill_on_drop(true);
        if cfg!(unix) {
            // The binary dlopens sibling libggml*.so; the loader does not
            // search the executable's own directory.
            let libs = lib_dir.display().to_string();
            let existing = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
            cmd.env("LD_LIBRARY_PATH", format!("{libs}:{existing}"));
        }
        let mut child = cmd
            .spawn()
            .with_context(|| format!("spawn {}", bin.display()))?;
        let deadline = tokio::time::Instant::now() + ready_timeout;
        loop {
            if tcp_alive(port).await {
                break;
            }
            // F95: a server that died mid-boot fails fast instead of
            // burning the whole ready timeout against a dead port.
            if let Ok(Some(status)) = child.try_wait() {
                return Err(anyhow!(
                    "whisper-server exited before serving on :{port}: {status}"
                ));
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
            last_used: tokio::time::Instant::now(),
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

    /// Idle reaper: kill the child when it has served no request for
    /// `max_idle`. Serialized against `ensure` by the same mutex, so a
    /// request racing the reap simply respawns the child (the lazy-lane
    /// contract; the loser pays one model load). A dead child is
    /// reaped here too — same hygiene `ensure` does on its next hit,
    /// just sooner. No-op when no child exists.
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
            tracing::info!(port = live.port, model = %live.loaded, "whisper-server had died; reaped the slot");
        } else {
            tracing::info!(
                port = live.port,
                model = %live.loaded,
                idle_secs,
                "whisper-server idle-reaped (respawns on next request)"
            );
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
    use crate::engine::gh::GhAsset;
    use sha2::Digest;

    fn release(tag: &str, prerelease: bool, assets: &[&str]) -> GhRelease {
        GhRelease {
            tag_name: tag.to_string(),
            prerelease,
            assets: assets
                .iter()
                .map(|n| GhAsset {
                    name: (*n).to_string(),
                    digest: None,
                    size: None,
                    browser_download_url: String::new(),
                })
                .collect(),
            published_at: None,
        }
    }

    /// The 2026-09-11 upstream shape, pinned: v1.9.4 published as a full
    /// (non-prerelease) release with ZERO assets while the prerelease
    /// b-tags carry the binaries. Asset-blind "latest" pointed installs
    /// at a tag that can never install (R2-25).
    #[test]
    fn unit__newest_with_asset__skips_assetless_latest() {
        let rels = vec![
            release("v1.9.4", false, &[]),
            release("b5130", true, &["whisper-bin-ubuntu-x64.tar.gz"]),
            release("b5127", true, &["whisper-bin-ubuntu-x64.tar.gz"]),
            release("v1.9.3", false, &[]),
            release("b4938", true, &["whisper-bin-ubuntu-x64.tar.gz"]),
        ];
        // Latest channel: newest installable wins, assetless skipped.
        assert_eq!(
            newest_with_asset(&rels, "whisper-bin-ubuntu-x64.tar.gz", false)
                .map(|r| r.tag_name.as_str()),
            Some("b5130")
        );
        // Stable channel: non-prerelease only — v1.9.4 is assetless, so
        // no stable release qualifies.
        assert!(newest_with_asset(&rels, "whisper-bin-ubuntu-x64.tar.gz", true).is_none());
        // A future stable release carrying assets wins over older b-tags.
        let with_stable = vec![
            release("v1.9.5", false, &["whisper-bin-ubuntu-x64.tar.gz"]),
            release("b5130", true, &["whisper-bin-ubuntu-x64.tar.gz"]),
        ];
        assert_eq!(
            newest_with_asset(&with_stable, "whisper-bin-ubuntu-x64.tar.gz", true)
                .map(|r| r.tag_name.as_str()),
            Some("v1.9.5")
        );
        // No release carries the wanted asset (renamed upstream).
        assert!(newest_with_asset(&rels, "whisper-bin-ubuntu-musl-x64.tar.gz", false).is_none());
    }

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
        // normalized: ggml- prefix (HF file naming) resolves to the bare size
        assert_eq!(resolve_model(Some("ggml-base"), &avail), None);
        let with_base: Vec<String> = ["base", "small"].iter().map(|s| (*s).to_string()).collect();
        assert_eq!(
            resolve_model(Some("ggml-base"), &with_base),
            Some("base".into())
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
        let models = BlazarDirs {
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

    #[test]
    fn unit__ggml_size_from_path__ggml_bins_only() {
        // Full tree paths reduce to the size stem (basename, ggml-/.bin).
        assert_eq!(
            ggml_size_from_path("ggml-large-v3-turbo-q5_0.bin").as_deref(),
            Some("large-v3-turbo-q5_0")
        );
        assert_eq!(
            ggml_size_from_path("ggml-base.bin").as_deref(),
            Some("base")
        );
        // CoreML sidecars and foreign files are not pullable models.
        assert!(ggml_size_from_path("ggml-small-encoder.mlmodelc.zip").is_none());
        assert!(ggml_size_from_path("README.md").is_none());
        assert!(ggml_size_from_path("ggml-naked").is_none());
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
            last_used: tokio::time::Instant::now(),
        });
        rt.shutdown().await;
        assert!(rt.status().await.is_none());
        // Reaped = wait resolved; process truly gone.
        let alive = std::path::Path::new(&format!("/proc/{pid}")).exists();
        assert!(!alive, "whisper child {pid} leaked past shutdown");
    }

    /// The idle reaper kills a child whose last serve is older than the
    /// budget, keeps a fresh one, and cleans a dead child's slot. Uses
    /// the same injected-fake-child pattern as the shutdown lifecycle
    /// test above (`sleep` stands in for whisper-server; port 1 never
    /// answers, which the reaper must treat as dead).
    #[tokio::test]
    async fn lifecycle__whisper_runtime__reap_idle_budget_and_dead_child() {
        if !cfg!(unix) {
            return;
        }
        let rt = WhisperRuntime::new();
        // Fresh child on an unreachable port: "dead" wins even though
        // last_used is now — a corpse must not squat the lane until the
        // idle budget also expires.
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
            last_used: tokio::time::Instant::now(),
        });
        rt.reap_idle(std::time::Duration::from_hours(1)).await;
        assert!(rt.status().await.is_none(), "dead child reaped");
        assert!(
            !std::path::Path::new(&format!("/proc/{pid}")).exists(),
            "dead child {pid} leaked past reap"
        );

        // Idle budget path: reachable port + stale last_used → reaped.
        // A port that answers TCP: bind our own listener.
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind");
        let port = listener.local_addr().expect("addr").port();
        let child = tokio::process::Command::new("sleep")
            .arg("60")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn sleep 2");
        let pid2 = child.id().expect("pid2");
        *rt.child.lock().await = Some(WhisperChild {
            child,
            port,
            loaded: "base".into(),
            last_used: tokio::time::Instant::now() - std::time::Duration::from_mins(10),
        });
        rt.reap_idle(std::time::Duration::from_mins(5)).await;
        assert!(rt.status().await.is_none(), "idle child reaped");
        assert!(
            !std::path::Path::new(&format!("/proc/{pid2}")).exists(),
            "idle child {pid2} leaked past reap"
        );

        // Fresh child on a reachable port survives the same budget.
        let child = tokio::process::Command::new("sleep")
            .arg("60")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn sleep 3");
        let pid3 = child.id().expect("pid3");
        *rt.child.lock().await = Some(WhisperChild {
            child,
            port,
            loaded: "base".into(),
            last_used: tokio::time::Instant::now(),
        });
        rt.reap_idle(std::time::Duration::from_mins(5)).await;
        assert!(rt.status().await.is_some(), "fresh child untouched");
        rt.shutdown().await;
        assert!(
            !std::path::Path::new(&format!("/proc/{pid3}")).exists(),
            "child {pid3} leaked past shutdown"
        );
    }

    /// Stage a fake installed tag dir containing a whisper-server file.
    fn stage_server(dirs: &BlazarDirs, tag: &str) -> std::path::PathBuf {
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
        let dirs = BlazarDirs {
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
        let dirs = BlazarDirs {
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
    fn unit__server_bin__engines_lane_row_beats_legacy_pin() {
        // The engines table is the primary lane once a whisper row
        // exists: even a legacy PIN must not shadow it (the pin belongs
        // to the legacy tree, a different install channel).
        let tmp = tempfile::tempdir().expect("tmp");
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        let legacy = stage_server(&dirs, "v1.0.0");
        std::fs::create_dir_all(bin_root(&dirs)).expect("root");
        std::fs::write(pin_path(&dirs), "v1.0.0\n").expect("pin");
        // Engines-lane install: nested tar-root shape, manifest the
        // resolver cannot use (decode falls through to a directory
        // search — the recursion is part of the contract).
        let lane_dir = dirs.engines_dir().join("b5130/whisper-bin-ubuntu-x64");
        std::fs::create_dir_all(&lane_dir).expect("lane");
        std::fs::write(lane_dir.join("whisper-server"), b"stub").expect("bin");
        let store = blazar_core::Store::open(&dirs).expect("store");
        store
            .upsert_engine(&blazar_core::store::EngineRow {
                tag: "b5130".to_string(),
                asset: "whisper-bin-ubuntu-x64.tar.gz".to_string(),
                sha256: "unverified".to_string(),
                installed_at: 1,
                active: true,
                manifest: "not-json".to_string(),
                kind: blazar_core::engine_kind::EngineKind::Whisper,
            })
            .expect("row");
        let (bin, dir) = server_bin(&dirs).expect("server");
        assert_eq!(dir, lane_dir);
        assert!(bin.starts_with(&lane_dir));
        assert_ne!(dir, legacy);
        assert!(!is_legacy_bin(&dirs, &bin));
    }

    #[test]
    fn unit__server_bin__degenerate_engines_row_falls_back_to_legacy() {
        // A row whose binary vanished must never brick the audio lane:
        // resolution falls through to the legacy tree.
        let tmp = tempfile::tempdir().expect("tmp");
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        let newest = stage_server(&dirs, "v2.0.0");
        std::fs::create_dir_all(bin_root(&dirs)).expect("root");
        let store = blazar_core::Store::open(&dirs).expect("store");
        store
            .upsert_engine(&blazar_core::store::EngineRow {
                tag: "b0001".to_string(),
                asset: "whisper-bin-ubuntu-x64.tar.gz".to_string(),
                sha256: "unverified".to_string(),
                installed_at: 1,
                active: true,
                manifest: "not-json".to_string(),
                kind: blazar_core::engine_kind::EngineKind::Whisper,
            })
            .expect("row");
        // No engines-tree dir staged: the row resolves to nothing.
        let (bin, dir) = server_bin(&dirs).expect("server");
        assert_eq!(dir, newest);
        assert!(is_legacy_bin(&dirs, &bin));
    }

    #[test]
    fn unit__set_pin__round_trip_selects_and_unpins() {
        let tmp = tempfile::tempdir().expect("tmp");
        let dirs = BlazarDirs {
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
        let dirs = BlazarDirs {
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
        let dirs = BlazarDirs {
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
        assert_eq!(installed_tags(&dirs).len(), crate::engine::KEEP_TAGS);
    }

    #[test]
    fn unit__prune__keeps_newest_keep_tags_and_pinned() {
        let tmp = tempfile::tempdir().expect("tmp");
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        for t in ["v1.1.0", "v1.2.0", "v1.3.0", "v1.4.0", "v1.5.0"] {
            stage_server(&dirs, t);
        }
        std::fs::create_dir_all(bin_root(&dirs)).expect("root");
        std::fs::write(pin_path(&dirs), "v1.1.0\n").expect("pin");
        prune(&dirs).expect("prune");
        // Newest KEEP_TAGS (newest-first list) plus the pinned old tag.
        let expected: Vec<String> = ["v1.5.0", "v1.4.0", "v1.3.0", "v1.2.0"]
            .into_iter()
            .take(crate::engine::KEEP_TAGS)
            .chain(["v1.1.0"])
            .map(String::from)
            .collect();
        assert_eq!(installed_tags(&dirs), expected);
    }

    #[test]
    fn unit__prune__semver_order_not_alphabetical() {
        let tmp = tempfile::tempdir().expect("tmp");
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        // Alphabetical order would call v1.9.0 "newest" and prune v1.10.0.
        for t in ["v1.7.0", "v1.8.0", "v1.9.0", "v1.10.0"] {
            stage_server(&dirs, t);
        }
        prune(&dirs).expect("prune");
        let expected: Vec<String> = ["v1.10.0", "v1.9.0", "v1.8.0", "v1.7.0"]
            .into_iter()
            .take(crate::engine::KEEP_TAGS)
            .map(String::from)
            .collect();
        assert_eq!(installed_tags(&dirs), expected);
    }

    /// Full wiremock cycle: `--tag` install pins, plain install unpins,
    /// prune keeps the newest `KEEP_TAGS`.
    #[tokio::test]
    #[allow(clippy::too_many_lines)] // one full install cycle, splitting hides the wire flow
    async fn install__tag_pins_latest_unpins_prunes() {
        async fn mount(
            api: &wiremock::MockServer,
            endpoint: &str,
            tag: &str,
            asset: &str,
            bytes: &[u8],
        ) {
            mount_list(api, endpoint, tag, asset, bytes, false).await;
        }

        // `install(None)` resolves the channel through the paginated
        // `repos/{repo}/releases` list endpoint (release_for_channel), so the
        // latest-lane mock must serve an array, not the single-object
        // `/releases/latest` shape.
        async fn mount_list(
            api: &wiremock::MockServer,
            endpoint: &str,
            tag: &str,
            asset: &str,
            bytes: &[u8],
            as_array: bool,
        ) {
            use wiremock::matchers::{method, path};
            let release = serde_json::json!({
                "tag_name": tag,
                "prerelease": false,
                "assets": [{
                    "name": asset,
                    "digest": format!("sha256:{:x}", sha2::Sha256::digest(bytes)),
                    "size": bytes.len(),
                    "browser_download_url": format!("{}/download/{}/{}", api.uri(), tag, asset),
                }]
            });
            let body = if as_array {
                serde_json::json!([release])
            } else {
                release
            };
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
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        let api = wiremock::MockServer::start().await;

        let archive = |tag: &str| -> Vec<u8> {
            use std::io::Write as _;
            let root = format!("whisper-{tag}");
            // Upstream ships zips to Windows and tar.gz everywhere else;
            // the fixture mirrors the format the host lane downloads.
            if cfg!(windows) {
                let mut buf = std::io::Cursor::new(Vec::new());
                let mut z = zip::ZipWriter::new(&mut buf);
                let opts: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default();
                z.add_directory(root.clone(), opts).unwrap();
                // The install lane probes the platform-specific binary name
                // (server_bin_in), so the fixture must carry it too.
                let bin = if cfg!(windows) {
                    "whisper-server.exe"
                } else {
                    "whisper-server"
                };
                z.start_file(format!("{root}/{bin}"), opts).unwrap();
                z.write_all(b"stub").unwrap();
                z.finish().unwrap();
                return buf.into_inner();
            }
            let mut tarbuf = Vec::new();
            {
                let mut builder = tar::Builder::new(&mut tarbuf);
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

        let v181 = archive("v1.8.1");
        mount(
            &api,
            "/repos/ggml-org/whisper.cpp/releases/tags/v1.8.1",
            "v1.8.1",
            asset,
            &v181,
        )
        .await;
        let v190 = archive("v1.9.0");
        mount_list(
            &api,
            "/repos/ggml-org/whisper.cpp/releases",
            "v1.9.0",
            asset,
            &v190,
            true,
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

        // Older staged dirs + a latest install: unpins, prunes to the
        // KEEP_TAGS newest.
        for t in ["v1.7.0", "v1.6.0", "v1.5.0"] {
            stage_server(&dirs, t);
        }
        let tag = install(&gh, &dirs, None, false)
            .await
            .expect("latest install");
        assert_eq!(tag, "v1.9.0");
        assert_eq!(pinned_tag(&dirs), None);
        let expected: Vec<String> = ["v1.9.0", "v1.8.1", "v1.7.0", "v1.6.0", "v1.5.0"]
            .into_iter()
            .take(crate::engine::KEEP_TAGS)
            .map(String::from)
            .collect();
        assert_eq!(installed_tags(&dirs), expected);
    }

    #[test]
    fn unit__installed_tag__engines_row_tag_never_leaks_lib_dir_name() {
        let tmp = tempfile::tempdir().expect("tmp");
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        // Engines-lane row with a nested extract subdir — the subdir name
        // (whisper-bin-ubuntu-x64) is exactly what must never reach a
        // currency verdict (live: an unclearable "update available").
        let nested = dirs.engines_dir().join("b5130/whisper-bin-ubuntu-x64");
        std::fs::create_dir_all(&nested).expect("nested");
        std::fs::write(nested.join("whisper-server"), "#!/bin/sh\n").expect("bin");
        let store = blazar_core::Store::open(&dirs).expect("store");
        store
            .upsert_engine(&blazar_core::store::EngineRow {
                tag: "b5130".into(),
                asset: "cpu".into(),
                sha256: "unverified".into(),
                installed_at: 1,
                active: true,
                manifest: "{}".into(),
                kind: blazar_core::engine_kind::EngineKind::Whisper,
            })
            .expect("row");

        assert_eq!(installed_tag(&dirs).as_deref(), Some("b5130"));
        assert!(engines_lane_installed(&dirs));
        // The serving binary's dir really is the nested one — the tag
        // derivation must not simply mirror server_bin's second element.
        let (bin, lib) = server_bin(&dirs).expect("serving");
        assert!(bin.ends_with("whisper-server"));
        assert!(lib.ends_with("whisper-bin-ubuntu-x64"));
    }

    #[test]
    fn unit__installed_tag__legacy_tree_without_rows_falls_back_to_tag_dirs() {
        let tmp = tempfile::tempdir().expect("tmp");
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        let legacy = bin_root(&dirs).join("v1.2.0");
        std::fs::create_dir_all(&legacy).expect("legacy dir");
        std::fs::write(legacy.join("whisper-server"), "#!/bin/sh\n").expect("bin");

        assert_eq!(installed_tag(&dirs).as_deref(), Some("v1.2.0"));
        assert!(!engines_lane_installed(&dirs));
    }
}
