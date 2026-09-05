//! Engine lifecycle: install upstream llama-server builds (sha-verified),
//! probe capabilities, activate/rollback, prune old tags. A local build
//! registers as pseudo-tag `local` and is never pruned.

pub mod gh;
pub mod manifest;

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};

use pallama_core::store::{EngineRow, Store};
use pallama_core::PallamaDirs;

use crate::events::{EventBus, PallamaEvent};
use gh::{GhClient, GhRelease};
use manifest::Manifest;

pub const KEEP_TAGS: usize = 3;
pub const LOCAL_TAG: &str = "local";

pub struct EngineManager {
    pub dirs: PallamaDirs,
    pub gh: GhClient,
    pub bus: EventBus,
    pub asset_override: String,
}

/// System-level GPU vendor hint used to pick the first engine asset
/// (before any engine exists to probe). Filesystem checks — no GPU libs
/// loaded, works on every distro.
#[must_use] 
pub fn system_vendor_hint() -> manifest::Vendor {
    let nvidia = Path::new("/proc/driver/nvidia").exists() || which_first(&["nvidia-smi"]);
    if nvidia {
        return manifest::Vendor::Nvidia;
    }
    if Path::new("/opt/rocm").exists() || Path::new("/dev/kfd").exists() {
        return manifest::Vendor::Amd;
    }
    // Intel oneAPI runtime marker
    if Path::new("/opt/intel/oneapi").exists() {
        return manifest::Vendor::Intel;
    }
    manifest::Vendor::Other
}

fn which_first(names: &[&str]) -> bool {
    names.iter().any(|n| {
        std::env::var_os("PATH")
            .is_some_and(|paths| {
                std::env::split_paths(&paths).any(|dir| dir.join(n).exists())
            })
    })
}

impl EngineManager {
    /// Install the requested (or newest) b-tag build and activate it.
    pub async fn update(&self, tag: Option<&str>) -> Result<EngineRow> {
        let release = match tag {
            Some(t) => self.gh.resolve_tag(t).await?,
            None => self.gh.latest_b_release().await?,
        };
        let tag_name = release.tag_name.clone();
        let suffix = self.pick_suffix(&release)?;
        self.install(&release, &suffix).await
            .with_context(|| format!("install {tag_name} asset {suffix}"))
    }

    /// Preference-ordered asset suffix for this machine, honoring
    /// `engine_asset` override ("auto" = detect).
    fn pick_suffix(&self, release: &GhRelease) -> Result<String> {
        if self.asset_override != "auto" && !self.asset_override.is_empty() {
            let name = gh::asset_filename(&release.tag_name, &self.asset_override);
            if release.assets.iter().any(|a| a.name == name) {
                return Ok(self.asset_override.clone());
            }
            return Err(anyhow!(
                "asset {} not present in release {}; available: {}",
                name,
                release.tag_name,
                release.assets.iter().map(|a| a.name.as_str()).collect::<Vec<_>>().join(", ")
            ));
        }
        let os = std::env::consts::OS;
        let arch = std::env::consts::ARCH;
        let vendor = system_vendor_hint();
        for suffix in gh::pick_asset(os, arch, Some(vendor)) {
            let name = gh::asset_filename(&release.tag_name, suffix);
            if release.assets.iter().any(|a| a.name == name) {
                return Ok(suffix.to_string());
            }
        }
        Err(anyhow!(
            "no usable asset for {os}/{arch} in release {}",
            release.tag_name
        ))
    }

    /// Download, verify, extract, probe, store, activate, prune.
    pub async fn install(&self, release: &GhRelease, suffix: &str) -> Result<EngineRow> {
        let name = gh::asset_filename(&release.tag_name, suffix);
        let asset = release
            .assets
            .iter()
            .find(|a| a.name == name)
            .ok_or_else(|| anyhow!("asset {name} missing from release {}", release.tag_name))?;
        let bytes = self.gh.download_asset_bytes(asset).await?;
        let digest = asset
            .digest
            .clone()
            .and_then(|d| d.strip_prefix("sha256:").map(str::to_string))
            .unwrap_or_else(|| "unverified".into());

        let dir = self.dirs.engines_dir().join(&release.tag_name);
        if dir.exists() {
            std::fs::remove_dir_all(&dir).context("replace existing engine dir")?;
        }
        std::fs::create_dir_all(&dir)?;
        extract_archive(&bytes, &dir, &name)?;
        let server = find_server(&dir)?;
        make_executable(&server);

        let m = manifest::probe(&server, &release.tag_name)?;
        if m.devices.is_empty() && suffix.contains("vulkan") || m.devices.is_empty() && suffix.contains("cuda") {
            tracing::warn!(
                "engine {} ({suffix}) reports 0 GPUs; a CPU asset may serve you better",
                release.tag_name
            );
        }
        let row = EngineRow {
            tag: release.tag_name.clone(),
            asset: suffix.to_string(),
            sha256: digest,
            installed_at: now_secs(),
            active: false,
            manifest: serde_json::to_string(&m)?,
        };
        let store = Store::open(&self.dirs)?;
        store.upsert_engine(&row)?;
        store.set_active_engine(&release.tag_name)?;
        self.bus.publish(PallamaEvent::EngineUpdated { tag: release.tag_name.clone() });
        self.prune(&store)?;
        Ok(row)
    }

    /// Activate an installed tag by name.
    pub fn use_tag(&self, tag: &str) -> Result<EngineRow> {
        let store = Store::open(&self.dirs)?;
        store.set_active_engine(tag)?;
        self.bus.publish(PallamaEvent::EngineUpdated { tag: tag.to_string() });
        store
            .list_engines()?
            .into_iter()
            .find(|e| e.tag == tag)
            .ok_or_else(|| anyhow!("tag {tag} vanished"))
    }

    /// Step back to the previous tag by install time.
    pub fn rollback(&self) -> Result<EngineRow> {
        let store = Store::open(&self.dirs)?;
        let engines = store.list_engines()?;
        let active_idx = engines
            .iter()
            .position(|e| e.active)
            .ok_or_else(|| anyhow!("no active engine to roll back from"))?;
        if active_idx + 1 >= engines.len() {
            return Err(anyhow!(
                "no older engine to roll back to (active: {})",
                engines[active_idx].tag
            ));
        }
        let target = engines[active_idx + 1].tag.clone();
        drop(store);
        self.use_tag(&target)
    }

    /// Keep the newest `KEEP_TAGS` engines; `local` and the active tag are
    /// never pruned.
    pub fn prune(&self, store: &Store) -> Result<()> {
        let engines = store.list_engines()?; // newest first
        let active = engines.iter().find(|e| e.active).map(|e| e.tag.clone());
        for e in engines.iter().skip(KEEP_TAGS) {
            if e.tag == LOCAL_TAG || Some(&e.tag) == active.as_ref() {
                continue;
            }
            let dir = self.dirs.engines_dir().join(&e.tag);
            if dir.exists() {
                std::fs::remove_dir_all(&dir)
                    .with_context(|| format!("prune engine dir {}", dir.display()))?;
            }
            store.delete_engine(&e.tag)?;
            tracing::info!("pruned old engine {}", e.tag);
            self.bus.publish(PallamaEvent::EngineRemoved { tag: e.tag.clone() });
        }
        Ok(())
    }

    /// Register a locally built llama-server (`PALLAMA_ENGINE_PATH`) under the
    /// pseudo-tag `local`. Never pruned; activation follows `use_tag`.
    pub fn register_local(&self, server: &Path) -> Result<EngineRow> {
        if !server.exists() {
            return Err(anyhow!("PALLAMA_ENGINE_PATH {} does not exist", server.display()));
        }
        let m = manifest::probe(server, LOCAL_TAG)?;
        let row = EngineRow {
            tag: LOCAL_TAG.to_string(),
            asset: "local".into(),
            sha256: "local".into(),
            installed_at: now_secs(),
            active: false,
            manifest: serde_json::to_string(&m)?,
        };
        let store = Store::open(&self.dirs)?;
        store.upsert_engine(&row)?;
        Ok(row)
    }

    /// Active engine's manifest (probed capabilities).
    pub fn active_manifest(&self) -> Result<Option<Manifest>> {
        let store = Store::open(&self.dirs)?;
        let Some(row) = store.active_engine()? else { return Ok(None) };
        let m: Manifest = serde_json::from_str(&row.manifest)
            .with_context(|| format!("decode manifest for {}", row.tag))?;
        Ok(Some(m))
    }
}

fn now_secs() -> i64 {
    i64::try_from(SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs())
        .unwrap_or(i64::MAX)
}

/// Extract tar.gz or zip into `dir` (strip nothing; roots are discovered).
#[allow(clippy::case_sensitive_file_extension_comparisons)] // exact upstream asset names
fn extract_archive(bytes: &[u8], dir: &Path, asset_name: &str) -> Result<()> {
    if asset_name.ends_with(".zip") {
        let reader = std::io::Cursor::new(bytes);
        let mut zip = zip::ZipArchive::new(reader).context("open zip")?;
        zip.extract(dir).context("extract zip")?;
        return Ok(());
    }
    let gz = flate2::read::GzDecoder::new(bytes);
    let mut archive = tar::Archive::new(gz);
    archive.set_preserve_permissions(true);
    archive.unpack(dir).context("extract tar.gz")?;
    Ok(())
}

/// Find the llama-server binary anywhere under the extracted dir
/// (release archives use `llama-<tag>/llama-server` roots).
fn find_server(dir: &Path) -> Result<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.file_name().is_some_and(|n| {
                n == "llama-server" || n == "llama-server.exe"
            }) {
                out.push(p);
            }
        }
    }
    let mut found = Vec::new();
    walk(dir, &mut found);
    found
        .into_iter()
        .min_by_key(|p| p.components().count())
        .ok_or_else(|| anyhow!("no llama-server binary inside extracted archive"))
}

fn make_executable(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = std::fs::metadata(path) {
            let mut perms = meta.permissions();
            perms.set_mode(perms.mode() | 0o755);
            let _ = std::fs::set_permissions(path, perms);
        }
    }
    #[cfg(not(unix))]
    let _ = path;
}
