//! Engine lifecycle: install upstream llama-server builds (sha-verified),
//! probe capabilities, activate/rollback, prune old tags. A local build
//! registers as pseudo-tag `local` and is never pruned.

pub mod build;
pub mod gh;
pub mod manifest;

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};

use pallama_core::config::UpdateChannel;
use pallama_core::engine_kind::EngineKind;
use pallama_core::store::{EngineRow, Store};
use pallama_core::PallamaDirs;

use crate::events::{EventBus, PallamaEvent};
use gh::{GhClient, GhRelease};
use manifest::Manifest;

pub const KEEP_TAGS: usize = 3;
pub const LOCAL_TAG: &str = "local";
/// Wait between asset-list re-fetches while a fresh release finishes
/// uploading (observed: full asset matrix lands ~75-120 s after publish).
pub const ASSET_UPLOAD_RETRY_DELAY: std::time::Duration = std::time::Duration::from_secs(20);
pub const ASSET_UPLOAD_RETRY_ATTEMPTS: usize = 6;
/// A release younger than this is considered "still uploading".
pub const FRESH_RELEASE_SECS: i64 = 600;

fn release_is_fresh(release: &GhRelease) -> bool {
    release
        .published_epoch()
        .is_some_and(|t| now_secs().saturating_sub(t) < FRESH_RELEASE_SECS)
}

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
            .is_some_and(|paths| std::env::split_paths(&paths).any(|dir| dir.join(n).exists()))
    })
}

impl EngineManager {
    /// Install the requested tag (or the channel's target) and activate it.
    pub async fn update(&self, tag: Option<&str>, channel: UpdateChannel) -> Result<EngineRow> {
        self.update_with_retry_delay(tag, channel, ASSET_UPLOAD_RETRY_DELAY)
            .await
    }

    /// `update` with an injectable inter-attempt delay (tests pin the
    /// fresh-release wait without sleeping 20 s).
    pub async fn update_with_retry_delay(
        &self,
        tag: Option<&str>,
        channel: UpdateChannel,
        retry_delay: std::time::Duration,
    ) -> Result<EngineRow> {
        let release = match tag {
            Some(t) => self.gh.resolve_tag(t).await?,
            None => self.gh.channel_b_release(channel).await?,
        };
        self.install_with_retries(release, retry_delay).await
    }

    /// Install an already-resolved release (single-fetch entry for callers
    /// that needed the `GhRelease` up front, e.g. downgrade gating).
    pub async fn update_resolved(&self, release: GhRelease) -> Result<EngineRow> {
        self.install_with_retries(release, ASSET_UPLOAD_RETRY_DELAY)
            .await
    }

    /// Fresh-asset retry loop shared by every update entry point.
    async fn install_with_retries(
        &self,
        mut release: GhRelease,
        retry_delay: std::time::Duration,
    ) -> Result<EngineRow> {
        let tag_name = release.tag_name.clone();
        for attempt in 0..=ASSET_UPLOAD_RETRY_ATTEMPTS {
            match self.pick(&release)? {
                Some(pick) => {
                    let last = attempt == ASSET_UPLOAD_RETRY_ATTEMPTS;
                    if pick.cpu_fallback && release_is_fresh(&release) && !last {
                        // Fresh release: the GPU asset is probably still
                        // uploading. Wait for it instead of degrading now.
                        tracing::warn!(
                            "release {} is fresh and its GPU asset is not up yet; \
                             waiting {:?} for the upload (attempt {}/{})",
                            tag_name,
                            retry_delay,
                            attempt + 1,
                            ASSET_UPLOAD_RETRY_ATTEMPTS + 1
                        );
                        tokio::time::sleep(retry_delay).await;
                        release = self.gh.release_by_tag(&tag_name).await?;
                        continue;
                    }
                    if pick.cpu_fallback {
                        tracing::warn!(
                            "no GPU asset for this machine in release {}; installed the \
                             CPU build ({}) instead — re-run `pallama engine update` \
                             later to pick up the GPU build",
                            tag_name,
                            pick.label
                        );
                    }
                    return self
                        .install_picked(&release, &pick)
                        .await
                        .with_context(|| format!("install {tag_name} asset {}", pick.label));
                }
                None if release_is_fresh(&release) && attempt < ASSET_UPLOAD_RETRY_ATTEMPTS => {
                    tracing::warn!(
                        "release {} published moments ago and its assets are still \
                         uploading; waiting {:?} (attempt {}/{})",
                        tag_name,
                        retry_delay,
                        attempt + 1,
                        ASSET_UPLOAD_RETRY_ATTEMPTS + 1
                    );
                    tokio::time::sleep(retry_delay).await;
                    release = self.gh.release_by_tag(&tag_name).await?;
                }
                None => {
                    return Err(anyhow!(
                        "no usable asset for {}/{} in release {} (assets may still be \
                         uploading; retry in a minute). available: {}",
                        std::env::consts::OS,
                        std::env::consts::ARCH,
                        tag_name,
                        release
                            .assets
                            .iter()
                            .map(|a| a.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
            }
        }
        unreachable!("retry loop always returns or errors")
    }

    /// Resolve the asset for this machine: explicit `engine_asset`
    /// override first (exact name, teaching error on miss), then the
    /// auto matrix against the release's actual assets.
    fn pick(&self, release: &GhRelease) -> Result<Option<gh::AssetPick>> {
        if self.asset_override != "auto" && !self.asset_override.is_empty() {
            let name = gh::asset_filename(&release.tag_name, &self.asset_override);
            if release.assets.iter().any(|a| a.name == name) {
                return Ok(Some(gh::AssetPick {
                    name,
                    label: self.asset_override.clone(),
                    cpu_fallback: false,
                }));
            }
            return Err(anyhow!(
                "asset {} not present in release {}; available: {}",
                name,
                release.tag_name,
                release
                    .assets
                    .iter()
                    .map(|a| a.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        let os = std::env::consts::OS;
        let arch = std::env::consts::ARCH;
        let vendor = system_vendor_hint();
        Ok(gh::resolve_asset(release, os, arch, Some(vendor)))
    }

    /// Download, verify, extract, then register (probe + store +
    /// activate + prune).
    pub async fn install_picked(
        &self,
        release: &GhRelease,
        pick: &gh::AssetPick,
    ) -> Result<EngineRow> {
        let asset = release
            .assets
            .iter()
            .find(|a| a.name == pick.name)
            .ok_or_else(|| {
                anyhow!(
                    "asset {} missing from release {}",
                    pick.name,
                    release.tag_name
                )
            })?;
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
        extract_archive(&bytes, &dir, &pick.name)?;
        self.register_engine(
            &dir,
            &release.tag_name,
            &pick.label,
            &digest,
            EngineKind::LlamaCpp,
        )
    }

    /// Install one mistralrs release asset. Same shape as
    /// `install_picked` but file-streamed (CUDA assets are GiB-class)
    /// and registered as the mistralrs engine kind.
    pub async fn install_picked_mistralrs(
        &self,
        release: &GhRelease,
        pick: &gh::AssetPick,
    ) -> Result<EngineRow> {
        let asset = release
            .assets
            .iter()
            .find(|a| a.name == pick.name)
            .ok_or_else(|| {
                anyhow!(
                    "asset {} missing from release {}",
                    pick.name,
                    release.tag_name
                )
            })?;
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
        let archive = dir.join(&pick.name);
        self.gh.download_asset_file(asset, &archive).await?;
        let extracted = extract_archive_file(&archive, &dir, &pick.name);
        std::fs::remove_file(&archive).context("remove downloaded archive")?;
        extracted?;
        if pick.cpu_fallback {
            tracing::warn!(
                "installed the CPU mistralrs asset {} — this machine's driver/GPU \
                 does not qualify for a CUDA prebuilt; expect CPU-only speed",
                pick.name
            );
        }
        self.register_engine(
            &dir,
            &release.tag_name,
            &pick.label,
            &digest,
            EngineKind::MistralRs,
        )
    }

    /// Resolve and install a mistralrs release: explicit `vX.Y.Z` tag or
    /// latest. Asset choice derives from the live driver CUDA version +
    /// compute cap (never a hardcoded compatibility matrix); a fresh
    /// release whose assets have not fully landed yet is retried on the
    /// same cadence as the llama.cpp lane.
    pub async fn update_mistralrs(&self, tag: Option<&str>) -> Result<EngineRow> {
        let release = if let Some(t) = tag {
            // Exact mistral.rs tag (vX.Y.Z); partial tags are rejected by
            // the repo API with a teaching error naming the release.
            self.gh.release_by_tag_repo(gh::MISTRALRS_REPO, t).await?
        } else {
            // The list endpoint trims `assets` — resolve the newest v-tag
            // there, then single-fetch the full release (same resolved-
            // release shape as the llamacpp update lane).
            let latest = self.gh.latest_mistralrs_release().await?;
            self.gh
                .release_by_tag_repo(gh::MISTRALRS_REPO, &latest.tag_name)
                .await?
        };
        let (driver_cuda, compute_cap) = build::nvidia_gpu_facts().await;
        let os = std::env::consts::OS;
        let arch = std::env::consts::ARCH;
        let vendor = system_vendor_hint();

        let mut last_missing = None;
        for attempt in 0..=ASSET_UPLOAD_RETRY_ATTEMPTS {
            let picks =
                gh::mistralrs_asset_picks(os, arch, Some(vendor), driver_cuda, compute_cap)?;
            if let Some(pick) = gh::resolve_mistralrs_asset(&release, &picks) {
                return self.install_picked_mistralrs(&release, &pick).await;
            }
            last_missing = Some(picks);
            if !release_is_fresh(&release) || attempt == ASSET_UPLOAD_RETRY_ATTEMPTS {
                break;
            }
            tracing::info!(
                "mistral.rs {} assets still uploading; retry {}/{} in {:?}",
                release.tag_name,
                attempt + 1,
                ASSET_UPLOAD_RETRY_ATTEMPTS,
                ASSET_UPLOAD_RETRY_DELAY
            );
            tokio::time::sleep(ASSET_UPLOAD_RETRY_DELAY).await;
        }
        Err(anyhow!(
            "no usable mistralrs asset in release {} (wanted one of: {}; available: {})",
            release.tag_name,
            last_missing.map_or_else(
                || "n/a".into(),
                |p| {
                    p.iter()
                        .map(|x| x.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                },
            ),
            release
                .assets
                .iter()
                .map(|a| a.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ))
    }

    /// Shared install tail for every engine source (release asset,
    /// source build): probe the binary, warn on GPU-asset-sees-no-GPU,
    /// store the row, activate it, publish, prune old tags.
    pub fn register_engine(
        &self,
        dir: &Path,
        tag: &str,
        asset_label: &str,
        sha256: &str,
        kind: EngineKind,
    ) -> Result<EngineRow> {
        let server = match kind {
            EngineKind::LlamaCpp => find_server(dir)?,
            EngineKind::MistralRs => find_engine_binary(dir, &["mistralrs", "mistralrs.exe"])?,
        };
        make_executable(&server);

        let m = manifest::probe_kind(&server, tag, &kind)?;
        match kind {
            EngineKind::LlamaCpp => {
                if m.devices.is_empty()
                    && (asset_label.contains("vulkan") || asset_label.contains("cuda"))
                {
                    tracing::warn!(
                        "engine {} ({}) reports 0 GPUs; a CPU asset may serve you better",
                        tag,
                        asset_label
                    );
                }
            }
            // mistralrs builds cannot enumerate devices; GPU use follows
            // from the asset label (cuda*/metal), so there is nothing to
            // cross-check here — an unusable GPU build fails at spawn.
            EngineKind::MistralRs => {
                tracing::debug!(target: "pallama::engine", "registered mistralrs {tag} ({asset_label}); device use follows the asset label");
            }
        }
        let row = EngineRow {
            tag: tag.to_string(),
            asset: asset_label.to_string(),
            sha256: sha256.to_string(),
            installed_at: now_secs(),
            active: false,
            manifest: serde_json::to_string(&m)?,
            kind,
        };
        let store = Store::open(&self.dirs)?;
        store.upsert_engine(&row)?;
        store.set_active_engine(tag)?;
        self.bus.publish(PallamaEvent::EngineUpdated {
            tag: tag.to_string(),
        });
        self.prune(&store)?;
        // The flip above happened after `row` was built; the caller's
        // contract ("install activates") expects the returned row to
        // reflect the post-install store state.
        let row = EngineRow {
            active: true,
            ..row
        };
        Ok(row)
    }

    /// Activate an installed tag by name.
    pub fn use_tag(&self, tag: &str) -> Result<EngineRow> {
        let store = Store::open(&self.dirs)?;
        store.set_active_engine(tag)?;
        self.bus.publish(PallamaEvent::EngineUpdated {
            tag: tag.to_string(),
        });
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
            self.bus
                .publish(PallamaEvent::EngineRemoved { tag: e.tag.clone() });
        }
        Ok(())
    }

    /// Register a locally built llama-server (`PALLAMA_ENGINE_PATH`) under the
    /// pseudo-tag `local`. Never pruned; activation follows `use_tag`.
    pub fn register_local(
        &self,
        server: &Path,
        extra_env: &std::collections::BTreeMap<String, String>,
    ) -> Result<EngineRow> {
        if !server.exists() {
            return Err(anyhow!(
                "PALLAMA_ENGINE_PATH {} does not exist",
                server.display()
            ));
        }
        // Probe under the configured engine env (e.g. GGML_BACKEND_PATH so
        // a CUDA build actually discovers its GPU).
        let m = {
            let prev: Vec<(String, String)> = extra_env
                .iter()
                .filter(|(k, _)| std::env::var_os(k).is_none())
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            for (k, v) in &prev {
                std::env::set_var(k, v);
            }
            let r = manifest::probe(server, LOCAL_TAG);
            for (k, _) in &prev {
                std::env::remove_var(k);
            }
            r?
        };
        let row = EngineRow {
            tag: LOCAL_TAG.to_string(),
            asset: "local".into(),
            sha256: "local".into(),
            installed_at: now_secs(),
            active: false,
            manifest: serde_json::to_string(&m)?,
            kind: EngineKind::LlamaCpp,
        };
        let store = Store::open(&self.dirs)?;
        store.upsert_engine(&row)?;
        Ok(row)
    }

    /// Active engine's manifest (probed capabilities).
    pub fn active_manifest(&self) -> Result<Option<Manifest>> {
        let store = Store::open(&self.dirs)?;
        let Some(row) = store.active_engine()? else {
            return Ok(None);
        };
        let m: Manifest = serde_json::from_str(&row.manifest)
            .with_context(|| format!("decode manifest for {}", row.tag))?;
        Ok(Some(m))
    }
}

fn now_secs() -> i64 {
    i64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    )
    .unwrap_or(i64::MAX)
}

/// Extract tar.gz or zip into `dir` (strip nothing; roots are discovered).
#[allow(clippy::case_sensitive_file_extension_comparisons)] // exact upstream asset names
pub(crate) fn extract_archive(bytes: &[u8], dir: &Path, asset_name: &str) -> Result<()> {
    extract_reader(std::io::Cursor::new(bytes), dir, asset_name)
}

/// File-backed variant for assets too large to buffer (mistralrs CUDA
/// archives are GiB-class); mirrors `extract_archive` semantics.
pub(crate) fn extract_archive_file(path: &Path, dir: &Path, asset_name: &str) -> Result<()> {
    let f = std::fs::File::open(path)
        .with_context(|| format!("open downloaded archive {}", path.display()))?;
    extract_reader(f, dir, asset_name)
}

fn extract_reader<R: std::io::Read + std::io::Seek>(
    r: R,
    dir: &Path,
    asset_name: &str,
) -> Result<()> {
    if std::path::Path::new(asset_name)
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("zip"))
    {
        let mut zip = zip::ZipArchive::new(r).context("open zip")?;
        zip.extract(dir).context("extract zip")?;
        return Ok(());
    }
    let gz = flate2::read::GzDecoder::new(r);
    let mut archive = tar::Archive::new(gz);
    archive.set_preserve_permissions(true);
    archive.unpack(dir).context("extract tar.gz")?;
    Ok(())
}

/// Find the llama-server binary anywhere under the extracted dir
/// (release archives use `llama-<tag>/llama-server` roots).
pub(crate) fn find_server(dir: &Path) -> Result<PathBuf> {
    find_engine_binary(dir, &["llama-server", "llama-server.exe"])
}

/// Find an engine binary by exact file name anywhere under `dir`,
/// shallowest match first.
pub(crate) fn find_engine_binary(dir: &Path, names: &[&str]) -> Result<PathBuf> {
    fn walk(dir: &Path, names: &[&str], out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, names, out);
            } else if p
                .file_name()
                .is_some_and(|n| names.iter().any(|want| n == *want))
            {
                out.push(p);
            }
        }
    }
    let mut found = Vec::new();
    walk(dir, names, &mut found);
    found
        .into_iter()
        .min_by_key(|p| p.components().count())
        .ok_or_else(|| {
            anyhow!(
                "no {} binary inside extracted archive",
                names.first().copied().unwrap_or("engine")
            )
        })
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
