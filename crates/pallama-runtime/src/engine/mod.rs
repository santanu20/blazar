//! Engine lifecycle: install upstream llama-server builds (sha-verified),
//! probe capabilities, activate/rollback, prune old tags. A local build
//! registers as pseudo-tag `local` and is never pruned.

pub mod build;
pub mod gh;
pub mod manifest;
pub mod sglang_install;

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

/// Retention for engine dirs: the newest 2 survive auto-prune — the
/// fresh build plus one rollback anchor (~215 MiB each). `local` and the
/// active tag are always kept on top of this.
pub const KEEP_TAGS: usize = 2;

/// Recursive byte size of an engine dir (for the update-prune summary).
fn engine_dir_bytes(p: &Path) -> u64 {
    let Ok(rd) = std::fs::read_dir(p) else {
        return 0;
    };
    let mut n = 0u64;
    for e in rd.flatten() {
        match e.file_type() {
            Ok(ft) if ft.is_dir() => n += engine_dir_bytes(&e.path()),
            Ok(_) => n += e.metadata().map_or(0, |m| m.len()),
            Err(_) => {}
        }
    }
    n
}
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

/// Dry-run report for `pallama engine update --check` (see
/// [`EngineManager::check_lane`]): what an update would target for this
/// box, with nothing fetched beyond release metadata.
#[derive(Debug, Clone)]
pub struct LaneCheck {
    /// Channel target tag (e.g. `b10985`).
    pub target_tag: String,
    /// Derived overlay tag an NVIDIA/linux box would install
    /// (`b10985-cuda`); `None` when the CUDA lane does not apply
    /// (asset pin, non-NVIDIA, or pre-CUDA-12 driver).
    pub overlay_tag: Option<String>,
    /// Prebuilt asset this box would install from that overlay.
    pub cuda_asset: Option<gh::AssetPick>,
    /// Newest CUDA toolkit the overlay was built with, when the overlay
    /// exists but nothing fits this driver.
    pub newest_cuda: Option<(u32, u32)>,
    /// This box's driver CUDA version (`None` = no NVIDIA driver).
    pub driver_cuda: Option<(u32, u32)>,
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

/// Is this engine row a CUDA build? Tag convention (`bNNNN-cuda` overlay
/// and source-build tags) or the asset label (`ubuntu-cuda-12.8-x64`).
#[must_use]
pub fn is_cuda_engine(tag: &str, asset_label: &str) -> bool {
    tag.ends_with("-cuda") || asset_label.contains("cuda")
}

/// Pre-download mirror of the keep-CUDA activation guard: on a
/// Linux-x86_64-NVIDIA box whose active engine is a llama.cpp CUDA
/// build, a standard-lane (Vulkan) asset can only ever register
/// dormant — the guard in `register_engine_with_vendor` refuses to
/// activate it after the fact. Deciding the same thing BEFORE the
/// download skips the ~28 MiB fetch + probe of an engine that will
/// Result of the overlay-lag fallback probe (see
/// `overlay_lag_fallback`): which of the three lanes the run
/// actually took, so narration matches the decision.
enum LagOutcome {
    /// A published build older than the channel target was installed.
    Installed(EngineRow),
    /// The newest published runnable build IS the active engine —
    /// nothing to do until the overlay drops the next build.
    AlreadyActive(EngineRow),
    /// Nothing runnable published (or the probe failed): the
    /// standard Vulkan lane is the next resort.
    NothingRunnable,
}

/// never serve. The `engine_asset` config pin bypasses, mirroring
/// `maybe_cuda_overlay`.
#[must_use]
pub fn keep_cuda_skip_pred(
    active: Option<&EngineRow>,
    vendor: manifest::Vendor,
    os: &str,
    arch: &str,
    asset_override: &str,
) -> bool {
    let Some(active) = active else { return false };
    active.active
        && active.kind == EngineKind::LlamaCpp
        && is_cuda_engine(&active.tag, &active.asset)
        && vendor == manifest::Vendor::Nvidia
        && (os, arch) == ("linux", "x86_64")
        && (asset_override == "auto" || asset_override.is_empty())
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
        // Explicit `-cuda` pins address the overlay repo directly (our
        // CI publishes `bNNNN-cuda` releases there); every other tag
        // resolves upstream first and probes the overlay afterwards.
        if let Some(t) = tag {
            if t.ends_with("-cuda") {
                return self.install_cuda_overlay_tag(t).await;
            }
        }
        let release = match tag {
            Some(t) => self.gh.resolve_tag(t).await?,
            None => self.gh.channel_b_release(channel).await?,
        };
        if let Some(row) = self.maybe_cuda_overlay(&release, tag.is_some()).await? {
            return Ok(row);
        }
        // Channel automation never benefits from the standard (Vulkan)
        // lane while the keep-CUDA guard holds — explicit tag pins
        // still download so `pallama engine use <tag>` can reach them.
        if tag.is_none() {
            if let Some(row) = self.try_keep_cuda_skip(&release.tag_name, system_vendor_hint())? {
                return Ok(row);
            }
        }
        self.install_with_retries(release, retry_delay).await
    }

    /// Install an already-resolved release (single-fetch entry for callers
    /// that needed the `GhRelease` up front, e.g. downgrade gating).
    pub async fn update_resolved(&self, release: GhRelease, exact_pin: bool) -> Result<EngineRow> {
        self.update_resolved_with_vendor(release, system_vendor_hint(), exact_pin)
            .await
    }

    /// Read-only dry-run of the update lane for `engine update --check`:
    /// resolves exactly what an update WOULD target — the derived CUDA
    /// overlay tag and the prebuilt asset this box would install —
    /// without downloading, installing, or writing anything. The guards
    /// mirror `maybe_cuda_overlay` so the dry-run and the real lane can
    /// never disagree about applicability.
    pub async fn check_lane(&self, release: &GhRelease) -> Result<LaneCheck> {
        let mut out = LaneCheck {
            target_tag: release.tag_name.clone(),
            overlay_tag: None,
            cuda_asset: None,
            newest_cuda: None,
            driver_cuda: None,
        };
        if (self.asset_override != "auto" && !self.asset_override.is_empty())
            || std::env::consts::OS != "linux"
            || std::env::consts::ARCH != "x86_64"
            || system_vendor_hint() != manifest::Vendor::Nvidia
        {
            return Ok(out); // standard asset lane; no CUDA story to report
        }
        let Some(number) = gh::btag_number(&release.tag_name) else {
            return Ok(out); // non-b upstream tags never have overlays
        };
        let (driver_cuda, cc) = build::nvidia_gpu_facts().await;
        // Compute capability (8,9) -> sm 89, same as install_cuda_overlay_tag.
        let sm = cc.map(|(maj, min)| maj * 10 + min);
        out.driver_cuda = driver_cuda;
        let Some(dc) = driver_cuda else {
            return Ok(out);
        };
        if dc.0 < 12 {
            return Ok(out);
        }
        let overlay_tag = format!("b{number}-cuda");
        out.overlay_tag = Some(overlay_tag.clone());
        if let Ok(overlay) = self
            .gh
            .release_by_tag_repo(&gh::engine_overlay_repo(), &overlay_tag)
            .await
        {
            out.cuda_asset = gh::resolve_cuda_asset(&overlay, dc, sm);
            out.newest_cuda = gh::newest_asset_cuda(&overlay);
        }
        // A missing overlay release stays overlay_tag=Some + asset=None:
        // the CLI reports the hourly-cadence lag instead of pretending
        // the lane was evaluated.
        Ok(out)
    }

    /// `update_resolved` with an injectable vendor hint so the keep-CUDA
    /// skip is deterministically testable on any box (same injection
    /// pattern as `register_engine_with_vendor`). `exact_pin`: the
    /// release came from a user-pinned tag — the overlay-lag fallback
    /// must not swap an exact pin for an older build.
    pub async fn update_resolved_with_vendor(
        &self,
        release: GhRelease,
        vendor_hint: manifest::Vendor,
        exact_pin: bool,
    ) -> Result<EngineRow> {
        if let Some(row) = self.maybe_cuda_overlay(&release, exact_pin).await? {
            return Ok(row);
        }
        if let Some(row) = self.try_keep_cuda_skip(&release.tag_name, vendor_hint)? {
            return Ok(row);
        }
        self.install_with_retries(release, ASSET_UPLOAD_RETRY_DELAY)
            .await
    }

    /// When `keep_cuda_skip_pred` holds, skip the standard asset lane and
    /// return the kept-active CUDA row instead of downloading an engine
    /// that would register dormant.
    fn try_keep_cuda_skip(
        &self,
        release_tag: &str,
        vendor: manifest::Vendor,
    ) -> Result<Option<EngineRow>> {
        let Some(active) = Store::open(&self.dirs)?.active_engine()? else {
            return Ok(None);
        };
        if !keep_cuda_skip_pred(
            Some(&active),
            vendor,
            std::env::consts::OS,
            std::env::consts::ARCH,
            &self.asset_override,
        ) {
            return Ok(None);
        }
        tracing::warn!(
            "NVIDIA box with active CUDA engine {} — skipped the standard (Vulkan) asset \
             download for {release_tag}: the keep-CUDA guard would leave it dormant. Refresh \
             the CUDA lane with `pallama engine build cuda`, or pin engine_asset \
             = \"ubuntu-vulkan-x64\" in config.toml to force the Vulkan lane",
            active.tag
        );
        Ok(Some(active))
    }

    /// Direct install of an overlay `bNNNN-cuda` tag the user pinned
    /// explicitly (`pallama engine install b10896-cuda`).
    async fn install_cuda_overlay_tag(&self, tag: &str) -> Result<EngineRow> {
        let repo = gh::engine_overlay_repo();
        let release = self
            .gh
            .release_by_tag_repo(&repo, tag)
            .await
            .with_context(|| {
                format!(
                    "overlay release {tag} not found in {repo}; the repo's engine-cuda \
                     workflow publishes bNNNN-cuda releases (set {} to point at a fork)",
                    gh::ENGINE_OVERLAY_REPO_ENV
                )
            })?;
        let (driver_cuda, cc) = build::nvidia_gpu_facts().await;
        // Compute capability (8,9) -> sm 89 for the per-arch asset rank.
        let sm = cc.map(|(maj, min)| maj * 10 + min);
        let Some(driver_cuda) = driver_cuda else {
            return Err(anyhow!(
                "cannot select a CUDA overlay asset: no NVIDIA driver CUDA \
                 capability probed (nvidia-smi absent or failed)"
            ));
        };
        let pick = gh::resolve_cuda_asset(&release, driver_cuda, sm)
            .ok_or_else(|| anyhow!("no driver-runnable CUDA asset in overlay release {tag}"))?;
        self.install_picked(&release, &pick)
            .await
            .with_context(|| format!("install overlay {tag} asset {}", pick.label))
    }

    /// Prebuilt CUDA overlay: when this machine is Linux-x86_64-NVIDIA
    /// with a CUDA-capable driver and our CI has published a
    /// `bNNNN-cuda` release for the resolved upstream build, install
    /// that instead of the Vulkan asset (~4% decode uplift, bundled
    /// cudart/cublas, no toolkit needed). Any miss — release absent,
    /// asset not runnable — is a quiet return to the normal lane: the
    /// Vulkan path stays the universal fallback. Zero-touch: the
    /// overlay repo defaults to the project home; `PALLAMA_ENGINE_REPO`
    /// exists purely for forks.
    async fn maybe_cuda_overlay(
        &self,
        release: &GhRelease,
        exact_pin: bool,
    ) -> Result<Option<EngineRow>> {
        if self.asset_override != "auto" && !self.asset_override.is_empty() {
            return Ok(None); // explicit asset pin wins over every heuristic
        }
        if std::env::consts::OS != "linux" || std::env::consts::ARCH != "x86_64" {
            return Ok(None);
        }
        if system_vendor_hint() != manifest::Vendor::Nvidia {
            return Ok(None);
        }
        let repo = gh::engine_overlay_repo();
        let Some(number) = gh::btag_number(&release.tag_name) else {
            return Ok(None); // non-b upstream tags never have overlays
        };
        let (driver_cuda, cc) = build::nvidia_gpu_facts().await;
        // Compute capability (8,9) -> sm 89 for the per-arch asset rank.
        let sm = cc.map(|(maj, min)| maj * 10 + min);
        let Some(driver_cuda) = driver_cuda else {
            tracing::warn!(
                "NVIDIA GPU present but no CUDA driver capability probed \
                 (driver installed but not rebooted?); staying on the Vulkan \
                 lane — after a driver install, reboot then run: pallama \
                 engine update"
            );
            return Ok(None);
        };
        if driver_cuda.0 < 12 {
            tracing::warn!(
                "driver CUDA {}.{} predates CUDA 12 — the prebuilt CUDA lane \
                 is never eligible for it; staying on the Vulkan lane",
                driver_cuda.0,
                driver_cuda.1
            );
            return Ok(None);
        }
        let overlay_tag = format!("b{number}-cuda");
        let overlay = match self.gh.release_by_tag_repo(&repo, &overlay_tag).await {
            Ok(r) => r,
            Err(e) => {
                // Overlay-lag fallback: a channel update must never shunt
                // an NVIDIA user into the hour-class source lane just
                // because the overlay has not published the brand-new tag
                // yet — install the newest PUBLISHED build instead.
                tracing::warn!(
                    "no CUDA overlay release {overlay_tag} in {repo} yet \
                     ({e:#}) — probing published overlay builds as a fallback"
                );
                if !exact_pin {
                    match self.overlay_lag_fallback(driver_cuda, sm, number).await? {
                        LagOutcome::Installed(row) => return Ok(Some(row)),
                        LagOutcome::AlreadyActive(row) => {
                            tracing::warn!(
                                "overlay hasn't published {overlay_tag} yet ({repo} drops \
                                 hourly); newest published CUDA build {} is already active — \
                                 rerun `pallama engine update` after the next overlay drop, \
                                 or run `pallama engine build cuda` to compile {} locally now",
                                row.tag,
                                release.tag_name
                            );
                            return Ok(Some(row));
                        }
                        LagOutcome::NothingRunnable => {} // Vulkan warn below is truthful
                    }
                }
                tracing::warn!(
                    "overlay fallback found nothing runnable; using the \
                     Vulkan lane this update (the overlay publishes on an \
                     hourly cadence). Local CUDA for THIS driver: pallama \
                     engine build cuda"
                );
                return Ok(None);
            }
        };
        if let Some(pick) = gh::resolve_cuda_asset(&overlay, driver_cuda, sm) {
            tracing::info!(
                "installing prebuilt CUDA engine from {repo} {overlay_tag} ({})",
                pick.label
            );
            let row = self
                .install_picked(&overlay, &pick)
                .await
                .with_context(|| format!("install overlay {overlay_tag}"))?;
            Ok(Some(row))
        } else {
            let need = gh::newest_asset_cuda(&overlay)
                .map_or_else(|| "unknown".into(), |(a, b)| format!("{a}.{b}"));
            tracing::warn!(
                "CUDA overlay {overlay_tag} needs CUDA {need}; this driver \
                 runs {}.{} — staying on the Vulkan lane. Options: pallama \
                 engine build cuda (builds locally for THIS driver), or pin \
                 an older overlay tag: pallama engine install b<N>-cuda",
                driver_cuda.0,
                driver_cuda.1
            );
            Ok(None)
        }
    }

    /// Overlay-lag fallback for channel updates: the channel target is
    /// not published in the overlay yet, so install the newest PUBLISHED
    /// overlay build the driver can run — never newer than the target,
    /// never the already-active tag (no reinstall churn). A
    /// minutes-class download beats the hour-class source lane while
    /// the overlay's hourly freshness watcher catches up.
    ///
    /// The outcome is tri-state so the caller narrates each arm
    /// distinctly: "newest published build is already active" used to
    /// collapse into `None` and read as "found nothing runnable",
    /// printing a Vulkan-lane switch the keep-CUDA guard then
    /// cancelled — three contradictory decisions in one run.
    async fn overlay_lag_fallback(
        &self,
        driver_cuda: (u32, u32),
        sm: Option<u32>,
        target: u64,
    ) -> Result<LagOutcome> {
        let repo = gh::engine_overlay_repo();
        let releases = match self.gh.list_releases_repo(&repo).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("overlay fallback probe of {repo} failed: {e:#}");
                return Ok(LagOutcome::NothingRunnable);
            }
        };
        let Some(pick_rel) = gh::newest_runnable_overlay(&releases, driver_cuda, sm, target) else {
            return Ok(LagOutcome::NothingRunnable);
        };
        if let Some(row) = Store::open(&self.dirs)?.active_engine()? {
            if row.tag == pick_rel.tag_name {
                tracing::info!(
                    "newest runnable overlay build {} is already active",
                    pick_rel.tag_name
                );
                return Ok(LagOutcome::AlreadyActive(row));
            }
        }
        let Some(pick) = gh::resolve_cuda_asset(pick_rel, driver_cuda, sm) else {
            return Ok(LagOutcome::NothingRunnable); // consistency guard; selection pre-filtered
        };
        let behind = target.saturating_sub(gh::btag_number(&pick_rel.tag_name).unwrap_or(0));
        tracing::warn!(
            "overlay lags the channel target by {behind} build(s): installing \
             published {} instead ({} asset; fresh overlay builds land hourly)",
            pick_rel.tag_name,
            pick.label
        );
        let row = self
            .install_picked(pick_rel, &pick)
            .await
            .with_context(|| format!("install overlay fallback {}", pick_rel.tag_name))?;
        Ok(LagOutcome::Installed(row))
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
        let dir = self.dirs.engines_dir().join(&release.tag_name);
        if dir.exists() {
            std::fs::remove_dir_all(&dir).context("replace existing engine dir")?;
        }
        std::fs::create_dir_all(&dir)?;
        // F87: stream to disk — llama.cpp release assets reach ~400 MB
        // and must not be buffered whole in RAM (the mistralrs lane has
        // streamed since day one). F88: a failed download or extract
        // removes the half-populated dir instead of orphaning it.
        let archive = dir.join(&pick.name);
        if let Err(e) = self.gh.download_asset_file(asset, &archive).await {
            let _ = std::fs::remove_dir_all(&dir);
            return Err(e);
        }
        let digest = asset
            .digest
            .clone()
            .and_then(|d| d.strip_prefix("sha256:").map(str::to_string))
            .unwrap_or_else(|| "unverified".into());
        let extracted = extract_archive_file(&archive, &dir, &pick.name);
        std::fs::remove_file(&archive).context("remove downloaded archive")?;
        if let Err(e) = extracted {
            let _ = std::fs::remove_dir_all(&dir);
            return Err(e);
        }
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
        if let Err(e) = self.gh.download_asset_file(asset, &archive).await {
            let _ = std::fs::remove_dir_all(&dir); // F88: no orphan dir
            return Err(e);
        }
        let extracted = extract_archive_file(&archive, &dir, &pick.name);
        std::fs::remove_file(&archive).context("remove downloaded archive")?;
        if let Err(e) = extracted {
            let _ = std::fs::remove_dir_all(&dir); // F88: no orphan dir
            return Err(e);
        }
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

    /// Install a sglang engine: venv + pip + shim under
    /// `engines/sglang-<version>`, then the shared register/probe tail.
    /// `version = None` pins to [`sglang_install::SGLANG_DEFAULT_VERSION`]
    /// (verified flag contract; see its doc for why not auto-latest).
    /// F88: every failure path after dir creation removes the dir.
    pub async fn install_sglang(&self, version: Option<&str>) -> Result<EngineRow> {
        if !cfg!(target_os = "linux") {
            anyhow::bail!(
                "sglang upstream supports Linux (CUDA/ROCm) only; {} is not \
                 installable here — llamacpp (default) and mistralrs serve \
                 Windows/macOS",
                std::env::consts::OS
            );
        }
        let version = version.unwrap_or(sglang_install::SGLANG_DEFAULT_VERSION);
        // Reject malformed pins early: the tag IS the version string that
        // probe_sglang's semver parse and pip both consume.
        if version.is_empty() || !version.chars().all(|c| c.is_ascii_digit() || c == '.') {
            anyhow::bail!("sglang version must be dotted digits (e.g. 0.5.19), got {version:?}");
        }
        let tag = format!("sglang-{version}");
        let dir = self.dirs.engines_dir().join(&tag);
        if dir.exists() {
            std::fs::remove_dir_all(&dir).context("replace existing engine dir")?;
        }
        std::fs::create_dir_all(&dir)?;
        if let Err(e) = sglang_install::install_into(&dir, version).await {
            let _ = std::fs::remove_dir_all(&dir); // F88: no orphan dir
            return Err(e);
        }
        self.register_engine(
            &dir,
            &tag,
            &format!("pip:sglang=={version}"),
            "unverified",
            EngineKind::Sglang,
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
    /// Register a probed engine dir and activate it (see the
    /// CUDA-dethrone guard inside for the NVIDIA exception).
    pub fn register_engine(
        &self,
        dir: &Path,
        tag: &str,
        asset_label: &str,
        sha256: &str,
        kind: EngineKind,
    ) -> Result<EngineRow> {
        self.register_engine_with_vendor(dir, tag, asset_label, sha256, kind, system_vendor_hint())
    }

    /// `register_engine` with an injectable vendor hint so the CUDA
    /// dethrone guard is deterministically testable on any box.
    pub fn register_engine_with_vendor(
        &self,
        dir: &Path,
        tag: &str,
        asset_label: &str,
        sha256: &str,
        kind: EngineKind,
        vendor_hint: manifest::Vendor,
    ) -> Result<EngineRow> {
        let server = match kind {
            EngineKind::LlamaCpp => find_server(dir)?,
            EngineKind::MistralRs => find_engine_binary(dir, &["mistralrs", "mistralrs.exe"])?,
            // The install lane writes the shim; anything else is a
            // hand-copied dir, and the shim name is the contract.
            EngineKind::Sglang => find_engine_binary(dir, &["sglang-server"])?,
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
            EngineKind::Sglang => {
                tracing::debug!(target: "pallama::engine", "registered sglang {tag} ({asset_label})");
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
        // Live-measured on a 4070 (BENCHMARK.md 2026-09-11): the Vulkan
        // asset costs ~34 ms first-token vs CUDA. Activation used to be
        // hardware-blind, so one `engine update` (which falls back to the
        // Vulkan asset while the CUDA overlay repo is not yet publishing)
        // silently DEMOTED an installed CUDA engine to inactive. On NVIDIA
        // boxes a Vulkan install never dethrones CUDA; `pallama engine
        // use <tag>` stays the explicit override.
        let keep_cuda = row.kind == EngineKind::LlamaCpp
            && !is_cuda_engine(tag, asset_label)
            && vendor_hint == manifest::Vendor::Nvidia
            && store
                .list_engines()?
                .iter()
                .any(|e| e.kind == EngineKind::LlamaCpp && is_cuda_engine(&e.tag, &e.asset));
        let activated = !keep_cuda;
        if activated {
            store.set_active_engine(tag)?;
        }
        self.bus.publish(PallamaEvent::EngineUpdated {
            tag: tag.to_string(),
        });
        self.prune(&store)?;
        if keep_cuda {
            tracing::warn!(
                "NVIDIA GPU present — registered engine {tag} ({asset_label}) but KEPT the \
                 installed CUDA engine active (Vulkan first-token is measurably slower); run \
                 `pallama engine use {tag}` to switch anyway"
            );
        }
        // The flip above happened after `row` was built; the caller's
        // contract expects the returned row to reflect the post-install
        // store state.
        let row = EngineRow {
            active: activated,
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
        // Retention is scoped per engine KIND: a mistral.rs build is never
        // a rollback anchor for an active llama.cpp engine (and vice
        // versa), so each lane keeps its own newest KEEP_TAGS. Kind-blind
        // retention deleted cross-lane engines on back-to-back installs
        // (mistralrs install pruned the active CUDA engine; the CUDA
        // reinstall then pruned sglang).
        let mut kept_per_kind: std::collections::BTreeMap<&str, usize> =
            std::collections::BTreeMap::new();
        for e in &engines {
            let seen = kept_per_kind.entry(e.kind.as_str()).or_insert(0);
            *seen += 1;
            if *seen <= KEEP_TAGS || e.tag == LOCAL_TAG || Some(&e.tag) == active.as_ref() {
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

    /// Delete every OTHER engine of the same kind: a verified, activated
    /// update leaves exactly one build per lane (the user-facing "why do
    /// I see two llama.cpp engines after updating" contract). The `local`
    /// pseudo-tag and `keep_tag` itself survive; cross-kind rows are
    /// untouched. Returns the freed (tag, bytes) pairs for the summary.
    pub fn prune_siblings(&self, kind: &str, keep_tag: &str) -> Result<Vec<(String, u64)>> {
        let store = Store::open(&self.dirs)?;
        let engines = store.list_engines()?;
        let mut freed = Vec::new();
        for e in engines {
            if e.kind.as_str() != kind || e.tag == keep_tag || e.tag == LOCAL_TAG {
                continue;
            }
            let dir = self.dirs.engines_dir().join(&e.tag);
            let mut bytes = 0u64;
            if dir.exists() {
                bytes = engine_dir_bytes(&dir);
                std::fs::remove_dir_all(&dir)
                    .with_context(|| format!("prune engine dir {}", dir.display()))?;
            }
            store.delete_engine(&e.tag)?;
            tracing::info!(
                "pruned superseded engine {} ({} bytes) after update to {}",
                e.tag,
                bytes,
                keep_tag
            );
            self.bus
                .publish(PallamaEvent::EngineRemoved { tag: e.tag.clone() });
            freed.push((e.tag, bytes));
        }
        Ok(freed)
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

    /// Active engine's manifest (probed capabilities). The recorded
    /// server path is re-anchored to this data dir's engines root so rows
    /// installed elsewhere (moved data dir, copied DB) still resolve.
    pub fn active_manifest(&self) -> Result<Option<Manifest>> {
        let store = Store::open(&self.dirs)?;
        let Some(row) = store.active_engine()? else {
            return Ok(None);
        };
        let mut m: Manifest = serde_json::from_str(&row.manifest)
            .with_context(|| format!("decode manifest for {}", row.tag))?;
        m.re_root_server_path(&self.dirs.engines_dir());
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
            // F86: `DirEntry::file_type` does NOT follow symlinks — a
            // symlinked dir cycles forever under `p.is_dir()` (which
            // does). Only REAL dirs recurse; symlinks are treated as
            // leaf files (matchable, never descended).
            match e.file_type() {
                Ok(ft) if ft.is_dir() => walk(&p, names, out),
                _ => {
                    if p.file_name()
                        .is_some_and(|n| names.iter().any(|want| n == *want))
                    {
                        out.push(p);
                    }
                }
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
            if let Err(e) = std::fs::set_permissions(path, perms) {
                tracing::warn!(target: "pallama::engine", path = %path.display(), error = %e, "chmod +x failed");
            }
        }
    }
    #[cfg(not(unix))]
    let _ = path;
}
