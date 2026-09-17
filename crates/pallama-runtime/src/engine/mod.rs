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

/// Error-context marker for the PROBE phase of engine registration
/// (binary discovery + version probe). The asset-install lanes match it
/// to decide an extracted dir is unusable garbage — a SIGILL-class
/// instruction mismatch or a missing server binary — and remove it
/// instead of orphaning hundreds of MB (`engines/b11005-cuda` lesson).
/// Store-phase failures never carry it: a store row may already
/// reference the dir, so deleting it would desync dir and row.
pub const ENGINE_PROBE_FAILED: &str = "engine probe failed";
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
    /// Upstream official ubuntu-cuda asset this box would install from
    /// the target release itself (lane 1); `None` when the release
    /// ships no driver-runnable upstream CUDA asset.
    pub upstream_cuda: Option<gh::AssetPick>,
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
    std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths).any(|dir| names.iter().any(|n| build::bin_on_path(&dir, n)))
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
/// Preconditions every CUDA lane probe in `maybe_cuda_overlay` shares:
/// the driver's CUDA ceiling, the GPU's compute capability (as sm), the
/// asset arch lane (`x64`/`arm64`), and the upstream build number the
/// update targets.
struct CudaLane {
    driver_cuda: (u32, u32),
    sm: Option<u32>,
    arch: &'static str,
    number: u64,
}

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
            upstream_cuda: None,
            overlay_tag: None,
            cuda_asset: None,
            newest_cuda: None,
            driver_cuda: None,
        };
        // Same arch universe as the install-time lane (x64/arm64 linux).
        let arch = match std::env::consts::ARCH {
            "x86_64" => "x64",
            "aarch64" => "arm64",
            _ => "",
        };
        if (self.asset_override != "auto" && !self.asset_override.is_empty())
            || std::env::consts::OS != "linux"
            || arch.is_empty()
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
        // Lane 1 report: the release's own ubuntu-cuda assets.
        out.upstream_cuda = gh::resolve_cuda_asset(release, dc, sm, arch);
        let overlay_tag = format!("b{number}-cuda");
        out.overlay_tag = Some(overlay_tag.clone());
        if let Ok(overlay) = self
            .gh
            .release_by_tag_repo(&gh::engine_overlay_repo(), &overlay_tag)
            .await
        {
            out.cuda_asset = gh::resolve_cuda_asset(&overlay, dc, sm, arch);
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
        let arch = match std::env::consts::ARCH {
            "x86_64" => "x64",
            "aarch64" => "arm64",
            other => {
                return Err(anyhow!(
                    "no CUDA overlay asset for CPU arch {other} (prebuilt lane \
                     publishes x64 and arm64 only) — `pallama engine build cuda` \
                     compiles locally"
                ));
            }
        };
        let pick = gh::resolve_cuda_asset(&release, driver_cuda, sm, arch)
            .ok_or_else(|| anyhow!("no driver-runnable CUDA asset in overlay release {tag}"))?;
        self.install_picked(&release, &pick, None)
            .await
            .with_context(|| format!("install overlay {tag} asset {}", pick.label))
    }

    /// Prebuilt CUDA lane: when this machine is Linux-NVIDIA (x64 or
    /// arm64) with a CUDA-capable driver, prefer a prebuilt CUDA engine
    /// over the Vulkan asset (~4% decode uplift). Chain, first wins:
    /// (1) upstream official ubuntu-cuda asset from the SAME release
    /// (generic SASS + CPU dispatch — runs on nearly every box, needs
    /// a system CUDA runtime or the cudart companion), (2) our CI's
    /// `bNNNN-cuda` overlay for the same tag (sm-slim SASS, bundled
    /// cudart), (3) the newest published overlay build (overlay-lag
    /// fallback), else the Vulkan universal fallback. Any miss — asset
    /// absent, not runnable — is a quiet return to the next lane.
    /// Zero-touch: the overlay repo defaults to the project home;
    /// `PALLAMA_ENGINE_REPO` exists purely for forks.
    async fn maybe_cuda_overlay(
        &self,
        release: &GhRelease,
        exact_pin: bool,
    ) -> Result<Option<EngineRow>> {
        if self.asset_override != "auto" && !self.asset_override.is_empty() {
            return Ok(None); // explicit asset pin wins over every heuristic
        }
        if std::env::consts::OS != "linux" {
            return Ok(None); // upstream/overlay CUDA assets are linux-only tarballs
        }
        // Upstream names its assets ...-cuda-{X.Y}-{arch}.tar.gz with
        // arch in {x64, arm64}; keep the box's own lane.
        let arch = if std::env::consts::ARCH == "aarch64" {
            "arm64"
        } else if std::env::consts::ARCH == "x86_64" {
            "x64"
        } else {
            return Ok(None); // no upstream CUDA asset for this CPU arch
        };
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
        let lane = CudaLane {
            driver_cuda,
            sm,
            arch,
            number,
        };
        // Lane 1 — upstream official CUDA (same tag, driver-capped,
        // generic SASS + CPU dispatch). Falls through to the overlay on
        // a decline or a probe-class failure; infra errors propagate.
        if let Some(row) = self.upstream_lane_or_none(release, &lane).await? {
            return Ok(Some(row));
        }
        let overlay_tag = format!("b{}-cuda", lane.number);
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
                if let Some(row) = self
                    .overlay_lag_or_none(&overlay_tag, exact_pin, &lane, &release.tag_name)
                    .await?
                {
                    return Ok(Some(row));
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
        if let Some(pick) = gh::resolve_cuda_asset(&overlay, lane.driver_cuda, lane.sm, lane.arch) {
            tracing::info!(
                "installing prebuilt CUDA engine from {repo} {overlay_tag} ({})",
                pick.label
            );
            let row = self
                .install_picked(&overlay, &pick, None)
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
                lane.driver_cuda.0,
                lane.driver_cuda.1
            );
            Ok(None)
        }
    }

    /// Lane 1 of the prebuilt chain: try the upstream release's own
    /// official ubuntu-cuda asset before any overlay probe. `None`
    /// means the lane declined or the binary proved unusable here —
    /// the overlay lanes are the next resort; hard errors propagate.
    async fn upstream_lane_or_none(
        &self,
        release: &GhRelease,
        lane: &CudaLane,
    ) -> Result<Option<EngineRow>> {
        let Some(pick) = gh::resolve_cuda_asset(release, lane.driver_cuda, lane.sm, lane.arch)
        else {
            return Ok(None);
        };
        match self.install_upstream_cuda(release, &pick).await {
            Ok(Some(row)) => Ok(Some(row)),
            // lane politely declined (no companion) — try overlay
            Ok(None) => Ok(None),
            Err(e)
                if e.chain()
                    .any(|c| c.to_string().contains(ENGINE_PROBE_FAILED)) =>
            {
                tracing::warn!(
                    "upstream CUDA asset {} unusable on this machine \
                     ({e:#}) — trying the overlay lane",
                    pick.label
                );
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }

    /// The overlay-lag arm of [`EngineManager::maybe_cuda_overlay`]:
    /// when the same-tag overlay release is missing, run the lag
    /// fallback (never for an exact pin — a user-pinned tag must not be
    /// silently swapped for an older build) and narrate each outcome.
    /// `Some(row)` = a lag lane resolved the update; `None` = fall
    /// through to the Vulkan warn.
    async fn overlay_lag_or_none(
        &self,
        overlay_tag: &str,
        exact_pin: bool,
        lane: &CudaLane,
        target_tag: &str,
    ) -> Result<Option<EngineRow>> {
        if exact_pin {
            return Ok(None);
        }
        match self.overlay_lag_fallback(lane).await? {
            LagOutcome::Installed(row) => Ok(Some(row)),
            LagOutcome::AlreadyActive(row) => {
                tracing::warn!(
                    "overlay hasn't published {overlay_tag} yet ({} drops hourly); \
                     newest published CUDA build {} is already active — rerun \
                     `pallama engine update` after the next overlay drop, or run \
                     `pallama engine build cuda` to compile {target_tag} locally now",
                    gh::engine_overlay_repo(),
                    row.tag
                );
                Ok(Some(row))
            }
            LagOutcome::NothingRunnable => Ok(None), // Vulkan warn is truthful
        }
    }

    /// Lane 1 of the prebuilt chain: install the upstream release's
    /// official ubuntu-cuda asset under the overlay's `bNNNN-cuda` tag
    /// (same lane separation, keep-CUDA guard, and prune semantics).
    /// Skips (returns `Ok(None)`) when the system lacks the CUDA runtime
    /// for the pick's major AND the release ships no cudart companion —
    /// better a quiet lane decline than a guaranteed-dead 168 MB
    /// download. The cudart companion extracts beside the binaries;
    /// RUNPATH=$ORIGIN resolves it with zero env wiring.
    async fn install_upstream_cuda(
        &self,
        release: &GhRelease,
        pick: &gh::AssetPick,
    ) -> Result<Option<EngineRow>> {
        // AssetPick::label is `ubuntu-cuda-{major.minor}-{arch}` — the
        // major decides which runtime libs must be present.
        let cuda_major = pick
            .label
            .split_once("ubuntu-cuda-")
            .and_then(|(_, rest)| rest.split('.').next().map(str::to_owned))
            .and_then(|m| m.parse::<u32>().ok());
        let Some(cuda_major) = cuda_major else {
            tracing::warn!(
                "cannot parse CUDA major from asset label {} — skipping the \
                 upstream CUDA lane this update",
                pick.label
            );
            return Ok(None);
        };
        let runtime_present = build::system_cuda_runtime_complete(cuda_major).await;
        let companion = gh::resolve_cudart_companion(release, pick);
        if companion.is_none() && !runtime_present {
            tracing::warn!(
                "system lacks the CUDA {} runtime and release {} ships no \
                 cudart companion for {} — skipping the upstream CUDA lane \
                 (overlay bundles its own runtime)",
                cuda_major,
                release.tag_name,
                pick.label
            );
            return Ok(None);
        }
        // Register under the overlay tag so the row lands in the CUDA
        // lane (is_cuda_build, keep-CUDA guard, one-per-lane prune) —
        // install_picked finds the asset by pick.name, so the cloned
        // release with a rewritten tag resolves identically.
        let mut cuda_release = release.clone();
        if !cuda_release.tag_name.ends_with("-cuda") {
            cuda_release.tag_name = format!("{}-cuda", cuda_release.tag_name);
        }
        tracing::info!(
            "installing upstream CUDA engine {} ({}){}",
            cuda_release.tag_name,
            pick.label,
            if companion.is_some() {
                " + cudart companion"
            } else {
                " (system runtime)"
            }
        );
        let row = self
            .install_picked(&cuda_release, pick, companion.as_ref())
            .await
            .with_context(|| format!("install upstream {}", cuda_release.tag_name))?;
        Ok(Some(row))
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
    async fn overlay_lag_fallback(&self, lane: &CudaLane) -> Result<LagOutcome> {
        let repo = gh::engine_overlay_repo();
        let releases = match self.gh.list_releases_repo(&repo).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("overlay fallback probe of {repo} failed: {e:#}");
                return Ok(LagOutcome::NothingRunnable);
            }
        };
        let Some(pick_rel) = gh::newest_runnable_overlay(
            &releases,
            lane.driver_cuda,
            lane.sm,
            lane.number,
            lane.arch,
        ) else {
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
        let Some(pick) = gh::resolve_cuda_asset(pick_rel, lane.driver_cuda, lane.sm, lane.arch)
        else {
            return Ok(LagOutcome::NothingRunnable); // consistency guard; selection pre-filtered
        };
        let behind = lane
            .number
            .saturating_sub(gh::btag_number(&pick_rel.tag_name).unwrap_or(0));
        tracing::warn!(
            "overlay lags the channel target by {behind} build(s): installing \
             published {} instead ({} asset; fresh overlay builds land hourly)",
            pick_rel.tag_name,
            pick.label
        );
        let row = self
            .install_picked(pick_rel, &pick, None)
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
            match self.pick(&release).await? {
                Some(pick) => {
                    // A Windows CUDA build dlopens cudart/cublas DLLs that
                    // Windows never ships system-wide — no companion means
                    // a guaranteed probe failure, so name it up front
                    // instead of shipping a 150+ MiB corpse.
                    let companion = if std::env::consts::OS == "windows"
                        && pick.label.starts_with("win-cuda")
                    {
                        let comp = gh::resolve_cudart_companion(&release, &pick);
                        if comp.is_none() {
                            return Err(anyhow!(
                                "release {tag_name} ships {} but no cudart \
                                 companion — the DLLs are not system-provided on \
                                 Windows and the engine cannot boot; pin an older \
                                 release or use the Vulkan lane (pallama engine \
                                 install <tag> after `pallama config set \
                                 engine_asset win-vulkan-x64`)",
                                pick.label
                            ));
                        }
                        tracing::info!(
                            "queuing cudart companion for {} — Windows never \
                             ships the CUDA DLLs system-wide",
                            pick.label
                        );
                        comp
                    } else {
                        None
                    };
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
                        .install_picked(&release, &pick, companion.as_ref())
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
    /// auto matrix against the release's actual assets. On the Windows
    /// NVIDIA lane the driver's CUDA ceiling caps the win-cuda version
    /// choice (a newer prebuilt cannot start); Linux-NVIDIA never lands
    /// here with a CUDA asset — the upstream/overlay chain in
    /// `maybe_cuda_overlay` owns that decision.
    async fn pick(&self, release: &GhRelease) -> Result<Option<gh::AssetPick>> {
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
        let driver_cuda = if os == "windows" && vendor == manifest::Vendor::Nvidia {
            let (driver_cuda, _) = build::nvidia_gpu_facts().await;
            if driver_cuda.is_none() {
                tracing::warn!(
                    "NVIDIA GPU present but the driver's CUDA ceiling could not \
                     be probed (nvidia-smi missing or unparsable); picking the \
                     newest win-cuda asset uncapped — install the driver's \
                     nvidia-smi or pin an older release if the probe fails"
                );
            }
            driver_cuda
        } else {
            None
        };
        Ok(gh::resolve_asset(
            release,
            os,
            arch,
            Some(vendor),
            driver_cuda,
        ))
    }

    /// Download, verify, extract, then register (probe + store +
    /// activate + prune). `companion` is the optional cudart runtime
    /// archive (upstream splits the CUDA runtime out of the main
    /// tarball): downloaded after the main extract and flattened into
    /// the server binary's dir — the binaries carry RUNPATH=$ORIGIN
    /// (verified b11011), so libs beside the binary resolve with no env
    /// wiring, and Windows resolves DLLs from the exe dir natively.
    pub async fn install_picked(
        &self,
        release: &GhRelease,
        pick: &gh::AssetPick,
        companion: Option<&gh::GhAsset>,
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
        if let Some(comp) = companion {
            let comp_size = comp
                .size
                .map_or_else(String::new, |b| format!(" ({} MiB)", b / 1_048_576));
            tracing::info!(
                "downloading CUDA runtime companion {}{} — the system lacks \
                 the runtime for this build and the binary dlopens it at boot",
                comp.name,
                comp_size
            );
            let comp_archive = dir.join(&comp.name);
            if let Err(e) = self.gh.download_asset_file(comp, &comp_archive).await {
                let _ = std::fs::remove_dir_all(&dir);
                return Err(e);
            }
            let scratch = dir.join("cudart-incoming");
            std::fs::create_dir_all(&scratch).context("stage companion extract")?;
            let comp_extracted = extract_archive_file(&comp_archive, &scratch, &comp.name);
            let _ = std::fs::remove_file(&comp_archive);
            let merged = comp_extracted.and_then(|()| {
                let server = find_server(&dir).map_err(|e| {
                    e.context(ENGINE_PROBE_FAILED)
                        .context("companion merge needs the server binary location")
                })?;
                let bin_dir = server
                    .parent()
                    .ok_or_else(|| anyhow!("server path {} has no parent", server.display()))?
                    .to_path_buf();
                flatten_payload_into(&scratch, &bin_dir)
            });
            let _ = std::fs::remove_dir_all(&scratch);
            if let Err(e) = merged {
                let _ = std::fs::remove_dir_all(&dir);
                return Err(e);
            }
        }
        self.register_or_clean(
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
        self.register_or_clean(
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
        self.register_or_clean(
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
            EngineKind::LlamaCpp => find_server(dir),
            EngineKind::MistralRs => find_engine_binary(dir, &["mistralrs", "mistralrs.exe"]),
            // The install lane writes the shim; anything else is a
            // hand-copied dir, and the shim name is the contract.
            EngineKind::Sglang => find_engine_binary(dir, &["sglang-server"]),
        }
        .map_err(|e| e.context(ENGINE_PROBE_FAILED))?;
        make_executable(&server);

        let m = manifest::probe_kind(&server, tag, &kind)
            .map_err(|e| e.context(ENGINE_PROBE_FAILED))?;
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

    /// Register a dir the caller JUST extracted for this install, removing
    /// the dir when the probe phase rejects the binary (unusable asset:
    /// SIGILL-class instruction mismatch, missing server binary) — F88's
    /// no-orphan contract extended through the register tail. Store-phase
    /// failures keep the dir: a store row may already reference it.
    fn register_or_clean(
        &self,
        dir: &Path,
        tag: &str,
        asset_label: &str,
        sha256: &str,
        kind: EngineKind,
    ) -> Result<EngineRow> {
        match self.register_engine(dir, tag, asset_label, sha256, kind) {
            Ok(row) => Ok(row),
            Err(e) => {
                let probe_class = e
                    .chain()
                    .any(|c| c.to_string().contains(ENGINE_PROBE_FAILED));
                if probe_class {
                    let _ = std::fs::remove_dir_all(dir);
                }
                Err(e)
            }
        }
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

/// Move every file of an extracted runtime payload (cudart companion)
/// into `dst`, flattening whatever root layout the archive used —
/// upstream wraps it in its own top-level dir, and the layout differs
/// per platform, so the merge must not depend on it. Same-filesystem
/// renames; an existing file of the same name wins (the main tarball
/// is authoritative) and the incoming copy is dropped.
fn flatten_payload_into(src: &Path, dst: &Path) -> Result<()> {
    let Ok(entries) = std::fs::read_dir(src) else {
        return Ok(()); // nothing staged: nothing to merge
    };
    for entry in entries.flatten() {
        let from = entry.path();
        if entry.file_type().is_ok_and(|t| t.is_dir()) {
            flatten_payload_into(&from, dst)?;
            let _ = std::fs::remove_dir(&from);
        } else {
            let to = dst.join(entry.file_name());
            if to.exists() {
                tracing::debug!(
                    "runtime payload file {} already present — keeping the main tarball's copy",
                    to.display()
                );
                let _ = std::fs::remove_file(&from);
            } else {
                std::fs::rename(&from, &to)
                    .with_context(|| format!("merge {} into {}", from.display(), to.display()))?;
            }
        }
    }
    Ok(())
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

/// Run a read-only version probe under a time budget. Returns false on
/// spawn failure, non-zero exit, or budget exhaustion.
///
/// No kill on timeout, by design: these probes are self-exiting version
/// checks (never GPU/HTTP work), the worker thread reaps the child
/// whenever it does finish, and sharing the `Child` handle across the
/// waiter and a killer thread reintroduces the classic wait-vs-kill lock
/// race. Bounding the DECISION (not the child's life) is the contract —
/// the previous probe blocked the spawn path indefinitely instead.
fn exec_version_probe(bin: &Path, args: &[&str], budget: std::time::Duration) -> bool {
    let Ok(child) = std::process::Command::new(bin)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    else {
        return false;
    };
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut child = child;
        let _ = tx.send(child.wait().ok());
    });
    matches!(rx.recv_timeout(budget), Ok(Some(st)) if st.success())
}

/// Post-crash liveness check of the ACTIVE engine's own binary, per
/// kind. This gates the supervisor's engine rollback: a false verdict
/// silently de-thrones a healthy engine, so the probe must exec what the
/// lane actually ships — historically this walked for `llama-server` by
/// name only, which structurally failed for sglang (shim is
/// `sglang-server`) and mistral.rs, making EVERY spawn failure on those
/// lanes a guaranteed false rollback.
///
/// Per-kind probes (cheap, no GPU, no torch import):
/// - llamacpp: `llama-server --version` (native exec, 5 s)
/// - sglang: the venv's `importlib.metadata` version read — NOT the
///   shim, which boots python+torch and parses `launch_server` args
///   (slow, and `--version` is not a `launch_server` flag) (15 s)
/// - mistral.rs: `mistralrs --version` (clap, native exec, 15 s)
///
/// `manifest_json` is the engine row's manifest; its `server_path`
/// names the lane binary. llamacpp falls back to a name walk for
/// manifest-less rows; the venv lanes cannot (their layout is the
/// install contract).
#[must_use]
pub fn verify_engine_binary(
    kind: &EngineKind,
    engines_dir: &Path,
    manifest_json: Option<&str>,
) -> bool {
    let mut manifest: Option<crate::engine::manifest::Manifest> = manifest_json
        .and_then(|raw| serde_json::from_str::<crate::engine::manifest::Manifest>(raw).ok());
    // Rows written before re-rooting carry stale absolute paths;
    // `re_root_server_path` adopts the live engines dir when the
    // recorded one is gone.
    if let Some(m) = manifest.as_mut() {
        m.re_root_server_path(engines_dir);
    }
    match kind {
        EngineKind::LlamaCpp => {
            let bin = manifest
                .map(|m| PathBuf::from(m.server_path))
                .or_else(|| find_server(engines_dir).ok());
            bin.is_some_and(|b| {
                exec_version_probe(&b, &["--version"], std::time::Duration::from_secs(5))
            })
        }
        EngineKind::Sglang => {
            // server_path = <engines>/<tag>/sglang-server → the venv sits
            // next to the shim (sglang_install layout contract).
            let py = manifest
                .map(|m| PathBuf::from(m.server_path))
                .and_then(|shim| shim.parent().map(|d| d.join("venv/bin/python")));
            py.is_some_and(|p| {
                exec_version_probe(
                    &p,
                    &[
                        "-c",
                        "import importlib.metadata as m; print(m.version(\"sglang\"))",
                    ],
                    std::time::Duration::from_secs(15),
                )
            })
        }
        EngineKind::MistralRs => manifest
            .map(|m| PathBuf::from(m.server_path))
            .is_some_and(|b| {
                exec_version_probe(&b, &["--version"], std::time::Duration::from_secs(15))
            }),
    }
}

#[cfg(test)]
mod verify_tests {
    use super::*;
    use pallama_core::engine_kind::EngineKind;
    use std::path::PathBuf;

    fn fake_bin(dir: &std::path::Path, rel: &str, body: &str) -> PathBuf {
        let p = dir.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
        make_executable(&p);
        p
    }

    fn manifest_for(server_path: &std::path::Path) -> String {
        serde_json::json!({
            "tag": "t-test",
            "build_number": 1,
            "version_raw": "t",
            "devices": [],
            "flags": [],
            "spec_types": [],
            "server_path": server_path.display().to_string(),
        })
        .to_string()
    }

    #[test]
    fn unit__verify_engine_binary__llamacpp_version_probe() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = fake_bin(tmp.path(), "llama-b/llama-server", "exit 0");
        assert!(verify_engine_binary(
            &EngineKind::LlamaCpp,
            tmp.path(),
            Some(&manifest_for(&bin))
        ));
        let bad = fake_bin(tmp.path(), "llama-bad/llama-server", "exit 3");
        assert!(!verify_engine_binary(
            &EngineKind::LlamaCpp,
            tmp.path(),
            Some(&manifest_for(&bad))
        ));
    }

    #[test]
    fn unit__verify_engine_binary__sglang_venv_metadata_probe() {
        let tmp = tempfile::tempdir().unwrap();
        // The flashinfer-class regression pin: a sglang dir with NO
        // llama-server anywhere and a WORKING venv must verify TRUE —
        // the old name-walk probe returned false unconditionally for
        // this lane and rolled healthy engines back.
        let shim = fake_bin(tmp.path(), "sglang-server", "exec venv/bin/python \"$@\"");
        fake_bin(
            tmp.path(),
            "venv/bin/python",
            "echo 0.5.19 # fake metadata read",
        );
        assert!(verify_engine_binary(
            &EngineKind::Sglang,
            tmp.path(),
            Some(&manifest_for(&shim))
        ));
        // Broken venv python (non-zero exit) must fail the probe.
        let tmp2 = tempfile::tempdir().unwrap();
        let shim2 = fake_bin(tmp2.path(), "sglang-server", "exec venv/bin/python \"$@\"");
        fake_bin(tmp2.path(), "venv/bin/python", "exit 1");
        assert!(!verify_engine_binary(
            &EngineKind::Sglang,
            tmp2.path(),
            Some(&manifest_for(&shim2))
        ));
    }

    #[test]
    fn unit__verify_engine_binary__mistralrs_version_probe() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = fake_bin(tmp.path(), "mistralrs", "echo mistralrs 0.9.3; exit 0");
        assert!(verify_engine_binary(
            &EngineKind::MistralRs,
            tmp.path(),
            Some(&manifest_for(&bin))
        ));
        // No manifest (legacy row): the venv lanes cannot fall back to a
        // name walk — their layout is the install contract.
        assert!(!verify_engine_binary(
            &EngineKind::MistralRs,
            tmp.path(),
            None
        ));
    }

    #[test]
    fn unit__exec_version_probe__budget_bounds_the_decision_not_the_child() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = fake_bin(tmp.path(), "slow-probe", "sleep 30; exit 0");
        let t0 = std::time::Instant::now();
        let ok = exec_version_probe(&bin, &["--version"], std::time::Duration::from_secs(1));
        assert!(!ok, "budget exhaustion must read as a failed probe");
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(5),
            "probe must return at roughly the budget, not the child lifetime"
        );
    }

    /// register_or_clean pins: an unusable binary (exec-format /
    /// SIGILL-class probe failure) must take its whole engine dir with
    /// it — the engines/b11005-cuda 927 MiB orphan lesson — while a
    /// healthy dir registers normally. Linux-only: the fixtures are
    /// /bin/sh scripts.
    #[cfg(target_os = "linux")]
    mod register_or_clean_tests {
        use super::super::*;
        use super::fake_bin;
        use pallama_core::engine_kind::EngineKind;

        fn manager_in(tmp: &tempfile::TempDir) -> EngineManager {
            EngineManager {
                dirs: pallama_core::PallamaDirs {
                    config_dir: tmp.path().join("cfg"),
                    data_dir: tmp.path().join("data"),
                },
                gh: GhClient::with_base("http://127.0.0.1", None).expect("gh client"),
                bus: crate::events::EventBus::default(),
                asset_override: "auto".into(),
            }
        }

        /// A llama-server that satisfies the probe contract: `--version`
        /// prints a `version:` line with a build number, `--help` exits 0.
        fn healthy_server(dir: &std::path::Path) {
            fake_bin(
                dir,
                "llama-server",
                "case \"$1\" in --version) echo 'version: 4242 (stub)';; --help) echo 'usage: stub';; esac; exit 0",
            );
        }

        #[test]
        fn unit__register_or_clean__probe_failure_removes_the_dir() {
            let tmp = tempfile::tempdir().unwrap();
            let mgr = manager_in(&tmp);
            let dir = tmp.path().join("data/engines/b1-cuda");
            // Garbage bytes with the +x bit: spawn fails at exec — the
            // same register-tail failure class as a SIGILL'd asset.
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("llama-server"), b"not an elf").unwrap();
            make_executable(&dir.join("llama-server"));
            let err = mgr
                .register_or_clean(
                    &dir,
                    "b1-cuda",
                    "ubuntu-cuda-12.8-x64",
                    "x",
                    EngineKind::LlamaCpp,
                )
                .expect_err("garbage binary must fail the probe");
            assert!(
                format!("{err:#}").contains(ENGINE_PROBE_FAILED),
                "probe-class failures carry the marker: {err:#}"
            );
            assert!(!dir.exists(), "probe-failed engine dir must be removed");
        }

        #[test]
        fn unit__register_or_clean__healthy_stub_registers_and_keeps_dir() {
            let tmp = tempfile::tempdir().unwrap();
            let mgr = manager_in(&tmp);
            let dir = tmp.path().join("data/engines/b2-cuda");
            std::fs::create_dir_all(&dir).unwrap();
            healthy_server(&dir);
            let row = mgr
                .register_or_clean(
                    &dir,
                    "b2-cuda",
                    "ubuntu-cuda-12.8-x64",
                    "x",
                    EngineKind::LlamaCpp,
                )
                .expect("healthy stub registers");
            assert_eq!(row.tag, "b2-cuda");
            assert!(dir.exists(), "healthy engine dir survives");
            assert!(dir.join("llama-server").exists(), "binary survives");
        }
    }
}
