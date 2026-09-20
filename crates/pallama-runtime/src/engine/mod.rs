//! Engine lifecycle: install upstream llama-server builds (sha-verified),
//! probe capabilities, activate/rollback, prune old tags. A local build
//! registers as pseudo-tag `local` and is never pruned.

pub mod arch_miner;
pub mod build;
pub mod capability_registry;
pub mod gh;
pub mod manifest;
pub mod sglang_install;

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};

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
/// Does this error chain carry the probe marker? (SIGILL-class binary
/// mismatch, missing server binary — the asset is fine for other boxes,
/// just unusable on THIS one.) Lanes convert it to a decline so the next
/// lane (overlay, then Vulkan) gets a chance; infra errors propagate.
fn is_probe_failure(e: &anyhow::Error) -> bool {
    e.chain()
        .any(|c| c.to_string().contains(ENGINE_PROBE_FAILED))
}
/// Pure lane-1 scan-back selection (test seam, no network): the newest
/// release STRICTLY behind the lane's channel number whose assets
/// resolve for this driver/arch, within [`UPSTREAM_SCANBACK_DEPTH`].
/// The channel release itself is excluded — the caller already tried
/// its own assets before scanning.
fn scanback_release<'a>(releases: &'a [GhRelease], lane: &CudaLane) -> Option<&'a GhRelease> {
    let mut behind: Vec<(u64, &GhRelease)> = releases
        .iter()
        .filter_map(|r| gh::btag_number(&r.tag_name).map(|n| (n, r)))
        .filter(|(n, _)| *n < lane.number)
        .collect();
    behind.sort_by_key(|(n, _)| std::cmp::Reverse(*n));
    behind
        .into_iter()
        .take(UPSTREAM_SCANBACK_DEPTH)
        .find_map(|(_, cand)| {
            gh::resolve_cuda_asset(cand, lane.driver_cuda, lane.sm, lane.arch).map(|_| cand)
        })
}
/// How many releases behind the channel target lane 1 may scan for an
/// upstream ubuntu-cuda asset. Most upstream releases ship none; the
/// scan stays shallow so we never wander into stale runtimes.
const UPSTREAM_SCANBACK_DEPTH: usize = 5;
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

/// Aside-dir prefix for an engine preserved across a replacement
/// install. Dot-prefixed: invisible to dir scans, and DB rows (the only
/// thing that names engine dirs) never point at it.
const RETIRED_ENGINE_PREFIX: &str = ".retired-";

/// Rename the installed `engines/<tag>` aside so a replacement build can
/// run at the FINAL path — venvs and extracted archives embed absolute
/// paths, so building under a scratch name and renaming in is not an
/// option. The aside copy is the rollback: `restore_retired_engine`
/// puts it back when the replacement fails, `discard_retired_engine`
/// deletes it once the new engine registers. Asides from CRASHED runs
/// of the same tag are swept here: nothing else references them and
/// they would leak GiB. Returns `None` when no installed dir existed
/// (fresh install — nothing to preserve).
fn retire_engine_dir(dir: &Path) -> Result<Option<PathBuf>> {
    if !dir.exists() {
        return Ok(None);
    }
    let engines = dir.parent().context("engine dir has no parent")?;
    let tag = dir
        .file_name()
        .and_then(|n| n.to_str())
        .context("engine tag is not UTF-8")?;
    let aside_prefix = format!("{RETIRED_ENGINE_PREFIX}{tag}-");
    for entry in std::fs::read_dir(engines)
        .with_context(|| format!("scan {}", engines.display()))?
        .flatten()
    {
        if entry
            .file_name()
            .to_str()
            .is_some_and(|n| n.starts_with(&aside_prefix))
        {
            // A crashed run's rollback copy: the replacement about to
            // run supersedes anything it held. (Concurrent installs of
            // the same tag are already undefined — both write this same
            // final dir.)
            std::fs::remove_dir_all(entry.path())
                .with_context(|| format!("sweep stale aside {}", entry.path().display()))?;
        }
    }
    let aside = engines.join(format!("{aside_prefix}{}", std::process::id()));
    std::fs::rename(dir, &aside).with_context(|| format!("retire {} aside", dir.display()))?;
    Ok(Some(aside))
}

/// Put a retired engine back after a failed replacement. The failed
/// replacement's remains go first (`register_or_clean` already removed
/// probe-class dirs; store-class keeps them — neither may block the
/// rename). A restore failure means both copies are stranded: logged at
/// error level with the aside left on disk for the next same-tag
/// install (or a human) to recover — never silently dropped.
fn restore_retired_engine(aside: Option<&Path>, dir: &Path) {
    if dir.exists() {
        let _ = std::fs::remove_dir_all(dir);
    }
    let Some(aside) = aside else { return };
    if let Err(e) = std::fs::rename(aside, dir) {
        tracing::error!(
            "cannot restore retired engine {} -> {}: {e} — the previous engine \
             is preserved at {} until the next install of this tag",
            aside.display(),
            dir.display(),
            aside.display()
        );
    }
}

/// Delete the superseded engine copy after its replacement registered.
/// A leak here wastes disk but breaks nothing — warn, never fail the
/// install that already succeeded.
fn discard_retired_engine(aside: Option<&Path>) {
    let Some(aside) = aside else { return };
    if let Err(e) = std::fs::remove_dir_all(aside) {
        tracing::warn!(
            "leaked retired engine dir {} ({e}): remove it to reclaim disk",
            aside.display()
        );
    }
}

/// Delete an engine's directory AND its store row as one unit: retire the
/// dir aside first, delete the row second, discard the aside last. Any
/// failure restores the previous state — a row never survives over a
/// deleted dir (ghost row), and a dir never disappears while its row
/// lives. Returns the bytes reclaimed. Both retirement paths (the boot
/// sweep and manual `engine rm`) go through here so the ordering
/// invariant has a single owner.
pub fn remove_engine_row_and_tree(store: &Store, tag: &str, dir: &Path) -> Result<u64> {
    let bytes = engine_dir_bytes(dir);
    let aside = retire_engine_dir(dir)?;
    if let Err(e) = store.delete_engine(tag) {
        restore_retired_engine(aside.as_deref(), dir);
        return Err(anyhow!(e).context(format!(
            "cannot delete engine row {tag} — the engine dir was restored"
        )));
    }
    discard_retired_engine(aside.as_deref());
    Ok(bytes)
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

/// Is this engine row a fork capability lane? Recognized by the
/// `fork-*` tag convention or by the recorded manifest naming a fork
/// source (the tag check is the fast path; the manifest check keeps the
/// classification honest for lanes registered by older/other tooling).
/// Fork lanes are user-installed escape hatches for capabilities
/// upstream has not merged: they never count against mainstream
/// retention budgets and are never auto-pruned — removal is explicit
/// (`pallama engine rm <tag>`).
#[must_use]
pub fn is_fork_lane(row: &EngineRow) -> bool {
    row.tag.starts_with("fork-")
        || serde_json::from_str::<manifest::Manifest>(&row.manifest)
            .is_ok_and(|m| m.source == manifest::EngineSource::Fork)
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
    pub async fn check_lane(&self, release: &GhRelease, exact_pin: bool) -> Result<LaneCheck> {
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
        // Lane 1 report: the SAME helper the install lane uses (own
        // asset, else scan-back for the newest release that ships one —
        // never behind an exact pin).
        let lane = CudaLane {
            driver_cuda: dc,
            sm,
            arch,
            number,
        };
        out.upstream_cuda = self
            .upstream_cuda_pick(release, &lane, exact_pin)
            .await
            .map(|(_, p)| p);
        // Overlay lanes are self-hosted-only (PALLAMA_ENGINE_REPO); with
        // no overlay configured the report stops at the upstream pick.
        if let Some(repo) = gh::engine_overlay_repo() {
            let overlay_tag = format!("b{number}-cuda");
            out.overlay_tag = Some(overlay_tag.clone());
            if let Ok(overlay) = self.gh.release_by_tag_repo(&repo, &overlay_tag).await {
                out.cuda_asset = gh::resolve_cuda_asset(&overlay, dc, sm, arch);
                out.newest_cuda = gh::newest_asset_cuda(&overlay);
            }
            // A missing overlay release stays overlay_tag=Some + asset=None:
            // the CLI reports the publish lag instead of pretending the
            // lane was evaluated.
        }
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
        let Some(repo) = gh::engine_overlay_repo() else {
            bail!(
                "{tag} looks like a self-hosted overlay tag, but no overlay \
                 repo is configured — set {} to the repo publishing \
                 bNNNN-cuda releases. The default channel serves upstream's \
                 official CUDA assets via: pallama engine update",
                gh::ENGINE_OVERLAY_REPO_ENV
            );
        };
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
    /// a system CUDA runtime or the cudart companion; falls back to a
    /// scan-back for the newest release that ships one), (2) a
    /// self-hosted `bNNNN-cuda` overlay for the same tag when
    /// `PALLAMA_ENGINE_REPO` points at one (sm-slim SASS, bundled
    /// cudart), (3) the newest published overlay build (overlay-lag
    /// fallback), else the Vulkan universal fallback. Any miss — asset
    /// absent, not runnable, overlay unset — is a quiet return to the
    /// next lane; the project publishes no overlay of its own.
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
        if let Some(row) = self
            .upstream_lane_or_none(release, &lane, exact_pin)
            .await?
        {
            return Ok(Some(row));
        }
        // Overlay lanes are self-hosted-only: with PALLAMA_ENGINE_REPO
        // unset there is nothing to probe — the Vulkan lane follows
        // (the caller narrates that drop).
        let Some(repo) = repo else {
            return Ok(None);
        };
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
                     Vulkan lane this update. Local CUDA for THIS driver: \
                     pallama engine build cuda"
                );
                return Ok(None);
            }
        };
        if let Some(pick) = gh::resolve_cuda_asset(&overlay, lane.driver_cuda, lane.sm, lane.arch) {
            self.install_same_tag_overlay(&overlay, &overlay_tag, &pick, &repo)
                .await
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

    /// Lane 2 — the same-tag overlay asset. A probe-class failure
    /// (SIGILL-class baseline mismatch) means this box cannot run the
    /// overlay build: fall to the Vulkan lane instead of failing the
    /// whole update. Infra errors still propagate.
    async fn install_same_tag_overlay(
        &self,
        overlay: &GhRelease,
        overlay_tag: &str,
        pick: &gh::AssetPick,
        repo: &str,
    ) -> Result<Option<EngineRow>> {
        tracing::info!(
            "installing prebuilt CUDA engine from {repo} {overlay_tag} ({})",
            pick.label
        );
        match self.install_picked(overlay, pick, None).await {
            Ok(row) => Ok(Some(row)),
            Err(e) if is_probe_failure(&e) => {
                tracing::warn!(
                    "overlay CUDA asset {overlay_tag}/{} unusable on \
                     this machine ({e:#}) — using the Vulkan lane this \
                     update; `pallama engine build cuda` compiles a \
                     local build for this exact CPU",
                    pick.label
                );
                Ok(None)
            }
            Err(e) => Err(e).with_context(|| format!("install overlay {overlay_tag}")),
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
        exact_pin: bool,
    ) -> Result<Option<EngineRow>> {
        let Some((asset_release, pick)) = self.upstream_cuda_pick(release, lane, exact_pin).await
        else {
            return Ok(None);
        };
        match self.install_upstream_cuda(&asset_release, &pick).await {
            Ok(Some(row)) => Ok(Some(row)),
            // lane politely declined (no companion) — try overlay
            Ok(None) => Ok(None),
            Err(e) if is_probe_failure(&e) => {
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

    /// The lane-1 asset choice, shared by the install lane and the
    /// `--check` dry-run so they can never disagree: the release's own
    /// ubuntu-cuda asset, else (auto updates only — never an exact pin)
    /// the newest release at or behind the channel target that ships
    /// one. Roughly two thirds of upstream releases carry no ubuntu-cuda
    /// assets; without the scan an asset-less channel target would punt
    /// CUDA users to the sm-slim overlay or Vulkan for no reason.
    async fn upstream_cuda_pick(
        &self,
        release: &GhRelease,
        lane: &CudaLane,
        exact_pin: bool,
    ) -> Option<(GhRelease, gh::AssetPick)> {
        if let Some(pick) = gh::resolve_cuda_asset(release, lane.driver_cuda, lane.sm, lane.arch) {
            return Some((release.clone(), pick));
        }
        if exact_pin {
            return None; // a pinned tag must not silently become an older build
        }
        let releases = match self.gh.list_releases_repo(gh::LLAMA_CPP_REPO).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(
                    "cannot list {} for the upstream-asset scan-back \
                     ({e:#}) — continuing on the overlay lanes",
                    gh::LLAMA_CPP_REPO
                );
                return None;
            }
        };
        if let Some(cand) = scanback_release(&releases, lane) {
            let pick = gh::resolve_cuda_asset(cand, lane.driver_cuda, lane.sm, lane.arch)
                .expect("scanback only returns releases that resolve");
            tracing::info!(
                "channel release {} ships no ubuntu-cuda asset for this \
                 driver/arch — using upstream {} ({}) instead",
                release.tag_name,
                cand.tag_name,
                pick.label
            );
            return Some((cand.clone(), pick));
        }
        None
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
        let lag = match self.overlay_lag_fallback(lane).await {
            Ok(outcome) => outcome,
            // Same contract as the same-tag overlay arm: a probe-class
            // failure (the lagged overlay's sm-slim baseline does not
            // run on this CPU) falls through to Vulkan, not a dead end.
            Err(e) if is_probe_failure(&e) => {
                tracing::warn!(
                    "lagged overlay CUDA asset unusable on this machine \
                     ({e:#}) — using the Vulkan lane this update; \
                     `pallama engine build cuda` compiles a local build \
                     for this exact CPU"
                );
                return Ok(None);
            }
            Err(e) => return Err(e),
        };
        match lag {
            LagOutcome::Installed(row) => Ok(Some(row)),
            LagOutcome::AlreadyActive(row) => {
                tracing::warn!(
                    "overlay hasn't published {overlay_tag} yet; \
                     newest published CUDA build {} is already active — rerun \
                     `pallama engine update` after the overlay publishes, or run \
                     `pallama engine build cuda` to compile {target_tag} locally now",
                    row.tag
                );
                Ok(Some(row))
            }
            LagOutcome::NothingRunnable => Ok(None), // Vulkan warn is truthful
        }
    }

    /// Lane 1 of the prebuilt chain: install the upstream release's
    /// official ubuntu-cuda asset under a local `bNNNN-cuda` tag
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
    /// the overlay catches up.
    ///
    /// The outcome is tri-state so the caller narrates each arm
    /// distinctly: "newest published build is already active" used to
    /// collapse into `None` and read as "found nothing runnable",
    /// printing a Vulkan-lane switch the keep-CUDA guard then
    /// cancelled — three contradictory decisions in one run.
    async fn overlay_lag_fallback(&self, lane: &CudaLane) -> Result<LagOutcome> {
        let Some(repo) = gh::engine_overlay_repo() else {
            return Ok(LagOutcome::NothingRunnable);
        };
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
             published {} instead ({} asset)",
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
        let digest = asset
            .digest
            .clone()
            .and_then(|d| d.strip_prefix("sha256:").map(str::to_string))
            .unwrap_or_else(|| "unverified".into());
        // F87: stream to disk — llama.cpp release assets reach ~400 MB
        // and must not be buffered whole in RAM (the mistralrs lane has
        // streamed since day one). F88 + rollback: a failed download or
        // extract restores the previously installed engine instead of
        // orphaning a half-populated dir.
        let tag = release.tag_name.clone();
        let pick_name = pick.name.clone();
        let pick_label = pick.label.clone();
        self.install_with_rollback(
            &tag,
            &pick_label,
            &digest,
            EngineKind::LlamaCpp,
            |dir| async move {
                std::fs::create_dir_all(&dir)?;
                let archive = dir.join(&pick_name);
                self.gh.download_asset_file(asset, &archive).await?;
                let extracted = extract_archive_file(&archive, &dir, &pick_name);
                std::fs::remove_file(&archive).context("remove downloaded archive")?;
                extracted?;
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
                    self.gh.download_asset_file(comp, &comp_archive).await?;
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
                            .ok_or_else(|| {
                                anyhow!("server path {} has no parent", server.display())
                            })?
                            .to_path_buf();
                        flatten_payload_into(&scratch, &bin_dir)
                    });
                    let _ = std::fs::remove_dir_all(&scratch);
                    merged?;
                }
                Ok(())
            },
        )
        .await
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
        let tag = release.tag_name.clone();
        let pick_name = pick.name.clone();
        let pick_label = pick.label.clone();
        let cpu_fallback = pick.cpu_fallback;
        self.install_with_rollback(
            &tag,
            &pick_label,
            &digest,
            EngineKind::MistralRs,
            |dir| async move {
                std::fs::create_dir_all(&dir)?;
                let archive = dir.join(&pick_name);
                self.gh.download_asset_file(asset, &archive).await?;
                let extracted = extract_archive_file(&archive, &dir, &pick_name);
                std::fs::remove_file(&archive).context("remove downloaded archive")?;
                extracted?;
                if cpu_fallback {
                    tracing::warn!(
                        "installed the CPU mistralrs asset {} — this machine's driver/GPU \
                         does not qualify for a CUDA prebuilt; expect CPU-only speed",
                        pick_name
                    );
                }
                Ok(())
            },
        )
        .await
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
        // Rollback-safe: a rebuild that dies mid-venv (a 100%-full disk
        // ate one live, 2026-09-18) restores the previously installed
        // engine instead of destroying it.
        let asset_label = format!("pip:sglang=={version}");
        self.install_with_rollback(
            &tag,
            &asset_label,
            "unverified",
            EngineKind::Sglang,
            |dir| async move {
                std::fs::create_dir_all(&dir)?;
                sglang_install::install_into(&dir, version)
                    .await
                    .map(|_| ())
            },
        )
        .await
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
        self.register_engine_inner(
            dir,
            tag,
            asset_label,
            sha256,
            kind,
            system_vendor_hint(),
            None,
            manifest::TrustTier::default(),
        )
    }

    /// `register_engine` plus lane provenance: build lanes (upstream
    /// b-tags and fork pins) bake their source repo, commit SHA, and
    /// mined architecture set into the row's manifest — the capability
    /// currency the supervisor's unknown-architecture re-route consumes.
    /// `trust` marks registry-installed curated lanes (auto-retirable
    /// once mainline covers them); user builds pass the default.
    // Mirrors register_engine_inner's surface 1:1.
    #[allow(clippy::too_many_arguments)]
    pub fn register_engine_provenanced(
        &self,
        dir: &Path,
        tag: &str,
        asset_label: &str,
        sha256: &str,
        kind: EngineKind,
        prov: &manifest::LaneProvenance,
        trust: manifest::TrustTier,
    ) -> Result<EngineRow> {
        self.register_engine_inner(
            dir,
            tag,
            asset_label,
            sha256,
            kind,
            system_vendor_hint(),
            Some(prov),
            trust,
        )
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
        self.register_engine_inner(
            dir,
            tag,
            asset_label,
            sha256,
            kind,
            vendor_hint,
            None,
            manifest::TrustTier::default(),
        )
    }

    // Argument list maps 1:1 onto the public register_* API; bundling
    // into a seed struct would hide that correspondence.
    #[allow(clippy::too_many_arguments)]
    fn register_engine_inner(
        &self,
        dir: &Path,
        tag: &str,
        asset_label: &str,
        sha256: &str,
        kind: EngineKind,
        vendor_hint: manifest::Vendor,
        prov: Option<&manifest::LaneProvenance>,
        trust: manifest::TrustTier,
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

        let mut m = manifest::probe_kind(&server, tag, &kind)
            .map_err(|e| e.context(ENGINE_PROBE_FAILED))?;
        if let Some(p) = prov {
            m.merge_provenance(p);
        }
        // Trust tier distinguishes registry-installed curated lanes
        // (auto-retirable) from user-pinned forks (never auto-deleted).
        if trust == manifest::TrustTier::Curated {
            m.trust = trust;
        }
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
        // Fork lanes are additive: installing one never dethrones the
        // active engine of its kind. Routing picks the newest lane per
        // spawn when the capability is actually needed, and
        // `pallama engine use <tag>` stays the explicit switch. A fork
        // lane only activates when it is the first of its kind.
        let fork_additive = m.source == manifest::EngineSource::Fork
            && store
                .list_engines()?
                .iter()
                .any(|e| e.kind == row.kind && e.active);
        let activated = !keep_cuda && !fork_additive;
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
        if fork_additive {
            tracing::info!(
                "fork lane {tag} registered without activating — the active {} engine stays; \
                 `pallama engine use {tag}` switches explicitly",
                row.kind.as_str()
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

    /// Rollback-safe engine install for the release/asset lanes: the
    /// installed `engines/<tag>` dir is renamed aside (instant) before
    /// `build` runs at the FINAL path — venvs and extracted archives
    /// embed absolute paths, so a scratch-name build cannot be renamed
    /// in — and restored untouched when anything fails: build error
    /// (disk-full mid-venv), probe rejection, store failure. The aside
    /// copy is deleted only after the replacement registers, so a
    /// replacement needs headroom for BOTH copies at peak — that disk
    /// cost is the price of never destroying a working engine (live
    /// incident 2026-09-18: a 100%-full disk ate the sglang venv
    /// mid-rebuild under the old rm-first flow).
    async fn install_with_rollback<B, F>(
        &self,
        tag: &str,
        asset_label: &str,
        sha256: &str,
        kind: EngineKind,
        build: B,
    ) -> Result<EngineRow>
    where
        B: FnOnce(PathBuf) -> F,
        F: std::future::Future<Output = Result<()>>,
    {
        let dir = self.dirs.engines_dir().join(tag);
        let aside = retire_engine_dir(&dir)?;
        let outcome = build(dir.clone())
            .await
            .and_then(|()| self.register_or_clean(&dir, tag, asset_label, sha256, kind));
        match outcome {
            Ok(row) => {
                discard_retired_engine(aside.as_deref());
                Ok(row)
            }
            Err(e) => {
                restore_retired_engine(aside.as_deref(), &dir);
                Err(e)
            }
        }
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
    /// never pruned. Fork capability lanes are exempt entirely — see
    /// [`is_fork_lane`].
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
            // A user-installed fork lane never consumes a mainstream
            // retention slot: skipping BEFORE the count keeps KEEP_TAGS
            // reserved for upstream currency.
            if is_fork_lane(e) {
                continue;
            }
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

    /// Phase-3 supersede lifecycle, run at daemon start and after any
    /// roster change:
    ///
    /// (a) one-time architecture mining for binary-installed upstream
    ///     lanes (release assets ship no manifest architectures; the
    ///     raw `src/llama-arch.cpp` at the lane's tag is fetched once
    ///     and the mined set persisted — never refetched);
    /// (b) supersede marking — a fork lane whose advertised architecture
    ///     set is fully covered by mainstream lanes gets
    ///     `superseded_by`/`superseded_at` stamped, so `engine list` can
    ///     show the graduation and the supervisor can drop learned pins;
    /// (c) retirement sweep — curated fork lanes past
    ///     `fork_retire_days` (0 = never) are deleted, EXCEPT rows that
    ///     are active or referenced by `pinned_tags` (user pins and
    ///     in-flight rescue pins override the lifecycle; user-built
    ///     forks are never auto-deleted at all).
    ///
    /// Every step fails open: supersede bookkeeping must never break
    /// engine registration or serving.
    pub async fn refresh_supersede_state(
        &self,
        fork_retire_days: u64,
        pinned_tags: &[String],
    ) -> Result<()> {
        let store = Store::open(&self.dirs)?;
        self.mine_missing_architectures(&store).await;
        Self::mark_superseded_lanes(&store)?;
        self.sweep_retired_lanes(&store, fork_retire_days, pinned_tags)?;
        Ok(())
    }

    /// Supersede step (a): one-time architecture mining for
    /// binary-installed upstream lanes (release assets ship no manifest
    /// architectures; source builds mine at build time). Fetch failures
    /// warn and move on — supersede coverage catches up on the next
    /// restart once the raw host is reachable again.
    async fn mine_missing_architectures(&self, store: &Store) {
        let rows = store.list_engines().unwrap_or_default(); // newest first
        for row in &rows {
            if row.kind != EngineKind::LlamaCpp {
                continue;
            }
            let Ok(mut manifest) = serde_json::from_str::<Manifest>(&row.manifest) else {
                continue;
            };
            if manifest.source != manifest::EngineSource::Upstream
                || !manifest.architectures.is_empty()
            {
                continue;
            }
            // Engine tags carry an asset suffix (b11026-cuda); the raw
            // source lives at the bare upstream tag.
            let source_tag = match gh::btag_number(&row.tag) {
                Some(n) => format!("b{n}"),
                None => row.tag.clone(),
            };
            match self.gh.fetch_llama_arch_source(&source_tag).await {
                Ok(text) => {
                    let archs = arch_miner::parse_arch_table(&text);
                    if archs.is_empty() {
                        tracing::warn!(
                            "lane {}: mined 0 architectures from upstream {} — leaving manifest untouched",
                            row.tag,
                            source_tag
                        );
                        continue;
                    }
                    manifest.architectures = archs;
                    match serde_json::to_string(&manifest)
                        .context("encode mined manifest")
                        .and_then(|encoded| {
                            store
                                .update_engine_manifest(&row.tag, &encoded)
                                .context("persist mined architectures")
                        }) {
                        Ok(()) => tracing::info!(
                            "lane {}: mined {} architectures from upstream {} (one-time)",
                            row.tag,
                            manifest.architectures.len(),
                            source_tag
                        ),
                        Err(e) => tracing::warn!(
                            "lane {}: could not persist mined architectures ({e:#})",
                            row.tag
                        ),
                    }
                }
                Err(e) => {
                    // Fail open: an unreachable raw host must not block
                    // registration or the rest of the supersede pass.
                    tracing::warn!(
                        "lane {}: one-time arch mining skipped ({e:#}) — supersede coverage may be incomplete until next restart",
                        row.tag
                    );
                }
            }
        }
    }

    /// Supersede step (b): stamp fork lanes whose advertised
    /// architecture set is fully covered by mainstream lanes. Partial
    /// coverage keeps the fork active — partial mainstream support is
    /// exactly the self-correcting case (unknown-arch rescue re-pins
    /// the fork for the archs mainline still rejects).
    fn mark_superseded_lanes(store: &Store) -> Result<()> {
        let rows = store.list_engines()?; // newest first
        let mut decoded: Vec<(EngineRow, Manifest)> = rows
            .iter()
            .filter_map(|row| {
                serde_json::from_str::<Manifest>(&row.manifest)
                    .ok()
                    .map(|manifest| (row.clone(), manifest))
            })
            .collect();
        // Mainstream coverage set, newest-first (list_engines is
        // installed_at DESC): upstream + local llamacpp lanes that
        // advertise architectures. Owned (tag, arch set) pairs so the
        // fork rows below can be mutated while mainstream stays alive.
        let mainstream: Vec<(String, std::collections::BTreeSet<String>)> = decoded
            .iter()
            .filter(|(row, manifest)| {
                row.kind == EngineKind::LlamaCpp
                    && manifest.source != manifest::EngineSource::Fork
                    && !manifest.architectures.is_empty()
            })
            .map(|(row, manifest)| (row.tag.clone(), manifest.architectures.clone()))
            .collect();
        let covers = |arch: &str| mainstream.iter().any(|(_, archs)| archs.contains(arch));
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs().cast_signed());
        for (row, manifest) in &mut decoded {
            if !is_fork_lane(row) || manifest.architectures.is_empty() {
                continue;
            }
            if !manifest.architectures.iter().all(|arch| covers(arch)) {
                continue;
            }
            // Graduation target: the newest mainstream lane covering
            // any of the fork's archs (mainstream is newest-first).
            let Some(newest_coverer) = mainstream
                .iter()
                .find(|(_, archs)| manifest.architectures.iter().any(|a| archs.contains(a)))
                .map(|(tag, _)| tag.clone())
            else {
                continue;
            };
            if manifest.superseded_by.as_deref() == Some(newest_coverer.as_str()) {
                continue; // already stamped for this coverer
            }
            manifest.superseded_by = Some(newest_coverer.clone());
            manifest.superseded_at_epoch = Some(now);
            let encoded = serde_json::to_string(manifest).context("encode superseded manifest")?;
            store.update_engine_manifest(&row.tag, &encoded)?;
            tracing::warn!(
                "fork lane {} superseded by {} — mainline now covers its architectures; model pins clear on next resolve",
                row.tag,
                newest_coverer
            );
        }
        Ok(())
    }

    /// Supersede step (c): retirement sweep — curated fork lanes past
    /// the grace period are deleted, EXCEPT rows that are active or
    /// referenced by `pinned_tags` (user pins and in-flight rescue pins
    /// override the lifecycle). User-built forks (trust User) outlive
    /// everything but an explicit `engine rm`.
    fn sweep_retired_lanes(
        &self,
        store: &Store,
        fork_retire_days: u64,
        pinned_tags: &[String],
    ) -> Result<()> {
        if fork_retire_days == 0 {
            return Ok(()); // disabled: never auto-delete
        }
        let cutoff = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs().cast_signed())
            .saturating_sub(
                i64::try_from(fork_retire_days.saturating_mul(86_400)).unwrap_or(i64::MAX),
            );
        for row in store.list_engines()? {
            let Ok(manifest) = serde_json::from_str::<Manifest>(&row.manifest) else {
                continue;
            };
            if manifest.superseded_at_epoch.is_none_or(|at| at >= cutoff) {
                continue;
            }
            if manifest.trust != manifest::TrustTier::Curated
                || row.active
                || pinned_tags.contains(&row.tag)
            {
                continue;
            }
            let dir = self.dirs.engines_dir().join(&row.tag);
            // Retire-to-aside, then row-delete, then discard — the row
            // only disappears once the directory is safely out of the
            // way, and any failure restores the previous state (no
            // ghost row over a deleted dir). A lane that cannot be
            // removed (locked dir, busy store) warns and frees the rest
            // of the pass: one poisoned lane must not block retirement
            // of its siblings on every sweep.
            match remove_engine_row_and_tree(store, &row.tag, &dir) {
                Ok(bytes) => {
                    tracing::warn!(
                        "retired curated fork lane {} ({} bytes) — rebuild any time via the registry",
                        row.tag,
                        bytes
                    );
                    self.bus.publish(PallamaEvent::EngineRemoved {
                        tag: row.tag.clone(),
                    });
                }
                Err(e) => {
                    tracing::warn!(
                        "cannot retire curated fork lane {} dir {} ({e:#}) — skipped this pass",
                        row.tag,
                        dir.display()
                    );
                }
            }
        }
        Ok(())
    }

    /// Delete every OTHER engine of the same kind: a verified, activated
    /// update leaves exactly one build per lane (the user-facing "why do
    /// I see two llama.cpp engines after updating" contract). The `local`
    /// pseudo-tag, `keep_tag` itself, and fork capability lanes survive;
    /// cross-kind rows are untouched. Returns the freed (tag, bytes)
    /// pairs for the summary.
    pub fn prune_siblings(&self, kind: &str, keep_tag: &str) -> Result<Vec<(String, u64)>> {
        let store = Store::open(&self.dirs)?;
        let engines = store.list_engines()?;
        let mut freed = Vec::new();
        for e in engines {
            if e.kind.as_str() != kind
                || e.tag == keep_tag
                || e.tag == LOCAL_TAG
                // Fork lanes are additive siblings, not superseded
                // currency: an upstream update must not delete a lane
                // some model still depends on.
                || is_fork_lane(&e)
            {
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
    let mut cmd = std::process::Command::new(bin);
    cmd.args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    // Bounded transient retry (EAGAIN-class under parallel load) — a
    // permanent failure (missing/garbage binary) still fails attempt one.
    let Ok(child) = crate::probe::with_spawn_retry(|| cmd.spawn()) else {
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
    // Established unit__scenario__expected naming convention for the suite.
    #![allow(non_snake_case)]
    use super::*;
    use std::path::PathBuf;

    fn fake_bin(dir: &std::path::Path, rel: &str, body: &str) -> PathBuf {
        let p = dir.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
        make_executable(&p);
        p
    }

    // Only the unix-gated probe tests read the manifest; Windows has no
    // success-path tests in this module (fake bins are shell scripts).
    #[cfg(unix)]
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
    // The fake bins are unix shell scripts; Windows cannot spawn an
    // extensionless script, so the success-path probes run on unix only.
    #[cfg(unix)]
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
    // The fake bins are unix shell scripts; Windows cannot spawn an
    // extensionless script, so the success-path probes run on unix only.
    #[cfg(unix)]
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
    // The fake bins are unix shell scripts; Windows cannot spawn an
    // extensionless script, so the success-path probes run on unix only.
    #[cfg(unix)]
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

    /// `register_or_clean` pins: an unusable binary (exec-format /
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

    /// `install_with_rollback` contract: a failed replacement must never
    /// cost the previously installed engine (dir + row), and a successful
    /// one must not leak the retired copy. Linux-only: the fixtures are
    /// /bin/sh scripts.
    #[cfg(target_os = "linux")]
    mod install_with_rollback_tests {
        use super::super::*;
        use super::fake_bin;
        use pallama_core::engine_kind::EngineKind;
        use pallama_core::store::{EngineRow, Store};

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

        /// Installed engine to protect: healthy binary, a marker file,
        /// and the store row that serves it.
        fn seed_installed_engine(mgr: &EngineManager, dir: &std::path::Path, tag: &str) {
            std::fs::create_dir_all(dir).unwrap();
            healthy_server(dir);
            std::fs::write(dir.join("marker"), b"previous build").unwrap();
            let store = Store::open(&mgr.dirs).unwrap();
            store
                .upsert_engine(&EngineRow {
                    tag: tag.to_string(),
                    asset: "old".into(),
                    sha256: "old".into(),
                    installed_at: 1,
                    active: true,
                    manifest: "{}".into(),
                    kind: EngineKind::LlamaCpp,
                })
                .unwrap();
        }

        fn aside_leftovers(mgr: &EngineManager) -> Vec<String> {
            let engines_dir = mgr.dirs.engines_dir();
            if !engines_dir.exists() {
                return Vec::new();
            }
            std::fs::read_dir(engines_dir)
                .unwrap()
                .flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n.starts_with(".retired-"))
                .collect()
        }

        #[tokio::test]
        async fn unit__install_with_rollback__build_failure_restores_existing_engine() {
            let tmp = tempfile::tempdir().unwrap();
            let mgr = manager_in(&tmp);
            let dir = tmp.path().join("data/engines/b1-cuda");
            seed_installed_engine(&mgr, &dir, "b1-cuda");
            let err = mgr
                .install_with_rollback("b1-cuda", "lbl", "x", EngineKind::LlamaCpp, |_dir| async {
                    Err(anyhow!("simulated disk-full mid-build"))
                })
                .await
                .expect_err("failed build must surface");
            assert!(
                format!("{err:#}").contains("disk-full"),
                "original error must survive the rollback: {err:#}"
            );
            assert!(dir.join("llama-server").exists(), "old binary survives");
            assert!(dir.join("marker").exists(), "old dir content survives");
            let store = Store::open(&mgr.dirs).unwrap();
            assert_eq!(store.list_engines().unwrap().len(), 1, "row survives");
            assert!(aside_leftovers(&mgr).is_empty(), "no aside lingered");
        }

        #[tokio::test]
        async fn unit__install_with_rollback__probe_failure_restores_existing_engine() {
            let tmp = tempfile::tempdir().unwrap();
            let mgr = manager_in(&tmp);
            let dir = tmp.path().join("data/engines/b2-cuda");
            seed_installed_engine(&mgr, &dir, "b2-cuda");
            let err = mgr
                .install_with_rollback(
                    "b2-cuda",
                    "lbl",
                    "x",
                    EngineKind::LlamaCpp,
                    |dir| async move {
                        // Garbage bytes with the +x bit: spawn fails at
                        // exec — the register-tail failure class a
                        // SIGILL'd asset hits.
                        std::fs::create_dir_all(&dir).unwrap();
                        std::fs::write(dir.join("llama-server"), b"not an elf").unwrap();
                        make_executable(&dir.join("llama-server"));
                        Ok(())
                    },
                )
                .await
                .expect_err("garbage binary must fail the probe");
            assert!(
                format!("{err:#}").contains(ENGINE_PROBE_FAILED),
                "probe-class failures carry the marker: {err:#}"
            );
            assert!(
                dir.join("marker").exists(),
                "old dir content restored after probe failure"
            );
            let store = Store::open(&mgr.dirs).unwrap();
            assert_eq!(store.list_engines().unwrap().len(), 1, "row survives");
            assert!(aside_leftovers(&mgr).is_empty(), "no aside lingered");
        }

        #[tokio::test]
        async fn unit__install_with_rollback__success_replaces_and_discards_retired() {
            let tmp = tempfile::tempdir().unwrap();
            let mgr = manager_in(&tmp);
            let dir = tmp.path().join("data/engines/b3-cuda");
            seed_installed_engine(&mgr, &dir, "b3-cuda");
            let row = mgr
                .install_with_rollback(
                    "b3-cuda",
                    "lbl",
                    "x",
                    EngineKind::LlamaCpp,
                    |dir| async move {
                        std::fs::create_dir_all(&dir).unwrap();
                        healthy_server(&dir);
                        Ok(())
                    },
                )
                .await
                .expect("healthy replacement registers");
            assert_eq!(row.tag, "b3-cuda");
            assert!(!dir.join("marker").exists(), "old content replaced");
            assert!(dir.join("llama-server").exists(), "new binary in place");
            let store = Store::open(&mgr.dirs).unwrap();
            assert_eq!(
                store.list_engines().unwrap().len(),
                1,
                "one row for the tag"
            );
            assert!(aside_leftovers(&mgr).is_empty(), "retired copy discarded");
        }

        #[tokio::test]
        async fn unit__install_with_rollback__fresh_install_failure_leaves_no_dir() {
            let tmp = tempfile::tempdir().unwrap();
            let mgr = manager_in(&tmp);
            let err = mgr
                .install_with_rollback(
                    "fresh-cuda",
                    "lbl",
                    "x",
                    EngineKind::LlamaCpp,
                    |_dir| async { Err(anyhow!("boom")) },
                )
                .await
                .expect_err("failure surfaces");
            assert!(
                format!("{err:#}").contains("boom"),
                "original error must survive: {err:#}"
            );
            assert!(
                !mgr.dirs.engines_dir().join("fresh-cuda").exists(),
                "F88: no orphan dir from a failed fresh install"
            );
            assert!(aside_leftovers(&mgr).is_empty());
        }
    }

    mod scanback_tests {
        use super::super::*;

        fn cuda_release(tag: &str, asset_names: &[&str]) -> GhRelease {
            GhRelease {
                tag_name: tag.to_string(),
                prerelease: false,
                assets: asset_names
                    .iter()
                    .map(|n| gh::GhAsset {
                        name: (*n).to_string(),
                        digest: None,
                        size: None,
                        browser_download_url: format!("https://x/{n}"),
                    })
                    .collect(),
                published_at: None,
            }
        }

        fn x64_lane(channel_number: u64) -> CudaLane {
            CudaLane {
                driver_cuda: (13, 0),
                sm: Some(89),
                arch: "x64",
                number: channel_number,
            }
        }

        #[test]
        fn unit__scanback_release__picks_newest_assetful_strictly_behind() {
            // The exact 09-17 shape: channel b11027 ships no ubuntu-cuda
            // assets; b11026 does; a NEWER b11028 exists upstream but is
            // not behind the channel target; b11020 also has assets but
            // is older than b11026.
            let releases = vec![
                cuda_release("b11028", &["llama-b11028-bin-ubuntu-cuda-12.8-x64.tar.gz"]),
                cuda_release("b11027", &["llama-b11027-bin-ubuntu-vulkan-x64.tar.gz"]),
                cuda_release("b11026", &["llama-b11026-bin-ubuntu-cuda-12.8-x64.tar.gz"]),
                cuda_release("b11025", &["llama-b11025-bin-ubuntu-vulkan-x64.tar.gz"]),
                cuda_release("b11020", &["llama-b11020-bin-ubuntu-cuda-12.8-x64.tar.gz"]),
            ];
            let hit = scanback_release(&releases, &x64_lane(11027))
                .expect("b11026 is the newest assetful release behind");
            assert_eq!(hit.tag_name, "b11026");
        }

        #[test]
        fn unit__scanback_release__depth_cap_returns_none() {
            // Assets exist 17 builds behind — beyond UPSTREAM_SCANBACK_DEPTH.
            let mut releases = Vec::new();
            for n in (11010..=11027).rev() {
                let assets: Vec<String> = if n == 11010 {
                    vec!["llama-b11010-bin-ubuntu-cuda-12.8-x64.tar.gz".to_string()]
                } else {
                    vec![format!("llama-b{n}-bin-ubuntu-vulkan-x64.tar.gz")]
                };
                releases.push(cuda_release(
                    &format!("b{n}"),
                    &assets.iter().map(String::as_str).collect::<Vec<_>>(),
                ));
            }
            assert!(
                scanback_release(&releases, &x64_lane(11027)).is_none(),
                "scan must not wander past the depth cap into stale runtimes"
            );
        }

        #[test]
        fn unit__scanback_release__driver_cap_skips_overcap_candidates() {
            // b11026 ships only 13.3 assets — over this driver's 13.0
            // ceiling — so the scan continues to b11020's 12.8 asset.
            let releases = vec![
                cuda_release("b11026", &["llama-b11026-bin-ubuntu-cuda-13.3-x64.tar.gz"]),
                cuda_release("b11020", &["llama-b11020-bin-ubuntu-cuda-12.8-x64.tar.gz"]),
            ];
            let hit = scanback_release(&releases, &x64_lane(11027))
                .expect("13.3-only release is skipped, 12.8 picked");
            assert_eq!(hit.tag_name, "b11020");
        }
    }
}
