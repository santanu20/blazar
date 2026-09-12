//! Instance supervisor: spawn llama-server children per model, keep them
//! healthy, evict on the idle ladder, respawn after crashes with a circuit
//! breaker, and shut everything down cleanly.
//!
//! Invariants (§9): every spawned child has a named stop path (evict /
//! shutdown), teardown is idempotent, SIGTERM→grace→SIGKILL bounded, and
//! every state transition publishes an event.

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use dashmap::DashMap;
use tokio::sync::Notify;

use pallama_core::profile::{self, Endpoint, ProfileInput};
use pallama_core::store::Store;
use pallama_core::{Config, GpuInfo, Hardware, ModelRow, PallamaDirs};

use crate::events::{EventBus, InstanceState, PallamaEvent};

/// Resolve the draft-model file path for `model` under `spec_mode`: the
/// catalog's typed pair (eagle3/dflash/dspark) or the generic pair
/// (auto), materialized only when that draft is a pulled store row.
/// Modes that never use an external draft (off/mtp/ngram*) return None
/// without touching the store. Shared by serve `ensure`, the router
/// preset, and the CLI bench/tune lanes so every spawn path resolves
/// drafts identically — profile-compile still gates emission on the
/// engine manifest and hard-errors on stale (missing) files.
pub fn resolve_draft_path(store: &Store, model: &str, spec_mode: &str) -> Option<String> {
    let pair = match spec_mode {
        "eagle3" => pallama_core::spec_pair_for_typed(model, "draft-eagle3"),
        "dflash" => pallama_core::spec_pair_for_typed(model, "draft-dflash"),
        "dspark" => pallama_core::spec_pair_for_typed(model, "draft-dspark"),
        // off/mtp/ngram never consume an external draft file.
        "off" | "mtp" | "ngram" | "ngram-map-k" | "ngram-map-k4v" | "ngram-mod" | "ngram-cache" => {
            None
        }
        _ => pallama_core::spec_pair_for(model),
    }?;
    let (repo_part, file_part) = pair
        .draft_repo
        .split_once(':')
        .unwrap_or((&pair.draft_repo, ""));
    let draft_name = crate::hf::draft_aware_name(repo_part, file_part);
    store.get_model(&draft_name).ok().flatten().map(|r| r.path)
}

/// Instance key for the single router-mode child (never a model name:
/// underscore prefix is invalid in HF repo names).
pub const ROUTER_KEY: &str = "_router";

/// Hard cap on `replicas` per model (B1). Beyond this, capacity fights
/// the point of replication; users wanting more can raise
/// `max_loaded_models` and stack models.
pub const MAX_REPLICAS: u32 = 8;

/// Prompt-prefix identity for cache-aware replica routing (B1/F8):
/// `sys` = shared-prefix class (head of the system prompt), `convo` =
/// conversation identity (system + first user turn head). Identical
/// `convo` sticks to its warm replica; a *new* conversation whose `sys`
/// class matches a live replica coalesces onto it (shared system-prompt
/// KV reuse) instead of growing a cold child.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixKey {
    pub sys: u64,
    pub convo: u64,
}

/// Recent `sys` classes remembered per replica key (F8 overlap routing).
const SYS_RING_CAP: usize = 8;
/// Bound on the prefix→replica affinity table (B1): one entry per
/// distinct prompt prefix; eviction keeps it from growing unbounded.
const PREFIX_AFFINITY_CAP: usize = 512;
/// LC1: minimum A→B transitions before B is considered a confident
/// preload target.
const PRELOAD_MIN_TRANSITIONS: u64 = 3;
/// LC1: cap on the transition table (arbitrary-evict on overflow).
const TRANSITIONS_CAP: usize = 1024;
/// LC1: cooldown after a failed speculative spawn of a model.
const PRELOAD_BACKOFF: Duration = Duration::from_mins(5);
/// F12: `keep_alive: -1` cap — 100 years, far enough to be forever in
/// practice without risking `Instant` overflow arithmetic.
const KEEP_ALIVE_FOREVER: Duration = Duration::from_hours(876_600);
/// LC4: reaper ticks (10s each) of sustained concurrent load before a
/// slots bump is adopted.
const SLOTS_STREAK_TICKS: u32 = 6;
/// LC4: max in-memory slots bump (`tune --slots` for higher). Raised
/// 4 → 8 on 2026-09-12: the np8 shape measured +19% system t/s and a
/// 5.4x concurrency TTFT-p99 win over np4 under 8-stream load
/// (scaling flag-space study, /tmp/opencode/flagprobe/results2.json).
const SLOTS_ADOPT_CAP: u32 = 8;
/// Quiet reaper ticks before an adoption decays back to the natural
/// shape (30 x 10s = 5 min of zero pressure and zero in-flight). The
/// asymmetry vs [`SLOTS_STREAK_TICKS`] is deliberate anti-flap hysteresis:
/// scaling up must be eager (demand is now), scaling down must be lazy
/// (the cost of waiting is only per-stream latency, never queueing).
const SLOTS_DECAY_TICKS: u32 = 30;

/// Model name behind an instance key: `"qwen#2"` → `"qwen"`. Plain keys
/// (no `#`) pass through unchanged, so `replicas = 1` stays
/// byte-identical with the pre-replica world.
/// A trailing `@vision` marker (projector-carrying respawn of a
/// text-only instance, see `ensure_vision`) is stripped too, so every
/// consumer sees the plain model name.
fn model_of_key(key: &str) -> &str {
    let key = key.split('@').next().unwrap_or(key);
    match key.split_once('#') {
        Some((model, _)) => model,
        None => key,
    }
}

/// Not-found payload with the flat-name teaching line for ollama
/// `model:tag` input (reached only when BOTH forms missed, so the
/// swapped spelling is a suggestion, never a promise).
fn not_found_name(name: &str) -> String {
    if name.contains(':') {
        format!(
            "{name} — pallama names are flat; `{}` would be its flat form (see `pallama list`)",
            name.replace(':', "-")
        )
    } else {
        name.to_string()
    }
}

/// Split an instance key into (model, replica index). `None` for plain
/// model keys and malformed suffixes. A `@vision` suffix after the
/// replica index is tolerated so `evict_model`-style filters match.
fn split_replica(key: &str) -> Option<(&str, u32)> {
    let key = key.split('@').next().unwrap_or(key);
    let (model, idx) = key.split_once('#')?;
    let idx = idx.parse::<u32>().ok()?;
    Some((model, idx))
}

use crate::engine_impl::{ChildHandle, Engine};

/// Typed failures the gateway maps onto HTTP statuses.
#[derive(Debug, thiserror::Error)]
pub enum SupervisionError {
    #[error("no such model: {0} (try `pallama pull`)")]
    ModelNotFound(String),
    #[error("engine crashed while loading or serving {0}")]
    EngineCrashed(String),
    #[error("model {0} did not become healthy in time")]
    ModelLoadTimeout(String),
    #[error("circuit open for {0}: engine restarted >3 times in 60s; run `pallama ps --reset`")]
    CircuitOpen(String),
    #[error("all slots busy: capacity reached and nothing idle to evict")]
    AllSlotsBusy,
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}

#[derive(Debug)]
pub struct Instance {
    pub name: String,
    pub endpoint: Endpoint,
    /// Shared interior state: mutated by ensure/reaper under this lock.
    pub state: std::sync::RwLock<InstanceState>,
    pub last_used: std::sync::RwLock<Instant>,
    pub in_flight: AtomicI64,
    /// Client-requested pin window (ollama `keep_alive`): `Some(t)` keeps
    /// the instance exempt from idle evict AND idle sleep until `t`
    /// (F12). `-1` maps to a far-future cap; `0` clears and the explicit
    /// evict-after-response path owns teardown.
    pub keep_until: std::sync::RwLock<Option<Instant>>,
    pub started_at: Instant,
    pub argv: Vec<String>,
    pub model: ModelRow,
    pub profile_ctx: u32,
    /// Resolved offload label from the compiled profile ("full" | "cpu" |
    /// "partial" | "auto") — what `ps` shows so silent CPU fallback is
    /// never silent.
    pub gpu: String,
    /// Auto-picked GPU card name (LC2 discrete-first pick); `None` when
    /// placement was manual (devices config) or unknown. Feeds the
    /// card-scoped co-residency planner.
    pub device: Option<String>,
    /// Post-quantization KV-cache estimate from the compiled profile —
    /// feeds the co-residency planner (A15).
    pub kv_est_bytes: Option<u64>,
    /// Profile-compile warnings (unified-KV fit, slot auto, gpu-offload
    /// rationale…) — kept on the instance so `ps` and the run path can
    /// surface them; the daemon tracing loop stays for logs.
    pub warnings: Vec<String>,
    /// Measured card free-VRAM delta from the spawn settle report — the
    /// footprint the child ACTUALLY took (weights + context + compute +
    /// KV working set). 0 = not settled yet. The bytes admission prefers
    /// this over the weights sum: on 2026-09-11 a 9B VL spawn measured
    /// 7302 MiB against 5417 weights, and a weights-only admission let a
    /// 0.5B sibling (~977 MiB actual) join it into an `NVRM NO_MEMORY`
    /// storm. Relaxed ordering: telemetry-grade hint, not a lock.
    pub settled_mib: std::sync::atomic::AtomicU64,
    /// Per-child bearer secret (child `--api-key` hardening). Lifecycle
    /// = child lifecycle; `None` on UDS children (filesystem perms
    /// already gate the socket) and on engines lacking `--api-key`.
    pub auth: Option<String>,
    /// Whether this child was spawned WITH the multimodal projector
    /// (`-mm`/`--mmproj` in the compiled argv). Under the default Lazy
    /// policy text-only spawns skip the projector (measured: 875 MiB
    /// file, ~3.9 s cold TTFT, 1126 MiB VRAM) and `ensure_vision`
    /// respawns projector-carrying on the first image request. Derived
    /// from the argv the compiler actually emitted — never assumed.
    pub projector: bool,
    child: tokio::sync::Mutex<ChildHandle>,
    pid: u32,
}

/// Measured prompt-cache hit rate, shared between the gateway poller
/// (writer) and profile compiles (reader). Fixed-point milli-units so
/// the read path is a single atomic load — no lock on the spawn hot path.
#[derive(Debug)]
pub struct CacheHint {
    rate_milli: std::sync::atomic::AtomicU32,
}

/// Auto-pick one GPU: most free VRAM among DISCRETE cards; integrated
/// cards are considered only when no discrete card exists (their "free"
/// is shared system RAM — bandwidth-starved for serving). Returns the
/// index of the pick plus whether a higher-free integrated card was
/// skipped (caller logs the teaching note). `None` = nothing to pick.
fn pick_gpu(gpus: &[pallama_core::GpuInfo]) -> Option<(usize, bool)> {
    if gpus.is_empty() {
        return None;
    }
    let (top_free, _) = gpus
        .iter()
        .enumerate()
        .max_by_key(|(_, g)| g.free_mib)
        .expect("non-empty gpu slice has a max");
    match gpus
        .iter()
        .enumerate()
        .filter(|(_, g)| !g.is_integrated())
        .max_by_key(|(_, g)| g.free_mib)
    {
        Some((idx, _)) => Some((idx, gpus[top_free].is_integrated())),
        // Integrated-only box: serve anyway (laptop iGPU is still a GPU).
        None => Some((top_free, false)),
    }
}

/// Last-resort auto tensor-split planning (#28c): when weights+KV exceed
/// the best single discrete card's MEASURED free VRAM but fit the
/// discrete cards COMBINED, return a `--tensor-split` ratio string
/// proportional to each card's free VRAM. `None` = no split (fits the
/// best card, fewer than two discrete cards, or not fixable by
/// splitting). Integrated cards never join the pool (shared-RAM
/// bandwidth fiction). Callers gate on every manual pin being unset —
/// this fn is placement math, not policy.
fn plan_auto_tensor_split(
    gpus: &[pallama_core::GpuInfo],
    weights_mib: u64,
    kv_mib: u64,
) -> Option<String> {
    let discrete: Vec<&pallama_core::GpuInfo> =
        gpus.iter().filter(|g| !g.is_integrated()).collect();
    if discrete.len() < 2 {
        return None; // splitting needs at least two discrete cards
    }
    let need = weights_mib.saturating_add(kv_mib);
    let best_free = discrete.iter().map(|g| g.free_mib).max()?;
    if need <= best_free {
        return None; // fits the best card: single-card placement wins
    }
    let combined: u64 = discrete.iter().map(|g| g.free_mib).sum();
    if need > combined {
        return None; // beyond splitting: the CPU-spill lane teaches instead
    }
    // Ratios relative to the smallest free card (llama.cpp takes relative
    // weights): 8188 + 3996 free → "2,1". Clamped ≥1 so a nearly-full
    // card still receives layers (upstream needs every listed device).
    let min_free = discrete.iter().map(|g| g.free_mib).min()?.max(1);
    Some(
        discrete
            .iter()
            .map(|g| (g.free_mib / min_free).clamp(1, 9999).to_string())
            .collect::<Vec<_>>()
            .join(","),
    )
}

/// Post-spawn settle report (#28a): compare the picked card's free VRAM
/// before spawn vs after the child came healthy, returning
/// (card name, measured take in MiB, card-used percentage). `device`
/// names the picked card; `None` falls back to the max-free card of the
/// AFTER snapshot (manual-pin boxes have no pick to attribute to).
fn settle_report(
    pre: &pallama_core::Hardware,
    post: &pallama_core::Hardware,
    device: Option<&str>,
) -> Option<(String, u64, u64)> {
    let pre_free_of = |name: &str| {
        pre.gpus
            .iter()
            .find(|g| g.name == name)
            .map_or(0, |g| g.free_mib)
    };
    // A named pick must exist in the AFTER snapshot (a card that vanished
    // from the census reports garbage, not zero). Without a pick, report
    // the card whose free VRAM dropped the most — the card the spawn
    // plausibly landed on.
    let card = match device {
        Some(name) => post.gpus.iter().find(|g| g.name == name)?,
        None => post
            .gpus
            .iter()
            .max_by_key(|g| pre_free_of(&g.name).saturating_sub(g.free_mib))?,
    };
    let pre_free = pre_free_of(&card.name);
    let taken = pre_free.saturating_sub(card.free_mib);
    let used_pct = ((card.total_mib.saturating_sub(card.free_mib)) * 100)
        .checked_div(card.total_mib)
        .unwrap_or(0);
    Some((card.name.clone(), taken, used_pct))
}

/// Per-card co-residency pressure (A15, card-scoped): would the candidate
/// (weights + f16 KV) plus everything already resident on the TARGET card
/// exceed 95% of that card? Instances whose placement is unknown (spawned
/// pre-device-recording, or manual multi-card splits) count
/// conservatively against the target. CPU/partial splits live elsewhere
/// and never count.
#[allow(clippy::type_complexity)]
fn card_pressure_exceeds(
    target: Option<&str>,
    gpus: &[pallama_core::GpuInfo],
    residents: &[(Option<String>, u64, u64, bool)], // (device, weights, kv, gpu_resident)
    candidate_bytes: u64,
    candidate_kv: u64,
) -> Option<bool> {
    if gpus.is_empty() {
        return None;
    }
    // The target card's VRAM — fall back to the summed pool only when
    // the placement is unknown (None), matching pre-scoping behavior.
    let vram = pallama_core::Hardware::bytes(match target {
        Some(name) => gpus.iter().find(|g| g.name == name)?.total_mib,
        None => gpus.iter().map(|g| g.total_mib).sum(),
    });
    let mut resident = candidate_bytes.saturating_add(candidate_kv);
    for (device, weights, kv, gpu_resident) in residents {
        if !gpu_resident {
            continue;
        }
        let same_or_unknown = match (&device, target) {
            (Some(d), Some(t)) => d == t,
            // Unknown placement (or no target scope): count conservatively.
            (None, _) | (Some(_), None) => true,
        };
        if same_or_unknown {
            resident = resident.saturating_add(*weights).saturating_add(*kv);
        }
    }
    Some(resident > vram / 100 * 95)
}

impl Default for CacheHint {
    fn default() -> Self {
        // MAX = "never measured": `get()` yields None until the poller
        // has observed a full window.
        Self {
            rate_milli: std::sync::atomic::AtomicU32::new(u32::MAX),
        }
    }
}

impl CacheHint {
    /// Record the current window's hit rate (0.0..=1.0).
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "clamped to [0, 1000] before the cast; rounding is the point"
    )]
    pub fn set(&self, rate: f64) {
        let clamped = rate.clamp(0.0, 1.0);
        let milli = (clamped * 1000.0).round();
        let milli = u32::try_from(milli.min(f64::from(u32::MAX)) as u64).unwrap_or(0);
        self.rate_milli.store(milli, Ordering::Relaxed);
    }
    /// Latest hit rate; None until the poller has seen a full window.
    #[must_use]
    pub fn get(&self) -> Option<f64> {
        match self.rate_milli.load(Ordering::Relaxed) {
            u32::MAX => None,
            m => Some(f64::from(m) / 1000.0),
        }
    }
}

/// Row for `pallama ps` / `/api/ps`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PsRow {
    pub name: String,
    /// Replica index when the model runs multi-instance (`model#N`,
    /// B1); `None` for plain single-instance models.
    pub replica: Option<u32>,
    pub state: &'static str,
    pub endpoint: String,
    pub idle_secs: u64,
    pub in_flight: i64,
    pub ctx: u32,
    /// Offload label ("full" | "cpu" | "partial" | "auto") — surfaces
    /// silent CPU fallback, ollama's most-common complaint.
    pub gpu: String,
    /// Card the instance was placed on (auto-pick or `devices` config);
    /// `None` for router mode / unknown placement. `ps` renders it as
    /// `full@<card>`.
    pub device: Option<String>,
    /// Profile-compile warnings for this instance (see Instance.warnings).
    pub warnings: Vec<String>,
    pub pid: u32,
    /// Model bytes on disk (0 in router mode — the front child serves
    /// many models and owns no single size).
    pub bytes: i64,
    /// Prefix heat (computed under the INSTANCE key so replica rows
    /// carry their own warmth).
    pub heat: u64,
    /// Remaining ollama `keep_alive` pin window (F12); `None` when the
    /// instance follows the default idle policy. `ps` surfaces it as the
    /// countdown behind `expires_at`.
    pub keep_alive_secs: Option<u64>,
}

/// Throttle/dedup state for the measured-pressure feedback loop.
#[derive(Default)]
struct MeasuredTick {
    /// Last hardware re-probe (probes spawn `--list-devices`; ≥60s apart).
    last_probe: Option<Instant>,
    /// card name → last pressure-teach instant (10-minute dedup).
    warned_cards: std::collections::HashMap<String, Instant>,
}

pub struct Supervisor {
    pub dirs: PallamaDirs,
    /// Total evictions (ladder + capacity) for the metrics gauge.
    pub evictions: std::sync::atomic::AtomicU64,
    /// Measured prompt-cache hit rate (gateway poller writes, profile
    /// compiles read — drives the adaptive `--cache-ram` clamp, A16).
    pub cache_hint: std::sync::Arc<CacheHint>,
    /// Speculative-decoding acceptance rate (accepted / drafted tokens,
    /// 0.0..=1.0, EWMA'd by the gateway poller). `None` until a
    /// spec-decoding child has reported counters — drives the
    /// `pallama_spec_accept_rate` gauge (G3).
    pub spec_accept: std::sync::Arc<CacheHint>,
    /// J2 self-healing: distinct models whose spawn ULTIMATELY failed
    /// since the last successful spawn. Cleared on every success; drives
    /// the crash-loop engine rollback (probe-gated, see
    /// `note_engine_failure`).
    engine_failures: std::sync::Mutex<std::collections::HashSet<String>>,
    pub config: Config,
    pub bus: EventBus,
    pub hardware: Hardware,
    pub engine: Arc<dyn Engine>,
    instances: DashMap<String, Arc<Instance>>,
    /// F1: names with an evict currently in flight. A spawn that wins the
    /// race against a slow terminate must NOT insert under the name being
    /// torn down — the evict's cleanup would delete the fresh instance
    /// (untracked child, `SIGKILL`ed mid-request by `kill_on_drop`) and shred
    /// its pidfile/apikey. Spins retry after the evict completes.
    /// In-flight evictions (F1). tokio Mutex: the RAII mark lives
    /// across the whole async evict (bank save + terminate grace can
    /// run ~12s worst-case) — a std Mutex held across those awaits
    /// pins every worker that touches the set (spawn deferral check,
    /// idle ladder) and panics cascade on poisoning.
    evicting: tokio::sync::Mutex<std::collections::HashSet<String>>,
    /// In-flight spawns: waiters subscribe to the Notify for completion.
    loading: DashMap<String, Arc<Notify>>,
    /// Completed load results for waiters: Ok(port) or error text.
    load_results: DashMap<String, Result<EngineRef, String>>,
    /// Crash-restart timestamps per model (circuit breaker). Recorded
    /// ONLY for spawns that consume an unclean-death mark — a user
    /// stop→run churn is a cold start, not a crash restart.
    restarts: DashMap<String, Vec<Instant>>,
    /// Models whose child died UNCLEANLY (non-zero exit) and whose next
    /// successful spawn therefore counts as a crash restart. Set by
    /// `reap_dead_children`, consumed by the spawn success path,
    /// cleared by a clean evict (user stop / idle / capacity).
    unclean_dead: DashMap<String, ()>,
    /// One-shot ctx override for the NEXT spawn of a model (per-request
    /// `options.num_ctx` — complaint #13). Consumed on use.
    pending_ctx: DashMap<String, u32>,
    /// Prefix heat per model: (hits, last-hit instant); decayed on read.
    heat: std::sync::Mutex<std::collections::HashMap<String, (u64, Instant)>>,
    /// Prompt-prefix → replica-key affinity (B1): hashes of
    /// (system + first user turn) pin a conversation to one warm-cache
    /// replica. Bounded, best-effort — a miss just re-routes.
    prefix_affinity: DashMap<u64, String>,
    /// Per-replica ring of recently served `sys` prefix classes (F8).
    sys_rings: DashMap<String, std::collections::VecDeque<u64>>,
    /// Model→model transition counts (LC1 predictive pre-loading):
    /// increments when a successful request for B follows one for A.
    transitions: DashMap<(String, String), u64>,
    /// Most recently served model (LC1 transition edge source).
    last_requested: std::sync::Mutex<Option<String>>,
    /// Models whose speculative preload failed recently (LC1 backoff).
    preload_failures: DashMap<String, Instant>,
    /// Consecutive reaper ticks with concurrent in-flight load on a
    /// single-slot model (LC4 adaptive slots).
    busy_streak: DashMap<String, u32>,
    /// Consecutive fully-quiet ticks per instance key (no admission
    /// pressure, zero in-flight) feeding the adoption decay rule.
    idle_streak: DashMap<String, u32>,
    /// In-memory slots bumps adopted by LC4 (restart resets; `tune
    /// --slots` is the permanent path).
    adopted_slots: DashMap<String, u32>,
    /// LC4 respawn queue: model → instance key adopted this pass. The
    /// reaper drain respawns each entry once its streams drain
    /// (in-flight = 0), so a slot-heavy reshape never kills live
    /// streams; under 24/7 saturation it defers to the natural
    /// idle-evict respawn instead.
    reshape_queue: DashMap<String, String>,
    /// Admission-pressure events per model since the last reaper tick:
    /// every gateway request that hit `AllSlotsBusy` and queued (the
    /// demand signal for adaptive slots — engine `in_flight` can never
    /// exceed the slot count because admission caps it, so queue-wait
    /// is the only reachable saturation indicator; live-proven
    /// 2026-09-12 when a 6-stream load left `in_flight` pinned at slots).
    slot_pressure: DashMap<String, u32>,
    /// Session pins (R3): sessions that recently carried
    /// `x-pallama-session` per model. The idle ladder and capacity
    /// pressure consult this before evicting; force stop releases.
    pub sessions: crate::sessionreg::SessionRegistry,
    /// Measured-pressure feedback loop: throttle state for the reaper
    /// tick's hardware re-probe + per-card teach dedup (warn once per
    /// card per 10 minutes, probe at most once per minute).
    measured: std::sync::Mutex<MeasuredTick>,
    /// Live-census TTL cache: back-to-back spawns (cold-start burst,
    /// preload, pressure tick within seconds) each paid the ~0.2-0.26 s
    /// `--list-devices` subprocess — measured on the cold path (2026-09-11
    /// forensics). Cached for [`CENSUS_TTL`]; invalidated on spawn
    /// failure so the pre-spawn guard never trusts a census that
    /// predates a failed boot.
    census_cache: std::sync::Mutex<Option<(Instant, Hardware)>>,
    // Test knobs (prod defaults from config).
    pub load_timeout: Duration,
    pub shutdown_grace: Duration,
    pub reaper_interval: Duration,
    pub circuit_window: Duration,
    pub max_restarts: usize,
}

#[derive(Debug, Clone)]
pub struct EngineRef {
    pub name: String,
    pub endpoint: Endpoint,
    /// Per-child bearer secret, stamped on every gateway call to this
    /// child (`proxy::child_auth` is the single choke point). Never
    /// serialized into `ps`/API output.
    pub auth: Option<String>,
}

/// Minted child-auth bundle from [`Supervisor::mint_child_auth`].
struct ChildAuth {
    /// argv fragment appended to the spawn command.
    argv: Vec<String>,
    /// The secret itself (Instance/EngineRef cargo).
    secret: String,
    /// Keyfile removed at teardown (None when argv carries the secret).
    keyfile: Option<std::path::PathBuf>,
}

impl Supervisor {
    #[allow(clippy::duration_suboptimal_units)] // plain second counts
    pub fn new(
        dirs: PallamaDirs,
        config: Config,
        bus: EventBus,
        hardware: Hardware,
        engine: Arc<dyn Engine>,
    ) -> Self {
        Self {
            evictions: std::sync::atomic::AtomicU64::new(0),
            cache_hint: std::sync::Arc::new(CacheHint::default()),
            spec_accept: std::sync::Arc::new(CacheHint::default()),
            engine_failures: std::sync::Mutex::new(std::collections::HashSet::new()),
            measured: std::sync::Mutex::new(MeasuredTick::default()),
            census_cache: std::sync::Mutex::new(None),
            load_timeout: Duration::from_secs(3 * 60),
            shutdown_grace: Duration::from_secs(10),
            reaper_interval: Duration::from_secs(10),
            circuit_window: Duration::from_secs(60),
            max_restarts: 3,
            dirs,
            config,
            bus,
            hardware,
            engine,
            instances: DashMap::new(),
            evicting: tokio::sync::Mutex::new(std::collections::HashSet::new()),
            loading: DashMap::new(),
            load_results: DashMap::new(),
            restarts: DashMap::new(),
            unclean_dead: DashMap::new(),
            pending_ctx: DashMap::new(),
            heat: std::sync::Mutex::new(std::collections::HashMap::new()),
            prefix_affinity: DashMap::new(),
            sys_rings: DashMap::new(),
            transitions: DashMap::new(),
            last_requested: std::sync::Mutex::new(None),
            preload_failures: DashMap::new(),
            busy_streak: DashMap::new(),
            idle_streak: DashMap::new(),
            adopted_slots: DashMap::new(),
            reshape_queue: DashMap::new(),
            slot_pressure: DashMap::new(),
            sessions: crate::sessionreg::SessionRegistry::new(),
        }
    }

    /// Hard instance-count cap: explicit `max_loaded_models` wins as a
    /// pure count; CPU-only boxes pin 1 (page cache is shared, but the
    /// evictor's heat model does not reason about it). `None` on GPU
    /// boxes in auto mode — the bytes admission below governs there.
    #[must_use]
    pub fn instance_cap(&self) -> Option<usize> {
        if self.config.max_loaded_models > 0 {
            return Some(self.config.max_loaded_models as usize);
        }
        if !self.hardware.has_gpu() {
            return Some(1);
        }
        None
    }

    /// Bytes admission active exactly when the count cap is not: GPU box,
    /// auto mode. Heterogeneous-friendly: what matters is the SUM of
    /// resident weights vs the VRAM budget, not VRAM divided by the
    /// largest model (a 0.5B + 9B pair co-resides where the old
    /// floor-division formula said capacity 1).
    #[must_use]
    pub fn bytes_admission_active(&self) -> bool {
        self.config.max_loaded_models == 0 && self.hardware.has_gpu()
    }

    /// Sum of resident instance footprints for the bytes admission:
    /// the MEASURED settle delta once known (weights, context, compute
    /// and KV working set), floored at the weights sum while the settle
    /// probe has not run yet. CPU-placed children never count (their
    /// pages live in system RAM, guarded by the J3 `MemAvailable`
    /// floor); each replica mmaps its own copy, so GPU-resident VRAM
    /// sums.
    #[must_use]
    pub fn resident_bytes(&self) -> u64 {
        use std::sync::atomic::Ordering;
        self.instances
            .iter()
            .filter(|e| e.value().gpu != "cpu")
            .map(|e| {
                let measured =
                    pallama_core::Hardware::bytes(e.value().settled_mib.load(Ordering::Relaxed));
                let weights = u64::try_from(e.value().model.bytes.max(0)).unwrap_or(u64::MAX);
                measured.max(weights)
            })
            .sum()
    }

    /// Total-VRAM planning budget for the bytes admission. Planning, not
    /// guarding: the fresh free-VRAM probe (spawn guard J3 / preload
    /// preflight) still owns refuse-vs-warn at spawn time.
    #[must_use]
    pub fn vram_budget_bytes(&self) -> u64 {
        pallama_core::Hardware::bytes(self.hardware.total_vram_mib())
    }

    /// Capacity-eviction victim: the coldest evictable instance — zero
    /// in-flight, not the incoming key, and not pinned (overlay
    /// `pin = true`, A13). Ordering: session-pinned models LAST (R3 —
    /// demoted, never excluded, so pressure can still land on them when
    /// nothing else is free), then prefix heat, then longest idle.
    /// `None` = nothing evictable (caller surfaces `AllSlotsBusy`).
    fn victim_key(&self, key: &str) -> Option<String> {
        let ttl = self.session_ttl();
        self.instances
            .iter()
            .filter(|e| {
                e.in_flight.load(Ordering::SeqCst) <= 0
                    && e.key() != key
                    && !self
                        .config
                        .overlay_for(model_of_key(e.key()))
                        .pin
                        .unwrap_or(false)
            })
            .min_by_key(|e| {
                (
                    self.sessions.pins(model_of_key(e.key()), ttl).live,
                    self.heat_of(e.key()),
                    *e.last_used.read().expect("idle lock"),
                )
            })
            .map(|e| e.key().clone())
    }

    /// Session-pin window (R3); zero = feature off.
    fn session_ttl(&self) -> Duration {
        Duration::from_secs(self.config.session_keep_secs)
    }

    /// Co-residency check (A15): would the candidate (weights + f16 KV)
    /// join the already-resident models within 95% of VRAM? Only counts
    /// GPU-resident instances (cpu/"partial" splits live elsewhere).
    fn coresidency_needs_kv_quant(
        &self,
        target: Option<&str>,
        candidate_bytes: u64,
        candidate_kv: Option<u64>,
    ) -> bool {
        if !self.hardware.has_gpu() {
            return false;
        }
        let Some(candidate_kv) = candidate_kv else {
            return false; // no geometry: never guess (H7)
        };
        // Card-scoped: LC2 picks ONE card per spawn, so pressure must be
        // computed against that card only — a summed-all-GPUs denominator
        // fires late (or never) on mixed iGPU+dGPU boxes.
        let residents: Vec<(Option<String>, u64, u64, bool)> = self
            .instances
            .iter()
            .map(|inst| {
                let weights = u64::try_from(inst.model.bytes.max(0)).unwrap_or(u64::MAX);
                (
                    inst.device.clone(),
                    weights,
                    inst.kv_est_bytes.unwrap_or(0),
                    !matches!(inst.gpu.as_str(), "cpu" | "partial"),
                )
            })
            .collect();
        card_pressure_exceeds(
            target,
            &self.hardware.gpus,
            &residents,
            candidate_bytes,
            candidate_kv,
        )
        .unwrap_or(false)
    }

    /// Live (non-evicted) TCP children as [`EngineRef`]s (endpoint +
    /// child auth) — the cache-hint poller's and metrics merger's fetch
    /// list. UDS children are skipped (no HTTP lane).
    #[must_use]
    pub fn live_http_endpoints(&self) -> Vec<EngineRef> {
        self.instances
            .iter()
            .filter(|i| matches!(i.endpoint, Endpoint::Tcp { .. }))
            .map(|i| self.engine_ref(i.value()))
            .collect()
    }

    /// Ensure a model is loaded and ready; returns its endpoint. Waits on
    /// a concurrent load instead of double-spawning.
    pub async fn ensure(&self, name: &str) -> Result<EngineRef, SupervisionError> {
        self.ensure_routed(name, None).await
    }

    /// `ensure` with prompt-prefix routing (B1): chat handlers hash
    /// (system + first user turn); identical conversations stick to the
    /// same warm-cache replica instead of round-robining cold children.
    /// `prefix = None` = no affinity, plain load-balance.
    pub async fn ensure_routed(
        &self,
        name: &str,
        prefix: Option<PrefixKey>,
    ) -> Result<EngineRef, SupervisionError> {
        // ollama API clients send `model:tag`; pallama rows are flat.
        // Same rule as the CLI boundary (`Store::resolve_model_name`),
        // colon-gated so the per-request hot path pays nothing for
        // canonical names: the store opens only when a ':' is present.
        let resolved;
        let name = if name.contains(':') {
            let store =
                Store::open(&self.dirs).map_err(|e| SupervisionError::Internal(anyhow!("{e}")))?;
            resolved = store.resolve_model_name(name);
            resolved.as_str()
        } else {
            name
        };
        // Router mode: every model name resolves to the ONE router child
        // (upstream autoloads the model on request, LRU-evicts at
        // models-max). Unknown names still fail fast against the store.
        if self.config.router && name != ROUTER_KEY {
            let store =
                Store::open(&self.dirs).map_err(|e| SupervisionError::Internal(anyhow!("{e}")))?;
            store
                .get_model(name)
                .map_err(|e| SupervisionError::Internal(anyhow!("{e}")))?;
            return Box::pin(self.ensure_routed(ROUTER_KEY, None)).await;
        }

        let key = self.replica_key(name, prefix);
        let result = self.ensure_key(&key).await;
        // Best-effort affinity record: pin this prefix to the replica
        // that served it, so the next turn hits its warm cache.
        if let Some(pk) = prefix {
            if result.is_ok() {
                if !self.prefix_affinity.contains_key(&pk.convo) {
                    if self.prefix_affinity.len() >= PREFIX_AFFINITY_CAP {
                        if let Some(oldest) = self.prefix_affinity.iter().next().map(|e| *e.key()) {
                            self.prefix_affinity.remove(&oldest);
                        }
                    }
                    self.prefix_affinity.insert(pk.convo, key.clone());
                }
                // F8: remember the sys class this replica has warm.
                let mut ring = self.sys_rings.entry(key).or_default();
                if ring.len() >= SYS_RING_CAP {
                    ring.pop_front();
                }
                if !ring.contains(&pk.sys) {
                    ring.push_back(pk.sys);
                }
            }
        }
        // LC1 predictive pre-loading: record the A→B edge on every
        // successful serve (router pseudo-model excluded — it has no
        // model identity of its own).
        if name != ROUTER_KEY && result.is_ok() {
            self.note_transition(name);
        }
        result
    }

    /// `ensure` for requests that carry images — the mmproj lazy-attach
    /// path. Policy (same resolution as the profile compiler's rule 19):
    /// `attach` → plain `ensure_routed` (projector already on the
    /// spawn); `skip` → loud teaching error (vision is OFF by config,
    /// never a silent 404); `lazy` → serve any live projector sibling
    /// of the same base model, else spawn the projector variant under a
    /// `@vision` key: projector-less live siblings of the same base are
    /// evicted first (the projector costs ~1.1 GiB measured VRAM —
    /// keeping both children of one model is double-loading), then the
    /// KV bank carries the conversation across the respawn.
    pub async fn ensure_vision(
        &self,
        name: &str,
        prefix: Option<PrefixKey>,
    ) -> Result<EngineRef, SupervisionError> {
        if self.config.router && name != ROUTER_KEY {
            // router child is forced-attach (preset loop) — plain path
            return Box::pin(self.ensure_routed(name, prefix)).await;
        }
        let policy =
            profile::mmproj_policy_effective(&self.config, name, &self.config.overlay_for(name));
        match policy {
            pallama_core::config::MmprojPolicy::Attach => {
                Box::pin(self.ensure_routed(name, prefix)).await
            }
            pallama_core::config::MmprojPolicy::Skip => Err(SupervisionError::Internal(anyhow!(
                "vision disabled: mmproj = skip spawns {name} text-only — set \
                 mmproj = true (or \"lazy\") to serve images"
            ))),
            pallama_core::config::MmprojPolicy::Lazy => {
                // (a) a live projector sibling of this base model serves
                // immediately (sticky @vision replica from an earlier
                // vision request)
                let mut serve: Option<Arc<Instance>> = None;
                let mut text_sibs: Vec<String> = Vec::new();
                for entry in &self.instances {
                    if model_of_key(entry.key()) != name {
                        continue;
                    }
                    let inst = entry.value();
                    let state = inst.state.read().unwrap();
                    if matches!(*state, InstanceState::Ready | InstanceState::Sleeping) {
                        if inst.projector {
                            serve = Some(Arc::clone(inst));
                        } else {
                            text_sibs.push(entry.key().clone());
                        }
                    }
                }
                if let Some(inst) = serve {
                    return Ok(self.engine_ref(&inst));
                }
                // (b) respawn WITH projector: drop the text-only sibling
                // first (VRAM honesty — never two children of one model)
                for k in &text_sibs {
                    tracing::warn!(
                        target: "pallama::supervisor",
                        "mmproj lazy: vision request for {name} — respawning \
                         text-only instance {k} with the projector (KV bank \
                         carries the conversation)"
                    );
                    if let Err(e) = self.evict(k).await {
                        tracing::warn!(
                            target: "pallama::supervisor",
                            error = %e,
                            "pre-vision evict failed for {k}"
                        );
                    }
                }
                let vision_key = format!("{}@vision", self.replica_key(name, prefix));
                Box::pin(self.ensure_key(&vision_key)).await
            }
        }
    }

    /// LC1: bump the (prev, current) transition count. Bounded table;
    /// overflow arbitrarily evicts one entry (a dropped edge just means
    /// a missed preload opportunity, never a wrong spawn).
    fn note_transition(&self, name: &str) {
        let prev = {
            let mut last = self.last_requested.lock().unwrap();
            let prev = last.clone();
            *last = Some(name.to_string());
            prev
        };
        let Some(prev) = prev else { return };
        if prev == name {
            return;
        }
        let edge = (prev, name.to_string());
        if !self.transitions.contains_key(&edge) && self.transitions.len() >= TRANSITIONS_CAP {
            if let Some(k) = self.transitions.iter().next().map(|e| e.key().clone()) {
                self.transitions.remove(&k);
            }
        }
        *self.transitions.entry(edge).or_insert(0) += 1;
    }

    /// Pick the instance key for a request: `"model"` when
    /// `replicas <= 1` (byte-identical legacy), else `"model#N"`.
    /// Order: prefix affinity → new-conversation grows a fresh replica
    /// (own KV cache) → anonymous traffic reuses an idle live replica,
    /// else grows while all live are busy → least-loaded existing anyway
    /// (join its queue).
    fn replica_key(&self, name: &str, prefix: Option<PrefixKey>) -> String {
        if name == ROUTER_KEY {
            return name.to_string();
        }
        let replicas = self
            .config
            .overlay_for(name)
            .replicas
            .unwrap_or(1)
            .clamp(1, MAX_REPLICAS);
        if replicas <= 1 {
            return name.to_string();
        }
        // (a) Sticky prefix: same conversation → same warm cache.
        if let Some(pk) = prefix {
            if let Some(hit) = self.prefix_affinity.get(&pk.convo) {
                let key = hit.value().clone();
                drop(hit);
                if let Some(inst) = self.instances.get(&key) {
                    let state = *inst.state.read().expect("state lock");
                    if matches!(state, InstanceState::Ready | InstanceState::Sleeping) {
                        return key;
                    }
                }
            }
        }
        // (b) Scan live replicas: least-loaded + highest replica index.
        // F8: while scanning, also score replicas that recently served the
        // same `sys` prefix class — a new conversation sharing the system
        // prompt coalesces onto the replica that already holds that KV.
        let mut best: Option<(String, i64)> = None;
        let mut scored: Option<(String, i64)> = None;
        let mut max_idx: u32 = 0;
        for e in &self.instances {
            let Some((model, idx)) = split_replica(e.key()) else {
                continue;
            };
            if model != name {
                continue;
            }
            max_idx = max_idx.max(idx);
            let state = *e.value().state.read().expect("state lock");
            if !matches!(state, InstanceState::Ready | InstanceState::Sleeping) {
                continue;
            }
            let load = e.value().in_flight.load(Ordering::SeqCst);
            if best.as_ref().is_none_or(|b| load < b.1) {
                best = Some((e.key().clone(), load));
            }
            if let Some(pk) = prefix {
                if self
                    .sys_rings
                    .get(e.key())
                    .is_some_and(|ring| ring.contains(&pk.sys))
                    && scored.as_ref().is_none_or(|sc| load < sc.1)
                {
                    scored = Some((e.key().clone(), load));
                }
            }
        }
        // (a2) Cache-aware coalesce: same system-prompt class, new
        // conversation → the replica holding that prefix absorbs it
        // instead of growing a cold child.
        if let Some((key, _)) = scored {
            return key;
        }
        match (best, prefix.is_some()) {
            // A NEW conversation (prefix present, unpinned): give it its
            // own cold cache rather than polluting another replica's —
            // grow first, fall back to least-loaded at capacity.
            (Some(best), true) => {
                if max_idx < replicas {
                    return format!("{name}#{}", max_idx + 1);
                }
                return best.0;
            }
            // Anonymous traffic: an IDLE live replica absorbs it —
            // sequential single-user traffic never spawns a second child.
            (Some((key, load)), false) if load <= 0 => return key,
            // Anonymous + every live replica busy: scale out first so
            // distinct requests get their own KV cache instead of
            // queueing behind a warm one.
            (Some((key, _)), false) => {
                if max_idx < replicas {
                    return format!("{name}#{}", max_idx + 1);
                }
                return key; // at capacity: join the least-loaded queue
            }
            // (c) No live replica: scale out until `replicas` children.
            (None, _) => {
                if max_idx < replicas {
                    return format!("{name}#{}", max_idx + 1);
                }
            }
        }
        // (d) All slots exist but none ready (loading/error): join the
        // least-loaded one's queue rather than fail.
        let mut least: Option<(String, i64)> = None;
        for e in &self.instances {
            let Some((model, _)) = split_replica(e.key()) else {
                continue;
            };
            if model != name {
                continue;
            }
            let load = e.value().in_flight.load(Ordering::SeqCst);
            if least.as_ref().is_none_or(|b| load < b.1) {
                least = Some((e.key().clone(), load));
            }
        }
        least.map_or_else(|| format!("{name}#1"), |(key, _)| key)
    }

    /// Load/wait core, keyed by INSTANCE key (model or model#N).
    async fn ensure_key(&self, key: &str) -> Result<EngineRef, SupervisionError> {
        // Fast path: running (or sleeping — the child wakes on traffic).
        if let Some(inst) = self.instances.get(key) {
            let snapshot = (
                *inst.state.read().expect("state lock"),
                inst.last_used.read().expect("idle lock").elapsed(),
            );
            if matches!(snapshot.0, InstanceState::Ready | InstanceState::Sleeping) {
                let was_sleeping = snapshot.0 == InstanceState::Sleeping;
                let live = inst.clone();
                drop(inst);
                *live.last_used.write().expect("idle lock") = Instant::now();
                if was_sleeping {
                    *live.state.write().expect("state lock") = InstanceState::Ready;
                    self.bus.publish(PallamaEvent::InstanceStateChanged {
                        name: key.to_string(),
                        state: InstanceState::Ready,
                    });
                }
                return Ok(self.engine_ref(&live));
            }
        }

        // Circuit breaker.
        if let Some(ts) = self.restarts.get(key) {
            let recent = ts
                .iter()
                .filter(|t| t.elapsed() < self.circuit_window)
                .count();
            if recent > self.max_restarts {
                return Err(SupervisionError::CircuitOpen(key.to_string()));
            }
        }

        // Join an in-flight load.
        if let Some(existing) = self.loading.get(key).map(|l| l.clone()) {
            drop(existing);
            let notify = self.loading.get(key).map(|l| l.clone());
            if let Some(n) = notify {
                let notified = n.notified();
                // Re-check results after registering interest.
                if let Some(res) = self.load_results.get(key) {
                    return clone_load_result(res.value(), key);
                }
                notified.await;
                if let Some(res) = self.load_results.get(key) {
                    return clone_load_result(res.value(), key);
                }
                // Loader finished between checks.
                if let Some(inst) = self.instances.get(key) {
                    return Ok(self.engine_ref(inst.value()));
                }
                return Err(SupervisionError::Internal(anyhow!(
                    "load of {key} vanished without result"
                )));
            }
        }

        // We are the loader.
        let notify = Arc::new(Notify::new());
        self.loading.insert(key.to_string(), notify.clone());
        // Cancellation safety: if THIS loader future is dropped before the
        // completion protocol below runs (the caller went away), parked
        // waiters would sleep forever on a Notify that never fires —
        // live-repro'd wedge (client timeout during cold spawn wedged every
        // later generate). The gateway detaches its loads, so this guard is
        // defense in depth for any other caller that can be dropped.
        let mut abort = LoadAbort {
            sup: self,
            key,
            notify: notify.clone(),
            armed: true,
        };
        let result = if self.config.router && key == ROUTER_KEY {
            self.spawn_router_instance().await
        } else {
            self.spawn_instance(key).await
        };
        // Waiters get a cloneable copy; the loader returns the typed error
        // itself (callers match on ModelLoadTimeout / CircuitOpen / ...).
        self.load_results
            .insert(key.to_string(), map_result(&result));
        notify.notify_waiters();
        // Give waiters a moment to drain, then clear loading state.
        tokio::time::sleep(Duration::from_millis(50)).await;
        self.loading.remove(key);
        self.load_results.remove(key);
        abort.armed = false;
        result
    }
}

/// Loader-protocol cancellation guard for [`Supervisor::ensure_key`]:
/// a loader dropped mid-spawn must wake parked waiters with a loud
/// error and clear the loading slot, never strand the Notify.
/// One-shot advisory: an NVIDIA box running a Vulkan llama.cpp engine
/// with NO CUDA engine installed is leaving ~18% first-token latency on
/// the table (live-measured, BENCHMARK.md 2026-09-11). Upstream ships no
/// Linux CUDA prebuilts, so the auto-overlay cannot fix this until the
/// overlay repo publishes — the in-tree build can, today. Once per
/// process; the dethrone guard in `register_engine_with_vendor` keeps an
/// installed CUDA engine from being silently replaced later.
fn advise_cuda_build(store: &Store, active: &pallama_core::store::EngineRow) {
    static ADVISED: AtomicBool = AtomicBool::new(false);
    if ADVISED.load(Ordering::Relaxed) {
        return;
    }
    if crate::engine::system_vendor_hint() != crate::engine::manifest::Vendor::Nvidia {
        return;
    }
    if active.kind != pallama_core::engine_kind::EngineKind::LlamaCpp
        || crate::engine::is_cuda_engine(&active.tag, &active.asset)
    {
        return;
    }
    let has_cuda = store.list_engines().is_ok_and(|rows| {
        rows.iter().any(|e| {
            e.kind == pallama_core::engine_kind::EngineKind::LlamaCpp
                && crate::engine::is_cuda_engine(&e.tag, &e.asset)
        })
    });
    if has_cuda {
        return; // CUDA installed but not active: a deliberate `engine use`
    }
    ADVISED.store(true, Ordering::Relaxed);
    tracing::warn!(
        "NVIDIA GPU with a Vulkan-only llama.cpp engine active — CUDA is ~18% faster on \
         first token here; build it once with: pallama engine build cuda"
    );
}

struct LoadAbort<'a> {
    sup: &'a Supervisor,
    key: &'a str,
    notify: Arc<Notify>,
    armed: bool,
}

impl Drop for LoadAbort<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        self.sup.load_results.insert(
            self.key.to_string(),
            Err("load cancelled before completion (loader dropped mid-spawn)".to_string()),
        );
        self.notify.notify_waiters();
        self.sup.loading.remove(self.key);
        self.sup.load_results.remove(self.key);
    }
}

impl Supervisor {
    #[allow(clippy::unused_self)] // symmetrical with future instance methods
    fn engine_ref(&self, inst: &Instance) -> EngineRef {
        EngineRef {
            name: inst.name.clone(),
            endpoint: inst.endpoint.clone(),
            auth: inst.auth.clone(),
        }
    }

    /// Mint a per-child auth secret (child `--api-key` hardening).
    /// `--api-key-file` is preferred when the engine has it: the secret
    /// then lives in a 0600 file, not on the child's `/proc` cmdline.
    /// The argv fallback (`--api-key <secret>`) still closes the open
    /// child, just less privately. Engines with neither flag warn-skip
    /// (same contract as every other manifest-gated emission) instead
    /// of breaking the spawn.
    fn mint_child_auth(
        &self,
        key: &str,
        endpoint: &Endpoint,
        manifest: &crate::engine::manifest::Manifest,
    ) -> Result<Option<ChildAuth>, SupervisionError> {
        let enabled = match self.config.child_auth {
            Some(v) => v,
            // Auto: TCP children are reachable by any local process;
            // UDS sockets already enforce filesystem permissions.
            None => matches!(endpoint, Endpoint::Tcp { .. }),
        };
        if !enabled {
            return Ok(None);
        }
        if !manifest.flags.contains("--api-key-file") && !manifest.flags.contains("--api-key") {
            tracing::warn!(
                model = key,
                "child_auth: engine {} lacks --api-key/--api-key-file; child stays \
                 unauthenticated (run: pallama engine update)",
                manifest.tag
            );
            return Ok(None);
        }
        // Entropy failure is a hard error, never a silently weaker key.
        let mut raw = [0u8; 24];
        getrandom::fill(&mut raw)
            .map_err(|e| SupervisionError::Internal(anyhow!("child_auth entropy: {e}")))?;
        let mut secret = String::with_capacity(4 + raw.len() * 2);
        secret.push_str("plm_");
        for b in &raw {
            use std::fmt::Write as _;
            let _ = write!(secret, "{b:02x}");
        }
        if manifest.flags.contains("--api-key-file") {
            let path = self.dirs.run_dir().join(format!("{key}.apikey"));
            std::fs::write(&path, &secret).map_err(|e| {
                SupervisionError::Internal(anyhow!("write {}: {e}", path.display()))
            })?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                if let Err(e) =
                    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
                {
                    tracing::warn!(target: "pallama::supervisor", error = %e, "apikey chmod 0600 failed");
                }
            }
            Ok(Some(ChildAuth {
                argv: vec!["--api-key-file".into(), path.display().to_string()],
                secret,
                keyfile: Some(path),
            }))
        } else {
            tracing::warn!(
                model = key,
                "child_auth: engine lacks --api-key-file; falling back to --api-key \
                 (secret visible in /proc/<pid>/cmdline)"
            );
            Ok(Some(ChildAuth {
                argv: vec!["--api-key".into(), secret.clone()],
                secret,
                keyfile: None,
            }))
        }
    }

    /// Router-mode spawn: ONE child, no `-m`, `--models-preset` INI
    /// generated from every loadable store model. Models whose GGUF
    /// metadata or manifest-gated profile fail to compile are SKIPPED
    /// with a warning — one bad model must not take the router down.
    #[allow(clippy::too_many_lines)]
    async fn spawn_router_instance(&self) -> Result<EngineRef, SupervisionError> {
        let store =
            Store::open(&self.dirs).map_err(|e| SupervisionError::Internal(anyhow!("{e}")))?;
        let models = store
            .list_models()
            .map_err(|e| SupervisionError::Internal(anyhow!("{e}")))?;
        let manifest = self.engine.capabilities();
        if !manifest.flags.contains("--models-preset") {
            return Err(SupervisionError::Internal(anyhow!(
                "router mode: engine {} lacks --models-preset; run: pallama engine update",
                manifest.tag
            )));
        }
        let _ = self.dirs.ensure();
        let data_dir_str = self.dirs.data_dir.to_string_lossy().into_owned();

        let mut sections: Vec<(String, Vec<String>)> = Vec::new();
        let mut global: Vec<String> = Vec::new();
        for m in &models {
            let gguf = match pallama_core::read_metadata_file(std::path::Path::new(&m.path)) {
                Ok(g) => g,
                Err(e) => {
                    tracing::warn!(model = %m.name, "router preset skips model (gguf metadata): {e}");
                    continue;
                }
            };
            let overlay = self.config.overlay_for(&m.name);
            let loras: Vec<(String, f64)> = store
                .list_loras(Some(&m.name))
                .map_err(|e| SupervisionError::Internal(anyhow!("{e}")))?
                .into_iter()
                .map(|l| (l.path, l.scale))
                .collect();
            // Same shared draft resolver as `ensure`: a spec=auto router
            // must not silently drop catalog-paired models with a false
            // "not pulled" error.
            let spec_mode = overlay
                .spec
                .clone()
                .unwrap_or_else(|| self.config.spec.clone());
            let draft_path = resolve_draft_path(&store, &m.name, &spec_mode);
            let input = ProfileInput {
                engine_kind: self.engine.kind(),
                sibling_devices: Vec::new(),
                auto_tensor_split: None,
                model_name: &m.name,
                instance_key: &m.name,
                model_path: &m.path,
                model_bytes: u64::try_from(m.bytes.max(0)).unwrap_or(u64::MAX),
                gguf: &gguf,
                hardware: &self.hardware,
                config: &self.config,
                overlay: &overlay,
                loras: &loras,
                draft_path: draft_path.as_deref(),
                draft_gguf: None, // router preset: one child, no co-residency planning
                mmproj_path: m.mmproj_path.as_deref(),
                // router preset: every pulled model rides one child incl
                // VL rows — force the projector on regardless of policy
                mmproj_force: true,
                engine_tag: &manifest.tag,
                supported_flags: &manifest.flags,
                spec_types: &manifest.spec_types,
                endpoint: Endpoint::Tcp {
                    host: "127.0.0.1".into(),
                    port: 0,
                },
                data_dir: &data_dir_str,
                cache_hit_rate: self.cache_hint.get(),
                device_hint: None, // router preset: no per-GPU scoping
                engine_census: self.hardware.gpus.clone(),
            };
            match profile::compile(&input, &pallama_core::TuningOverrides::default()) {
                Ok(p) => {
                    if global.is_empty() {
                        global.clone_from(&p.argv);
                    }
                    sections.push((m.name.clone(), p.argv));
                }
                Err(e) => {
                    tracing::warn!(model = %m.name, "router preset skips model (profile): {e}");
                }
            }
        }
        if sections.is_empty() {
            return Err(SupervisionError::Internal(anyhow!(
                "router mode: no loadable models (metadata or manifest failures for all pulled models)"
            )));
        }
        let ini = pallama_core::profile::generate_router_preset(&sections, &global);
        let ini_path = self.dirs.run_dir().join("router-preset.ini");
        std::fs::write(&ini_path, &ini).map_err(|e| {
            SupervisionError::Internal(anyhow!("write {}: {e}", ini_path.display()))
        })?;
        tracing::info!(
            ini = %ini_path.display(),
            models = sections.len(),
            "router preset generated"
        );

        let synthetic_model = pallama_core::ModelRow {
            name: ROUTER_KEY.to_string(),
            repo: String::new(),
            quant: String::new(),
            path: ini_path.display().to_string(),
            bytes: 0,
            sha256: None,
            mmproj_path: None,
            shards: 1,
            arch: None,
            params: None,
            ctx_train: None,
            pulled_at: 0,
        };

        let mut auth_keyfile: Option<std::path::PathBuf> = None;
        let mut child_died_during_load = false;
        // Dying router child's log tail (see spawn_instance for rationale).
        let mut last_load_tail = String::new();
        for _attempt in 0..2 {
            let endpoint = self.pick_endpoint(ROUTER_KEY);
            // Child auth dies with the child: minted per attempt (same
            // keyfile path, so retries overwrite), removed when this
            // attempt or the whole spawn fails.
            let auth = match self.mint_child_auth(ROUTER_KEY, &endpoint, manifest) {
                Ok(a) => a,
                Err(e) => {
                    if let Some(p) = &auth_keyfile {
                        let _ = std::fs::remove_file(p);
                    }
                    return Err(e);
                }
            };
            if let Some(p) = auth.as_ref().and_then(|a| a.keyfile.clone()) {
                auth_keyfile = Some(p);
            }
            let mut argv: Vec<String> = Vec::new();
            match &endpoint {
                Endpoint::Tcp { host, port } => {
                    argv.extend([
                        "--host".into(),
                        host.clone(),
                        "--port".into(),
                        port.to_string(),
                    ]);
                }
                Endpoint::Unix { socket } => {
                    argv.extend(["--host".into(), socket.clone()]);
                }
            }
            argv.extend(["--models-preset".into(), ini_path.display().to_string()]);
            // NOTE: no front-level --slot-save-path — a CLI value cascades
            // to model children and would OVERRIDE their per-model INI
            // paths (verified live). Sessions stay per-model via the INI.
            if self.config.router_max_models > 0 {
                argv.extend([
                    "--models-max".into(),
                    self.config.router_max_models.to_string(),
                ]);
            }
            // Router LRU autoload: None keeps the upstream default (off).
            match self.config.models_autoload {
                Some(true) if manifest.flags.contains("--models-autoload") => {
                    argv.push("--models-autoload".into());
                }
                Some(true) => {
                    tracing::warn!(
                        "models_autoload set but engine {} lacks --models-autoload; run: pallama engine update",
                        manifest.tag
                    );
                }
                Some(false) if manifest.flags.contains("--no-models-autoload") => {
                    argv.push("--no-models-autoload".into());
                }
                Some(false) | None => {}
            }
            if let Some(a) = &auth {
                argv.extend(a.argv.iter().cloned());
            }
            self.remap_device_argv(&mut argv, ROUTER_KEY).await;
            let mut child = self.engine.spawn(&argv, &endpoint).await.map_err(|e| {
                if let Some(p) = &auth_keyfile {
                    let _ = std::fs::remove_file(p);
                }
                SupervisionError::Internal(anyhow!("{e}"))
            })?;
            if let Ok(Some(status)) = child.try_status() {
                tracing::warn!(
                    router = ROUTER_KEY,
                    "router died at spawn ({status}); retrying"
                );
                let _ = child.kill().await;
                let _ = child.reap().await;
                continue;
            }
            let (health, child_died) = self.wait_healthy(&endpoint, &mut child).await;
            match health {
                Ok(()) => {
                    let pid = child.id().ok_or_else(|| {
                        SupervisionError::Internal(anyhow!(
                            "router child has no pid; refusing to track instance"
                        ))
                    })?;
                    if pid <= 1 {
                        if let Some(p) = &auth_keyfile {
                            let _ = std::fs::remove_file(p);
                        }
                        return Err(SupervisionError::Internal(anyhow!(
                            "router child pid {pid} is not a safe process-group id"
                        )));
                    }
                    let inst = Arc::new(Instance {
                        name: ROUTER_KEY.to_string(),
                        endpoint,
                        state: std::sync::RwLock::new(InstanceState::Ready),
                        last_used: std::sync::RwLock::new(Instant::now()),
                        in_flight: AtomicI64::new(0),
                        keep_until: std::sync::RwLock::new(None),
                        started_at: Instant::now(),
                        // router serves EVERY pulled model incl VL rows —
                        // the preset compile forces attach, so reflect it
                        projector: argv.windows(2).any(|w| w[0] == "-mm" || w[0] == "--mmproj"),
                        argv,
                        model: synthetic_model.clone(),
                        profile_ctx: 0,
                        gpu: "router".to_string(),
                        device: None,
                        kv_est_bytes: None,
                        warnings: Vec::new(),
                        settled_mib: std::sync::atomic::AtomicU64::new(0),
                        auth: auth.as_ref().map(|a| a.secret.clone()),
                        child: tokio::sync::Mutex::new(child),
                        pid,
                    });
                    let _ = std::fs::write(
                        self.dirs.run_dir().join(format!("{ROUTER_KEY}.pid")),
                        pid.to_string(),
                    );
                    self.instances.insert(ROUTER_KEY.to_string(), inst);
                    if self.unclean_dead.remove(ROUTER_KEY).is_some() {
                        self.record_restart(ROUTER_KEY);
                    }
                    self.bus.publish(PallamaEvent::InstanceStateChanged {
                        name: ROUTER_KEY.to_string(),
                        state: InstanceState::Ready,
                    });
                    let inst = self.instances.get(ROUTER_KEY).expect("just inserted");
                    return Ok(self.engine_ref(inst.value()));
                }
                Err(e) => {
                    child_died_during_load |= child_died;
                    if child_died {
                        tracing::error!(router = ROUTER_KEY, "router died during load: {e:#}");
                    } else {
                        tracing::warn!(router = ROUTER_KEY, "router health check failed: {e:#}");
                    }
                    last_load_tail = child.tail_joined();
                    let _ = child.kill().await;
                    let _ = child.reap().await;
                }
            }
        }
        if let Some(p) = &auth_keyfile {
            let _ = std::fs::remove_file(p);
        }
        Err(if child_died_during_load {
            SupervisionError::EngineCrashed(format!(
                "{ROUTER_KEY}: {}",
                Self::tail_excerpt(&last_load_tail)
            ))
        } else {
            SupervisionError::ModelLoadTimeout(ROUTER_KEY.to_string())
        })
    }

    /// Race the health poll against child liveness: a child that dies
    /// mid-load must fail the spawn IMMEDIATELY, carrying its last output
    /// lines — never keep polling a corpse for the full `model_load_timeout`
    /// (mislabeling an instant crash as a slow model load).
    /// Returns the health result plus `true` when the failure is child death.
    async fn wait_healthy(
        &self,
        endpoint: &Endpoint,
        child: &mut ChildHandle,
    ) -> (anyhow::Result<()>, bool) {
        let mut health = Box::pin(self.engine.health_check(endpoint, self.load_timeout));
        let mut liveness = Box::pin(async {
            loop {
                if let Ok(Some(status)) = child.try_status() {
                    return status;
                }
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
            }
        });
        let outcome = tokio::select! {
            res = &mut health => return (res, false),
            status = &mut liveness => status,
        };
        // Release the liveness coroutine's unique child borrow before
        // reading the tail.
        drop(health);
        drop(liveness);
        let tail = child.tail_joined();
        (
            Err(anyhow::anyhow!(
                "engine exited during load ({outcome}); last output: {tail}"
            )),
            true,
        )
    }

    /// Map `--device` names in argv from manifest-era (install-context)
    /// names to the names the SERVING child context enumerates. The two
    /// enumerations can differ (install ran from a shell; serving spawns
    /// from the daemon) and the child is the authority — a name it does
    /// not accept is fatal to it. Name still enumerated: keep. Same
    /// device re-enumerated under a different name: rewrite (identity =
    /// vendor/description overlap + `total_mib` proximity). No live
    /// counterpart: drop the pair (llama.cpp auto-picks) and warn.
    fn map_devices(
        argv: &[String],
        frozen: &[crate::engine::manifest::DeviceDesc],
        live: &[crate::engine::manifest::DeviceDesc],
    ) -> (Vec<String>, Vec<String>) {
        use crate::engine::manifest::DeviceDesc;
        fn same_device(a: &DeviceDesc, b: &DeviceDesc) -> bool {
            use crate::engine::manifest::Vendor;
            let mem_close = a.total_mib.abs_diff(b.total_mib) <= 64;
            let vendor_match = a.vendor() == b.vendor() && a.vendor() != Vendor::Other;
            let desc_overlap = !a.description.is_empty()
                && (a.description.contains(&b.description)
                    || b.description.contains(&a.description));
            (vendor_match || desc_overlap) && mem_close
        }
        let mut out = Vec::with_capacity(argv.len());
        let mut warnings = Vec::new();
        let mut i = 0;
        while i < argv.len() {
            if argv[i] == "--device" {
                if let Some(name) = argv.get(i + 1) {
                    if live.iter().any(|d| &d.name == name) {
                        out.push(argv[i].clone());
                        out.push(name.clone());
                    } else {
                        let mapped = frozen
                            .iter()
                            .find(|d| &d.name == name)
                            .and_then(|old| live.iter().find(|l| same_device(old, l)));
                        if let Some(l) = mapped {
                            out.push(argv[i].clone());
                            out.push(l.name.clone());
                            warnings.push(format!(
                            "device {name} re-enumerated as {} in the serving child context; using it",
                            l.name
                        ));
                        } else {
                            let sees = live
                                .iter()
                                .map(|d| d.name.as_str())
                                .collect::<Vec<_>>()
                                .join(", ");
                            warnings.push(format!(
                            "device {name} not visible to the serving child (sees: {sees}); dropping --device, llama.cpp will auto-pick"
                        ));
                        }
                    }
                    i += 2;
                    continue;
                }
            }
            out.push(argv[i].clone());
            i += 1;
        }
        (out, warnings)
    }

    /// Validate/rewrite `--device` entries in argv against the serving
    /// child's own enumeration. No-ops when the argv has no devices or
    /// the engine cannot enumerate (old builds, listing failure).
    async fn remap_device_argv(&self, argv: &mut Vec<String>, who: &str) {
        if !argv.iter().any(|a| a == "--device") {
            return;
        }
        let Ok(Some(live)) = self.engine.enumerate_devices().await else {
            return;
        };
        let (mapped, warnings) =
            Self::map_devices(argv, &self.engine.capabilities().devices, &live);
        for w in &warnings {
            tracing::warn!(ctx = who, "device-map: {w}");
        }
        *argv = mapped;
    }

    /// One cohesive spawn path (model load → profile → capacity → spawn →
    /// health → track); splitting it would thread six one-use locals.
    #[allow(clippy::too_many_lines)]
    /// Spawn by INSTANCE key (`model` or `model#N`, B1): everything
    /// user-facing resolves through `model_of_key`, everything
    /// instance-scoped (pidfile, socket, sessions dir, maps, events)
    /// uses the full key so replicas never collide.
    async fn spawn_instance(&self, key: &str) -> Result<EngineRef, SupervisionError> {
        let name = model_of_key(key);
        let store =
            Store::open(&self.dirs).map_err(|e| SupervisionError::Internal(anyhow!("{e}")))?;
        let model = store
            .get_model(name)
            .map_err(|e| SupervisionError::Internal(anyhow!("{e}")))?
            .ok_or_else(|| SupervisionError::ModelNotFound(not_found_name(name)))?;
        let gguf = pallama_core::read_metadata_file(std::path::Path::new(&model.path))
            .map_err(|e| SupervisionError::Internal(anyhow!("gguf metadata: {e}")))?;
        let mut overlay = self.config.overlay_for(name);
        // LC4: in-memory adaptive slots fill in ONLY where the user left
        // slots unset (manual overlay always wins; `tune --slots` writes
        // the overlay, which then shadows any adoption).
        if overlay.slots.is_none() {
            if let Some(adopted) = self.adopted_slots.get(name) {
                overlay.slots = Some(*adopted);
            }
        }
        let loras: Vec<(String, f64)> = store
            .list_loras(Some(name))
            .map_err(|e| SupervisionError::Internal(anyhow!("{e}")))?
            .into_iter()
            .map(|l| (l.path, l.scale))
            .collect();
        // Draft resolution mirrors every other spawn path (router
        // preset, CLI bench/tune): one shared resolver, catalog pair →
        // pulled store row. Compile-time freshness + manifest gates
        // live in profile::compile.
        let spec_mode = overlay
            .spec
            .clone()
            .unwrap_or_else(|| self.config.spec.clone());
        let draft_path = resolve_draft_path(&store, name, &spec_mode);
        // Draft header for the planner's device-KV charge (A15): the
        // spec pair allocates its own KV at the compiled ctx. An
        // unreadable header degrades to dense-only with a warn — it
        // must never block the spawn itself.
        let draft_gguf =
            draft_path.as_deref().and_then(|p| {
                match pallama_core::read_metadata_file(std::path::Path::new(p)) {
                    Ok(g) => Some(g),
                    Err(e) => {
                        tracing::warn!(
                            model = name,
                            draft = p,
                            "draft header unreadable; co-residency planner charges dense-only: {e}"
                        );
                        None
                    }
                }
            });
        // Capacity: evict the COLDEST instance first — recency-weighted
        // prefix heat (hot models keep their warm cache across capacity
        // pressure; the radix-lite lane), ties broken by longest-idle.
        // Pinned models (overlay `pin = true`, A13) are never victims:
        // capacity pressure falls on unpinned instances instead, and
        // when everything live is pinned+busy the spawn fails loudly.
        // Gates: explicit `max_loaded_models` counts instances; auto on
        // GPU charges MEASURED resident footprints (settle report) plus
        // the incoming model's admission floor — weights + projector +
        // KV floor + spawn overhead, the same standing charges the
        // unified gpu-layers pin trusts. Weights-only planning
        // oversubscribed an 8 GiB card on 2026-09-11 (measured 7302 +
        // 977 actual vs 5854 weights-sum → NVRM NO_MEMORY crash-loop);
        // a cold box always admits one model, and the J3 spawn guard
        // owns the honest-refusal path for loads that cannot fit at all.
        let incoming_bytes = {
            let mmproj_bytes = model
                .mmproj_path
                .as_deref()
                .and_then(|p| std::fs::metadata(p).ok())
                .map_or(0, |m| m.len());
            pallama_core::profile::admission_floor_bytes(
                u64::try_from(model.bytes.max(0)).unwrap_or(u64::MAX),
                mmproj_bytes,
            )
        };
        loop {
            let count_blocked = self
                .instance_cap()
                .is_some_and(|cap| self.instances.len() >= cap);
            let bytes_blocked = self.bytes_admission_active()
                && !self.instances.is_empty()
                && self.resident_bytes().saturating_add(incoming_bytes) > self.vram_budget_bytes();
            if !count_blocked && !bytes_blocked {
                break;
            }
            match self.victim_key(key) {
                Some(v) => {
                    self.evict(&v)
                        .await
                        .map_err(|e| SupervisionError::Internal(anyhow!("{e}")))?;
                }
                None => return Err(SupervisionError::AllSlotsBusy),
            }
        }

        // Spawn-time memory guard (J3): below a hard floor the mmap
        // engine will thrash swap for minutes instead of loading — fail
        // fast with a named error. Floor = half the model + 512 MiB
        // (zero-false-positive: mmap streaming needs roughly that
        // resident even in the best case).
        if self.config.spawn_mem_guard {
            let avail = crate::probe::mem_available_mib();
            let model_mib = u64::try_from(model.bytes.max(0)).unwrap_or(u64::MAX) / (1024 * 1024);
            let need = (model_mib / 2) + 512;
            if avail < need {
                return Err(SupervisionError::Internal(anyhow!(
                    "insufficient memory to load {:?}: MemAvailable {avail} MiB < floor {need} MiB \
                     (model {} MiB). Stop co-resident engines (pallama ps / ollama stop) or free RAM; \
                     disable this guard with spawn_mem_guard = false",
                    name,
                    model.bytes / (1024 * 1024)
                )));
            }
        }

        let manifest = self.engine.capabilities();
        let model_bytes = u64::try_from(model.bytes.max(0)).unwrap_or(u64::MAX);
        // One fresh hardware probe serves both spawn-time decisions: the
        // J3 VRAM preflight and the LC2 auto GPU pick. The boot-time
        // snapshot can be stale (other engines came and went).
        let wants_pick = self.config.effective_devices(name).is_empty()
            && self.hardware.gpus.len() > 1
            && manifest.flags.contains("--device");
        let fresh = if self.config.spawn_mem_guard && self.hardware.has_gpu() || wants_pick {
            Some(self.live_hardware())
        } else {
            None
        };
        // Spawn-time VRAM preflight (J3): warn when even the weights
        // alone exceed free VRAM — the child will either spill layers to
        // CPU (slow) or die mid-load. Teaching warn only: partial offload
        // is legitimate, probe failure is skipped, never blocks a spawn.
        if let Some(f) = &fresh {
            if self.config.spawn_mem_guard && self.hardware.has_gpu() {
                let free_vram: u64 = f.gpus.iter().map(|g| g.free_mib).sum();
                let weights_mib = model_bytes / (1024 * 1024);
                if free_vram > 0 && weights_mib > free_vram * 95 / 100 {
                    tracing::warn!(
                        model = name,
                        weights_mib,
                        free_vram_mib = free_vram,
                        "model weights exceed free VRAM at spawn time — the engine will spill \
                         layers to CPU (slow) or fail mid-load; consider a smaller quant \
                         (pallama fit), freeing GPU memory (pallama ps), or kv quantization"
                    );
                }
            }
        }
        // Auto GPU bin-packing (LC2): with several cards and no manual
        // `devices` choice, place the child on the DISCRETE card with the
        // most free VRAM (integrated GPUs report shared-RAM "free" as if
        // it were dedicated — a bandwidth-starved trap; they serve only
        // when no discrete card exists) and scope every downstream VRAM
        // estimate (ngl ladder, KV, cache-ram, coresidency) to THAT card
        // instead of the summed pool — a child lands on one card, not the
        // sum. Manual overlay or global `devices` always wins; fail-open
        // (no pick on probe weirdness).
        let mut scoped_hw: Option<Hardware> = None;
        let mut picked_device: Option<String> = None;
        // Display twin of `picked_device` (census description, not the
        // backend id) for the ps card label.
        let mut picked_display: Option<String> = None;
        let mut sibling_devices: Vec<String> = Vec::new();
        let mut auto_split: Option<String> = None;
        // Tuning overrides are consumed exactly once, ABOVE the endpoint
        // retry loop (a retry attempt used to re-read an already-removed
        // `pending_ctx` entry), because the auto-split decision below
        // needs the candidate KV estimate before any card scoping.
        let mut tuning = self
            .pending_ctx
            .remove(name)
            .map(|(_, ctx)| pallama_core::TuningOverrides {
                ctx: Some(ctx),
                ..Default::default()
            })
            .unwrap_or_default();
        // Candidate f16 KV over the FULL (unscoped) hardware: the split
        // decision compares weights+KV against the combined discrete
        // pool before single-card scoping exists. Sibling literal of the
        // per-attempt input below (that one carries the real endpoint
        // and the scoping decided here); this probe only feeds the KV
        // estimator, endpoint is irrelevant to it.
        let data_dir_str = self.dirs.data_dir.to_string_lossy().into_owned();
        let candidate_kv_mib = {
            let probe = ProfileInput {
                engine_kind: self.engine.kind(),
                sibling_devices: Vec::new(),
                auto_tensor_split: None,
                model_name: name,
                instance_key: key,
                model_path: &model.path,
                model_bytes: u64::try_from(model.bytes.max(0)).unwrap_or(u64::MAX),
                gguf: &gguf,
                hardware: fresh.as_ref().unwrap_or(&self.hardware),
                config: &self.config,
                overlay: &overlay,
                loras: &loras,
                draft_path: draft_path.as_deref(),
                draft_gguf: draft_gguf.as_ref(),
                mmproj_path: model.mmproj_path.as_deref(),
                // KV-estimate probe: policy-neutral (mirror the spawn's
                // own key-derived force below for estimate honesty)
                mmproj_force: key.ends_with("@vision"),
                engine_tag: &manifest.tag,
                supported_flags: &manifest.flags,
                spec_types: &manifest.spec_types,
                endpoint: Endpoint::Tcp {
                    host: "127.0.0.1".into(),
                    port: 0,
                },
                data_dir: &data_dir_str,
                cache_hit_rate: self.cache_hint.get(),
                device_hint: None,
                engine_census: fresh.as_ref().unwrap_or(&self.hardware).gpus.clone(),
            };
            pallama_core::profile::estimate_kv_vram_charge(&probe, tuning.ctx)
                .map_or(0, |b| b / (1024 * 1024))
        };
        // Last-resort auto tensor-split (#28c): only when EVERY manual
        // pin is unset (tensor_split / devices / non-default main_gpu)
        // and the measured pool says weights+KV fit ONLY when spread.
        let split_pinned = !self.config.tensor_split.is_empty()
            || !self.config.effective_devices(name).is_empty()
            || self.config.main_gpu != pallama_core::Config::default().main_gpu;
        if !split_pinned && self.hardware.has_gpu() {
            let hw = fresh.as_ref().unwrap_or(&self.hardware);
            if let Some(ratios) =
                plan_auto_tensor_split(&hw.gpus, model_bytes / (1024 * 1024), candidate_kv_mib)
            {
                tracing::info!(
                    model = name,
                    ratios = %ratios,
                    "auto tensor-split: weights+KV exceed the best single card's free VRAM but fit the discrete cards combined (manual pins unset; split trades inter-card bandwidth for capacity)"
                );
                auto_split = Some(ratios);
            }
        }
        if wants_pick && auto_split.is_none() {
            let hw = fresh.as_ref().unwrap_or(&self.hardware);
            if let Some((idx, skipped_integrated)) = pick_gpu(&hw.gpus) {
                let best = &hw.gpus[idx];
                let mut scoped = hw.clone();
                scoped.gpus = vec![best.clone()];
                picked_device = Some(best.name.clone());
                picked_display = Some(best.display_name().to_string());
                scoped_hw = Some(scoped);
                // Spare discrete cards (manual picks only reserve what
                // `devices` names): the profile surfaces them as draft/
                // mmproj placement recommendations.
                sibling_devices = hw
                    .gpus
                    .iter()
                    .enumerate()
                    .filter(|(i, g)| *i != idx && !g.is_integrated())
                    .map(|(_, g)| g.name.clone())
                    .collect();
                tracing::info!(
                    model = name,
                    device = %best.name,
                    free_mib = best.free_mib,
                    "auto GPU pick: most free VRAM (manual `devices` unset)"
                );
                if skipped_integrated {
                    let igpu = hw
                        .gpus
                        .iter()
                        .find(|g| g.is_integrated())
                        .map_or("integrated", |g| g.name.as_str());
                    tracing::info!(
                        model = name,
                        "integrated GPU {igpu} reported more free memory but was skipped (shared-RAM bandwidth); pin it explicitly with `devices` if intended"
                    );
                }
            }
        }
        // Placement label for `ps` (`full@<card>`): auto-pick names its
        // card above; cover the other placements — manual `devices`
        // (joined, a multi-card pin spans cards), single-GPU boxes (the
        // child lands on the only card), and auto tensor-split (spans
        // all discrete cards). Nulled for pure-CPU spawns at Instance
        // construction.
        let card_label = picked_display
            .clone()
            .or_else(|| {
                let manual = self.config.effective_devices(name);
                (!manual.is_empty()).then(|| manual.join("+"))
            })
            .or_else(|| {
                let hw = fresh.as_ref().unwrap_or(&self.hardware);
                (hw.gpus.len() == 1).then(|| hw.gpus[0].display_name().to_string())
            })
            .or_else(|| {
                (auto_split.is_some()).then(|| {
                    let hw = fresh.as_ref().unwrap_or(&self.hardware);
                    hw.gpus
                        .iter()
                        .map(|g| g.display_name().to_string())
                        .collect::<Vec<_>>()
                        .join("+")
                })
            });
        // Cache-file dirs (speccache/, sessions/) must exist before the
        // child opens them; profile emission names these paths. Upstream
        // validates --slot-save-path IS a directory, so the per-model
        // subdir must pre-exist.
        let _ = self.dirs.ensure();
        let _ = std::fs::create_dir_all(
            self.dirs
                .sessions_dir()
                .join(pallama_core::profile::path_safe(key)),
        );
        // Retry once on immediate port-race death (bind fail).
        let mut auth_keyfile: Option<std::path::PathBuf> = None;
        let mut child_died_during_load = false;
        // Last dying child's log tail: the single most useful artifact
        // when a spawn fails (upstream's own error, e.g. "failed to
        // create context") — carried into the EngineCrashed payload.
        let mut last_load_tail = String::new();
        for _attempt in 0..2 {
            let endpoint = self.pick_endpoint(key);
            // Child auth dies with the child: minted per attempt (same
            // keyfile path, so retries overwrite), removed when this
            // attempt or the whole spawn fails.
            let auth = match self.mint_child_auth(key, &endpoint, manifest) {
                Ok(a) => a,
                Err(e) => {
                    if let Some(p) = &auth_keyfile {
                        let _ = std::fs::remove_file(p);
                    }
                    return Err(e);
                }
            };
            if let Some(p) = auth.as_ref().and_then(|a| a.keyfile.clone()) {
                auth_keyfile = Some(p);
            }
            let input = ProfileInput {
                engine_kind: self.engine.kind(),
                sibling_devices: sibling_devices.clone(),
                auto_tensor_split: auto_split.clone(),
                model_name: name,
                // Per-replica paths: the argv's sessions/speccache dirs
                // must match the dir created for THIS instance key.
                instance_key: key,
                model_path: &model.path,
                model_bytes: u64::try_from(model.bytes.max(0)).unwrap_or(u64::MAX),
                gguf: &gguf,
                // Scoped (picked card) > fresh census > boot snapshot:
                // single-GPU boxes previously planned against the
                // BOOT-TIME free VRAM (live-repro'd: daemon booted
                // while a foreign 5 GiB context held the card → every
                // later spawn needlessly CPU-split and declined its
                // draft, even with 7.8 GiB actually free).
                hardware: scoped_hw
                    .as_ref()
                    .or(fresh.as_ref())
                    .unwrap_or(&self.hardware),
                config: &self.config,
                overlay: &overlay,
                loras: &loras,
                draft_path: draft_path.as_deref(),
                draft_gguf: draft_gguf.as_ref(),
                mmproj_path: model.mmproj_path.as_deref(),
                // @vision respawn = caller demanded a projector-carrying
                // child (ensure_vision); every other spawn honors policy
                mmproj_force: key.ends_with("@vision"),
                engine_tag: &manifest.tag,
                supported_flags: &manifest.flags,
                spec_types: &manifest.spec_types,
                endpoint: endpoint.clone(),
                data_dir: &data_dir_str,
                cache_hit_rate: self.cache_hint.get(),
                device_hint: picked_device.as_deref(),
                // Build-class detection reads the FULL census: scoped
                // `hardware` above sizes capacity against the picked
                // card, but whether the engine binary is vulkan-class
                // (fat ctx-scaled compute buffers with a projector) is
                // a property of everything it enumerates.
                engine_census: fresh.as_ref().unwrap_or(&self.hardware).gpus.clone(),
            };
            // Co-residency planner (A15): when other models are already
            // resident, sum their (weights + post-quant KV) with the
            // candidate's f16 KV; if the box would not fit, downgrade the
            // candidate to q8_0 KV before spawning instead of OOMing mid-
            // load. Explicit kv_quant (bench-adopted) is never overridden.
            if tuning.kv_quant.is_none() {
                let candidate_kv =
                    pallama_core::profile::estimate_kv_vram_charge(&input, tuning.ctx);
                if self.coresidency_needs_kv_quant(
                    picked_device.as_deref(),
                    model_bytes,
                    candidate_kv,
                ) {
                    tracing::warn!(model = name, "co-residency planner: KV downgraded to q8_0 to fit alongside resident models (weights+KV exceed VRAM)");
                    tuning.kv_quant = Some(true);
                }
            }
            let profile = profile::compile(&input, &tuning)
                .map_err(|e| SupervisionError::Internal(anyhow!("profile: {e}")))?;
            for w in &profile.warnings {
                tracing::warn!(model = name, "profile: {w}");
            }
            let mut argv = self.engine.build_argv(&model, &profile, &endpoint);
            if let Some(a) = &auth {
                argv.extend(a.argv.iter().cloned());
            }
            self.remap_device_argv(&mut argv, name).await;
            let mut child = self.engine.spawn(&argv, &endpoint).await.map_err(|e| {
                if let Some(p) = &auth_keyfile {
                    let _ = std::fs::remove_file(p);
                }
                SupervisionError::Internal(anyhow!("{e}"))
            })?;

            // Child died instantly (port race)? Retry with a fresh port.
            if let Ok(Some(status)) = child.try_status() {
                tracing::warn!(model = name, "engine died at spawn ({status}); retrying");
                let _ = child.kill().await;
                let _ = child.reap().await;
                continue;
            }

            let (health, child_died) = self.wait_healthy(&endpoint, &mut child).await;
            match health {
                Ok(()) => {
                    // F1: an evict is tearing this name down right now —
                    // inserting here would race its cleanup (map remove +
                    // pidfile/apikey deletion). Kill this child and let
                    // the spawn loop retry; the winner inserts after the
                    // evict completes. Checked BEFORE the child moves
                    // into the Instance so it can still be signalled.
                    if self.evicting.lock().await.contains(key) {
                        tracing::info!(
                            model = name,
                            "spawn deferred: evict in progress for this name"
                        );
                        let _ = child.kill().await;
                        let _ = child.reap().await;
                        continue;
                    }
                    // NEVER default the pid: a 0 here would later target
                    // process group 0 (the whole session) on teardown.
                    let pid = child.id().ok_or_else(|| {
                        SupervisionError::Internal(anyhow!(
                            "engine child has no pid; refusing to track instance"
                        ))
                    })?;
                    if pid <= 1 {
                        if let Some(p) = &auth_keyfile {
                            let _ = std::fs::remove_file(p);
                        }
                        return Err(SupervisionError::Internal(anyhow!(
                            "engine child pid {pid} is not a safe process-group id"
                        )));
                    }
                    let autofit_model_name = model.name.clone();
                    let inst = Arc::new(Instance {
                        name: key.to_string(),
                        endpoint: endpoint.clone(),
                        state: std::sync::RwLock::new(InstanceState::Ready),
                        last_used: std::sync::RwLock::new(Instant::now()),
                        in_flight: AtomicI64::new(0),
                        keep_until: std::sync::RwLock::new(None),
                        started_at: Instant::now(),
                        projector: argv.windows(2).any(|w| w[0] == "-mm" || w[0] == "--mmproj"),
                        argv,
                        model,
                        profile_ctx: profile.ctx,
                        gpu: profile.gpu.to_string(),
                        device: if profile.gpu == "cpu" {
                            None
                        } else {
                            card_label.clone()
                        },
                        kv_est_bytes: profile.kv_est_bytes,
                        warnings: profile.warnings.clone(),
                        settled_mib: std::sync::atomic::AtomicU64::new(0),
                        auth: auth.as_ref().map(|a| a.secret.clone()),
                        child: tokio::sync::Mutex::new(child),
                        pid,
                    });
                    // Auto-fit visibility: the spawn traded ctx depth for
                    // parallel slots — surface it as an event so SSE
                    // (/api/events) and API consumers see WHY per-slot
                    // ctx shrank.
                    if let Some((per_slot, slots)) = profile.ctx_autofit {
                        let _ = self.bus.publish(PallamaEvent::SlotsCtxAutoFit {
                            model: autofit_model_name,
                            per_slot_ctx: per_slot,
                            slots,
                            total_ctx: slots * per_slot,
                        });
                    }
                    // Bank restore, choke point #2: warm KV before the
                    // first request prefills (ctx-matched only). Banks
                    // are per-replica (key-scoped files).
                    self.bank_restore(
                        key,
                        &endpoint,
                        profile.ctx,
                        auth.as_ref().map(|a| a.secret.as_str()),
                    )
                    .await;
                    let _ = std::fs::write(
                        self.dirs.run_dir().join(format!("{key}.pid")),
                        pid.to_string(),
                    );
                    self.instances.insert(key.to_string(), inst);
                    // A restart only counts when it follows an unclean
                    // death — churn (stop→run) is a cold start, not a
                    // crash loop (live-repro'd: 4 clean churns in 60s
                    // used to open the breaker and 503 the 5th).
                    if self.unclean_dead.remove(key).is_some() {
                        self.record_restart(key);
                    }
                    self.bus.publish(PallamaEvent::InstanceStateChanged {
                        name: key.to_string(),
                        state: InstanceState::Ready,
                    });
                    let inst = self.instances.get(key).expect("just inserted");
                    // Measured settle check (#28a), OFF the first-request
                    // critical path: the child is healthy and its
                    // weights+KV are resident — re-probe the card and
                    // report what the spawn ACTUALLY took vs the
                    // pre-spawn baseline. Teaching only (drift
                    // visibility); needs the pre-spawn probe as a
                    // baseline, skipped silently without one. Runs as a
                    // detached task AFTER the instance is visible so the
                    // cold first token never waits on the census
                    // subprocess — until the task stores the measured
                    // number, the bytes admission charges the
                    // weights-sum floor (the same pre-settle state that
                    // exists today).
                    if let Some(pre) = fresh {
                        let engine = Arc::clone(&self.engine);
                        let settled_inst = Arc::clone(inst.value());
                        let picked = picked_device.clone();
                        let model_name = name.to_string();
                        let predicted_mib = model_bytes / (1024 * 1024)
                            + profile.kv_est_bytes.map_or(0, |b| b / (1024 * 1024));
                        tokio::spawn(async move {
                            let post = Supervisor::live_hardware_with(&engine);
                            if let Some((card, taken_mib, used_pct)) =
                                settle_report(&pre, &post, picked.as_deref())
                            {
                                settled_inst
                                    .settled_mib
                                    .store(taken_mib, std::sync::atomic::Ordering::Relaxed);
                                tracing::info!(
                                    model = %model_name,
                                    card = %card,
                                    measured_mib = taken_mib,
                                    predicted_mib,
                                    used_pct,
                                    "spawn settle: card free-VRAM delta after load (predicted = weights+KV estimate)"
                                );
                                if used_pct > 95 {
                                    tracing::warn!(
                                        model = %model_name,
                                        card = %card,
                                        used_pct,
                                        "card is over 95% committed after this load — expect KV pressure; consider kv quantization, a smaller quant (pallama fit), or freeing co-resident engines (pallama ps)"
                                    );
                                }
                            }
                        });
                    }
                    // Engine proven healthy: clear the J2 crash-loop tracker.
                    self.engine_failures.lock().unwrap().clear();
                    return Ok(self.engine_ref(inst.value()));
                }
                Err(e) => {
                    // Health never came up: kill and retry once (port
                    // race), then surface the failure — classified by
                    // WHAT failed, not a blanket timeout label.
                    child_died_during_load |= child_died;
                    if child_died {
                        tracing::error!(model = name, "engine died during load: {e:#}");
                    } else {
                        tracing::warn!(model = name, "health check failed: {e:#}");
                    }
                    last_load_tail = child.tail_joined();
                    let _ = child.kill().await;
                    let _ = child.reap().await;
                }
            }
        }
        // Ultimate spawn failure (all in-spawn retries exhausted): feed the
        // J2 crash-loop detector before surfacing the failure.
        if let Some(p) = &auth_keyfile {
            let _ = std::fs::remove_file(p);
        }
        self.note_engine_failure(name);
        Err(if child_died_during_load {
            SupervisionError::EngineCrashed(format!(
                "{key}: {}",
                Self::tail_excerpt(&last_load_tail)
            ))
        } else {
            SupervisionError::ModelLoadTimeout(key.to_string())
        })
    }

    /// Last few non-empty lines of a dead child's log tail, for the
    /// `EngineCrashed` payload — the upstream error line usually sits
    /// right at the end (e.g. "failed to create context").
    fn tail_excerpt(tail: &str) -> String {
        let lines: Vec<&str> = tail
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .collect();
        let take = lines.len().saturating_sub(3)..;
        lines[take].join(" | ")
    }

    /// J2 crash-loop detection: record that `model` failed to spawn and
    /// roll the active engine back to the previous install when the
    /// evidence points at the ENGINE (not the model):
    ///
    /// - `llama-server --version` fails on the active binary → immediate
    ///   rollback (direct evidence; one model is enough), or
    /// - the binary probes fine but ≥2 DISTINCT models keep failing to
    ///   spawn → circumstantial engine evidence (e.g. a serving-path
    ///   regression that `--version` does not exercise).
    ///
    /// Single-model failures with a healthy probe stay model-level
    /// (VRAM, bad GGUF, ctx) — no rollback, no misattribution. Affects
    /// future spawns only; loud (`EngineRolledBack` event + warn) and
    /// reversible (`pallama engine use <tag>`).
    fn note_engine_failure(&self, model: &str) {
        // The failed boot invalidates the census TTL cache: the next
        // spawn must re-measure (the pre-spawn guard never trusts a
        // census that predates a failure — e.g. an external VRAM
        // squatter that appeared after the cached reading).
        *self.census_cache.lock().expect("census cache lock") = None;
        // LC4 rollback: a spawn failure after an adaptive adoption is
        // the capacity answer — the bumped shape did not fit. Drop the
        // adoption (and any queued reshape) so the next spawn returns
        // to the last-known-good shape instead of crash-looping on it.
        if self.adopted_slots.remove(model).is_some() {
            self.reshape_queue.remove(model);
            tracing::warn!(
                model = model,
                "adaptive slots rolled back after spawn failure — the adopted shape did not fit"
            );
        }
        let trigger = {
            let mut fails = self.engine_failures.lock().unwrap();
            fails.insert(model.to_string());
            if !self.probe_active_engine() {
                Some("active engine binary failed its --version probe".to_string())
            } else if fails.len() >= 2 {
                Some(format!(
                    "spawn failures across {} distinct models with a healthy binary probe",
                    fails.len()
                ))
            } else {
                None
            }
        };
        if let Some(reason) = trigger {
            if let Err(e) = self.rollback_active_engine(&reason) {
                tracing::warn!("engine rollback skipped: {e:#}");
            }
        }
    }

    /// Exec the active engine's `llama-server --version` (5 s budget).
    /// Any clean exit = healthy; timeout/crash/non-zero = broken.
    fn probe_active_engine(&self) -> bool {
        let Ok(store) = Store::open(&self.dirs) else {
            return true; // cannot probe: assume innocent (H1: no false rollbacks)
        };
        let Ok(Some(row)) = store.active_engine() else {
            return true;
        };
        advise_cuda_build(&store, &row);
        let dir = self.dirs.engines_dir().join(&row.tag);
        let Ok(bin) = crate::engine::find_server(&dir) else {
            return false; // active engine dir has no server binary: broken
        };
        let ok = std::process::Command::new(&bin)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success());
        ok
    }

    /// Switch the active engine to the previous install (same ordering as
    /// `EngineManager::rollback`: next entry after active = older). Leaves
    /// the tracker armed for a fresh verdict on the new engine.
    fn rollback_active_engine(&self, reason: &str) -> Result<(String, String)> {
        let store = Store::open(&self.dirs)?;
        let engines = store.list_engines()?;
        let active = engines
            .iter()
            .find(|e| e.active)
            .ok_or_else(|| anyhow!("no active engine to roll back from"))?;
        let from = active.tag.clone();
        let target = engines
            .iter()
            .position(|e| e.active)
            .and_then(|idx| engines.get(idx + 1))
            .map(|e| e.tag.clone())
            .ok_or_else(|| anyhow!("no older engine to roll back to (active: {from})"))?;
        drop(store);
        Store::open(&self.dirs)?.set_active_engine(&target)?;
        self.engine_failures.lock().unwrap().clear();
        tracing::warn!(
            "engine {from} rolled back to {target}: {reason} — new spawns use {target}; \
             `pallama engine use {from}` restores it"
        );
        self.bus.publish(PallamaEvent::EngineRolledBack {
            from: from.clone(),
            to: target.clone(),
            reason: reason.to_string(),
        });
        Ok((from, target))
    }

    /// Free TCP port (bind 0, read, drop) or a unix socket path.
    fn pick_endpoint(&self, name: &str) -> Endpoint {
        if self.config.child_transport == "unix" {
            return Endpoint::Unix {
                socket: self
                    .dirs
                    .run_dir()
                    .join(format!("{name}.sock"))
                    .display()
                    .to_string(),
            };
        }
        let port = std::net::TcpListener::bind(("127.0.0.1", 0))
            .and_then(|l| l.local_addr())
            .map_or(0, |a| a.port());
        Endpoint::Tcp {
            host: "127.0.0.1".into(),
            port,
        }
    }

    fn record_restart(&self, name: &str) {
        let mut entry = self.restarts.entry(name.to_string()).or_default();
        entry.push(Instant::now());
        entry.retain(|t| t.elapsed() < self.circuit_window);
    }

    /// True when no name is mid-teardown — leaked marks would defer
    /// every future spawn of that name (starvation class).
    pub async fn evicting_is_empty(&self) -> bool {
        self.evicting.lock().await.is_empty()
    }

    /// Idempotent; publishes
    /// Evicted.
    pub async fn evict(&self, name: &str) -> Result<()> {
        let Some(inst) = self.instances.get(name).map(|i| i.clone()) else {
            return Ok(());
        };
        // F1: mark the name as being torn down for the whole evict; a
        // concurrent spawn retries instead of inserting a fresh child
        // that our cleanup would then race (map remove + pidfile/apikey
        // deletion under it). Cleared explicitly on every exit path —
        // see EvictingName (no Drop: it cannot await).
        let evicting = EvictingName::guard(self, name).await;
        let result = self.evict_teardown(name, &inst).await;
        evicting.release().await;
        result
    }

    async fn evict_teardown(&self, name: &str, inst: &Arc<Instance>) -> Result<()> {
        // Our own teardown just (re)claimed card capacity: the census
        // cache must not survive it, or the next spawn plans against the
        // PRE-evict free-VRAM reading (live-repro'd: 7637 MiB free, a
        // stale census said 3177, and the 8B spawn needlessly split
        // layers / declined its draft).
        *self.census_cache.lock().expect("census cache lock") = None;
        // A clean teardown (user stop / idle ladder / capacity) resets
        // the crash history for this key: the next spawn is a cold
        // start, not a respawn-after-crash.
        self.unclean_dead.remove(name);
        // Session bank, choke point #1 (FIX1): EVERY eviction path —
        // gateway requests, idle ladder, capacity pressure — flows
        // through here, so the `_auto-<ctx>` checkpoint is saved exactly
        // once, bounded, and only for idle instances (live requests own
        // their KV).
        self.bank_save(inst).await;
        *inst.state.write().expect("state lock") = InstanceState::Evicted;
        self.evictions.fetch_add(1, Ordering::Relaxed);
        self.bus.publish(PallamaEvent::InstanceStateChanged {
            name: name.to_string(),
            state: InstanceState::Evicted,
        });
        {
            let mut child = inst.child.lock().await;
            terminate_group(inst.pid, self.shutdown_grace, &mut child).await?;
        }
        // F1: a concurrent respawn can insert a FRESH instance under
        // this name while we waited in the terminate grace; only tear
        // down the map entry + runtime files when the live entry is
        // still OURS (ptr identity). Deleting a fresh child here would
        // orphan it (kill_on_drop SIGKILL mid-request) and remove its
        // pidfile/apikey under it.
        if self
            .instances
            .remove_if(name, |_k, live| Arc::ptr_eq(live, inst))
            .is_some()
        {
            let _ = std::fs::remove_file(self.dirs.run_dir().join(format!("{name}.pid")));
            // Child-auth keyfile dies with the child (its secret too).
            let _ = std::fs::remove_file(self.dirs.run_dir().join(format!("{name}.apikey")));
        } else {
            tracing::info!(
                model = name,
                "evict: map entry was replaced by a respawn during termination — fresh instance left intact"
            );
        }
        Ok(())
    }

    /// Model-level evict (B1): stop the model AND all its replicas
    /// (`qwen`, `qwen#1`, `qwen#2`, …). This is the user-facing stop
    /// (gateway /api/stop, CLI); internal paths (reaper, capacity)
    /// call [`Supervisor::evict`] with one exact key. Force wins over
    /// session pins: they die with the model (R3).
    pub async fn evict_model(&self, model: &str) -> Result<()> {
        let released = self.sessions.release_model(model);
        if released > 0 {
            tracing::info!(target: "pallama::sessions", model = %model, released, "force stop released session pins");
        }
        let keys: Vec<String> = self
            .instances
            .iter()
            .map(|e| e.key().clone())
            .filter(|k| k == model || split_replica(k).is_some_and(|(m, _)| m == model))
            .collect();
        for k in keys {
            self.evict(&k).await?;
        }
        Ok(())
    }

    /// Bank filename for an instance: ctx-scoped (FIX4) so a checkpoint
    /// saved at ctx A never restores into a child spawned at ctx B.
    fn bank_file(&self, name: &str, ctx: u32) -> std::path::PathBuf {
        self.dirs
            .sessions_dir()
            .join(pallama_core::profile::path_safe(name))
            .join(format!("_auto-{ctx}"))
    }

    /// Best-effort `_auto-<ctx>` save before termination. Bounded (2s);
    // failures cost a re-prefill, never the evict.
    async fn bank_save(&self, inst: &Arc<Instance>) {
        if !self.config.session_bank || inst.in_flight.load(Ordering::SeqCst) > 0 {
            return;
        }
        let file = self.bank_file(&inst.name, inst.profile_ctx);
        let url = match &inst.endpoint {
            Endpoint::Tcp { host, port } => {
                format!(
                    "http://{host}:{port}/slots/0?action=save&filename=_auto-{}",
                    inst.profile_ctx
                )
            }
            Endpoint::Unix { .. } => return, // no HTTP lane on UDS children
        };
        let body = if self.config.router {
            serde_json::json!({"filename": format!("_auto-{}", inst.profile_ctx), "model": inst.name})
        } else {
            serde_json::json!({"filename": format!("_auto-{}", inst.profile_ctx)})
        };
        let mut req = reqwest::Client::new()
            .post(&url)
            .json(&body)
            .timeout(std::time::Duration::from_secs(2));
        if let Some(secret) = &inst.auth {
            req = req.bearer_auth(secret);
        }
        let ok = req.send().await.is_ok_and(|r| r.status().is_success());
        if ok {
            // Hoard guard: a per-model bank over 512 MiB is storage
            // abuse, not a cache — drop it and say so once.
            if let Ok(md) = std::fs::metadata(&file) {
                if md.len() > 512 * 1024 * 1024 {
                    let _ = std::fs::remove_file(&file);
                    tracing::warn!(target: "pallama::bank", model = %inst.name, "banked checkpoint > 512 MiB — dropped");
                }
            }
            // #20 identity manifest: stamp the shape so bank restores
            // can refuse stale KV (built from the LIVE instance ctx).
            match pallama_core::session_identity::build(&self.dirs, &self.config, &inst.name) {
                Some(mut id) => {
                    id.ctx = inst.profile_ctx;
                    if let Err(e) = pallama_core::session_identity::write_manifest(&file, &id) {
                        tracing::warn!(target: "pallama::bank", model = %inst.name, "bank identity write failed: {e}");
                    }
                }
                None => {
                    tracing::warn!(target: "pallama::bank", model = %inst.name, "bank identity indeterminate — checkpoint left unverified");
                }
            }
        }
    }

    /// Bank restore, choke point #2: after a fresh spawn reaches
    /// readiness, a matching `_auto-<ctx>` is restored so the first
    /// request rides warm KV. Warn-continue on any failure. A #20
    /// identity mismatch SKIPS the restore — injecting KV from a
    /// different runtime shape is worse than a cold start.
    async fn bank_restore(&self, name: &str, endpoint: &Endpoint, ctx: u32, auth: Option<&str>) {
        if !self.config.session_bank {
            return;
        }
        let file = self.bank_file(name, ctx);
        if !file.exists() {
            return;
        }
        // Shape check before touching the child: refuse silently-stale
        // banks (engine swap, model re-pull, ctx/cache change).
        if let Some(saved) = pallama_core::session_identity::read_manifest(&file) {
            if let Some(mut cur) =
                pallama_core::session_identity::build(&self.dirs, &self.config, name)
            {
                cur.ctx = ctx;
                let diffs = pallama_core::session_identity::verify(&saved, &cur);
                if !diffs.is_empty() {
                    tracing::warn!(
                        target: "pallama::bank",
                        model = name,
                        "bank identity mismatch — SKIPPING restore ({}); continuing cold",
                        diffs.join("; ")
                    );
                    return;
                }
            }
        }
        let url = match endpoint {
            Endpoint::Tcp { host, port } => {
                format!("http://{host}:{port}/slots/0?action=restore&filename=_auto-{ctx}")
            }
            Endpoint::Unix { .. } => return, // no HTTP lane on UDS children
        };
        let body = if self.config.router {
            serde_json::json!({"filename": format!("_auto-{ctx}"), "model": name})
        } else {
            serde_json::json!({"filename": format!("_auto-{ctx}")})
        };
        let mut req = reqwest::Client::new()
            .post(&url)
            .json(&body)
            .timeout(std::time::Duration::from_secs(2));
        if let Some(secret) = auth {
            req = req.bearer_auth(secret);
        }
        match req.send().await {
            Ok(r) if r.status().is_success() => {
                tracing::info!(target: "pallama::bank", model = name, "restored banked session _auto-{ctx}");
            }
            Ok(r) => {
                tracing::warn!(target: "pallama::bank", model = name, "bank restore HTTP {} — continuing cold", r.status());
            }
            Err(e) => {
                tracing::warn!(target: "pallama::bank", model = name, "bank restore failed: {e:#} — continuing cold");
            }
        }
    }

    /// Stop every instance (daemon shutdown). Bounded by grace per child.
    pub async fn shutdown_all(&self) -> Result<()> {
        let names: Vec<String> = self.instances.iter().map(|e| e.key().clone()).collect();
        for n in names {
            self.evict(&n).await?;
        }
        Ok(())
    }

    /// Reaper: implements the ladder (Ready→Sleeping at `idle_sleep_secs`,
    /// →Evicted at `idle_timeout_secs`). Runs until the manager is dropped.
    #[must_use]
    pub fn spawn_reaper(self: &Arc<Self>) -> tokio::task::JoinHandle<()> {
        let mgr = Arc::clone(self);
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(mgr.reaper_interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                ticker.tick().await;
                mgr.reap_dead_children().await;
                mgr.reap_once().await;
            }
        })
    }

    async fn reap_once(&self) {
        let now = Instant::now();
        // R3 session pins: drop expired windows first, then honor the
        // rest — a model with a live session skips the idle-eviction
        // branch (sleep-marking still applies: the process + RAM cache
        // survive a sleep, wake is cheap).
        let session_ttl = self.session_ttl();
        for expired in self.sessions.sweep(session_ttl) {
            tracing::debug!(target: "pallama::sessions", session = %expired, "session pin expired");
        }
        let mut evictions: Vec<String> = Vec::new();
        for entry in &self.instances {
            let inst = entry.value();
            if inst.in_flight.load(Ordering::SeqCst) > 0 {
                continue; // never evict or sleep-mark under load
            }
            let idle = now.duration_since(*inst.last_used.read().expect("idle lock"));
            let state = *inst.state.read().expect("state lock");
            let session_pinned = self
                .sessions
                .pins(model_of_key(&inst.name), session_ttl)
                .live;
            // F12 ollama keep_alive pin: same protection class as a live
            // session — a client that asked to keep the model loaded gets
            // no evict and no sleep-mark until its window lapses.
            let keep_alive = inst
                .keep_until
                .read()
                .expect("idle lock")
                .is_some_and(|t| t > now);
            if idle >= Duration::from_secs(self.config.idle_timeout_secs)
                && !session_pinned
                && !keep_alive
            {
                evictions.push(inst.name.clone());
            } else if idle >= Duration::from_secs(self.config.idle_sleep_secs)
                && state == InstanceState::Ready
                && self.hardware.has_gpu()
                && !keep_alive
            {
                // Child sleeps itself (--sleep-idle-seconds); mark observed.
                let inst2 = inst.clone();
                drop(entry);
                *inst2.state.write().expect("state lock") = InstanceState::Sleeping;
                self.bus.publish(PallamaEvent::InstanceStateChanged {
                    name: inst2.name.clone(),
                    state: InstanceState::Sleeping,
                });
            }
        }
        for name in evictions {
            if let Err(e) = self.evict(&name).await {
                tracing::error!(model = %name, "evict: {e:#}");
            }
        }
        // LC1/LC4 ride the 10s reaper tick.
        self.maybe_preload().await;
        self.adaptive_slots_reap().await;
        self.measured_pressure_tick();
    }

    /// Measured-pressure feedback loop (#28b): re-probe the GPU census
    /// Live hardware view: CPU/RAM via sysinfo + a REAL `--list-devices`
    /// census from the engine binary (llama lane; mistral.rs has no
    /// census flag). Falls back to the manifest's install-day snapshot
    /// when the census yields nothing (binary missing, backend init
    /// failed). The census spawns the engine binary — callers throttle
    /// to spawn-time and ≥60s periodic.
    /// Live census cache lifetime: short enough that the pre-spawn
    /// guard re-measures on any normal (non-burst) spawn, long enough
    /// that a cold-start burst (spawn → settle → pressure tick → next
    /// model's spawn) doesn't exec the census subprocess per step.
    const CENSUS_TTL: Duration = Duration::from_secs(10);

    /// TTL cache read (pure; testable without spawning the engine).
    fn census_from_cache(
        cache: &std::sync::Mutex<Option<(Instant, Hardware)>>,
        ttl: Duration,
    ) -> Option<Hardware> {
        let guard = cache.lock().expect("census cache lock");
        match guard.as_ref() {
            Some((at, hw)) if at.elapsed() < ttl => Some(hw.clone()),
            _ => None,
        }
    }

    fn live_hardware(&self) -> Hardware {
        if let Some(hw) = Self::census_from_cache(&self.census_cache, Self::CENSUS_TTL) {
            return hw;
        }
        let fresh = Self::live_hardware_with(&self.engine);
        *self.census_cache.lock().expect("census cache lock") =
            Some((Instant::now(), fresh.clone()));
        fresh
    }

    /// Engine-only live census, no `self` borrow: lets the post-spawn
    /// settle task run detached from the supervisor while the first
    /// request is already being served.
    fn live_hardware_with(engine: &Arc<dyn Engine>) -> Hardware {
        let manifest = engine.capabilities();
        let live = if engine.kind() == pallama_core::engine_kind::EngineKind::LlamaCpp {
            crate::engine::manifest::run_list_devices(std::path::Path::new(&manifest.server_path))
        } else {
            Vec::new()
        };
        if live.is_empty() {
            return crate::probe::probe_hardware(Some(manifest));
        }
        let gpus = live
            .iter()
            .map(|d| GpuInfo {
                name: d.name.clone(),
                description: d.description.clone(),
                total_mib: d.total_mib,
                free_mib: d.free_mib,
            })
            .collect();
        crate::probe::hardware_with(gpus)
    }

    /// (at most once per minute — the probe spawns `--list-devices`) and
    /// teach loudly when any card runs over 95% committed. Card-LEVEL by
    /// design: a co-resident squatter (ollama, a desktop app) counts the
    /// same as our own children — the number that matters is what the
    /// NEXT spawn would see. Teach-only; spawn-time policy stays with the
    /// card-scoped planner (A15), whose inputs the spawn-path probe
    /// already refreshes.
    fn measured_pressure_tick(&self) {
        if self.instances.is_empty() || !self.hardware.has_gpu() {
            return;
        }
        let due = {
            let mut m = self.measured.lock().expect("measured lock");
            let due = m
                .last_probe
                .is_none_or(|t| t.elapsed() >= Duration::from_mins(1));
            if due {
                m.last_probe = Some(Instant::now());
            }
            due
        };
        if !due {
            return;
        }
        let now = self.live_hardware();
        for g in &now.gpus {
            if g.total_mib == 0 {
                continue;
            }
            let used_pct = (g.total_mib.saturating_sub(g.free_mib)) * 100 / g.total_mib;
            if used_pct <= 95 {
                continue;
            }
            let mut m = self.measured.lock().expect("measured lock");
            let stale = m
                .warned_cards
                .get(&g.name)
                .is_none_or(|t| t.elapsed() >= Duration::from_mins(10));
            if stale {
                m.warned_cards.insert(g.name.clone(), Instant::now());
                drop(m);
                tracing::warn!(
                    card = %g.name,
                    used_pct,
                    free_mib = g.free_mib,
                    "measured VRAM pressure: card is over 95% committed — new spawns will spill or degrade; free co-resident engines (pallama ps / ollama stop) or kv-quantize"
                );
            }
        }
    }

    /// LC1 predictive pre-loading: when exactly one model is resident
    /// and idle, and history says B reliably follows A, spawn B NOW so
    /// the user's next switch is warm instead of a cold start. Every
    /// guard is fail-open: any miss (no history, no capacity, no VRAM,
    /// backoff) simply skips this tick.
    async fn maybe_preload(&self) {
        if !self.config.predictive_preload {
            return;
        }
        // Exactly one live non-router instance, idle.
        let mut live: Vec<(String, u64)> = Vec::new(); // (key, model bytes)
        for e in &self.instances {
            let i = e.value();
            if model_of_key(&i.name) == ROUTER_KEY {
                continue;
            }
            let state = *i.state.read().expect("state lock");
            if matches!(state, InstanceState::Ready | InstanceState::Sleeping)
                && i.in_flight.load(Ordering::SeqCst) == 0
            {
                live.push((
                    i.name.clone(),
                    u64::try_from(i.model.bytes.max(0)).unwrap_or(u64::MAX),
                ));
            }
        }
        if live.len() != 1 || !self.loading.is_empty() {
            return;
        }
        let (key, _a_bytes) = live[0].clone();
        let current = model_of_key(&key).to_string();
        // Confident transition target.
        let Some(target) = self
            .transitions
            .iter()
            .filter(|e| e.key().0 == current && e.key().1 != current)
            .filter(|e| {
                // Not already live and not the router pseudo-model.
                e.key().1 != ROUTER_KEY
                    && !self
                        .instances
                        .iter()
                        .any(|i| model_of_key(i.key()) == e.key().1)
            })
            .max_by_key(|e| *e.value())
            .map(|e| (e.key().1.clone(), *e.value()))
        else {
            return;
        };
        let (next, count) = target;
        if count < PRELOAD_MIN_TRANSITIONS {
            return;
        }
        // Recent-failure backoff.
        if let Some(t) = self.preload_failures.get(&next) {
            if t.elapsed() < PRELOAD_BACKOFF {
                return;
            }
        }
        // Known model + capacity headroom.
        let Ok(store) = Store::open(&self.dirs) else {
            return;
        };
        let Ok(Some(row)) = store.get_model(&next) else {
            return;
        };
        let b_bytes = u64::try_from(row.bytes.max(0)).unwrap_or(u64::MAX);
        let b_mmproj = row
            .mmproj_path
            .as_deref()
            .and_then(|p| std::fs::metadata(p).ok())
            .map_or(0, |m| m.len());
        let b_floor = pallama_core::profile::admission_floor_bytes(b_bytes, b_mmproj);
        // Early-exit gates only (the spawn admission re-checks under the
        // instances lock; the fresh free-VRAM preflight below still owns
        // the co-residency verdict on live numbers). Both sides charge
        // admission floors: A's measured footprint (weights floored) and
        // B's floor — weights-sum planning is what crashed the 8 GiB
        // card on 2026-09-11.
        if self
            .instance_cap()
            .is_some_and(|cap| self.instances.len() >= cap)
        {
            return;
        }
        if self.bytes_admission_active()
            && self.resident_bytes().saturating_add(b_floor) > self.vram_budget_bytes()
        {
            return;
        }
        // Fresh VRAM check: both footprints must fit on the GPU pool.
        let fresh = self.live_hardware();
        let free_vram: u64 = fresh.gpus.iter().map(|g| g.free_mib).sum();
        let need_mib = (self.resident_bytes().saturating_add(b_floor)) / (1024 * 1024);
        if fresh.has_gpu() && free_vram > 0 && need_mib > free_vram * 95 / 100 {
            tracing::debug!(model = %next, need_mib, free_vram, "preload skipped: VRAM");
            return;
        }
        tracing::info!(model = %next, from = %current, count, "predictive preload");
        match self.spawn_instance(&next).await {
            Ok(_) => {
                let _ = self.bus.publish(PallamaEvent::ModelPreloaded {
                    model: next,
                    from: current,
                });
            }
            Err(e) => {
                tracing::warn!("predictive preload of {next} failed: {e:#}");
                self.preload_failures.insert(next, Instant::now());
            }
        }
    }

    /// LC4 adaptive slots: a model under sustained concurrent load
    /// (queueing visible as in-flight > live slots across ticks) earns
    /// an in-memory slots bump. The live slot count is read from the
    /// spawned child's argv (`-np`), so capacity-shaped auto spawns
    /// (config `slots = 0`) adapt too — the pre-2026-09-12 form only
    /// saw `slots = 1` configs and left the default auto shape inert.
    /// Explicit overlay slots, replicas, and `deterministic` exclude
    /// the model (a pin is a policy statement); opt-in via config.
    /// Adoption queues a respawn (see [`Self::drain_reshape_queue`]).
    fn adaptive_slots_tick(&self) {
        if !self.config.adaptive_slots {
            return;
        }
        for e in &self.instances {
            let key = e.key().clone();
            let model = model_of_key(&key).to_string();
            let i = e.value();
            let state = *i.state.read().expect("state lock");
            if !matches!(state, InstanceState::Ready) {
                self.busy_streak.remove(&key);
                self.idle_streak.remove(&key);
                continue;
            }
            let overlay = self.config.overlay_for(&model);
            // Manual slots, replicas, or deterministic mode exclude the
            // model entirely (deterministic spawns pin slots = 1 by
            // design — adoption would fight the pin every tick).
            if overlay.slots.is_some()
                || overlay.replicas.unwrap_or(1) > 1
                || overlay.deterministic.unwrap_or(self.config.deterministic)
            {
                self.busy_streak.remove(&key);
                self.idle_streak.remove(&key);
                continue;
            }
            // Live shape from the child's own argv (source of truth);
            // fabricated test instances without argv fall back to the
            // config value, preserving the pre-generalization semantics.
            let resolved = i
                .argv
                .windows(2)
                .find(|w| w[0] == "-np" || w[0] == "--parallel")
                .and_then(|w| w[1].parse::<u32>().ok())
                .unwrap_or(self.config.slots);
            let effective = self
                .adopted_slots
                .get(&model)
                .map_or(resolved, |v| *v.value());
            let in_flight = i.in_flight.load(Ordering::SeqCst);
            // Demand signal: admission pressure since the last tick (the
            // gateway reports every request that queued on AllSlotsBusy).
            // in_flight alone is unreachable past the slot count —
            // admission caps it there (live-proven 2026-09-12: 6-stream
            // load left in_flight pinned at 4 while 2 requests queued).
            // gauge PEEK, not a drain: entries live for as long as the
            // queued request is parked, so sustained queueing shows up
            // on every tick in the window (an event-drain resets the
            // streak after one tick — live-caught 2026-09-12)
            let pressure = self.slot_pressure.get(&model).map_or(0, |v| *v);
            let saturated = pressure > 0 || (effective > 0 && in_flight > i64::from(effective));
            if saturated {
                // Saturation cancels any pending decay count.
                self.idle_streak.remove(&key);
                // Scope the entry guard: it must drop BEFORE the remove
                // below, or the DashMap shard self-deadlocks.
                let hit_threshold = {
                    let mut streak = self.busy_streak.entry(key.clone()).or_insert(0);
                    *streak += 1;
                    *streak >= SLOTS_STREAK_TICKS
                };
                if hit_threshold {
                    let from = effective;
                    let to = (from + 1).min(SLOTS_ADOPT_CAP);
                    self.busy_streak.remove(&key);
                    if to > from {
                        self.adopted_slots.insert(model.clone(), to);
                        // Queue the respawn: the drain below performs it
                        // once live streams finish (in-flight = 0), so
                        // the reshape never kills an active stream.
                        self.reshape_queue.insert(model.clone(), key.clone());
                        tracing::info!(model = %model, from, to, "adaptive slots adopted");
                        self.bus
                            .publish(PallamaEvent::SlotsAutoAdopted { model, from, to });
                    }
                }
            } else if in_flight == 0 && self.adopted_slots.contains_key(&model) {
                // Decay: the adopted shape buys queue-latency under load
                // and costs per-stream ITL (np8 ITL 60ms vs np4 37ms,
                // /tmp/opencode/flagprobe/results2.json) — after a long
                // fully-quiet window, fall back to the natural shape so
                // single-stream latency recovers. One-shot to base: the
                // adopt path re-raises in 60s if demand returns.
                let decayed = {
                    let mut streak = self.idle_streak.entry(key.clone()).or_insert(0);
                    *streak += 1;
                    *streak >= SLOTS_DECAY_TICKS
                };
                if decayed {
                    self.idle_streak.remove(&key);
                    self.busy_streak.remove(&key);
                    self.adopted_slots.remove(&model);
                    self.reshape_queue.insert(model.clone(), key.clone());
                    tracing::info!(
                        model = %model,
                        to = resolved,
                        "adaptive slots decayed: sustained quiet — respawning at the natural shape (per-stream latency recovers)"
                    );
                }
            } else {
                self.busy_streak.remove(&key);
                self.idle_streak.remove(&key);
            }
        }
    }

    /// LC4 reaper half: observe saturation, then reshape. Runs on the
    /// 10s tick; see [`Self::adaptive_slots_tick`] for the adoption
    /// rule and [`Self::drain_reshape_queue`] for the respawn.
    async fn adaptive_slots_reap(&self) {
        self.adaptive_slots_tick();
        self.drain_reshape_queue().await;
    }

    /// Respawn adopted models whose streams have drained. An entry
    /// whose instance is still busy stays queued for the next tick
    /// (sustained 24/7 load defers the reshape to the natural
    /// idle-evict respawn — honest, never disruptive).
    async fn drain_reshape_queue(&self) {
        let entries: Vec<(String, String)> = self
            .reshape_queue
            .iter()
            .map(|e| (e.key().clone(), e.value().clone()))
            .collect();
        for (model, key) in entries {
            let busy = self
                .instances
                .get(&key)
                .is_some_and(|i| i.in_flight.load(Ordering::SeqCst) > 0);
            if busy {
                continue;
            }
            self.reshape_queue.remove(&model);
            tracing::warn!(
                model = %model,
                "adaptive slots: respawning with the adopted slot count \
                 (the KV bank carries conversations across the reshape)"
            );
            if let Err(e) = self.evict(&key).await {
                tracing::warn!(model = %model, "adaptive reshape evict: {e:#}");
                continue;
            }
            if let Err(e) = self.ensure(&model).await {
                tracing::warn!(model = %model, "adaptive reshape respawn: {e:#}");
            }
        }
    }

    /// Gateway admission brackets queue-waits with enter/leave calls —
    /// this is a GAUGE of requests currently parked in the admission
    /// queue, not an event counter. Live-proven 2026-09-12: an event
    /// counter drains after the first reaper tick (each queued stream
    /// reports exactly once on `AllSlotsBusy` and then parks inside
    /// `queue.wait` without retrying), so a per-tick streak built on
    /// accumulated events can never reach its threshold. The reaper
    /// needs the LEVEL (how many are waiting right now), which only a
    /// gauge carries across ticks.
    pub fn note_slot_pressure(&self, model: &str) {
        *self.slot_pressure.entry(model.to_string()).or_insert(0) += 1;
    }

    /// The leaving half of the admission-pressure gauge: the queued
    /// request either acquired a slot or gave up — either way it is no
    /// longer demand the slot reshaper should react to.
    pub fn note_slot_pressure_release(&self, model: &str) {
        if let Some(mut e) = self.slot_pressure.get_mut(model) {
            let left = e.value().saturating_sub(1);
            *e.value_mut() = left;
            if left == 0 {
                drop(e);
                self.slot_pressure.remove(model);
            }
        }
    }

    /// Live slot ceiling for the gateway admission gate. The static
    /// `config.slots` is the AUTO sentinel (0) most of the time — a
    /// hardcoded fallback there would silently cap admission below an
    /// adopted/reshaped child (live-caught 2026-09-12: `slots == 0`
    /// mapped to a flat 4 while the reshape path can adopt up to 8).
    /// Resolution order: adopted count (the reshaper's decision) → the
    /// live child argv `-np` (source of truth, same parse as the
    /// adaptive tick) → explicit config value → modest auto default.
    pub fn slot_cap(&self, model: &str) -> i64 {
        // Canonicalize first: the gateway may hand a replica key
        // ("m#1") while adoption is recorded against the base model.
        let model = model_of_key(model);
        if let Some(v) = self.adopted_slots.get(model) {
            return i64::from(*v);
        }
        let argv_cap = self.instances.iter().find_map(|e| {
            if model_of_key(e.key()) != model {
                return None;
            }
            e.value()
                .argv
                .windows(2)
                .find(|w| w[0] == "-np" || w[0] == "--parallel")
                .and_then(|w| w[1].parse::<u32>().ok())
                .map(i64::from)
        });
        argv_cap
            .or_else(|| {
                let s = self.config.slots;
                if s > 0 {
                    Some(i64::from(s))
                } else {
                    None
                }
            })
            .unwrap_or(4)
    }

    /// Request accounting: gateway brackets proxied calls with these.
    pub fn begin_request(&self, name: &str) {
        if let Some(i) = self.instances.get(name) {
            i.in_flight.fetch_add(1, Ordering::SeqCst);
            *i.last_used.write().expect("idle lock") = Instant::now();
        }
    }

    pub fn end_request(&self, name: &str) {
        if let Some(i) = self.instances.get(name) {
            // F1 companion: a request that began on a PREVIOUS generation
            // of this name can end after a respawn replaced the map entry;
            // never let the shared counter go negative (a negative value
            // would read as permanently-busy in ps/admission).
            let mut cur = i.in_flight.load(Ordering::SeqCst);
            while cur > 0 {
                match i
                    .in_flight
                    .compare_exchange(cur, cur - 1, Ordering::SeqCst, Ordering::SeqCst)
                {
                    Ok(_) => break,
                    Err(now) => cur = now,
                }
            }
            *i.last_used.write().expect("idle lock") = Instant::now();
        }
    }

    /// ps rows for CLI/API.
    #[must_use]
    pub fn ps(&self) -> Vec<PsRow> {
        self.instances
            .iter()
            .map(|e| {
                let i = e.value();
                let heat = self.heat_of(&i.name);
                PsRow {
                    name: model_of_key(&i.name).to_string(),
                    replica: split_replica(&i.name).map(|(_, idx)| idx),
                    state: i.state.read().expect("state lock").as_str(),
                    endpoint: match &i.endpoint {
                        Endpoint::Tcp { host, port } => format!("{host}:{port}"),
                        Endpoint::Unix { socket } => socket.clone(),
                    },
                    idle_secs: i.last_used.read().expect("idle lock").elapsed().as_secs(),
                    in_flight: i.in_flight.load(Ordering::SeqCst),
                    ctx: i.profile_ctx,
                    gpu: i.gpu.clone(),
                    device: i.device.clone(),
                    warnings: i.warnings.clone(),
                    pid: i.pid,
                    bytes: i.model.bytes,
                    heat,
                    keep_alive_secs: i
                        .keep_until
                        .read()
                        .expect("idle lock")
                        .map(|t| t.saturating_duration_since(Instant::now()).as_secs()),
                }
            })
            .collect()
    }

    /// Startup sweep: a SIGKILL'd daemon leaves engine children behind
    /// (own process groups). `<run>/<model>.pid` markers whose recorded
    /// process is still alive get a single-pid TERM; markers are cleared.
    /// The daemon's own `pallama.pid` and pull locks are NOT engines.
    pub fn sweep_orphans(&self) -> Vec<String> {
        let mut swept = Vec::new();
        let Ok(entries) = std::fs::read_dir(self.dirs.run_dir()) else {
            return swept;
        };
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            #[allow(clippy::case_sensitive_file_extension_comparisons)] // our own marker names
            if name == "pallama.pid" || name.starts_with("pull-") || !name.ends_with(".pid") {
                continue;
            }
            let Ok(content) = std::fs::read_to_string(e.path()) else {
                continue;
            };
            let Ok(pid) = content.trim().parse::<u32>() else {
                continue;
            };
            if pid > 1 && crate::daemon::process_alive_by_pid(pid) {
                tracing::warn!("orphan engine pid {pid} ({name}) from a dead daemon; terminating");
                #[cfg(unix)]
                #[allow(unsafe_code)]
                // SAFETY: single positive pid from our own marker file.
                unsafe {
                    libc::kill(i32::try_from(pid).unwrap_or(-1), libc::SIGTERM);
                }
                swept.push(format!("{name}={pid}"));
            }
            let _ = std::fs::remove_file(e.path());
        }
        swept
    }

    /// Queue a ctx override for the model's next spawn (per-request
    /// `options.num_ctx`). Consumed once; clamped by the profile compiler
    /// against the model's trained context.
    pub fn set_next_ctx(&self, model: &str, ctx: u32) {
        self.pending_ctx.insert(model.to_string(), ctx);
    }

    /// Apply an ollama `keep_alive` request to a live instance (F12):
    /// `secs > 0` pins for that window, `secs < 0` pins to a far-future
    /// cap ("forever"), `0` clears any prior pin (the response lane then
    /// evicts explicitly). Returns whether the instance was found live.
    pub fn set_keep_alive(&self, key: &str, secs: i64) -> bool {
        let Some(inst) = self.instances.get(key) else {
            return false;
        };
        let window = match secs {
            0 => None,
            n if n > 0 => Some(Duration::from_secs(
                u64::try_from(n).unwrap_or(u64::MAX / 2),
            )),
            _ => Some(KEEP_ALIVE_FOREVER),
        };
        let window = window.map(|d| Instant::now() + d.min(KEEP_ALIVE_FOREVER));
        *inst.keep_until.write().expect("idle lock") = window;
        true
    }

    /// Prefix heat: +1 per chat-family hit (saturating), decaying with a
    /// 10-minute half-life on read — recent traffic dominates old, an
    /// hour-old hot model cools to evictable. Drives capacity-eviction
    /// victim choice (hot caches survive).
    pub fn note_prefix_hit(&self, model: &str) {
        let mut heat = self.heat.lock().expect("heat lock");
        let e = heat.entry(model.to_string()).or_insert((0, Instant::now()));
        e.0 = e.0.saturating_add(1).min(1000);
        e.1 = Instant::now();
        // Bound the table: models come and go; 256 entries is far beyond
        // any live box's model count.
        if heat.len() > 256 {
            if let Some((coldest, _)) = heat.iter().min_by_key(|(_, v)| v.0) {
                let coldest = coldest.clone();
                heat.remove(&coldest);
            }
        }
    }

    /// Decay-adjusted heat of a model (0 = cold / never seen). Half-life
    /// 10 min: heat >> whole half-lives elapsed since the last hit.
    pub fn heat_of(&self, model: &str) -> u64 {
        self.heat
            .lock()
            .expect("heat lock")
            .get(model)
            .map_or(0, |(h, at)| h >> (at.elapsed().as_secs() / 600).min(63))
    }

    /// Circuit reset (`pallama ps --reset`).
    pub fn reset_circuit(&self, name: Option<&str>) {
        match name {
            Some(n) => {
                self.restarts.remove(n);
            }
            None => self.restarts.clear(),
        }
    }

    /// Detect a dead child (crash) and drop it from the map so the next
    /// `ensure()` respawns. Called by the reaper.
    pub async fn reap_dead_children(&self) {
        // Snapshot FIRST, then await: iterating a DashMap holds a read
        // guard on the current shard, and `child.lock().await` below
        // would park that guard across an await — every write to the
        // shard (spawn `insert`, evict `remove`) then blocks its worker
        // thread in a sync futex. A tick coinciding with an evict that
        // holds the child mutex through a slow terminate was observed
        // stalling the whole runtime (accept, /health, SIGTERM drain);
        // this snapshot removes the map guard from the await path.
        let snapshot: Vec<std::sync::Arc<Instance>> =
            self.instances.iter().map(|e| e.value().clone()).collect();
        let mut crashed: Vec<std::sync::Arc<Instance>> = Vec::new();
        for inst in &snapshot {
            let mut child = inst.child.lock().await;
            match child.try_status() {
                Ok(Some(status)) => {
                    // Exit 0 = the child shut itself down on purpose
                    // (--sleep-idle-seconds idle exit, upstream clean
                    // shutdown) — not a crash; ERROR here would cry wolf.
                    if status.success() {
                        tracing::info!(
                            model = %inst.name,
                            "engine exited cleanly ({status}) — idle sleep or upstream shutdown; next request will respawn"
                        );
                    } else {
                        tracing::error!(
                            model = %inst.name,
                            "engine crashed ({status}); next request will respawn"
                        );
                        // Unclean death: the next successful spawn of
                        // this key counts toward the circuit breaker.
                        self.unclean_dead.insert(inst.name.clone(), ());
                    }
                    crashed.push(inst.clone());
                }
                Ok(None) => {}
                Err(e) => tracing::warn!(model = %inst.name, "child poll: {e}"),
            }
        }
        for inst in crashed {
            // F1: only drop the map entry (and runtime files) when the
            // live entry is still the CRASHED instance — a concurrent
            // ensure() may have already respawned a fresh child under
            // this name; deleting it here would orphan it. F3: the
            // crash lane now also removes the child-auth keyfile (it
            // previously leaked as a stale secret file).
            let name = inst.name.clone();
            if self
                .instances
                .remove_if(&name, |_k, live| Arc::ptr_eq(live, &inst))
                .is_some()
            {
                let _ = std::fs::remove_file(self.dirs.run_dir().join(format!("{name}.pid")));
                let _ = std::fs::remove_file(self.dirs.run_dir().join(format!("{name}.apikey")));
            }
            self.bus.publish(PallamaEvent::InstanceStateChanged {
                name,
                state: InstanceState::Crashed,
            });
        }
    }
}

/// F1: clears the evict-in-progress mark when the evict lane ends, on
/// every path (success, error, early return).
/// F1: in-flight eviction marks. NOT RAII — Drop cannot await, and a
/// best-effort sync clear could leak the mark (permanently deferring
/// spawns). `evict` clears explicitly on every path; the set is
/// advisory (spawn deferral), so a same-name re-evict race at worst
/// lets a spawn through mid-teardown, which the F1 ptr-identity check
/// in the teardown itself already handles.
struct EvictingName<'a>(&'a Supervisor, String);
impl<'a> EvictingName<'a> {
    async fn guard(s: &'a Supervisor, name: &str) -> Self {
        s.evicting.lock().await.insert(name.to_string());
        Self(s, name.to_string())
    }
    async fn release(&self) {
        self.0.evicting.lock().await.remove(&self.1);
    }
}

fn clone_load_result(
    res: &Result<EngineRef, String>,
    name: &str,
) -> Result<EngineRef, SupervisionError> {
    match res {
        Ok(ep) => Ok(ep.clone()),
        Err(text) => Err(SupervisionError::Internal(anyhow!("{text} (from {name})"))),
    }
}

fn map_result(r: &Result<EngineRef, SupervisionError>) -> Result<EngineRef, String> {
    match r {
        Ok(ep) => Ok(ep.clone()),
        Err(e) => Err(e.to_string()),
    }
}

/// Audit trail that survives session death: every signal decision is
/// appended here so any future incident has exact evidence.
fn audit_signal(action: &str, pid: u32, note: &str) {
    use std::io::Write as _;
    let line = format!(
        "{} action={} pid={} note={}\n",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        action,
        pid,
        note
    );
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/pallama-signal-audit.log")
    {
        let _ = f.write_all(line.as_bytes());
    }
}

/// Terminate one child by SINGLE-PID signals only (TERM, then KILL).
///
/// Design rule after two session-death incidents: NEVER signal a process
/// group from Pallama. Group math (-pgid) is one removed step from
/// signalling an unrelated group (pid reuse, group 0 = the session);
/// a single positive pid of our own unreaped child cannot miss.
#[allow(unsafe_code)] // one audited libc::kill call below
async fn terminate_group(pid: u32, grace: Duration, child: &mut ChildHandle) -> Result<()> {
    let still_ours_and_alive = pid > 1 && matches!(child.try_status(), Ok(None));
    if still_ours_and_alive {
        audit_signal("TERM", pid, "graceful stop of own child");
        #[cfg(unix)]
        // SAFETY: pid > 1 was verified against our own unreaped child two
        // lines above; libc::kill signals exactly that pid, no group.
        unsafe {
            libc::kill(i32::try_from(pid).unwrap_or(-1), libc::SIGTERM);
        }
    }
    let deadline = tokio::time::Instant::now() + grace;
    loop {
        // Exited (or not forked): nothing more to signal either way.
        if !matches!(child.try_status(), Ok(None)) {
            return Ok(());
        }
        if tokio::time::Instant::now() >= deadline {
            tracing::warn!(pid, "engine child did not exit within grace; SIGKILL");
            let _ = child.kill().await;
            let _ = child.reap().await;
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[cfg(test)]
#[allow(non_snake_case)] // scenario-style test names match the repo convention
mod routing_tests {
    use super::*;
    use crate::engine::manifest::Manifest;
    use pallama_core::{ModelOverride, Profile};

    /// Engine stub: routing never spawns, so every method is inert.
    struct FakeEngine(Manifest);
    #[async_trait::async_trait]
    impl Engine for FakeEngine {
        fn kind(&self) -> pallama_core::engine_kind::EngineKind {
            pallama_core::engine_kind::EngineKind::LlamaCpp
        }
        fn capabilities(&self) -> &Manifest {
            &self.0
        }
        fn build_argv(
            &self,
            _model: &ModelRow,
            _profile: &Profile,
            _endpoint: &Endpoint,
        ) -> Vec<String> {
            vec![]
        }
        async fn spawn(&self, _argv: &[String], _endpoint: &Endpoint) -> Result<ChildHandle> {
            Err(anyhow!("FakeEngine never spawns"))
        }
        async fn health_check(&self, _endpoint: &Endpoint, _timeout: Duration) -> Result<()> {
            Ok(())
        }
    }

    fn routing_sup(replicas: u32) -> Supervisor {
        let manifest = Manifest {
            tag: "fake".into(),
            build_number: 1,
            version_raw: "b1".into(),
            devices: vec![],
            flags: std::collections::BTreeSet::new(),
            spec_types: vec![],
            server_path: String::new(),
        };
        let mut config = Config::default();
        config.model_overrides.insert(
            "m".into(),
            ModelOverride {
                replicas: Some(replicas),
                ..Default::default()
            },
        );
        Supervisor::new(
            PallamaDirs {
                config_dir: std::path::PathBuf::from("/tmp/pallama-routing-test-cfg"),
                data_dir: std::path::PathBuf::from("/tmp/pallama-routing-test-data"),
            },
            config,
            EventBus::default(),
            Hardware {
                physical_cores: 1,
                total_ram_mib: 1024,
                gpus: vec![],
            },
            Arc::new(FakeEngine(manifest)),
        )
    }

    /// J2 harness: real temp dirs + store rows + a script posing as
    /// llama-server under engines/{tag}/llama-{tag}/.
    fn j2_sup(exit_code: i32) -> (Supervisor, tempfile::TempDir, EventBus) {
        use std::os::unix::fs::PermissionsExt;
        let bus = EventBus::default();
        let root = tempfile::TempDir::new().unwrap();
        let dirs = PallamaDirs {
            config_dir: root.path().join("cfg"),
            data_dir: root.path().join("data"),
        };
        std::fs::create_dir_all(&dirs.config_dir).unwrap();
        std::fs::create_dir_all(dirs.engines_dir().join("b_bad/llama-b_bad")).unwrap();
        std::fs::create_dir_all(dirs.engines_dir().join("b_good/llama-b_good")).unwrap();
        for tag in ["b_bad", "b_good"] {
            let bin = dirs
                .engines_dir()
                .join(tag)
                .join(format!("llama-{tag}"))
                .join("llama-server");
            std::fs::write(&bin, format!("#!/bin/sh\nexit {exit_code}\n")).unwrap();
            std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let store = Store::open(&dirs).unwrap();
        let now = 1_000_000;
        store
            .upsert_engine(&pallama_core::store::EngineRow {
                tag: "b_bad".into(),
                asset: "a".into(),
                sha256: String::new(),
                installed_at: now + 1,
                active: true,
                manifest: String::new(),
                kind: pallama_core::engine_kind::EngineKind::default(),
            })
            .unwrap();
        store
            .upsert_engine(&pallama_core::store::EngineRow {
                tag: "b_good".into(),
                asset: "a".into(),
                sha256: String::new(),
                installed_at: now,
                active: false,
                manifest: String::new(),
                kind: pallama_core::engine_kind::EngineKind::default(),
            })
            .unwrap();
        let sup = Supervisor::new(
            dirs,
            Config::default(),
            bus.clone(),
            Hardware {
                physical_cores: 1,
                total_ram_mib: 1024,
                gpus: vec![],
            },
            Arc::new(FakeEngine(Manifest {
                tag: "fake".into(),
                build_number: 1,
                version_raw: "b1".into(),
                devices: vec![],
                flags: std::collections::BTreeSet::new(),
                spec_types: vec![],
                server_path: String::new(),
            })),
        );
        (sup, root, bus)
    }

    #[test]
    fn unit__map_devices__keeps_live_rewrites_reenumerated_drops_invisible() {
        fn dev(name: &str, desc: &str, mib: u64) -> crate::engine::manifest::DeviceDesc {
            crate::engine::manifest::DeviceDesc {
                name: name.into(),
                description: desc.into(),
                total_mib: mib,
                free_mib: mib,
            }
        }
        let argv = |d: &str| {
            vec![
                "-m".to_string(),
                "x".to_string(),
                "--device".to_string(),
                d.to_string(),
                "-t".to_string(),
                "4".to_string(),
            ]
        };
        let frozen = vec![dev("Vulkan1", "NVIDIA GeForce RTX 4070 Laptop GPU", 8188)];
        // (a) name still enumerated: kept verbatim, no warnings.
        let live = vec![
            dev("Vulkan0", "Intel(R) Graphics (RPL-S)", 10256),
            dev("Vulkan1", "NVIDIA GeForce RTX 4070 Laptop GPU", 8188),
        ];
        let (out, w) = Supervisor::map_devices(&argv("Vulkan1"), &frozen, &live);
        assert!(out.contains(&"Vulkan1".to_string()));
        assert!(w.is_empty());
        // (b) same hardware re-enumerated under a new name: rewritten.
        let live2 = vec![dev("Vulkan0", "NVIDIA GeForce RTX 4070 Laptop GPU", 8188)];
        let (out, w) = Supervisor::map_devices(&argv("Vulkan1"), &frozen, &live2);
        assert!(out.contains(&"Vulkan0".to_string()));
        assert!(!out.contains(&"Vulkan1".to_string()));
        assert_eq!(w.len(), 1);
        // (c) no live counterpart: pair dropped with an explanatory warn.
        let (out, w) = Supervisor::map_devices(&argv("Vulkan1"), &frozen, &[]);
        assert!(!out.iter().any(|a| a == "--device"));
        assert_eq!(w.len(), 1);
        assert!(w[0].contains("not visible"));
    }

    #[test]
    fn unit__j2__broken_binary_probe_rolls_back_immediately() {
        let (sup, _root, bus) = j2_sup(1); // --version exits 1: broken
        let mut events = bus.subscribe();
        sup.note_engine_failure("model-a");
        let active = Store::open(&sup.dirs)
            .unwrap()
            .active_engine()
            .unwrap()
            .unwrap();
        assert_eq!(active.tag, "b_good");
        assert!(matches!(
            events.try_recv().unwrap(),
            PallamaEvent::EngineRolledBack { from, .. } if from == "b_bad"
        ));
    }

    #[test]
    fn unit__j2__healthy_binary_single_model_failure_no_rollback() {
        let (sup, _root, _bus) = j2_sup(0); // healthy probe
        sup.note_engine_failure("model-a");
        let active = Store::open(&sup.dirs)
            .unwrap()
            .active_engine()
            .unwrap()
            .unwrap();
        assert_eq!(active.tag, "b_bad"); // untouched: model-level failure
    }

    #[test]
    fn unit__j2__healthy_binary_two_distinct_models_rolls_back() {
        let (sup, _root, _bus) = j2_sup(0);
        sup.note_engine_failure("model-a");
        sup.note_engine_failure("model-b");
        let active = Store::open(&sup.dirs)
            .unwrap()
            .active_engine()
            .unwrap()
            .unwrap();
        assert_eq!(active.tag, "b_good");
    }

    /// GPU supervisor with one 8188 MiB card — the 2026-09-11 crash box
    /// shape (`NVRM NO_MEMORY` storm from a weights-only admission).
    fn gpu_sup() -> (Supervisor, tempfile::TempDir) {
        let bus = EventBus::default();
        let root = tempfile::TempDir::new().unwrap();
        let dirs = PallamaDirs {
            config_dir: root.path().join("cfg"),
            data_dir: root.path().join("data"),
        };
        std::fs::create_dir_all(&dirs.config_dir).unwrap();
        let sup = Supervisor::new(
            dirs,
            Config::default(),
            bus,
            Hardware {
                physical_cores: 1,
                total_ram_mib: 16_000,
                gpus: vec![GpuInfo {
                    name: "g".into(),
                    description: "S".into(),
                    total_mib: 8_188,
                    free_mib: 8_188,
                }],
            },
            Arc::new(FakeEngine(Manifest {
                tag: "fake".into(),
                build_number: 1,
                version_raw: "b1".into(),
                devices: vec![],
                flags: std::collections::BTreeSet::new(),
                spec_types: vec![],
                server_path: String::new(),
            })),
        );
        (sup, root)
    }

    /// GPU-resident fabricated instance carrying a settle-measured
    /// footprint — mirrors `fake_instance` but with real weights and a
    /// measured card delta.
    fn gpu_instance(key: &str, weights_bytes: i64, settled_mib: u64) -> (Arc<Instance>, u32) {
        let proc = dummy_process();
        let pid = proc.id().expect("fabricated child pid");
        let endpoint = Endpoint::Tcp {
            host: "127.0.0.1".into(),
            port: 0,
        };
        let inst = Instance {
            name: key.to_string(),
            endpoint: endpoint.clone(),
            state: std::sync::RwLock::new(InstanceState::Ready),
            last_used: std::sync::RwLock::new(Instant::now()),
            in_flight: AtomicI64::new(0),
            keep_until: std::sync::RwLock::new(None),
            started_at: Instant::now(),
            argv: vec![],
            projector: false,
            model: ModelRow {
                name: key.to_string(),
                repo: String::new(),
                quant: String::new(),
                path: String::new(),
                bytes: weights_bytes,
                sha256: None,
                mmproj_path: None,
                shards: 1,
                arch: None,
                params: None,
                ctx_train: None,
                pulled_at: 0,
            },
            profile_ctx: 8,
            gpu: "full".into(),
            device: None,
            kv_est_bytes: None,
            warnings: Vec::new(),
            settled_mib: std::sync::atomic::AtomicU64::new(settled_mib),
            auth: None,
            child: tokio::sync::Mutex::new(ChildHandle::new(endpoint, proc)),
            pid,
        };
        (Arc::new(inst), pid)
    }

    #[tokio::test]
    async fn unit__ps_row__carries_instance_device() {
        let (sup, _root) = gpu_sup();
        let (mut inst, _pid) = gpu_instance("dev-model", 1000, 0);
        // Auto-pick records the card; ps() must surface it so `full@card`
        // renders in the CLI GPU column.
        if let Some(i) = Arc::get_mut(&mut inst) {
            i.device = Some("RTX 4070".into());
        }
        sup.instances.insert("dev-model".to_string(), inst);
        let rows = sup.ps();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].gpu, "full");
        assert_eq!(rows[0].device.as_deref(), Some("RTX 4070"));
    }

    #[tokio::test]
    #[allow(non_snake_case)]
    async fn unit__bytes_admission__measured_resident_refuses_oversubscription() {
        // 2026-09-11 crash shape: the 9B VL model settled at 7302 MiB
        // measured (5800 weights + context/compute/KV working set the
        // weights sum never sees). Weights-only arithmetic admitted the
        // 0.5B sibling (6300 <= 8188 budget) and the card died in an
        // NVRM NO_MEMORY storm; measured-resident + floor-charged
        // admission must refuse that exact pair.
        let mib = |m: u64| m * 1024 * 1024;
        let (sup, _root) = gpu_sup();
        let big_weights = i64::try_from(mib(5800)).expect("weights fit i64");
        let (big, _p1) = gpu_instance("big", big_weights, 7302);
        sup.instances.insert("big".to_string(), big);
        assert_eq!(
            sup.resident_bytes(),
            mib(7302),
            "measured settle delta must win over the weights sum"
        );
        // CPU-placed siblings never count against the VRAM budget.
        let (cpu_sibling, _p2) = fake_instance("cpu-model", InstanceState::Ready, 0);
        cpu_sibling
            .settled_mib
            .store(9999, std::sync::atomic::Ordering::Relaxed);
        sup.instances.insert("cpu-model".to_string(), cpu_sibling);
        assert_eq!(
            sup.resident_bytes(),
            mib(7302),
            "cpu-placed children are excluded from VRAM admission"
        );
        let small_floor = pallama_core::profile::admission_floor_bytes(mib(500), 0);
        assert_eq!(
            small_floor,
            mib(500 + 512 + 700),
            "floor = weights + KV floor + spawn overhead"
        );
        assert!(sup.bytes_admission_active());
        assert!(
            sup.resident_bytes().saturating_add(small_floor) > sup.vram_budget_bytes(),
            "the pair that crashed the 8 GiB card must be refused"
        );
        // And the refusal is NOT the old largest-model heuristic: the
        // weights-only sum would have admitted it (that was the bug).
        assert!(mib(5800) + mib(500) <= sup.vram_budget_bytes());
    }

    /// Fabricated map entry: real child (so teardown paths stay honest),
    /// throwaway argv/profile. Caller owns the pid for cleanup.
    fn fake_instance(key: &str, state: InstanceState, load: i64) -> (Arc<Instance>, u32) {
        let proc = dummy_process();
        let pid = proc.id().expect("fabricated child pid");
        let endpoint = Endpoint::Tcp {
            host: "127.0.0.1".into(),
            port: 0,
        };
        let inst = Instance {
            name: key.to_string(),
            endpoint: endpoint.clone(),
            state: std::sync::RwLock::new(state),
            last_used: std::sync::RwLock::new(Instant::now()),
            in_flight: AtomicI64::new(load),
            keep_until: std::sync::RwLock::new(None),
            started_at: Instant::now(),
            argv: vec![],
            projector: false,
            model: ModelRow {
                name: model_of_key(key).to_string(),
                repo: String::new(),
                quant: String::new(),
                path: String::new(),
                bytes: 1,
                sha256: None,
                mmproj_path: None,
                shards: 1,
                arch: None,
                params: None,
                ctx_train: None,
                pulled_at: 0,
            },
            profile_ctx: 8,
            gpu: "cpu".into(),
            device: None,
            kv_est_bytes: None,
            warnings: Vec::new(),
            settled_mib: std::sync::atomic::AtomicU64::new(0),
            auth: None,
            child: tokio::sync::Mutex::new(ChildHandle::new(endpoint, proc)),
            pid,
        };
        (Arc::new(inst), pid)
    }

    fn dummy_process() -> tokio::process::Child {
        #[cfg(not(windows))]
        return tokio::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn sleep");
        #[cfg(windows)]
        return tokio::process::Command::new("cmd")
            .args(["/C", "pause"])
            .spawn()
            .expect("spawn cmd");
    }

    /// Fabricated children are real OS processes: reap them so the test
    /// binary never leaks sleepers.
    #[allow(unsafe_code)] // audited libc::kill on fabricated test-child pids
    fn kill_all(pids: &[u32]) {
        for &pid in pids {
            #[cfg(unix)]
            // SAFETY: test-fabricated child pids only; SIGKILL, no group.
            unsafe {
                libc::kill(i32::try_from(pid).unwrap_or(-1), libc::SIGKILL);
            }
            #[cfg(not(unix))]
            {
                let _ = pid;
            }
        }
    }

    #[tokio::test]
    async fn unit__evict__respawn_during_grace_survives() {
        // F1 pin: a fresh instance inserted under the same name while an
        // evict waits in the child mutex (terminate grace) must survive
        // the evict's cleanup — pre-fix, instances.remove(name) deleted
        // it and the pidfile/apikey went with it.
        let sup = std::sync::Arc::new(routing_sup(1));
        let (old, po) = fake_instance("m", InstanceState::Ready, 0);
        sup.instances.insert("m".to_string(), old.clone());
        std::fs::create_dir_all(sup.dirs.run_dir()).unwrap();
        let pidfile = sup.dirs.run_dir().join("m.pid");
        std::fs::write(&pidfile, b"1").unwrap();
        // Park the evict mid-teardown by holding the child lock.
        let held = old.child.lock().await;
        let task = tokio::spawn({
            let sup = std::sync::Arc::clone(&sup);
            async move { sup.evict("m").await }
        });
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        // Simulate the respawn that won the TOCTOU between the evict's
        // map snapshot and its cleanup (direct insert = a spawn that
        // started before the evicting mark was set).
        let (fresh, pf) = fake_instance("m", InstanceState::Ready, 0);
        sup.instances.insert("m".to_string(), fresh.clone());
        drop(held);
        task.await.unwrap().unwrap();
        let live = sup
            .instances
            .get("m")
            .expect("fresh instance survived")
            .clone();
        assert!(
            std::sync::Arc::ptr_eq(&live, &fresh),
            "evict must not delete a respawned generation"
        );
        assert!(
            pidfile.exists(),
            "pidfile of the fresh generation must not be shredded"
        );
        assert!(
            sup.evicting.lock().await.is_empty(),
            "evicting mark cleared on completion"
        );
        drop(live);
        sup.instances.remove("m");
        kill_all(&[po, pf]);
    }

    #[tokio::test]
    async fn unit__model_of_key__plain_and_replica() {
        assert_eq!(model_of_key("m"), "m");
        assert_eq!(model_of_key("m#3"), "m");
        assert_eq!(model_of_key("m#3#4"), "m"); // first '#' wins
                                                // @vision suffix (lazy projector respawn key) must never leak
                                                // into model resolution — evict_model-style filters stay correct
        assert_eq!(model_of_key("m@vision"), "m");
        assert_eq!(model_of_key("m#1@vision"), "m");
        assert_eq!(model_of_key("m@vision#1"), "m"); // '@' stripped first
    }

    #[tokio::test]
    async fn unit__split_replica__parse_and_reject() {
        assert_eq!(split_replica("m"), None);
        assert_eq!(split_replica("m#2"), Some(("m", 2)));
        assert_eq!(split_replica("m#0"), Some(("m", 0)));
        assert_eq!(split_replica("m#x"), None); // non-numeric suffix
        assert_eq!(split_replica("m#"), None); // empty suffix
                                               // vision keys keep their replica index underneath the suffix
        assert_eq!(split_replica("m#1@vision"), Some(("m", 1)));
        assert_eq!(split_replica("m@vision"), None); // no replica part
    }

    #[tokio::test]
    async fn unit__replica_key__legacy_no_overlay__plain_name() {
        let sup = routing_sup(1);
        assert_eq!(sup.replica_key("m", None), "m");
        assert_eq!(
            sup.replica_key("m", Some(PrefixKey { sys: 7, convo: 7 })),
            "m"
        );
    }

    #[test]
    fn unit__resolve_draft_path__untyped_typed_and_none() {
        // R2-2/R2-3: ONE resolver for serve, router and bench. The
        // untyped auto pair resolves the draft-simple sibling row; the
        // typed eagle3 mode resolves the speculator row; modes that
        // never use an external draft return None without a store hit.
        let (_, root, _) = j2_sup(0);
        let dirs = PallamaDirs {
            config_dir: root.path().join("cfg2"),
            data_dir: root.path().join("data2"),
        };
        let store = Store::open(&dirs).unwrap();
        // Before any pull: family pair resolves but no row exists, and
        // no-pair / self-drafting / off modes never hit the store.
        assert_eq!(resolve_draft_path(&store, "qwen3-14b", "auto"), None);
        assert_eq!(resolve_draft_path(&store, "gemma3-4b", "auto"), None);
        assert_eq!(resolve_draft_path(&store, "qwen3-8b", "off"), None);
        assert_eq!(resolve_draft_path(&store, "qwen3-8b", "mtp"), None);
        assert_eq!(resolve_draft_path(&store, "qwen3-8b", "ngram"), None);
        let row = |name: &str, path: &str| pallama_core::ModelRow {
            name: name.into(),
            repo: name.into(),
            quant: "Q4_0".into(),
            path: path.into(),
            bytes: 1,
            sha256: None,
            mmproj_path: None,
            shards: 1,
            arch: None,
            params: None,
            ctx_train: None,
            pulled_at: 0,
        };
        store
            .upsert_model(&row("qwen3-0.6b", "/models/qwen3-0.6b-q4_0.gguf"))
            .unwrap();
        store
            .upsert_model(&row(
                "qwen3-8b-eagle3-speculator",
                "/models/qwen3-8b-eagle3-f16.gguf",
            ))
            .unwrap();

        // auto: generic draft-simple pair (catalog precedence).
        assert_eq!(
            resolve_draft_path(&store, "qwen3-8b", "auto").as_deref(),
            Some("/models/qwen3-0.6b-q4_0.gguf")
        );
        // eagle3: the TYPED speculator head, not the generic sibling.
        assert_eq!(
            resolve_draft_path(&store, "qwen3-8b", "eagle3").as_deref(),
            Some("/models/qwen3-8b-eagle3-f16.gguf")
        );
        // Pulled rows reach every family member via the shared prefix.
        assert_eq!(
            resolve_draft_path(&store, "qwen3-14b", "auto").as_deref(),
            Some("/models/qwen3-0.6b-q4_0.gguf")
        );
    }

    #[tokio::test]
    async fn unit__replica_key__router_key__passthrough() {
        let sup = routing_sup(4);
        assert_eq!(
            sup.replica_key(ROUTER_KEY, Some(PrefixKey { sys: 7, convo: 7 })),
            ROUTER_KEY
        );
    }

    #[tokio::test]
    async fn unit__replica_key__cold_start__grows_first_replica() {
        let sup = routing_sup(2);
        assert_eq!(sup.replica_key("m", None), "m#1");
    }

    #[tokio::test]
    async fn unit__replica_key__cold_concurrent__grows_second() {
        let sup = routing_sup(2);
        let (a, pa) = fake_instance("m#1", InstanceState::Loading, 0);
        sup.instances.insert("m#1".into(), a);
        // No live replica yet (#1 loading): scale out.
        assert_eq!(sup.replica_key("m", None), "m#2");
        kill_all(&[pa]);
    }

    #[tokio::test]
    async fn unit__replica_key__idle_live__reused_not_scaled() {
        let sup = routing_sup(2);
        let (a, pa) = fake_instance("m#1", InstanceState::Ready, 0);
        sup.instances.insert("m#1".into(), a);
        assert_eq!(sup.replica_key("m", None), "m#1");
        kill_all(&[pa]);
    }

    #[tokio::test]
    async fn unit__replica_key__busy_live__scales_out() {
        let sup = routing_sup(2);
        let (a, pa) = fake_instance("m#1", InstanceState::Ready, 3);
        sup.instances.insert("m#1".into(), a);
        assert_eq!(sup.replica_key("m", None), "m#2");
        kill_all(&[pa]);
    }

    #[tokio::test]
    async fn unit__victim_key__pinned_never_chosen() {
        // "hot" is pinned (overlay), "cold" is not: pressure falls on
        // "cold" even though it is the hotter cache.
        let mut sup = routing_sup(1);
        sup.config.model_overrides.insert(
            "hot".into(),
            ModelOverride {
                pin: Some(true),
                ..Default::default()
            },
        );
        let (hot, ph) = fake_instance("hot", InstanceState::Ready, 0);
        let (cold, pc) = fake_instance("cold", InstanceState::Ready, 0);
        sup.instances.insert("hot".into(), hot);
        sup.instances.insert("cold".into(), cold);
        sup.note_prefix_hit("hot");
        sup.note_prefix_hit("hot");
        assert_eq!(sup.victim_key("incoming"), Some("cold".into()));
        kill_all(&[ph, pc]);
    }

    #[tokio::test]
    async fn unit__victim_key__all_pinned__none() {
        let mut sup = routing_sup(1);
        sup.config.model_overrides.insert(
            "hot".into(),
            ModelOverride {
                pin: Some(true),
                ..Default::default()
            },
        );
        let (hot, ph) = fake_instance("hot", InstanceState::Ready, 0);
        sup.instances.insert("hot".into(), hot);
        assert_eq!(sup.victim_key("incoming"), None);
        kill_all(&[ph]);
    }

    #[test]
    fn unit__note_transition__edges_and_self_loops() {
        let sup = routing_sup(1);
        sup.note_transition("a");
        assert!(sup.transitions.is_empty(), "first request has no edge");
        sup.note_transition("b");
        sup.note_transition("b");
        let edge0 = ("a".to_string(), "b".to_string());
        assert_eq!(sup.transitions.get(&edge0).map(|v| *v), Some(1));
        sup.note_transition("a"); // b -> a
        sup.note_transition("b"); // a -> b again
        let edge = ("a".to_string(), "b".to_string());
        assert_eq!(sup.transitions.get(&edge).map(|v| *v), Some(2));
        // Self-loops never create edges.
        sup.note_transition("b");
        sup.note_transition("b");
        assert!(!sup.transitions.contains_key(&("b".into(), "b".into())));
    }

    #[tokio::test]
    async fn unit__adaptive_slots__adopts_after_streak_and_respects_manual() {
        let mut sup = routing_sup(1);
        sup.config.adaptive_slots = true;
        // LC4 bumps SINGLE-slot models; the slots=0 auto default makes it
        // intentionally inert (spawn-time sizing owns that case).
        sup.config.slots = 1;
        let (inst, ph) = fake_instance("m", InstanceState::Ready, 2);
        sup.instances.insert("m".into(), inst);
        for _ in 0..(SLOTS_STREAK_TICKS - 1) {
            sup.adaptive_slots_tick();
        }
        assert!(
            sup.adopted_slots.get("m").is_none(),
            "one tick short of the threshold"
        );
        sup.adaptive_slots_tick();
        assert_eq!(sup.adopted_slots.get("m").map(|v| *v), Some(2));
        kill_all(&[ph]);

        // Manual overlay slots exclude the model entirely.
        let mut sup = routing_sup(1);
        sup.config.adaptive_slots = true;
        sup.config.slots = 1;
        sup.config.model_overrides.insert(
            "m".into(),
            ModelOverride {
                slots: Some(1),
                ..Default::default()
            },
        );
        let (inst, ph) = fake_instance("m", InstanceState::Ready, 2);
        sup.instances.insert("m".into(), inst);
        for _ in 0..(SLOTS_STREAK_TICKS * 2) {
            sup.adaptive_slots_tick();
        }
        assert!(sup.adopted_slots.get("m").is_none());
        kill_all(&[ph]);
    }

    #[tokio::test]
    async fn unit__adaptive_slots__off_by_default_is_noop() {
        let sup = routing_sup(1); // adaptive_slots defaults false
        let (inst, ph) = fake_instance("m", InstanceState::Ready, 2);
        sup.instances.insert("m".into(), inst);
        for _ in 0..(SLOTS_STREAK_TICKS * 2) {
            sup.adaptive_slots_tick();
        }
        assert!(sup.adopted_slots.is_empty());
        kill_all(&[ph]);
    }

    /// [`slot_cap`] resolution order: adopted > live argv `-np` > explicit
    /// config > modest auto default. A stale hardcoded auto cap would
    /// silently defeat an adopted reshape (live-caught 2026-09-12).
    #[tokio::test]
    async fn unit__slot_cap__adopted_argv_config_default_resolution() {
        let sup = routing_sup(1);
        // No instance, config slots=0 (auto): modest default.
        assert_eq!(sup.slot_cap("m"), 4);
        // Live child argv is the source of truth once spawned.
        let (mut inst, ph) = fake_instance("m", InstanceState::Ready, 0);
        Arc::get_mut(&mut inst).expect("sole owner").argv =
            vec!["llama-server".into(), "-np".into(), "4".into()];
        sup.instances.insert("m".to_string(), inst);
        assert_eq!(sup.slot_cap("m"), 4);
        // Adoption wins over the argv shape (the respawn carries it).
        sup.adopted_slots.insert("m".to_string(), 6);
        assert_eq!(sup.slot_cap("m"), 6);
        // Replica keys of the same model resolve identically.
        assert_eq!(sup.slot_cap("m#1"), 6);
        kill_all(&[ph]);
    }

    /// Decay rule: an adopted shape (np6 over a natural np4) survives
    /// 29 fully-quiet ticks untouched, then the 30th removes the
    /// adoption and queues the reshape back to the natural shape.
    /// Saturation mid-window resets the decay count.
    #[tokio::test]
    async fn unit__adaptive_slots__quiet_decay_restores_natural_shape() {
        let mut sup = routing_sup(1);
        sup.config.adaptive_slots = true;
        sup.adopted_slots.insert("m".to_string(), 6);
        let (mut inst, ph) = fake_instance("m", InstanceState::Ready, 0);
        Arc::get_mut(&mut inst).expect("sole owner").argv =
            vec!["llama-server".into(), "-np".into(), "4".into()];
        sup.instances.insert("m".to_string(), inst);
        for _ in 0..(SLOTS_DECAY_TICKS - 1) {
            sup.adaptive_slots_tick();
        }
        assert!(sup.adopted_slots.get("m").is_some());
        assert!(sup.reshape_queue.get("m").is_none());
        // One more quiet tick crosses the threshold.
        sup.adaptive_slots_tick();
        assert!(sup.adopted_slots.get("m").is_none());
        assert!(sup.reshape_queue.get("m").is_some());
        // Mid-window saturation resets the count: re-adopt, then verify
        // a short quiet stretch does NOT decay again.
        sup.reshape_queue.remove("m");
        sup.adopted_slots.insert("m".to_string(), 6);
        sup.note_slot_pressure("m");
        sup.adaptive_slots_tick();
        sup.slot_pressure.remove("m");
        for _ in 0..10 {
            sup.adaptive_slots_tick();
        }
        assert!(sup.adopted_slots.get("m").is_some());
        kill_all(&[ph]);
    }

    /// 2026-09-12 live-proven: admission caps engine in-flight at the
    /// slot count, so queue-wait pressure (gateway `AllSlotsBusy` reports)
    /// is the reachable saturation signal. The pressure is a GAUGE: two
    /// parked requests stay visible on every tick (no per-tick drain),
    /// six saturated ticks adopt even though `in-flight` never exceeds
    /// the shape, and releasing the gauge resets the streak.
    #[tokio::test]
    async fn unit__adaptive_slots__admission_pressure_adopts_without_inflight_excess() {
        let mut sup = routing_sup(1);
        sup.config.adaptive_slots = true;
        sup.config.slots = 0;
        let (mut inst, ph) = fake_instance("m", InstanceState::Ready, 4);
        Arc::get_mut(&mut inst).expect("sole owner").argv =
            vec!["llama-server".into(), "-np".into(), "4".into()];
        sup.instances.insert("m".into(), inst);
        // two requests enter the admission queue and STAY parked
        sup.note_slot_pressure("m");
        sup.note_slot_pressure("m");
        for _ in 0..SLOTS_STREAK_TICKS {
            sup.adaptive_slots_tick();
        }
        assert_eq!(sup.adopted_slots.get("m").map(|v| *v), Some(5));
        assert_eq!(
            sup.reshape_queue.get("m").map(|v| v.value().clone()),
            Some("m".to_string())
        );
        // gauge release: the queued demand leaving must clear the signal
        sup.note_slot_pressure_release("m");
        sup.note_slot_pressure_release("m");
        assert!(sup.slot_pressure.get("m").is_none());
        kill_all(&[ph]);
    }

    /// 2026-09-12 generalization: capacity-shaped auto spawns (child
    /// argv carries `-np 4`, config slots = 0) must earn bumps too —
    /// the old `effective == 1` guard left the default shape inert.
    /// Saturation at in-flight 5 > 4 slots adopts 5 and queues the
    /// reshape respawn.
    #[tokio::test]
    async fn unit__adaptive_slots__auto_shape_saturation_adopts_and_queues() {
        let mut sup = routing_sup(1);
        sup.config.adaptive_slots = true;
        sup.config.slots = 0; // auto: the argv is the source of truth
        let (mut inst, ph) = fake_instance("m", InstanceState::Ready, 5);
        Arc::get_mut(&mut inst).expect("sole owner").argv = vec![
            "llama-server".into(),
            "-np".into(),
            "4".into(),
            "--ctx-size".into(),
            "65536".into(),
        ];
        sup.instances.insert("m".into(), inst);
        for _ in 0..SLOTS_STREAK_TICKS {
            sup.adaptive_slots_tick();
        }
        assert_eq!(sup.adopted_slots.get("m").map(|v| *v), Some(5));
        assert_eq!(
            sup.reshape_queue.get("m").map(|v| v.value().clone()),
            Some("m".to_string())
        );
        kill_all(&[ph]);
    }

    /// At the adoption cap the saturation streak must not re-adopt
    /// (to <= from skips) — no event spam, no queue churn.
    #[tokio::test]
    async fn unit__adaptive_slots__at_cap_does_not_adopt() {
        let mut sup = routing_sup(1);
        sup.config.adaptive_slots = true;
        sup.config.slots = 0;
        let (mut inst, ph) = fake_instance("m", InstanceState::Ready, 9);
        Arc::get_mut(&mut inst).expect("sole owner").argv =
            vec!["llama-server".into(), "-np".into(), "8".into()];
        sup.instances.insert("m".into(), inst);
        for _ in 0..(SLOTS_STREAK_TICKS * 2) {
            sup.adaptive_slots_tick();
        }
        assert!(sup.adopted_slots.get("m").is_none());
        assert!(sup.reshape_queue.is_empty());
        kill_all(&[ph]);
    }

    /// `deterministic = true` models are excluded: their spawns pin
    /// slots = 1 by design and adoption would fight the pin every tick.
    #[tokio::test]
    async fn unit__adaptive_slots__deterministic_excluded() {
        let mut sup = routing_sup(1);
        sup.config.adaptive_slots = true;
        sup.config.deterministic = true;
        let (inst, ph) = fake_instance("m", InstanceState::Ready, 4);
        sup.instances.insert("m".into(), inst);
        for _ in 0..(SLOTS_STREAK_TICKS * 2) {
            sup.adaptive_slots_tick();
        }
        assert!(sup.adopted_slots.is_empty());
        kill_all(&[ph]);
    }

    /// Rollback: a spawn failure after an adoption drops the bump (the
    /// capacity answer) instead of crash-looping the adopted shape.
    #[tokio::test]
    async fn unit__adaptive_slots__spawn_failure_rolls_back_adoption() {
        let sup = routing_sup(1);
        sup.adopted_slots.insert("m".into(), 6);
        sup.reshape_queue.insert("m".into(), "m".into());
        sup.note_engine_failure("m");
        assert!(sup.adopted_slots.get("m").is_none());
        assert!(sup.reshape_queue.get("m").is_none());
    }

    /// The 2026-09-08 runtime-freeze pin: `reap_dead_children` must not
    /// hold a `DashMap` iteration across `child.lock().await`. The old
    /// loop parked a shard read-guard on the child mutex; with the mutex
    /// held by a concurrent evict (slow terminate), every write to that
    /// shard parked its worker thread in a sync futex — accept, /health
    /// and the SIGTERM drain all starved. Regression shape: while the
    /// reaper awaits a held child lock, instance-map writes must still
    /// go through.
    #[tokio::test]
    async fn unit__reap_dead_children__does_not_park_shard_guard_across_child_lock() {
        let sup = std::sync::Arc::new(routing_sup(1));
        let (inst, pid) = fake_instance("guard-probe", InstanceState::Ready, 0);
        sup.instances
            .insert("guard-probe".to_string(), inst.clone());

        // Simulate an evict inside terminate_group: hold the child mutex
        // for a while.
        let holder = tokio::spawn({
            let inst = inst.clone();
            async move {
                let _guard = inst.child.lock().await;
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        });
        tokio::time::sleep(Duration::from_millis(100)).await; // holder has the lock

        let reaper = tokio::spawn({
            let sup = std::sync::Arc::clone(&sup);
            async move { sup.reap_dead_children().await }
        });
        // Parked on the child lock by now; instance-map writes (spawn
        // inserts) must NOT block on a snapshot-based reap.
        tokio::time::sleep(Duration::from_millis(150)).await;
        let inserts = tokio::task::spawn_blocking({
            let sup = std::sync::Arc::clone(&sup);
            let inst = inst.clone();
            move || {
                // 256 keys blanket every shard of the map.
                for i in 0..256 {
                    sup.instances.insert(format!("k{i}"), inst.clone());
                }
            }
        });
        let inserted = tokio::time::timeout(Duration::from_secs(1), inserts).await;
        assert!(
            inserted.is_ok(),
            "instance-map writes stalled while reap awaited the child lock"
        );

        let _ = tokio::time::timeout(Duration::from_secs(4), reaper).await;
        let _ = tokio::time::timeout(Duration::from_secs(4), holder).await;
        kill_all(&[pid]);
    }

    #[tokio::test]
    async fn unit__replica_key__at_capacity__joins_least_loaded() {
        let sup = routing_sup(2);
        let (a, pa) = fake_instance("m#1", InstanceState::Ready, 5);
        let (b, pb) = fake_instance("m#2", InstanceState::Ready, 2);
        sup.instances.insert("m#1".into(), a);
        sup.instances.insert("m#2".into(), b);
        assert_eq!(sup.replica_key("m", None), "m#2");
        kill_all(&[pa, pb]);
    }

    #[tokio::test]
    async fn unit__replica_key__clamp__never_beyond_max() {
        // Overlay asks for 20; routing must clamp to MAX_REPLICAS (8)
        // and never hand out #9.
        let sup = routing_sup(20);
        let mut pids = Vec::new();
        for idx in 1..=MAX_REPLICAS {
            let (a, p) = fake_instance(&format!("m#{idx}"), InstanceState::Ready, 1);
            sup.instances.insert(format!("m#{idx}"), a);
            pids.push(p);
        }
        let picked = sup.replica_key("m", None);
        let (_, idx) = split_replica(&picked).expect("replica key");
        assert!(idx <= MAX_REPLICAS, "picked beyond clamp: {picked}");
        kill_all(&pids);
    }

    #[tokio::test]
    async fn unit__replica_key__affinity_hit__sticky_even_when_idle_other() {
        let sup = routing_sup(2);
        let (a, pa) = fake_instance("m#1", InstanceState::Ready, 0);
        let (b, pb) = fake_instance("m#2", InstanceState::Ready, 0);
        sup.instances.insert("m#1".into(), a);
        sup.instances.insert("m#2".into(), b);
        sup.prefix_affinity.insert(42, "m#2".to_string());
        assert_eq!(
            sup.replica_key("m", Some(PrefixKey { sys: 42, convo: 42 })),
            "m#2"
        );
        kill_all(&[pa, pb]);
    }

    #[tokio::test]
    async fn unit__replica_key__affinity_stale__grows_new_replica() {
        let sup = routing_sup(2);
        let (a, pa) = fake_instance("m#1", InstanceState::Ready, 0);
        sup.instances.insert("m#1".into(), a);
        // Affinity points at an evicted key: the conversation is new to
        // every live replica, so it grows its own cache instead of
        // squatting on m#1's.
        sup.prefix_affinity.insert(42, "m#9".to_string());
        assert_eq!(
            sup.replica_key("m", Some(PrefixKey { sys: 43, convo: 42 })),
            "m#2"
        );
        kill_all(&[pa]);
    }

    #[tokio::test]
    async fn unit__replica_key__same_sys_class__coalesces_onto_warm_replica() {
        let sup = routing_sup(3);
        let (a, pa) = fake_instance("m#1", InstanceState::Ready, 0);
        let (b, pb) = fake_instance("m#2", InstanceState::Ready, 0);
        sup.instances.insert("m#1".into(), a);
        sup.instances.insert("m#2".into(), b);
        // m#1 recently served sys-class 100: a NEW conversation from the
        // same system prompt reuses its warm system-prompt KV instead of
        // growing a cold m#3 (F8 coalesce).
        sup.sys_rings
            .entry("m#1".to_string())
            .or_default()
            .push_back(100);
        let picked = sup.replica_key(
            "m",
            Some(PrefixKey {
                sys: 100,
                convo: 999,
            }),
        );
        assert_eq!(picked, "m#1");
        kill_all(&[pa, pb]);
    }

    #[tokio::test]
    async fn unit__replica_key__unknown_sys_class__grows_per_policy() {
        let sup = routing_sup(3);
        let (a, pa) = fake_instance("m#1", InstanceState::Ready, 0);
        let (b, pb) = fake_instance("m#2", InstanceState::Ready, 0);
        sup.instances.insert("m#1".into(), a);
        sup.instances.insert("m#2".into(), b);
        sup.sys_rings
            .entry("m#1".to_string())
            .or_default()
            .push_back(100);
        // Neither replica has seen sys-class 777: pinned conversation with
        // no ring hit grows its own replica (prefix-aware grow policy).
        let picked = sup.replica_key(
            "m",
            Some(PrefixKey {
                sys: 777,
                convo: 888,
            }),
        );
        assert_eq!(picked, "m#3");
        kill_all(&[pa, pb]);
    }

    #[tokio::test]
    async fn unit__replica_key__other_model_instances__ignored() {
        let sup = routing_sup(2);
        let (other, po) = fake_instance("other#1", InstanceState::Ready, 0);
        sup.instances.insert("other#1".into(), other);
        assert_eq!(sup.replica_key("m", None), "m#1");
        kill_all(&[po]);
    }

    #[tokio::test]
    async fn unit__replica_key__all_loading__joins_least_loaded_queue() {
        let sup = routing_sup(2);
        let (a, pa) = fake_instance("m#1", InstanceState::Loading, 2);
        let (b, pb) = fake_instance("m#2", InstanceState::Loading, 0);
        sup.instances.insert("m#1".into(), a);
        sup.instances.insert("m#2".into(), b);
        assert_eq!(sup.replica_key("m", None), "m#2");
        kill_all(&[pa, pb]);
    }

    // ---- LC2 discrete-first auto pick + card-scoped pressure ----------

    const MIB: u64 = 1024 * 1024;

    fn gpu(name: &str, desc: &str, total_mib: u64, free_mib: u64) -> pallama_core::GpuInfo {
        pallama_core::GpuInfo {
            name: name.to_string(),
            description: desc.to_string(),
            total_mib,
            free_mib,
        }
    }

    #[test]
    fn unit__pick_gpu__discrete_preferred_over_higher_free_integrated() {
        let igpu = gpu("GPU1", "Intel(R) Graphics (RPL-S)", 16_384, 10_000);
        let dgpu = gpu("GPU0", "NVIDIA GeForce RTX 4070", 8_188, 4_000);
        let (idx, skipped) = pick_gpu(&[igpu, dgpu]).expect("a pick exists");
        assert_eq!(idx, 1, "discrete card wins despite less free VRAM");
        assert!(skipped, "the higher-free integrated card was skipped");
    }

    #[test]
    fn unit__pick_gpu__integrated_only_box_falls_back_to_overall_max() {
        let a = gpu("a", "Intel(R) Iris Xe Graphics", 8_192, 1_000);
        let b = gpu("b", "AMD Radeon(TM) Graphics", 8_192, 2_000);
        let (idx, skipped) = pick_gpu(&[a, b]).expect("a pick exists");
        assert_eq!(idx, 1, "fallback = overall max-free");
        assert!(!skipped);
    }

    #[test]
    fn unit__pick_gpu__empty_gpu_list_is_none() {
        assert!(pick_gpu(&[]).is_none());
    }

    #[test]
    fn unit__card_pressure__same_card_counts_other_card_does_not() {
        let gpus = [
            gpu("ig", "Intel(R) Graphics (RPL-S)", 16_384, 8_000),
            gpu("dg", "NVIDIA GeForce RTX 4070", 8_188, 4_000),
        ];
        let residents = vec![(Some("dg".to_string()), 4_000 * MIB, 0, true)];
        // 4000 resident + 3500 weights + 500 KV = 8000 MiB > 95% of 8188
        let same = card_pressure_exceeds(Some("dg"), &gpus, &residents, 3_500 * MIB, 500 * MIB);
        assert_eq!(same, Some(true));
        // Same residents pinned to the OTHER card: nothing on "dg" → fits.
        let others = vec![(Some("ig".to_string()), 4_000 * MIB, 0, true)];
        let free = card_pressure_exceeds(Some("dg"), &gpus, &others, 3_500 * MIB, 500 * MIB);
        assert_eq!(free, Some(false));
    }

    #[test]
    fn unit__card_pressure__unknown_placement_counts_conservatively() {
        let gpus = [gpu("dg", "NVIDIA GeForce RTX 4070", 8_188, 4_000)];
        let residents = vec![(None, 4_000 * MIB, 0, true)];
        let press = card_pressure_exceeds(Some("dg"), &gpus, &residents, 3_500 * MIB, 500 * MIB);
        assert_eq!(press, Some(true), "unknown placement must count");
    }

    #[test]
    fn unit__card_pressure__ninety_five_percent_boundary_is_exclusive() {
        let gpus = [gpu("dg", "NVIDIA GeForce RTX 4070", 1_000, 1_000)];
        // Exactly 95% of a 1000 MiB card: NOT over.
        let at = card_pressure_exceeds(Some("dg"), &gpus, &[], 900 * MIB, 50 * MIB);
        assert_eq!(at, Some(false));
        // One byte over: over.
        let over = card_pressure_exceeds(Some("dg"), &gpus, &[], 900 * MIB, 50 * MIB + 1);
        assert_eq!(over, Some(true));
    }

    // ---- #28c auto tensor-split planning --------------------------------

    fn two_discrete() -> Vec<pallama_core::GpuInfo> {
        vec![
            gpu("A", "NVIDIA GeForce RTX 4090", 24_000, 16_000),
            gpu("B", "NVIDIA GeForce RTX 4070", 8_188, 4_000),
        ]
    }

    #[test]
    fn unit__auto_split__fires_when_over_best_but_fits_combined() {
        // 17_000 MiB need > best card's 16_000 free, <= 20_000 combined.
        let ratios = plan_auto_tensor_split(&two_discrete(), 16_000, 1_000);
        assert_eq!(ratios.as_deref(), Some("4,1"), "proportional to free VRAM");
    }

    #[test]
    fn unit__auto_split__none_when_fits_best_card() {
        assert_eq!(plan_auto_tensor_split(&two_discrete(), 15_000, 500), None);
    }

    #[test]
    fn unit__auto_split__none_when_beyond_combined() {
        assert_eq!(plan_auto_tensor_split(&two_discrete(), 20_000, 1_000), None);
    }

    #[test]
    fn unit__auto_split__needs_two_discrete_cards() {
        let one = vec![gpu("A", "NVIDIA GeForce RTX 4090", 24_000, 16_000)];
        assert_eq!(plan_auto_tensor_split(&one, 20_000, 1_000), None);
        assert_eq!(plan_auto_tensor_split(&[], 20_000, 1_000), None);
    }

    #[test]
    fn unit__auto_split__integrated_cards_never_join_the_pool() {
        // iGPU reports huge shared-RAM free; it must not widen the pool
        // nor appear in the ratio string.
        let mut gpus = two_discrete();
        gpus.push(gpu("ig", "Intel(R) Graphics (RPL-S)", 16_384, 12_000));
        let ratios = plan_auto_tensor_split(&gpus, 16_000, 1_000);
        assert_eq!(ratios.as_deref(), Some("4,1"), "discrete-only ratios");
    }

    #[test]
    fn unit__auto_split__ratio_clamps_to_one_for_nearly_full_card() {
        let gpus = vec![
            gpu("A", "NVIDIA GeForce RTX 4090", 24_000, 16_000),
            gpu("B", "NVIDIA GeForce RTX 4070", 8_188, 100),
        ];
        // need 16_050: over A's 16_000 free, within the 16_100 combined.
        // min_free = 100 → ratios 160:1 (B clamps to ≥1).
        let ratios = plan_auto_tensor_split(&gpus, 16_000, 50);
        assert_eq!(ratios.as_deref(), Some("160,1"));
    }

    // ---- #28a settle report ----------------------------------------------

    fn hw_of(gpus: Vec<pallama_core::GpuInfo>) -> Hardware {
        Hardware {
            physical_cores: 8,
            total_ram_mib: 32_000,
            gpus,
        }
    }

    #[test]
    fn unit__settle_report__picked_card_delta_and_used_pct() {
        let pre = hw_of(vec![gpu("dg", "NVIDIA GeForce RTX 4070", 8_188, 6_000)]);
        let post = hw_of(vec![gpu("dg", "NVIDIA GeForce RTX 4070", 8_188, 1_000)]);
        let (card, taken, used_pct) =
            settle_report(&pre, &post, Some("dg")).expect("card present both sides");
        assert_eq!(card, "dg");
        assert_eq!(taken, 5_000, "free delta = what the spawn took");
        assert_eq!(used_pct, (8_188 - 1_000) * 100 / 8_188);
    }

    #[test]
    fn unit__census_cache__ttl_hit_miss_and_freshness_window() {
        // The cache exists so a cold-start burst pays the ~0.25 s census
        // subprocess once, not per step; TTL expiry re-measures, and an
        // empty (invalidated) cache is always a miss.
        use std::sync::Mutex;
        let hw = hw_of(vec![gpu("dg", "NVIDIA GeForce RTX 4070", 8_188, 6_000)]);
        let ttl = Duration::from_secs(10);
        let cache: Mutex<Option<(Instant, Hardware)>> = Mutex::new(None);
        assert!(
            Supervisor::census_from_cache(&cache, ttl).is_none(),
            "empty cache = miss"
        );
        *cache.lock().unwrap() = Some((Instant::now(), hw.clone()));
        assert!(
            Supervisor::census_from_cache(&cache, ttl).is_some(),
            "fresh entry = hit"
        );
        *cache.lock().unwrap() = Some((
            Instant::now().checked_sub(Duration::from_secs(11)).unwrap(),
            hw,
        ));
        assert!(
            Supervisor::census_from_cache(&cache, ttl).is_none(),
            "expired entry = miss"
        );
    }

    #[test]
    fn unit__settle_report__no_pick_falls_back_to_max_free_card() {
        let pre = hw_of(vec![
            gpu("full", "NVIDIA A", 8_000, 7_000),
            gpu("idle", "NVIDIA B", 8_000, 8_000),
        ]);
        let post = hw_of(vec![
            gpu("full", "NVIDIA A", 8_000, 7_000),
            gpu("idle", "NVIDIA B", 8_000, 3_000),
        ]);
        let (card, taken, _) = settle_report(&pre, &post, None).expect("a card exists");
        assert_eq!(card, "idle", "max-free card of the AFTER snapshot");
        assert_eq!(taken, 5_000);
    }

    #[test]
    fn unit__settle_report__missing_card_is_none() {
        let pre = hw_of(vec![gpu("dg", "NVIDIA A", 8_000, 7_000)]);
        let post = hw_of(Vec::new());
        assert!(settle_report(&pre, &post, Some("dg")).is_none());
    }
}
