//! Shared gateway state.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use blazar_core::{BlazarDirs, Config};
use blazar_runtime::{EventBus, Supervisor};

use crate::histogram::Histogram;
use crate::keys::KeysLimiter;
use crate::otlp::Otlp;
use crate::queue::PriorityQueue;
use crate::remotes::RemoteHealth;
use crate::semcache::{SemMetrics, SemanticCache};
use crate::sentinel::Sentinel;

/// Gateway-observed prompt-cache classification (R6/R7-lite): per-response
/// usage counters + warm/cold TTFT split. `record` runs exactly once per
/// COMPLETED response; a response whose usage never surfaced (upstream
/// omission, aborted before the usage chunk) lands in `unclassified`
/// instead of guessing a side. Atomics only - lock-free on the hot path.
pub struct CacheObs {
    pub prompt_tokens: AtomicU64,
    pub cached_tokens: AtomicU64,
    /// Completed responses we could NOT classify warm/cold (no usage seen).
    pub unclassified: AtomicU64,
    pub ttft_warm: Histogram,
    pub ttft_cold: Histogram,
    /// Per-model split of the same token counters (keyed by the resolved
    /// model name): powers per-model cache-hit surfaces (`ps` HIT column,
    /// `/api/ps`, per-model metrics) without touching child scrapes. One
    /// entry per distinct model - bounded by the store, not by traffic.
    pub per_model: dashmap::DashMap<String, ModelCacheObs>,
}

/// Per-model prompt-cache counters; same lock-free shape as the global.
pub struct ModelCacheObs {
    pub prompt_tokens: AtomicU64,
    pub cached_tokens: AtomicU64,
}

impl Default for CacheObs {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheObs {
    #[must_use]
    pub fn new() -> Self {
        Self {
            prompt_tokens: AtomicU64::new(0),
            cached_tokens: AtomicU64::new(0),
            unclassified: AtomicU64::new(0),
            ttft_warm: crate::histogram::ttft_warm(),
            ttft_cold: crate::histogram::ttft_cold(),
            per_model: dashmap::DashMap::new(),
        }
    }

    /// Classify one completed response. `cached > 0` = warm. A missing
    /// TTFT with known usage still counts tokens but skips both
    /// histograms (the split stays honest).
    pub fn record(&self, model: &str, prompt: u64, cached: u64, ttft_secs: Option<f64>) {
        self.prompt_tokens.fetch_add(prompt, Ordering::Relaxed);
        self.cached_tokens.fetch_add(cached, Ordering::Relaxed);
        if let Some(e) = self.per_model.get(model) {
            e.prompt_tokens.fetch_add(prompt, Ordering::Relaxed);
            e.cached_tokens.fetch_add(cached, Ordering::Relaxed);
        } else {
            let e = self
                .per_model
                .entry(model.to_string())
                .or_insert_with(|| ModelCacheObs {
                    prompt_tokens: AtomicU64::new(0),
                    cached_tokens: AtomicU64::new(0),
                });
            // Loser of an insert race re-adds its tokens onto the
            // winner's entry - totals stay exact.
            e.prompt_tokens.fetch_add(prompt, Ordering::Relaxed);
            e.cached_tokens.fetch_add(cached, Ordering::Relaxed);
        }
        let warm = cached > 0;
        if let Some(t) = ttft_secs {
            if warm {
                self.ttft_warm.observe_secs(t);
            } else {
                self.ttft_cold.observe_secs(t);
            }
        }
    }

    /// Lifetime cache-hit ratio for one model: `None` until the gateway
    /// has observed at least one prompt token for it (the ps HIT column
    /// renders `-`, never a lying 0%).
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // integer counters -> ratio
    pub fn hit_ratio(&self, model: &str) -> Option<f64> {
        let e = self.per_model.get(model)?;
        let prompt = e.prompt_tokens.load(Ordering::Relaxed);
        let cached = e.cached_tokens.load(Ordering::Relaxed);
        (prompt > 0).then(|| cached as f64 / prompt as f64)
    }

    /// Render the per-model gateway-observed counters (sorted by model
    /// for stable scrapes). Distinct name-space from the scrape-derived
    /// `blazar_cache_hit_ratio`: that one is the engine-side truth,
    /// these are what completed responses actually reported.
    #[allow(clippy::cast_precision_loss)] // integer counters -> ratio
    pub fn render_per_model(&self, out: &mut String) {
        use std::fmt::Write as _;
        let mut rows: Vec<(String, u64, u64)> = self
            .per_model
            .iter()
            .map(|e| {
                (
                    e.key().clone(),
                    e.value().prompt_tokens.load(Ordering::Relaxed),
                    e.value().cached_tokens.load(Ordering::Relaxed),
                )
            })
            .collect();
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        let mut help_written = false;
        for (model, prompt, cached) in rows {
            if prompt == 0 {
                continue;
            }
            if !help_written {
                let _ = write!(
                    out,
                    "# HELP blazar_gateway_cache_hit_ratio Cache-hit ratio of prompt tokens observed at the gateway, per model (completed responses only; scrape-based blazar_cache_hit_ratio remains the engine-side truth)\n# TYPE blazar_gateway_cache_hit_ratio gauge\n"
                );
                help_written = true;
            }
            let ratio = cached as f64 / prompt as f64;
            let _ = writeln!(
                out,
                "blazar_gateway_cache_hit_ratio{{model=\"{model}\"}} {ratio:.4}"
            );
        }
    }

    /// A response completed without any usage information: counted as
    /// unclassified rather than silently dropped or forced cold.
    pub fn miss(&self) {
        self.unclassified.fetch_add(1, Ordering::Relaxed);
    }
}

pub struct AppState {
    pub dirs: BlazarDirs,
    pub sup: Arc<Supervisor>,
    pub bus: EventBus,
    /// One-use join codes for guided onboarding (`blazar adopt` /
    /// `blazar join`); self-initializing and ephemeral.
    pub invites: crate::fleet::InviteRegistry,
    pub queue: Arc<PriorityQueue>,
    /// Loopback client for child traffic + metrics scrape.
    pub http: reqwest::Client,
    /// Timeout-less client for media forwards: a sync image/video
    /// request holds the connection for the WHOLE render - minutes,
    /// past any total timeout. Liveness comes from the loopback
    /// socket itself (a dead child closes it immediately) plus the
    /// child header/evict lanes, not from a client deadline.
    pub media_http: reqwest::Client,
    /// Cached per-socket clients for unix-transport children
    /// (`child_transport = "unix"`): reqwest pins one socket path per
    /// client, so each unix child gets its own entry here. Empty and
    /// untouched in the default TCP mode.
    pub uds_http: blazar_runtime::uds::UdsClients,
    /// J5 eviction lane sender (sentinel body-stall fires through it):
    /// the consumer task in `new` debounces (1/min per model) and reaps
    /// the wedged child. Proxy header-stall evicts synchronously in-band
    /// instead - a debounce is too slow to gate an in-band retry.
    pub evict_tx: tokio::sync::mpsc::UnboundedSender<String>,
    /// vLLM-style evidence loop: measured at the proxy, owned by the gateway.
    pub ttft: Histogram,
    pub tpot: Histogram,
    /// Request stage timing (F5): per-request boundary stamps from the
    /// middleware chokepoints, aggregated here - splits gateway overhead
    /// (auth/body/admission/dispatch) from engine time (ttft/stream).
    pub stage: crate::stage_timing::StageTimings,
    /// Workload descriptor counters (B6): one per request, labeled by
    /// the dialect-neutral kind - `blazar_workload_requests_total`.
    pub workload: [std::sync::atomic::AtomicU64; crate::workload::WorkloadKind::ALL.len()],
    /// Prompt-cache observability (R6/R7-lite): gateway-side usage
    /// classification + warm/cold TTFT split. Arc so stream closures and
    /// drop-finishers can own it without the whole state.
    pub obs: Arc<CacheObs>,
    /// R4: opt-in semantic cache (in-memory, TTL + LRU).
    pub semcache: Arc<SemanticCache>,
    /// R4: semantic-cache counters for /metrics.
    pub sem: Arc<SemMetrics>,
    /// Warn-only response-semantics observation layer (`blazar why`).
    pub sentinel: Arc<Sentinel>,
    /// System/tools fingerprint churn detector (agent-client cache-bust
    /// teaching): flags conversations whose prefix mutates every turn.
    /// Advisory only - never mutates requests.
    pub cache_bust: Arc<crate::cache_bust::CacheBustTracker>,
    /// Per-key scoping / rate limits / usage accounting (`[[keys]]`).
    pub keys: Arc<KeysLimiter>,
    /// OTLP trace export (default off; bounded, never blocks).
    pub otlp: Arc<Otlp>,
    /// Responses API conversation state (`previous_response_id`
    /// chaining; bounded LRU - upstream has no storage of its own).
    pub responses: std::sync::Mutex<crate::responses::ResponsesRegistry>,
    /// Local whisper.cpp lane (H8): lazily-spawned whisper-server child
    /// for /v1/audio/transcriptions; killed at serve teardown or by the
    /// idle reaper (`whisper_idle_secs`). Arc so the reaper task shares
    /// the same child slot every request path uses.
    pub whisper: std::sync::Arc<blazar_runtime::whisper::WhisperRuntime>,
    /// Kokoro TTS lane: same lazy-child lifecycle as whisper, driven by
    /// `/v1/audio/speech` and the shared idle reaper.
    pub kokoro: std::sync::Arc<blazar_runtime::kokoro::KokoroRuntime>,
    /// Gateway-owned async audio jobs (`"async": true` on the audio
    /// routes): upstream /inference is sync-only, so long files run as
    /// spawned tasks polled at /v1/audio/jobs/{id}. Bounded registry;
    /// jobs die with the gateway process.
    pub audio_jobs: crate::whisper::AudioJobs,
    /// Durable job ledger (SQLite write-through behind every async lane:
    /// audio tasks, image/video child jobs) plus the unified
    /// `/v1/jobs` read/cancel/events plane. Owns artifact spillover on
    /// disk; holds no locks and no tasks - the lanes stay authoritative
    /// for live state, this outlives them.
    pub jobs: std::sync::Arc<crate::jobs::JobRuntime>,
    /// Request lifecycle cards (v0.15): in-flight generation tracking +
    /// programmatic cancel/interrupt via `/v1/requests`. Pure memory -
    /// cards live and die with the process by design (durable history is
    /// the job ledger's role, not this one).
    pub requests: std::sync::Arc<crate::requests::RequestRuntime>,
    /// Single-flight for identical NON-STREAM requests (model + body
    /// hash): concurrent duplicates wait for the leader, then ride the
    /// leader's warm prefix instead of double-prefilling. Bounded. The
    /// outer `Arc<Mutex<..>>` lets the guard remove its own entry in
    /// `Drop` (F31: abort-safe cleanup).
    pub singleflight: std::sync::Arc<
        std::sync::Mutex<std::collections::HashMap<u64, std::sync::Arc<tokio::sync::Mutex<()>>>>,
    >,
    /// The ACTUAL bound HTTP listener address (loopback-reachable form),
    /// set by `serve()` after bind. The batch worker needs this: config
    /// port 0 / dynamic ports must not be guessed from `config`.
    pub http_addr: std::sync::OnceLock<(String, u16)>,
    /// Remote-fleet health (C2/C3): per-remote circuit state + in-flight
    /// counter, keyed `"{name}|{url}"`. Arc so `RemoteLease` can drop it
    /// without owning the whole state.
    pub remote_health:
        std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, RemoteHealth>>>,
    /// C4 cache-aware remote routing: conversation-prefix hash -> the
    /// `health_key` of the remote that last served it. Sticky while that
    /// remote stays live; bounded (one entry per distinct prefix).
    pub remote_affinity: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<u64, String>>>,
    /// Federation presence (per-remote `/v1/models` listings, TTL-bound
    /// in `remotes::refresh_presence`): backs bare-name fallback routing
    /// and the `/v1/models` peer merge. Keyed by the same `health_key`.
    pub remote_presence: std::sync::Arc<
        std::sync::Mutex<std::collections::HashMap<String, crate::remotes::PeerPresence>>,
    >,
    /// v0.16 federation: per-peer `/api/capacity` snapshots, refreshed
    /// alongside presence (same TTL window). Absent entry = non-Blazar
    /// peer or not yet probed - capacity-aware routing degrades to the
    /// load/circuit signals above, never errors.
    pub remote_capacity: std::sync::Arc<
        std::sync::Mutex<std::collections::HashMap<String, crate::remotes::PeerCapacity>>,
    >,
    /// Model-pool hysteresis: each `adaptive` pool's last pick (member
    /// effective string + instant), consumed by [`crate::pools::pick`]
    /// to keep a healthy near-best member sticky inside its window.
    /// Keyed by pool name; written only on the request path (`expand`).
    pub pools_sticky: std::sync::Arc<
        std::sync::Mutex<std::collections::HashMap<String, crate::pools::StickyPick>>,
    >,
    /// Local-lane service-time EWMA per model (milliseconds), stamped
    /// when local chat streams complete (see `keys::LocalServiceStamp`)
    /// and consumed by pool `adaptive` scoring - the local twin of the
    /// remote lane's per-peer `service_ewma_ms`.
    pub service_ewma_local: std::sync::Arc<
        std::sync::Mutex<std::collections::HashMap<String, (f64, std::time::Instant)>>,
    >,
    /// Local-lane time-to-first-token EWMA per model (milliseconds),
    /// stamped by the usage tap at the first observed stream byte and
    /// consumed by pool scoring: local members multiply their queue
    /// waves by this measured TTFT instead of the declared default.
    /// Same tuple shape as `service_ewma_local` so staleness decay
    /// applies uniformly.
    pub ttft_ewma_local: std::sync::Arc<
        std::sync::Mutex<std::collections::HashMap<String, (f64, std::time::Instant)>>,
    >,
    /// Per-prefix cache affinity for adaptive pools: conversation
    /// opener hash -> (member effective string, when it served).
    /// Consulted and recorded by `pools::expand_request`; bounded and
    /// expired by the pools module's own hygiene rules.
    pub pools_affinity: std::sync::Arc<
        std::sync::Mutex<std::collections::HashMap<String, (String, std::time::Instant)>>,
    >,
    /// Runtime operator overrides per pool member ("{pool}:{effective}"
    /// -> ops): drain (temporary cordon, optional TTL deadline) and pin
    /// (outrank strategy while healthy). In-memory - a restart clears
    /// them; permanent membership changes belong in `[[model_pools]]`.
    pub pools_ops: std::sync::Arc<
        std::sync::Mutex<std::collections::HashMap<String, crate::pools::MemberOps>>,
    >,
    /// Affinity steering counter ("{pool}:{effective}" -> hits): how
    /// many requests routed to the member already holding their
    /// conversation opener's prefix KV. Session-only telemetry (a
    /// restart zeroes it); surfaced per member in the route table.
    pub pools_affinity_hits:
        std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, u64>>>,
    /// Routed-request counter per pool name: the denominator for the
    /// steering hit-rate (`affinity_hits / requests`). Session-only
    /// telemetry like the hits map; every successful pool expansion
    /// counts one request, whatever lane initiated it.
    pub pools_requests: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, u64>>>,
    /// Timestamped request arrivals per pool for the rolling steering
    /// rate (paired with `pools_hit_window`; pruned to the window and
    /// capped, unlike the lifetime `pools_requests` counter).
    pub pools_req_window: std::sync::Arc<
        std::sync::Mutex<
            std::collections::HashMap<String, std::collections::VecDeque<std::time::Instant>>,
        >,
    >,
    /// Timestamped steering hits per pool - the numerator of the
    /// rolling rate, same window and cap as `pools_req_window`. Each
    /// event carries the member it steered to, so per-member trends
    /// split out of the same events.
    #[allow(clippy::type_complexity)]
    // one tagged deque; a type alias would hide the member-tag shape
    pub pools_hit_window: std::sync::Arc<
        std::sync::Mutex<
            std::collections::HashMap<
                String,
                std::collections::VecDeque<(std::time::Instant, String)>,
            >,
        >,
    >,
    /// E4 audit: line sender (`None` = audit off; no request-path tax
    /// when disabled). Dropped-line counter is shared with the writer.
    pub audit_tx: Option<tokio::sync::mpsc::Sender<crate::audit::AuditLine>>,
    /// Audit lines dropped because the writer channel was full -
    /// surfaced via `/metrics` as `blazar_audit_dropped_total`.
    pub audit_dropped: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Reused `SQLite` handle for per-request store reads (engine kind,
    /// tag lists, usage rows). Previously ~15 gateway sites opened a
    /// fresh connection (open + 3 pragmas) per request. `Mutex`-wrapped
    /// (rusqlite `Connection` is `Send + !Sync`); all uses are
    /// synchronous - the guard never crosses an `.await`. We cache the
    /// CONNECTION, not data: WAL + per-call queries keep CLI-side
    /// writes immediately visible cross-process.
    pub store: std::sync::Mutex<Option<blazar_core::Store>>,
    /// Warn-once latch for `with_store` open failures: the first
    /// failure of an incident logs (store errors otherwise read as
    /// "absent" at every call site); a successful re-open re-arms it.
    pub store_open_warned: std::sync::atomic::AtomicBool,
    /// Ordered failover chains (`[[failover]]`): alias resolution,
    /// anti-flap benching, pin/unpin, bounded switch history. Empty
    /// registry = zero request-path cost (one map miss on the alias).
    pub failover: crate::failover::Registry,
    /// MCP stdio server registry: lazily spawned sessions behind the
    /// `mcp` request opt-in. Empty config = zero request-path cost.
    pub mcp: crate::mcp::Registry,
    /// F4 usage metering: per-(day, model) token/latency rollups.
    /// Recording is an O(1) hash bump on the response path; a periodic
    /// task flushes buckets into `usage_daily`.
    pub usage: std::sync::Arc<crate::usage::UsageMeter>,
    /// Admission-accuracy samples (R-admission): the last poller join of
    /// "what the fit gate charged" vs "what the driver actually
    /// reports", per resident engine process. Purely observational -
    /// never feeds a gate, so a stale or missing sample can only make
    /// the surface less informed, never wrong.
    pub admission: std::sync::Arc<AdmissionAccuracy>,
}

/// One predicted-vs-actual VRAM sample for a resident engine child.
/// `predicted_mib` is the admission floor the gate charged (weights +
/// KV floor + spawn overhead); `actual_mib` is the driver-reported
/// compute footprint of that exact pid, when the platform exposes one.
#[derive(Debug, Clone)]
pub struct AccuracyRow {
    /// Display name from `ps` (includes any `+lora` suffix).
    pub model: String,
    /// Engine child pid - the join key against driver tenant lists.
    pub pid: u32,
    /// Device the child was placed on (e.g. `cuda:0`), when known.
    pub device: Option<String>,
    /// `admission_floor_bytes` / MiB - the gate's charge.
    pub predicted_mib: u64,
    /// Driver-reported compute usage of `pid`, when a tenant census ran.
    /// `None` = platform without a census tool (or a skipped tick): the
    /// prediction side stays visible, honestly unverified.
    pub actual_mib: Option<u64>,
    /// Unix seconds of the poller tick that produced this row.
    pub sampled_unix: i64,
}

/// Latest admission-accuracy snapshot, wholesale-replaced by the
/// gateway poller each tick. Wholesale replacement (not merge) is the
/// pid-reuse defense: rows never outlive the `ps` census that keyed
/// them, and staleness is explicit via `sampled_unix`.
#[derive(Debug, Default)]
pub struct AdmissionAccuracy {
    rows: std::sync::Mutex<Vec<AccuracyRow>>,
}

impl AdmissionAccuracy {
    /// Replace the whole sample set (poller tick boundary).
    pub fn replace(&self, rows: Vec<AccuracyRow>) {
        *self.rows.lock().expect("admission accuracy poisoned") = rows;
    }

    /// Point-in-time copy of the current samples.
    #[must_use]
    pub fn snapshot(&self) -> Vec<AccuracyRow> {
        self.rows
            .lock()
            .expect("admission accuracy poisoned")
            .clone()
    }
}

const MIB: u64 = 1024 * 1024;

/// Join the supervisor's `ps` census with a driver compute-tenant list
/// into accuracy rows. Pure so the join is unit-testable without a
/// daemon: the poller supplies live inputs, tests supply table rows.
/// CPU rows, routers (`bytes == 0`) and unspawned entries are skipped -
/// the accuracy question only exists for GPU-charged residents.
/// Partial-offload residents are KEPT: the gate charges the full floor,
/// so `actual < predicted` there is exactly the signal worth seeing.
#[must_use]
pub fn join_accuracy_rows(
    ps: &[blazar_runtime::supervisor::PsRow],
    tenants: Option<&[blazar_runtime::probe::GpuTenant]>,
    mmproj_of: impl Fn(&str) -> u64,
    now_unix: i64,
) -> Vec<AccuracyRow> {
    ps.iter()
        .filter(|p| p.gpu != "cpu" && p.pid > 0 && p.bytes > 0)
        .map(|p| {
            // Store keys drop the `+lora` display suffix; the predictor
            // charges the mmproj of the base row either way.
            let (base, _lora) = blazar_core::catalog::split_lora_suffix(&p.name);
            let mmproj = mmproj_of(base);
            let bytes = u64::try_from(p.bytes).unwrap_or(0);
            let predicted_mib = blazar_core::profile::admission_floor_bytes(bytes, mmproj) / MIB;
            AccuracyRow {
                model: p.name.clone(),
                pid: p.pid,
                device: p.device.clone(),
                predicted_mib,
                actual_mib: tenants
                    .and_then(|list| list.iter().find(|t| t.pid == p.pid).map(|t| t.used_mib)),
                sampled_unix: now_unix,
            }
        })
        .collect()
}

/// Client for dialing ONE child endpoint: the shared TCP pool, or the
/// cached socket-pinned client when the endpoint rides the unix
/// transport. Pair with `proxy::child_base` - that base names a
/// placeholder host the unix connector ignores, only the path is dialed.
#[must_use]
pub fn child_client(state: &AppState, ep: &blazar_core::profile::Endpoint) -> reqwest::Client {
    match ep {
        blazar_core::profile::Endpoint::Tcp { .. } => state.http.clone(),
        blazar_core::profile::Endpoint::Unix { socket } => state.uds_http.get(socket),
    }
}

/// Client for media forwards that hold one request open across the
/// whole generation - the timeout-less media client on TCP, the
/// (already deadline-free) per-socket pool on unix transport.
pub fn media_child_client(
    state: &AppState,
    ep: &blazar_core::profile::Endpoint,
) -> reqwest::Client {
    match ep {
        blazar_core::profile::Endpoint::Tcp { .. } => state.media_http.clone(),
        blazar_core::profile::Endpoint::Unix { socket } => state.uds_http.get(socket),
    }
}

impl AppState {
    /// Key admission for the gated lanes that live outside the proxy
    /// (images, videos, local whisper, local TTS): scope + rpm/tpm/daily
    /// check, then the request charge - the exact bracket the text lanes
    /// run inline. Local compute is not a free lane on an authed gateway
    /// (audit MM1). No-op when no key context rode the request (open
    /// gateway) or the key is unknown (the auth middleware already
    /// rejected those before any handler ran).
    pub fn admit_or_respond(
        &self,
        key_ext: Option<&axum::Extension<crate::keys::KeyCtx>>,
        model: &str,
    ) -> Result<(), Box<axum::response::Response>> {
        let Some(axum::Extension(k)) = key_ext else {
            return Ok(());
        };
        let Some(entry) = self.keys.entry(&k.name) else {
            return Ok(());
        };
        self.keys
            .admit(&entry, model)
            .map_err(|rej| Box::new(rej.to_response()))?;
        Ok(())
    }

    /// Current config generation, delegated to the supervisor's
    /// shared handle - the gateway has no second copy to drift stale
    /// when a hot reload swaps the file.
    #[must_use]
    pub fn cfg(&self) -> std::sync::Arc<Config> {
        self.sup.cfg()
    }

    #[must_use]
    #[allow(clippy::duration_suboptimal_units)] // 10-minute ceiling mirrors long generations
    #[allow(clippy::too_many_lines)] // constructor: field-by-field wiring, splitting would not reduce it
    pub fn new(dirs: BlazarDirs, config: &Config, sup: Arc<Supervisor>, bus: EventBus) -> Self {
        let failover = crate::failover::Registry::from_config(&config.failover);
        let mcp = crate::mcp::Registry::from_config(
            &config.mcp,
            config.mcp_default.clone(),
            config.mcp_max_request_secs,
        );
        let http = crate::http_pool::tuned(reqwest::Client::builder())
            .timeout(std::time::Duration::from_secs(10 * 60))
            .build()
            .expect("gateway http client");
        // No total timeout on purpose (see the field doc); connect
        // stays bounded so a wedged child endpoint fails to connect.
        let media_http = crate::http_pool::tuned(reqwest::Client::builder())
            .connect_timeout(std::time::Duration::from_secs(15))
            .build()
            .expect("gateway media http client");
        let run_dir = dirs.run_dir();
        let sentinel = Sentinel::new(config.sentinel, config.sentinel_stall_secs, Some(&run_dir));
        // E4 audit: spawn the file writer when the knob is on; the
        // sender rides AppState, the counter survives either way.
        let (audit_tx, audit_dropped) = if config.audit_log {
            let (tx, rx) = tokio::sync::mpsc::channel(crate::audit::AUDIT_CHANNEL_CAP);
            let dropped = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
            let log_dir = dirs.data_dir.join("log");
            tokio::spawn(crate::audit::writer_task(
                rx,
                log_dir,
                std::sync::Arc::clone(&dropped),
            ));
            (Some(tx), dropped)
        } else {
            (
                None,
                std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            )
        };
        // Best-effort pre-load of today's usage: a store failure must not
        // block boot (counters restart at zero, budgets loosen). The
        // handle seeds the per-request cache below instead of dropping.
        let store = blazar_core::Store::open(&dirs).ok();
        let keys = KeysLimiter::loaded(store.as_ref(), config.keys.clone());
        // Audio lane: one child slot shared by request paths and the
        // idle reaper. 0 disables the reaper (child lives until
        // teardown - the pre-reaper behavior).
        let whisper = std::sync::Arc::new(blazar_runtime::whisper::WhisperRuntime::new());
        let kokoro = std::sync::Arc::new(blazar_runtime::kokoro::KokoroRuntime::new());
        if config.whisper_idle_secs > 0 {
            // One reaper, both audio lanes: TTS and STT children are
            // equally idle-fodable, and the shared budget keeps the
            // config surface at one knob (whisper_idle_secs).
            let reaper = std::sync::Arc::clone(&whisper);
            let kokoro_reaper = std::sync::Arc::clone(&kokoro);
            let max_idle = std::time::Duration::from_secs(config.whisper_idle_secs);
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                    reaper.reap_idle(max_idle).await;
                    kokoro_reaper.reap_idle(max_idle).await;
                }
            });
        }
        let otlp = Arc::new(Otlp::new(config));
        // J5: consume stall-eviction requests; a wedged child is reaped
        // (no bank - a stalled child may not answer a save request) and
        // the next request respawns it clean.
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        // `tx` rides the AppState as `evict_tx`: the proxy lane fires
        // the same debounced reap for header-stalled children.
        sentinel.set_evict_channel(tx.clone());
        {
            let sup = Arc::clone(&sup);
            tokio::spawn(async move {
                let mut last_evict: std::collections::HashMap<String, std::time::Instant> =
                    std::collections::HashMap::new();
                while let Some(model) = rx.recv().await {
                    // FIX7 debounce: one transient stall must not churn a
                    // healthy model - at most one stall-evict per model
                    // per minute (ring still records every detection).
                    if last_evict
                        .get(&model)
                        .is_some_and(|t| t.elapsed() < std::time::Duration::from_secs(60))
                    {
                        tracing::info!(target: "blazar::sentinel", model = %model, "stall-evict debounced (<60s since last)");
                        continue;
                    }
                    last_evict.insert(model.clone(), std::time::Instant::now());
                    match sup.evict(&model).await {
                        Ok(()) => {
                            tracing::warn!(target: "blazar::sentinel", model = %model, "evicted stalled child (auto-recover)");
                        }
                        Err(e) => {
                            tracing::warn!(target: "blazar::sentinel", model = %model, "stall-evict failed: {e:#}");
                        }
                    }
                }
            });
        }
        // Job ledger owns `<data>/jobs` artifact spillover; constructed
        // before the struct literal because `dirs` moves into it.
        let jobs = std::sync::Arc::new(crate::jobs::JobRuntime::new(&dirs.data_dir));
        let requests = std::sync::Arc::new(crate::requests::RequestRuntime::new());
        // Usage meter opens its own WAL connection before `dirs` moves.
        let usage = std::sync::Arc::new(crate::usage::UsageMeter::new(&dirs));
        Self {
            dirs,
            sup,
            bus,
            invites: crate::fleet::InviteRegistry::default(),
            queue: Arc::new(PriorityQueue::new()),
            http,
            media_http,
            uds_http: blazar_runtime::uds::UdsClients::new(),
            evict_tx: tx,
            cache_bust: Arc::new(crate::cache_bust::CacheBustTracker::new()),
            ttft: crate::histogram::ttft(),
            tpot: crate::histogram::tpot(),
            stage: crate::stage_timing::StageTimings::new(),
            workload: core::array::from_fn(|_| std::sync::atomic::AtomicU64::new(0)),
            obs: Arc::new(CacheObs::new()),
            semcache: Arc::new(SemanticCache::new()),
            sem: Arc::new(SemMetrics::default()),
            sentinel,
            keys: Arc::new(keys),
            otlp,
            responses: std::sync::Mutex::new(crate::responses::ResponsesRegistry::new()),
            whisper,
            kokoro,
            audio_jobs: crate::whisper::AudioJobs::new(),
            jobs,
            requests,
            http_addr: std::sync::OnceLock::new(),
            remote_health: std::sync::Arc::default(),
            remote_affinity: std::sync::Arc::default(),
            remote_presence: std::sync::Arc::default(),
            remote_capacity: std::sync::Arc::default(),
            pools_sticky: std::sync::Arc::default(),
            service_ewma_local: std::sync::Arc::default(),
            ttft_ewma_local: std::sync::Arc::default(),
            pools_affinity: std::sync::Arc::default(),
            pools_ops: std::sync::Arc::default(),
            pools_affinity_hits: std::sync::Arc::default(),
            pools_requests: std::sync::Arc::default(),
            pools_req_window: std::sync::Arc::default(),
            pools_hit_window: std::sync::Arc::default(),
            audit_tx,
            audit_dropped,
            store: std::sync::Mutex::new(store),
            store_open_warned: std::sync::atomic::AtomicBool::new(false),
            failover,
            mcp,
            usage,
            admission: Arc::new(AdmissionAccuracy::default()),
            singleflight: std::sync::Arc::new(std::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
        }
    }

    /// Run `f` against the cached store connection, opening lazily on
    /// first use. `None` = store unavailable (callers keep their existing
    /// degraded paths). A failed open is retried on the next call - same
    /// per-request retry semantics as the old open-per-request sites, at
    /// zero cost once healthy.
    pub fn with_store<T>(&self, f: impl FnOnce(&blazar_core::Store) -> T) -> Option<T> {
        let mut guard = self.store.lock().expect("store handle poisoned");
        if guard.is_none() {
            match blazar_core::Store::open(&self.dirs) {
                Ok(store) => {
                    *guard = Some(store);
                    // Re-arm the warn-once latch: a failure after this
                    // recovery is a new incident, not a repeat.
                    self.store_open_warned
                        .store(false, std::sync::atomic::Ordering::Relaxed);
                }
                Err(e) => {
                    // Store-open failures retry per call (below), but a
                    // broken store makes every persisted read look like
                    // "absent" - that conflation must be visible at
                    // least once per incident, not silently per request.
                    if !self
                        .store_open_warned
                        .swap(true, std::sync::atomic::Ordering::Relaxed)
                    {
                        tracing::warn!(
                            target: "blazar::store",
                            "store open failed ({}) - persisted state reads as absent until it recovers: {e}",
                            self.dirs.data_dir.display()
                        );
                    }
                }
            }
        }
        guard.as_ref().map(f)
    }

    /// Memory-first stored-response lookup with SQLite fallback (restart
    /// durability for `previous_response_id` chaining). Lock order is a
    /// fixed law: responses mutex -> store mutex, never reversed - the
    /// store mutex is only ever taken while already inside a responses
    /// guard (here and in `store_response`), so no cycle can form.
    pub fn stored_response(&self, id: &str) -> Option<crate::responses::StoredResponse> {
        let mut reg = self.responses.lock().expect("responses registry");
        if let Some(s) = reg.get(id) {
            return Some(s.clone());
        }
        self.with_store(|s| reg.promote_from_store(s, id)).flatten()
    }

    /// Persisted put: hot LRU always, then SQLite write-through + prune
    /// (same 24h TTL / cap as memory). Store failures log loud - durable
    /// chain degradation must be visible - but never fail the request:
    /// memory stays authoritative for the live process, the ledger is
    /// the restart path. Lock order: responses -> store (see
    /// `stored_response`).
    pub fn store_response(&self, id: &str, r: &crate::responses::StoredResponse) {
        let mut reg = self.responses.lock().expect("responses registry");
        reg.put(id.to_string(), r.clone());
        let durable = self.with_store(|s| {
            s.put_response(&crate::responses::ResponsesRegistry::to_row(id, r))
                .and_then(|()| {
                    s.prune_responses(
                        crate::responses::RESPONSES_TTL_SECS,
                        crate::responses::RESPONSES_CAP,
                    )
                })
        });
        match durable {
            Some(Ok(())) => {}
            Some(Err(e)) => tracing::warn!(
                "responses: durable write failed: {e:#} - {id} is memory-only until the next successful put"
            ),
            None => tracing::warn!(
                "responses: store unavailable - {id} is memory-only (will not survive a restart)"
            ),
        }
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    fn rendered(h: &Histogram) -> String {
        let mut out = String::new();
        h.render(&mut out);
        out
    }

    #[test]
    fn unit__cache_obs__warm_routes_to_warm_histogram() {
        let obs = CacheObs::new();
        obs.record("m", 100, 96, Some(0.05));
        let warm = rendered(&obs.ttft_warm);
        let cold = rendered(&obs.ttft_cold);
        assert!(warm.contains("_count 1"), "{warm}");
        assert!(cold.contains("_count 0"), "{cold}");
        assert_eq!(obs.prompt_tokens.load(Ordering::Relaxed), 100);
        assert_eq!(obs.cached_tokens.load(Ordering::Relaxed), 96);
        assert_eq!(obs.unclassified.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn unit__cache_obs__cold_routes_to_cold_histogram() {
        let obs = CacheObs::new();
        obs.record("m", 100, 0, Some(0.5));
        assert!(rendered(&obs.ttft_warm).contains("_count 0"));
        assert!(rendered(&obs.ttft_cold).contains("_count 1"));
    }

    #[test]
    fn unit__cache_obs__missing_ttft_counts_tokens_skips_histograms() {
        // Known usage but no TTFT (e.g. abort between usage and drain):
        // tokens counted, the warm/cold split stays honest (neither hist).
        let obs = CacheObs::new();
        obs.record("m", 70, 40, None);
        assert_eq!(obs.prompt_tokens.load(Ordering::Relaxed), 70);
        assert_eq!(obs.cached_tokens.load(Ordering::Relaxed), 40);
        assert!(rendered(&obs.ttft_warm).contains("_count 0"));
        assert!(rendered(&obs.ttft_cold).contains("_count 0"));
    }

    #[test]
    fn unit__cache_obs__miss_counts_unclassified_only() {
        let obs = CacheObs::new();
        obs.miss();
        obs.miss();
        assert_eq!(obs.unclassified.load(Ordering::Relaxed), 2);
        assert_eq!(obs.prompt_tokens.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn unit__cache_obs__per_model_ratio_isolated_and_null_before_data() {
        let obs = CacheObs::new();
        // No traffic yet: ratio is None (the ps HIT column renders `-`,
        // never a lying 0%).
        assert_eq!(obs.hit_ratio("a"), None);
        obs.record("a", 100, 96, Some(0.05));
        obs.record("a", 100, 4, Some(0.05));
        obs.record("b", 50, 0, Some(0.5));
        // Per-model split: a = 100/200, b = 0/50 - independent buckets.
        assert_eq!(obs.hit_ratio("a"), Some(0.5));
        assert_eq!(obs.hit_ratio("b"), Some(0.0));
        assert_eq!(obs.hit_ratio("c"), None);
        // Globals still sum across models.
        assert_eq!(obs.prompt_tokens.load(Ordering::Relaxed), 250);
        assert_eq!(obs.cached_tokens.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn unit__cache_obs__render_per_model_sorted_and_skips_empty() {
        let obs = CacheObs::new();
        obs.record("b-model", 100, 100, None);
        obs.record("a-model", 80, 40, None);
        obs.per_model.insert(
            "z-empty".to_string(),
            super::ModelCacheObs {
                prompt_tokens: std::sync::atomic::AtomicU64::new(0),
                cached_tokens: std::sync::atomic::AtomicU64::new(0),
            },
        );
        let mut out = String::new();
        obs.render_per_model(&mut out);
        // Sorted scrape-stable order; zero-traffic models stay silent.
        let a = out.find("blazar_gateway_cache_hit_ratio{model=\"a-model\"} 0.5000");
        let b = out.find("blazar_gateway_cache_hit_ratio{model=\"b-model\"} 1.0000");
        assert!(a.is_some(), "{out}");
        assert!(b.is_some(), "{out}");
        assert!(a.unwrap() < b.unwrap(), "{out}");
        assert!(!out.contains("z-empty"), "{out}");
        // HELP/TYPE emitted exactly once for the family.
        assert_eq!(
            out.matches("# HELP blazar_gateway_cache_hit_ratio").count(),
            1
        );
    }

    /// Minimal `ps` row builder mirroring the supervisor struct field
    /// order; only the fields the join reads vary per scenario.
    fn row(name: &str, pid: u32, bytes: i64, gpu: &str) -> blazar_runtime::supervisor::PsRow {
        blazar_runtime::supervisor::PsRow {
            name: name.to_string(),
            engine: "llama".to_string(),
            replica: None,
            state: "ready",
            endpoint: "http://127.0.0.1:1".to_string(),
            idle_secs: 0,
            in_flight: 0,
            ctx: 4096,
            capacity: None,
            slots: None,
            slots_configured: None,
            slots_source: None,
            gpu: gpu.to_string(),
            device: Some("cuda:0".to_string()),
            device_id: Some("CUDA0".to_string()),
            warnings: Vec::new(),
            spec_mode: String::new(),
            draft: None,
            pid,
            bytes,
            heat: 0,
            keep_alive_secs: None,
        }
    }

    #[allow(clippy::cast_possible_wrap)] // test weights are tiny positives
    fn gpu_row(name: &str, pid: u32, weights_mib: u64) -> blazar_runtime::supervisor::PsRow {
        row(name, pid, (weights_mib * super::MIB) as i64, "full")
    }

    #[allow(clippy::cast_possible_wrap)] // test weights are tiny positives
    fn cpu_row(name: &str) -> blazar_runtime::supervisor::PsRow {
        row(name, 99, 500 * super::MIB as i64, "cpu")
    }

    fn router_row() -> blazar_runtime::supervisor::PsRow {
        row("router", 0, 0, "auto")
    }

    fn tenant(pid: u32, used_mib: u64) -> blazar_runtime::probe::GpuTenant {
        blazar_runtime::probe::GpuTenant {
            pid,
            process_name: "llama-server".to_string(),
            used_mib,
        }
    }

    #[test]
    fn unit__join_accuracy_rows__pid_match_joins_actual() {
        let rows = vec![gpu_row("qwen", 42, 100), cpu_row("cpu-model"), router_row()];
        let tenants = vec![tenant(7, 10), tenant(42, 1234)];
        let out = join_accuracy_rows(&rows, Some(&tenants), |_| 0, 1_700_000_000);
        assert_eq!(out.len(), 1, "cpu/router rows must be filtered");
        let r = &out[0];
        assert_eq!(r.model, "qwen");
        assert_eq!(r.pid, 42);
        assert_eq!(r.actual_mib, Some(1234));
        // floor = 100 weights + 512 KV floor + 700 spawn overhead.
        assert_eq!(r.predicted_mib, 100 + 512 + 700);
        assert_eq!(r.sampled_unix, 1_700_000_000);
    }

    #[test]
    fn unit__join_accuracy_rows__no_census_keeps_prediction_honest_none() {
        let rows = vec![gpu_row("qwen", 42, 100)];
        let out = join_accuracy_rows(&rows, None, |_| 0, 0);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].actual_mib, None);
        // An empty census list is still a census: unmatched pid = no
        // tenant visible (not "uses 0 MiB").
        let out = join_accuracy_rows(&rows, Some(&[]), |_| 0, 0);
        assert_eq!(out[0].actual_mib, None);
    }

    #[test]
    fn unit__join_accuracy_rows__lora_display_looks_up_base_and_folds_mmproj() {
        let rows = vec![gpu_row("qwen+my-adapter", 42, 100)];
        let out = join_accuracy_rows(
            &rows,
            None,
            |base| {
                assert_eq!(base, "qwen", "store lookup must use the base name");
                3 * super::MIB // vision projector bytes
            },
            0,
        );
        assert_eq!(out[0].model, "qwen+my-adapter", "display name kept");
        assert_eq!(out[0].predicted_mib, 100 + 3 + 512 + 700);
    }
}
