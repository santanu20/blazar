//! Shared gateway state.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

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
/// instead of guessing a side. Atomics only — lock-free on the hot path.
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
    /// entry per distinct model — bounded by the store, not by traffic.
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
            // winner's entry — totals stay exact.
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
    pub config: Config,
    pub sup: Arc<Supervisor>,
    pub bus: EventBus,
    pub queue: Arc<PriorityQueue>,
    /// Loopback client for child traffic + metrics scrape.
    pub http: reqwest::Client,
    /// Timeout-less client for media forwards: a sync image/video
    /// request holds the connection for the WHOLE render — minutes,
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
    /// instead — a debounce is too slow to gate an in-band retry.
    pub evict_tx: tokio::sync::mpsc::UnboundedSender<String>,
    /// vLLM-style evidence loop: measured at the proxy, owned by the gateway.
    pub ttft: Histogram,
    pub tpot: Histogram,
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
    /// Advisory only — never mutates requests.
    pub cache_bust: Arc<crate::cache_bust::CacheBustTracker>,
    /// Per-key scoping / rate limits / usage accounting (`[[keys]]`).
    pub keys: Arc<KeysLimiter>,
    /// OTLP trace export (default off; bounded, never blocks).
    pub otlp: Arc<Otlp>,
    /// Responses API conversation state (`previous_response_id`
    /// chaining; bounded LRU — upstream has no storage of its own).
    pub responses: std::sync::Mutex<crate::responses::ResponsesRegistry>,
    /// Local whisper.cpp lane (H8): lazily-spawned whisper-server child
    /// for /v1/audio/transcriptions; killed at serve teardown or by the
    /// idle reaper (`whisper_idle_secs`). Arc so the reaper task shares
    /// the same child slot every request path uses.
    pub whisper: std::sync::Arc<blazar_runtime::whisper::WhisperRuntime>,
    /// Gateway-owned async audio jobs (`"async": true` on the audio
    /// routes): upstream /inference is sync-only, so long files run as
    /// spawned tasks polled at /v1/audio/jobs/{id}. Bounded registry;
    /// jobs die with the gateway process.
    pub audio_jobs: crate::whisper::AudioJobs,
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
    /// E4 audit: line sender (`None` = audit off; no request-path tax
    /// when disabled). Dropped-line counter is shared with the writer.
    pub audit_tx: Option<tokio::sync::mpsc::Sender<crate::audit::AuditLine>>,
    /// Audit lines dropped because the writer channel was full —
    /// surfaced via `/metrics` as `blazar_audit_dropped_total`.
    pub audit_dropped: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Reused `SQLite` handle for per-request store reads (engine kind,
    /// tag lists, usage rows). Previously ~15 gateway sites opened a
    /// fresh connection (open + 3 pragmas) per request. `Mutex`-wrapped
    /// (rusqlite `Connection` is `Send + !Sync`); all uses are
    /// synchronous — the guard never crosses an `.await`. We cache the
    /// CONNECTION, not data: WAL + per-call queries keep CLI-side
    /// writes immediately visible cross-process.
    pub store: std::sync::Mutex<Option<blazar_core::Store>>,
}

/// Client for dialing ONE child endpoint: the shared TCP pool, or the
/// cached socket-pinned client when the endpoint rides the unix
/// transport. Pair with `proxy::child_base` — that base names a
/// placeholder host the unix connector ignores, only the path is dialed.
#[must_use]
pub fn child_client(state: &AppState, ep: &blazar_core::profile::Endpoint) -> reqwest::Client {
    match ep {
        blazar_core::profile::Endpoint::Tcp { .. } => state.http.clone(),
        blazar_core::profile::Endpoint::Unix { socket } => state.uds_http.get(socket),
    }
}

/// Client for media forwards that hold one request open across the
/// whole generation — the timeout-less media client on TCP, the
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
    /// check, then the request charge — the exact bracket the text lanes
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
            .check(&entry, model)
            .map_err(|rej| Box::new(rej.to_response()))?;
        self.keys.charge_request(&k.name);
        Ok(())
    }

    #[must_use]
    #[allow(clippy::duration_suboptimal_units)] // 10-minute ceiling mirrors long generations
    pub fn new(dirs: BlazarDirs, config: Config, sup: Arc<Supervisor>, bus: EventBus) -> Self {
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
        // teardown — the pre-reaper behavior).
        let whisper = std::sync::Arc::new(blazar_runtime::whisper::WhisperRuntime::new());
        if config.whisper_idle_secs > 0 {
            let reaper = std::sync::Arc::clone(&whisper);
            let max_idle = std::time::Duration::from_secs(config.whisper_idle_secs);
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                    reaper.reap_idle(max_idle).await;
                }
            });
        }
        let otlp = Arc::new(Otlp::new(&config));
        // J5: consume stall-eviction requests; a wedged child is reaped
        // (no bank — a stalled child may not answer a save request) and
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
                    // healthy model — at most one stall-evict per model
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
        Self {
            dirs,
            config,
            sup,
            bus,
            queue: Arc::new(PriorityQueue::new()),
            http,
            media_http,
            uds_http: blazar_runtime::uds::UdsClients::new(),
            evict_tx: tx,
            cache_bust: Arc::new(crate::cache_bust::CacheBustTracker::new()),
            ttft: crate::histogram::ttft(),
            tpot: crate::histogram::tpot(),
            obs: Arc::new(CacheObs::new()),
            semcache: Arc::new(SemanticCache::new()),
            sem: Arc::new(SemMetrics::default()),
            sentinel,
            keys: Arc::new(keys),
            otlp,
            responses: std::sync::Mutex::new(crate::responses::ResponsesRegistry::new()),
            whisper,
            audio_jobs: crate::whisper::AudioJobs::new(),
            http_addr: std::sync::OnceLock::new(),
            remote_health: std::sync::Arc::default(),
            remote_affinity: std::sync::Arc::default(),
            audit_tx,
            audit_dropped,
            store: std::sync::Mutex::new(store),
            singleflight: std::sync::Arc::new(std::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
        }
    }

    /// Run `f` against the cached store connection, opening lazily on
    /// first use. `None` = store unavailable (callers keep their existing
    /// degraded paths). A failed open is retried on the next call — same
    /// per-request retry semantics as the old open-per-request sites, at
    /// zero cost once healthy.
    pub fn with_store<T>(&self, f: impl FnOnce(&blazar_core::Store) -> T) -> Option<T> {
        let mut guard = self.store.lock().expect("store handle poisoned");
        if guard.is_none() {
            *guard = blazar_core::Store::open(&self.dirs).ok();
        }
        guard.as_ref().map(f)
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
        // Per-model split: a = 100/200, b = 0/50 — independent buckets.
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
}
