//! Shared gateway state.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use pallama_core::{Config, PallamaDirs};
use pallama_runtime::{EventBus, Supervisor};

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
        }
    }

    /// Classify one completed response. `cached > 0` = warm. A missing
    /// TTFT with known usage still counts tokens but skips both
    /// histograms (the split stays honest).
    pub fn record(&self, prompt: u64, cached: u64, ttft_secs: Option<f64>) {
        self.prompt_tokens.fetch_add(prompt, Ordering::Relaxed);
        self.cached_tokens.fetch_add(cached, Ordering::Relaxed);
        let warm = cached > 0;
        if let Some(t) = ttft_secs {
            if warm {
                self.ttft_warm.observe_secs(t);
            } else {
                self.ttft_cold.observe_secs(t);
            }
        }
    }

    /// A response completed without any usage information: counted as
    /// unclassified rather than silently dropped or forced cold.
    pub fn miss(&self) {
        self.unclassified.fetch_add(1, Ordering::Relaxed);
    }
}

pub struct AppState {
    pub dirs: PallamaDirs,
    pub config: Config,
    pub sup: Arc<Supervisor>,
    pub bus: EventBus,
    pub queue: Arc<PriorityQueue>,
    /// Loopback client for child traffic + metrics scrape.
    pub http: reqwest::Client,
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
    /// Warn-only response-semantics observation layer (`pallama why`).
    pub sentinel: Arc<Sentinel>,
    /// Per-key scoping / rate limits / usage accounting (`[[keys]]`).
    pub keys: Arc<KeysLimiter>,
    /// OTLP trace export (default off; bounded, never blocks).
    pub otlp: Arc<Otlp>,
    /// Responses API conversation state (`previous_response_id`
    /// chaining; bounded LRU — upstream has no storage of its own).
    pub responses: std::sync::Mutex<crate::responses::ResponsesRegistry>,
    /// Local whisper.cpp lane (H8): lazily-spawned whisper-server child
    /// for /v1/audio/transcriptions; killed at serve teardown.
    pub whisper: pallama_runtime::whisper::WhisperRuntime,
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
    /// surfaced via `/metrics` as `pallama_audit_dropped_total`.
    pub audit_dropped: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Reused `SQLite` handle for per-request store reads (engine kind,
    /// tag lists, usage rows). Previously ~15 gateway sites opened a
    /// fresh connection (open + 3 pragmas) per request. `Mutex`-wrapped
    /// (rusqlite `Connection` is `Send + !Sync`); all uses are
    /// synchronous — the guard never crosses an `.await`. We cache the
    /// CONNECTION, not data: WAL + per-call queries keep CLI-side
    /// writes immediately visible cross-process.
    pub store: std::sync::Mutex<Option<pallama_core::Store>>,
}

impl AppState {
    #[must_use]
    #[allow(clippy::duration_suboptimal_units)] // 10-minute ceiling mirrors long generations
    pub fn new(dirs: PallamaDirs, config: Config, sup: Arc<Supervisor>, bus: EventBus) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10 * 60))
            .build()
            .expect("gateway http client");
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
        let store = pallama_core::Store::open(&dirs).ok();
        let keys = KeysLimiter::loaded(store.as_ref(), config.keys.clone());
        let otlp = Arc::new(Otlp::new(&config));
        // J5: consume stall-eviction requests; a wedged child is reaped
        // (no bank — a stalled child may not answer a save request) and
        // the next request respawns it clean.
        {
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
            sentinel.set_evict_channel(tx);
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
                        tracing::info!(target: "pallama::sentinel", model = %model, "stall-evict debounced (<60s since last)");
                        continue;
                    }
                    last_evict.insert(model.clone(), std::time::Instant::now());
                    match sup.evict(&model).await {
                        Ok(()) => {
                            tracing::warn!(target: "pallama::sentinel", model = %model, "evicted stalled child (auto-recover)");
                        }
                        Err(e) => {
                            tracing::warn!(target: "pallama::sentinel", model = %model, "stall-evict failed: {e:#}");
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
            ttft: crate::histogram::ttft(),
            tpot: crate::histogram::tpot(),
            obs: Arc::new(CacheObs::new()),
            semcache: Arc::new(SemanticCache::new()),
            sem: Arc::new(SemMetrics::default()),
            sentinel,
            keys: Arc::new(keys),
            otlp,
            responses: std::sync::Mutex::new(crate::responses::ResponsesRegistry::new()),
            whisper: pallama_runtime::whisper::WhisperRuntime::new(),
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
    pub fn with_store<T>(&self, f: impl FnOnce(&pallama_core::Store) -> T) -> Option<T> {
        let mut guard = self.store.lock().expect("store handle poisoned");
        if guard.is_none() {
            *guard = pallama_core::Store::open(&self.dirs).ok();
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
        obs.record(100, 96, Some(0.05));
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
        obs.record(100, 0, Some(0.5));
        assert!(rendered(&obs.ttft_warm).contains("_count 0"));
        assert!(rendered(&obs.ttft_cold).contains("_count 1"));
    }

    #[test]
    fn unit__cache_obs__missing_ttft_counts_tokens_skips_histograms() {
        // Known usage but no TTFT (e.g. abort between usage and drain):
        // tokens counted, the warm/cold split stays honest (neither hist).
        let obs = CacheObs::new();
        obs.record(70, 40, None);
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
}
