//! Shared gateway state.

use std::sync::Arc;

use pallama_core::{Config, PallamaDirs};
use pallama_runtime::{EventBus, Supervisor};

use crate::histogram::Histogram;
use crate::keys::KeysLimiter;
use crate::otlp::Otlp;
use crate::queue::PriorityQueue;
use crate::sentinel::Sentinel;

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
    /// leader's warm prefix instead of double-prefilling. Bounded.
    pub singleflight:
        tokio::sync::Mutex<std::collections::HashMap<u64, std::sync::Arc<tokio::sync::Mutex<()>>>>,
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
        // Best-effort pre-load of today's usage: a store failure must not
        // block boot (counters restart at zero, budgets loosen).
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
            sentinel,
            keys: Arc::new(keys),
            otlp,
            responses: std::sync::Mutex::new(crate::responses::ResponsesRegistry::new()),
            whisper: pallama_runtime::whisper::WhisperRuntime::new(),
            singleflight: tokio::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }
}
