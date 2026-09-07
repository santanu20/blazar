//! Instance supervisor: spawn llama-server children per model, keep them
//! healthy, evict on the idle ladder, respawn after crashes with a circuit
//! breaker, and shut everything down cleanly.
//!
//! Invariants (§9): every spawned child has a named stop path (evict /
//! shutdown), teardown is idempotent, SIGTERM→grace→SIGKILL bounded, and
//! every state transition publishes an event.

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use dashmap::DashMap;
use tokio::sync::Notify;

use pallama_core::profile::{self, Endpoint, ProfileInput};
use pallama_core::store::Store;
use pallama_core::{Config, Hardware, ModelRow, PallamaDirs};

use crate::events::{EventBus, InstanceState, PallamaEvent};

/// Instance key for the single router-mode child (never a model name:
/// underscore prefix is invalid in HF repo names).
pub const ROUTER_KEY: &str = "_router";

/// Hard cap on `replicas` per model (B1). Beyond this, capacity fights
/// the point of replication; users wanting more can raise
/// `max_loaded_models` and stack models.
pub const MAX_REPLICAS: u32 = 8;
/// Bound on the prefix→replica affinity table (B1): one entry per
/// distinct prompt prefix; eviction keeps it from growing unbounded.
const PREFIX_AFFINITY_CAP: usize = 512;

/// Model name behind an instance key: `"qwen#2"` → `"qwen"`. Plain keys
/// (no `#`) pass through unchanged, so `replicas = 1` stays
/// byte-identical with the pre-replica world.
fn model_of_key(key: &str) -> &str {
    match key.split_once('#') {
        Some((model, _)) => model,
        None => key,
    }
}

/// Split an instance key into (model, replica index). `None` for plain
/// model keys and malformed suffixes.
fn split_replica(key: &str) -> Option<(&str, u32)> {
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
    pub started_at: Instant,
    pub argv: Vec<String>,
    pub model: ModelRow,
    pub profile_ctx: u32,
    /// Resolved offload label from the compiled profile ("full" | "cpu" |
    /// "partial" | "auto") — what `ps` shows so silent CPU fallback is
    /// never silent.
    pub gpu: String,
    /// Post-quantization KV-cache estimate from the compiled profile —
    /// feeds the co-residency planner (A15).
    pub kv_est_bytes: Option<u64>,
    child: tokio::sync::Mutex<ChildHandle>,
    pid: u32,
}

/// Measured prompt-cache hit rate, shared between the gateway poller
/// (writer) and profile compiles (reader). Fixed-point milli-units so the
/// read path is a single atomic load — no lock on the spawn hot path.
#[derive(Debug)]
pub struct CacheHint {
    rate_milli: std::sync::atomic::AtomicU32,
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
    pub pid: u32,
    /// Model bytes on disk (0 in router mode — the front child serves
    /// many models and owns no single size).
    pub bytes: i64,
    /// Prefix heat (computed under the INSTANCE key so replica rows
    /// carry their own warmth).
    pub heat: u64,
}

pub struct Supervisor {
    pub dirs: PallamaDirs,
    /// Total evictions (ladder + capacity) for the metrics gauge.
    pub evictions: std::sync::atomic::AtomicU64,
    /// Measured prompt-cache hit rate (gateway poller writes, profile
    /// compiles read — drives the adaptive `--cache-ram` clamp, A16).
    pub cache_hint: std::sync::Arc<CacheHint>,
    pub config: Config,
    pub bus: EventBus,
    pub hardware: Hardware,
    pub engine: Arc<dyn Engine>,
    instances: DashMap<String, Arc<Instance>>,
    /// In-flight spawns: waiters subscribe to the Notify for completion.
    loading: DashMap<String, Arc<Notify>>,
    /// Completed load results for waiters: Ok(port) or error text.
    load_results: DashMap<String, Result<EngineRef, String>>,
    /// Crash-restart timestamps per model (circuit breaker).
    restarts: DashMap<String, Vec<Instant>>,
    /// One-shot ctx override for the NEXT spawn of a model (per-request
    /// `options.num_ctx` — complaint #13). Consumed on use.
    pending_ctx: DashMap<String, u32>,
    /// Prefix heat per model: (hits, last-hit instant); decayed on read.
    heat: std::sync::Mutex<std::collections::HashMap<String, (u64, Instant)>>,
    /// Prompt-prefix → replica-key affinity (B1): hashes of
    /// (system + first user turn) pin a conversation to one warm-cache
    /// replica. Bounded, best-effort — a miss just re-routes.
    prefix_affinity: DashMap<u64, String>,
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
            loading: DashMap::new(),
            load_results: DashMap::new(),
            restarts: DashMap::new(),
            pending_ctx: DashMap::new(),
            heat: std::sync::Mutex::new(std::collections::HashMap::new()),
            prefix_affinity: DashMap::new(),
        }
    }

    /// Capacity for a given model size: explicit config wins; auto = 1 on
    /// CPU-only, else max(1, floor(VRAM / `model_bytes`)).
    #[must_use]
    pub fn capacity_for(&self, model_bytes: i64) -> usize {
        if self.config.max_loaded_models > 0 {
            return self.config.max_loaded_models as usize;
        }
        if !self.hardware.has_gpu() || model_bytes <= 0 {
            return 1;
        }
        let vram = pallama_core::Hardware::bytes(self.hardware.total_vram_mib());
        let model = u64::try_from(model_bytes).unwrap_or(u64::MAX);
        (usize::try_from(vram / model).unwrap_or(1)).max(1)
    }

    /// Co-residency check (A15): would the candidate (weights + f16 KV)
    /// join the already-resident models within 95% of VRAM? Only counts
    /// GPU-resident instances (cpu/"partial" splits live elsewhere).
    fn coresidency_needs_kv_quant(&self, candidate_bytes: u64, candidate_kv: Option<u64>) -> bool {
        if !self.hardware.has_gpu() {
            return false;
        }
        let Some(candidate_kv) = candidate_kv else {
            return false; // no geometry: never guess (H7)
        };
        let mut resident: u64 = candidate_bytes.saturating_add(candidate_kv);
        for inst in &self.instances {
            if matches!(inst.gpu.as_str(), "cpu" | "partial") {
                continue;
            }
            let weights = u64::try_from(inst.model.bytes.max(0)).unwrap_or(u64::MAX);
            resident = resident
                .saturating_add(weights)
                .saturating_add(inst.kv_est_bytes.unwrap_or(0));
        }
        let vram = pallama_core::Hardware::bytes(self.hardware.total_vram_mib());
        resident > vram / 100 * 95
    }

    /// TCP endpoints of live (non-evicted) children — the cache-hint
    /// poller's fetch list. UDS children are skipped (no HTTP lane).
    #[must_use]
    pub fn live_http_endpoints(&self) -> Vec<(String, Endpoint)> {
        self.instances
            .iter()
            .filter(|i| matches!(i.endpoint, Endpoint::Tcp { .. }))
            .map(|i| (i.name.clone(), i.endpoint.clone()))
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
        prefix: Option<u64>,
    ) -> Result<EngineRef, SupervisionError> {
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
        if let Some(h) = prefix {
            if result.is_ok() && !self.prefix_affinity.contains_key(&h) {
                if self.prefix_affinity.len() >= PREFIX_AFFINITY_CAP {
                    if let Some(oldest) = self.prefix_affinity.iter().next().map(|e| *e.key()) {
                        self.prefix_affinity.remove(&oldest);
                    }
                }
                self.prefix_affinity.insert(h, key);
            }
        }
        result
    }

    /// Pick the instance key for a request: `"model"` when
    /// `replicas <= 1` (byte-identical legacy), else `"model#N"`.
    /// Order: prefix affinity → new-conversation grows a fresh replica
    /// (own KV cache) → anonymous traffic reuses an idle live replica,
    /// else grows while all live are busy → least-loaded existing anyway
    /// (join its queue).
    fn replica_key(&self, name: &str, prefix: Option<u64>) -> String {
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
        if let Some(h) = prefix {
            if let Some(hit) = self.prefix_affinity.get(&h) {
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
        let mut best: Option<(String, i64)> = None;
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
        result
    }

    #[allow(clippy::unused_self)] // symmetrical with future instance methods
    fn engine_ref(&self, inst: &Instance) -> EngineRef {
        EngineRef {
            name: inst.name.clone(),
            endpoint: inst.endpoint.clone(),
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
            let input = ProfileInput {
                model_name: &m.name,
                instance_key: &m.name,
                model_path: &m.path,
                model_bytes: u64::try_from(m.bytes.max(0)).unwrap_or(u64::MAX),
                gguf: &gguf,
                hardware: &self.hardware,
                config: &self.config,
                overlay: &overlay,
                loras: &loras,
                draft_path: None,
                mmproj_path: m.mmproj_path.as_deref(),
                engine_tag: &manifest.tag,
                supported_flags: &manifest.flags,
                endpoint: Endpoint::Tcp {
                    host: "127.0.0.1".into(),
                    port: 0,
                },
                data_dir: &data_dir_str,
                cache_hit_rate: self.cache_hint.get(),
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

        for _attempt in 0..2 {
            let endpoint = self.pick_endpoint(ROUTER_KEY);
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
            let mut child = self
                .engine
                .spawn(&argv, &endpoint)
                .await
                .map_err(|e| SupervisionError::Internal(anyhow!("{e}")))?;
            if let Ok(Some(status)) = child.try_status() {
                tracing::warn!(
                    router = ROUTER_KEY,
                    "router died at spawn ({status}); retrying"
                );
                let _ = child.kill().await;
                let _ = child.reap().await;
                continue;
            }
            match self.engine.health_check(&endpoint, self.load_timeout).await {
                Ok(()) => {
                    let pid = child.id().ok_or_else(|| {
                        SupervisionError::Internal(anyhow!(
                            "router child has no pid; refusing to track instance"
                        ))
                    })?;
                    if pid <= 1 {
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
                        started_at: Instant::now(),
                        argv,
                        model: synthetic_model.clone(),
                        profile_ctx: 0,
                        gpu: "router".to_string(),
                        kv_est_bytes: None,
                        child: tokio::sync::Mutex::new(child),
                        pid,
                    });
                    let _ = std::fs::write(
                        self.dirs.run_dir().join(format!("{ROUTER_KEY}.pid")),
                        pid.to_string(),
                    );
                    self.instances.insert(ROUTER_KEY.to_string(), inst);
                    self.record_restart(ROUTER_KEY);
                    self.bus.publish(PallamaEvent::InstanceStateChanged {
                        name: ROUTER_KEY.to_string(),
                        state: InstanceState::Ready,
                    });
                    let inst = self.instances.get(ROUTER_KEY).expect("just inserted");
                    return Ok(self.engine_ref(inst.value()));
                }
                Err(e) => {
                    let _ = child.kill().await;
                    let _ = child.reap().await;
                    tracing::warn!(router = ROUTER_KEY, "router health check failed: {e:#}");
                }
            }
        }
        Err(SupervisionError::ModelLoadTimeout(ROUTER_KEY.to_string()))
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
            .ok_or_else(|| SupervisionError::ModelNotFound(name.to_string()))?;
        let gguf = pallama_core::read_metadata_file(std::path::Path::new(&model.path))
            .map_err(|e| SupervisionError::Internal(anyhow!("gguf metadata: {e}")))?;
        let overlay = self.config.overlay_for(name);
        let loras: Vec<(String, f64)> = store
            .list_loras(Some(name))
            .map_err(|e| SupervisionError::Internal(anyhow!("{e}")))?
            .into_iter()
            .map(|l| (l.path, l.scale))
            .collect();
        // Draft resolution: same-name model row for the catalog pair repo.
        let draft_path = pallama_core::spec_pair_for(name).and_then(|pair| {
            let draft_name =
                crate::hf::registry_name(pair.draft_repo.split(':').next().unwrap_or(""));
            store.get_model(&draft_name).ok().flatten().map(|r| r.path)
        });

        // Capacity: evict the COLDEST instance first — recency-weighted
        // prefix heat (hot models keep their warm cache across capacity
        // pressure; the radix-lite lane), ties broken by longest-idle.
        let cap = self.capacity_for(model.bytes);
        while self.instances.len() >= cap {
            let victim = self
                .instances
                .iter()
                .filter(|e| e.in_flight.load(Ordering::SeqCst) <= 0 && e.key() != key)
                .min_by_key(|e| {
                    (
                        self.heat_of(e.key()),
                        *e.last_used.read().expect("idle lock"),
                    )
                })
                .map(|e| e.key().clone());
            match victim {
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
        let data_dir_str = self.dirs.data_dir.to_string_lossy().into_owned();
        // Retry once on immediate port-race death (bind fail).
        for _attempt in 0..2 {
            let endpoint = self.pick_endpoint(key);
            let input = ProfileInput {
                model_name: name,
                // Per-replica paths: the argv's sessions/speccache dirs
                // must match the dir created for THIS instance key.
                instance_key: key,
                model_path: &model.path,
                model_bytes: u64::try_from(model.bytes.max(0)).unwrap_or(u64::MAX),
                gguf: &gguf,
                hardware: &self.hardware,
                config: &self.config,
                overlay: &overlay,
                loras: &loras,
                draft_path: draft_path.as_deref(),
                mmproj_path: model.mmproj_path.as_deref(),
                engine_tag: &manifest.tag,
                supported_flags: &manifest.flags,
                endpoint: endpoint.clone(),
                data_dir: &data_dir_str,
                cache_hit_rate: self.cache_hint.get(),
            };
            let mut tuning = self
                .pending_ctx
                .remove(name)
                .map(|(_, ctx)| pallama_core::TuningOverrides {
                    ctx: Some(ctx),
                    ..Default::default()
                })
                .unwrap_or_default();
            // Co-residency planner (A15): when other models are already
            // resident, sum their (weights + post-quant KV) with the
            // candidate's f16 KV; if the box would not fit, downgrade the
            // candidate to q8_0 KV before spawning instead of OOMing mid-
            // load. Explicit kv_quant (bench-adopted) is never overridden.
            if tuning.kv_quant.is_none() {
                let candidate_kv = pallama_core::profile::estimate_kv_f16(&input, tuning.ctx);
                if self.coresidency_needs_kv_quant(model_bytes, candidate_kv) {
                    tracing::warn!(model = name, "co-residency planner: KV downgraded to q8_0 to fit alongside resident models (weights+KV exceed VRAM)");
                    tuning.kv_quant = Some(true);
                }
            }
            let profile = profile::compile(&input, &tuning)
                .map_err(|e| SupervisionError::Internal(anyhow!("profile: {e}")))?;
            for w in &profile.warnings {
                tracing::warn!(model = name, "profile: {w}");
            }
            let argv = self.engine.build_argv(&model, &profile, &endpoint);
            let mut child = self
                .engine
                .spawn(&argv, &endpoint)
                .await
                .map_err(|e| SupervisionError::Internal(anyhow!("{e}")))?;

            // Child died instantly (port race)? Retry with a fresh port.
            if let Ok(Some(status)) = child.try_status() {
                tracing::warn!(model = name, "engine died at spawn ({status}); retrying");
                let _ = child.kill().await;
                let _ = child.reap().await;
                continue;
            }

            match self.engine.health_check(&endpoint, self.load_timeout).await {
                Ok(()) => {
                    // NEVER default the pid: a 0 here would later target
                    // process group 0 (the whole session) on teardown.
                    let pid = child.id().ok_or_else(|| {
                        SupervisionError::Internal(anyhow!(
                            "engine child has no pid; refusing to track instance"
                        ))
                    })?;
                    if pid <= 1 {
                        return Err(SupervisionError::Internal(anyhow!(
                            "engine child pid {pid} is not a safe process-group id"
                        )));
                    }
                    let inst = Arc::new(Instance {
                        name: key.to_string(),
                        endpoint: endpoint.clone(),
                        state: std::sync::RwLock::new(InstanceState::Ready),
                        last_used: std::sync::RwLock::new(Instant::now()),
                        in_flight: AtomicI64::new(0),
                        started_at: Instant::now(),
                        argv,
                        model,
                        profile_ctx: profile.ctx,
                        gpu: profile.gpu.to_string(),
                        kv_est_bytes: profile.kv_est_bytes,
                        child: tokio::sync::Mutex::new(child),
                        pid,
                    });
                    // Bank restore, choke point #2: warm KV before the
                    // first request prefills (ctx-matched only). Banks
                    // are per-replica (key-scoped files).
                    self.bank_restore(key, &endpoint, profile.ctx).await;
                    let _ = std::fs::write(
                        self.dirs.run_dir().join(format!("{key}.pid")),
                        pid.to_string(),
                    );
                    self.instances.insert(key.to_string(), inst);
                    self.record_restart(key);
                    self.bus.publish(PallamaEvent::InstanceStateChanged {
                        name: key.to_string(),
                        state: InstanceState::Ready,
                    });
                    let inst = self.instances.get(key).expect("just inserted");
                    return Ok(self.engine_ref(inst.value()));
                }
                Err(e) => {
                    // Health never came up: kill and retry once (port race),
                    // then surface a load-timeout failure.
                    let _ = child.kill().await;
                    let _ = child.reap().await;
                    tracing::warn!(model = name, "health check failed: {e:#}");
                }
            }
        }
        Err(SupervisionError::ModelLoadTimeout(key.to_string()))
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

    /// Kill an instance: SIGTERM process group → grace → SIGKILL.
    /// Idempotent; publishes Evicted.
    pub async fn evict(&self, name: &str) -> Result<()> {
        let Some(inst) = self.instances.get(name).map(|i| i.clone()) else {
            return Ok(());
        };
        // Session bank, choke point #1 (FIX1): EVERY eviction path —
        // gateway requests, idle ladder, capacity pressure — flows
        // through here, so the `_auto-<ctx>` checkpoint is saved exactly
        // once, bounded, and only for idle instances (live requests own
        // their KV).
        self.bank_save(&inst).await;
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
        let _ = std::fs::remove_file(self.dirs.run_dir().join(format!("{name}.pid")));
        self.instances.remove(name);
        Ok(())
    }

    /// Model-level evict (B1): stop the model AND all its replicas
    /// (`qwen`, `qwen#1`, `qwen#2`, …). This is the user-facing stop
    /// (gateway /api/stop, CLI); internal paths (reaper, capacity)
    /// call [`Supervisor::evict`] with one exact key.
    pub async fn evict_model(&self, model: &str) -> Result<()> {
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
        let ok = reqwest::Client::new()
            .post(&url)
            .json(&body)
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await
            .is_ok_and(|r| r.status().is_success());
        if ok {
            // Hoard guard: a per-model bank over 512 MiB is storage
            // abuse, not a cache — drop it and say so once.
            if let Ok(md) = std::fs::metadata(&file) {
                if md.len() > 512 * 1024 * 1024 {
                    let _ = std::fs::remove_file(&file);
                    tracing::warn!(target: "pallama::bank", model = %inst.name, "banked checkpoint > 512 MiB — dropped");
                }
            }
        }
    }

    /// Bank restore, choke point #2: after a fresh spawn reaches
    /// readiness, a matching `_auto-<ctx>` is restored so the first
    /// request rides warm KV. Warn-continue on any failure.
    async fn bank_restore(&self, name: &str, endpoint: &Endpoint, ctx: u32) {
        if !self.config.session_bank {
            return;
        }
        let file = self.bank_file(name, ctx);
        if !file.exists() {
            return;
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
        match reqwest::Client::new()
            .post(&url)
            .json(&body)
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await
        {
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
        let mut evictions: Vec<String> = Vec::new();
        for entry in &self.instances {
            let inst = entry.value();
            if inst.in_flight.load(Ordering::SeqCst) > 0 {
                continue; // never evict or sleep-mark under load
            }
            let idle = now.duration_since(*inst.last_used.read().expect("idle lock"));
            let state = *inst.state.read().expect("state lock");
            if idle >= Duration::from_secs(self.config.idle_timeout_secs) {
                evictions.push(inst.name.clone());
            } else if idle >= Duration::from_secs(self.config.idle_sleep_secs)
                && state == InstanceState::Ready
                && self.hardware.has_gpu()
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
            i.in_flight.fetch_sub(1, Ordering::SeqCst);
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
                    pid: i.pid,
                    bytes: i.model.bytes,
                    heat,
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
        let mut crashed: Vec<String> = Vec::new();
        for entry in &self.instances {
            let inst = entry.value();
            let mut child = inst.child.lock().await;
            match child.try_status() {
                Ok(Some(status)) => {
                    tracing::error!(
                        model = %inst.name,
                        "engine crashed ({status}); next request will respawn"
                    );
                    crashed.push(inst.name.clone());
                }
                Ok(None) => {}
                Err(e) => tracing::warn!(model = %inst.name, "child poll: {e}"),
            }
        }
        for name in crashed {
            let _ = std::fs::remove_file(self.dirs.run_dir().join(format!("{name}.pid")));
            self.instances.remove(&name);
            self.bus.publish(PallamaEvent::InstanceStateChanged {
                name,
                state: InstanceState::Crashed,
            });
        }
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
            tracing::warn!(pid, "engine group did not exit within grace; SIGKILL");
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
            started_at: Instant::now(),
            argv: vec![],
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
            kv_est_bytes: None,
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
    async fn unit__model_of_key__plain_and_replica() {
        assert_eq!(model_of_key("m"), "m");
        assert_eq!(model_of_key("m#3"), "m");
        assert_eq!(model_of_key("m#3#4"), "m"); // first '#' wins
    }

    #[tokio::test]
    async fn unit__split_replica__parse_and_reject() {
        assert_eq!(split_replica("m"), None);
        assert_eq!(split_replica("m#2"), Some(("m", 2)));
        assert_eq!(split_replica("m#0"), Some(("m", 0)));
        assert_eq!(split_replica("m#x"), None); // non-numeric suffix
        assert_eq!(split_replica("m#"), None); // empty suffix
    }

    #[tokio::test]
    async fn unit__replica_key__legacy_no_overlay__plain_name() {
        let sup = routing_sup(1);
        assert_eq!(sup.replica_key("m", None), "m");
        assert_eq!(sup.replica_key("m", Some(7)), "m");
    }

    #[tokio::test]
    async fn unit__replica_key__router_key__passthrough() {
        let sup = routing_sup(4);
        assert_eq!(sup.replica_key(ROUTER_KEY, Some(7)), ROUTER_KEY);
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
        assert_eq!(sup.replica_key("m", Some(42)), "m#2");
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
        assert_eq!(sup.replica_key("m", Some(42)), "m#2");
        kill_all(&[pa]);
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
}
