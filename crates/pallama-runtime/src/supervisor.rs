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
    child: tokio::sync::Mutex<ChildHandle>,
    pid: u32,
}

/// Row for `pallama ps` / `/api/ps`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PsRow {
    pub name: String,
    pub state: &'static str,
    pub endpoint: String,
    pub idle_secs: u64,
    pub in_flight: i64,
    pub ctx: u32,
    pub pid: u32,
    /// Model bytes on disk (0 in router mode — the front child serves
    /// many models and owns no single size).
    pub bytes: i64,
}

pub struct Supervisor {
    pub dirs: PallamaDirs,
    /// Total evictions (ladder + capacity) for the metrics gauge.
    pub evictions: std::sync::atomic::AtomicU64,
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

    /// Ensure a model is loaded and ready; returns its endpoint. Waits on
    /// a concurrent load instead of double-spawning.
    pub async fn ensure(&self, name: &str) -> Result<EngineRef, SupervisionError> {
        // Router mode: every model name resolves to the ONE router child
        // (upstream autoloads the model on request, LRU-evicts at
        // models-max). Unknown names still fail fast against the store.
        if self.config.router && name != ROUTER_KEY {
            let store = Store::open(&self.dirs)
                .map_err(|e| SupervisionError::Internal(anyhow!("{e}")))?;
            store
                .get_model(name)
                .map_err(|e| SupervisionError::Internal(anyhow!("{e}")))?
                .ok_or_else(|| SupervisionError::ModelNotFound(name.to_string()))?;
            return Box::pin(self.ensure(ROUTER_KEY)).await;
        }

        // Fast path: running (or sleeping — the child wakes on traffic).
        if let Some(inst) = self.instances.get(name) {
            let snapshot = (*inst.state.read().expect("state lock"), inst.last_used.read().expect("idle lock").elapsed());
            if matches!(snapshot.0, InstanceState::Ready | InstanceState::Sleeping) {
                let was_sleeping = snapshot.0 == InstanceState::Sleeping;
                let live = inst.clone();
                drop(inst);
                *live.last_used.write().expect("idle lock") = Instant::now();
                if was_sleeping {
                    *live.state.write().expect("state lock") = InstanceState::Ready;
                    self.bus.publish(PallamaEvent::InstanceStateChanged {
                        name: name.to_string(),
                        state: InstanceState::Ready,
                    });
                }
                return Ok(self.engine_ref(&live));
            }
        }

        // Circuit breaker.
        if let Some(ts) = self.restarts.get(name) {
            let recent = ts
                .iter()
                .filter(|t| t.elapsed() < self.circuit_window)
                .count();
            if recent > self.max_restarts {
                return Err(SupervisionError::CircuitOpen(name.to_string()));
            }
        }

        // Join an in-flight load.
        if let Some(existing) = self.loading.get(name).map(|l| l.clone()) {
            drop(existing);
            let notify = self.loading.get(name).map(|l| l.clone());
            if let Some(n) = notify {
                let notified = n.notified();
                // Re-check results after registering interest.
                if let Some(res) = self.load_results.get(name) {
                    return clone_load_result(res.value(), name);
                }
                notified.await;
                if let Some(res) = self.load_results.get(name) {
                    return clone_load_result(res.value(), name);
                }
                // Loader finished between checks.
                if let Some(inst) = self.instances.get(name) {
                    return Ok(self.engine_ref(inst.value()));
                }
                return Err(SupervisionError::Internal(anyhow!(
                    "load of {name} vanished without result"
                )));
            }
        }

        // We are the loader.
        let notify = Arc::new(Notify::new());
        self.loading.insert(name.to_string(), notify.clone());
        let result = if self.config.router && name == ROUTER_KEY {
            self.spawn_router_instance().await
        } else {
            self.spawn_instance(name).await
        };
        // Waiters get a cloneable copy; the loader returns the typed error
        // itself (callers match on ModelLoadTimeout / CircuitOpen / ...).
        self.load_results
            .insert(name.to_string(), map_result(&result));
        notify.notify_waiters();
        // Give waiters a moment to drain, then clear loading state.
        tokio::time::sleep(Duration::from_millis(50)).await;
        self.loading.remove(name);
        self.load_results.remove(name);
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
        let store = Store::open(&self.dirs).map_err(|e| SupervisionError::Internal(anyhow!("{e}")))?;
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
                endpoint: Endpoint::Tcp { host: "127.0.0.1".into(), port: 0 },
                data_dir: &data_dir_str,
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
        std::fs::write(&ini_path, &ini)
            .map_err(|e| SupervisionError::Internal(anyhow!("write {}: {e}", ini_path.display())))?;
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
                    argv.extend(["--host".into(), host.clone(), "--port".into(), port.to_string()]);
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
                tracing::warn!(router = ROUTER_KEY, "router died at spawn ({status}); retrying");
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
    async fn spawn_instance(&self, name: &str) -> Result<EngineRef, SupervisionError> {
        let store = Store::open(&self.dirs).map_err(|e| SupervisionError::Internal(anyhow!("{e}")))?;
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
            let draft_name = crate::hf::registry_name(pair.draft_repo.split(':').next().unwrap_or(""));
            store.get_model(&draft_name).ok().flatten().map(|r| r.path)
        });

        // Capacity: evict the longest-idle non-inflight instance first.
        let cap = self.capacity_for(model.bytes);
        while self.instances.len() >= cap {
            let victim = self
                .instances
                .iter()
                .filter(|e| e.in_flight.load(Ordering::SeqCst) <= 0 && e.key() != name)
                .min_by_key(|e| *e.last_used.read().expect("idle lock"))
                .map(|e| e.key().clone());
            match victim {
                Some(v) => {
                    self.evict(&v).await.map_err(|e| SupervisionError::Internal(anyhow!("{e}")))?;
                }
                None => return Err(SupervisionError::AllSlotsBusy),
            }
        }

        let manifest = self.engine.capabilities();
        // Cache-file dirs (speccache/, sessions/) must exist before the
        // child opens them; profile emission names these paths. Upstream
        // validates --slot-save-path IS a directory, so the per-model
        // subdir must pre-exist.
        let _ = self.dirs.ensure();
        let _ = std::fs::create_dir_all(
            self.dirs.sessions_dir().join(pallama_core::profile::path_safe(name)),
        );
        let data_dir_str = self.dirs.data_dir.to_string_lossy().into_owned();
        // Retry once on immediate port-race death (bind fail).
        for _attempt in 0..2 {
            let endpoint = self.pick_endpoint(name);
            let input = ProfileInput {
                model_name: name,
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
            };
            let tuning = self
                .pending_ctx
                .remove(name)
                .map(|(_, ctx)| pallama_core::TuningOverrides { ctx: Some(ctx), ..Default::default() })
                .unwrap_or_default();
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
                        name: name.to_string(),
                        endpoint,
                        state: std::sync::RwLock::new(InstanceState::Ready),
                        last_used: std::sync::RwLock::new(Instant::now()),
                        in_flight: AtomicI64::new(0),
                        started_at: Instant::now(),
                        argv,
                        model,
                        profile_ctx: profile.ctx,
                        child: tokio::sync::Mutex::new(child),
                        pid,
                    });
                    let _ = std::fs::write(self.dirs.run_dir().join(format!("{name}.pid")), pid.to_string());
                    self.instances.insert(name.to_string(), inst);
                    self.record_restart(name);
                    self.bus.publish(PallamaEvent::InstanceStateChanged {
                        name: name.to_string(),
                        state: InstanceState::Ready,
                    });
                    let inst = self.instances.get(name).expect("just inserted");
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
        Err(SupervisionError::ModelLoadTimeout(name.to_string()))
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
        Endpoint::Tcp { host: "127.0.0.1".into(), port }
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
                PsRow {
                    name: i.name.clone(),
                    state: i.state.read().expect("state lock").as_str(),
                    endpoint: match &i.endpoint {
                        Endpoint::Tcp { host, port } => format!("{host}:{port}"),
                        Endpoint::Unix { socket } => socket.clone(),
                    },
                    idle_secs: i.last_used.read().expect("idle lock").elapsed().as_secs(),
                    in_flight: i.in_flight.load(Ordering::SeqCst),
                    ctx: i.profile_ctx,
                    pid: i.pid,
                    bytes: i.model.bytes,
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
            let Ok(content) = std::fs::read_to_string(e.path()) else { continue };
            let Ok(pid) = content.trim().parse::<u32>() else { continue };
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
async fn terminate_group(
    pid: u32,
    grace: Duration,
    child: &mut ChildHandle,
) -> Result<()> {
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
