//! The `Engine` abstraction: everything Pallama needs from a serving
//! backend. Implementations: `LlamaCppEngine` (upstream llama-server)
//! and `MistralRsEngine` (upstream mistralrs `serve`). vLLM/SGLang
//! adapters later implement the same trait — gateway, supervisor and
//! lifecycle stay engine-agnostic.

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;

use pallama_core::profile::{Endpoint, Profile};

/// Ring buffer of the child's last output lines (stdout + stderr share it),
/// so a child that dies mid-load can be diagnosed from its final words.
pub type LogTail = std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<String>>>;

const TAIL_CAP: usize = 32;

fn new_tail() -> LogTail {
    std::sync::Arc::new(std::sync::Mutex::new(std::collections::VecDeque::new()))
}

/// A spawned engine child process: how to reach it and how to stop it.
#[derive(Debug)]
pub struct ChildHandle {
    pub endpoint: Endpoint,
    child: tokio::process::Child,
    log_tail: LogTail,
}

impl ChildHandle {
    #[must_use]
    pub fn new(endpoint: Endpoint, child: tokio::process::Child) -> Self {
        Self {
            endpoint,
            child,
            log_tail: new_tail(),
        }
    }

    /// `new` with a pre-created tail (spawn wires the pipe tasks into it).
    #[must_use]
    pub fn with_tail(endpoint: Endpoint, child: tokio::process::Child, log_tail: LogTail) -> Self {
        Self {
            endpoint,
            child,
            log_tail,
        }
    }

    /// Last child output lines, oldest first, joined for one-line logs.
    #[must_use]
    pub fn tail_joined(&self) -> String {
        let tail = self.log_tail.lock().expect("log tail lock");
        tail.iter().cloned().collect::<Vec<_>>().join(" | ")
    }

    pub async fn wait(&mut self) -> std::io::Result<std::process::ExitStatus> {
        self.child.wait().await
    }

    /// Non-blocking exit check.
    pub fn try_status(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        self.child.try_wait()
    }

    /// Hard kill (teardown last resort).
    pub async fn kill(&mut self) -> std::io::Result<()> {
        self.child.kill().await
    }

    /// Reap the exited process (alias of `wait`, kept for teardown
    /// call-site readability).
    pub async fn reap(&mut self) -> std::io::Result<std::process::ExitStatus> {
        self.wait().await
    }

    #[must_use]
    pub fn id(&self) -> Option<u32> {
        self.child.id()
    }
}

/// One running instance of a model, as the supervisor sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineEndpoint {
    pub host: String,
    pub port: u16,
}

#[async_trait]
pub trait Engine: Send + Sync {
    /// Capabilities of the active engine build (manifest).
    fn capabilities(&self) -> &crate::engine::manifest::Manifest;

    /// Dialect selector for the profile compiler: the engines.kind
    /// column's in-process mirror. No kind lives in the manifest —
    /// the store row is the single source of truth (b6 rev 1).
    fn kind(&self) -> pallama_core::engine_kind::EngineKind;

    /// Build the argv (without the binary) for one model instance.
    fn build_argv(
        &self,
        model: &pallama_core::ModelRow,
        profile: &Profile,
        endpoint: &Endpoint,
    ) -> Vec<String>;

    /// Spawn a child for the compiled argv. Implementations own process
    /// group setup and log piping.
    async fn spawn(&self, argv: &[String], endpoint: &Endpoint) -> Result<ChildHandle>;

    /// Poll the child's /health until {"status":"ok"} or timeout.
    async fn health_check(&self, endpoint: &Endpoint, timeout: std::time::Duration) -> Result<()>;

    /// Devices the SERVING child context can see (`--list-devices` run
    /// exactly as a serving spawn would). `Ok(None)` = engine build has
    /// no `--list-devices` or listing failed — callers skip validation.
    async fn enumerate_devices(&self) -> Result<Option<Vec<crate::engine::manifest::DeviceDesc>>> {
        Ok(None)
    }
}

/// llama.cpp llama-server engine.
pub struct LlamaCppEngine {
    pub manifest: crate::engine::manifest::Manifest,
    /// HTTP client for health polls (children are loopback).
    http: reqwest::Client,
    /// Extra env injected into children (tests: STUB_* knobs).
    pub child_env: Vec<(String, String)>,
}

impl LlamaCppEngine {
    #[must_use]
    pub fn new(manifest: crate::engine::manifest::Manifest) -> Self {
        Self::with_env(manifest, Vec::new())
    }

    /// `env` pairs apply to every spawned child (config `engine_env`).
    #[must_use]
    pub fn with_env(
        manifest: crate::engine::manifest::Manifest,
        env: Vec<(String, String)>,
    ) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .expect("health client");
        Self {
            manifest,
            http,
            child_env: env,
        }
    }

    fn base_url(endpoint: &Endpoint) -> String {
        match endpoint {
            Endpoint::Tcp { host, port } => format!("http://{host}:{port}"),
            Endpoint::Unix { .. } => String::new(), // unix sockets: probe skipped
        }
    }
}

#[async_trait]
impl Engine for LlamaCppEngine {
    fn kind(&self) -> pallama_core::engine_kind::EngineKind {
        pallama_core::engine_kind::EngineKind::LlamaCpp
    }

    fn capabilities(&self) -> &crate::engine::manifest::Manifest {
        &self.manifest
    }

    fn build_argv(
        &self,
        _model: &pallama_core::ModelRow,
        profile: &Profile,
        endpoint: &Endpoint,
    ) -> Vec<String> {
        // The profile already carries the full argv for this endpoint
        // (it was compiled with the endpoint baked in).
        let _ = endpoint;
        profile.argv.clone()
    }

    async fn spawn(&self, argv: &[String], endpoint: &Endpoint) -> Result<ChildHandle> {
        // Preflight --rpc: upstream llama-server connects RPC backends
        // EAGERLY at argv-parse and SIGABRTs on a dead endpoint
        // (ggml-rpc.cpp rpc_dispatcher::start), which crash-loops the
        // child into 502s. Refuse the spawn with a teaching error naming
        // the dead endpoint(s) instead — the engine binary itself is
        // healthy, so this returns before any child exists (no
        // crash-loop, no engine rollback).
        let dead = probe_rpc_endpoints(argv).await;
        if !dead.is_empty() {
            return Err(anyhow!(
                "--rpc endpoint(s) unreachable: {} — llama-server aborts at \
                 startup when an RPC backend is down; start the RPC worker(s) \
                 or fix `rpc_servers` in the config",
                dead.join(", ")
            ));
        }
        spawn_child(&self.manifest.server_path, argv, endpoint, &self.child_env)
    }

    async fn enumerate_devices(&self) -> Result<Option<Vec<crate::engine::manifest::DeviceDesc>>> {
        if !self.manifest.flags.iter().any(|f| f == "--list-devices") {
            return Ok(None);
        }
        let mut cmd = tokio::process::Command::new(&self.manifest.server_path);
        cmd.arg("--list-devices")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null());
        for (k, v) in &self.child_env {
            cmd.env(k, v);
        }
        let listed = tokio::time::timeout(std::time::Duration::from_secs(10), cmd.output()).await;
        match listed {
            Ok(Ok(out)) => {
                let text = String::from_utf8_lossy(&out.stdout).to_string();
                let devs = crate::engine::manifest::parse_devices(&text);
                if devs.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(devs))
                }
            }
            // Listing failure must never block serving: skip validation.
            Ok(Err(_)) | Err(_) => Ok(None),
        }
    }

    async fn health_check(&self, endpoint: &Endpoint, timeout: std::time::Duration) -> Result<()> {
        let url = Self::base_url(endpoint);
        if url.is_empty() {
            return Ok(()); // unix transport: health via socket handled by caller
        }
        let deadline = tokio::time::Instant::now() + timeout;
        let started = std::time::Instant::now();
        // Adaptive poll: fast at first (the child usually turns healthy
        // within a few hundred ms of its socket opening), backing off to
        // the steady 150ms cadence for long loads. Cuts the average
        // post-ready overshoot from ~75ms to ~12ms on cold first token.
        let mut poll = std::time::Duration::from_millis(25);
        loop {
            match self.http.get(format!("{url}/health")).send().await {
                Ok(resp) if resp.status().is_success() => {
                    let body: serde_json::Value = resp.json().await.unwrap_or_default();
                    if body["status"] == "ok" {
                        tracing::debug!("engine healthy at {url} after {:?}", started.elapsed());
                        return Ok(());
                    }
                }
                _ => {}
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(anyhow!(
                    "engine at {url} did not become healthy within {timeout:?} (model_load_timeout)"
                ));
            }
            tokio::time::sleep(poll).await;
            poll = std::cmp::min(poll.mul_f64(1.6), std::time::Duration::from_millis(150));
        }
    }
}

/// How long each `--rpc` endpoint preflight probe may take. Probes run
/// in parallel; a live loopback/LAN worker answers in well under a
/// second, while a dead host with dropped SYNs burns the full budget
/// once — still cheaper than the child SIGABRT loop it prevents.
const RPC_PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

/// Preflight every `--rpc` endpoint in the FINAL child argv (all
/// occurrences — config knob, model overlay and `extra_args` can each
/// add one). TCP-connect each in parallel; a failed connect means
// upstream's eager connect would SIGABRT the child at argv-parse.
/// Returns the offending entries (`host:port (reason)`), empty when
/// every endpoint answered or no `--rpc` flag is present.
async fn probe_rpc_endpoints(argv: &[String]) -> Vec<String> {
    let targets: Vec<String> = argv
        .windows(2)
        .filter(|w| w[0] == "--rpc")
        .flat_map(|w| w[1].split(','))
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    let probes = targets.into_iter().map(|target| async move {
        let (host, port) = match target.rsplit_once(':') {
            Some((h, p)) if !h.is_empty() => match p.parse::<u16>() {
                Ok(p) => (h.to_string(), p),
                Err(_) => return Some(format!("{target} (malformed port)")),
            },
            _ => return Some(format!("{target} (malformed host:port)")),
        };
        let connected = tokio::time::timeout(
            RPC_PROBE_TIMEOUT,
            tokio::net::TcpStream::connect((host.as_str(), port)),
        )
        .await;
        if matches!(connected, Ok(Ok(_))) {
            None
        } else {
            Some(format!("{target} (unreachable)"))
        }
    });
    futures::future::join_all(probes)
        .await
        .into_iter()
        .flatten()
        .collect()
}

/// Shared child-process mechanics for every engine kind: null stdin,
/// piped stdio into tracing + the log tail, kill-on-drop, own process
/// group (§5 H19: acquired = released by construction).
fn spawn_child(
    server_path: &str,
    argv: &[String],
    endpoint: &Endpoint,
    child_env: &[(String, String)],
) -> Result<ChildHandle> {
    let mut cmd = tokio::process::Command::new(server_path);
    cmd.args(argv)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        // If the owning process dies without teardown, the child must
        // not linger (test leakage, daemon crash).
        .kill_on_drop(true);
    #[cfg(unix)]
    {
        // Own process group: Ctrl-C on the daemon never reaches the
        // child. Termination is single-pid by design (terminate()
        // signals only this pid; never a group signal).
        cmd.process_group(0);
    }
    for (k, v) in child_env {
        cmd.env(k, v);
    }
    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawn {server_path}"))?;

    // Pipe child logs into tracing; the shared tail keeps the last lines
    // around for death diagnostics.
    let tail = new_tail();
    if let Some(stdout) = child.stdout.take() {
        let t = tail.clone();
        tokio::spawn(pipe_logs(stdout, "stdout", t));
    }
    if let Some(stderr) = child.stderr.take() {
        let t = tail.clone();
        tokio::spawn(pipe_logs(stderr, "stderr", t));
    }
    Ok(ChildHandle::with_tail(endpoint.clone(), child, tail))
}

/// mistral.rs engine (`mistralrs serve`). Dialect differences from
/// llama.cpp that shaped this impl (verified against mistral.rs docs):
/// - no `--list-devices` equivalent (device enumeration unavailable)
/// - `/health` answers 200 as soon as HTTP is up, BEFORE the model is
///   served — readiness MUST come from `/v1/models` (per-model status
///   "loaded")
/// - child binds `0.0.0.0` by default and has NO auth; Pallama forces
///   `--host 127.0.0.1` and the gateway stays the only public face
/// - no unix-socket transport: endpoints are always TCP.
pub struct MistralRsEngine {
    pub manifest: crate::engine::manifest::Manifest,
    /// HTTP client for health polls (children are loopback).
    http: reqwest::Client,
    /// Extra env injected into children (config `engine_env`).
    pub child_env: Vec<(String, String)>,
    /// Per-model staging roots (`<run_dir>/mistralrs-staging`). mistralrs
    /// discovers projectors/draft GGUFs by scanning the model file's
    /// whole directory; Pallama's flat shared models dir would feed it
    /// every OTHER model's projector (observed live: a stray mmproj
    /// force-fed a multimodal load that crashed on qwen2). Staging gives
    /// each child a directory containing only its own files. `None` =
    /// no staging (argv uses raw store paths; unit-test convenience).
    staging_root: Option<std::path::PathBuf>,
}

impl MistralRsEngine {
    #[must_use]
    pub fn new(manifest: crate::engine::manifest::Manifest) -> Self {
        Self::with_env(manifest, Vec::new())
    }

    /// `env` pairs apply to every spawned child (config `engine_env`).
    #[must_use]
    pub fn with_env(
        manifest: crate::engine::manifest::Manifest,
        env: Vec<(String, String)>,
    ) -> Self {
        Self::with_staging(manifest, env, None)
    }

    /// Production constructor: `staging_root` is
    /// `<run_dir>/mistralrs-staging`.
    #[must_use]
    pub fn with_staging(
        manifest: crate::engine::manifest::Manifest,
        env: Vec<(String, String)>,
        staging_root: Option<std::path::PathBuf>,
    ) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .expect("health client");
        Self {
            manifest,
            http,
            child_env: env,
            staging_root,
        }
    }

    /// Build the per-model view directory (symlink farm): the model's
    /// shard set plus its own projector, nothing else. Returns
    /// (`view_dir`, `staged_mmproj`); falls back to raw paths on any fs
    /// error by returning the un-staged originals.
    fn stage_model_view(
        &self,
        model: &pallama_core::ModelRow,
    ) -> (std::path::PathBuf, Option<std::path::PathBuf>) {
        let raw_mmproj = model.mmproj_path.as_deref().map(std::path::PathBuf::from);
        // HF-style directory rows (safetensors trees) serve from their raw
        // location: the shard-view farm below stages only *.gguf siblings,
        // so a directory row would produce an empty view and hand the
        // engine a nonexistent path.
        if std::path::Path::new(&model.path).is_dir() {
            return (std::path::PathBuf::from(&model.path), raw_mmproj);
        }
        let Some(root) = self.staging_root.as_deref() else {
            return (std::path::PathBuf::from(&model.path), raw_mmproj);
        };
        let Some(file_name) = std::path::Path::new(&model.path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
        else {
            return (std::path::PathBuf::from(&model.path), raw_mmproj);
        };
        let view = root.join(sanitize_dir_name(&model.name));
        let Some(src_dir) = std::path::Path::new(&model.path).parent() else {
            return (std::path::PathBuf::from(&model.path), raw_mmproj);
        };
        // Rebuild from scratch each spawn: replicas race benignly (same
        // content) and deleted source files cannot linger as stale links.
        if std::fs::remove_dir_all(&view).is_ok() || !view.exists() {
            if std::fs::create_dir_all(&view).is_err() {
                return (std::path::PathBuf::from(&model.path), raw_mmproj);
            }
        } else {
            return (std::path::PathBuf::from(&model.path), raw_mmproj);
        }
        let base = split_shard_base(&file_name);
        let mut staged_mmproj = None;
        let Ok(entries) = std::fs::read_dir(src_dir) else {
            return (std::path::PathBuf::from(&model.path), raw_mmproj);
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !shard_of(&name, &base) {
                continue;
            }
            if link_entry(&entry.path(), &view.join(&name)).is_err() {
                return (std::path::PathBuf::from(&model.path), raw_mmproj);
            }
        }
        if let Some(mm) = raw_mmproj.as_deref() {
            let Some(mm_name) = mm.file_name().map(|n| n.to_string_lossy().into_owned()) else {
                return (view, None);
            };
            let target = view.join(mm_name);
            if link_entry(mm, &target).is_ok() {
                staged_mmproj = Some(target);
            }
        }
        (view.join(&file_name), staged_mmproj)
    }
}

/// Filesystem link for staging: symlink on unix; Windows falls back
/// hardlink -> copy (symlinks need privileges there).
fn link_entry(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(src, dst)
    }
    #[cfg(windows)]
    {
        let _ = std::fs::remove_file(dst);
        if std::fs::hard_link(src, dst).is_ok() {
            return Ok(());
        }
        std::fs::copy(src, dst).map(|_| ())
    }
}

/// Directory-safe model name (store names are unique per model).
fn sanitize_dir_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '.' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Strip a split-GGUF `-00001-of-00002` suffix from a file stem so the
/// whole shard set shares one base (`x-00001-of-00003.gguf` ->
/// `x`). Non-split names pass through unchanged.
fn split_shard_base(file_name: &str) -> String {
    let stem = file_name.strip_suffix(".gguf").unwrap_or(file_name);
    let Some(of) = stem.rfind("-of-") else {
        return stem.to_string();
    };
    let (head, total) = (&stem[..of], &stem[of + 4..]);
    let is_digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if !is_digits(total) {
        return stem.to_string();
    }
    if let Some(dash) = head.rfind('-') {
        if is_digits(&head[dash + 1..]) {
            return head[..dash].to_string();
        }
    }
    stem.to_string()
}

/// Is `name` part of the shard set identified by `base`? Matches the
/// exact single file, or `<base>-<N>-of-<M>.gguf` siblings.
fn shard_of(name: &str, base: &str) -> bool {
    let stem = name.strip_suffix(".gguf").unwrap_or(name);
    if stem == base {
        return true;
    }
    stem.strip_prefix(&format!("{base}-")).is_some_and(|tail| {
        if let Some(of) = tail.find("-of-") {
            let (idx, total) = (&tail[..of], &tail[of + 4..]);
            !idx.is_empty()
                && idx.bytes().all(|b| b.is_ascii_digit())
                && !total.is_empty()
                && total.bytes().all(|b| b.is_ascii_digit())
        } else {
            false
        }
    })
}

/// Translate a compiled llama.cpp-style profile into `mistralrs serve`
/// argv. Only the knobs with a clean semantic mapping carry over
/// (model file, port, ctx, parallelism, mmproj); llama.cpp-only tuning
/// flags are intentionally dropped, not approximated.
#[must_use]
pub fn mistralrs_argv(
    model: &pallama_core::ModelRow,
    profile: &Profile,
    endpoint: &Endpoint,
    flags: &std::collections::BTreeSet<String>,
) -> Vec<String> {
    let port = match endpoint {
        Endpoint::Tcp { port, .. } => *port,
        // Supervisor rejects unix endpoints for mistralrs engines before
        // argv assembly; a placeholder here cannot produce a valid child.
        Endpoint::Unix { .. } => 0,
    };
    // mistral.rs dialect: GGUF rows ride `-f` (the filename suffix selects
    // GGUF/GGML); HF-style safetensors directories ride `-m/--model-id`,
    // which accepts a local model directory (v0.9.3 `serve --help`:
    // "Hugging Face model ID or local model directory"). A directory under
    // `-f` aborts the child: "Cannot infer model format from
    // `--quantized-file`".
    let is_hf_dir = std::path::Path::new(&model.path).is_dir();
    let mut argv = vec![
        "serve".to_string(),
        if is_hf_dir { "-m" } else { "-f" }.to_string(),
        model.path.clone(),
        // Forced loopback: the child is unauthenticated by design; the
        // gateway is the only public face (binding revision 3).
        "--host".to_string(),
        "127.0.0.1".to_string(),
        "--port".to_string(),
        port.to_string(),
        "--no-ui".to_string(),
    ];
    // `--max-model-len` is a GGUF-loader knob in mistral.rs (v0.9.3): the
    // HF/safetensors loader aborts with "max_model_len=N is not supported
    // by this model loader" and serves the model's native context from
    // config.json instead. Only GGUF rows carry the flag.
    if profile.ctx > 0 && !is_hf_dir && flags.contains("--max-model-len") {
        argv.push("--max-model-len".into());
        argv.push(profile.ctx.to_string());
    }
    // mistral.rs's paged-attention scheduler processes at most
    // `--max-num-batched-tokens` tokens per step and defaults to 4096:
    // any prompt longer than that dies as an HTTP 200 with an empty
    // choices array and `error: service_unavailable` (live-proven on
    // v0.9.3 — a real-world ~4.3k-token system prompt tripped it). Raise the
    // step to the serving context so prefill is never chunk-starved.
    if profile.ctx > 0 && flags.contains("--max-num-batched-tokens") {
        argv.push("--max-num-batched-tokens".into());
        argv.push(profile.ctx.max(4096).to_string());
    }
    // Parallelism: mine the compiled llama.cpp argv for `-np N` /
    // `--parallel=N` (the profile's slot count).
    let np = profile
        .argv
        .windows(2)
        .find(|w| w[0] == "-np")
        .and_then(|w| w[1].parse::<u32>().ok())
        .or_else(|| {
            profile.argv.iter().find_map(|a| {
                a.strip_prefix("--parallel=")
                    .and_then(|v| v.parse::<u32>().ok())
            })
        });
    if let Some(n) = np.filter(|_| flags.contains("--max-seqs")) {
        argv.push("--max-seqs".into());
        argv.push(n.to_string());
    }
    if let Some(mm) = model.mmproj_path.as_deref() {
        if flags.contains("--mmproj") {
            argv.push("--mmproj".into());
            argv.push(mm.to_string());
        }
    }
    // Engine-tuning passthrough: the mistral.rs profile dialect emits
    // tuning flags (pa-memory-fraction, paged-attn, scheduler/MTP/
    // vision/LoRA knobs, --lora pairs); forward them verbatim when the
    // binary advertises them. The profile compiler already warned when
    // the engine lacks one.
    for w in profile.argv.windows(2) {
        if (w[0] == "--pa-memory-fraction" && flags.contains("--pa-memory-fraction"))
            || (w[0] == "--paged-attn" && flags.contains("--paged-attn"))
        {
            argv.push(w[0].clone());
            argv.push(w[1].clone());
        }
    }
    mistralrs_tuning_passthrough(&profile.argv, flags, &mut argv);
    // User extra_args ride profile.argv after compile-time strict
    // validation (manifest-gated, reserved-flag refusal). Forward them
    // verbatim here — skipping everything this function owns or the
    // tuning passthrough already translated — with the spawn-time
    // manifest check as belt-and-braces against stale profiles.
    let owned: &[&str] = &[
        "-m",
        "-f",
        "--host",
        "--port",
        "--no-ui",
        "--max-model-len",
        "--max-num-batched-tokens",
        "--max-seqs",
        "--mmproj",
        "-np",
        "--paged-attn",
        "--pa-memory-fraction",
    ];
    for (i, tok) in profile.argv.iter().enumerate() {
        let t = tok.as_str();
        if !t.starts_with('-')
            || t.starts_with("--parallel=")
            || owned.contains(&t)
            || pallama_core::profile::MISTRALRS_TUNING_BOOL_FLAGS.contains(&t)
            || pallama_core::profile::MISTRALRS_TUNING_VALUE_FLAGS.contains(&t)
        {
            continue;
        }
        if flags.contains(t) {
            argv.push(t.to_string());
            if let Some(v) = profile.argv.get(i + 1) {
                if !v.starts_with('-') {
                    argv.push(v.clone());
                }
            }
        }
    }
    argv
}

/// Forward the dialect's tuning tokens from the compiled profile argv
/// into the child argv, manifest-gated per flag. Pair flags carry a
/// value token; bool flags stand alone. The flag lists live in
/// pallama-core next to `compile_mistralrs` — one source of truth.
fn mistralrs_tuning_passthrough(
    dialect: &[String],
    flags: &std::collections::BTreeSet<String>,
    out: &mut Vec<String>,
) {
    use pallama_core::profile::{MISTRALRS_TUNING_BOOL_FLAGS, MISTRALRS_TUNING_VALUE_FLAGS};
    let mut i = 0;
    while i < dialect.len() {
        let tok = dialect[i].as_str();
        if MISTRALRS_TUNING_BOOL_FLAGS.contains(&tok) {
            if flags.contains(tok) {
                out.push(tok.to_string());
            }
            i += 1;
        } else if MISTRALRS_TUNING_VALUE_FLAGS.contains(&tok) && i + 1 < dialect.len() {
            if flags.contains(tok) {
                out.push(tok.to_string());
                out.push(dialect[i + 1].clone());
            }
            i += 2;
        } else {
            i += 1;
        }
    }
}

#[async_trait]
impl Engine for MistralRsEngine {
    fn kind(&self) -> pallama_core::engine_kind::EngineKind {
        pallama_core::engine_kind::EngineKind::MistralRs
    }

    fn capabilities(&self) -> &crate::engine::manifest::Manifest {
        &self.manifest
    }

    fn build_argv(
        &self,
        model: &pallama_core::ModelRow,
        profile: &Profile,
        endpoint: &Endpoint,
    ) -> Vec<String> {
        let (staged_path, staged_mmproj) = self.stage_model_view(model);
        let mut staged = model.clone();
        staged.path = staged_path.to_string_lossy().into_owned();
        staged.mmproj_path = staged_mmproj.map(|p| p.to_string_lossy().into_owned());
        mistralrs_argv(&staged, profile, endpoint, &self.manifest.flags)
    }

    async fn spawn(&self, argv: &[String], endpoint: &Endpoint) -> Result<ChildHandle> {
        if matches!(endpoint, Endpoint::Unix { .. }) {
            return Err(anyhow!(
                "mistralrs engines have no unix-socket transport; set \
                 child_transport = \"tcp\" in the pallama config"
            ));
        }
        spawn_child(&self.manifest.server_path, argv, endpoint, &self.child_env)
    }

    async fn health_check(&self, endpoint: &Endpoint, timeout: std::time::Duration) -> Result<()> {
        let Endpoint::Tcp { host, port } = endpoint else {
            return Err(anyhow!("mistralrs engines require a TCP endpoint"));
        };
        let url = format!("http://{host}:{port}");
        let deadline = tokio::time::Instant::now() + timeout;
        let started = std::time::Instant::now();
        // Adaptive poll, mirroring the llamacpp health lane: fast first
        // probes, back off to 150ms steady state.
        let mut poll = std::time::Duration::from_millis(25);
        loop {
            if let Ok(resp) = self.http.get(format!("{url}/v1/models")).send().await {
                if resp.status().is_success() {
                    let body: serde_json::Value = resp.json().await.unwrap_or_default();
                    // mistralrs reports per-model status; "loaded" is the
                    // serving-ready state. Some builds omit the field —
                    // treat a listed model as loaded then.
                    let loaded = body["data"].as_array().is_some_and(|models| {
                        models
                            .iter()
                            .any(|m| m["status"].as_str().unwrap_or("loaded") == "loaded")
                    });
                    if loaded {
                        tracing::debug!("mistralrs healthy at {url} after {:?}", started.elapsed());
                        return Ok(());
                    }
                }
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(anyhow!(
                    "mistralrs at {url} did not report a loaded model on /v1/models \
                     within {timeout:?} (model_load_timeout)"
                ));
            }
            tokio::time::sleep(poll).await;
            poll = std::cmp::min(poll.mul_f64(1.6), std::time::Duration::from_millis(150));
        }
    }
}

/// sglang engine (`python -m sglang.launch_server` behind an executable
/// shim). Dialect differences that shaped this impl (verified against
/// sglang v0.5.19 `http_server.py` + `server_args.py`):
/// - `/health` and `/health_generate` share one handler: 503 while the
///   server is `Starting`, 200 once serving (generate-backed). Readiness
///   is the STATUS alone — there is no `{"status":"ok"}` body contract.
/// - no `--list-devices` equivalent; device enumeration unavailable.
/// - no unix-socket transport: endpoints are always TCP.
/// - auth exists upstream (`--api-key`): the supervisor's minted
///   per-child secret rides the connection quintet like llamacpp.
pub struct SglangEngine {
    pub manifest: crate::engine::manifest::Manifest,
    /// HTTP client for health polls (children are loopback).
    http: reqwest::Client,
    /// Extra env injected into children (config `engine_env`).
    pub child_env: Vec<(String, String)>,
}

impl SglangEngine {
    #[must_use]
    pub fn new(manifest: crate::engine::manifest::Manifest) -> Self {
        Self::with_env(manifest, Vec::new())
    }

    /// `env` pairs apply to every spawned child (config `engine_env`).
    #[must_use]
    pub fn with_env(
        manifest: crate::engine::manifest::Manifest,
        env: Vec<(String, String)>,
    ) -> Self {
        // 5s, not the 2s the fast lanes use: sglang 0.5.19's default
        // /health is a GENERATE probe (SGLANG_ENABLE_HEALTH_ENDPOINT_
        // GENERATION defaults true, environ.py) that sleeps in 1s steps
        // waiting for a 1-token round trip — ~1.2s idle, >2s under load.
        // A 2s total timeout never completes one 200 and the poll reports
        // a healthy child as dead for the whole model_load_timeout window
        // (seen live: uvicorn access log full of 200s the daemon never
        // received within budget).
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .expect("health client");
        Self {
            manifest,
            http,
            child_env: env,
        }
    }
}

/// sglang connection quintet + compiled profile. The profile argv is
/// already `launch_server` grammar (context-length, max-running-requests,
/// mem-fraction-static, ...), so this only PREPENDS what the supervisor
/// owns: model dir, loopback bind, port, public name. The per-child
/// `--api-key` secret is appended by the supervisor's child-auth mint
/// (same as llamacpp) — one choke point, not engine business.
fn sglang_argv(
    model: &pallama_core::ModelRow,
    profile: &Profile,
    endpoint: &Endpoint,
) -> Vec<String> {
    let port = match endpoint {
        Endpoint::Tcp { port, .. } => *port,
        // Supervisor rejects unix endpoints for sglang engines before
        // argv assembly; a placeholder here cannot produce a valid child.
        Endpoint::Unix { .. } => 0,
    };
    let mut argv = vec![
        "--model-path".to_string(),
        model.path.clone(),
        // Forced loopback: even with the API key set, the child is not
        // the public face — the gateway is (binding revision 3).
        "--host".to_string(),
        "127.0.0.1".to_string(),
        "--port".to_string(),
        port.to_string(),
        "--served-model-name".to_string(),
        model.name.clone(),
    ];
    argv.extend(profile.argv.iter().cloned());
    argv
}

#[async_trait]
impl Engine for SglangEngine {
    fn kind(&self) -> pallama_core::engine_kind::EngineKind {
        pallama_core::engine_kind::EngineKind::Sglang
    }

    fn capabilities(&self) -> &crate::engine::manifest::Manifest {
        &self.manifest
    }

    fn build_argv(
        &self,
        model: &pallama_core::ModelRow,
        profile: &Profile,
        endpoint: &Endpoint,
    ) -> Vec<String> {
        sglang_argv(model, profile, endpoint)
    }

    async fn spawn(&self, argv: &[String], endpoint: &Endpoint) -> Result<ChildHandle> {
        if matches!(endpoint, Endpoint::Unix { .. }) {
            return Err(anyhow!(
                "sglang engines have no unix-socket transport; set \
                 child_transport = \"tcp\" in the pallama config"
            ));
        }
        // No --rpc preflight: sglang has no rpc-worker flag surface.
        spawn_child(&self.manifest.server_path, argv, endpoint, &self.child_env)
    }

    async fn health_check(&self, endpoint: &Endpoint, timeout: std::time::Duration) -> Result<()> {
        let Endpoint::Tcp { host, port } = endpoint else {
            return Err(anyhow!("sglang engines require a TCP endpoint"));
        };
        let url = format!("http://{host}:{port}");
        let deadline = tokio::time::Instant::now() + timeout;
        let started = std::time::Instant::now();
        // Same adaptive poll as the other engines. Readiness = HTTP 200
        // on /health: sglang answers 503 while `Starting` (weights
        // loading, warmup generate) and 200 once it serves — the body is
        // not a contract (v0.5.19 http_server.py:662).
        let mut poll = std::time::Duration::from_millis(25);
        loop {
            if let Ok(resp) = self.http.get(format!("{url}/health")).send().await {
                if resp.status().is_success() {
                    tracing::debug!("sglang healthy at {url} after {:?}", started.elapsed());
                    return Ok(());
                }
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(anyhow!(
                    "sglang at {url} did not turn healthy (HTTP 200 on /health) \
                     within {timeout:?} (model_load_timeout)"
                ));
            }
            tokio::time::sleep(poll).await;
            // 1s cap, not the 150ms the fast lanes use: every sglang
            // /health poll costs the child a 1-token generate on the GPU
            // (generate-probe default), so a 150ms hammer queues ~7
            // probes/second against a booting scheduler and self-congests
            // the exact responses the poll is waiting for.
            poll = std::cmp::min(poll.mul_f64(1.6), std::time::Duration::from_secs(1));
        }
    }
}

async fn pipe_logs<R: tokio::io::AsyncRead + Unpin + Send + 'static>(
    r: R,
    stream: &str,
    tail: LogTail,
) {
    use tokio::io::AsyncBufReadExt;
    // Raw-byte reads + lossy decode (F102): `lines()` aborts the whole
    // pipe task on one invalid-UTF-8 line and the child goes dark.
    let mut reader = tokio::io::BufReader::new(r);
    let mut raw = Vec::new();
    loop {
        raw.clear();
        match reader.read_until(b'\n', &mut raw).await {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                let line = String::from_utf8_lossy(&raw);
                let line = line.trim_end_matches(['\n', '\r']);
                tracing::debug!(target: "pallama::engine", stream, "{line}");
                if let Ok(mut t) = tail.lock() {
                    if t.len() >= TAIL_CAP {
                        t.pop_front();
                    }
                    t.push_back(line.to_string());
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(non_snake_case)] // repo convention: unit__scenario__expected
mod tests {
    use super::*;

    fn rpc_argv(value: &str) -> Vec<String> {
        vec![
            "llama-server".to_string(),
            "--rpc".to_string(),
            value.to_string(),
        ]
    }

    /// A loopback port that is VERIFIED unreachable at hand-off. The
    /// naive bind-drop pattern races the suite's parallel port-0 binds:
    /// another test can rebind the just-released port mid-test and the
    /// "dead" endpoint quietly turns alive (seen live: `probe_rpc` and
    /// `dead_rpc` spawn tests failed ~1-in-10 full runs, a different one
    /// each time). Here the port is re-checked by an actual connect and
    /// swapped out if anyone grabbed it.
    async fn dead_port() -> u16 {
        for _ in 0..64 {
            let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
                .await
                .expect("bind");
            let port = listener.local_addr().unwrap().port();
            drop(listener);
            match tokio::time::timeout(
                std::time::Duration::from_millis(250),
                tokio::net::TcpStream::connect(("127.0.0.1", port)),
            )
            .await
            {
                // timeout (still closed) or refused: verifiably dead
                Err(_) | Ok(Err(_)) => return port,
                // someone rebound it: try another port
                Ok(Ok(_)) => {}
            }
        }
        panic!("could not secure a dead loopback port in 64 tries");
    }

    #[tokio::test]
    async fn unit__probe_rpc__no_flag_is_noop() {
        let argv = vec![
            "llama-server".to_string(),
            "-np".to_string(),
            "1".to_string(),
        ];
        assert!(probe_rpc_endpoints(&argv).await.is_empty());
    }

    #[tokio::test]
    async fn unit__probe_rpc__alive_endpoint_passes() {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let port = listener.local_addr().unwrap().port();
        let argv = rpc_argv(&format!("127.0.0.1:{port}"));
        assert!(probe_rpc_endpoints(&argv).await.is_empty());
    }

    #[tokio::test]
    async fn unit__probe_rpc__dead_endpoint_reported() {
        // Bind then drop: the port is (almost certainly) closed again.
        let port = dead_port().await;
        let argv = rpc_argv(&format!("127.0.0.1:{port}"));
        let dead = probe_rpc_endpoints(&argv).await;
        assert_eq!(dead.len(), 1, "{dead:?}");
        assert!(dead[0].contains("(unreachable)"), "{dead:?}");
    }

    #[tokio::test]
    async fn unit__probe_rpc__mixed_list_reports_only_dead() {
        let alive = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let alive_port = alive.local_addr().unwrap().port();
        let dead_port = dead_port().await;
        let argv = rpc_argv(&format!("127.0.0.1:{alive_port},127.0.0.1:{dead_port}"));
        let reported = probe_rpc_endpoints(&argv).await;
        assert_eq!(reported.len(), 1, "{reported:?}");
        assert!(
            reported[0].contains(&format!("127.0.0.1:{dead_port}")),
            "{reported:?}"
        );
    }

    #[tokio::test]
    async fn unit__probe_rpc__all_flag_occurrences_probed() {
        // Config knob + extra_args can each emit --rpc; every occurrence
        // must be checked, not just the first.
        let alive = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let alive_port = alive.local_addr().unwrap().port();
        let dead_port = dead_port().await;
        let argv = vec![
            "llama-server".to_string(),
            "--rpc".to_string(),
            format!("127.0.0.1:{alive_port}"),
            "--rpc".to_string(),
            format!("127.0.0.1:{dead_port}"),
        ];
        let reported = probe_rpc_endpoints(&argv).await;
        assert_eq!(reported.len(), 1, "{reported:?}");
        assert!(
            reported[0].contains(&format!("127.0.0.1:{dead_port}")),
            "{reported:?}"
        );
    }

    #[tokio::test]
    async fn unit__probe_rpc__malformed_entries_reported_without_probe() {
        let argv = rpc_argv("nohostport,box1:notaport, ,127.0.0.1:1");
        let reported = probe_rpc_endpoints(&argv).await;
        // 127.0.0.1:1 parses as host:port and connect is refused on
        // loopback -> unreachable; the other two are malformed; the
        // empty entry is dropped.
        assert_eq!(reported.len(), 3, "{reported:?}");
        assert!(reported
            .iter()
            .any(|r| r.contains("nohostport (malformed host:port)")));
        assert!(reported
            .iter()
            .any(|r| r.contains("box1:notaport (malformed port)")));
        assert!(reported
            .iter()
            .any(|r| r.contains("127.0.0.1:1 (unreachable)")));
    }

    #[tokio::test]
    async fn unit__llamacpp_spawn__dead_rpc_refuses_before_exec() {
        let port = dead_port().await;
        let manifest = crate::engine::manifest::Manifest {
            tag: "fake".into(),
            build_number: 1,
            version_raw: "b1".into(),
            devices: vec![],
            flags: std::collections::BTreeSet::new(),
            spec_types: vec![],
            // Deliberately a nonexistent binary: the preflight must bail
            // BEFORE any exec, so this path is never touched.
            server_path: "/nonexistent/llama-server".into(),
            ..Default::default()
        };
        let engine = LlamaCppEngine::new(manifest);
        let argv = rpc_argv(&format!("127.0.0.1:{port}"));
        let endpoint = Endpoint::Tcp {
            host: "127.0.0.1".into(),
            port: 0,
        };
        let err = engine
            .spawn(&argv, &endpoint)
            .await
            .expect_err("dead rpc endpoint must refuse the spawn");
        let msg = format!("{err:#}");
        assert!(msg.contains("--rpc endpoint(s) unreachable"), "{msg}");
        assert!(msg.contains("rpc_servers"), "{msg}");
    }

    /// sglang 0.5.19 /health is a generate probe: the handler sleeps in
    /// 1s steps waiting for a 1-token round trip, so a 200 takes ~1.2s
    /// idle and 2s+ under load. A stub answers every /health with 200
    /// AFTER a probe-like latency — the health client's total timeout
    /// must outlive it or the poll declares a healthy child dead (the
    /// live failure: uvicorn logged a steady stream of 200s the daemon
    /// never completed within its old 2s budget, so every spawn was
    /// killed at `model_load_timeout`).
    async fn probe_latency_health_stub(latency: std::time::Duration) -> u16 {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind stub");
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    return;
                };
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                    let mut buf = [0u8; 4096];
                    loop {
                        let Ok(n) = sock.read(&mut buf).await else {
                            return;
                        };
                        if n == 0 || buf[..n].windows(4).any(|w| w == b"\r\n\r\n") {
                            break;
                        }
                    }
                    tokio::time::sleep(latency).await;
                    let _ = sock
                        .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n")
                        .await;
                });
            }
        });
        port
    }

    #[tokio::test]
    async fn unit__sglang_health__slow_generate_probe_200_still_passes() {
        let port = probe_latency_health_stub(std::time::Duration::from_millis(2500)).await;
        let engine = SglangEngine::new(crate::engine::manifest::Manifest {
            tag: "fake".into(),
            build_number: 1,
            version_raw: "b1".into(),
            devices: vec![],
            flags: std::collections::BTreeSet::new(),
            spec_types: vec![],
            server_path: String::new(),
            ..Default::default()
        });
        let endpoint = Endpoint::Tcp {
            host: "127.0.0.1".into(),
            port,
        };
        // 2.5s response latency: the old 2s-total client timed out on
        // EVERY attempt and never returned Ok within any budget.
        engine
            .health_check(&endpoint, std::time::Duration::from_secs(30))
            .await
            .expect("probe-latency 200 must count as healthy");
    }

    #[tokio::test]
    async fn unit__sglang_health__deadline_still_bounded() {
        let port = dead_port().await;
        let engine = SglangEngine::new(crate::engine::manifest::Manifest {
            tag: "fake".into(),
            build_number: 1,
            version_raw: "b1".into(),
            devices: vec![],
            flags: std::collections::BTreeSet::new(),
            spec_types: vec![],
            server_path: String::new(),
            ..Default::default()
        });
        let endpoint = Endpoint::Tcp {
            host: "127.0.0.1".into(),
            port,
        };
        let err = engine
            .health_check(&endpoint, std::time::Duration::from_secs(2))
            .await
            .expect_err("dead port must fail");
        assert!(err.to_string().contains("model_load_timeout"), "{err:#}");
    }

    fn mistralrs_row(path: &str) -> pallama_core::ModelRow {
        pallama_core::ModelRow {
            name: "qwen2.5-0.5b-instruct".into(),
            repo: "registry.ollama.ai/library/qwen2.5-0.5b-instruct".into(),
            quant: "BF16".into(),
            path: path.into(),
            bytes: 988_000_000,
            sha256: None,
            mmproj_path: None,
            shards: 1,
            arch: Some("Qwen2ForCausalLM".into()),
            params: None,
            ctx_train: Some(32_768),
            pulled_at: 0,
        }
    }

    fn mistralrs_profile() -> Profile {
        Profile::default()
    }

    fn mistralrs_flags() -> std::collections::BTreeSet<String> {
        std::collections::BTreeSet::new()
    }

    #[test]
    fn unit__mistralrs_argv__gguf_file_rides_dash_f() {
        let row = mistralrs_row("/store/qwen2.5-0.5b Q4_K_M.gguf");
        let argv = mistralrs_argv(
            &row,
            &mistralrs_profile(),
            &Endpoint::Tcp {
                host: "127.0.0.1".into(),
                port: 8123,
            },
            &mistralrs_flags(),
        );
        assert_eq!(argv[0], "serve");
        assert_eq!(argv[1], "-f");
        assert_eq!(argv[2], row.path);
        // GGUF loader keeps the ctx knob (mirror of the HF-dir pin).
        let mut p = mistralrs_profile();
        p.ctx = 32768;
        let flags = [
            "--max-model-len".to_string(),
            "--max-num-batched-tokens".to_string(),
        ]
        .into_iter()
        .collect();
        let argv = mistralrs_argv(
            &row,
            &p,
            &Endpoint::Tcp {
                host: "127.0.0.1".into(),
                port: 8123,
            },
            &flags,
        );
        let i = argv
            .iter()
            .position(|a| a == "--max-model-len")
            .expect("GGUF rows keep --max-model-len");
        assert_eq!(argv[i + 1], "32768");
        // The batch-step floor rides both loaders: v0.9.3 defaults to
        // 4096 tokens per paged-attention step, so any longer prompt
        // returns an empty-choices 200 (live-proven on the matrix run).
        let b = argv
            .iter()
            .position(|a| a == "--max-num-batched-tokens")
            .expect("GGUF rows raise the scheduler batch step");
        assert_eq!(argv[b + 1], "32768");
    }

    #[test]
    fn unit__mistralrs_argv__hf_directory_rides_model_id() {
        // The exact live failure: a safetensors directory under `-f`
        // aborted the child with "Cannot infer model format from
        // `--quantized-file`" (first live mistralrs serve, matrix pilot).
        let tmp = std::env::temp_dir().join(format!("pallama-mistralrs-hf-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).expect("hf dir");
        let row = mistralrs_row(tmp.to_str().unwrap_or("/tmp/x"));
        let argv = mistralrs_argv(
            &row,
            &mistralrs_profile(),
            &Endpoint::Tcp {
                host: "127.0.0.1".into(),
                port: 8123,
            },
            &mistralrs_flags(),
        );
        assert_eq!(argv[1], "-m", "HF dirs must ride --model-id, not -f");
        assert_eq!(argv[2], row.path);
        // The second live failure: the HF loader rejects the GGUF-only
        // `--max-model-len` ("not supported by this model loader").
        let mut p = mistralrs_profile();
        p.ctx = 32768;
        let flags = ["--max-model-len".to_string()].into_iter().collect();
        let argv = mistralrs_argv(
            &row,
            &p,
            &Endpoint::Tcp {
                host: "127.0.0.1".into(),
                port: 8123,
            },
            &flags,
        );
        assert!(
            !argv.contains(&"--max-model-len".to_string()),
            "HF loader must not receive --max-model-len"
        );
        // The batch-step raise is loader-independent (same paged
        // scheduler serves HF weights).
        let flags = ["--max-num-batched-tokens".to_string()]
            .into_iter()
            .collect();
        let argv = mistralrs_argv(
            &row,
            &p,
            &Endpoint::Tcp {
                host: "127.0.0.1".into(),
                port: 8123,
            },
            &flags,
        );
        let b = argv
            .iter()
            .position(|a| a == "--max-num-batched-tokens")
            .expect("HF rows raise the scheduler batch step too");
        assert_eq!(argv[b + 1], "32768");
        // Small contexts never LOWER the engine default below the floor.
        let mut tiny = mistralrs_profile();
        tiny.ctx = 1024;
        let argv = mistralrs_argv(
            &row,
            &tiny,
            &Endpoint::Tcp {
                host: "127.0.0.1".into(),
                port: 8123,
            },
            &flags,
        );
        let b = argv
            .iter()
            .position(|a| a == "--max-num-batched-tokens")
            .expect("tiny ctx still raises the floor");
        assert_eq!(argv[b + 1], "4096");
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn unit__mistralrs_staging__hf_directory_bypasses_shard_view() {
        // With a staging root set (production shape), a directory row must
        // come back UNTOUCHED: the shard-view farm would stage an empty
        // view (only *.gguf siblings match) and hand the engine a
        // nonexistent path.
        let tmp =
            std::env::temp_dir().join(format!("pallama-mistralrs-stage-{}", std::process::id()));
        let hf = tmp.join("model.d");
        let staging = tmp.join("staging-root");
        std::fs::create_dir_all(&hf).expect("hf dir");
        std::fs::create_dir_all(&staging).expect("staging root");
        let mut manifest = crate::engine::manifest::Manifest {
            tag: "v0.9.3".into(),
            build_number: 0,
            version_raw: "v0.9.3".into(),
            devices: vec![],
            flags: std::collections::BTreeSet::new(),
            spec_types: vec![],
            server_path: String::new(),
            ..Default::default()
        };
        manifest.server_path = "/nonexistent/mistralrs".into();
        let engine = MistralRsEngine::with_staging(manifest, Vec::new(), Some(staging));
        let (staged, mmproj) = engine.stage_model_view(&mistralrs_row(hf.to_str().unwrap_or("")));
        assert_eq!(staged, hf, "HF dir must serve from its raw location");
        assert!(mmproj.is_none());
        assert!(
            !staged.starts_with(tmp.join("staging-root")),
            "no view must be built for directory rows"
        );
        std::fs::remove_dir_all(&tmp).ok();
    }
}
