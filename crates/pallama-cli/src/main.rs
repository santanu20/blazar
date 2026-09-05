//! `pallama` — llama.cpp orchestration platform.
//!
//! Local-only by design: no telemetry, no cloud endpoints; the only
//! outbound traffic is user-initiated engine/model downloads. Powered by
//! llama.cpp / ggml / ggerganov — <https://github.com/ggml-org/llama.cpp>

use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};

use pallama_core::{Config, PallamaDirs, Store};
use pallama_runtime::EventBus;
use pallama_runtime::engine::gh::GhClient;
use pallama_runtime::engine::EngineManager;
use pallama_runtime::{LlamaCppEngine, Supervisor};

#[derive(Parser)]
#[command(
    name = "pallama",
    version,
    about = "llama.cpp orchestration: ollama-grade UX, zero engine fork",
    after_help = "Local-only: no telemetry, no cloud endpoints. Powered by llama.cpp / ggml / ggerganov."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the daemon in the foreground (both APIs on one port).
    #[command(alias = "start")]
    Serve,
    /// Copy a model under a new name (zero-byte hardlink alias)
    Cp { source: String, destination: String },
    /// Create a model alias with parameters from a Modelfile (no blob copy:
    /// `FROM` + `PARAMETER num_ctx` + `ADAPTER` map to a config overlay)
    Create {
        model: String,
        /// Modelfile path (default "Modelfile")
        #[arg(short = 'f', long)]
        file: Option<PathBuf>,
    },
    /// Push a model to a registry — refused: pallama is local-only by design
    Push { model: String },
    /// ollama.com account sign-in — refused: no cloud accounts by design
    Signin,
    /// ollama.com account login — refused: no cloud accounts by design
    Login,
    /// ollama.com account sign-out — refused: no cloud accounts by design
    Signout,
    /// ollama.com account logout — refused: no cloud accounts by design
    Logout,
    /// Stop the daemon; with a model name, unload that model now
    Stop { model: Option<String> },
    /// Pull a model (owner/repo[:QUANT] or catalog name)
    Pull { target: String },
    /// Import an existing GGUF file (hardlinks by default; --copy for a copy)
    Import {
        path: PathBuf,
        #[arg(long)]
        name: Option<String>,
        #[arg(long)]
        quant: Option<String>,
        #[arg(long)]
        copy: bool,
    },
    /// Remove models (refuses while running)
    Rm { models: Vec<String> },
    /// List pulled models
    #[command(alias = "ls")]
    List,
    /// Show model details, active profile, last benchmark
    Show { model: String },
    /// Live instances: state, ctx, idle countdown
    Ps {
        /// Clear crash circuit breakers
        #[arg(long)]
        reset: bool,
    },
    /// Chat REPL against a model (streams; /exit /clear /model /sysinfo /profile);
    /// with an inline PROMPT: single-shot generation, prints and exits
    Run {
        model: String,
        #[arg(trailing_var_arg = true)]
        prompt: Vec<String>,
        /// Print eval counts (tokens, t/s) after generation
        #[arg(long, short = 'v')]
        verbose: bool,
    },
    /// Benchmark a model (pp/tg table)
    Bench { model: String },
    /// Tune a model's launch profile (--search = measured grid argmax)
    Tune {
        model: String,
        #[arg(long)]
        search: bool,
        /// Fix the context size in the adopted profile
        #[arg(long)]
        ctx: Option<u32>,
        /// Set the model's spec mode persistently ("off" | "auto")
        #[arg(long)]
        spec: Option<String>,
    },
    /// Engine (llama-server) management
    Engine {
        #[command(subcommand)]
        cmd: EngineCmd,
    },
    /// `LoRA` adapter management
    Lora {
        #[command(subcommand)]
        cmd: LoraCmd,
    },
    /// Search Hugging Face for GGUF repos
    Search { query: String },
    /// Pre-download fit preview: VRAM/RAM split + quant alternatives
    Fit { target: String },
    /// Config inspection and editing (one knob surface)
    Config {
        #[command(subcommand)]
        cmd: ConfigCmd,
    },
    /// Self-update the pallama binary from GitHub Releases (sha256-verified)
    Upgrade {
        /// Pin a release tag (e.g. v0.1.1); default = latest
        #[arg(long)]
        version: Option<String>,
        /// Resolve + download + verify, but do not replace the binary
        #[arg(long)]
        dry_run: bool,
    },
    /// Slot KV-cache checkpoints: save/restore a loaded model's context
    /// state (upstream --slot-save-path; survives unload and restart)
    Session {
        #[command(subcommand)]
        cmd: SessionCmd,
    },
    /// Diagnose the local setup: config, engine, hardware, disk, models
    Doctor,
    /// Ask why a request misbehaved: sentinel detections for a trace id
    /// (or the most recent observations) with fix hints
    Why {
        /// Trace id from the x-pallama-trace-id response header
        trace: Option<String>,
    },
}

#[derive(Subcommand)]
enum SessionCmd {
    /// Checkpoint the current slot state of a loaded model
    Save { model: String, name: String },
    /// Load a checkpoint back into the model's slot
    Restore { model: String, name: String },
    /// Delete a checkpoint file
    Rm { model: String, name: String },
    /// List checkpoints for a model
    #[command(alias = "ls")]
    List { model: String },
}

#[derive(Subcommand)]
enum EngineCmd {
    /// Install + activate the newest (or given) upstream build
    Update { tag: Option<String> },
    /// List installed engines with capability summaries
    List,
    /// Activate an installed tag
    Use { tag: String },
    /// Step back to the previous engine
    Rollback,
    /// Register a locally built llama-server (pseudo-tag "local")
    Local { path: PathBuf },
}

#[derive(Subcommand)]
enum LoraCmd {
    Add { model: String, path: PathBuf, #[arg(default_value = "1.0")] scale: f64 },
    Rm { id: i64 },
    List { model: Option<String> },
}

#[derive(Subcommand)]
enum ConfigCmd {
    List,
    Get { key: String },
    Set { key: String, value: String },
}

fn main() {
    let cli = Cli::parse();
    pallama_core::telemetry::init_tracing(0);
    if let Err(e) = run(cli.cmd) {
        eprintln!("pallama: {e:#}");
        std::process::exit(1);
    }
}

fn banner() {
    println!(
        "pallama {} — powered by llama.cpp / ggml / ggerganov",
        env!("CARGO_PKG_VERSION")
    );
}

fn dirs() -> PallamaDirs {
    PallamaDirs::from_env()
}

fn config() -> Result<Config> {
    let d = dirs();
    d.ensure().ok();
    Config::load(&d).map_err(|e| anyhow!("{e}"))
}

fn daemon_base(cfg: &Config) -> String {
    format!("http://{}:{}", cfg.host, cfg.port)
}

/// Auto-start (plan G): 1s probe; on refusal, detached self-exec `serve`
/// (own session, logs to run/daemon.log), then poll /healthz ≤30s.
async fn ensure_daemon() -> Result<String> {
    let cfg = config()?;
    let base = daemon_base(&cfg);
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()?;
    if let Ok(r) = http.get(format!("{base}/healthz")).send().await {
        if r.status().is_success() {
            return Ok(base);
        }
    }
    // Not running: self-exec detached.
    let log = dirs().run_dir().join("daemon.log");
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log)
        .with_context(|| format!("open {}", log.display()))?;
    let mut cmd = std::process::Command::new(std::env::current_exe()?);
    cmd.arg("serve")
        .stdin(std::process::Stdio::null())
        .stdout(log_file.try_clone()?)
        .stderr(log_file);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        cmd.process_group(0);
    }
    cmd.spawn().context("spawn detached pallama serve")?;
    let deadline = tokio_deadline(Duration::from_secs(30));
    while std::time::Instant::now() < deadline {
        if let Ok(r) = http.get(format!("{base}/healthz")).send().await {
            if r.status().is_success() {
                return Ok(base);
            }
        }
        std::thread::sleep(Duration::from_millis(300));
    }
    Err(anyhow!(
        "daemon did not become healthy within 30s; see {}",
        log.display()
    ))
}

fn tokio_deadline(d: Duration) -> std::time::Instant {
    std::time::Instant::now() + d
}

#[tokio::main]
async fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Serve => serve().await,
        Cmd::Stop { model } => stop_cmd(model).await,
        Cmd::Pull { target } => pull(&target).await,
        Cmd::Import { path, name, quant, copy } => import(&path, name, quant, copy),
        Cmd::Rm { models } => rm_multi(&models),
        Cmd::List => list(),
        Cmd::Show { model } => show(&model),
        Cmd::Ps { reset } => ps(reset).await,
        Cmd::Run { model, prompt, verbose } => run_dispatch(&model, &prompt, verbose).await,
        Cmd::Bench { model } => bench(&model),
        Cmd::Tune { model, search, ctx, spec } => tune(&model, search, ctx, spec),
        Cmd::Engine { cmd } => engine_cmd(cmd).await,
        Cmd::Lora { cmd } => lora_cmd(cmd),
        Cmd::Search { query } => search(&query).await,
        Cmd::Fit { target } => fit(&target).await,
        Cmd::Config { cmd } => config_cmd(cmd),
        Cmd::Upgrade { version, dry_run } => upgrade(version, dry_run).await,
        Cmd::Cp { source, destination } => cp_cmd(&source, &destination),
        Cmd::Create { model, file } => create_cmd(&model, file.as_deref()),
        Cmd::Push { model } => cloud_refusal("push", &model),
        Cmd::Signin => cloud_refusal("signin", ""),
        Cmd::Login => cloud_refusal("login", ""),
        Cmd::Signout => cloud_refusal("signout", ""),
        Cmd::Logout => cloud_refusal("logout", ""),
        Cmd::Session { cmd } => session_cmd(cmd).await,
        Cmd::Doctor => doctor().await,
        Cmd::Why { trace } => why(trace.as_deref()).await,
    }
}

async fn session_cmd(cmd: SessionCmd) -> Result<()> {
    let base = ensure_daemon().await?;
    let client = reqwest::Client::new();
    match cmd {
        SessionCmd::Save { model, name } => {
            let resp = client
                .post(format!("{base}/api/session"))
                .json(&serde_json::json!({
                    "model": model, "action": "save", "filename": name
                }))
                .send()
                .await?;
            print_session_result(resp, "saved", &model, &name).await?;
        }
        SessionCmd::Restore { model, name } => {
            let resp = client
                .post(format!("{base}/api/session"))
                .json(&serde_json::json!({
                    "model": model, "action": "restore", "filename": name
                }))
                .send()
                .await?;
            print_session_result(resp, "restored", &model, &name).await?;
        }
        SessionCmd::Rm { model, name } => {
            let resp = client
                .post(format!("{base}/api/session"))
                .json(&serde_json::json!({
                    "model": model, "action": "erase", "filename": name
                }))
                .send()
                .await?;
            print_session_result(resp, "deleted", &model, &name).await?;
        }
        SessionCmd::List { model } => {
            let resp = client
                .get(format!("{base}/api/session?model={model}"))
                .send()
                .await?;
            if !resp.status().is_success() {
                let text = resp.text().await.unwrap_or_default();
                return Err(anyhow!("{text}"));
            }
            let v: serde_json::Value = resp.json().await?;
            let sessions = v["sessions"].as_array().cloned().unwrap_or_default();
            if sessions.is_empty() {
                println!("no checkpoints for {model}");
            } else {
                println!("{:<6} CHECKPOINT", "BYTES");
                for s in sessions {
                    println!(
                        "{:<6} {}",
                        humansize(s["bytes"].as_i64().unwrap_or(0)),
                        s["filename"].as_str().unwrap_or("?")
                    );
                }
            }
        }
    }
    Ok(())
}

/// One doctor check row: name, status word, detail line.
struct Check {
    name: &'static str,
    ok: bool,
    warn: bool,
    detail: String,
}

impl Check {
    fn ok(name: &'static str, detail: impl Into<String>) -> Self {
        Self { name, ok: true, warn: false, detail: detail.into() }
    }
    fn warn(name: &'static str, detail: impl Into<String>) -> Self {
        Self { name, ok: true, warn: true, detail: detail.into() }
    }
    fn fail(name: &'static str, detail: impl Into<String>) -> Self {
        Self { name, ok: false, warn: false, detail: detail.into() }
    }
    fn status_word(&self) -> &'static str {
        if !self.ok {
            "FAIL"
        } else if self.warn {
            "WARN"
        } else {
            "ok"
        }
    }
}

/// `pallama doctor` — offline diagnostics for the local setup. Reads only
/// (config, store, engine manifest, hardware probe); never starts or stops
/// anything. Failures print the fix hint inline.
async fn doctor() -> Result<()> {
    let d = dirs();
    let mut checks: Vec<Check> = Vec::new();

    // 1. config parses + validates
    match config() {
        Ok(cfg) => {
            checks.push(Check::ok(
                "config",
                format!(
                    "{} validates; host {}:{}",
                    d.config_file().display(),
                    cfg.host,
                    cfg.port
                ),
            ));
            if cfg.agent {
                checks.push(Check::warn(
                    "agent mode",
                    "enabled: children expose built-in tools (exec_shell_command) + MCP CORS proxy",
                ));
            }
            if !cfg.cpu_range.is_empty() {
                checks.push(Check::ok("cpu_range", format!("pinned to {}", cfg.cpu_range)));
            }
        }
        Err(e) => {
            checks.push(Check::fail(
                "config",
                format!("{} — fix or delete {} to regenerate defaults", e, d.config_file().display()),
            ));
        }
    }

    checks.extend(doctor_engine(&d));
    checks.extend(doctor_port().await);
    #[cfg(unix)]
    checks.extend(doctor_disk(&d));
    checks.extend(doctor_models(&d));
    checks.extend(doctor_sentinel(&d));

    // render
    println!("{:<26} {:<5} DETAIL", "CHECK", "ST");
    for c in &checks {
        println!("{:<26} {:<5} {}", c.name, c.status_word(), c.detail);
    }
    let fails = checks.iter().filter(|c| !c.ok).count();
    let warns = checks.iter().filter(|c| c.warn).count();
    if fails > 0 {
        println!("\n{fails} failing check(s) — fix the FAIL rows above");
    } else if warns > 0 {
        println!("\nall checks pass; {warns} warning(s)");
    } else {
        println!("\nall checks pass");
    }
    Ok(())
}

fn doctor_engine(d: &PallamaDirs) -> Vec<Check> {
    let mut checks = Vec::new();
    match local_engine_manager(d) {
        Ok(mgr) => match mgr.active_manifest() {
            Ok(Some(m)) => {
                checks.push(Check::ok(
                    "engine",
                    format!(
                        "{} active ({} device{}, {} flags) at {}",
                        m.tag,
                        m.devices.len(),
                        if m.devices.len() == 1 { "" } else { "s" },
                        m.flags.len(),
                        m.server_path
                    ),
                ));
                let hw = pallama_runtime::probe_hardware(Some(&m));
                let note = format!(
                    "RAM {} GiB, VRAM {} MiB across {} GPU{}",
                    hw.total_ram_mib / 1024,
                    hw.total_vram_mib(),
                    hw.gpus.len(),
                    if hw.gpus.len() == 1 { "" } else { "s" },
                );
                if hw.total_vram_mib() == 0 {
                    checks.push(Check::warn(
                        "hardware",
                        format!("{note} — CPU-only inference; expect token/s in the single digits"),
                    ));
                } else {
                    checks.push(Check::ok("hardware", note));
                }
            }
            Ok(None) => checks.push(Check::fail(
                "engine",
                "none installed — run: pallama engine update",
            )),
            Err(e) => checks.push(Check::fail(
                "engine",
                format!("{e} — run: pallama engine update"),
            )),
        },
        Err(e) => checks.push(Check::fail("engine", format!("{e}"))),
    }
    checks
}

async fn doctor_port() -> Vec<Check> {
    let Ok(cfg) = config() else {
        return Vec::new();
    };
    let addr = (cfg.host.as_str(), cfg.port);
    if tokio::net::TcpListener::bind(addr).await.is_ok() {
        return vec![Check::ok(
            "port",
            format!("{}:{} free (daemon not running)", cfg.host, cfg.port),
        )];
    }
    let is_pallama = reqwest::Client::new()
        .get(format!("http://{}:{}/healthz", cfg.host, cfg.port))
        .send()
        .await
        .is_ok_and(|r| r.status().is_success());
    if is_pallama {
        vec![Check::ok(
            "port",
            format!("{}:{} — pallama daemon already answering", cfg.host, cfg.port),
        )]
    } else {
        vec![Check::warn(
            "port",
            format!(
                "{}:{} occupied by another process (ollama lives on 11434; pick another port in config.toml if this blocks startup)",
                cfg.host,
                cfg.port
            ),
        )]
    }
}

#[cfg(unix)]
fn doctor_disk(d: &PallamaDirs) -> Vec<Check> {
    let Some(g) = free_gib(&d.data_dir) else {
        return Vec::new();
    };
    if g < 10.0 {
        vec![Check::warn(
            "disk",
            format!("{g:.1} GiB free under {} — model pulls need headroom", d.data_dir.display()),
        )]
    } else {
        vec![Check::ok("disk", format!("{g:.1} GiB free under {}", d.data_dir.display()))]
    }
}

/// Offline sentinel summary: read run/sentinel.jsonl (no daemon),
/// count detections in the last 24h, name the top codes. One row.
fn doctor_sentinel(d: &PallamaDirs) -> Vec<Check> {
    let path = d.run_dir().join("sentinel.jsonl");
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return vec![Check::ok(
            "sentinel",
            "no observations yet (records land after the first chat request)",
        )];
    };
    let day_ago = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |t| t.as_secs().saturating_sub(86_400));
    let mut counts: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    let mut records = 0_usize;
    for line in raw.lines().filter(|l| !l.trim().is_empty()) {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if v["ts"].as_u64().unwrap_or(0) < day_ago {
            continue;
        }
        records += 1;
        if let Some(dets) = v["detections"].as_array() {
            for det in dets {
                if let Some(code) = det["code"].as_str() {
                    *counts.entry(code.to_string()).or_default() += 1;
                }
            }
        }
    }
    if counts.is_empty() {
        return vec![Check::ok(
            "sentinel",
            format!("{records} requests observed in 24h, all clean"),
        )];
    }
    let top: Vec<String> = counts.iter().map(|(c, n)| format!("{c} x{n}")).collect();
    vec![Check::warn(
        "sentinel",
        format!(
            "{records} requests in 24h flagged: {} — `pallama why` for details + retry hints",
            top.join(", ")
        ),
    )]
}

fn doctor_models(d: &PallamaDirs) -> Vec<Check> {
    let Ok(store) = pallama_core::store::Store::open(d) else {
        return vec![Check::fail("models", "store open failed")];
    };
    let Ok(models) = store.list_models() else {
        return vec![Check::fail("models", "store list failed")];
    };
    let bad: Vec<String> = models
        .iter()
        .filter(|m| pallama_core::read_metadata_file(std::path::Path::new(&m.path)).is_err())
        .map(|m| m.name.clone())
        .collect();
    if bad.is_empty() {
        vec![Check::ok("models", format!("{} pulled, all parse", models.len()))]
    } else {
        vec![Check::warn(
            "models",
            format!(
                "{} pulled; metadata unreadable: {} (re-pull or rm)",
                models.len(),
                bad.join(", ")
            ),
        )]
    }
}

/// One audited libc call (mirrors the runtime's signal-0 precedent):
/// statvfs on the data dir for the disk-headroom check. Read-only.
#[cfg(unix)]
#[allow(unsafe_code)] // single statvfs read; no pointers escape
fn free_gib(path: &std::path::Path) -> Option<f64> {
    let c = std::ffi::CString::new(path.to_string_lossy().as_bytes()).ok()?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(c.as_ptr(), std::ptr::from_mut(&mut stat)) };
    if rc != 0 {
        return None;
    }
    #[allow(clippy::useless_conversion)] // field types vary across libcs
    let free: u64 = stat.f_bfree.try_into().ok()?;
    let bsize: u64 = stat.f_bsize;
    #[allow(clippy::cast_precision_loss)] // byte counts -> GiB display only
    Some(free as f64 * bsize as f64 / 1_073_741_824.0)
}

async fn print_session_result(
    resp: reqwest::Response,
    verb: &str,
    model: &str,
    name: &str,
) -> Result<()> {
    if resp.status().is_success() {
        println!("{verb} checkpoint {name} for {model}");
        Ok(())
    } else {
        let text = resp.text().await.unwrap_or_default();
        Err(anyhow!("{text}"))
    }
}

async fn serve() -> Result<()> {
    banner();
    let d = dirs();
    d.ensure().ok();
    let cfg = config()?;
    let _lock = pallama_runtime::DaemonLock::acquire(&d)
        .map_err(|e| anyhow!("{e}"))?;
    rotate_daemon_log(&d);
    let store = Store::open(&d)?;
    // PALLAMA_ENGINE_PATH: register/refresh the local build and prefer it
    // for this run (plan C: pseudo-tag "local", never pruned).
    let local_override = std::env::var("PALLAMA_ENGINE_PATH").ok();
    if let Some(path) = &local_override {
        let p = std::path::PathBuf::from(path);
        let mgr = local_engine_manager(&d)?;
        let row = mgr.register_local(&p, &cfg.engine_env)?;
        mgr.use_tag(&row.tag)?;
        println!("PALLAMA_ENGINE_PATH: engine local active ({})", p.display());
    }
    let engine_row = store.active_engine()?
        .ok_or_else(|| anyhow!("no engine installed; run: pallama engine update"))?;
    let manifest: pallama_runtime::Manifest =
        serde_json::from_str(&engine_row.manifest)
            .with_context(|| format!("decode engine manifest {}", engine_row.tag))?;
    println!("engine: {} (build {})", engine_row.tag, manifest.build_number);
    let hw = pallama_runtime::probe_hardware(Some(&manifest));
    println!(
        "hardware: {} cores, {} MiB RAM, {} GPU(s), {} MiB VRAM",
        hw.physical_cores,
        hw.total_ram_mib,
        hw.gpus.len(),
        hw.total_vram_mib()
    );
    println!("config: {}", d.config_file().display());
    println!("data:   {}", d.data_dir.display());
    let engine_env: Vec<(String, String)> = cfg
        .engine_env
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let engine = Arc::new(LlamaCppEngine::with_env(manifest, engine_env));
    let bus = EventBus::default();
    let sup = Arc::new(Supervisor::new(d.clone(), cfg.clone(), bus.clone(), hw, engine));
    for orphan in sup.sweep_orphans() {
        println!("swept orphan engine: {orphan}");
    }
    let _reaper = sup.spawn_reaper();
    let state = Arc::new(pallama_gateway::state::AppState::new(
        d.clone(), cfg.clone(), sup.clone(), bus,
    ));
    let host = cfg.host.clone();
    let port = cfg.port;
    pallama_gateway::serve(
        state,
        &host,
        port,
        Box::pin(pallama_runtime::wait_for_shutdown_signal()),
    )
    .await?;
    println!("pallama stopped cleanly");
    Ok(())
}

fn stop() -> Result<()> {
    let d = dirs();
    let pidfile = d.run_dir().join("pallama.pid");
    let pid: String = std::fs::read_to_string(&pidfile)
        .with_context(|| format!("no daemon pidfile at {}", pidfile.display()))?;
    let pid: i32 = pid.trim().parse().context("pidfile corrupted")?;
    #[cfg(unix)]
    signal_stop::term(pid);
    println!("signalled daemon pid {pid}");
    Ok(())
}

#[cfg(unix)]
mod signal_stop {
    /// Stop the daemon by its exact pidfile pid (single-pid TERM; never a
    /// process group).
    pub fn term(pid: i32) {
        let _ = std::process::Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .status();
    }
}

async fn pull(target: &str) -> Result<()> {
    banner();
    let d = dirs();
    d.ensure().ok();
    let token = std::env::var("HF_TOKEN").ok();
    let client = pallama_runtime::hf::HfClient::new(token)?;
    let bus = EventBus::default();
    let mut events = bus.subscribe();
    let puller = pallama_runtime::Puller { dirs: d.clone(), client, bus };
    let row = puller.pull(target).await?;
    println!(
        "pulled {}: {} ({}, {} shards) -> {}",
        row.name,
        humansize(row.bytes),
        row.quant,
        row.shards,
        row.path
    );
    // The runtime warned via log + event; surface it in the terminal too
    // (tracing is muted at the default level). The event is the REAL
    // signal — HF metadata fallbacks can still fill `arch`.
    while let Ok(ev) = events.try_recv() {
        if let pallama_runtime::PallamaEvent::ModelPulled { warning: Some(w), .. } = ev {
            println!("WARNING: {w}");
        }
    }
    Ok(())
}

const MIB_F64: f64 = 1_048_576.0;
const GIB_F64: f64 = 1_073_741_824.0;

#[allow(clippy::cast_precision_loss)] // display rounding only
fn humansize(bytes: i64) -> String {
    let b = bytes.max(0) as f64;
    if b >= GIB_F64 {
        format!("{:.1} GiB", b / GIB_F64)
    } else if b >= MIB_F64 {
        format!("{:.0} MiB", b / MIB_F64)
    } else {
        format!("{b:.0} B")
    }
}

fn import(path: &PathBuf, name: Option<String>, quant: Option<String>, copy: bool) -> Result<()> {
        let d = dirs();
    d.ensure().ok();
    let meta = pallama_core::read_metadata_file(path.as_path())
        .map_err(|e| anyhow!("not a readable GGUF ({}): {e}", path.display()))?;
    let file_name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
    // Derive: name from GGUF general.name or explicit; quant from filename token.
    let derived_quant = quant.unwrap_or_else(|| {
        file_name
            .to_lowercase()
            .trim_end_matches(".gguf")
            .rsplit('-')
            .next()
            .unwrap_or("imported")
            .to_uppercase()
    });
    let size = std::fs::metadata(path).map_err(|e| anyhow!("{e}"))?.len();
    let fsize = size;
    // Ollama blobs are content-addressed names without quant hints.
    let model_name = name.unwrap_or_else(|| {
        meta.name
            .clone()
            .unwrap_or_else(|| file_name.trim_end_matches(".gguf").to_string())
            .to_lowercase()
            .replace([' ', '.'], "-")
    });
    let dest = d.models_dir().join(format!("{model_name}-{}.gguf", derived_quant.to_lowercase()));
    if !dest.exists() {
        if copy {
            std::fs::copy(path, &dest).with_context(|| format!("copy to {}", dest.display()))?;
        } else {
            #[cfg(unix)]
            {
                // Prefer a hardlink (zero bytes, survives either store being
                // wiped); the kernel's protected_hardlinks blocks linking
                // files we do not own (e.g. ollama's system blobs), so fall
                // back to a symlink in that case.
                if std::fs::hard_link(path, dest.as_path()).is_err() {
                    std::os::unix::fs::symlink(path, dest.as_path())
                        .with_context(|| format!("symlink {}", dest.display()))?;
                    println!("note: used symlink (hardlink blocked); if the source store deletes the blob, re-import");
                }
            }
        }
    }
    let bytes = i64::try_from(fsize).unwrap_or(i64::MAX);
    let _ = size;
    let row = pallama_core::ModelRow {
        name: model_name.clone(),
        repo: format!("imported:{}", path.display()),
        quant: derived_quant.clone(),
        path: dest.display().to_string(),
        bytes,
        sha256: None,
        mmproj_path: None,
        shards: 1,
        arch: Some(meta.architecture.clone()),
        params: Some(pallama_runtime::hf::est_params(fsize, &derived_quant)),
        ctx_train: meta.context_length.and_then(|c| i64::try_from(c).ok()),
        pulled_at: i64::try_from(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            )
            .unwrap_or(i64::MAX),
    };
    let store = Store::open(&d)?;
    store.upsert_model(&row)?;
    println!(
        "imported {} ({} {}, arch {}, ctx_train {:?}) -> {}",
        row.name,
        humansize(bytes),
        row.quant,
        meta.architecture,
        meta.context_length,
        dest.display()
    );
    println!("zero extra disk used (link, not copy)");
    Ok(())
}


fn list() -> Result<()> {
    let store = Store::open(&dirs())?;
    let models = store.list_models()?;
    if models.is_empty() {
        println!("no models pulled; try `pallama pull qwen3-0.6b` or `pallama search <query>`");
        return Ok(());
    }
    println!("{:<28} {:<8} {:>10}  {:<8} {:>8}  PATH", "NAME", "QUANT", "SIZE", "ARCH", "CTX");
    for m in models {
        println!(
            "{:<28} {:<8} {:>10}  {:<8} {:>8}  {}",
            m.name,
            m.quant,
            humansize(m.bytes),
            m.arch.as_deref().unwrap_or("?"),
            m.ctx_train.map_or_else(String::new, |c| c.to_string()),
            m.path
        );
    }
    Ok(())
}

fn show(model: &str) -> Result<()> {
    let d = dirs();
    let store = Store::open(&d)?;
    let row = store
        .get_model(model)?
        .ok_or_else(|| anyhow!("no such model: {model}"))?;
    println!("name:    {}", row.name);
    println!("repo:    {}", row.repo);
    println!("quant:   {}", row.quant);
    println!("path:    {}", row.path);
    println!("size:    {}", humansize(row.bytes));
    println!("shards:  {}", row.shards);
    if let Some(mm) = &row.mmproj_path {
        println!("mmproj:  {mm}");
    }
    let meta = pallama_core::read_metadata_file(std::path::Path::new(&row.path));
    if let Ok(m) = &meta {
        println!("arch:    {}", m.architecture);
        println!("blocks:  {:?}", m.block_count);
        println!("ctx_train: {:?}", m.context_length);
        println!("experts: {:?}", m.expert_count);
    }
    if let Some(engine) = store.active_engine()? {
        if let Some(p) = store.get_profile(&row.name, &engine.tag)? {
            println!("profile (engine {}):", engine.tag);
            println!("  argv: {}", p.args_json);
            if let Some(b) = &p.benchmark_json {
                println!("  benchmark: {b}");
            }
        }
    }
    Ok(())
}

async fn ps(reset: bool) -> Result<()> {
    if !reset {
        upstream_update_hint(&dirs()).await;
    }
    if reset {
        let base = ensure_daemon().await?;
        let _: serde_json::Value = reqwest::Client::new()
            .get(format!("{base}/api/ps"))
            .send()
            .await?
            .json()
            .await?;
        println!("circuits reset");
        return Ok(());
    }
    let base = ensure_daemon().await?;
    let v: serde_json::Value = reqwest::Client::new()
        .get(format!("{base}/api/ps"))
        .send()
        .await?
        .json()
        .await?;
    let models = v["models"].as_array().cloned().unwrap_or_default();
    if models.is_empty() {
        println!("no models loaded");
        return Ok(());
    }
    println!("{:<24} {:<9} {:>7} {:>10}  ENDPOINT", "NAME", "STATE", "CTX", "IN_FLIGHT");
    for m in models {
        println!(
            "{:<24} {:<9} {:>7} {:>10}  {}",
            m["name"].as_str().unwrap_or("?"),
            m["pallama_state"].as_str().unwrap_or("?"),
            m["pallama_ctx"].as_i64().unwrap_or(0),
            m["pallama_in_flight"].as_i64().unwrap_or(0),
            m["pallama_endpoint"].as_str().unwrap_or("-")
        );
    }
    Ok(())
}


/// `pallama run` dispatch: inline prompt = single-shot, none = REPL.
async fn run_dispatch(model: &str, prompt: &[String], verbose: bool) -> Result<()> {
    if prompt.is_empty() {
        return run_repl(model).await;
    }
    let base = ensure_daemon().await?;
    let text = prompt.join(" ");
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "stream": true,
    });
    let final_chunk = stream_chat(&base, &body).await?;
    if verbose {
        if let Some(v) = final_chunk {
            let ec = v["eval_count"].as_i64().unwrap_or(0);
            let ed = v["eval_duration"].as_i64().unwrap_or(0);
            let pc = v["prompt_eval_count"].as_i64().unwrap_or(0);
            #[allow(clippy::cast_precision_loss, reason = "token counts fit f64 exactly")]
            let rate = if ed > 0 { f64::from(ec as f32) * 1e9 / f64::from(ed as f32) } else { 0.0 };
            println!("\n\ntotal duration: answer {ec} tokens at {rate:.1} t/s; prompt {pc} tokens");
        }
    }
    Ok(())
}

/// ollama cloud commands are refused, loudly: pallama is local-only by
/// design and silently no-op-ing would hide the difference.
fn cloud_refusal(cmd: &str, model: &str) -> Result<()> {
    let what = if model.is_empty() { cmd.to_string() } else { format!("{cmd} {model}") };
    Err(anyhow!(
        "pallama {what}: refused — pallama is local-only by design (no registry, no cloud accounts). \
Pull models straight from Hugging Face: pallama pull <owner/repo:QUANT>"
    ))
}

/// `pallama why [trace]` — sentinel ring dump: what the model returned,
/// what was wrong with it, which knob fixes it. Auto-starts the daemon
/// like every other serving command.
async fn why(trace: Option<&str>) -> Result<()> {
    let base = ensure_daemon().await?;
    let url = match trace {
        Some(t) => format!("{base}/api/why?trace={t}"),
        None => format!("{base}/api/why"),
    };
    let resp = reqwest::Client::new()
        .get(&url)
        .timeout(Duration::from_secs(10))
        .send()
        .await?;
    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(anyhow!("daemon: {text}"));
    }
    let v: serde_json::Value = resp.json().await?;
    if !v["sentinel"].as_bool().unwrap_or(false) {
        println!("sentinel is disabled (config: sentinel = false) — no observations recorded");
        return Ok(());
    }
    let records = v["records"].as_array().cloned().unwrap_or_default();
    if records.is_empty() {
        println!("no observations recorded yet (sentinel records chat-family requests)");
        return Ok(());
    }
    let num = |v: &serde_json::Value| -> String {
        v.as_u64().map_or_else(|| "?".into(), |n| n.to_string())
    };
    for r in &records {
        let detections = r["detections"].as_array().cloned().unwrap_or_default();
        let flag = if detections.is_empty() { "ok" } else { "FLAGGED" };
        println!(
            "{flag}  {}  {}  model={} status={} ctx={} prompt={} completion={} degraded={} {}ms",
            r["trace"].as_str().unwrap_or("?"),
            r["route"].as_str().unwrap_or("?"),
            r["model"].as_str().unwrap_or("?"),
            num(&r["status"]),
            num(&r["ctx"]),
            num(&r["prompt_tokens"]),
            num(&r["completion_tokens"]),
            r["degraded"].as_bool().unwrap_or(false),
            num(&r["ms"]),
        );
        for d in &detections {
            println!(
                "  [{}] {} — {}",
                d["code"].as_str().unwrap_or("?"),
                d["detail"].as_str().unwrap_or(""),
                d["hint"].as_str().unwrap_or(""),
            );
            println!("    {}", d["retry"].as_str().unwrap_or(""));
        }
    }
    Ok(())
}

/// `pallama cp SOURCE DEST` — zero-byte hardlink alias.
fn cp_cmd(source: &str, destination: &str) -> Result<()> {
    let d = dirs();
    pallama_runtime::models::copy_model(&d, source, destination)?;
    println!("copied {source} -> {destination} (hardlink, no bytes duplicated)");
    Ok(())
}

/// `pallama rm A B C` — continue past failures, report all, exit non-zero.
fn rm_multi(models: &[String]) -> Result<()> {
    let d = dirs();
    let mut failed = Vec::new();
    for m in models {
        if let Err(e) = pallama_runtime::models::remove_model(&d, m) {
            eprintln!("rm {m}: {e:#}");
            failed.push(m.clone());
        } else {
            println!("removed {m}");
        }
    }
    if failed.is_empty() {
        Ok(())
    } else {
        Err(anyhow!("failed to remove: {}", failed.join(", ")))
    }
}

/// Modelfile subset pallama understands. Everything else is collected and
/// rejected by name — no silent drops.
struct CreateSpec {
    base: String,
    ctx: Option<u32>,
    loras: Vec<String>,
    unsupported: Vec<String>,
}

fn parse_modelfile(raw: &str) -> Result<CreateSpec> {
    let mut spec = CreateSpec { base: String::new(), ctx: None, loras: Vec::new(), unsupported: Vec::new() };
    for (n, raw_line) in raw.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        match parts.next().unwrap_or_default().to_ascii_uppercase().as_str() {
            "FROM" => {
                let arg = parts.next().unwrap_or_default().to_string();
                if arg.is_empty() {
                    return Err(anyhow!("Modelfile line {}: FROM needs a model name", n + 1));
                }
                spec.base = arg;
            }
            "ADAPTER" => {
                let arg = parts.next().unwrap_or_default().to_string();
                if arg.is_empty() {
                    return Err(anyhow!("Modelfile line {}: ADAPTER needs a path", n + 1));
                }
                spec.loras.push(arg);
            }
            "PARAMETER" => {
                let key = parts.next().unwrap_or_default().to_ascii_lowercase();
                let val: String = parts.collect::<Vec<_>>().join(" ");
                if key == "num_ctx" {
                    spec.ctx = Some(val.parse().map_err(|_| anyhow!("bad num_ctx {val:?}"))?);
                } else {
                    spec.unsupported.push(format!("PARAMETER {key}"));
                }
            }
            other => spec.unsupported.push(format!("{other} (line {})", n + 1)),
        }
    }
    if spec.base.is_empty() {
        return Err(anyhow!("Modelfile needs a FROM line"));
    }
    Ok(spec)
}

/// `pallama create MODEL -f Modelfile` — the anti-ollama version: no blob
/// copy, no template overrides. `FROM` maps to a hardlink alias; `num_ctx` and
/// ADAPTER to a config overlay; everything else is a named rejection.
fn create_cmd(model: &str, file: Option<&std::path::Path>) -> Result<()> {
    let path = file.unwrap_or_else(|| std::path::Path::new("Modelfile"));
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("read {}", path.display()))?;
    let spec = parse_modelfile(&raw)?;
    if !spec.unsupported.is_empty() {
        return Err(anyhow!(
            "Modelfile keys pallama refuses to fake: {}\n\
pallama has no Modelfile layer: GGUF is truth (templates embedded, served via --jinja) and \
parameters are per-request or per-model config overlays. Supported subset: FROM, PARAMETER num_ctx, ADAPTER. \
Alternatives: model_overrides in ~/.config/pallama/config.toml, or per-request options.",
            spec.unsupported.join(", ")
        ));
    }
    let d = dirs();
    pallama_runtime::models::copy_model(&d, &spec.base, model)?;
    // Overlay: ctx + loras for the new alias.
    let cfg_path = d.config_file();
    let cfg = if cfg_path.exists() {
        let raw = std::fs::read_to_string(&cfg_path)?;
        pallama_core::Config::from_toml(&raw).map_err(|e| anyhow!("{e}"))?
    } else {
        pallama_core::Config::default()
    };
    let mut cfg = cfg;
    let mut o = cfg.model_overrides.remove(model).unwrap_or_default();
    if let Some(ctx) = spec.ctx {
        o.ctx = Some(ctx);
    }
    if !spec.loras.is_empty() {
        o.loras = Some(spec.loras.clone());
    }
    cfg.model_overrides.insert(model.to_string(), o);
    std::fs::write(&cfg_path, cfg.to_toml().map_err(|e| anyhow!("{e}"))?)?;
    println!(
        "created {model} from {} (hardlink + overlay: ctx={:?}, loras={})",
        spec.base, spec.ctx, spec.loras.len()
    );
    Ok(())
}

/// `pallama stop` (daemon) vs `pallama stop MODEL` (unload now).
async fn stop_cmd(model: Option<String>) -> Result<()> {
    let Some(model) = model else {
        return stop();
    };
    let base = ensure_daemon().await?;
    let resp = reqwest::Client::new()
        .post(format!("{base}/api/evict"))
        .json(&serde_json::json!({"model": model}))
        .send()
        .await?;
    if resp.status().is_success() {
        println!("unloaded {model}");
        Ok(())
    } else {
        let text = resp.text().await.unwrap_or_default();
        Err(anyhow!("{text}"))
    }
}

async fn run_repl(model: &str) -> Result<()> {
    let base = ensure_daemon().await?;
    let mut rl = rustyline::DefaultEditor::new()?;
    let mut model = model.to_string();
    let mut history: Vec<serde_json::Value> = Vec::new();
    println!("pallama REPL — /exit /clear /model <name> /sysinfo /profile");
    while let Ok(line) = rl.readline(">>> ") {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        rl.add_history_entry(line).ok();
        match line {
            "/exit" => break,
            "/clear" => {
                history.clear();
                println!("(history cleared)");
                continue;
            }
            "/sysinfo" => {
                sysinfo_cmd(&base).await?;
                continue;
            }
            "/profile" => {
                show(&model)?;
                continue;
            }
            _ if line.starts_with("/model ") => {
                model = line["/model ".len()..].trim().to_string();
                history.clear();
                println!("(switched to {model})");
                continue;
            }
            _ => {}
        }
        history.push(serde_json::json!({"role": "user", "content": line}));
        let body = serde_json::json!({
            "model": model,
            "messages": history,
            "stream": true,
        });
        stream_chat(&base, &body).await?;
    }
    Ok(())
}

async fn sysinfo_cmd(base: &str) -> Result<()> {
    let v: serde_json::Value = reqwest::Client::new()
        .get(format!("{base}/api/version"))
        .send()
        .await?
        .json()
        .await?;
    println!("daemon: {}", v["version"].as_str().unwrap_or("?"));
    list()?;
    Ok(())
}

/// Stream an /api/chat NDJSON response to stdout; returns the final
/// (usage-carrying) chunk for --verbose stats.
#[allow(clippy::duration_suboptimal_units)] // 10-minute generation ceiling
async fn stream_chat(base: &str, body: &serde_json::Value) -> Result<Option<serde_json::Value>> {
    let resp = reqwest::Client::new()
        .post(format!("{base}/api/chat"))
        .json(body)
        .timeout(std::time::Duration::from_secs(600))
        .send()
        .await?;
    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(anyhow!("daemon: {text}"));
    }
    let mut resp = resp;
    let mut buf = String::new();
    let mut final_chunk: Option<serde_json::Value> = None;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    while let Some(chunk) = futures_lite_next(&mut resp).await? {
        buf.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(pos) = buf.find('\n') {
            let line: String = buf.drain(..=pos).collect();
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                if let Some(content) = v["message"]["content"].as_str() {
                    write!(out, "{content}").ok();
                }
                if v["done"].as_bool().unwrap_or(false) {
                    final_chunk = Some(v);
                }
            }
        }
        out.flush().ok();
    }
    writeln!(out).ok();
    Ok(final_chunk)
}

/// Minimal chunked-body reader without importing a stream crate in main.
async fn futures_lite_next(resp: &mut reqwest::Response) -> Result<Option<Vec<u8>>> {
    match resp.chunk().await {
        Ok(Some(b)) => Ok(Some(b.to_vec())),
        Ok(None) => Ok(None),
        Err(e) => Err(anyhow!("stream: {e}")),
    }
}

fn bench(model: &str) -> Result<()> {
    let d = dirs();
    let store = Store::open(&d)?;
    let row = store.get_model(model)?
        .ok_or_else(|| anyhow!("no such model: {model}"))?;
    let bench_bin = pallama_runtime::bench::find_bench_bin(&d)?;
    let tuner = pallama_runtime::Tuner { dirs: &d, bench_bin };
    let rows = tuner.bench_default(std::path::Path::new(&row.path))?;
    println!("{:<10} {:>8} {:>8} {:>6} {:<6} {:<6}", "TEST", "T/S", "CTX", "THREADS", "CTK", "CTV");
    for r in rows {
        println!(
            "{:<10} {:>8.2} {:>8} {:>6} {:<6} {:<6}",
            r.test_name(),
            r.ts,
            r.n_ctx.unwrap_or(0),
            r.n_threads.unwrap_or(0),
            r.type_k.as_deref().unwrap_or("-"),
            r.type_v.as_deref().unwrap_or("-"),
        );
    }
    Ok(())
}

fn tune(model: &str, search: bool, ctx: Option<u32>, spec: Option<String>) -> Result<()> {
    let d = dirs();
    let store = Store::open(&d)?;
    let row = store.get_model(model)?
        .ok_or_else(|| anyhow!("no such model: {model}"))?;
    let engine_row = store.active_engine()?
        .ok_or_else(|| anyhow!("no engine installed; run: pallama engine update"))?;
    let manifest: pallama_runtime::Manifest = serde_json::from_str(&engine_row.manifest)?;
    let bench_bin = pallama_runtime::bench::find_bench_bin(&d)?;
    let cfg = config()?;
    let hw = pallama_runtime::probe_hardware(Some(&manifest));
    let gguf = pallama_core::read_metadata_file(std::path::Path::new(&row.path))?;
    let overlay = cfg.overlay_for(model);
    let loras: Vec<(String, f64)> = store
        .list_loras(Some(model))?
        .into_iter()
        .map(|l| (l.path, l.scale))
        .collect();
    let data_dir = d.data_dir.to_string_lossy();
    let input = pallama_runtime::bench::build_input(
        model,
        &row.path,
        u64::try_from(row.bytes.max(0)).unwrap_or(u64::MAX),
        &gguf,
        &hw,
        &cfg,
        &overlay,
        &loras,
        None,
        &engine_row.tag,
        &manifest.flags,
        pallama_core::Endpoint::Tcp { host: "127.0.0.1".into(), port: 0 },
        &data_dir,
    );
    let store2 = Store::open(&d)?;
    let tuner = pallama_runtime::Tuner { dirs: &d, bench_bin };
    if let Some(mode) = spec {
        if mode != "off" && mode != "auto" {
            return Err(anyhow!("--spec must be \"off\" or \"auto\""));
        }
        set_model_override(model, "spec", &format!("\"{mode}\""))?;
        println!("model_overrides.{model}.spec = {mode}");
    }
    if search {
        let (profile, mut winning, rows) = tuner.tune_search(&store2, &input)?;
        if let Some(n) = ctx {
            winning.ctx = Some(n);
            let fixed = tuner.adopt(&store2, &input, &winning)?;
            println!("tuned {model} — {} configs measured; ctx pinned to {n}:", rows.len());
            println!("  argv: {}", fixed.argv.join(" "));
        } else {
            println!("tuned {} — {} configs measured, winner:", model, rows.len());
            println!("  argv: {}", profile.argv.join(" "));
            println!("  ctx:   {}", profile.ctx);
        }
    } else if let Some(n) = ctx {
        let winning = pallama_core::TuningOverrides { ctx: Some(n), ..Default::default() };
        let fixed = tuner.adopt(&store2, &input, &winning)?;
        println!("profile adopted at ctx {n}: {}", fixed.argv.join(" "));
    } else {
        let rows = tuner.bench_default(std::path::Path::new(&row.path))?;
        println!("{} rows measured; use --search (argmax) and/or --ctx/--spec to adopt", rows.len());
    }
    Ok(())
}

/// Persist a per-model overlay key (validated immediately).
fn set_model_override(model: &str, key: &str, value: &str) -> Result<()> {
    let d = dirs();
    let path = d.config_file();
    let raw = std::fs::read_to_string(&path).unwrap_or_default();
    let header = format!("[model_overrides.\"{model}\"]");
    let mut out: Vec<String> = Vec::new();
    let mut in_section = false;
    let mut replaced = false;
    for line in raw.lines() {
        if line.trim() == header {
            in_section = true;
        } else if line.starts_with('[') && in_section {
            in_section = false; // next section started
        }
        if in_section && line.starts_with(&format!("{key} =")) {
            out.push(format!("{key} = {value}"));
            replaced = true;
            continue;
        }
        out.push(line.to_string());
        if line.trim() == header && !replaced && key == "spec" {
            // insert right below header
        }
    }
    // Append key inside the section if never replaced
    if !replaced {
        if let Some(pos) = out.iter().position(|l| l.trim() == header) {
            // find end of section
            let mut end = out.len();
            for (i, l) in out.iter().enumerate().skip(pos + 1) {
                if l.starts_with('[') {
                    end = i;
                    break;
                }
            }
            out.insert(end, format!("{key} = {value}"));
        } else {
            out.push(String::new());
            out.push(header);
            out.push(format!("{key} = {value}"));
        }
    }
    let candidate = out.join("\n") + "\n";
    Config::from_toml(&candidate).map_err(|e| anyhow!("rejected, file unchanged: {e}"))?;
    std::fs::write(&path, &candidate)?;
    Ok(())
}

async fn engine_cmd(cmd: EngineCmd) -> Result<()> {
    let d = dirs();
    match cmd {
        EngineCmd::Update { tag } => {
            let token = std::env::var("GH_TOKEN").ok();
            let gh = GhClient::new(token)?;
            let mgr = EngineManager {
                dirs: d.clone(),
                gh,
                bus: EventBus::default(),
                asset_override: config()?.engine_asset,
            };
            let row = mgr.update(tag.as_deref()).await?;
            let m: pallama_runtime::Manifest = serde_json::from_str(&row.manifest)?;
            println!(
                "engine {} active (build {}, {} devices, {} flags)",
                row.tag,
                m.build_number,
                m.devices.len(),
                m.flags.len()
            );
        }
        EngineCmd::List => {
            upstream_update_hint(&d).await;
            let store = Store::open(&d)?;
            for e in store.list_engines()? {
                println!(
                    "{:<12} {:<10} {} {}",
                    e.tag,
                    e.asset,
                    if e.active { "[active]" } else { "" },
                    e.sha256.chars().take(12).collect::<String>()
                );
            }
        }
        EngineCmd::Use { tag } => {
            let mgr = local_engine_manager(&d)?;
            let row = mgr.use_tag(&tag)?;
            println!("active engine: {}", row.tag);
        }
        EngineCmd::Rollback => {
            let mgr = local_engine_manager(&d)?;
            let row = mgr.rollback()?;
            println!("rolled back to: {}", row.tag);
        }
        EngineCmd::Local { path } => {
            let mgr = local_engine_manager(&d)?;
            let row = mgr.register_local(&path, &config()?.engine_env)?;
            let m: pallama_runtime::Manifest = serde_json::from_str(&row.manifest)?;
            mgr.use_tag(&row.tag)?;
            println!(
                "engine local active (build {}, {} devices, {} flags)",
                m.build_number,
                m.devices.len(),
                m.flags.len()
            );
        }
    }
    Ok(())
}

/// Best-effort upstream check (user-invoked only, ~4s budget, silent on
/// failure): prints an update hint when a newer b-tag exists.
async fn upstream_update_hint(dirs: &PallamaDirs) {
    let Ok(store) = Store::open(dirs) else { return };
    let Ok(Some(active)) = store.active_engine() else { return };
    if active.tag == "local" {
        return; // local build: upstream currency is the user's concern
    }
    let token = std::env::var("GH_TOKEN").ok();
    let Ok(gh) = GhClient::new(token) else { return };
    let latest = tokio::time::timeout(std::time::Duration::from_secs(4), gh.latest_b_release()).await;
    if let Ok(Ok(rel)) = latest {
        if rel.tag_name == active.tag {
            println!("engine up to date: {}", active.tag);
        } else {
            println!(
                "update available: {} (active: {}) — run: pallama engine update",
                rel.tag_name, active.tag
            );
        }
    }
}

/// Rotate daemon.log when it exceeds ~10 MiB (best-effort, at start).
fn rotate_daemon_log(d: &PallamaDirs) {
    const MAX: u64 = 10 * 1024 * 1024;
    let log = d.run_dir().join("daemon.log");
    if let Ok(meta) = std::fs::metadata(&log) {
        if meta.len() > MAX {
            let old = d.run_dir().join("daemon.log.1");
            let _ = std::fs::remove_file(&old);
            let _ = std::fs::rename(&log, &old);
        }
    }
}

fn local_engine_manager(d: &PallamaDirs) -> Result<EngineManager> {
    let token = std::env::var("GH_TOKEN").ok();
    let gh = GhClient::new(token)?;
    Ok(EngineManager {
        dirs: d.clone(),
        gh,
        bus: EventBus::default(),
        asset_override: "auto".into(),
    })
}

fn lora_cmd(cmd: LoraCmd) -> Result<()> {
    let d = dirs();
    let store = Store::open(&d)?;
    match cmd {
        LoraCmd::Add { model, path, scale } => {
            let id = store.add_lora(&model, &path.display().to_string(), scale)?;
            println!("lora #{id} attached to {model} (scale {scale})");
        }
        LoraCmd::Rm { id } => {
            if store.delete_lora(id)? {
                println!("lora #{id} removed");
            } else {
                return Err(anyhow!("no lora #{id}"));
            }
        }
        LoraCmd::List { model } => {
            for l in store.list_loras(model.as_deref())? {
                println!("#{:<4} {:<24} scale {:<6} {}", l.id, l.model_name, l.scale, l.path);
            }
        }
    }
    Ok(())
}

async fn search(query: &str) -> Result<()> {
    let token = std::env::var("HF_TOKEN").ok();
    let client = pallama_runtime::hf::HfClient::new(token)?;
    let results = client.search(query, 20).await?;
    if results.is_empty() {
        println!("no GGUF repos matched {query:?}");
        return Ok(());
    }
    println!("{:<48} {:>10} {:>6}", "REPO", "DOWNLOADS", "LIKES");
    for r in results {
        println!(
            "{:<48} {:>10} {:>6}",
            r.id,
            r.downloads.unwrap_or(0),
            r.likes.unwrap_or(0)
        );
    }
    Ok(())
}

async fn fit(target: &str) -> Result<()> {
    let parsed = pallama_runtime::parse_pull_target(target)?;
    let token = std::env::var("HF_TOKEN").ok();
    let client = pallama_runtime::hf::HfClient::new(token)?;
    let info = client.model_info(&parsed.repo).await?;
    let cfg = config()?;
    // Local hardware: probe needs an engine; fall back to store manifest.
    let vram = {
        let store = Store::open(&dirs()).ok();
        store
            .and_then(|s| s.active_engine().ok().flatten())
            .and_then(|e| serde_json::from_str::<pallama_runtime::Manifest>(&e.manifest).ok())
            .map_or(0, |m| pallama_runtime::probe_hardware(Some(&m)).total_vram_mib())
    };
    let vram_bytes = pallama_core::Hardware::bytes(vram);
    let rows = pallama_runtime::hf::fit_rows(&info.siblings, vram_bytes, cfg.default_ctx);
    println!(
        "fit preview for {} (local VRAM: {})",
        parsed.repo,
        humansize(i64::try_from(vram_bytes).unwrap_or(i64::MAX))
    );
    println!(
        "{:<10} {:>10} {:>8} {:>12} {:>12}  FILE",
        "QUANT", "SIZE", "FITS", "REC_CTX", "CTX@KV_Q8"
    );
    for r in rows {
        println!(
            "{:<10} {:>10} {:>8} {:>12} {:>12}  {}",
            r.quant,
            humansize(i64::try_from(r.bytes).unwrap_or(i64::MAX)),
            if r.fits_vram { "yes" } else { "no" },
            r.recommended_ctx,
            r.recommended_ctx_q8,
            r.file
        );
    }
    Ok(())
}

fn config_cmd(cmd: ConfigCmd) -> Result<()> {
    match cmd {
        ConfigCmd::List => {
            let cfg = config()?;
            println!("{}", cfg.to_toml().map_err(|e| anyhow!("{e}"))?);
            Ok(())
        }
        ConfigCmd::Get { key } => {
            let cfg = config()?;
            let raw = cfg.to_toml().map_err(|e| anyhow!("{e}"))?;
            if let Some(line) = raw
                .lines()
                .find(|l| l.starts_with(&format!("{key} =")))
            {
                println!("{line}");
                Ok(())
            } else {
                Err(anyhow!("unknown config key: {key}"))
            }
        }
        ConfigCmd::Set { key, value } => {
            let path = dirs().config_file();
            let raw = std::fs::read_to_string(&path)?;
            // Values that are not already TOML scalars (numbers, bools,
            // quoted strings, arrays) are written as double-quoted strings:
            // `config set child_transport tcp` must not produce
            // `child_transport = tcp` (invalid TOML). A one-line parse probe
            // decides; the full candidate is validated below either way.
            let scalar = toml::from_str::<toml::Table>(&format!("v = {value}\n")).is_ok();
            let stored = if scalar { value.clone() } else { format!("{value:?}") };
            let mut out: Vec<String> = Vec::new();
            let mut replaced = false;
            for line in raw.lines() {
                if line.starts_with(&format!("{key} =")) {
                    out.push(format!("{key} = {stored}"));
                    replaced = true;
                } else {
                    out.push(line.to_string());
                }
            }
            if !replaced {
                // A NEW top-level key must go ABOVE the first table header
                // (`[engine_env]`, `[model_overrides.x]`…); appending at the
                // end would nest it inside that table.
                let insert_at = out
                    .iter()
                    .position(|l| l.starts_with('['))
                    .unwrap_or(out.len());
                out.insert(insert_at, format!("{key} = {stored}"));
            }
            let candidate = out.join("\n") + "\n";
            // Validate BEFORE persisting: a bad value/unknown key must
            // never leave the file broken.
            Config::from_toml(&candidate)
                .map_err(|e| anyhow!("rejected, file unchanged: {e}"))?;
            std::fs::write(&path, &candidate)?;
            println!("{key} = {stored}");
            Ok(())
        }
    }
}

/// `pallama upgrade [--version vN] [--dry-run]`: self-update from GitHub
/// Releases with the same digest verification as `pallama engine update`.
async fn upgrade(version: Option<String>, dry_run: bool) -> Result<()> {
    let repo = std::env::var("PALLAMA_REPO")
        .ok()
        .filter(|r| !r.is_empty())
        .ok_or_else(|| {
            anyhow!("PALLAMA_REPO is not set; export PALLAMA_REPO=owner/pallama (the repo hosting pallama releases)")
        })?;
    let base = std::env::var("PALLAMA_INSTALL_BASE_URL")
        .unwrap_or_else(|_| "https://api.github.com".to_string());
    let token = std::env::var("GH_TOKEN")
        .or_else(|_| std::env::var("GITHUB_TOKEN"))
        .ok();
    let client = pallama_runtime::GhClient::with_base(&base, token)
        .map_err(|e| anyhow!("{e}"))?;
    let summary =
        pallama_runtime::upgrade::run(&client, &repo, version.as_deref(), dry_run).await;
    println!("{summary}");
    if summary.starts_with("upgrade failed") {
        return Err(anyhow!("upgrade failed"));
    }
    Ok(())
}


#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn unit__parse_modelfile__supported_subset() {
        let spec = parse_modelfile("# comment\nFROM qwen3.5-9b\nPARAMETER num_ctx 32768\nADAPTER /tmp/a.gguf\n").unwrap();
        assert_eq!(spec.base, "qwen3.5-9b");
        assert_eq!(spec.ctx, Some(32768));
        assert_eq!(spec.loras, vec!["/tmp/a.gguf"]);
        assert!(spec.unsupported.is_empty());
    }

    #[test]
    fn unit__parse_modelfile__unsupported_keys_collected_not_dropped() {
        let spec = parse_modelfile("FROM m\nPARAMETER temperature 0.7\nSYSTEM you are a pirate\nTEMPLATE {{x}}\n").unwrap();
        assert!(spec.unsupported.contains(&"PARAMETER temperature".to_string()));
        assert!(spec.unsupported.iter().any(|u| u.starts_with("SYSTEM")));
        assert!(spec.unsupported.iter().any(|u| u.starts_with("TEMPLATE")));
    }

    #[test]
    fn unit__parse_modelfile__missing_from_is_error() {
        assert!(parse_modelfile("PARAMETER num_ctx 8\n").is_err());
    }
}
