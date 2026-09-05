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
    Serve,
    /// Stop the daemon
    Stop,
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
    /// Remove a model (refuses while running)
    Rm { model: String },
    /// List pulled models
    List,
    /// Show model details, active profile, last benchmark
    Show { model: String },
    /// Live instances: state, ctx, idle countdown
    Ps {
        /// Clear crash circuit breakers
        #[arg(long)]
        reset: bool,
    },
    /// Chat REPL against a model (streams; /exit /clear /model /sysinfo /profile)
    Run { model: String },
    /// Benchmark a model (pp/tg table)
    Bench { model: String },
    /// Tune a model's launch profile (--search = measured grid argmax)
    Tune {
        model: String,
        #[arg(long)]
        search: bool,
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
        Cmd::Stop => stop(),
        Cmd::Pull { target } => pull(&target).await,
        Cmd::Import { path, name, quant, copy } => import(&path, name, quant, copy),
        Cmd::Rm { model } => rm(&model),
        Cmd::List => list(),
        Cmd::Show { model } => show(&model),
        Cmd::Ps { reset } => ps(reset).await,
        Cmd::Run { model } => run_repl(&model).await,
        Cmd::Bench { model } => bench(&model),
        Cmd::Tune { model, search } => tune(&model, search),
        Cmd::Engine { cmd } => engine_cmd(cmd).await,
        Cmd::Lora { cmd } => lora_cmd(cmd),
        Cmd::Search { query } => search(&query).await,
        Cmd::Fit { target } => fit(&target).await,
        Cmd::Config { cmd } => config_cmd(cmd),
    }
}

async fn serve() -> Result<()> {
    banner();
    let d = dirs();
    d.ensure().ok();
    let cfg = config()?;
    let _lock = pallama_runtime::DaemonLock::acquire(&d)
        .map_err(|e| anyhow!("{e}"))?;
    let store = Store::open(&d)?;
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
    let puller = pallama_runtime::Puller { dirs: d.clone(), client, bus: EventBus::default() };
    let row = puller.pull(target).await?;
    println!(
        "pulled {}: {} ({}, {} shards) -> {}",
        row.name,
        humansize(row.bytes),
        row.quant,
        row.shards,
        row.path
    );
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

fn rm(model: &str) -> Result<()> {
    let d = dirs();
    pallama_runtime::remove_model(&d, model).map_err(|e| anyhow!("{e:#}"))?;
    println!("removed {model}");
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

/// Stream an /api/chat NDJSON response to stdout.
#[allow(clippy::duration_suboptimal_units)] // 10-minute generation ceiling
async fn stream_chat(base: &str, body: &serde_json::Value) -> Result<()> {
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
            }
        }
        out.flush().ok();
    }
    writeln!(out).ok();
    Ok(())
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

fn tune(model: &str, search: bool) -> Result<()> {
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
    );
    let store2 = Store::open(&d)?;
    let tuner = pallama_runtime::Tuner { dirs: &d, bench_bin };
    if search {
        let (profile, _winning, rows) = tuner.tune_search(&store2, &input)?;
        println!("tuned {} — {} configs measured, winner:", model, rows.len());
        println!("  argv: {}", profile.argv.join(" "));
        println!("  ctx:   {}", profile.ctx);
        if !profile.warnings.is_empty() {
            for w in &profile.warnings {
                println!("  note:  {w}");
            }
        }
    } else {
        let rows = tuner.bench_default(std::path::Path::new(&row.path))?;
        println!("{:.0} rows; run with --search to adopt the argmax profile", rows.len());
    }
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
    println!("{:<10} {:>10} {:>8} {:>12}  FILE", "QUANT", "SIZE", "FITS", "REC_CTX");
    for r in rows {
        println!(
            "{:<10} {:>10} {:>8} {:>12}  {}",
            r.quant,
            humansize(i64::try_from(r.bytes).unwrap_or(i64::MAX)),
            if r.fits_vram { "yes" } else { "no" },
            r.recommended_ctx,
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
            let raw = std::fs::read_to_string(&path).unwrap_or_default();
            let mut out: Vec<String> = Vec::new();
            let mut replaced = false;
            for line in raw.lines() {
                if line.starts_with(&format!("{key} =")) {
                    out.push(format!("{key} = {value}"));
                    replaced = true;
                } else {
                    out.push(line.to_string());
                }
            }
            if !replaced {
                out.push(format!("{key} = {value}"));
            }
            let candidate = out.join("\n") + "\n";
            // Validate BEFORE persisting: a bad value/unknown key must
            // never leave the file broken.
            Config::from_toml(&candidate).map_err(|e| anyhow!("rejected, file unchanged: {e}"))?;
            std::fs::write(&path, &candidate)?;
            println!("{key} = {value}");
            Ok(())
        }
    }
}

