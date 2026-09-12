//! `pallama` — llama.cpp orchestration platform.
//!
//! Local-only by design: no telemetry, no cloud endpoints; the only
//! outbound traffic is user-initiated engine/model downloads. Powered by
//! llama.cpp / ggml / ggerganov — <https://github.com/ggml-org/llama.cpp>

use anyhow::{anyhow, Context, Result};
use clap::{CommandFactory, Parser, Subcommand};
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use pallama_core::engine_kind::EngineKind;
use pallama_core::{Config, PallamaDirs, Store};
use pallama_runtime::engine::build::{
    detect_toolchain, nvidia_gpu_facts, path_dirs, BuildBackend, BuildOpts, Toolchain,
};
use pallama_runtime::engine::gh::{btag_number, same_build, GhClient};
use pallama_runtime::engine::EngineManager;
use pallama_runtime::engine_impl::Engine;
use pallama_runtime::EventBus;
use pallama_runtime::{LlamaCppEngine, MistralRsEngine, Supervisor};

#[derive(Parser)]
#[command(
    name = "pallama",
    version,
    about = "llama.cpp orchestration: pallama-grade UX, zero engine fork",
    after_help = "Quickstart: pallama pull <model> · pallama run <model> · pallama doctor\n\nLocal-only: no telemetry, no cloud endpoints. Powered by llama.cpp / ggml / ggerganov."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the daemon in the foreground (`OpenAI` + `Anthropic` + ollama
    /// APIs on one port)
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
    /// Manage API keys (list / add / rm / rotate against the running daemon)
    Keys {
        #[command(subcommand)]
        action: Option<KeysAction>,
    },
    /// Derive a new quantization of a pulled model using the engine's
    /// own llama-quantize (output registered in the store)
    Quantize {
        model: String,
        /// Target quantization (e.g. `Q4_K_M`, `Q5_K_S`, `IQ4_XS`, f16)
        #[arg(short = 't', long)]
        qtype: String,
        /// Registered name for the output (default: <model-base>-<qtype>)
        #[arg(long)]
        name: Option<String>,
        /// Importance-matrix quantization: calibration text file for
        /// llama-imatrix (better Q4 accuracy than plain k-quants)
        #[arg(long)]
        imatrix: Option<PathBuf>,
        /// Allow quantizing a source that is already quantized (upstream
        /// llama-quantize flag; quality risk — pair with --verify)
        #[arg(long)]
        allow_requantize: bool,
        /// Verify the output with llama-perplexity before registering:
        /// reject (and delete) it when perplexity degrades beyond the gate
        #[arg(long)]
        verify: bool,
        /// Maximum tolerated relative perplexity degradation for --verify,
        /// in percent (default 10.0; improvements always pass)
        #[arg(long, default_value_t = 10.0)]
        max_degradation: f64,
    },
    /// Run an agent CLI against the daemon: sets OpenAI/Anthropic/ollama
    /// base-URL env, ensures the daemon is up, then execs COMMAND
    /// (e.g. `pallama launch claude` / `pallama launch dsh`)
    Launch {
        /// The CLI to exec (searched on PATH; args after `--`)
        #[arg(trailing_var_arg = true)]
        command: Vec<String>,
        /// Model to pre-warm before exec (optional)
        #[arg(long)]
        warm: Option<String>,
        /// API key to hand the CLI (`PALLAMA_KEY` env or a key name here)
        #[arg(long)]
        key: Option<String>,
    },
    /// pallama account sign-in — refused: no cloud accounts by design
    Signin,
    /// pallama account login — refused: no cloud accounts by design
    Login,
    /// pallama account sign-out — refused: no cloud accounts by design
    Signout,
    /// pallama account logout — refused: no cloud accounts by design
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
        /// Vision projector GGUF to attach alongside the imported model
        #[arg(long)]
        mmproj: Option<PathBuf>,
    },
    /// Attach a vision projector (mmproj GGUF) to an existing model;
    /// refuses while the model is running — next spawn picks it up
    Mmproj { model: String, path: PathBuf },
    /// Remove models (refuses while running)
    Rm { models: Vec<String> },
    /// List pulled models
    #[command(alias = "ls")]
    List,
    /// Show model details, active profile, last benchmark
    Show { model: String },
    /// Live instances: state, ctx, in-flight requests, idle countdown
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
        /// Cap single-shot generation (tokens); REPL mode is unaffected.
        /// Without it a reasoning model can ramble to the context limit.
        #[arg(long)]
        max_tokens: Option<u64>,
    },
    /// Benchmark a model (pp/tg table; history kept for tune gates)
    Bench { model: String },
    /// Tune a model's launch profile (--search = measured grid argmax;
    /// live probes: --slots/--ngram/--load/--replicas/--cache-reuse)
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
        /// Live-concurrency slots search: C concurrent clients against
        /// -np candidates {1,2,4}; prints the table and (with --search)
        /// adopts the winner as the global `slots` config
        #[arg(long)]
        slots: Option<u32>,
        /// Live n-gram tuning search: spawns real children across
        /// (`size_m`, `min_hits`) candidates and adopts the measured winner
        /// as the global ngram_* config (requires/sets spec = "ngram")
        #[arg(long)]
        ngram: bool,
        /// Live warmup A/B: spawn the model twice (with/without
        /// --no-warmup), measure ready+first-chat, and adopt
        /// `model_overrides.<model>.warmup = false` only if >5% faster
        #[arg(long)]
        load: bool,
        /// Live replica scaling search: aggregate tok/s with 1 vs 2
        /// identical children (8 clients x 3 gens); adopts
        /// `model_overrides.<model>.replicas = 2` only if >1.3x
        #[arg(long)]
        replicas: bool,
        /// Live `--cache-reuse N` grid {0, 256, 512}: warm second-chat wall
        /// seconds on a long repeated prompt; adopts the winner only if it
        /// beats the current default (256) by >5%
        #[arg(long)]
        cache_reuse: bool,
    },
    /// Persist config migrations (legacy `api_keys` -> [[keys]]), with a
    /// timestamped backup; idempotent
    Migrate,
    /// Backup config + store + sessions manifest to a timestamped dir
    Snapshot,
    /// Generate shell completions (bash | zsh | fish | powershell) to
    /// stdout — `pallama completions bash > .../pallama.bash`
    Completions { shell: clap_complete::Shell },
    /// Suggest draft-model candidates (EAGLE3/MTP heads + small
    /// same-family models) for speculative decoding of MODEL
    Drafts { model: String },
    /// Co-residency plan: which local models fit in VRAM together
    /// (weights + f16 KV at each model's ctx; hot-first greedy)
    Coreside,
    /// Transcribe an audio file (local whisper.cpp lane, else a whisper:
    /// [[remotes]] entry); lane management: --install/--pull/--list/--pin
    Whisper {
        /// Audio file (wav/mp3/flac/...) — omit with --install/--pull/--list
        file: Option<PathBuf>,
        /// Model name (default: local installed whisper model, else whisper-1 remote)
        #[arg(long)]
        model: Option<String>,
        /// Install the whisper.cpp server binary from its GitHub release
        #[arg(long)]
        install: bool,
        /// Install this specific release tag and pin the runtime to it
        /// (plain --install tracks the latest release)
        #[arg(long, requires = "install")]
        tag: Option<String>,
        /// Download a ggml model by size (base, small.en, large-v3-turbo, ...)
        #[arg(long)]
        pull: Option<String>,
        /// List installed whisper server tag + local models
        #[arg(long)]
        list: bool,
        /// Pin the whisper server to an installed tag ("none" unpins —
        /// tracks the newest installed tag). No download involved.
        #[arg(
            long,
            value_name = "TAG|none",
            conflicts_with_all = ["install", "tag", "pull", "list", "file"]
        )]
        pin: Option<String>,
    },
    /// Engine management: llama.cpp releases, mistral.rs lane, source
    /// builds (`build cuda|cpu`), rollback + update channels
    Engine {
        #[command(subcommand)]
        cmd: EngineCmd,
    },
    /// `LoRA` adapter management
    Lora {
        #[command(subcommand)]
        cmd: LoraCmd,
    },
    /// Search Hugging Face for GGUF repos (multiple words are joined)
    Search {
        #[arg(trailing_var_arg = true, num_args = 1..)]
        query: Vec<String>,
    },
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
    /// state (survives unload and restart; restore is identity-checked
    /// against the engine/model/ctx shape it was saved under)
    Session {
        #[command(subcommand)]
        cmd: SessionCmd,
    },
    /// Diagnose the local setup: config, engine, keys, remotes, whisper,
    /// hardware, disk, models
    Doctor,
    /// Ask why a request misbehaved: sentinel detections for a trace id
    /// (or the most recent observations) with fix hints
    Why {
        /// Trace id from the x-pallama-trace-id response header
        trace: Option<String>,
        /// Tail sentinel detections live instead (same as `pallama watch`)
        #[arg(long, conflicts_with = "trace")]
        watch: bool,
    },
    /// Live tail of sentinel detections as they happen (Ctrl-C to stop)
    Watch,
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
    Update {
        tag: Option<String>,
        /// Skip the decode-regression gate (tune-baseline bench compare)
        #[arg(long)]
        no_gate: bool,
    },
    /// List installed engines with capability summaries
    List,
    /// Activate an installed tag
    Use { tag: String },
    /// Step back to the previous engine
    Rollback,
    /// Register a locally built llama-server (pseudo-tag "local")
    Local { path: PathBuf },
    /// Compile llama.cpp from source into an installable engine
    /// (Linux-CUDA prebuilts don't exist upstream; the backend is in-tree)
    Build {
        /// Backend to compile: cuda | cpu
        backend: String,
        /// Upstream b-tag to build (default: the update channel's target)
        tag: Option<String>,
        /// CUDA architectures override (e.g. 89, or 80;86), for
        /// cross-builds without a local GPU
        #[arg(long)]
        arch: Option<String>,
        /// CUDA host compiler override (e.g. /usr/bin/g++-12)
        #[arg(long)]
        cuda_host_compiler: Option<PathBuf>,
        /// Parallel compile jobs (default: CPU count)
        #[arg(long)]
        jobs: Option<usize>,
        /// Skip the decode-regression gate (tune-baseline bench compare)
        #[arg(long)]
        no_gate: bool,
    },
    /// Install + activate a mistral.rs engine (prebuilt upstream binary;
    /// picks CPU/Metal/CUDA asset from the local GPU + driver)
    Install { tag: Option<String> },
}

#[derive(Subcommand)]
enum LoraCmd {
    Add {
        model: String,
        path: PathBuf,
        #[arg(default_value = "1.0")]
        scale: f64,
    },
    Rm {
        id: i64,
    },
    List {
        model: Option<String>,
    },
}

#[derive(Subcommand)]
enum ConfigCmd {
    /// Print the full effective config as TOML (one knob surface)
    List,
    /// Print one knob's current line (`config get slots`)
    Get { key: String },
    /// Set a knob (`config set slots 1`); the candidate is validated
    /// against the config schema before the file is touched — a bad
    /// value or unknown key is rejected with the file unchanged.
    /// Model-scoped overrides live in `[model_overrides."<model>"]`
    /// tables; edit those in the file directly.
    Set { key: String, value: String },
}

/// Grouping table for the top-level help. Descriptions and aliases come
/// live from clap (single source of truth); this table owns ONLY the
/// grouping. `unit__grouped_help__covers_every_subcommand` pins the two
/// together: a command added to the enum but not to a group (or vice
/// versa) fails `cargo test`.
const HELP_GROUPS: &[(&str, &[&str])] = &[
    (
        "Serve & Chat",
        &["serve", "run", "ps", "stop", "launch", "session"],
    ),
    (
        "Model Management",
        &[
            "pull", "import", "cp", "create", "rm", "list", "show", "quantize", "mmproj", "search",
            "fit", "lora",
        ],
    ),
    (
        "Tuning & Benchmarks",
        &["bench", "tune", "drafts", "coreside"],
    ),
    (
        "Engine & Config",
        &[
            "engine",
            "config",
            "keys",
            "whisper",
            "doctor",
            "migrate",
            "snapshot",
            "upgrade",
            "completions",
        ],
    ),
    ("Observability", &["why", "watch"]),
    (
        "Refused by design (local-only)",
        &["push", "signin", "login", "signout", "logout"],
    ),
    ("General", &["help"]),
];

/// Render the top-level help with commands grouped by category. clap has
/// no native subcommand grouping (verified against clap 4.6); the enums
/// stay authoritative for parsing, per-command help and completions.
fn render_grouped_help() -> String {
    let mut cmd = Cli::command();
    // Materialize clap's auto `help` subcommand (added lazily at build).
    cmd.build();
    let mut about = std::collections::BTreeMap::new();
    let mut aliases = std::collections::BTreeMap::new();
    for sc in cmd.get_subcommands() {
        about.insert(
            sc.get_name().to_string(),
            sc.get_about()
                .map(ToString::to_string)
                .unwrap_or_default()
                .replace('\n', " "),
        );
        let al: Vec<String> = sc.get_all_aliases().map(str::to_string).collect();
        if !al.is_empty() {
            aliases.insert(sc.get_name().to_string(), format!(" ({})", al.join(", ")));
        }
    }
    // Width: longest "name (aliases)" across ALL groups, for one aligned
    // column (clap-style two-space indent, two-space gutter).
    let entry = |n: &str| format!("{n}{}", aliases.get(n).map_or("", String::as_str));
    let width = HELP_GROUPS
        .iter()
        .flat_map(|(_, ns)| ns.iter())
        .map(|n| entry(n).len())
        .max()
        .unwrap_or(0);
    let mut out = String::new();
    writeln!(
        out,
        "pallama {} — llama.cpp orchestration: pallama-grade UX, zero engine fork",
        env!("CARGO_PKG_VERSION")
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "Usage: pallama <COMMAND>").unwrap();
    for (heading, names) in HELP_GROUPS {
        writeln!(out, "\n{heading}:").unwrap();
        for n in *names {
            let a = about
                .get(*n)
                .unwrap_or_else(|| panic!("help group lists unknown command {n:?}"));
            writeln!(out, "  {:<width$}  {}", entry(n), a, width = width).unwrap();
        }
    }
    writeln!(out, "\nOptions:").unwrap();
    writeln!(out, "  -h, --help     Print help").unwrap();
    writeln!(out, "  -V, --version  Print version").unwrap();
    writeln!(
        out,
        "\nQuickstart: pallama pull <model> · pallama run <model> · pallama doctor"
    )
    .unwrap();
    writeln!(
        out,
        "\nLocal-only: no telemetry, no cloud endpoints. Powered by llama.cpp / ggml / ggerganov."
    )
    .unwrap();
    out
}

fn main() {
    // Grouped help intercept: clap renders an ungrouped 38-command wall.
    // Only the TOP-level listing is replaced; `pallama help <cmd>`,
    // `pallama <cmd> --help` and error usage stay clap-native.
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let grouped_help = match argv.first().map(String::as_str) {
        Some("--help" | "-h") => true,
        Some("help") => argv.len() == 1,
        _ => false,
    };
    if grouped_help {
        print!("{}", render_grouped_help());
        return;
    }
    let cli = Cli::parse();
    // Peek log_level from the config file WITHOUT creating it
    // (Config::load writes a fresh file when absent; a plain read must not).
    let log_level = std::fs::read_to_string(dirs().config_file())
        .ok()
        .and_then(|raw| Config::from_toml(&raw).ok())
        .and_then(|c| c.log_level);
    pallama_core::telemetry::init_tracing(0, log_level.as_deref());
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

/// Bounded HTTP client for one-shot CLI calls (F126): a wedged daemon
/// must fail the command, not hang it. Streaming lanes (watch, chat
/// stream) deliberately build their own unbounded/600s clients.
fn cli_http() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap_or_default()
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

/// The serving daemon keeps the engine it booted with (supervisor holds
/// the engine at construction) — after any engine switch, a running
/// daemon must be restarted before it serves the new binary. Probe-only:
/// silent when no daemon is up.
/// Restart command lanes after an engine switch, privilege-free first:
/// a user-scope service restarts without elevation; a system-scope one
/// is attempted once via passwordless sudo (`sudo -n`, fails fast when a
/// password would be needed) — never prompting mid-command.
fn restart_lanes(os: &str, home: &str, system_unit: bool, user_unit: bool) -> Vec<Vec<String>> {
    fn cmd(parts: &[&str]) -> Vec<String> {
        parts.iter().map(std::string::ToString::to_string).collect()
    }
    let mut lanes: Vec<Vec<String>> = Vec::new();
    match os {
        "linux" => {
            if user_unit {
                lanes.push(cmd(&["systemctl", "--user", "restart", "pallama"]));
            }
            if system_unit {
                lanes.push(cmd(&["sudo", "-n", "systemctl", "restart", "pallama"]));
            }
        }
        "macos" => {
            // crate is forbid(unsafe_code): resolve the uid without libc
            let uid = std::process::Command::new("id")
                .arg("-u")
                .output()
                .ok()
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .unwrap_or_default();
            if !uid.is_empty() {
                lanes.push(cmd(&[
                    "launchctl",
                    "kickstart",
                    "-k",
                    &format!("gui/{uid}/dev.pallama"),
                ]));
            }
            if system_unit {
                lanes.push(cmd(&[
                    "sudo",
                    "-n",
                    "launchctl",
                    "kickstart",
                    "-k",
                    "system/dev.pallama",
                ]));
            }
        }
        _ => {}
    }
    let _ = home;
    lanes
}

/// Post-engine-switch daemon action. Knob off (default): one-line hint.
/// Knob on (`auto_restart_engine_switch = true`) and the daemon is
/// alive: restart it through the first working lane and wait for
/// /healthz; when no lane works (manual daemon, locked-down system
/// unit) fall back to the printed hint with the reason.
async fn restart_hint() {
    let Ok(cfg) = config() else { return };
    let base = daemon_base(&cfg);
    let Ok(http) = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
    else {
        return;
    };
    let alive = match http.get(format!("{base}/healthz")).send().await {
        Ok(r) => r.status().is_success(),
        Err(_) => false,
    };
    if !alive {
        return; // nothing serving: the next start picks the engine up
    }
    let hint = "a running daemon serves with the engine it booted with — \
         restart it to pick up the switch (systemd: `systemctl restart pallama`)";
    if !cfg.auto_restart_engine_switch {
        println!("note: {hint}");
        return;
    }
    let home = std::env::var("HOME").unwrap_or_default();
    let lanes = restart_lanes(
        std::env::consts::OS,
        &home,
        Path::new("/etc/systemd/system/pallama.service").exists()
            || Path::new("/Library/LaunchDaemons/dev.pallama.plist").exists(),
        Path::new(&format!("{home}/.config/systemd/user/pallama.service")).exists(),
    );
    let mut last_why = "no service manager lane for this platform".to_string();
    for lane in &lanes {
        match tokio::process::Command::new(&lane[0])
            .args(&lane[1..])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .await
        {
            Ok(s) if s.success() => {
                // wait for the daemon to answer /healthz again (≤30s)
                for _ in 0..60 {
                    if let Ok(r) = http.get(format!("{base}/healthz")).send().await {
                        if r.status().is_success() {
                            println!(
                                "note: daemon restarted (auto_restart_engine_switch) — it now serves the new engine"
                            );
                            return;
                        }
                    }
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
                last_why = "restart command ran but /healthz never came back within 30s".into();
            }
            Ok(s) => {
                last_why = format!("`{}` exited with {s} (needs a password?)", lane.join(" "));
            }
            Err(e) => {
                last_why = format!("`{}` failed to spawn: {e}", lane.join(" "));
            }
        }
    }
    println!("note: could not auto-restart the daemon ({last_why}) — {hint}");
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
    #[cfg(windows)]
    {
        // F132: without DETACHED_PROCESS the auto-started daemon shares
        // the CLI's console — Ctrl-C at the prompt kills it too.
        use std::os::windows::process::CommandExt as _;
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        cmd.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP);
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
// Pure one-call-per-arm dispatch over 30+ commands; splitting arms into
// helpers would add indirection without lowering complexity.
#[allow(clippy::too_many_lines)]
async fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Serve => serve().await,
        Cmd::Stop { model } => stop_cmd(model).await,
        Cmd::Pull { target } => pull(&target).await,
        Cmd::Import {
            path,
            name,
            quant,
            copy,
            mmproj,
        } => import(&path, name, quant, copy, mmproj.as_ref()),
        Cmd::Mmproj { model, path } => mmproj_cmd(&model, &path),
        Cmd::Rm { models } => rm_multi(&models),
        Cmd::List => list(),
        Cmd::Show { model } => show(&model),
        Cmd::Ps { reset } => ps(reset).await,
        Cmd::Run {
            model,
            prompt,
            verbose,
            max_tokens,
        } => run_dispatch(&model, &prompt, verbose, max_tokens).await,
        Cmd::Bench { model } => bench(&model),
        Cmd::Tune {
            model,
            search,
            ctx,
            spec,
            slots,
            ngram,
            load,
            replicas,
            cache_reuse,
        } => tune_full(
            &model,
            search,
            ctx,
            spec,
            slots,
            ngram,
            load,
            replicas,
            cache_reuse,
        ),
        Cmd::Engine { cmd } => engine_cmd(cmd).await,
        Cmd::Lora { cmd } => lora_cmd(cmd),
        Cmd::Search { query } => search(&query.join(" ")).await,
        Cmd::Fit { target } => fit(&target).await,
        Cmd::Config { cmd } => config_cmd(cmd),
        Cmd::Upgrade { version, dry_run } => upgrade(version, dry_run).await,
        Cmd::Cp {
            source,
            destination,
        } => cp_cmd(&source, &destination),
        Cmd::Create { model, file } => create_cmd(&model, file.as_deref()),
        Cmd::Push { model } => cloud_refusal("push", &model),
        Cmd::Keys { action } => keys_cmd(action).await,
        Cmd::Quantize {
            model,
            qtype,
            name,
            imatrix,
            allow_requantize,
            verify,
            max_degradation,
        } => {
            let gate = verify.then_some(max_degradation);
            quantize_cmd(
                &model,
                &qtype,
                name.as_deref(),
                imatrix.as_deref(),
                allow_requantize,
                gate,
            )
        }
        Cmd::Launch { command, warm, key } => launch_cmd(command, warm, key).await,
        Cmd::Whisper {
            file,
            model,
            install,
            tag,
            pull,
            list,
            pin,
        } => whisper_cmd(file.as_ref(), model, install, tag, pull, list, pin).await,
        Cmd::Coreside => coreside_cmd(),
        Cmd::Drafts { model } => drafts_cmd(&model).await,
        Cmd::Migrate => migrate_cmd(),
        Cmd::Snapshot => snapshot_cmd(),
        Cmd::Completions { shell } => {
            let mut cmd = Cli::command();
            let name = cmd.get_name().to_string();
            let mut buf = std::io::stdout().lock();
            clap_complete::generate(shell, &mut cmd, name, &mut buf);
            Ok(())
        }
        Cmd::Signin => cloud_refusal("signin", ""),
        Cmd::Login => cloud_refusal("login", ""),
        Cmd::Signout => cloud_refusal("signout", ""),
        Cmd::Logout => cloud_refusal("logout", ""),
        Cmd::Session { cmd } => session_cmd(cmd).await,
        Cmd::Doctor => doctor().await,
        Cmd::Why { trace, watch: live } => {
            if live {
                watch().await
            } else {
                why(trace.as_deref()).await
            }
        }
        Cmd::Watch => watch().await,
    }
}

async fn session_cmd(cmd: SessionCmd) -> Result<()> {
    let base = ensure_daemon().await?;
    let client = cli_http();
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
        Self {
            name,
            ok: true,
            warn: false,
            detail: detail.into(),
        }
    }
    fn warn(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            ok: true,
            warn: true,
            detail: detail.into(),
        }
    }
    fn fail(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            ok: false,
            warn: false,
            detail: detail.into(),
        }
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

/// `pallama doctor` — local-state diagnostics: config, store, engine
/// manifest, hardware probe, plus 4s-capped upstream currency probes
/// (engine/whisper/app). Warn-only — never installs or starts anything
/// (the engine row executes the engine binary with `--version`, nothing
/// more).
async fn doctor() -> Result<()> {
    let d = dirs();
    let mut checks: Vec<Check> = Vec::new();

    // 1. config parses + validates
    match config() {
        Ok(cfg) => {
            let legacy = std::fs::read_to_string(d.config_file())
                .is_ok_and(|raw| pallama_core::Config::raw_has_legacy_keys(&raw));
            if legacy {
                checks.push(Check::warn(
                    "config",
                    "parsed with an IN-MEMORY migration (legacy api_keys) — persist with `pallama migrate`".to_string(),
                ));
            } else {
                checks.push(Check::ok(
                    "config",
                    format!(
                        "{} validates; host {}:{}",
                        d.config_file().display(),
                        cfg.host,
                        cfg.port
                    ),
                ));
            }
            if cfg.agent {
                checks.push(Check::warn(
                    "agent mode",
                    "enabled: children expose built-in tools (exec_shell_command) + MCP CORS proxy",
                ));
            }
            if !cfg.cpu_range.is_empty() {
                checks.push(Check::ok(
                    "cpu_range",
                    format!("pinned to {}", cfg.cpu_range),
                ));
            }
        }
        Err(e) => {
            checks.push(Check::fail(
                "config",
                format!(
                    "{} — fix or delete {} to regenerate defaults",
                    e,
                    d.config_file().display()
                ),
            ));
        }
    }

    checks.extend(doctor_engine(&d).await);
    checks.extend(doctor_whisper_currency(&d).await);
    checks.extend(doctor_whisper_models(&d));
    checks.extend(doctor_binary_shadow());
    checks.extend(doctor_app_currency().await);
    checks.extend(doctor_port().await);
    checks.extend(doctor_service());
    #[cfg(unix)]
    checks.extend(doctor_disk(&d));
    checks.extend(doctor_store(&d));
    checks.extend(doctor_models(&d));
    checks.extend(doctor_sentinel(&d));
    checks.extend(doctor_exposure(&d));
    checks.extend(doctor_keys(&d));
    checks.extend(doctor_remotes().await);

    // render
    println!("{:<26} {:<5} DETAIL", "CHECK", "ST");
    for c in &checks {
        println!("{:<26} {:<5} {}", c.name, c.status_word(), c.detail);
    }
    let fails = checks.iter().filter(|c| !c.ok).count();
    let warns = checks.iter().filter(|c| c.warn).count();
    if fails > 0 {
        println!("\n{fails} failing check(s) — fix the FAIL rows above");
    } else {
        if warns > 0 {
            println!("\nall checks pass; {warns} warning(s)");
        } else {
            println!("\nall checks pass");
        }
        for step in doctor_next_steps(&checks) {
            println!("next: {step}");
        }
    }
    Ok(())
}

/// Adaptive "what to do next" footer for `pallama doctor`. Reuses the
/// computed checks instead of re-querying state: the port row encodes
/// daemon liveness, the models row encodes the pull count ("0 pulled").
/// Empty when the failing-checks branch already told the user what to do.
fn doctor_next_steps(checks: &[Check]) -> Vec<String> {
    let port_up = checks.iter().any(|c| c.name == "port" && c.ok);
    let no_models = checks
        .iter()
        .any(|c| c.name == "models" && c.ok && c.detail.starts_with("0 pulled"));
    let mut steps = Vec::new();
    if !port_up {
        steps.push("start the daemon: pallama serve (or: systemctl start pallama)".to_string());
    }
    if no_models {
        steps
            .push("pull a model: pallama pull <name> (find one: pallama search qwen3)".to_string());
    }
    if steps.is_empty() {
        steps.push("chat: pallama run <model>".to_string());
    }
    steps
}

/// F8: keys health — parse, uniqueness, admin existence, gateway
/// enforcement state (a config with keys only bites when the daemon
/// actually runs this config).
fn doctor_keys(d: &PallamaDirs) -> Vec<Check> {
    let mut out = Vec::new();
    let Ok(cfg) = pallama_core::Config::from_toml(
        &std::fs::read_to_string(d.config_file()).unwrap_or_default(),
    ) else {
        return out; // config check already failed loudly
    };
    if cfg.keys.is_empty() {
        out.push(Check::ok(
            "keys",
            "none configured (authless loopback) — add via `pallama keys add` before exposing the port",
        ));
        return out;
    }
    let admins = cfg.keys.iter().filter(|k| k.models.is_empty()).count();
    if admins == 0 {
        out.push(Check::warn(
            "keys",
            "no unscoped (admin) key — /api/keys management is unreachable",
        ));
    }
    let budgets = cfg
        .keys
        .iter()
        .filter(|k| k.rpm > 0 || k.tpm > 0 || k.daily_tokens > 0)
        .count();
    out.push(Check::ok(
        "keys",
        format!(
            "{} key(s), {admins} admin, {budgets} with budgets; auth: Bearer + x-api-key",
            cfg.keys.len()
        ),
    ));
    out
}

/// F8: remotes health — one 3s probe per [[remotes]] entry (the same
/// probe `ps` uses; dead remotes warn, not fail — they are optional
/// lanes).
async fn doctor_remotes() -> Vec<Check> {
    let mut out = Vec::new();
    let Ok(cfg) = config() else {
        return out;
    };
    if cfg.remotes.is_empty() {
        return out;
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .expect("doctor client");
    for r in &cfg.remotes {
        let mut req = client.get(format!("{}/v1/models", r.url.trim_end_matches('/')));
        if !r.key.is_empty() {
            req = req.bearer_auth(&r.key);
        }
        match req.send().await {
            Ok(resp) if resp.status().is_success() => {
                out.push(Check::ok(
                    "remote",
                    format!("{} reachable at {}", r.name, r.url),
                ));
            }
            Ok(resp) => out.push(Check::warn(
                "remote",
                format!(
                    "{} answered HTTP {} — check its auth/model list",
                    r.name,
                    resp.status()
                ),
            )),
            Err(e) => out.push(Check::warn(
                "remote",
                format!("{} unreachable: {e:#}", r.name),
            )),
        }
    }
    out
}

/// Engine binary smoke: execute the active engine's server binary with
/// `--version`. The manifest can say "installed" while the binary is
/// corrupted or linked against a glibc the box no longer has — this is
/// the row that catches it. Read-only probe: `--version` prints and
/// exits without touching the GPU.
fn engine_smoke_check(server_path: &str) -> Check {
    match std::process::Command::new(server_path)
        .arg("--version")
        .output()
    {
        Ok(out) if out.status.success() => {
            let text = format!(
                "{}\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr),
            );
            let line = text
                .lines()
                .find(|l| !l.trim().is_empty())
                .map_or_else(|| "?".to_string(), |l| l.trim().to_string());
            Check::ok("engine binary", format!("executes ({line})"))
        }
        Ok(out) => Check::warn(
            "engine binary",
            format!(
                "exited {status} — reinstall: pallama engine update",
                status = out.status
            ),
        ),
        Err(e) => Check::warn(
            "engine binary",
            format!("cannot execute {server_path}: {e} — reinstall: pallama engine update"),
        ),
    }
}

/// Exposure: join the bind host with auth state — the security fact the
/// config and keys rows separately can't say. Authless loopback is the
/// default and fine; anything off-loopback without keys is an open
/// inference endpoint for the whole network.
fn doctor_exposure(d: &PallamaDirs) -> Vec<Check> {
    let Ok(cfg) = Config::load(d).map_err(|e| anyhow!("{e}")) else {
        return Vec::new(); // config row already failed loudly
    };
    let mut checks = vec![exposure_verdict(&cfg.host, cfg.port, cfg.keys.len())];
    // mistral.rs children accept-and-ignore credentials: there is no
    // child-auth lane at all. Pallama forces their bind to loopback and
    // the gateway stays the only authenticated surface — say so loudly
    // while one is active, so the security model is never silent.
    if let Ok(store) = Store::open(d) {
        if let Ok(Some(row)) = store.active_engine() {
            if row.kind == EngineKind::MistralRs {
                checks.push(Check::warn(
                    "exposure",
                    "mistral.rs engine active: the child has NO authentication and ignores \
                     bearer keys; pallama binds it to 127.0.0.1 and the gateway remains the \
                     only authenticated surface",
                ));
            }
        }
    }
    checks
}

/// Pure verdict over (host, port, `n_keys`): loopback needs no auth;
/// wildcard/LAN without keys is exposed to the network. Non-loopback
/// rows also note pallama speaks plain HTTP — TLS belongs in a reverse
/// proxy in front.
fn exposure_verdict(host: &str, port: u16, n_keys: usize) -> Check {
    let h = host.trim();
    if matches!(h, "localhost" | "127.0.0.1" | "::1") {
        return if n_keys == 0 {
            Check::ok("exposure", "loopback bind, no auth needed".to_string())
        } else {
            Check::ok(
                "exposure",
                format!("loopback bind + {n_keys} key(s) (belt and braces)"),
            )
        };
    }
    let binding = if h.is_empty() || h == "0.0.0.0" || h == "::" {
        "all interfaces"
    } else {
        h
    };
    if n_keys == 0 {
        Check::warn(
            "exposure",
            format!(
                "{binding}:{port} reachable off-box with NO auth — anyone on the network can \
                 use this box; set host=127.0.0.1 or add keys (`pallama keys add`); pallama \
                 is HTTP-only, put TLS in front (reverse proxy) if you stay exposed"
            ),
        )
    } else {
        Check::ok(
            "exposure",
            format!(
                "{binding}:{port} exposed with {n_keys} key(s) enforcing auth (HTTP — front \
                 it with a TLS proxy for secrets in transit)"
            ),
        )
    }
}

/// Store health: sqlite `quick_check` on pallama.db. Absent = fresh
/// install (ok row); unreadable or failing = warn with the regenerate
/// hint — usage history is the only casualty.
fn doctor_store(d: &PallamaDirs) -> Vec<Check> {
    let db = d.db_file();
    if !db.is_file() {
        return vec![Check::ok(
            "store",
            "no pallama.db yet (fresh install)".to_string(),
        )];
    }
    let size_kib = std::fs::metadata(&db).map_or(0, |m| m.len() / 1024);
    match Store::open(d) {
        Ok(store) => {
            let verdict = store
                .conn()
                .query_row("PRAGMA quick_check", [], |r| r.get::<_, String>(0))
                .unwrap_or_else(|e| format!("quick_check query failed: {e}"));
            if verdict == "ok" {
                vec![Check::ok(
                    "store",
                    format!("pallama.db ok ({size_kib} KiB, quick_check passed)"),
                )]
            } else {
                vec![Check::warn(
                    "store",
                    format!("quick_check: {verdict} — back up, then delete {} to regenerate (usage history lost)", db.display()),
                )]
            }
        }
        Err(e) => vec![Check::warn(
            "store",
            format!(
                "pallama.db unreadable ({e:#}) — back up, then delete {} to regenerate (usage history lost)",
                db.display()
            ),
        )],
    }
}

/// Service-manager state: the systemd/launchd unit the installer lanes
/// create. `PALLAMA_SYSTEMCTL` overrides the systemctl binary (test
/// seam, mirrors install.sh). No manager found (windows/dev) → no row;
/// no unit is an informational ok, not a warning — running manually is
/// a legitimate lane.
fn doctor_service() -> Vec<Check> {
    let systemctl = std::env::var("PALLAMA_SYSTEMCTL").unwrap_or_else(|_| "systemctl".into());
    if std::process::Command::new(&systemctl)
        .arg("--version")
        .output()
        .is_ok()
    {
        let enabled = std::process::Command::new(&systemctl)
            .args(["is-enabled", "pallama"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "enabled");
        let active = std::process::Command::new(&systemctl)
            .args(["is-active", "pallama"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "active");
        return service_verdict(enabled, active, "systemd")
            .into_iter()
            .collect();
    }
    #[cfg(target_os = "macos")]
    {
        let uid = std::process::Command::new("id")
            .arg("-u")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
        let Some(uid) = uid else {
            return Vec::new();
        };
        // Installer label: LaunchAgent (user) or LaunchDaemon (root).
        let label = if uid == "0" {
            format!("system/dev.pallama")
        } else {
            format!("gui/{uid}/dev.pallama")
        };
        let loaded = std::process::Command::new("launchctl")
            .args(["print", &label])
            .output()
            .ok()
            .map(|o| o.status.success());
        return service_verdict(Some(true), loaded, "launchd")
            .into_iter()
            .collect();
    }
    #[cfg(not(target_os = "macos"))]
    Vec::new()
}

/// Pure verdict over manager probe results: both set = managed, both
/// unset-but-probed = manual lane, `None` = probe inconclusive → no row.
fn service_verdict(enabled: Option<bool>, active: Option<bool>, manager: &str) -> Option<Check> {
    match (enabled, active) {
        (Some(true), Some(true)) => Some(Check::ok(
            "service",
            format!("{manager} unit active + enabled (starts on boot, restarts on crash)"),
        )),
        (Some(true), Some(false)) => Some(Check::warn(
            "service",
            format!("{manager} unit enabled but not running — logs: journalctl -u pallama"),
        )),
        (Some(false), Some(true)) => Some(Check::warn(
            "service",
            "daemon running but unit not enabled — may not survive reboot; run: sh \
             scripts/install.sh (or systemctl enable pallama)",
        )),
        (Some(false), Some(false)) => Some(Check::ok(
            "service",
            format!("no {manager} unit (manual/dev lane — start with `pallama serve`)"),
        )),
        _ => None,
    }
}

// Probe aggregator: store row + manifest + live --version + GPU pick in
// one diagnostic sweep; splitting scatters one logical verdict (same
// precedent as the other long gateway handlers).
#[allow(clippy::too_many_lines)]
async fn doctor_engine(d: &PallamaDirs) -> Vec<Check> {
    let mut checks = Vec::new();
    let mut active_tag: Option<String> = None;
    // Install-time (manifest) GPU names — what auto-pick derives from.
    let mut frozen_gpu_names: Vec<String> = Vec::new();
    match local_engine_manager(d) {
        Ok(mgr) => match mgr.active_manifest() {
            Ok(Some(m)) => {
                active_tag = Some(m.tag.clone());
                frozen_gpu_names = m.devices.iter().map(|dev| dev.name.clone()).collect();
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
                checks.push(engine_smoke_check(&m.server_path));
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
    // The engines row carries the kind (the manifest does not — single
    // source of truth lives in the store).
    let active_kind = pallama_core::Store::open(d)
        .ok()
        .and_then(|s| s.active_engine().ok().flatten())
        .map(|r| r.kind);
    // CUDA opportunity + toolchain readiness (NVIDIA boxes only): a
    // Vulkan prebuilt on an NVIDIA GPU leaves measured decode/prefill on
    // the table. mistral.rs-active boxes skip — that lane picks its
    // asset by driver at install time.
    let (driver_cuda, compute_cap) = nvidia_gpu_facts().await;
    checks.extend(cuda_opportunity_rows(
        active_tag.as_deref(),
        active_kind,
        driver_cuda,
        compute_cap,
        &detect_toolchain(&path_dirs()),
    ));
    // Engine currency: prefer reconciling the daemon's last upstream
    // survey against the CURRENT active engine — the marker's own
    // verdict went stale the moment `engine update`/`use`/`rollback`
    // flipped the store between daily ticks. Marker missing or >48h
    // old (daemon never surveyed / long offline): probe upstream live,
    // same 4s-capped warn-only shape as the whisper/app currency rows.
    // The daily survey tracks llama.cpp only — a mistral.rs-active box
    // gets its own live mistral.rs check instead.
    if active_kind == Some(EngineKind::MistralRs) {
        if let Some(active) = active_tag.as_deref() {
            checks.push(live_mistralrs_currency(active).await);
        }
        return checks;
    }
    let mut currency: Option<Check> = None;
    let mut marker_usable = false;
    // A marker is only trustworthy for the CURRENT channel: one written by a
    // pre-channel daemon (no "channel" key) or under a different channel
    // reports the other channel's target — reconciling it after a switch
    // nags about builds the user deliberately left behind.
    let cfg_channel = pallama_core::Config::load(d)
        .map(|c| c.update_channel)
        .unwrap_or_default();
    // Lane-aware update hint: the active engine's asset decides which
    // command can actually refresh it (source builds vs prebuilt lanes).
    let active_asset = active_engine_asset(d, active_tag.as_deref());
    if let Ok(raw) = std::fs::read_to_string(d.run_dir().join("engine-check.json")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |n| n.as_secs());
            marker_usable = now.saturating_sub(v["checked_at"].as_u64().unwrap_or(0)) <= 2 * 86_400
                && marker_channel_matches(&v, cfg_channel);
            if marker_usable {
                currency = currency_verdict(active_tag.as_deref(), &v, now, &active_asset);
                // Enumeration drift: install-time probe names the serving
                // child can no longer see (the daemon's spawn context is
                // the authority — spawns auto-remap, but the user should
                // know the probe view is stale).
                if let Some(live) = census_names(&v) {
                    let drift = device_drift(&frozen_gpu_names, &live);
                    if !drift.is_empty() {
                        checks.push(Check::warn(
                            "engine devices",
                            format!(
                                "enumeration drift: {} known to the install-time probe but \
                                 invisible to the serving child — spawns auto-remap, \
                                 `pallama engine update` refreshes the probe",
                                drift.join(", ")
                            ),
                        ));
                    }
                }
            }
        }
    }
    if currency.is_none() && !marker_usable {
        if let Some(active) = active_tag.as_deref() {
            if active == pallama_runtime::LOCAL_TAG {
                currency = Some(Check::ok(
                    "engine currency",
                    "local build — upstream currency not tracked",
                ));
            } else {
                currency = Some(live_engine_currency(active, &active_asset).await);
            }
        }
    }
    if let Some(c) = currency {
        checks.push(c);
    }
    // CUDA-channel hint: an NVIDIA box serving the Vulkan asset is
    // leaving the measured ~4% Vulkan delta on the table. The channel
    // is zero-touch (default repo, probed automatically at every
    // `engine update`) — this row only tells the user the lane exists
    // and where its assets come from.
    if std::env::consts::OS == "linux"
        && std::env::consts::ARCH == "x86_64"
        && pallama_runtime::engine::system_vendor_hint()
            == pallama_runtime::engine::manifest::Vendor::Nvidia
        && !active_asset_is_cuda(d, active_tag.as_deref())
    {
        let repo = pallama_runtime::engine::gh::engine_overlay_repo();
        checks.push(Check::ok(
            "engine cuda channel",
            format!(
                "NVIDIA GPU on the Vulkan asset — prebuilt CUDA engines are preferred \
                 automatically once {repo} publishes bNNNN-cuda releases (engine-cuda \
                 workflow); set PALLAMA_ENGINE_REPO to use a fork"
            ),
        ));
    }
    checks
}

/// Active engine's asset string ("" when unknown: no engine, store error,
/// or tag mismatch). Feeds the lane-aware update hint.
fn active_engine_asset(d: &PallamaDirs, active_tag: Option<&str>) -> String {
    let Some(tag) = active_tag else {
        return String::new();
    };
    Store::open(d)
        .ok()
        .and_then(|s| s.active_engine().ok().flatten())
        .filter(|row| row.tag == tag)
        .map_or_else(String::new, |row| row.asset)
}

/// True when the ACTIVE llama.cpp engine already serves a CUDA asset
/// (overlay prebuilt or source-built) — the cuda-channel hint only
/// applies to Vulkan/CPU assets. Probe failures read as "not cuda":
/// the hint is advisory, a store error must not hide it.
fn active_asset_is_cuda(d: &PallamaDirs, active_tag: Option<&str>) -> bool {
    active_engine_asset(d, active_tag).contains("cuda")
}

/// Frozen (install-time) names missing from the serving child's census.
/// Superset child views and CPU-only empties are healthy — only names the
/// child CANNOT see get flagged.
fn device_drift(frozen: &[String], live: &[String]) -> Vec<String> {
    frozen
        .iter()
        .filter(|name| !live.iter().any(|l| l == *name))
        .cloned()
        .collect()
}

/// Device names from the engine-check marker's census, `None` when the
/// census is unknown (key absent or null: pre-census daemon or failed
/// probe). Unknown must NOT be reported as drift — only a census that
/// actually ran can say a device is invisible.
fn census_names(marker: &serde_json::Value) -> Option<Vec<String>> {
    marker["devices"].as_array().map(|a| {
        a.iter()
            .filter_map(|dev| dev["name"].as_str().map(str::to_string))
            .collect()
    })
}

/// True when the marker was written for the channel the user is on.
/// Absent key = the pre-channel default ("latest"), matching the
/// reconcile path's own `unwrap_or("latest")` read.
fn marker_channel_matches(
    marker: &serde_json::Value,
    cfg: pallama_core::config::UpdateChannel,
) -> bool {
    marker["channel"].as_str().unwrap_or("latest") == cfg.to_string()
}

/// Direction word for update messaging: llama b-tags compare by build
/// number (upgrade/downgrade); anything else stays neutral ("update").
/// Semver tuple for `vX.Y.Z` tags (mistral.rs lane): numeric compare so
/// "v0.10.0" is an upgrade over "v0.9.3" instead of lexicographic noise.
use pallama_runtime::engine::gh::vtag_semver;

fn channel_word(active: &str, target: &str) -> &'static str {
    match (btag_number(active), btag_number(target)) {
        (Some(a), Some(t)) if t > a => "upgrade",
        (Some(a), Some(t)) if t < a => "downgrade",
        _ => match (vtag_semver(active), vtag_semver(target)) {
            (Some(a), Some(t)) if t > a => "upgrade",
            (Some(a), Some(t)) if t < a => "downgrade",
            _ => "update",
        },
    }
}

/// The command that updates the ACTIVE engine's lane: source-built
/// engines (`asset` = `built-*`) refresh by rebuilding the source lane —
/// `engine update` would install the Vulkan prebuilt instead. Overlay
/// prebuilts (`bNNNN-cuda` from the CUDA channel, `asset` =
/// `ubuntu-cuda-*`) update through the normal `engine update` lane,
/// which re-probes the overlay for the channel's build number.
fn engine_update_command(active_tag: &str, asset: &str) -> &'static str {
    if active_tag.ends_with("-cuda") {
        if asset.starts_with("built-") {
            "pallama engine build cuda"
        } else {
            "pallama engine update"
        }
    } else if active_tag.ends_with("-cpu") {
        "pallama engine build cpu"
    } else {
        "pallama engine update"
    }
}

/// Live engine-currency probe (marker missing or >48h stale): 4s cap,
/// warn-only — the update itself stays a human action.
async fn live_engine_currency(active: &str, asset: &str) -> Check {
    let token = std::env::var("GH_TOKEN")
        .or_else(|_| std::env::var("GITHUB_TOKEN"))
        .ok();
    let Ok(gh) = GhClient::new(token) else {
        return Check::warn("engine currency", "cannot build GitHub client");
    };
    let channel = config().map(|c| c.update_channel).unwrap_or_default();
    let fetched = tokio::time::timeout(
        std::time::Duration::from_secs(4),
        gh.channel_b_release(channel),
    )
    .await;
    match fetched {
        Ok(Ok(rel)) => {
            // b-tags compare by build number: lane suffixes (-cuda/-cpu)
            // are the SAME build — string equality would nag a current
            // source-built engine forever.
            if same_build(active, &rel.tag_name) {
                Check::ok(
                    "engine currency",
                    format!("up to date ({active}, channel: {channel})"),
                )
            } else {
                Check::warn(
                    "engine currency",
                    format!(
                        "{} available: {} (active: {}, channel: {}) — run: {}",
                        channel_word(active, &rel.tag_name),
                        rel.tag_name,
                        active,
                        channel,
                        engine_update_command(active, asset)
                    ),
                )
            }
        }
        Ok(Err(e)) => Check::warn(
            "engine currency",
            format!("check failed ({e:#}) — offline? set GH_TOKEN if rate limited"),
        ),
        Err(_) => Check::warn("engine currency", "GitHub check timed out after 4s"),
    }
}

/// mistral.rs engine currency (mistral.rs engine active): live 4s-capped
/// warn-only probe of the mistral.rs release list, mirroring the llama.cpp
/// live lane — the daemon's daily survey does not track this repo.
async fn live_mistralrs_currency(active: &str) -> Check {
    let token = std::env::var("GH_TOKEN")
        .or_else(|_| std::env::var("GITHUB_TOKEN"))
        .ok();
    let Ok(gh) = GhClient::new(token) else {
        return Check::warn("engine currency", "cannot build GitHub client");
    };
    let fetched = tokio::time::timeout(
        std::time::Duration::from_secs(4),
        gh.latest_mistralrs_release(),
    )
    .await;
    match fetched {
        Ok(Ok(rel)) => {
            if rel.tag_name == active {
                Check::ok(
                    "engine currency",
                    format!("up to date ({active}, mistral.rs)"),
                )
            } else {
                Check::warn(
                    "engine currency",
                    format!(
                        "{} available: {} (active: {}, mistral.rs) — run: pallama engine install",
                        channel_word(active, &rel.tag_name),
                        rel.tag_name,
                        active
                    ),
                )
            }
        }
        Ok(Err(e)) => Check::warn(
            "engine currency",
            format!("check failed ({e:#}) — offline? set GH_TOKEN if rate limited"),
        ),
        Err(_) => Check::warn("engine currency", "GitHub check timed out after 4s"),
    }
}

/// CUDA-opportunity + toolchain-readiness rows for doctor. Pure: GPU
/// facts and toolchain are passed in (the nvidia-smi probe is the
/// caller's job), so the decision matrix is unit-testable.
fn cuda_opportunity_rows(
    active_tag: Option<&str>,
    active_kind: Option<EngineKind>,
    driver_cuda: Option<(u32, u32)>,
    compute_cap: Option<(u32, u32)>,
    tc: &Toolchain,
) -> Vec<Check> {
    let Some((dmaj, dmin)) = driver_cuda else {
        return Vec::new(); // no NVIDIA driver: nothing to suggest
    };
    let mut rows = Vec::new();
    // Backend nudge: llamacpp engine that is not the CUDA build (a
    // `-cuda` tag already IS the fast lane; mistral.rs picked its
    // asset by driver at install time).
    if active_kind != Some(EngineKind::MistralRs)
        && active_tag.is_some_and(|t| !t.ends_with("-cuda"))
    {
        let cc = compute_cap.map_or_else(String::new, |(a, b)| format!(", sm {a}{b}"));
        rows.push(Check::ok(
            "engine backend",
            format!(
                "NVIDIA GPU (driver CUDA {dmaj}.{dmin}{cc}) — a CUDA engine is faster: \
                 `pallama engine build cuda` (compiles upstream) or \
                 `pallama engine install` (mistral.rs prebuilt)"
            ),
        ));
    }
    // Toolchain readiness for the source-build lane.
    let missing: Vec<&str> = [
        ("git", tc.git.is_none()),
        ("cmake", tc.cmake.is_none()),
        ("nvcc", tc.nvcc.is_none()),
    ]
    .iter()
    .filter(|(_, m)| *m)
    .map(|(n, _)| *n)
    .collect();
    if missing.is_empty() {
        rows.push(Check::ok(
            "cuda toolchain",
            "ready (git, cmake, nvcc) — `pallama engine build cuda` can run now",
        ));
    } else {
        rows.push(Check::warn(
            "cuda toolchain",
            format!(
                "missing {} — install the distro git/cmake packages + the CUDA toolkit \
                 (https://developer.nvidia.com/cuda-downloads); prebuilt alternative: \
                 `pallama engine install`",
                missing.join(", ")
            ),
        ));
    }
    rows
}

/// Doctor engine-currency verdict. The marker's `latest`/`checked_at` are
/// durable (the last upstream survey); its `active`/`update_available` are
/// write-time derivations that go stale the instant an engine switch lands
/// between daily daemon ticks — so the verdict is derived here against the
/// live active tag, never read from the marker.
fn currency_verdict(
    active_tag: Option<&str>,
    marker: &serde_json::Value,
    now_secs: u64,
    asset: &str,
) -> Option<Check> {
    let active = active_tag?; // no engine installed: the engine row already FAILs
    let latest = marker["latest"].as_str()?;
    if active == pallama_runtime::LOCAL_TAG {
        return Some(Check::ok(
            "engine currency",
            "local build — upstream currency not tracked",
        ));
    }
    let checked_at = marker["checked_at"].as_u64().unwrap_or(0);
    let stale = now_secs.saturating_sub(checked_at) > 2 * 86_400;
    // The marker's `channel` is what `latest` was resolved against at
    // write time (old markers predate the knob: default latest).
    let channel = marker["channel"].as_str().unwrap_or("latest");
    if !same_build(active, latest) {
        Some(Check::warn(
            "engine currency",
            format!(
                "{} available: {latest} (active: {active}, channel: {channel}) — run: {}",
                channel_word(active, latest),
                engine_update_command(active, asset)
            ),
        ))
    } else if stale {
        Some(Check::warn(
            "engine currency",
            "no successful upstream check in >48h (offline? engine_check_secs=0?)",
        ))
    } else {
        Some(Check::ok(
            "engine currency",
            format!("up to date ({active})"),
        ))
    }
}

/// App currency: the newest pallama release on the same repo/URL the
/// `pallama upgrade` lane installs from. Warn-only — never installs
/// (`pallama upgrade` stays a human action, exactly like engine updates).
/// `PALLAMA_REPO` unset → informational row (self-upgrade lane not
/// configured; dev checkouts and manual installs live here).
async fn doctor_app_currency() -> Vec<Check> {
    let Some(repo) = std::env::var("PALLAMA_REPO").ok().filter(|r| !r.is_empty()) else {
        return vec![Check::ok(
            "app currency",
            "PALLAMA_REPO unset — self-upgrade lane not configured",
        )];
    };
    let base = std::env::var("PALLAMA_INSTALL_BASE_URL")
        .unwrap_or_else(|_| "https://api.github.com".to_string());
    let token = std::env::var("GH_TOKEN")
        .or_else(|_| std::env::var("GITHUB_TOKEN"))
        .ok();
    let Ok(gh) = pallama_runtime::GhClient::with_base(&base, token) else {
        return vec![Check::warn("app currency", "cannot build GitHub client")];
    };
    let channel = config().map(|c| c.update_channel).unwrap_or_default();
    let fetched = tokio::time::timeout(
        std::time::Duration::from_secs(4),
        gh.channel_repo_release(&repo, channel),
    )
    .await;
    match fetched {
        Ok(Ok(rel)) => vec![version_currency_verdict(
            "app currency",
            env!("CARGO_PKG_VERSION"),
            &rel.tag_name,
            "pallama upgrade",
        )],
        Ok(Err(e)) => vec![Check::warn(
            "app currency",
            format!("check failed ({e:#}) — offline? set GH_TOKEN if rate limited"),
        )],
        Err(_) => vec![Check::warn(
            "app currency",
            "GitHub check timed out after 4s",
        )],
    }
}

/// Whisper.cpp currency: installs are versioned by tag dir under
/// `data/whisper/bin/<tag>/`; compare the newest against the latest
/// upstream release (`WHISPER_REPO`). Optional component — not installed
/// means no row (same policy as remotes). Warn-only; `pallama whisper
/// --install` stays a human action.
async fn doctor_whisper_currency(d: &PallamaDirs) -> Vec<Check> {
    let Some((_, tag_dir)) = pallama_runtime::whisper::server_bin(d) else {
        // Optional lane absent: say so instead of silently omitting the
        // row — doctor is where users discover the lane exists (observed
        // live: all-green doctor while transcription had no backend).
        return vec![Check::warn(
            "whisper lane",
            "not installed — optional: `pallama whisper --install` enables \
             local /v1/audio/transcriptions",
        )];
    };
    let installed = tag_dir
        .file_name()
        .map_or_else(|| "?".into(), |t| t.to_string_lossy().into_owned());
    let token = std::env::var("GH_TOKEN")
        .or_else(|_| std::env::var("GITHUB_TOKEN"))
        .ok();
    let Ok(gh) = pallama_runtime::GhClient::new(token) else {
        return vec![Check::warn(
            "whisper currency",
            "cannot build GitHub client",
        )];
    };
    let channel = config().map(|c| c.update_channel).unwrap_or_default();
    // Asset-aware: whisper.cpp tags assetless v-releases (v1.9.4) while
    // the b-tags carry the binaries — currency must never advertise a
    // tag the install lane cannot install.
    let fetched = tokio::time::timeout(
        std::time::Duration::from_secs(4),
        pallama_runtime::whisper::channel_target(&gh, channel),
    )
    .await;
    match fetched {
        Ok(Ok(latest)) => vec![whisper_pin_verdict(d, &installed, &latest)],
        Ok(Err(e)) => vec![Check::warn(
            "whisper currency",
            format!("check failed ({e:#}) — offline? set GH_TOKEN if rate limited"),
        )],
        Err(_) => vec![Check::warn(
            "whisper currency",
            "GitHub check timed out after 4s",
        )],
    }
}

/// Pinned-whisper currency verdict (pure): deliberate pin named as such —
/// plain --install silently unpins, so updates point at the
/// pin-preserving --tag form.
fn whisper_pin_verdict(d: &PallamaDirs, installed: &str, latest: &str) -> Check {
    match pallama_runtime::whisper::pinned_tag(d) {
        Some(pin) if pin == latest => {
            Check::ok("whisper currency", format!("pinned to {pin} (up to date)"))
        }
        Some(pin) => {
            let ahead =
                matches!((ver_triple(&pin), ver_triple(latest)), (Some(a), Some(b)) if a > b);
            if ahead {
                Check::ok(
                    "whisper currency",
                    format!("pinned to {pin} (ahead of latest release {latest})"),
                )
            } else {
                Check::warn(
                    "whisper currency",
                    format!(
                        "update available: {latest} (pinned: {pin}) — run: \
                         pallama whisper --install --tag {latest}"
                    ),
                )
            }
        }
        None => version_currency_verdict(
            "whisper currency",
            installed,
            latest,
            "pallama whisper --install",
        ),
    }
}

/// Local whisper models health: the server binary alone transcribes
/// nothing — without a pulled ggml model every request 501s (observed
/// live in validation: installed server + zero models = broken lane).
fn doctor_whisper_models(d: &PallamaDirs) -> Vec<Check> {
    if pallama_runtime::whisper::server_bin(d).is_none() {
        return Vec::new(); // covered by the "whisper lane" row above
    }
    let pulled = pallama_runtime::whisper::list_models(d);
    if pulled.is_empty() {
        vec![Check::warn(
            "whisper models",
            "none pulled — transcription will 501 until `pallama whisper --pull base`",
        )]
    } else {
        vec![Check::ok(
            "whisper models",
            format!("pulled: {}", pulled.join(", ")),
        )]
    }
}

/// Generic currency verdict (pure): semver-compare when both tags parse;
/// otherwise fall back to tag inequality (whisper.cpp alternates `vX.Y.Z`
/// and `bNNNN` tag shapes — a differing latest tag IS an update). Equal
/// tags are always up to date; the verdict is never silently skipped.
fn version_currency_verdict(
    name: &'static str,
    running: &str,
    latest: &str,
    action: &str,
) -> Check {
    match (ver_triple(running), ver_triple(latest)) {
        (Some(r), Some(l)) if l > r => warn_update(name, running, latest, action),
        (Some(r), Some(l)) if r > l => Check::ok(
            name,
            format!("running ahead of latest release ({running} > {latest}; dev build?)"),
        ),
        (Some(_), Some(_)) => Check::ok(name, format!("up to date ({running})")),
        _ if running == latest => Check::ok(name, format!("up to date ({running})")),
        _ => warn_update(name, running, latest, action),
    }
}

fn warn_update(name: &'static str, running: &str, latest: &str, action: &str) -> Check {
    Check::warn(
        name,
        format!("update available: {latest} (running: {running}) — run: {action}"),
    )
}

/// `vX.Y.Z` / `X.Y.Z` → `(X, Y, Z)`. Strict: exactly three numeric parts —
/// anything else (suffixes, partial tags) is None so callers warn.
fn ver_triple(v: &str) -> Option<(u64, u64, u64)> {
    let v = v.trim().strip_prefix('v').unwrap_or(v.trim());
    let mut it = v.split('.');
    let triple = (
        it.next()?.parse().ok()?,
        it.next()?.parse().ok()?,
        it.next()?.parse().ok()?,
    );
    it.next().is_none().then_some(triple)
}

/// Multiple pallama binaries on one box: your shell resolves one order,
/// system services (gateways, systemd units, cron) resolve another — and
/// the stale copy silently wins the daemon race (live incident class).
/// Found by scanning PATH plus the usual suspects, versions compared by
/// executing each candidate.
fn doctor_binary_shadow() -> Vec<Check> {
    #[cfg(unix)]
    const SUSPECTS: &[&str] = &["/usr/local/bin", "/usr/bin", "/snap/bin"];
    #[cfg(not(unix))]
    const SUSPECTS: &[&str] = &[];
    let mut dirs: Vec<std::path::PathBuf> = SUSPECTS.iter().map(std::path::PathBuf::from).collect();
    if let Some(path) = std::env::var_os("PATH") {
        dirs.extend(std::env::split_paths(&path));
    }
    if let Some(home) = std::env::var_os("HOME") {
        let mut h = std::path::PathBuf::from(home);
        h.push(".local/bin");
        dirs.push(h);
    }
    let mut seen: Vec<std::path::PathBuf> = Vec::new(); // grows via push below
    let mut found: Vec<(std::path::PathBuf, String)> = Vec::new();
    for dir in dirs {
        let cand = dir.join("pallama");
        if cand.is_file() && !seen.contains(&cand) {
            let ver = std::process::Command::new(&cand)
                .arg("--version")
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map_or_else(|| "?".to_string(), |s| s.trim().to_string());
            seen.push(cand.clone());
            found.push((cand, ver));
        }
    }
    if found.len() <= 1 {
        return vec![Check::ok("binary", "single pallama on PATH".to_string())];
    }
    let current = std::env::current_exe()
        .ok()
        .and_then(|p| p.canonicalize().ok());
    let mut detail = String::new();
    let mut stale = false;
    for (path, ver) in &found {
        let marker = current
            .as_ref()
            .and_then(|c| path.canonicalize().ok().map(|p| p == *c))
            .unwrap_or(false);
        let _ = write!(
            detail,
            "\n  {}{} ({ver})",
            path.display(),
            if marker { "  <- this binary" } else { "" }
        );
        if path.starts_with("/usr/local/bin") && !marker {
            stale = true;
        }
    }
    if stale {
        vec![Check::warn(
            "binary",
            format!(
                "multiple pallama binaries; system services with their own PATH resurrect the stale copy:{detail}\n  fix: sudo rm the stale path (or reinstall with --system)"
            ),
        )]
    } else {
        vec![Check::warn(
            "binary",
            format!("multiple pallama binaries on PATH:{detail}"),
        )]
    }
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
    let is_pallama = cli_http()
        .get(format!("http://{}:{}/healthz", cfg.host, cfg.port))
        .send()
        .await
        .is_ok_and(|r| r.status().is_success());
    if is_pallama {
        vec![Check::ok(
            "port",
            format!(
                "{}:{} — pallama daemon already answering",
                cfg.host, cfg.port
            ),
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
            format!(
                "{g:.1} GiB free under {} — model pulls need headroom",
                d.data_dir.display()
            ),
        )]
    } else {
        vec![Check::ok(
            "disk",
            format!("{g:.1} GiB free under {}", d.data_dir.display()),
        )]
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
    let mut flagged = 0_usize;
    for line in raw.lines().filter(|l| !l.trim().is_empty()) {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if v["ts"].as_u64().unwrap_or(0) < day_ago {
            continue;
        }
        records += 1;
        if let Some(dets) = v["detections"].as_array() {
            if !dets.is_empty() {
                flagged += 1;
            }
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
            "{flagged} of {records} requests in 24h flagged: {} — `pallama why` for details + retry hints",
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
        vec![Check::ok(
            "models",
            format!("{} pulled, all parse", models.len()),
        )]
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

#[allow(clippy::too_many_lines)] // one cohesive startup: hook wiring, flushers, listener
async fn serve() -> Result<()> {
    banner();
    let d = dirs();
    d.ensure().ok();
    let cfg = config()?;
    let _lock = pallama_runtime::DaemonLock::acquire(&d).map_err(|e| anyhow!("{e}"))?;
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
    let engine_row = store
        .active_engine()?
        .ok_or_else(|| anyhow!("no engine installed; run: pallama engine update"))?;
    let manifest: pallama_runtime::Manifest = serde_json::from_str(&engine_row.manifest)
        .with_context(|| format!("decode engine manifest {}", engine_row.tag))?;
    println!(
        "engine: {} (build {})",
        engine_row.tag, manifest.build_number
    );
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
    // Engine fork: the active row's kind picks the adapter. Restart is
    // required to switch kinds (the supervisor holds the engine for its
    // lifetime) — the restart hint covers that.
    let engine: Arc<dyn Engine> = match engine_row.kind {
        pallama_core::engine_kind::EngineKind::MistralRs => {
            Arc::new(MistralRsEngine::with_staging(
                manifest,
                engine_env,
                Some(d.run_dir().join("mistralrs-staging")),
            ))
        }
        pallama_core::engine_kind::EngineKind::LlamaCpp => {
            Arc::new(LlamaCppEngine::with_env(manifest, engine_env))
        }
    };
    let bus = EventBus::default();
    let sup = Arc::new(Supervisor::new(
        d.clone(),
        cfg.clone(),
        bus.clone(),
        hw,
        engine.clone(),
    ));
    for orphan in sup.sweep_orphans() {
        println!("swept orphan engine: {orphan}");
    }
    let _reaper = sup.spawn_reaper();

    if cfg.engine_check_secs > 0 {
        spawn_engine_check_task(&d, cfg.engine_check_secs, engine.clone());
    }
    let state = Arc::new(pallama_gateway::state::AppState::new(
        d.clone(),
        cfg.clone(),
        sup.clone(),
        bus,
    ));
    let host = cfg.host.clone();

    // Which binary owns this daemon — one-command visibility into the
    // stale-copy race (gateways auto-start pallama from their own PATH).
    if let Ok(exe) = std::env::current_exe() {
        let _ = std::fs::write(d.run_dir().join("daemon.path"), exe.display().to_string());
    }
    let port = cfg.port;
    // J4: SIGHUP -> live keys registry re-reads config.toml (children
    // stay up; budgets/scopes apply on the next request).
    {
        let state = Arc::clone(&state);
        let dirs = d.clone();
        let hook: std::sync::Arc<dyn Fn() + Send + Sync> = std::sync::Arc::new(move || {
            let raw = std::fs::read_to_string(dirs.config_file()).unwrap_or_default();
            if let Ok(fresh) = pallama_core::Config::from_toml(&raw) {
                for k in &fresh.keys {
                    // Upsert by name (scopes/budgets refresh); removals
                    // handled below.
                    state.keys.upsert(k.clone());
                }
                let live: std::collections::HashSet<String> =
                    fresh.keys.iter().map(|k| k.name.clone()).collect();
                for e in state.keys.entries() {
                    if !live.contains(&e.name) {
                        state.keys.remove(&e.name);
                    }
                }
            }
        });
        *pallama_runtime::daemon::SIGHUP_HOOK.lock().expect("hook") = Some(hook);
    }
    // Keys usage write-behind: dirty counters flush every 60s and once
    // more at shutdown (daily budgets survive restarts). Each flush
    // opens its own store connection (rusqlite is !Sync; WAL handles
    // the concurrency).
    let flusher = {
        let state = Arc::clone(&state);
        let flush_dirs = d.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(std::time::Duration::from_mins(1));
            loop {
                tick.tick().await;
                let dirs = flush_dirs.clone();
                let keys = Arc::clone(&state.keys);
                tokio::task::spawn_blocking(move || keys.flush(&dirs))
                    .await
                    .ok();
            }
        })
    };
    let served = pallama_gateway::serve(
        state.clone(),
        &host,
        port,
        Box::pin(pallama_runtime::wait_for_shutdown_signal()),
    )
    .await;
    flusher.abort();
    state.keys.flush(&d);
    // A hard bind conflict (ollama owns the port) must not feed the
    // systemd Restart=always loop: exit 3 + RestartPreventExitStatus.
    if let Err(e) = &served {
        if e.downcast_ref::<pallama_gateway::BindConflict>().is_some() {
            eprintln!("pallama: {e} — fix config.toml `port` (or PALLAMA_PORT) and start again");
            std::process::exit(pallama_gateway::EXIT_BIND_CONFLICT);
        }
    }
    served?;
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
    #[cfg(windows)]
    // F127: `taskkill` (no /F) asks for a close first; a console daemon
    // without a message pump ignores it, so finish with a hard kill after
    // a grace beat. Never a group — exact pid only, like the unix lane.
    {
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string()])
            .status();
        std::thread::sleep(std::time::Duration::from_millis(500));
        let still_alive = pallama_runtime::process_alive_by_pid(u32::try_from(pid).unwrap_or(0));
        if still_alive {
            std::process::Command::new("taskkill")
                .args(["/F", "/PID", &pid.to_string()])
                .status()
                .context("taskkill /F")?;
        }
    }
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
    let cfg = config()?;
    let client = pallama_runtime::hf::HfClient::new(token)?
        .with_download_connections(cfg.download_connections);
    let bus = EventBus::default();
    let mut events = bus.subscribe();
    let puller = pallama_runtime::Puller {
        dirs: d.clone(),
        client,
        bus,
    };
    // Foreground by contract: the pull runs in THIS terminal and dies with
    // it. On interrupt the future is dropped, which releases the pull
    // lock and keeps the `.part` for a later resume.
    let outcome = tokio::select! {
        r = puller.route_pull(target) => r?,
        () = pallama_runtime::events::interrupted() => {
            return Err(anyhow::anyhow!(
                "pull interrupted — partial file kept; re-run `pallama pull {target}` to resume"
            ));
        }
    };
    let row = outcome.row;
    if outcome.already_present {
        println!(
            "already present — {}: {} ({}, {} shards) -> {}",
            row.name,
            humansize(row.bytes),
            row.quant,
            row.shards,
            row.path
        );
    } else {
        println!(
            "pulled {}: {} ({}, {} shards) -> {}",
            row.name,
            humansize(row.bytes),
            row.quant,
            row.shards,
            row.path
        );
    }
    // The runtime warned via log + event; surface it in the terminal too
    // (tracing is muted at the default level). The event is the REAL
    // signal — HF metadata fallbacks can still fill `arch`.
    while let Ok(ev) = events.try_recv() {
        if let pallama_runtime::PallamaEvent::ModelPulled {
            warning: Some(w), ..
        } = ev
        {
            println!("WARNING: {w}");
        }
    }
    // Structural GGUF lint (H4): warn-only, explains degraded sizing.
    if let Ok(m) = pallama_core::read_metadata_file(std::path::Path::new(&row.path)) {
        for w in m.lint() {
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

/// Install a model-side file (GGUF or mmproj) into the store: copy, or the
/// unix default of hardlink with symlink fallback (zero bytes, survives
/// either store being wiped; `protected_hardlinks` blocks linking files we
/// do not own, e.g. ollama's system blobs).
fn install_model_file(path: &std::path::Path, dest: &std::path::Path, copy: bool) -> Result<()> {
    if dest.exists() {
        return Ok(());
    }
    if copy {
        std::fs::copy(path, dest).with_context(|| format!("copy to {}", dest.display()))?;
        return Ok(());
    }
    #[cfg(unix)]
    {
        if std::fs::hard_link(path, dest).is_err() {
            std::os::unix::fs::symlink(path, dest)
                .with_context(|| format!("symlink {}", dest.display()))?;
            println!("note: used symlink (hardlink blocked); if the source store deletes the blob, re-import");
        }
    }
    Ok(())
}

fn mmproj_cmd(model: &str, path: &std::path::Path) -> Result<()> {
    let d = dirs();
    d.ensure().ok();
    if pallama_runtime::models::instance_running(&d, model) {
        return Err(anyhow!(
            "model {model} is currently running; stop it first (`pallama stop {model}` or wait for eviction)"
        ));
    }
    let store = Store::open(&d)?;
    let mut row = store
        .get_model(model)?
        .ok_or_else(|| anyhow!("no such model: {model} (see `pallama list`)"))?;
    let mmproj_meta = pallama_core::read_metadata_file(path)
        .map_err(|e| anyhow!("not a readable GGUF projector ({}): {e}", path.display()))?;
    // Vision projectors are CLIP-based GGUFs; anything else (e.g. a language
    // model file) attaches fine but hangs the engine at spawn — reject loudly.
    if mmproj_meta.architecture != "clip" {
        return Err(anyhow!(
            "not a vision projector: {} has architecture `{}`, expected `clip`; use the mmproj GGUF shipped by the model publisher",
            path.display(),
            mmproj_meta.architecture
        ));
    }
    let dest = d.models_dir().join(format!("{model}-mmproj.gguf"));
    install_model_file(path, &dest, false)?;
    row.mmproj_path = Some(dest.display().to_string());
    store.upsert_model(&row)?;
    println!("mmproj attached: {} → {model}", dest.display());
    Ok(())
}

fn import(
    path: &PathBuf,
    name: Option<String>,
    quant: Option<String>,
    copy: bool,
    mmproj: Option<&PathBuf>,
) -> Result<()> {
    let d = dirs();
    d.ensure().ok();
    let meta = pallama_core::read_metadata_file(path.as_path())
        .map_err(|e| anyhow!("not a readable GGUF ({}): {e}", path.display()))?;
    // Structural GGUF lint (H4): warn-only, explains degraded sizing.
    for w in meta.lint() {
        println!("WARNING: {w}");
    }
    let file_name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
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
    // Ollama blobs are content-addressed names without quant hints.
    let model_name = name.unwrap_or_else(|| {
        meta.name
            .clone()
            .unwrap_or_else(|| file_name.trim_end_matches(".gguf").to_string())
            .to_lowercase()
            .replace([' ', '.'], "-")
    });
    // GGUF-embedded names are untrusted input too (F98): a crafted
    // general.name with '/' would write outside the models dir.
    pallama_runtime::models::ensure_portable_name(&model_name)?;
    let dest = d.models_dir().join(format!(
        "{model_name}-{}.gguf",
        derived_quant.to_lowercase()
    ));
    install_model_file(path, &dest, copy)?;
    // Optional vision projector: validated loudly, installed alongside.
    let mmproj_dest = mmproj
        .as_ref()
        .map(|p| {
            let proj_meta = pallama_core::read_metadata_file(p.as_path())
                .map_err(|e| anyhow!("--mmproj not a readable GGUF ({}): {e}", p.display()))?;
            if proj_meta.architecture != "clip" {
                return Err(anyhow!(
                    "--mmproj not a vision projector: {} has architecture `{}`, expected `clip`",
                    p.display(),
                    proj_meta.architecture
                ));
            }
            let mdest = d.models_dir().join(format!("{model_name}-mmproj.gguf"));
            install_model_file(p.as_path(), &mdest, copy)?;
            Ok::<_, anyhow::Error>(mdest)
        })
        .transpose()?;
    let bytes = i64::try_from(size).unwrap_or(i64::MAX);
    let row = pallama_core::ModelRow {
        name: model_name.clone(),
        repo: format!("imported:{}", path.display()),
        quant: derived_quant.clone(),
        path: dest.display().to_string(),
        bytes,
        sha256: None,
        mmproj_path: mmproj_dest.map(|p| p.display().to_string()),
        shards: 1,
        arch: Some(meta.architecture.clone()),
        params: Some(pallama_runtime::hf::est_params(size, &derived_quant)),
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
        println!("no models pulled");
        return Ok(());
    }
    // Adaptive name width (capped) keeps columns aligned for long names;
    // oversize names truncate with an ellipsis instead of shifting the row.
    let name_cap = 26usize;
    let name_w = models
        .iter()
        .map(|m| m.name.len().min(name_cap))
        .max()
        .unwrap_or(0)
        .max("NAME".len());
    println!(
        "{:<name_w$}  {:<8}  {:<9}  {:<10}  {:<8}  {:>7}  PATH",
        "NAME",
        "QUANT",
        "SIZE",
        "VISION",
        "ARCH",
        "CTX",
        name_w = name_w
    );
    for m in models {
        // Multimodal visibility: the projector is a real on-disk cost the
        // user otherwise cannot see anywhere (list was LLM-bytes only).
        let vision = m.mmproj_path.as_ref().map_or_else(
            || "-".to_string(),
            |p| {
                let mm = std::fs::metadata(p)
                    .map_or(0, |md| i64::try_from(md.len()).unwrap_or(i64::MAX));
                if mm > 0 {
                    format!("+{}", humansize(mm))
                } else {
                    "missing!".to_string()
                }
            },
        );
        let name = if m.name.len() > name_cap {
            format!("{}…", &m.name[..name_cap - 1])
        } else {
            m.name.clone()
        };
        println!(
            "{:<name_w$}  {:<8}  {:<9}  {:<10}  {:<8}  {:>7}  {}",
            name,
            m.quant,
            humansize(m.bytes),
            vision,
            m.arch.as_deref().unwrap_or("?"),
            m.ctx_train.map_or_else(String::new, |c| c.to_string()),
            m.path,
            name_w = name_w
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
        if let Some(q) = &m.quantized_by {
            println!("quantized_by: {q}");
        }
        if let Some(v) = &m.general_version {
            println!("version: {v}");
        }
        // Structural GGUF lint (H4): warn-only, explains degraded sizing.
        for w in m.lint() {
            println!("WARNING: {w}");
        }
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
        let _: serde_json::Value = cli_http()
            .get(format!("{base}/api/ps"))
            .send()
            .await?
            .json()
            .await?;
        println!("circuits reset");
        return Ok(());
    }
    let base = ensure_daemon().await?;
    let v: serde_json::Value = cli_http()
        .get(format!("{base}/api/ps"))
        .send()
        .await?
        .json()
        .await?;
    let models = v["models"].as_array().cloned().unwrap_or_default();
    // Which binary owns the daemon (stale-copy race visibility).
    let d = dirs();
    if let (Ok(pid), Ok(path)) = (
        std::fs::read_to_string(d.run_dir().join("pallama.pid")),
        std::fs::read_to_string(d.run_dir().join("daemon.path")),
    ) {
        println!("daemon pid {} ({})", pid.trim(), path.trim());
    }
    if models.is_empty() {
        println!("no models loaded");
        return Ok(());
    }
    println!(
        "{:<24} {:<9} {:>7} {:>6} {:>10}  ENDPOINT",
        "NAME", "STATE", "CTX", "GPU", "IN_FLIGHT"
    );
    for m in models {
        let display = match m["pallama_replica"].as_i64() {
            Some(r) => format!("{}#{}", m["name"].as_str().unwrap_or("?"), r),
            None => m["name"].as_str().unwrap_or("?").to_string(),
        };
        println!(
            "{:<24} {:<9} {:>7} {:>6} {:>10}  {}",
            display,
            m["pallama_state"].as_str().unwrap_or("?"),
            m["pallama_ctx"].as_i64().unwrap_or(0),
            m["pallama_gpu"].as_str().unwrap_or("-"),
            m["pallama_in_flight"].as_i64().unwrap_or(0),
            m["pallama_endpoint"].as_str().unwrap_or("-")
        );
    }
    Ok(())
}

/// `pallama run` dispatch: inline prompt = single-shot, none = REPL.
async fn run_dispatch(
    model: &str,
    prompt: &[String],
    verbose: bool,
    max_tokens: Option<u64>,
) -> Result<()> {
    if prompt.is_empty() {
        return run_repl(model).await;
    }
    let base = ensure_daemon().await?;
    let text = prompt.join(" ");
    let mut body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "stream": true,
    });
    if let Some(n) = max_tokens {
        // /api/chat is the ollama dialect: the generation cap lives in
        // options.num_predict (translated to max_tokens for the engine),
        // a top-level max_tokens key would be silently ignored.
        body["options"] = serde_json::json!({ "num_predict": n });
    }
    let final_chunk = stream_chat(&base, &body).await?;
    if verbose {
        if let Some(v) = final_chunk {
            let ec = v["eval_count"].as_i64().unwrap_or(0);
            let ed = v["eval_duration"].as_i64().unwrap_or(0);
            let pc = v["prompt_eval_count"].as_i64().unwrap_or(0);
            #[allow(clippy::cast_precision_loss, reason = "token counts fit f64 exactly")]
            let rate = if ed > 0 {
                f64::from(ec as f32) * 1e9 / f64::from(ed as f32)
            } else {
                0.0
            };
            println!("\n\ntotal duration: answer {ec} tokens at {rate:.1} t/s; prompt {pc} tokens");
        }
    }
    Ok(())
}

/// ollama cloud commands are refused, loudly: pallama is local-only by
/// design and silently no-op-ing would hide the difference.
fn cloud_refusal(cmd: &str, model: &str) -> Result<()> {
    let what = if model.is_empty() {
        cmd.to_string()
    } else {
        format!("{cmd} {model}")
    };
    Err(anyhow!(
        "pallama {what}: refused — pallama is local-only by design (no registry, no cloud accounts). \
Pull models straight from Hugging Face: pallama pull <owner/repo:QUANT> \
· discover GGUFs first: pallama search <terms>"
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
    let resp = cli_http()
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
    for r in &records {
        print_record(r);
    }
    Ok(())
}

/// One compact observation line + detection/retry lines (why + watch).
fn print_record(r: &serde_json::Value) {
    let num = |v: &serde_json::Value| -> String {
        v.as_u64().map_or_else(|| "?".into(), |n| n.to_string())
    };
    let detections = r["detections"].as_array().cloned().unwrap_or_default();
    let flag = if detections.is_empty() {
        "ok"
    } else {
        "FLAGGED"
    };
    // Response confidence (R2): mean/min token logprob when requested.
    let conf = match (
        r["logprob_mean"].as_f64(),
        r["logprob_min"].as_f64(),
        r["logprob_tokens"].as_u64(),
    ) {
        (Some(mean), Some(min), Some(n)) => {
            format!(" conf={mean:.2}/{min:.2}({n}tok)")
        }
        _ => String::new(),
    };
    println!(
        "{flag}  {}  {}  model={} status={} finish={} ctx={} prompt={} completion={} degraded={} {ms}{conf}",
        r["trace"].as_str().unwrap_or("?"),
        r["route"].as_str().unwrap_or("?"),
        r["model"].as_str().unwrap_or("?"),
        num(&r["status"]),
        r["finish"].as_str().unwrap_or("-"),
        num(&r["ctx"]),
        num(&r["prompt_tokens"]),
        num(&r["completion_tokens"]),
        r["degraded"].as_bool().unwrap_or(false),
        ms = num(&r["ms"]),
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

/// `pallama watch` — live SSE tail of sentinel records. One line per
/// observation, detections + retry hints underneath. Ctrl-C stops.
async fn watch() -> Result<()> {
    let base = ensure_daemon().await?;
    // Default reqwest client carries no total-request timeout — exactly
    // what a long-lived tail needs; Ctrl-C is the off switch.
    let mut resp = reqwest::Client::new()
        .get(format!("{base}/api/watch"))
        .send()
        .await?;
    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(anyhow!("daemon: {text}"));
    }
    println!("watching sentinel — Ctrl-C to stop");
    let mut buf = String::new();
    let mut chunk_lines = pallama_gateway::translate::LineBuffer::new();
    while let Some(chunk) = futures_lite_next(&mut resp).await? {
        buf.push_str(&chunk_lines.feed(&chunk));
        while let Some(pos) = buf.find("\n\n") {
            let frame: String = buf.drain(..pos + 2).collect();
            for line in frame.lines() {
                if let Some(payload) = line.strip_prefix("data: ") {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) {
                        print_record(&v);
                    }
                }
            }
        }
        std::io::Write::flush(&mut std::io::stdout()).ok();
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
    let mut spec = CreateSpec {
        base: String::new(),
        ctx: None,
        loras: Vec::new(),
        unsupported: Vec::new(),
    };
    for (n, raw_line) in raw.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        match parts
            .next()
            .unwrap_or_default()
            .to_ascii_uppercase()
            .as_str()
        {
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
    let raw = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
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
    pallama_core::persist_config(&cfg_path, &cfg.to_toml().map_err(|e| anyhow!("{e}"))?)?;
    println!(
        "created {model} from {} (hardlink + overlay: ctx={:?}, loras={})",
        spec.base,
        spec.ctx,
        spec.loras.len()
    );
    Ok(())
}

/// `pallama keys [list]|add|rm` — HTTP client to the daemon's
/// `/api/keys` (the daemon is the single config writer: no file races).
#[derive(clap::Subcommand)]
enum KeysAction {
    /// List keys with redacted secrets + today's usage
    List,
    /// Add a key; prints the generated secret ONCE
    Add {
        name: String,
        /// Comma-separated model allowlist (empty = all models / admin)
        #[arg(long)]
        models: Option<String>,
        /// Requests-per-minute cap (0 = unlimited)
        #[arg(long)]
        rpm: Option<u32>,
        /// Tokens-per-minute cap (0 = unlimited)
        #[arg(long)]
        tpm: Option<u64>,
        /// Total tokens per UTC day (0 = unlimited)
        #[arg(long)]
        daily_tokens: Option<u64>,
        /// Max simultaneous in-flight requests (0 = unlimited)
        #[arg(long)]
        max_concurrent: Option<u32>,
    },
    /// Remove a key by name (refuses the last one)
    Rm { name: String },
    /// Mint a fresh secret for a key (scopes/limits/usage kept)
    Rotate { name: String },
}

#[allow(clippy::too_many_lines)] // one match arm per subcommand, flat by design
/// Admin bearer for CLI-to-gateway calls that key-gate once [[keys]] exist:
/// first non-empty `name:key` from `PALLAMA_KEYS`, else the first unscoped
/// (admin) [[keys]] entry from config.toml.
fn admin_bearer() -> Option<String> {
    if let Ok(v) = std::env::var("PALLAMA_KEYS") {
        // F133: a colon-less entry has no secret half — using the entry
        // itself as the bearer would send the NAME as an auth token.
        // Skip with a loud teaching line, keep scanning valid entries.
        for entry in v.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            match entry.split_once(':') {
                Some((_name, secret)) => return Some(secret.to_string()),
                None => eprintln!(
                    "pallama: ignoring malformed PALLAMA_KEYS entry '{entry}' \
                     (expected name:key); falling back to config keys"
                ),
            }
        }
    }
    config().ok().and_then(|c| {
        c.keys
            .iter()
            .find(|k| k.models.is_empty())
            .map(|k| k.key.clone())
    })
}

#[allow(clippy::too_many_lines)]
async fn keys_cmd(action: Option<KeysAction>) -> Result<()> {
    let base = ensure_daemon().await?;
    let client = cli_http();
    // `/api/keys` demands an unscoped (admin) bearer once keys exist.
    // PALLAMA_KEYS entries are admin by construction; otherwise the
    // first unscoped [[keys]] entry from config.toml.
    let bearer = admin_bearer();
    match action.unwrap_or(KeysAction::List) {
        KeysAction::List => {
            let mut req = client.get(format!("{base}/api/keys"));
            if let Some(b) = &bearer {
                req = req.bearer_auth(b);
            }
            let resp = req.send().await?;
            let status = resp.status();
            let body: serde_json::Value = resp.json().await.unwrap_or(serde_json::Value::Null);
            if !status.is_success() {
                return Err(anyhow!(
                    "{}",
                    body["error"]["message"].as_str().unwrap_or("list failed")
                ));
            }
            let keys = body["keys"].as_array().cloned().unwrap_or_default();
            if keys.is_empty() {
                println!("no keys configured (gateway is authless)");
                return Ok(());
            }
            println!(
                "{:<12} {:<16} {:<24} {:>8} {:>10} {:>12} {:>6}",
                "NAME", "KEY", "MODELS", "RPM", "TPM", "DAILY", "CONC"
            );
            for k in &keys {
                let models = k["models"].as_array().cloned().unwrap_or_default();
                let models = if models.is_empty() {
                    "* (admin)".to_string()
                } else {
                    models
                        .iter()
                        .filter_map(|m| m.as_str())
                        .collect::<Vec<_>>()
                        .join(",")
                };
                println!(
                    "{:<12} {:<16} {:<24} {:>8} {:>10} {:>12} {:>6}",
                    k["name"].as_str().unwrap_or("?"),
                    k["key"].as_str().unwrap_or("?"),
                    models,
                    k["rpm"].as_u64().unwrap_or(0),
                    k["tpm"].as_u64().unwrap_or(0),
                    k["daily_tokens"].as_u64().unwrap_or(0),
                    k["max_concurrent"].as_u64().unwrap_or(0),
                );
                let u = &k["usage"];
                println!(
                    "  usage today {}: {} requests, {} tokens",
                    u["day"].as_str().unwrap_or("-"),
                    u["requests"].as_u64().unwrap_or(0),
                    u["tokens"].as_u64().unwrap_or(0),
                );
            }
            Ok(())
        }
        KeysAction::Add {
            name,
            models,
            rpm,
            tpm,
            daily_tokens,
            max_concurrent,
        } => {
            let mut body = serde_json::json!({"name": name});
            if let Some(m) = models {
                body["models"] = serde_json::Value::Array(
                    m.split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(String::from)
                        .map(serde_json::Value::String)
                        .collect(),
                );
            }
            if let Some(v) = rpm {
                body["rpm"] = v.into();
            }
            if let Some(v) = tpm {
                body["tpm"] = v.into();
            }
            if let Some(v) = daily_tokens {
                body["daily_tokens"] = v.into();
            }
            if let Some(v) = max_concurrent {
                body["max_concurrent"] = v.into();
            }
            let mut req = client.post(format!("{base}/api/keys")).json(&body);
            if let Some(b) = &bearer {
                req = req.bearer_auth(b);
            }
            let resp = req.send().await?;
            let status = resp.status();
            let out: serde_json::Value = resp.json().await.unwrap_or(serde_json::Value::Null);
            if !status.is_success() {
                return Err(anyhow!(
                    "{}",
                    out["error"]["message"].as_str().unwrap_or("add failed")
                ));
            }
            println!(
                "created key {} — secret (shown once):",
                out["created"].as_str().unwrap_or(&name)
            );
            println!("  {}", out["key"].as_str().unwrap_or("?"));
            Ok(())
        }
        KeysAction::Rotate { name } => {
            let mut req = client.post(format!("{base}/api/keys/rotate?name={name}"));
            if let Some(b) = &bearer {
                req = req.bearer_auth(b);
            }
            let resp = req.send().await?;
            let status = resp.status();
            let out: serde_json::Value = resp.json().await.unwrap_or(serde_json::Value::Null);
            if !status.is_success() {
                return Err(anyhow!(
                    "{}",
                    out["error"]["message"].as_str().unwrap_or("rotate failed")
                ));
            }
            println!(
                "rotated key {} — new secret (shown once): {}",
                out["rotated"].as_str().unwrap_or(&name),
                out["key"].as_str().unwrap_or("?")
            );
            Ok(())
        }
        KeysAction::Rm { name } => {
            let mut req = client.delete(format!("{base}/api/keys?name={name}"));
            if let Some(b) = &bearer {
                req = req.bearer_auth(b);
            }
            let resp = req.send().await?;
            let status = resp.status();
            let out: serde_json::Value = resp.json().await.unwrap_or(serde_json::Value::Null);
            if !status.is_success() {
                return Err(anyhow!(
                    "{}",
                    out["error"]["message"].as_str().unwrap_or("rm failed")
                ));
            }
            println!("removed key {name}");
            Ok(())
        }
    }
}

/// `pallama quantize <model> -t Q4_K_M` — run the engine's own
/// llama-quantize and register the output in the store.
#[allow(clippy::too_many_lines)]
fn quantize_cmd(
    model: &str,
    qtype: &str,
    name: Option<&str>,
    imatrix: Option<&std::path::Path>,
    allow_requantize: bool,
    verify_gate_pct: Option<f64>,
) -> Result<()> {
    if !pallama_runtime::quantize::plausible_quant_type(qtype) {
        return Err(anyhow!(
            "implausible quant type {qtype:?}: expected something like Q4_K_M, Q5_K_S, IQ4_XS, f16"
        ));
    }
    let d = dirs();
    let store = Store::open(&d)?;
    let row = store
        .get_model(model)?
        .ok_or_else(|| anyhow!("unknown model {model:?} (pallama list)"))?;
    let bin = pallama_runtime::quantize::find_quantize_bin(&d)?;
    let base = row.name.split('-').next().unwrap_or(&row.name).to_string();
    let out_name = name.map_or_else(|| format!("{base}-{qtype}"), str::to_string);
    let dst = d.models_dir().join(format!("{out_name}.gguf"));
    if dst.exists() {
        return Err(anyhow!(
            "output {} already exists (rm it first or pass --name)",
            dst.display()
        ));
    }
    println!("quantizing {} -> {} ({qtype})", row.path, dst.display());
    let src_bytes = std::fs::metadata(&row.path)?.len();
    // Optional importance matrix: calibrate first, then quantize with it
    // (verified upstream CLI: llama-imatrix -m -f -o --output-format gguf;
    // llama-quantize --imatrix file).
    let mut imatrix_args: Vec<String> = match &imatrix {
        Some(calib) => {
            let ibin = pallama_runtime::quantize::find_imatrix_bin(&d)?;
            let ipath = d.data_dir.join(format!(
                "{}-imatrix.gguf",
                row.name.replace(['/', '\\', ':'], "_")
            ));
            println!(
                "calibrating imatrix from {} (a full forward pass — takes minutes)...",
                calib.display()
            );
            let im = pallama_runtime::quantize::imatrix(
                &ibin,
                std::path::Path::new(&row.path),
                calib,
                &ipath,
                0,
                |line| {
                    println!("  {line}");
                },
            )?;
            vec!["--imatrix".to_string(), im.display().to_string()]
        }
        None => Vec::new(),
    };
    if allow_requantize {
        imatrix_args.insert(0, "--allow-requantize".to_string());
    }
    let out = pallama_runtime::quantize::quantize_im(
        &bin,
        std::path::Path::new(&row.path),
        &dst,
        qtype,
        &imatrix_args,
        |line| {
            println!("  {line}");
        },
    )?;
    // R5: perplexity gate — measure base + output on a fixed probe corpus;
    // a failing output is deleted and never registered.
    if let Some(max_degradation_pct) = verify_gate_pct {
        let pbin = pallama_runtime::quantize::find_perplexity_bin(&d)?;
        let probe = pallama_runtime::quantize::write_verify_probe(&d)?;
        println!("verify: perplexity pass 1/2 (base) — full forward passes, this takes a while...");
        let base_ppl = pallama_runtime::quantize::perplexity(
            &pbin,
            std::path::Path::new(&row.path),
            &probe,
            |line| {
                println!("  {line}");
            },
        )?;
        println!("verify: perplexity pass 2/2 (output)...");
        let out_ppl = pallama_runtime::quantize::perplexity(&pbin, &out, &probe, |line| {
            println!("  {line}");
        })?;
        if let Err(e) =
            pallama_runtime::quantize::verify_gate(base_ppl, out_ppl, max_degradation_pct / 100.0)
        {
            let _ = std::fs::remove_file(&out);
            return Err(e.context(format!("quantized output removed ({})", dst.display())));
        }
        #[allow(clippy::cast_precision_loss)] // display only
        let delta_pct = 100.0 * (out_ppl - base_ppl) / base_ppl;
        println!(
            "verify: PPL {base_ppl:.4} -> {out_ppl:.4} ({delta_pct:+.1}% vs gate {max_degradation_pct:.1}%) — pass"
        );
    }
    // Read the result's own metadata (fails loud on truncated output).
    let meta = pallama_core::read_metadata_file(&out)
        .map_err(|e| anyhow!("output not a readable GGUF ({e}): {}", out.display()))?;
    let out_bytes = std::fs::metadata(&out)?.len();
    let bytes = i64::try_from(out_bytes).unwrap_or(i64::MAX);
    store.upsert_model(&pallama_core::ModelRow {
        name: out_name.clone(),
        repo: format!("{}/local-quant", row.repo),
        quant: qtype.to_string(),
        path: out.display().to_string(),
        bytes,
        sha256: None,
        mmproj_path: row.mmproj_path.clone(),
        shards: 1,
        arch: Some(meta.architecture.clone()),
        params: Some(pallama_runtime::hf::est_params(out_bytes, qtype)),
        ctx_train: meta.context_length.and_then(|c| i64::try_from(c).ok()),
        pulled_at: i64::try_from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        )
        .unwrap_or(0),
    })?;
    #[allow(clippy::cast_precision_loss)] // percent display only
    let pct = 100.0 * out_bytes as f64 / src_bytes as f64;
    println!(
        "registered {out_name} — {} -> {} MiB ({pct:.0}% of source); projectors inherited: {}",
        src_bytes / (1024 * 1024),
        out_bytes / (1024 * 1024),
        row.mmproj_path.is_some()
    );
    Ok(())
}

/// `pallama launch <cli> [args...]` — ollama-`launch`-class ergonomics
/// without ollama's coupling: point any OpenAI/Anthropic/ollama-speaking
/// CLI at the daemon via env, optionally pre-warm a model, exec.
async fn launch_cmd(command: Vec<String>, warm: Option<String>, key: Option<String>) -> Result<()> {
    let Some(program) = command.first().cloned() else {
        return Err(anyhow!(
            "usage: pallama launch [--warm MODEL] [--key KEY] <cli> [args...]\n\
             examples: pallama launch claude\n\
                        pallama launch --warm qwen3.5-9b dsh\n\
             the CLI inherits OPENAI_BASE_URL / ANTHROPIC_BASE_URL / OLLAMA_HOST"
        ));
    };
    let base = ensure_daemon().await?;
    if let Some(model) = &warm {
        println!("pre-warming {model} ...");
        let resp = cli_http()
            .post(format!("{base}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": model, "max_tokens": 1,
                "messages": [{"role": "user", "content": "warm"}],
            }))
            .send()
            .await?;
        if !resp.status().is_success() {
            return Err(anyhow!(
                "warm-up failed: HTTP {} — is {model:?} pulled?",
                resp.status()
            ));
        }
    }
    let http = base.replacen("http://", "", 1);
    let mut cmd = std::process::Command::new(&program);
    cmd.args(&command[1..]);
    for (k, v) in [
        ("OPENAI_BASE_URL", base.clone()),
        ("OPENAI_API_BASE", format!("{base}/v1")),
        ("ANTHROPIC_BASE_URL", base.clone()),
        ("OLLAMA_HOST", http.clone()),
    ] {
        cmd.env(k, v);
    }
    if let Some(k) = &key {
        // A literal plm_ secret passes through; a key NAME resolves
        // against config (secret must exist there).
        let secret = if k.starts_with("plm_") {
            k.clone()
        } else {
            let cfg = config()?;
            cfg.keys
                .iter()
                .find(|e| &e.name == k)
                .map(|e| e.key.clone())
                .ok_or_else(|| anyhow!("no key named {k:?} in config.toml [[keys]]"))?
        };
        cmd.env("OPENAI_API_KEY", &secret);
        cmd.env("ANTHROPIC_API_KEY", &secret);
    }
    println!("launching {program} against {base}");
    let status = cmd
        .status()
        .map_err(|e| anyhow!("exec {program:?}: {e} — is it on PATH?"))?;
    std::process::exit(status.code().unwrap_or(1));
}

/// `pallama drafts <model>` — speculative-decoding draft candidates:
/// EAGLE3/MTP head repos first (they exist for common families), then
/// tiny same-family instruct models (the ngram-class fallback). Ranked
/// by HF downloads; the bench (`tune --spec`) proves net-positive
/// before adoption — suggestions are candidates, not pairings.
#[allow(clippy::too_many_lines)] // flat query loop by design
async fn drafts_cmd(model: &str) -> Result<()> {
    let client = pallama_runtime::hf::HfClient::new(std::env::var("HF_TOKEN").ok())?;
    // Family prefix: "qwen3.5-9b" -> "qwen3", "llama-3.1-8b" -> "llama".
    let family = model
        .split(['-', '.', ':'])
        .next()
        .unwrap_or(model)
        .to_lowercase();
    let queries = [
        format!("{family} eagle3 gguf"),
        format!("{family} mtp gguf"),
        format!("{family} 0.5b gguf"),
        format!("{family} 0.6b gguf"),
    ];
    let mut seen = std::collections::HashSet::new();
    println!("draft candidates for {model:?} (verify with `pallama tune {model} --spec`):");
    let mut any = false;
    for q in &queries {
        for e in client.search(q, 5).await? {
            let repo = e.id.clone();
            if seen.insert(repo.clone()) {
                let dl = e.downloads.unwrap_or(0);
                println!(
                    "  {:<58} {:>10} downloads  [{q}]",
                    repo,
                    humansize(i64::try_from(dl).unwrap_or(i64::MAX))
                );
                any = true;
            }
        }
    }
    if !any {
        println!(
            "  (no candidates found — pull a small same-family model and set spec = \"draft\")"
        );
    }
    Ok(())
}

/// `pallama migrate` — one-click config persistence: parses (which
/// auto-migrates legacy fields in memory), writes the canonical TOML,
/// keeps a timestamped backup of the previous file. Idempotent: a
/// second run is a no-op.
fn migrate_cmd() -> Result<()> {
    let d = dirs();
    let path = d.config_file();
    if !path.exists() {
        println!(
            "no config at {} — nothing to migrate (defaults apply)",
            path.display()
        );
        return Ok(());
    }
    let raw = std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let cfg = pallama_core::Config::from_toml(&raw)?;
    if !pallama_core::Config::raw_has_legacy_keys(&raw) {
        println!("config already canonical — nothing to migrate");
        return Ok(());
    }
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let backup = path.with_extension(format!("toml.bak-{ts}"));
    std::fs::copy(&path, &backup)
        .with_context(|| format!("backup {} -> {}", path.display(), backup.display()))?;
    let keys = cfg.keys.len();
    pallama_core::persist_config(&path, &cfg.to_toml()?)?;
    println!(
        "migrated: {} api_keys entr{} -> [[keys]] (admin power kept); backup at {}",
        keys,
        if keys == 1 { "y" } else { "ies" },
        backup.display()
    );
    Ok(())
}

/// `pallama snapshot` — backup the small state (config, store,
/// sessions list) to `<data>/snapshots/<ts>/`. Models/engines stay in
/// place (they ARE the bulk; re-pull or re-copy them deliberately).
/// Snapshots kept on disk; older ones are pruned on each `pallama snapshot`.
const SNAPSHOTS_KEEP: usize = 10;

fn snapshot_cmd() -> Result<()> {
    let d = dirs();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let dir = d.data_dir.join("snapshots").join(ts.to_string());
    std::fs::create_dir_all(&dir)?;
    let mut copied = vec![];
    if d.config_file().exists() {
        let dst = dir.join("config.toml");
        std::fs::copy(d.config_file(), &dst)
            .with_context(|| format!("copy config -> {}", dst.display()))?;
        copied.push("config.toml".to_string());
    }
    if d.db_file().exists() {
        // F130: VACUUM INTO, never a raw copy of the live WAL database.
        let store = Store::open(&d)?;
        store
            .snapshot_db_to(&dir.join("pallama.db"))
            .with_context(|| "snapshot pallama.db (VACUUM INTO)".to_string())?;
        copied.push("pallama.db".to_string());
    }
    prune_snapshots(&d, SNAPSHOTS_KEEP);
    println!(
        "snapshot at {} ({}); models/engines not copied (bulk)",
        dir.display(),
        copied.join(", ")
    );
    Ok(())
}

/// Remove oldest timestamped snapshot dirs beyond `keep`. Only touches
/// entries whose name parses as a unix-seconds stamp — never stray files.
fn prune_snapshots(d: &PallamaDirs, keep: usize) {
    let Ok(entries) = std::fs::read_dir(d.data_dir.join("snapshots")) else {
        return;
    };
    let mut stamps: Vec<u64> = entries
        .flatten()
        .filter_map(|e| e.file_name().to_str().and_then(|n| n.parse::<u64>().ok()))
        .collect();
    stamps.sort_unstable();
    while stamps.len() > keep {
        let oldest = stamps.remove(0);
        let _ = std::fs::remove_dir_all(d.data_dir.join("snapshots").join(oldest.to_string()));
    }
}

/// `pallama coreside` — which local models can stay loaded together.
fn coreside_cmd() -> Result<()> {
    let d = dirs();
    let store = Store::open(&d)?;
    let cfg = config()?;
    let models = store.list_models()?;
    if models.is_empty() {
        return Err(anyhow!("no models pulled (pallama pull <model>)"));
    }
    let manifest = store
        .active_engine()
        .ok()
        .flatten()
        .and_then(|e| serde_json::from_str::<pallama_runtime::Manifest>(&e.manifest).ok());
    let vram = manifest.as_ref().map_or(0, |m| {
        pallama_runtime::probe_hardware(Some(m)).total_vram_mib()
    });
    if vram == 0 {
        return Err(anyhow!(
            "no GPU detected — coreside planning is a VRAM question; CPU boxes load one model at a time anyway"
        ));
    }
    let mut fps = Vec::new();
    for m in &models {
        let ctx = cfg.effective_ctx(&m.name);
        // Same unified decision as the spawn profile: --kv-unified hosts
        // the KV buffer in --cache-ram (system RAM), so the VRAM plan
        // charges the measured working-set floor, not f16-at-ctx — f16
        // here made unified models read as 2-4x their real VRAM demand
        // and get deferred on phantom pressure.
        let unified = manifest
            .as_ref()
            .is_some_and(|mf| pallama_core::profile::kv_unified_for(&cfg, &m.name, &mf.flags));
        let kv =
            pallama_core::read_metadata_file(std::path::Path::new(&m.path)).map_or(0, |meta| {
                if unified {
                    pallama_core::profile::KV_UNIFIED_VRAM_FLOOR_BYTES / (1024 * 1024)
                } else {
                    pallama_core::coreside::kv_f16_mib(&meta, u64::from(ctx))
                }
            });
        fps.push(pallama_core::coreside::Footprint {
            name: m.name.clone(),
            weights_mib: (u64::try_from(m.bytes.max(0)).unwrap_or(u64::MAX)) / (1024 * 1024),
            kv_mib: kv,
            ctx,
            heat: 0,
        });
    }
    let (resident, deferred) = pallama_core::coreside::plan(&fps, vram);
    println!("VRAM {vram} MiB — co-residency plan (weights + KV @ ctx, 512 MiB headroom; unified = RAM-hosted KV, VRAM floor 512M):");
    println!(
        "{:<24} {:>8} {:>9} {:>7}",
        "MODEL", "WEIGHTS", "KV@CTX", "CTX"
    );
    for f in &resident {
        println!(
            "{:<24} {:>7}M {:>8}M {:>7}",
            f.name, f.weights_mib, f.kv_mib, f.ctx
        );
    }
    if deferred.is_empty() {
        println!("all models co-resident");
    } else {
        println!("deferred (swap in on demand):");
        for f in &deferred {
            println!(
                "{:<24} {:>7}M {:>8}M {:>7}",
                f.name, f.weights_mib, f.kv_mib, f.ctx
            );
        }
    }
    Ok(())
}

/// `pallama whisper file.mp3` — STT via the `whisper` [[remotes]] entry.
/// The gateway already forwards `/v1/audio/transcriptions`; this is the
/// local CLI convenience for it (no separate engine lane to manage).
#[allow(clippy::too_many_lines)]
async fn whisper_cmd(
    file: Option<&PathBuf>,
    model: Option<String>,
    install: bool,
    tag: Option<String>,
    pull: Option<String>,
    list: bool,
    pin: Option<String>,
) -> Result<()> {
    let d = dirs();
    if let Some(value) = pin {
        let unpin = value.trim().eq_ignore_ascii_case("none");
        let target = if unpin { None } else { Some(value.as_str()) };
        pallama_runtime::whisper::set_pin(&d, target)?;
        if unpin {
            println!("whisper pin removed — tracking the newest installed tag");
        } else {
            println!("whisper server pinned to {}", value.trim());
        }
        return Ok(());
    }
    if list {
        match pallama_runtime::whisper::server_bin(&d) {
            Some((bin, _)) => {
                let tag = bin
                    .parent()
                    .and_then(|p| p.parent())
                    .and_then(|p| p.file_name())
                    .map_or_else(|| "?".into(), |t| t.to_string_lossy().into_owned());
                let pin = if pallama_runtime::whisper::pinned_tag(&d).is_some() {
                    " (pinned)"
                } else {
                    ""
                };
                println!("server: {tag}{pin} ({})", bin.display());
            }
            None => println!("server: not installed (pallama whisper --install)"),
        }
        let installed = pallama_runtime::whisper::installed_tags(&d);
        if installed.len() > 1 {
            println!("installed: {}", installed.join(", "));
        }
        let models = pallama_runtime::whisper::list_models(&d);
        if models.is_empty() {
            println!("models: none (pallama whisper --pull base)");
        } else {
            println!("models:");
            for m in models {
                println!("  {m}");
            }
        }
        return Ok(());
    }
    if install {
        let pinned = tag.is_some();
        let token = std::env::var("GH_TOKEN").ok();
        let gh = GhClient::new(token)?;
        // Channel semantics for whisper mirror the engine lane: explicit
        // --tag pins the install; otherwise the configured channel
        // (latest = newest incl. prereleases, stable = GitHub's
        // releases/latest) resolves the newest release that actually
        // ships this platform's server binary (asset-aware: whisper.cpp
        // tags assetless v-releases) without pinning.
        let target = match tag.as_deref() {
            Some(t) => Some(t.to_string()),
            None => {
                Some(pallama_runtime::whisper::channel_target(&gh, config()?.update_channel).await?)
            }
        };
        let tag = pallama_runtime::whisper::install(&gh, &d, target.as_deref(), pinned).await?;
        let pin = if pinned { " (pinned)" } else { "" };
        println!("whisper.cpp server installed{pin}: {tag}");
        return Ok(());
    }
    if let Some(size) = pull {
        let token = std::env::var("HF_TOKEN").ok();
        let hf = pallama_runtime::hf::HfClient::new(token)?
            .with_download_connections(config()?.download_connections);
        let dest = pallama_runtime::whisper::pull(&hf, &d, &size, |done, total| {
            use std::io::Write as _;
            print!("\rpulling ggml-{size}.bin: {done}/{total} bytes");
            let _ = std::io::stdout().flush();
        })
        .await?;
        println!("\npulled {}", dest.display());
        return Ok(());
    }
    let file = file
        .ok_or_else(|| anyhow!("no audio file given — pass one, or use --install/--pull/--list"))?;
    let bytes = std::fs::read(file).with_context(|| format!("read {}", file.display()))?;
    let base = ensure_daemon().await?;
    // `/v1/audio/transcriptions` is key-gated once [[keys]] exist; pick the
    // same admin bearer the keys CLI uses so the local lane keeps working.
    let bearer = admin_bearer();
    // Local-first default: use the installed whisper lane when present,
    // else the historical whisper: remote prefix.
    let local_ready = pallama_runtime::whisper::server_bin(&d).is_some()
        && !pallama_runtime::whisper::list_models(&d).is_empty();
    let model = match model {
        Some(m) => m,
        None if local_ready => "whisper-1".to_string(),
        None => "whisper:whisper-1".to_string(),
    };
    // multipart: file + model fields (boundary-safe via reqwest)
    let part = reqwest::multipart::Part::bytes(bytes)
        .file_name(file.display().to_string())
        .mime_str("application/octet-stream")?;
    let form = reqwest::multipart::Form::new()
        .text("model", model)
        .part("file", part);
    let mut req = cli_http()
        .post(format!("{base}/v1/audio/transcriptions"))
        .multipart(form);
    if let Some(b) = &bearer {
        req = req.bearer_auth(b);
    }
    let resp = req.send().await?;
    let status = resp.status();
    let v: serde_json::Value = resp.json().await.unwrap_or(serde_json::Value::Null);
    if !status.is_success() {
        return Err(anyhow!(
            "transcription failed (HTTP {status}): {}",
            v["error"]["message"].as_str().unwrap_or("?")
        ));
    }
    match v.get("text").and_then(|t| t.as_str()) {
        Some(text) => {
            println!("{text}");
            Ok(())
        }
        None => Err(anyhow!("no \"text\" field in response: {v}")),
    }
}

/// `pallama stop` (daemon) vs `pallama stop MODEL` (unload now).
async fn stop_cmd(model: Option<String>) -> Result<()> {
    let Some(model) = model else {
        return stop();
    };
    let base = ensure_daemon().await?;
    let resp = cli_http()
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
    let v: serde_json::Value = cli_http()
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
    // Streaming lane: no total-request ceiling; the 600s per-request
    // timeout below is the bound (F126 exemption).
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
    let mut chunk_lines = pallama_gateway::translate::LineBuffer::new();
    let mut final_chunk: Option<serde_json::Value> = None;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    while let Some(chunk) = futures_lite_next(&mut resp).await? {
        buf.push_str(&chunk_lines.feed(&chunk));
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
    let row = store
        .get_model(model)?
        .ok_or_else(|| anyhow!("no such model: {model}"))?;
    let bench_bin = pallama_runtime::bench::find_bench_bin(&d)?;
    let tuner = pallama_runtime::Tuner {
        dirs: &d,
        bench_bin,
    };
    let rows = tuner.bench_default(std::path::Path::new(&row.path))?;
    println!(
        "{:<10} {:>8} {:>8} {:>6} {:<6} {:<6}",
        "TEST", "T/S", "CTX", "THREADS", "CTK", "CTV"
    );
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

#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_arguments)]
// Lane switches mirroring the clap flags one-to-one; a flags struct would
// just re-spell the same four booleans.
#[allow(clippy::fn_params_excessive_bools)]
fn tune_full(
    model: &str,
    search: bool,
    ctx: Option<u32>,
    spec: Option<String>,
    slots: Option<u32>,
    ngram: bool,
    load: bool,
    replicas: bool,
    cache_reuse: bool,
) -> Result<()> {
    let d = dirs();
    let store = Store::open(&d)?;
    let row = store
        .get_model(model)?
        .ok_or_else(|| anyhow!("no such model: {model}"))?;
    let engine_row = store
        .active_engine()?
        .ok_or_else(|| anyhow!("no engine installed; run: pallama engine update"))?;
    let manifest: pallama_runtime::Manifest = serde_json::from_str(&engine_row.manifest)?;
    let bench_bin = pallama_runtime::bench::find_bench_bin(&d)?;
    let mut cfg = config()?;
    // Persist --spec BEFORE building the profile input so the compiled
    // argv (and its draft resolution) matches what the daemon will
    // serve on the next run — mirrors the --ngram lane's
    // set-then-reload discipline.
    if let Some(mode) = spec {
        if mode != "off" && mode != "auto" {
            return Err(anyhow!("--spec must be \"off\" or \"auto\""));
        }
        set_model_override(model, "spec", &format!("\"{mode}\""))?;
        cfg = config()?;
        println!("model_overrides.{model}.spec = {mode}");
    }
    let hw = pallama_runtime::probe_hardware(Some(&manifest));
    let gguf = pallama_core::read_metadata_file(std::path::Path::new(&row.path))?;
    let overlay = cfg.overlay_for(model);
    let loras: Vec<(String, f64)> = store
        .list_loras(Some(model))?
        .into_iter()
        .map(|l| (l.path, l.scale))
        .collect();
    let data_dir = d.data_dir.to_string_lossy();
    // Draft resolution identical to the serve path: catalog pair ->
    // pulled store row. A paired draft must not brick `bench --spec
    // auto` with a false "not pulled" compile error.
    let spec_mode = overlay.spec.clone().unwrap_or_else(|| cfg.spec.clone());
    let draft = pallama_runtime::resolve_draft_path(&store, model, &spec_mode);
    let input = pallama_runtime::bench::build_input(
        model,
        &row.path,
        u64::try_from(row.bytes.max(0)).unwrap_or(u64::MAX),
        &gguf,
        &hw,
        &cfg,
        &overlay,
        &loras,
        draft.as_deref(),
        &engine_row.tag,
        &manifest.flags,
        &manifest.spec_types,
        pallama_core::Endpoint::Tcp {
            host: "127.0.0.1".into(),
            port: 0,
        },
        &data_dir,
    );
    let store2 = Store::open(&d)?;
    let tuner = pallama_runtime::Tuner {
        dirs: &d,
        bench_bin,
    };
    if search {
        let (profile, mut winning, rows) = tuner.tune_search(&store2, &input)?;
        // F6: record the winner so `engine update` can gate future
        // activations against a measured baseline (regression guard).
        if let Some(w) = rows
            .iter()
            .filter(|r| r.test == "tg128")
            .max_by(|a, b| a.ts.total_cmp(&b.ts))
        {
            let _ = store2.record_bench(
                &engine_row.tag,
                model,
                w.ts,
                rows.iter()
                    .find(|r| r.test == "pp512" && r.n_threads == w.n_threads)
                    .map_or(0.0, |r| r.ts),
                i64::from(profile.ctx),
            );
        }
        if let Some(n) = ctx {
            winning.ctx = Some(n);
            let fixed = tuner.adopt(&store2, &input, &winning)?;
            println!(
                "tuned {model} — {} configs measured; ctx pinned to {n}:",
                rows.len()
            );
            println!("  argv: {}", fixed.argv.join(" "));
        } else {
            println!("tuned {} — {} configs measured, winner:", model, rows.len());
            println!("  argv: {}", profile.argv.join(" "));
            println!("  ctx:   {}", profile.ctx);
        }
    } else if let Some(n) = ctx {
        let winning = pallama_core::TuningOverrides {
            ctx: Some(n),
            ..Default::default()
        };
        let fixed = tuner.adopt(&store2, &input, &winning)?;
        println!("profile adopted at ctx {n}: {}", fixed.argv.join(" "));
    } else {
        let rows = tuner.bench_default(std::path::Path::new(&row.path))?;
        println!(
            "{} rows measured; use --search (argmax) and/or --ctx/--spec to adopt",
            rows.len()
        );
    }
    if let Some(clients) = slots {
        // Live-concurrency slots axis: spawn real children per -np and
        // measure aggregate throughput under `clients` parallel streams.
        let profile = store2
            .get_profile(model, &engine_row.tag)?
            .ok_or_else(|| anyhow!("no profile yet — run `pallama tune {model} --search` first"))?;
        let argv: Vec<String> = serde_json::from_str::<serde_json::Value>(&profile.args_json)?
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let candidates = [1, 2, 4, 8];
        println!(
            "slots search: {clients} concurrent clients against -np {candidates:?} (live servers, ~1 min) ..."
        );
        let rows = tuner.slots_search(&argv, clients, &candidates, 1)?;
        println!("{:<8} {:>14}", "-np", "agg tok/s");
        for (np, tps) in &rows {
            println!("{np:<8} {tps:>14.1}");
        }
        let (best_np, best_tps) = rows
            .iter()
            .fold((0u32, 0.0), |acc, r| if r.1 > acc.1 { *r } else { acc });
        let second = rows
            .iter()
            .filter(|r| r.0 != best_np)
            .map(|r| r.1)
            .fold(0.0_f64, f64::max);
        if best_tps > second * 1.05 {
            // Adopt: meaningful winner (>5% over the runner-up).
            let mut cfg2 = config()?;
            cfg2.slots = best_np;
            pallama_core::persist_config(&d.config_file(), &cfg2.to_toml()?)?;
            println!(
                "adopted slots = {best_np} ({best_tps:.1} tok/s, +{:.0}% over runner-up) — restart the daemon to apply",
                (best_tps / second - 1.0) * 100.0
            );
        } else {
            println!(
                "no clear winner (best {best_tps:.1} vs {second:.1}) — slots stays {}; single-slot avoids ctx splitting",
                cfg.slots
            );
        }
    }
    if ngram {
        // Live n-gram tuning axis: real children across (size_m, min_hits),
        // single-stream (speculation is a c=1 win). Requires spec=ngram.
        let effective_spec = cfg
            .overlay_for(model)
            .spec
            .unwrap_or_else(|| cfg.spec.clone());
        if effective_spec != "ngram" {
            set_model_override(model, "spec", "\"ngram\"")?;
            cfg = config()?;
            println!("model_overrides.{model}.spec = ngram (set by --ngram)");
        }
        let overlay2 = cfg.overlay_for(model);
        // n-gram speculation self-drafts from context n-grams — no
        // external draft model to resolve.
        let input2 = pallama_runtime::bench::build_input(
            model,
            &row.path,
            u64::try_from(row.bytes.max(0)).unwrap_or(u64::MAX),
            &gguf,
            &hw,
            &cfg,
            &overlay2,
            &loras,
            None,
            &engine_row.tag,
            &manifest.flags,
            &manifest.spec_types,
            pallama_core::Endpoint::Tcp {
                host: "127.0.0.1".into(),
                port: 0,
            },
            &data_dir,
        );
        let base =
            pallama_core::profile::compile(&input2, &pallama_core::TuningOverrides::default())
                .map_err(|e| anyhow!("profile: {e}"))?;
        let candidates: [(u32, u32); 4] = [(16, 1), (16, 2), (32, 1), (32, 2)];
        println!(
            "ngram search: live servers across (size_m, min_hits) {candidates:?} (~1 min each) ..."
        );
        let rows = tuner.ngram_search(&base.argv, &candidates, 1)?;
        let ((bm, bh), best_tps) =
            rows.iter().fold(
                ((0u32, 0u32), 0.0),
                |acc, r| if r.1 > acc.1 { *r } else { acc },
            );
        let second = rows
            .iter()
            .filter(|r| r.0 != (bm, bh))
            .map(|r| r.1)
            .fold(0.0_f64, f64::max);
        println!("winner: size_m={bm} min_hits={bh} at {best_tps:.1} tok/s");
        if best_tps > second * 1.05 {
            let mut cfg2 = config()?;
            cfg2.ngram_size_m = bm;
            cfg2.ngram_min_hits = bh;
            pallama_core::persist_config(&d.config_file(), &cfg2.to_toml()?)?;
            println!(
                "adopted ngram_size_m = {bm}, ngram_min_hits = {bh} — restart the daemon to apply"
            );
        } else {
            println!("no clear winner (best {best_tps:.1} vs {second:.1}) — engine defaults stay");
        }
    }
    if load || replicas || cache_reuse {
        // Shared base profile for the spawn-your-own-children lanes: both
        // searches strip endpoint/session/warmup flags themselves, so the
        // children differ only in the axis under test.
        let overlay3 = cfg.overlay_for(model);
        let spec_mode3 = overlay3.spec.clone().unwrap_or_else(|| cfg.spec.clone());
        let draft3 = pallama_runtime::resolve_draft_path(&store, model, &spec_mode3);
        let input3 = pallama_runtime::bench::build_input(
            model,
            &row.path,
            u64::try_from(row.bytes.max(0)).unwrap_or(u64::MAX),
            &gguf,
            &hw,
            &cfg,
            &overlay3,
            &loras,
            draft3.as_deref(),
            &engine_row.tag,
            &manifest.flags,
            &manifest.spec_types,
            pallama_core::Endpoint::Tcp {
                host: "127.0.0.1".into(),
                port: 0,
            },
            &data_dir,
        );
        let base =
            pallama_core::profile::compile(&input3, &pallama_core::TuningOverrides::default())
                .map_err(|e| anyhow!("profile: {e}"))?;
        if load {
            // Warmup A/B axis: the two children differ ONLY in --no-warmup.
            // Orthogonal to spec mode — no ngram-style precondition.
            println!("load search: warmup A/B — 2 spawns, ready + first-chat each ...");
            let probe = tuner.load_search(&base.argv, 1)?;
            println!("warmup on : {:>6.1}s", probe.warmup_secs);
            println!("warmup off: {:>6.1}s", probe.no_warmup_secs);
            if probe.no_warmup_secs < probe.warmup_secs * 0.95 {
                set_model_override(model, "warmup", "false")?;
                println!(
                    "adopted model_overrides.{model}.warmup = false ({:.0}% faster to first token) — restart the daemon to apply",
                    (1.0 - probe.no_warmup_secs / probe.warmup_secs) * 100.0
                );
            } else {
                println!("warmup stays on (off not >5% faster) — first-token latency wins");
            }
        }
        if replicas {
            // Scaling axis: aggregate tok/s from 8 concurrent clients x 3
            // generations against 1 vs 2 identical children.
            println!(
                "replica search: 1 vs 2 children — 3 spawns total, 8 clients x 3 gens each ..."
            );
            let probe = tuner.replica_search(&base.argv, 1)?;
            let ratio = probe.r2_tps / probe.r1_tps.max(1e-6);
            println!("1 child   : {:>6.1} tok/s", probe.r1_tps);
            println!("2 children: {:>6.1} tok/s ({ratio:.2}x)", probe.r2_tps);
            if probe.r2_tps > probe.r1_tps * 1.3 {
                set_model_override(model, "replicas", "2")?;
                println!(
                    "adopted model_overrides.{model}.replicas = 2 — restart the daemon to apply"
                );
            } else {
                println!("keep replicas = 1 (2 children not >1.3x aggregate)");
            }
        }
        if cache_reuse {
            // Prompt-cache reuse axis: warm second-chat wall seconds on a
            // long repeated prompt across {0, 256, 512}. 256 is Pallama's
            // shipped default, so it is the bar to beat.
            println!(
                "cache-reuse search: grid {{0, 256, 512}} — 3 spawns, long repeated prompt x2 each ..."
            );
            let probe = tuner.cache_reuse_search(&base.argv, 1)?;
            for (n, secs) in &probe.grid {
                let mark = if *n == probe.best { "<- best" } else { "" };
                println!("  N={n:<4}: warm chat {secs:.2}s {mark}");
            }
            let default_secs = probe.grid.iter().find(|(n, _)| *n == 256).map(|(_, s)| *s);
            let adopt = probe.best != 256
                && default_secs.is_some_and(|d| {
                    let b = probe
                        .grid
                        .iter()
                        .find(|(n, _)| *n == probe.best)
                        .map_or(f64::MAX, |(_, s)| *s);
                    // Relative bar AND an absolute floor: sub-50 ms deltas on
                    // tiny models are timer noise, not cache effects.
                    b < d * 0.95 && (d - b) > 0.05
                });
            if adopt {
                // cache_reuse is a global knob (no overlay field): adopt via
                // the same config-rewrite path as the ngram lane.
                let mut cfg2 = config()?;
                cfg2.cache_reuse = probe.best;
                pallama_core::persist_config(&d.config_file(), &cfg2.to_toml()?)?;
                println!(
                    "adopted cache_reuse = {} — restart the daemon to apply",
                    probe.best
                );
            } else {
                println!("keep cache_reuse = 256 (no candidate >5% faster and >50 ms >5% faster)");
            }
        }
    }
    Ok(())
}

/// Key part of a `key = value` TOML line (trimmed), None when no `=`.
/// Tolerant of `key="v"`, `key = "v"` and leading indentation.
fn key_before_eq(line: &str) -> Option<&str> {
    let (k, _) = line.split_once('=')?;
    let k = k.trim();
    (!k.is_empty()).then_some(k)
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
        // F128: match `key = v`, `key="v"` and indented forms alike —
        // anything whose text before the first `=` trims to the key.
        if in_section && key_before_eq(line).is_some_and(|k| k == key) {
            out.push(format!("{key} = {value}"));
            replaced = true;
            continue;
        }
        out.push(line.to_string());
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
    pallama_core::persist_config(&path, &candidate)?;
    Ok(())
}

/// Gate baseline: the most recent measured (engine, tg, model) — the
/// row `tune --search` recorded. None when never tuned (gate no-ops).
fn gate_baseline(store: &Store) -> Result<Option<(String, f64, String)>> {
    // FIX5: MOST RECENT tune row (highest-tg picked the tiny model).
    Ok(store.latest_bench_by_time()?)
}

/// Single-shot tg128 on a specific engine's llama-bench.
fn quick_tg(d: &PallamaDirs, _row: &pallama_core::EngineRow, model_path: &str) -> Result<f64> {
    // F129: go through the shared exe-aware discovery instead of
    // hand-building a unix-only path.
    let bench_bin = pallama_runtime::bench::find_bench_bin(d)?;
    let tuner = pallama_runtime::bench::Tuner { dirs: d, bench_bin };
    let rows = tuner.bench_default(std::path::Path::new(model_path))?;
    let tg = rows
        .iter()
        .filter(|r| r.test == "tg128")
        .map(|r| r.ts)
        .fold(0.0_f64, f64::max);
    if tg > 0.0 {
        Ok(tg)
    } else {
        Err(anyhow!("gate bench produced no tg128 row"))
    }
}

/// F7 regression gate: DEFAULT-config tg128 on the freshly installed
/// engine vs the recorded baseline engine. Skips silently when no tune
/// baseline exists (nothing to compare). On a >10% decode drop it rolls
/// the previous engine back to active and errors.
fn engine_regression_gate(
    mgr: &EngineManager,
    d: &PallamaDirs,
    row: &pallama_core::EngineRow,
) -> Result<()> {
    let store = Store::open(d)?;
    if let Some((prev_tag, _recorded_tg, model)) = gate_baseline(&store)? {
        let mrow = store
            .get_model(&model)?
            .ok_or_else(|| anyhow!("gate baseline model {model:?} no longer pulled"))?;
        let prev_row = store
            .list_engines()?
            .into_iter()
            .find(|e| e.tag == prev_tag)
            .ok_or_else(|| anyhow!("gate: baseline engine {prev_tag} pruned"))?;
        let prev_tg = quick_tg(d, &prev_row, &mrow.path)?;
        let new_tg = quick_tg(d, row, &mrow.path)?;
        println!(
            "gate: {model} tg128 (default cfg) {prev_tg:.1} -> {new_tg:.1} t/s ({prev_tag} -> {})",
            row.tag
        );
        if new_tg < prev_tg * 0.9 {
            mgr.use_tag(&prev_tag)?;
            anyhow::bail!(
                "REGRESSION GATE TRIPPED: {:.0}% decode drop on {model} — \
                 rolled back to {prev_tag} (now active). {row_tag} stays \
                 installed for `pallama engine use {row_tag}` to force.",
                (1.0 - new_tg / prev_tg) * 100.0,
                row_tag = row.tag,
            );
        }
        let _ = store.record_bench(&row.tag, &model, new_tg, 0.0, 0);
    }
    Ok(())
}

async fn engine_cmd(cmd: EngineCmd) -> Result<()> {
    let d = dirs();
    match cmd {
        EngineCmd::Update { tag, no_gate } => engine_update(&d, tag, no_gate).await?,
        EngineCmd::List => {
            upstream_update_hint(&d).await;
            let store = Store::open(&d)?;
            for e in store.list_engines()? {
                println!(
                    "{:<12} {:<9} {:<10} {} {}",
                    e.tag,
                    e.kind.as_str(),
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
            restart_hint().await;
        }
        EngineCmd::Rollback => {
            let mgr = local_engine_manager(&d)?;
            let row = mgr.rollback()?;
            println!("rolled back to: {}", row.tag);
            restart_hint().await;
        }
        EngineCmd::Build {
            backend,
            tag,
            arch,
            cuda_host_compiler,
            jobs,
            no_gate,
        } => {
            engine_build(
                &d,
                BackendArg {
                    backend,
                    tag,
                    arch,
                    cuda_host_compiler,
                    jobs,
                    no_gate,
                },
            )
            .await?;
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
        EngineCmd::Install { tag } => engine_install_mistralrs(&d, tag).await?,
    }
    Ok(())
}

/// `pallama engine install` — install + activate a prebuilt mistral.rs
/// engine. Asset choice is GPU-aware (CPU / Metal / CUDA-by-driver). The
/// F7 decode-regression gate is llama-server-only: skipped, and SAID so
/// (never silently) — llama-bench cannot drive a mistral.rs child.
async fn engine_install_mistralrs(d: &PallamaDirs, tag: Option<String>) -> Result<()> {
    let mgr = local_engine_manager(d)?;
    let wanted = tag.clone().unwrap_or_else(|| "latest".to_string());
    println!("installing mistral.rs {wanted} (prebuilt upstream binary)");
    let row = mgr.update_mistralrs(tag.as_deref()).await?;
    let m: pallama_runtime::Manifest = serde_json::from_str(&row.manifest)?;
    println!(
        "engine {} active ({} flags probed, build {})",
        row.tag,
        m.flags.len(),
        m.build_number
    );
    println!("note: decode-regression gate is llama-server-only — skipped for mistral.rs engines");
    restart_hint().await;
    Ok(())
}

/// `pallama engine update` — install + activate the newest (or given)
/// upstream build, gated by the F7 decode-regression bench.
async fn engine_update(d: &PallamaDirs, tag: Option<String>, no_gate: bool) -> Result<()> {
    let token = std::env::var("GH_TOKEN").ok();
    let gh = GhClient::new(token)?;
    let cfg = config()?;
    // Explicit --tag bypasses the configured channel (power-user
    // override); otherwise the channel resolves the target. The
    // resolved release is passed through to the manager so the
    // whole flow is a single upstream fetch.
    let from_channel = tag.is_none();
    let resolved = match &tag {
        Some(_) => None,
        None => Some(gh.channel_b_release(cfg.update_channel).await?),
    };
    let target_tag = match (&tag, &resolved) {
        (Some(t), _) => t.clone(),
        (None, Some(rel)) => rel.tag_name.clone(),
        (None, None) => unreachable!("channel lane always resolves"),
    };
    // Channels are pins, not floors: switching latest -> stable
    // re-targets downward by design. The F7 gate compares perf
    // and would trip on any intentional downgrade, so it is
    // skipped when the target is older than the active engine.
    let active_tag = Store::open(d)?.active_engine()?.map(|e| e.tag);
    let downgrade = match &active_tag {
        Some(a) => matches!(
            (btag_number(a), btag_number(&target_tag)),
            (Some(x), Some(y)) if y < x
        ),
        None => false,
    };
    let mgr = EngineManager {
        dirs: d.clone(),
        gh,
        bus: EventBus::default(),
        asset_override: cfg.engine_asset,
    };
    let row = match resolved {
        Some(rel) => mgr.update_resolved(rel).await?,
        None => mgr.update(tag.as_deref(), cfg.update_channel).await?,
    };
    // Keep-CUDA skip (or a same-tag reinstall) resolves to the pre-call
    // active engine: nothing new was installed, so there is no fresh
    // binary to gate, compare, or restart the daemon for.
    let unchanged = active_tag.as_deref() == Some(row.tag.as_str());
    // F7 gate (FIX5): DEFAULT-config tg128 on BOTH engines —
    // comparing new-default vs baseline-argmax was apples-to-
    // oranges, biased to trip. Baseline = most recent tune row.
    // Skip: --no-gate, PALLAMA_ENGINE_GATE=0, or channel
    // downgrade (an older build losing to a newer one is the
    // point of the switch, not a regression).
    let gate_on =
        !no_gate && std::env::var("PALLAMA_ENGINE_GATE").as_deref() != Ok("0") && !downgrade;
    if unchanged {
        println!("engine {} already active — nothing new installed (see the warning above for lane options)", row.tag);
    } else if gate_on {
        engine_regression_gate(&mgr, d, &row)?;
    } else if downgrade {
        println!("channel switch: downgrade to {target_tag} — regression gate skipped");
        if let Some(prev) = &active_tag {
            println!("previous {prev} stays installed — `pallama engine use {prev}` restores it");
        }
    } else {
        println!("engine {} installed; regression gate skipped", row.tag);
    }
    let m: pallama_runtime::Manifest = serde_json::from_str(&row.manifest)?;
    if from_channel {
        println!("update channel: {}", cfg.update_channel);
    }
    if row.active {
        println!(
            "engine {} active (build {}, {} devices, {} flags)",
            row.tag,
            m.build_number,
            m.devices.len(),
            m.flags.len()
        );
    } else {
        // keep-cuda guard fired: an installed CUDA engine stays active.
        println!(
            "engine {} registered (build {}, {} devices, {} flags); the active CUDA \
             engine was kept — run `pallama engine use {}` to switch",
            row.tag,
            m.build_number,
            m.devices.len(),
            m.flags.len(),
            row.tag
        );
    }
    if !unchanged {
        restart_hint().await;
    }
    Ok(())
}

/// `pallama engine build` — compile llama.cpp from source into an
/// installable engine (see `engine::build`), then the same F7 gate and
/// summary as `engine update`.
struct BackendArg {
    backend: String,
    tag: Option<String>,
    arch: Option<String>,
    cuda_host_compiler: Option<PathBuf>,
    jobs: Option<usize>,
    no_gate: bool,
}

async fn engine_build(d: &PallamaDirs, a: BackendArg) -> Result<()> {
    let backend = match a.backend.as_str() {
        "cuda" => BuildBackend::Cuda,
        "cpu" => BuildBackend::Cpu,
        other => return Err(anyhow!("unknown backend {other:?} — supported: cuda, cpu")),
    };
    // Resolve the source tag exactly like Update: explicit tag, else the
    // channel target. Must land on a concrete b-tag.
    let token = std::env::var("GH_TOKEN").ok();
    let gh = GhClient::new(token)?;
    let cfg = config()?;
    let resolved = match &a.tag {
        Some(t) => gh.resolve_tag(t).await?.tag_name,
        None => gh.channel_b_release(cfg.update_channel).await?.tag_name,
    };
    if btag_number(&resolved).is_none() {
        return Err(anyhow!(
            "engine build needs a b-tag; channel resolved {resolved:?}"
        ));
    }
    println!(
        "building llama.cpp {resolved} (backend {}, from source — this needs \
         git + cmake + a C++ compiler{cuda_note})",
        backend.as_str(),
        cuda_note = if backend == BuildBackend::Cuda {
            " + the CUDA toolkit (nvcc)"
        } else {
            ""
        }
    );
    let mut opts = BuildOpts::new(backend, &resolved);
    opts.arch = a.arch;
    opts.cuda_host_compiler = a.cuda_host_compiler;
    if let Some(j) = a.jobs {
        opts.jobs = j;
    }
    let mgr = EngineManager {
        dirs: d.clone(),
        gh,
        bus: EventBus::default(),
        asset_override: cfg.engine_asset,
    };
    let row = mgr
        .build_and_install(&opts, &mut |line| println!("  {line}"))
        .await?;
    // Same F7 gate as Update: skip on --no-gate, env knob, or an
    // intentionally older build than the active engine.
    let active_tag = Store::open(d)?.active_engine()?.map(|e| e.tag);
    let downgrade = match &active_tag {
        Some(act) => matches!(
            (btag_number(act), btag_number(&resolved)),
            (Some(x), Some(y)) if y < x
        ),
        None => false,
    };
    let gate_on =
        !a.no_gate && std::env::var("PALLAMA_ENGINE_GATE").as_deref() != Ok("0") && !downgrade;
    if gate_on {
        engine_regression_gate(&mgr, d, &row)?;
    } else if downgrade {
        println!("older build {resolved}: regression gate skipped");
    } else {
        println!("engine {} installed; regression gate skipped", row.tag);
    }
    let m: pallama_runtime::Manifest = serde_json::from_str(&row.manifest)?;
    if row.active {
        println!(
            "engine {} active (build {}, {} devices, {} flags)",
            row.tag,
            m.build_number,
            m.devices.len(),
            m.flags.len()
        );
    } else {
        // keep-cuda guard fired: an installed CUDA engine stays active.
        println!(
            "engine {} registered (build {}, {} devices, {} flags); the active CUDA \
             engine was kept — run `pallama engine use {}` to switch",
            row.tag,
            m.build_number,
            m.devices.len(),
            m.flags.len(),
            row.tag
        );
    }
    restart_hint().await;
    Ok(())
}

/// Best-effort upstream check (user-invoked only, ~4s budget, silent on
/// failure): prints an update hint when a newer b-tag exists.
async fn upstream_update_hint(dirs: &PallamaDirs) {
    let Ok(store) = Store::open(dirs) else { return };
    let Ok(Some(active)) = store.active_engine() else {
        return;
    };
    if active.tag == "local" {
        return; // local build: upstream currency is the user's concern
    }
    let Ok(cfg) = Config::load(dirs) else { return };
    let token = std::env::var("GH_TOKEN").ok();
    let Ok(gh) = GhClient::new(token) else { return };
    let latest = tokio::time::timeout(
        std::time::Duration::from_secs(4),
        gh.channel_b_release(cfg.update_channel),
    )
    .await;
    if let Ok(Ok(rel)) = latest {
        if same_build(&active.tag, &rel.tag_name) {
            println!(
                "engine up to date: {} (channel: {})",
                active.tag, cfg.update_channel
            );
        } else {
            println!(
                "{} available: {} (active: {}, channel: {}) — run: {}",
                channel_word(&active.tag, &rel.tag_name),
                rel.tag_name,
                active.tag,
                cfg.update_channel,
                engine_update_command(&active.tag, &active.asset)
            );
        }
    }
}

/// Background engine-currency check (`engine_check_secs`, default daily):
/// asks GitHub for the newest llama.cpp b-release and writes a marker the
/// doctor/ps surfaces read. Never auto-installs — `pallama engine update`
/// stays a human action; failures log at debug and retry next tick.
fn spawn_engine_check_task(
    dirs: &PallamaDirs,
    every_secs: u64,
    engine: Arc<dyn pallama_runtime::Engine>,
) {
    let dirs = dirs.clone();
    tokio::spawn(async move {
        // First check fires immediately; a successful check repeats every
        // `every_secs`, a FAILED one retries in 10 minutes instead of
        // waiting a full day (a cold-start timeout must not blind the
        // daemon's currency marker until tomorrow).
        let retry_delay = std::time::Duration::from_mins(10).min(Duration::from_secs(every_secs));
        let full_delay = Duration::from_secs(every_secs);
        let mut delay = Duration::ZERO;
        loop {
            tokio::time::sleep(delay).await;
            let Ok(store) = Store::open(&dirs) else {
                delay = retry_delay;
                continue;
            };
            let Ok(Some(active)) = store.active_engine() else {
                delay = retry_delay;
                continue;
            };
            if active.tag == pallama_runtime::LOCAL_TAG {
                delay = full_delay;
                continue; // local build: currency is the user's concern
            }
            if active.kind == EngineKind::MistralRs {
                delay = full_delay;
                continue; // the llamacpp channel survey is meaningless
                          // against a mistral.rs tag (it would nag
                          // "update available: bNNNN" cross-kind);
                          // mistral.rs currency lives in `pallama doctor`
            }
            let Ok(cfg) = Config::load(&dirs) else {
                delay = retry_delay;
                continue;
            };
            let token = std::env::var("GH_TOKEN").ok();
            let Ok(gh) = GhClient::new(token) else {
                delay = retry_delay;
                continue;
            };
            // Fresh config every tick: switching channels in config.toml
            // takes effect on the next check without a daemon restart.
            // Budget covers the full stable-channel resolve chain
            // (latest -> v-tag -> nightly-tag.txt -> concrete b-tag).
            let latest = tokio::time::timeout(
                std::time::Duration::from_secs(15),
                gh.channel_b_release(cfg.update_channel),
            )
            .await;
            let Ok(Ok(rel)) = latest else {
                let reason = match latest.as_ref() {
                    Ok(Err(e)) => format!("{e:#}"),
                    Err(_) => "timed out after 15s".to_string(),
                    Ok(Ok(_)) => unreachable!("guarded by the let-else"),
                };
                tracing::warn!(
                    target: "pallama::engine",
                    "engine check failed ({reason}) — retrying in 10 min"
                );
                delay = retry_delay;
                continue;
            };
            let newer = !same_build(&active.tag, &rel.tag_name);
            if newer {
                tracing::info!(
                    target: "pallama::engine",
                    "{} available: {} (active: {}, channel: {}) — run: {}",
                    channel_word(&active.tag, &rel.tag_name),
                    rel.tag_name,
                    active.tag,
                    cfg.update_channel,
                    engine_update_command(&active.tag, &active.asset)
                );
            }
            // Child-context device census: the serving child's own view of
            // the GPU world. Doctor compares these against the install-time
            // (manifest) names to flag enumeration drift. A failed census
            // writes null — drift detection simply waits for the next tick.
            let devices: Option<Vec<serde_json::Value>> =
                engine.enumerate_devices().await.ok().flatten().map(|ds| {
                    ds.iter()
                        .map(
                            |dev| serde_json::json!({"name": dev.name, "total_mib": dev.total_mib}),
                        )
                        .collect()
                });
            let marker = serde_json::json!({
                "checked_at": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_secs()),
                "latest": rel.tag_name,
                "active": active.tag,
                "channel": cfg.update_channel.to_string(),
                "update_available": newer,
                "devices": devices,
            });
            let _ = std::fs::write(dirs.run_dir().join("engine-check.json"), marker.to_string());
            delay = full_delay;
        }
    });
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
                println!(
                    "#{:<4} {:<24} scale {:<6} {}",
                    l.id, l.model_name, l.scale, l.path
                );
            }
        }
    }
    Ok(())
}

/// Search-table repo-id display: shrink the OWNER (leading ellipsis),
/// never the model name — the model tail is the discriminator users copy
/// for `pallama pull` (`…-heretic-i1-GGUF` vs `…-heretic-GGUF` are
/// different repos). Slash-less ids and oversized model names fall back
/// to a tail cut. Char-safe: never slices mid-UTF-8.
fn truncate_repo_id(id: &str, cap: usize) -> String {
    if id.chars().count() <= cap {
        return id.to_string();
    }
    if let Some((owner, model)) = id.split_once('/') {
        let model_len = model.chars().count();
        if model_len + 2 <= cap {
            let keep = cap - model_len - 2; // room for "…/"
            let head: String = owner.chars().take(keep).collect();
            return format!("{head}…/{model}");
        }
    }
    let head: String = id.chars().take(cap - 1).collect();
    format!("{head}…")
}

/// Downloads/likes display: round-half-up to one decimal (`2395` ->
/// `"2.4k"`, not the floored `"2.3k"`), with carry promotion at unit
/// boundaries (`999_950` -> `"1.0M"`). Integer-only — no float casts.
fn human_count(n: u64) -> String {
    if n >= 999_950 {
        let t = (n + 50_000) / 100_000; // tenths of a million, rounded
        if t / 10 >= 100 {
            format!("{}M", t / 10)
        } else {
            format!("{}.{}M", t / 10, t % 10)
        }
    } else if n >= 995 {
        let t = (n + 50) / 100; // tenths of a thousand, rounded
        format!("{}.{}k", t / 10, t % 10)
    } else {
        n.to_string()
    }
}

async fn search(query: &str) -> Result<()> {
    let token = std::env::var("HF_TOKEN").ok();
    let client = pallama_runtime::hf::HfClient::new(token)?;
    let results = client.search(query, 20).await?;
    if results.is_empty() {
        println!("no GGUF repos matched {query:?}");
        return Ok(());
    }
    // Column width adapts to the longest repo id (capped) so numbers never
    // drift out of alignment; oversize ids shrink the owner, keeping the
    // model name — the pull discriminator — fully visible.
    let cap = 64usize;
    let width = results
        .iter()
        .map(|r| r.id.len().min(cap))
        .max()
        .unwrap_or(0)
        .max("REPO".len());
    let human_ctx = |c: u64| {
        if c >= 1024 * 1024 {
            format!("{}M", c / (1024 * 1024))
        } else if c >= 1024 {
            format!("{}k", c / 1024)
        } else {
            c.to_string()
        }
    };
    println!(
        "{:<width$}  {:>10}  {:>6}  {:<10}  {:<7}  {:>5}  {:<22}",
        "REPO",
        "DOWNLOADS",
        "LIKES",
        "SIZE",
        "ARCH",
        "CTX",
        "QUANTS",
        width = width
    );
    for r in results {
        let id = truncate_repo_id(&r.id, cap);
        let (size, arch, ctx) = r.gguf.map_or_else(
            || ("-".to_string(), "?".to_string(), "-".to_string()),
            |g| {
                (
                    humansize(i64::try_from(g.total.unwrap_or(0)).unwrap_or(i64::MAX)),
                    g.architecture.unwrap_or_else(|| "?".to_string()),
                    g.context_length.map_or_else(|| "-".to_string(), &human_ctx),
                )
            },
        );
        let names =
            pallama_runtime::hf::quant_tokens(r.siblings.iter().map(|s| s.rfilename.as_str()));
        let quants = if names.is_empty() {
            "-".to_string()
        } else if names.len() <= 3 {
            names.join(",")
        } else {
            format!("{},+{}", names[..3].join(","), names.len() - 3)
        };
        println!(
            "{:<width$}  {:>10}  {:>6}  {:<10}  {:<7}  {:>5}  {:<22}",
            id,
            human_count(r.downloads.unwrap_or(0)),
            r.likes.unwrap_or(0),
            size,
            arch,
            ctx,
            quants,
            width = width
        );
    }
    println!("\n# pull one: pallama pull <REPO>[:quant]   (size = all quants in repo; QUANTS lists the choices)");
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
            .map_or(0, |m| {
                pallama_runtime::probe_hardware(Some(&m)).total_vram_mib()
            })
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
                .find(|l| key_before_eq(l).is_some_and(|k| k == key))
            {
                println!("{line}");
                Ok(())
            } else {
                Err(anyhow!("unknown config key: {key}"))
            }
        }
        ConfigCmd::Set { key, value } => {
            let path = dirs().config_file();
            // Fresh-box bootstrap: `config set` may be the first command
            // ever run — ensure the dirs exist and a config file is
            // present (Config::load writes a fresh default), instead of
            // crashing with a raw ENOENT on read below.
            let d = dirs();
            d.ensure().ok();
            if !path.exists() {
                Config::load(&d).map_err(|e| anyhow!("{e}"))?;
            }
            let raw = std::fs::read_to_string(&path)?;
            // Values that are not already TOML scalars (numbers, bools,
            // quoted strings, arrays) are written as double-quoted strings:
            // `config set child_transport tcp` must not produce
            // `child_transport = tcp` (invalid TOML). A one-line parse probe
            // decides; the full candidate is validated below either way.
            let scalar = toml::from_str::<toml::Table>(&format!("v = {value}\n")).is_ok();
            let stored = if scalar {
                value.clone()
            } else {
                format!("{value:?}")
            };
            let mut out: Vec<String> = Vec::new();
            let mut replaced = false;
            for line in raw.lines() {
                // F128: tolerant key match (compact `key="v"`, indented).
                if key_before_eq(line).is_some_and(|k| k == key) {
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
            Config::from_toml(&candidate).map_err(|e| anyhow!("rejected, file unchanged: {e}"))?;
            pallama_core::persist_config(&path, &candidate)?;
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
    let client = pallama_runtime::GhClient::with_base(&base, token).map_err(|e| anyhow!("{e}"))?;
    let channel = config()?.update_channel;
    let summary =
        pallama_runtime::upgrade::run(&client, &repo, version.as_deref(), channel, dry_run).await;
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
    fn unit__grouped_help__covers_every_subcommand_exactly_once() {
        let rendered = render_grouped_help();
        // Rendered command lines: two-space indent, lowercase name first
        // (Options lines start with '-'; headings and footer are not
        // indented).
        let mut listed: Vec<String> = rendered
            .lines()
            .filter(|l| l.starts_with("  ") && !l.trim_start().starts_with('-'))
            .map(|l| l.split_whitespace().next().unwrap().to_string())
            .collect();
        listed.sort();
        let mut clap_names: Vec<String> = {
            let mut c = Cli::command();
            c.build();
            c.get_subcommands()
                .map(|s| s.get_name().to_string())
                .collect()
        };
        clap_names.sort();
        assert_eq!(
            listed, clap_names,
            "rendered help and the clap enum disagree (missing from a HELP_GROUPS entry, \
             or a new command not grouped)"
        );
        // validate.py asserts the llama.cpp credit on --help; keep it here
        // so a renderer rewrite cannot silently drop it.
        assert!(rendered.contains("llama.cpp"));
    }

    #[test]
    fn unit__truncate_repo_id__shrinks_owner_keeps_model_tail() {
        // The incident rows: two different repos whose old tail-cut
        // rendering was byte-identical AND un-copy-pasteable for pull.
        let i1 = "mradermacher/Parable-Nanbeige4.2-3B-Claude-Fable-5-heretic-i1-GGUF";
        let plain = "mradermacher/Parable-Nanbeige4.2-3B-Claude-Fable-5-heretic-GGUF";
        let t = truncate_repo_id(i1, 64);
        assert!(t.ends_with("heretic-i1-GGUF"), "{t}");
        assert!(t.contains("…/"), "owner must be the part that shrinks: {t}");
        assert!(t.chars().count() <= 64);
        // Fits: untouched (this row previously truncated at 49 chars).
        assert_eq!(
            truncate_repo_id(plain, 64),
            "mradermacher/Parable-Nanbeige4.2-3B-Claude-Fable-5-heretic-GGUF"
        );
        assert_eq!(
            truncate_repo_id("owao/Nanbeige4.2-3B-GGUF", 64),
            "owao/Nanbeige4.2-3B-GGUF"
        );
        // Slash-less / oversized-model fallback: tail cut, never a panic.
        let no_slash = "a-very-long-repository-name-without-any-slash-at-all-0123456789";
        let t2 = truncate_repo_id(no_slash, 32);
        assert!(t2.chars().count() <= 32 && t2.ends_with('…'), "{t2}");
    }

    #[test]
    fn unit__human_count__rounds_half_up_with_carry() {
        assert_eq!(human_count(0), "0");
        assert_eq!(human_count(994), "994"); // below 0.995k: verbatim
        assert_eq!(human_count(995), "1.0k"); // 0.995 rounds up
        assert_eq!(human_count(2_395), "2.4k"); // was floored "2.3k"
        assert_eq!(human_count(142_502), "142.5k");
        assert_eq!(human_count(999_949), "999.9k"); // last k row
        assert_eq!(human_count(999_950), "1.0M"); // carry promotes unit
        assert_eq!(human_count(2_414_570), "2.4M");
    }

    #[test]
    fn unit__parse_modelfile__supported_subset() {
        let spec = parse_modelfile(
            "# comment\nFROM qwen3.5-9b\nPARAMETER num_ctx 32768\nADAPTER /tmp/a.gguf\n",
        )
        .unwrap();
        assert_eq!(spec.base, "qwen3.5-9b");
        assert_eq!(spec.ctx, Some(32768));
        assert_eq!(spec.loras, vec!["/tmp/a.gguf"]);
        assert!(spec.unsupported.is_empty());
    }

    #[test]
    fn unit__parse_modelfile__unsupported_keys_collected_not_dropped() {
        let spec = parse_modelfile(
            "FROM m\nPARAMETER temperature 0.7\nSYSTEM you are a pirate\nTEMPLATE {{x}}\n",
        )
        .unwrap();
        assert!(spec
            .unsupported
            .contains(&"PARAMETER temperature".to_string()));
        assert!(spec.unsupported.iter().any(|u| u.starts_with("SYSTEM")));
        assert!(spec.unsupported.iter().any(|u| u.starts_with("TEMPLATE")));
    }

    #[test]
    fn unit__parse_modelfile__missing_from_is_error() {
        assert!(parse_modelfile("PARAMETER num_ctx 8\n").is_err());
    }

    #[test]
    fn unit__currency_verdict__stale_marker_after_update_reports_current() {
        // Live incident shape: daemon surveyed while b10819 was active, the
        // user then updated to b10831 — the marker's own verdict must NOT
        // be trusted against the current store.
        let marker = serde_json::json!({
            "checked_at": 1_788_759_955_u64,
            "latest": "b10831",
            "active": "b10819",
            "update_available": true,
        });
        let c = currency_verdict(Some("b10831"), &marker, 1_788_760_000, "").unwrap();
        assert!(c.ok && !c.warn);
        assert_eq!(c.detail, "up to date (b10831)");
    }

    #[test]
    fn unit__currency_verdict__genuine_pending_update_warns_with_current_active() {
        let marker = serde_json::json!({
            "checked_at": 1_000_u64,
            "latest": "b10831",
            "active": "b10819",
            "update_available": true,
        });
        let c = currency_verdict(Some("b10819"), &marker, 2_000, "").unwrap();
        assert!(c.warn);
        assert_eq!(
            c.detail,
            "upgrade available: b10831 (active: b10819, channel: latest) — run: pallama engine update"
        );
    }

    #[test]
    fn unit__currency_verdict__lane_suffix_same_build_and_lane_aware_hint() {
        // Live incident: a source-built bNNNN-cuda engine IS the same
        // build as upstream bNNNN (lane suffix ignored), and the remedy
        // must name the lane that can actually refresh it — `engine
        // update` would install the Vulkan prebuilt instead.
        let same = serde_json::json!({
            "checked_at": 1_000_u64,
            "latest": "b10909",
            "active": "b10909-cuda",
            "update_available": false,
        });
        let c = currency_verdict(Some("b10909-cuda"), &same, 2_000, "built-cuda").unwrap();
        assert!(c.ok && !c.warn, "{}", c.detail);

        let pending = serde_json::json!({
            "checked_at": 1_000_u64,
            "latest": "b10912",
            "active": "b10909-cuda",
            "update_available": true,
        });
        let c = currency_verdict(Some("b10909-cuda"), &pending, 2_000, "built-cuda").unwrap();
        assert!(c.warn, "{}", c.detail);
        assert!(
            c.detail.contains("run: pallama engine build cuda"),
            "{}",
            c.detail
        );

        // Overlay prebuilts (ubuntu-cuda-*) refresh through engine update.
        let c = currency_verdict(Some("b10909-cuda"), &pending, 2_000, "ubuntu-cuda-x64").unwrap();
        assert!(
            c.detail.contains("run: pallama engine update"),
            "{}",
            c.detail
        );

        // Plain prebuilt tags keep the standard hint.
        let plain = serde_json::json!({
            "checked_at": 1_000_u64,
            "latest": "b10912",
            "active": "b10909",
            "update_available": true,
        });
        let c = currency_verdict(Some("b10909"), &plain, 2_000, "ubuntu-vulkan-x64").unwrap();
        assert!(
            c.detail.contains("run: pallama engine update"),
            "{}",
            c.detail
        );
    }

    #[test]
    fn unit__currency_verdict__downgrade_after_check_still_warns() {
        // Marker written while b10819 was active AND current; the user then
        // rolled back to b10817 — marker claims up-to-date, store disagrees.
        let marker = serde_json::json!({
            "checked_at": 1_000_u64,
            "latest": "b10819",
            "active": "b10819",
            "update_available": false,
        });
        let c = currency_verdict(Some("b10817"), &marker, 1_100, "").unwrap();
        assert!(c.warn);
        assert!(c.detail.contains("active: b10817"), "{}", c.detail);
    }

    #[test]
    fn unit__currency_verdict__survey_older_than_48h_warns_stale() {
        let marker = serde_json::json!({
            "checked_at": 1_000_u64,
            "latest": "b10831",
            "active": "b10831",
            "update_available": false,
        });
        let c = currency_verdict(Some("b10831"), &marker, 1_000 + 2 * 86_400 + 1, "").unwrap();
        assert!(c.warn);
        assert_eq!(
            c.detail,
            "no successful upstream check in >48h (offline? engine_check_secs=0?)"
        );
    }

    #[test]
    fn unit__currency_verdict__local_engine_untracked() {
        let marker = serde_json::json!({"checked_at": 1_u64, "latest": "b10831"});
        let c = currency_verdict(Some("local"), &marker, 2, "").unwrap();
        assert!(c.ok && !c.warn);
        assert_eq!(c.detail, "local build — upstream currency not tracked");
    }

    #[test]
    fn unit__currency_verdict__no_engine_or_malformed_marker_skips_row() {
        let marker = serde_json::json!({"checked_at": 1_u64, "active": "b1"});
        assert!(currency_verdict(None, &marker, 2, "").is_none());
        assert!(currency_verdict(Some("b1"), &marker, 2, "").is_none()); // no latest
    }

    #[test]
    fn unit__ver_triple__strict_three_numeric_parts() {
        assert_eq!(ver_triple("0.3.0"), Some((0, 3, 0)));
        assert_eq!(ver_triple("v1.2.3"), Some((1, 2, 3)));
        assert_eq!(ver_triple(" v2.0.1 "), Some((2, 0, 1)));
        assert_eq!(ver_triple("1.2"), None);
        assert_eq!(ver_triple("1.2.3.4"), None);
        assert_eq!(ver_triple("1.2.x"), None);
        assert_eq!(ver_triple("0.3.0-beta"), None);
        assert_eq!(ver_triple("nightly"), None);
    }

    #[test]
    fn unit__version_currency_verdict__update_available_warns_with_run_hint() {
        let c = version_currency_verdict("app currency", "0.3.0", "v0.4.0", "pallama upgrade");
        assert!(c.warn);
        assert_eq!(
            c.detail,
            "update available: v0.4.0 (running: 0.3.0) — run: pallama upgrade"
        );
        let w = version_currency_verdict(
            "whisper currency",
            "v1.7.5",
            "v1.7.6",
            "pallama whisper --install",
        );
        assert!(w.warn);
        assert_eq!(
            w.detail,
            "update available: v1.7.6 (running: v1.7.5) — run: pallama whisper --install"
        );
    }

    #[test]
    fn unit__version_currency_verdict__equal_versions_ok_with_v_prefix() {
        let c = version_currency_verdict("app currency", "0.3.0", "v0.3.0", "pallama upgrade");
        assert!(c.ok && !c.warn);
        assert_eq!(c.detail, "up to date (0.3.0)");
    }

    #[test]
    fn unit__version_currency_verdict__running_ahead_is_ok() {
        let c = version_currency_verdict("app currency", "0.5.1", "v0.4.0", "pallama upgrade");
        assert!(c.ok && !c.warn);
        assert!(c.detail.contains("ahead"), "{}", c.detail);
    }

    #[test]
    fn unit__version_currency_verdict__mixed_tag_shapes_fall_back_to_inequality() {
        // whisper.cpp reality: installed v-tag vs latest b-tag.
        let c = version_currency_verdict(
            "whisper currency",
            "v1.7.5",
            "b4938",
            "pallama whisper --install",
        );
        assert!(c.warn);
        assert_eq!(
            c.detail,
            "update available: b4938 (running: v1.7.5) — run: pallama whisper --install"
        );
        // Same unparseable tag on both sides = up to date, not a warn.
        let c = version_currency_verdict("app currency", "nightly", "nightly", "pallama upgrade");
        assert!(c.ok && !c.warn);
        assert_eq!(c.detail, "up to date (nightly)");
    }

    #[test]
    fn unit__exposure_verdict__loopback_authless_ok() {
        let c = exposure_verdict("127.0.0.1", 11434, 0);
        assert!(c.ok && !c.warn);
        assert_eq!(c.detail, "loopback bind, no auth needed");
    }

    #[test]
    fn unit__exposure_verdict__wildcard_authless_warns() {
        let c = exposure_verdict("0.0.0.0", 11434, 0);
        assert!(c.warn, "{}", c.detail);
        assert!(c.detail.contains("all interfaces"), "{}", c.detail);
        assert!(c.detail.to_lowercase().contains("no auth"), "{}", c.detail);
        // Empty host means wildcard too.
        let c = exposure_verdict("", 11434, 0);
        assert!(c.warn, "{}", c.detail);
    }

    #[test]
    fn unit__exposure_verdict__lan_with_keys_ok() {
        let c = exposure_verdict("192.168.1.10", 11434, 2);
        assert!(c.ok && !c.warn, "{}", c.detail);
        assert!(c.detail.contains("2 key(s)"), "{}", c.detail);
        assert!(c.detail.contains("TLS"), "{}", c.detail);
    }

    #[test]
    fn unit__device_drift__flags_only_missing() {
        let f = vec!["Vulkan0".to_string(), "Vulkan1".to_string()];
        let l = vec!["Vulkan0".to_string()];
        assert_eq!(device_drift(&f, &l), vec!["Vulkan1".to_string()]);
        // child sees a superset: nothing flagged
        assert!(device_drift(&f, &f).is_empty());
        assert!(device_drift(&f, &["Vulkan0".into(), "Vulkan1".into(), "CUDA0".into()]).is_empty());
        // both empty (CPU-only): healthy
        assert!(device_drift(&[], &[]).is_empty());
    }

    #[test]
    fn unit__doctor_next_steps__daemon_models_ready() {
        let port_ok = vec![
            Check::ok("port", "127.0.0.1:11435 — pallama daemon already answering"),
            Check::ok("models", "2 pulled, all parse"),
        ];
        // daemon up + models present -> the ready line
        assert_eq!(
            doctor_next_steps(&port_ok),
            vec!["chat: pallama run <model>"]
        );
        // daemon down -> serve hint first, model hint second
        let down = vec![
            Check::fail("port", "nothing answering"),
            Check::ok("models", "0 pulled, all parse"),
        ];
        assert_eq!(
            doctor_next_steps(&down)[0],
            "start the daemon: pallama serve (or: systemctl start pallama)"
        );
        assert_eq!(
            doctor_next_steps(&down)[1],
            "pull a model: pallama pull <name> (find one: pallama search qwen3)"
        );
        // "10 pulled" must NOT match the zero-models wording
        let ten = vec![
            Check::ok("port", "pallama daemon already answering"),
            Check::ok("models", "10 pulled, all parse"),
        ];
        assert_eq!(doctor_next_steps(&ten), vec!["chat: pallama run <model>"]);
    }

    #[test]
    fn unit__census_names__absent_null_array_empty() {
        // Legacy marker (pre-census daemon): unknown, never drift.
        assert!(census_names(&serde_json::json!({})).is_none());
        // Probe failed at census time (writer emits null): unknown.
        assert!(census_names(&serde_json::json!({"devices": null})).is_none());
        // Census ran: names extracted.
        let m = serde_json::json!({
            "devices": [{"name": "Vulkan0", "total_mib": 8188}],
            "checked_at": 1_788_886_202,
        });
        assert_eq!(census_names(&m), Some(vec!["Vulkan0".to_string()]));
        // Census ran and found nothing: known-empty, not unknown.
        assert_eq!(
            census_names(&serde_json::json!({"devices": []})),
            Some(Vec::new())
        );
    }

    #[test]
    fn unit__marker_channel_matches__absent_match_mismatch() {
        use pallama_core::config::UpdateChannel;
        // Legacy marker (pre-channel daemon) defaults to "latest".
        assert!(marker_channel_matches(
            &serde_json::json!({}),
            UpdateChannel::Latest
        ));
        assert!(!marker_channel_matches(
            &serde_json::json!({}),
            UpdateChannel::Stable
        ));
        // Explicit channel key.
        let m = serde_json::json!({"channel": "stable"});
        assert!(marker_channel_matches(&m, UpdateChannel::Stable));
        assert!(!marker_channel_matches(&m, UpdateChannel::Latest));
    }

    #[test]
    fn unit__service_verdict__shapes() {
        // Managed and healthy.
        let c = service_verdict(Some(true), Some(true), "systemd").unwrap();
        assert!(c.ok && !c.warn);
        assert!(c.detail.contains("active + enabled"), "{}", c.detail);
        // Enabled but dead: warn.
        let c = service_verdict(Some(true), Some(false), "systemd").unwrap();
        assert!(c.warn);
        assert!(c.detail.contains("not running"), "{}", c.detail);
        // Running unenabled: warn.
        let c = service_verdict(Some(false), Some(true), "systemd").unwrap();
        assert!(c.warn);
        assert!(c.detail.contains("not enabled"), "{}", c.detail);
        // Manual lane: informational ok, not a warning.
        let c = service_verdict(Some(false), Some(false), "systemd").unwrap();
        assert!(c.ok && !c.warn, "{}", c.detail);
        // Inconclusive probes: no row at all.
        assert!(service_verdict(None, Some(true), "systemd").is_none());
        assert!(service_verdict(Some(true), None, "systemd").is_none());
    }

    #[test]
    fn unit__doctor_store__absent_fresh_ok() {
        let tmp = tempfile::tempdir().unwrap();
        let d = PallamaDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        let checks = doctor_store(&d);
        assert_eq!(checks.len(), 1);
        assert!(checks[0].ok && !checks[0].warn, "{}", checks[0].detail);
        assert!(
            checks[0].detail.contains("fresh install"),
            "{}",
            checks[0].detail
        );
    }

    #[test]
    fn unit__doctor_store__healthy_quick_check_ok() {
        let tmp = tempfile::tempdir().unwrap();
        let d = PallamaDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        std::fs::create_dir_all(&d.data_dir).unwrap();
        drop(Store::open(&d).unwrap()); // creates + migrates pallama.db
        let checks = doctor_store(&d);
        assert_eq!(checks.len(), 1);
        assert!(checks[0].ok && !checks[0].warn, "{}", checks[0].detail);
        assert!(
            checks[0].detail.contains("quick_check passed"),
            "{}",
            checks[0].detail
        );
    }

    #[test]
    fn unit__doctor_store__corrupt_db_warns() {
        let tmp = tempfile::tempdir().unwrap();
        let d = PallamaDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        std::fs::create_dir_all(&d.data_dir).unwrap();
        std::fs::write(d.db_file(), b"this is not a sqlite file").unwrap();
        let checks = doctor_store(&d);
        assert_eq!(checks.len(), 1);
        assert!(checks[0].warn, "{}", checks[0].detail);
        assert!(
            checks[0].detail.contains("regenerate"),
            "{}",
            checks[0].detail
        );
    }

    #[cfg(unix)]
    #[test]
    fn unit__engine_smoke_check__executes_warns_and_misses() {
        let tmp = tempfile::tempdir().unwrap();
        // Healthy: a script that prints a version banner.
        let good = tmp.path().join("server-good");
        std::fs::write(&good, "#!/bin/sh\necho 'llama-server b10857'\n").unwrap();
        make_executable(&good);
        let c = engine_smoke_check(good.to_str().unwrap());
        assert!(c.ok && !c.warn, "{}", c.detail);
        assert!(c.detail.contains("executes"), "{}", c.detail);
        assert!(c.detail.contains("b10857"), "{}", c.detail);
        // Broken: exits non-zero.
        let bad = tmp.path().join("server-bad");
        std::fs::write(&bad, "#!/bin/sh\nexit 3\n").unwrap();
        make_executable(&bad);
        let c = engine_smoke_check(bad.to_str().unwrap());
        assert!(c.warn, "{}", c.detail);
        assert!(c.detail.contains("exited"), "{}", c.detail);
        // Missing path entirely.
        let c = engine_smoke_check(tmp.path().join("nope").to_str().unwrap());
        assert!(c.warn, "{}", c.detail);
        assert!(c.detail.contains("cannot execute"), "{}", c.detail);
    }

    #[cfg(unix)]
    fn make_executable(path: &std::path::Path) {
        use std::os::unix::fs::PermissionsExt;
        let mut perm = std::fs::metadata(path).unwrap().permissions();
        perm.set_mode(0o755);
        std::fs::set_permissions(path, perm).unwrap();
    }

    #[test]
    fn unit__engine_update_command__lane_aware() {
        // Source-built lane: tag suffix + built-* asset -> rebuild.
        assert_eq!(
            engine_update_command("b10809-cuda", "built-cuda"),
            "pallama engine build cuda"
        );
        assert_eq!(
            engine_update_command("b4242-cpu", "built-cpu"),
            "pallama engine build cpu"
        );
        // Overlay prebuilt lane: same tag suffix, ubuntu-* asset ->
        // normal update lane (re-probes the overlay by build number).
        assert_eq!(
            engine_update_command("b10809-cuda", "ubuntu-cuda-12.8-x64"),
            "pallama engine update"
        );
        // Upstream prebuilts and local builds.
        assert_eq!(
            engine_update_command("b10809", "ubuntu-vulkan-x64"),
            "pallama engine update"
        );
        assert_eq!(
            engine_update_command("local", "built-cuda"),
            "pallama engine update"
        );
    }

    #[test]
    fn unit__restart_lanes__privilege_free_before_sudo() {
        // linux: user unit restarts without elevation and must come
        // BEFORE the passwordless-sudo system lane; neither unit ->
        // no lane (manual daemons get the printed hint).
        let both = restart_lanes("linux", "/home/u", true, true);
        assert_eq!(both[0][0..4], ["systemctl", "--user", "restart", "pallama"]);
        assert_eq!(both[1][0..2], ["sudo", "-n"]);
        assert!(restart_lanes("linux", "/home/u", false, false).is_empty());
        let sys_only = restart_lanes("linux", "/home/u", true, false);
        assert_eq!(sys_only.len(), 1);
        assert_eq!(sys_only[0][0], "sudo");
        // macos: launch agent lane is uid-scoped; system daemon via sudo -n
        let mac = restart_lanes("macos", "/Users/u", true, false);
        assert_eq!(mac.len(), 2);
        assert!(mac[0][1] == "kickstart" && mac[0][2] == "-k");
        assert!(mac[1][0] == "sudo");
        // unknown platform: nothing to drive
        assert!(restart_lanes("windows", "C:/u", true, true).is_empty());
    }

    #[test]
    fn unit__vtag_semver__parses_or_none() {
        assert_eq!(vtag_semver("v0.9.3"), Some((0, 9, 3)));
        assert_eq!(vtag_semver("v0.10.0"), Some((0, 10, 0)));
        assert_eq!(vtag_semver("v1.8"), Some((1, 8, 0)));
        assert_eq!(vtag_semver("v2.0.0-rc.1"), Some((2, 0, 0)));
        assert_eq!(vtag_semver("b10857"), None);
        assert_eq!(vtag_semver("vx.y.z"), None);
    }

    #[test]
    fn unit__cuda_opportunity_rows__nudge_only_for_non_cuda_llamacpp() {
        let full = Toolchain {
            git: Some(std::path::PathBuf::from("/usr/bin/git")),
            cmake: Some(std::path::PathBuf::from("/usr/bin/cmake")),
            cxx: Some(std::path::PathBuf::from("/usr/bin/g++")),
            nvcc: Some(std::path::PathBuf::from("/usr/bin/nvcc")),
            nvidia_smi: Some(std::path::PathBuf::from("/usr/bin/nvidia-smi")),
        };
        // NVIDIA + Vulkan-prebuilt active: nudge + ready toolchain.
        let rows = cuda_opportunity_rows(
            Some("b10809"),
            Some(EngineKind::LlamaCpp),
            Some((13, 0)),
            Some((8, 9)),
            &full,
        );
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].name, "engine backend");
        assert!(rows[0].detail.contains("driver CUDA 13.0, sm 89"));
        assert!(rows[0].detail.contains("pallama engine build cuda"));
        assert!(rows[0].detail.contains("pallama engine install"));
        assert_eq!(rows[1].name, "cuda toolchain");
        assert!(rows[1].detail.contains("ready"));
        // Already on the CUDA build: no nudge, toolchain row stays.
        let rows = cuda_opportunity_rows(
            Some("b10809-cuda"),
            Some(EngineKind::LlamaCpp),
            Some((13, 0)),
            Some((8, 9)),
            &full,
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].name, "cuda toolchain");
        // mistral.rs active: asset was driver-picked at install, no nudge.
        let rows = cuda_opportunity_rows(
            Some("v0.9.3"),
            Some(EngineKind::MistralRs),
            Some((13, 0)),
            Some((8, 9)),
            &full,
        );
        assert_eq!(rows.len(), 1);
        // No NVIDIA driver: nothing at all.
        assert!(cuda_opportunity_rows(
            Some("b10809"),
            Some(EngineKind::LlamaCpp),
            None,
            None,
            &full
        )
        .is_empty());
    }

    #[test]
    fn unit__cuda_opportunity_rows__missing_toolchain_names_bins() {
        let empty = Toolchain::default();
        let rows = cuda_opportunity_rows(
            Some("b10809"),
            Some(EngineKind::LlamaCpp),
            Some((13, 0)),
            None,
            &empty,
        );
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1].name, "cuda toolchain");
        // Missing compute cap must not garble the nudge line.
        assert!(rows[0].detail.contains("driver CUDA 13.0)"));
        assert!(rows[1].detail.contains("git, cmake, nvcc"));
        assert!(rows[1].detail.contains("pallama engine install"));
    }

    #[test]
    fn unit__channel_word__btag_direction() {
        assert_eq!(channel_word("b10857", "b10865"), "upgrade");
        assert_eq!(channel_word("b10857", "b10780"), "downgrade");
        assert_eq!(channel_word("b10857", "b10857"), "update");
        // v-tags compare numerically per component: the lexicographic bug
        // ("0.10.0" < "0.9.3") is what this pins.
        assert_eq!(channel_word("v0.9.3", "v0.10.0"), "upgrade");
        assert_eq!(channel_word("v1.9.0", "v1.8.0"), "downgrade");
        assert_eq!(channel_word("v1.8.0", "v1.8.0"), "update");
        // Mixed shapes have no ordering: neutral word.
        assert_eq!(channel_word("b10857", "v1.36.0"), "update");
        assert_eq!(channel_word("v1.8.0", "b10865"), "update");
    }

    #[test]
    fn unit__currency_verdict__direction_and_channel() {
        let now = 1_000_000u64;
        // Marker older than active, stable channel: a downgrade hint with
        // the channel named — switching is the point, not a regression.
        let marker = serde_json::json!({
            "checked_at": now - 60,
            "latest": "b10780",
            "active": "b10857",
            "channel": "stable",
            "update_available": true,
        });
        let c = currency_verdict(Some("b10857"), &marker, now, "").unwrap();
        assert!(c.warn, "{}", c.detail);
        assert!(c.detail.contains("downgrade available"), "{}", c.detail);
        assert!(c.detail.contains("channel: stable"), "{}", c.detail);
        // Old marker without the channel key: defaults to latest, still
        // carries a direction word.
        let legacy = serde_json::json!({
            "checked_at": now - 60,
            "latest": "b10865",
            "active": "b10857",
            "update_available": true,
        });
        let c = currency_verdict(Some("b10857"), &legacy, now, "").unwrap();
        assert!(c.warn, "{}", c.detail);
        assert!(c.detail.contains("upgrade available"), "{}", c.detail);
        assert!(c.detail.contains("channel: latest"), "{}", c.detail);
        // Up to date: ok row regardless of channel.
        let fresh = serde_json::json!({
            "checked_at": now - 60,
            "latest": "b10857",
            "active": "b10857",
            "channel": "latest",
            "update_available": false,
        });
        let c = currency_verdict(Some("b10857"), &fresh, now, "").unwrap();
        assert!(c.ok && !c.warn, "{}", c.detail);
    }
}
