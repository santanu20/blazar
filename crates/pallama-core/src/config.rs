use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::dirs::PallamaDirs;
use crate::error::{CoreError, CoreResult};

/// Tunables exposed in config.toml. One file, one surface: every Pallama
/// knob lives here or in a per-model overlay; the documented `PALLAMA_*`
/// env vars override file values. Secrets (`HF_TOKEN` / `GH_TOKEN`) are env-only
/// and never persisted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
// Flat one-knob-per-line is the product decision here ("every knob in one
// documented config.toml"); grouping the switches into sub-tables would
// churn every user's file for a lint.
#[allow(clippy::struct_excessive_bools)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub default_ctx: u32,
    pub idle_sleep_secs: u64,
    pub idle_timeout_secs: u64,
    /// 0 = auto: derived from hardware probe at runtime.
    pub max_loaded_models: u32,
    /// "tcp" (curl-debuggable children) | "unix" (socket files).
    pub child_transport: String,
    /// "auto" or explicit engine asset suffix (e.g. "ubuntu-vulkan-x64").
    pub engine_asset: String,
    /// "" = newest b-tag; else pin like "b10816".
    pub engine_pin: String,
    /// "off" | "auto" (auto = adopt spec decode when a draft pair is pulled).
    pub spec: String,
    /// Min chunk size for KV-shift prefix reuse; 0 disables.
    pub cache_reuse: u32,
    /// API keys (virtual keys): empty = no auth (loopback default).
    /// Each entry scopes a bearer key to models + rate/token budgets
    /// (`pallama keys add`). The legacy flat `api_keys` list is gone.
    #[serde(default)]
    pub keys: Vec<ApiKey>,
    /// Comma-separated RPC servers, e.g. "box1:50052,box2:50052".
    pub rpc_servers: String,
    /// Child prompt-cache budget in MiB; 0 = unlimited.
    pub cache_ram_mb: i64,
    /// CPU affinity range for child threads, "lo-hi" (e.g. "0-15" for the
    /// P-core threads on an 8P+8E hybrid; discover with `lscpu -e`).
    /// Empty = OS default scheduling.
    pub cpu_range: String,
    /// Busy-poll level 0-100 while waiting for child work (0 = off).
    /// >0 trades idle CPU for lower per-token latency (TTFT).
    pub poll: u32,
    /// Child reasoning output format: "none" | "deepseek" | "deepseek-legacy".
    /// Empty = upstream auto (detect from template).
    pub reasoning_format: String,
    /// Server slots (`-np`). 1 = full-speed single client (default);
    /// larger = concurrent clients sharing ctx. 0 = auto.
    #[serde(default = "default_slots")]
    pub slots: u32,
    /// KV cache quantization: "" = auto ladder (`q8_0` when KV+weights near
    /// VRAM, `q4_0` when still tight), or an explicit type: f32, f16, bf16,
    /// `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`. Quantized V requires flash
    /// attention (always emitted when the engine supports it).
    #[serde(default)]
    pub cache_type: String,
    /// KV buffer layout: None = engine default (unified when slots are
    /// auto); Some(true) = `--kv-unified` (one shared buffer; K-shift
    /// prefix reuse across sequences — the cheap radix); Some(false) =
    /// `--no-kv-unified` (isolated per-slot buffers). Manifest-gated.
    #[serde(default)]
    pub kv_unified: Option<bool>,
    /// `--kv-unified-per-slot N`: per-slot token budget inside the
    /// unified buffer (0 = off). Manifest-gated.
    #[serde(default)]
    pub kv_unified_per_slot: u32,
    /// `--swa-full`: keep FULL KV for sliding-window layers (quality at
    /// memory cost; SWA models only). Manifest-gated.
    #[serde(default)]
    pub swa_full: bool,
    /// `--ctx-checkpoints N`: rolling context checkpoints for SWA models
    /// (0 = off). Manifest-gated.
    #[serde(default)]
    pub ctx_checkpoints: u32,
    /// `--no-kv-offload`: keep all KV on GPU — fail the load when it
    /// does not fit instead of spilling to CPU (latency determinism).
    #[serde(default)]
    pub no_kv_offload: bool,
    /// `--load-mode`: "" (engine default) | "mmap" | "mlock" | "direct-io".
    #[serde(default)]
    pub load_mode: String,
    /// Refuse model loads when `MemAvailable` is under half the model +
    /// 512 MiB (swap-death prevention; the box would thrash for minutes).
    /// Set false to restore load-anyway behavior.
    #[serde(default = "default_true")]
    pub spawn_mem_guard: bool,
    /// Session bank: checkpoint slot KV to `_auto` on evict and restore
    /// it on the next spawn — agent conversations resume warm after a
    /// capacity cycle instead of re-prefilling from scratch. Banks over
    /// 512 MiB are dropped (not a bank, a hoard).
    #[serde(default = "default_true")]
    pub session_bank: bool,
    /// Single-flight: identical NON-STREAM chat requests coalesce at the
    /// child-call boundary (the twin awaits the leader, then rides the
    /// warm prefix). 5s bound — long generations never serialize their
    /// duplicates indefinitely.
    #[serde(default = "default_true")]
    pub singleflight: bool,
    /// Refuse chat requests whose prompt cannot fit the effective
    /// context (pre-hoc; the alternative is the engine's silent
    /// truncation). Exact /tokenize when the cheap estimate crosses 90%.
    #[serde(default = "default_true")]
    pub prompt_preflight: bool,
    /// Persist the n-gram speculative cache across restarts
    /// (`--lookup-cache-dynamic`, one file per model).
    #[serde(default = "default_true")]
    pub spec_cache: bool,
    /// `YaRN` `RoPE` context-extension factor: 0 = off; e.g. 2.0 doubles the
    /// usable context beyond the trained window at some quality cost.
    #[serde(default)]
    pub ctx_extend: f64,
    /// Keep N `MoE` expert tensors on CPU (`--n-cpu-moe`): finer-grained
    /// cousin of the boolean MoE-offload heuristic. 0 = off.
    #[serde(default)]
    pub cpu_moe_n: i32,
    /// Upstream `--override-tensor` entries ("PATTERN=DEVICE", e.g.
    /// `".ffn_.*_exps.=CPU"`); applied per model via overlay.
    #[serde(default)]
    pub override_tensor: Vec<String>,
    /// Enable the child's agent mode (`--agent`): built-in tools + MCP CORS
    /// proxy. WARNING: tools include `exec_shell_command` — only enable on
    /// machines where that is acceptable.
    #[serde(default)]
    pub agent: bool,
    /// Enable session checkpoints (`--slot-save-path`): `pallama session`
    /// save/restore of slot KV state. Harmless when unsupported by the
    /// engine (emission is manifest-gated with a named warning).
    #[serde(default = "default_true")]
    pub sessions: bool,
    /// Router mode: ONE llama-server child serves every pulled model
    /// (upstream `--models-preset` INI generated from the store; the
    /// engine autoloads on request and LRU-evicts at `router_max_models`).
    /// Default false = one child per model.
    #[serde(default)]
    pub router: bool,
    /// Router mode: max concurrently loaded models (`--models-max`).
    /// 0 = upstream default (4).
    #[serde(default)]
    pub router_max_models: u32,
    /// Explicit engine device selection (`--device <name>` per entry), for
    /// multi-GPU boxes where auto-pick lands on the wrong card. Names come
    /// from `pallama doctor`'s device list. Empty = engine auto.
    #[serde(default)]
    pub devices: Vec<String>,
    /// How often the daemon checks GitHub for a newer llama.cpp b-release
    /// (0 = off). Never auto-installs: the result surfaces in `doctor`,
    /// `ps`, and the log — `pallama engine update` stays a human action.
    #[serde(default = "default_engine_check_secs")]
    pub engine_check_secs: u64,
    /// Slot prompt-similarity threshold (`--slot-prompt-similarity`):
    /// how closely a request's prompt must match a slot's cached prompt to
    /// reuse it (prefix affinity at slots > 1). 0 = emit nothing (upstream
    /// default 0.1). Range (0.0..=1.0].
    #[serde(default)]
    pub slot_prompt_similarity: f64,
    /// Sentinel: warn-only response-semantics observation (truncation,
    /// tool-call validity, schema violations, empty replies, stalls).
    /// Never alters request or response bytes; `false` disables every hook.
    #[serde(default = "default_true")]
    pub sentinel: bool,
    /// Sentinel stalled-stream threshold in seconds. 0 disables stall
    /// detection specifically (other detections stay on).
    #[serde(default = "default_stall_secs")]
    pub sentinel_stall_secs: u64,
    /// Sentinel enforce: hard violations (invalid tool args, unknown tool
    /// names, schema violations) become 422s on NON-STREAMING chat
    /// requests (streaming bytes are already on the wire — warn-only
    /// there). Per-request `X-Pallama-Enforce: 1|0` overrides.
    #[serde(default)]
    pub sentinel_enforce: bool,
    /// TLS: PEM certificate chain path. Empty = plain HTTP. Must be set
    /// together with `tls_key` (both or neither — validated).
    #[serde(default)]
    pub tls_cert: String,
    /// TLS: PEM private key path matching `tls_cert`.
    #[serde(default)]
    pub tls_key: String,
    /// CORS: allowed origin list, e.g. a single `https://chat.example`
    /// entry. Empty = no CORS headers (the previous behavior). The string
    /// `*` = any origin. Never affects non-browser clients.
    #[serde(default)]
    pub cors_origins: Vec<String>,
    /// OTLP trace export: collector base URL (e.g.
    /// "<http://127.0.0.1:4318>"). Empty = off (default). One span per
    /// gateway request, batched, bounded — observability never becomes
    /// backpressure.
    #[serde(default)]
    pub otlp_endpoint: String,
    /// OTLP service.name tag (default "pallama").
    #[serde(default)]
    pub otlp_service: String,
    /// Remote OpenAI-compatible endpoints (`[[remotes]]`); models named
    /// `<remote>:<model>` route there instead of loading locally.
    #[serde(default)]
    pub remotes: Vec<Remote>,
    /// Extra env for engine children + probes (e.g. `GGML_BACKEND_PATH` for
    /// a local CUDA build).
    #[serde(default)]
    pub engine_env: BTreeMap<String, String>,
    #[serde(default)]
    pub model_overrides: BTreeMap<String, ModelOverride>,

    // ---- spec-draft placement (draft model CPU/VRAM placement; only
    // meaningful with spec = "auto" or a draft model configured). All
    // default = engine defaults (unset = nothing emitted).
    /// Pin draft-model threads to a "lo-hi" CPU set (P/E hybrid boxes).
    #[serde(default)]
    pub spec_draft_cpu_range: String,
    /// Strict CPU placement for the draft model (upstream: 0|1).
    #[serde(default)]
    pub spec_draft_cpu_strict: bool,
    /// Dedicated device for the draft model (multi-GPU boxes).
    #[serde(default)]
    pub spec_draft_device: String,
    /// Draft-model VRAM layers: exact number, "auto" or "all".
    #[serde(default)]
    pub spec_draft_ngl: String,
    /// Draft-model thread count (0 = engine default).
    #[serde(default)]
    pub spec_draft_threads: u32,
    /// Minimum draft probability below which speculation is not verified
    /// (greedy accept). None = engine default (0.0).
    #[serde(default)]
    pub spec_draft_p_min: Option<f64>,
    /// Probability of splitting speculation at a draft token.
    /// None = engine default (0.10).
    #[serde(default)]
    pub spec_draft_p_split: Option<f64>,
    /// Draft-model poll level 0..=100 (None = follows `poll`).
    #[serde(default)]
    pub spec_draft_poll: Option<u32>,
    /// Draft-model process priority -1..=3 (0 = unset).
    #[serde(default)]
    pub spec_draft_prio: i32,
    /// Draft-model batch-thread priority -1..=3 (0 = unset).
    #[serde(default)]
    pub spec_draft_prio_batch: i32,
    /// Poll for draft batch work (None = follows `spec_draft_poll`).
    #[serde(default)]
    pub spec_draft_poll_batch: Option<bool>,
    /// Strict CPU placement for draft batch threads.
    #[serde(default)]
    pub spec_draft_cpu_strict_batch: bool,
    /// Draft-model batch thread count (0 = engine default).
    #[serde(default)]
    pub spec_draft_threads_batch: u32,
    /// Draft-model KV cache type K ("" = engine default).
    #[serde(default)]
    pub spec_draft_type_k: String,
    /// Draft-model KV cache type V ("" = engine default).
    #[serde(default)]
    pub spec_draft_type_v: String,
    /// Draft-model `--override-tensor` entries.
    #[serde(default)]
    pub spec_draft_override_tensor: Vec<String>,
    /// Draft-model `MoE` experts on CPU (count; 0 = off).
    #[serde(default)]
    pub spec_draft_n_cpu_moe: i32,
    /// Draft-model boolean MoE-CPU offload.
    #[serde(default)]
    pub spec_draft_cpu_moe: bool,
    /// Offload draft sampling to the backend (upstream default true).
    #[serde(default = "default_true")]
    pub spec_draft_backend_sampling: bool,
    /// Draft-cache adaptive decay (0 = engine default).
    #[serde(default)]
    pub adaptive_decay: i32,
    /// Draft-cache adaptive target acceptance rate (0 = engine default).
    #[serde(default)]
    pub adaptive_target: f64,

    // ---- n-gram speculation tuning (spec = "ngram"). Upstream b10833
    // REMOVED the generic --spec-ngram-* forms; these emit the typed
    // --spec-ngram-simple-* flags. 0 = engine default (16/8/2 upstream).
    /// n-gram lookup table size (tokens of context hashed).
    #[serde(default)]
    pub ngram_size_m: u32,
    /// n-gram length.
    #[serde(default)]
    pub ngram_size_n: u32,
    /// Minimum table hits before a draft is trusted.
    #[serde(default)]
    pub ngram_min_hits: u32,

    // ---- reasoning control (server-side thinking budget; works on
    // reasoning models, cuts wasted thinking tokens on agent traffic).
    /// Token budget for thinking: -1 = unrestricted (upstream default),
    /// 0 = immediate end, N>0 = budget.
    #[serde(default = "default_reasoning_budget")]
    pub reasoning_budget: i64,
    /// Message injected when the thinking budget is exhausted.
    #[serde(default)]
    pub reasoning_budget_message: String,
    /// Reasoning effort level passed to the chat template:
    /// "" (keep template default) | minimal|low|medium|high|xhigh|max.
    #[serde(default)]
    pub reasoning_effort: String,
    /// Keep reasoning content in responses. None = engine default.
    #[serde(default)]
    pub reasoning_preserve: Option<bool>,

    // ---- vision / multimodal tuning (models with an mmproj).
    /// Max tokens per image (dynamic-resolution vision models). 0 = model.
    #[serde(default)]
    pub image_max_tokens: u32,
    /// Min tokens per image. 0 = model.
    #[serde(default)]
    pub image_min_tokens: u32,
    /// Max image tokens per encode batch. 0 = engine default.
    #[serde(default)]
    pub mtmd_batch_max_tokens: u32,
    /// GPU-offload the multimodal projector (upstream default true).
    #[serde(default = "default_true")]
    pub mmproj_offload: bool,
    /// Auto-use a discovered projector (upstream default true).
    #[serde(default = "default_true")]
    pub mmproj_auto: bool,
    /// Dedicated device for the projector ("" = engine default).
    #[serde(default)]
    pub mmproj_device: String,
    /// Embeddings L2 normalization: 0 = off (engine), 1 = on.
    #[serde(default)]
    pub embd_normalize: u32,

    // ---- YaRN fine-tuning (independent of ctx_extend; each 0/empty =
    // engine default).
    /// Original trained context (0 = from GGUF).
    #[serde(default)]
    pub yarn_orig_ctx: u32,
    /// Extrapolation mix factor (>= 0.0 to set; upstream default -1).
    #[serde(default = "default_yarn_ext_factor")]
    pub yarn_ext_factor: f64,
    /// Attention scaling factor (0 = engine default).
    #[serde(default)]
    pub yarn_attn_factor: f64,
    /// `YaRN` beta fast (0 = engine default).
    #[serde(default)]
    pub yarn_beta_fast: f64,
    /// `YaRN` beta slow (0 = engine default).
    #[serde(default)]
    pub yarn_beta_slow: f64,

    // ---- scheduling extras (server-level).
    /// Strict CPU placement for the main model (upstream: 0|1).
    #[serde(default)]
    pub cpu_strict: bool,
    /// Process/thread priority -1..=3 (low..realtime). 0 = unset.
    #[serde(default)]
    pub prio: i32,
    /// Batch-thread priority -1..=3. 0 = unset.
    #[serde(default)]
    pub prio_batch: i32,
    /// Poll while waiting for batch work. None = follows `poll`.
    #[serde(default)]
    pub poll_batch: Option<bool>,
    /// HTTP-server thread count (0 = engine default).
    #[serde(default)]
    pub threads_http: u32,

    // ---- engine behavior toggles (upstream defaults preserved; the
    // knob exists to disable).
    /// Warmup run at startup (upstream default true).
    #[serde(default = "default_true")]
    pub warmup: bool,
    /// Weight repacking for CPU/GPU layout (upstream default true).
    #[serde(default = "default_true")]
    pub repack: bool,
    /// Save idle slots to the prompt cache on new tasks (upstream
    /// default true; requires --cache-ram, which Pallama sets). Disable
    /// to keep idle slots from consuming prompt-cache budget.
    #[serde(default = "default_true")]
    pub cache_idle_slots: bool,
    /// Read-only static prompt-cache file for --lookup-cache-static
    /// (e.g. a system prompt pre-baked with `llama-lookup-save/merge`).
    /// The engine loads it but never updates it.
    #[serde(default)]
    pub lookup_cache_static: Option<String>,
    /// Writable prompt-cache file for --lookup-cache-dynamic; the
    /// engine updates it as generation runs. Both must exist on disk.
    #[serde(default)]
    pub lookup_cache_dynamic: Option<String>,
    /// Predictive pre-loading (LC1): track which model tends to follow
    /// which and pre-spawn the likely next model while the current one
    /// idles, so the switch is warm instead of a cold start. Off by
    /// default — it deliberately spends RAM/VRAM ahead of demand.
    #[serde(default)]
    pub predictive_preload: bool,
    /// Adaptive slots (LC4): when a single-slot model sustains
    /// concurrent load for ~60s, adopt slots+1 (in-memory, capped at 4;
    /// `tune --slots` remains the permanent path). Off by default.
    #[serde(default)]
    pub adaptive_slots: bool,
    /// Bypass host buffer for extra VRAM (upstream default false).
    #[serde(default)]
    pub no_host: bool,
    /// Offload host tensor ops to device (None = engine default true).
    #[serde(default)]
    pub op_offload: Option<bool>,
    /// Tokens kept from the initial prompt on ctx shift (0 = upstream
    /// default, -1 = all).
    #[serde(default)]
    pub keep_tokens: i32,

    // ---- power-user escapes.
    /// `KEY=TYPE:VALUE` GGUF metadata overrides (repeatable). Repairs
    /// broken quant metadata without re-pulling.
    #[serde(default)]
    pub override_kv: Vec<String>,
    /// Control-vector file paths (repeatable).
    #[serde(default)]
    pub control_vectors: Vec<String>,
    /// `FNAME:SCALE` control vectors (repeatable).
    #[serde(default)]
    pub control_vectors_scaled: Vec<String>,
    /// Layer range for control vectors (e.g. "0-10").
    #[serde(default)]
    pub control_vector_layer_range: String,
    /// Named `--override-tensor` preset: "" | "moe-cpu-offload".
    #[serde(default)]
    pub tensor_preset: String,
    /// Redact obvious PII (emails, bearer secrets, IPv4 addresses) from
    /// `why`/`watch` output. Opt-in; access logs never carry bodies.
    #[serde(default)]
    pub pii_scrub: bool,
    /// Video-in lane: directory containing the ffmpeg binary (models
    /// with video input; "" = engine default discovery).
    #[serde(default)]
    pub video_ffmpeg_dir: String,
    /// Video input sampling FPS (0 = engine default).
    #[serde(default)]
    pub video_fps: f64,
    /// Video timestamp annotation interval seconds (0 = off).
    #[serde(default)]
    pub video_timestamp_interval: f64,
    /// NUMA policy: "" | distribute | isolate (multi-socket boxes).
    #[serde(default)]
    pub numa: String,
    /// Validate tensor data at load (debug; slow).
    #[serde(default)]
    pub check_tensors: bool,
    /// Enable context shift on infinite generation (upstream b10833
    /// default: DISABLED — emits `--context-shift` only when true).
    #[serde(default)]
    pub context_shift: bool,
    /// Default sampler chain, semicolon-separated as upstream takes it
    /// ("" = engine default; e.g. `"top_k;top_p;typical"`).
    #[serde(default)]
    pub samplers: String,
}

fn default_reasoning_budget() -> i64 {
    -1
}
fn default_yarn_ext_factor() -> f64 {
    -1.0
}

/// Named `--override-tensor` presets. Keep the vocabulary tiny and
/// documented; unrecognized names fail validation (never silently no-op).
fn tensor_preset_entries(name: &str) -> Option<&'static [String]> {
    static MOE_CPU_OFFLOAD: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    match name {
        // MoE experts to CPU, attention on GPU: the classic >VRAM MoE
        // split (upstream docs' own example pattern).
        "moe-cpu-offload" => Some(MOE_CPU_OFFLOAD.get_or_init(|| vec!["exps=CPU".to_string()])),
        _ => None,
    }
}

const REASONING_EFFORT_LEVELS: &[&str] = &["minimal", "low", "medium", "high", "xhigh", "max"];

/// Per-model config overlay. Unknown keys are a hard error at parse time —
/// a typo'd knob must never be silently ignored.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelOverride {
    pub ctx: Option<u32>,
    /// Per-model parallel slots (`-np`); overrides the global `slots`
    /// for this model only. 0 keeps upstream auto.
    pub slots: Option<u32>,
    pub spec: Option<String>,
    pub loras: Option<Vec<String>>,
    /// Extra llama-server args, validated against the engine capability
    /// manifest at profile-compile time (unknown flag = error).
    pub extra_args: Option<Vec<String>>,
    /// Per-model KV cache type ("" or None = inherit the global ladder).
    pub cache_type: Option<String>,
    /// Per-model unified-KV override (None = inherit).
    #[serde(default)]
    pub kv_unified: Option<bool>,
    /// Per-model `YaRN` context-extension factor (None = inherit global).
    pub ctx_extend: Option<f64>,
    /// Per-model `MoE` expert CPU-offload count (None = inherit global).
    pub cpu_moe_n: Option<i32>,
    /// Per-model `--override-tensor` entries; replaces (not merges) the
    /// global list for this model.
    pub override_tensor: Option<Vec<String>>,
    /// Per-model GPU devices; replaces the global `devices` list.
    #[serde(default)]
    pub devices: Option<Vec<String>>,
    /// Per-model warmup override (None = inherit global).
    #[serde(default)]
    pub warmup: Option<bool>,
    /// Per-model thinking token budget (None = inherit global).
    #[serde(default)]
    pub reasoning_budget: Option<i64>,
    /// Per-model reasoning effort (None = inherit global).
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// Per-model replica count (1 = single instance — the default and
    /// the exact pre-replica behavior). >1 enables prefix-affinity
    /// routing across identical children of the same model.
    #[serde(default)]
    pub replicas: Option<u32>,
    /// Pin this model's instances against capacity eviction (A13): the
    /// supervisor's victim filter skips pinned instances entirely. Use
    /// for always-hot models that must not pay a cold reload; capacity
    /// pressure falls on unpinned models instead.
    #[serde(default)]
    pub pin: Option<bool>,
}

/// One external OpenAI-compatible server (another pallama, vLLM, MLX
/// server, llamactl — anything speaking `/v1/*`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Remote {
    /// Routing prefix: `name:model` requests land here.
    pub name: String,
    /// Base URL, e.g. "<http://10.0.0.4:8000>".
    pub url: String,
    /// Optional bearer key for the remote (empty = none).
    #[serde(default)]
    pub key: String,
}

/// One virtual API key: bearer identity + optional model scope and
/// rate/token budgets. 0-valued budgets are unlimited; an empty `models`
/// list means every model (admin semantics — such keys also manage the
/// key list itself via `/api/keys`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApiKey {
    /// Human label (`pallama keys add NAME`).
    pub name: String,
    /// The bearer secret itself (`plm_...`, shown once at creation).
    pub key: String,
    /// Model allowlist; empty = all models.
    #[serde(default)]
    pub models: Vec<String>,
    /// Requests per minute cap; 0 = unlimited.
    #[serde(default)]
    pub rpm: u32,
    /// Tokens per minute cap (prompt+completion, chat-family routes);
    /// 0 = unlimited.
    #[serde(default)]
    pub tpm: u64,
    /// Total tokens per UTC day; 0 = unlimited.
    #[serde(default)]
    pub daily_tokens: u64,
    /// Simultaneous in-flight requests cap; 0 = unlimited. The slot is
    /// leased at auth time and held until the response body finishes
    /// streaming (or the connection drops) — a held SSE stream IS the
    /// resource being bounded.
    #[serde(default)]
    pub max_concurrent: u32,
}

impl Default for Config {
    // A flat literal of every knob's default: one line each beats
    // splitting across helper fns that hide the table.
    #[allow(clippy::too_many_lines)]
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 11434,
            default_ctx: 16384,
            idle_sleep_secs: 300,
            cpu_range: String::new(),
            poll: 0,
            reasoning_format: String::new(),
            idle_timeout_secs: 1800,
            max_loaded_models: 0,
            child_transport: "tcp".to_string(),
            engine_asset: "auto".to_string(),
            engine_pin: String::new(),
            router_max_models: 0,
            devices: Vec::new(),
            engine_check_secs: default_engine_check_secs(),
            spec: "off".to_string(),
            cache_reuse: 256,
            keys: Vec::new(),
            rpc_servers: String::new(),
            cache_ram_mb: 8192,
            slots: 1,
            slot_prompt_similarity: 0.0,
            sentinel: true,
            sentinel_stall_secs: 30,
            sentinel_enforce: false,
            tls_cert: String::new(),
            tls_key: String::new(),
            cors_origins: Vec::new(),
            otlp_endpoint: String::new(),
            otlp_service: String::new(),
            remotes: Vec::new(),
            cache_type: String::new(),
            kv_unified: None,
            kv_unified_per_slot: 0,
            swa_full: false,
            ctx_checkpoints: 0,
            no_kv_offload: false,
            load_mode: String::new(),
            spawn_mem_guard: true,
            session_bank: true,
            singleflight: true,
            prompt_preflight: true,
            spec_cache: true,
            ctx_extend: 0.0,
            cpu_moe_n: 0,
            override_tensor: Vec::new(),
            agent: false,
            model_overrides: BTreeMap::new(),
            engine_env: BTreeMap::new(),
            sessions: true,
            router: false,
            spec_draft_cpu_range: String::new(),
            spec_draft_cpu_strict: false,
            spec_draft_device: String::new(),
            spec_draft_ngl: String::new(),
            spec_draft_threads: 0,
            spec_draft_p_min: None,
            spec_draft_p_split: None,
            spec_draft_poll: None,
            spec_draft_prio: 0,
            spec_draft_prio_batch: 0,
            spec_draft_poll_batch: None,
            spec_draft_cpu_strict_batch: false,
            spec_draft_threads_batch: 0,
            spec_draft_type_k: String::new(),
            spec_draft_type_v: String::new(),
            spec_draft_override_tensor: Vec::new(),
            spec_draft_n_cpu_moe: 0,
            spec_draft_cpu_moe: false,
            spec_draft_backend_sampling: true,
            adaptive_decay: 0,
            adaptive_target: 0.0,
            ngram_size_m: 0,
            ngram_size_n: 0,
            ngram_min_hits: 0,
            reasoning_budget: default_reasoning_budget(),
            reasoning_budget_message: String::new(),
            reasoning_effort: String::new(),
            reasoning_preserve: None,
            image_max_tokens: 0,
            image_min_tokens: 0,
            mtmd_batch_max_tokens: 0,
            mmproj_offload: true,
            mmproj_auto: true,
            mmproj_device: String::new(),
            embd_normalize: 0,
            yarn_orig_ctx: 0,
            yarn_ext_factor: default_yarn_ext_factor(),
            yarn_attn_factor: 0.0,
            yarn_beta_fast: 0.0,
            yarn_beta_slow: 0.0,
            cpu_strict: false,
            prio: 0,
            prio_batch: 0,
            poll_batch: None,
            threads_http: 0,
            warmup: true,
            repack: true,
            cache_idle_slots: true,
            lookup_cache_static: None,
            lookup_cache_dynamic: None,
            predictive_preload: false,
            adaptive_slots: false,
            no_host: false,
            op_offload: None,
            keep_tokens: 0,
            override_kv: Vec::new(),
            control_vectors: Vec::new(),
            control_vectors_scaled: Vec::new(),
            control_vector_layer_range: String::new(),
            tensor_preset: String::new(),
            pii_scrub: false,
            video_ffmpeg_dir: String::new(),
            video_fps: 0.0,
            video_timestamp_interval: 0.0,
            numa: String::new(),
            check_tensors: false,
            context_shift: false,
            samplers: String::new(),
        }
    }
}

impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            toml::to_string_pretty(self).map_err(|_| fmt::Error)?
        )
    }
}

impl Config {
    /// Load config from `<config_dir>/config.toml`, creating it with defaults
    /// if absent. Then apply `PALLAMA_*` env overrides.
    pub fn load(dirs: &PallamaDirs) -> CoreResult<Self> {
        let path = dirs.config_file();
        let cfg = if path.exists() {
            let raw = std::fs::read_to_string(&path)?;
            Self::from_toml(&raw)?
        } else {
            let cfg = Self::default();
            std::fs::create_dir_all(&dirs.config_dir)?;
            std::fs::write(&path, cfg.to_toml()?)?;
            cfg
        };
        cfg.with_env_overrides()
    }

    pub fn from_toml(raw: &str) -> CoreResult<Self> {
        // One-click upgrade: the removed flat `api_keys` list migrates
        // IN MEMORY (no file write — side-effect-free load), loudly.
        // `pallama migrate` persists the canonical form; installers and
        // self-upgrade call it. Every entry keeps its secret as an
        // unscoped admin key: same bearer power, nothing silently lost.
        let probe: toml::Table =
            toml::from_str(raw).map_err(|e| CoreError::Config(format!("parse: {e}")))?;
        let migrating = probe.contains_key("api_keys");
        let raw = if migrating {
            let legacy = probe
                .get("api_keys")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            let mut table = probe.clone();
            table.remove("api_keys");
            let mut cfg: Config = toml::from_str(&toml::to_string(&table).unwrap_or_default())
                .map_err(|e| CoreError::Config(format!("parse: {e}")))?;
            cfg.keys = legacy
                .iter()
                .enumerate()
                .filter_map(|(i, v)| {
                    v.as_str().map(str::to_string).map(|key| ApiKey {
                        name: format!("migrated-{i}"),
                        key,
                        ..ApiKey::default()
                    })
                })
                .collect();
            tracing::warn!(
                target: "pallama::config",
                "legacy api_keys migrated to [[keys]] ({} key(s), in memory) — persist with `pallama migrate`",
                cfg.keys.len()
            );
            return cfg.validate().map(|()| cfg);
        } else {
            raw.to_string()
        };
        let cfg: Config =
            toml::from_str(&raw).map_err(|e| CoreError::Config(format!("parse: {e}")))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn to_toml(&self) -> CoreResult<String> {
        toml::to_string_pretty(self).map_err(|e| CoreError::Config(format!("serialize: {e}")))
    }

    /// Overlay for a model: exact name match only. Overlays are validated at
    /// parse time (`deny_unknown_fields`), so no re-validation here.
    #[must_use]
    pub fn overlay_for(&self, model: &str) -> ModelOverride {
        self.model_overrides.get(model).cloned().unwrap_or_default()
    }

    /// Effective ctx for a model: overlay wins over global default.
    #[must_use]
    pub fn effective_ctx(&self, model: &str) -> u32 {
        let o = self.overlay_for(model);
        o.ctx.unwrap_or(self.default_ctx)
    }

    /// Effective spec mode for a model: overlay wins over global default.
    #[must_use]
    pub fn effective_spec(&self, model: &str) -> &str {
        match self
            .model_overrides
            .get(model)
            .and_then(|o| o.spec.as_deref())
        {
            Some(s) => s,
            None => self.spec.as_str(),
        }
    }

    /// True when the RAW text still carries legacy `api_keys` (needs
    /// `pallama migrate` to persist the canonical form).
    #[must_use]
    pub fn raw_has_legacy_keys(raw: &str) -> bool {
        toml::from_str::<toml::Table>(raw).is_ok_and(|t| t.contains_key("api_keys"))
    }

    /// Resolve a presented bearer secret to its key entry (None = unknown).
    #[must_use]
    pub fn key_for(&self, presented: &str) -> Option<&ApiKey> {
        self.keys.iter().find(|k| k.key == presented)
    }

    fn validate_keys(&self) -> CoreResult<()> {
        for (i, k) in self.keys.iter().enumerate() {
            if k.name.trim().is_empty() {
                return Err(CoreError::Config(format!(
                    "keys[{i}]: name must not be empty"
                )));
            }
            if k.key.trim().is_empty() {
                return Err(CoreError::Config(format!(
                    "keys[{}]: key secret must not be empty (generate with `pallama keys add {}`)",
                    i, k.name
                )));
            }
        }
        let mut names: Vec<&str> = self.keys.iter().map(|k| k.name.as_str()).collect();
        let n = names.len();
        names.sort_unstable();
        names.dedup();
        if names.len() != n {
            return Err(CoreError::Config(
                "keys: duplicate name — each key needs a unique name".to_string(),
            ));
        }
        let mut secrets: Vec<&str> = self.keys.iter().map(|k| k.key.as_str()).collect();
        let s = secrets.len();
        secrets.sort_unstable();
        secrets.dedup();
        if secrets.len() != s {
            return Err(CoreError::Config(
                "keys: duplicate key secret — regenerate one of them".to_string(),
            ));
        }
        Ok(())
    }

    /// Effective `kv_unified`: overlay beats global; None = engine default.
    #[must_use]
    pub fn effective_kv_unified(&self, model: &str) -> Option<bool> {
        self.model_overrides
            .get(model)
            .and_then(|o| o.kv_unified)
            .or(self.kv_unified)
    }

    /// Effective KV cache type: "" = auto ladder; explicit overlay beats
    /// explicit global beats auto.
    #[must_use]
    pub fn effective_cache_type(&self, model: &str) -> &str {
        if let Some(s) = self
            .model_overrides
            .get(model)
            .and_then(|o| o.cache_type.as_deref())
            .filter(|s| !s.is_empty())
        {
            return s;
        }
        self.cache_type.as_str()
    }

    /// Effective `YaRN` context-extension factor; overlay wins over global.
    #[must_use]
    pub fn effective_ctx_extend(&self, model: &str) -> f64 {
        let o = self.overlay_for(model);
        o.ctx_extend
            .filter(|v| *v != 0.0)
            .unwrap_or(self.ctx_extend)
    }

    /// Effective `MoE` CPU-offload expert count; overlay wins over global.
    #[must_use]
    pub fn effective_cpu_moe_n(&self, model: &str) -> i32 {
        let o = self.overlay_for(model);
        o.cpu_moe_n.filter(|v| *v > 0).unwrap_or(self.cpu_moe_n)
    }

    /// Effective `--override-tensor` entries; overlay list replaces the
    /// global list when present. A named `tensor_preset` (global or
    /// overlay) expands to its entries and is replaced by any explicit
    /// list.
    #[must_use]
    pub fn effective_override_tensor(&self, model: &str) -> &[String] {
        if let Some(list) = self
            .model_overrides
            .get(model)
            .and_then(|o| o.override_tensor.as_ref())
        {
            return list.as_slice();
        }
        if !self.tensor_preset.is_empty() {
            if let Some(entries) = tensor_preset_entries(&self.tensor_preset) {
                return entries;
            }
        }
        self.override_tensor.as_slice()
    }
    /// Effective device list; overlay replaces global (matches
    /// `override_tensor` semantics).
    #[must_use]
    pub fn effective_devices(&self, model: &str) -> &[String] {
        if let Some(list) = self
            .model_overrides
            .get(model)
            .and_then(|o| o.devices.as_ref())
        {
            return list.as_slice();
        }
        self.devices.as_slice()
    }

    /// Effective warmup: overlay beats global.
    #[must_use]
    pub fn effective_warmup(&self, model: &str) -> bool {
        self.model_overrides
            .get(model)
            .and_then(|o| o.warmup)
            .unwrap_or(self.warmup)
    }

    /// Effective thinking budget: overlay beats global.
    #[must_use]
    pub fn effective_reasoning_budget(&self, model: &str) -> i64 {
        self.model_overrides
            .get(model)
            .and_then(|o| o.reasoning_budget)
            .unwrap_or(self.reasoning_budget)
    }

    /// Effective reasoning effort: overlay beats global.
    #[must_use]
    pub fn effective_reasoning_effort(&self, model: &str) -> &str {
        self.model_overrides
            .get(model)
            .and_then(|o| o.reasoning_effort.as_deref())
            .unwrap_or(self.reasoning_effort.as_str())
    }

    /// Cross-field sanity. Violations are config errors, not warnings:
    /// fail fast rather than run with contradictory knobs.
    #[allow(clippy::too_many_lines)] // flat one-check-per-knob by design
    pub fn validate(&self) -> CoreResult<()> {
        if self.default_ctx == 0 {
            return Err(CoreError::Config("default_ctx must be > 0".into()));
        }
        self.validate_keys()?;
        if !LOAD_MODES.contains(&self.load_mode.as_str()) {
            return Err(CoreError::Config(format!(
                "load_mode must be one of mmap|mlock|direct-io (or empty), got {:?}",
                self.load_mode
            )));
        }
        match (self.tls_cert.is_empty(), self.tls_key.is_empty()) {
            (false, true) => {
                return Err(CoreError::Config(
                    "tls_cert set without tls_key — provide the matching PEM key or clear both"
                        .into(),
                ));
            }
            (true, false) => {
                return Err(CoreError::Config(
                    "tls_key set without tls_cert — provide the PEM chain or clear both".into(),
                ));
            }
            _ => {}
        }
        if self.idle_timeout_secs < self.idle_sleep_secs {
            return Err(CoreError::Config(format!(
                "idle_timeout_secs ({}) must be >= idle_sleep_secs ({}) — eviction must not fire before sleep",
                self.idle_timeout_secs, self.idle_sleep_secs
            )));
        }
        match self.child_transport.as_str() {
            "tcp" | "unix" => {}
            other => {
                return Err(CoreError::Config(format!(
                    "child_transport must be \"tcp\" or \"unix\", got {other:?}"
                )))
            }
        }
        match self.spec.as_str() {
            "off" | "auto" | "ngram" => {}
            other => {
                return Err(CoreError::Config(format!(
                    "spec must be \"off\", \"auto\" or \"ngram\", got {other:?}"
                )))
            }
        }
        if !self.cpu_range.is_empty() && !valid_cpu_range(&self.cpu_range) {
            return Err(CoreError::Config(format!(
                "cpu_range must be \"lo-hi\" with lo <= hi (e.g. \"0-15\"), got {:?}",
                self.cpu_range
            )));
        }
        if self.poll > 100 {
            return Err(CoreError::Config(format!(
                "poll must be 0..=100, got {}",
                self.poll
            )));
        }
        match self.reasoning_format.as_str() {
            "" | "none" | "deepseek" | "deepseek-legacy" => {}
            other => {
                return Err(CoreError::Config(format!(
                    "reasoning_format must be \"none\", \"deepseek\" or \"deepseek-legacy\", got {other:?}"
                )))
            }
        }
        if self.host.trim().is_empty() {
            return Err(CoreError::Config("host must not be empty".into()));
        }
        self.validate_new_knobs()?;
        self.validate_wire_knobs()?;
        for (name, o) in &self.model_overrides {
            if let Some(spec) = &o.spec {
                if spec != "off" && spec != "auto" && spec != "ngram" {
                    return Err(CoreError::Config(format!(
                        "model_overrides.{name}.spec must be \"off\", \"auto\" or \"ngram\", got {spec:?}"
                    )));
                }
            }
            if let Some(ctx) = o.ctx {
                if ctx == 0 {
                    return Err(CoreError::Config(format!(
                        "model_overrides.{name}.ctx must be > 0"
                    )));
                }
            }
            if let Some(ct) = &o.cache_type {
                if !valid_cache_type(ct) {
                    return Err(CoreError::Config(format!(
                        "model_overrides.{name}.cache_type must be one of {} (or empty), got {ct:?}",
                        CACHE_TYPES.join(", ")
                    )));
                }
            }
            if let Some(ce) = o.ctx_extend {
                if ce != 0.0 && !(1.0 < ce && ce <= 32.0) {
                    return Err(CoreError::Config(format!(
                        "model_overrides.{name}.ctx_extend must be 0 (off) or within (1.0..=32.0], got {ce}"
                    )));
                }
            }
            if let Some(n) = o.cpu_moe_n {
                if n < 0 {
                    return Err(CoreError::Config(format!(
                        "model_overrides.{name}.cpu_moe_n must be >= 0, got {n}"
                    )));
                }
            }
            if let Some(ots) = &o.override_tensor {
                for ot in ots {
                    if !valid_override_tensor(ot) {
                        return Err(CoreError::Config(format!(
                            "model_overrides.{name}.override_tensor entries must be PATTERN=DEVICE, got {ot:?}"
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Validation for the capacity/steering knobs added in the
    /// beat-ollama wave: `cache_type` vocabulary, `ctx_extend` range,
    /// expert counts, override-tensor shapes. Kept separate from `validate`
    /// to stay under the line budget with the base checks.
    fn validate_new_knobs(&self) -> CoreResult<()> {
        if self.sentinel
            && self.sentinel_stall_secs != 0
            && !(5..=600).contains(&self.sentinel_stall_secs)
        {
            return Err(CoreError::Config(format!(
                "sentinel_stall_secs must be 0 (off) or 5..=600, got {}",
                self.sentinel_stall_secs
            )));
        }
        if !valid_cache_type(&self.cache_type) {
            return Err(CoreError::Config(format!(
                "cache_type must be one of {} (or empty for auto), got {:?}",
                CACHE_TYPES.join(", "),
                self.cache_type
            )));
        }
        if self.devices.iter().any(|d| d.trim().is_empty()) {
            return Err(CoreError::Config(
                "devices entries must be non-empty device names (see `pallama doctor` for the list)"
                    .to_string(),
            ));
        }
        // 1.0 exactly is a no-op (yarn scale 1 = no extension): reject it
        // so a meaningless value can't masquerade as configured behavior.
        if self.ctx_extend != 0.0 && !(1.0 < self.ctx_extend && self.ctx_extend <= 32.0) {
            return Err(CoreError::Config(format!(
                "ctx_extend must be 0 (off) or within (1.0..=32.0], got {}",
                self.ctx_extend
            )));
        }
        if self.cpu_moe_n < 0 {
            return Err(CoreError::Config(format!(
                "cpu_moe_n must be >= 0, got {}",
                self.cpu_moe_n
            )));
        }
        for ot in &self.override_tensor {
            if !valid_override_tensor(ot) {
                return Err(CoreError::Config(format!(
                    "override_tensor entries must be PATTERN=DEVICE (e.g. \".ffn_.*_exps.=CPU\"), got {ot:?}"
                )));
            }
        }
        if !(0.0..=1.0).contains(&self.slot_prompt_similarity) {
            return Err(CoreError::Config(format!(
                "slot_prompt_similarity must be within 0.0..=1.0 (0 = upstream default), got {}",
                self.slot_prompt_similarity
            )));
        }
        Ok(())
    }

    /// Validation for the wire-everything knob wave: enums, ranges and
    /// cross-field sanity for spec-draft placement, ngram tuning,
    /// reasoning, vision, `YaRN`, scheduling and power-user escapes.
    /// One flat check per knob, fail-fast on contradiction.
    #[allow(clippy::too_many_lines)] // flat one-check-per-knob by design
    fn validate_wire_knobs(&self) -> CoreResult<()> {
        for (field, path) in [
            ("lookup_cache_static", &self.lookup_cache_static),
            ("lookup_cache_dynamic", &self.lookup_cache_dynamic),
        ] {
            if let Some(p) = path {
                if !std::path::Path::new(p).is_file() {
                    return Err(CoreError::Config(format!(
                        "{field} must point at an existing cache file, got {p:?}"
                    )));
                }
            }
        }
        if !self.spec_draft_cpu_range.is_empty() && !valid_cpu_range(&self.spec_draft_cpu_range) {
            return Err(CoreError::Config(format!(
                "spec_draft_cpu_range must be \"lo-hi\" with lo <= hi, got {:?}",
                self.spec_draft_cpu_range
            )));
        }
        if !self.spec_draft_ngl.is_empty()
            && self.spec_draft_ngl != "auto"
            && self.spec_draft_ngl != "all"
            && !self.spec_draft_ngl.chars().all(|c| c.is_ascii_digit())
        {
            return Err(CoreError::Config(format!(
                "spec_draft_ngl must be an exact layer count, \"auto\" or \"all\", got {:?}",
                self.spec_draft_ngl
            )));
        }
        if let Some(p) = self.spec_draft_p_min {
            if !(0.0..=1.0).contains(&p) {
                return Err(CoreError::Config(format!(
                    "spec_draft_p_min must be within 0.0..=1.0, got {p}"
                )));
            }
        }
        if let Some(p) = self.spec_draft_p_split {
            if !(0.0..=1.0).contains(&p) {
                return Err(CoreError::Config(format!(
                    "spec_draft_p_split must be within 0.0..=1.0, got {p}"
                )));
            }
        }
        if let Some(p) = self.spec_draft_poll {
            if p > 100 {
                return Err(CoreError::Config(format!(
                    "spec_draft_poll must be 0..=100, got {p}"
                )));
            }
        }
        if !(-1..=3).contains(&self.spec_draft_prio) {
            return Err(CoreError::Config(format!(
                "spec_draft_prio must be -1..=3 (low..realtime), got {}",
                self.spec_draft_prio
            )));
        }
        if !(-1..=3).contains(&self.spec_draft_prio_batch) {
            return Err(CoreError::Config(format!(
                "spec_draft_prio_batch must be -1..=3, got {}",
                self.spec_draft_prio_batch
            )));
        }
        for (label, t) in [
            ("spec_draft_type_k", &self.spec_draft_type_k),
            ("spec_draft_type_v", &self.spec_draft_type_v),
        ] {
            if !t.is_empty() && !valid_cache_type(t) {
                return Err(CoreError::Config(format!(
                    "{label} must be one of {} (or empty), got {t:?}",
                    CACHE_TYPES.join(", ")
                )));
            }
        }
        for ot in &self.spec_draft_override_tensor {
            if !valid_override_tensor(ot) {
                return Err(CoreError::Config(format!(
                    "spec_draft_override_tensor entries must be PATTERN=DEVICE, got {ot:?}"
                )));
            }
        }
        if !self.numa.is_empty() && !matches!(self.numa.as_str(), "distribute" | "isolate") {
            return Err(CoreError::Config(format!(
                "numa must be \"distribute\" or \"isolate\" (or empty), got {:?}",
                self.numa
            )));
        }
        if !self.samplers.is_empty()
            && self.samplers.split(',').any(|s| {
                !s.trim()
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_')
            })
        {
            return Err(CoreError::Config(format!(
                "samplers must be comma-separated sampler names (e.g. \"top_k,top_p,typical,min_p\"), got {:?}",
                self.samplers
            )));
        }
        if self.adaptive_target != 0.0 && !(0.0..=1.0).contains(&self.adaptive_target) {
            return Err(CoreError::Config(format!(
                "adaptive_target must be within 0.0..=1.0 (0 = off), got {}",
                self.adaptive_target
            )));
        }
        let effort = self.reasoning_effort.trim();
        if !effort.is_empty() && !REASONING_EFFORT_LEVELS.contains(&effort) {
            return Err(CoreError::Config(format!(
                "reasoning_effort must be one of {} (or empty), got {:?}",
                REASONING_EFFORT_LEVELS.join("|"),
                self.reasoning_effort
            )));
        }
        if self.image_max_tokens > 0
            && self.image_min_tokens > 0
            && self.image_max_tokens < self.image_min_tokens
        {
            return Err(CoreError::Config(format!(
                "image_max_tokens ({}) must be >= image_min_tokens ({})",
                self.image_max_tokens, self.image_min_tokens
            )));
        }
        // -1.0 exactly = engine default sentinel; any other negative is
        // invalid. Margin guards the float compare.
        if (self.yarn_ext_factor - default_yarn_ext_factor()).abs() > f64::EPSILON
            && self.yarn_ext_factor.is_sign_negative()
        {
            return Err(CoreError::Config(format!(
                "yarn_ext_factor must be >= 0.0 (or -1 = engine default), got {}",
                self.yarn_ext_factor
            )));
        }
        if self.yarn_attn_factor < 0.0 || self.yarn_beta_fast < 0.0 || self.yarn_beta_slow < 0.0 {
            return Err(CoreError::Config(
                "yarn_attn_factor/yarn_beta_fast/yarn_beta_slow must be >= 0.0 (0 = engine default)"
                    .into(),
            ));
        }
        if !(-1..=3).contains(&self.prio) {
            return Err(CoreError::Config(format!(
                "prio must be -1..=3 (low..realtime), got {}",
                self.prio
            )));
        }
        if !(-1..=3).contains(&self.prio_batch) {
            return Err(CoreError::Config(format!(
                "prio_batch must be -1..=3, got {}",
                self.prio_batch
            )));
        }
        for kv in &self.override_kv {
            // KEY=TYPE:VALUE with TYPE in int|float|bool|str (upstream spec).
            let shape = kv.contains('=') && kv.contains(':');
            let type_ok = ["int:", "float:", "bool:", "str:"]
                .iter()
                .any(|t| kv.contains(t));
            if !shape || !type_ok {
                return Err(CoreError::Config(format!(
                    "override_kv entries must be KEY=TYPE:VALUE with TYPE int|float|bool|str (e.g. \"tokenizer.ggml.add_bos_token=bool:false\"), got {kv:?}"
                )));
            }
        }
        for cv in self
            .control_vectors_scaled
            .iter()
            .chain(&self.control_vectors)
        {
            if cv.trim().is_empty() {
                return Err(CoreError::Config(
                    "control_vectors/control_vectors_scaled entries must be non-empty paths".into(),
                ));
            }
        }
        if !self.control_vector_layer_range.is_empty()
            && !valid_cpu_range(&self.control_vector_layer_range)
        {
            return Err(CoreError::Config(format!(
                "control_vector_layer_range must be \"lo-hi\" with lo <= hi, got {:?}",
                self.control_vector_layer_range
            )));
        }
        if !self.tensor_preset.is_empty() && tensor_preset_entries(&self.tensor_preset).is_none() {
            return Err(CoreError::Config(format!(
                "tensor_preset must be one of moe-cpu-offload (or empty), got {:?}",
                self.tensor_preset
            )));
        }
        for (name, o) in &self.model_overrides {
            if let Some(e) = o.reasoning_effort.as_deref() {
                let e = e.trim();
                if !e.is_empty() && !REASONING_EFFORT_LEVELS.contains(&e) {
                    return Err(CoreError::Config(format!(
                        "model_overrides.{name}.reasoning_effort must be one of {} (or empty), got {e:?}",
                        REASONING_EFFORT_LEVELS.join("|")
                    )));
                }
            }
            if let Some(d) = o.devices.as_ref() {
                if d.iter().any(|x| x.trim().is_empty()) {
                    return Err(CoreError::Config(format!(
                        "model_overrides.{name}.devices entries must be non-empty device names"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Apply documented `PALLAMA_*` env overrides on top of file values.
    /// Env wins over file; each override fails fast on a malformed value.
    #[allow(clippy::too_many_lines)] // one env var per knob, flat by design
    pub fn with_env_overrides(&self) -> CoreResult<Self> {
        let mut cfg = self.clone();
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());

        if let Some(v) = env("PALLAMA_HOST") {
            cfg.host = v;
        }
        if let Some(v) = env("PALLAMA_PORT") {
            cfg.port = parse_u16("PALLAMA_PORT", &v)?;
        }
        if let Some(v) = env("PALLAMA_DEFAULT_CTX") {
            cfg.default_ctx = parse_u32("PALLAMA_DEFAULT_CTX", &v)?;
        }
        if let Some(v) = env("PALLAMA_IDLE_SLEEP_SECS") {
            cfg.idle_sleep_secs = parse_u64("PALLAMA_IDLE_SLEEP_SECS", &v)?;
        }
        if let Some(v) = env("PALLAMA_IDLE_TIMEOUT_SECS") {
            cfg.idle_timeout_secs = parse_u64("PALLAMA_IDLE_TIMEOUT_SECS", &v)?;
        }
        if let Some(v) = env("PALLAMA_MAX_LOADED_MODELS") {
            cfg.max_loaded_models = parse_u32("PALLAMA_MAX_LOADED_MODELS", &v)?;
        }
        if let Some(v) = env("PALLAMA_CHILD_TRANSPORT") {
            cfg.child_transport = v;
        }
        if let Some(v) = env("PALLAMA_ENGINE_ASSET") {
            cfg.engine_asset = v;
        }
        if let Some(v) = env("PALLAMA_ENGINE_PIN") {
            cfg.engine_pin = v;
        }
        if let Some(v) = env("PALLAMA_SPEC") {
            cfg.spec = v;
        }
        if let Some(v) = env("PALLAMA_CACHE_REUSE") {
            cfg.cache_reuse = parse_u32("PALLAMA_CACHE_REUSE", &v)?;
        }
        if let Some(v) = env("PALLAMA_KEYS") {
            // `name:key[,name:key...]` — machine-composition-friendly form
            // of the [[keys]] tables; budgets/scopes stay file-only.
            cfg.keys = v
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(|s| {
                    let (name, key) = s.split_once(':').unwrap_or((s, s));
                    ApiKey {
                        name: name.to_string(),
                        key: key.to_string(),
                        ..ApiKey::default()
                    }
                })
                .collect();
        }
        if let Some(v) = env("PALLAMA_RPC_SERVERS") {
            cfg.rpc_servers = v;
        }
        if let Some(v) = env("PALLAMA_CACHE_RAM_MB") {
            cfg.cache_ram_mb = parse_i64("PALLAMA_CACHE_RAM_MB", &v)?;
        }
        if let Some(v) = env("PALLAMA_CPU_RANGE") {
            cfg.cpu_range = v;
        }
        if let Some(v) = env("PALLAMA_POLL") {
            cfg.poll = parse_u32("PALLAMA_POLL", &v)?;
        }
        if let Some(v) = env("PALLAMA_REASONING_FORMAT") {
            cfg.reasoning_format = v;
        }
        if let Some(v) = env("PALLAMA_SLOTS") {
            cfg.slots = parse_u32("PALLAMA_SLOTS", &v)?;
        }
        if let Some(v) = env("PALLAMA_CACHE_TYPE") {
            cfg.cache_type = v;
        }
        if let Some(v) = env("PALLAMA_SPEC_CACHE") {
            cfg.spec_cache = parse_bool("PALLAMA_SPEC_CACHE", &v)?;
        }
        if let Some(v) = env("PALLAMA_CTX_EXTEND") {
            cfg.ctx_extend = parse_f64("PALLAMA_CTX_EXTEND", &v)?;
        }
        if let Some(v) = env("PALLAMA_CPU_MOE_N") {
            cfg.cpu_moe_n = parse_i32("PALLAMA_CPU_MOE_N", &v)?;
        }
        if let Some(v) = env("PALLAMA_OVERRIDE_TENSOR") {
            cfg.override_tensor = v
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect();
        }

        Self::apply_wave_env_overrides(&mut cfg)?;

        if let Some(v) = env("PALLAMA_SLOT_PROMPT_SIMILARITY") {
            cfg.slot_prompt_similarity = parse_f64("PALLAMA_SLOT_PROMPT_SIMILARITY", &v)?;
        }
        if let Some(v) = env("PALLAMA_SENTINEL") {
            cfg.sentinel = parse_bool("PALLAMA_SENTINEL", &v)?;
        }
        if let Some(v) = env("PALLAMA_SENTINEL_STALL_SECS") {
            cfg.sentinel_stall_secs = parse_u64("PALLAMA_SENTINEL_STALL_SECS", &v)?;
        }
        if let Some(v) = env("PALLAMA_SENTINEL_ENFORCE") {
            cfg.sentinel_enforce = parse_bool("PALLAMA_SENTINEL_ENFORCE", &v)?;
        }
        cfg.validate()?;
        Ok(cfg)
    }

    /// Later-wave env knobs (`PALLAMA_AGENT` ..= `PALLAMA_ENGINE_CHECK_SECS`),
    /// split out only to keep `with_env_overrides` under the line budget.
    fn apply_wave_env_overrides(cfg: &mut Self) -> CoreResult<()> {
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        if let Some(v) = env("PALLAMA_AGENT") {
            cfg.agent = parse_bool("PALLAMA_AGENT", &v)?;
        }
        if let Some(v) = env("PALLAMA_SESSIONS") {
            cfg.sessions = parse_bool("PALLAMA_SESSIONS", &v)?;
        }
        if let Some(v) = env("PALLAMA_ROUTER") {
            cfg.router = parse_bool("PALLAMA_ROUTER", &v)?;
        }
        if let Some(v) = env("PALLAMA_ROUTER_MAX_MODELS") {
            cfg.router_max_models = parse_u32("PALLAMA_ROUTER_MAX_MODELS", &v)?;
        }
        if let Some(v) = env("PALLAMA_DEVICES") {
            cfg.devices = v
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect();
        }
        if let Some(v) = env("PALLAMA_ENGINE_CHECK_SECS") {
            cfg.engine_check_secs = parse_u64("PALLAMA_ENGINE_CHECK_SECS", &v)?;
        }
        Ok(())
    }
}

fn parse_u16(key: &str, raw: &str) -> CoreResult<u16> {
    raw.parse::<u16>()
        .map_err(|e| CoreError::Config(format!("invalid {key} {raw:?}: {e}")))
}
fn parse_u32(key: &str, raw: &str) -> CoreResult<u32> {
    raw.parse::<u32>()
        .map_err(|e| CoreError::Config(format!("invalid {key} {raw:?}: {e}")))
}
fn parse_u64(key: &str, raw: &str) -> CoreResult<u64> {
    raw.parse::<u64>()
        .map_err(|e| CoreError::Config(format!("invalid {key} {raw:?}: {e}")))
}
fn parse_i64(key: &str, raw: &str) -> CoreResult<i64> {
    raw.parse::<i64>()
        .map_err(|e| CoreError::Config(format!("invalid {key} {raw:?}: {e}")))
}
fn parse_f64(key: &str, raw: &str) -> CoreResult<f64> {
    raw.parse::<f64>()
        .map_err(|e| CoreError::Config(format!("invalid {key} {raw:?}: {e}")))
}
fn parse_bool(key: &str, raw: &str) -> CoreResult<bool> {
    raw.parse::<bool>()
        .map_err(|_| CoreError::Config(format!("invalid {key} {raw:?}: expected true or false")))
}
fn parse_i32(key: &str, raw: &str) -> CoreResult<i32> {
    raw.parse::<i32>()
        .map_err(|e| CoreError::Config(format!("invalid {key} {raw:?}: {e}")))
}

/// `lo-hi` decimal CPU range with lo <= hi (upstream `--cpu-range` syntax).
fn default_stall_secs() -> u64 {
    30
}

fn valid_cpu_range(s: &str) -> bool {
    let Some((lo, hi)) = s.split_once('-') else {
        return false;
    };
    match (lo.parse::<u32>(), hi.parse::<u32>()) {
        (Ok(lo), Ok(hi)) => lo <= hi,
        _ => false,
    }
}

fn default_slots() -> u32 {
    1
}

fn default_true() -> bool {
    true
}

fn default_engine_check_secs() -> u64 {
    86_400 // daily; 0 disables
}

/// KV cache types upstream accepts for K and V (`-ctk`/`-ctv`).
pub const CACHE_TYPES: &[&str] = &[
    "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
];

fn valid_cache_type(s: &str) -> bool {
    s.is_empty() || CACHE_TYPES.contains(&s)
}

const LOAD_MODES: [&str; 4] = ["", "mmap", "mlock", "direct-io"];

fn valid_override_tensor(s: &str) -> bool {
    match s.split_once('=') {
        Some((pat, dev)) => !pat.is_empty() && !dev.is_empty(),
        None => false,
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn unit__lookup_cache__validation_requires_existing_file() {
        let good = std::env::temp_dir().join("pallama-lc-test.bin");
        std::fs::write(&good, b"ggml").unwrap();
        let cfg = Config {
            lookup_cache_static: Some(good.display().to_string()),
            lookup_cache_dynamic: Some("/definitely/not/present/lc.bin".to_string()),
            ..Config::default()
        };
        let err = cfg.validate().unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("lookup_cache_dynamic") && msg.contains("/definitely/not/present/lc.bin"),
            "error names the field and path: {msg}"
        );
        let _ = std::fs::remove_file(&good);

        // Both existing -> validation passes.
        let d = std::env::temp_dir().join("pallama-lc-test-dyn.bin");
        std::fs::write(&d, b"ggml").unwrap();
        let cfg = Config {
            lookup_cache_static: Some(good.display().to_string()),
            lookup_cache_dynamic: Some(d.display().to_string()),
            ..Config::default()
        };
        let _ = std::fs::write(&good, b"ggml");
        assert!(cfg.validate().is_ok());
        let _ = std::fs::remove_file(&d);
    }

    #[test]
    fn unit__wire_validation__enum_and_range_rejects() {
        for bad in ["turbo", "ultra"] {
            let cfg = Config {
                reasoning_effort: bad.into(),
                ..Default::default()
            };
            assert!(cfg.validate().is_err(), "{bad} must be rejected");
        }
        Config {
            reasoning_effort: "xhigh".into(),
            ..Default::default()
        }
        .validate()
        .unwrap();
        assert!(
            Config {
                prio: 4,
                ..Default::default()
            }
            .validate()
            .is_err(),
            "prio range"
        );
        assert!(
            Config {
                prio_batch: -2,
                ..Default::default()
            }
            .validate()
            .is_err(),
            "prio_batch range"
        );
        assert!(
            Config {
                spec_draft_ngl: "many".into(),
                ..Default::default()
            }
            .validate()
            .is_err(),
            "spec_draft_ngl vocabulary"
        );
        Config {
            spec_draft_ngl: "all".into(),
            ..Default::default()
        }
        .validate()
        .unwrap();
        assert!(
            Config {
                spec_draft_p_min: Some(1.5),
                ..Default::default()
            }
            .validate()
            .is_err(),
            "p_min range"
        );
    }

    #[test]
    fn unit__wire_validation__override_kv_shape() {
        assert!(
            Config {
                override_kv: vec!["key=value".into()],
                ..Default::default()
            }
            .validate()
            .is_err(),
            "missing TYPE colon"
        );
        Config {
            override_kv: vec!["tokenizer.ggml.add_bos_token=bool:false".into()],
            ..Default::default()
        }
        .validate()
        .unwrap();
        Config {
            override_kv: vec!["k=int:notanumber".into()],
            ..Default::default()
        }
        .validate()
        .unwrap(); // shape ok; the ENGINE parses the value
    }

    #[test]
    fn unit__wire_validation__tensor_preset_vocabulary() {
        assert!(Config {
            tensor_preset: "everything-on-cpu".into(),
            ..Default::default()
        }
        .validate()
        .is_err());
        let cfg = Config {
            tensor_preset: "moe-cpu-offload".into(),
            ..Default::default()
        };
        cfg.validate().unwrap();
        assert_eq!(
            cfg.effective_override_tensor("any-model"),
            &["exps=CPU".to_string()]
        );
    }

    #[test]
    fn unit__wire_validation__image_token_cross_check() {
        assert!(
            Config {
                image_max_tokens: 256,
                image_min_tokens: 512,
                ..Default::default()
            }
            .validate()
            .is_err(),
            "max < min"
        );
        Config {
            image_max_tokens: 256,
            image_min_tokens: 64,
            ..Default::default()
        }
        .validate()
        .unwrap();
    }

    #[test]
    fn unit__wire_overlay__effective_warmup_and_reasoning() {
        let mut cfg = Config::default();
        assert!(cfg.effective_warmup("m"), "global default true");
        cfg.warmup = false;
        assert!(!cfg.effective_warmup("m"));
        cfg.model_overrides.insert(
            "m".into(),
            ModelOverride {
                warmup: Some(true),
                reasoning_budget: Some(128),
                reasoning_effort: Some("high".into()),
                ..Default::default()
            },
        );
        assert!(cfg.effective_warmup("m"), "overlay wins");
        assert_eq!(cfg.effective_reasoning_budget("m"), 128);
        assert_eq!(cfg.effective_reasoning_effort("m"), "high");
        // Other models inherit the global.
        assert!(!cfg.effective_warmup("other"));
        assert_eq!(cfg.effective_reasoning_budget("other"), -1);
    }

    #[test]
    fn unit__defaults_roundtrip_toml__stable() {
        let cfg = Config::default();
        let raw = cfg.to_toml().unwrap();
        let back = Config::from_toml(&raw).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn unit__default_file_contains_documented_defaults() {
        let raw = Config::default().to_toml().unwrap();
        assert!(raw.contains("port = 11434"));
        assert!(raw.contains("default_ctx = 16384"));
        assert!(raw.contains("idle_sleep_secs = 300"));
        assert!(raw.contains("idle_timeout_secs = 1800"));
    }

    #[test]
    fn unit__new_knobs__validated() {
        // Old config files (missing the new keys) still parse: serde default.
        let old = r"port = 11434
default_ctx = 16384
";
        let cfg = Config::from_toml(old).unwrap();
        assert_eq!(cfg.cpu_range, "");
        assert_eq!(cfg.poll, 0);

        // spec ngram accepted globally and per-overlay.
        Config::from_toml("spec = \"ngram\"\n").unwrap();
        Config::from_toml("[model_overrides.m]\nspec = \"ngram\"\n").unwrap();

        for bad in ["0", "5-1", "lo-hi", "0-15-3"] {
            assert!(
                Config::from_toml(&format!("cpu_range = \"{bad}\"\n")).is_err(),
                "{bad}"
            );
        }
        Config::from_toml("cpu_range = \"0-15\"\n").unwrap();

        assert!(Config::from_toml("poll = 101\n").is_err());
        Config::from_toml("poll = 100\n").unwrap();

        assert!(Config::from_toml("reasoning_format = \"bogus\"\n").is_err());
        Config::from_toml("reasoning_format = \"deepseek\"\n").unwrap();
        assert!(
            Config::from_toml("spec = \"ngram-simple\"\n").is_err(),
            "raw spec types are not config values"
        );
    }

    #[test]
    fn unit__unknown_top_level_key__config_error() {
        let raw = "port = 1234\nbogus_knob = 3\n";
        let err = Config::from_toml(raw).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("bogus_knob"),
            "error must name the unknown key: {msg}"
        );
    }

    #[test]
    fn unit__overlay_unknown_key__config_error() {
        let raw = "[model_overrides.m]\nctx = 8192\nwrong = true\n";
        let err = Config::from_toml(raw).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("wrong"),
            "error must name the unknown overlay key: {msg}"
        );
    }

    #[test]
    fn unit__overlay_merge__overlay_wins() {
        let mut cfg = Config::default();
        cfg.model_overrides.insert(
            "qwen3-coder-30b".into(),
            ModelOverride {
                ctx: Some(32768),
                ..Default::default()
            },
        );
        assert_eq!(cfg.effective_ctx("qwen3-coder-30b"), 32768);
        assert_eq!(cfg.effective_ctx("other-model"), cfg.default_ctx);
        assert_eq!(cfg.effective_spec("qwen3-coder-30b"), "off");
    }

    #[test]
    fn unit__validation__eviction_before_sleep_rejected() {
        let cfg = Config {
            idle_sleep_secs: 600,
            idle_timeout_secs: 300,
            ..Config::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("idle_timeout_secs"));
    }

    #[test]
    fn unit__validation__bad_spec_mode_rejected() {
        let cfg = Config {
            spec: "maybe".into(),
            ..Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn unit__validation__overlay_bad_spec_rejected() {
        let mut cfg = Config::default();
        cfg.model_overrides.insert(
            "m".into(),
            ModelOverride {
                spec: Some("turbo".into()),
                ..Default::default()
            },
        );
        assert!(cfg.validate().unwrap_err().to_string().contains("m.spec"));
    }

    #[test]
    fn unit__env_override__file_plus_env() {
        // Scoped env mutation: serial test, restored unconditionally.
        let _g = env_lock();
        let cfg = Config {
            port: 1,
            ..Config::default()
        };
        std::env::set_var("PALLAMA_PORT", "12345");
        std::env::set_var("PALLAMA_DEFAULT_CTX", "4096");
        let merged = cfg.with_env_overrides().unwrap();
        std::env::remove_var("PALLAMA_PORT");
        std::env::remove_var("PALLAMA_DEFAULT_CTX");
        assert_eq!(merged.port, 12345);
        assert_eq!(merged.default_ctx, 4096);
    }

    #[test]
    fn unit__env_override__malformed_value__named_error() {
        let _g = env_lock();
        std::env::set_var("PALLAMA_PORT", "not-a-port");
        let err = Config::default().with_env_overrides().unwrap_err();
        std::env::remove_var("PALLAMA_PORT");
        let msg = err.to_string();
        assert!(
            msg.contains("PALLAMA_PORT") && msg.contains("not-a-port"),
            "{msg}"
        );
    }

    #[test]
    fn unit__new_knobs__defaults_are_conservative() {
        let c = Config::default();
        assert_eq!(c.cache_type, ""); // auto ladder
        assert!(c.spec_cache);
        assert_eq!(
            c.ctx_extend.to_bits(),
            0.0_f64.to_bits(),
            "default ctx_extend not zero"
        );
        assert_eq!(c.cpu_moe_n, 0);
        assert!(c.override_tensor.is_empty());
        assert!(!c.agent);
        assert!(c.sessions);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn unit__cache_type__vocabulary_enforced() {
        let ok = Config {
            cache_type: "q4_0".into(),
            ..Config::default()
        };
        assert!(ok.validate().is_ok());
        let bad = Config {
            cache_type: "q9_9".into(),
            ..Config::default()
        };
        let err = bad.validate().unwrap_err().to_string();
        assert!(err.contains("cache_type") && err.contains("q8_0"), "{err}");
    }

    #[test]
    fn unit__ctx_extend__range_enforced() {
        for ok in [0.0, 2.0, 16.0, 32.0] {
            let c = Config {
                ctx_extend: ok,
                ..Config::default()
            };
            assert!(c.validate().is_ok(), "{ok} should validate");
        }
        for bad in [0.5, 1.0, 33.0, -2.0] {
            let c = Config {
                ctx_extend: bad,
                ..Config::default()
            };
            assert!(c.validate().is_err(), "{bad} should reject");
        }
    }

    #[test]
    fn unit__override_tensor__shape_enforced() {
        let ok = Config {
            override_tensor: vec![".ffn_.*_exps.=CPU".into()],
            ..Config::default()
        };
        assert!(ok.validate().is_ok());
        for bad in ["no-device-separator", "=CPU"] {
            let c = Config {
                override_tensor: vec![bad.into()],
                ..Config::default()
            };
            assert!(c.validate().is_err(), "{bad} should reject");
        }
    }

    #[test]
    fn unit__effective_knobs__overlay_beats_global() {
        let c = Config {
            cache_type: "q8_0".into(),
            ctx_extend: 2.0,
            cpu_moe_n: 4,
            override_tensor: vec!["global=CPU".into()],
            model_overrides: BTreeMap::from([(
                "m1".into(),
                ModelOverride {
                    cache_type: Some("q5_0".into()),
                    ctx_extend: Some(4.0),
                    cpu_moe_n: Some(8),
                    override_tensor: Some(vec!["local=GPU".into()]),
                    ..ModelOverride::default()
                },
            )]),
            ..Config::default()
        };
        assert_eq!(c.effective_cache_type("m1"), "q5_0");
        assert_eq!(c.effective_ctx_extend("m1").to_bits(), 4.0_f64.to_bits());
        assert_eq!(c.effective_cpu_moe_n("m1"), 8);
        assert_eq!(
            c.effective_override_tensor("m1"),
            &["local=GPU".to_string()]
        );
        // other models fall through to globals
        assert_eq!(c.effective_cache_type("m2"), "q8_0");
        assert_eq!(
            c.effective_override_tensor("m2"),
            &["global=CPU".to_string()]
        );
    }

    #[test]
    fn unit__keys_env__name_key_pairs() {
        let _g = env_lock();
        std::env::set_var("PALLAMA_KEYS", " alice:plm_a , bob:plm_b ,");
        let cfg = Config::default().with_env_overrides().unwrap();
        std::env::remove_var("PALLAMA_KEYS");
        assert_eq!(cfg.keys.len(), 2);
        assert_eq!(cfg.keys[0].name, "alice");
        assert_eq!(cfg.keys[0].key, "plm_a");
        assert!(cfg.key_for("plm_b").is_some_and(|k| k.name == "bob"));
    }

    #[test]
    fn unit__keys_tables__parse_scope_and_budgets() {
        let raw = r#"
port = 11500
[[keys]]
name = "ci"
key = "plm_ci"
models = ["qwen3.5-9b"]
rpm = 12
tpm = 4000
daily_tokens = 1_000_000
[[keys]]
name = "admin"
key = "plm_admin"
"#;
        let cfg = Config::from_toml(raw).unwrap();
        assert_eq!(cfg.keys.len(), 2);
        let ci = cfg.key_for("plm_ci").unwrap();
        assert_eq!(ci.models, vec!["qwen3.5-9b".to_string()]);
        assert_eq!(ci.rpm, 12);
        assert_eq!(ci.tpm, 4000);
        assert_eq!(ci.daily_tokens, 1_000_000);
        assert!(cfg
            .key_for("plm_admin")
            .is_some_and(|k| k.models.is_empty()));
    }

    #[test]
    fn unit__legacy_api_keys__in_memory_migration() {
        let raw = "port = 11500\napi_keys = [\"k1\", \"k2\"]\n";
        assert!(Config::raw_has_legacy_keys(raw));
        let cfg = Config::from_toml(raw).unwrap();
        assert_eq!(cfg.port, 11500, "the rest of the config survives");
        assert_eq!(cfg.keys.len(), 2);
        assert_eq!(cfg.keys[0].name, "migrated-0");
        assert_eq!(cfg.keys[0].key, "k1");
        assert!(
            cfg.keys.iter().all(|k| k.models.is_empty()),
            "admin power preserved"
        );
        assert!(cfg.key_for("k2").is_some());
        // Canonical round-trip is clean and legacy-free.
        let canon = cfg.to_toml().unwrap();
        assert!(!Config::raw_has_legacy_keys(&canon));
        assert!(canon.contains("[[keys]]"));
        // Absent legacy: normal parse, no false positive.
        assert!(!Config::raw_has_legacy_keys("port = 11500\n"));
    }

    #[test]
    fn unit__keys_validation__rejects_empty_and_duplicates() {
        let cfg = |keys: Vec<ApiKey>| Config {
            keys,
            ..Config::default()
        };
        let c = cfg(vec![ApiKey {
            name: "a".into(),
            key: String::new(),
            ..ApiKey::default()
        }]);
        assert!(c.validate().unwrap_err().to_string().contains("secret"));

        let c = cfg(vec![
            ApiKey {
                name: "a".into(),
                key: "k1".into(),
                ..ApiKey::default()
            },
            ApiKey {
                name: "a".into(),
                key: "k2".into(),
                ..ApiKey::default()
            },
        ]);
        assert!(c
            .validate()
            .unwrap_err()
            .to_string()
            .contains("duplicate name"));
    }

    #[test]
    fn unit__load_creates_default_file_when_absent() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = PallamaDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        let cfg = Config::load(&dirs).unwrap();
        assert_eq!(cfg, Config::default());
        assert!(
            dirs.config_file().exists(),
            "default config must be written"
        );
        // Second load reads the same file back.
        assert_eq!(Config::load(&dirs).unwrap(), Config::default());
    }

    #[test]
    fn unit__load_existing_file__values_honored() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = PallamaDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        std::fs::create_dir_all(&dirs.config_dir).unwrap();
        std::fs::write(dirs.config_file(), "port = 9999\ndefault_ctx = 8192\n").unwrap();
        let cfg = Config::load(&dirs).unwrap();
        assert_eq!((cfg.port, cfg.default_ctx), (9999, 8192));
    }

    /// Tests mutate process env; serialize with a global lock so parallel
    /// test threads never race on the same variable.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}
