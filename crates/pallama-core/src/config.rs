use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::dirs::PallamaDirs;
use crate::error::{CoreError, CoreResult};

/// Engine/app update channel. See `Config::update_channel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UpdateChannel {
    /// Newest release including prereleases (llama.cpp nightly b-tags).
    #[default]
    Latest,
    /// Newest non-prerelease (GitHub `/releases/latest`).
    Stable,
}

impl std::fmt::Display for UpdateChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            UpdateChannel::Latest => "latest",
            UpdateChannel::Stable => "stable",
        })
    }
}

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
    /// Per-child bearer auth (`--api-key-file` on every spawned engine
    /// child): None = auto (TCP children authenticated — any local
    /// process could otherwise bypass the gateway's keys by hitting the
    /// child port directly; UDS children already get filesystem
    /// permissions), Some(true) = always, Some(false) = never.
    #[serde(default)]
    pub child_auth: Option<bool>,
    /// Tracing `EnvFilter` directive for the daemon, e.g. `"debug"` or
    /// `"pallama=trace,pallama::engine=debug"`. `None` = built-in default
    /// (`INFO`). The `RUST_LOG` env var always wins when set (escape hatch
    /// for systemd units and one-off debugging).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub log_level: Option<String>,
    /// Engine/app update channel. `latest` = newest release including
    /// prereleases (llama.cpp nightly b-tags); `stable` = newest
    /// non-prerelease. The channel is a PIN, not a floor: switching
    /// `latest` -> `stable` re-targets (downgrades) to stable's current
    /// release; `pallama engine update`/`pallama upgrade` then move to it.
    #[serde(default)]
    pub update_channel: UpdateChannel,
    /// "auto" or explicit engine asset suffix (e.g. "ubuntu-vulkan-x64").
    pub engine_asset: String,
    /// Parallel byte-range connections used for large model downloads
    /// (>= 32 MiB). 1 = classic single-stream resume lane. Servers that
    /// reject Range get the single-stream lane regardless.
    pub download_connections: u32,
    /// "auto" (default: opportunistic — embedded MTP head used when the
    /// GGUF carries one, catalog draft pair used when pulled, dense with a
    /// teaching warning otherwise) | "off" | "mtp" | "eagle3" | "dflash" |
    /// "dspark" (explicit modes fail fast when their inputs are missing).
    /// n-gram variants: see ngram-* keys. Default flipped off→auto
    /// 2026-09-11: speculation is verified-lossless (target-side
    /// verification) and MTP measured +50% decode on Qwen3.5-9B; ollama
    /// auto-enables MTP the same way.
    pub spec: String,
    /// On-demand tensor loading (`--lazy-mode`): "auto" (engine default:
    /// on-demand only for tensors > 4 GiB), "on" (all such tensors from
    /// disk via mmap — big-MoE RAM relief), "off" (fully resident).
    pub lazy_mode: String,
    /// EXPERIMENTAL upstream server-side agent tools (`--tools` CSV or
    /// "all"): the engine gains read/grep/exec/write capabilities —
    /// opt-in, never defaulted, engine also limits CORS to localhost
    /// when set. Do not enable in untrusted environments.
    #[serde(default)]
    pub server_tools: Option<String>,
    /// Runtime sandbox for `server_tools` (`--tools-runtime`), one of
    /// docker:<image> | podman:<image> | docker-container:<id> |
    /// podman-container:<id> | ssh:<target>. Passthrough only — pallama
    /// itself never requires docker.
    #[serde(default)]
    pub server_tools_runtime: Option<String>,
    /// Path to a Cursor-compatible JSON of MCP server definitions passed
    /// to the engine (`--mcp-servers-config`). Exclusive with
    /// `mcp_servers_json`. EXPERIMENTAL upstream; untrusted-input risk.
    #[serde(default)]
    pub mcp_servers_config: Option<String>,
    /// Inline JSON form of `mcp_servers_config` (`--mcp-servers-json`).
    #[serde(default)]
    pub mcp_servers_json: Option<String>,
    /// Min chunk size for KV-shift prefix reuse; 0 disables. Default 0:
    /// the engine's native slot prompt-cache already covers identical
    /// prefixes (16x on re-ask, measured) at zero cost, while the
    /// `--cache-reuse` path measured ~0.6s SLOWER cold loads (elim
    /// sweep 2026-09-12). Opt back in for cross-slot prefix sharing.
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
    /// Server slots (`-np`). 0 = pallama auto (default): capacity-aware
    /// concurrency — the total ctx scales so each slot keeps the resolved
    /// per-slot ctx, and the slot count is clamped by the KV budget and the
    /// model's trained context (up to 4). Concurrent streams then batch on
    /// the GPU instead of queueing (measured ~2.5x system throughput).
    /// 1 = full-speed single client pin; N = manual pin (upstream slices
    /// the resolved ctx across N slots).
    #[serde(default = "default_slots")]
    pub slots: u32,
    /// Deterministic decoding pin: `true` forces slots = 1 for every
    /// model without a per-model override. Multi-slot batches perturb
    /// logits in near-tie positions, so greedy runs under `slots > 1`
    /// do not reproduce token-for-token (measured: auto-slots np=4
    /// flipped 14/20 greedy probes vs the same child at slots = 1).
    /// Costs single-client nothing; concurrent streams queue instead
    /// of batching (~2.5x system throughput left on the table).
    #[serde(default)]
    pub deterministic: bool,
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
    /// Warm-peg JIT-class engines (sglang) at spawn: after /health turns
    /// 200, fire one tiny single completion (drains the residual warmup
    /// queue) plus a small concurrent burst (pegs the bs=N batch shapes)
    /// BEFORE Ready publishes — the JIT cost lands in spawn instead of on
    /// a random first request. No-op for engines that ship precompiled
    /// kernels (llamacpp/mistralrs).
    #[serde(default = "default_true")]
    pub warm_after_spawn: bool,
    /// `YaRN` `RoPE` context-extension factor: 0 = off; e.g. 2.0 doubles the
    /// usable context beyond the trained window at some quality cost.
    #[serde(default)]
    pub ctx_extend: f64,
    /// Keep N `MoE` expert tensors on CPU (`--n-cpu-moe`): finer-grained
    /// cousin of the boolean MoE-offload heuristic. 0 = off.
    #[serde(default)]
    pub cpu_moe_n: i32,
    /// Keep the dense `FFN` weights of the first N layers on CPU
    /// (`--n-cpu-ffn`, dense models; the `--n-cpu-moe` analogue for
    /// non-MoE architectures). 0 = off.
    #[serde(default)]
    pub cpu_ffn_n: i32,
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
    /// Late chunking (R1): hard cap on total doc tokens a late-chunking
    /// embed request may carry. Beyond it the gateway answers 413 instead
    /// of streaming a per-token embedding matrix through JSON. Applies to
    /// models opted in via `[model_overrides.<name>] late_chunking = true`.
    #[serde(default = "default_late_chunking_max_tokens")]
    pub late_chunking_max_tokens: usize,
    /// Session pins (R3): how long a model stays protected from idle
    /// eviction after a request carrying `x-pallama-session: <name>`.
    /// Each such request refreshes the window; `POST /api/session
    /// {"action":"close","session":name}` releases immediately, and
    /// `pallama stop` / `/api/stop` always wins. 0 disables pinning
    /// entirely (header ignored, zero overhead).
    #[serde(default = "default_session_keep_secs")]
    pub session_keep_secs: u64,
    /// R4: opt-in semantic cache for non-stream chat responses.
    /// Disabled by default — semantic similarity can serve a near-miss
    /// where an exact match was required; correctness-sensitive lanes
    /// must stay off (per-request `x-pallama-cache: off` escape hatch).
    #[serde(default)]
    pub semantic_cache: SemanticCacheConfig,
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
    /// Restart the daemon automatically after an engine switch (use/
    /// update/build/install/rollback activated a different engine).
    /// false (default) = print the restart hint only: the daemon keeps
    /// serving with the engine it booted with until restarted.
    #[serde(default)]
    pub auto_restart_engine_switch: bool,
    /// mistral.rs paged-attention KV budget as a fraction of GPU memory
    /// (`--pa-memory-fraction`): the upstream default (0.90) claims ~90%
    /// of VRAM for KV and hard-fails at load ("Num GPU blocks is 0")
    /// when weights + projector crowd an 8 GB card — lower it for big
    /// vision models. None (default) = upstream default; 0.05..=0.95.
    #[serde(default)]
    pub mistralrs_pa_memory_fraction: Option<f32>,
    /// mistral.rs attention backend override (`--paged-attn on|off`):
    /// None (default) = upstream auto (paged on CUDA). `Some(false)`
    /// sizes the classic KV cache from the context instead of a
    /// VRAM fraction — the only configuration that fits big vision
    /// models (9B + projector) on 8 GB cards, live-proven.
    #[serde(default)]
    pub mistralrs_paged_attn: Option<bool>,
    /// mistral.rs engine tuning knobs (global defaults; per-model
    /// `model_overrides.<name>.mistralrs` replaces this whole struct
    /// when present). Every field maps 1:1 to a `mistralrs serve` flag
    /// and is emitted only when the active engine's capability manifest
    /// advertises it — unknown-on-this-version flags skip with a
    /// warning instead of erroring. The two legacy globals
    /// (`mistralrs_pa_memory_fraction`, `mistralrs_paged_attn`) predate
    /// the struct and keep their names.
    #[serde(default)]
    pub mistralrs: MistralrsTuning,
    /// sglang engine tuning knobs (global defaults; per-model
    /// `model_overrides.<name>.sglang` replaces this whole struct when
    /// present). Every field maps 1:1 to a `python -m sglang.launch_server`
    /// flag and is emitted only when the active engine's capability
    /// manifest advertises it — unknown-on-this-version flags skip with a
    /// warning instead of erroring. The VRAM ladder flags
    /// (`--mem-fraction-static`, `--cpu-offload-gb`, `--kv-cache-dtype`,
    /// `--context-length`) are owned by the profile compiler; the tuning
    /// struct only overrides `mem_fraction_static` and `kv_cache_dtype`
    /// explicitly (explicit user pin always wins over the ladder).
    #[serde(default)]
    pub sglang: SglangTuning,
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
    /// E4 audit log: append one JSON line per GENERATION request
    /// (trace id, key, model, status, latency, priority, queue depth)
    /// to `<data>/log/audit.jsonl`. Off by default; rotates at 16 MiB.
    #[serde(default)]
    pub audit_log: bool,
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

    // ---- n-gram speculation tuning (spec = "ngram" and the typed
    // variants "ngram-map-k" | "ngram-map-k4v" | "ngram-mod"). Upstream
    // b10833 REMOVED the generic --spec-ngram-* forms; these emit the
    // typed --spec-ngram-<family>-* flags. 0 = engine default.
    /// n-gram lookup table size (tokens of context hashed). Applies to
    /// ngram, ngram-map-k and ngram-map-k4v (upstream default 48/48/12).
    #[serde(default)]
    pub ngram_size_m: u32,
    /// n-gram length. Applies to ngram, ngram-map-k and ngram-map-k4v.
    #[serde(default)]
    pub ngram_size_n: u32,
    /// Minimum table hits before a draft is trusted. Same families.
    #[serde(default)]
    pub ngram_min_hits: u32,
    /// ngram-mod only: lookup length. 0 = engine default (24). Range 1..=1024.
    #[serde(default)]
    pub ngram_mod_n_match: u32,
    /// ngram-mod only: max drafted tokens. 0 = engine default (64). Range 0..=1024.
    #[serde(default)]
    pub ngram_mod_n_max: u32,
    /// ngram-mod only: min drafted tokens. 0 = engine default (48). Range 0..=1024.
    #[serde(default)]
    pub ngram_mod_n_min: u32,

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
    /// Server-side reasoning switch: "" (engine default = auto-detect
    /// from the chat template) | "on" | "off" | "auto". Authoritative
    /// for templates that ignore the `thinking`/`enable_thinking`
    /// request variables.
    #[serde(default)]
    pub reasoning: String,

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

    // ---- child HTTP server behavior (llama-server lane).
    /// SSE keep-alive ping interval in seconds (-1 = disabled; upstream
    /// default 30). Keeps proxies and clients from idling out a stream
    /// during long generations.
    #[serde(default)]
    pub sse_ping_interval: Option<i64>,
    /// Child server read/write timeout in seconds (upstream default
    /// 3600). Lower values fail stuck connections sooner.
    #[serde(default)]
    pub server_timeout_secs: Option<u64>,
    /// Extra params for the child's JSON chat-template parser, as a JSON
    /// object string (e.g. `{"enable_thinking": false}`). The engine
    /// rejects a non-object string at boot.
    #[serde(default)]
    pub chat_template_kwargs: Option<String>,
    /// Continuous (dynamic) batching (upstream default true). Disabling
    /// serializes slots — useful only for debugging near-tie logits.
    #[serde(default)]
    pub cont_batching: Option<bool>,
    /// `SO_REUSEPORT` on the child listener (upstream default false). Lets
    /// a replacement child bind while the old one drains.
    #[serde(default)]
    pub reuse_port: bool,
    /// Load `LoRA` adapters without applying them at spawn (apply later
    /// per-request). Upstream default false.
    #[serde(default)]
    pub lora_init_without_apply: bool,

    // ---- engine behavior toggles (upstream defaults preserved; the
    // knob exists to disable).
    /// Warmup run at startup (upstream default true).
    #[serde(default = "default_true")]
    pub warmup: bool,

    /// Global projector policy default (None = `lazy`: text-only spawn,
    /// projector attaches on the first vision request via a KV-bank
    /// respawn — measured 2026-09-11: 875 MiB projector costs +3.9 s
    /// cold TTFT + 1126 MiB VRAM on a 9B VL row). Per-model
    /// `model_overrides.<name>.mmproj` wins.
    #[serde(default, deserialize_with = "de_mmproj")]
    pub mmproj_policy: Option<MmprojPolicy>,
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
    /// ("" = engine default; e.g. `"top_k;top_p;typical"`). The char-encoded
    /// `--sampler-seq` alias is deliberately not a knob: it writes the same
    /// upstream param this knob controls.
    #[serde(default)]
    pub samplers: String,

    // ---- compute plumbing / multi-GPU split (0/""/-1 = engine default).
    /// Logical batch size for prompt processing (0 = engine default).
    /// Bench-adopted tuning still wins over this knob.
    #[serde(default)]
    pub batch_size: u32,
    /// Physical micro-batch ceiling for prefill (0 = engine default).
    /// Bench-adopted tuning still wins over this knob.
    #[serde(default)]
    pub ubatch_size: u32,
    /// Batch-phase thread count (0 = follow `threads`).
    #[serde(default)]
    pub threads_batch: u32,
    /// Preferred GPU index for weights/KV in split mode (-1 = engine
    /// default). Only meaningful on multi-GPU boxes with `devices` set.
    #[serde(default = "default_main_gpu")]
    pub main_gpu: i32,
    /// Multi-GPU split strategy: "" | none | layer | row | tensor
    /// ("" = engine default `layer`; `tensor` is upstream-experimental).
    #[serde(default)]
    pub split_mode: String,
    /// Comma-separated per-GPU split ratios ("" = even split; e.g. "3,1"
    /// gives GPU0 three shares per one of GPU1).
    #[serde(default)]
    pub tensor_split: String,
    /// Router-mode model autoload into the child's own LRU (None =
    /// upstream default off). Complements `router` + `router_max_models`.
    #[serde(default)]
    pub models_autoload: Option<bool>,
}

fn default_reasoning_budget() -> i64 {
    -1
}
fn default_main_gpu() -> i32 {
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
    /// Per-model determinism pin (None = inherit the global
    /// `deterministic`). `true` forces slots = 1 for this model.
    #[serde(default)]
    pub deterministic: Option<bool>,
    pub spec: Option<String>,
    /// Per-model on-demand tensor loading (None = inherit `lazy_mode`).
    pub lazy_mode: Option<String>,
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
    /// Per-model dense-`FFN` CPU-offload count (None = inherit global).
    pub cpu_ffn_n: Option<i32>,
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
    /// Per-model server-side reasoning switch (None = inherit global).
    #[serde(default)]
    pub reasoning: Option<String>,
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
    /// Per-model chat template override (`--chat-template`): a Jinja
    /// template string. Mutually exclusive with `chat_template_file`.
    #[serde(default)]
    pub chat_template: Option<String>,
    /// Per-model chat template file (`--chat-template-file`). Mutually
    /// exclusive with `chat_template`. Must exist at config load.
    #[serde(default)]
    pub chat_template_file: Option<String>,
    /// Per-model sampling defaults, emitted as argv so they apply to
    /// every request that does not override the param in its body.
    #[serde(default)]
    pub sampler_defaults: Option<SamplerDefaults>,
    /// Per-model infill token-order toggle (`--spm-infill`): use
    /// Suffix/Prefix/Middle instead of Prefix/Suffix/Middle for `/infill`.
    /// Some coder models (e.g. `CodeGemma`) require it. None = upstream
    /// default (off).
    #[serde(default)]
    pub spm_infill: Option<bool>,
    /// Late chunking (R1): launch this model's child with `--embeddings
    /// --pooling none` and terminate all embed lanes at the gateway.
    /// List input on `/api/embed` embeds the joined document ONCE and
    /// mean-pools per-chunk token spans (jina late-chunking, arXiv
    /// 2409.04701). Opt-in: normal models keep byte-proxied embeds.
    #[serde(default)]
    pub late_chunking: Option<bool>,
    /// Per-model RPC server list (`--rpc`): comma-separated `host:port`
    /// entries pointing at `llama-server -r` workers. Replaces (not
    /// merges) the global `rpc_servers` list for this model — C6:
    /// different models can lean on different GPU boxes without a
    /// fleet-wide config change. Empty/None = inherit the global.
    #[serde(default)]
    pub rpc_servers: Option<String>,
    /// Per-model projector policy (None = the global default, `lazy`).
    /// `true`/`"attach"` spawns with the row's mmproj as before;
    /// `false`/`"skip"` spawns text-only and vision fails loudly;
    /// `"lazy"` spawns text-only for the measured cold-start/VRAM win
    /// (2026-09-11: 875 MiB projector ≈ +3.9 s cold TTFT + 1126 MiB VRAM
    /// on a 9B VL row) and RESPAWNS with the projector on the first
    /// vision request — conversation state rides the KV bank across the
    /// respawn, so nothing is lost beyond one warm-cache reload. An
    /// explicit `-mm` in `extra_args` still wins.
    #[serde(default, deserialize_with = "de_mmproj")]
    pub mmproj: Option<MmprojPolicy>,
    /// Per-model sglang tuning; replaces (not merges) the global
    /// `sglang` struct for this model. Only read when the model spawns on
    /// a `sglang` engine.
    #[serde(default)]
    pub sglang: Option<SglangTuning>,
    /// mistral.rs tuning override; replaces the global `[mistralrs]`
    /// struct for this model. Ignored unless the active engine is a
    /// `mistralrs` engine.
    #[serde(default)]
    pub mistralrs: Option<MistralrsTuning>,
}

/// Projector attach policy. Accepts the bool spellings the suppress knob
/// shipped with (`true`/`false`) plus `"attach"` / `"skip"` / `"lazy"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MmprojPolicy {
    Attach,
    Skip,
    Lazy,
}

impl From<bool> for MmprojPolicy {
    fn from(b: bool) -> Self {
        if b {
            Self::Attach
        } else {
            Self::Skip
        }
    }
}

impl MmprojPolicy {
    /// Resolve the effective policy from the model override + global
    /// default (`None` everywhere = `Lazy`).
    #[must_use]
    pub fn effective(override_: Option<Self>, global: Option<Self>) -> Self {
        override_.or(global).unwrap_or(Self::Lazy)
    }
}

/// sglang `launch_server` tuning knobs. All-`Option` on purpose: `None`
/// never emits a flag (upstream default applies); values are passed
/// through verbatim (validated as flag-gated at profile-compile, so a
/// value this Pallama build does not know still reaches a newer sglang
/// that supports it).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SglangTuning {
    /// `--attention-backend` (triton, fa3, fa4, flashinfer, `trtllm_mha`...).
    pub attention_backend: Option<String>,
    /// `--sampling-backend` (pytorch, flashinfer, ascend).
    pub sampling_backend: Option<String>,
    /// `--tool-call-parser` (qwen25, mistral, llama4...).
    pub tool_call_parser: Option<String>,
    /// `--reasoning-parser` (deepseek-r1, qwen3...).
    pub reasoning_parser: Option<String>,
    /// `--tokenizer-path`: separate tokenizer dir/file.
    pub tokenizer_path: Option<String>,
    /// `--dtype` (auto, half, bfloat16, float...).
    pub dtype: Option<String>,
    /// `--quantization` (awq, gptq, fp8, marlin...) — overrides the
    /// checkpoint's own quantization config.
    pub quantization: Option<String>,
    /// `--kv-cache-dtype` (auto, bf16, `fp8_e5m2`, `fp8_e4m3`...). Explicit
    /// pin wins over the profile compiler's VRAM ladder.
    pub kv_cache_dtype: Option<String>,
    /// `--mem-fraction-static` (0.05..=0.95). Explicit pin wins over the
    /// ladder's derived fraction.
    pub mem_fraction_static: Option<f32>,
    /// `--cpu-offload-gb`: weights GBs pinned to host RAM (low-VRAM
    /// ladder engages this automatically; an explicit pin wins).
    pub cpu_offload_gb: Option<f32>,
    /// `--page-size` (tokens per KV page).
    pub page_size: Option<u64>,
    /// `--schedule-policy` (lpm, fcfs, dfs-weight...).
    pub schedule_policy: Option<String>,
    /// `--schedule-conservativeness`.
    pub schedule_conservativeness: Option<f64>,
    /// `--chunked-prefill-size` (0.5.19 default 8192; the ladder lowers
    /// it on tight fits).
    pub chunked_prefill_size: Option<u64>,
    /// `--max-prefill-tokens`.
    pub max_prefill_tokens: Option<u64>,
    /// `--stream-interval` (tokens between stream chunks).
    pub stream_interval: Option<u32>,
    /// `--random-seed`.
    pub random_seed: Option<i64>,
    /// `--cuda-graph-max-bs` (0.5.19 default 256 — expensive on small
    /// cards; the ladder lowers it on tight fits).
    pub cuda_graph_max_bs: Option<u32>,
    /// `--enable-hierarchical-cache`: KV hierarchical caching to host
    /// RAM (`HiCache`). Off by default.
    pub hicache_enable: Option<bool>,
    /// `--hicache-ratio` (host:device KV size ratio, default 2.0).
    pub hicache_ratio: Option<f64>,
    /// `--hicache-size` (host KV cache size in GB).
    pub hicache_size: Option<f64>,
    /// `--enable-metrics` (Prometheus /metrics).
    pub metrics: Option<bool>,
    /// `--skip-server-warmup` (faster cold start; first request pays it).
    pub skip_warmup: Option<bool>,
    /// `--enable-torch-compile`.
    pub torch_compile: Option<bool>,
    /// `--tokenizer-mode` (auto, slow).
    pub tokenizer_mode: Option<String>,
    /// `--tokenizer-backend` (huggingface, fastokens).
    pub tokenizer_backend: Option<String>,
    /// `--tokenizer-worker-num`: dedicated tokenizer processes.
    pub tokenizer_worker_num: Option<u32>,
    /// `--detokenizer-worker-num`: dedicated detokenizer processes.
    pub detokenizer_worker_num: Option<u32>,
    /// `--enable-dynamic-batch-tokenizer`: batch encodes concurrent
    /// prompts instead of one-by-one.
    pub dynamic_batch_tokenizer: Option<bool>,
    /// `--dynamic-batch-tokenizer-batch-size`.
    pub dynamic_batch_tokenizer_batch_size: Option<u32>,
    /// `--dynamic-batch-tokenizer-batch-timeout` (seconds).
    pub dynamic_batch_tokenizer_batch_timeout: Option<f64>,
    /// `--grammar-backend` (xgrammar, outlines, llguidance, none) for
    /// structured/JSON output.
    pub grammar_backend: Option<String>,
    /// `--radix-eviction-policy` (lru, lfu, slru, priority) for the
    /// prefix (radix) cache.
    pub radix_eviction_policy: Option<String>,
    /// `--enable-session-radix-cache`: per-session radix cache
    /// isolation (session-aware prefix reuse).
    pub session_radix_cache: Option<bool>,
    /// `--enable-mixed-chunk`: let prefill chunks mix with decode
    /// batches (lower latency under mixed load).
    pub mixed_chunk: Option<bool>,
    /// `--sleep-on-idle`: reduce CPU usage when idle.
    pub sleep_on_idle: Option<bool>,
    /// `--enable-memory-saver`: release weights VRAM while idle
    /// (co-residency hygiene; reload cost on wake).
    pub memory_saver: Option<bool>,
    /// `--watchdog-timeout` (seconds).
    pub watchdog_timeout: Option<f64>,
    /// `--enable-cache-report`: report prefix-cache hits in
    /// `usage.prompt_tokens_details`.
    pub cache_report: Option<bool>,
    /// `--batch-notify-size`: asyncio notification batching under high
    /// concurrency (upstream default 16).
    pub batch_notify_size: Option<u32>,
    /// `--scheduler-recv-interval`: scheduler request-poll interval;
    /// > 1 reduces CPU overhead at a latency cost.
    pub scheduler_recv_interval: Option<u32>,
    /// `--cuda-graph-bs`: explicit capture list of decode batch sizes
    /// (e.g. `[1, 2, 4]`). Overrides the ladder's derived
    /// cuda-graph-max-bs on tight fits.
    pub cuda_graph_bs: Option<Vec<u32>>,
    /// `--max-total-tokens`: hard cap on KV tokens (ctx * capacity).
    pub max_total_tokens: Option<u64>,
    /// `--tp-size`: tensor parallelism. Emits only when > 1.
    pub tp_size: Option<u32>,
    /// `--dp-size`: data parallelism. Emits only when > 1.
    pub dp_size: Option<u32>,
    /// `--pp-size`: pipeline parallelism. Emits only when > 1.
    pub pp_size: Option<u32>,
    /// `--ep-size`: expert parallelism (`MoE` models). Emits only when > 1.
    pub ep_size: Option<u32>,
    /// `--max-lora-rank`: rank ceiling for served `LoRA` adapters.
    pub max_lora_rank: Option<u32>,
    /// `--lora-backend` (pytorch, flashinfer).
    pub lora_backend: Option<String>,
}

impl SglangTuning {
    /// Per-model resolution: override replaces global (not merges),
    /// mirroring `override_tensor` semantics.
    #[must_use]
    pub fn effective(override_: Option<&SglangTuning>, global: &SglangTuning) -> SglangTuning {
        override_.cloned().unwrap_or_else(|| global.clone())
    }

    fn validate(&self, where_: &str) -> Result<(), CoreError> {
        self.validate_scalar_ranges(where_)?;
        self.validate_choice_fields(where_)?;
        self.validate_count_floors(where_)?;
        Ok(())
    }

    /// Numeric envelopes for knobs the engine would only reject
    /// (or silently misbehave with) at spawn time.
    fn validate_scalar_ranges(&self, where_: &str) -> Result<(), CoreError> {
        if let Some(frac) = self.mem_fraction_static {
            if !(0.05..=0.95).contains(&frac) {
                return Err(CoreError::Config(format!(
                    "{where_}.mem_fraction_static must be 0.05..=0.95, got {frac}"
                )));
            }
        }
        if let Some(gb) = self.cpu_offload_gb {
            if gb < 0.0 {
                return Err(CoreError::Config(format!(
                    "{where_}.cpu_offload_gb must be >= 0, got {gb}"
                )));
            }
        }
        if let Some(r) = self.hicache_ratio {
            if r <= 0.0 {
                return Err(CoreError::Config(format!(
                    "{where_}.hicache_ratio must be > 0, got {r}"
                )));
            }
        }
        if let Some(t) = self.watchdog_timeout {
            if t <= 0.0 {
                return Err(CoreError::Config(format!(
                    "{where_}.watchdog_timeout must be > 0, got {t}"
                )));
            }
        }
        if let Some(t) = self.dynamic_batch_tokenizer_batch_timeout {
            if t <= 0.0 {
                return Err(CoreError::Config(format!(
                    "{where_}.dynamic_batch_tokenizer_batch_timeout must be > 0, got {t}"
                )));
            }
        }
        if let Some(t) = self.max_total_tokens {
            if t == 0 {
                return Err(CoreError::Config(format!(
                    "{where_}.max_total_tokens must be >= 1, got 0"
                )));
            }
        }
        Ok(())
    }

    /// Choice lists mirror the installed sglang 0.5.19 `--help`
    /// grammar — rejecting at config-write time beats an opaque
    /// argparse death at spawn.
    fn validate_choice_fields(&self, where_: &str) -> Result<(), CoreError> {
        let choice = |field: &str, v: &Option<String>, allowed: &[&str]| -> Option<CoreError> {
            let s = v.as_ref()?;
            if allowed.contains(&s.as_str()) {
                None
            } else {
                Some(CoreError::Config(format!(
                    "{where_}.{field} must be one of {allowed:?}, got {s:?}"
                )))
            }
        };
        if let Some(e) = choice("tokenizer_mode", &self.tokenizer_mode, &["auto", "slow"]) {
            return Err(e);
        }
        if let Some(e) = choice(
            "tokenizer_backend",
            &self.tokenizer_backend,
            &["huggingface", "fastokens"],
        ) {
            return Err(e);
        }
        if let Some(e) = choice(
            "grammar_backend",
            &self.grammar_backend,
            &["xgrammar", "outlines", "llguidance", "none"],
        ) {
            return Err(e);
        }
        if let Some(e) = choice(
            "radix_eviction_policy",
            &self.radix_eviction_policy,
            &["lru", "lfu", "slru", "priority"],
        ) {
            return Err(e);
        }
        Ok(())
    }

    /// `>= 1` floors for worker counts, batch sizes, parallel degrees,
    /// plus the cuda-graph capture list shape.
    fn validate_count_floors(&self, where_: &str) -> Result<(), CoreError> {
        let min_one_u32 = |field: &str, v: Option<u32>| -> Option<CoreError> {
            v.filter(|n| *n == 0)
                .map(|_| CoreError::Config(format!("{where_}.{field} must be >= 1, got 0")))
        };
        for (field, v) in [
            ("tokenizer_worker_num", self.tokenizer_worker_num),
            ("detokenizer_worker_num", self.detokenizer_worker_num),
            (
                "dynamic_batch_tokenizer_batch_size",
                self.dynamic_batch_tokenizer_batch_size,
            ),
            ("batch_notify_size", self.batch_notify_size),
            ("scheduler_recv_interval", self.scheduler_recv_interval),
            ("tp_size", self.tp_size),
            ("dp_size", self.dp_size),
            ("pp_size", self.pp_size),
            ("ep_size", self.ep_size),
            ("max_lora_rank", self.max_lora_rank),
        ] {
            if let Some(e) = min_one_u32(field, v) {
                return Err(e);
            }
        }
        if let Some(list) = &self.cuda_graph_bs {
            if list.is_empty() {
                return Err(CoreError::Config(format!(
                    "{where_}.cuda_graph_bs must list at least one batch size, got []"
                )));
            }
            if let Some(bad) = list.iter().find(|bs| **bs == 0) {
                return Err(CoreError::Config(format!(
                    "{where_}.cuda_graph_bs batch sizes must be >= 1, got {bad}"
                )));
            }
        }
        Ok(())
    }
}

/// mistral.rs `serve` tuning knobs. All-`Option` on purpose: `None`
/// never emits a flag (upstream default applies); emission is
/// manifest-flag-gated at profile-compile with the usual warn-skip.
/// Grammar mirrors the installed v0.9.3 `mistralrs serve --help`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MistralrsTuning {
    /// `--max-batch-size`: max batch for automatic device mapping
    /// (upstream default 1).
    pub max_batch_size: Option<u64>,
    /// `--max-prefill-chunk-tokens`: CUDA prompt-token quantum while
    /// decode is resident (upstream default 512).
    pub max_prefill_chunk_tokens: Option<u64>,
    /// `--max-decode-steps-before-prefill`: decode steps admitted
    /// before a waiting prefill batch (upstream default 8; higher
    /// favors token throughput, lower favors TTFT).
    pub max_decode_steps_before_prefill: Option<u64>,
    /// `--prefix-cache-n`: prefix cache entries (0 disables; upstream
    /// default 16).
    pub prefix_cache_n: Option<u64>,
    /// `--pa-block-size`: tokens per paged-KV block (upstream default
    /// 32 on CUDA).
    pub pa_block_size: Option<u32>,
    /// `--pa-cache-type`: paged KV quantization (upstream default
    /// `auto`; e.g. `f16`, `q8_0`).
    pub pa_cache_type: Option<String>,
    /// `--pa-context-len`: allocate paged KV for this context length
    /// instead of a VRAM fraction.
    pub pa_context_len: Option<u64>,
    /// `--lora-max-rank`: rank ceiling for served `LoRA` adapters
    /// (upstream default 256).
    pub lora_max_rank: Option<u32>,
    /// `--lora-max-adapters`: loaded `LoRA` aliases and resident adapter
    /// generations (upstream default 16).
    pub lora_max_adapters: Option<u32>,
    /// `--lora-max-bytes`: memory cap for loaded adapters (upstream
    /// default 8 GiB).
    pub lora_max_bytes: Option<u64>,
    /// `--mtp`: MTP speculative decoding with the head built into the
    /// checkpoint (qwen3.5-class MTP rows).
    pub mtp: Option<bool>,
    /// `--mtp-model`: MTP assistant model id or path (external head).
    pub mtp_model: Option<String>,
    /// `--mtp-n-predict`: draft tokens proposed per target step.
    pub mtp_n_predict: Option<u32>,
    /// `--mtp-draft-sampling` (auto, greedy, probabilistic).
    pub mtp_draft_sampling: Option<String>,
    /// `--encoder-cache-memory-mb`: MiB cap for the multimodal encoder
    /// cache.
    pub encoder_cache_memory_mb: Option<u64>,
    /// `--max-num-images`: images per request.
    pub max_num_images: Option<u32>,
    /// `--max-image-length`: max image dimension for device mapping.
    pub max_image_length: Option<u32>,
    /// `--disable-metrics`: turn the child's Prometheus recorder off
    /// (default on upstream).
    pub disable_metrics: Option<bool>,
    /// `--disable-access-log`: turn the child's HTTP access log off.
    pub disable_access_log: Option<bool>,
    /// `--device-layers`: layer mapping `ORD:NUM;...` (e.g.
    /// `0:10;1:20`). Single-GPU boxes get a teaching warning.
    pub device_layers: Option<String>,
}

impl MistralrsTuning {
    /// Per-model resolution: override replaces global (not merges),
    /// mirroring `SglangTuning::effective`.
    #[must_use]
    pub fn effective(
        override_: Option<&MistralrsTuning>,
        global: &MistralrsTuning,
    ) -> MistralrsTuning {
        override_.cloned().unwrap_or_else(|| global.clone())
    }

    fn validate(&self, where_: &str) -> Result<(), CoreError> {
        let min_one = |field: &str, zero: bool| -> Option<CoreError> {
            zero.then(|| CoreError::Config(format!("{where_}.{field} must be >= 1, got 0")))
        };
        for (field, zero) in [
            ("max_batch_size", self.max_batch_size == Some(0)),
            (
                "max_prefill_chunk_tokens",
                self.max_prefill_chunk_tokens == Some(0),
            ),
            (
                "max_decode_steps_before_prefill",
                self.max_decode_steps_before_prefill == Some(0),
            ),
            ("pa_block_size", self.pa_block_size == Some(0)),
            ("pa_context_len", self.pa_context_len == Some(0)),
            ("lora_max_rank", self.lora_max_rank == Some(0)),
            ("lora_max_adapters", self.lora_max_adapters == Some(0)),
            ("lora_max_bytes", self.lora_max_bytes == Some(0)),
            ("mtp_n_predict", self.mtp_n_predict == Some(0)),
            (
                "encoder_cache_memory_mb",
                self.encoder_cache_memory_mb == Some(0),
            ),
            ("max_num_images", self.max_num_images == Some(0)),
            ("max_image_length", self.max_image_length == Some(0)),
        ] {
            if let Some(e) = min_one(field, zero) {
                return Err(e);
            }
        }
        // prefix_cache_n: 0 is a valid "disable" — only negative is
        // impossible in u64, so no floor check.
        if let Some(s) = &self.mtp_draft_sampling {
            if !["auto", "greedy", "probabilistic"].contains(&s.as_str()) {
                return Err(CoreError::Config(format!(
                    "{where_}.mtp_draft_sampling must be one of [\"auto\", \"greedy\", \"probabilistic\"], got {s:?}"
                )));
            }
        }
        if let Some(layers) = &self.device_layers {
            let pair = |p: &str| -> bool {
                let mut parts = p.split(':');
                match (parts.next(), parts.next(), parts.next()) {
                    (Some(a), Some(b), None) => {
                        let digits =
                            |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit());
                        digits(a) && digits(b)
                    }
                    _ => false,
                }
            };
            if layers.is_empty() || !layers.split(';').all(pair) {
                return Err(CoreError::Config(format!(
                    "{where_}.device_layers must be ORD:NUM pairs joined by ';' (e.g. \"0:10;1:20\"), got {layers:?}"
                )));
            }
        }
        if self.mtp_model.is_some() && self.mtp != Some(true) {
            // Not an error: the model pin is inert without --mtp —
            // surface as a config-time footgun catch.
            return Err(CoreError::Config(format!(
                "{where_}.mtp_model set but mtp is not true — the MTP head stays inactive; set mtp = true or drop mtp_model"
            )));
        }
        Ok(())
    }
}

/// Deserializes `MmprojPolicy` from either the bool spellings the
/// suppress knob shipped with (`mmproj = false`) or the string forms
/// (`"attach"` / `"skip"` / `"lazy"`).
fn de_mmproj<'de, D>(deserializer: D) -> Result<Option<MmprojPolicy>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum Raw {
        Bool(bool),
        Word(MmprojPolicy),
    }
    match Option::<Raw>::deserialize(deserializer)? {
        None => Ok(None),
        Some(Raw::Bool(b)) => Ok(Some(MmprojPolicy::from(b))),
        Some(Raw::Word(p)) => Ok(Some(p)),
    }
}

/// Model-level sampling defaults, compiled to `--temp`, `--top-k`, ...
/// argv flags. Body params still win per request: these are defaults,
/// not caps. Field names follow the OpenAI/ollama spelling users know.
/// `mirostat` accepts 0 (off), 1, or 2. No mirostat tau/eta: the engine
/// server manifest does not expose those flags.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplerDefaults {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_n_sigma: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub typical_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeat_last_n: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dry_multiplier: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dry_base: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dry_allowed_length: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dry_penalty_last_n: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub xtc_probability: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub xtc_threshold: Option<f64>,
    /// 0 = off, 1 = Mirostat, 2 = Mirostat 2.0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mirostat: Option<i32>,
    /// -1 = random seed each request (upstream default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
}

impl SamplerDefaults {
    /// Range checks mirroring upstream sampler semantics: probabilities
    /// within [0,1], temperatures/multipliers finite and non-negative,
    /// 1|1|1
    /// frequency penalties accept negative values (they reward rather
    /// than penalize), so only finiteness is checked there.
    pub fn validate(&self, ctx: &str) -> CoreResult<()> {
        let prob = |field: &str, v: f64| -> CoreResult<()> {
            if v.is_nan() || !(0.0..=1.0).contains(&v) {
                Err(CoreError::Config(format!(
                    "{ctx}.{field} must be within [0.0, 1.0], got {v}"
                )))
            } else {
                Ok(())
            }
        };
        let nonneg = |field: &str, v: f64| -> CoreResult<()> {
            if v.is_nan() || v < 0.0 {
                Err(CoreError::Config(format!(
                    "{ctx}.{field} must be >= 0.0, got {v}"
                )))
            } else {
                Ok(())
            }
        };
        let count = |field: &str, v: i32| -> CoreResult<()> {
            if v < 0 {
                Err(CoreError::Config(format!(
                    "{ctx}.{field} must be >= 0, got {v}"
                )))
            } else {
                Ok(())
            }
        };
        if let Some(v) = self.temperature {
            nonneg("temperature", v)?;
        }
        if let Some(v) = self.top_k {
            count("top_k", v)?;
        }
        if let Some(v) = self.top_p {
            prob("top_p", v)?;
        }
        if let Some(v) = self.min_p {
            prob("min_p", v)?;
        }
        if let Some(v) = self.top_n_sigma {
            nonneg("top_n_sigma", v)?;
        }
        if let Some(v) = self.typical_p {
            prob("typical_p", v)?;
        }
        if let Some(v) = self.repeat_penalty {
            nonneg("repeat_penalty", v)?;
        }
        if let Some(v) = self.repeat_last_n {
            count("repeat_last_n", v)?;
        }
        for field in ["presence_penalty", "frequency_penalty"] {
            let v = if field == "presence_penalty" {
                self.presence_penalty
            } else {
                self.frequency_penalty
            };
            if let Some(v) = v {
                if !v.is_finite() {
                    return Err(CoreError::Config(format!(
                        "{ctx}.{field} must be finite, got {v}"
                    )));
                }
            }
        }
        if let Some(v) = self.dry_multiplier {
            nonneg("dry_multiplier", v)?;
        }
        if let Some(v) = self.dry_base {
            if v.is_nan() || v <= 1.0 {
                return Err(CoreError::Config(format!(
                    "{ctx}.dry_base must be > 1.0, got {v}"
                )));
            }
        }
        if let Some(v) = self.dry_allowed_length {
            count("dry_allowed_length", v)?;
        }
        if let Some(v) = self.dry_penalty_last_n {
            count("dry_penalty_last_n", v)?;
        }
        if let Some(v) = self.xtc_probability {
            prob("xtc_probability", v)?;
        }
        if let Some(v) = self.xtc_threshold {
            nonneg("xtc_threshold", v)?;
        }
        if let Some(v) = self.mirostat {
            if !(0..=2).contains(&v) {
                return Err(CoreError::Config(format!(
                    "{ctx}.mirostat must be 0 (off), 1 or 2, got {v}"
                )));
            }
        }
        Ok(())
    }
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
    /// Scheduling weight under contention (B5): when requests from
    /// multiple keys queue for the same slot class, admission share is
    /// proportional to weight. 1 = default fair share; 0 is invalid
    /// (treated as 1); 10 = ~10x the admission rate of a weight-1 key
    /// in the same priority class. Rate limits still apply on top.
    #[serde(default)]
    pub weight: u32,
}

impl ApiKey {
    /// Scheduling weight clamped to >= 1 (B5): an unset or 0 weight is
    /// a plain fair share, never a division-by-zero footgun.
    #[must_use]
    pub fn effective_weight(&self) -> u32 {
        self.weight.max(1)
    }
}

fn default_late_chunking_max_tokens() -> usize {
    8192
}

fn default_session_keep_secs() -> u64 {
    900
}

fn default_semantic_ttl_secs() -> u64 {
    600
}

fn default_semantic_threshold() -> f64 {
    0.90
}

fn default_semantic_max_entries() -> usize {
    256
}

/// `[semantic_cache]` — opt-in semantic response cache (R4). The `model`
/// is a registered embedding-capable model used to embed prompts; the
/// cache stores L2-normalized prompt vectors and serves stored responses
/// when cosine similarity meets `threshold` for the same chat model and
/// API key on the same lane.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticCacheConfig {
    /// Master switch. Never default-on (correctness risk).
    #[serde(default)]
    pub enabled: bool,
    /// Registered model used to embed prompts (must spawn `--embeddings`,
    /// e.g. an embedding GGUF or a `late_chunking` override). Required
    /// when enabled.
    pub model: Option<String>,
    /// Entry lifetime in seconds (per-request `x-pallama-cache-ttl`
    /// overrides, 1..=86400).
    #[serde(default = "default_semantic_ttl_secs")]
    pub ttl_secs: u64,
    /// Cosine similarity gate in (0, 1] (per-request
    /// `x-pallama-cache-threshold` overrides).
    #[serde(default = "default_semantic_threshold")]
    pub threshold: f64,
    /// Maximum entries (LRU eviction past the cap).
    #[serde(default = "default_semantic_max_entries")]
    pub max_entries: usize,
}

impl Default for SemanticCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: None,
            ttl_secs: default_semantic_ttl_secs(),
            threshold: default_semantic_threshold(),
            max_entries: default_semantic_max_entries(),
        }
    }
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
            child_auth: None,
            log_level: None,
            update_channel: UpdateChannel::default(),
            engine_asset: "auto".to_string(),
            download_connections: 8,
            router_max_models: 0,
            late_chunking_max_tokens: default_late_chunking_max_tokens(),
            session_keep_secs: default_session_keep_secs(),
            semantic_cache: SemanticCacheConfig::default(),
            devices: Vec::new(),
            engine_check_secs: default_engine_check_secs(),
            auto_restart_engine_switch: false,
            mistralrs_pa_memory_fraction: None,
            mistralrs_paged_attn: None,
            mistralrs: MistralrsTuning::default(),
            sglang: SglangTuning::default(),
            spec: "auto".to_string(),
            lazy_mode: "auto".to_string(),
            server_tools: None,
            server_tools_runtime: None,
            mcp_servers_config: None,
            mcp_servers_json: None,
            cache_reuse: 0,
            keys: Vec::new(),
            rpc_servers: String::new(),
            cache_ram_mb: 8192,
            slots: default_slots(),
            deterministic: false,
            slot_prompt_similarity: 0.0,
            sentinel: true,
            sentinel_stall_secs: 30,
            sentinel_enforce: false,
            audit_log: false,
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
            warm_after_spawn: true,
            ctx_extend: 0.0,
            cpu_moe_n: 0,
            cpu_ffn_n: 0,
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
            ngram_mod_n_match: 0,
            ngram_mod_n_max: 0,
            ngram_mod_n_min: 0,
            reasoning_budget: default_reasoning_budget(),
            reasoning_budget_message: String::new(),
            reasoning_effort: String::new(),
            reasoning: String::new(),
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
            sse_ping_interval: None,
            server_timeout_secs: None,
            chat_template_kwargs: None,
            cont_batching: None,
            reuse_port: false,
            lora_init_without_apply: false,
            warmup: true,
            mmproj_policy: None,
            repack: true,
            cache_idle_slots: true,
            lookup_cache_static: None,
            lookup_cache_dynamic: None,
            predictive_preload: false,
            adaptive_slots: true,
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
            batch_size: 0,
            ubatch_size: 0,
            threads_batch: 0,
            main_gpu: -1,
            split_mode: String::new(),
            tensor_split: String::new(),
            models_autoload: None,
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

/// Persist a `config.toml` body safely. Every config write path routes
/// through here so nothing can lose user settings silently:
/// - previous file backed up to `config.toml.bak-<unix-ms>` (5 newest
///   kept, older pruned) — an uninstall/reinstall cycle or operator
///   mistake is recoverable;
/// - write is temp-file + atomic rename (a crash mid-write never leaves
///   a truncated config behind).
pub fn persist_config(path: &std::path::Path, body: &str) -> CoreResult<()> {
    if path.exists() {
        let ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis());
        if let Some(dir) = path.parent() {
            let mut baks: Vec<_> = std::fs::read_dir(dir)?
                .filter_map(std::result::Result::ok)
                .filter(|e| {
                    e.file_name()
                        .to_string_lossy()
                        .starts_with("config.toml.bak-")
                })
                .collect();
            // Timestamp names sort lexically = chronologically.
            baks.sort_by_key(std::fs::DirEntry::file_name);
            while baks.len() >= 5 {
                let _ = std::fs::remove_file(baks.remove(0).path());
            }
        }
        let bak = path.with_file_name(format!("config.toml.bak-{ms}"));
        std::fs::copy(path, &bak)?;
    }
    let tmp = path.with_file_name("config.toml.tmp-write");
    std::fs::write(&tmp, body)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
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
            persist_config(&path, &cfg.to_toml()?)?;
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

    /// Effective lazy-mode for a model: overlay wins over global default.
    #[must_use]
    pub fn effective_lazy_mode(&self, model: &str) -> &str {
        match self
            .model_overrides
            .get(model)
            .and_then(|o| o.lazy_mode.as_deref())
        {
            Some(m) => m,
            None => self.lazy_mode.as_str(),
        }
    }

    /// Effective late-chunking mode for a model: overlay wins (default off).
    #[must_use]
    pub fn effective_late_chunking(&self, model: &str) -> bool {
        self.overlay_for(model).late_chunking.unwrap_or(false)
    }

    /// Effective `--rpc` server list for a model (C6): a non-empty
    /// overlay replaces the global list; empty/absent inherits it.
    #[must_use]
    pub fn effective_rpc_servers(&self, model: &str) -> &str {
        // Borrow the stored overlay (overlay_for returns an owned clone —
        // borrowing that temporary would dangle).
        let overlay = self
            .model_overrides
            .get(model)
            .and_then(|o| o.rpc_servers.as_deref())
            .map(str::trim)
            .filter(|s| !s.is_empty());
        overlay.unwrap_or(self.rpc_servers.trim())
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

    /// Config pins whose value mirrors a RETIRED default: the file was
    /// written when the old value was the default, the default has
    /// since changed, and the pin now silently keeps the old behavior.
    /// Returns one human-readable line per offending pin (empty = none)
    /// for `pallama doctor` to surface as a WARN — a deliberate pin is
    /// legitimate, the point is visibility.
    ///
    /// MAINTENANCE CONTRACT: whenever a default value changes, add a
    /// check here naming the knob, the retired value and the new
    /// default (both global and `model_overrides` spellings when the
    /// knob has an overlay field).
    #[must_use]
    pub fn retired_default_pins(&self) -> Vec<String> {
        let mut pins = Vec::new();
        if self.cache_reuse == 256 {
            pins.push("cache_reuse = 256 (a retired default; current default: 0)".to_string());
        }
        if self.spec == "off" {
            pins.push("spec = \"off\" (a retired default; current default: \"auto\")".to_string());
        }
        for (model, overlay) in &self.model_overrides {
            if overlay.spec.as_deref() == Some("off") {
                pins.push(format!(
                    "model_overrides.\"{model}\".spec = \"off\" (a retired default; current default: \"auto\")"
                ));
            }
        }
        pins
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
        // F114: Some(0.0) is a VALID explicit off (validate teaches
        // "0 (off)") — it must override the global, not fall through it.
        o.ctx_extend.unwrap_or(self.ctx_extend)
    }

    /// Effective `MoE` CPU-offload expert count; overlay wins over global.
    #[must_use]
    pub fn effective_cpu_moe_n(&self, model: &str) -> i32 {
        let o = self.overlay_for(model);
        // F114: Some(0) is a VALID explicit off — override, not inherit.
        o.cpu_moe_n.unwrap_or(self.cpu_moe_n)
    }

    /// Effective dense-`FFN` CPU-offload layer count; overlay wins over
    /// global.
    #[must_use]
    pub fn effective_cpu_ffn_n(&self, model: &str) -> i32 {
        let o = self.overlay_for(model);
        // F114: Some(0) is a VALID explicit off — override, not inherit.
        o.cpu_ffn_n.unwrap_or(self.cpu_ffn_n)
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

    /// Effective server-side reasoning switch: overlay beats global.
    #[must_use]
    pub fn effective_reasoning(&self, model: &str) -> &str {
        self.model_overrides
            .get(model)
            .and_then(|o| o.reasoning.as_deref())
            .unwrap_or(self.reasoning.as_str())
    }

    /// Cross-field sanity. Violations are config errors, not warnings:
    /// fail fast rather than run with contradictory knobs.
    #[allow(clippy::too_many_lines)] // flat one-check-per-knob by design
    pub fn validate(&self) -> CoreResult<()> {
        if self.default_ctx == 0 {
            return Err(CoreError::Config("default_ctx must be > 0".into()));
        }
        self.validate_keys()?;
        let sc = &self.semantic_cache;
        if sc.enabled && sc.model.as_deref().unwrap_or_default().is_empty() {
            return Err(CoreError::Config(
                "semantic_cache.enabled requires semantic_cache.model (a registered embedding-capable model)"
                    .into(),
            ));
        }
        // F115: 0.0 must be rejected — the header path enforces
        // 0.01..=1.0 and 0.0 would make nearly everything a "hit".
        if sc.enabled && !(sc.threshold > 0.0 && sc.threshold <= 1.0) {
            return Err(CoreError::Config(format!(
                "semantic_cache.threshold must be in (0, 1], got {}",
                sc.threshold
            )));
        }
        if sc.enabled && sc.ttl_secs == 0 {
            return Err(CoreError::Config(
                "semantic_cache.ttl_secs must be > 0 when enabled".into(),
            ));
        }
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
        if !(1..=32).contains(&self.download_connections) {
            return Err(CoreError::Config(format!(
                "download_connections must be within 1..=32 (1 = single-stream resume), got {}",
                self.download_connections
            )));
        }
        match self.child_transport.as_str() {
            // F35: "unix" was accepted by validation but no lane dials a
            // UDS — every request would 500. Fail loud at config load
            // instead of at first request.
            "tcp" => {}
            "unix" => {
                return Err(CoreError::Config(
                    "child_transport = \"unix\" is not implemented yet — use \"tcp\"".to_string(),
                ));
            }
            other => {
                return Err(CoreError::Config(format!(
                    "child_transport must be \"tcp\" or \"unix\", got {other:?}"
                )))
            }
        }
        match self.spec.as_str() {
            "off" | "auto" | "ngram" | "ngram-map-k" | "ngram-map-k4v" | "ngram-mod"
            | "ngram-cache" | "mtp" | "eagle3" | "dflash" | "dspark" => {}
            other => {
                return Err(CoreError::Config(format!(
                    "spec must be \"off\", \"auto\", \"ngram\", \"ngram-map-k\", \"ngram-map-k4v\", \"ngram-mod\", \"ngram-cache\", \"mtp\", \"eagle3\", \"dflash\" or \"dspark\", got {other:?}"
                )))
            }
        }
        if !matches!(self.lazy_mode.as_str(), "auto" | "on" | "off") {
            return Err(CoreError::Config(format!(
                "lazy_mode must be \"auto\", \"on\" or \"off\", got {:?}",
                self.lazy_mode
            )));
        }
        // Experimental upstream agent-tooling surface. Global-only by
        // design: a security posture must not vary silently per model.
        if let Some(rt) = &self.server_tools_runtime {
            let ok = [
                "docker:",
                "podman:",
                "docker-container:",
                "podman-container:",
                "ssh:",
            ]
            .iter()
            .any(|p| rt.starts_with(p) && rt.len() > p.len());
            if !ok {
                return Err(CoreError::Config(format!(
                    "server_tools_runtime must be docker:<image>, podman:<image>, \
                     docker-container:<id>, podman-container:<id> or ssh:<target>, got {rt:?}"
                )));
            }
            if self.server_tools.is_none() {
                return Err(CoreError::Config(
                    "server_tools_runtime is set but server_tools is not — a runtime \
                     without tools is meaningless; set server_tools (or drop the runtime)"
                        .into(),
                ));
            }
        }
        if let Some(json) = &self.mcp_servers_json {
            if serde_json::from_str::<serde_json::Value>(json).is_err() {
                return Err(CoreError::Config(
                    "mcp_servers_json must be valid JSON (Cursor-compatible MCP \
                     server definitions), parse failed"
                        .into(),
                ));
            }
        }
        if self.mcp_servers_config.is_some() && self.mcp_servers_json.is_some() {
            return Err(CoreError::Config(
                "mcp_servers_config and mcp_servers_json are mutually exclusive — \
                 pick the file path or the inline JSON"
                    .into(),
            ));
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
        match self.reasoning.as_str() {
            "" | "on" | "off" | "auto" => {}
            other => {
                return Err(CoreError::Config(format!(
                    "reasoning must be \"on\", \"off\" or \"auto\", got {other:?}"
                )))
            }
        }
        if let Some(frac) = self.mistralrs_pa_memory_fraction {
            if !(0.05..=0.95).contains(&frac) {
                return Err(CoreError::Config(format!(
                    "mistralrs_pa_memory_fraction must be 0.05..=0.95, got {frac}"
                )));
            }
        }
        self.sglang.validate("sglang")?;
        self.mistralrs.validate("mistralrs")?;
        for (name, o) in &self.model_overrides {
            if let Some(t) = &o.sglang {
                t.validate(&format!("model_overrides.{name}.sglang"))?;
            }
            if let Some(t) = &o.mistralrs {
                t.validate(&format!("model_overrides.{name}.mistralrs"))?;
            }
        }
        // Child HTTP server behavior knobs.
        if let Some(p) = self.sse_ping_interval {
            if p < -1 {
                return Err(CoreError::Config(format!(
                    "sse_ping_interval must be >= -1 (-1 disables), got {p}"
                )));
            }
        }
        if let Some(t) = self.server_timeout_secs {
            if t == 0 {
                return Err(CoreError::Config(
                    "server_timeout_secs must be >= 1, got 0".into(),
                ));
            }
        }
        if let Some(kwargs) = &self.chat_template_kwargs {
            let trimmed = kwargs.trim();
            if trimmed.is_empty() {
                return Err(CoreError::Config(
                    "chat_template_kwargs must be a JSON object string, got \"\"".into(),
                ));
            }
            if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
                return Err(CoreError::Config(format!(
                    "chat_template_kwargs must be a JSON object string (e.g. \"{{\\\"enable_thinking\\\": false}}\"), got {kwargs:?}"
                )));
            }
        }
        if self.host.trim().is_empty() {
            return Err(CoreError::Config("host must not be empty".into()));
        }
        self.validate_new_knobs()?;
        self.validate_wire_knobs()?;
        for (name, o) in &self.model_overrides {
            if let Some(spec) = &o.spec {
                if !matches!(
                    spec.as_str(),
                    "off"
                        | "auto"
                        | "ngram"
                        | "ngram-map-k"
                        | "ngram-map-k4v"
                        | "ngram-mod"
                        | "ngram-cache"
                        | "mtp"
                        | "eagle3"
                        | "dflash"
                        | "dspark"
                ) {
                    return Err(CoreError::Config(format!(
                        "model_overrides.{name}.spec must be \"off\", \"auto\", \"ngram\", \"ngram-map-k\", \"ngram-map-k4v\", \"ngram-mod\", \"ngram-cache\", \"mtp\", \"eagle3\", \"dflash\" or \"dspark\", got {spec:?}"
                    )));
                }
            }
            if let Some(lm) = &o.lazy_mode {
                if !matches!(lm.as_str(), "auto" | "on" | "off") {
                    return Err(CoreError::Config(format!(
                        "model_overrides.{name}.lazy_mode must be \"auto\", \"on\" or \"off\", got {lm:?}"
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
            if let Some(n) = o.cpu_ffn_n {
                if n < 0 {
                    return Err(CoreError::Config(format!(
                        "model_overrides.{name}.cpu_ffn_n must be >= 0, got {n}"
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
            if o.chat_template.is_some() && o.chat_template_file.is_some() {
                return Err(CoreError::Config(format!(
                    "model_overrides.{name}: set only one of chat_template or chat_template_file"
                )));
            }
            if let Some(f) = &o.chat_template_file {
                if !std::path::Path::new(f).exists() {
                    return Err(CoreError::Config(format!(
                        "model_overrides.{name}.chat_template_file not found: {f:?}"
                    )));
                }
            }
            if let Some(sd) = &o.sampler_defaults {
                sd.validate(&format!("model_overrides.{name}.sampler_defaults"))?;
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
        if self.cpu_ffn_n < 0 {
            return Err(CoreError::Config(format!(
                "cpu_ffn_n must be >= 0, got {}",
                self.cpu_ffn_n
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
        // Upstream clamps ngram-mod knobs to [0,1024] and rejects a zero
        // lookup length (arg.cpp: n_min/n_max 0..=1024, n_match 1..=1024).
        if self.ngram_mod_n_match > 1024 {
            return Err(CoreError::Config(format!(
                "ngram_mod_n_match must be 0 (engine default) or 1..=1024, got {}",
                self.ngram_mod_n_match
            )));
        }
        if self.ngram_mod_n_max > 1024 {
            return Err(CoreError::Config(format!(
                "ngram_mod_n_max must be 0 (engine default) or within 0..=1024, got {}",
                self.ngram_mod_n_max
            )));
        }
        if self.ngram_mod_n_min > 1024 {
            return Err(CoreError::Config(format!(
                "ngram_mod_n_min must be 0 (engine default) or within 0..=1024, got {}",
                self.ngram_mod_n_min
            )));
        }
        if self.main_gpu < -1 {
            return Err(CoreError::Config(format!(
                "main_gpu must be >= -1 (-1 = engine default), got {}",
                self.main_gpu
            )));
        }
        if !self.split_mode.is_empty()
            && !matches!(
                self.split_mode.as_str(),
                "none" | "layer" | "row" | "tensor"
            )
        {
            return Err(CoreError::Config(format!(
                "split_mode must be one of none|layer|row|tensor (or empty), got {:?}",
                self.split_mode
            )));
        }
        if !self.tensor_split.is_empty() {
            for part in self.tensor_split.split(',') {
                let ok = part
                    .trim()
                    .parse::<f64>()
                    .is_ok_and(|v| v.is_finite() && v > 0.0);
                if !ok {
                    return Err(CoreError::Config(format!(
                        "tensor_split entries must be positive numbers (e.g. \"3,1\"), got {part:?}"
                    )));
                }
            }
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
            if let Some(r) = o.reasoning.as_deref() {
                let r = r.trim();
                if !matches!(r, "on" | "off" | "auto" | "") {
                    return Err(CoreError::Config(format!(
                        "model_overrides.{name}.reasoning must be \"on\", \"off\" or \"auto\", got {r:?}"
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
        if let Some(v) = env("PALLAMA_DOWNLOAD_CONNECTIONS") {
            cfg.download_connections = parse_u32("PALLAMA_DOWNLOAD_CONNECTIONS", &v)?;
        }
        if let Some(v) = env("PALLAMA_SPEC") {
            cfg.spec = v;
        }
        if let Some(v) = env("PALLAMA_LAZY_MODE") {
            cfg.lazy_mode = v;
        }
        if let Some(v) = env("PALLAMA_SERVER_TOOLS") {
            cfg.server_tools = Some(v);
        }
        if let Some(v) = env("PALLAMA_SERVER_TOOLS_RUNTIME") {
            cfg.server_tools_runtime = Some(v);
        }
        if let Some(v) = env("PALLAMA_MCP_SERVERS_CONFIG") {
            cfg.mcp_servers_config = Some(v);
        }
        if let Some(v) = env("PALLAMA_MCP_SERVERS_JSON") {
            cfg.mcp_servers_json = Some(v);
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
        if let Some(v) = env("PALLAMA_CPU_FFN_N") {
            cfg.cpu_ffn_n = parse_i32("PALLAMA_CPU_FFN_N", &v)?;
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
        if let Some(v) = env("PALLAMA_AUDIT_LOG") {
            cfg.audit_log = parse_bool("PALLAMA_AUDIT_LOG", &v)?;
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

fn default_stall_secs() -> u64 {
    30
}

/// `lo-hi` decimal CPU range with lo <= hi (upstream `--cpu-range` syntax).
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
    0
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
    fn unit__retired_pins__default_config_has_none() {
        assert!(Config::default().retired_default_pins().is_empty());
    }

    #[test]
    fn unit__retired_pins__each_retired_global_value_fires() {
        let pins = Config {
            cache_reuse: 256,
            spec: "off".into(),
            ..Config::default()
        }
        .retired_default_pins();
        assert_eq!(pins.len(), 2, "{pins:?}");
        assert!(
            pins.iter().any(|p| p.contains("cache_reuse = 256")),
            "{pins:?}"
        );
        assert!(
            pins.iter().any(|p| p.contains("spec = \"off\"")),
            "{pins:?}"
        );
        // Current-default values never fire, explicit or not.
        assert!(Config {
            cache_reuse: 0,
            spec: "auto".into(),
            ..Config::default()
        }
        .retired_default_pins()
        .is_empty());
    }

    #[test]
    fn unit__retired_pins__overlay_spec_names_the_model() {
        let mut cfg = Config::default();
        cfg.model_overrides.insert(
            "qwen3.5-9b".into(),
            ModelOverride {
                spec: Some("off".into()),
                ..ModelOverride::default()
            },
        );
        let pins = cfg.retired_default_pins();
        assert_eq!(pins.len(), 1, "{pins:?}");
        assert!(pins[0].contains("qwen3.5-9b"), "{}", pins[0]);
        assert!(pins[0].contains("spec"), "{}", pins[0]);
    }

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
        // Server-side reasoning switch: tri-state vocabulary, empty = off.
        for bad in ["enabled", "true", "0"] {
            let cfg = Config {
                reasoning: bad.into(),
                ..Default::default()
            };
            assert!(
                cfg.validate().is_err(),
                "reasoning {bad:?} must be rejected"
            );
        }
        for good in ["", "on", "off", "auto"] {
            Config {
                reasoning: good.into(),
                ..Default::default()
            }
            .validate()
            .unwrap_or_else(|e| panic!("reasoning {good:?} must validate: {e}"));
        }
        let mut overrides = std::collections::BTreeMap::new();
        overrides.insert(
            "m1".to_string(),
            ModelOverride {
                reasoning: Some("yes".into()),
                ..Default::default()
            },
        );
        assert!(
            Config {
                model_overrides: overrides,
                ..Default::default()
            }
            .validate()
            .is_err(),
            "overlay reasoning vocabulary must be enforced"
        );
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
    #[allow(clippy::too_many_lines)] // manifest snapshot: one cohesive table
    fn unit__full_manifest_roundtrip_toml__stable() {
        // Golden contract: a config that sets a representative value on
        // EVERY knob family (Options, containers, model override + sampler
        // defaults) must survive to_toml -> from_toml identically, and the
        // serialization itself must be byte-stable (second pass equal).
        // Mirrors scripts/validate.py gate (c) at the type level.
        let lks = std::env::temp_dir().join("pallama-gold-lks.bin");
        let lkd = std::env::temp_dir().join("pallama-gold-lkd.bin");
        std::fs::write(&lks, b"lk").unwrap();
        std::fs::write(&lkd, b"lk").unwrap();
        let raw = format!(
            r#"
host = "127.0.0.1"
port = 11499
default_ctx = 2048
idle_sleep_secs = 77
cache_reuse = 128
poll = 77
reasoning_format = "deepseek"
slot_prompt_similarity = 0.6
cpu_range = "0-7"
cpu_moe_n = 2
cpu_ffn_n = 1
override_tensor = [".ffn_.*_exps.=CPU"]
spec = "off"
kv_unified_per_slot = 4096
swa_full = true
ctx_checkpoints = 16
no_kv_offload = true
load_mode = "mlock"
agent = true
cache_idle_slots = false
warmup = false
image_max_tokens = 4096
image_min_tokens = 64
embd_normalize = 2
threads_http = 2
no_host = true
op_offload = true
sessions = true
batch_size = 512
ubatch_size = 256
threads_batch = 2
main_gpu = 0
split_mode = "layer"
tensor_split = "3,1"
prio = 2
prio_batch = 2
ctx_extend = 2.0
yarn_orig_ctx = 4096
yarn_ext_factor = 1.5
yarn_attn_factor = 1.75
child_auth = true
lookup_cache_static = "{lks}"
lookup_cache_dynamic = "{lkd}"
models_autoload = false
poll_batch = true
spec_draft_p_min = 0.1
spec_draft_p_split = 0.1
spec_draft_poll = 2
spec_draft_poll_batch = true
reasoning_preserve = true

[[keys]]
name = "gatekey"
key = "plm-gate"
models = []

[[remotes]]
name = "edge"
url = "http://127.0.0.1:1/v1"
key = "rk"

[engine_env]
PROBE = "marker"

[model_overrides.m]
ctx = 3072
slots = 1
spec = "ngram"
loras = []
extra_args = ["--probe"]
cache_type = "q8_0"
kv_unified = true
ctx_extend = 2.0
cpu_moe_n = 1
override_tensor = [".ffn_.*_exps.=CPU"]
devices = ["Vulkan1"]
warmup = false
reasoning_budget = 512
reasoning_effort = "low"
replicas = 1
pin = true
chat_template = "chatml"
spm_infill = true

[model_overrides.m.sampler_defaults]
temperature = 0.7
top_k = 40
top_p = 0.9
min_p = 0.05
top_n_sigma = 0.0
typical_p = 1.0
repeat_penalty = 1.1
repeat_last_n = 64
presence_penalty = 0.0
frequency_penalty = 0.0
dry_multiplier = 0.8
dry_base = 1.75
dry_allowed_length = 2
dry_penalty_last_n = 256
xtc_probability = 0.0
xtc_threshold = 0.1
mirostat = 0
seed = 42
"#,
            lks = lks.display(),
            lkd = lkd.display(),
        );
        let cfg = Config::from_toml(&raw).unwrap();
        let out1 = cfg.to_toml().unwrap();
        let back = Config::from_toml(&out1).unwrap();
        assert_eq!(cfg, back, "struct round-trip must be lossless");
        let out2 = back.to_toml().unwrap();
        assert_eq!(out1, out2, "toml serialization must be byte-stable");
        assert!(out1.contains("plm-gate"));
        assert!(out1.contains("chat_template = \"chatml\""));
        let _ = std::fs::remove_file(&lks);
        let _ = std::fs::remove_file(&lkd);
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

        // Typed n-gram variants accepted at both scopes ("ngram" stays the
        // only shorthand; "ngram-simple" remains rejected below).
        for spec in ["ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache"] {
            Config::from_toml(&format!("spec = \"{spec}\"\n")).unwrap();
            Config::from_toml(&format!("[model_overrides.m]\nspec = \"{spec}\"\n")).unwrap();
        }
        assert!(Config::from_toml("spec = \"ngram-map\"\n").is_err());

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
    fn unit__sglang_knobs__validated() {
        // Accepted: the full post-28-flag knob surface parses + validates.
        Config::from_toml(
            "[sglang]\ngrammar_backend = \"xgrammar\"\ntokenizer_mode = \"slow\"\n\
             tokenizer_backend = \"fastokens\"\nradix_eviction_policy = \"lfu\"\n\
             sleep_on_idle = true\nmemory_saver = true\nsession_radix_cache = true\n\
             mixed_chunk = true\ncache_report = true\ndynamic_batch_tokenizer = true\n\
             tokenizer_worker_num = 2\ndetokenizer_worker_num = 2\n\
             dynamic_batch_tokenizer_batch_size = 8\n\
             dynamic_batch_tokenizer_batch_timeout = 0.01\nbatch_notify_size = 32\n\
             scheduler_recv_interval = 2\nwatchdog_timeout = 300.0\n\
             cuda_graph_bs = [1, 2, 4]\nmax_total_tokens = 65536\n\
             tp_size = 2\nmax_lora_rank = 64\nlora_backend = \"pytorch\"\n",
        )
        .unwrap();

        // Choice rejections name the field and the allowed set.
        for (field, raw) in [
            ("grammar_backend", "[sglang]\ngrammar_backend = \"regex\"\n"),
            ("tokenizer_mode", "[sglang]\ntokenizer_mode = \"turbo\"\n"),
            (
                "tokenizer_backend",
                "[sglang]\ntokenizer_backend = \"slowmatic\"\n",
            ),
            (
                "radix_eviction_policy",
                "[sglang]\nradix_eviction_policy = \"fifo\"\n",
            ),
        ] {
            let err = Config::from_toml(raw).unwrap_err().to_string();
            assert!(err.contains(field), "{field}: {err}");
            assert!(err.contains("must be one of"), "{field}: {err}");
        }

        // Numeric floors + list sanity, global scope.
        for (field, raw) in [
            ("batch_notify_size", "[sglang]\nbatch_notify_size = 0\n"),
            (
                "scheduler_recv_interval",
                "[sglang]\nscheduler_recv_interval = 0\n",
            ),
            ("watchdog_timeout", "[sglang]\nwatchdog_timeout = 0.0\n"),
            ("max_total_tokens", "[sglang]\nmax_total_tokens = 0\n"),
            ("cuda_graph_bs", "[sglang]\ncuda_graph_bs = []\n"),
            ("cuda_graph_bs", "[sglang]\ncuda_graph_bs = [1, 0]\n"),
        ] {
            let err = Config::from_toml(raw).unwrap_err().to_string();
            assert!(err.contains(field), "{field}: {err}");
        }

        // Per-overlay scope validates with the full where_ path.
        let err = Config::from_toml("[model_overrides.m.sglang]\ntokenizer_mode = \"turbo\"\n")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("model_overrides.m.sglang.tokenizer_mode"),
            "{err}"
        );
    }

    #[test]
    fn unit__llama_server_knobs__validated() {
        // Full llama-server surface parses at both scopes.
        Config::from_toml(
            "sse_ping_interval = -1\nserver_timeout_secs = 600\n\
             chat_template_kwargs = \"{\\\"enable_thinking\\\": false}\"\n\
             cont_batching = false\nreuse_port = true\nlora_init_without_apply = true\n",
        )
        .unwrap();
        Config::from_toml("cont_batching = true\nsse_ping_interval = 30\n").unwrap();

        // -1 (disable) is the floor for the SSE ping; 0-second server timeout
        // would kill every stream; the kwargs string must be a JSON object.
        for (field, raw) in [
            ("sse_ping_interval", "sse_ping_interval = -2\n"),
            ("server_timeout_secs", "server_timeout_secs = 0\n"),
            ("chat_template_kwargs", "chat_template_kwargs = \"[]\"\n"),
            (
                "chat_template_kwargs",
                "chat_template_kwargs = \"not json\"\n",
            ),
        ] {
            let err = Config::from_toml(raw).unwrap_err().to_string();
            assert!(err.contains(field), "{field}: {err}");
        }
    }

    #[test]
    fn unit__mistralrs_knobs__validated() {
        // Accepted: the full 20-knob surface parses + validates, globally and
        // per-overlay.
        let full = "[mistralrs]\nmax_batch_size = 4\nmax_prefill_chunk_tokens = 1024\n\
                    max_decode_steps_before_prefill = 16\nprefix_cache_n = 0\n\
                    pa_block_size = 64\npa_cache_type = \"bf16\"\npa_context_len = 4096\n\
                    lora_max_rank = 64\nlora_max_adapters = 2\nlora_max_bytes = 1073741824\n\
                    mtp = true\nmtp_model = \"mtp-draft\"\nmtp_n_predict = 3\n\
                    mtp_draft_sampling = \"greedy\"\nencoder_cache_memory_mb = 512\n\
                    max_num_images = 2\nmax_image_length = 1024\ndisable_metrics = true\n\
                    disable_access_log = true\ndevice_layers = \"0:12;1:24\"\n";
        Config::from_toml(full).unwrap();
        Config::from_toml("[model_overrides.m.mistralrs]\nmax_batch_size = 2\n").unwrap();

        // Floors name the field.
        for (field, raw) in [
            ("max_batch_size", "[mistralrs]\nmax_batch_size = 0\n"),
            (
                "prefix_cache_n_is_exempt",
                "[mistralrs]\nprefix_cache_n = 0\n",
            ),
            ("pa_block_size", "[mistralrs]\npa_block_size = 0\n"),
            ("lora_max_rank", "[mistralrs]\nlora_max_rank = 0\n"),
            ("mtp_n_predict", "[mistralrs]\nmtp_n_predict = 0\n"),
        ] {
            let ok = field == "prefix_cache_n_is_exempt";
            let res = Config::from_toml(raw);
            assert_eq!(res.is_ok(), ok, "{field}: {res:?}");
        }

        // Draft-sampling choices + the mtp_model-without-mtp footgun + the
        // device_layers ORD:NUM grammar.
        for (field, raw) in [
            (
                "mtp_draft_sampling",
                "[mistralrs]\nmtp_draft_sampling = \"chaotic\"\n",
            ),
            ("mtp_model", "[mistralrs]\nmtp_model = \"orphan-draft\"\n"),
            ("device_layers", "[mistralrs]\ndevice_layers = \"0-12\"\n"),
            ("device_layers", "[mistralrs]\ndevice_layers = \"all\"\n"),
        ] {
            let err = Config::from_toml(raw).unwrap_err().to_string();
            assert!(err.contains(field), "{field}: {err}");
        }
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
        assert_eq!(cfg.effective_spec("qwen3-coder-30b"), "auto");
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
    fn unit__validation__mtp_spec_accepted_both_scopes() {
        let mut cfg = Config {
            spec: "mtp".into(),
            ..Config::default()
        };
        cfg.model_overrides.insert(
            "m".into(),
            ModelOverride {
                spec: Some("mtp".into()),
                ..Default::default()
            },
        );
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn unit__validation__server_tools_runtime_shape_and_pairing() {
        // Well-formed runtime + tools = ok.
        let cfg = Config {
            server_tools: Some("all".into()),
            server_tools_runtime: Some("ssh:gpu-box".into()),
            ..Config::default()
        };
        assert!(cfg.validate().is_ok());
        // Bad prefix rejected.
        let cfg = Config {
            server_tools: Some("all".into()),
            server_tools_runtime: Some("jail:strict".into()),
            ..Config::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("server_tools_runtime"), "{err}");
        // Bare prefix (no value after colon) rejected.
        let cfg = Config {
            server_tools: Some("all".into()),
            server_tools_runtime: Some("docker:".into()),
            ..Config::default()
        };
        assert!(cfg.validate().is_err());
        // Runtime without tools = meaningless, rejected.
        let cfg = Config {
            server_tools_runtime: Some("ssh:gpu-box".into()),
            ..Config::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("server_tools"), "{err}");
    }

    #[test]
    fn unit__validation__mcp_json_syntax_and_exclusivity() {
        let cfg = Config {
            mcp_servers_json: Some(r#"{"servers": {"fs": {"command": "x"}}}"#.into()),
            ..Config::default()
        };
        assert!(cfg.validate().is_ok());
        let cfg = Config {
            mcp_servers_json: Some("{not json".into()),
            ..Config::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("mcp_servers_json"), "{err}");
        let cfg = Config {
            mcp_servers_config: Some("/tmp/mcp.json".into()),
            mcp_servers_json: Some("{}".into()),
            ..Config::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("mutually exclusive"), "{err}");
    }

    #[test]
    fn unit__validation__eagle3_spec_accepted_both_scopes() {
        let mut cfg = Config {
            spec: "eagle3".into(),
            ..Config::default()
        };
        cfg.model_overrides.insert(
            "m".into(),
            ModelOverride {
                spec: Some("eagle3".into()),
                ..Default::default()
            },
        );
        assert!(cfg.validate().is_ok());
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
            cpu_ffn_n: 1,
            override_tensor: vec!["global=CPU".into()],
            model_overrides: BTreeMap::from([(
                "m1".into(),
                ModelOverride {
                    cache_type: Some("q5_0".into()),
                    ctx_extend: Some(4.0),
                    cpu_moe_n: Some(8),
                    cpu_ffn_n: Some(2),
                    override_tensor: Some(vec!["local=GPU".into()]),
                    ..ModelOverride::default()
                },
            )]),
            ..Config::default()
        };
        assert_eq!(c.effective_cache_type("m1"), "q5_0");
        assert_eq!(c.effective_ctx_extend("m1").to_bits(), 4.0_f64.to_bits());
        assert_eq!(c.effective_cpu_moe_n("m1"), 8);
        assert_eq!(c.effective_cpu_ffn_n("m1"), 2);
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

    /// Test-only resolver (F118: the live auth path is gateway keys.rs
    /// `ct_eq`, constant-time — this plain `==` must never be reachable
    /// from request handling).
    impl Config {
        fn key_for(&self, presented: &str) -> Option<&ApiKey> {
            self.keys.iter().find(|k| k.key == presented)
        }
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
        // Same env-leak guard as the file-load test below.
        let _g = env_lock();
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
        // Config::load applies env overrides; hold the env lock so the
        // parallel env-override tests can't leak PALLAMA_PORT into us.
        let _g = env_lock();
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

    #[test]
    fn unit__log_level__absent_and_present() {
        let absent = Config::from_toml("port = 9999\n").unwrap();
        assert_eq!(absent.log_level, None);
        let present =
            Config::from_toml("port = 9999\nlog_level = \"pallama=trace,pallama::engine=debug\"\n")
                .unwrap();
        assert_eq!(
            present.log_level.as_deref(),
            Some("pallama=trace,pallama::engine=debug")
        );
    }

    #[test]
    fn unit__update_channel__absent_stable_invalid() {
        // Absent = latest (zero behavior change for existing configs).
        let absent = Config::from_toml("port = 9999\n").unwrap();
        assert_eq!(absent.update_channel, UpdateChannel::Latest);
        // Explicit stable parses.
        let stable = Config::from_toml("port = 9999\nupdate_channel = \"stable\"\n").unwrap();
        assert_eq!(stable.update_channel, UpdateChannel::Stable);
        let latest = Config::from_toml("port = 9999\nupdate_channel = \"latest\"\n").unwrap();
        assert_eq!(latest.update_channel, UpdateChannel::Latest);
        // Typos fail loudly instead of silently falling back.
        assert!(Config::from_toml("port = 9999\nupdate_channel = \"nightly\"\n").is_err());
    }

    #[test]
    fn unit__overlay_chat_template__xor_and_missing_file_rejected() {
        let base = || Config {
            model_overrides: BTreeMap::from([("m".to_string(), ModelOverride::default())]),
            ..Config::default()
        };
        let with = |o: ModelOverride| {
            let mut c = base();
            c.model_overrides.insert("m".to_string(), o);
            c
        };
        let both = ModelOverride {
            chat_template: Some("{{}}".into()),
            chat_template_file: Some("/x.tpl".into()),
            ..Default::default()
        };
        let err = with(both).validate().unwrap_err();
        assert!(
            err.to_string().contains("only one of chat_template"),
            "{err}"
        );

        let missing = ModelOverride {
            chat_template_file: Some("/definitely/not/present.tpl".into()),
            ..Default::default()
        };
        let err = with(missing).validate().unwrap_err();
        assert!(
            err.to_string().contains("/definitely/not/present.tpl"),
            "{err}"
        );

        let tpl = std::env::temp_dir().join("pallama-tpl-test.j2");
        std::fs::write(&tpl, "{{ message }}").unwrap();
        let good = ModelOverride {
            chat_template_file: Some(tpl.display().to_string()),
            ..Default::default()
        };
        with(good).validate().unwrap();
        let _ = std::fs::remove_file(&tpl);
    }

    #[test]
    fn unit__gpu_split_knobs__validated() {
        let c = Config {
            split_mode: "diagonal".into(),
            ..Config::default()
        };
        let err = c.validate().unwrap_err();
        assert!(err.to_string().contains("split_mode"), "{err}");

        let c = Config {
            tensor_split: "3,zero,1".into(),
            ..Config::default()
        };
        let err = c.validate().unwrap_err();
        assert!(err.to_string().contains("tensor_split"), "{err}");

        let c = Config {
            tensor_split: "3,-1".into(),
            ..Config::default()
        };
        let err = c.validate().unwrap_err();
        assert!(err.to_string().contains("tensor_split"), "{err}");

        let c = Config {
            main_gpu: -2,
            ..Config::default()
        };
        let err = c.validate().unwrap_err();
        assert!(err.to_string().contains("main_gpu"), "{err}");

        let c = Config {
            split_mode: "tensor".into(),
            tensor_split: "3,1".into(),
            main_gpu: 0,
            ..Config::default()
        };
        c.validate().unwrap();
    }

    #[test]
    fn unit__ngram_mod_knobs__validated() {
        for (field, value) in [
            ("ngram_mod_n_match", 1025u32),
            ("ngram_mod_n_max", 2000),
            ("ngram_mod_n_min", 4096),
        ] {
            let mut c = Config::default();
            match field {
                "ngram_mod_n_match" => c.ngram_mod_n_match = value,
                "ngram_mod_n_max" => c.ngram_mod_n_max = value,
                _ => c.ngram_mod_n_min = value,
            }
            let err = c.validate().unwrap_err();
            assert!(err.to_string().contains(field), "{field}: {err}");
        }
        let c = Config {
            ngram_mod_n_match: 24,
            ngram_mod_n_max: 64,
            ngram_mod_n_min: 48,
            ..Config::default()
        };
        c.validate().unwrap();
    }

    #[test]
    #[allow(clippy::too_many_lines)] // one assertion per sampler field, flat table
    fn unit__sampler_defaults__range_validation() {
        let cfg = |sd: SamplerDefaults| {
            let mut c = Config::default();
            c.model_overrides.insert(
                "m".to_string(),
                ModelOverride {
                    sampler_defaults: Some(sd),
                    ..Default::default()
                },
            );
            c.validate()
        };
        // Valid extremes pass: bounded probabilities, off-values, any seed.
        cfg(SamplerDefaults {
            temperature: Some(0.0),
            top_p: Some(1.0),
            min_p: Some(0.0),
            typical_p: Some(1.0),
            presence_penalty: Some(-2.0),
            frequency_penalty: Some(2.0),
            mirostat: Some(2),
            seed: Some(-1),
            ..Default::default()
        })
        .unwrap();

        for (field, sd) in [
            (
                "top_p",
                SamplerDefaults {
                    top_p: Some(1.5),
                    ..Default::default()
                },
            ),
            (
                "min_p",
                SamplerDefaults {
                    min_p: Some(-0.1),
                    ..Default::default()
                },
            ),
            (
                "typical_p",
                SamplerDefaults {
                    typical_p: Some(2.0),
                    ..Default::default()
                },
            ),
            (
                "temperature",
                SamplerDefaults {
                    temperature: Some(-1.0),
                    ..Default::default()
                },
            ),
            (
                "top_k",
                SamplerDefaults {
                    top_k: Some(-5),
                    ..Default::default()
                },
            ),
            (
                "repeat_penalty",
                SamplerDefaults {
                    repeat_penalty: Some(-0.5),
                    ..Default::default()
                },
            ),
            (
                "repeat_last_n",
                SamplerDefaults {
                    repeat_last_n: Some(-1),
                    ..Default::default()
                },
            ),
            (
                "dry_multiplier",
                SamplerDefaults {
                    dry_multiplier: Some(-1.0),
                    ..Default::default()
                },
            ),
            (
                "dry_base",
                SamplerDefaults {
                    dry_base: Some(1.0),
                    ..Default::default()
                },
            ),
            (
                "dry_allowed_length",
                SamplerDefaults {
                    dry_allowed_length: Some(-2),
                    ..Default::default()
                },
            ),
            (
                "dry_penalty_last_n",
                SamplerDefaults {
                    dry_penalty_last_n: Some(-1),
                    ..Default::default()
                },
            ),
            (
                "xtc_probability",
                SamplerDefaults {
                    xtc_probability: Some(1.2),
                    ..Default::default()
                },
            ),
            (
                "xtc_threshold",
                SamplerDefaults {
                    xtc_threshold: Some(-0.5),
                    ..Default::default()
                },
            ),
            (
                "mirostat",
                SamplerDefaults {
                    mirostat: Some(3),
                    ..Default::default()
                },
            ),
        ] {
            let err = cfg(sd).unwrap_err();
            assert!(
                err.to_string().contains(field),
                "error must name {field}: {err}"
            );
        }
    }

    #[test]
    fn unit__sampler_defaults__unknown_field_rejected_by_serde() {
        let err = serde_json::from_str::<SamplerDefaults>("{\"temp\": 0.5}");
        assert!(err.is_err(), "typo'd field must hard-error, not ignore");
        let sd: SamplerDefaults = serde_json::from_str(
            "{\"temperature\": 0.7, \"top_k\": 40, \"mirostat\": 1, \"seed\": -1}",
        )
        .unwrap();
        assert_eq!(sd.temperature, Some(0.7));
        assert_eq!(sd.mirostat, Some(1));
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

#[cfg(test)]
#[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
mod persist_tests {
    use super::persist_config;

    #[test]
    fn unit__persist_config__backs_up_keeps_five_atomic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        for i in 0..7 {
            persist_config(&path, &format!("port = 1143{i}\n")).unwrap();
        }
        // Final body wins atomically; no temp residue.
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "port = 11436\n");
        assert!(
            !path.with_file_name("config.toml.tmp-write").exists(),
            "temp file must be renamed away"
        );
        // Backups exist and never exceed keep-5 (same-ms writes may
        // collapse onto one name, so only bound the range).
        let baks = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(std::result::Result::ok)
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with("config.toml.bak-")
            })
            .count();
        assert!((1..=5).contains(&baks), "baks = {baks}");
    }
}
