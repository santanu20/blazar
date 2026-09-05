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
    /// Empty = no auth (loopback default). Non-empty = Bearer required.
    pub api_keys: Vec<String>,
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
    /// Extra env for engine children + probes (e.g. `GGML_BACKEND_PATH` for
    /// a local CUDA build).
    #[serde(default)]
    pub engine_env: BTreeMap<String, String>,
    pub model_overrides: BTreeMap<String, ModelOverride>,
}

/// Per-model config overlay. Unknown keys are a hard error at parse time —
/// a typo'd knob must never be silently ignored.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelOverride {
    pub ctx: Option<u32>,
    pub spec: Option<String>,
    pub loras: Option<Vec<String>>,
    /// Extra llama-server args, validated against the engine capability
    /// manifest at profile-compile time (unknown flag = error).
    pub extra_args: Option<Vec<String>>,
    /// Per-model KV cache type ("" or None = inherit the global ladder).
    pub cache_type: Option<String>,
    /// Per-model `YaRN` context-extension factor (None = inherit global).
    pub ctx_extend: Option<f64>,
    /// Per-model `MoE` expert CPU-offload count (None = inherit global).
    pub cpu_moe_n: Option<i32>,
    /// Per-model `--override-tensor` entries; replaces (not merges) the
    /// global list for this model.
    pub override_tensor: Option<Vec<String>>,
}

impl Default for Config {
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
            spec: "off".to_string(),
            cache_reuse: 256,
            api_keys: Vec::new(),
            rpc_servers: String::new(),
            cache_ram_mb: 8192,
            slots: 1,
            slot_prompt_similarity: 0.0,
            sentinel: true,
            sentinel_stall_secs: 30,
            sentinel_enforce: false,
            cache_type: String::new(),
            spec_cache: true,
            ctx_extend: 0.0,
            cpu_moe_n: 0,
            override_tensor: Vec::new(),
            agent: false,
            sessions: true,
            router: false,
            router_max_models: 0,
            engine_env: BTreeMap::new(),
            model_overrides: BTreeMap::new(),
        }
    }
}

impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", toml::to_string_pretty(self).map_err(|_| fmt::Error)?)
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
        let cfg: Config =
            toml::from_str(raw).map_err(|e| CoreError::Config(format!("parse: {e}")))?;
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
        o.ctx_extend.filter(|v| *v != 0.0).unwrap_or(self.ctx_extend)
    }

    /// Effective `MoE` CPU-offload expert count; overlay wins over global.
    #[must_use]
    pub fn effective_cpu_moe_n(&self, model: &str) -> i32 {
        let o = self.overlay_for(model);
        o.cpu_moe_n.filter(|v| *v > 0).unwrap_or(self.cpu_moe_n)
    }

    /// Effective `--override-tensor` entries; overlay list replaces the
    /// global list when present.
    #[must_use]
    pub fn effective_override_tensor(&self, model: &str) -> &[String] {
        if let Some(list) = self.model_overrides.get(model).and_then(|o| o.override_tensor.as_ref()) {
            return list.as_slice();
        }
        self.override_tensor.as_slice()
    }

    /// Cross-field sanity. Violations are config errors, not warnings:
    /// fail fast rather than run with contradictory knobs.
    pub fn validate(&self) -> CoreResult<()> {
        if self.default_ctx == 0 {
            return Err(CoreError::Config("default_ctx must be > 0".into()));
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
        if self.sentinel && self.sentinel_stall_secs != 0 && !(5..=600).contains(&self.sentinel_stall_secs) {
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

    /// Apply documented `PALLAMA_*` env overrides on top of file values.
    /// Env wins over file; each override fails fast on a malformed value.
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
        if let Some(v) = env("PALLAMA_API_KEYS") {
            cfg.api_keys = v.split(',').map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).collect();
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
            cfg.override_tensor =
                v.split(',').map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).collect();
        }

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

/// KV cache types upstream accepts for K and V (`-ctk`/`-ctv`).
pub const CACHE_TYPES: &[&str] = &[
    "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
];

fn valid_cache_type(s: &str) -> bool {
    s.is_empty() || CACHE_TYPES.contains(&s)
}

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
            assert!(Config::from_toml(&format!("cpu_range = \"{bad}\"\n")).is_err(), "{bad}");
        }
        Config::from_toml("cpu_range = \"0-15\"\n").unwrap();

        assert!(Config::from_toml("poll = 101\n").is_err());
        Config::from_toml("poll = 100\n").unwrap();

        assert!(Config::from_toml("reasoning_format = \"bogus\"\n").is_err());
        Config::from_toml("reasoning_format = \"deepseek\"\n").unwrap();
        assert!(Config::from_toml("spec = \"ngram-simple\"\n").is_err(), "raw spec types are not config values");
    }

    #[test]
    fn unit__unknown_top_level_key__config_error() {
        let raw = "port = 1234\nbogus_knob = 3\n";
        let err = Config::from_toml(raw).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("bogus_knob"), "error must name the unknown key: {msg}");
    }

    #[test]
    fn unit__overlay_unknown_key__config_error() {
        let raw = "[model_overrides.m]\nctx = 8192\nwrong = true\n";
        let err = Config::from_toml(raw).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("wrong"), "error must name the unknown overlay key: {msg}");
    }

    #[test]
    fn unit__overlay_merge__overlay_wins() {
        let mut cfg = Config::default();
        cfg.model_overrides.insert(
            "qwen3-coder-30b".into(),
            ModelOverride { ctx: Some(32768), ..Default::default() },
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
            ModelOverride { spec: Some("turbo".into()), ..Default::default() },
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
        assert!(msg.contains("PALLAMA_PORT") && msg.contains("not-a-port"), "{msg}");
    }

    #[test]
    fn unit__new_knobs__defaults_are_conservative() {
        let c = Config::default();
        assert_eq!(c.cache_type, ""); // auto ladder
        assert!(c.spec_cache);
        assert_eq!(c.ctx_extend.to_bits(), 0.0_f64.to_bits(), "default ctx_extend not zero");
        assert_eq!(c.cpu_moe_n, 0);
        assert!(c.override_tensor.is_empty());
        assert!(!c.agent);
        assert!(c.sessions);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn unit__cache_type__vocabulary_enforced() {
        let ok = Config { cache_type: "q4_0".into(), ..Config::default() };
        assert!(ok.validate().is_ok());
        let bad = Config { cache_type: "q9_9".into(), ..Config::default() };
        let err = bad.validate().unwrap_err().to_string();
        assert!(err.contains("cache_type") && err.contains("q8_0"), "{err}");
    }

    #[test]
    fn unit__ctx_extend__range_enforced() {
        for ok in [0.0, 2.0, 16.0, 32.0] {
            let c = Config { ctx_extend: ok, ..Config::default() };
            assert!(c.validate().is_ok(), "{ok} should validate");
        }
        for bad in [0.5, 1.0, 33.0, -2.0] {
            let c = Config { ctx_extend: bad, ..Config::default() };
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
            let c = Config { override_tensor: vec![bad.into()], ..Config::default() };
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
        assert_eq!(c.effective_override_tensor("m1"), &["local=GPU".to_string()]);
        // other models fall through to globals
        assert_eq!(c.effective_cache_type("m2"), "q8_0");
        assert_eq!(c.effective_override_tensor("m2"), &["global=CPU".to_string()]);
    }

    #[test]
    fn unit__api_keys_env__comma_split() {
        let _g = env_lock();
        std::env::set_var("PALLAMA_API_KEYS", " k1 , k2 ,");
        let cfg = Config::default().with_env_overrides().unwrap();
        std::env::remove_var("PALLAMA_API_KEYS");
        assert_eq!(cfg.api_keys, vec!["k1".to_string(), "k2".to_string()]);
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
        assert!(dirs.config_file().exists(), "default config must be written");
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
        ENV_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}
