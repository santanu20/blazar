use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::dirs::PallamaDirs;
use crate::error::{CoreError, CoreResult};

/// Tunables exposed in config.toml. One file, one surface: every Pallama
/// knob lives here or in a per-model overlay; the documented `PALLAMA_*`
/// env vars override file values. Secrets (HF_TOKEN / GH_TOKEN) are env-only
/// and never persisted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
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
}

impl Default for Config {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 11434,
            default_ctx: 16384,
            idle_sleep_secs: 300,
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
    /// parse time (deny_unknown_fields), so no re-validation here.
    pub fn overlay_for(&self, model: &str) -> ModelOverride {
        self.model_overrides.get(model).cloned().unwrap_or_default()
    }

    /// Effective ctx for a model: overlay wins over global default.
    pub fn effective_ctx(&self, model: &str) -> u32 {
        let o = self.overlay_for(model);
        o.ctx.unwrap_or(self.default_ctx)
    }

    /// Effective spec mode for a model: overlay wins over global default.
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
            "off" | "auto" => {}
            other => {
                return Err(CoreError::Config(format!(
                    "spec must be \"off\" or \"auto\", got {other:?}"
                )))
            }
        }
        if self.host.trim().is_empty() {
            return Err(CoreError::Config("host must not be empty".into()));
        }
        for (name, o) in &self.model_overrides {
            if let Some(spec) = &o.spec {
                if spec != "off" && spec != "auto" {
                    return Err(CoreError::Config(format!(
                        "model_overrides.{name}.spec must be \"off\" or \"auto\", got {spec:?}"
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
        let mut cfg = Config::default();
        cfg.idle_sleep_secs = 600;
        cfg.idle_timeout_secs = 300;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("idle_timeout_secs"));
    }

    #[test]
    fn unit__validation__bad_spec_mode_rejected() {
        let mut cfg = Config::default();
        cfg.spec = "maybe".into();
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
        let mut cfg = Config::default();
        cfg.port = 1;
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
        ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner())
    }
}
