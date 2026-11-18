//! Engine-neutral KV cache contract types.
//!
//! These describe WHAT may be shared between serving instances: the
//! model computation identity (weights, tokenizer, adapters, `RoPE`
//! configuration) and the memory layout the engine materialized for
//! it. Transport and engine provenance live with the connectors in
//! `blazar-runtime::kv`, not here - a v0.9 llama.cpp bank and a v0.10
//! one share the same model identity but different provenance.
//!
//! The compatibility policy over these types is deliberately strict:
//! cross-engine sharing requires the same model computation, and the
//! burden of proof sits on a validated conversion, never on hope.

use serde::{Deserialize, Serialize};

/// Everything that must match for two engines to agree they computed
/// the same key/value tensors for a shared prefix.
///
/// `model_sha256` refers to the model artifact (GGUF/safetensors bundle)
/// as recorded in the store's model row. `tokenizer_sha256` and
/// `model_config_sha256` pin the preprocessing that produced the token
/// stream - a tokenizer drift silently changes every prefix. `rope_fingerprint`
/// summarizes positional encoding parameters that make cached positions
/// meaningless when they differ.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvModelIdentity {
    pub model_sha256: String,
    pub tokenizer_sha256: String,
    /// Adapter (`LoRA`) digest when the cache was produced under an
    /// adapter; `None` for base-model caches.
    pub adapter_sha256: Option<String>,
    pub model_config_sha256: String,
    pub rope_fingerprint: String,
}

/// How one engine laid the cache out in memory. Two layouts for the
/// same [`KvModelIdentity`] may still be sharable when a converter
/// exists - that decision belongs to the compatibility policy, not to
/// string-comparing engine names.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvLayout {
    pub layers: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub key_dtype: String,
    pub value_dtype: String,
    pub block_tokens: u32,
    pub tensor_parallel: u32,
    pub pipeline_parallel: u32,
    pub attention_layout: String,
}

/// The verdict of comparing an available cache against a target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvCompatibility {
    /// Byte-compatible (or engine-defined equivalent): restore as-is.
    Exact,
    /// Sharable after a registered, validated conversion.
    Transformable { converter_id: String },
    /// Do not share; the string teaches why.
    Incompatible { reason: String },
}

/// What one connector can honestly do. Registries report these; they
/// never simulate an unsupported operation.
#[allow(clippy::struct_excessive_bools)] // capability flags by definition
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCapabilities {
    pub export: bool,
    pub import: bool,
    pub persistent: bool,
    pub prefix_blocks: bool,
    pub cross_process: bool,
    pub supported_formats: Vec<String>,
}

/// Errors from KV connector operations. `Unsupported` is a capability
/// statement, not a failure - callers fall back to cold prefill.
#[derive(Debug, thiserror::Error)]
pub enum KvError {
    #[error("engine connector does not support this operation: {0}")]
    Unsupported(String),
    #[error("kv transport failed: {0}")]
    Transport(String),
    #[error("kv identity mismatch: {0}")]
    Identity(String),
    #[error("kv engine rejected the request: {0}")]
    Engine(String),
}

impl KvModelIdentity {
    /// An identity is only exact-shareable when every digest that the
    /// compatibility policy treats as mandatory is present.
    #[must_use]
    pub fn complete(&self) -> bool {
        !self.model_sha256.is_empty()
            && !self.tokenizer_sha256.is_empty()
            && !self.model_config_sha256.is_empty()
            && !self.rope_fingerprint.is_empty()
    }
}

/// Prefix-coverage snapshot for one candidate route: how much of the
/// incoming prompt a cache with a compatible identity already holds.
///
/// The gateway's measured reuse (`cached_tokens` in usage) is the truth
/// this decision input is built from; routing layers consume the score
/// to prefer the instance whose cache covers the request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KvRouteScore {
    /// Prompt tokens the request will process.
    pub prompt_tokens: u64,
    /// Tokens a compatible cache holds as a prefix of that prompt
    /// (never more than `prompt_tokens`; callers clamp, readers trust).
    pub cached_prefix_tokens: u64,
    /// Compatibility-policy verdict for the cache behind this route:
    /// only an `Exact`/`Transformable` identity may feed the ratio.
    pub identity_compatible: bool,
}

impl KvRouteScore {
    /// Cached-prefix fraction in `0.0..=1.0`. `0.0` when the identity
    /// did not match or there is no prompt to cover.
    #[allow(clippy::cast_precision_loss)] // token counts live far below 2^53
    #[must_use]
    pub fn ratio(&self) -> f64 {
        if !self.identity_compatible || self.prompt_tokens == 0 {
            return 0.0;
        }
        (self.cached_prefix_tokens as f64 / self.prompt_tokens as f64).min(1.0)
    }

    /// Prefer the cached route only when the identity matched and the
    /// covered fraction reaches `min_ratio` (`0.0..=1.0`, inclusive).
    /// An incompatible or empty cache never routes on - cold prefill is
    /// the fallback, not a gamble.
    #[must_use]
    pub fn should_use_cached_route(&self, min_ratio: f64) -> bool {
        self.ratio() >= min_ratio && self.identity_compatible
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // suite convention: unit__scenario__expected (6b)
    #[allow(non_snake_case)]
    fn unit__kv_route_score__full_coverage_ratio_one() {
        let s = KvRouteScore {
            prompt_tokens: 100,
            cached_prefix_tokens: 100,
            identity_compatible: true,
        };
        assert!((s.ratio() - 1.0).abs() < f64::EPSILON);
        assert!(s.should_use_cached_route(0.5));
        assert!(s.should_use_cached_route(1.0));
    }

    #[test]
    // suite convention: unit__scenario__expected (6b)
    #[allow(non_snake_case)]
    fn unit__kv_route_score__partial_coverage_meets_threshold_at_boundary() {
        let s = KvRouteScore {
            prompt_tokens: 200,
            cached_prefix_tokens: 100,
            identity_compatible: true,
        };
        assert!((s.ratio() - 0.5).abs() < f64::EPSILON);
        // Inclusive boundary: exactly-at-threshold routes on.
        assert!(s.should_use_cached_route(0.5));
        assert!(!s.should_use_cached_route(0.51));
    }

    #[test]
    // suite convention: unit__scenario__expected (6b)
    #[allow(non_snake_case)]
    fn unit__kv_route_score__incompatible_identity_never_routes_on() {
        let s = KvRouteScore {
            prompt_tokens: 100,
            cached_prefix_tokens: 100,
            identity_compatible: false,
        };
        assert!((s.ratio() - 0.0).abs() < f64::EPSILON);
        assert!(!s.should_use_cached_route(0.0));
    }

    #[test]
    // suite convention: unit__scenario__expected (6b)
    #[allow(non_snake_case)]
    fn unit__kv_route_score__empty_prompt_and_overreport_clamp() {
        let empty = KvRouteScore {
            prompt_tokens: 0,
            cached_prefix_tokens: 0,
            identity_compatible: true,
        };
        assert!((empty.ratio() - 0.0).abs() < f64::EPSILON);
        assert!(!empty.should_use_cached_route(0.5));
        // A lying count above the prompt clamps to full coverage, never
        // beyond 1.0.
        let over = KvRouteScore {
            prompt_tokens: 10,
            cached_prefix_tokens: 50,
            identity_compatible: true,
        };
        assert!((over.ratio() - 1.0).abs() < f64::EPSILON);
    }
}
