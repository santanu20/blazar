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
