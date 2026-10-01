//! HF safetensors model-directory metadata. Reads `config.json` (and the
//! shard index when present) from a pulled `models/<name>.d/` directory and
//! extracts exactly what the sglang profile compiler needs: architecture
//! identity, training context ceiling, dtype, quantization bits, and the KV
//! geometry (layers / kv-heads / head-dim) for VRAM estimation.
//!
//! Same contract as `gguf.rs`: anything absent stays `None` — downstream
//! rules must skip-with-warning, never estimate. The one derivation here
//! (`head_dim` from `hidden_size / num_attention_heads`) mirrors HF loader
//! semantics and `GgufMeta::derived_head_dim()`.

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::error::{CoreError, CoreResult};
use crate::gguf::GgufMeta;

/// KV-relevant transformer geometry from `config.json`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct KvGeom {
    /// `num_hidden_layers`, reduced to the full-attention layer count on
    /// hybrid-attention models (`full_attention_interval` >= 2): only
    /// those layers carry the context-scaling KV cache.
    pub layers: Option<u64>,
    /// `num_key_value_heads`; falls back to `num_attention_heads` (HF MHA
    /// semantics — absent GQA field means every attention head is a KV head).
    pub kv_heads: Option<u64>,
    /// Explicit `head_dim` when present; else derived `hidden_size /
    /// num_attention_heads`.
    pub head_dim: Option<u64>,
}

/// HF safetensors model metadata Blazar consumes.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct HfMeta {
    /// `architectures[0]` (e.g. `Qwen2ForCausalLM`), else `model_type`.
    pub architecture: String,
    /// `max_position_embeddings` — the training context ceiling.
    pub ctx_train: Option<u64>,
    /// `torch_dtype` / `dtype` (e.g. `bfloat16`).
    pub dtype: Option<String>,
    /// Effective weight bits when a `quantization_config` is present
    /// (`bits`, bnb `load_in_4bit`/`load_in_8bit`, fp8 → 8).
    pub quant_bits: Option<u8>,
    pub kv: KvGeom,
}

impl HfMeta {
    /// Bytes per KV element for the given cache dtype suffix
    /// (fp8 = 1, everything 16-bit = 2).
    #[must_use]
    pub fn kv_elem_bytes(cache_dtype: &str) -> u64 {
        if cache_dtype.starts_with("fp8") {
            1
        } else {
            2
        }
    }
}

/// Model metadata for either storage lane. The profile compiler branches on
/// this instead of assuming GGUF.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelMeta<'a> {
    Gguf(&'a GgufMeta),
    Hf(&'a HfMeta),
}

impl ModelMeta<'_> {
    #[must_use]
    pub fn architecture(&self) -> &str {
        match self {
            ModelMeta::Gguf(g) => &g.architecture,
            ModelMeta::Hf(h) => &h.architecture,
        }
    }

    /// Training context ceiling, if the metadata carries one.
    #[must_use]
    pub fn context_length(&self) -> Option<u64> {
        match self {
            ModelMeta::Gguf(g) => g.context_length,
            ModelMeta::Hf(h) => h.ctx_train,
        }
    }
}

fn u64_of(v: &Value, key: &str) -> Option<u64> {
    v.get(key).and_then(Value::as_u64)
}

fn str_of(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(Value::as_str).map(str::to_owned)
}

/// Per-key nested lookup for multimodal-wrapper configs: Qwen3.5,
/// Qwen3-VL, Qwen3-Next and Llama-4-class repos keep the LM geometry
/// inside `text_config`, leaving the top level to `architectures` and
/// the tower configs. Top-level keys keep absolute priority — only an
/// ABSENT key descends, so flat causal-LM configs behave identically.
fn u64_of_text_config(v: &Value, key: &str) -> Option<u64> {
    u64_of(v, key).or_else(|| v.get("text_config").and_then(|t| u64_of(t, key)))
}

fn str_of_text_config(v: &Value, key: &str) -> Option<String> {
    str_of(v, key).or_else(|| v.get("text_config").and_then(|t| str_of(t, key)))
}

/// Quantization bits from a HF `quantization_config` object. Only reads
/// fields the quantizers themselves define — no arch guessing.
fn quant_bits_of(q: &Value) -> Option<u8> {
    if let Some(bits) = q.get("bits").and_then(Value::as_u64) {
        return u8::try_from(bits).ok();
    }
    if q.get("load_in_4bit").and_then(Value::as_bool) == Some(true) {
        return Some(4);
    }
    if q.get("load_in_8bit").and_then(Value::as_bool) == Some(true) {
        return Some(8);
    }
    match q.get("quant_method").and_then(Value::as_str) {
        Some("fp8" | "mxfp8") => Some(8),
        Some("nvfp4" | "fp4") => Some(4),
        _ => None,
    }
}

/// Read `config.json` from a safetensors model directory.
///
/// Errors (missing dir / missing config / malformed JSON) are the caller's
/// signal that this is not a usable HF model directory — the supervisor
/// skips-with-teach, mirroring the GGUF read-failure path.
pub fn read_hf_config(dir: &Path) -> CoreResult<HfMeta> {
    let config_path = dir.join("config.json");
    let raw = std::fs::read_to_string(&config_path).map_err(|e| {
        CoreError::Config(format!("hf model dir {} unreadable: {e}", dir.display()))
    })?;
    let v: Value = serde_json::from_str(&raw)
        .map_err(|e| CoreError::Config(format!("{} config.json parse: {e}", dir.display())))?;

    let architecture = v
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| str_of(&v, "model_type"))
        .unwrap_or_default();

    let attn_heads = u64_of_text_config(&v, "num_attention_heads");
    let kv_heads = u64_of_text_config(&v, "num_key_value_heads").or(attn_heads);
    let head_dim = u64_of_text_config(&v, "head_dim").or_else(|| {
        let hidden = u64_of_text_config(&v, "hidden_size")?;
        let heads = attn_heads?;
        (hidden > 0 && heads > 0).then(|| hidden / heads)
    });

    // Hybrid attention (Qwen3.5 / Qwen3-Next GDN): only every
    // `full_attention_interval`-th layer runs full attention and carries
    // the context-scaling KV cache; the linear/GDN layers hold a constant
    // state that never grows with ctx. Counting all layers would
    // overestimate KV by the interval factor and misplace the fit ladder.
    let kv_layers = u64_of_text_config(&v, "num_hidden_layers").map(|layers| {
        match u64_of_text_config(&v, "full_attention_interval") {
            Some(interval) if interval >= 2 && layers > 0 => layers.div_ceil(interval),
            _ => layers,
        }
    });

    Ok(HfMeta {
        architecture,
        ctx_train: u64_of_text_config(&v, "max_position_embeddings"),
        dtype: str_of_text_config(&v, "torch_dtype").or_else(|| str_of_text_config(&v, "dtype")),
        quant_bits: v.get("quantization_config").and_then(quant_bits_of),
        kv: KvGeom {
            layers: kv_layers,
            kv_heads,
            head_dim,
        },
    })
}

/// Root-level `.safetensors` weights of an HF model dir — the pull
/// lane's selection rule (shards live at the root beside a mandatory
/// `config.json`). Empty when the dir is unreadable or holds no
/// weights; callers decide what an empty listing means.
#[must_use]
pub fn root_safetensors(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.file_name()
                    .is_some_and(|f| f.to_string_lossy().to_lowercase().ends_with(".safetensors"))
        })
        .collect();
    files.sort();
    files
}

/// Summed bytes of the root-level `.safetensors` weights, or `None`
/// when there are none — an empty sum is indistinguishable from "not a
/// model dir" (or a download in flight) and must never be written back
/// as a row size or fed to planning.
#[must_use]
pub fn root_safetensors_bytes(dir: &Path) -> Option<u64> {
    let weights = root_safetensors(dir);
    if weights.is_empty() {
        return None;
    }
    Some(
        weights
            .iter()
            .filter_map(|w| std::fs::metadata(w).ok())
            .map(|m| m.len())
            .sum(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write_dir(files: &[(&str, &str)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (name, body) in files {
            fs::write(dir.path().join(name), body).unwrap();
        }
        dir
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__hf_config_nested_text_config__geometry_descended() {
        // Qwen3.5-9B shape (live config.json): top level carries only
        // architectures/model_type + tower configs; geometry nests in
        // text_config, hybrid attention keeps KV in every 4th layer.
        let dir = write_dir(&[(
            "config.json",
            r#"{
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "text_config": {
                    "num_hidden_layers": 32,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 4,
                    "head_dim": 256,
                    "hidden_size": 4096,
                    "max_position_embeddings": 262144,
                    "full_attention_interval": 4,
                    "torch_dtype": "bfloat16"
                },
                "vision_config": {}
            }"#,
        )]);
        let m = read_hf_config(dir.path()).unwrap();
        assert_eq!(m.architecture, "Qwen3_5ForConditionalGeneration");
        assert_eq!(m.ctx_train, Some(262_144));
        assert_eq!(m.dtype.as_deref(), Some("bfloat16"));
        assert_eq!(m.kv.layers, Some(8), "32 layers / interval 4 = 8 KV layers");
        assert_eq!(m.kv.kv_heads, Some(4));
        assert_eq!(m.kv.head_dim, Some(256));
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__hf_config_full_attention_interval__layer_count_math() {
        // interval < 2 is "no hybrid" (every layer full-attention);
        // non-multiplicative layer counts round up (full-attn layer 0
        // plus every interval-th after it).
        let cases = [
            (
                r#"{"num_hidden_layers": 32, "full_attention_interval": 4}"#,
                Some(8),
            ),
            (
                r#"{"num_hidden_layers": 48, "full_attention_interval": 3}"#,
                Some(16),
            ),
            (
                r#"{"num_hidden_layers": 31, "full_attention_interval": 4}"#,
                Some(8),
            ),
            (
                r#"{"num_hidden_layers": 32, "full_attention_interval": 1}"#,
                Some(32),
            ),
            (r#"{"num_hidden_layers": 32}"#, Some(32)),
        ];
        for (body, expect) in cases {
            let dir = write_dir(&[("config.json", body)]);
            assert_eq!(
                read_hf_config(dir.path()).unwrap().kv.layers,
                expect,
                "body: {body}"
            );
        }
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__hf_config_nested_top_level_priority__flat_keys_win() {
        // A key present at BOTH levels resolves top-level first — the
        // wrapper object never overrides the LM section.
        let dir = write_dir(&[(
            "config.json",
            r#"{
                "max_position_embeddings": 4096,
                "text_config": {"max_position_embeddings": 262144}
            }"#,
        )]);
        assert_eq!(read_hf_config(dir.path()).unwrap().ctx_train, Some(4096));
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__root_safetensors__listing_and_bytes_sum() {
        let dir = write_dir(&[]);
        assert_eq!(root_safetensors(dir.path()).len(), 0);
        assert_eq!(root_safetensors_bytes(dir.path()), None);
        fs::write(dir.path().join("model-00001-of-00002.safetensors"), "aaaa").unwrap();
        fs::write(dir.path().join("model-00002-of-00002.safetensors"), "aa").unwrap();
        fs::write(dir.path().join("config.json"), "{}").unwrap();
        fs::create_dir(dir.path().join("nested.safetensors")).unwrap(); // dirs never count
        let files = root_safetensors(dir.path());
        assert_eq!(files.len(), 2);
        assert_eq!(root_safetensors_bytes(dir.path()), Some(6));
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__hf_config_qwen2__full_geometry() {
        let dir = write_dir(&[(
            "config.json",
            r#"{
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "max_position_embeddings": 32768,
                "torch_dtype": "bfloat16",
                "num_hidden_layers": 28,
                "num_attention_heads": 14,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "hidden_size": 1536
            }"#,
        )]);
        let m = read_hf_config(dir.path()).unwrap();
        assert_eq!(m.architecture, "Qwen2ForCausalLM");
        assert_eq!(m.ctx_train, Some(32768));
        assert_eq!(m.dtype.as_deref(), Some("bfloat16"));
        assert_eq!(m.kv.layers, Some(28));
        assert_eq!(m.kv.kv_heads, Some(2));
        assert_eq!(m.kv.head_dim, Some(128));
        assert_eq!(m.quant_bits, None);
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__hf_config_mha_derive__head_dim_from_hidden() {
        // No num_key_value_heads (MHA) and no explicit head_dim: kv_heads
        // falls back to attention heads, head_dim derives hidden/heads.
        let dir = write_dir(&[(
            "config.json",
            r#"{"model_type": "llama", "hidden_size": 4096,
                "num_attention_heads": 32, "num_hidden_layers": 26}"#,
        )]);
        let m = read_hf_config(dir.path()).unwrap();
        assert_eq!(m.architecture, "llama");
        assert_eq!(m.kv.kv_heads, Some(32));
        assert_eq!(m.kv.head_dim, Some(128));
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__hf_config_gptq_bits__quant_extracted() {
        let dir = write_dir(&[(
            "config.json",
            r#"{"architectures": ["Qwen2ForCausalLM"],
                "quantization_config": {"quant_method": "gptq", "bits": 4},
                "num_hidden_layers": 28, "num_attention_heads": 14}"#,
        )]);
        assert_eq!(read_hf_config(dir.path()).unwrap().quant_bits, Some(4));
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__hf_config_missing_file__teaching_error() {
        let dir = write_dir(&[]);
        let err = read_hf_config(dir.path()).unwrap_err();
        assert!(err.to_string().contains("unreadable"), "{err}");
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__hf_config_malformed_json__error() {
        let dir = write_dir(&[("config.json", "{ nope")]);
        assert!(read_hf_config(dir.path()).is_err());
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__kv_elem_bytes__fp8_vs_16bit() {
        assert_eq!(HfMeta::kv_elem_bytes("fp8_e5m2"), 1);
        assert_eq!(HfMeta::kv_elem_bytes("bf16"), 2);
        assert_eq!(HfMeta::kv_elem_bytes("auto"), 2);
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__model_meta__context_length_both_lanes() {
        let g = GgufMeta {
            architecture: "llama".into(),
            context_length: Some(8192),
            ..Default::default()
        };
        assert_eq!(ModelMeta::Gguf(&g).context_length(), Some(8192));
        let h = HfMeta {
            architecture: "llama".into(),
            ctx_train: Some(4096),
            ..Default::default()
        };
        assert_eq!(ModelMeta::Hf(&h).context_length(), Some(4096));
        assert_eq!(ModelMeta::Hf(&h).architecture(), "llama");
    }
}
