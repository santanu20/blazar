//! Built-in KV layout translation: typed converters, eligibility
//! predicates, and the identity/layout constructors for each modeled
//! lane. This is the policy layer the compatibility verdict and the
//! doctor's cross-lane table consume; data-plane bridges (actually
//! re-blocking stored tensors) are deliberately NOT here yet - a
//! `Transformable` verdict describes what a bridge MAY do, never a
//! transfer that runs today.
//!
//! Layout constants cite live registrations observed on 2026-11-19
//! (qwen2.5-1.5b-instruct, sglang 0.5.23 + vllm 0.34.0 dialing a
//! `lmcache server`): sglang registered `NL_X_NB_BS_NH_HS` (separate
//! K/V tensors, block-major) while vllm registered `NL_X_NB_BS_NH_CS`
//! (fused K|V, `cs = 2 * head_dim`) - the exact pair that corrupted
//! cross-engine reuse upstream and motivated this gate.

use std::path::Path;

use blazar_core::gguf::GgufMeta;
use blazar_core::hfmeta::{HfMeta, read_hf_config};
use blazar_core::kv::{KV_TRANSLATION_CONVERTERS, KvCompatibility, KvLayout, KvModelIdentity};
use sha2::{Digest, Sha256};

/// The two built-in converters. Plain enum on purpose: no plugin
/// surface, no trait objects - every admission decision resolves to a
/// statically known predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltInConverter {
    /// Fuse or split the K|V pair per head (`attention_layout` delta
    /// only): separate-K/V layouts <-> interleaved-fused layouts.
    Defuse,
    /// Re-block between token-granular and paged layouts
    /// (`block_tokens` delta only).
    Repage,
}

impl BuiltInConverter {
    /// Canonical id - matches `KV_TRANSLATION_CONVERTERS`.
    #[must_use]
    pub fn id(self) -> &'static str {
        match self {
            Self::Defuse => "defuse",
            Self::Repage => "repage",
        }
    }

    /// Resolve a config-admitted id to the built-in it names; unknown
    /// ids (impossible after config validation) resolve to `None` and
    /// are skipped rather than guessed at.
    #[must_use]
    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "defuse" => Some(Self::Defuse),
            "repage" => Some(Self::Repage),
            _ => None,
        }
    }

    /// Eligibility: the converter has work to do toward `target` - its
    /// own field differs. Chained steps compose from there (defuse a
    /// split pair first, re-block second); a step that changed nothing
    /// would loop, and a step touching a foreign field does not exist.
    #[must_use]
    pub fn eligible(self, saved: &KvLayout, target: &KvLayout) -> bool {
        match self {
            Self::Defuse => saved.attention_layout != target.attention_layout,
            Self::Repage => saved.block_tokens != target.block_tokens,
        }
    }

    /// Apply to `current`, pulling the converter's own field toward
    /// `target`. Eligibility is checked by the caller; applying to a
    /// layout that does not differ in this field is a no-op clone.
    #[must_use]
    pub fn apply(self, current: &KvLayout, target: &KvLayout) -> KvLayout {
        let mut next = current.clone();
        match self {
            Self::Defuse => {
                next.attention_layout.clone_from(&target.attention_layout);
            }
            Self::Repage => next.block_tokens = target.block_tokens,
        }
        next
    }
}

/// Layout verdict after identity already matched and layouts already
/// differ: hard blocks first (no converter may bridge them), then a
/// bounded composition search over the admitted registry.
#[must_use]
pub fn translation_verdict(
    saved: &KvLayout,
    target: &KvLayout,
    registry: &[BuiltInConverter],
) -> KvCompatibility {
    // Tensor geometry defines the arithmetic of attention; a converter
    // that changed layer/head counts would be re-computing the model,
    // not re-laying-out its cache.
    if (saved.layers, saved.kv_heads, saved.head_dim)
        != (target.layers, target.kv_heads, target.head_dim)
    {
        return KvCompatibility::Incompatible {
            reason: format!(
                "different tensor geometry ({}x{}x{} vs {}x{}x{}) - no converter may bridge layer/head counts",
                saved.layers,
                saved.kv_heads,
                saved.head_dim,
                target.layers,
                target.kv_heads,
                target.head_dim
            ),
        };
    }
    if saved.key_dtype != target.key_dtype || saved.value_dtype != target.value_dtype {
        return KvCompatibility::Incompatible {
            reason: format!(
                "dtype cast is value-changing ({}->{} keys) - never bridged silently",
                saved.key_dtype, target.key_dtype
            ),
        };
    }
    if saved.tensor_parallel != target.tensor_parallel
        || saved.pipeline_parallel != target.pipeline_parallel
    {
        return KvCompatibility::Incompatible {
            reason: format!(
                "parallel reshuffle not modeled (tp {}/pp {} vs tp {}/pp {}) - no blazar lane reports tp>1, so the semantics stay unverified",
                saved.tensor_parallel,
                saved.pipeline_parallel,
                target.tensor_parallel,
                target.pipeline_parallel
            ),
        };
    }

    // Bounded composition: at most two built-ins, explored in registry
    // order so the recorded chain id is deterministic and canonical
    // (e.g. `defuse+repage` - defuse first, matching the const order).
    for first in registry {
        if !first.eligible(saved, target) {
            continue;
        }
        let mid = first.apply(saved, target);
        if mid == *target {
            return KvCompatibility::Transformable {
                converter_id: first.id().to_owned(),
            };
        }
        for second in registry {
            if second == first || !second.eligible(&mid, target) {
                continue;
            }
            if second.apply(&mid, target) == *target {
                return KvCompatibility::Transformable {
                    converter_id: format!("{}+{}", first.id(), second.id()),
                };
            }
        }
    }
    KvCompatibility::Incompatible {
        reason: format!(
            "layout differs ({} tpb={} vs {} tpb={}) and no admitted converter bridges it",
            saved.attention_layout,
            saved.block_tokens,
            target.attention_layout,
            target.block_tokens
        ),
    }
}

/// Admitted built-ins from the `[kv.translation]` section: disabled =
/// empty registry (Exact-only sharing); `allow = []` = every built-in
/// in canonical order; otherwise the listed ids in canonical order
/// (duplicates collapse).
#[must_use]
pub fn builtin_registry(
    section: &blazar_core::config::TranslationSection,
) -> Vec<BuiltInConverter> {
    if !section.enabled {
        return Vec::new();
    }
    if section.allow.is_empty() {
        return vec![BuiltInConverter::Defuse, BuiltInConverter::Repage];
    }
    KV_TRANSLATION_CONVERTERS
        .iter()
        .filter_map(|id| {
            section
                .allow
                .iter()
                .any(|a| a == id)
                .then(|| BuiltInConverter::from_id(id))
                .flatten()
        })
        .collect()
}

/// Attention-layout marker: llama.cpp writes per-layer K then V
/// tensors, token-granular (a slot's cache grows one token at a time;
/// there is no paging). Verified against `llama-server` slot-state
/// saves (the bank protocol's own byte format).
pub const LLAMACPP_LAYOUT: &str = "sep-kv-token-major";
/// Attention-layout marker: sglang/LMCache `NL_X_NB_BS_NH_HS` -
/// separate K and V tensors per layer, block-major (qwen2.5 live
/// registration, sglang 0.5.23).
pub const SGLANG_LAYOUT: &str = "sep-kv-block-major";
/// Attention-layout marker: vLLM/LMCache `NL_X_NB_BS_NH_CS` - fused
/// K|V per head (`cs = 2 * head_dim`), page-granular (qwen2.5 live
/// registration, vllm 0.34.0, page 64).
pub const VLLM_LAYOUT: &str = "fused-kv-block-major";

/// Cache dtypes every modeled lane uses for a bf16/f16 base model when
/// no cache-quantization knob overrides it. Cache-dtype knobs exist
/// per lane; when two lanes disagree there, the dtype hard block
/// fires honestly instead of this constant bridging it.
pub const KV_DTYPE_F16: &str = "f16";

/// KV layout as llama.cpp materializes a GGUF model: full-block layers
/// (loop architectures multiply `block_count`), derived head geometry,
/// token-granular separate K/V. `None` when the GGUF does not carry
/// enough geometry to describe the cache - callers teach, never guess.
#[must_use]
pub fn llamacpp_kv_layout(meta: &GgufMeta) -> Option<KvLayout> {
    let loops = meta.num_loops.unwrap_or(1);
    let layers = u32::try_from(meta.block_count?).ok()?.saturating_mul(loops);
    let kv_heads = u32::try_from(meta.head_count_kv.or(meta.head_count)?).ok()?;
    let head_dim = u32::try_from(meta.derived_head_dim()?).ok()?;
    Some(KvLayout {
        layers,
        kv_heads,
        head_dim,
        key_dtype: KV_DTYPE_F16.into(),
        value_dtype: KV_DTYPE_F16.into(),
        block_tokens: 1,
        tensor_parallel: 1,
        pipeline_parallel: 1,
        attention_layout: LLAMACPP_LAYOUT.into(),
    })
}

/// KV layout as sglang materializes an HF model with the given page
/// size (`page_size` tuning knob; the live qwen2.5 registration used
/// 1). `None` when the HF config lacks KV geometry.
#[must_use]
pub fn sglang_kv_layout(meta: &HfMeta, page_tokens: u32) -> Option<KvLayout> {
    Some(KvLayout {
        layers: u32::try_from(meta.kv.layers?).ok()?,
        kv_heads: u32::try_from(meta.kv.kv_heads?).ok()?,
        head_dim: u32::try_from(meta.kv.head_dim?).ok()?,
        key_dtype: KV_DTYPE_F16.into(),
        value_dtype: KV_DTYPE_F16.into(),
        block_tokens: page_tokens,
        tensor_parallel: 1,
        pipeline_parallel: 1,
        attention_layout: SGLANG_LAYOUT.into(),
    })
}

/// KV layout as vLLM materializes an HF model with the given block
/// size (qwen2.5 live registration: 64). `None` when the HF config
/// lacks KV geometry.
#[must_use]
pub fn vllm_kv_layout(meta: &HfMeta, page_tokens: u32) -> Option<KvLayout> {
    Some(KvLayout {
        layers: u32::try_from(meta.kv.layers?).ok()?,
        kv_heads: u32::try_from(meta.kv.kv_heads?).ok()?,
        head_dim: u32::try_from(meta.kv.head_dim?).ok()?,
        key_dtype: KV_DTYPE_F16.into(),
        value_dtype: KV_DTYPE_F16.into(),
        block_tokens: page_tokens,
        tensor_parallel: 1,
        pipeline_parallel: 1,
        attention_layout: VLLM_LAYOUT.into(),
    })
}

/// SHA-256 of one file, hex-encoded; `None` when unreadable.
fn sha256_file(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Some(hex::encode(hasher.finalize()))
}

/// Identity for an HF safetensors directory: the store-recorded model
/// digest, the tokenizer and config file digests, and the rope
/// fingerprint from the parsed config. `None` when any mandatory leg
/// is missing (no recorded digest, absent `tokenizer.json`, unreadable
/// `config.json`, or a config with no rope declaration) - an
/// incomplete identity is never shareable, only teachable.
#[must_use]
pub fn hf_kv_identity(dir: &Path, model_sha: Option<&str>) -> Option<KvModelIdentity> {
    let model_sha256 = model_sha?.trim().to_owned();
    if model_sha256.is_empty() {
        return None;
    }
    let tokenizer = sha256_file(&dir.join("tokenizer.json"))?;
    let config = sha256_file(&dir.join("config.json"))?;
    let meta = read_hf_config(dir).ok()?;
    Some(KvModelIdentity {
        model_sha256,
        tokenizer_sha256: tokenizer,
        adapter_sha256: None,
        model_config_sha256: config,
        rope_fingerprint: meta.rope_fingerprint?,
    })
}

/// Identity for a GGUF file: the recorded file digest, with tokenizer
/// and config legs carrying honest `gguf-embedded:<sha>` markers - the
/// tokenizer and rope configuration live INSIDE the file, so the file
/// digest already pins them; fabricating sub-digests would add no
/// information. `None` when the file digest is unrecorded or the
/// metadata carries no rope fingerprint.
#[must_use]
pub fn gguf_kv_identity(meta: &GgufMeta, file_sha: Option<&str>) -> Option<KvModelIdentity> {
    let sha = file_sha?.trim().to_owned();
    if sha.is_empty() {
        return None;
    }
    Some(KvModelIdentity {
        tokenizer_sha256: format!("gguf-embedded:{sha}"),
        model_config_sha256: format!("gguf-embedded:{sha}"),
        model_sha256: sha,
        adapter_sha256: None,
        rope_fingerprint: meta.rope_fingerprint.clone()?,
    })
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;

    fn layout(attention: &str, tpb: u32) -> KvLayout {
        KvLayout {
            layers: 28,
            kv_heads: 2,
            head_dim: 128,
            key_dtype: KV_DTYPE_F16.into(),
            value_dtype: KV_DTYPE_F16.into(),
            block_tokens: tpb,
            tensor_parallel: 1,
            pipeline_parallel: 1,
            attention_layout: attention.into(),
        }
    }

    fn section(enabled: bool, allow: &[&str]) -> blazar_core::config::TranslationSection {
        blazar_core::config::TranslationSection {
            enabled,
            allow: allow.iter().map(|s| (*s).to_owned()).collect(),
        }
    }

    #[test]
    fn unit__translation__each_converter_is_eligible_exactly_when_its_own_field_differs() {
        let sep1 = layout(SGLANG_LAYOUT, 1);
        let sep64 = layout(SGLANG_LAYOUT, 64);
        let fused1 = layout(VLLM_LAYOUT, 1);
        // Repage: work exists whenever the page granularity differs -
        // even when the layout family differs too (it runs second).
        assert!(BuiltInConverter::Repage.eligible(&sep1, &sep64));
        assert!(BuiltInConverter::Repage.eligible(&sep1, &layout(VLLM_LAYOUT, 64)));
        assert!(!BuiltInConverter::Repage.eligible(&sep1, &fused1));
        // Defuse: work exists whenever the K|V layout family differs.
        assert!(BuiltInConverter::Defuse.eligible(&sep1, &fused1));
        assert!(BuiltInConverter::Defuse.eligible(&sep1, &layout(VLLM_LAYOUT, 64)));
        assert!(!BuiltInConverter::Defuse.eligible(&sep1, &sep64));
    }

    #[test]
    fn unit__translation__geometry_dtype_and_parallelism_never_bridge() {
        let all = vec![BuiltInConverter::Defuse, BuiltInConverter::Repage];
        let base = layout(SGLANG_LAYOUT, 1);
        // geometry: layer count delta
        let mut geom = layout(VLLM_LAYOUT, 1);
        geom.layers = 32;
        assert!(matches!(
            translation_verdict(&base, &geom, &all),
            KvCompatibility::Incompatible { reason } if reason.contains("tensor geometry")
        ));
        // dtype: value-changing cast
        let mut dt = layout(VLLM_LAYOUT, 1);
        dt.key_dtype = "bf16".into();
        assert!(matches!(
            translation_verdict(&base, &dt, &all),
            KvCompatibility::Incompatible { reason } if reason.contains("value-changing")
        ));
        // parallelism: reshuffle unmodeled
        let mut par = layout(VLLM_LAYOUT, 1);
        par.tensor_parallel = 2;
        assert!(matches!(
            translation_verdict(&base, &par, &all),
            KvCompatibility::Incompatible { reason } if reason.contains("parallel reshuffle")
        ));
    }

    #[test]
    fn unit__translation__corruption_pair_composes_defuse_then_repage() {
        // The live-registered qwen2.5 pair that corrupted cross-engine
        // reuse upstream: sglang sep-K/V tpb=1 vs vllm fused page=64.
        let sglang = layout(SGLANG_LAYOUT, 1);
        let vllm = layout(VLLM_LAYOUT, 64);
        let registry = builtin_registry(&section(true, &[]));
        assert_eq!(
            translation_verdict(&sglang, &vllm, &registry),
            KvCompatibility::Transformable {
                converter_id: "defuse+repage".into()
            }
        );
        // Reverse direction composes the same canonical chain.
        assert_eq!(
            translation_verdict(&vllm, &sglang, &registry),
            KvCompatibility::Transformable {
                converter_id: "defuse+repage".into()
            }
        );
    }

    #[test]
    fn unit__translation__registry_admission_follows_the_knob() {
        // Off: Exact-only posture, nothing admitted.
        assert_eq!(
            builtin_registry(&section(false, &[])),
            [] as [crate::kv::translation::BuiltInConverter; 0]
        );
        // On with empty allow: every built-in, canonical order.
        assert_eq!(
            builtin_registry(&section(true, &[])),
            vec![BuiltInConverter::Defuse, BuiltInConverter::Repage]
        );
        // Narrowed: only the listed id, still canonical order.
        assert_eq!(
            builtin_registry(&section(true, &["repage"])),
            vec![BuiltInConverter::Repage]
        );
    }

    #[test]
    fn unit__translation__narrowed_registry_cannot_compose_the_corruption_pair() {
        let sglang = layout(SGLANG_LAYOUT, 1);
        let vllm = layout(VLLM_LAYOUT, 64);
        let repage_only = builtin_registry(&section(true, &["repage"]));
        assert!(matches!(
            translation_verdict(&sglang, &vllm, &repage_only),
            KvCompatibility::Incompatible { reason } if reason.contains("no admitted converter")
        ));
    }

    #[test]
    fn unit__translation__lane_constructors_carry_the_live_registered_constants() {
        let gguf = GgufMeta {
            base_model_id: None,
            block_count: Some(28),
            head_count: Some(12),
            head_count_kv: Some(2),
            embedding_length: Some(1536),
            rope_fingerprint: Some("freq_base=1000000".into()),
            ..GgufMeta::default()
        };
        let lc = llamacpp_kv_layout(&gguf).unwrap();
        assert_eq!((lc.layers, lc.kv_heads, lc.head_dim), (28, 2, 128));
        assert_eq!(
            (lc.block_tokens, lc.attention_layout.as_str()),
            (1, LLAMACPP_LAYOUT)
        );

        let hf = HfMeta {
            kv: blazar_core::hfmeta::KvGeom {
                layers: Some(28),
                kv_heads: Some(2),
                head_dim: Some(128),
            },
            ..HfMeta::default()
        };
        let sg = sglang_kv_layout(&hf, 1).unwrap();
        assert_eq!(
            (sg.block_tokens, sg.attention_layout.as_str()),
            (1, SGLANG_LAYOUT)
        );
        let vl = vllm_kv_layout(&hf, 64).unwrap();
        assert_eq!(
            (vl.block_tokens, vl.attention_layout.as_str()),
            (64, VLLM_LAYOUT)
        );

        // Missing geometry stays None - never guessed.
        assert!(sglang_kv_layout(&HfMeta::default(), 1).is_none());
        assert!(llamacpp_kv_layout(&GgufMeta::default()).is_none());
    }

    #[test]
    fn unit__translation__hf_identity_pins_all_four_digests() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            r#"{"architectures":["Qwen2ForCausalLM"],"num_hidden_layers":28,
                "num_key_value_heads":2,"hidden_size":1536,"num_attention_heads":12,
                "rope_theta": 1000000.0}"#,
        )
        .unwrap();
        std::fs::write(tmp.path().join("tokenizer.json"), r#"{"version":"1"}"#).unwrap();
        let id = hf_kv_identity(tmp.path(), Some("abc123")).unwrap();
        assert_eq!(id.model_sha256, "abc123");
        assert_eq!(id.rope_fingerprint, "freq_base=1000000");
        assert_ne!(id.tokenizer_sha256, "");
        assert_ne!(id.model_config_sha256, "");
        assert!(id.complete());
        // Any missing leg (no recorded digest / no tokenizer.json /
        // no rope declaration) yields None - teachable, never partial.
        assert!(hf_kv_identity(tmp.path(), None).is_none());
        let no_tok = tempfile::tempdir().unwrap();
        std::fs::write(no_tok.path().join("config.json"), r#"{"rope_theta": 1.0}"#).unwrap();
        assert!(hf_kv_identity(no_tok.path(), Some("abc123")).is_none());
        let no_rope = tempfile::tempdir().unwrap();
        std::fs::write(
            no_rope.path().join("config.json"),
            r#"{"num_hidden_layers":28}"#,
        )
        .unwrap();
        std::fs::write(no_rope.path().join("tokenizer.json"), "{}").unwrap();
        assert!(hf_kv_identity(no_rope.path(), Some("abc123")).is_none());
    }

    #[test]
    fn unit__translation__gguf_identity_uses_honest_embedded_markers() {
        let meta = GgufMeta {
            base_model_id: None,
            rope_fingerprint: Some("freq_base=10000000;scaling=yarn,factor=4".into()),
            ..GgufMeta::default()
        };
        let id = gguf_kv_identity(&meta, Some("deadbeef")).unwrap();
        assert_eq!(id.model_sha256, "deadbeef");
        assert_eq!(id.tokenizer_sha256, "gguf-embedded:deadbeef");
        assert_eq!(id.model_config_sha256, "gguf-embedded:deadbeef");
        assert!(id.complete());
        assert!(gguf_kv_identity(&meta, None).is_none());
        assert!(gguf_kv_identity(&GgufMeta::default(), Some("deadbeef")).is_none());
    }
}
