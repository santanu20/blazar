//! Compatibility policy over [`blazar_core::kv`] identities and
//! layouts.
//!
//! Strict by construction: identical identities and layouts are
//! `Exact`; a registered converter can make differing layouts
//! `Transformable`; everything else - including any missing mandatory
//! digest - is `Incompatible` with a teaching reason. Callers never
//! decide this by comparing engine names or versions.

use blazar_core::kv::{KvCompatibility, KvLayout, KvModelIdentity};

/// Registered layout converters, keyed by an id that later waves can
/// resolve to an implementation. Empty registry = only `Exact` ever
/// fires; that is the honest v1 posture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutConverter {
    pub id: String,
}

/// Compare a produced cache (identity + layout) against a target.
#[must_use]
pub fn compatibility(
    saved: &KvModelIdentity,
    saved_layout: &KvLayout,
    target: &KvModelIdentity,
    target_layout: &KvLayout,
    converters: &[LayoutConverter],
) -> KvCompatibility {
    // Mandatory identity first: missing digests can never establish
    // that the same computation produced both caches.
    if !saved.complete() || !target.complete() {
        return KvCompatibility::Incompatible {
            reason: "mandatory identity digest missing (model/tokenizer/config/rope)".into(),
        };
    }
    if saved.model_sha256 != target.model_sha256 {
        return KvCompatibility::Incompatible {
            reason: format!(
                "model artifact differs ({} vs {})",
                &saved.model_sha256[..8.min(saved.model_sha256.len())],
                &target.model_sha256[..8.min(target.model_sha256.len())],
            ),
        };
    }
    if saved.tokenizer_sha256 != target.tokenizer_sha256 {
        return KvCompatibility::Incompatible {
            reason: "tokenizer differs - token streams are not comparable".into(),
        };
    }
    if saved.model_config_sha256 != target.model_config_sha256 {
        return KvCompatibility::Incompatible {
            reason: "model config differs - hidden-state meaning may have shifted".into(),
        };
    }
    if saved.rope_fingerprint != target.rope_fingerprint {
        return KvCompatibility::Incompatible {
            reason: "rope/positional configuration differs - cached positions are meaningless"
                .into(),
        };
    }
    // Adapters: base vs adapter and adapter vs adapter both change the
    // projection weights; only a validated pair-specific mapper could
    // ever bridge that, which is exactly what later waves gate on.
    if saved.adapter_sha256 != target.adapter_sha256 {
        return KvCompatibility::Incompatible {
            reason: "adapter (LoRA) digest differs - projection weights differ".into(),
        };
    }

    // Same computation: can the target read the layout?
    if saved_layout == target_layout {
        return KvCompatibility::Exact;
    }
    if let Some(conv) = converters.first() {
        // A registered converter claims it can bridge these layouts.
        KvCompatibility::Transformable {
            converter_id: conv.id.clone(),
        }
    } else {
        KvCompatibility::Incompatible {
            reason: format!(
                "layout differs ({}x{}x{} {} vs {}x{}x{} {}) and no converter is registered",
                saved_layout.layers,
                saved_layout.kv_heads,
                saved_layout.head_dim,
                saved_layout.attention_layout,
                target_layout.layers,
                target_layout.kv_heads,
                target_layout.head_dim,
                target_layout.attention_layout,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ident(model: &str) -> KvModelIdentity {
        KvModelIdentity {
            model_sha256: model.into(),
            tokenizer_sha256: "tok".into(),
            adapter_sha256: None,
            model_config_sha256: "cfg".into(),
            rope_fingerprint: "rope".into(),
        }
    }

    fn layout(heads: u32) -> KvLayout {
        KvLayout {
            layers: 24,
            kv_heads: heads,
            head_dim: 128,
            key_dtype: "f16".into(),
            value_dtype: "f16".into(),
            block_tokens: 32,
            tensor_parallel: 1,
            pipeline_parallel: 1,
            attention_layout: "k-major".into(),
        }
    }

    #[test]
    // suite convention: unit__scenario__expected (6b)
    #[allow(non_snake_case)]
    fn unit__compatibility__identical_identity_and_layout_is_exact() {
        let c = compatibility(&ident("m"), &layout(8), &ident("m"), &layout(8), &[]);
        assert_eq!(c, KvCompatibility::Exact);
    }

    #[test]
    // suite convention: unit__scenario__expected (6b)
    #[allow(non_snake_case)]
    fn unit__compatibility__missing_digest_is_never_shareable() {
        let mut saved = ident("m");
        saved.tokenizer_sha256 = String::new();
        let c = compatibility(&saved, &layout(8), &ident("m"), &layout(8), &[]);
        assert!(matches!(c, KvCompatibility::Incompatible { .. }));
    }

    #[test]
    // suite convention: unit__scenario__expected (6b)
    #[allow(non_snake_case)]
    fn unit__compatibility__tokenizer_drift_rejects() {
        let mut target = ident("m");
        target.tokenizer_sha256 = "tok2".into();
        let c = compatibility(&ident("m"), &layout(8), &target, &layout(8), &[]);
        assert!(
            matches!(c, KvCompatibility::Incompatible { reason } if reason.contains("tokenizer"))
        );
    }

    #[test]
    // suite convention: unit__scenario__expected (6b)
    #[allow(non_snake_case)]
    fn unit__compatibility__adapter_drift_rejects() {
        let mut target = ident("m");
        target.adapter_sha256 = Some("lora-a".into());
        let c = compatibility(&ident("m"), &layout(8), &target, &layout(8), &[]);
        assert!(
            matches!(c, KvCompatibility::Incompatible { reason } if reason.contains("adapter"))
        );
    }

    #[test]
    // suite convention: unit__scenario__expected (6b)
    #[allow(non_snake_case)]
    fn unit__compatibility__layout_gap_needs_a_registered_converter() {
        let no_conv = compatibility(&ident("m"), &layout(8), &ident("m"), &layout(4), &[]);
        assert!(
            matches!(no_conv, KvCompatibility::Incompatible { reason } if reason.contains("no converter"))
        );
        let conv = compatibility(
            &ident("m"),
            &layout(8),
            &ident("m"),
            &layout(4),
            &[LayoutConverter {
                id: "gqa-split".into(),
            }],
        );
        assert_eq!(
            conv,
            KvCompatibility::Transformable {
                converter_id: "gqa-split".into()
            }
        );
    }
}
