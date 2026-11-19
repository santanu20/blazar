//! Compatibility policy over [`blazar_core::kv`] identities and
//! layouts.
//!
//! Strict by construction: identical identities and layouts are
//! `Exact`; a registered converter can make differing layouts
//! `Transformable`; everything else - including any missing mandatory
//! digest - is `Incompatible` with a teaching reason. Callers never
//! decide this by comparing engine names or versions.

use blazar_core::kv::{KvCompatibility, KvLayout, KvModelIdentity};

use super::translation::BuiltInConverter;

/// Registered layout converters, keyed by the id config validation
/// admits (`defuse`, `repage`). An empty registry = only `Exact` ever
/// fires; that is the knob-off posture of `[kv.translation]`.
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
    let builtin: Vec<BuiltInConverter> = converters
        .iter()
        .filter_map(|c| BuiltInConverter::from_id(&c.id))
        .collect();
    if builtin.is_empty() {
        return KvCompatibility::Incompatible {
            reason: format!(
                "layout differs ({}x{}x{} {} tpb={} vs {}x{}x{} {} tpb={}) and no converter is admitted - [kv.translation] enabled admits the built-in defuse/repage pair",
                saved_layout.layers,
                saved_layout.kv_heads,
                saved_layout.head_dim,
                saved_layout.attention_layout,
                saved_layout.block_tokens,
                target_layout.layers,
                target_layout.kv_heads,
                target_layout.head_dim,
                target_layout.attention_layout,
                target_layout.block_tokens,
            ),
        };
    }
    super::translation::translation_verdict(saved_layout, target_layout, &builtin)
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

    fn reblocked(heads: u32, tpb: u32) -> KvLayout {
        KvLayout {
            layers: 24,
            kv_heads: heads,
            head_dim: 128,
            key_dtype: "f16".into(),
            value_dtype: "f16".into(),
            block_tokens: tpb,
            tensor_parallel: 1,
            pipeline_parallel: 1,
            attention_layout: "k-major".into(),
        }
    }

    #[test]
    // suite convention: unit__scenario__expected (6b)
    #[allow(non_snake_case)]
    fn unit__compatibility__layout_gap_needs_an_admitted_eligible_converter() {
        // Knob off (empty registry): the teaching names [kv.translation].
        let off = compatibility(
            &ident("m"),
            &reblocked(8, 1),
            &ident("m"),
            &reblocked(8, 64),
            &[],
        );
        assert!(
            matches!(off, KvCompatibility::Incompatible { reason } if reason.contains("no converter is admitted")
                && reason.contains("[kv.translation]"))
        );
        // Admitted + eligible (pure reblocking delta): Transformable.
        let on = compatibility(
            &ident("m"),
            &reblocked(8, 1),
            &ident("m"),
            &reblocked(8, 64),
            &[LayoutConverter {
                id: "repage".into(),
            }],
        );
        assert_eq!(
            on,
            KvCompatibility::Transformable {
                converter_id: "repage".into()
            }
        );
        // Unknown ids resolve to nothing - never guessed at.
        let unknown = compatibility(
            &ident("m"),
            &reblocked(8, 1),
            &ident("m"),
            &reblocked(8, 64),
            &[LayoutConverter {
                id: "gqa-split".into(),
            }],
        );
        assert!(
            matches!(unknown, KvCompatibility::Incompatible { reason } if reason.contains("no converter is admitted"))
        );
    }

    #[test]
    // suite convention: unit__scenario__expected (6b)
    #[allow(non_snake_case)]
    fn unit__compatibility__geometry_gap_stays_incompatible_even_with_converters() {
        // Head-count deltas change the arithmetic of attention - the
        // old placeholder fired Transformable for this pair; the real
        // policy refuses it no matter what is admitted.
        let c = compatibility(
            &ident("m"),
            &reblocked(8, 32),
            &ident("m"),
            &reblocked(4, 32),
            &[
                LayoutConverter {
                    id: "defuse".into(),
                },
                LayoutConverter {
                    id: "repage".into(),
                },
            ],
        );
        assert!(
            matches!(c, KvCompatibility::Incompatible { reason } if reason.contains("tensor geometry"))
        );
    }
}
