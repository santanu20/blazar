//! Canonical KV interchange carrier ("BKI"): the lane-neutral representation
//! every cross-engine codec converts through. One codec per lane instead of
//! one converter per lane pair - the pair matrix grows linearly as lanes are
//! added (`defuse`/`repage` eligibility still comes from the PR4 translation
//! verdict, which stays the single gate for any transfer).
//!
//! The carrier carries what no single engine file carries alone: the real
//! token ids (the MLX prompt-cache file omits them; vLLM tier files are only
//! hash-addressed BY them), the tensor geometry, and the memory layout the
//! rows are stored in. Provenance/rope/geometry gates run at codec entry -
//! a carrier never exists for a pair those gates refuse.

use std::path::Path;

use blazar_core::config::TranslationSection;

use super::repack::{RepackGeometry, RepackOutcome, SlotSaveBank, VllmFsSpec, repack_into_fs_tier};
use super::translation::{KV_DTYPE_F16, LLAMACPP_LAYOUT};

/// Lane-neutral KV payload: tokens plus per-layer K/V rows in the layout the
/// `attention_layout` string names. Rows are token-major (one contiguous row
/// per cached position) with `block_tokens` recording the granularity the
/// source engine stored them at (llama.cpp banks are per-token, so 1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCarrier {
    /// Real token ids of the cached prefix (post-envelope, no gap markers,
    /// non-negative - llama.cpp banks are validated at parse).
    pub tokens: Vec<u32>,
    /// Cached cell count (may trail `tokens.len()`; the tail is uncached).
    pub cells: u32,
    /// Tensor geometry the gates verified (layers, kv heads, head dim).
    pub geometry: RepackGeometry,
    /// KV element dtype - `f16` is the only admitted dtype today.
    pub dtype: &'static str,
    /// Storage granularity of the source cache (rows are still token-major).
    pub block_tokens: u32,
    /// PR4 layout descriptor for the row encoding (`sep-kv-token-major`, ...).
    pub attention_layout: &'static str,
    /// K bytes, layer-major, cell-within-layer - the llama.cpp token-major
    /// row order (`layers * cells * row_bytes` total).
    pub k_rows: Vec<u8>,
    /// V bytes in the same shape contract as `k_rows`.
    pub v_rows: Vec<u8>,
}

impl KvCarrier {
    /// Lift a parsed llama.cpp slot-save bank into the carrier. The bank is
    /// already the canonical shape (token-major separate K/V rows), so the
    /// lift is a move plus the layout descriptor.
    #[must_use]
    pub fn from_slot_bank(bank: SlotSaveBank, geometry: RepackGeometry) -> Self {
        let cells = bank.cells;
        // Parser already refuses gap markers inside the cached prefix, so
        // every id is non-negative; map into the lane-neutral u32 space.
        let tokens = bank
            .tokens
            .into_iter()
            .map(|t| u32::try_from(t).expect("parse validated non-negative tokens"))
            .collect::<Vec<_>>();
        Self {
            tokens,
            cells,
            geometry,
            dtype: KV_DTYPE_F16,
            block_tokens: 1,
            attention_layout: LLAMACPP_LAYOUT,
            k_rows: bank.k_rows,
            v_rows: bank.v_rows,
        }
    }

    /// Project the carrier back onto a llama.cpp slot-save bank shape (the
    /// sink side of the reverse codec): geometry and layout are re-validated
    /// by the writer, not trusted from the carrier.
    #[must_use]
    pub fn into_slot_bank(self) -> SlotSaveBank {
        SlotSaveBank {
            tokens: self
                .tokens
                .into_iter()
                .map(|t| i32::try_from(t).expect("carrier token ids fit llama_token"))
                .collect(),
            cells: self.cells,
            layers: self.geometry.layers,
            k_rows: self.k_rows,
            v_rows: self.v_rows,
        }
    }

    /// Row width in bytes for one cached position in one layer
    /// (`kv_heads * head_dim * 2` for `f16`).
    #[must_use]
    pub fn row_bytes(&self) -> usize {
        self.geometry.kv_heads as usize * self.geometry.head_dim as usize * 2
    }
}

/// Export the carrier into vLLM's persistent FS offload tier. Byte-identical
/// delegation to the shipped repack writer - the carrier adds no encoding of
/// its own, so everything proven live for `blazar kv import` (gguf -> HF)
/// holds for this path unchanged.
pub fn carrier_into_vllm_tier(
    carrier: &KvCarrier,
    spec: &VllmFsSpec,
    root: &Path,
) -> Result<RepackOutcome, String> {
    let bank = SlotSaveBank {
        tokens: carrier
            .tokens
            .iter()
            .map(|&t| {
                i32::try_from(t)
                    .map_err(|_| format!("carrier token id {t} does not fit llama_token"))
            })
            .collect::<Result<Vec<_>, _>>()?,
        cells: carrier.cells,
        layers: carrier.geometry.layers,
        k_rows: carrier.k_rows.clone(),
        v_rows: carrier.v_rows.clone(),
    };
    repack_into_fs_tier(&bank, &carrier.geometry, spec, root)
}

/// A cross-lane codec pair the interchange admits. Pairs appear here only
/// once their codec has live evidence; anything absent refuses with the
/// pairs that do exist named in the teaching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecLane {
    /// llama.cpp GGUF lane -> vLLM HF lane (shipped, live-proven
    /// `cached_tokens` restore through the gateway).
    LlamaCppToVllm,
}

impl CodecLane {
    /// Human-readable pair name used by CLI reports and doctor rows.
    #[must_use]
    pub fn pair_name(self) -> &'static str {
        match self {
            Self::LlamaCppToVllm => "llama.cpp -> vllm",
        }
    }
}

/// The codec pairs admitted under the current `[kv.translation]` knob. The
/// knob gates converter admission exactly as in the PR4 verdict registry;
/// with the knob off no pair is admitted and every transfer refuses with the
/// opt-in teaching.
#[must_use]
pub fn admitted_pairs(cfg: &TranslationSection) -> Vec<CodecLane> {
    if cfg.enabled {
        vec![CodecLane::LlamaCppToVllm]
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use super::*;

    fn geo() -> RepackGeometry {
        RepackGeometry {
            layers: 2,
            kv_heads: 2,
            head_dim: 4,
        }
    }

    fn bank() -> SlotSaveBank {
        // Two layers, 128 cached cells (two full 64-token blocks), 16-byte
        // F16 rows (2 heads x 4 dim) - big enough to exercise real block
        // writes in the byte-identity test below.
        // Flat byte vecs are layer-major: both layers concatenated, each
        // 128 cells x 16-byte rows (kv_heads 2 x head_dim 4 x F16).
        let layer_bytes = |l: u32| {
            (0..128)
                .flat_map(|t| [u8::try_from((l << 6) | (t & 0x3f)).unwrap(); 16])
                .collect::<Vec<u8>>()
        };
        let two_layers = |l: u32| [layer_bytes(l), layer_bytes(l)].concat();
        SlotSaveBank {
            tokens: (0..128).map(|i| 1000 + i).collect(),
            cells: 128,
            layers: 2,
            k_rows: two_layers(0),
            v_rows: two_layers(1),
        }
    }

    #[test]
    fn unit__carrier__slot_bank_roundtrip_is_lossless() {
        let b = bank();
        let carrier = KvCarrier::from_slot_bank(b.clone(), geo());
        assert_eq!(carrier.dtype, KV_DTYPE_F16);
        assert_eq!(carrier.attention_layout, LLAMACPP_LAYOUT);
        assert_eq!(carrier.block_tokens, 1);
        assert_eq!(carrier.row_bytes(), 16);
        assert_eq!(carrier.into_slot_bank(), b);
    }

    #[test]
    fn unit__carrier__vllm_export_is_byte_identical_to_repack() {
        let b = bank();
        let spec = VllmFsSpec {
            model_name: "m".into(),
            tokens_per_hash: 64,
            blocks_per_file: 1,
            canonical_format: "v1-nhd".into(),
        };
        let left = tempfile::tempdir().unwrap();
        let right = tempfile::tempdir().unwrap();
        let via_carrier = carrier_into_vllm_tier(
            &KvCarrier::from_slot_bank(b.clone(), geo()),
            &spec,
            left.path(),
        )
        .unwrap();
        let direct = repack_into_fs_tier(&b, &geo(), &spec, right.path()).unwrap();
        assert_eq!(via_carrier.blocks_written, direct.blocks_written);
        // Both sides must write the same config.json AND the same block
        // files (two 64-token blocks: 2 layers x 128 tokens x 16-byte rows
        // x K|V).
        let files = |d: &Path| -> Vec<Vec<u8>> {
            let mut out = vec![];
            for e in std::fs::read_dir(d.join(spec.rank_dir(d))).unwrap() {
                let p = e.unwrap().path();
                if p.is_file() {
                    out.push(std::fs::read(&p).unwrap());
                }
            }
            out.sort();
            out
        };
        assert_eq!(files(left.path()), files(right.path()));
    }

    #[test]
    fn unit__admitted_pairs__follow_the_translation_knob() {
        assert!(admitted_pairs(&TranslationSection::default()).is_empty());
        let on = TranslationSection {
            enabled: true,
            ..TranslationSection::default()
        };
        assert_eq!(admitted_pairs(&on), vec![CodecLane::LlamaCppToVllm]);
        assert_eq!(CodecLane::LlamaCppToVllm.pair_name(), "llama.cpp -> vllm");
    }
}
