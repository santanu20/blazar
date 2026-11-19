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

use super::repack::{
    RepackGeometry, RepackOutcome, SLOT_SAVE_MAGIC, SLOT_SAVE_VERSION, SlotSaveBank, VllmFsSpec,
    repack_into_fs_tier, vllm_block_hash, vllm_none_hash,
};
use super::translation::{KV_DTYPE_F16, LLAMACPP_LAYOUT, VLLM_LAYOUT};

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

/// What a tier read actually recovered - vLLM stores (and therefore we can
/// read back) only full blocks, so the report names the prefix depth the
/// restored bank will cover.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TierReadReport {
    /// Full blocks found and read.
    pub blocks_read: u32,
    /// Block granularity (tokens per block, from the spec).
    pub block_tokens: u32,
    /// Blocks whose hash-addressed file is absent (stops the walk; the
    /// prefix beyond this point was never offloaded).
    pub blocks_missing: u32,
}

/// Read a vLLM FS offload tier back into a carrier. The tier is
/// content-addressed BY token ids but never stores them, so the caller
/// supplies the prefix's real ids (provenance gates ran before this point;
/// a wrong id sequence simply finds no files and imports nothing). The
/// walk stops at the first missing block - vLLM offloads whatever full
/// blocks it happened to evict, and honest import takes exactly that
/// prefix. The de-interleave is the exact inverse of the forward page
/// writer (layer-major, per head per token: K slice then V slice).
pub fn vllm_tier_to_carrier(
    root: &Path,
    spec: &VllmFsSpec,
    tokens: &[u32],
    geo: &RepackGeometry,
) -> Result<(KvCarrier, TierReadReport), String> {
    let block_tokens = spec.tokens_per_hash;
    if block_tokens == 0 {
        return Err("spec tokens_per_hash must be nonzero".into());
    }
    let full_blocks = u32::try_from(tokens.len() / block_tokens as usize)
        .map_err(|_| "token sequence too long")?;
    if full_blocks == 0 {
        return Err(format!(
            "token sequence holds {} ids - a full {}-token block is the minimum the tier stores",
            tokens.len(),
            block_tokens
        ));
    }
    let row = geo.row_bytes();
    let head_bytes = geo.head_dim as usize * 2;
    let layer_page = geo.kv_heads as usize * block_tokens as usize * 2 * head_bytes;

    let rank_dir = spec.rank_dir(root);
    let mut parent = vllm_none_hash();
    let mut blocks_read = 0u32;
    let mut blocks_missing = 0u32;
    // Carrier rows are token-major (canonical), pre-sized to the full
    // candidate prefix; the never-offloaded tail is truncated after the
    // walk stops at the first missing block.
    let block_stride = block_tokens as usize * row;
    let mut k_rows = vec![0u8; geo.layers as usize * full_blocks as usize * block_stride];
    let mut v_rows = k_rows.clone();

    'blocks: for b in 0..full_blocks {
        let ids =
            &tokens[b as usize * block_tokens as usize..(b + 1) as usize * block_tokens as usize];
        let hash = vllm_block_hash(parent, ids);
        let rel = VllmFsSpec::block_rel_path(&hex::encode(hash), 0);
        // A missing block ends the walk: the tier stores whole blocks only,
        // so everything from here on was never offloaded.
        let Ok(page) = std::fs::read(rank_dir.join(&rel)) else {
            blocks_missing = full_blocks - b;
            break 'blocks;
        };
        if page.len() != geo.layers as usize * layer_page {
            return Err(format!(
                "block file {:#?} holds {} bytes - expected {} for {} layers x {} heads x {} tokens",
                rel.display(),
                page.len(),
                geo.layers as usize * layer_page,
                geo.layers,
                geo.kv_heads,
                block_tokens
            ));
        }
        let mut off = 0usize;
        for l in 0..geo.layers as usize {
            let layer_base = l * full_blocks as usize * block_stride;
            for h in 0..geo.kv_heads as usize {
                for t in 0..block_tokens as usize {
                    let dst = layer_base + b as usize * block_stride + t * row + h * head_bytes;
                    k_rows[dst..dst + head_bytes].copy_from_slice(&page[off..off + head_bytes]);
                    off += head_bytes;
                    v_rows[dst..dst + head_bytes].copy_from_slice(&page[off..off + head_bytes]);
                    off += head_bytes;
                }
            }
        }
        parent = hash;
        blocks_read += 1;
    }

    if blocks_read == 0 {
        return Err(format!(
            "no hash-addressed block files under {:#?} for this token prefix - the tier never offloaded it (serve the prefix on the vllm lane first)",
            rank_dir.display()
        ));
    }
    let cells = blocks_read * block_tokens;
    k_rows.truncate(geo.layers as usize * cells as usize * row);
    v_rows.truncate(k_rows.len());
    let carrier = KvCarrier {
        tokens: tokens[..cells as usize].to_vec(),
        cells,
        geometry: *geo,
        dtype: KV_DTYPE_F16,
        block_tokens,
        // Rows are de-interleaved into the canonical token-major separate
        // K/V encoding on the way in, so the carrier speaks the canonical
        // layout regardless of which side it came from.
        attention_layout: VLLM_LAYOUT,
        k_rows,
        v_rows,
    };
    Ok((
        carrier,
        TierReadReport {
            blocks_read,
            block_tokens,
            blocks_missing,
        },
    ))
}

/// Serialize the carrier into a llama.cpp slot-save bank file: the
/// `server_tokens` envelope (marker, version 1, real ids, zero media
/// keys) followed by one cache section exactly as llama.cpp's
/// `state_write` emits it at rest (`v_trans = 0`, F16 rows). The engine's
/// restore accepts this shape verbatim - it is the same container the
/// supervisor's own `_auto` banks use.
pub fn carrier_to_slot_save(carrier: &KvCarrier) -> Result<Vec<u8>, String> {
    let row = carrier.row_bytes();
    let cells = carrier.cells as usize;
    let expect = carrier.geometry.layers as usize * cells * row;
    if carrier.k_rows.len() != expect || carrier.v_rows.len() != expect {
        return Err(format!(
            "carrier rows hold {} K / {} V bytes - geometry wants {} each (layers {} x cells {} x row {})",
            carrier.k_rows.len(),
            carrier.v_rows.len(),
            expect,
            carrier.geometry.layers,
            cells,
            row
        ));
    }
    let mut out = Vec::with_capacity(12 + cells * 4 + expect * 2);
    out.extend_from_slice(&SLOT_SAVE_MAGIC.to_le_bytes());
    out.extend_from_slice(&SLOT_SAVE_VERSION.to_le_bytes());
    // server_tokens::serialize envelope: -1 marker, version 1, n_real ids,
    // then zero media keys.
    let cells_u32 = u32::try_from(cells).map_err(|_| "bank too large for llama.cpp")?;
    out.extend_from_slice(&(cells_u32 + 4).to_le_bytes());
    out.extend_from_slice(&(-1i32).to_le_bytes());
    out.extend_from_slice(&1i32.to_le_bytes());
    out.extend_from_slice(&cells_u32.to_le_bytes());
    for &t in &carrier.tokens[..cells] {
        let id = i32::try_from(t).map_err(|_| format!("token id {t} does not fit llama_token"))?;
        out.extend_from_slice(&id.to_le_bytes());
    }
    out.extend_from_slice(&0u32.to_le_bytes()); // media_keys_count
    // One stream, contiguous cells from position 0, single sequence id 0.
    out.extend_from_slice(&1u32.to_le_bytes()); // n_stream
    out.extend_from_slice(&carrier.cells.to_le_bytes()); // cell_count
    for pos in 0..carrier.cells {
        out.extend_from_slice(&pos.cast_signed().to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes()); // n_seq_id
        out.extend_from_slice(&0i32.to_le_bytes()); // seq id
    }
    out.extend_from_slice(&0u32.to_le_bytes()); // v_trans
    out.extend_from_slice(&carrier.geometry.layers.to_le_bytes());
    // The engine banks on this lane write 0/0 here (post-RoPE rows carry
    // no rotary dimension; live-verified against a slot-save oracle - a
    // nonzero value is rejected at restore with a space error).
    out.extend_from_slice(&0u32.to_le_bytes()); // n_rot_k
    out.extend_from_slice(&0u32.to_le_bytes()); // n_rot_v
    for layer_bytes in carrier.k_rows.chunks(cells * row) {
        out.extend_from_slice(&1i32.to_le_bytes()); // ggml_type F16
        out.extend_from_slice(&(row as u64).to_le_bytes());
        out.extend_from_slice(layer_bytes);
    }
    for layer_bytes in carrier.v_rows.chunks(cells * row) {
        out.extend_from_slice(&1i32.to_le_bytes());
        out.extend_from_slice(&(row as u64).to_le_bytes());
        out.extend_from_slice(layer_bytes);
    }
    Ok(out)
}

/// A cross-lane codec pair the interchange admits. Pairs appear here only
/// once their codec has live evidence; anything absent refuses with the
/// pairs that do exist named in the teaching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecLane {
    /// llama.cpp GGUF lane -> vLLM HF lane (shipped, live-proven
    /// `cached_tokens` restore through the gateway).
    LlamaCppToVllm,
    /// vLLM HF lane -> llama.cpp GGUF lane: reads the engine's own fs-tier
    /// block files back into a slot-save bank the llama.cpp server
    /// restores natively.
    VllmToLlamaCpp,
}

impl CodecLane {
    /// Human-readable pair name used by CLI reports and doctor rows.
    #[must_use]
    pub fn pair_name(self) -> &'static str {
        match self {
            Self::LlamaCppToVllm => "llama.cpp -> vllm",
            Self::VllmToLlamaCpp => "vllm -> llama.cpp",
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
        vec![CodecLane::LlamaCppToVllm, CodecLane::VllmToLlamaCpp]
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
        assert_eq!(
            admitted_pairs(&on),
            vec![CodecLane::LlamaCppToVllm, CodecLane::VllmToLlamaCpp]
        );
        assert_eq!(CodecLane::LlamaCppToVllm.pair_name(), "llama.cpp -> vllm");
        assert_eq!(CodecLane::VllmToLlamaCpp.pair_name(), "vllm -> llama.cpp");
    }

    #[test]
    fn unit__reverse_codec__tier_roundtrip_restores_the_original_bank() {
        let b = bank();
        let spec = VllmFsSpec {
            model_name: "m".into(),
            tokens_per_hash: 64,
            blocks_per_file: 1,
            canonical_format: "v1-nhd".into(),
        };
        let geo = geo();
        let dir = tempfile::tempdir().unwrap();
        // Forward: bank -> tier (two full blocks).
        repack_into_fs_tier(&b, &geo, &spec, dir.path()).unwrap();
        // Reverse: tier -> carrier -> slot-save file, tokens supplied by the
        // caller (the tier is only hash-addressed BY them).
        let tokens: Vec<u32> = b
            .tokens
            .iter()
            .map(|&t| u32::try_from(t).unwrap())
            .collect();
        let (carrier, report) = vllm_tier_to_carrier(dir.path(), &spec, &tokens, &geo).unwrap();
        assert_eq!(report.blocks_read, 2);
        assert_eq!(report.blocks_missing, 0);
        let file = carrier_to_slot_save(&carrier).unwrap();
        // The written file must parse back through the production parser
        // into exactly the original bank bytes and tokens.
        let parsed = super::super::repack::parse_slot_save(&file, &geo).unwrap();
        assert_eq!(parsed, b);
    }

    #[test]
    fn unit__reverse_codec__stops_at_first_missing_block_and_reports() {
        let b = bank();
        let spec = VllmFsSpec {
            model_name: "m".into(),
            tokens_per_hash: 64,
            blocks_per_file: 1,
            canonical_format: "v1-nhd".into(),
        };
        let geo = geo();
        let dir = tempfile::tempdir().unwrap();
        repack_into_fs_tier(&b, &geo, &spec, dir.path()).unwrap();
        // Delete the SECOND block's file by re-walking the chain.
        let mut parent = vllm_none_hash();
        let mut second_rel = None;
        for bidx in 0..2u32 {
            let ids = &b.tokens[bidx as usize * 64..bidx as usize * 64 + 64];
            let ids: Vec<u32> = ids.iter().map(|&t| u32::try_from(t).unwrap()).collect();
            let hash = vllm_block_hash(parent, &ids);
            parent = hash;
            if bidx == 1 {
                second_rel = Some(VllmFsSpec::block_rel_path(&hex::encode(hash), 0));
            }
        }
        std::fs::remove_file(
            dir.path()
                .join(spec.rank_dir(dir.path()))
                .join(second_rel.unwrap()),
        )
        .unwrap();
        let tokens: Vec<u32> = b
            .tokens
            .iter()
            .map(|&t| u32::try_from(t).unwrap())
            .collect();
        let (carrier, report) = vllm_tier_to_carrier(dir.path(), &spec, &tokens, &geo).unwrap();
        assert_eq!(report.blocks_read, 1);
        assert_eq!(report.blocks_missing, 1);
        assert_eq!(carrier.cells, 64);
        // One-block bank still round-trips through the parser.
        let file = carrier_to_slot_save(&carrier).unwrap();
        let parsed = super::super::repack::parse_slot_save(&file, &geo).unwrap();
        assert_eq!(parsed.cells, 64);
        assert_eq!(&parsed.tokens[..64], &b.tokens[..64]);
    }

    #[test]
    fn unit__reverse_codec__empty_tier_refuses_with_a_teaching() {
        let spec = VllmFsSpec {
            model_name: "m".into(),
            tokens_per_hash: 64,
            blocks_per_file: 1,
            canonical_format: "v1-nhd".into(),
        };
        let dir = tempfile::tempdir().unwrap();
        let tokens: Vec<u32> = (0..128).collect();
        let err = vllm_tier_to_carrier(dir.path(), &spec, &tokens, &geo()).unwrap_err();
        assert!(err.contains("never offloaded"), "{err}");
    }
}
