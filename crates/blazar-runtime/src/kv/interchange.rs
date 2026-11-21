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

use std::path::{Path, PathBuf};

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
    /// mlx-lm lane -> vLLM HF lane: reads an official mlx-lm
    /// `--prompt-cache-file` safetensors export and lands it in the vLLM
    /// fs tier.
    MlxToVllm,
    /// vLLM HF lane -> mlx-lm lane: reads the engine's fs-tier blocks and
    /// stages an mlx-lm prompt-cache file the supervisor's spawn wrapper
    /// pre-loads into the mlx server's LRU.
    VllmToMlx,
    SglangToVllm,
    VllmToSglang,
}

impl CodecLane {
    /// Human-readable pair name used by CLI reports and doctor rows.
    #[must_use]
    pub fn pair_name(self) -> &'static str {
        match self {
            Self::LlamaCppToVllm => "llama.cpp -> vllm",
            Self::VllmToLlamaCpp => "vllm -> llama.cpp",
            Self::MlxToVllm => "mlx -> vllm",
            Self::VllmToMlx => "vllm -> mlx",
            Self::SglangToVllm => "sglang -> vllm",
            Self::VllmToSglang => "vllm -> sglang",
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
        vec![
            CodecLane::LlamaCppToVllm,
            CodecLane::VllmToLlamaCpp,
            CodecLane::MlxToVllm,
            CodecLane::VllmToMlx,
            CodecLane::SglangToVllm,
            CodecLane::VllmToSglang,
        ]
    } else {
        Vec::new()
    }
}

/// What the mlx-lm cache reader found, for reporting and provenance.
#[derive(Debug)]
pub struct MlxCacheReport {
    /// Exact model path recorded in the file metadata (`0.model`).
    pub model_path: Option<String>,
    /// Number of per-layer caches found (tensor tree `{i}.0`/`{i}.1`).
    pub layers: u32,
    /// Used token count (the per-layer `offset` scalars, `{i}.2`).
    pub cells: u32,
    /// Element dtype of the stored tensors, `f16` or `bf16`.
    pub source_dtype: &'static str,
    /// Whether rows were numerically converted bf16 -> f16 on the way in.
    pub converted_to_f16: bool,
}

/// Decode one little-endian bf16 pair into an f32.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Convert an f32 to IEEE 754 binary16 bits (round-to-nearest-even).
/// Works entirely in the integer bit domain so every rounding step is
/// exact by construction. Returns `None` when the value cannot be
/// represented (|v| > 65504, or NaN/inf) - the caller refuses instead of
/// emitting corrupt rows.
fn f32_to_f16_bits(v: f32) -> Option<u16> {
    if !v.is_finite() {
        return None;
    }
    let abs = v.abs();
    if abs > 65504.0 {
        return None;
    }
    let sign = if v.is_sign_negative() { 1u16 << 15 } else { 0 };
    let bits = abs.to_bits();
    let exp = ((bits >> 23) & 0xFF).cast_signed() - 127; // unbiased exponent
    let mant = bits & 0x7F_FFFF; // 23-bit mantissa
    if exp >= -14 {
        // Normal f16: keep 10 mantissa bits, round-to-nearest-even on the
        // 13 dropped bits. The +0x0FFF (one below half) plus the kept LSB
        // implements the tie rule: ties round to an even kept mantissa.
        let e = u32::try_from(exp + 15).ok()?;
        let rounded = mant + 0x0FFF + ((mant >> 13) & 1);
        let e = if rounded > 0x7F_FFFF { e + 1 } else { e };
        if e >= 0x1F {
            // Rounded past 65504 (max f16); the caller refuses.
            return None;
        }
        let frac = (rounded >> 13) & 0x3FF;
        return Some(sign | u16::try_from((e << 10) | frac).ok()?);
    }
    // Subnormal f16 territory: quantum is 2^-24, so the full significand
    // (implicit one included) shifted right by -(exp+1) lands in quanta.
    if exp < -25 {
        // Strictly below half a quantum even at max significand.
        return Some(sign);
    }
    let sig = (1u32 << 23) | mant;
    let shift = u32::try_from(-14 - exp + 13).ok()?;
    let q = sig >> shift;
    let dropped = sig & ((1u32 << shift) - 1);
    let half = 1u32 << (shift - 1);
    let mut q = q;
    if dropped > half || (dropped == half && (q & 1) == 1) {
        q += 1;
    }
    if q >= 0x400 {
        // Rounded up to the smallest normal (2^-14).
        return Some(sign | 0x0400);
    }
    Some(sign | u16::try_from(q).ok()?)
}

/// Convert a bf16 row to f16 bytes, refusing values f16 cannot hold.
fn bf16_row_to_f16(src: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(src.len());
    for pair in src.as_chunks::<2>().0 {
        let bits = u16::from_le_bytes(*pair);
        let f = bf16_to_f32(bits);
        let h = f32_to_f16_bits(f).ok_or_else(|| {
            format!(
                "value {f} overflows the f16 range the sink stores - the \
                 prefix was computed in a dtype the target lane cannot hold"
            )
        })?;
        out.extend_from_slice(&h.to_le_bytes());
    }
    Ok(out)
}

/// Read an mlx-lm `--prompt-cache-file` export (safetensors written by
/// `mlx_lm`'s `save_prompt_cache`) into a carrier. The file records the
/// model path and per-layer `(keys, values, offset)` states; tensors are
/// `[1, n_kv_heads, capacity, head_dim]` where capacity is padded to the
/// cache step (256) and the per-layer offset scalar carries the used
/// length. Token ids are NOT stored - the caller supplies them and the
/// count must equal the used length exactly.
/// Read the safetensors header of an mlx-lm prompt-cache file and the
/// byte offset where tensor data starts.
fn mlx_header(bytes: &[u8]) -> Result<(serde_json::Value, usize), String> {
    if bytes.len() < 8 {
        return Err("mlx cache file is too short to be a safetensors archive".into());
    }
    let header_len = usize::try_from(u64::from_le_bytes(bytes[..8].try_into().unwrap()))
        .map_err(|_| "mlx cache header length exceeds the addressable size".to_string())?;
    let data_start = 8 + header_len;
    if data_start > bytes.len() {
        return Err("mlx cache header length exceeds the file".into());
    }
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..data_start])
        .map_err(|e| format!("mlx cache header is not valid json: {e}"))?;
    Ok((header, data_start))
}

/// Resolve one tensor entry: (heads, seq, dim, byte range, dtype).
fn mlx_shape_of(
    tensors: &serde_json::Map<String, serde_json::Value>,
    name: &str,
) -> Result<(u32, u32, u32, u64, u64, &'static str), String> {
    let t = tensors
        .get(name)
        .and_then(|v| v.as_object())
        .ok_or_else(|| format!("mlx cache tensor {name} missing"))?;
    let dtype = t
        .get("dtype")
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("mlx cache tensor {name} has no dtype"))?;
    let dtype: &'static str = match dtype {
        "F16" => "f16",
        "BF16" => "bf16",
        other => {
            return Err(format!(
                "mlx cache tensor {name} is {other} - only f16 and bf16 \
                 KV exports are admitted"
            ));
        }
    };
    let shape = t
        .get("shape")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("mlx cache tensor {name} has no shape"))?;
    if shape.len() != 4 {
        return Err(format!(
            "mlx cache tensor {name} is not a [1, heads, seq, dim] state"
        ));
    }
    let dims: Vec<u64> = shape.iter().filter_map(serde_json::Value::as_u64).collect();
    if dims.len() != 4 || dims[0] != 1 {
        return Err(format!(
            "mlx cache tensor {name} shape must be [1, heads, seq, dim]"
        ));
    }
    let offsets = t
        .get("data_offsets")
        .and_then(|v| v.as_array())
        .and_then(|a| {
            let s = a.first()?.as_u64()?;
            let e = a.get(1)?.as_u64()?;
            Some((s, e))
        })
        .ok_or_else(|| format!("mlx cache tensor {name} has no data offsets"))?;
    let dim = |i: usize| {
        u32::try_from(dims[i]).map_err(|_| format!("mlx cache tensor {name} dimension too large"))
    };
    Ok((dim(1)?, dim(2)?, dim(3)?, offsets.0, offsets.1, dtype))
}

/// Walk the per-layer `{i}.2` offset scalars; every layer must agree on
/// the used token count (the plain `KVCache` invariant).
fn mlx_used_cells(
    tensors: &serde_json::Map<String, serde_json::Value>,
    bytes: &[u8],
    data_start: usize,
    layers: u32,
    cap: u32,
) -> Result<u32, String> {
    let mut cells: Option<i64> = None;
    for i in 0..layers {
        let t = tensors
            .get(&format!("{i}.2"))
            .and_then(|v| v.as_object())
            .ok_or_else(|| format!("mlx cache offset scalar {i}.2 missing"))?;
        let dtype = t.get("dtype").and_then(|v| v.as_str()).unwrap_or("");
        if dtype != "I32" {
            return Err(format!(
                "mlx cache offset scalar {i}.2 is {dtype:?} - expected the \
                 plain KVCache i32 offset"
            ));
        }
        let (s, e) = t
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .and_then(|a| {
                let s = a.first()?.as_u64()?;
                let e = a.get(1)?.as_u64()?;
                Some((s, e))
            })
            .ok_or_else(|| format!("mlx cache offset scalar {i}.2 has no offsets"))?;
        if e - s != 4 {
            return Err(format!("mlx cache offset scalar {i}.2 is not 4 bytes"));
        }
        let start = data_start + usize::try_from(s).unwrap_or(usize::MAX);
        let raw = bytes
            .get(start..start + 4)
            .ok_or("mlx cache offset scalar out of bounds")?;
        let value = i32::from_le_bytes(raw.try_into().unwrap());
        if value < 0 {
            return Err(format!(
                "mlx cache layer {i} offset is negative - the file is corrupt"
            ));
        }
        match cells {
            None => cells = Some(i64::from(value)),
            Some(prev) if prev == i64::from(value) => {}
            Some(prev) => {
                return Err(format!(
                    "mlx cache layer offsets disagree ({prev} vs {value}) - \
                     the file is corrupt"
                ));
            }
        }
    }
    let cells = u32::try_from(cells.unwrap_or(0))
        .map_err(|_| "mlx cache used length is absurd".to_string())?;
    if cells == 0 || cells > cap {
        return Err(format!(
            "mlx cache used length {cells} is outside its capacity {cap}"
        ));
    }
    Ok(cells)
}

/// Gather mlx [1, heads, capacity, dim] tensor pairs into the carrier's
/// layer-major token rows, converting bf16 to f16 when needed.
struct MlxGather<'a> {
    tensors: &'a serde_json::Map<String, serde_json::Value>,
    bytes: &'a [u8],
    data_start: usize,
    layers: u32,
    kv_heads: u32,
    cap: u32,
    head_dim: u32,
    cells: u32,
    row_bytes: usize,
    converted: bool,
    k_dtype: &'static str,
    k_off: u64,
    k_end: u64,
    v_off: u64,
    v_end: u64,
}

fn mlx_gather_rows(g: &MlxGather<'_>) -> Result<(Vec<u8>, Vec<u8>), String> {
    let MlxGather {
        tensors,
        bytes,
        data_start,
        layers,
        kv_heads,
        cap,
        head_dim,
        cells,
        row_bytes,
        converted,
        k_dtype,
        k_off,
        k_end,
        v_off,
        v_end,
    } = g;
    let (layers, kv_heads, cap, head_dim, cells) = (*layers, *kv_heads, *cap, *head_dim, *cells);
    let (row_bytes, converted, k_dtype) = (*row_bytes, *converted, *k_dtype);
    let (data_start, k_off, k_end, v_off, v_end) = (*data_start, *k_off, *k_end, *v_off, *v_end);
    let slice = |off: u64, end: u64| -> Result<&[u8], String> {
        let s = data_start + usize::try_from(off).unwrap_or(usize::MAX);
        let e = data_start + usize::try_from(end).unwrap_or(usize::MAX);
        bytes
            .get(s..e)
            .ok_or_else(|| "mlx cache tensor data out of bounds".to_string())
    };
    let k0 = slice(k_off, k_end)?;
    let v0 = slice(v_off, v_end)?;
    let tensor_bytes = row_bytes.saturating_mul(usize::try_from(cap).unwrap_or(usize::MAX));
    if k0.len() != tensor_bytes || v0.len() != tensor_bytes {
        return Err("mlx cache tensor byte length disagrees with its shape".into());
    }
    let mut k_rows = vec![0u8; layers as usize * cells as usize * row_bytes];
    let mut v_rows = k_rows.clone();
    let head_bytes = head_dim as usize * 2;
    for i in 0..layers {
        // Shapes were validated on layer 0; later layers only need their
        // data offsets (plus the dtype consistency check below).
        let (_, _, _, ko, ke, kd) = mlx_shape_of(tensors, &format!("{i}.0"))?;
        if kd != k_dtype {
            return Err("mlx cache layers mix dtypes".into());
        }
        let (_, _, _, vo, ve, _) = mlx_shape_of(tensors, &format!("{i}.1"))?;
        let k = slice(ko, ke)?;
        let v = slice(vo, ve)?;
        let layer_base = i as usize * cells as usize * row_bytes;
        for t in 0..cells as usize {
            let row_base = layer_base + t * row_bytes;
            for h in 0..kv_heads as usize {
                let src = (h * cap as usize + t) * head_bytes;
                let dst = row_base + h * head_bytes;
                if converted {
                    k_rows[dst..dst + head_bytes]
                        .copy_from_slice(&bf16_row_to_f16(&k[src..src + head_bytes])?);
                    v_rows[dst..dst + head_bytes]
                        .copy_from_slice(&bf16_row_to_f16(&v[src..src + head_bytes])?);
                } else {
                    k_rows[dst..dst + head_bytes].copy_from_slice(&k[src..src + head_bytes]);
                    v_rows[dst..dst + head_bytes].copy_from_slice(&v[src..src + head_bytes]);
                }
            }
        }
    }
    Ok((k_rows, v_rows))
}

pub fn mlx_cache_to_carrier(
    bytes: &[u8],
    tokens: &[u32],
) -> Result<(KvCarrier, MlxCacheReport), String> {
    let (header, data_start) = mlx_header(bytes)?;
    let tensors = header
        .as_object()
        .ok_or("mlx cache header must be an object")?;
    let metadata = tensors
        .get("__metadata__")
        .and_then(|m| m.as_object())
        .ok_or("mlx cache file carries no metadata - was it written by mlx_lm cache_prompt?")?;
    let model_path = metadata
        .get("0.model")
        .and_then(|v| v.as_str())
        .map(str::to_owned);

    // Contiguous per-layer states {i}.0 (K), {i}.1 (V), {i}.2 (offset).
    let mut layers: u32 = 0;
    loop {
        if tensors.contains_key(&format!("{layers}.0"))
            && tensors.contains_key(&format!("{layers}.1"))
            && tensors.contains_key(&format!("{layers}.2"))
        {
            layers += 1;
        } else {
            break;
        }
    }
    if layers == 0 {
        return Err("mlx cache has no per-layer (K, V, offset) states".into());
    }
    for i in 0..layers {
        let class = metadata
            .get(&format!("1.{i}"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if class != "KVCache" {
            return Err(format!(
                "mlx cache layer {i} is a {class:?} cache - only plain \
                 mlx-lm KVCache exports are admitted (rotating/quantized \
                 variants carry state the interchange does not model)"
            ));
        }
    }

    let (kv_heads, cap, head_dim, k_off, k_end, k_dtype) = mlx_shape_of(tensors, "0.0")?;
    let (_, cap_v, _, v_off, v_end, _) = mlx_shape_of(tensors, "0.1")?;
    if cap != cap_v {
        return Err("mlx cache K and V capacity disagree".into());
    }
    let row_bytes = usize::try_from(
        kv_heads
            .checked_mul(head_dim)
            .and_then(|e| e.checked_mul(2))
            .ok_or("mlx cache geometry overflows")?,
    )
    .map_err(|_| "mlx cache row size exceeds the addressable size".to_string())?;

    let cells = mlx_used_cells(tensors, bytes, data_start, layers, cap)?;
    if tokens.len() != cells as usize {
        return Err(format!(
            "token file carries {} ids but the cache holds {cells} tokens - \
             supply exactly the tokenization of the cached prompt",
            tokens.len()
        ));
    }

    let source_dtype: &'static str = if k_dtype == "f16" { "f16" } else { "bf16" };
    let converted = source_dtype == "bf16";

    let (k_rows, v_rows) = mlx_gather_rows(&MlxGather {
        tensors,
        bytes,
        data_start,
        layers,
        kv_heads,
        cap,
        head_dim,
        cells,
        row_bytes,
        converted,
        k_dtype,
        k_off,
        k_end,
        v_off,
        v_end,
    })?;

    let carrier = KvCarrier {
        tokens: tokens.to_vec(),
        cells,
        geometry: RepackGeometry {
            layers,
            kv_heads,
            head_dim,
        },
        dtype: KV_DTYPE_F16,
        block_tokens: 64,
        attention_layout: LLAMACPP_LAYOUT,
        k_rows,
        v_rows,
    };
    let report = MlxCacheReport {
        model_path,
        layers,
        cells,
        source_dtype,
        converted_to_f16: converted,
    };
    Ok((carrier, report))
}

/// Widen f16 bits to f32 bits exactly (bit-domain: every f16 value is
/// representable in f32; subnormals normalize, inf/NaN map structurally).
fn f16_bits_to_f32_bits(h: u16) -> u32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = ((h >> 10) & 0x1F).cast_signed();
    let frac = u32::from(h & 0x3FF);
    if exp == 0 {
        if frac == 0 {
            return sign; // +-0
        }
        // Subnormal f16: value = frac * 2^-24. Normalize into an f32
        // normal by shifting the significand until bit 10 is the
        // implicit-one position.
        let mut e = -1i32;
        let mut f = frac;
        while f & 0x400 == 0 {
            f <<= 1;
            e -= 1;
        }
        f &= 0x3FF;
        let f32_exp = 127 - 15 + e - 10;
        return sign | (u32::try_from(f32_exp).unwrap_or(0) << 23) | (f << 13);
    }
    if exp == 0x1F {
        return sign | 0x7F80_0000 | (frac << 13); // inf / NaN
    }
    sign | (u32::try_from(exp - 15 + 127).unwrap_or(0) << 23) | (frac << 13)
}

/// Narrow f32 bits to bf16 bits with round-to-nearest-even. The source is
/// always an f16-widened value, so overflow past bf16's range is
/// impossible and non-finite values cannot appear (the reader refuses
/// them) - the rounding bias word alone is safe.
fn f32_bits_to_bf16_bits(bits: u32) -> u16 {
    u16::try_from((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).unwrap_or(0)
}

/// Scatter-carrier report for the staged mlx direction.
#[derive(Debug)]
pub struct MlxStageReport {
    pub layers: u32,
    pub cells: u32,
    pub capacity: u32,
    /// `bf16` or `f16` - the cache dtype written, mirroring the model
    /// weights the way mlx-lm's own cache does.
    pub dtype: &'static str,
}

/// Write a carrier back out as an mlx-lm prompt-cache safetensors file:
/// per-layer `{i}.0` (K), `{i}.1` (V) shaped `[1, heads, capacity, dim]`
/// with the used-length scalar `{i}.2`, and the `__metadata__` tree
/// `load_prompt_cache` reconstructs from. Capacity rounds up to the
/// `KVCache` step (256) exactly like mlx-lm's own allocation.
///
/// `weights_dtype` is the HF config dtype string; the cache mirrors the
/// weights (mlx-lm behavior), so `bfloat16` emits bf16 rows (converted
/// from the carrier's f16 with round-to-nearest-even - f16 values are
/// always in bf16's range) and `float16` emits f16 rows verbatim.
/// Build the mlx-lm safetensors header (and the total data length it
/// describes) for a staged prompt cache - the exact schema
/// `load_prompt_cache` reconstructs from.
fn mlx_stage_header(
    geo: &RepackGeometry,
    dtype: &str,
    capacity: u32,
    model_path: &str,
) -> Result<(Vec<u8>, u64), String> {
    let tensor_bytes = usize::try_from(geo.head_dim.checked_mul(2).ok_or("head size overflows")?)
        .map_err(|_| "head size too large".to_string())?
        .checked_mul(capacity as usize)
        .and_then(|v| v.checked_mul(geo.kv_heads as usize))
        .ok_or_else(|| "tensor size overflows".to_string())?;
    let mut tensors = serde_json::Map::new();
    let mut data_len = 0u64;
    for i in 0..geo.layers {
        for (leg, dtype_name) in [("0", dtype_upper(dtype)), ("1", dtype_upper(dtype))] {
            tensors.insert(
                format!("{i}.{leg}"),
                serde_json::json!({
                    "dtype": dtype_name,
                    "shape": [1, geo.kv_heads, capacity, geo.head_dim],
                    "data_offsets": [data_len, data_len + tensor_bytes as u64],
                }),
            );
            data_len += tensor_bytes as u64;
        }
        tensors.insert(
            format!("{i}.2"),
            serde_json::json!({
                "dtype": "I32",
                "shape": [],
                "data_offsets": [data_len, data_len + 4],
            }),
        );
        data_len += 4;
    }
    let mut metadata = serde_json::Map::new();
    metadata.insert("0.model".to_string(), serde_json::json!(model_path));
    metadata.insert(
        "0.tokenizer_config".to_string(),
        serde_json::json!(r#"{"trust_remote_code": false}"#),
    );
    for i in 0..geo.layers {
        metadata.insert(format!("1.{i}"), serde_json::json!("KVCache"));
    }
    metadata.insert("2.0".to_string(), serde_json::json!(""));
    for i in 0..geo.layers {
        metadata.insert(
            format!("2.{}.0", i + 1),
            serde_json::json!(format!("{i}.2")),
        );
        metadata.insert(format!("2.{}.1", i + 1), serde_json::json!("scalar"));
    }
    let mut header_map = serde_json::Map::new();
    header_map.append(&mut tensors);
    header_map.insert(
        "__metadata__".to_string(),
        serde_json::Value::Object(metadata),
    );
    let header_bytes =
        serde_json::to_vec(&serde_json::Value::Object(header_map)).map_err(|e| e.to_string())?;
    Ok((header_bytes, data_len))
}

pub fn carrier_to_mlx_cache(
    carrier: &KvCarrier,
    model_path: &str,
    weights_dtype: &str,
) -> Result<(Vec<u8>, MlxStageReport), String> {
    let dtype: &'static str = match weights_dtype {
        "bfloat16" => "bf16",
        "float16" => "f16",
        other => {
            return Err(format!(
                "the mlx cache dtype follows the model weights - {other:?} \
                 weights are not admitted (only bfloat16 and float16)"
            ));
        }
    };
    let geo = &carrier.geometry;
    let cells_us = usize::try_from(carrier.cells).map_err(|_| "cell count too large")?;
    let layer_count = usize::try_from(geo.layers).map_err(|_| "layer count too large")?;
    if carrier.k_rows.len() != layer_count * cells_us * geo.row_bytes() {
        return Err("carrier K rows do not match the declared geometry".into());
    }
    if carrier.k_rows.len() != carrier.v_rows.len() {
        return Err("carrier K and V rows disagree in size".into());
    }
    let cells = carrier.cells;
    if cells == 0 {
        return Err("carrier holds no cached tokens".into());
    }
    let capacity = cells.div_ceil(256) * 256;
    let head_bytes = usize::try_from(geo.head_dim.checked_mul(2).ok_or("head size overflows")?)
        .map_err(|_| "head size too large")?;
    let row_bytes = geo.row_bytes();
    let (header_bytes, data_len) = mlx_stage_header(geo, dtype, capacity, model_path)?;
    let mut out =
        Vec::with_capacity(8 + header_bytes.len() + usize::try_from(data_len).unwrap_or(0));
    out.extend_from_slice(
        &u64::try_from(header_bytes.len())
            .map_err(|_| "mlx cache header exceeds the addressable size".to_string())?
            .to_le_bytes(),
    );
    out.extend_from_slice(&header_bytes);
    let cap_us = capacity as usize;
    // Data follows the header order exactly: per layer K, then V, then the
    // 4-byte offset scalar - the layout mlx-lm's tree_flatten produces.
    let scatter_layer = |rows: &[u8], layer: usize, out: &mut Vec<u8>| {
        for h in 0..geo.kv_heads as usize {
            for t in 0..cells_us {
                let src = layer * cells_us * row_bytes + t * row_bytes + h * head_bytes;
                let sl = &rows[src..src + head_bytes];
                if dtype == "bf16" {
                    for pair in sl.as_chunks::<2>().0 {
                        let f = f16_bits_to_f32_bits(u16::from_le_bytes(*pair));
                        out.extend_from_slice(&f32_bits_to_bf16_bits(f).to_le_bytes());
                    }
                } else {
                    out.extend_from_slice(sl);
                }
            }
            // Capacity padding: mlx-lm allocates in 256-token steps; the
            // unused tail is zeros by construction.
            out.resize(out.len() + (cap_us - cells_us) * head_bytes, 0);
        }
    };
    for l in 0..layer_count {
        scatter_layer(&carrier.k_rows, l, &mut out);
        scatter_layer(&carrier.v_rows, l, &mut out);
        out.extend_from_slice(&cells.to_le_bytes());
    }

    let report = MlxStageReport {
        layers: geo.layers,
        cells,
        capacity,
        dtype,
    };
    Ok((out, report))
}

/// safetensors dtype name for a logical cache dtype.
fn dtype_upper(dtype: &str) -> &'static str {
    if dtype == "bf16" { "BF16" } else { "F16" }
}

// ---------------------------------------------------------------------------
// sglang HiCache L3 file backend (0.5.23) - page-hash-addressed import/export
//
// The engine's file backend names every stored page after its radix page
// hash: SHA-256 over the parent page digest (raw 32 bytes, absent for the
// first page) followed by the page's token ids as little-endian u32 words,
// chained per page. Files land as `{hash}_{model-dashes}_{tp_rank}_{tp_size}.bin`
// (tp=1/pp=1 spawns) holding the raw page-unified object: axes
// (head_group, layer, K|V, page_token, head_in_group, head_dim) with K=0, V=1.

/// One page digest of the sglang hash chain (native ext semantics, byte-exact).
#[must_use]
pub fn sglang_page_hash(parent: Option<[u8; 32]>, page_tokens: &[u32]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    if let Some(p) = parent {
        hasher.update(p);
    }
    for t in page_tokens {
        hasher.update(t.to_le_bytes());
    }
    hasher.finalize().into()
}

/// Naming inputs for the sglang file backend on a tp=1/pp=1 spawn.
pub struct SglangPageSpec {
    /// The `--model` argv value with slashes joined by dashes, as the engine embeds it.
    pub model_name: String,
}

impl SglangPageSpec {
    #[must_use]
    pub fn model_name_of(model_path: &str) -> String {
        model_path.replace('/', "-")
    }
    /// tp=1/pp=1 suffix: `_{model}_{tp_rank}_{tp_size}`.
    #[must_use]
    pub fn config_suffix(&self) -> String {
        format!("_{}_1_1", self.model_name)
    }
    fn page_path(&self, dir: &Path, hash_hex: &str) -> PathBuf {
        dir.join(format!("{hash_hex}{}.bin", self.config_suffix()))
    }
}

/// Report of a sglang page walk, mirroring [`TierReadReport`].
pub type SglangPageReport = TierReadReport;

fn sglang_object_bytes(geo: &RepackGeometry, page: usize) -> Result<usize, String> {
    geo.layers
        .checked_mul(geo.kv_heads)
        .and_then(|v| v.checked_mul(2))
        .and_then(|v| usize::try_from(v).ok())
        .and_then(|v| v.checked_mul(page))
        .and_then(|v| v.checked_mul(geo.head_dim as usize))
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| "page object size overflows".to_string())
}

/// Read chained page files back into a carrier. Walks full pages only and
/// stops at the first missing file (the tier never stored the tail).
pub fn sglang_pages_to_carrier(
    storage_dir: &Path,
    spec: &SglangPageSpec,
    tokens: &[u32],
    geo: &RepackGeometry,
    page_tokens: u32,
) -> Result<(KvCarrier, SglangPageReport), String> {
    let page = usize::try_from(page_tokens).map_err(|_| "page tokens overflow".to_string())?;
    if tokens.len() < page {
        return Err(format!(
            "the tokens cover {} of the {page_tokens} tokens a page holds - the file tier stores full pages only",
            tokens.len()
        ));
    }
    let full_pages = tokens.len() / page;
    let head_bytes = geo.head_dim as usize * 2;
    let row_bytes = geo.kv_heads as usize * head_bytes;
    let object_bytes = sglang_object_bytes(geo, page)?;
    let layers = geo.layers as usize;
    let mut parent: Option<[u8; 32]> = None;
    // allocate at the full-walk stride up front so layer-major offsets never
    // shift as pages arrive, then truncate to what the walk actually read
    let stride_cells = full_pages * page;
    let row_volume = layers * stride_cells * row_bytes;
    let mut k_rows = vec![0u8; row_volume];
    let mut v_rows = vec![0u8; row_volume];
    let mut blocks_read = 0usize;
    for b in 0..full_pages {
        let digest = sglang_page_hash(parent, &tokens[b * page..(b + 1) * page]);
        let path = spec.page_path(storage_dir, &hex::encode(digest));
        let Ok(object) = std::fs::read(&path) else {
            if blocks_read == 0 {
                return Err(format!(
                    "no page file for the first page under {} - the sglang lane never wrote this prefix",
                    storage_dir.display()
                ));
            }
            break;
        };
        if object.len() != object_bytes {
            return Err(format!(
                "page object for block {b} holds {} bytes, expected {object_bytes} - different tensor geometry or engine version",
                object.len()
            ));
        }
        for l in 0..layers {
            for t in 0..page {
                for h in 0..geo.kv_heads as usize {
                    // axes (head_group, layer, K|V, page_token, head_in_group=1, head_dim)
                    let k_base = ((h * layers + l) * 2 * page + t) * head_bytes;
                    let v_base = ((h * layers + l) * 2 * page + page + t) * head_bytes;
                    let dst =
                        l * stride_cells * row_bytes + (b * page + t) * row_bytes + h * head_bytes;
                    k_rows[dst..dst + head_bytes]
                        .copy_from_slice(&object[k_base..k_base + head_bytes]);
                    v_rows[dst..dst + head_bytes]
                        .copy_from_slice(&object[v_base..v_base + head_bytes]);
                }
            }
        }
        parent = Some(digest);
        blocks_read += 1;
    }
    let cells = blocks_read * page;
    let keep = layers * cells * row_bytes;
    k_rows.truncate(keep);
    v_rows.truncate(keep);
    let report = SglangPageReport {
        blocks_read: u32::try_from(blocks_read).unwrap_or(u32::MAX),
        block_tokens: page_tokens,
        blocks_missing: u32::try_from(full_pages - blocks_read).unwrap_or(u32::MAX),
    };
    let carrier = KvCarrier {
        tokens: tokens[..cells].to_vec(),
        cells: u32::try_from(cells).map_err(|_| "cell count overflows".to_string())?,
        geometry: RepackGeometry {
            layers: geo.layers,
            kv_heads: geo.kv_heads,
            head_dim: geo.head_dim,
        },
        dtype: KV_DTYPE_F16,
        block_tokens: page_tokens,
        attention_layout: VLLM_LAYOUT,
        k_rows,
        v_rows,
    };
    Ok((carrier, report))
}

/// Outcome of writing carrier rows out as chained page files.
#[derive(Debug)]
pub struct SglangWriteReport {
    pub blocks_written: u32,
    pub block_tokens: u32,
    pub storage_dir: PathBuf,
}

/// Write a carrier into the sglang file backend: per full page, interleave
/// rows into the page-unified object and store it under its chained hash.
pub fn carrier_to_sglang_pages(
    carrier: &KvCarrier,
    spec: &SglangPageSpec,
    storage_dir: &Path,
    page_tokens: u32,
) -> Result<SglangWriteReport, String> {
    let page = usize::try_from(page_tokens).map_err(|_| "page tokens overflow".to_string())?;
    let full_pages = (carrier.cells as usize) / page;
    if full_pages == 0 {
        return Err(format!(
            "the carrier holds {} tokens - under one full {page_tokens}-token page, and the file tier stores full pages only",
            carrier.cells
        ));
    }
    std::fs::create_dir_all(storage_dir)
        .map_err(|e| format!("create {}: {e}", storage_dir.display()))?;
    let head_bytes = carrier.geometry.head_dim as usize * 2;
    let row_bytes = carrier.geometry.kv_heads as usize * head_bytes;
    let object_bytes = sglang_object_bytes(&carrier.geometry, page)?;
    let layers = carrier.geometry.layers as usize;
    let total_cells = carrier.cells as usize;
    let mut parent: Option<[u8; 32]> = None;
    for b in 0..full_pages {
        let digest = sglang_page_hash(parent, &carrier.tokens[b * page..(b + 1) * page]);
        let mut object = vec![0u8; object_bytes];
        for l in 0..layers {
            for t in 0..page {
                for h in 0..carrier.geometry.kv_heads as usize {
                    let k_base = ((h * layers + l) * 2 * page + t) * head_bytes;
                    let v_base = ((h * layers + l) * 2 * page + page + t) * head_bytes;
                    let src =
                        l * total_cells * row_bytes + (b * page + t) * row_bytes + h * head_bytes;
                    object[k_base..k_base + head_bytes]
                        .copy_from_slice(&carrier.k_rows[src..src + head_bytes]);
                    object[v_base..v_base + head_bytes]
                        .copy_from_slice(&carrier.v_rows[src..src + head_bytes]);
                }
            }
        }
        let path = spec.page_path(storage_dir, &hex::encode(digest));
        let tmp = path.with_extension("part");
        std::fs::write(&tmp, &object).map_err(|e| format!("write {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &path).map_err(|e| format!("rename {}: {e}", path.display()))?;
        parent = Some(digest);
    }
    Ok(SglangWriteReport {
        blocks_written: u32::try_from(full_pages).unwrap_or(u32::MAX),
        block_tokens: page_tokens,
        storage_dir: storage_dir.to_path_buf(),
    })
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
        let spec = VllmFsSpec::direct("m".into(), 64, 2, "bfloat16");
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
            vec![
                CodecLane::LlamaCppToVllm,
                CodecLane::VllmToLlamaCpp,
                CodecLane::MlxToVllm,
                CodecLane::VllmToMlx,
                CodecLane::SglangToVllm,
                CodecLane::VllmToSglang
            ]
        );
        assert_eq!(CodecLane::LlamaCppToVllm.pair_name(), "llama.cpp -> vllm");
        assert_eq!(CodecLane::VllmToLlamaCpp.pair_name(), "vllm -> llama.cpp");
    }

    #[test]
    fn unit__reverse_codec__tier_roundtrip_restores_the_original_bank() {
        let b = bank();
        let spec = VllmFsSpec::direct("m".into(), 64, 2, "bfloat16");
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
        let spec = VllmFsSpec::direct("m".into(), 64, 2, "bfloat16");
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
        let spec = VllmFsSpec::direct("m".into(), 64, 2, "bfloat16");
        let dir = tempfile::tempdir().unwrap();
        let tokens: Vec<u32> = (0..128).collect();
        let err = vllm_tier_to_carrier(dir.path(), &spec, &tokens, &geo()).unwrap_err();
        assert!(err.contains("never offloaded"), "{err}");
    }

    // ---- mlx codec tests -------------------------------------------------

    fn bf16_bits(v: f32) -> u16 {
        // Truncate the f32 mantissa - exact for the round values tests use.
        (v.to_bits() >> 16) as u16
    }

    /// Build a synthetic mlx-lm prompt-cache file: header JSON +
    /// `{i}.0`/`{i}.1` tensors `[1, heads, cap, dim]` + `{i}.2` offset
    /// scalars, mirroring `save_prompt_cache` on mlx-lm 0.32.x.
    struct MlxFixture<'a> {
        layers: u32,
        heads: u32,
        cap: u32,
        dim: u32,
        cells: u32,
        bf16: bool,
        model_path: &'a str,
        class_override: Option<&'a str>,
    }

    fn mlx_file(f: &MlxFixture) -> Vec<u8> {
        let (layers, heads, cap, dim, cells, bf16, model_path, class_override) = (
            f.layers,
            f.heads,
            f.cap,
            f.dim,
            f.cells,
            f.bf16,
            f.model_path,
            f.class_override,
        );
        let head_bytes = dim as usize * 2;
        let tensor_bytes = cap as usize * heads as usize * head_bytes;
        let mut tensors = serde_json::Map::new();
        let mut metadata = serde_json::Map::new();
        metadata.insert("0.model".into(), model_path.into());
        metadata.insert(
            "0.tokenizer_config".into(),
            "{\"trust_remote_code\":false}".into(),
        );
        let mut cursor = 0usize;
        for i in 0..layers {
            metadata.insert(format!("1.{i}"), class_override.unwrap_or("KVCache").into());
            for (leg, _fill) in [(".0", 0x11u8), (".1", 0x22u8)] {
                let name = format!("{i}{leg}");
                let start = cursor;
                cursor += tensor_bytes;
                tensors.insert(
                    name.clone(),
                    serde_json::json!({
                        "dtype": if bf16 { "BF16" } else { "F16" },
                        "shape": [1, heads, cap, dim],
                        "data_offsets": [start, cursor],
                    }),
                );
            }
            let name = format!("{i}.2");
            let start = cursor;
            cursor += 4;
            tensors.insert(
                name,
                serde_json::json!({
                    "dtype": "I32",
                    "shape": [],
                    "data_offsets": [start, cursor],
                }),
            );
        }
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_string(), metadata.into());
        for (k, v) in tensors {
            header.insert(k, v);
        }
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut out = Vec::with_capacity(8 + header_bytes.len() + cursor);
        out.extend_from_slice(&u64::try_from(header_bytes.len()).unwrap().to_le_bytes());
        out.extend_from_slice(&header_bytes);
        // Layer data: per head, per token, `dim` elements. Values encode the
        // coordinates so byte-gathering is assertable: value = (h+1) + t/10.
        for _ in 0..layers {
            for fill in [0x11u8, 0x22u8] {
                for _ in 0..(cap * heads) {
                    for _ in 0..dim {
                        if bf16 {
                            out.extend_from_slice(&[0, 0]);
                        } else {
                            out.push(fill);
                            out.push(0);
                        }
                    }
                }
            }
            out.extend_from_slice(&cells.cast_signed().to_le_bytes());
        }
        if bf16 {
            // Rewrite the bf16 payloads with real values (second pass):
            // rebuild data region for layers, tokens, heads.
            let data_start = out.len() - cursor;
            let mut w = data_start;
            for _ in 0..layers {
                for _leg in 0..2 {
                    for h in 0..heads {
                        for t in 0..cap {
                            for _d in 0..dim {
                                let v = f32::from(u16::try_from(h + 1).unwrap())
                                    + f32::from(u16::try_from(t).unwrap()) / 10.0;
                                let bits = bf16_bits(v).to_le_bytes();
                                out[w] = bits[0];
                                out[w + 1] = bits[1];
                                w += 2;
                            }
                        }
                    }
                }
                w += 4; // offset scalar
            }
        }
        out
    }

    fn f16_of(v: f32) -> u16 {
        f32_to_f16_bits(v).unwrap()
    }

    #[test]
    fn unit__f32_to_f16__pins_the_ieee_boundary_vectors() {
        assert_eq!(f16_of(0.0), 0x0000);
        assert_eq!(f16_of(-0.0), 0x8000);
        assert_eq!(f16_of(1.0), 0x3C00);
        assert_eq!(f16_of(-2.5), 0xC100);
        assert_eq!(f16_of(65504.0), 0x7BFF);
        assert_eq!(f16_of(0.5), 0x3800);
        // Round-to-nearest-even at the dropped-mantissa midpoint.
        assert_eq!(f16_of(1.0 + 2.0f32.powi(-11)), 0x3C00); // exact tie rounds to even
        assert_eq!(f16_of(2048.0 + 1.0), 0x6800); // 2049 ties to even -> 2048
        // Subnormals.
        assert_eq!(f16_of(2.0f32.powi(-24)), 0x0001);
        assert_eq!(f16_of(3.0 * 2.0f32.powi(-24)), 0x0003);
        assert_eq!(f16_of(2.0f32.powi(-25)), 0x0000); // below half-quantum
        assert_eq!(f16_of(3.0 * 2.0f32.powi(-25)), 0x0002); // tie -> even
        // Refusals.
        assert!(f32_to_f16_bits(65505.0).is_none());
        assert!(f32_to_f16_bits(f32::INFINITY).is_none());
        assert!(f32_to_f16_bits(f32::NAN).is_none());
    }

    #[test]
    fn unit__mlx_codec__parses_a_synthetic_f16_cache_without_conversion() {
        // 2 layers, 2 heads, dim 4, capacity 8, 5 used cells.
        let file = mlx_file(&MlxFixture {
            layers: 2,
            heads: 2,
            cap: 8,
            dim: 4,
            cells: 5,
            bf16: false,
            model_path: "/models/qwen.d",
            class_override: None,
        });
        let tokens: Vec<u32> = (0..5).collect();
        let (carrier, report) = mlx_cache_to_carrier(&file, &tokens).unwrap();
        assert_eq!(report.layers, 2);
        assert_eq!(report.cells, 5);
        assert_eq!(report.source_dtype, "f16");
        assert!(!report.converted_to_f16);
        assert_eq!(report.model_path.as_deref(), Some("/models/qwen.d"));
        assert_eq!(carrier.cells, 5);
        assert_eq!(carrier.geometry.layers, 2);
        assert_eq!(carrier.geometry.kv_heads, 2);
        assert_eq!(carrier.geometry.head_dim, 4);
        let row = 2 * 4 * 2; // heads * dim * 2 bytes
        assert_eq!(carrier.k_rows.len(), 2 * 5 * row);
        // F16 rows pass through untouched: token 0 of layer 0 is the first
        // `row` bytes, K fill 0x11.
        assert!(carrier.k_rows[..row].iter().step_by(2).all(|&b| b == 0x11));
        assert!(carrier.v_rows[..row].iter().step_by(2).all(|&b| b == 0x22));
    }

    #[test]
    fn unit__mlx_codec__converts_bf16_rows_and_gathers_heads_per_token() {
        // bf16 fill: value = (h+1) + t/10 per element.
        let file = mlx_file(&MlxFixture {
            layers: 2,
            heads: 2,
            cap: 8,
            dim: 4,
            cells: 2,
            bf16: true,
            model_path: "/models/qwen.d",
            class_override: None,
        });
        let tokens: Vec<u32> = (0..2).collect();
        let (carrier, report) = mlx_cache_to_carrier(&file, &tokens).unwrap();
        assert_eq!(report.source_dtype, "bf16");
        assert!(report.converted_to_f16);
        // Layer 0, token 0, head 0: value 1.0 for all 4 dims -> f16 0x3C00 LE.
        assert_eq!(
            &carrier.k_rows[0..8],
            &[0x00, 0x3C, 0x00, 0x3C, 0x00, 0x3C, 0x00, 0x3C]
        );
        // Head 1 of the same token: value 2.0 -> 0x4000.
        let row = 2 * 4 * 2;
        let head1: Vec<u8> = [0x00u8, 0x40].repeat(4);
        assert_eq!(&carrier.k_rows[8..16], &head1[..]);
        // V rows carry the same values (bf16 fill ignores the leg constant).
        assert_eq!(
            &carrier.v_rows[0..8],
            &[0x00, 0x3C, 0x00, 0x3C, 0x00, 0x3C, 0x00, 0x3C]
        );
        assert_eq!(row, 16);
    }

    #[test]
    fn unit__mlx_codec__roundtrips_into_the_vllm_tier_writer() {
        // 64 cells so the carrier floors to exactly one full block.
        let file = mlx_file(&MlxFixture {
            layers: 2,
            heads: 2,
            cap: 128,
            dim: 4,
            cells: 64,
            bf16: true,
            model_path: "/models/qwen.d",
            class_override: None,
        });
        let tokens: Vec<u32> = (1000..1064).collect();
        let (carrier, _report) = mlx_cache_to_carrier(&file, &tokens).unwrap();
        let dir = std::env::temp_dir().join("mlx-tier-roundtrip");
        let spec = VllmFsSpec {
            model_name: "/models/qwen.d".into(),
            tokens_per_hash: 64,
            blocks_per_file: 1,
            canonical_format: None,
            dtype: "bfloat16".into(),
            layers: 2,
            pcp_size: 1,
            dcp_size: 1,
            inference_engine: "vllm".into(),
            parallel_agnostic: false,
        };
        let outcome = carrier_into_vllm_tier(&carrier, &spec, &dir).unwrap();
        assert_eq!(outcome.blocks_written, 1);
        let none = super::vllm_none_hash();
        let hash = super::vllm_block_hash(none, &carrier.tokens[..64]);
        let hex = hex::encode(hash);
        let rel = VllmFsSpec::block_rel_path(&hex, 0);
        let rank = spec.rank_dir(&dir);
        let block = dir.join(&rank).join(&rel);
        let page = std::fs::read(block).unwrap();
        let _ = page.len();
        // layers(2) * heads(2) * tokens(64) * (K|V fused 8 B) = 2048 bytes.
        assert_eq!(page.len(), 2 * 2 * 64 * 2 * 4 * 2);
        assert_eq!(outcome.tail_tokens_dropped, 0);
    }

    #[test]
    fn unit__mlx_codec__refuses_variant_classes_and_mismatched_inputs() {
        let tokens: Vec<u32> = (0..5).collect();
        // RotatingKVCache class metadata.
        let file = mlx_file(&MlxFixture {
            layers: 2,
            heads: 2,
            cap: 8,
            dim: 4,
            cells: 5,
            bf16: false,
            model_path: "/m",
            class_override: Some("RotatingKVCache"),
        });
        let err = mlx_cache_to_carrier(&file, &tokens).unwrap_err();
        assert!(err.contains("RotatingKVCache"), "{err}");
        // Token count mismatch.
        let file = mlx_file(&MlxFixture {
            layers: 2,
            heads: 2,
            cap: 8,
            dim: 4,
            cells: 5,
            bf16: false,
            model_path: "/m",
            class_override: None,
        });
        let err = mlx_cache_to_carrier(&file, &[0u32; 3]).unwrap_err();
        assert!(err.contains("holds 5 tokens"), "{err}");
        // No metadata at all.
        let mut bare = mlx_file(&MlxFixture {
            layers: 1,
            heads: 1,
            cap: 4,
            dim: 2,
            cells: 2,
            bf16: false,
            model_path: "/m",
            class_override: None,
        });
        bare.truncate(8 + 2); // header claims more than remains
        let err = mlx_cache_to_carrier(&bare, &[0u32; 2]).unwrap_err();
        assert!(!err.is_empty());
    }
    #[test]
    fn unit__f16_bits_to_bf16__widens_and_rounds_the_boundary_vectors() {
        // widen f16 -> f32 -> bf16; values chosen representable in both formats
        let cases: &[(u16, u16)] = &[
            (0x0000, 0x0000), // zero
            (0x8000, 0x8000), // -zero
            (0x3C00, 0x3F80), // 1.0
            (0x4000, 0x4000), // 2.0 (f32 0x40000000 top bits)
            (0xC100, 0xC020), // -2.5
            (0x3800, 0x3F00), // 0.5
            (0x7BFF, 0x4780), // 65504: f32 0x477FE000 rounds up to bf16 0x4780
        ];
        for &(f16h, bf16h) in cases {
            let widened = f16_bits_to_f32_bits(f16h);
            assert_eq!(f32_bits_to_bf16_bits(widened), bf16h, "f16 {f16h:#06x}");
        }
    }

    #[test]
    fn unit__mlx_stage__reader_writer_roundtrip_is_lossless_for_bf16_clean_values() {
        // fixture values (h+1)+t/10 are exact in bf16 -> f16 widens exactly ->
        // f16 -> bf16 RNE returns the identical bits
        let fx = MlxFixture {
            layers: 2,
            heads: 2,
            cap: 256,
            dim: 4,
            cells: 70,
            bf16: true,
            model_path: "/models/qwen",
            class_override: None,
        };
        let tokens: Vec<u32> = (1000..1070).collect();
        let file = mlx_file(&fx);
        let (carrier, report) = mlx_cache_to_carrier(&file, &tokens).expect("parse");
        assert_eq!(report.source_dtype, "bf16");
        let (round, stage_report) =
            carrier_to_mlx_cache(&carrier, "/models/qwen", "bfloat16").expect("stage");
        assert_eq!(stage_report.layers, 2);
        assert_eq!(stage_report.cells, 70);
        assert_eq!(stage_report.capacity, 256);
        assert_eq!(stage_report.dtype, "bf16");
        let (carrier2, _) = mlx_cache_to_carrier(&round, &tokens).expect("reparse");
        assert_eq!(carrier2.geometry.layers, carrier.geometry.layers);
        assert_eq!(carrier2.cells, carrier.cells);
        assert_eq!(
            carrier2.k_rows, carrier.k_rows,
            "K rows must survive bf16->f16->bf16"
        );
        assert_eq!(
            carrier2.v_rows, carrier.v_rows,
            "V rows must survive bf16->f16->bf16"
        );
    }

    #[test]
    fn unit__mlx_stage__capacity_rounds_up_to_the_256_step() {
        let fx = MlxFixture {
            layers: 1,
            heads: 2,
            cap: 512,
            dim: 4,
            cells: 342,
            bf16: true,
            model_path: "/models/qwen",
            class_override: None,
        };
        let tokens: Vec<u32> = (0..342).collect();
        let (carrier, _) = mlx_cache_to_carrier(&mlx_file(&fx), &tokens).expect("parse");
        let (_, report) =
            carrier_to_mlx_cache(&carrier, "/models/qwen", "bfloat16").expect("stage");
        assert_eq!(report.capacity, 512);
    }
    #[test]
    #[allow(non_snake_case)]
    fn unit__sglang_page_hash__pins_the_native_extension_golden_vectors() {
        // vectors generated through sglang 0.5.23's own native hash extension
        let t1: Vec<u32> = (1000..1064).collect();
        let h1 = sglang_page_hash(None, &t1);
        assert_eq!(
            hex::encode(h1),
            "5202e60f6130ac4d1a719da4699af7be8be6db2720efe114f00388ae4df4ba1f"
        );
        let t2: Vec<u32> = (2000..2128).collect();
        let p0 = sglang_page_hash(None, &t2[..64]);
        assert_eq!(
            hex::encode(p0),
            "514bb5fa54d1b42dce09995f07b8f1ad36c67c4fa5b116c85814d173d27b0760"
        );
        let p1 = sglang_page_hash(Some(p0), &t2[64..]);
        assert_eq!(
            hex::encode(p1),
            "9b9a66563db02f172bba272f3cf8b099f2b541749c13e8ebf496b382d3aeb05e"
        );
        let part = sglang_page_hash(None, &[5, 6, 7]);
        assert_eq!(
            hex::encode(part),
            "2e4c4603cb0dea8636f8e802bfa0f24caef5f9943377f91d89fd6fbd9577cd1c"
        );
    }

    fn sglang_geo() -> RepackGeometry {
        RepackGeometry {
            layers: 2,
            kv_heads: 2,
            head_dim: 4,
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__sglang_codec__suffix_and_names_follow_the_model_path() {
        let name = SglangPageSpec::model_name_of("/t/models/qwen.d");
        assert_eq!(name, "-t-models-qwen.d");
        let spec = SglangPageSpec { model_name: name };
        assert_eq!(spec.config_suffix(), "_-t-models-qwen.d_1_1");
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__sglang_codec__write_read_roundtrip_restores_the_carrier() {
        // 128 tokens, geometry 2L x 2H x 4D (row 16B): two full 64-token pages
        let dir = tempfile::tempdir().unwrap();
        let geo = sglang_geo();
        let bank = bank();
        let carrier = KvCarrier::from_slot_bank(bank.clone(), sglang_geo());
        let spec = SglangPageSpec {
            model_name: "qwen-test".into(),
        };
        let w = carrier_to_sglang_pages(&carrier, &spec, dir.path(), 64).unwrap();
        assert_eq!(w.blocks_written, 2);
        let tokens: Vec<u32> = bank
            .tokens
            .iter()
            .map(|t| u32::try_from(*t).unwrap())
            .collect();
        let (restored, report) =
            sglang_pages_to_carrier(dir.path(), &spec, &tokens, &geo, 64).unwrap();
        assert_eq!(report.blocks_read, 2);
        assert_eq!(report.blocks_missing, 0);
        assert_eq!(restored.cells, carrier.cells);
        assert_eq!(restored.k_rows, carrier.k_rows);
        assert_eq!(restored.v_rows, carrier.v_rows);
        // two page files exist, addressed by the chained hashes
        let h0 = hex::encode(sglang_page_hash(None, &tokens[..64]));
        assert!(dir.path().join(format!("{h0}_qwen-test_1_1.bin")).is_file());
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__sglang_codec__stops_at_missing_pages_and_refuses_short_prompts() {
        let dir = tempfile::tempdir().unwrap();
        let spec = SglangPageSpec {
            model_name: "qwen-test".into(),
        };
        // nothing written: first page missing -> teaching error
        let err =
            sglang_pages_to_carrier(dir.path(), &spec, &[7; 64], &sglang_geo(), 64).unwrap_err();
        assert!(err.contains("never wrote this prefix"), "{err}");
        // short prompt: under one page
        let err =
            sglang_pages_to_carrier(dir.path(), &spec, &[1, 2, 3], &sglang_geo(), 64).unwrap_err();
        assert!(err.contains("full pages only"), "{err}");
        // write one page of 128 tokens then delete the second file: partial walk
        let bank = bank();
        let carrier = KvCarrier::from_slot_bank(bank.clone(), sglang_geo());
        carrier_to_sglang_pages(&carrier, &spec, dir.path(), 64).unwrap();
        let tokens: Vec<u32> = bank
            .tokens
            .iter()
            .map(|t| u32::try_from(*t).unwrap())
            .collect();
        let h1 = hex::encode(sglang_page_hash(
            Some(sglang_page_hash(None, &tokens[..64])),
            &tokens[64..],
        ));
        std::fs::remove_file(dir.path().join(format!("{h1}_qwen-test_1_1.bin"))).unwrap();
        let (partial, report) =
            sglang_pages_to_carrier(dir.path(), &spec, &tokens, &sglang_geo(), 64).unwrap();
        assert_eq!(report.blocks_read, 1);
        assert_eq!(report.blocks_missing, 1);
        assert_eq!(partial.cells, 64);
    }
}
