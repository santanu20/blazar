//! One-way KV data plane: llama.cpp slot-save banks into the vllm
//! filesystem offload tier.
//!
//! This is the byte mover behind `[kv.translation]`: a llama.cpp bank
//! (server slot save, container magic `0x6767_7371` version 4) holds
//! per-layer K/V tensor rows for a contiguous token prefix; the vllm
//! fs tier expects one file per cached block, addressed by vllm's own
//! chain hash over the token ids. Repacking is therefore three honest
//! replicas of upstream formats:
//!
//! 1. the bank container grammar (llama.cpp `llama_kv_cache::state_write`
//!    at the engine revision blazar lanes pin, empirically re-verified
//!    against a real bank on disk),
//! 2. vllm's block hashing (`vllm.v1.core.kv_cache_utils.hash_block_tokens`
//!    - SHA-256 over a pickle protocol-5 frame of
//!      `(parent_hash, token_tuple, None)`; every encoder branch below is
//!      pinned by golden vectors generated through vllm itself),
//! 3. vllm's fs-tier layout (`FileMapper`: digest-addressed base dir,
//!    `config.json` fields, `<h3>/<h2>/<hash>_g<group>.bin` files).
//!
//! Safety posture: an import can only ever *miss* (wrong token ids give
//! wrong hashes, vllm finds no file, cold prefill) - it can never feed
//! vllm bytes it did not itself hash-match against live prompt tokens.
//! Value fidelity is gated separately and fail-closed: lossless GGUF
//! container dtype (`general.file_type`), same upstream base model,
//! matching `RoPE` fingerprint, F16 KV tensors, equal tensor geometry,
//! and the PR4 layout verdict (`defuse`+`repage`) over the real lane
//! layouts. Quantized GGUF weights are numerically distinct from the HF
//! source and are refused with a teaching, not imported.
//!
//! The head-interleave convention (post-RoPE K row byte order) cannot be
//! proven from source reading alone, so this module shipped only after a
//! live end-to-end proof: a banked prefix restored through a real vllm
//! spawn as `cached_tokens > 0` with a sane reply.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Container magic of a llama-server slot save ('gsgg' little-endian).
pub const SLOT_SAVE_MAGIC: u32 = 0x6767_7371;
/// Container version written by the lane-pinned llama.cpp builds.
pub const SLOT_SAVE_VERSION: u32 = 4;
/// Appendix magic ('SCKP') - rewound slot checkpoints trail the KV
/// payload; import stops here and ignores the rest of the file.
pub const SLOT_CKPT_MAGIC: u32 = 0x504b_4353;
/// `GGML_TYPE_F16` - the only KV tensor dtype import admits (values are
/// copied byte-for-byte; any cast would silently change them).
pub const GGML_TYPE_F16: i32 = 1;

/// GGUF container dtypes whose weights can be byte-faithful to an HF
/// source: `LLAMA_FTYPE_ALL_F32` (0) and `LLAMA_FTYPE_MOSTLY_F16` (1).
/// F16 holds every in-range BF16 value losslessly (7-bit mantissa into
/// 10); F32 is the lossless superset. Every quantized class is refused.
pub const GGUF_FTYPE_LOSSLESS: [u64; 2] = [0, 1];

/// vllm's none-hash seed string (`init_none_hash`, deterministic path
/// with `PYTHONHASHSEED` unset and a cryptographic algo configured -
/// both true for blazar spawns).
const VLLM_NONE_HASH_SEED: &str = "vllm-none-hash";
/// Length of the hex digest suffix vllm appends to a tier base dir.
const VLLM_BASE_PATH_HASH_LEN: usize = 12;

// ---------------------------------------------------------------------------
// vllm block-hash replica
// ---------------------------------------------------------------------------

/// Append a pickled protocol-5 integer (BININT1 / BININT2 / BININT),
/// exactly as `pickle.dumps(..., protocol=5)` encodes them.
/// BININT covers up to `i32::MAX`; beyond it pickle switches to LONG1
/// (length-prefixed little-endian). Token ids live far below the switch -
/// the branch exists so the encoder stays total, not because any vocab
/// reaches it.
const BININT_MAX_U32: u32 = 2_147_483_647;

fn pickle_int(out: &mut Vec<u8>, v: u32) {
    if v <= 0xff {
        out.push(0x4b);
        out.push(u8::try_from(v).expect("guarded by the <= 0xff branch"));
    } else if v <= 0xffff {
        out.push(0x4d);
        out.extend_from_slice(
            &u16::try_from(v)
                .expect("guarded by the <= 0xffff branch")
                .to_le_bytes(),
        );
    } else if v <= BININT_MAX_U32 {
        out.push(0x4a);
        out.extend_from_slice(
            &i32::try_from(v)
                .expect("guarded by the <= BININT_MAX branch")
                .to_le_bytes(),
        );
    } else {
        let bytes = v.to_le_bytes();
        out.push(0x8a);
        out.push(u8::try_from(bytes.len()).expect("a u32 is four bytes"));
        out.extend_from_slice(&bytes);
    }
}

/// Pickle the exact shape vllm hashes: a framed protocol-5 dump of
/// `(parent_hash_bytes, token_tuple, None)`.
fn pickle_hash_frame(parent: &[u8; 32], tokens: &[u32]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(16 + 8 * tokens.len());
    // SHORT_BINUNICODE for the parent bytes + MEMOIZE
    payload.push(0x43);
    payload.push(32); // SHORT_BINBYTES length: the parent hash is fixed 32B
    payload.extend_from_slice(parent);
    payload.push(0x94);
    // The token tuple: pickle emits the fixed-arity TUPLE1/2/3 opcodes
    // for short tuples and MARK+TUPLE only for four or more items
    // (pinned by vllm-generated frames).
    let mark = tokens.len() >= 4;
    if mark {
        payload.push(0x28);
    }
    for t in tokens {
        pickle_int(&mut payload, *t);
    }
    match tokens.len() {
        1 => payload.push(0x85),
        2 => payload.push(0x86),
        3 => payload.push(0x87),
        _ => payload.push(0x74),
    }
    payload.push(0x94);
    // None, outer TUPLE3 + MEMOIZE
    payload.push(0x4e);
    payload.push(0x87);
    payload.push(0x94);

    // Same framing rule as the seed vector: STOP rides inside the
    // FRAME segment.
    payload.push(0x2e); // STOP
    let mut out = Vec::with_capacity(payload.len() + 11);
    out.extend_from_slice(&[0x80, 0x05]); // PROTO 5
    out.push(0x95); // FRAME with u64 LE segment length
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(&payload);
    out
}

/// The chain seed every first block hashes against.
#[must_use]
pub fn vllm_none_hash() -> [u8; 32] {
    // Pickled SHORT_BINUNICODE of the seed string, framed.
    let seed = VLLM_NONE_HASH_SEED.as_bytes();
    let mut payload = Vec::with_capacity(seed.len() + 4);
    payload.push(0x8c); // SHORT_BINUNICODE
    payload.push(u8::try_from(seed.len()).expect("the none-hash seed fits one byte"));
    payload.extend_from_slice(seed);
    payload.push(0x94); // MEMOIZE

    // The FRAME segment length covers every byte after the frame
    // header - the STOP terminator included (pinned by the vllm goldens).
    payload.push(0x2e); // STOP
    let mut frame = Vec::with_capacity(payload.len() + 11);
    frame.extend_from_slice(&[0x80, 0x05]);
    frame.push(0x95);
    frame.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    frame.extend_from_slice(&payload);
    let mut hasher = Sha256::new();
    hasher.update(&frame);
    hasher.finalize().into()
}

/// vllm `hash_block_tokens`: SHA-256 over the pickled
/// `(parent, tokens, None)` frame.
#[must_use]
pub fn vllm_block_hash(parent: [u8; 32], tokens: &[u32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(pickle_hash_frame(&parent, tokens));
    hasher.finalize().into()
}

/// vllm `make_block_hash_with_group_id`: block hash + big-endian group
/// id - the full on-disk key.
#[must_use]
pub fn vllm_block_key(hash: [u8; 32], group_id: u32) -> [u8; 36] {
    let mut key = [0u8; 36];
    key[..32].copy_from_slice(&hash);
    key[32..].copy_from_slice(&group_id.to_be_bytes());
    key
}

// ---------------------------------------------------------------------------
// vllm fs-tier layout replica (FileMapper)
// ---------------------------------------------------------------------------

/// The spawn parameters vllm folds into a tier's identity digest. These
/// mirror `FileMapper.from_offloading_spec(..., parallel_agnostic=True)`:
/// tp/pp are pinned to 1 and the agnostic keys are omitted entirely.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VllmFsSpec {
    /// Model name exactly as passed to vllm - for blazar spawns this is
    /// the HF directory path string.
    pub model_name: String,
    /// Tokens per hash block (vllm block size; 64 on the flash-attn
    /// lane this was proven against).
    pub tokens_per_hash: u32,
    /// Blocks per written file (offloading spec `blocks_per_chunk`).
    pub blocks_per_file: u32,
    /// Canonical format id (`v1-nhd` for the LBNHC layout flash-attn
    /// resolves to; present in the digest once the layout resolves).
    pub canonical_format: String,
}

impl VllmFsSpec {
    /// The exact canonical JSON `json.dumps(fields, sort_keys=True,
    /// separators=(',', ':'))` shape vllm digests and writes to
    /// `config.json`.
    #[must_use]
    pub fn fields_json(&self) -> String {
        let mut fields: Vec<(&str, String)> = vec![
            ("blocks_per_file", self.blocks_per_file.to_string()),
            (
                "model_name",
                format!(
                    "\"{}\"",
                    self.model_name.replace('\\', "\\\\").replace('"', "\\\"")
                ),
            ),
            ("pp_size", "1".into()),
            ("tokens_per_hash", self.tokens_per_hash.to_string()),
            ("tp_size", "1".into()),
        ];
        fields.push(("canonical_format", format!("\"{}\"", self.canonical_format)));
        fields.sort_by(|a, b| a.0.cmp(b.0));
        let body: Vec<String> = fields.iter().map(|(k, v)| format!("\"{k}\":{v}")).collect();
        format!("{{{}}}", body.join(","))
    }

    /// The 12-hex-char digest suffix of the tier base dir name.
    #[must_use]
    pub fn digest12(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.fields_json().as_bytes());
        let full: String = hex::encode(hasher.finalize());
        full[..VLLM_BASE_PATH_HASH_LEN].to_string()
    }

    /// Rank-0 tier dir under a tier root: `<safe_model>_<digest>_r0`.
    #[must_use]
    pub fn rank_dir(&self, root: &Path) -> PathBuf {
        let safe = self.model_name.replace('/', "_");
        root.join(format!("{}_{}_r0", safe, self.digest12()))
    }

    /// Relative path of one block file inside the rank dir.
    #[must_use]
    pub fn block_rel_path(hash_hex: &str, group_idx: u32) -> PathBuf {
        PathBuf::from(format!(
            "{}/{}/{}_g{}.bin",
            &hash_hex[..3],
            &hash_hex[3..5],
            hash_hex,
            group_idx
        ))
    }
}

// ---------------------------------------------------------------------------
// Bank parsing
// ---------------------------------------------------------------------------

/// Tensor geometry shared by the bank cross-checks (from GGUF metadata).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RepackGeometry {
    pub layers: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
}

impl RepackGeometry {
    /// Bytes in one F16 K (or V) row: `kv_heads * head_dim` elements.
    #[must_use]
    pub fn row_bytes(&self) -> usize {
        self.kv_heads as usize * self.head_dim as usize * 2
    }
}

/// A parsed, import-ready bank: contiguous prefix, F16 tensors,
/// row-major K and V.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotSaveBank {
    /// Sequence tokens; `tokens[..cells]` are the hashed ids.
    pub tokens: Vec<i32>,
    /// Cells (positions 0..cells-1) covered by the cache.
    pub cells: u32,
    pub layers: u32,
    /// K bytes: `layers * cells * row_bytes`, layer-major, cell-within-
    /// layer - exactly the llama.cpp token-major row order.
    pub k_rows: Vec<u8>,
    /// V bytes regrouped to the same shape (column-major stores are
    /// transposed during parse).
    pub v_rows: Vec<u8>,
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.pos.checked_add(n).ok_or("bank offset overflow")?;
        if end > self.buf.len() {
            return Err(format!(
                "bank truncated: need {} bytes at offset {}, file ends at {}",
                n,
                self.pos,
                self.buf.len()
            ));
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn i32(&mut self) -> Result<i32, String> {
        Ok(self.u32()?.cast_signed())
    }
    fn u64(&mut self) -> Result<u64, String> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }
    fn peek_u32(&self) -> Option<u32> {
        if self.pos + 4 > self.buf.len() {
            return None;
        }
        Some(u32::from_le_bytes([
            self.buf[self.pos],
            self.buf[self.pos + 1],
            self.buf[self.pos + 2],
            self.buf[self.pos + 3],
        ]))
    }
}

/// One raw KV section as laid out on disk (pre-regroup).
struct RawSection {
    cells: u32,
    v_trans: bool,
    layers: u32,
    /// K payload, layer-major rows.
    k: Vec<u8>,
    /// V payload in stored order.
    v: Vec<u8>,
    v_size_el: usize,
}

/// Walk one section starting at `cur`. `cell_ext` selects the DSA-class
/// 12-byte-per-cell extension record; llama.cpp emits it only for
/// models whose KV cells carry (x, y, token) coordinates, so import
/// tries the common no-ext walk first and retries with ext on failure.
/// Walk the per-stream cell metadata and return the validated cell
/// count. `cell_ext` selects the DSA-class 12-byte-per-cell extension
/// record. Import hashes blocks chained from sequence position zero, so
/// the prefix-rooted, contiguous invariants are proven right here.
fn parse_section_cells(cur: &mut Cursor, cell_ext: bool) -> Result<u32, String> {
    let n_stream = cur.u32()?;
    if n_stream != 1 {
        return Err(format!(
            "bank stream count {n_stream} - import handles the single-stream banks blazar writes"
        ));
    }
    let cells = cur.u32()?;
    if cells == 0 || cells > 1 << 24 {
        return Err(format!("bank cell count {cells} outside sane bounds"));
    }
    let mut positions = Vec::with_capacity(cells as usize);
    for _ in 0..cells {
        let pos = cur.i32()?;
        if pos < 0 {
            return Err("bank cell position negative".into());
        }
        let n_seq = cur.u32()?;
        if n_seq == 0 || n_seq > 1 << 16 {
            return Err(format!("cell seq count {n_seq} outside sane bounds"));
        }
        if cell_ext {
            cur.take(12)?;
        }
        cur.take(n_seq as usize * 4)?;
        positions.push(pos.cast_unsigned());
    }
    if positions.first().copied() != Some(0) {
        return Err(format!(
            "bank cells start at position {} - vllm hashes from position 0, so only prefix-rooted banks import",
            positions.first().copied().unwrap_or(0)
        ));
    }
    for w in positions.windows(2) {
        if w[1] != w[0] + 1 {
            return Err(format!(
                "bank positions fragment at {} -> {} - import wants one contiguous prefix",
                w[0], w[1]
            ));
        }
    }
    Ok(cells)
}

fn parse_section(cur: &mut Cursor, cell_ext: bool, row_bytes: usize) -> Result<RawSection, String> {
    let cells = parse_section_cells(cur, cell_ext)?;
    let v_trans_raw = cur.u32()?;
    if v_trans_raw > 1 {
        return Err(format!("bank v_trans flag {v_trans_raw} is not boolean"));
    }
    let v_trans = v_trans_raw == 1;
    let layers = cur.u32()?;
    if layers == 0 || layers > 1 << 12 {
        return Err(format!("bank layer count {layers} outside sane bounds"));
    }
    let n_rot_k = cur.u32()?;
    let n_rot_v = cur.u32()?;
    let _ = (n_rot_k, n_rot_v); // informational in the container; RoPE equality is gated on metadata

    let mut k = Vec::new();
    for _ in 0..layers {
        let ggml_type = cur.i32()?;
        if ggml_type != GGML_TYPE_F16 {
            return Err(format!(
                "bank K tensor dtype ggml_type={ggml_type} - import copies F16 bytes only, any cast would change values"
            ));
        }
        let size_row = usize::try_from(cur.u64()?)
            .map_err(|_| "bank K row length exceeds addressable memory".to_string())?;
        if size_row != row_bytes {
            return Err(format!(
                "bank K row is {size_row} bytes, model geometry says {row_bytes} (kv_heads * head_dim * 2)"
            ));
        }
        k.extend_from_slice(cur.take(cells as usize * size_row)?);
    }

    let mut v = Vec::new();
    let mut v_size_el = 0usize;

    for _ in 0..layers {
        let ggml_type = cur.i32()?;
        if ggml_type != GGML_TYPE_F16 {
            return Err(format!(
                "bank V tensor dtype ggml_type={ggml_type} - import copies F16 bytes only, any cast would change values"
            ));
        }
        if v_trans {
            v_size_el = cur.u32()? as usize;
            let v_cols = cur.u32()? as usize;
            // Column-major store: every column holds one element per
            // cell; the walk already proved positions contiguous, so
            // each column is one `cells * v_size_el` run.
            if v_cols.checked_mul(v_size_el) != Some(row_bytes) {
                return Err(format!(
                    "bank V geometry {v_cols} columns x {v_size_el} bytes disagrees with model row bytes {row_bytes}"
                ));
            }
            v.extend_from_slice(cur.take(v_cols * cells as usize * v_size_el)?);
        } else {
            let size_row = usize::try_from(cur.u64()?)
                .map_err(|_| "bank V row length exceeds addressable memory".to_string())?;
            if size_row != row_bytes {
                return Err(format!(
                    "bank V row is {size_row} bytes, model geometry says {row_bytes}"
                ));
            }
            v.extend_from_slice(cur.take(cells as usize * size_row)?);
        }
    }

    Ok(RawSection {
        cells,
        v_trans,
        layers,
        k,
        v,
        v_size_el,
    })
}

/// Parse a llama-server slot save into an import-ready bank, validating
/// every length against `geo` (from GGUF metadata).
// Binary slot-save grammar: magic/version/token header, per-stream cell
// metadata, per-layer F16 rows, optional SCKP appendix - section for
// section the mirror of the engine's writer.
#[allow(clippy::too_many_lines)]
pub fn parse_slot_save(buf: &[u8], geo: &RepackGeometry) -> Result<SlotSaveBank, String> {
    let mut cur = Cursor::new(buf);
    if cur.u32()? != SLOT_SAVE_MAGIC {
        return Err("not a llama.cpp slot-save bank (magic mismatch)".into());
    }
    let version = cur.u32()?;
    if version != SLOT_SAVE_VERSION {
        return Err(format!(
            "slot-save container version {version} - import pins version {SLOT_SAVE_VERSION} (the lane engines write)"
        ));
    }
    let n_tokens = cur.u32()? as usize;
    if n_tokens == 0 || n_tokens > 1 << 24 {
        return Err(format!("bank token count {n_tokens} outside sane bounds"));
    }
    let mut raw = Vec::with_capacity(n_tokens);
    for _ in 0..n_tokens {
        raw.push(cur.i32()?);
    }
    // The engine's slot save serializes its token list with a small header
    // (`server_tokens::serialize` in llama.cpp's server-common.cpp): a -1
    // format marker, a u32 state version, a u32 real-token count, the real
    // ids, then a u32 media-key count. Older engines wrote a plain id list;
    // both parse. Real ids are exactly what the engine tokenized - the ids
    // the receiving lane will hash.
    let tokens: Vec<i32> = if raw.first() == Some(&-1) {
        if raw.len() < 4 {
            return Err("bank token header truncated (server_tokens serialization)".into());
        }
        if raw[1] != 1 {
            return Err(format!(
                "bank token serialization version {} - import pins version 1 (the lane engines write)",
                raw[1]
            ));
        }
        let Ok(n_real) = u32::try_from(raw[2]) else {
            return Err("bank real-token count is negative".into());
        };
        let n_real = n_real as usize;
        if n_real == 0 || raw.len() != 3 + n_real + 1 {
            return Err(format!(
                "bank token serialization declares {n_real} real ids in a {}-slot array - malformed",
                raw.len()
            ));
        }
        if raw[3 + n_real] != 0 {
            return Err(
                "bank carries multimodal media chunks - KV import handles text-only banks".into(),
            );
        }
        raw[3..3 + n_real].to_vec()
    } else {
        raw
    };

    let row_bytes = geo.row_bytes();
    let mut first: Option<RawSection> = None;
    while let Some(next) = cur.peek_u32() {
        if next == SLOT_CKPT_MAGIC {
            // Rewound slot checkpoints trail the payload; irrelevant to
            // import by design.
            break;
        }
        if first.is_some() {
            return Err(
                "bank carries multiple KV sections (hybrid/iswa cache) - import handles the single-section dense layout only".into(),
            );
        }
        // Try the common no-ext walk first; only DSA-class cells carry
        // the 12-byte extension record, so an ext bank fails the plain
        // walk structurally and is retried with the records enabled.
        let start = cur.pos;
        let sec = match parse_section(&mut cur, false, row_bytes) {
            Ok(s) => s,
            Err(no_ext_err) => {
                cur.pos = start;
                parse_section(&mut cur, true, row_bytes).map_err(|ext_err| {
                    format!("{no_ext_err} (with DSA cell-ext records: {ext_err})")
                })?
            }
        };
        if sec.layers != geo.layers {
            return Err(format!(
                "bank holds {} layers, model geometry says {}",
                sec.layers, geo.layers
            ));
        }
        first = Some(sec);
    }
    let Some(sec) = first else {
        return Err("bank holds no KV section".into());
    };
    if (sec.cells as usize) > tokens.len() {
        return Err(format!(
            "bank covers {} cells but only {} tokens - ids needed for hashing are missing",
            sec.cells,
            tokens.len()
        ));
    }
    for t in &tokens[..sec.cells as usize] {
        if *t < 0 {
            return Err(
                "bank token stream has gap markers (-1) inside the cached prefix - hashable ids are missing".into(),
            );
        }
    }

    let cell_bytes = sec.cells as usize;
    let layer_stride = cell_bytes * row_bytes;
    let mut v_rows = Vec::with_capacity(sec.v.len());
    if sec.v_trans {
        // Transpose every layer's (columns, cells) element-major store
        // into token-major rows.
        for l in 0..sec.layers as usize {
            let base = l * layer_stride;
            for t in 0..cell_bytes {
                for j in 0..(row_bytes / sec.v_size_el) {
                    let col_base = base + j * cell_bytes * sec.v_size_el;
                    let start = col_base + t * sec.v_size_el;
                    v_rows.extend_from_slice(&sec.v[start..start + sec.v_size_el]);
                }
            }
        }
    } else {
        v_rows = sec.v;
    }

    Ok(SlotSaveBank {
        tokens,
        cells: sec.cells,
        layers: sec.layers,
        k_rows: sec.k,
        v_rows,
    })
}

// ---------------------------------------------------------------------------
// Gates
// ---------------------------------------------------------------------------

/// Repo-id normalization for the provenance compare: converters spell the
/// same upstream id with cosmetic variance (`Qwen/Qwen2.5-1.5B-Instruct`
/// vs `qwen2.5_1.5b-instruct`), so the gate compares lowercase-alphanumeric
/// forms while still requiring the SAME model identity.
fn normalize_model_id(id: &str) -> String {
    id.chars()
        .filter(char::is_ascii_alphanumeric)
        .collect::<String>()
        .to_lowercase()
}

/// Weight-provenance gate: the GGUF must be a lossless container of the
/// same upstream model as the HF target.
pub fn provenance_gate(
    gguf_file_type: Option<u64>,
    gguf_base_model: Option<&str>,
    hf_repo: &str,
) -> Result<(), String> {
    let ft = gguf_file_type.ok_or(
        "GGUF carries no general.file_type - cannot prove lossless weights, refusing import",
    )?;
    if !GGUF_FTYPE_LOSSLESS.contains(&ft) {
        return Err(format!(
            "GGUF file_type={ft} is a lossy quantization of the HF weights - KV bytes computed from it never transfer honestly; use an f16 conversion of the same HF source"
        ));
    }
    let base = gguf_base_model
        .ok_or("GGUF carries no base-model id - cannot tie it to the HF source, refusing import (set general.base_model_id to the HF repo id, e.g. via llama.cpp's gguf-set-metadata)")?;
    if normalize_model_id(base) != normalize_model_id(hf_repo) {
        return Err(format!(
            "GGUF base model {base:?} is not the HF target {hf_repo:?} - refusing to import across different models"
        ));
    }
    Ok(())
}

/// `RoPE` fingerprint equality: cached positions are meaningless across
/// different positional configurations.
pub fn rope_gate(gguf_rope: Option<&str>, hf_rope: Option<&str>) -> Result<(), String> {
    match (gguf_rope, hf_rope) {
        (Some(g), Some(h)) if g == h => Ok(()),
        (Some(g), Some(h)) => Err(format!(
            "rope fingerprints differ ({g} vs {h}) - cached positions are meaningless, refusing import"
        )),
        _ => Err("rope fingerprint missing on one side - cannot prove positional equality, refusing import".into()),
    }
}

// ---------------------------------------------------------------------------
// Repack + write
// ---------------------------------------------------------------------------

/// What an import wrote, for the CLI report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepackOutcome {
    pub blocks_written: u32,
    pub tail_tokens_dropped: u32,
    pub block_tokens: u32,
    pub files: Vec<PathBuf>,
    pub config_json_written: bool,
}

/// Repack a parsed bank into a vllm fs tier under `root`.
///
/// Layout written: one file per full block at
/// `<rank_dir>/<h3>/<h2>/<hash>_g0.bin`, each holding `layers` pages in
/// layer order, every page the canonical `v1-nhd` block
/// `(heads, block_tokens, 2*head_size)` - K half then V half per
/// (head, token). Tail tokens that never fill a block are dropped:
/// vllm stores full blocks only.
pub fn repack_into_fs_tier(
    bank: &SlotSaveBank,
    geo: &RepackGeometry,
    spec: &VllmFsSpec,
    root: &Path,
) -> Result<RepackOutcome, String> {
    let block_tokens = spec.tokens_per_hash;
    if block_tokens == 0 || block_tokens > 4096 {
        return Err(format!(
            "tokens_per_hash={block_tokens} outside sane bounds"
        ));
    }
    let blocks = bank.cells / block_tokens;
    if blocks == 0 {
        return Err(format!(
            "bank holds {} cells - a full {}-token block is the minimum vllm stores",
            bank.cells, block_tokens
        ));
    }
    let tail = bank.cells - blocks * block_tokens;

    let row = geo.row_bytes();
    let head_bytes = geo.head_dim as usize * 2;
    let layer_page = geo.kv_heads as usize * block_tokens as usize * 2 * head_bytes;
    let block_file = bank.layers as usize * layer_page;
    let mut page = Vec::with_capacity(block_file);

    let rank_dir = spec.rank_dir(root);
    let config_path = rank_dir.join("config.json");
    if rank_dir.exists() && config_path.exists() {
        let existing = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("tier config.json unreadable: {e}"))?;
        if existing.trim() != spec.fields_json() {
            return Err(format!(
                "tier dir {} already belongs to a different engine configuration - remove it or point the import at its exact spawn parameters",
                rank_dir.display()
            ));
        }
    } else if !config_path.exists() {
        std::fs::create_dir_all(&rank_dir)
            .map_err(|e| format!("cannot create tier dir {}: {e}", rank_dir.display()))?;
        std::fs::write(&config_path, format!("{}\n", spec.fields_json()))
            .map_err(|e| format!("cannot write tier config.json: {e}"))?;
    }

    let mut parent = vllm_none_hash();
    let mut files = Vec::new();
    for b in 0..blocks {
        let ids: Vec<u32> = bank.tokens
            [(b * block_tokens) as usize..(b + 1) as usize * block_tokens as usize]
            .iter()
            .map(|&t| t.cast_unsigned())
            .collect();
        let hash = vllm_block_hash(parent, &ids);
        parent = hash;

        page.clear();
        let layer_stride = bank.cells as usize * row;
        for l in 0..bank.layers as usize {
            let k_base = l * layer_stride;
            let v_base = l * layer_stride;
            for h in 0..geo.kv_heads as usize {
                let h_off = h * head_bytes;
                for t in 0..block_tokens as usize {
                    let k_row = k_base + t * row + h_off;
                    page.extend_from_slice(&bank.k_rows[k_row..k_row + head_bytes]);
                    let v_row = v_base + t * row + h_off;
                    page.extend_from_slice(&bank.v_rows[v_row..v_row + head_bytes]);
                }
            }
        }

        let hash_hex = hex::encode(hash);
        let rel = VllmFsSpec::block_rel_path(&hash_hex, 0);
        let dest = rank_dir.join(&rel);
        if let Some(dir) = dest.parent() {
            std::fs::create_dir_all(dir)
                .map_err(|e| format!("cannot create tier subdir {}: {e}", dir.display()))?;
        }
        // Same discipline as the engine: write aside, rename in.
        let tmp = dest.with_extension("bin.part");
        std::fs::write(&tmp, &page).map_err(|e| format!("cannot write block file: {e}"))?;
        std::fs::rename(&tmp, &dest).map_err(|e| format!("cannot publish block file: {e}"))?;
        files.push(dest);
    }

    Ok(RepackOutcome {
        blocks_written: blocks,
        tail_tokens_dropped: tail,
        block_tokens,
        files,
        config_json_written: true,
    })
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use super::*;

    fn hexhash(s: &str) -> String {
        let mut h = Sha256::new();
        h.update(s.as_bytes());
        hex::encode(h.finalize())
    }

    #[test]
    fn unit__vllm_none_hash__matches_the_vllm_generated_golden() {
        // Vector generated through vllm 0.34.0's own init_none_hash.
        assert_eq!(
            hex::encode(vllm_none_hash()),
            "066b1e8987e027138e98701c46128929378935f9b6cd1549105ae6b4f0d71198"
        );
    }

    #[test]
    fn unit__vllm_block_hash__reproduces_vllm_golden_vectors() {
        // (1, 2, 3, 99) against the none hash, generated through vllm.
        let h = vllm_block_hash(vllm_none_hash(), &[1, 2, 3, 99]);
        assert_eq!(
            hex::encode(h),
            "92629fdf3bd78db0e4934e5bb97bb20230f2b4b4a558cbf701b1b0eb998fa78e"
        );
        // Integer encodings across the BININT1/BININT2/BININT classes:
        // every id below was hashed through vllm's own path.
        for (ids, want_prefix) in [
            (vec![255u32], "1d560901"),
            (vec![256], "ae2a0712"),
            (vec![151_643], "5882e043"),
        ] {
            let got = hex::encode(vllm_block_hash(vllm_none_hash(), &ids));
            assert!(
                got.starts_with(want_prefix),
                "{ids:?} -> {got} (want {want_prefix}..)"
            );
        }
    }

    #[test]
    fn unit__vllm_block_hash__chains_two_sixty_four_token_blocks_like_vllm() {
        // Chain vectors from vllm over range(1000, 1064) then
        // range(2000, 2064): proves frame layout holds at real block
        // size and that chaining is parent-hash inclusion, not reseed.
        let t1: Vec<u32> = (1000..1064).collect();
        let h1 = vllm_block_hash(vllm_none_hash(), &t1);
        assert_eq!(
            hex::encode(h1),
            "3c80a9847617782fa8ff4e9408d048639ccf2d53b0ba4ddbf34be62b67340f84"
        );
        let t2: Vec<u32> = (2000..2064).collect();
        let h2 = vllm_block_hash(h1, &t2);
        assert_eq!(
            hex::encode(h2),
            "02ee619ec6e92b7834606f049f5618337a6b45847a31ff6d2b3bebb22e9716eb"
        );
    }

    #[test]
    fn unit__vllm_fs_spec__digest_matches_vllm_canonical_json() {
        // Digest computed by vllm's FileMapper over these fields.
        let spec = VllmFsSpec {
            model_name: "qwen".into(),
            tokens_per_hash: 64,
            blocks_per_file: 1,
            canonical_format: "v1-nhd".into(),
        };
        assert_eq!(spec.digest12(), "b3bb9d5340d7");
        assert_eq!(
            spec.fields_json(),
            "{\"blocks_per_file\":1,\"canonical_format\":\"v1-nhd\",\"model_name\":\"qwen\",\"pp_size\":1,\"tokens_per_hash\":64,\"tp_size\":1}"
        );
    }

    #[test]
    fn unit__vllm_fs_spec__model_paths_are_slash_safe_and_ranked() {
        let spec = VllmFsSpec {
            model_name: "/data/models/qwen.d".into(),
            tokens_per_hash: 64,
            blocks_per_file: 1,
            canonical_format: "v1-nhd".into(),
        };
        let d = spec.rank_dir(Path::new("/kv"));
        assert!(d.to_string_lossy().starts_with("/kv/_data_models_qwen.d_"));
        assert!(d.to_string_lossy().ends_with("_r0"));
        assert_eq!(
            VllmFsSpec::block_rel_path("abcdef0123456789", 0),
            PathBuf::from("abc/de/abcdef0123456789_g0.bin")
        );
    }

    fn geo() -> RepackGeometry {
        RepackGeometry {
            layers: 2,
            kv_heads: 2,
            head_dim: 4,
        } // row = 16 B
    }

    /// Build a minimal valid slot save: `cells` cells pos 0..cells-1,
    /// single seq id, F16 rows filled layer-major with per-cell markers.
    /// Re-wrap a legacy synthetic bank's plain token list in the
    /// `server_tokens::serialize` envelope the engine actually writes:
    /// [-1 marker][state version 1][real count][real ids][media count 0].
    fn with_serialized_tokens(legacy: &[u8], version: i32, media_keys: u32) -> Vec<u8> {
        let n = u32::from_le_bytes(legacy[8..12].try_into().unwrap());
        // u32 -> usize only widens; keep byte math in usize for indexing.
        let n_us = n as usize;
        let toks: Vec<i32> = (0..n)
            .map(|i| {
                let at = 12 + 4 * i as usize;
                i32::from_le_bytes(legacy[at..at + 4].try_into().unwrap())
            })
            .collect();
        let mut out = Vec::new();
        out.extend_from_slice(&legacy[..8]);
        out.extend_from_slice(&(n + 4).to_le_bytes());
        out.extend_from_slice(&(-1i32).to_le_bytes());
        out.extend_from_slice(&version.to_le_bytes());
        out.extend_from_slice(&n.to_le_bytes());
        for t in toks {
            out.extend_from_slice(&t.to_le_bytes());
        }
        out.extend_from_slice(&media_keys.to_le_bytes());
        out.extend_from_slice(&legacy[12 + 4 * n_us..]);
        out
    }

    fn synthetic_bank(cells: u32, v_trans: bool) -> Vec<u8> {
        let g = geo();
        let row = g.row_bytes();
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&SLOT_SAVE_MAGIC.to_le_bytes());
        buf.extend_from_slice(&SLOT_SAVE_VERSION.to_le_bytes());
        buf.extend_from_slice(&cells.to_le_bytes());
        for t in 0..cells {
            buf.extend_from_slice(&(1000 + t).to_le_bytes());
        }
        // section
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_stream
        buf.extend_from_slice(&cells.to_le_bytes());
        for p in 0..cells {
            buf.extend_from_slice(&p.cast_signed().to_le_bytes());
            buf.extend_from_slice(&1u32.to_le_bytes()); // one seq id
            buf.extend_from_slice(&0i32.to_le_bytes());
        }
        buf.extend_from_slice(&u32::from(v_trans).to_le_bytes());
        buf.extend_from_slice(&g.layers.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes()); // n_rot_k
        buf.extend_from_slice(&8u32.to_le_bytes()); // n_rot_v
        // K: layer-major, per cell a row of 0x10 | cell, per layer a
        // base offset so cross-layer order is asserted.
        for l in 0..g.layers {
            buf.extend_from_slice(&GGML_TYPE_F16.to_le_bytes());
            buf.extend_from_slice(&(row as u64).to_le_bytes());
            for t in 0..cells {
                for b in 0..row {
                    let tag = (l << 6)
                        | ((t & 0x3f) << 2)
                        | u32::try_from(b & 0x3).expect("2-bit marker");
                    buf.push(u8::try_from(tag).expect("tag fits a byte"));
                }
            }
        }
        if v_trans {
            for l in 0..g.layers {
                buf.extend_from_slice(&GGML_TYPE_F16.to_le_bytes());
                buf.extend_from_slice(&2u32.to_le_bytes()); // v_size_el (F16)
                let v_cols = u32::try_from(row / 2).expect("row fits u32");
                buf.extend_from_slice(&v_cols.to_le_bytes()); // n_embd_v_gqa columns
                for j in 0..row / 2 {
                    let col = u32::try_from(j).expect("column fits u32");
                    for _t in 0..cells {
                        let low = (l << 6) | 0x20 | col;
                        let high = (l << 6) | 0x21 | col;
                        buf.push(u8::try_from(low).expect("tag fits a byte"));
                        buf.push(u8::try_from(high).expect("tag fits a byte"));
                    }
                }
            }
        } else {
            for l in 0..g.layers {
                buf.extend_from_slice(&GGML_TYPE_F16.to_le_bytes());
                buf.extend_from_slice(&(row as u64).to_le_bytes());
                for t in 0..cells {
                    for b in 0..row {
                        let tag = (l << 6)
                            | 0x10
                            | ((t & 0x3f) << 2)
                            | u32::try_from(b & 0x3).expect("2-bit marker");
                        buf.push(u8::try_from(tag).expect("tag fits a byte"));
                    }
                }
            }
        }
        buf
    }

    #[test]
    fn unit__parse_slot_save__walks_a_synthetic_bank_to_the_exact_eof() {
        let buf = synthetic_bank(3, false);
        let bank = parse_slot_save(&buf, &geo()).expect("walk");
        assert_eq!(bank.cells, 3);
        assert_eq!(bank.layers, 2);
        assert_eq!(bank.tokens, vec![1000, 1001, 1002]);
        assert_eq!(bank.k_rows.len(), 2 * 3 * 16);
        // layer-major: first 48 bytes are layer 0.
        assert!(bank.k_rows[..48].iter().all(|b| b & 0xc0 == 0));
        assert!(bank.k_rows[48..].iter().all(|b| b & 0xc0 == 0x40));
    }

    #[test]
    fn unit__parse_slot_save__regroups_column_major_values_into_rows() {
        let buf = synthetic_bank(2, true);
        let bank = parse_slot_save(&buf, &geo()).expect("walk");
        // Row t must interleave one element from each column: col marker
        // 0x2j followed by its 0x21 partner.
        assert_eq!(bank.v_rows.len(), 2 * 2 * 16);
        let row0 = &bank.v_rows[..16];
        for j in 0..8 {
            assert_eq!(
                row0[2 * j] & 0x0f,
                u8::try_from(j).expect("column marker fits a byte"),
                "col {j} must land in row 0 slot {j}"
            );
        }
    }

    #[test]
    fn unit__parse_slot_save__stops_at_the_sckp_appendix() {
        let mut buf = synthetic_bank(2, false);
        buf.extend_from_slice(&SLOT_CKPT_MAGIC.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&[0xffu8; 64]); // junk appendix payload
        let bank = parse_slot_save(&buf, &geo()).expect("appendix ignored");
        assert_eq!(bank.cells, 2);
    }

    #[test]
    fn unit__parse_slot_save__refuses_broken_banks_with_teachings() {
        let good = synthetic_bank(2, false);
        // bad magic
        let mut m = good.clone();
        m[0] = 0;
        assert!(parse_slot_save(&m, &geo()).unwrap_err().contains("magic"));
        // bad version
        let mut v = good.clone();
        v[4..8].copy_from_slice(&9u32.to_le_bytes());
        assert!(parse_slot_save(&v, &geo()).unwrap_err().contains("version"));
        // truncated payload
        assert!(
            parse_slot_save(&good[..good.len() - 10], &geo())
                .unwrap_err()
                .contains("truncated")
        );
        // geometry mismatch: parser built for a different row size
        let wide = RepackGeometry {
            layers: 2,
            kv_heads: 4,
            head_dim: 4,
        };
        assert!(parse_slot_save(&good, &wide).unwrap_err().contains("row"));
        // gap marker inside the cached prefix (index 0 is the serialized
        // format marker in real banks, so punch the second id)
        let mut g = good.clone();
        let tok_at = 16; // header 12 + one i32
        g[tok_at..tok_at + 4].copy_from_slice(&(-1i32).to_le_bytes());
        assert!(
            parse_slot_save(&g, &geo())
                .unwrap_err()
                .contains("gap markers")
        );
    }

    #[test]
    fn unit__provenance_and_rope_gates__refuse_everything_unprovable() {
        assert!(
            provenance_gate(
                None,
                Some("Qwen/Qwen2.5-1.5B-Instruct"),
                "Qwen/Qwen2.5-1.5B-Instruct"
            )
            .unwrap_err()
            .contains("file_type")
        );
        assert!(
            provenance_gate(Some(7), Some("x"), "x")
                .unwrap_err()
                .contains("lossy")
        );
        assert!(
            provenance_gate(Some(1), None, "x")
                .unwrap_err()
                .contains("base-model")
        );
        assert!(
            provenance_gate(Some(1), Some("a"), "b")
                .unwrap_err()
                .contains("different models")
        );
        assert!(
            provenance_gate(
                Some(1),
                Some("Qwen/Qwen2.5-1.5B-Instruct"),
                "Qwen/Qwen2.5-1.5B-Instruct"
            )
            .is_ok()
        );
        assert!(rope_gate(Some("r"), Some("r")).is_ok());
        assert!(
            rope_gate(Some("r"), Some("s"))
                .unwrap_err()
                .contains("rope")
        );
    }

    #[test]
    fn unit__provenance_gate__cosmetic_repo_id_spelling_variants_pass() {
        // Converters write the same upstream id with case/dash/underscore
        // variance; the gate compares normalized forms so the legit pair is
        // importable while a genuinely different model still refuses.
        assert!(
            provenance_gate(
                Some(1),
                Some("qwen/qwen2.5-1.5b-instruct"),
                "Qwen/Qwen2.5-1.5B-Instruct"
            )
            .is_ok()
        );
        assert!(
            provenance_gate(
                Some(0),
                Some("Qwen/Qwen2.5-1.5B-Instruct"),
                "qwen/qwen2.5_1.5b_instruct"
            )
            .is_ok()
        );
        // Missing the org namespace is a different identity string, not
        // a spelling variant - the gate stays fail-closed on it.
        assert!(
            provenance_gate(
                Some(1),
                Some("qwen2.5-1.5b-instruct"),
                "Qwen/Qwen2.5-1.5B-Instruct"
            )
            .unwrap_err()
            .contains("different models")
        );
        // Same family, different member: must still refuse.
        assert!(
            provenance_gate(
                Some(1),
                Some("Qwen/Qwen2.5-0.5B-Instruct"),
                "Qwen/Qwen2.5-1.5B-Instruct"
            )
            .unwrap_err()
            .contains("different models")
        );
        assert!(
            provenance_gate(Some(1), None, "Qwen/Qwen2.5-1.5B-Instruct")
                .unwrap_err()
                .contains("gguf-set-metadata")
        );
        assert!(rope_gate(Some("r"), None).unwrap_err().contains("missing"));
    }

    #[test]
    fn unit__repack__writes_hash_addressed_pages_and_drops_the_tail() {
        let dir = tempfile::tempdir().expect("tmp");
        let cells = 130u32; // 2 full 64-blocks + 2 tail tokens
        let buf = synthetic_bank(cells, false);
        let bank = parse_slot_save(&buf, &geo()).expect("walk");
        let spec = VllmFsSpec {
            model_name: "m".into(),
            tokens_per_hash: 64,
            blocks_per_file: 1,
            canonical_format: "v1-nhd".into(),
        };
        let out = repack_into_fs_tier(&bank, &geo(), &spec, dir.path()).expect("repack");
        assert_eq!(out.blocks_written, 2);
        assert_eq!(out.tail_tokens_dropped, 2);
        assert_eq!(out.files.len(), 2);
        // Page geometry: layers * heads * tokens * 2 halves * head bytes.
        let f = std::fs::read(&out.files[0]).expect("block file");
        assert_eq!(f.len(), 2 /* layers */ * 2 /* heads */ * 64 * 2 * 8);
        // First bytes must be layer 0, head 0, token 0 K half: the
        // synthetic bank encodes layer in bits 6-7, so a mis-layered
        // repack shows 0x40 bits here.
        assert_eq!(f[0] & 0xc0, 0x00);
        // Config written next to the rank dir and stable across reruns.
        let cfg =
            std::fs::read_to_string(spec.rank_dir(dir.path()).join("config.json")).expect("cfg");
        assert_eq!(cfg.trim(), spec.fields_json());
        // Files land at the hash-addressed path the engine derives.
        let h1 = hex::encode(vllm_block_hash(
            vllm_none_hash(),
            &(1000..1064).collect::<Vec<u32>>(),
        ));
        assert_eq!(
            out.files[0].file_name().unwrap().to_string_lossy(),
            format!("{h1}_g0.bin")
        );
        let _ = hexhash("unused");
    }

    #[test]
    fn unit__repack__refuses_when_another_configuration_owns_the_tier() {
        let dir = tempfile::tempdir().expect("tmp");
        let buf = synthetic_bank(64, false);
        let bank = parse_slot_save(&buf, &geo()).expect("walk");
        let spec = VllmFsSpec {
            model_name: "m".into(),
            tokens_per_hash: 64,
            blocks_per_file: 1,
            canonical_format: "v1-nhd".into(),
        };
        repack_into_fs_tier(&bank, &geo(), &spec, dir.path()).expect("first");
        // Same digest, hand-tampered identity file: the defense catches
        // a tier dir whose config.json disagrees with the spec.
        let cfg_path = spec.rank_dir(dir.path()).join("config.json");
        std::fs::write(&cfg_path, "{\"tampered\":true}").expect("tamper");
        let err = repack_into_fs_tier(&bank, &geo(), &spec, dir.path()).unwrap_err();
        assert!(err.contains("different engine configuration"), "{err}");
        // Sub-64-token banks never produce a block.
        let tiny = synthetic_bank(10, false);
        let bank2 = parse_slot_save(&tiny, &geo()).expect("walk");
        let fresh = tempfile::tempdir().expect("tmp2");
        assert!(
            repack_into_fs_tier(&bank2, &geo(), &spec, fresh.path())
                .unwrap_err()
                .contains("full 64-token block")
        );
    }

    #[test]
    fn unit__parse_slot_save__serialized_server_tokens_header_yields_real_ids() {
        let g = geo();
        let legacy = synthetic_bank(8, false);
        let expected = parse_slot_save(&legacy, &g).expect("legacy walk");
        let serialized = with_serialized_tokens(&legacy, 1, 0);
        let bank = parse_slot_save(&serialized, &g).expect("serialized walk");
        assert_eq!(bank.tokens, expected.tokens);
        assert_eq!(bank.cells, expected.cells);
        assert_eq!(bank.k_rows, expected.k_rows);
        // The envelope ids are the hashable stream: no marker, no count.
        assert!(bank.tokens.iter().all(|&t| t >= 0));
    }

    #[test]
    fn unit__parse_slot_save__serialized_header_refuses_media_and_version_drift() {
        let g = geo();
        let media = with_serialized_tokens(&synthetic_bank(8, false), 1, 2);
        let err = parse_slot_save(&media, &g).unwrap_err();
        assert!(err.contains("multimodal media chunks"), "{err}");
        let drifted = with_serialized_tokens(&synthetic_bank(8, false), 7, 0);
        let err = parse_slot_save(&drifted, &g).unwrap_err();
        assert!(err.contains("serialization version 7"), "{err}");
    }
}
