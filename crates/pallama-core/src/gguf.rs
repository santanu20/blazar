//! Minimal GGUF metadata reader. Parses only the header + KV section
//! (tensor infos and data are seeked past, never read). Format truth:
//! references/llama.cpp-master/ggml/src/gguf.cpp (v2 and v3 both use u64
//! lengths/counts; v3 added big-endian support only).
//!
//! Extracts exactly what Pallama needs: architecture identity + the fields
//! the profile compiler and `pallama fit` consume. Anything absent stays
//! `None` — downstream rules must skip-with-warning, never estimate.

use std::io::Read;
use std::path::Path;

use crate::error::{CoreError, CoreResult};

/// GGUF metadata Pallama consumes, resolved against the file's `{arch}.*`
/// prefix.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GgufMeta {
    pub architecture: String,
    pub name: Option<String>,
    pub block_count: Option<u64>,
    pub context_length: Option<u64>,
    pub expert_count: Option<u64>,
    pub head_count: Option<u64>,
    pub head_count_kv: Option<u64>,
    pub embedding_length: Option<u64>,
    /// Explicit `{arch}.head_dim` when present; else derived by
    /// `derived_head_dim()`.
    pub head_dim: Option<u64>,
    /// Per-head K width (`{arch}.attention.key_length`). MLA-class models
    /// (`DeepSeek`, `Qwen3.5`) carry a latent width that differs from
    /// `head_dim`; the loader allocates from THIS, so KV math uses it as
    /// the K width (upper bound — absorbed/FA paths allocate less).
    pub key_length: Option<u64>,
    /// Per-head V width (`{arch}.attention.value_length`); see `key_length`.
    pub value_length: Option<u64>,
    /// Scalar `{arch}.attention.sliding_window` — uniform SWA window in
    /// tokens. Alone it proves nothing about WHICH layers are windowed
    /// (gemma-2 alternates windowed/full layers with no metadata trace),
    /// so it only shrinks the KV estimate when paired with
    /// `full_attention_interval` or a per-layer array.
    pub sliding_window: Option<u64>,
    /// Per-layer `{arch}.attention.sliding_window` array (0 = that layer is
    /// full attention). Fully explicit, so it alone is provable.
    pub sliding_window_per_layer: Option<Vec<u64>>,
    /// `{arch}.full_attention_interval` — every Nth layer attends over the
    /// full context, the rest over the sliding window (Qwen3.5-9B ships 4).
    pub full_attention_interval: Option<u64>,
    /// `general.quantized_by` — quantizer identity (e.g. "Unsloth");
    /// consumed by the known-bad-quantizer lint.
    pub quantized_by: Option<String>,
    /// `general.version` — upstream version string.
    pub general_version: Option<String>,
    /// `{arch}.pooling_type` — present only on embedding-class models
    /// (bert/nomic-bert/...). Drives the `--embeddings` profile rule so
    /// `/v1/embeddings` works without manual flags.
    pub pooling_type: Option<u64>,
    /// Embedded chat-template text (`tokenizer.chat_template` and/or the
    /// raw-jinja `chat_template.jinja` key; array variants concatenated).
    /// Consumed only by capability heuristics (sentinel tool precheck).
    pub chat_template: Option<String>,
}

impl GgufMeta {
    /// `head_dim`: metadata value, else `embedding_length` / `head_count` when
    /// both present and divisible.
    #[must_use]
    pub fn derived_head_dim(&self) -> Option<u64> {
        if let Some(dim) = self.head_dim {
            return Some(dim);
        }
        match (self.embedding_length, self.head_count) {
            (Some(len), Some(head)) if head > 0 && len % head == 0 => Some(len / head),
            _ => None,
        }
    }

    /// Sum over layers of each layer's KV token capacity for a serving
    /// context of `ctx` tokens. `None` = geometry ambiguous — the caller
    /// must fall back to whole-context math for EVERY layer. Never guesses
    /// LOW: an underestimated cache OOMs at runtime, an overestimated one
    /// only costs headroom.
    ///
    /// Provable shapes only: a per-layer array covering every block, or a
    /// scalar window paired with `full_attention_interval`. A scalar window
    /// alone is NOT enough — gemma-2 alternates windowed/full layers with
    /// no metadata trace — and an interval without a window (Qwen3.5 GGUFs
    /// omit the size) leaves the windowed width unknown.
    #[must_use]
    pub fn swa_token_sum(&self, ctx: u64) -> Option<u64> {
        let blocks = self.block_count?;
        let layer_tokens = |window: u64| {
            if window == 0 {
                ctx
            } else {
                window.min(ctx)
            }
        };
        if let Some(per_layer) = &self.sliding_window_per_layer {
            if per_layer.len() as u64 == blocks {
                return Some(per_layer.iter().map(|w| layer_tokens(*w)).sum());
            }
            // Array shorter/longer than the layer count: ambiguous.
            return None;
        }
        match (self.sliding_window, self.full_attention_interval) {
            (Some(window), Some(interval)) if interval > 0 => {
                let full_layers = blocks.div_ceil(interval);
                Some(full_layers * ctx + (blocks - full_layers) * layer_tokens(window))
            }
            _ => None,
        }
    }

    /// f16 KV-cache bytes at `ctx`: per-head widths from explicit
    /// `key_length`/`value_length` (MLA latents — the loader allocates from
    /// these; upper bound, since absorbed/FA paths allocate less), else the
    /// classic `head_dim`. Layer token capacity from `swa_token_sum` when
    /// the SWA shape is provable, else whole-context for every layer.
    /// `None` when the GGUF lacks the geometry — callers skip-with-warning,
    /// never estimate.
    #[must_use]
    pub fn kv_f16_bytes(&self, ctx: u64) -> Option<u64> {
        let blocks = self.block_count?;
        let kv_heads = self.head_count_kv.or(self.head_count)?;
        let head_dim = self.derived_head_dim()?;
        let k_len = self.key_length.unwrap_or(head_dim);
        let v_len = self.value_length.unwrap_or(head_dim);
        let per_token = kv_heads * (k_len + v_len) * 2; // K + V, f16 = 2 B/elem
        let tokens = self.swa_token_sum(ctx).unwrap_or(blocks * ctx);
        Some(per_token.saturating_mul(tokens))
    }

    /// Structural metadata lint (H4): verifiable completeness warnings only.
    /// These explain WHY downstream sizing (pallama fit, cache-ram math,
    /// coresidency) degrades — they never guess at producer quality, which
    /// cannot be verified from the file alone. Warn-level by contract;
    /// callers must never block on a finding.
    #[must_use]
    pub fn lint(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.context_length.is_none() {
            out.push("context_length missing — context sizing falls back to engine defaults");
        }
        if self.kv_f16_bytes(4096).is_none() {
            out.push(
                "attention geometry incomplete — KV/VRAM estimates are disabled (fit, cache-ram, coresidency run blind)",
            );
        }
        if let (Some(blocks), Some(windows)) = (self.block_count, &self.sliding_window_per_layer) {
            if windows.len() != usize::try_from(blocks).unwrap_or(usize::MAX) {
                out.push("sliding_window array length != block_count — SWA KV savings ignored");
            }
        }
        out
    }
}

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
/// Caps against crafted files: absurd sizes are rejected, not allocated.
const MAX_STRING_BYTES: u64 = 1 << 20;
const MAX_ARRAY_ITEMS: u64 = 1 << 20;
const MAX_KV_PAIRS: u64 = 100_000;

/// Value types 0..=12 (`gguf_type` enum). Arrays are parsed recursively.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    /// Integer view. Arrays yield the max of their integer members — some
    /// models ship `context_length` as an array of candidates.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U8(v) => Some(u64::from(*v)),
            Self::U16(v) => Some(u64::from(*v)),
            Self::U32(v) => Some(u64::from(*v)),
            Self::U64(v) => Some(*v),
            Self::I8(v) => u64::try_from(i64::from(*v)).ok(),
            Self::I16(v) => u64::try_from(i64::from(*v)).ok(),
            Self::I32(v) => u64::try_from(i64::from(*v)).ok(),
            Self::I64(v) => u64::try_from(*v).ok(),
            Self::Bool(v) => Some(u64::from(*v)),
            Self::Array(items) => items.iter().filter_map(Self::as_u64).max(),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> CoreResult<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| bad("offset overflow"))?;
        if end > self.buf.len() {
            return Err(bad("unexpected end of GGUF metadata section"));
        }
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u8(&mut self) -> CoreResult<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> CoreResult<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes(b.try_into().expect("2 bytes")))
    }

    fn i16(&mut self) -> CoreResult<i16> {
        let b = self.take(2)?;
        Ok(i16::from_le_bytes(b.try_into().expect("2 bytes")))
    }

    fn u32(&mut self) -> CoreResult<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes(b.try_into().expect("4 bytes")))
    }

    fn i32(&mut self) -> CoreResult<i32> {
        let b = self.take(4)?;
        Ok(i32::from_le_bytes(b.try_into().expect("4 bytes")))
    }

    fn u64(&mut self) -> CoreResult<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes(b.try_into().expect("8 bytes")))
    }

    fn i64(&mut self) -> CoreResult<i64> {
        let b = self.take(8)?;
        Ok(i64::from_le_bytes(b.try_into().expect("8 bytes")))
    }

    fn f32(&mut self) -> CoreResult<f32> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes(b.try_into().expect("4 bytes")))
    }

    fn f64(&mut self) -> CoreResult<f64> {
        let b = self.take(8)?;
        Ok(f64::from_le_bytes(b.try_into().expect("8 bytes")))
    }

    fn string(&mut self) -> CoreResult<String> {
        let len = self.u64()?;
        if len > MAX_STRING_BYTES {
            return Err(bad(&format!(
                "GGUF string length {len} exceeds cap {MAX_STRING_BYTES}"
            )));
        }
        let n = usize::try_from(len).map_err(|_| bad("string length exceeds addressable size"))?;
        let bytes = self.take(n)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| CoreError::Config("gguf: string is not valid UTF-8".into()))
    }
}

fn bad(msg: &str) -> CoreError {
    CoreError::Config(format!("gguf: {msg}"))
}

fn read_value(cur: &mut Cursor<'_>, vtype: u32) -> CoreResult<GgufValue> {
    Ok(match vtype {
        0 => GgufValue::U8(cur.u8()?),
        1 => GgufValue::I8(i8::from_le_bytes([cur.u8()?])),
        2 => GgufValue::U16(cur.u16()?),
        3 => GgufValue::I16(cur.i16()?),
        4 => GgufValue::U32(cur.u32()?),
        5 => GgufValue::I32(cur.i32()?),
        6 => GgufValue::F32(cur.f32()?),
        7 => GgufValue::Bool(cur.u8()? != 0),
        8 => GgufValue::String(cur.string()?),
        9 => {
            let elem_type = cur.u32()?;
            let count = cur.u64()?;
            if count > MAX_ARRAY_ITEMS {
                return Err(bad(&format!(
                    "GGUF array count {count} exceeds cap {MAX_ARRAY_ITEMS}"
                )));
            }
            let mut items = Vec::new();
            for _ in 0..count {
                items.push(read_value(cur, elem_type)?);
            }
            GgufValue::Array(items)
        }
        10 => GgufValue::U64(cur.u64()?),
        11 => GgufValue::I64(cur.i64()?),
        12 => GgufValue::F64(cur.f64()?),
        other => return Err(bad(&format!("unknown GGUF value type {other}"))),
    })
}

/// Parse GGUF metadata from an in-memory buffer. Only the KV section is
/// read. Returns the meta and the byte offset where tensor infos begin.
pub fn parse_metadata(buf: &[u8]) -> CoreResult<(GgufMeta, usize)> {
    let mut cur = Cursor::new(buf);
    let magic = cur.take(4)?;
    if magic != GGUF_MAGIC {
        return Err(bad(&format!(
            "bad magic {:?}, expected GGUF",
            String::from_utf8_lossy(magic)
        )));
    }
    let version = cur.u32()?;
    if !(2..=3).contains(&version) {
        return Err(bad(&format!(
            "unsupported GGUF version {version} (supported: 2, 3)"
        )));
    }
    let _tensor_count = cur.u64()?;
    let kv_count = cur.u64()?;
    if kv_count > MAX_KV_PAIRS {
        return Err(bad(&format!("GGUF kv count {kv_count} exceeds cap")));
    }

    let mut kvs: Vec<(String, GgufValue)> = Vec::new();
    for _ in 0..kv_count {
        let key = cur.string()?;
        let vtype = cur.u32()?;
        let value = read_value(&mut cur, vtype)?;
        kvs.push((key, value));
    }
    let meta_end = cur.pos;

    let arch = kvs
        .iter()
        .find(|(k, _)| k == "general.architecture")
        .and_then(|(_, v)| v.as_str())
        .ok_or_else(|| bad("missing general.architecture"))?
        .to_string();

    let get = |field: String| -> Option<u64> {
        kvs.iter()
            .find(|(k, _)| *k == field)
            .and_then(|(_, v)| v.as_u64())
    };
    // Real GGUFs ship attention geometry under `{arch}.attention.*`
    // (verified against Qwen2.5 + Qwen3.5 files); the bare `{arch}.*`
    // spellings stay as legacy fallback so old fixtures keep parsing.
    let att = |field: &str| -> Option<u64> {
        get(format!("{arch}.attention.{field}")).or_else(|| get(format!("{arch}.{field}")))
    };
    let find = |field: String| -> Option<&GgufValue> {
        kvs.iter().find(|(k, _)| *k == field).map(|(_, v)| v)
    };
    let sliding_window_per_layer = find(format!("{arch}.attention.sliding_window"))
        .and_then(|v| match v {
            GgufValue::Array(items) => Some(
                items
                    .iter()
                    .filter_map(GgufValue::as_u64)
                    .collect::<Vec<u64>>(),
            ),
            _ => None,
        })
        .filter(|items| !items.is_empty());

    let meta = GgufMeta {
        name: kvs
            .iter()
            .find(|(k, _)| k == "general.name")
            .and_then(|(_, v)| v.as_str())
            .map(str::to_string),
        quantized_by: kvs
            .iter()
            .find(|(k, _)| k == "general.quantized_by")
            .and_then(|(_, v)| v.as_str())
            .map(str::to_string),
        general_version: kvs
            .iter()
            .find(|(k, _)| k == "general.version")
            .and_then(|(_, v)| v.as_str())
            .map(str::to_string),
        block_count: get(format!("{arch}.block_count")),
        context_length: get(format!("{arch}.context_length")),
        expert_count: get(format!("{arch}.expert_count")),
        head_count: att("head_count"),
        head_dim: att("head_dim"),
        pooling_type: get(format!("{arch}.pooling_type"))
            .or_else(|| get("*.pooling_type".to_string())),
        head_count_kv: att("head_count_kv"),
        embedding_length: get(format!("{arch}.embedding_length")),
        key_length: att("key_length"),
        value_length: att("value_length"),
        sliding_window: if sliding_window_per_layer.is_some() {
            None
        } else {
            att("sliding_window")
        },
        sliding_window_per_layer,
        full_attention_interval: get(format!("{arch}.full_attention_interval")),
        chat_template: extract_chat_template(&kvs),
        architecture: arch,
    };
    Ok((meta, meta_end))
}

/// Concatenate template text from every known template KV. String and
/// array-of-string shapes both occur in the wild (multi-template files);
/// only marker-substring heuristics consume the result, so concatenation
/// is safe.
fn extract_chat_template(kvs: &[(String, GgufValue)]) -> Option<String> {
    let mut out = String::new();
    for (key, value) in kvs {
        if key != "tokenizer.chat_template" && key != "chat_template.jinja" {
            continue;
        }
        match value {
            GgufValue::String(s) => out.push_str(s),
            GgufValue::Array(items) => {
                for item in items {
                    if let GgufValue::String(s) = item {
                        out.push_str(s);
                    }
                }
            }
            _ => {}
        }
    }
    (!out.is_empty()).then_some(out)
}

/// Read metadata from a GGUF file on disk. Only a bounded prefix is read
/// (grown on demand); tensor data is never touched.
pub fn read_metadata_file(path: &Path) -> CoreResult<GgufMeta> {
    // Bounded read with growth: 4 MiB covers real KV sections (chat
    // templates included); grow x4 up to 64 MiB for pathological files.
    let mut cap = 4 * 1024 * 1024usize;
    loop {
        let mut file = std::fs::File::open(path)
            .map_err(|e| CoreError::Config(format!("open {}: {e}", path.display())))?;
        let mut buf = vec![0u8; cap];
        let mut read = 0;
        while read < buf.len() {
            let got = file.read(&mut buf[read..])?;
            if got == 0 {
                break;
            }
            read += got;
        }
        buf.truncate(read);
        match parse_metadata(&buf) {
            Ok((meta, _)) => return Ok(meta),
            Err(e) if is_truncated(&e) && cap < 64 * 1024 * 1024 => cap *= 4,
            Err(e) if is_truncated(&e) => return Err(bad("GGUF metadata section exceeds 64 MiB")),
            Err(e) => return Err(e),
        }
    }
}

fn is_truncated(e: &CoreError) -> bool {
    e.to_string().contains("unexpected end of GGUF metadata")
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    /// GGUF v3 fixture builder: header + arbitrary KV pairs.
    fn build_gguf(kvs: &[(&str, GgufValue)]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        b.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (k, v) in kvs {
            put_str(&mut b, k);
            put_value(&mut b, v);
        }
        b
    }

    fn put_str(b: &mut Vec<u8>, s: &str) {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    }

    fn put_value(b: &mut Vec<u8>, v: &GgufValue) {
        match v {
            GgufValue::U8(x) => {
                b.extend_from_slice(&0u32.to_le_bytes());
                b.push(*x);
            }
            GgufValue::U16(x) => {
                b.extend_from_slice(&2u32.to_le_bytes());
                b.extend_from_slice(&x.to_le_bytes());
            }
            GgufValue::U32(x) => {
                b.extend_from_slice(&4u32.to_le_bytes());
                b.extend_from_slice(&x.to_le_bytes());
            }
            GgufValue::I32(x) => {
                b.extend_from_slice(&5u32.to_le_bytes());
                b.extend_from_slice(&x.to_le_bytes());
            }
            GgufValue::F32(x) => {
                b.extend_from_slice(&6u32.to_le_bytes());
                b.extend_from_slice(&x.to_le_bytes());
            }
            GgufValue::Bool(x) => {
                b.extend_from_slice(&7u32.to_le_bytes());
                b.push(u8::from(*x));
            }
            GgufValue::String(s) => {
                b.extend_from_slice(&8u32.to_le_bytes());
                put_str(b, s);
            }
            GgufValue::Array(items) => {
                b.extend_from_slice(&9u32.to_le_bytes());
                let elem_type = type_id(&items[0]);
                b.extend_from_slice(&elem_type.to_le_bytes());
                b.extend_from_slice(&(items.len() as u64).to_le_bytes());
                for it in items {
                    put_raw(b, it);
                }
            }
            GgufValue::U64(x) => {
                b.extend_from_slice(&10u32.to_le_bytes());
                b.extend_from_slice(&x.to_le_bytes());
            }
            other => panic!("fixture builder: unsupported {other:?}"),
        }
    }

    /// Array elements carry NO type tag — `elem_type` is stated once for the
    /// whole array (this is what tripped the first fixture version).
    fn put_raw(b: &mut Vec<u8>, v: &GgufValue) {
        match v {
            GgufValue::String(s) => put_str(b, s),
            GgufValue::U8(x) => b.push(*x),
            GgufValue::U16(x) => b.extend_from_slice(&x.to_le_bytes()),
            GgufValue::U32(x) => b.extend_from_slice(&x.to_le_bytes()),
            other => panic!("raw fixture writer: unsupported {other:?}"),
        }
    }

    fn type_id(v: &GgufValue) -> u32 {
        match v {
            GgufValue::U8(_) => 0,
            GgufValue::U16(_) => 2,
            GgufValue::U32(_) => 4,
            GgufValue::I32(_) => 5,
            GgufValue::F32(_) => 6,
            GgufValue::Bool(_) => 7,
            GgufValue::String(_) => 8,
            GgufValue::Array(_) => 9,
            GgufValue::U64(_) => 10,
            _ => panic!("no type id"),
        }
    }

    /// Real-shape fixture: attention geometry under `{arch}.attention.*`,
    /// mirroring actual Qwen2.5/Qwen3.5 GGUF files on disk.
    fn qwen_like() -> Vec<(&'static str, GgufValue)> {
        vec![
            ("general.architecture", GgufValue::String("qwen3".into())),
            ("general.name", GgufValue::String("Qwen3 0.6B".into())),
            ("qwen3.block_count", GgufValue::U32(28)),
            ("qwen3.context_length", GgufValue::U32(40960)),
            ("qwen3.expert_count", GgufValue::U32(128)),
            ("qwen3.attention.head_count", GgufValue::U32(16)),
            ("qwen3.attention.head_count_kv", GgufValue::U32(8)),
            ("qwen3.embedding_length", GgufValue::U32(1024)),
        ]
    }

    /// Qwen3.5-9B geometry (read from the real file): MLA latents + a
    /// full-attention interval with NO sliding-window size in metadata.
    fn qwen35_mla() -> Vec<(&'static str, GgufValue)> {
        vec![
            ("general.architecture", GgufValue::String("qwen35".into())),
            ("general.name", GgufValue::String("Qwen3.5-9B".into())),
            ("general.quantized_by", GgufValue::String("Unsloth".into())),
            ("qwen35.block_count", GgufValue::U32(32)),
            ("qwen35.context_length", GgufValue::U32(262_144)),
            ("qwen35.embedding_length", GgufValue::U32(4096)),
            ("qwen35.attention.head_count", GgufValue::U32(16)),
            ("qwen35.attention.head_count_kv", GgufValue::U32(4)),
            ("qwen35.attention.key_length", GgufValue::U32(256)),
            ("qwen35.attention.value_length", GgufValue::U32(256)),
            ("qwen35.full_attention_interval", GgufValue::U32(4)),
        ]
    }

    #[test]
    fn unit__gguf_v3_full_meta__parsed() {
        let buf = build_gguf(&qwen_like());
        let (meta, end) = parse_metadata(&buf).unwrap();
        assert_eq!(meta.architecture, "qwen3");
        assert_eq!(meta.name.as_deref(), Some("Qwen3 0.6B"));
        assert_eq!(meta.block_count, Some(28));
        assert_eq!(meta.context_length, Some(40960));
        assert_eq!(meta.expert_count, Some(128));
        assert_eq!(meta.head_count, Some(16));
        assert_eq!(meta.head_count_kv, Some(8));
        assert_eq!(meta.embedding_length, Some(1024));
        assert_eq!(end, buf.len());
    }

    #[test]
    fn unit__gguf_head_dim__derived_from_embedding_and_heads() {
        let (meta, _) = parse_metadata(&build_gguf(&qwen_like())).unwrap();
        assert_eq!(meta.head_dim, None);
        assert_eq!(meta.derived_head_dim(), Some(64)); // 1024 / 16
    }

    #[test]
    fn unit__gguf_legacy_bare_attention_keys__still_parsed() {
        // Old-fixture spelling: bare `{arch}.head_count[_kv]` keeps parsing.
        let kvs = vec![
            ("general.architecture", GgufValue::String("llama".into())),
            ("llama.block_count", GgufValue::U32(32)),
            ("llama.head_count", GgufValue::U32(32)),
            ("llama.head_count_kv", GgufValue::U32(8)),
        ];
        let (meta, _) = parse_metadata(&build_gguf(&kvs)).unwrap();
        assert_eq!(meta.head_count, Some(32));
        assert_eq!(meta.head_count_kv, Some(8));
    }

    #[test]
    fn unit__gguf_mla_geometry__parsed_real_qwen35_shape() {
        let (meta, _) = parse_metadata(&build_gguf(&qwen35_mla())).unwrap();
        assert_eq!(meta.head_count, Some(16));
        assert_eq!(meta.head_count_kv, Some(4));
        assert_eq!(meta.key_length, Some(256));
        assert_eq!(meta.value_length, Some(256));
        assert_eq!(meta.full_attention_interval, Some(4));
        assert_eq!(meta.quantized_by.as_deref(), Some("Unsloth"));
        assert_eq!(meta.sliding_window, None);
        assert_eq!(meta.sliding_window_per_layer, None);
        // Interval present but window size absent (real Qwen3.5 files):
        // ambiguous → caller falls back to whole-context math.
        assert_eq!(meta.swa_token_sum(4096), None);
    }

    #[test]
    fn unit__gguf_swa_token_sum__scalar_window_needs_interval() {
        let mut meta = GgufMeta {
            block_count: Some(32),
            sliding_window: Some(512),
            ..GgufMeta::default()
        };
        // Scalar window alone proves nothing (gemma-2 alternates without
        // metadata): whole-context fallback.
        assert_eq!(meta.swa_token_sum(4096), None);
        meta.full_attention_interval = Some(4);
        // 8 full layers × 4096 + 24 windowed × 512.
        assert_eq!(meta.swa_token_sum(4096), Some(8 * 4096 + 24 * 512));
        // Window larger than ctx clamps to ctx.
        meta.sliding_window = Some(8192);
        assert_eq!(meta.swa_token_sum(4096), Some(32 * 4096));
    }

    #[test]
    fn unit__gguf_swa_token_sum__per_layer_array_elementwise() {
        let meta = GgufMeta {
            block_count: Some(4),
            sliding_window_per_layer: Some(vec![0, 512, 0, 512]),
            ..GgufMeta::default()
        };
        // 0 = full-attention layer; 512 windowed; elementwise sum.
        assert_eq!(meta.swa_token_sum(4096), Some(2 * 4096 + 2 * 512));
    }

    #[test]
    fn unit__gguf_swa_token_sum__array_length_mismatch_is_ambiguous() {
        let meta = GgufMeta {
            block_count: Some(8),
            sliding_window_per_layer: Some(vec![512; 4]),
            ..GgufMeta::default()
        };
        assert_eq!(meta.swa_token_sum(4096), None);
    }

    #[test]
    fn unit__gguf_swa_token_sum__per_layer_array_parsed_from_file() {
        let kvs = vec![
            ("general.architecture", GgufValue::String("gemma2".into())),
            ("gemma2.block_count", GgufValue::U32(3)),
            (
                "gemma2.attention.sliding_window",
                GgufValue::Array(vec![
                    GgufValue::U32(512),
                    GgufValue::U32(0),
                    GgufValue::U32(512),
                ]),
            ),
        ];
        let (meta, _) = parse_metadata(&build_gguf(&kvs)).unwrap();
        assert_eq!(
            meta.sliding_window, None,
            "array shape must not fake scalar"
        );
        assert_eq!(meta.sliding_window_per_layer, Some(vec![512, 0, 512]));
        assert_eq!(meta.swa_token_sum(4096), Some(2 * 512 + 4096));
    }

    #[test]
    fn unit__kv_f16_bytes__classic_shape_matches_legacy_formula() {
        // No MLA widths, no SWA: 2 * blocks * kv_heads * head_dim * ctx * 2.
        let meta = GgufMeta {
            block_count: Some(32),
            head_count: Some(32),
            head_count_kv: Some(8),
            embedding_length: Some(4096),
            ..GgufMeta::default()
        };
        assert_eq!(
            meta.kv_f16_bytes(32_768),
            Some(2 * 32 * 8 * 128 * 32_768 * 2)
        );
    }

    #[test]
    fn unit__kv_f16_bytes__mla_latent_widths_override_head_dim() {
        // DeepSeek2-flavored: kv_heads=1, explicit K/V latents wider and
        // narrower than the derived head_dim — both are loader truth.
        let meta = GgufMeta {
            block_count: Some(2),
            head_count: Some(16),
            head_count_kv: Some(1),
            embedding_length: Some(4096), // derived head_dim = 256
            key_length: Some(576),
            value_length: Some(512),
            ..GgufMeta::default()
        };
        // per-token = 1 * (576 + 512) * 2 = 2176; tokens = 2 * 1024.
        assert_eq!(meta.kv_f16_bytes(1024), Some(2176 * 2048));
    }

    #[test]
    fn unit__kv_f16_bytes__hybrid_swa_shrinks_cache_provably() {
        let meta = GgufMeta {
            block_count: Some(32),
            head_count: Some(16),
            head_count_kv: Some(4),
            embedding_length: Some(4096), // derived head_dim = 256
            sliding_window: Some(512),
            full_attention_interval: Some(4),
            ..GgufMeta::default()
        };
        // tokens = 8*4096 + 24*512 = 45056; per-token = 4*(256+256)*2 = 4096.
        assert_eq!(meta.kv_f16_bytes(4096), Some(4096 * 45_056));
        // Sanity vs whole-ctx: strictly smaller.
        assert!(meta.kv_f16_bytes(4096).unwrap() < 4096 * 32 * 4096);
    }

    #[test]
    fn unit__kv_f16_bytes__missing_geometry_is_none() {
        assert_eq!(GgufMeta::default().kv_f16_bytes(4096), None);
        let no_dim = GgufMeta {
            block_count: Some(32),
            head_count_kv: Some(8),
            ..GgufMeta::default()
        };
        assert_eq!(no_dim.kv_f16_bytes(4096), None);
    }

    #[test]
    fn unit__lint__complete_geometry_is_silent() {
        let meta = GgufMeta {
            block_count: Some(32),
            context_length: Some(32_768),
            head_count: Some(32),
            head_count_kv: Some(8),
            embedding_length: Some(4096),
            ..GgufMeta::default()
        };
        assert!(meta.lint().is_empty());
    }

    #[test]
    fn unit__lint__missing_ctx_and_geometry_each_warn_once() {
        let empty = GgufMeta::default();
        let findings = empty.lint();
        assert_eq!(findings.len(), 2);
        assert!(findings[0].contains("context_length missing"));
        assert!(findings[1].contains("KV/VRAM estimates are disabled"));
    }

    #[test]
    fn unit__lint__swa_array_length_mismatch_warns() {
        let meta = GgufMeta {
            block_count: Some(4),
            context_length: Some(4096),
            head_count: Some(16),
            head_count_kv: Some(4),
            embedding_length: Some(4096),
            // 3 windows for 4 blocks: malformed — provable warning.
            sliding_window_per_layer: Some(vec![512, 0, 512]),
            ..GgufMeta::default()
        };
        let findings = meta.lint();
        assert_eq!(findings.len(), 1);
        assert!(findings[0].contains("SWA KV savings ignored"));
    }

    #[test]
    fn unit__gguf_array_context_length__max_taken() {
        let kvs = vec![
            ("general.architecture", GgufValue::String("llama".into())),
            (
                "llama.context_length",
                GgufValue::Array(vec![GgufValue::U32(8192), GgufValue::U32(131_072)]),
            ),
        ];
        let (meta, _) = parse_metadata(&build_gguf(&kvs)).unwrap();
        assert_eq!(meta.context_length, Some(131_072));
    }

    #[test]
    fn unit__gguf_v2_header__accepted() {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&2u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        put_str(&mut b, "general.architecture");
        put_value(&mut b, &GgufValue::String("gemma".into()));
        let (meta, _) = parse_metadata(&b).unwrap();
        assert_eq!(meta.architecture, "gemma");
        assert_eq!(meta.context_length, None);
    }

    #[test]
    fn unit__gguf_u16_value__roundtrip() {
        let kvs = vec![
            ("general.architecture", GgufValue::String("x".into())),
            ("x.head_count", GgufValue::U16(40)),
        ];
        let (meta, _) = parse_metadata(&build_gguf(&kvs)).unwrap();
        assert_eq!(meta.head_count, Some(40));
    }

    #[test]
    fn unit__gguf_bad_magic__named_error() {
        let err = parse_metadata(b"JUNK----").unwrap_err();
        assert!(err.to_string().contains("magic"));
    }

    #[test]
    fn unit__gguf_bad_version__named_error() {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&1u32.to_le_bytes());
        let err = parse_metadata(&b).unwrap_err();
        assert!(err.to_string().contains("version 1"));
    }

    #[test]
    fn unit__gguf_truncated__bounded_error() {
        let buf = build_gguf(&qwen_like());
        let err = parse_metadata(&buf[..buf.len() - 4]).unwrap_err();
        assert!(err.to_string().contains("unexpected end"));
    }

    #[test]
    fn unit__gguf_missing_architecture__named_error() {
        let kvs = vec![("general.name", GgufValue::String("x".into()))];
        let err = parse_metadata(&build_gguf(&kvs)).unwrap_err();
        assert!(err.to_string().contains("general.architecture"));
    }

    #[test]
    fn unit__gguf_string_array_kv__parsed() {
        let tokens: Vec<GgufValue> = ["a", "b", "c"]
            .iter()
            .map(|s| GgufValue::String((*s).to_string()))
            .collect();
        let kvs = vec![
            ("general.architecture", GgufValue::String("llama".into())),
            ("tokenizer.ggml.tokens", GgufValue::Array(tokens)),
        ];
        let (meta, _) = parse_metadata(&build_gguf(&kvs)).unwrap();
        assert_eq!(meta.architecture, "llama");
    }

    #[test]
    fn unit__gguf_file_read__real_path() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("m.gguf");
        std::fs::write(&p, build_gguf(&qwen_like())).unwrap();
        let meta = read_metadata_file(&p).unwrap();
        assert_eq!(meta.architecture, "qwen3");
    }
}
