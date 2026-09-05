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
        let end = self.pos.checked_add(n).ok_or_else(|| bad("offset overflow"))?;
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

    let meta = GgufMeta {
        name: kvs
            .iter()
            .find(|(k, _)| k == "general.name")
            .and_then(|(_, v)| v.as_str())
            .map(str::to_string),
        block_count: get(format!("{arch}.block_count")),
        context_length: get(format!("{arch}.context_length")),
        expert_count: get(format!("{arch}.expert_count")),
        head_count: get(format!("{arch}.head_count")),
        head_count_kv: get(format!("{arch}.head_count_kv")),
        embedding_length: get(format!("{arch}.embedding_length")),
        head_dim: get(format!("{arch}.head_dim")),
        architecture: arch,
    };
    Ok((meta, meta_end))
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

    fn qwen_like() -> Vec<(&'static str, GgufValue)> {
        vec![
            ("general.architecture", GgufValue::String("qwen3".into())),
            ("general.name", GgufValue::String("Qwen3 0.6B".into())),
            ("qwen3.block_count", GgufValue::U32(28)),
            ("qwen3.context_length", GgufValue::U32(40960)),
            ("qwen3.expert_count", GgufValue::U32(128)),
            ("qwen3.head_count", GgufValue::U32(16)),
            ("qwen3.head_count_kv", GgufValue::U32(8)),
            ("qwen3.embedding_length", GgufValue::U32(1024)),
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
