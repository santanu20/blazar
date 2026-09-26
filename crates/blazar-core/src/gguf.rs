//! Minimal GGUF metadata reader. Parses only the header + KV section
//! (tensor infos and data are seeked past, never read). Format truth:
//! ggml-org/llama.cpp `ggml/src/gguf.cpp` (v2 and v3 both use u64
//! lengths/counts; v3 added big-endian support only).
//!
//! Extracts exactly what Blazar needs: architecture identity + the fields
//! the profile compiler and `blazar fit` consume. Anything absent stays
//! `None` — downstream rules must skip-with-warning, never estimate.

use std::io::Read;
use std::path::Path;

use crate::error::{CoreError, CoreResult};

/// GGUF metadata Blazar consumes, resolved against the file's `{arch}.*`
/// prefix.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GgufMeta {
    pub architecture: String,
    pub name: Option<String>,
    /// `general.basename` — upstream repo id with the size suffix stripped
    /// (`Qwen/Qwen3.5` written as `Qwen_Qwen3.5`). Sole source for the
    /// mistral.rs `--tok-model-id` derivation on repackaged multimodal
    /// GGUFs.
    pub basename: Option<String>,
    /// `general.size_label` — size tag (`9B`) recombined with `basename`.
    pub size_label: Option<String>,
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
    /// `{arch}.attention.recurrent_layers` — per-layer boolean array marking
    /// linear-attention (gated-delta/SSM) layers that hold a constant-size
    /// recurrent state instead of a ctx-growing KV cache. Read by upstream
    /// `models/qwen35.cpp`, `qwen3next.cpp`, `qwen4exp.cpp`, `minimax-01.cpp`
    /// via `get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, n_layer_all)`;
    /// entries beyond `block_count` are MTP tail layers, not trunk KV.
    pub recurrent_layers: Option<Vec<bool>>,
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
    /// MTP (multi-token-prediction) head layers baked into the weights:
    /// `{arch}.n_predict_layers` (llama.cpp master) or
    /// `{arch}.nextn_predict_layers` (ollama converter) — both accepted,
    /// first non-zero wins. Drives the `spec = "auto"` draft-mtp lane;
    /// KV-gated upstream, so tensor-only legacy grafts without the key
    /// report None (the engine could not boot draft-mtp on them either).
    pub mtp_layers: Option<u64>,
    /// `{arch}.num_loops` — loop architectures (e.g. nanbeige) replay each
    /// block N times: the loader builds `n_layer` = `block_count` × `num_loops`
    /// KV-holding layers. Live-verified on nanbeige4.2: 22 blocks × 2
    /// loops = 44 layers → 5632 MiB f16 KV @ 32768 ctx, exactly 2× the
    /// block-only estimate. Clamped 1..=8 at parse; absent = 1. Only the
    /// dense per-block token math scales with it — per-layer arrays
    /// (recurrent/swa) count physical layers and stay unscaled.
    pub num_loops: Option<u32>,
}

/// Architectures whose every layer is recurrent (no ctx-growing KV at all).
/// Mirrored from vendored `llama-arch.cpp:1052 llm_arch_is_recurrent`.
const RECURRENT_ARCHS: &[&str] = &["mamba", "mamba2", "rwkv6", "rwkv6qwen2", "rwkv7", "arwkv7"];

/// Hybrid linear-attention architectures: a mix of recurrent (constant
/// state) and full-attention (ctx-growing KV) layers. Mirrored from vendored
/// `llama-arch.cpp:1066 llm_arch_is_hybrid`. Only a fraction of layers holds
/// KV; without per-layer metadata the fraction is not provable and callers
/// must stay conservative (count every layer).
const HYBRID_LINEAR_ARCHS: &[&str] = &[
    "jamba",
    "falcon-h1",
    "plamo2",
    "granitehybrid",
    "lfm2",
    "lfm2moe",
    "nemotron-h",
    "nemotron-h-moe",
    "qwen3next",
    "kimi-linear",
    "bailingmoe3",
    "kimi-k3",
    "qwen35",
    "qwen35moe",
    "qwen4exp",
    "deepseek4",
    "minimax-01",
];

/// Architectures proven to interpret `full_attention_interval` as
/// recurrent-layer spacing: every `(i + 1) % interval == 0` trunk layer is
/// full attention, the rest are recurrent (gated delta net) — NOT windowed
/// SWA. Sources: vendored `models/qwen35.cpp:17-24`, `qwen3next.cpp:17-24`,
/// `qwen4exp.cpp:128-136`, `minimax-01.cpp:12-19` (interval fallback loop).
/// Other hybrid archs reading interval differently are deliberately absent:
/// unlisted archs keep the conservative whole-block estimate.
const INTERVAL_RECURRENT_ARCHS: &[&str] =
    &["qwen35", "qwen35moe", "qwen3next", "qwen4exp", "minimax-01"];

/// Attention layout class resolved from `general.architecture`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionClass {
    /// Every layer holds ctx-growing KV (classic transformer).
    Full,
    /// Mix of recurrent (constant state) and full-attention layers; only
    /// the full-attention fraction grows KV.
    HybridLinear,
    /// Pure recurrent state; no ctx-growing KV.
    Recurrent,
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

    /// Attention layout class by architecture name (upstream lists, see the
    /// const docs). Unknown architectures classify as [`AttentionClass::Full`]
    /// — the conservative default that keeps the whole-block KV estimate.
    #[must_use]
    pub fn attention_class(&self) -> AttentionClass {
        if RECURRENT_ARCHS.contains(&self.architecture.as_str()) {
            AttentionClass::Recurrent
        } else if HYBRID_LINEAR_ARCHS.contains(&self.architecture.as_str()) {
            AttentionClass::HybridLinear
        } else {
            AttentionClass::Full
        }
    }

    /// Provable count of trunk layers that hold a ctx-growing KV cache.
    /// When the recurrent/full split is NOT provable from metadata this
    /// returns `Some(block_count)` — the conservative all-layers assumption
    /// callers fall back to; see [`Self::recurrent_split_provable`] to
    /// distinguish the two.
    ///
    /// Precedence: an explicit `recurrent_layers` array (trunk = first
    /// `block_count` entries; a longer array's tail is MTP layers, outside
    /// trunk KV), else the recurrent-interval formula for the archs proven
    /// to use it (`blocks / interval`, floor — upstream marks layer i full
    /// iff `(i + 1) % interval == 0`). A too-short array is ambiguous →
    /// `None`.
    #[must_use]
    pub fn full_attn_layers(&self) -> Option<u64> {
        let blocks = self.block_count?;
        if let Some(trunk) = self.provable_trunk() {
            return Some(trunk.iter().filter(|rec| !**rec).count() as u64);
        }
        if self.recurrent_layers.is_some() {
            // An array exists but does not cover the trunk: ambiguous.
            return None;
        }
        Some(blocks)
    }

    /// Whether the recurrent/full-attention layer split is provable from
    /// metadata (explicit array, or the arch-proven interval formula).
    /// Hybrid archs without a provable split keep conservative whole-block
    /// KV estimates — callers warn on exactly that case.
    #[must_use]
    pub fn recurrent_split_provable(&self) -> bool {
        self.provable_trunk().is_some()
    }

    /// Trunk-layer token budget for KV math: recurrent layers contribute 0,
    /// full-attention layers contribute `min(window, ctx)` when a per-layer
    /// window array proves it, else the whole `ctx` (conservative — a scalar
    /// window is never paired with array typing; gemma-2 lesson).
    fn recurrent_aware_token_sum(&self, ctx: u64, trunk: &[bool]) -> u64 {
        let windows = self
            .sliding_window_per_layer
            .as_ref()
            .filter(|w| w.len() == trunk.len());
        trunk
            .iter()
            .enumerate()
            .map(|(i, rec)| {
                if *rec {
                    0
                } else {
                    let window = windows.map_or(0, |w| w[i]);
                    if window == 0 {
                        ctx
                    } else {
                        window.min(ctx)
                    }
                }
            })
            .sum()
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
    ///
    /// Recurrent/hybrid-linear awareness (R8): recurrent layers hold a
    /// constant-size state, not ctx-growing KV. Pure-recurrent archs return
    /// `Some(0)` for the growing term (the constant state rides caller
    /// headroom); hybrid archs scale by the provable full-attention
    /// fraction via `full_attn_layers`. Unprovable splits stay
    /// conservative — every layer, whole context (never guess low).
    #[must_use]
    pub fn kv_f16_bytes(&self, ctx: u64) -> Option<u64> {
        if self.attention_class() == AttentionClass::Recurrent {
            // Proven by arch class: the recurrent cache is constant-size
            // (vendored llama-memory-recurrent); nothing grows with ctx.
            return Some(0);
        }
        // Loop archs multiply the KV-holding layer count: the dense
        // fallback (and only it — see the field doc) scales with loops.
        let loops = u64::from(self.num_loops.unwrap_or(1));
        let blocks = self.block_count?.saturating_mul(loops);
        let kv_heads = self.head_count_kv.or(self.head_count)?;
        let head_dim = self.derived_head_dim()?;
        let k_len = self.key_length.unwrap_or(head_dim);
        let v_len = self.value_length.unwrap_or(head_dim);
        let per_token = kv_heads * (k_len + v_len) * 2; // K + V, f16 = 2 B/elem
        let tokens = if let Some(trunk) = self.provable_trunk() {
            self.recurrent_aware_token_sum(ctx, &trunk)
        } else {
            self.swa_token_sum(ctx).unwrap_or(blocks * ctx)
        };
        Some(per_token.saturating_mul(tokens))
    }

    /// Trunk layer-typing array when provable: an explicit `recurrent_layers`
    /// array covering the trunk (first `block_count` entries; a longer
    /// array's tail is MTP layers outside trunk KV), else the interval
    /// formula for archs proven to use it. Empty when the split is not
    /// provable — callers keep the conservative whole-block estimate.
    #[must_use]
    fn provable_trunk(&self) -> Option<Vec<bool>> {
        let blocks = self.block_count?;
        let blocks_idx = usize::try_from(blocks).unwrap_or(usize::MAX);
        if let Some(arr) = &self.recurrent_layers {
            if arr.len() >= blocks_idx {
                return Some(arr[..blocks_idx].to_vec());
            }
            return None;
        }
        if INTERVAL_RECURRENT_ARCHS.contains(&self.architecture.as_str())
            && self.full_attention_interval.is_some_and(|i| i > 0)
        {
            // Upstream marks trunk layer i full iff (i+1) % interval == 0.
            let interval = self.full_attention_interval.unwrap_or(1);
            let trunk: Vec<bool> = (0..blocks).map(|i| (i + 1) % interval != 0).collect();
            return Some(trunk);
        }
        None
    }

    /// Structural metadata lint (H4): verifiable completeness warnings only.
    /// These explain WHY downstream sizing (blazar fit, cache-ram math,
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
        if self.attention_class() == AttentionClass::HybridLinear
            && !self.recurrent_split_provable()
        {
            out.push(
                "hybrid-linear architecture without a provable recurrent/full layer split — KV \
                 estimates count every layer (upper bound)",
            );
        }
        out
    }

    /// Hugging Face base-model id for the mistral.rs `--tok-model-id`
    /// fallback: repackaged multimodal GGUFs (lmstudio-community style)
    /// embed no adapter identity mistral.rs can read, so a forced
    /// mistral.rs lane aborts with "multimodal GGUF requires its original
    /// config.json". `general.basename` carries the upstream repo as
    /// `org_model` (slash flattened to `_`) and `general.size_label` the
    /// size tag; `Qwen_Qwen3.5` + `9B` recombine to `Qwen/Qwen3.5-9B`.
    /// Only derivable when the org separator is present — single-segment
    /// basenames (`unsloth` style) stay `None` and the engine's own
    /// teaching error surfaces unchanged.
    #[must_use]
    pub fn hf_base_model_id(&self) -> Option<String> {
        let base = self.basename.as_deref()?.trim();
        let sep = base.find('_')?;
        let (org, repo) = (&base[..sep], &base[sep + 1..]);
        if org.is_empty() || repo.is_empty() {
            return None;
        }
        let mut id = format!("{org}/{repo}");
        if let Some(size) = self.size_label.as_deref().map(str::trim) {
            if !size.is_empty() && !id.ends_with(size) {
                id.push('-');
                id.push_str(size);
            }
        }
        Some(id)
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

/// Real metadata nests arrays at most ~2 deep; 8 is a generous ceiling.
/// Without it a crafted file of nested-array headers recurses one stack
/// frame per ~12 bytes until the stack overflows (F125).
const MAX_ARRAY_DEPTH: u8 = 8;

fn read_value(cur: &mut Cursor<'_>, vtype: u32) -> CoreResult<GgufValue> {
    read_value_at(cur, vtype, 0)
}

fn read_value_at(cur: &mut Cursor<'_>, vtype: u32, depth: u8) -> CoreResult<GgufValue> {
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
            if depth >= MAX_ARRAY_DEPTH {
                return Err(bad(&format!(
                    "GGUF array nesting exceeds depth cap {MAX_ARRAY_DEPTH}"
                )));
            }
            let elem_type = cur.u32()?;
            let count = cur.u64()?;
            if count > MAX_ARRAY_ITEMS {
                return Err(bad(&format!(
                    "GGUF array count {count} exceeds cap {MAX_ARRAY_ITEMS}"
                )));
            }
            let mut items = Vec::new();
            for _ in 0..count {
                items.push(read_value_at(cur, elem_type, depth + 1)?);
            }
            GgufValue::Array(items)
        }
        10 => GgufValue::U64(cur.u64()?),
        11 => GgufValue::I64(cur.i64()?),
        12 => GgufValue::F64(cur.f64()?),
        other => return Err(bad(&format!("unknown GGUF value type {other}"))),
    })
}

/// `{arch}.attention.recurrent_layers` — per-layer boolean array marking
/// recurrent (linear-attention / SSM) layers; present on hybrid-linear
/// conversions (R8). Empty or malformed arrays are ignored (None).
fn recurrent_layer_array(kvs: &[(String, GgufValue)], arch: &str) -> Option<Vec<bool>> {
    kvs.iter()
        .find(|(k, _)| k == &format!("{arch}.attention.recurrent_layers"))
        .and_then(|(_, v)| match v {
            // F124: strict per-layer arrays — one wrong-typed item makes
            // the WHOLE array absent (conservative default) instead of
            // silently shifting every later layer index.
            GgufValue::Array(items) => items
                .iter()
                .map(|i| match i {
                    GgufValue::Bool(b) => Some(*b),
                    _ => None,
                })
                .collect::<Option<Vec<bool>>>(),
            _ => None,
        })
        .filter(|items| !items.is_empty())
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
            // F124: strict — see recurrent_layers note above.
            GgufValue::Array(items) => items
                .iter()
                .map(GgufValue::as_u64)
                .collect::<Option<Vec<u64>>>(),
            _ => None,
        })
        .filter(|items| !items.is_empty());
    let recurrent_layers = recurrent_layer_array(&kvs, &arch);

    let meta = GgufMeta {
        name: kvs
            .iter()
            .find(|(k, _)| k == "general.name")
            .and_then(|(_, v)| v.as_str())
            .map(str::to_string),
        basename: kvs
            .iter()
            .find(|(k, _)| k == "general.basename")
            .and_then(|(_, v)| v.as_str())
            .map(str::to_string),
        size_label: kvs
            .iter()
            .find(|(k, _)| k == "general.size_label")
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
        recurrent_layers,
        sliding_window: if sliding_window_per_layer.is_some() {
            None
        } else {
            att("sliding_window")
        },
        sliding_window_per_layer,
        full_attention_interval: get(format!("{arch}.full_attention_interval")),
        chat_template: extract_chat_template(&kvs),
        num_loops: get(format!("{arch}.num_loops"))
            .map(|n| u32::try_from(n.clamp(1, 8)).unwrap_or(1)),
        mtp_layers: get(format!("{arch}.n_predict_layers"))
            .filter(|&n| n > 0)
            .or_else(|| get(format!("{arch}.nextn_predict_layers")).filter(|&n| n > 0)),
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

/// Structural GGUF check: valid magic, a supported version, and at least
/// one tensor. Diffusion-lane `.gguf` exports (`DiT` weights paired with
/// `--vae`/`--llm` components) legitimately carry zero metadata KVs — the
/// architecture is supplied by the row's component set, not the file — so
/// they can never satisfy [`read_metadata_file`]'s `general.architecture`
/// requirement. This answers the narrower question: is this a GGUF
/// container at all?
#[must_use]
pub fn is_gguf_container(path: &Path) -> bool {
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    // 24 bytes = magic(4) + version(4) + tensor_count(8) + kv_count(8).
    let mut head = [0u8; 24];
    let mut read = 0;
    while read < head.len() {
        match file.read(&mut head[read..]) {
            Ok(0) | Err(_) => return false,
            Ok(n) => read += n,
        }
    }
    &head[..4] == GGUF_MAGIC
        && (2..=3).contains(&u32::from_le_bytes(
            head[4..8].try_into().expect("4-byte slice"),
        ))
        && u64::from_le_bytes(head[8..16].try_into().expect("8-byte slice")) >= 1
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

    #[test]
    fn unit__gguf_mtp_layers__master_key_parsed() {
        let buf = build_gguf(&[
            ("general.architecture", GgufValue::String("qwen35".into())),
            ("qwen35.n_predict_layers", GgufValue::U32(1)),
        ]);
        let (m, _) = parse_metadata(&buf).unwrap();
        assert_eq!(m.mtp_layers, Some(1));
    }

    #[test]
    fn unit__gguf_mtp_layers__ollama_converter_key_parsed() {
        let buf = build_gguf(&[
            ("general.architecture", GgufValue::String("qwen35".into())),
            ("qwen35.nextn_predict_layers", GgufValue::U32(1)),
        ]);
        let (m, _) = parse_metadata(&buf).unwrap();
        assert_eq!(m.mtp_layers, Some(1));
    }

    #[test]
    fn unit__gguf_mtp_layers__master_key_wins_and_zero_is_absent() {
        let both = build_gguf(&[
            ("general.architecture", GgufValue::String("qwen35".into())),
            ("qwen35.n_predict_layers", GgufValue::U32(1)),
            ("qwen35.nextn_predict_layers", GgufValue::U32(3)),
        ]);
        let (m, _) = parse_metadata(&both).unwrap();
        assert_eq!(m.mtp_layers, Some(1)); // master key takes precedence
        let zero = build_gguf(&[
            ("general.architecture", GgufValue::String("qwen35".into())),
            ("qwen35.n_predict_layers", GgufValue::U32(0)),
        ]);
        let (m, _) = parse_metadata(&zero).unwrap();
        assert_eq!(m.mtp_layers, None); // 0 = not an MTP model
        let plain = build_gguf(&[("general.architecture", GgufValue::String("qwen35".into()))]);
        let (m, _) = parse_metadata(&plain).unwrap();
        assert_eq!(m.mtp_layers, None);
    }

    #[test]
    fn unit__hf_base_model_id__recombines_lmstudio_convention() {
        let buf = build_gguf(&[
            ("general.architecture", GgufValue::String("qwen35".into())),
            ("general.basename", GgufValue::String("Qwen_Qwen3.5".into())),
            ("general.size_label", GgufValue::String("9B".into())),
        ]);
        let (m, _) = parse_metadata(&buf).unwrap();
        assert_eq!(m.basename.as_deref(), Some("Qwen_Qwen3.5"));
        assert_eq!(m.size_label.as_deref(), Some("9B"));
        assert_eq!(m.hf_base_model_id().as_deref(), Some("Qwen/Qwen3.5-9B"));
    }

    #[test]
    fn unit__hf_base_model_id__size_suffix_not_doubled() {
        let buf = build_gguf(&[
            ("general.architecture", GgufValue::String("qwen35".into())),
            (
                "general.basename",
                GgufValue::String("Qwen_Qwen3.5-9B".into()),
            ),
            ("general.size_label", GgufValue::String("9B".into())),
        ]);
        let (m, _) = parse_metadata(&buf).unwrap();
        assert_eq!(m.hf_base_model_id().as_deref(), Some("Qwen/Qwen3.5-9B"));
    }

    #[test]
    fn unit__hf_base_model_id__underivable_shapes_stay_none() {
        // Single-segment basenames (unsloth style) have no org separator;
        // missing basename or empty org/repo halves are equally
        // underivable — the engine's teaching error stays the surface.
        for base in ["Qwen3.5-9B", "_repo", "org_", ""] {
            let buf = build_gguf(&[
                ("general.architecture", GgufValue::String("qwen35".into())),
                ("general.basename", GgufValue::String(base.into())),
                ("general.size_label", GgufValue::String("9B".into())),
            ]);
            let (m, _) = parse_metadata(&buf).unwrap();
            assert!(m.hf_base_model_id().is_none(), "{base} must not derive");
        }
        let bare = build_gguf(&[("general.architecture", GgufValue::String("q".into()))]);
        let (m, _) = parse_metadata(&bare).unwrap();
        assert!(m.hf_base_model_id().is_none());
    }

    #[test]
    fn unit__hf_base_model_id__missing_size_label_keeps_org_repo() {
        let buf = build_gguf(&[
            ("general.architecture", GgufValue::String("qwen35".into())),
            ("general.basename", GgufValue::String("Qwen_Qwen3.5".into())),
        ]);
        let (m, _) = parse_metadata(&buf).unwrap();
        assert_eq!(m.hf_base_model_id().as_deref(), Some("Qwen/Qwen3.5"));
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
            GgufValue::Bool(x) => b.push(u8::from(*x)),
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

    #[test]
    fn unit__gguf_container__metadataless_dit_file_passes() {
        // DiT (diffusion) exports ship zero KVs: header then tensor infos.
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes()); // tensor count
        b.extend_from_slice(&0u64.to_le_bytes()); // kv count — by design
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("dit.gguf");
        std::fs::write(&p, &b).unwrap();
        assert!(is_gguf_container(&p));
        // The LLM-lane reader must still refuse it: that distinction is
        // exactly why the container check exists.
        assert!(read_metadata_file(&p).is_err());

        let mut junk = b.clone();
        junk[..4].copy_from_slice(b"JUNK");
        let jp = tmp.path().join("junk.gguf");
        std::fs::write(&jp, &junk).unwrap();
        assert!(!is_gguf_container(&jp));
    }

    // --- R8: hybrid-linear / recurrent KV awareness ---

    fn hybrid_kv_meta(arch: &str, blocks: u32) -> GgufMeta {
        GgufMeta {
            architecture: arch.into(),
            block_count: Some(u64::from(blocks)),
            head_count: Some(16),
            head_count_kv: Some(8),
            embedding_length: Some(1024),
            head_dim: Some(64),
            ..GgufMeta::default()
        }
    }

    #[test]
    fn unit__recurrent_layers__parsed_from_bool_array() {
        let mut kvs = qwen_like();
        kvs.push((
            "qwen3.attention.recurrent_layers",
            GgufValue::Array(vec![
                GgufValue::Bool(true),
                GgufValue::Bool(false),
                GgufValue::Bool(true),
            ]),
        ));
        let (meta, _) = parse_metadata(&build_gguf(&kvs)).unwrap();
        assert_eq!(meta.recurrent_layers, Some(vec![true, false, true]));
    }

    #[test]
    fn unit__attention_class__arch_lists_mirror_upstream() {
        // Spot-checks against vendored llama-arch.cpp:1052/:1066.
        assert_eq!(
            hybrid_kv_meta("kimi-k3", 69).attention_class(),
            AttentionClass::HybridLinear
        );
        assert_eq!(
            hybrid_kv_meta("qwen3next", 48).attention_class(),
            AttentionClass::HybridLinear
        );
        assert_eq!(
            hybrid_kv_meta("mamba", 64).attention_class(),
            AttentionClass::Recurrent
        );
        assert_eq!(
            hybrid_kv_meta("rwkv7", 32).attention_class(),
            AttentionClass::Recurrent
        );
        assert_eq!(
            hybrid_kv_meta("qwen3", 28).attention_class(),
            AttentionClass::Full
        );
        // Unknown archs default conservative (Full).
        assert_eq!(
            hybrid_kv_meta("future-arch", 8).attention_class(),
            AttentionClass::Full
        );
    }

    #[test]
    fn unit__kv_f16_bytes__recurrent_array_counts_only_full_layers() {
        let mut m = hybrid_kv_meta("qwen3", 4);
        m.recurrent_layers = Some(vec![true, false, true, false]);
        // per_token = 8 kv_heads * (64+64) * 2 = 4096 B; only layers 1+3
        // hold KV: 4096 * 2 * 1024 = 8 MiB.
        assert_eq!(m.kv_f16_bytes(1024), Some(2048 * 2 * 1024));
    }

    #[test]
    fn unit__kv_f16_bytes__recurrent_array_with_per_layer_windows() {
        let mut m = hybrid_kv_meta("qwen3", 2);
        m.recurrent_layers = Some(vec![false, true]);
        m.sliding_window_per_layer = Some(vec![512, 512]);
        // Layer 0 full-attn windowed: min(512, 4096) = 512; layer 1
        // recurrent: 0. 4096 B/token * 512 = 2 MiB.
        assert_eq!(m.kv_f16_bytes(4096), Some(2048 * 512));
    }

    #[test]
    fn unit__kv_f16_bytes__qwen35_interval_counts_quarter_layers() {
        // Real Qwen3.5-9B shape: interval 4 → 8 of 32 trunk layers hold KV
        // (upstream qwen35.cpp: non-interval layers are recurrent gated
        // delta net, NOT windowed). per_token = 4 * (256+256) * 2 = 4096 B.
        let (m, _) = parse_metadata(&build_gguf(&qwen35_mla())).unwrap();
        assert!(m.recurrent_split_provable());
        assert_eq!(m.full_attn_layers(), Some(8));
        // 4096 * 8 * 8192 = 256 MiB (was 1 GiB whole-block).
        assert_eq!(m.kv_f16_bytes(8192), Some(4096 * 8 * 8192));
    }

    #[test]
    fn unit__kv_f16_bytes__interval_floor_division_edge() {
        // 33 trunk layers, interval 4: full at i=3,7,...,31 → 8 = floor.
        let mut kvs = qwen35_mla();
        kvs.retain(|(k, _)| !k.ends_with("full_attention_interval"));
        kvs.push(("qwen35.block_count", GgufValue::U32(33)));
        kvs.push(("qwen35.full_attention_interval", GgufValue::U32(4)));
        let (m, _) = parse_metadata(&build_gguf(&kvs)).unwrap();
        assert_eq!(m.full_attn_layers(), Some(8));
        // Interval exceeding the trunk: zero full layers, zero growing KV.
        let mut m2 = hybrid_kv_meta("qwen35", 32);
        m2.full_attention_interval = Some(128);
        assert_eq!(m2.full_attn_layers(), Some(0));
        assert_eq!(m2.kv_f16_bytes(4096), Some(0));
    }

    #[test]
    fn unit__full_attn_layers__mtp_tail_uses_trunk_prefix() {
        // Upstream arrays cover n_layer_all (trunk + MTP tail); trunk KV
        // is the first block_count entries.
        let mut m = hybrid_kv_meta("qwen4exp", 32);
        m.recurrent_layers = Some(vec![true; 32].into_iter().chain([false, false]).collect());
        assert_eq!(m.full_attn_layers(), Some(0));
        m.recurrent_layers = Some(vec![false; 32].into_iter().chain([true, true]).collect());
        assert_eq!(m.full_attn_layers(), Some(32));
    }

    #[test]
    fn unit__full_attn_layers__short_array_is_ambiguous() {
        let mut m = hybrid_kv_meta("qwen4exp", 32);
        m.recurrent_layers = Some(vec![true; 8]);
        assert_eq!(m.full_attn_layers(), None);
        assert!(!m.recurrent_split_provable());
        // KV math stays whole-block (conservative upper bound).
        assert_eq!(m.kv_f16_bytes(1024), Some(2048 * 32 * 1024));
    }

    #[test]
    fn unit__kv_f16_bytes__mamba_pure_recurrent_is_zero() {
        // Constant recurrent state rides caller headroom; the growing
        // term is provably zero (llama-memory-recurrent).
        let m = hybrid_kv_meta("mamba", 64);
        assert_eq!(m.kv_f16_bytes(1_000_000), Some(0));
    }

    #[test]
    fn unit__kv_f16_bytes__non_interval_hybrid_stays_conservative() {
        // kimi-k3 without a layer map: no provable split → every layer
        // counts (never guess low), and lint explains the bound.
        let m = hybrid_kv_meta("kimi-k3", 69);
        assert!(!m.recurrent_split_provable());
        assert_eq!(m.kv_f16_bytes(1024), Some(2048 * 69 * 1024));
        assert!(m
            .lint()
            .iter()
            .any(|w| w.contains("hybrid-linear architecture")));
        // Provable split (qwen35 interval): no lint noise.
        let (q, _) = parse_metadata(&build_gguf(&qwen35_mla())).unwrap();
        assert!(!q.lint().iter().any(|w| w.contains("hybrid-linear")));
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod depth_tests {
    use super::*;

    #[test]
    fn unit__gguf_depth__nested_arrays_capped_not_stack_overflow() {
        // F125: one KV whose value is a chain of 64 nested single-element
        // array headers (~12 bytes per level) — must error at the depth
        // cap instead of recursing the stack into oblivion.
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes()); // tensors
        b.extend_from_slice(&1u64.to_le_bytes()); // one KV pair
                                                  // key "deep"
        b.extend_from_slice(&4u64.to_le_bytes());
        b.extend_from_slice(b"deep");
        // Value: type 9 (array) ONCE — nested levels are just
        // [elem_type][count] pairs with no repeated type tag.
        b.extend_from_slice(&9u32.to_le_bytes());
        for _ in 0..64 {
            b.extend_from_slice(&9u32.to_le_bytes()); // elem type: array
            b.extend_from_slice(&1u64.to_le_bytes()); // count: 1
        }
        b.extend_from_slice(&0u32.to_le_bytes()); // innermost elem: u8
        b.push(7);
        let res = parse_metadata(&b);
        let Err(err) = res else {
            panic!("depth cap must fire");
        };
        assert!(err.to_string().contains("depth"), "{err}");
    }
}

/// Env-gated live check against a real MTP-bearing GGUF
/// (`BLAZAR_TEST_MTP_GGUF=/path/to/model.gguf cargo test mtp_real`).
/// Skips silently when the env var is unset (CI has no such file).
#[test]
#[allow(non_snake_case)] // integration__ prefix matches the suite convention
fn integration__gguf_mtp_layers__real_file_when_provided() {
    let Ok(path) = std::env::var("BLAZAR_TEST_MTP_GGUF") else {
        eprintln!("skipping: BLAZAR_TEST_MTP_GGUF not set");
        return;
    };
    let meta = read_metadata_file(std::path::Path::new(&path))
        .unwrap_or_else(|e| panic!("parse {path}: {e}"));
    assert!(
        meta.mtp_layers.is_some(),
        "real MTP GGUF reported no mtp_layers: arch {}",
        meta.architecture
    );
    eprintln!(
        "arch={} mtp_layers={:?} block_count={:?}",
        meta.architecture, meta.mtp_layers, meta.block_count
    );
}
