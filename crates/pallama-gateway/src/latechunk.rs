//! Late chunking (R1): embed the joined document ONCE with per-token
//! embeddings (`--pooling none` child), then mean-pool per-chunk token
//! spans at the gateway. Beats chunk-then-embed on retrieval because every
//! chunk vector sees the full-document context (jina late chunking,
//! arXiv 2409.04701). Opt-in per model:
//! `[model_overrides.<name>] late_chunking = true`.
//!
//! Wire protocol (verified against the vendored llama.cpp server):
//! - `POST /tokenize {content, add_special: false, with_pieces: true}` ->
//!   `{tokens: [{id, piece: "str" | [bytes]}]}` — exact token stream, no
//!   BOS; pieces reconstruct the input byte-for-byte for walkable offsets.
//! - `POST /embedding {content: [token ids]}` (legacy route; the OAI
//!   `/v1/embeddings` route rejects `--pooling none` as not OAI-compatible
//!   on stable b10809) — an id array is used
//!   raw (`tokenize_input_prompts` numbers branch), so matrix row i is
//!   embedding of token i, no special tokens added, alignment exact.

use std::sync::Arc;

use serde_json::{json, Value};

use pallama_runtime::EngineRef;

use crate::proxy::{child_auth, child_base};
use crate::state::AppState;

/// Everything the lanes need to shape their responses.
pub struct LateChunkOutput {
    pub embeddings: Vec<Vec<f64>>,
    pub total_tokens: u64,
}

/// One token's byte span in the joined document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenSpan {
    pub start: usize,
    pub end: usize,
}

/// Walk the tokenizer pieces over the document to recover each token's
/// byte span. Pieces must reconstruct the input exactly; any mismatch is
/// an explicit failure (never a guessed offset).
pub fn piece_walk(doc: &[u8], pieces: &[Vec<u8>]) -> Result<Vec<TokenSpan>, String> {
    let mut spans = Vec::with_capacity(pieces.len());
    let mut cursor = 0usize;
    for piece in pieces {
        let end = cursor + piece.len();
        if end > doc.len() || &doc[cursor..end] != piece.as_slice() {
            return Err(format!(
                "tokenizer pieces do not reconstruct the input at byte offset {cursor}; \
                 cannot late-chunk this document with this tokenizer"
            ));
        }
        spans.push(TokenSpan { start: cursor, end });
        cursor = end;
    }
    if cursor != doc.len() {
        return Err(format!(
            "tokenizer pieces cover {cursor} of {} input bytes; \
             cannot late-chunk this document with this tokenizer",
            doc.len()
        ));
    }
    Ok(spans)
}

/// Map tokens to chunks: a token belongs to the chunk that contains its
/// START byte (jina semantics — tokens merged across a chunk boundary
/// carry both chunks' context into the earlier chunk). `chunk_bytes` is
/// each chunk's byte length. Tokens arrive in document order, so a
/// monotonic cursor classifies them; a chunk whose every byte was merged
/// into a token starting earlier stays empty (`0..0`).
#[must_use]
pub fn chunk_token_spans(
    spans: &[TokenSpan],
    chunk_bytes: &[usize],
) -> Vec<std::ops::Range<usize>> {
    let mut upper = Vec::with_capacity(chunk_bytes.len()); // exclusive end byte per chunk
    let mut acc = 0usize;
    for &b in chunk_bytes {
        acc += b;
        upper.push(acc);
    }
    let mut out = vec![0usize..0; chunk_bytes.len()];
    let mut opened = vec![false; chunk_bytes.len()];
    let mut ci = 0usize;
    for (ti, ts) in spans.iter().enumerate() {
        while ci + 1 < upper.len() && ts.start >= upper[ci] {
            ci += 1;
        }
        if !opened[ci] {
            out[ci].start = ti;
            opened[ci] = true;
        }
        out[ci].end = ti + 1;
    }
    out
}

/// Mean-pool each chunk's token vectors, then L2-normalize (matching
/// upstream mean-pooling: normalize after pooling). A chunk that received
/// zero tokens (its text fully merged into a token starting in the
/// previous chunk) falls back to the vector of the token covering its
/// start byte — dimension stays correct and the context is the one the
/// chunk physically lives in.
pub fn pool_spans(
    matrix: &[Vec<f64>],
    spans: &[TokenSpan],
    chunk_ranges: &[std::ops::Range<usize>],
    chunk_bytes: &[usize],
) -> Vec<Vec<f64>> {
    let dims = matrix.first().map_or(0, Vec::len);
    let mut out = Vec::with_capacity(chunk_ranges.len());
    for (ci, range) in chunk_ranges.iter().enumerate() {
        if range.end > range.start {
            let n = f64::from(u32::try_from(range.end - range.start).unwrap_or(u32::MAX));
            let mut acc = vec![0.0f64; dims];
            for row in &matrix[range.clone()] {
                for (a, v) in acc.iter_mut().zip(row.iter()) {
                    *a += v;
                }
            }
            for a in &mut acc {
                *a /= n;
            }
            normalize_in_place(&mut acc);
            out.push(acc);
        } else {
            // empty span: borrow the token covering this chunk's start byte
            let start_byte = chunk_bytes[..ci].iter().sum::<usize>();
            let donor = spans
                .iter()
                .position(|t| t.start <= start_byte && start_byte < t.end)
                .or_else(|| spans.iter().position(|t| t.start >= start_byte));
            out.push(donor.map_or_else(
                || vec![0.0; dims],
                |ti| {
                    let mut v = matrix[ti].clone();
                    normalize_in_place(&mut v);
                    v
                },
            ));
        }
    }
    out
}

fn normalize_in_place(v: &mut [f64]) {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Pre-tokenized variant boundaries (all-int-array input on the `OpenAI`
/// lane): ids arrive per chunk, so boundaries are exact token counts and
/// no piece walk is needed. Spans are synthesized to satisfy `pool_spans`.
#[must_use]
pub fn synthetic_spans(n_tokens: usize) -> Vec<TokenSpan> {
    (0..n_tokens)
        .map(|i| TokenSpan {
            start: i,
            end: i + 1,
        })
        .collect()
}

/// Full late-chunking pipeline for string inputs.
///
/// Errors are `(http_status, message)` — explicit, never silent: 400 bad
/// input, 413 over token cap / context budget, 422 tokenizer cannot be
/// offset-walked, 502 child misbehaving (e.g. not actually pooling=none).
pub async fn late_embed(
    state: &Arc<AppState>,
    engine: &EngineRef,
    model: &str,
    inputs: &[String],
) -> Result<LateChunkOutput, (u16, String)> {
    if inputs.iter().any(String::is_empty) {
        return Err((
            400,
            "late chunking: \"input\" chunks must be non-empty strings".into(),
        ));
    }
    let full = inputs.concat();
    let chunk_byte_lens: Vec<usize> = inputs.iter().map(String::len).collect();
    let ids_pieces = tokenize_with_pieces(state, engine, &full).await?;
    let ids: Vec<u64> = ids_pieces.iter().map(|(id, _)| *id).collect();
    let pieces: Vec<Vec<u8>> = ids_pieces.into_iter().map(|(_, p)| p).collect();
    embed_and_pool(
        state,
        engine,
        model,
        full.as_bytes(),
        &ids,
        &pieces,
        &chunk_byte_lens,
    )
    .await
}

/// Shared cap + context-budget guard. Errors are 413 with the knob or
/// override that fixes them named in the message.
fn guard_token_budget(
    state: &Arc<AppState>,
    model: &str,
    n_tokens: usize,
) -> Result<(), (u16, String)> {
    let cap = state.config.late_chunking_max_tokens;
    if n_tokens >= cap {
        return Err((
            413,
            format!(
                "late chunking: document is {n_tokens} tokens, over the {cap}-token cap \
                 (config: late_chunking_max_tokens)"
            ),
        ));
    }
    let ctx = u64::from(state.config.effective_ctx(model));
    let n = u64::try_from(n_tokens).unwrap_or(u64::MAX);
    if n >= ctx.saturating_mul(9) / 10 {
        return Err((
            413,
            format!(
                "late chunking: document is {n} tokens, within 10% of the model's \
                 {ctx}-token context; raise [model_overrides.{model}] ctx or shorten the document"
            ),
        ));
    }
    // One doc must fit a single micro-batch (upstream embedding tasks
    // cannot split): the same number the profile rule assigns the child.
    let ubatch = if state.config.ubatch_size > 0 {
        u64::from(state.config.ubatch_size)
    } else {
        u64::from(pallama_core::profile::LATE_CHUNK_UBATCH_DEFAULT).min(ctx)
    };
    if n > ubatch {
        return Err((
            413,
            format!(
                "late chunking: document is {n} tokens, over the engine's \
                 {ubatch}-token micro-batch limit; raise config ubatch_size \
                 (and VRAM permitting) or shorten the document"
            ),
        ));
    }
    Ok(())
}

/// Shared tail: guards, matrix fetch, span math, pooling.
pub async fn embed_and_pool(
    state: &Arc<AppState>,
    engine: &EngineRef,
    model: &str,
    doc: &[u8],
    ids: &[u64],
    pieces: &[Vec<u8>],
    chunk_byte_lens: &[usize],
) -> Result<LateChunkOutput, (u16, String)> {
    guard_token_budget(state, model, ids.len())?;
    let spans = piece_walk(doc, pieces).map_err(|m| (422u16, m))?;
    let ranges = chunk_token_spans(&spans, chunk_byte_lens);
    let matrix = fetch_matrix(state, engine, ids).await?;
    let embeddings = pool_spans(&matrix, &spans, &ranges, chunk_byte_lens);
    Ok(LateChunkOutput {
        embeddings,
        total_tokens: ids.len() as u64,
    })
}

/// Pre-tokenized pipeline (`OpenAI` array-of-int-arrays input): ids already
/// chunked; boundaries exact; no tokenize/piece-walk round trip.
pub async fn late_embed_pretokenized(
    state: &Arc<AppState>,
    engine: &EngineRef,
    model: &str,
    chunk_ids: &[Vec<u64>],
) -> Result<LateChunkOutput, (u16, String)> {
    if chunk_ids.iter().any(Vec::is_empty) {
        return Err((
            400,
            "late chunking: pre-tokenized \"input\" chunks must be non-empty id arrays".into(),
        ));
    }
    let ids: Vec<u64> = chunk_ids.iter().flatten().copied().collect();
    guard_token_budget(state, model, ids.len())?;
    let spans = synthetic_spans(ids.len());
    let ranges: Vec<std::ops::Range<usize>> = {
        let mut out = Vec::with_capacity(chunk_ids.len());
        let mut acc = 0usize;
        for c in chunk_ids {
            out.push(acc..acc + c.len());
            acc += c.len();
        }
        out
    };
    // byte lens unused for synthetic spans (no empty-span donors possible:
    // every chunk has >= 1 token), pass token lens as placeholders
    let chunk_bytes: Vec<usize> = chunk_ids.iter().map(Vec::len).collect();
    let matrix = fetch_matrix(state, engine, &ids).await?;
    let embeddings = pool_spans(&matrix, &spans, &ranges, &chunk_bytes);
    Ok(LateChunkOutput {
        embeddings,
        total_tokens: ids.len() as u64,
    })
}

async fn tokenize_with_pieces(
    state: &Arc<AppState>,
    engine: &EngineRef,
    content: &str,
) -> Result<Vec<(u64, Vec<u8>)>, (u16, String)> {
    let url = format!("{}/tokenize", child_base(&engine.endpoint));
    let body = json!({"content": content, "add_special": false, "with_pieces": true});
    let resp = child_auth(state.http.post(&url), engine)
        .json(&body)
        .send()
        .await
        .map_err(|e| (502u16, format!("tokenize request failed: {e:#}")))?;
    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        return Err((status, format!("tokenize failed: {text}")));
    }
    let v: Value = resp
        .json()
        .await
        .map_err(|e| (502u16, format!("bad tokenize response: {e}")))?;
    let tokens = v["tokens"].as_array().ok_or_else(|| {
        (
            502u16,
            "tokenize response missing \"tokens\" array".to_string(),
        )
    })?;
    let mut out = Vec::with_capacity(tokens.len());
    for t in tokens {
        let id = t["id"]
            .as_u64()
            .ok_or_else(|| (502u16, "tokenize response token missing \"id\"".to_string()))?;
        let piece = match &t["piece"] {
            Value::String(s) => s.as_bytes().to_vec(),
            Value::Array(a) => a
                .iter()
                .filter_map(|b| b.as_u64().and_then(|b| u8::try_from(b).ok()))
                .collect(),
            _ => return Err((502, "tokenize response token missing \"piece\"".to_string())),
        };
        out.push((id, piece));
    }
    Ok(out)
}

async fn fetch_matrix(
    state: &Arc<AppState>,
    engine: &EngineRef,
    ids: &[u64],
) -> Result<Vec<Vec<f64>>, (u16, String)> {
    let url = format!("{}/embedding", child_base(&engine.endpoint));
    let body = json!({"content": ids});
    let resp = child_auth(state.http.post(&url), engine)
        .json(&body)
        .send()
        .await
        .map_err(|e| (502u16, format!("embedding request failed: {e:#}")))?;
    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        return Err((status, text));
    }
    let v: Value = resp
        .json()
        .await
        .map_err(|e| (502u16, format!("bad embedding response: {e}")))?;
    let matrix = v
        .as_array()
        .and_then(|a| a.first())
        .and_then(|e| e["embedding"].as_array())
        .ok_or_else(|| {
            (
                502u16,
                "embedding response missing per-token matrix".to_string(),
            )
        })?;
    let rows: Vec<Vec<f64>> = matrix
        .iter()
        .map(|row| {
            row.as_array()
                .map(|r| r.iter().filter_map(Value::as_f64).collect())
                .unwrap_or_default()
        })
        .collect();
    if rows.len() != ids.len() {
        return Err((
            502,
            format!(
                "engine returned {} embedding rows for {} tokens — child is not running \
                 --pooling none; evict and reload the model (pallama stop/start or keep_alive \
                 cycle) so the late_chunking profile applies",
                rows.len(),
                ids.len()
            ),
        ));
    }
    Ok(rows)
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    fn spans_of(pairs: &[(usize, usize)]) -> Vec<TokenSpan> {
        pairs
            .iter()
            .map(|&(s, e)| TokenSpan { start: s, end: e })
            .collect()
    }

    #[test]
    fn unit__piece_walk__exact_reconstruction() {
        let doc = b"hello world";
        let pieces: Vec<Vec<u8>> = vec![b"hello".to_vec(), b" ".to_vec(), b"world".to_vec()];
        let spans = piece_walk(doc, &pieces).expect("exact walk");
        assert_eq!(spans, spans_of(&[(0, 5), (5, 6), (6, 11)]));
    }

    #[test]
    fn unit__piece_walk__mismatch_is_error_with_offset() {
        let doc = b"hello world";
        let pieces: Vec<Vec<u8>> = vec![b"hello".to_vec(), b"WORLD".to_vec()];
        let err = piece_walk(doc, &pieces).expect_err("must fail");
        assert!(err.contains("byte offset 5"), "err: {err}");
    }

    #[test]
    fn unit__piece_walk__short_coverage_is_error() {
        let doc = b"hello world";
        let pieces: Vec<Vec<u8>> = vec![b"hello".to_vec()];
        let err = piece_walk(doc, &pieces).expect_err("must fail");
        assert!(err.contains("cover 5 of 11"), "err: {err}");
    }

    #[test]
    fn unit__chunk_token_spans__boundary_token_goes_to_chunk_of_start() {
        // doc "hello" split as ["hel","lo"] -> one token "hello" (start 0)
        // belongs to chunk 0; chunk 1 stays empty (0..0, donor path).
        let spans = spans_of(&[(0, 5)]);
        let ranges = chunk_token_spans(&spans, &[3, 2]);
        assert_eq!(ranges[0], 0..1);
        assert_eq!(ranges[1], 0..0);
    }

    #[test]
    fn unit__chunk_token_spans__clean_boundaries_partition() {
        // tokens: [0,3) [3,6) [6,9); chunks "abc","def" (3+3 bytes)
        let spans = spans_of(&[(0, 3), (3, 6), (6, 9)]);
        let ranges = chunk_token_spans(&spans, &[3, 3]);
        assert_eq!(ranges[0], 0..1);
        assert_eq!(ranges[1], 1..3);
    }

    #[test]
    fn unit__pool_spans__mean_then_l2_normalize() {
        // 2 tokens x 2 dims, chunk covers both: mean (2,4)/2 = (1,2),
        // norm sqrt(5) -> (1/sqrt(5), 2/sqrt(5))
        let matrix = vec![vec![2.0, 4.0], vec![0.0, 0.0]];
        let spans = spans_of(&[(0, 1), (1, 2)]);
        let out = pool_spans(&matrix, &spans, std::slice::from_ref(&(0..2)), &[1, 1]);
        let inv = 1.0 / 5.0f64.sqrt();
        assert!((out[0][0] - inv).abs() < 1e-12);
        assert!((out[0][1] - 2.0 * inv).abs() < 1e-12);
    }

    #[test]
    fn unit__pool_spans__zero_norm_stays_zero() {
        let matrix = vec![vec![0.0, 0.0]];
        let spans = spans_of(&[(0, 1)]);
        let out = pool_spans(&matrix, &spans, std::slice::from_ref(&(0..1)), &[1]);
        assert_eq!(out[0], vec![0.0, 0.0]);
    }

    #[test]
    fn unit__pool_spans__empty_chunk_borrows_covering_token() {
        // chunk 1 ("lo") empty: token 0 covers bytes 0..5; chunk 1 start
        // byte = 3 -> donor token 0 -> its normalized vector.
        let matrix = vec![vec![3.0, 4.0], vec![0.0, 0.0]];
        let spans = spans_of(&[(0, 5), (5, 6)]);
        let out = pool_spans(&matrix, &spans, &[0..1, 1..1], &[3, 2]);
        assert!((out[1][0] - 0.6).abs() < 1e-12);
        assert!((out[1][1] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn unit__synthetic_spans__identity_positions() {
        let s = synthetic_spans(3);
        assert_eq!(s, spans_of(&[(0, 1), (1, 2), (2, 3)]));
    }
}
