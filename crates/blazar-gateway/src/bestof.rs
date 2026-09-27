//! Best-of-N request fan-out: one client ask, N engine candidates, one
//! judged winner. Opt-in per request (`best_of` body field on the
//! OpenAI/Ollama dialects, `X-Blazar-Best-Of` header on every dialect);
//! the fan-out happens on the TRANSLATED child body, so the winning
//! response flows back through each dialect's unchanged response path.
//!
//! Contract:
//! - non-streaming only (`stream: true` + best-of is rejected 400 — a
//!   winner cannot be picked before completion, and streaming N
//!   candidates to the client is a different feature);
//! - the winner is chosen by a deterministic ladder: schema-valid (when
//!   the request carries a schema) > `finish_reason: stop` > candidate
//!   index (arrival order);
//! - usage is the SUM across candidates (every candidate generated and
//!   is billed; the response extension itemizes the split);
//! - the fan-out never queues behind live traffic: candidates are
//!   capped by the model's free decode slots, and any queue depth at
//!   admission collapses the fan-out to 1 with a response header
//!   saying so.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use axum::http::HeaderMap;
use serde_json::Value;

use crate::state::AppState;

pub const HEADER: &str = "x-blazar-best-of";
pub(crate) const MAX_BEST_OF: u64 = 4;

/// Process-wide counters surfaced as gauges in `/metrics` (every public
/// surface emits; these are plain totals, not histograms).
pub(crate) static FANOUT_ACTIVE: AtomicU64 = AtomicU64::new(0);
pub(crate) static FANOUT_DEGRADED: AtomicU64 = AtomicU64::new(0);

fn parse_u64(v: &str) -> Option<u64> {
    v.parse::<u64>().ok()
}

/// Merge the header and body spellings of the knob into one validated
/// value. `Ok(None)` = fan-out not requested (the default path).
/// Header wins on conflict (it is the explicit, dialect-universal
/// spelling); out-of-range values are errors, not clamps — a client
/// asking for 9 candidates should learn the ceiling, not silently
/// get 4.
pub fn resolve_best_of(
    headers: &HeaderMap,
    body_field: Option<&Value>,
) -> Result<Option<u64>, String> {
    let header_raw = headers.get(HEADER).and_then(|v| v.to_str().ok());
    let header_val = header_raw.and_then(parse_u64);
    if let (Some(raw), None) = (header_raw, header_val) {
        return Err(format!(
            "{HEADER} must be an integer in 2..={MAX_BEST_OF}, got {raw:?}"
        ));
    }
    let body_val = body_field
        .filter(|v| v.is_u64() || v.is_string())
        .and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(parse_u64)));
    if let Some(v) = body_field.filter(|v| !v.is_null()) {
        if v.as_u64().is_none() && v.as_str().and_then(parse_u64).is_none() {
            return Err(format!("best_of must be an integer in 2..={MAX_BEST_OF}"));
        }
    }
    if let (Some(h), Some(b)) = (header_val, body_val) {
        if h != b {
            return Err(format!(
                "{HEADER} header ({h}) and best_of body field ({b}) disagree — send one"
            ));
        }
    }
    Ok(header_val
        .or(body_val)
        .filter(|n| *n >= 2)
        .map(|n| n.min(MAX_BEST_OF)))
}

/// Fan-out size after the saturation guard: any queued work collapses
/// to 1 (candidates must never push real traffic back), otherwise the
/// model's free decode slots bound the count.
pub(crate) fn effective_n(state: &Arc<AppState>, model: &str, want: u64) -> u64 {
    if state.queue.depth() > 0 {
        FANOUT_DEGRADED.fetch_add(1, Ordering::Relaxed);
        return 1;
    }
    let headroom = u64::from(state.sup.slot_headroom(model));
    if headroom < want {
        FANOUT_DEGRADED.fetch_add(1, Ordering::Relaxed);
        return headroom.max(1);
    }
    want
}

/// The schema a candidate must satisfy, read off the TRANSLATED child
/// body: `response_format.json_schema.schema`, else the forced tool's
/// `parameters` when the request pins a specific function.
pub(crate) fn request_schema(child_body: &Value) -> Option<Value> {
    if let Some(schema) = child_body
        .pointer("/response_format/json_schema/schema")
        .filter(|s| s.is_object())
    {
        return Some(schema.clone());
    }
    let forced = child_body
        .pointer("/tool_choice/function/name")
        .and_then(Value::as_str)?;
    child_body
        .get("tools")
        .and_then(Value::as_array)?
        .iter()
        .find(|t| {
            t.pointer("/function/name")
                .and_then(Value::as_str)
                .is_some_and(|n| n == forced)
        })
        .and_then(|t| t.pointer("/function/parameters").cloned())
        .filter(Value::is_object)
}

/// Deterministic judge: schema-valid beats invalid, `finish_reason:
/// stop` beats other endings, lower candidate index breaks ties.
/// Non-2xx candidates rank last (they carry no content to judge).
pub(crate) fn pick_winner(
    sentinel: &crate::sentinel::Sentinel,
    candidates: &[(u16, Value)],
    schema: Option<&Value>,
) -> usize {
    let score = |(status, body): &(u16, Value)| -> (u8, u8) {
        if !(200..300).contains(status) {
            return (0, 0);
        }
        // No schema in the request: the rung is neutral (true for all).
        let schema_ok = match schema {
            None => true,
            Some(s) => {
                let content = body
                    .pointer("/choices/0/message/content")
                    .and_then(|c| c.as_str())
                    .and_then(|c| serde_json::from_str::<Value>(c).ok());
                match content {
                    Some(v) => sentinel.schema_check(s, &v).is_none(),
                    None => false,
                }
            }
        };
        let stop = body
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
            .is_some_and(|r| r == "stop");
        (u8::from(schema_ok), u8::from(stop))
    };
    candidates
        .iter()
        .enumerate()
        .max_by_key(|(i, c)| (score(c), std::cmp::Reverse(*i)))
        .map_or(0, |(i, _)| i)
}

/// Bodies above this size skip the fan-out (degrade to the normal
/// path): N copies of a huge prompt multiply child memory pressure for
/// little judging benefit (long prompts are prefix-cached anyway).
pub(crate) const MAX_FANOUT_BODY_BYTES: usize = 1024 * 1024;

/// One judged fan-out outcome: the winning candidate reconstructed as a
/// normal child response (winner body, usage summed across candidates,
/// real status preserved) so every dialect's unchanged response path
/// can consume it, plus what the caller needs for observability.
pub(crate) struct FanOutOutcome {
    pub resp: reqwest::Response,
    pub elapsed: std::time::Duration,
    pub hdr: String,
}

/// Run the fan-out and judge it. `None` = take the normal single-send
/// path instead: knob absent, saturation- or size-degraded, or the
/// first candidate failed on transport (the caller's existing
/// respawn-retry contract then owns crash recovery — re-running the
/// request once through that path beats failing a live lane because one
/// of N speculative copies died).
pub(crate) async fn fan_out(
    state: &Arc<AppState>,
    engine: &blazar_runtime::EngineRef,
    url: &str,
    child_body: &[u8],
    want: u64,
) -> Option<FanOutOutcome> {
    let n = effective_n(state, &engine.name, want);
    if n < 2 {
        return None;
    }
    if child_body.len() > MAX_FANOUT_BODY_BYTES {
        tracing::warn!(
            target: "blazar::bestof",
            model = %engine.name,
            "body {} bytes exceeds the fan-out cap — single-send path",
            child_body.len()
        );
        FANOUT_DEGRADED.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let parsed: Value = serde_json::from_slice(child_body).ok()?;
    let schema = request_schema(&parsed);
    FANOUT_ACTIVE.fetch_add(1, Ordering::Relaxed);
    let t0 = std::time::Instant::now();
    let sends = (0..n).map(|_| {
        let rb = crate::proxy::child_auth(
            crate::state::child_client(state, &engine.endpoint)
                .post(url)
                .header("content-type", "application/json")
                .body(child_body.to_vec()),
            engine,
        );
        crate::proxy::child_send(state, engine, rb.send())
    });
    let results = futures::future::join_all(sends).await;
    if results.first().is_some_and(std::result::Result::is_err) {
        tracing::warn!(
            target: "blazar::bestof",
            model = %engine.name,
            "candidate 0 failed transport ({}) — falling back to the single-send path",
            results[0].as_ref().unwrap_err()
        );
        return None;
    }
    // Buffer + parse the completed candidates; a candidate that fails
    // body-read is a lost candidate, not a failed request.
    let mut cands: Vec<(u16, Value, axum::body::Bytes)> = Vec::with_capacity(results.len());
    for r in results {
        let resp = r.ok()?;
        let status = resp.status().as_u16();
        let bytes = resp.bytes().await.ok()?;
        let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        cands.push((status, value, bytes));
    }
    if cands.is_empty() {
        return None;
    }
    let used = u64::try_from(cands.len()).unwrap_or(u64::MAX);
    let judged: Vec<(u16, Value)> = cands.iter().map(|c| (c.0, c.1.clone())).collect();
    let winner = pick_winner(&state.sentinel, &judged, schema.as_ref());
    let others: Vec<Value> = judged
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != winner)
        .map(|(_, c)| c.1.clone())
        .collect();
    let (status, winner_val, raw) = &mut cands[winner];
    let hdr = patch_winner(winner_val, &others, n, used);
    let status = *status;
    let raw = raw.clone();
    // A non-JSON winner body (degenerate case — the ladder ranks those
    // last) keeps its raw bytes so error text survives reconstruction.
    let body = if winner_val.is_null() {
        raw.clone()
    } else {
        serde_json::to_vec(winner_val).map_or_else(|_| raw.clone(), axum::body::Bytes::from)
    };
    let http_resp = axum::http::Response::builder()
        .status(
            axum::http::StatusCode::from_u16(status).unwrap_or(axum::http::StatusCode::BAD_GATEWAY),
        )
        .header("content-type", "application/json")
        .header(HEADER, &hdr)
        .body(body)
        .ok()?;
    Some(FanOutOutcome {
        resp: reqwest::Response::from(http_resp),
        elapsed: t0.elapsed(),
        hdr,
    })
}

/// Stamp the transparency header onto a client-facing response (the
/// child-facing copy already rides inside the reconstructed winner).
pub fn stamp(resp: &mut axum::response::Response, hdr: Option<&str>) {
    if let Some(h) = hdr {
        if let Ok(v) = axum::http::HeaderValue::from_str(h) {
            resp.headers_mut().insert(HEADER, v);
        }
    }
}

/// Sum usage across candidates onto the winner so every generated
/// token is billed exactly once, and stamp the transparency header.
pub(crate) fn patch_winner(winner: &mut Value, others: &[Value], asked: u64, used: u64) -> String {
    let mut prompt = winner
        .pointer("/usage/prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let mut completion = winner
        .pointer("/usage/completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    for o in others {
        prompt = prompt.saturating_add(
            o.pointer("/usage/prompt_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        );
        completion = completion.saturating_add(
            o.pointer("/usage/completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        );
    }
    if let Some(usage) = winner.get_mut("usage").and_then(|u| u.as_object_mut()) {
        usage.insert("prompt_tokens".into(), Value::from(prompt));
        usage.insert("completion_tokens".into(), Value::from(completion));
    }
    format!("asked={asked} used={used} usage=prompt:{prompt} completion:{completion}")
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // suite convention: unit__scenario__expected (§6b)

    use super::*;
    use serde_json::json;

    fn hdr(n: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(HEADER, n.parse().unwrap());
        h
    }

    #[test]
    fn unit__resolve_best_of__header_body_and_range_gates() {
        assert_eq!(resolve_best_of(&hdr("3"), None).unwrap(), Some(3));
        assert_eq!(
            resolve_best_of(&HeaderMap::new(), Some(&json!(2))).unwrap(),
            Some(2)
        );
        // below 2 = not a fan-out (1 copy is the normal path, not an error)
        assert_eq!(resolve_best_of(&hdr("1"), None).unwrap(), None);
        // above the cap clamps with the documented ceiling
        assert_eq!(resolve_best_of(&hdr("9"), None).unwrap(), Some(MAX_BEST_OF));
        // garbage fails fast
        assert!(resolve_best_of(&hdr("two"), None).is_err());
        // header/body disagreement fails fast (one knob, one source of truth)
        assert!(resolve_best_of(&hdr("2"), Some(&json!(3))).is_err());
    }

    #[test]
    fn unit__pick_winner__ladder_schema_then_stop_then_index() {
        let s = crate::sentinel::Sentinel::new(false, 0, None);
        let schema =
            json!({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]});
        let ok =
            |c: &str| json!({"choices": [{"message": {"content": c}, "finish_reason": "stop"}]});
        let schema_pass = ok(r#"{"a": 1}"#);
        let schema_fail = ok("not json at all");
        let truncated =
            json!({"choices": [{"message": {"content": "hi"}, "finish_reason": "length"}]});
        // schema-pass wins over everything
        let cands = vec![(200u16, schema_fail.clone()), (200, schema_pass.clone())];
        assert_eq!(pick_winner(&s, &cands, Some(&schema)), 1);
        // no schema: stop beats length
        let cands = vec![(200, truncated.clone()), (200, ok("fine"))];
        assert_eq!(pick_winner(&s, &cands, None), 1);
        // full tie: lowest index wins (deterministic)
        let cands = vec![(200, ok("x")), (200, ok("y"))];
        assert_eq!(pick_winner(&s, &cands, None), 0);
        // a 500 with valid shape ranks below a clean 200
        let cands = vec![(500, ok("partial")), (200, ok("fine"))];
        assert_eq!(pick_winner(&s, &cands, None), 1);
    }

    #[test]
    fn unit__request_schema__response_format_then_forced_tool() {
        let rf = json!({"response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}});
        assert_eq!(request_schema(&rf), Some(json!({"type": "object"})));
        let forced = json!({
            "tools": [{"type": "function", "function": {"name": "x", "parameters": {"type": "object"}}}],
            "tool_choice": {"type": "function", "function": {"name": "x"}}
        });
        assert_eq!(request_schema(&forced), Some(json!({"type": "object"})));
        assert_eq!(request_schema(&json!({"messages": []})), None);
    }

    #[test]
    fn unit__patch_winner__usage_sums_across_candidates() {
        let mut winner =
            json!({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}});
        let others = vec![
            json!({"usage": {"prompt_tokens": 10, "completion_tokens": 7}}),
            json!({"usage": {"prompt_tokens": 10, "completion_tokens": 9}}),
        ];
        let header = patch_winner(&mut winner, &others, 3, 3);
        assert_eq!(winner["usage"]["prompt_tokens"], json!(30));
        assert_eq!(winner["usage"]["completion_tokens"], json!(21));
        assert!(header.contains("asked=3 used=3"), "{header}");
    }
}
