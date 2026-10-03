//! Agentic cascade routing (A3): try a chain of models cheap-first and
//! escalate only when a candidate's answer fails the quality bar. The
//! judge reuses the best-of-N ladder semantics — a candidate passes when
//! it answered 2xx, matched the requested schema (if any) and finished
//! cleanly — so a request served by the first candidate costs exactly one
//! small-model call (AgentRouter-lineage behavior: most agentic steps
//! route to the smaller model; only the failures pay the big one).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::body::{Body, Bytes};
use axum::extract::{Extension, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::Response;
use serde_json::Value;

use crate::TraceId;
use crate::state::AppState;

pub const HEADER: &str = "x-blazar-cascade";

/// Process-wide counters surfaced in `/metrics`.
pub(crate) static CASCADE_RUNS: AtomicU64 = AtomicU64::new(0);
pub(crate) static CASCADE_ESCALATIONS: AtomicU64 = AtomicU64::new(0);

/// Candidates per cascade request (a chain longer than this is a topology
/// error, not a routing request).
pub(crate) const MAX_CASCADE: usize = 4;

/// Bodies above this size skip the cascade machinery: a prompt this large
/// has nothing small-model-shaped about it anymore.
pub(crate) const MAX_CASCADE_BODY_BYTES: usize = 1024 * 1024;

fn split_names(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Resolve the cascade chain: `x-blazar-cascade: a,b` header wins over the
/// body field `cascade: ["a","b"]`. Err = caller error (400). `Ok(None)` =
/// normal single-model path.
pub(crate) fn resolve_cascade(
    headers: &HeaderMap,
    req: &Value,
) -> Result<Option<Vec<String>>, String> {
    let header_names = headers
        .get(HEADER)
        .and_then(|v| v.to_str().ok())
        .map(split_names);
    let body_names = req.get("cascade").and_then(|c| c.as_array()).map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_str())
            .map(str::to_string)
            .collect::<Vec<_>>()
    });
    let header_pinned = header_names.is_some();
    let has_body_field = req.get("cascade").is_some();
    let names = match (header_names, body_names) {
        (Some(h), _) => h,
        (None, Some(b)) => b,
        (None, None) => {
            if has_body_field {
                return Err("cascade field must be a list of model names".into());
            }
            return Ok(None);
        }
    };
    if names.is_empty() {
        return Err("cascade list is empty — name at least one model".into());
    }
    if names.len() > MAX_CASCADE {
        return Err(format!(
            "cascade lists {} models — the cap is {MAX_CASCADE}",
            names.len()
        ));
    }
    if let Some(arr) = req.get("cascade").and_then(|c| c.as_array())
        && !header_pinned
        && arr.is_empty()
    {
        return Err("cascade list is empty — name at least one model".into());
    }
    Ok(Some(names))
}

/// The ollama-dialect schema ask: a non-null `format` object is the
/// JSON-schema constraint the candidate must satisfy.
pub(crate) fn ollama_schema(req: &Value) -> Option<&Value> {
    req.get("format").filter(|f| f.is_object())
}

/// Judge one ollama-dialect candidate: `(schema_ok, stop)`. Non-2xx fails
/// both rungs (it carries no answer to judge).
fn candidate_pass(
    sentinel: &crate::sentinel::Sentinel,
    status: StatusCode,
    body: &Value,
    schema: Option<&Value>,
) -> (bool, bool) {
    if !status.is_success() {
        return (false, false);
    }
    let content = body
        .pointer("/message/content")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let schema_ok = match schema {
        None => true,
        Some(s) => serde_json::from_str::<Value>(content)
            .is_ok_and(|v| sentinel.schema_check(s, &v).is_none()),
    };
    let stop = body
        .get("done_reason")
        .and_then(Value::as_str)
        .is_some_and(|r| r == "stop");
    (schema_ok, stop)
}

/// Header the client sees on every cascade response: the chain, the
/// winner, and why it stopped there.
fn cascade_header(tried: &[String], served_idx: usize, reason: &str) -> String {
    format!(
        "tried={} served={} reason={reason}",
        tried.join(","),
        tried[served_idx]
    )
}

/// Run the cascade: re-enter the normal per-model handler for each
/// candidate (full admission, translation, caching semantics per model),
/// judge each answer, serve the first pass. The cascade field and header
/// are stripped from the recursive request, so the depth is structurally 1;
/// candidates after the first winner are never sent (that is the entire
/// point of the chain). None-passed serves the LAST candidate's answer
/// with an honest marker.
///
/// Returns `None` when the request had no cascade ask — the caller's
/// normal path continues untouched.
#[allow(clippy::too_many_lines)]
pub(crate) async fn run(
    state: &Arc<AppState>,
    trace_ext: Option<Extension<TraceId>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    headers: HeaderMap,
    req: &Value,
) -> Option<Response> {
    let candidates = match resolve_cascade(&headers, req) {
        Ok(None) => return None,
        Ok(Some(c)) => c,
        Err(msg) => return Some(crate::ollama::api_error(400, &msg)),
    };
    let asks_stream = req.get("stream").and_then(Value::as_bool).unwrap_or(true);
    if asks_stream {
        return Some(crate::ollama::api_error(
            400,
            "cascade requires stream=false — a cascade winner cannot be judged before completion",
        ));
    }
    if serde_json::to_vec(req).map_or(true, |b| b.len() > MAX_CASCADE_BODY_BYTES) {
        return Some(crate::ollama::api_error(
            400,
            "cascade is unavailable for bodies over 1 MiB — send the prompt to one model directly",
        ));
    }
    CASCADE_RUNS.fetch_add(1, Ordering::Relaxed);

    // Cost ordering (A2 telemetry): when the BODY named the chain (no
    // header pin) and at least two candidates have measured decode
    // rates, try the cheapest first. The header always pins its order.
    let header_pinned = headers.get(HEADER).is_some();
    let mut tried = candidates;
    if !header_pinned {
        let hinted = tried
            .iter()
            .filter(|n| state.sup.decode_rate_for(n).is_some())
            .count();
        if hinted >= 2 {
            // Same request ⇒ queue/headroom/prompt/output are equal for
            // every candidate; only rate + cache differentiate.
            let cost = |name: &str| -> f64 {
                blazar_runtime::supervisor::route_cost(
                    state.sup.decode_rate_for(name),
                    state.sup.cache_hint_for(name),
                    0,
                    u32::MAX,
                    512,
                    256,
                )
            };
            tried.sort_by(|a, b| cost(a).total_cmp(&cost(b)));
        }
    }

    let mut cleaned = headers.clone();
    cleaned.remove(HEADER);
    let mut child_req = req.clone();
    child_req["model"] = Value::String(tried[0].clone());
    if let Some(obj) = child_req.as_object_mut() {
        obj.remove("cascade");
    }

    let schema = ollama_schema(req);
    let mut attempts: Vec<(StatusCode, Value)> = Vec::new();
    for (i, name) in tried.iter().enumerate() {
        if i > 0 {
            child_req["model"] = Value::String(name.clone());
        }
        let bytes = Bytes::from(serde_json::to_vec(&child_req).ok()?);
        // Boxed: this re-entry is the recursive edge (chat → run → chat);
        // the indirection keeps the future sized (E0733).
        let resp = Box::pin(crate::ollama::chat(
            State(state.clone()),
            trace_ext.clone(),
            key_ext.clone(),
            cleaned.clone(),
            bytes,
        ))
        .await;
        let status = resp.status();
        let body_bytes = axum::body::to_bytes(resp.into_body(), 8 * 1024 * 1024)
            .await
            .ok()?;
        let body: Value = serde_json::from_slice(&body_bytes).unwrap_or(Value::Null);
        let (schema_ok, stop) = candidate_pass(&state.sentinel, status, &body, schema);
        attempts.push((status, body));
        if schema_ok && stop {
            let escalated = i > 0;
            if escalated {
                CASCADE_ESCALATIONS.fetch_add(1, Ordering::Relaxed);
            }
            let hdr = cascade_header(
                &tried,
                i,
                if escalated { "escalated" } else { "first-pass" },
            );
            return Some(rebuild(attempts, i, hdr));
        }
    }
    let last = attempts.len().checked_sub(1)?;
    let hdr = cascade_header(&tried, last, "none-passed");
    Some(rebuild(attempts, last, hdr))
}

/// Rebuild the winner as a normal chat response: the winner's body with
/// token usage summed across every attempted candidate (failed candidates
/// burned real tokens; billing must not hide them), original status kept.
fn rebuild(mut attempts: Vec<(StatusCode, Value)>, served_idx: usize, hdr: String) -> Response {
    let (status, mut winner) = attempts.swap_remove(served_idx);
    let sum = |field: &str| -> u64 {
        attempts
            .iter()
            .chain(std::iter::once(&(status, winner.clone())))
            .map(|(_, b)| b.get(field).and_then(Value::as_u64).unwrap_or(0))
            .fold(0_u64, u64::saturating_add)
    };
    let prompt_total = sum("prompt_eval_count");
    let eval_total = sum("eval_count");
    if prompt_total > 0 {
        winner["prompt_eval_count"] = Value::from(prompt_total);
    }
    if eval_total > 0 {
        winner["eval_count"] = Value::from(eval_total);
    }
    let bytes = serde_json::to_vec(&winner).unwrap_or_default();
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .header(HEADER, hdr)
        .body(Body::from(bytes))
        .expect("static response parts")
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // suite convention: unit__scenario__expected (§6b)

    use super::*;

    fn hdr_map(v: Option<&str>) -> HeaderMap {
        let mut m = HeaderMap::new();
        if let Some(v) = v {
            m.insert(HEADER, v.parse().expect("test header value"));
        }
        m
    }

    fn body(cascade: Option<Vec<&str>>) -> Value {
        let mut v = serde_json::json!({"model": "m1", "stream": false, "messages": []});
        if let Some(c) = cascade {
            v["cascade"] = Value::Array(c.into_iter().map(|n| Value::String(n.into())).collect());
        }
        v
    }

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn unit__resolve_cascade__header_wins_body_cap_and_errors() {
        let none = serde_json::json!({"model": "m1"});
        assert!(resolve_cascade(&hdr_map(None), &none).unwrap().is_none());
        // Body field names the chain.
        assert_eq!(
            resolve_cascade(&hdr_map(None), &body(Some(vec!["a", "b"]))).unwrap(),
            Some(names(&["a", "b"]))
        );
        // Header wins over the body field.
        assert_eq!(
            resolve_cascade(&hdr_map(Some("z, y")), &body(Some(vec!["a"]))).unwrap(),
            Some(names(&["z", "y"]))
        );
        // Empty body list is a caller error.
        assert!(resolve_cascade(&hdr_map(None), &body(Some(vec![]))).is_err());
        // Garbage body shape (strings, not an array).
        let bad = serde_json::json!({"model": "m1", "cascade": "a,b"});
        assert!(resolve_cascade(&hdr_map(None), &bad).is_err());
        let too_many = serde_json::json!({"model": "m1", "cascade": ["a","b","c","d","e"]});
        let err = resolve_cascade(&hdr_map(None), &too_many).unwrap_err();
        assert!(err.contains("the cap is"), "{err}");
    }

    #[test]
    fn unit__candidate_pass__ladder_over_ollama_shape() {
        let sentinel = crate::sentinel::Sentinel::new(false, 0, None);
        let schema = serde_json::json!({"type": "object", "properties": {"ok": {"const": true}}, "required": ["ok"]});
        let good = serde_json::json!({
            "message": {"content": "{\"ok\": true}"},
            "done_reason": "stop"
        });
        let wrong = serde_json::json!({
            "message": {"content": "{\"ok\": false}"},
            "done_reason": "stop"
        });
        let truncated = serde_json::json!({
            "message": {"content": "{\"ok\": true}"},
            "done_reason": "length"
        });
        let no_schema_pass =
            serde_json::json!({"message": {"content": "hi"}, "done_reason": "stop"});
        // Schema asks discriminate; non-2xx fails both rungs.
        assert_eq!(
            candidate_pass(&sentinel, StatusCode::OK, &good, Some(&schema)),
            (true, true)
        );
        assert_eq!(
            candidate_pass(&sentinel, StatusCode::OK, &wrong, Some(&schema)),
            (false, true)
        );
        assert_eq!(
            candidate_pass(&sentinel, StatusCode::OK, &truncated, Some(&schema)),
            (true, false)
        );
        assert_eq!(
            candidate_pass(&sentinel, StatusCode::OK, &no_schema_pass, None),
            (true, true)
        );
        assert_eq!(
            candidate_pass(&sentinel, StatusCode::NOT_FOUND, &good, None),
            (false, false)
        );
    }

    #[test]
    fn unit__cascade_header__names_chain_winner_and_reason() {
        let tried = names(&["small", "big"]);
        assert_eq!(
            cascade_header(&tried, 0, "first-pass"),
            "tried=small,big served=small reason=first-pass"
        );
        assert_eq!(
            cascade_header(&tried, 1, "escalated"),
            "tried=small,big served=big reason=escalated"
        );
    }
}
