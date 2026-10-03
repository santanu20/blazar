//! POST /api/model-doctor — per-model capability certificate.
//!
//! Every probe runs through the REAL gateway path (admission queue,
//! dialect translation, `child_send`, sentinel) by invoking the ollama
//! handlers directly as functions. Auth is a middleware concern, so an
//! in-daemon call carries no key — the run is operator-initiated. No
//! self-HTTP loop: that would re-enter the body limit and lifecycle
//! middleware with a fake client disconnect semantics.
//!
//! The run itself is a `JobRuntime` job (kind `doctor`): durable row,
//! per-probe events, polling and cancel come free from the /v1/jobs
//! plane. The finished certificate is also upserted into `model_caps`
//! where routing surfaces can read it without scraping job history.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use axum::body::Bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use crate::state::AppState;

/// Cold-load allowance for the FIRST probe: weights + KV allocation for a
/// large offload ladder takes minutes on small-VRAM boxes. Subsequent
/// probes assume a warm resident.
const FIRST_PROBE_TIMEOUT: Duration = Duration::from_secs(300);
/// Warm probes: generation itself is seconds.
const PROBE_TIMEOUT: Duration = Duration::from_secs(120);
/// Hard ceiling for the whole run. A hung child cannot pin a doctor job
/// forever — probes left when the budget is gone are marked FAIL(timeout)
/// and the partial certificate is still stored.
const OVERALL_CAP: Duration = Duration::from_secs(600);
/// Probe replies are tiny by construction; anything larger means the lane
/// misbehaved and gets truncated, not buffered whole.
const PROBE_BODY_CAP: usize = 4 * 1024 * 1024;

static SEQ: AtomicU64 = AtomicU64::new(1);
static BOOT_NANOS: OnceLock<u128> = OnceLock::new();

fn next_job_id() -> String {
    let boot = *BOOT_NANOS.get_or_init(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    });
    format!("doc_{boot:x}_{}", SEQ.fetch_add(1, Ordering::Relaxed))
}

/// One probe verdict. `PASS` / `FAIL` / `N/A` — N/A is reserved for
/// capabilities the lane honestly does not serve (e.g. embeddings on a
/// generative-only lane), never to soften a failure.
pub(crate) struct Verdict {
    pub status: &'static str,
    pub receipt: String,
}

fn pass(receipt: String) -> Verdict {
    Verdict {
        status: "PASS",
        receipt,
    }
}

fn fail(receipt: String) -> Verdict {
    Verdict {
        status: "FAIL",
        receipt,
    }
}

fn na(receipt: String) -> Verdict {
    Verdict {
        status: "N/A",
        receipt,
    }
}

fn excerpt(body: &[u8], max: usize) -> String {
    let text = String::from_utf8_lossy(body);
    let text = text.trim();
    if text.len() <= max {
        text.to_owned()
    } else {
        let mut cut = max;
        while !text.is_char_boundary(cut) {
            cut -= 1;
        }
        format!("{}…", &text[..cut])
    }
}

// ---------------------------------------------------------------------------
// Judges — pure (status, body) classifiers so the wire vocabulary is
// unit-pinned without a live child.
// ---------------------------------------------------------------------------

/// chat probe: 200 with an assistant message carrying non-empty content.
fn judge_chat(code: u16, body: &[u8]) -> Verdict {
    if code != 200 {
        return fail(format!("HTTP {code}: {}", excerpt(body, 160)));
    }
    let parsed: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => return fail(format!("200 but body is not JSON: {e}")),
    };
    let content = parsed["message"]["content"]
        .as_str()
        .or(parsed["response"].as_str())
        .unwrap_or_default();
    if content.trim().is_empty() {
        return fail(format!("200 but empty generation: {}", excerpt(body, 160)));
    }
    pass(format!("200, {} chars", content.len()))
}

/// stream probe: 200 plus streaming evidence in EITHER wire dialect the
/// gateway emits. The ollama lane streams `application/x-ndjson` (one JSON
/// object per line); OpenAI-style surfaces stream `text/event-stream` with
/// `data:` frames. A buffered `application/json` body is NOT streaming.
fn judge_stream(code: u16, content_type: Option<&str>, body: &[u8]) -> Verdict {
    if code != 200 {
        return fail(format!("HTTP {code}: {}", excerpt(body, 160)));
    }
    let ct = content_type.unwrap_or("(none)");
    if ct.contains("text/event-stream") {
        let frames = body
            .split(|b| *b == b'\n')
            .filter(|l| l.starts_with(b"data:"))
            .count();
        return if frames == 0 {
            fail("event-stream carried no data frames".to_string())
        } else {
            pass(format!("200, {frames} SSE frames"))
        };
    }
    if ct.contains("x-ndjson") || ct.contains("json-lines") || ct.contains("jsonl") {
        let frames = body
            .split(|b| *b == b'\n')
            .filter(|l| l.iter().any(|b| !b.is_ascii_whitespace()))
            .count();
        return if frames == 0 {
            fail("ndjson stream carried no frames".to_string())
        } else {
            pass(format!("200, {frames} NDJSON frames"))
        };
    }
    fail(format!(
        "200 but content type is {ct} (expected a streaming dialect)"
    ))
}

/// structured-output probe: 200 and a JSON body whose .response parses as
/// JSON (the strict grammar should guarantee at least that).
fn judge_json(code: u16, body: &[u8]) -> Verdict {
    if code != 200 {
        return fail(format!("HTTP {code}: {}", excerpt(body, 160)));
    }
    let parsed: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => return fail(format!("200 but body is not JSON: {e}")),
    };
    let inner = parsed["response"]
        .as_str()
        .or(parsed["message"]["content"].as_str())
        .unwrap_or_default();
    match serde_json::from_str::<Value>(inner) {
        Ok(_) => pass("200, response parses as JSON".to_string()),
        Err(e) => fail(format!("response is not valid JSON: {e}: {inner}")),
    }
}

/// tool-call probe: 200 AND the model actually emitted a tool call. A
/// model that answers in prose when handed a tool is exactly the failure
/// the certificate exists to surface — that is a FAIL, not an N/A.
fn judge_tools(code: u16, body: &[u8]) -> Verdict {
    if code != 200 {
        return fail(format!("HTTP {code}: {}", excerpt(body, 160)));
    }
    let parsed: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => return fail(format!("200 but body is not JSON: {e}")),
    };
    let calls = parsed["message"]["tool_calls"].as_array();
    if calls.is_some_and(|c| !c.is_empty()) {
        return pass(format!(
            "200, {} tool call(s), first: {}",
            calls.map_or(0, Vec::len),
            calls
                .and_then(|c| {
                    c.first().map(|t| {
                        t["function"]["name"]
                            .as_str()
                            .unwrap_or("(unnamed)")
                            .to_string()
                    })
                })
                .unwrap_or_default()
        ));
    }
    fail(format!(
        "200 but no tool_calls — model answered in prose: {}",
        excerpt(body, 160)
    ))
}

/// embeddings probe: 200 with vectors; a lane refusal that names
/// embeddings is an honest N/A; anything else is a FAIL.
fn judge_embed(code: u16, body: &[u8]) -> Verdict {
    if code == 200 {
        let parsed: Value = match serde_json::from_slice(body) {
            Ok(v) => v,
            Err(e) => return fail(format!("200 but body is not JSON: {e}")),
        };
        let dims = parsed["embeddings"]
            .as_array()
            .and_then(|a| a.first())
            .and_then(|v| v.as_array())
            .map_or(0, Vec::len);
        if dims == 0 {
            return fail(format!("200 but no vectors: {}", excerpt(body, 160)));
        }
        return pass(format!("200, {dims} dims"));
    }
    let text = excerpt(body, 200);
    if code == 400 && text.to_lowercase().contains("embed") {
        return na(format!("lane refuses embeddings for this model: {text}"));
    }
    fail(format!("HTTP {code}: {text}"))
}

// ---------------------------------------------------------------------------
// Probe transport — direct handler invocation through the real path.
// ---------------------------------------------------------------------------

async fn call_chat(state: &Arc<AppState>, body: Value) -> (u16, Option<String>, Bytes) {
    let resp = crate::ollama::chat(
        State(Arc::clone(state)),
        None,
        None,
        HeaderMap::new(),
        Bytes::from(body.to_string()),
    )
    .await;
    let code = resp.status().as_u16();
    let ct = resp
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let bytes = axum::body::to_bytes(resp.into_body(), PROBE_BODY_CAP)
        .await
        .unwrap_or_default();
    (code, ct, bytes)
}

async fn call_embed(state: &Arc<AppState>, body: Value) -> (u16, Bytes) {
    let resp = crate::ollama::embed(
        State(Arc::clone(state)),
        None,
        Bytes::from(body.to_string()),
    )
    .await;
    let code = resp.status().as_u16();
    let bytes = axum::body::to_bytes(resp.into_body(), PROBE_BODY_CAP)
        .await
        .unwrap_or_default();
    (code, bytes)
}

// ---------------------------------------------------------------------------
// Handlers.
// ---------------------------------------------------------------------------

/// POST /api/model-doctor {"model": "..."} — queue the probe run as a
/// durable job; poll `/v1/jobs/{id}` for the certificate.
pub async fn run(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return crate::error_response(400, &format!("invalid JSON: {e}")),
    };
    let model = req["model"].as_str().unwrap_or_default().trim().to_string();
    if model.is_empty() {
        return crate::error_response(400, "missing \"model\"");
    }
    // Resolve through the canonical gateway ladder FIRST: unknown names 404
    // before any job row exists, and the static caps need the row anyway.
    // Same resolver as /api/chat so `name:quant` display forms resolve
    // identically here.
    let row = match state.with_store(|s| crate::proxy::resolve_model(s, &model)) {
        Some(Ok(r)) => r,
        Some(Err(teach)) => return crate::error_response(404, &teach),
        None => return crate::error_response(500, "store unavailable"),
    };

    // Static caps never spawn a child: vision is a pulled-mmproj fact,
    // thinking is a chat-template fact.
    let vision = if row.mmproj_path.is_some() {
        Verdict {
            status: "PASS",
            receipt: "mmproj present (capability not exercised by this run)".to_string(),
        }
    } else {
        na("no mmproj pulled — vision requests would be text-only".to_string())
    };
    let think = if crate::ollama::template_supports_thinking_cached(&row) {
        Verdict {
            status: "PASS",
            receipt: "chat template carries a thinking block (not exercised)".to_string(),
        }
    } else {
        na("chat template has no thinking markers".to_string())
    };

    let id = next_job_id();
    state.jobs.record_created(
        &state,
        &id,
        "doctor",
        Some(&model),
        json!({ "model": model }),
    );
    tokio::spawn(run_probes(
        Arc::clone(&state),
        id.clone(),
        model.clone(),
        vision,
        think,
    ));

    axum::Json(json!({
        "id": id,
        "object": "blazar.doctor",
        "model": model,
        "status": "queued",
        "poll_url": format!("/v1/jobs/{id}"),
        "cancel_url": format!("/v1/jobs/{id}/cancel"),
        "cert_url": format!("/api/model-doctor/{model}"),
    }))
    .into_response()
}

/// GET /api/model-doctor/{model} — the latest stored certificate.
pub async fn cert(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(model): axum::extract::Path<String>,
) -> Response {
    let record = state
        .with_store(|s| {
            // Canonical ladder so `name:quant` display forms resolve the
            // same way here as on /api/chat.
            let resolved = crate::proxy::resolve_model(s, &model).ok()?.name;
            s.get_model_caps(&resolved).ok().flatten()
        })
        .flatten();
    let Some((engine_tag, caps_json)) = record else {
        return crate::error_response(
            404,
            &format!(
                "no certificate on record — POST /api/model-doctor {{\"model\": \"{model}\"}} runs the probes"
            ),
        );
    };
    let caps: Value = serde_json::from_str(&caps_json).unwrap_or(Value::Null);
    axum::Json(json!({
        "object": "blazar.model-doctor",
        "model": model,
        "engine_tag": engine_tag,
        "caps": caps,
    }))
    .into_response()
}

#[allow(clippy::too_many_lines)]
async fn run_probes(
    state: Arc<AppState>,
    id: String,
    model: String,
    vision: Verdict,
    think: Verdict,
) {
    state.jobs.record_running(&state, &id);
    let started = Instant::now();
    let deadline = started + OVERALL_CAP;

    // Between probes: honor a cancel that landed on the row. The ledger
    // close is one-way, so this task's later writes would no-op anyway —
    // this stops the actual work instead of burning child time.
    let row_cancelled = |state: &Arc<AppState>| {
        state
            .with_store(|s| s.get_job(&id).ok().flatten())
            .flatten()
            .is_some_and(|r| r.state == "cancelled")
    };

    let mut caps = serde_json::Map::new();

    // chat / load — the first probe carries the cold-load allowance.
    let v_chat = {
        if row_cancelled(&state) {
            return;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            fail("overall time budget exhausted".to_string())
        } else {
            let t0 = Instant::now();
            let body = json!({
                "model": model,
                "stream": false,
                "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
                "options": {"num_predict": 8},
            });
            let verdict = match tokio::time::timeout(
                FIRST_PROBE_TIMEOUT.min(remaining),
                call_chat(&state, body),
            )
            .await
            {
                Ok((code, _, bytes)) => judge_chat(code, &bytes),
                Err(_) => fail(format!(
                    "no completion within {}s (includes cold load)",
                    FIRST_PROBE_TIMEOUT.as_secs()
                )),
            };
            if verdict.status == "PASS" {
                Verdict {
                    status: verdict.status,
                    receipt: format!(
                        "{} in {}ms (includes load)",
                        verdict.receipt,
                        t0.elapsed().as_millis()
                    ),
                }
            } else {
                verdict
            }
        }
    };
    state.jobs.record_event(
        &state,
        &id,
        "probe:chat",
        json!({"status": v_chat.status, "receipt": v_chat.receipt}),
    );
    caps.insert("chat".into(), verdict_json(&v_chat));

    // stream
    let v_stream = {
        if row_cancelled(&state) {
            return;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            fail("overall time budget exhausted".to_string())
        } else {
            let body = json!({
                "model": model,
                "stream": true,
                "messages": [{"role": "user", "content": "count: one two three"}],
                "options": {"num_predict": 4},
            });
            match tokio::time::timeout(PROBE_TIMEOUT.min(remaining), call_chat(&state, body)).await
            {
                Ok((code, ct, bytes)) => judge_stream(code, ct.as_deref(), &bytes),
                Err(_) => fail(format!("no completion within {}s", PROBE_TIMEOUT.as_secs())),
            }
        }
    };
    state.jobs.record_event(
        &state,
        &id,
        "probe:stream",
        json!({"status": v_stream.status, "receipt": v_stream.receipt}),
    );
    caps.insert("stream".into(), verdict_json(&v_stream));

    // structured JSON
    let v_json = {
        if row_cancelled(&state) {
            return;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            fail("overall time budget exhausted".to_string())
        } else {
            let body = json!({
                "model": model,
                "stream": false,
                "messages": [{"role": "user", "content": "What is the capital of France? Answer as JSON."}],
                "format": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": false
                },
                "options": {"num_predict": 32},
            });
            match tokio::time::timeout(PROBE_TIMEOUT.min(remaining), call_chat(&state, body)).await
            {
                Ok((code, _, bytes)) => judge_json(code, &bytes),
                Err(_) => fail(format!("no completion within {}s", PROBE_TIMEOUT.as_secs())),
            }
        }
    };
    state.jobs.record_event(
        &state,
        &id,
        "probe:json",
        json!({"status": v_json.status, "receipt": v_json.receipt}),
    );
    caps.insert("json".into(), verdict_json(&v_json));

    // tool elicitation — the probe agents actually care about.
    let v_tools = {
        if row_cancelled(&state) {
            return;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            fail("overall time budget exhausted".to_string())
        } else {
            let body = json!({
                "model": model,
                "stream": false,
                "messages": [{
                    "role": "user",
                    "content": "What is the weather in Paris? You MUST call the get_weather tool to answer."
                }],
                "tools": [{"type": "function", "function": {
                    "name": "get_weather",
                    "description": "current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }
                }}],
                "options": {"num_predict": 64},
            });
            match tokio::time::timeout(PROBE_TIMEOUT.min(remaining), call_chat(&state, body)).await
            {
                Ok((code, _, bytes)) => judge_tools(code, &bytes),
                Err(_) => fail(format!("no completion within {}s", PROBE_TIMEOUT.as_secs())),
            }
        }
    };
    state.jobs.record_event(
        &state,
        &id,
        "probe:tools",
        json!({"status": v_tools.status, "receipt": v_tools.receipt}),
    );
    caps.insert("tools".into(), verdict_json(&v_tools));

    // embeddings — separate handler, honest N/A on lane refusal.
    let v_embed = {
        if row_cancelled(&state) {
            return;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            fail("overall time budget exhausted".to_string())
        } else {
            match tokio::time::timeout(
                PROBE_TIMEOUT.min(remaining),
                call_embed(&state, json!({"model": model, "input": "doctor probe"})),
            )
            .await
            {
                Ok((code, bytes)) => judge_embed(code, &bytes),
                Err(_) => fail(format!("no completion within {}s", PROBE_TIMEOUT.as_secs())),
            }
        }
    };
    state.jobs.record_event(
        &state,
        &id,
        "probe:embeddings",
        json!({"status": v_embed.status, "receipt": v_embed.receipt}),
    );
    caps.insert("embeddings".into(), verdict_json(&v_embed));

    caps.insert("vision".into(), verdict_json(&vision));
    caps.insert("think".into(), verdict_json(&think));

    // Engine tag: prefer the LIVE resident (proof the probes exercised a
    // real child); a model that never became resident gets an honest tag.
    let engine_tag = state.sup.ps().iter().find(|r| r.name == model).map_or_else(
        || "unresolved (never resident this run)".to_string(),
        |r| r.engine.clone(),
    );

    let cert = json!({
        "object": "blazar.model-doctor",
        "model": model,
        "engine_tag": engine_tag,
        "tested_at": blazar_core::store::unix_now(),
        "total_ms": u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
        "caps": Value::Object(caps),
    });
    if let Some(err) = state
        .with_store(|s| {
            s.put_model_caps(
                cert["model"].as_str().unwrap_or_default(),
                &engine_tag,
                &cert.to_string(),
            )
            .err()
        })
        .flatten()
    {
        tracing::warn!(target: "blazar::model_doctor", job = %id, %err, "certificate store write failed — job result still carries it");
    }
    state
        .jobs
        .record_completed(&state, &id, cert.to_string().as_bytes(), "application/json");
}

fn verdict_json(v: &Verdict) -> Value {
    json!({"status": v.status, "receipt": v.receipt})
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use super::*;

    #[test]
    fn unit__judge_chat__pass_fail_shapes() {
        let ok = br#"{"message":{"role":"assistant","content":"ok"}}"#;
        assert_eq!(judge_chat(200, ok).status, "PASS");
        let empty = br#"{"message":{"role":"assistant","content":"  "}}"#;
        assert_eq!(judge_chat(200, empty).status, "FAIL");
        assert_eq!(judge_chat(500, b"boom").status, "FAIL");
        assert_eq!(judge_chat(200, b"not json").status, "FAIL");
    }

    #[test]
    fn unit__judge_stream__both_dialects_pass_buffered_fails() {
        // SSE dialect (OpenAI-style surfaces)
        let sse = b"data: {\"a\":1}\n\ndata: [DONE]\n\n";
        assert_eq!(
            judge_stream(200, Some("text/event-stream"), sse).status,
            "PASS"
        );
        // NDJSON dialect (the ollama lane the probes actually traverse)
        let ndjson = b"{\"message\":\"hi\"}\n{\"done\":true}\n";
        assert_eq!(
            judge_stream(200, Some("application/x-ndjson"), ndjson).status,
            "PASS"
        );
        // a buffered JSON body is not streaming, whatever the frames look like
        assert_eq!(
            judge_stream(200, Some("application/json"), sse).status,
            "FAIL"
        );
        // no data frames
        assert_eq!(
            judge_stream(200, Some("text/event-stream"), b":keepalive\n\n").status,
            "FAIL"
        );
    }

    #[test]
    fn unit__judge_json__inner_must_parse() {
        let ok = br#"{"message":{"content":"{\"answer\":\"Paris\"}"}}"#;
        assert_eq!(judge_json(200, ok).status, "PASS");
        let prose = br#"{"message":{"content":"Paris"}}"#;
        assert_eq!(judge_json(200, prose).status, "FAIL");
    }

    #[test]
    fn unit__judge_tools__prose_without_calls_fails() {
        let called = br#"{"message":{"tool_calls":[{"function":{"name":"get_weather"}}]}}"#;
        assert_eq!(judge_tools(200, called).status, "PASS");
        let prose = br#"{"message":{"content":"It is sunny."}}"#;
        let v = judge_tools(200, prose);
        assert_eq!(v.status, "FAIL");
        assert!(v.receipt.contains("no tool_calls"));
    }

    #[test]
    fn unit__judge_embed__refusal_is_na_only_when_named() {
        let ok = br#"{"embeddings":[[0.1,0.2]]}"#;
        assert_eq!(judge_embed(200, ok).status, "PASS");
        let refuse = br#"{"error":"model does not support embeddings"}"#;
        assert_eq!(judge_embed(400, refuse).status, "N/A");
        // a 400 that never mentions embeddings is a real failure
        assert_eq!(judge_embed(400, b"bad input").status, "FAIL");
        assert_eq!(judge_embed(200, br#"{"embeddings":[]}"#).status, "FAIL");
    }

    #[test]
    fn unit__excerpt__utf8_safe_and_bounded() {
        assert_eq!(excerpt(b"short", 32), "short");
        let long = "é".repeat(64);
        let cut = excerpt(long.as_bytes(), 10);
        // 5 é chars (10 bytes) + the 3-byte ellipsis: never splits a
        // char, never exceeds the cap by more than the ellipsis itself.
        assert!(cut.ends_with('…') && cut.chars().count() == 6);
    }
}
