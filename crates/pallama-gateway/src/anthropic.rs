//! D7 — Anthropic `/v1/messages` native translate lane.
//!
//! Claude-dialect clients (Claude Code, anthropic SDKs) speak
//! `/v1/messages`: top-level `system`, content-block arrays,
//! `tool_use`/`tool_result` blocks, `stop_reason` semantics and the
//! `message_start` → `content_block_*` → `message_delta` `SSE` event
//! family. The engine speaks `OpenAI` chat. This module translates BOTH
//! directions locally: request → `OpenAI` chat body, child response →
//! Anthropic shape (`JSON` + `SSE`). Local engine only — a remote-prefixed
//! model gets a teaching error (remote fleet speaks its own dialect).

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use serde_json::{json, Value};
use std::sync::Arc;

use crate::proxy::{admission_gate_slo, child_base, ensure_with_admission};
use crate::queue::Priority;

/// POST /v1/messages — full Anthropic Messages API translate.
#[allow(clippy::too_many_lines)]
pub async fn messages(
    State(state): State<Arc<crate::state::AppState>>,
    key_ext: Option<axum::extract::Extension<crate::keys::KeyCtx>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let parsed: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return anthropic_error(400, "invalid_request_error", &format!("invalid JSON: {e}"))
        }
    };
    let model = parsed
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if model.is_empty() {
        return anthropic_error(
            400,
            "invalid_request_error",
            "missing `model` field in request body",
        );
    }
    if crate::remotes::split_remote(&model, &state.config).is_some() {
        return anthropic_error(
            400,
            "invalid_request_error",
            "the /v1/messages translate lane serves the local engine only; \
             remote-prefixed models are not translated (their dialect is the remote's own)",
        );
    }
    // F46: the Anthropic schema makes `max_tokens` REQUIRED — accepting
    // requests without an output cap lets a reasoning model burn the
    // whole context before the client learns anything.
    if parsed.get("max_tokens").and_then(Value::as_u64).is_none() {
        return anthropic_error(
            400,
            "invalid_request_error",
            "max_tokens is required and must be a non-negative integer",
        );
    }
    if let Some(key) = key_ext.as_ref().map(|axum::extract::Extension(k)| k) {
        if let Some(entry) = state.keys.entry(&key.name) {
            if let Err(rej) = state.keys.check(&entry, &model) {
                return rej.to_response();
            }
            // F53: charge AFTER translate_request validation — 400s must
            // not consume a request from the key's budget.
        }
    }
    let stream = parsed
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let openai_body = match translate_request(&parsed, stream) {
        Ok(b) => b,
        Err(msg) => return anthropic_error(400, "invalid_request_error", &msg),
    };
    if let Some(key) = key_ext.as_ref().map(|axum::extract::Extension(k)| k) {
        state.keys.charge_request(&key.name);
    }
    if let Some(err) = state.sentinel.strict_tool_def_error_cached(&openai_body) {
        return anthropic_error(
            400,
            "invalid_request_error",
            &format!("invalid tools: {err}"),
        );
    }
    let eff = state
        .sup
        .ps()
        .into_iter()
        .find(|p| p.name == model)
        .map_or_else(|| state.config.effective_ctx(&model), |p| p.ctx);
    if let Err(resp) =
        crate::preflight::enforce_prompt_fits(&state, &model, &openai_body, eff).await
    {
        return resp;
    }
    let priority = Priority::from_header(
        headers
            .get("x-pallama-priority")
            .and_then(|v| v.to_str().ok()),
    );
    let prefix = crate::proxy::affinity_hash_bytes(
        &serde_json::to_vec(&openai_body).unwrap_or_else(|_| body.to_vec()),
    );
    let (engine, _load_ms) = match ensure_with_admission(&state, &model, priority, prefix).await {
        Ok(ok) => ok,
        Err(resp) => return resp,
    };
    let key_name = engine.name.clone();
    state.sup.note_prefix_hit(&key_name);
    let deadline_ms = headers
        .get("x-pallama-deadline-ms")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    let guard = match admission_gate_slo(
        &state,
        &key_name,
        priority,
        deadline_ms,
        body.len(),
        key_ext
            .as_ref()
            .map(|axum::extract::Extension(k)| (k.name.as_str(), k.weight)),
    )
    .await
    {
        Ok(g) => g,
        Err(resp) => return resp,
    };
    let url = format!("{}/v1/chat/completions", child_base(&engine.endpoint));
    // F44: pooled client (10-min total timeout) instead of a per-request
    // build — same transport every other child lane uses.
    let client = state.http.clone();
    let send = crate::proxy::child_auth(
        client
            .post(&url)
            .header("content-type", "application/json")
            .body(serde_json::to_vec(&openai_body).unwrap_or_default()),
        &engine,
    );
    if stream {
        let resp = match send.send().await {
            Ok(r) => r,
            Err(e) => return anthropic_error(502, "api_error", &format!("engine: {e}")),
        };
        // F45: a non-2xx child body is a JSON error, not an SSE stream —
        // surfacing it as 200+empty-events hangs clients. Translate the
        // status instead of framing the error bytes as events.
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            let detail = serde_json::from_str::<serde_json::Value>(&body)
                .ok()
                .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
                .unwrap_or_else(|| body.chars().take(200).collect());
            return anthropic_error(502, "api_error", &format!("engine {status}: {detail}"));
        }
        let id = format!("msg_{}", unique_suffix());
        let model_label = model.clone();
        let child_stream = resp.bytes_stream();
        let events = anthropic_sse_stream(child_stream, StreamState::new(&id, &model_label), guard);
        return Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/event-stream")
            .body(Body::from_stream(events))
            .unwrap_or_else(|e| anthropic_error(500, "api_error", &format!("stream: {e}")));
    }
    let resp = match send.send().await {
        Ok(r) => r,
        Err(e) => return anthropic_error(502, "api_error", &format!("engine: {e}")),
    };
    let status = resp.status();
    let bytes = match resp.bytes().await {
        Ok(b) => b,
        Err(e) => return anthropic_error(502, "api_error", &format!("engine: {e}")),
    };
    let openai: Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(_) => {
            return anthropic_error(
                status.as_u16(),
                "api_error",
                &String::from_utf8_lossy(&bytes[..bytes.len().min(400)]),
            )
        }
    };
    if !status.is_success() {
        let msg = openai
            .pointer("/error/message")
            .and_then(Value::as_str)
            .unwrap_or("engine error");
        return anthropic_error(status.as_u16(), "api_error", msg);
    }
    let translated = translate_response(&openai, &model);
    (StatusCode::OK, axum::Json(translated)).into_response()
}

/// POST `/v1/messages/count_tokens` — Anthropic counting over `/tokenize`.
pub async fn count_tokens(
    State(state): State<Arc<crate::state::AppState>>,
    axum::extract::Extension(trace): axum::extract::Extension<crate::TraceId>,
    headers: HeaderMap,
    key_ext: Option<axum::extract::Extension<crate::keys::KeyCtx>>,
    body: Bytes,
) -> Response {
    tracing::debug!(target: "pallama::anthropic", trace = %trace.0, "count_tokens");
    let _ = &headers;
    let parsed: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return anthropic_error(400, "invalid_request_error", &format!("invalid JSON: {e}"))
        }
    };
    let model = parsed
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if model.is_empty() {
        return anthropic_error(400, "invalid_request_error", "missing `model` field");
    }
    // F48: tokenizing a model SP loads it — scoped keys must not spawn
    // outside their scope through this lane.
    if let Some(key) = key_ext.as_ref().map(|axum::extract::Extension(k)| k) {
        if let Some(entry) = state.keys.entry(&key.name) {
            if let Err(rej) = state.keys.check(&entry, &model) {
                return rej.to_response();
            }
            state.keys.charge_request(&key.name);
        }
    }
    let mut text = String::new();
    if let Some(sys) = parsed.get("system") {
        if let Some(s) = system_to_text(sys) {
            text.push_str(&s);
            text.push_str("\n\n");
        }
    }
    if let Some(msgs) = parsed.get("messages").and_then(Value::as_array) {
        for m in msgs {
            if let Some(content) = m.get("content") {
                text.push_str(&content_to_text(content));
                text.push('\n');
            }
        }
    }
    let (engine, _) = match ensure_with_admission(&state, &model, Priority::Normal, None).await {
        Ok(ok) => ok,
        Err(resp) => return resp,
    };
    let url = format!("{}/tokenize", child_base(&engine.endpoint));
    // F44: pooled client — the old bare `Client::new()` had NO timeout,
    // so a dead child hung the count_tokens lane forever.
    let client = state.http.clone();
    let resp = match crate::proxy::child_auth(client.post(&url), &engine)
        .json(&json!({"content": text}))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return anthropic_error(502, "api_error", &format!("engine: {e}")),
    };
    let v: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => return anthropic_error(502, "api_error", &format!("engine: {e}")),
    };
    let n = v
        .get("tokens")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    (StatusCode::OK, axum::Json(json!({"input_tokens": n}))).into_response()
}

// ---------- request direction ----------

/// Anthropic Messages body → `OpenAI` chat body. `stream` is forced onto
/// the translated body; callers add `stream_options` themselves.
pub fn translate_request(v: &Value, stream: bool) -> Result<Value, String> {
    let mut out = json!({"model": v.get("model").and_then(Value::as_str).unwrap_or_default()});
    let obj = v
        .as_object()
        .ok_or_else(|| "request body must be a JSON object".to_string())?;
    let mut messages: Vec<Value> = Vec::new();
    if let Some(sys) = obj.get("system") {
        if let Some(text) = system_to_text(sys) {
            messages.push(json!({"role": "system", "content": text}));
        }
    }
    let msgs = obj
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| "missing `messages` array".to_string())?;
    for m in msgs {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        let content = m.get("content").unwrap_or(&Value::Null);
        translate_message(role, content, &mut messages)?;
    }
    out["messages"] = json!(messages);
    if let Some(mt) = obj.get("max_tokens").and_then(Value::as_u64) {
        out["max_tokens"] = json!(mt);
    }
    for key in ["temperature", "top_p"] {
        if let Some(x) = obj.get(key) {
            if !x.is_null() {
                out[key] = x.clone();
            }
        }
    }
    if let Some(stops) = obj.get("stop_sequences").and_then(Value::as_array) {
        let seqs: Vec<&str> = stops.iter().filter_map(Value::as_str).collect();
        if !seqs.is_empty() {
            out["stop"] = json!(seqs);
        }
    }
    if let Some(tools) = obj.get("tools").and_then(Value::as_array) {
        let mapped: Vec<Value> = tools
            .iter()
            .filter_map(|t| {
                let name = t.get("name")?.as_str()?;
                let desc = t.get("description").and_then(Value::as_str).unwrap_or("");
                let schema = t.get("input_schema").cloned().unwrap_or(json!({}));
                Some(json!({
                    "type": "function",
                    "function": {"name": name, "description": desc, "parameters": schema}
                }))
            })
            .collect();
        if !mapped.is_empty() {
            out["tools"] = json!(mapped);
            if let Some(choice) = obj.get("tool_choice") {
                out["tool_choice"] = translate_tool_choice(choice);
            }
        }
    }
    out["stream"] = json!(stream);
    if stream {
        out["stream_options"] = json!({"include_usage": true});
    }
    Ok(out)
}

fn translate_tool_choice(choice: &Value) -> Value {
    match choice.get("type").and_then(Value::as_str) {
        Some("any") => json!("required"),
        Some("tool") => {
            let name = choice
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            json!({"type": "function", "function": {"name": name}})
        }
        _ => json!("auto"),
    }
}

fn system_to_text(sys: &Value) -> Option<String> {
    match sys {
        Value::String(s) => Some(s.clone()),
        Value::Array(blocks) => {
            let parts: Vec<String> = blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(Value::as_str).map(str::to_string))
                .collect();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join("\n\n"))
            }
        }
        _ => None,
    }
}

/// One Anthropic message (string or block array) → one or more `OpenAI`
/// messages. `tool_result` blocks become standalone `OpenAI` `tool`
/// messages; assistant `tool_use` blocks fold into `tool_calls`.
fn translate_message(role: &str, content: &Value, out: &mut Vec<Value>) -> Result<(), String> {
    if content.is_string() || content.is_null() {
        out.push(json!({"role": role, "content": content}));
        return Ok(());
    }
    let blocks = content
        .as_array()
        .ok_or_else(|| "message.content must be a string or block array".to_string())?;
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut tool_results: Vec<Value> = Vec::new();
    let mut thinking: Vec<String> = Vec::new();
    for b in blocks {
        match b.get("type").and_then(Value::as_str).unwrap_or("text") {
            "text" => {
                if let Some(t) = b.get("text").and_then(Value::as_str) {
                    text_parts.push(t.to_string());
                }
            }
            "image" => {
                let src = b.get("source").unwrap_or(&Value::Null);
                if src.get("type").and_then(Value::as_str) == Some("base64") {
                    let mt = src
                        .get("media_type")
                        .and_then(Value::as_str)
                        .unwrap_or("image/png");
                    let data = src.get("data").and_then(Value::as_str).unwrap_or_default();
                    text_parts.push(format!("[image: {mt}, {} bytes]", data.len()));
                    // vision rides the engine's multimodal lane; the URL
                    // form is preserved for engines that honor it via a
                    // follow-up content part (kept minimal: text marker
                    // only — mmproj models receive the marker today).
                }
            }
            "tool_use" => {
                let id = b.get("id").and_then(Value::as_str).unwrap_or_default();
                let name = b.get("name").and_then(Value::as_str).unwrap_or_default();
                let args = b.get("input").cloned().unwrap_or(json!({}));
                tool_calls.push(json!({
                    "id": id, "type": "function",
                    "function": {"name": name, "arguments": args.to_string()}
                }));
            }
            "tool_result" => {
                let id = b
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let inner = b.get("content").map(content_to_text).unwrap_or_default();
                tool_results.push(json!({"role": "tool", "tool_call_id": id, "content": inner}));
            }
            // F47: preserve prior-turn reasoning — `thinking` text rides
            // the assistant message as `reasoning_content` (engines that
            // model reasoning read it; others ignore unknown fields).
            // `redacted_thinking` is an opaque encrypted payload with no
            // OpenAI representation, so dropping it is lossless.
            "thinking" => {
                if let Some(t) = b.get("thinking").and_then(Value::as_str) {
                    thinking.push(t.to_string());
                }
            }
            "redacted_thinking" | "document" => {}
            other => return Err(format!("unsupported content block type: {other}")),
        }
    }
    let mut pushed_text = false;
    if !text_parts.is_empty() || (tool_calls.is_empty() && tool_results.is_empty()) {
        let text = text_parts.join("");
        out.push(json!({"role": role, "content": text}));
        pushed_text = true;
    }
    if !tool_calls.is_empty() {
        if pushed_text {
            let last = out.last_mut().expect("text message pushed above");
            last["tool_calls"] = json!(tool_calls);
            if last["content"].as_str().is_some_and(str::is_empty) {
                last["content"] = Value::Null;
            }
        } else {
            // assistant message carrying only tool_use blocks: no text part,
            // but the tool calls must still ride an assistant message
            out.push(json!({"role": role, "content": Value::Null, "tool_calls": tool_calls}));
        }
    }
    out.extend(tool_results);
    if !thinking.is_empty() {
        // F47: attach preserved reasoning to the assistant message this
        // call produced (text-fold or tool-only push), never to a
        // trailing tool result.
        if let Some(slot) = out
            .iter()
            .rposition(|m| m.get("role").and_then(Value::as_str) == Some(role))
        {
            out[slot]["reasoning_content"] = json!(thinking.join(""));
        }
    }
    Ok(())
}

fn content_to_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .map(|b| b.get("text").and_then(Value::as_str).unwrap_or_default())
            .collect::<String>(),
        _ => String::new(),
    }
}

// ---------- response direction ----------

/// `OpenAI` chat response → Anthropic message shape.
pub fn translate_response(openai: &Value, model: &str) -> Value {
    let id = format!(
        "msg_{}",
        openai
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("chatcmpl")
            .trim_start_matches("chatcmpl-")
    );
    let choice = openai.pointer("/choices/0").cloned().unwrap_or(Value::Null);
    let msg = choice.get("message").cloned().unwrap_or(Value::Null);
    let mut content: Vec<Value> = Vec::new();
    // F47: engines exposing reasoning (DeepSeek-style `reasoning_content`)
    // surface it as a leading Anthropic `thinking` block, ahead of text.
    if let Some(rc) = msg.get("reasoning_content").and_then(Value::as_str) {
        if !rc.is_empty() {
            content.push(json!({"type": "thinking", "thinking": rc}));
        }
    }
    if let Some(text) = msg.get("content").and_then(Value::as_str) {
        if !text.is_empty() {
            content.push(json!({"type": "text", "text": text}));
        }
    }
    if let Some(calls) = msg.get("tool_calls").and_then(Value::as_array) {
        for c in calls {
            let input = c
                .pointer("/function/arguments")
                .and_then(Value::as_str)
                .and_then(|s| serde_json::from_str::<Value>(s).ok())
                .unwrap_or(json!({}));
            content.push(json!({
                "type": "tool_use",
                "id": c.get("id").and_then(Value::as_str).unwrap_or_default(),
                "name": c.pointer("/function/name").and_then(Value::as_str).unwrap_or_default(),
                "input": input
            }));
        }
    }
    if content.is_empty() {
        content.push(json!({"type": "text", "text": ""}));
    }
    let finish = choice
        .get("finish_reason")
        .and_then(Value::as_str)
        .unwrap_or("stop");
    let stop_reason = match finish {
        "length" => "max_tokens",
        "tool_calls" | "function_call" => "tool_use",
        "content_filter" => "refusal",
        _ => "end_turn",
    };
    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": Value::Null,
        "usage": {
            "input_tokens": openai.pointer("/usage/prompt_tokens").and_then(Value::as_u64).unwrap_or(0),
            "output_tokens": openai.pointer("/usage/completion_tokens").and_then(Value::as_u64).unwrap_or(0),
        }
    })
}

/// One `OpenAI` `SSE` chunk → zero or more Anthropic `SSE` events. The state
/// machine (which block indexes are open) lives in the stream wrapper.
// Chunk-to-SSE translation is one state machine over event kinds; splitting
// it scatters the per-state transitions. TODO(split-chunk-events): extract
// when the parallel gateway wave lands.
#[allow(clippy::too_many_lines)]
pub fn chunk_events(chunk: &Value, state: &mut StreamState) -> Vec<(String, Value)> {
    let mut events: Vec<(String, Value)> = Vec::new();
    if !state.started {
        state.started = true;
        let input = chunk
            .pointer("/usage/prompt_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        events.push((
            "message_start".into(),
            json!({"type": "message_start", "message": {
                "id": state.id, "type": "message", "role": "assistant",
                "model": state.model, "content": [],
                "stop_reason": Value::Null, "stop_sequence": Value::Null,
                "usage": {"input_tokens": input, "output_tokens": 1}
            }}),
        ));
    }
    let choice = chunk.pointer("/choices/0").cloned().unwrap_or(Value::Null);
    // F47: reasoning deltas ride a leading `thinking` block (Anthropic
    // orders thinking before text), same open-once semantics as text.
    if let Some(rc) = choice
        .pointer("/delta/reasoning_content")
        .and_then(Value::as_str)
    {
        if !rc.is_empty() {
            match state.thinking_idx {
                None => {
                    let i = state.output_blocks;
                    state.output_blocks += 1;
                    state.thinking_idx = Some(i);
                    events.push((
                        "content_block_start".into(),
                        json!({"type": "content_block_start", "index": i,
                               "content_block": {"type": "thinking", "thinking": ""}}),
                    ));
                    events.push((
                        "content_block_delta".into(),
                        json!({"type": "content_block_delta", "index": i,
                               "delta": {"type": "thinking_delta", "thinking": rc}}),
                    ));
                }
                Some(i) => events.push((
                    "content_block_delta".into(),
                    json!({"type": "content_block_delta", "index": i,
                           "delta": {"type": "thinking_delta", "thinking": rc}}),
                )),
            }
        }
    }
    if let Some(text) = choice.pointer("/delta/content").and_then(Value::as_str) {
        if !text.is_empty() {
            // F50: track the ACTUAL text block index — with tool-first
            // streams the text slot is not 0 and hardcoding it corrupted
            // the Anthropic indices.
            match state.text_idx {
                None => {
                    let i = state.output_blocks;
                    state.output_blocks += 1;
                    state.text_idx = Some(i);
                    events.push((
                        "content_block_start".into(),
                        json!({"type": "content_block_start", "index": i,
                               "content_block": {"type": "text", "text": ""}}),
                    ));
                    events.push(text_delta(i, text));
                }
                Some(i) => events.push(text_delta(i, text)),
            }
        }
    }
    if let Some(calls) = choice
        .pointer("/delta/tool_calls")
        .and_then(Value::as_array)
    {
        for c in calls {
            let oai_idx = c.get("index").and_then(Value::as_u64).unwrap_or(0);
            let slot = state.tool_slot(oai_idx, c, &mut events);
            if let Some(frag) = c.pointer("/function/arguments").and_then(Value::as_str) {
                if !frag.is_empty() {
                    events.push((
                        "content_block_delta".into(),
                        json!({"type": "content_block_delta", "index": slot,
                               "delta": {"type": "input_json_delta", "partial_json": frag}}),
                    ));
                }
            }
        }
    }
    let finish = choice.get("finish_reason").and_then(Value::as_str);
    if finish.is_some()
        || chunk
            .get("usage")
            .is_some_and(|u| u.pointer("/completion_tokens").is_some())
    {
        if state.thinking_idx.is_some() || state.text_idx.is_some() || state.open_tools > 0 {
            for i in 0..state.output_blocks {
                events.push((
                    "content_block_stop".into(),
                    json!({"type": "content_block_stop", "index": i}),
                ));
            }
            state.thinking_idx = None;
            state.text_idx = None;
            state.open_tools = 0;
        }
        let stop_reason = match finish.unwrap_or("stop") {
            "length" => "max_tokens",
            "tool_calls" | "function_call" => "tool_use",
            "content_filter" => "refusal",
            _ => "end_turn",
        };
        let output = chunk
            .pointer("/usage/completion_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        // F49: engines report usage on the FINAL chunk only — the
        // message_start snapshot stays 0, so correct input_tokens here
        // where Anthropic clients merge cumulative usage.
        let input = chunk
            .pointer("/usage/prompt_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let mut usage = json!({"output_tokens": output});
        if input > 0 {
            usage["input_tokens"] = json!(input);
        }
        events.push((
            "message_delta".into(),
            json!({"type": "message_delta",
                   "delta": {"stop_reason": stop_reason, "stop_sequence": Value::Null},
                   "usage": usage}),
        ));
        events.push(("message_stop".into(), json!({"type": "message_stop"})));
        state.done = true;
    }
    events
}

fn text_delta(index: u64, text: &str) -> (String, Value) {
    (
        "content_block_delta".into(),
        json!({"type": "content_block_delta", "index": index,
               "delta": {"type": "text_delta", "text": text}}),
    )
}

/// Per-response stream translation state.
#[derive(Default)]
pub struct StreamState {
    started: bool,
    done: bool,
    thinking_idx: Option<u64>,
    text_idx: Option<u64>,
    open_tools: u64,
    output_blocks: u64,
    tool_map: std::collections::HashMap<u64, u64>,
    id: String,
    model: String,
}

impl StreamState {
    #[must_use]
    pub fn new(id: &str, model: &str) -> Self {
        Self {
            id: id.to_string(),
            model: model.to_string(),
            ..Self::default()
        }
    }

    /// Map an `OpenAI` tool-call index to an Anthropic block index,
    /// opening a `tool_use` block on first sight.
    fn tool_slot(&mut self, oai_idx: u64, call: &Value, events: &mut Vec<(String, Value)>) -> u64 {
        if let Some(slot) = self.tool_map.get(&oai_idx) {
            return *slot;
        }
        let slot = self.output_blocks;
        self.output_blocks += 1;
        self.open_tools += 1;
        self.tool_map.insert(oai_idx, slot);
        events.push((
            "content_block_start".into(),
            json!({"type": "content_block_start", "index": slot,
                   "content_block": {"type": "tool_use",
                        "id": call.get("id").and_then(Value::as_str).unwrap_or_default(),
                        "name": call.pointer("/function/name").and_then(Value::as_str).unwrap_or_default(),
                        "input": {}}}),
        ));
        slot
    }
}

/// Child `OpenAI` `SSE` → Anthropic event `SSE`. The in-flight guard rides in
/// the stream state so accounting holds until the last frame ships.
fn anthropic_sse_stream<S>(
    child: S,
    st: StreamState,
    guard: crate::proxy::InFlightGuard,
) -> impl futures::Stream<Item = Result<Bytes, std::io::Error>>
where
    S: futures::Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    use std::collections::VecDeque;
    futures::stream::unfold(
        (
            child,
            st,
            String::new(),
            crate::translate::LineBuffer::new(),
            VecDeque::new(),
            Some(guard),
        ),
        |(mut child, mut st, mut buf, mut lines, mut queue, guard)| async move {
            loop {
                if let Some(frame) = queue.pop_front() {
                    return Some((Ok(frame), (child, st, buf, lines, queue, guard)));
                }
                match child.next().await {
                    Some(Ok(chunk)) => {
                        buf.push_str(&lines.feed(&chunk));
                        while let Some(pos) = buf.find('\n') {
                            let line: String = buf.drain(..=pos).collect();
                            let line = line.trim();
                            let Some(payload) = line.strip_prefix("data: ") else {
                                continue;
                            };
                            if payload == "[DONE]" {
                                continue;
                            }
                            let Ok(v) = serde_json::from_str::<Value>(payload) else {
                                continue;
                            };
                            for (kind, data) in chunk_events(&v, &mut st) {
                                queue.push_back(Bytes::from(format!(
                                    "event: {kind}\ndata: {data}\n\n"
                                )));
                            }
                        }
                    }
                    Some(Err(e)) => {
                        let frame = Bytes::from(format!(
                            "event: error\ndata: {}\n\n",
                            json!({"type": "error", "error": {"type": "api_error", "message": e.to_string()}})
                        ));
                        queue.push_back(frame);
                    }
                    None => {
                        // F51: child ended without a finish chunk — emit a
                        // terminal error event so clients see the truncation
                        // instead of a silent bare EOF.
                        if !st.done {
                            st.done = true;
                            queue.push_back(Bytes::from(format!(
                                "event: error\ndata: {}\n\n",
                                json!({"type": "error", "error": {"type": "api_error",
                                       "message": "upstream stream ended before completion"}})
                            )));
                            continue;
                        }
                        return None;
                    }
                }
            }
        },
    )
}

fn anthropic_error(status: u16, err_type: &str, message: &str) -> Response {
    let body = json!({"type": "error", "error": {"type": err_type, "message": message}});
    Response::builder()
        .status(StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR))
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap_or_else(|_| Response::new(Body::from("{}")))
}

fn unique_suffix() -> String {
    // cheap uniqueness: no uuid dep in gateway — timestamp + counter
    use std::sync::atomic::{AtomicU64, Ordering};
    static CTR: AtomicU64 = AtomicU64::new(0);
    let n = CTR.fetch_add(1, Ordering::Relaxed);
    format!(
        "{:x}-{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos()),
        n
    )
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    fn anthropic_req() -> Value {
        serde_json::from_str(
            r#"{
              "model": "m1", "max_tokens": 64, "stream": false,
              "system": "be terse",
              "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "sure"},
                    {"type": "tool_use", "id": "t1", "name": "get_weather",
                     "input": {"city": "sf"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1",
                     "content": [{"type": "text", "text": "72F"}]}
                ]}
              ],
              "tools": [{"name": "get_weather", "description": "w",
                         "input_schema": {"type": "object"}}],
              "tool_choice": {"type": "any"}
            }"#,
        )
        .expect("fixture")
    }

    #[test]
    fn unit__translate_request__system_messages_tools_tool_results() {
        let out = translate_request(&anthropic_req(), false).expect("ok");
        let msgs = out["messages"].as_array().expect("msgs");
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "be terse");
        assert_eq!(msgs[1]["content"], "hi");
        let assistant = &msgs[2];
        assert_eq!(assistant["role"], "assistant");
        let calls = assistant["tool_calls"].as_array().expect("calls");
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(calls[0]["function"]["arguments"], r#"{"city":"sf"}"#);
        assert_eq!(msgs[3]["role"], "tool");
        assert_eq!(msgs[3]["tool_call_id"], "t1");
        assert_eq!(msgs[3]["content"], "72F");
        let tools = out["tools"].as_array().expect("tools");
        assert_eq!(tools[0]["function"]["name"], "get_weather");
        assert_eq!(out["tool_choice"], "required");
        assert_eq!(out["max_tokens"], 64);
        assert_eq!(out["stream"], false);
    }

    #[test]
    fn unit__translate_request__stop_sequences_and_stream_options() {
        let mut v = anthropic_req();
        v["stop_sequences"] = json!(["END"]);
        let out = translate_request(&v, true).expect("ok");
        assert_eq!(out["stop"], json!(["END"]));
        assert_eq!(out["stream"], true);
        assert_eq!(out["stream_options"]["include_usage"], true);
    }

    #[test]
    fn unit__translate_response__text_and_usage() {
        let openai = serde_json::from_str(
            r#"{"id": "chatcmpl-abc", "choices": [{"index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 7, "completion_tokens": 3}}"#,
        )
        .expect("fixture");
        let out = translate_response(&openai, "m1");
        assert_eq!(out["type"], "message");
        assert_eq!(out["id"], "msg_abc");
        assert_eq!(out["content"][0]["text"], "hello");
        assert_eq!(out["stop_reason"], "end_turn");
        assert_eq!(out["usage"]["input_tokens"], 7);
        assert_eq!(out["usage"]["output_tokens"], 3);
    }

    #[test]
    fn unit__translate_response__tool_use_blocks() {
        let openai = serde_json::from_str(
            r#"{"id": "x", "choices": [{"index": 0,
                "message": {"role": "assistant", "content": null,
                  "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "f", "arguments": "{\"a\":1}"}}]},
                "finish_reason": "tool_calls"}],
               "usage": {"prompt_tokens": 1, "completion_tokens": 2}}"#,
        )
        .expect("fixture");
        let out = translate_response(&openai, "m1");
        assert_eq!(out["content"][0]["type"], "tool_use");
        assert_eq!(out["content"][0]["input"]["a"], 1);
        assert_eq!(out["stop_reason"], "tool_use");
    }

    #[test]
    fn unit__chunk_events__full_sse_sequence() {
        let mut st = StreamState::new("msg_t", "m1");
        let c1: Value = serde_json::from_str(
            r#"{"choices": [{"index": 0, "delta": {"content": "he"}, "finish_reason": null}]}"#,
        )
        .unwrap();
        let ev = chunk_events(&c1, &mut st);
        assert_eq!(ev[0].0, "message_start");
        assert_eq!(ev[1].0, "content_block_start");
        assert_eq!(ev[2].1["delta"]["text"], "he");
        let c2 = json!({
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [
                    {"index": 0, "id": "call_1", "function": {"name": "f", "arguments": "{\"a\""}}
                ]},
                "finish_reason": null
            }]
        });
        let ev = chunk_events(&c2, &mut st);
        assert_eq!(ev[0].0, "content_block_start");
        assert_eq!(ev[0].1["content_block"]["type"], "tool_use");
        assert_eq!(ev[1].1["delta"]["partial_json"], "{\"a\"");
        let c3: Value = serde_json::from_str(
            r#"{"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
               "usage": {"prompt_tokens": 5, "completion_tokens": 4}}"#,
        )
        .unwrap();
        let ev = chunk_events(&c3, &mut st);
        let kinds: Vec<&str> = ev.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(
            kinds,
            [
                "content_block_stop",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
        assert_eq!(ev[2].1["delta"]["stop_reason"], "tool_use");
        assert_eq!(ev[2].1["usage"]["output_tokens"], 4);
        assert!(st.done);
    }

    #[test]
    fn unit__chunk_events__final_chunk_input_tokens_corrected() {
        // F49: usage arrives only on the final chunk — message_start saw
        // 0; the closing message_delta must carry the real prompt_tokens.
        let mut st = StreamState::new("msg_t", "m1");
        let c1: Value = serde_json::from_str(
            r#"{"choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": null}]}"#,
        )
        .unwrap();
        let _ = chunk_events(&c1, &mut st);
        assert!(st.started);
        let c2: Value = serde_json::from_str(
            r#"{"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 77, "completion_tokens": 3}}"#,
        )
        .unwrap();
        let ev = chunk_events(&c2, &mut st);
        let delta = ev
            .iter()
            .find(|(k, _)| k == "message_delta")
            .map(|(_, v)| v.clone())
            .unwrap();
        assert_eq!(delta["usage"]["input_tokens"], 77, "{delta}");
        assert_eq!(delta["usage"]["output_tokens"], 3, "{delta}");
    }

    #[test]
    fn unit__chunk_events__tool_first_text_uses_real_index() {
        // F50: with tool calls opening first, text arrives at block 1 —
        // deltas must reference index 1, not the old hardcoded 0.
        let mut st = StreamState::new("msg_t", "m1");
        let c1 = json!({
            "choices": [{
                "index": 0,
                "delta": {"tool_calls": [
                    {"index": 0, "id": "call_1", "function": {"name": "f", "arguments": ""}}
                ]},
                "finish_reason": null
            }]
        });
        let _ = chunk_events(&c1, &mut st);
        let c2: Value = serde_json::from_str(
            r#"{"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": null}]}"#,
        )
        .unwrap();
        let ev = chunk_events(&c2, &mut st);
        let start = ev
            .iter()
            .find(|(k, _)| k == "content_block_start")
            .map(|(_, v)| v.clone())
            .unwrap();
        assert_eq!(start["index"], 1, "{start}");
        assert_eq!(start["content_block"]["type"], "text", "{start}");
        let text_delta = ev
            .iter()
            .find(|(k, v)| k == "content_block_delta" && v["delta"]["type"] == "text_delta")
            .map(|(_, v)| v.clone())
            .unwrap();
        assert_eq!(text_delta["index"], 1, "{text_delta}");
    }

    #[test]
    fn unit__chunk_events__reasoning_delta_opens_thinking_block() {
        // F47 stream direction: reasoning_content rides a leading
        // thinking block with thinking_delta frames.
        let mut st = StreamState::new("msg_t", "m1");
        let c1: Value = serde_json::from_str(
            r#"{"choices": [{"index": 0, "delta": {"reasoning_content": "hm"}, "finish_reason": null}]}"#,
        )
        .unwrap();
        let ev = chunk_events(&c1, &mut st);
        assert_eq!(ev[1].1["content_block"]["type"], "thinking", "{ev:?}");
        assert_eq!(ev[2].1["delta"]["thinking"], "hm", "{ev:?}");
        assert_eq!(ev[2].1["index"], 0);
        let c2: Value = serde_json::from_str(
            r#"{"choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": null}]}"#,
        )
        .unwrap();
        let ev = chunk_events(&c2, &mut st);
        let text_start = ev
            .iter()
            .find(|(k, _)| k == "content_block_start")
            .map(|(_, v)| v.clone())
            .unwrap();
        assert_eq!(text_start["index"], 1, "text after thinking");
    }

    #[test]
    fn unit__translate_response__reasoning_becomes_leading_thinking_block() {
        // F47 non-stream direction.
        let openai = json!({
            "choices": [{"index": 0, "message": {
                "role": "assistant",
                "reasoning_content": "ponder",
                "content": "final",
                "tool_calls": null
            }, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2}
        });
        let out = translate_response(&openai, "m1");
        let blocks = out["content"].as_array().unwrap();
        assert_eq!(blocks[0]["type"], "thinking");
        assert_eq!(blocks[0]["thinking"], "ponder");
        assert_eq!(blocks[1]["type"], "text");
        assert_eq!(blocks[1]["text"], "final");
    }

    #[test]
    fn unit__translate_message__thinking_preserved_as_reasoning_content() {
        // F47 request direction: assistant prefill thinking survives as
        // reasoning_content on the translated OpenAI message.
        let mut out: Vec<Value> = Vec::new();
        let content = json!([
            {"type": "thinking", "thinking": "prior thought"},
            {"type": "text", "text": "answer"},
        ]);
        translate_message("assistant", &content, &mut out).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["content"], "answer");
        assert_eq!(out[0]["reasoning_content"], "prior thought");
        // Tool-only assistant message still receives the field.
        let mut out: Vec<Value> = Vec::new();
        let content = json!([
            {"type": "thinking", "thinking": "t1"},
            {"type": "tool_use", "id": "c1", "name": "f", "input": {}},
        ]);
        translate_message("assistant", &content, &mut out).unwrap();
        assert_eq!(out[0]["reasoning_content"], "t1");
        assert_eq!(out[0]["tool_calls"].as_array().unwrap().len(), 1);
    }
}
