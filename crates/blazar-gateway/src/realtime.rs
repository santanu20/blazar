//! Realtime voice lane: one WebSocket per session, one utterance pipeline
//! per committed buffer - STT (whisper lane) -> chat (`OpenAI` lane) -> TTS
//! (piper lane). Every stage calls the gateway's own handlers, so admission,
//! queueing, and telemetry apply exactly as on the HTTP lanes.
//!
//! Event surface (documented in `docs/4.API_SPEC.md`):
//!
//! ```text
//! client -> server : {"type":"input_audio_buffer.append","audio":"<b64>"}
//! client -> server : {"type":"input_audio_buffer.commit"}
//! client -> server : {"type":"input_audio_buffer.clear"}
//! client -> server : {"type":"session.update","session":{voice?,stt_model?,instructions?,tools?,tool_choice?}}
//! client -> server : {"type":"response.create"}
//! client -> server : {"type":"response.cancel"}
//! client -> server : {"type":"conversation.item.create","item":{...}}
//! server -> client : {"type":"session.created","session":{...}}
//! server -> client : {"type":"session.updated","session":{...}}
//! server -> client : {"type":"input_audio_transcription.completed",...}
//! server -> client : {"type":"input_audio_transcription.failed","error":{...}}
//! server -> client : {"type":"response.audio_transcript.delta"/".done",...}
//! server -> client : {"type":"response.audio.delta"/".done",...}
//! server -> client : {"type":"response.output_item.added","item":{type:"function_call",...}}
//! server -> client : {"type":"response.function_call_arguments.done",...}
//! server -> client : {"type":"response.done","response":{"output":[...]}}
//! server -> client : {"type":"response.cancelled",...}
//! server -> client : {"type":"input_audio_buffer.cleared",...}
//! server -> client : {"type":"conversation.item.created","item":{...}}
//! server -> client : {"type":"error","error":{"type","message"}}
//! ```
//!
//! Audio is PCM16LE mono 24 kHz (the `OpenAI` realtime convention); the
//! lane wraps it into a WAV container before handing it to whisper.
//! Model selection rides `?model=`; the piper voice rides `?voice=`
//! (default `en_US-amy-medium`); the whisper size rides `?stt_model=`
//! (default `whisper-1`, resolved by the whisper lane's own ladder).
//!
//! Semantics worth knowing:
//! - `session.update` mutates voice / `stt_model` / instructions /
//!   `tools` / `tool_choice` and they apply from the NEXT turn on (a running
//!   turn snapshots the config).
//! - `response.cancel` sets a flag checked BETWEEN pipeline stages, so a
//!   cancel lands after the current stage finishes (chat streams are not
//!   cut mid-flight - each stage call is atomic here). A cancel that
//!   arrives too late (turn already finished) simply has no effect.
//! - Function calling: `session.update` tools are converted to the chat
//!   dialect and attached to every chat call. When the model answers with
//!   `tool_calls`, the turn emits `response.output_item.added` +
//!   `response.function_call_arguments.done` per call and one
//!   `response.done` carrying the `function_call` items; the calls and the
//!   trigger transcript persist into the conversation context. The client
//!   returns outputs via `conversation.item.create` items of type
//!   `function_call_output`, then `response.create` (empty buffer) runs a
//!   continuation turn on the remembered context - no new audio needed.
//!   A `response.create` while calls are still outstanding is a taught
//!   error; a new VOICE turn that interrupts pending calls auto-injects
//!   error tool outputs so the model sees a completed (failed) round.
//! - `conversation.item.create` accepts text messages, client-authored
//!   `function_call` items, and `function_call_output` items; other item
//!   types are taught. Context is capped at 16 items / 8 192 chars,
//!   oldest evicted first. Items created mid-turn apply to the next turn
//!   (tool outputs mid-turn are a taught race - send them after
//!   `response.done`).
//! - While a turn runs, only `response.cancel`, `input_audio_buffer.append`,
//!   and `conversation.item.create` are accepted; anything else gets an
//!   error event.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Extension, Query, State};
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::Response;
use futures::SinkExt;
use futures::StreamExt;
use futures::stream::{SplitSink, SplitStream};
use serde_json::{Value, json};

use crate::state::AppState;

/// No frame for this long closes the session (client vanished mid-phrase).
const IDLE_TIMEOUT_SECS: u64 = 120;
/// Hard session ceiling - a realtime socket is not a forever connection.
const SESSION_MAX_SECS: u64 = 600;
/// Append-buffer cap: beyond this the client is misbehaving (or not a
/// voice client at all) - error event, then close.
const AUDIO_BUFFER_CAP_BYTES: usize = 20 * 1024 * 1024;
/// TTS audio rides ~4 KiB base64 deltas.
const TTS_CHUNK_BYTES: usize = 4 * 1024;
/// Utterance budget for the reply text.
const CHAT_MAX_TOKENS: u64 = 512;
/// Conversation-context ceiling: at most this many remembered items ...
const CHAT_CTX_MAX_ITEMS: usize = 16;
/// ... and this many characters of item payload (oldest evicted first).
const CHAT_CTX_MAX_CHARS: usize = 8_192;
/// Session tool ceiling: count ...
const TOOLS_MAX_COUNT: usize = 16;
/// ... and serialized size of the converted chat-shape array.
const TOOLS_MAX_BYTES: usize = 32 * 1024;
const PCM_SAMPLE_RATE: u32 = 24_000;
const DEFAULT_VOICE: &str = "en_US-amy-medium";

/// One remembered conversation item. Text items are the historical
/// voice-turn shape; tool items ride the function-calling loop and map
/// onto the chat dialect (`tool_calls` / `role:"tool"`) at request time.
#[derive(Debug, Clone, PartialEq)]
enum CtxItem {
    Text {
        role: String,
        text: String,
    },
    ToolCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    ToolOutput {
        call_id: String,
        content: String,
    },
}

impl CtxItem {
    /// Payload size for the character cap (roles/ids stay out of the
    /// budget, matching the historical text-only accounting).
    fn chars(&self) -> usize {
        match self {
            CtxItem::Text { text, .. } => text.len(),
            CtxItem::ToolCall {
                name, arguments, ..
            } => name.len() + arguments.len(),
            CtxItem::ToolOutput { content, .. } => content.len(),
        }
    }
}

/// A function call the model issued and the client has not answered yet.
#[derive(Debug, Clone)]
struct PendingCall {
    call_id: String,
    name: String,
    arguments: String,
}

/// What a finished turn hands back to the session loop: the trigger
/// transcript to persist (voice tool rounds only) and any function calls
/// awaiting outputs.
struct TurnEnd {
    transcript: Option<String>,
    pending: Vec<PendingCall>,
}

impl TurnEnd {
    fn done() -> Self {
        Self {
            transcript: None,
            pending: Vec::new(),
        }
    }
}

/// Mutable per-session configuration (voice, stt size, system prompt,
/// tools). Snapshot-cloned at each turn start so `session.update`
/// mid-turn can only affect the next turn.
#[derive(Debug, Clone)]
struct SessionCfg {
    voice: String,
    stt_model: String,
    instructions: String,
    /// Function tools in CHAT shape (nested `function` object) - the
    /// dialect the local chat lane forwards verbatim.
    tools: Vec<Value>,
    /// `tool_choice` passthrough (`auto`/`none`/`required`/named fn).
    tool_choice: Option<Value>,
}

impl SessionCfg {
    fn new(voice: String, stt_model: String) -> Self {
        Self {
            voice,
            stt_model,
            instructions: String::new(),
            tools: Vec::new(),
            tool_choice: None,
        }
    }
}

/// The session object echoed by `session.created` / `session.updated`.
fn session_obj(cfg: &SessionCfg) -> Value {
    json!({
        "voice": cfg.voice,
        "stt_model": cfg.stt_model,
        "instructions": cfg.instructions,
        "tools": cfg.tools,
        "tool_choice": cfg.tool_choice,
    })
}

/// Convert one realtime tool (flat: `type`/`name`/`description`/
/// `parameters`) into the chat dialect (nested `function` object).
/// Non-function tools are rejected - the local chat lane speaks
/// `type:"function"` only.
fn tool_to_chat_shape(tool: &Value) -> Result<Value, String> {
    if tool.get("type").and_then(Value::as_str) != Some("function") {
        return Err(format!(
            "tool type '{}' is not supported - this lane forwards type:\"function\" tools only",
            tool.get("type").and_then(Value::as_str).unwrap_or("?")
        ));
    }
    let name = tool
        .get("name")
        .and_then(Value::as_str)
        .filter(|n| !n.trim().is_empty())
        .ok_or("function tool needs a non-empty `name`")?;
    let mut function = json!({"name": name});
    if let Some(d) = tool.get("description") {
        match d.as_str() {
            Some(s) => function["description"] = json!(s),
            None => return Err("`description` must be a string".into()),
        }
    }
    if let Some(p) = tool.get("parameters") {
        if !p.is_object() {
            return Err("`parameters` must be a JSON object (JSON-schema shape)".into());
        }
        function["parameters"] = p.clone();
    }
    Ok(json!({"type": "function", "function": function}))
}

/// Validate a `session.update` `tools` array: bounded count and size,
/// every entry converted to chat shape up front so turn time just
/// attaches the array.
fn validate_tools(raw: &Value) -> Result<Vec<Value>, String> {
    let arr = raw
        .as_array()
        .ok_or_else(|| "`tools` must be an array of function tool objects".to_string())?;
    if arr.len() > TOOLS_MAX_COUNT {
        return Err(format!(
            "`tools` holds {} entries - the session ceiling is {TOOLS_MAX_COUNT}",
            arr.len()
        ));
    }
    let mut out = Vec::with_capacity(arr.len());
    for tool in arr {
        out.push(tool_to_chat_shape(tool)?);
    }
    let serialized = serde_json::to_string(&out).unwrap_or_default();
    if serialized.len() > TOOLS_MAX_BYTES {
        return Err(format!(
            "serialized `tools` exceed {TOOLS_MAX_BYTES} bytes - trim descriptions/schemas"
        ));
    }
    Ok(out)
}

/// Validate `tool_choice`: null clears it; the three string modes and
/// the named-function object pass through; anything else is taught.
fn validate_tool_choice(raw: &Value) -> Result<Option<Value>, String> {
    match raw {
        Value::Null => Ok(None),
        Value::String(s)
            if matches!(s.as_str(), "auto" | "none" | "required") =>
        {
            Ok(Some(json!(s)))
        }
        Value::Object(_) if {
            raw.get("type").and_then(Value::as_str) == Some("function")
                && raw
                    .get("name")
                    .and_then(Value::as_str)
                    .is_some_and(|n| !n.trim().is_empty())
        } =>
        {
            Ok(Some(raw.clone()))
        }
        _ => Err(
            "`tool_choice` must be \"auto\" | \"none\" | \"required\" | {\"type\":\"function\",\"name\":...} | null"
                .into(),
        ),
    }
}

/// Apply a `session.update` payload to the config; returns the session
/// object to echo back in `session.updated`. Unknown session keys are
/// tolerated and ignored (`OpenAI` clients send many we do not consume);
/// known keys are validated.
fn apply_session_update(cfg: &mut SessionCfg, ev: &Value) -> Result<Value, String> {
    let Some(session) = ev.get("session") else {
        return Err("session.update needs a `session` object".into());
    };
    if let Some(v) = session.get("voice") {
        match v.as_str() {
            Some(s) if !s.trim().is_empty() => cfg.voice = s.trim().to_string(),
            _ => return Err("`voice` must be a non-empty string".into()),
        }
    }
    if let Some(m) = session.get("stt_model") {
        match m.as_str() {
            Some(s) if !s.trim().is_empty() => cfg.stt_model = s.trim().to_string(),
            _ => return Err("`stt_model` must be a non-empty string".into()),
        }
    }
    if let Some(i) = session.get("instructions") {
        match i.as_str() {
            Some(s) => cfg.instructions = s.to_string(),
            None => return Err("`instructions` must be a string".into()),
        }
    }
    if let Some(t) = session.get("tools") {
        cfg.tools = validate_tools(t)?;
    }
    if let Some(tc) = session.get("tool_choice") {
        cfg.tool_choice = validate_tool_choice(tc)?;
    }
    Ok(session_obj(cfg))
}

/// Extract `(role, text)` from a `conversation.item.create` message item.
/// Only text messages are consumable by the local chat lane directly;
/// tool items have their own intake (`item_tool_output` /
/// `item_client_tool_call`).
fn item_text(item: &Value) -> Result<(String, String), String> {
    let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
    let content = item.pointer("/content/0").ok_or(
        "item needs content[0] with type \"text\" (the local chat lane consumes text context only)",
    )?;
    if content.get("type").and_then(Value::as_str) != Some("text") {
        return Err(format!(
            "item content type '{}' is not supported - this lane consumes text context only",
            content.get("type").and_then(Value::as_str).unwrap_or("?")
        ));
    }
    let text = content
        .get("text")
        .and_then(Value::as_str)
        .ok_or("text content needs a `text` field")?;
    Ok((role.to_string(), text.to_string()))
}

/// Parse a `function_call_output` item: `call_id` names the call being
/// answered, `output` is a string or any JSON value (serialized to text
/// so the chat lane's `role:"tool"` content stays a plain string).
fn item_tool_output(item: &Value) -> Result<CtxItem, String> {
    let call_id = item
        .get("call_id")
        .and_then(Value::as_str)
        .filter(|c| !c.trim().is_empty())
        .ok_or("function_call_output needs a non-empty `call_id`")?;
    let content = match item.get("output") {
        Some(Value::String(s)) => s.clone(),
        Some(v @ (Value::Object(_) | Value::Array(_))) => {
            serde_json::to_string(v).unwrap_or_else(|_| v.to_string())
        }
        Some(Value::Number(n)) => n.to_string(),
        Some(Value::Bool(b)) => b.to_string(),
        Some(Value::Null) | None => {
            return Err(
                "function_call_output needs an `output` field (string or JSON value)".into(),
            );
        }
    };
    Ok(CtxItem::ToolOutput {
        call_id: call_id.to_string(),
        content,
    })
}

/// Parse a client-authored `function_call` item (history replay or a
/// harness injecting a call the model never made).
fn item_client_tool_call(item: &Value) -> Result<CtxItem, String> {
    let call_id = item
        .get("call_id")
        .and_then(Value::as_str)
        .filter(|c| !c.trim().is_empty())
        .ok_or("function_call item needs a non-empty `call_id`")?;
    let name = item
        .get("name")
        .and_then(Value::as_str)
        .filter(|n| !n.trim().is_empty())
        .ok_or("function_call item needs a non-empty `name`")?;
    let arguments = item
        .get("arguments")
        .and_then(Value::as_str)
        .unwrap_or("{}")
        .to_string();
    Ok(CtxItem::ToolCall {
        call_id: call_id.to_string(),
        name: name.to_string(),
        arguments,
    })
}

/// Push one item into the conversation context, enforcing both caps by
/// evicting oldest-first.
fn ctx_push(ctx: &mut Vec<CtxItem>, item: CtxItem) {
    ctx.push(item);
    let mut chars: usize = ctx.iter().map(CtxItem::chars).sum();
    while (ctx.len() > CHAT_CTX_MAX_ITEMS || chars > CHAT_CTX_MAX_CHARS) && !ctx.is_empty() {
        chars = chars.saturating_sub(ctx[0].chars());
        ctx.remove(0);
    }
}

/// Build the chat-completion `messages` array: optional system
/// instructions, remembered context items (text, tool calls, tool
/// outputs in chat dialect), then - for voice turns - the live
/// transcript as the final user message. Continuation turns pass
/// `None` and end on whatever the context holds.
fn ctx_messages(cfg: &SessionCfg, ctx: &[CtxItem], transcript: Option<&str>) -> Value {
    let mut messages = Vec::with_capacity(ctx.len() + 2);
    if !cfg.instructions.trim().is_empty() {
        messages.push(json!({"role": "system", "content": cfg.instructions}));
    }
    for item in ctx {
        match item {
            CtxItem::Text { role, text } => messages.push(json!({"role": role, "content": text})),
            CtxItem::ToolCall {
                call_id,
                name,
                arguments,
            } => messages.push(json!({
                "role": "assistant",
                "content": Value::Null,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }],
            })),
            CtxItem::ToolOutput { call_id, content } => messages.push(json!({
                "role": "tool",
                "tool_call_id": call_id,
                "content": content,
            })),
        }
    }
    if let Some(t) = transcript {
        messages.push(json!({"role": "user", "content": t}));
    }
    Value::Array(messages)
}

/// `GET /v1/realtime?model=...` - upgrade, then run the session loop.
#[allow(clippy::too_many_lines, clippy::implicit_hasher)] // one cohesive upgrade path; bind_remote precedent
pub async fn realtime_session(
    State(state): State<Arc<AppState>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    Query(params): Query<HashMap<String, String>>,
    ws: WebSocketUpgrade,
) -> Response {
    let Some(model) = params.get("model").cloned() else {
        return crate::proxy::openai_error(
            400,
            "missing ?model= - the realtime lane needs the chat model id as a query \
             parameter (e.g. /v1/realtime?model=qwen3-0.6b:q4_0)",
        );
    };
    let voice = params
        .get("voice")
        .cloned()
        .unwrap_or_else(|| DEFAULT_VOICE.into());
    let stt_model = params
        .get("stt_model")
        .cloned()
        .unwrap_or_else(|| "whisper-1".into());
    let key = key_ext.map(|axum::Extension(k)| k);
    ws.on_upgrade(move |socket| run_session(socket, state, model, voice, stt_model, key))
}
/// Emit one JSON text frame; failures (client gone) are quietly the end
/// of the session - the read loop notices the closed socket next. The
/// sink is shared between the turn future and the event loop, so it
/// rides an async mutex (short sends; held only per frame).
type WsSink = Arc<tokio::sync::Mutex<SplitSink<WebSocket, Message>>>;

async fn send_json(sink: &WsSink, v: &Value) {
    let mut guard = sink.lock().await;
    let _ = guard
        .send(Message::Text(
            serde_json::to_string(v).unwrap_or_default().into(),
        ))
        .await;
}

/// Close the shared sink (session teardown paths).
async fn sink_close(sink: &WsSink) {
    let _ = sink.lock().await.close().await;
}

fn error_event(etype: &str, message: &str) -> Value {
    json!({"type": "error", "error": {"type": etype, "message": message}})
}

/// Wrap raw PCM16LE mono bytes in a 44-byte RIFF/WAV header so the
/// whisper lane receives a container it accepts.
fn wav_wrap(pcm: &[u8]) -> Vec<u8> {
    let data_len = u32::try_from(pcm.len()).unwrap_or(u32::MAX);
    let byte_rate = PCM_SAMPLE_RATE * 2; // 16-bit mono
    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&36u32.saturating_add(data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&PCM_SAMPLE_RATE.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    out.extend_from_slice(pcm);
    out
}

/// Build the multipart body the whisper handler expects (file part with a
/// filename, model as a plain field).
fn transcription_multipart(wav: &[u8], stt_model: &str) -> (String, Vec<u8>) {
    let boundary = "blazarRealtimeAudio7f3b";
    let mut body = Vec::with_capacity(wav.len() + 256);
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{stt_model}\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; \
             filename=\"utterance.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(wav);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

/// Extract the assistant text from an `OpenAI` chat-completion body.
fn reply_text(chat: &Value) -> String {
    chat.pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// Extract well-formed function calls from a chat-completion body.
/// Entries missing an id or a function name are skipped (malformed
/// child output must not corrupt the calling loop); an absent array is
/// simply "no calls".
fn reply_tool_calls(chat: &Value) -> Vec<PendingCall> {
    chat.pointer("/choices/0/message/tool_calls")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|c| {
                    let call_id = c.get("id")?.as_str()?.to_string();
                    let name = c.pointer("/function/name")?.as_str()?.to_string();
                    let arguments = c
                        .pointer("/function/arguments")
                        .and_then(Value::as_str)
                        .unwrap_or("{}")
                        .to_string();
                    Some(PendingCall {
                        call_id,
                        name,
                        arguments,
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Emit the function-call event triad for one tool round: per call
/// `response.output_item.added` + `response.function_call_arguments.done`,
/// then one `response.done` whose `response.output` carries every
/// `function_call` item. Returns the emitted items (for callers that need
/// the shapes downstream).
async fn emit_tool_round(socket: &WsSink, calls: &[PendingCall], seq: &mut u64) -> Vec<Value> {
    let mut items = Vec::with_capacity(calls.len());
    for (i, c) in calls.iter().enumerate() {
        *seq += 1;
        let item = json!({
            "id": format!("fc_{seq}"),
            "type": "function_call",
            "call_id": c.call_id,
            "name": c.name,
            "arguments": c.arguments,
        });
        send_json(
            socket,
            &json!({
                "type": "response.output_item.added",
                "output_index": i,
                "item": item,
            }),
        )
        .await;
        send_json(
            socket,
            &json!({
                "type": "response.function_call_arguments.done",
                "call_id": c.call_id,
                "name": c.name,
                "arguments": c.arguments,
            }),
        )
        .await;
        items.push(item);
    }
    send_json(
        socket,
        &json!({
            "type": "response.done",
            "response": {"id": crate::batch::short_id("resp"), "output": items},
        }),
    )
    .await;
    items
}

/// Call the gateway's own chat-completion lane (admission, queueing,
/// child routing) and drain it to `(status, body)`.
async fn chat_call(
    state: &Arc<AppState>,
    key: Option<&crate::keys::KeyCtx>,
    model: &str,
    cfg: &SessionCfg,
    messages: Value,
) -> (StatusCode, axum::body::Bytes) {
    let mut body = json!({
        "model": model,
        "messages": messages,
        "max_tokens": CHAT_MAX_TOKENS,
        "stream": false,
    });
    if !cfg.tools.is_empty() {
        body["tools"] = json!(cfg.tools);
        if let Some(tc) = &cfg.tool_choice {
            body["tool_choice"] = tc.clone();
        }
    }
    let chat_body = serde_json::to_vec(&body).unwrap_or_default();
    let mut headers = HeaderMap::new();
    headers.insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("application/json"),
    );
    let chat = crate::openai::openai_proxy(
        State(state.clone()),
        None,
        key.cloned().map(axum::Extension),
        Uri::from_static("/v1/chat/completions"),
        Method::POST,
        headers,
        axum::body::Bytes::from(chat_body),
    )
    .await;
    response_parts(chat).await
}

/// Synthesize and stream the reply audio (piper lane); failure degrades
/// to text-only with an explicit error event (the transcript was already
/// delivered by the caller).
async fn speak_reply(
    socket: &WsSink,
    state: &Arc<AppState>,
    key: Option<&crate::keys::KeyCtx>,
    cfg: &SessionCfg,
    text: &str,
) {
    let tts_body = serde_json::to_vec(&json!({
        "model": cfg.voice,
        "input": text,
        "response_format": "wav",
    }))
    .unwrap_or_default();
    let mut tts_headers = HeaderMap::new();
    tts_headers.insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("application/json"),
    );
    let tts = crate::tts::audio_speech(
        State(state.clone()),
        key.cloned().map(axum::Extension),
        Uri::from_static("/v1/audio/speech"),
        Method::POST,
        tts_headers,
        axum::body::Bytes::from(tts_body),
    )
    .await;
    let (tts_status, tts_bytes) = response_parts(tts).await;
    if !tts_status.is_success() {
        send_json(
            socket,
            &error_event(
                "tts_failed",
                &format!(
                    "piper lane returned {} (voice '{}') - transcript delivered above, audio skipped",
                    tts_status.as_u16(),
                    cfg.voice
                ),
            ),
        )
        .await;
        return;
    }
    for chunk in tts_bytes.chunks(TTS_CHUNK_BYTES) {
        let b64 = base64_encode(chunk);
        send_json(
            socket,
            &json!({"type": "response.audio.delta", "delta": b64}),
        )
        .await;
    }
    send_json(socket, &json!({"type": "response.audio.done"})).await;
}

/// One committed utterance: STT -> chat -> TTS, each event streamed as it
/// lands. A stage failure emits an error event and returns - the socket
/// stays usable for the next utterance. `cancel` is checked between
/// stages (chat streams are not cut mid-flight); observing it emits
/// `response.cancelled` and stops the pipeline. When the model answers
/// with tool calls the TTS stage is skipped: the function-call triad is
/// emitted (unless cancelled - a cancelled round is dropped whole, no
/// pending recorded) and the calls ride back for the client to answer.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)] // one cohesive staged pipeline
async fn run_turn(
    socket: &WsSink,
    state: &Arc<AppState>,
    key: Option<&crate::keys::KeyCtx>,
    model: &str,
    cfg: &SessionCfg,
    ctx: &[CtxItem],
    cancel: &AtomicBool,
    buffer: &[u8],
    seq: &mut u64,
) -> TurnEnd {
    if buffer.is_empty() {
        send_json(
            socket,
            &error_event("invalid_request", "committed an empty audio buffer"),
        )
        .await;
        return TurnEnd::done();
    }
    let key_ext = key.cloned().map(axum::Extension);

    // 1. STT through the gateway's own whisper lane.
    let (ct, body) = transcription_multipart(&wav_wrap(buffer), &cfg.stt_model);
    let mut headers = HeaderMap::new();
    headers.insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_str(&ct)
            .unwrap_or_else(|_| axum::http::HeaderValue::from_static("application/octet-stream")),
    );
    let stt = crate::whisper::audio_transcriptions(
        State(state.clone()),
        key_ext.clone(),
        Uri::from_static("/v1/audio/transcriptions"),
        Method::POST,
        headers,
        axum::body::Bytes::from(body),
    )
    .await;
    let (stt_status, stt_bytes) = response_parts(stt).await;
    let transcript = if stt_status.is_success() {
        serde_json::from_slice::<Value>(&stt_bytes)
            .ok()
            .and_then(|v| v.get("text").and_then(Value::as_str).map(str::to_string))
    } else {
        None
    };
    let Some(transcript) = transcript.filter(|t| !t.trim().is_empty()) else {
        let message = format!(
            "whisper lane returned {} - is a whisper model pulled and the engine installed?",
            stt_status.as_u16()
        );
        send_json(
            socket,
            &json!({
                "type": "input_audio_transcription.failed",
                "error": {"code": "transcription_failed", "message": message},
            }),
        )
        .await;
        send_json(socket, &error_event("transcription_failed", &message)).await;
        return TurnEnd::done();
    };
    if cancel.load(Ordering::Relaxed) {
        send_json(socket, &json!({"type": "response.cancelled"})).await;
        return TurnEnd::done();
    }
    send_json(
        socket,
        &json!({"type": "input_audio_transcription.completed", "transcript": transcript}),
    )
    .await;

    // 2. Chat through the full OpenAI lane (admission, queue, sentinel).
    let (chat_status, chat_bytes) = chat_call(
        state,
        key,
        model,
        cfg,
        ctx_messages(cfg, ctx, Some(&transcript)),
    )
    .await;
    let chat_v = if chat_status.is_success() {
        serde_json::from_slice::<Value>(&chat_bytes).unwrap_or(Value::Null)
    } else {
        Value::Null
    };
    let calls = reply_tool_calls(&chat_v);
    if !calls.is_empty() {
        // Cancel-before-emit: a cancelled tool round disappears entirely -
        // no events, no pending, nothing persisted.
        if cancel.load(Ordering::Relaxed) {
            send_json(socket, &json!({"type": "response.cancelled"})).await;
            return TurnEnd::done();
        }
        emit_tool_round(socket, &calls, seq).await;
        return TurnEnd {
            transcript: Some(transcript),
            pending: calls,
        };
    }
    let text = if chat_status.is_success() {
        reply_text(&chat_v)
    } else {
        String::new()
    };
    if text.is_empty() {
        let why = String::from_utf8_lossy(&chat_bytes);
        send_json(
            socket,
            &error_event(
                "model_failed",
                &format!(
                    "chat lane returned {}: {}",
                    chat_status.as_u16(),
                    why.chars().take(240).collect::<String>()
                ),
            ),
        )
        .await;
        return TurnEnd::done();
    }
    if cancel.load(Ordering::Relaxed) {
        // Text exists but audio synthesis is the expensive stage - a
        // cancel here keeps the transcript delivered and skips voice.
        send_json(
            socket,
            &json!({"type": "response.cancelled", "transcript_delivered": true}),
        )
        .await;
        return TurnEnd::done();
    }
    send_json(
        socket,
        &json!({"type": "response.audio_transcript.delta", "delta": text}),
    )
    .await;
    send_json(
        socket,
        &json!({"type": "response.audio_transcript.done", "transcript": text}),
    )
    .await;

    // 3. TTS through the piper lane.
    speak_reply(socket, state, key, cfg, &text).await;
    TurnEnd::done()
}

/// Continuation turn (tool-loop or typed-context): no audio, no STT -
/// the remembered context (tool outputs last) drives the chat call and
/// the reply rides the normal transcript + TTS events. Chained tool
/// rounds are emitted exactly like first-round calls.
#[allow(clippy::too_many_arguments)] // staged pipeline mirror of run_turn
async fn run_continuation_turn(
    socket: &WsSink,
    state: &Arc<AppState>,
    key: Option<&crate::keys::KeyCtx>,
    model: &str,
    cfg: &SessionCfg,
    ctx: &[CtxItem],
    cancel: &AtomicBool,
    seq: &mut u64,
) -> TurnEnd {
    let messages = ctx_messages(cfg, ctx, None);
    if messages.as_array().is_none_or(Vec::is_empty) {
        send_json(
            socket,
            &error_event(
                "invalid_request",
                "continuation turn needs remembered context - create conversation items first",
            ),
        )
        .await;
        return TurnEnd::done();
    }
    let (chat_status, chat_bytes) = chat_call(state, key, model, cfg, messages).await;
    let chat_v = if chat_status.is_success() {
        serde_json::from_slice::<Value>(&chat_bytes).unwrap_or(Value::Null)
    } else {
        Value::Null
    };
    let calls = reply_tool_calls(&chat_v);
    if !calls.is_empty() {
        if cancel.load(Ordering::Relaxed) {
            send_json(socket, &json!({"type": "response.cancelled"})).await;
            return TurnEnd::done();
        }
        emit_tool_round(socket, &calls, seq).await;
        return TurnEnd {
            transcript: None,
            pending: calls,
        };
    }
    let text = if chat_status.is_success() {
        reply_text(&chat_v)
    } else {
        String::new()
    };
    if text.is_empty() {
        let why = String::from_utf8_lossy(&chat_bytes);
        send_json(
            socket,
            &error_event(
                "model_failed",
                &format!(
                    "chat lane returned {}: {}",
                    chat_status.as_u16(),
                    why.chars().take(240).collect::<String>()
                ),
            ),
        )
        .await;
        return TurnEnd::done();
    }
    if cancel.load(Ordering::Relaxed) {
        send_json(
            socket,
            &json!({"type": "response.cancelled", "transcript_delivered": true}),
        )
        .await;
        return TurnEnd::done();
    }
    send_json(
        socket,
        &json!({"type": "response.audio_transcript.delta", "delta": text}),
    )
    .await;
    send_json(
        socket,
        &json!({"type": "response.audio_transcript.done", "transcript": text}),
    )
    .await;
    speak_reply(socket, state, key, cfg, &text).await;
    TurnEnd::done()
}

/// Drain an axum Response into (status, body) for in-process lane calls.
async fn response_parts(resp: Response) -> (StatusCode, axum::body::Bytes) {
    let status = resp.status();
    let body = axum::body::to_bytes(resp.into_body(), AUDIO_BUFFER_CAP_BYTES)
        .await
        .unwrap_or_default();
    (status, body)
}

/// Decode one append payload into the session buffer. `Err(Cap)` is the
/// fatal case (the 20 MiB memory bound was blown - the session resets);
/// `Err(Decode)` is a one-frame client encoding bug that must NOT cost
/// the conversation context: the caller teaches and keeps the session.
#[derive(Debug)]
enum AppendErr {
    Decode(String),
    Cap,
}

fn append_audio(buffer: &mut Vec<u8>, b64: &str) -> Result<(), AppendErr> {
    let decoded =
        base64_decode(b64).map_err(|e| AppendErr::Decode(format!("audio is not base64: {e}")))?;
    if buffer.len() + decoded.len() > AUDIO_BUFFER_CAP_BYTES {
        return Err(AppendErr::Cap);
    }
    buffer.extend_from_slice(&decoded);
    Ok(())
}

/// Teach one append failure: decode bugs stay recoverable (`invalid_request`),
/// over-capacity is fatal (`buffer_overflow`) - the caller closes after a Cap.
async fn append_failed(socket: &WsSink, err: &AppendErr) -> bool {
    match err {
        AppendErr::Decode(msg) => {
            send_json(socket, &error_event("invalid_request", msg)).await;
            false
        }
        AppendErr::Cap => {
            send_json(
                socket,
                &error_event(
                    "buffer_overflow",
                    "audio buffer exceeded 20 MiB - this lane is for utterances, not files",
                ),
            )
            .await;
            true
        }
    }
}

/// Echo a created conversation item back with a server-assigned id.
async fn item_created(socket: &WsSink, item: &Value, seq: u64) {
    let mut echoed = item.clone();
    if echoed.get("id").and_then(Value::as_str).is_none() {
        echoed["id"] = json!(format!("item_{seq}"));
    }
    if echoed.get("object").is_none() {
        echoed["object"] = json!("realtime.item");
    }
    send_json(
        socket,
        &json!({"type": "conversation.item.created", "item": echoed}),
    )
    .await;
}

/// Mid-turn event routing: only cancel / append / item-create are live
/// while a turn runs; everything else is taught, nothing is dropped
/// silently. Tool items mid-turn are a client race (outputs belong to
/// the round that just produced them - send them after `response.done`)
/// and are taught, while text items keep applying to the next turn.
/// Returns `true` when the session must close.
async fn dispatch_mid_turn(
    raw: &str,
    socket: &WsSink,
    buffer: &mut Vec<u8>,
    ctx: &mut Vec<CtxItem>,
    item_seq: &mut u64,
    cancel: &AtomicBool,
) -> bool {
    let ev: Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(e) => {
            send_json(
                socket,
                &error_event("invalid_request", &format!("event is not JSON: {e}")),
            )
            .await;
            return false;
        }
    };
    match ev["type"].as_str().unwrap_or_default() {
        "response.cancel" => {
            // Silent by design: the running turn emits
            // `response.cancelled` at its next between-stage checkpoint.
            cancel.store(true, Ordering::Relaxed);
        }
        "input_audio_buffer.append" => match ev["audio"].as_str() {
            Some(b64) => {
                let fatal = match append_audio(buffer, b64) {
                    Err(err) => append_failed(socket, &err).await,
                    Ok(()) => false,
                };
                if fatal {
                    return true;
                }
            }
            None => {
                send_json(
                    socket,
                    &error_event("invalid_request", "append needs a base64 `audio` field"),
                )
                .await;
            }
        },
        "conversation.item.create" => {
            let item_type = ev["item"].get("type").and_then(Value::as_str).unwrap_or("");
            if item_type == "function_call_output" || item_type == "function_call" {
                send_json(
                    socket,
                    &error_event(
                        "invalid_request",
                        &format!(
                            "item type '{item_type}' mid-turn is a race - function-call items \
                             belong between turns: send them after response.done"
                        ),
                    ),
                )
                .await;
            } else {
                match item_text(&ev["item"]) {
                    Ok((role, text)) => {
                        *item_seq += 1;
                        ctx_push(ctx, CtxItem::Text { role, text });
                        item_created(socket, &ev["item"], *item_seq).await;
                    }
                    Err(msg) => {
                        send_json(socket, &error_event("invalid_request", &msg)).await;
                    }
                }
            }
        }
        other => {
            send_json(
                socket,
                &error_event(
                    "invalid_request",
                    &format!(
                        "event '{other}' is not accepted while a response is running - \
                         mid-turn this lane accepts response.cancel, \
                         input_audio_buffer.append, and conversation.item.create"
                    ),
                ),
            )
            .await;
        }
    }
    false
}

/// Drive one turn future to completion while keeping the read half of
/// the socket live (mid-turn routing above). `None` means the session
/// must close (client gone or fatal append).
async fn drive_turn<F>(
    turn: F,
    stream: &mut SplitStream<WebSocket>,
    sink: &WsSink,
    buffer: &mut Vec<u8>,
    ctx: &mut Vec<CtxItem>,
    item_seq: &mut u64,
    cancel: &AtomicBool,
) -> Option<TurnEnd>
where
    F: std::future::Future<Output = TurnEnd>,
{
    tokio::pin!(turn);
    loop {
        tokio::select! {
            done = &mut turn => return Some(done),
            maybe = stream.next() => {
                let Some(Ok(next)) = maybe else { return None };
                match next {
                    Message::Text(raw) => {
                        if dispatch_mid_turn(&raw, sink, buffer, ctx, item_seq, cancel).await {
                            return None;
                        }
                    }
                    Message::Close(_) => return None,
                    _ => {}
                }
            }
        }
    }
}

/// Session loop: bounded idle, hard TTL, bounded buffer, concurrent-turn
/// cancel. The turn future is driven alongside the read half so a
/// `response.cancel` frame can land while stages are still running.
#[allow(clippy::too_many_lines)] // one cohesive event loop
async fn run_session(
    socket: WebSocket,
    state: Arc<AppState>,
    model: String,
    voice: String,
    stt_model: String,
    key: Option<crate::keys::KeyCtx>,
) {
    let (sink, mut stream) = socket.split();
    let sink: WsSink = Arc::new(tokio::sync::Mutex::new(sink));
    let session_id = crate::batch::short_id("rt");
    let birth = {
        let mut obj = session_obj(&SessionCfg::new(voice.clone(), stt_model.clone()));
        obj["id"] = json!(session_id);
        obj["model"] = json!(model);
        obj
    };
    send_json(&sink, &json!({"type": "session.created", "session": birth})).await;
    let started = std::time::Instant::now();
    let mut cfg = SessionCfg::new(voice, stt_model);
    let mut ctx: Vec<CtxItem> = Vec::new();
    let mut item_seq: u64 = 0;
    // Independent counter for server-generated function_call item ids -
    // the turn future owns it while drive_turn owns item_seq (mid-turn
    // item.create), so the two mutable borrows never alias.
    let mut fc_seq: u64 = 0;
    let cancel = AtomicBool::new(false);
    let mut buffer: Vec<u8> = Vec::new();
    // Function calls the model issued and the client has not answered.
    let mut pending: Vec<PendingCall> = Vec::new();
    loop {
        if started.elapsed().as_secs() >= SESSION_MAX_SECS {
            send_json(
                &sink,
                &error_event(
                    "session_expired",
                    "session hit its 10-minute ceiling - reconnect",
                ),
            )
            .await;
            sink_close(&sink).await;
            return;
        }
        let frame = tokio::time::timeout(
            std::time::Duration::from_secs(IDLE_TIMEOUT_SECS),
            stream.next(),
        )
        .await;
        let Ok(Some(Ok(msg))) = frame else {
            // Idle timeout or client-gone: both end the session quietly.
            sink_close(&sink).await;
            return;
        };
        match msg {
            Message::Text(t) => {
                let ev: Value = match serde_json::from_str(&t) {
                    Ok(v) => v,
                    Err(e) => {
                        send_json(
                            &sink,
                            &error_event("invalid_request", &format!("event is not JSON: {e}")),
                        )
                        .await;
                        continue;
                    }
                };
                match ev["type"].as_str().unwrap_or_default() {
                    "input_audio_buffer.append" => {
                        let Some(b64) = ev["audio"].as_str() else {
                            send_json(
                                &sink,
                                &error_event(
                                    "invalid_request",
                                    "append needs a base64 `audio` field",
                                ),
                            )
                            .await;
                            continue;
                        };
                        let fatal = match append_audio(&mut buffer, b64) {
                            Err(err) => append_failed(&sink, &err).await,
                            Ok(()) => false,
                        };
                        if fatal {
                            sink_close(&sink).await;
                            return;
                        }
                    }
                    "input_audio_buffer.clear" => {
                        buffer.clear();
                        send_json(&sink, &json!({"type": "input_audio_buffer.cleared"})).await;
                    }
                    "input_audio_buffer.commit" | "response.create" => {
                        let is_create = ev["type"].as_str() == Some("response.create");
                        if buffer.is_empty() {
                            if !is_create {
                                send_json(
                                    &sink,
                                    &error_event(
                                        "invalid_request",
                                        "cannot start a response from an empty audio buffer - \
                                         append audio first",
                                    ),
                                )
                                .await;
                                continue;
                            }
                            // response.create with an empty buffer: the
                            // tool loop's continuation path. Outstanding
                            // calls are a hard taught error; remembered
                            // context alone drives a text continuation.
                            let outstanding: Vec<String> = pending
                                .iter()
                                .filter(|p| {
                                    !ctx.iter().any(|i| {
                                        matches!(i, CtxItem::ToolOutput { call_id, .. } if call_id == &p.call_id)
                                    })
                                })
                                .map(|p| p.call_id.clone())
                                .collect();
                            if !outstanding.is_empty() {
                                send_json(
                                    &sink,
                                    &error_event(
                                        "invalid_request",
                                        &format!(
                                            "{} outstanding function call(s) await \
                                             function_call_output items before the next \
                                             response: {}",
                                            outstanding.len(),
                                            outstanding.join(", ")
                                        ),
                                    ),
                                )
                                .await;
                                continue;
                            }
                            if ctx.is_empty() {
                                send_json(
                                    &sink,
                                    &error_event(
                                        "invalid_request",
                                        "cannot start a response from an empty audio buffer - \
                                         append audio first, create conversation items, or \
                                         answer pending function calls",
                                    ),
                                )
                                .await;
                                continue;
                            }
                            cancel.store(false, Ordering::Relaxed);
                            let turn_cfg = cfg.clone();
                            let turn_ctx = ctx.clone();
                            let turn = run_continuation_turn(
                                &sink,
                                &state,
                                key.as_ref(),
                                &model,
                                &turn_cfg,
                                &turn_ctx,
                                &cancel,
                                &mut fc_seq,
                            );
                            let Some(end) = drive_turn(
                                turn,
                                &mut stream,
                                &sink,
                                &mut buffer,
                                &mut ctx,
                                &mut item_seq,
                                &cancel,
                            )
                            .await
                            else {
                                sink_close(&sink).await;
                                return;
                            };
                            persist_round(&mut ctx, &mut pending, &end);
                            continue;
                        }
                        // Voice turn. A barge-in over pending calls
                        // auto-injects error tool outputs first: the
                        // model must see a completed (failed) round.
                        if !pending.is_empty() {
                            for p in &pending {
                                ctx_push(
                                    &mut ctx,
                                    CtxItem::ToolOutput {
                                        call_id: p.call_id.clone(),
                                        content:
                                            r#"{"error":"user interrupted before tool output"}"#
                                                .to_string(),
                                    },
                                );
                            }
                            pending.clear();
                        }
                        cancel.store(false, Ordering::Relaxed);
                        let turn_buf = std::mem::take(&mut buffer);
                        let turn_cfg = cfg.clone();
                        let turn_ctx = ctx.clone();
                        let turn = run_turn(
                            &sink,
                            &state,
                            key.as_ref(),
                            &model,
                            &turn_cfg,
                            &turn_ctx,
                            &cancel,
                            &turn_buf,
                            &mut fc_seq,
                        );
                        let Some(end) = drive_turn(
                            turn,
                            &mut stream,
                            &sink,
                            &mut buffer,
                            &mut ctx,
                            &mut item_seq,
                            &cancel,
                        )
                        .await
                        else {
                            sink_close(&sink).await;
                            return;
                        };
                        persist_round(&mut ctx, &mut pending, &end);
                    }
                    "session.update" => match apply_session_update(&mut cfg, &ev) {
                        Ok(session) => {
                            send_json(
                                &sink,
                                &json!({"type": "session.updated", "session": session}),
                            )
                            .await;
                        }
                        Err(msg) => {
                            send_json(&sink, &error_event("invalid_request", &msg)).await;
                        }
                    },
                    "response.cancel" => {
                        send_json(
                            &sink,
                            &error_event(
                                "invalid_request",
                                "no response in flight - cancel is honored while a \
                                 turn is running (sent mid-turn) and lands at the next \
                                 stage checkpoint",
                            ),
                        )
                        .await;
                    }
                    "conversation.item.create" => {
                        let item_type = ev["item"]
                            .get("type")
                            .and_then(Value::as_str)
                            .unwrap_or("message");
                        let parsed = match item_type {
                            "function_call_output" => item_tool_output(&ev["item"]),
                            "function_call" => item_client_tool_call(&ev["item"]),
                            _ => item_text(&ev["item"])
                                .map(|(role, text)| CtxItem::Text { role, text }),
                        };
                        match parsed {
                            Ok(item) => {
                                item_seq += 1;
                                ctx_push(&mut ctx, item);
                                item_created(&sink, &ev["item"], item_seq).await;
                            }
                            Err(msg) => {
                                send_json(&sink, &error_event("invalid_request", &msg)).await;
                            }
                        }
                    }
                    other => {
                        send_json(
                            &sink,
                            &error_event(
                                "unknown_event",
                                &format!(
                                    "unsupported event '{other}' - this lane speaks \
                                     input_audio_buffer.append/commit/clear, session.update, \
                                     response.create/cancel, and conversation.item.create \
                                     (see docs/4.API_SPEC.md)"
                                ),
                            ),
                        )
                        .await;
                    }
                }
            }
            Message::Binary(b) => {
                // Raw PCM frames are accepted as a convenience for
                // non-browser clients (same buffer, same cap).
                if buffer.len() + b.len() > AUDIO_BUFFER_CAP_BYTES {
                    send_json(
                        &sink,
                        &error_event("buffer_overflow", "audio buffer exceeded 20 MiB"),
                    )
                    .await;
                    sink_close(&sink).await;
                    return;
                }
                buffer.extend_from_slice(&b);
            }
            Message::Close(_) => {
                sink_close(&sink).await;
                return;
            }
            Message::Ping(_) | Message::Pong(_) => {}
        }
    }
}

/// After a turn that produced function calls: persist the trigger
/// transcript (voice rounds) and the `ToolCall` items, and mark the calls
/// pending. Text/voice turns without calls keep the historical
/// no-auto-persist behavior.
fn persist_round(ctx: &mut Vec<CtxItem>, pending: &mut Vec<PendingCall>, end: &TurnEnd) {
    if end.pending.is_empty() {
        return;
    }
    if let Some(t) = &end.transcript {
        ctx_push(
            ctx,
            CtxItem::Text {
                role: "user".to_string(),
                text: t.clone(),
            },
        );
    }
    for c in &end.pending {
        ctx_push(
            ctx,
            CtxItem::ToolCall {
                call_id: c.call_id.clone(),
                name: c.name.clone(),
                arguments: c.arguments.clone(),
            },
        );
    }
    end.pending.clone_into(pending);
}

// Minimal base64 (standard alphabet, padding) - keeps the realtime lane
// free of an extra dependency for two small helpers.
const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(B64[(n >> 18) as usize & 63] as char);
        out.push(B64[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            B64[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            B64[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    fn val(c: u8) -> Result<u32, String> {
        match c {
            b'A'..=b'Z' => Ok(u32::from(c - b'A')),
            b'a'..=b'z' => Ok(u32::from(c - b'a') + 26),
            b'0'..=b'9' => Ok(u32::from(c - b'0') + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            other => Err(format!("invalid base64 byte {other:#x}")),
        }
    }
    let clean: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if !clean.len().is_multiple_of(4) {
        return Err("length must be a multiple of 4".into());
    }
    let mut out = Vec::with_capacity(clean.len() / 4 * 3);
    for chunk in clean.chunks(4) {
        // Padding can only occupy positions 2/3 of a quad; anything else
        // is malformed interior padding.
        let pad = usize::from(chunk[2] == b'=') + usize::from(chunk[3] == b'=');
        if chunk[..4 - pad].contains(&b'=') {
            return Err("padding only allowed at the end".into());
        }
        let v2 = if chunk[2] == b'=' { 0 } else { val(chunk[2])? };
        let v3 = if chunk[3] == b'=' { 0 } else { val(chunk[3])? };
        let n = (val(chunk[0])? << 18) | (val(chunk[1])? << 12) | (v2 << 6) | v3;
        out.extend_from_slice(&n.to_be_bytes()[1..]);
        out.truncate(out.len() - pad);
    }
    Ok(out)
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    fn cfg() -> SessionCfg {
        SessionCfg::new("en_US-amy-medium".into(), "whisper-1".into())
    }

    #[test]
    fn unit__wav_wrap__header_fields_and_sizes() {
        let pcm = vec![0u8; 4800]; // 0.1s of silence
        let wav = wav_wrap(&pcm);
        assert_eq!(wav.len(), 44 + 4800);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 4800);
        assert_eq!(
            u32::from_le_bytes(wav[24..28].try_into().unwrap()),
            PCM_SAMPLE_RATE
        );
        assert_eq!(
            u16::from_le_bytes(wav[22..24].try_into().unwrap()),
            1,
            "mono"
        );
        assert_eq!(
            u16::from_le_bytes(wav[34..36].try_into().unwrap()),
            16,
            "16-bit"
        );
    }

    #[test]
    fn unit__base64__roundtrip_against_known_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
        for probe in [
            &b""[..],
            b"x",
            b"xy",
            b"xyz",
            b"this is a longer probe!!".as_slice(),
        ] {
            assert_eq!(base64_decode(&base64_encode(probe)).unwrap(), probe);
        }
        assert!(base64_decode("A").is_err(), "bad length");
        assert!(base64_decode("A===").is_err(), "interior padding");
        assert!(base64_decode("////").is_ok(), "valid alphabet");
        assert!(base64_decode("?!??").is_err(), "invalid byte");
    }

    #[test]
    fn unit__append_audio__decode_taught_cap_fatal() {
        let mut buffer = Vec::new();
        // A decode bug is recoverable: the buffer must stay untouched so
        // the session (and its context) survives the frame.
        match append_audio(&mut buffer, "??") {
            Err(AppendErr::Decode(msg)) => assert!(msg.contains("not base64"), "{msg}"),
            other => panic!("expected Decode, got {other:?}"),
        }
        assert!(buffer.is_empty(), "failed append must not grow the buffer");
        // Valid appends accumulate.
        append_audio(&mut buffer, &base64_encode(b"abc")).unwrap();
        assert_eq!(buffer, b"abc");
        // Blowing the 20 MiB bound is the one fatal append.
        let huge = vec![0u8; AUDIO_BUFFER_CAP_BYTES];
        match append_audio(&mut buffer, &base64_encode(&huge)) {
            Err(AppendErr::Cap) => {}
            other => panic!("expected Cap, got {other:?}"),
        }
        assert_eq!(buffer, b"abc", "capped append must not grow the buffer");
    }

    #[test]
    fn unit__transcription_multipart__shape_the_handler_parses() {
        let (ct, body) = transcription_multipart(b"WAVBYTES", "whisper-1");
        assert!(ct.starts_with("multipart/form-data; boundary="));
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("name=\"model\"\r\n\r\nwhisper-1"), "{text}");
        assert!(text.contains("filename=\"utterance.wav\""), "{text}");
        assert!(text.ends_with("--\r\n"), "closed boundary: {text}");
        assert!(body.windows(8).any(|w| w == b"WAVBYTES"));
    }

    #[test]
    fn unit__reply_text__reads_choice_zero_content() {
        let v = json!({"choices": [{"message": {"role": "assistant", "content": "hi"}}]});
        assert_eq!(reply_text(&v), "hi");
        assert_eq!(reply_text(&json!({})), "");
    }

    #[test]
    fn unit__reply_tool_calls__parses_calls_skips_malformed() {
        let v = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_1", "type": "function",
                         "function": {"name": "get_weather", "arguments": "{\"city\":\"paris\"}"}},
                        {"id": "call_2", "type": "function",
                         "function": {"name": "no_args"}},
                        {"type": "function", "function": {"name": "no_id"}},
                        "junk"
                    ]
                }
            }]
        });
        let calls = reply_tool_calls(&v);
        assert_eq!(calls.len(), 2, "malformed entries are skipped, not fatal");
        assert_eq!(calls[0].call_id, "call_1");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, "{\"city\":\"paris\"}");
        assert_eq!(
            calls[1].arguments, "{}",
            "missing arguments default to an empty object"
        );
        assert!(reply_tool_calls(&json!({})).is_empty());
    }

    #[test]
    fn unit__validate_tools__converts_shape_and_enforces_caps() {
        let ok = json!([
            {"type": "function", "name": "get_weather",
             "description": "weather lookup",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}
        ]);
        let out = validate_tools(&ok).unwrap();
        assert_eq!(
            out[0],
            json!({
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "weather lookup",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                }
            })
        );

        let custom = json!([{"type": "custom", "name": "x"}]);
        let err = validate_tools(&custom).unwrap_err();
        assert!(err.contains("function"), "{err}");

        let no_name = json!([{"type": "function"}]);
        assert!(validate_tools(&no_name).unwrap_err().contains("name"));

        let bad_params = json!([{"type": "function", "name": "x", "parameters": 7}]);
        assert!(
            validate_tools(&bad_params)
                .unwrap_err()
                .contains("parameters")
        );

        let too_many: Vec<Value> = (0..=TOOLS_MAX_COUNT)
            .map(|i| json!({"type": "function", "name": format!("t{i}")}))
            .collect();
        assert!(
            validate_tools(&json!(too_many))
                .unwrap_err()
                .contains("ceiling")
        );

        let huge = json!([
            {"type": "function", "name": "t", "description": "x".repeat(TOOLS_MAX_BYTES)}
        ]);
        assert!(validate_tools(&huge).unwrap_err().contains("bytes"));

        assert!(validate_tools(&json!({})).unwrap_err().contains("array"));
    }

    #[test]
    fn unit__validate_tool_choice__accepts_modes_teaches_rest() {
        assert_eq!(validate_tool_choice(&Value::Null).unwrap(), None);
        for mode in ["auto", "none", "required"] {
            assert_eq!(
                validate_tool_choice(&json!(mode)).unwrap(),
                Some(json!(mode))
            );
        }
        let named = json!({"type": "function", "name": "get_weather"});
        assert_eq!(validate_tool_choice(&named).unwrap(), Some(named.clone()));
        assert!(validate_tool_choice(&json!("banana")).is_err());
        assert!(validate_tool_choice(&json!({"type": "function"})).is_err());
        assert!(validate_tool_choice(&json!(7)).is_err());
    }

    #[test]
    fn unit__apply_session_update__mutates_known_keys_and_tolerates_unknown() {
        let mut c = cfg();
        let ev = json!({
            "type": "session.update",
            "session": {
                "voice": "en_US-lessac-medium",
                "stt_model": "whisper-large-v3",
                "instructions": "Answer in one word.",
                "modalities": ["text", "audio"],   // OpenAI key we ignore
                "turn_detection": null,             // ... on purpose
            }
        });
        let echo = apply_session_update(&mut c, &ev).unwrap();
        assert_eq!(c.voice, "en_US-lessac-medium");
        assert_eq!(c.stt_model, "whisper-large-v3");
        assert_eq!(c.instructions, "Answer in one word.");
        assert_eq!(echo["voice"], "en_US-lessac-medium");
        assert_eq!(echo["stt_model"], "whisper-large-v3");
        assert_eq!(echo["instructions"], "Answer in one word.");
        assert_eq!(echo["tools"], json!([]), "tools echo starts empty");

        // tools + tool_choice round-trip into the echo.
        let ev2 = json!({
            "session": {
                "tools": [{"type": "function", "name": "get_weather"}],
                "tool_choice": "auto",
            }
        });
        let echo2 = apply_session_update(&mut c, &ev2).unwrap();
        assert_eq!(echo2["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(echo2["tool_choice"], "auto");
        assert_eq!(c.tools.len(), 1);

        // tool_choice null clears.
        let echo3 =
            apply_session_update(&mut c, &json!({"session": {"tool_choice": null}})).unwrap();
        assert_eq!(echo3["tool_choice"], Value::Null);
        assert!(c.tool_choice.is_none());

        assert!(apply_session_update(&mut c, &json!({"type": "session.update"})).is_err());
        assert!(
            apply_session_update(&mut c, &json!({"session": {"voice": 7}})).is_err(),
            "non-string voice is rejected"
        );
        assert!(
            apply_session_update(&mut c, &json!({"session": {"voice": "  "}})).is_err(),
            "blank voice is rejected"
        );
        assert!(
            apply_session_update(
                &mut c,
                &json!({"session": {"tools": [{"type": "custom", "name": "x"}]}})
            )
            .is_err(),
            "non-function tools are rejected"
        );
    }

    #[test]
    fn unit__item_text__accepts_text_messages_teaches_everything_else() {
        let ok = json!({
            "type": "message",
            "role": "system",
            "content": [{"type": "text", "text": "prefer short answers"}],
        });
        assert_eq!(
            item_text(&ok).unwrap(),
            ("system".to_string(), "prefer short answers".to_string())
        );
        let default_role = json!({"content": [{"type": "text", "text": "hi"}]});
        assert_eq!(
            item_text(&default_role).unwrap(),
            ("user".to_string(), "hi".to_string())
        );

        let function = json!({
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
        });
        let err = item_text(&function).unwrap_err();
        assert!(err.contains("text context"), "{err}");

        let no_text = json!({"content": [{"type": "input_audio", "audio": "AA=="}]});
        assert!(item_text(&no_text).unwrap_err().contains("not supported"));

        let empty = json!({});
        assert!(item_text(&empty).unwrap_err().contains("content[0]"));
    }

    #[test]
    fn unit__item_tool_output__string_and_json_value_shapes() {
        let s = item_tool_output(&json!({
            "type": "function_call_output", "call_id": "c1", "output": "sunny"
        }))
        .unwrap();
        assert_eq!(
            s,
            CtxItem::ToolOutput {
                call_id: "c1".into(),
                content: "sunny".into()
            }
        );
        let obj = item_tool_output(&json!({
            "type": "function_call_output", "call_id": "c2",
            "output": {"temp_c": 18, "ok": true}
        }))
        .unwrap();
        match obj {
            CtxItem::ToolOutput { content, .. } => {
                let parsed: Value = serde_json::from_str(&content).unwrap();
                assert_eq!(parsed["temp_c"], 18);
                assert_eq!(parsed["ok"], true);
            }
            other => panic!("expected ToolOutput, got {other:?}"),
        }
        assert!(
            item_tool_output(&json!({"output": "x"})).is_err(),
            "no call_id"
        );
        assert!(
            item_tool_output(&json!({"call_id": "c"})).is_err(),
            "no output"
        );
        assert!(
            item_tool_output(&json!({"call_id": "  ", "output": "x"})).is_err(),
            "blank call_id"
        );
    }

    #[test]
    fn unit__item_client_tool_call__defaults_arguments_teaches_missing() {
        let call = item_client_tool_call(&json!({
            "type": "function_call", "call_id": "c1", "name": "get_weather",
            "arguments": "{\"city\":\"oslo\"}"
        }))
        .unwrap();
        assert_eq!(
            call,
            CtxItem::ToolCall {
                call_id: "c1".into(),
                name: "get_weather".into(),
                arguments: "{\"city\":\"oslo\"}".into()
            }
        );
        let bare = item_client_tool_call(&json!({"call_id": "c2", "name": "ping"})).unwrap();
        assert_eq!(
            bare,
            CtxItem::ToolCall {
                call_id: "c2".into(),
                name: "ping".into(),
                arguments: "{}".into()
            },
            "missing arguments default to an empty object"
        );
        assert!(item_client_tool_call(&json!({"name": "x"})).is_err());
        assert!(item_client_tool_call(&json!({"call_id": "c"})).is_err());
    }

    #[test]
    fn unit__ctx_push__caps_evict_oldest_first() {
        let mut ctx = Vec::new();
        for i in 0..CHAT_CTX_MAX_ITEMS {
            ctx_push(
                &mut ctx,
                CtxItem::Text {
                    role: "user".into(),
                    text: format!("item-{i:02}"),
                },
            );
        }
        assert_eq!(ctx.len(), CHAT_CTX_MAX_ITEMS);
        assert_eq!(
            ctx[0],
            CtxItem::Text {
                role: "user".into(),
                text: "item-00".into()
            }
        );
        ctx_push(
            &mut ctx,
            CtxItem::Text {
                role: "user".into(),
                text: "one-more".into(),
            },
        );
        assert_eq!(ctx.len(), CHAT_CTX_MAX_ITEMS, "item cap holds");
        assert_eq!(
            ctx[0],
            CtxItem::Text {
                role: "user".into(),
                text: "item-01".into()
            },
            "oldest evicted"
        );
        assert_eq!(
            ctx.last().unwrap(),
            &CtxItem::Text {
                role: "user".into(),
                text: "one-more".into()
            }
        );

        // Character cap: one big item larger than the whole budget keeps
        // only its tail once smaller items age out (never silent growth).
        let mut wide = Vec::new();
        let big = "x".repeat(CHAT_CTX_MAX_CHARS + 100);
        ctx_push(
            &mut wide,
            CtxItem::Text {
                role: "user".into(),
                text: big,
            },
        );
        ctx_push(
            &mut wide,
            CtxItem::Text {
                role: "user".into(),
                text: "small".into(),
            },
        );
        let chars: usize = wide.iter().map(CtxItem::chars).sum();
        assert!(chars <= CHAT_CTX_MAX_CHARS + "small".len());
        assert_eq!(
            wide.last().unwrap(),
            &CtxItem::Text {
                role: "user".into(),
                text: "small".into()
            },
            "newest always kept"
        );
    }

    #[test]
    fn unit__ctx_messages__instructions_head_context_middle_transcript_last() {
        let mut c = cfg();
        c.instructions = "be terse".into();
        let ctx = vec![
            CtxItem::Text {
                role: "system".into(),
                text: "ctx-a".into(),
            },
            CtxItem::Text {
                role: "user".into(),
                text: "ctx-b".into(),
            },
        ];
        let msgs = ctx_messages(&c, &ctx, Some("live question"));
        let arr = msgs.as_array().unwrap();
        assert_eq!(arr[0]["role"], "system");
        assert_eq!(arr[0]["content"], "be terse");
        assert_eq!(arr[1]["content"], "ctx-a");
        assert_eq!(arr[2]["content"], "ctx-b");
        assert_eq!(arr[3]["role"], "user");
        assert_eq!(arr[3]["content"], "live question");

        let no_instructions = SessionCfg {
            instructions: "   ".into(),
            ..cfg()
        };
        let bare = ctx_messages(&no_instructions, &[], Some("solo"));
        assert_eq!(bare.as_array().unwrap().len(), 1, "no empty system slot");

        // Continuation shape: no trailing user slot; tool items map onto
        // the chat dialect verbatim.
        let mut tooly = cfg();
        tooly.instructions = "use tools".into();
        let tctx = vec![
            CtxItem::Text {
                role: "user".into(),
                text: "weather?".into(),
            },
            CtxItem::ToolCall {
                call_id: "call_9".into(),
                name: "get_weather".into(),
                arguments: "{\"city\":\"oslo\"}".into(),
            },
            CtxItem::ToolOutput {
                call_id: "call_9".into(),
                content: "sunny".into(),
            },
        ];
        let cont = ctx_messages(&tooly, &tctx, None);
        let arr = cont.as_array().unwrap();
        assert_eq!(arr.len(), 4, "instructions + 3 items, no trailing user");
        assert_eq!(arr[1]["role"], "user");
        assert_eq!(arr[2]["role"], "assistant");
        assert_eq!(arr[2]["content"], Value::Null);
        assert_eq!(arr[2]["tool_calls"][0]["id"], "call_9");
        assert_eq!(arr[2]["tool_calls"][0]["function"]["name"], "get_weather");
        assert_eq!(arr[3]["role"], "tool");
        assert_eq!(arr[3]["tool_call_id"], "call_9");
        assert_eq!(arr[3]["content"], "sunny");
    }

    #[test]
    fn unit__persist_round__voice_round_persists_transcript_and_calls() {
        let mut ctx = Vec::new();
        let mut pending = Vec::new();
        let end = TurnEnd {
            transcript: Some("what is the weather".into()),
            pending: vec![PendingCall {
                call_id: "call_1".into(),
                name: "get_weather".into(),
                arguments: "{}".into(),
            }],
        };
        persist_round(&mut ctx, &mut pending, &end);
        assert_eq!(ctx.len(), 2, "trigger transcript + the call item");
        assert_eq!(
            ctx[0],
            CtxItem::Text {
                role: "user".into(),
                text: "what is the weather".into()
            }
        );
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].call_id, "call_1");

        // Text turns persist nothing.
        let before = ctx.len();
        persist_round(&mut ctx, &mut pending, &TurnEnd::done());
        assert_eq!(ctx.len(), before);
        assert_eq!(pending.len(), 1, "pending untouched by plain turns");
    }
}
