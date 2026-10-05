//! Realtime voice lane: one WebSocket per session, one utterance pipeline
//! per committed buffer — STT (whisper lane) → chat (`OpenAI` lane) → TTS
//! (piper lane). Every stage calls the gateway's own handlers, so admission,
//! queueing, and telemetry apply exactly as on the HTTP lanes.
//!
//! Event surface (documented in `docs/4.API_SPEC.md`):
//!
//! ```text
//! client → server : {"type":"input_audio_buffer.append","audio":"<b64>"}
//! client → server : {"type":"input_audio_buffer.commit"}
//! client → server : {"type":"input_audio_buffer.clear"}
//! client → server : {"type":"session.update","session":{voice?,stt_model?,instructions?}}
//! client → server : {"type":"response.create"}
//! client → server : {"type":"response.cancel"}
//! client → server : {"type":"conversation.item.create","item":{...}}
//! server → client : {"type":"session.created","session":{...}}
//! server → client : {"type":"session.updated","session":{...}}
//! server → client : {"type":"input_audio_transcription.completed",...}
//! server → client : {"type":"input_audio_transcription.failed","error":{...}}
//! server → client : {"type":"response.audio_transcript.delta"/".done",...}
//! server → client : {"type":"response.audio.delta"/".done",...}
//! server → client : {"type":"response.cancelled",...}
//! server → client : {"type":"input_audio_buffer.cleared",...}
//! server → client : {"type":"conversation.item.created","item":{...}}
//! server → client : {"type":"error","error":{"type","message"}}
//! ```
//!
//! Audio is PCM16LE mono 24 kHz (the `OpenAI` realtime convention); the
//! lane wraps it into a WAV container before handing it to whisper.
//! Model selection rides `?model=`; the piper voice rides `?voice=`
//! (default `en_US-amy-medium`); the whisper size rides `?stt_model=`
//! (default `whisper-1`, resolved by the whisper lane's own ladder).
//!
//! Semantics worth knowing:
//! - `session.update` mutates voice / `stt_model` / instructions and they
//!   apply from the NEXT turn on (a running turn snapshots the config).
//! - `response.cancel` sets a flag checked BETWEEN pipeline stages, so a
//!   cancel lands after the current stage finishes (chat streams are not
//!   cut mid-flight — each stage call is atomic here). A cancel that
//!   arrives too late (turn already finished) simply has no effect.
//! - `conversation.item.create` accepts text messages only (the local
//!   chat lane consumes text context); context is capped at 16 items /
//!   8 192 chars, oldest evicted first. Items created mid-turn apply to
//!   the next turn.
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
use futures::stream::SplitSink;
use serde_json::{Value, json};

use crate::state::AppState;

/// No frame for this long closes the session (client vanished mid-phrase).
const IDLE_TIMEOUT_SECS: u64 = 120;
/// Hard session ceiling — a realtime socket is not a forever connection.
const SESSION_MAX_SECS: u64 = 600;
/// Append-buffer cap: beyond this the client is misbehaving (or not a
/// voice client at all) — error event, then close.
const AUDIO_BUFFER_CAP_BYTES: usize = 20 * 1024 * 1024;
/// TTS audio rides ~4 KiB base64 deltas.
const TTS_CHUNK_BYTES: usize = 4 * 1024;
/// Utterance budget for the reply text.
const CHAT_MAX_TOKENS: u64 = 512;
/// Conversation-context ceiling: at most this many remembered items ...
const CHAT_CTX_MAX_ITEMS: usize = 16;
/// ... and this many characters of item text (oldest evicted first).
const CHAT_CTX_MAX_CHARS: usize = 8_192;
const PCM_SAMPLE_RATE: u32 = 24_000;
const DEFAULT_VOICE: &str = "en_US-amy-medium";

/// Mutable per-session configuration (voice, stt size, system prompt).
/// Snapshot-cloned at each turn start so `session.update` mid-turn can
/// only affect the next turn.
#[derive(Debug, Clone)]
struct SessionCfg {
    voice: String,
    stt_model: String,
    instructions: String,
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
    Ok(json!({
        "voice": cfg.voice,
        "stt_model": cfg.stt_model,
        "instructions": cfg.instructions,
    }))
}

/// Extract `(role, text)` from a `conversation.item.create` item. Only
/// text messages are consumable by the local chat lane; everything else
/// gets a teaching error instead of a silent drop.
fn item_text(item: &Value) -> Result<(String, String), String> {
    let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
    let content = item.pointer("/content/0").ok_or(
        "item needs content[0] with type \"text\" (the local chat lane consumes text context only)",
    )?;
    if content.get("type").and_then(Value::as_str) != Some("text") {
        return Err(format!(
            "item content type '{}' is not supported — this lane consumes text context only",
            content.get("type").and_then(Value::as_str).unwrap_or("?")
        ));
    }
    let text = content
        .get("text")
        .and_then(Value::as_str)
        .ok_or("text content needs a `text` field")?;
    Ok((role.to_string(), text.to_string()))
}

/// Push one item into the conversation context, enforcing both caps by
/// evicting oldest-first.
fn ctx_push(ctx: &mut Vec<(String, String)>, role: &str, text: &str) {
    ctx.push((role.to_string(), text.to_string()));
    let mut chars: usize = ctx.iter().map(|(_, t)| t.len()).sum();
    while (ctx.len() > CHAT_CTX_MAX_ITEMS || chars > CHAT_CTX_MAX_CHARS) && !ctx.is_empty() {
        chars = chars.saturating_sub(ctx[0].1.len());
        ctx.remove(0);
    }
}

/// Build the chat-completion `messages` array: optional system
/// instructions, remembered context items, then the live transcript.
fn ctx_messages(cfg: &SessionCfg, ctx: &[(String, String)], transcript: &str) -> Value {
    let mut messages = Vec::with_capacity(ctx.len() + 2);
    if !cfg.instructions.trim().is_empty() {
        messages.push(json!({"role": "system", "content": cfg.instructions}));
    }
    for (role, text) in ctx {
        messages.push(json!({"role": role, "content": text}));
    }
    messages.push(json!({"role": "user", "content": transcript}));
    Value::Array(messages)
}

/// `GET /v1/realtime?model=...` — upgrade, then run the session loop.
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
            "missing ?model= — the realtime lane needs the chat model id as a query \
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
/// of the session — the read loop notices the closed socket next. The
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

/// One committed utterance: STT → chat → TTS, each event streamed as it
/// lands. A stage failure emits an error event and returns — the socket
/// stays usable for the next utterance. `cancel` is checked between
/// stages (chat streams are not cut mid-flight); observing it emits
/// `response.cancelled` and stops the pipeline.
#[allow(clippy::too_many_lines)] // one cohesive staged pipeline
#[allow(clippy::too_many_arguments)] // staged pipeline: socket, state, key, model, cfg, ctx, cancel, buffer
async fn run_turn(
    socket: &WsSink,
    state: &Arc<AppState>,
    key: Option<&crate::keys::KeyCtx>,
    model: &str,
    cfg: &SessionCfg,
    ctx: &[(String, String)],
    cancel: &AtomicBool,
    buffer: &[u8],
) {
    if buffer.is_empty() {
        send_json(
            socket,
            &error_event("invalid_request", "committed an empty audio buffer"),
        )
        .await;
        return;
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
            "whisper lane returned {} — is a whisper model pulled and the engine installed?",
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
        return;
    };
    if cancel.load(Ordering::Relaxed) {
        send_json(socket, &json!({"type": "response.cancelled"})).await;
        return;
    }
    send_json(
        socket,
        &json!({"type": "input_audio_transcription.completed", "transcript": transcript}),
    )
    .await;

    // 2. Chat through the full OpenAI lane (admission, queue, sentinel).
    let chat_body = serde_json::to_vec(&json!({
        "model": model,
        "messages": ctx_messages(cfg, ctx, &transcript),
        "max_tokens": CHAT_MAX_TOKENS,
        "stream": false,
    }))
    .unwrap_or_default();
    let chat = crate::openai::openai_proxy(
        State(state.clone()),
        None,
        key_ext.clone(),
        Uri::from_static("/v1/chat/completions"),
        Method::POST,
        HeaderMap::new(),
        axum::body::Bytes::from(chat_body),
    )
    .await;
    let (chat_status, chat_bytes) = response_parts(chat).await;
    let text = if chat_status.is_success() {
        serde_json::from_slice::<Value>(&chat_bytes)
            .map(|v| reply_text(&v))
            .unwrap_or_default()
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
        return;
    }
    if cancel.load(Ordering::Relaxed) {
        // Text exists but audio synthesis is the expensive stage — a
        // cancel here keeps the transcript delivered and skips voice.
        send_json(
            socket,
            &json!({"type": "response.cancelled", "transcript_delivered": true}),
        )
        .await;
        return;
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

    // 3. TTS through the piper lane; failure degrades to text-only with
    // an explicit error event (the transcript was already delivered).
    let tts_body = serde_json::to_vec(&json!({
        "model": cfg.voice,
        "input": text,
        "response_format": "wav",
    }))
    .unwrap_or_default();
    let tts = crate::tts::audio_speech(
        State(state.clone()),
        key_ext,
        Uri::from_static("/v1/audio/speech"),
        Method::POST,
        HeaderMap::new(),
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
                    "piper lane returned {} (voice '{}') — transcript delivered above, audio skipped",
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

/// Drain an axum Response into (status, body) for in-process lane calls.
async fn response_parts(resp: Response) -> (StatusCode, axum::body::Bytes) {
    let status = resp.status();
    let body = axum::body::to_bytes(resp.into_body(), AUDIO_BUFFER_CAP_BYTES)
        .await
        .unwrap_or_default();
    (status, body)
}

/// Decode one append payload into the session buffer; `Err` is a fatal
/// misbehavior (bad base64 or over-capacity) and the message says which.
fn append_audio(buffer: &mut Vec<u8>, b64: &str) -> Result<(), String> {
    let decoded = base64_decode(b64).map_err(|e| format!("audio is not base64: {e}"))?;
    if buffer.len() + decoded.len() > AUDIO_BUFFER_CAP_BYTES {
        return Err("audio buffer exceeded 20 MiB — this lane is for utterances, not files".into());
    }
    buffer.extend_from_slice(&decoded);
    Ok(())
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
/// silently. Returns `true` when the session must close.
async fn dispatch_mid_turn(
    raw: &str,
    socket: &WsSink,
    buffer: &mut Vec<u8>,
    ctx: &mut Vec<(String, String)>,
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
                if let Err(msg) = append_audio(buffer, b64) {
                    send_json(socket, &error_event("buffer_overflow", &msg)).await;
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
        "conversation.item.create" => match item_text(&ev["item"]) {
            Ok((role, text)) => {
                *item_seq += 1;
                ctx_push(ctx, &role, &text);
                item_created(socket, &ev["item"], *item_seq).await;
            }
            Err(msg) => {
                send_json(socket, &error_event("invalid_request", &msg)).await;
            }
        },
        other => {
            send_json(
                socket,
                &error_event(
                    "invalid_request",
                    &format!(
                        "event '{other}' is not accepted while a response is running — \
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

/// Session loop: bounded idle, hard TTL, bounded buffer, concurrent-turn
/// cancel. The turn future is pinned and polled alongside the read half
/// so a `response.cancel` frame can land while stages are still running.
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
    send_json(
        &sink,
        &json!({
            "type": "session.created",
            "session": {"id": session_id, "model": model, "voice": voice, "stt_model": stt_model},
        }),
    )
    .await;
    let started = std::time::Instant::now();
    let mut cfg = SessionCfg {
        voice,
        stt_model,
        instructions: String::new(),
    };
    let mut ctx: Vec<(String, String)> = Vec::new();
    let mut item_seq: u64 = 0;
    let cancel = AtomicBool::new(false);
    let mut buffer: Vec<u8> = Vec::new();
    loop {
        if started.elapsed().as_secs() >= SESSION_MAX_SECS {
            send_json(
                &sink,
                &error_event(
                    "session_expired",
                    "session hit its 10-minute ceiling — reconnect",
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
                        if let Err(msg) = append_audio(&mut buffer, b64) {
                            send_json(&sink, &error_event("buffer_overflow", &msg)).await;
                            sink_close(&sink).await;
                            return;
                        }
                    }
                    "input_audio_buffer.clear" => {
                        buffer.clear();
                        send_json(&sink, &json!({"type": "input_audio_buffer.cleared"})).await;
                    }
                    "input_audio_buffer.commit" | "response.create" => {
                        if buffer.is_empty() {
                            send_json(
                                &sink,
                                &error_event(
                                    "invalid_request",
                                    "cannot start a response from an empty audio buffer — \
                                     append audio first",
                                ),
                            )
                            .await;
                            continue;
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
                        );
                        tokio::pin!(turn);
                        loop {
                            tokio::select! {
                                () = &mut turn => break,
                                maybe = stream.next() => {
                                    let Some(Ok(next)) = maybe else {
                                        sink_close(&sink).await;
                                        return;
                                    };
                                    match next {
                                        Message::Text(raw) => {
                                            if dispatch_mid_turn(
                                                &raw,
                                                &sink,
                                                &mut buffer,
                                                &mut ctx,
                                                &mut item_seq,
                                                &cancel,
                                            )
                                            .await
                                            {
                                                sink_close(&sink).await;
                                                return;
                                            }
                                        }
                                        Message::Close(_) => {
                                            sink_close(&sink).await;
                                            return;
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
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
                                "no response in flight — cancel is honored while a \
                                 turn is running (sent mid-turn) and lands at the next \
                                 stage checkpoint",
                            ),
                        )
                        .await;
                    }
                    "conversation.item.create" => match item_text(&ev["item"]) {
                        Ok((role, text)) => {
                            item_seq += 1;
                            ctx_push(&mut ctx, &role, &text);
                            item_created(&sink, &ev["item"], item_seq).await;
                        }
                        Err(msg) => {
                            send_json(&sink, &error_event("invalid_request", &msg)).await;
                        }
                    },
                    other => {
                        send_json(
                            &sink,
                            &error_event(
                                "unknown_event",
                                &format!(
                                    "unsupported event '{other}' — this lane speaks \
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

// Minimal base64 (standard alphabet, padding) — keeps the realtime lane
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
    fn unit__apply_session_update__mutates_known_keys_and_tolerates_unknown() {
        let mut cfg = SessionCfg {
            voice: "en_US-amy-medium".into(),
            stt_model: "whisper-1".into(),
            instructions: String::new(),
        };
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
        let echo = apply_session_update(&mut cfg, &ev).unwrap();
        assert_eq!(cfg.voice, "en_US-lessac-medium");
        assert_eq!(cfg.stt_model, "whisper-large-v3");
        assert_eq!(cfg.instructions, "Answer in one word.");
        assert_eq!(echo["voice"], "en_US-lessac-medium");
        assert_eq!(echo["stt_model"], "whisper-large-v3");
        assert_eq!(echo["instructions"], "Answer in one word.");

        assert!(apply_session_update(&mut cfg, &json!({"type": "session.update"})).is_err());
        assert!(
            apply_session_update(&mut cfg, &json!({"session": {"voice": 7}})).is_err(),
            "non-string voice is rejected"
        );
        assert!(
            apply_session_update(&mut cfg, &json!({"session": {"voice": "  "}})).is_err(),
            "blank voice is rejected"
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
    fn unit__ctx_push__caps_evict_oldest_first() {
        let mut ctx = Vec::new();
        for i in 0..CHAT_CTX_MAX_ITEMS {
            ctx_push(&mut ctx, "user", &format!("item-{i:02}"));
        }
        assert_eq!(ctx.len(), CHAT_CTX_MAX_ITEMS);
        assert_eq!(ctx[0].1, "item-00");
        ctx_push(&mut ctx, "user", "one-more");
        assert_eq!(ctx.len(), CHAT_CTX_MAX_ITEMS, "item cap holds");
        assert_eq!(ctx[0].1, "item-01", "oldest evicted");
        assert_eq!(ctx.last().unwrap().1, "one-more");

        // Character cap: one big item larger than the whole budget keeps
        // only its tail once smaller items age out (never silent growth).
        let mut wide = Vec::new();
        let big = "x".repeat(CHAT_CTX_MAX_CHARS + 100);
        ctx_push(&mut wide, "user", &big);
        ctx_push(&mut wide, "user", "small");
        let chars: usize = wide.iter().map(|(_, t)| t.len()).sum();
        assert!(chars <= CHAT_CTX_MAX_CHARS + "small".len());
        assert_eq!(wide.last().unwrap().1, "small", "newest always kept");
    }

    #[test]
    fn unit__ctx_messages__instructions_head_context_middle_transcript_last() {
        let cfg = SessionCfg {
            voice: "v".into(),
            stt_model: "whisper-1".into(),
            instructions: "be terse".into(),
        };
        let ctx = vec![
            ("system".to_string(), "ctx-a".to_string()),
            ("user".to_string(), "ctx-b".to_string()),
        ];
        let msgs = ctx_messages(&cfg, &ctx, "live question");
        let arr = msgs.as_array().unwrap();
        assert_eq!(arr[0]["role"], "system");
        assert_eq!(arr[0]["content"], "be terse");
        assert_eq!(arr[1]["content"], "ctx-a");
        assert_eq!(arr[2]["content"], "ctx-b");
        assert_eq!(arr[3]["role"], "user");
        assert_eq!(arr[3]["content"], "live question");

        let no_instructions = SessionCfg {
            voice: "v".into(),
            stt_model: "whisper-1".into(),
            instructions: "   ".into(),
        };
        let bare = ctx_messages(&no_instructions, &[], "solo");
        assert_eq!(bare.as_array().unwrap().len(), 1, "no empty system slot");
    }
}
