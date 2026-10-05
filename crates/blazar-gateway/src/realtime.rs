//! Realtime voice lane: one WebSocket per session, one utterance pipeline
//! per commit — STT (whisper lane) → chat (`OpenAI` lane) → TTS (piper
//! lane). Every stage calls the gateway's own handlers, so admission,
//! queueing, and telemetry apply exactly as on the HTTP lanes.
//!
//! MVP protocol (documented in `docs/4.API_SPEC.md`):
//!
//! ```text
//! client → server : {"type":"input_audio_buffer.append","audio":"<b64>"}
//! client → server : {"type":"input_audio_buffer.commit"}
//! server → client : {"type":"session.created","session":{...}}
//! server → client : {"type":"input_audio_transcription.completed",...}
//! server → client : {"type":"response.audio_transcript.delta"/".done",...}
//! server → client : {"type":"response.audio.delta"/".done",...}
//! server → client : {"type":"error","error":{"type","message"}}
//! ```
//!
//! Audio is PCM16LE mono 24 kHz (the `OpenAI` realtime convention); the
//! lane wraps it into a WAV container before handing it to whisper.
//! Model selection rides `?model=`; the piper voice rides `?voice=`
//! (default `en_US-amy-medium`); the whisper size rides `?stt_model=`
//! (default `whisper-1`, resolved by the whisper lane's own ladder).

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Extension, Query, State};
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::Response;
use futures::SinkExt;
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
const PCM_SAMPLE_RATE: u32 = 24_000;
const DEFAULT_VOICE: &str = "en_US-amy-medium";

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
/// of the session — the read loop notices the closed socket next.
async fn send_json(socket: &mut WebSocket, v: &Value) {
    let _ = socket
        .send(Message::Text(
            serde_json::to_string(v).unwrap_or_default().into(),
        ))
        .await;
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
/// stays usable for the next utterance.
#[allow(clippy::too_many_lines)] // one cohesive staged pipeline
async fn run_turn(
    socket: &mut WebSocket,
    state: &Arc<AppState>,
    key: Option<&crate::keys::KeyCtx>,
    model: &str,
    voice: &str,
    stt_model: &str,
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
    let (ct, body) = transcription_multipart(&wav_wrap(buffer), stt_model);
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
        send_json(
            socket,
            &error_event(
                "transcription_failed",
                &format!(
                    "whisper lane returned {} — is a whisper model pulled and the engine installed?",
                    stt_status.as_u16()
                ),
            ),
        )
        .await;
        return;
    };
    send_json(
        socket,
        &json!({"type": "input_audio_transcription.completed", "transcript": transcript}),
    )
    .await;

    // 2. Chat through the full OpenAI lane (admission, queue, sentinel).
    let chat_body = serde_json::to_vec(&json!({
        "model": model,
        "messages": [{"role": "user", "content": transcript}],
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
        "model": voice,
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
                    "piper lane returned {} (voice '{voice}') — transcript delivered above, audio skipped",
                    tts_status.as_u16()
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

/// Session loop: bounded idle, hard TTL, bounded buffer.
#[allow(clippy::too_many_lines)] // one cohesive event loop
async fn run_session(
    mut socket: WebSocket,
    state: Arc<AppState>,
    model: String,
    voice: String,
    stt_model: String,
    key: Option<crate::keys::KeyCtx>,
) {
    let session_id = crate::batch::short_id("rt");
    send_json(
        &mut socket,
        &json!({
            "type": "session.created",
            "session": {"id": session_id, "model": model, "voice": voice, "stt_model": stt_model},
        }),
    )
    .await;
    let started = std::time::Instant::now();
    let mut buffer: Vec<u8> = Vec::new();
    loop {
        if started.elapsed().as_secs() >= SESSION_MAX_SECS {
            send_json(
                &mut socket,
                &error_event(
                    "session_expired",
                    "session hit its 10-minute ceiling — reconnect",
                ),
            )
            .await;
            let _ = socket.close().await;
            return;
        }
        let frame = tokio::time::timeout(
            std::time::Duration::from_secs(IDLE_TIMEOUT_SECS),
            socket.recv(),
        )
        .await;
        let Ok(Some(Ok(msg))) = frame else {
            // Idle timeout or client-gone: both end the session quietly.
            let _ = socket.close().await;
            return;
        };
        match msg {
            Message::Text(t) => {
                let ev: Value = match serde_json::from_str(&t) {
                    Ok(v) => v,
                    Err(e) => {
                        send_json(
                            &mut socket,
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
                                &mut socket,
                                &error_event(
                                    "invalid_request",
                                    "append needs a base64 `audio` field",
                                ),
                            )
                            .await;
                            continue;
                        };
                        let decoded = match base64_decode(b64) {
                            Ok(b) => b,
                            Err(e) => {
                                send_json(
                                    &mut socket,
                                    &error_event(
                                        "invalid_request",
                                        &format!("audio is not base64: {e}"),
                                    ),
                                )
                                .await;
                                continue;
                            }
                        };
                        if buffer.len() + decoded.len() > AUDIO_BUFFER_CAP_BYTES {
                            send_json(
                                &mut socket,
                                &error_event(
                                    "buffer_overflow",
                                    "audio buffer exceeded 20 MiB — this lane is for utterances, not files",
                                ),
                            )
                            .await;
                            let _ = socket.close().await;
                            return;
                        }
                        buffer.extend_from_slice(&decoded);
                    }
                    "input_audio_buffer.commit" => {
                        run_turn(
                            &mut socket,
                            &state,
                            key.as_ref(),
                            &model,
                            &voice,
                            &stt_model,
                            &buffer,
                        )
                        .await;
                        buffer.clear();
                    }
                    other => {
                        send_json(
                            &mut socket,
                            &error_event(
                                "unknown_event",
                                &format!(
                                    "unsupported event '{other}' — this lane speaks \
                                     input_audio_buffer.append/commit (see docs/4.API_SPEC.md)"
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
                        &mut socket,
                        &error_event("buffer_overflow", "audio buffer exceeded 20 MiB"),
                    )
                    .await;
                    let _ = socket.close().await;
                    return;
                }
                buffer.extend_from_slice(&b);
            }
            Message::Close(_) => return,
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
}
