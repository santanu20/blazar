//! /v1/realtime WebSocket session contract at the GATEWAY boundary over
//! the real router + supervisor + stub-llama-server children. The voice
//! stages (whisper STT, piper TTS) are exercised through their real
//! in-process lanes: with no whisper engine installed the turn must fail
//! with the teaching transcription error — never a silent drop — and the
//! socket must stay usable for the next utterance.
#![allow(non_snake_case)] // scenario__behavior naming
#![allow(clippy::similar_names)] // ws / ws2 stream handles

mod support;

use std::time::Duration;

use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::StatusCode;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};

type Ws = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

/// Recv the next JSON event, bounding each wait so a dropped frame fails
/// the test instead of hanging the suite.
async fn next_ev(ws: &mut Ws, what: &str) -> Value {
    let frame = tokio::time::timeout(Duration::from_secs(20), ws.next())
        .await
        .expect("recv timeout");
    let frame = frame.unwrap_or_else(|| panic!("stream ended while waiting for {what}"));
    let frame = frame.unwrap_or_else(|e| panic!("ws error while waiting for {what}: {e}"));
    let Message::Text(t) = frame else {
        panic!("expected text frame for {what}, got {frame:?}");
    };
    serde_json::from_str(&t).unwrap_or_else(|e| panic!("{what} is not JSON ({t}): {e}"))
}

async fn send_ev(ws: &mut Ws, ev: Value) {
    let text = ev.to_string();
    ws.send(Message::Text(text.into()))
        .await
        .expect("send event");
}

/// Minimal standard base64 (test-side only; the lane has its own).
fn b64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for c in data.chunks(3) {
        let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
        let n = u32::from(b[0]) << 16 | u32::from(b[1]) << 8 | u32::from(b[2]);
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 {
            T[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if c.len() > 2 {
            T[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

async fn connect_realtime(port_base: &str, query: &str) -> Ws {
    let url = format!("ws://{port_base}/v1/realtime{query}");
    let shown = url.clone();
    let mut req = url
        .into_client_request()
        .unwrap_or_else(|e| panic!("build request {shown}: {e}"));
    // The realtime lane sits behind the same key admission as every HTTP
    // lane — ride the unscoped admin key from the test config.
    req.headers_mut().insert(
        "authorization",
        "Bearer plm_admin".parse().expect("bearer header value"),
    );
    let (ws, _resp) = connect_async(req)
        .await
        .unwrap_or_else(|e| panic!("connect {shown}: {e}"));
    ws
}

#[tokio::test]
async fn realtime__missing_model_param__rejected_before_upgrade() {
    let server = support::start(support::config_with_keys()).await;
    // A proper WS upgrade WITH no ?model= must surface the lane's 400
    // (the teaching message names the missing parameter) — the handshake
    // never completes.
    let mut req = format!("ws://{}/v1/realtime", base_port(&server.base))
        .into_client_request()
        .unwrap();
    req.headers_mut().insert(
        "authorization",
        "Bearer plm_admin".parse().expect("bearer header value"),
    );
    let err = connect_async(req)
        .await
        .expect_err("must reject without model");
    match err {
        tokio_tungstenite::tungstenite::Error::Http(resp) => {
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        }
        other => panic!("expected HTTP rejection, got {other:?}"),
    }
}

#[tokio::test]
async fn realtime__session_lifecycle__created_update_items_clear_unknown() {
    let server = support::start(support::config_with_keys()).await;
    let mut ws = connect_realtime(&base_port(&server.base), "?model=m1").await;

    // Birth event carries the negotiated session shape.
    let created = next_ev(&mut ws, "session.created").await;
    assert_eq!(created["type"], "session.created");
    assert_eq!(created["session"]["model"], "m1");
    assert_eq!(created["session"]["voice"], "en_US-amy-medium");
    assert_eq!(created["session"]["stt_model"], "whisper-1");
    assert!(
        created["session"]["id"]
            .as_str()
            .is_some_and(|i| i.starts_with("rt")),
        "session id should carry the rt prefix: {}",
        created["session"]["id"]
    );

    // session.update round-trips the negotiated config.
    send_ev(
        &mut ws,
        json!({
            "type": "session.update",
            "session": {"voice": "en_US-ryan-high", "instructions": "Be terse."}
        }),
    )
    .await;
    let updated = next_ev(&mut ws, "session.updated").await;
    assert_eq!(updated["type"], "session.updated");
    assert_eq!(updated["session"]["voice"], "en_US-ryan-high");
    assert_eq!(updated["session"]["instructions"], "Be terse.");

    // A text conversation item joins the context and is echoed with a
    // server-assigned id.
    send_ev(
        &mut ws,
        json!({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Prior answer."}]
            }
        }),
    )
    .await;
    let item = next_ev(&mut ws, "conversation.item.created").await;
    assert_eq!(item["type"], "conversation.item.created");
    assert_eq!(item["item"]["id"], "item_1");
    assert_eq!(item["item"]["object"], "realtime.item");
    assert_eq!(item["item"]["content"][0]["text"], "Prior answer.");

    // Buffer clear acknowledges.
    send_ev(&mut ws, json!({"type": "input_audio_buffer.clear"})).await;
    let cleared = next_ev(&mut ws, "input_audio_buffer.cleared").await;
    assert_eq!(cleared["type"], "input_audio_buffer.cleared");

    // Unknown events are taught, not dropped.
    send_ev(&mut ws, json!({"type": "rate_limit.check"})).await;
    let unknown = next_ev(&mut ws, "unknown_event error").await;
    assert_eq!(unknown["type"], "error");
    assert_eq!(unknown["error"]["type"], "unknown_event");
    assert!(
        unknown["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("rate_limit.check")),
        "error should name the rejected event: {unknown}"
    );

    // Idle response.cancel teaches the mid-turn contract.
    send_ev(&mut ws, json!({"type": "response.cancel"})).await;
    let idle = next_ev(&mut ws, "idle cancel error").await;
    assert_eq!(idle["error"]["type"], "invalid_request");
    assert!(
        idle["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("no response in flight")),
        "{idle}"
    );

    ws.close(None).await.ok();
}

#[tokio::test]
async fn realtime__commit_empty_buffer__invalid_request() {
    let server = support::start(support::config_with_keys()).await;
    let mut ws = connect_realtime(&base_port(&server.base), "?model=m1").await;
    let _ = next_ev(&mut ws, "session.created").await;

    send_ev(&mut ws, json!({"type": "input_audio_buffer.commit"})).await;
    let err = next_ev(&mut ws, "empty-commit error").await;
    assert_eq!(err["type"], "error");
    assert_eq!(err["error"]["type"], "invalid_request");
    assert!(
        err["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("empty audio buffer")),
        "{err}"
    );

    // response.create is the alias for commit — same guard.
    send_ev(&mut ws, json!({"type": "response.create"})).await;
    let alias = next_ev(&mut ws, "empty response.create error").await;
    assert_eq!(alias["error"]["type"], "invalid_request");
}

#[tokio::test]
async fn realtime__junk_audio_turn__transcription_failed_and_socket_survives() {
    let server = support::start(support::config_with_keys()).await;
    let mut ws = connect_realtime(&base_port(&server.base), "?model=m1").await;
    let _ = next_ev(&mut ws, "session.created").await;

    // Malformed base64 is a one-frame client bug: taught as
    // invalid_request WITHOUT killing the session (context survives).
    send_ev(
        &mut ws,
        json!({"type": "input_audio_buffer.append", "audio": "!!not-base64!!"}),
    )
    .await;
    let bad = next_ev(&mut ws, "bad base64 error").await;
    assert_eq!(bad["type"], "error");
    assert_eq!(bad["error"]["type"], "invalid_request");
    assert!(
        bad["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("not base64")),
        "{bad}"
    );

    // Valid base64 silence, committed: the whisper lane is absent in this
    // harness, so the turn must emit the structured transcription failure
    // events and keep the session usable.
    let silence = vec![0u8; 4096];
    send_ev(
        &mut ws,
        json!({"type": "input_audio_buffer.append", "audio": b64(&silence)}),
    )
    .await;
    send_ev(&mut ws, json!({"type": "input_audio_buffer.commit"})).await;

    let failed = next_ev(&mut ws, "input_audio_transcription.failed").await;
    assert_eq!(failed["type"], "input_audio_transcription.failed");
    assert_eq!(failed["error"]["code"], "transcription_failed");
    assert!(
        failed["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("whisper lane")),
        "teaching message should point at the whisper lane: {failed}"
    );
    let dup = next_ev(&mut ws, "error transcription_failed").await;
    assert_eq!(dup["error"]["type"], "transcription_failed");

    // The socket survived the failed turn: a fresh clear still acks.
    send_ev(&mut ws, json!({"type": "input_audio_buffer.clear"})).await;
    let cleared = next_ev(&mut ws, "post-failure cleared").await;
    assert_eq!(cleared["type"], "input_audio_buffer.cleared");

    ws.close(None).await.ok();
}

/// `http://127.0.0.1:PORT` → `127.0.0.1:PORT` for ws:// URL assembly.
fn base_port(base: &str) -> String {
    base.trim_start_matches("http://").to_string()
}
