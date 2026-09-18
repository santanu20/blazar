//! Session pins (R3): opt-in request header `x-pallama-session: <name>`
//! marks a model as serving an ACTIVE session. The reaper's idle ladder
//! then holds its eviction for the pin's TTL window (refreshed by every
//! pinned request), capacity pressure demotes pinned models to
//! last-resort victims, and `POST /api/session {"action":"close"}`
//! releases the pin immediately. Force stop always wins.
//!
//! The touch lives in ONE middleware so every lane (`ollama`, `OpenAI`,
//! Anthropic, embeddings, batch replays) gets pinning for free and no
//! future handler can forget it. Header-gated: requests without the
//! header pay nothing (no buffering, no lock, no parse).
//!
//! Scope note (F78): the middleware sees the header on ANY lane, but
//! only the JSON-body lanes (chat/generate/embeddings) carry the model
//! name inside the parsed body the pin reads — multipart lanes (whisper
//! audio, file uploads) never pin because they carry no model field to
//! pin against.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::state::AppState;

/// Header name (lowercase, HTTP/1.1 canonical form as received).
pub const SESSION_HEADER: &str = "x-pallama-session";

/// Session-name ceiling — same policy as checkpoint filenames.
const MAX_SESSION_NAME: usize = 128;

/// Buffer cap for the header-carrying path. Matches the gateway's global
/// `DefaultBodyLimit` (50 MiB): a body past this cap would be rejected by
/// the limit anyway; here the error names the header so the client knows
/// the exact contract.
const PIN_BODY_CAP: usize = 50 * 1024 * 1024;

/// Pure decision core of [`pin_mw`]: given the feature switch, the raw
/// header value, the buffered body and the model store, return the
/// `(session, canonical model)` to pin — or `None` (pass through
/// untouched). Every gate (feature off, invalid/oversized name,
/// unparseable body, missing model, unresolvable model) is `None`.
fn pin_target(
    keep_secs: u64,
    session: Option<&str>,
    body: &[u8],
    store: &pallama_core::Store,
) -> Option<(String, String)> {
    if keep_secs == 0 {
        return None;
    }
    let session = session.filter(|s| !s.is_empty() && s.len() <= MAX_SESSION_NAME)?;
    let model = serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| {
            v.get("model")
                .and_then(serde_json::Value::as_str)
                .filter(|m| !m.is_empty())
                .map(str::to_string)
        })?;
    let row = crate::proxy::resolve_model(store, &model).ok()?;
    Some((session.to_string(), row.name))
}

/// Session-pin touch point (see module docs). Feature off
/// (`session_keep_secs = 0`) or header absent/invalid → untouched
/// pass-through, zero work; otherwise the body is buffered once, the
/// target model resolved to its canonical name, and the registry
/// touched before the request continues downstream.
pub async fn pin_mw(
    State(state): State<Arc<AppState>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    // Cheap gates first: feature off or no usable header → the request
    // flows on untouched, no buffering, no locks (H4).
    if state.config.session_keep_secs == 0 {
        return next.run(req).await;
    }
    let raw_session = req
        .headers()
        .get(SESSION_HEADER)
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty() && s.len() <= MAX_SESSION_NAME)
        .map(str::to_string);
    if raw_session.is_none() {
        return next.run(req).await;
    }
    let (parts, body) = req.into_parts();
    let bytes = match axum::body::to_bytes(body, PIN_BODY_CAP).await {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::PAYLOAD_TOO_LARGE,
                axum::Json(serde_json::json!({
                    "error": {"type": "pallama_error", "code": 413, "message": format!(
                        "x-pallama-session pinning buffers the request body (cap {PIN_BODY_CAP} bytes): {e} — resend without the header or shrink the body"
                    )}
                })),
            )
                .into_response();
        }
    };
    // Best-effort model resolution: unresolvable models (remote routes,
    // typos) never pin — the request passes through untouched. The
    // cached-connection visit is fully synchronous (guard never crosses
    // the `next.run` await below).
    if let Some((session, model)) = state
        .with_store(|s| {
            pin_target(
                state.config.session_keep_secs,
                raw_session.as_deref(),
                &bytes,
                s,
            )
        })
        .flatten()
    {
        state.sup.sessions.touch(&session, &model);
    }
    next.run(Request::from_parts(parts, Body::from(bytes)))
        .await
}

/// GET /api/sessions — live pins (observability: `pallama` tooling, live
/// validation, agent dashboards). Oldest idle first.
pub async fn list(State(state): State<Arc<AppState>>) -> Response {
    let ttl = std::time::Duration::from_secs(state.config.session_keep_secs);
    let sessions: Vec<serde_json::Value> = state
        .sup
        .sessions
        .list(ttl)
        .into_iter()
        .map(|s| {
            serde_json::json!({
                "session": s.session,
                "model": s.model,
                "idle_secs": s.idle_secs,
                "remaining_secs": s.remaining_secs,
            })
        })
        .collect();
    axum::Json(serde_json::json!({
        "sessions": sessions,
        "keep_secs": state.config.session_keep_secs,
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    #[allow(non_snake_case)]
    #[allow(clippy::duration_suboptimal_units)]
    mod pin_target {
        use super::super::{pin_target, MAX_SESSION_NAME};
        use pallama_core::{ModelRow, PallamaDirs, Store};

        fn fixture_store() -> (tempfile::TempDir, Store) {
            let tmp = tempfile::tempdir().unwrap();
            let dirs = PallamaDirs {
                config_dir: tmp.path().join("c"),
                data_dir: tmp.path().join("d"),
            };
            let store = Store::open(&dirs).unwrap();
            store
                .upsert_model(&ModelRow {
                    name: "m1".into(),
                    repo: "o/m1".into(),
                    quant: "Q4_K_M".into(),
                    path: "/nonexistent/m1.gguf".into(),
                    bytes: 1,
                    sha256: None,
                    mmproj_path: None,
                    shards: 1,
                    arch: None,
                    params: None,
                    ctx_train: None,
                    pulled_at: 1,
                })
                .unwrap();
            (tmp, store)
        }

        fn body(model: &str) -> Vec<u8> {
            serde_json::json!({ "model": model, "messages": [] })
                .to_string()
                .into_bytes()
        }

        #[test]
        fn unit__pin_target__pins_canonical_model_via_header() {
            let (_tmp, store) = fixture_store();
            let got = pin_target(900, Some("agent1"), &body("M1"), &store);
            assert_eq!(got, Some(("agent1".into(), "m1".into())));
        }

        #[test]
        fn unit__pin_target__feature_off_never_pins() {
            let (_tmp, store) = fixture_store();
            assert!(pin_target(0, Some("agent1"), &body("m1"), &store).is_none());
        }

        #[test]
        fn unit__pin_target__unknown_model_never_pins() {
            let (_tmp, store) = fixture_store();
            assert!(pin_target(900, Some("s"), &body("no-such-model"), &store).is_none());
        }

        #[test]
        fn unit__pin_target__invalid_session_names_rejected() {
            let (_tmp, store) = fixture_store();
            assert!(pin_target(900, None, &body("m1"), &store).is_none());
            assert!(pin_target(900, Some(""), &body("m1"), &store).is_none());
            let long = "x".repeat(MAX_SESSION_NAME + 1);
            assert!(pin_target(900, Some(&long), &body("m1"), &store).is_none());
        }

        #[test]
        fn unit__pin_target__malformed_body_never_pins() {
            let (_tmp, store) = fixture_store();
            assert!(pin_target(900, Some("s"), b"not json", &store).is_none());
            assert!(pin_target(900, Some("s"), b"{\"messages\":[]}", &store).is_none());
            assert!(pin_target(900, Some("s"), br#"{"model":""}"#, &store).is_none());
        }
    }
}
