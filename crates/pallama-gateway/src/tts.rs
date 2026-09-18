//! POST `/v1/audio/speech` — `OpenAI` TTS shape on the local piper lane.
//!
//! `model` IS the piper voice id (honest mapping: `en_US-amy-medium`),
//! `response_format` speaks WAV only in this phase (the lane is a
//! one-shot binary writing a WAV; container conversion is a codec
//! concern, not a synthesis one), `speed` maps to `piper`'s inverse
//! length scale. Remote intent (`name:model`) forwards like every
//! other lane.

use axum::body::Bytes;
use axum::extract::{Extension, State};
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use serde_json::Value;
use std::sync::Arc;

use crate::proxy::openai_error;
use crate::remotes::split_remote;
use crate::state::AppState;

/// Bound mirrors the runtime cap; the route validates first so clients
/// get a 400 before any spawn.
const MAX_INPUT_CHARS: usize = 10_000;

pub async fn audio_speech(
    State(state): State<Arc<AppState>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    uri: Uri,
    method: Method,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return openai_error(400, &format!("body is not valid JSON: {e}")),
    };
    let Some(voice) = req.get("model").and_then(Value::as_str) else {
        return openai_error(
            400,
            "missing required field: model (a piper voice id, e.g. en_US-amy-medium)",
        );
    };
    let Some(text) = req.get("input").and_then(Value::as_str) else {
        return openai_error(400, "missing required field: input");
    };
    if text.trim().is_empty() {
        return openai_error(400, "input must be non-empty text");
    }
    if text.chars().count() > MAX_INPUT_CHARS {
        return openai_error(
            400,
            &format!(
                "input is {} chars (max {MAX_INPUT_CHARS}) — split long documents",
                text.chars().count()
            ),
        );
    }
    match req.get("response_format").and_then(Value::as_str) {
        None | Some("wav") => {}
        Some(other) => {
            return openai_error(
                400,
                &format!(
                    "response_format {other:?} is not supported on the local lane — \
                     piper speaks WAV (omit response_format); convert downstream if you need {other}"
                ),
            )
        }
    }
    let speed = match req.get("speed").and_then(Value::as_f64) {
        None => None,
        Some(s) if (0.25..=4.0).contains(&s) => Some(s),
        Some(s) => return openai_error(400, &format!("speed {s} out of range (0.25..=4.0)")),
    };

    // Remote intent wins: `name:model` never goes local.
    if split_remote(voice, &state.config).is_some() {
        if let Some(Extension(k)) = &key_ext {
            if let Some(entry) = state.keys.entry(&k.name) {
                if let Err(rej) = state.keys.check(&entry, voice) {
                    return rej.to_response();
                }
                state.keys.charge_request(&k.name);
            }
        }
        return crate::remotes::forward_with_health(
            &state,
            voice,
            &method,
            uri.path_and_query()
                .map_or("/v1/audio/speech", axum::http::uri::PathAndQuery::as_str),
            &headers,
            body,
        )
        .await;
    }

    // Local lane: the runtime error carries the exact teaching (install
    // vs pull vs voice list) — pass it through with the right status.
    match pallama_runtime::piper::synthesize(
        &state.dirs,
        voice,
        text,
        speed,
        std::time::Duration::from_secs(120),
    )
    .await
    {
        Ok(wav) => Response::builder()
            .status(StatusCode::OK)
            .header(axum::http::header::CONTENT_TYPE, "audio/wav")
            .body(axum::body::Body::from(wav))
            .unwrap_or_else(|_| openai_error(500, "response build").into_response()),
        Err(e) => {
            let msg = format!("{e:#}");
            let status = if msg.contains("not installed") || msg.contains("is not pulled") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::BAD_REQUEST
            };
            openai_error(status.as_u16(), &msg)
        }
    }
}
