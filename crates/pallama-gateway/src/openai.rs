//! OpenAI-compatible surface: byte-faithful proxy to the child engine.
//! Pallama never parses or rewrites these bodies — tool calls, structured
//! output, logprobs, `stream_options` ride through untouched (complaint #1
//! fidelity argument).

use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use serde_json::json;

use pallama_core::store::Store;

use crate::proxy::{ensure_with_admission, openai_error, path_and_query, proxy_request, with_accounting};
use crate::queue::Priority;
use crate::state::AppState;

/// GET /v1/models — synthesized from the local store (any pulled model is
/// servable; the engine is hot-swapped underneath).
pub async fn models(State(state): State<Arc<AppState>>) -> Response {
    let store = match Store::open(&state.dirs) {
        Ok(s) => s,
        Err(e) => return openai_error(500, &e.to_string()),
    };
    let list = match store.list_models() {
        Ok(l) => l,
        Err(e) => return openai_error(500, &e.to_string()),
    };
    let data: Vec<serde_json::Value> = list
        .iter()
        .map(|m| {
            json!({
                "id": format!("{}:{}", m.name, m.quant.to_lowercase()),
                "object": "model",
                "owned_by": "pallama",
                "created": m.pulled_at,
            })
        })
        .collect();
    axum::Json(json!({"object": "list", "data": data})).into_response()
}

/// All POST /v1/* traffic: one handler, one proxy path, zero body
/// rewriting. `X-Pallama-Priority` orders admission under load.
pub async fn openai_proxy(
    State(state): State<Arc<AppState>>,
    uri: Uri,
    method: Method,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let model = match extract_model(&body) {
        Some(m) => m,
        None => return openai_error(400, "missing `model` field in request body"),
    };
    let priority = Priority::from_header(
        headers
            .get("x-pallama-priority")
            .and_then(|v| v.to_str().ok()),
    );

    let (engine, load_ms) = match ensure_with_admission(&state, &model, priority).await {
        Ok(ok) => ok,
        Err(resp) => return resp,
    };
    let model_name = engine.name.clone();

    with_accounting(&state, &model_name, async {
        proxy_request(
            &state,
            &engine,
            &model_name,
            method,
            &path_and_query(&uri),
            &headers,
            body,
            load_ms,
        )
        .await
    })
    .await
}

fn extract_model(body: &[u8]) -> Option<String> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    v.get("model")?.as_str().map(str::to_string)
}

/// GET|POST /v1/adapters -> child /lora-adapters (route verified in
/// upstream server.cpp).
pub async fn lora_adapters(
    State(state): State<Arc<AppState>>,
    uri: Uri,
    method: Method,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let store = match Store::open(&state.dirs) {
        Ok(s) => s,
        Err(e) => return openai_error(500, &e.to_string()),
    };
    let first = match store.list_models() {
        Ok(l) => l.into_iter().next(),
        Err(_) => None,
    };
    let Some(m) = first else {
        return openai_error(404, "no models pulled; adapters apply to a running engine");
    };
    let (engine, load_ms) = match ensure_with_admission(&state, &m.name, Priority::Normal).await {
        Ok(ok) => ok,
        Err(resp) => return resp,
    };
    let url = format!("/lora-adapters{}", path_and_query(&uri).trim_start_matches("/v1/adapters"));
    with_accounting(&state, &engine.name.clone(), async {
        proxy_request(&state, &engine, &engine.name, method, &url, &headers, body, load_ms).await
    })
    .await
}

/// GET /healthz — liveness of the gateway itself (children not required).
pub async fn healthz() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

/// Type alias to satisfy the router signature expectations.
pub type BodyBytes = Bytes;
pub type StreamBody = Body;
