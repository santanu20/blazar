//! sglang-native `/generate` surface. Blazar's own API families are
//! OpenAI/Ollama/Anthropic-shaped, but the disaggregation wire contract
//! rides sglang's native `GenerateReqInput` (`bootstrap_host`/
//! `bootstrap_room` travel in the request, not in any chat-compat
//! body). This route passes the native body through to the serving
//! child - and when the model asks for the PD pair, it runs the
//! router-role two-phase dispatch instead.

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{Query, State};
use axum::response::Response;
use serde::Deserialize;

use crate::state::AppState;

#[derive(Deserialize)]
pub(crate) struct GenerateQuery {
    /// The native body carries no model field; Blazar is multi-model,
    /// so the route takes the target from the query string.
    model: Option<String>,
}

/// POST /generate?model=<name> - native passthrough (or PD two-phase
/// when the model's sglang tuning enables `disagg_pair`).
#[allow(clippy::too_many_lines)]
pub(crate) async fn generate(
    State(state): State<Arc<AppState>>,
    Query(query): Query<GenerateQuery>,
    body: Bytes,
) -> Response {
    let Some(model) = query.model.filter(|m| !m.is_empty()) else {
        return crate::proxy::openai_error(
            400,
            "sglang-native /generate needs the model in the query string: POST /generate?model=<name> \
             (the GenerateReqInput body carries no model field)",
        );
    };
    match serde_json::from_slice::<serde_json::Value>(&body) {
        Ok(v) if v.is_object() => {}
        Ok(_) => {
            return crate::proxy::openai_error(
                400,
                "the sglang-native /generate body must be a JSON object (GenerateReqInput)",
            );
        }
        Err(e) => {
            return crate::proxy::openai_error(
                400,
                &format!("the sglang-native /generate body must be JSON: {e}"),
            );
        }
    }

    if crate::pd::pd_pair_requested(&state, &model) {
        return crate::pd::pd_generate_two_phase(&state, &model, &body).await;
    }

    // Plain passthrough: hand the native body to the serving child and
    // relay the response stream verbatim.
    let engine = match state.sup.ensure_routed(&model, None).await {
        Ok(engine) => engine,
        Err(e) => {
            return crate::proxy::openai_error(
                503,
                &format!("the serving child for {model} could not come up: {e}"),
            );
        }
    };
    let url = format!("{}/generate", crate::proxy::child_base(&engine.endpoint));
    let req = crate::proxy::child_auth(
        crate::state::child_client(&state, &engine.endpoint)
            .post(&url)
            .header("content-type", "application/json"),
        &engine,
    )
    .body(body);
    let resp = match crate::proxy::child_send(&state, &engine, req.send()).await {
        Ok(resp) => resp,
        Err(e) => {
            return crate::proxy::openai_error(
                e.status_u16(),
                &format!("child /generate failed: {e}"),
            );
        }
    };
    let status = axum::http::StatusCode::from_u16(resp.status().as_u16())
        .unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR);
    let mut headers = resp.headers().clone();
    headers.remove(axum::http::header::CONTENT_LENGTH);
    let mut response = Response::new(axum::body::Body::from_stream(resp.bytes_stream()));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response
}
