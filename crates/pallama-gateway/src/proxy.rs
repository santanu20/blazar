//! Byte-stream reverse proxy: the zero-tax path. Bodies stream between
//! client and child verbatim (no aggregation, no SSE parsing) for the
//! `OpenAI` surface; only the ollama-compat layer translates.
//!
//! Cancellation: when the client disconnects, the handler future is
//! dropped, dropping the upstream reqwest stream — llama-server frees the
//! slot on connection close. No orphan generations.

use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use futures::{Future, StreamExt};

use pallama_core::store::Store;
use pallama_core::ModelRow;

use crate::queue::Priority;
use crate::state::AppState;
use pallama_runtime::{EngineRef, SupervisionError};

/// Headers never forwarded client->child (hop-by-hop / pallama-internal).
const STRIP_REQUEST: &[&str] = &["host", "authorization", "connection", "content-length", "transfer-encoding", "x-pallama-priority", "accept-encoding"];
const STRIP_RESPONSE: &[&str] = &["connection", "content-length", "transfer-encoding", "content-encoding", "keep-alive"];

#[must_use] 
pub fn child_base(ep: &pallama_core::Endpoint) -> String {
    match ep {
        pallama_core::Endpoint::Tcp { host, port } => format!("http://{host}:{port}"),
        pallama_core::Endpoint::Unix { .. } => String::new(),
    }
}

/// Resolve a user-visible model name (with optional `:quant` suffix) to
/// the stored row: exact -> unique prefix -> levenshtein-3 suggestion.
pub fn resolve_model(store: &Store, requested: &str) -> Result<ModelRow, String> {
    let bare = requested.split(':').next().unwrap_or(requested).to_lowercase();
    let models = store.list_models().map_err(|e| e.to_string())?;
    if let Some(m) = models.iter().find(|m| m.name == bare) {
        return Ok(m.clone());
    }
    let prefixed: Vec<&ModelRow> = models.iter().filter(|m| m.name.starts_with(&bare)).collect();
    match prefixed.len() {
        1 => return Ok(prefixed[0].clone()),
        0 => {}
        _ => {
            let names: Vec<&str> = prefixed.iter().map(|m| m.name.as_str()).collect();
            return Err(format!(
                "model {requested:?} is ambiguous: {names:?}"
            ));
        }
    }
    // Levenshtein-3 suggestion across names (hand-rolled in core catalog).
    let near: Vec<&str> = models
        .iter()
        .filter(|m| pallama_core::catalog::levenshtein(&m.name, &bare) <= 3)
        .map(|m| m.name.as_str())
        .collect();
    if near.is_empty() {
        Err(format!("model {requested:?} not found; try `pallama list`"))
    } else {
        Err(format!("model {requested:?} not found; did you mean: {}?", near.join(", ")))
    }
}

pub struct ProxyOutcome {
    pub status: u16,
    pub model: String,
    pub duration_ms: u128,
}

/// Ensure the model is running (priority-aware admission) and hand back
/// the engine reference. Errors map to typed HTTP statuses.
#[allow(clippy::duration_suboptimal_units)] // 120s admission bound per plan
pub async fn ensure_with_admission(
    state: &Arc<AppState>,
    model: &str,
    priority: Priority,
) -> Result<(EngineRef, u128), Response> {
    let started = Instant::now();
    let store = Store::open(&state.dirs).map_err(|e| openai_error(500, &e.to_string()))?;
    let row = resolve_model(&store, model).map_err(|e| match e.as_str() {
        msg if msg.contains("not found") || msg.contains("ambiguous") => {
            openai_error(StatusCode::NOT_FOUND.as_u16(), msg)
        }
        msg => openai_error(500, msg),
    })?;

    let first = state.sup.ensure(&row.name).await;
    let engine = match first {
        Ok(ep) => ep,
        Err(SupervisionError::AllSlotsBusy) => {
            // Capacity exhausted: queue at our priority, bounded wait.
            state.bus.publish(pallama_runtime::PallamaEvent::QueueDepth {
                n: state.queue.depth() + 1,
            });
            state
                .queue
                .wait(&row.name, priority, std::time::Duration::from_secs(2 * 60))
                .await
                .map_err(|e| openai_error(503, &e))?;
            state
                .sup
                .ensure(&row.name)
                .await
                .map_err(|e| supervision_error(&e))?
        }
        Err(e) => return Err(supervision_error(&e)),
    };
    Ok((engine, started.elapsed().as_millis()))
}

#[must_use] 
pub fn supervision_error(e: &SupervisionError) -> Response {
    match e {
        SupervisionError::ModelNotFound(m) => openai_error(404, &format!("no such model: {m}")),
        SupervisionError::ModelLoadTimeout(m) => {
            openai_error(503, &format!("model {m} failed to become healthy (model_load_timeout); check `pallama ps`"))
        }
        SupervisionError::CircuitOpen(m) => openai_error(
            503,
            &format!("circuit open for {m}: engine keeps crashing; run `pallama ps --reset`"),
        ),
        SupervisionError::AllSlotsBusy => openai_error(503, "all slots busy"),
        SupervisionError::EngineCrashed(m) => openai_error(502, &format!("engine crashed: {m}")),
        SupervisionError::Internal(e) => openai_error(500, &format!("{e:#}")),
    }
}

/// OpenAI-shaped error body.
#[must_use] 
pub fn openai_error(status: u16, message: &str) -> Response {
    let body = serde_json::json!({
        "error": {"message": message, "type": "pallama_error", "code": status}
    });
    (StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), axum::Json(body))
        .into_response()
}

/// Forward a request to the child byte-for-byte and stream the response
/// back. `path_query` includes the leading `/`.
/// Forward one request to the child. Eight distinct request components
/// (engine, model, method, path, headers, body, load timing) — a struct
/// here would only shuffle the same data.
#[allow(clippy::too_many_arguments)]
pub async fn proxy_request(
    state: &Arc<AppState>,
    engine: &EngineRef,
    model: &str,
    method: &axum::http::Method,
    path_query: &str,
    headers: &HeaderMap,
    body: axum::body::Bytes,
    load_ms: u128,
    body_guard: Option<InFlightGuard>,
) -> Response {
    let base = child_base(&engine.endpoint);
    if base.is_empty() {
        return openai_error(500, "unix-socket child transport not supported by this proxy path yet");
    }
    let url = format!("{base}{path_query}");

    let mut req = state.http.request(method.clone(), &url);
    for (name, value) in headers {
        if !STRIP_REQUEST.contains(&name.as_str()) {
            req = req.header(name, value);
        }
    }
    let upstream = req
        .body(reqwest::Body::wrap_stream(futures::stream::once(async move {
            Ok::<_, std::io::Error>(body)
        })))
        .send()
        .await;
    let resp = match upstream {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(model, "proxy {path_query}: {e:#}");
            // The child may have crashed: reap it now so the NEXT request
            // respawns instead of 502-looping on a stale entry.
            state.sup.reap_dead_children().await;
            return openai_error(502, &format!("engine request failed: {e:#}"));
        }
    };

    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut builder = Response::builder().status(status);
    for (name, value) in resp.headers() {
        if !STRIP_RESPONSE.contains(&name.as_str()) {
            builder = builder.header(name, value);
        }
    }
    // Loading transparency (complaint #10): requests that waited on a
    // cold load say so.
    if load_ms > 100 {
        builder = builder.header("x-pallama-status", "loading");
    }
    let stream = resp.bytes_stream().map(|r| {
        r.map_err(|e| std::io::Error::other(e.to_string()))
    });
    // Hold in-flight accounting for the body's lifetime: the guard drops
    // when the client drains (or aborts) the stream.
    let stream = stream.chain(futures::stream::unfold(body_guard, |g| async {
        drop(g);
        None
    }));
    builder
        .body(Body::from_stream(stream))
        .unwrap_or_else(|e| openai_error(500, &format!("proxy body: {e}")))
}

/// Holds in-flight accounting until dropped. The gateway wraps every
/// streamed response body in one, so a 30-minute generation still counts
/// as in-flight (the reaper never evicts under load).
pub struct InFlightGuard {
    state: Arc<AppState>,
    model: String,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.state.sup.end_request(&self.model);
        self.state.queue.signal_free();
    }
}

/// Slots-aware admission: when the instance already has `slots` requests
/// in flight (1 by default), WAIT at the caller's priority instead of
/// colliding with the engine's single slot (instant rejects). Bounded by
/// the queue timeout -> 503.
pub async fn admission_gate(
    state: &Arc<AppState>,
    model: &str,
    priority: Priority,
) -> Result<InFlightGuard, Response> {
    let max_inflight: i64 = if state.config.slots == 0 {
        4 // auto multi-slot: allow modest concurrency
    } else {
        i64::from(state.config.slots)
    };
    loop {
        let busy = state
            .sup
            .ps()
            .into_iter()
            .find(|p| p.name == model)
            .map_or(0, |p| p.in_flight);
        if busy < max_inflight {
            return Ok(begin_accounting(state, model));
        }
        state
            .queue
            .wait(model, priority, std::time::Duration::from_secs(2 * 60))
            .await
            .map_err(|e| openai_error(503, &e))?;
    }
}


/// Begin accounting and return the guard; the response path holds it for
/// the body's lifetime.
#[must_use]
pub fn begin_accounting(state: &Arc<AppState>, model: &str) -> InFlightGuard {
    state.sup.begin_request(model);
    InFlightGuard { state: state.clone(), model: model.to_string() }
}

/// Bracket a NON-streaming request (accounting ends with the future).
pub async fn with_accounting<T>(
    state: &Arc<AppState>,
    model: &str,
    f: impl Future<Output = T>,
) -> T {
    state.sup.begin_request(model);
    let out = f.await;
    state.sup.end_request(model);
    state.queue.signal_free();
    out
}

#[must_use] 
pub fn header_value(v: &str) -> axum::http::HeaderValue {
    axum::http::HeaderValue::from_str(v).unwrap_or(axum::http::HeaderValue::from_static("invalid"))
}

/// Helper for handlers that need the raw path+query from the request uri.
pub fn path_and_query(uri: &axum::http::Uri) -> String {
    let pq = uri.path_and_query().map_or("/", axum::http::uri::PathAndQuery::as_str);
    pq.to_string()
}

pub type GatewayResult = Result<Response, Response>;
#[must_use] 
pub fn internal(m: &str) -> Response {
    openai_error(500, m)
}
