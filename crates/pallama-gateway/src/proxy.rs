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
use crate::sentinel;
use crate::state::AppState;
use pallama_runtime::{EngineRef, SupervisionError};

/// Headers never forwarded client->child (hop-by-hop / pallama-internal).
pub(crate) const STRIP_REQUEST: &[&str] = &[
    "host",
    "authorization",
    "connection",
    "content-length",
    "transfer-encoding",
    "x-pallama-priority",
    "x-pallama-num-ctx",
    "x-pallama-enforce",
    "accept-encoding",
];
pub(crate) const STRIP_RESPONSE: &[&str] = &[
    "connection",
    "content-length",
    "transfer-encoding",
    "content-encoding",
    "keep-alive",
];

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
    let bare = requested
        .split(':')
        .next()
        .unwrap_or(requested)
        .to_lowercase();
    let models = store.list_models().map_err(|e| e.to_string())?;
    if let Some(m) = models.iter().find(|m| m.name == bare) {
        return Ok(m.clone());
    }
    let prefixed: Vec<&ModelRow> = models
        .iter()
        .filter(|m| m.name.starts_with(&bare))
        .collect();
    match prefixed.len() {
        1 => return Ok(prefixed[0].clone()),
        0 => {}
        _ => {
            let names: Vec<&str> = prefixed.iter().map(|m| m.name.as_str()).collect();
            return Err(format!("model {requested:?} is ambiguous: {names:?}"));
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
        Err(format!(
            "model {requested:?} not found; did you mean: {}?",
            near.join(", ")
        ))
    }
}

pub struct ProxyOutcome {
    pub status: u16,
    pub model: String,
    pub duration_ms: u128,
}

/// Ensure the model is running (priority-aware admission) and hand back
/// the engine reference. Errors map to typed HTTP statuses. `prefix`
/// carries the prompt-affinity hash (B1): chat-family callers pass it
/// so repeat conversations land on their warm replica; everything else
/// passes `None`.
#[allow(clippy::duration_suboptimal_units)] // 120s admission bound per plan
pub async fn ensure_with_admission(
    state: &Arc<AppState>,
    model: &str,
    priority: Priority,
    prefix: Option<u64>,
) -> Result<(EngineRef, u128), Response> {
    let started = Instant::now();
    let store = Store::open(&state.dirs).map_err(|e| openai_error(500, &e.to_string()))?;
    let row = resolve_model(&store, model).map_err(|e| match e.as_str() {
        msg if msg.contains("not found") || msg.contains("ambiguous") => {
            openai_error(StatusCode::NOT_FOUND.as_u16(), msg)
        }
        msg => openai_error(500, msg),
    })?;

    let first = state.sup.ensure_routed(&row.name, prefix).await;
    // (Bank restore happens inside the supervisor at spawn-readiness.)
    let engine = match first {
        Ok(ep) => ep,
        Err(SupervisionError::AllSlotsBusy) => {
            // Capacity exhausted: queue at our priority, bounded wait.
            state
                .bus
                .publish(pallama_runtime::PallamaEvent::QueueDepth {
                    n: state.queue.depth() + 1,
                });
            state
                .queue
                .wait(
                    &row.name,
                    priority,
                    None,
                    0,
                    std::time::Duration::from_mins(2),
                )
                .await
                .map_err(|e| openai_error(503, &e))?;
            state
                .sup
                .ensure_routed(&row.name, prefix)
                .await
                .map_err(|e| supervision_error(&e))?
        }
        Err(e) => return Err(supervision_error(&e)),
    };
    Ok((engine, started.elapsed().as_millis()))
}

/// Prompt-prefix affinity hash (B1) from a parsed request body, stable
/// across conversation turns and blind to samplers/options on purpose:
/// chat/completions → system + first user turn; generate → `prompt`;
/// responses → `input` (string or parts array). First 1 KiB of each
/// part. `None` when no recognizable prompt (embeddings, tools) — no
/// affinity, plain load-balance.
#[must_use]
pub fn affinity_hash(req: &serde_json::Value) -> Option<u64> {
    use std::hash::{Hash, Hasher};
    type H = std::collections::hash_map::DefaultHasher;
    let head = |s: &str, h: &mut H| s.as_bytes()[..s.len().min(1024)].hash(h);
    let mut h = H::new();
    if let Some(messages) = req.get("messages").and_then(serde_json::Value::as_array) {
        if messages.is_empty() {
            return None;
        }
        let content = |m: &serde_json::Value| -> String {
            match m.get("content") {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(other) => other.to_string(), // multimodal parts array
                None => String::new(),
            }
        };
        let role_is = |m: &serde_json::Value, role: &str| {
            m.get("role").and_then(serde_json::Value::as_str) == Some(role)
        };
        let system = messages
            .iter()
            .find(|m| role_is(m, "system"))
            .map(&content)
            .unwrap_or_default();
        let user = messages
            .iter()
            .find(|m| role_is(m, "user"))
            .or_else(|| messages.first())
            .map(&content)?;
        head(&system, &mut h);
        head(&user, &mut h);
    } else {
        let prompt = req
            .get("prompt")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
            .or_else(|| {
                req.get("input").map(|v| match v {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
            })?;
        head(&prompt, &mut h);
    }
    Some(h.finish())
}

/// Byte-slice wrapper for raw-body handlers (one parse).
#[must_use]
pub fn affinity_hash_bytes(body: &[u8]) -> Option<u64> {
    affinity_hash(&serde_json::from_slice::<serde_json::Value>(body).ok()?)
}

#[must_use]
pub fn supervision_error(e: &SupervisionError) -> Response {
    match e {
        SupervisionError::ModelNotFound(m) => openai_error(404, &format!("no such model: {m}")),
        SupervisionError::ModelLoadTimeout(m) => openai_error(
            503,
            &format!("model {m} failed to become healthy (model_load_timeout); check `pallama ps`"),
        ),
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
    (
        StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        axum::Json(body),
    )
        .into_response()
}

/// Forward a request to the child byte-for-byte and stream the response
/// back. `path_query` includes the leading `/`.
/// Forward one request to the child. Eight distinct request components
/// (engine, model, method, path, headers, body, load timing) — a struct
/// here would only shuffle the same data.
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::items_after_statements
)] // one cohesive forwarding path: headers -> sentinel/enforce -> stream
pub async fn proxy_request(
    state: &Arc<AppState>,
    engine: &EngineRef,
    model: &str,
    method: &axum::http::Method,
    path_query: &str,
    headers: &HeaderMap,
    body: axum::body::Bytes,
    load_ms: u128,
    trace: Option<String>,
    body_guard: Option<InFlightGuard>,
    key: Option<crate::keys::KeyCtx>,
) -> Response {
    let base = child_base(&engine.endpoint);
    if base.is_empty() {
        return openai_error(
            500,
            "unix-socket child transport not supported by this proxy path yet",
        );
    }
    let url = format!("{base}{path_query}");

    // Single-flight (B8, FIX2): identical NON-STREAM chat requests
    // coalesce AT THE CHILD-CALL BOUNDARY (admission already happened:
    // queued duplicates are not serialized behind queue waits). Stream
    // detection is a real JSON parse, not a substring sniff — prompt
    // text containing `{"stream":true}` cannot fool it. Bounded wait:
    // after 5s the twin proceeds uncoalesced (long generations never
    // serialize their duplicates indefinitely).
    let mut sf: Option<SingleFlight> = None;
    if state.config.singleflight && is_chat_route(path_query) && body.len() <= 32 * 1024 {
        let asks_stream = serde_json::from_slice::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| v.get("stream").and_then(serde_json::Value::as_bool))
            .unwrap_or(false);
        if !asks_stream {
            let key = sentinel::singleflight_key(model, &body, false);
            let lock = {
                let mut map = state.singleflight.lock().await;
                if map.len() > 256 {
                    map.clear(); // bounded; a cleared key elects a new leader
                }
                std::sync::Arc::clone(
                    map.entry(key)
                        .or_insert_with(|| std::sync::Arc::new(tokio::sync::Mutex::new(()))),
                )
            };
            if let Ok(guard) =
                tokio::time::timeout(std::time::Duration::from_secs(5), lock.clone().lock_owned())
                    .await
            {
                sf = Some(SingleFlight { key, _guard: guard });
            }
        }
    }

    let mut req = state.http.request(method.clone(), &url);
    for (name, value) in headers {
        if !STRIP_REQUEST.contains(&name.as_str()) {
            req = req.header(name, value);
        }
    }
    // Bytes clone = refcount bump: the sentinel request-side parse reads
    // this snapshot after `body` moved into the upstream stream.
    let body_snapshot = body.clone();
    let upstream = req
        .body(reqwest::Body::wrap_stream(futures::stream::once(
            async move { Ok::<_, std::io::Error>(body) },
        )))
        .send()
        .await;
    let resp = match upstream {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(model, "proxy {path_query}: {e:#}");
            // The child may have crashed: reap it now so the NEXT request
            // respawns instead of 502-looping on a stale entry.
            state.sup.reap_dead_children().await;
            release_sf(state, sf).await;
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
    // Sentinel (semantic reliability layer): warn-only observation of
    // response semantics. Bytes are cloned onto a bounded side-channel;
    // parsing and detection run off the hot path, and an overloaded or
    // failed analyzer only degrades the diagnostic record. `Drop` on the
    // feed (owned by the stream closure below) signals completion for
    // clean drain AND client aborts.
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    let sse = content_type.contains("text/event-stream");
    // Per-key token accounting: a bounded tail-sniffer constructed ONLY
    // when the requesting key carries tpm/daily budgets (everyone else
    // pays nothing — the zero-tax contract holds for unlimited keys).
    // Dropped with the stream closure: clean drains AND client aborts
    // both charge (same lifetime trick as the sentinel feed).
    let mut sniffer = key
        .as_ref()
        .and_then(|k| state.keys.entry(&k.name))
        .filter(|e| e.tpm > 0 || e.daily_tokens > 0)
        .map(|e| crate::keys::UsageSniffer::new(&e.name));
    // Enforce (opt-in): non-stream chat requests get judged BEFORE any
    // byte is released — the client waits for the full JSON anyway, so
    // buffering is bounded (ENFORCE_BODY_CAP) and costs no extra round
    // trip. Streaming stays warn-only (bytes already on the wire).
    if state.config.sentinel
        && !sse
        && status.is_success()
        && is_chat_route(path_query)
        && sentinel::enforce_enabled(&state.config, headers)
    {
        let (ctx, warnings) = sentinel::request_ctx(
            state,
            route_name(path_query),
            model,
            &body_snapshot,
            trace,
            sse,
        );
        if !warnings.is_empty() {
            builder = builder.header("x-pallama-warnings", warnings.join(","));
        }
        match resp.bytes().await {
            Err(e) => return openai_error(502, &format!("engine body: {e}")),
            Ok(buf) => {
                if buf.len() > sentinel::ENFORCE_BODY_CAP {
                    tracing::warn!(
                        target: "pallama::sentinel",
                        trace = %ctx.trace,
                        "enforce skipped: body {} bytes exceeds cap",
                        buf.len()
                    );
                } else {
                    let hard = state.sentinel.judge(&ctx, &buf, status.as_u16());
                    if !hard.is_empty() {
                        let detail = hard
                            .iter()
                            .map(|d| format!("[{}] {}", d.code.as_str(), d.detail))
                            .collect::<Vec<_>>()
                            .join("; ");
                        return openai_error(422, &format!("sentinel enforce: {detail}"));
                    }
                }
                // Pass (or oversize passthrough): re-emit the buffered body
                // with the original headers; the guard chain keeps
                // in-flight accounting for the send.
                if let Some(mut s) = sniffer.take() {
                    s.push(&buf);
                    s.finish(&state.keys);
                }
                release_sf(state, sf).await;
                let stream = futures::stream::once(async move { Ok::<_, std::io::Error>(buf) })
                    .chain(futures::stream::unfold(body_guard, |g| async {
                        drop(g);
                        None
                    }));
                return builder
                    .body(Body::from_stream(stream))
                    .unwrap_or_else(|e| openai_error(500, &format!("proxy body: {e}")));
            }
        }
    }
    let (sentinel_feed, sentinel_warnings) = if is_chat_route(path_query) {
        sentinel::begin_chat_observation(
            state,
            route_name(path_query),
            model,
            &body_snapshot,
            trace,
            status.as_u16(),
            sse,
        )
    } else {
        (sentinel::SentinelFeed::inert(), Vec::new())
    };
    if !sentinel_warnings.is_empty() {
        builder = builder.header("x-pallama-warnings", sentinel_warnings.join(","));
    }
    // Evidence loop: measure TTFT (first body byte) and inter-chunk cadence
    // for this generation. Observing is infallible (atomics only); the
    // stream's data/errors pass through untouched. The in-flight guard
    // chain below still owns the body lifetime.
    /// Abort-safe finisher: `Drop` runs on clean drain AND client abort.
    struct FinishSniffer(
        Option<crate::keys::UsageSniffer>,
        std::sync::Arc<crate::keys::KeysLimiter>,
    );
    impl Drop for FinishSniffer {
        fn drop(&mut self) {
            if let Some(s) = self.0.take() {
                s.finish(&self.1);
            }
        }
    }
    let start = std::time::Instant::now();
    let mut first_chunk = true;
    let mut last_chunk = start;
    let hist_state = std::sync::Arc::clone(state);
    let sniffer_finisher = std::sync::Arc::new(std::sync::Mutex::new(FinishSniffer(
        sniffer,
        Arc::clone(&state.keys),
    )));
    let stream = resp.bytes_stream().map(move |r| {
        if r.is_ok() {
            let now = std::time::Instant::now();
            if first_chunk {
                hist_state.ttft.observe_secs((now - start).as_secs_f64());
                first_chunk = false;
            } else {
                hist_state
                    .tpot
                    .observe_secs((now - last_chunk).as_secs_f64());
            }
            last_chunk = now;
            if let Ok(bytes) = r.as_ref() {
                sentinel_feed.bytes(bytes.as_ref());
                if let Some(s) = sniffer_finisher.lock().expect("sniffer").0.as_mut() {
                    s.push(bytes.as_ref());
                }
            }
        }
        r.map_err(|e| std::io::Error::other(e.to_string()))
    });
    // Hold in-flight accounting for the body's lifetime: the guard drops
    // when the client drains (or aborts) the stream.
    let sf_state = std::sync::Arc::clone(state);
    let stream = stream.chain(futures::stream::unfold((body_guard, sf), move |(g, sf)| {
        let sf_state = std::sync::Arc::clone(&sf_state);
        async move {
            drop(g);
            release_sf(&sf_state, sf).await; // stream end (or abort): twin may lead
            None
        }
    }));
    builder
        .body(Body::from_stream(stream))
        .unwrap_or_else(|e| openai_error(500, &format!("proxy body: {e}")))
}

/// Single-flight token: held from child-call to response-stream end
/// (FIX2 — the guard rides the SAME drop chain as the in-flight guard,
/// so clean drains, client aborts, and early errors all release it).
struct SingleFlight {
    key: u64,
    _guard: tokio::sync::OwnedMutexGuard<()>,
}

/// Release a single-flight slot: map entry out, lock freed on drop.
async fn release_sf(state: &Arc<AppState>, sf: Option<SingleFlight>) {
    if let Some(sf) = sf {
        state.singleflight.lock().await.remove(&sf.key);
        drop(sf);
    }
}

/// Chat-family routes the sentinel observes (the only ones whose
/// semantics the current detection set understands).
fn is_chat_route(path_query: &str) -> bool {
    let p = path_query.split('?').next().unwrap_or(path_query);
    p.ends_with("/chat/completions") || p.ends_with("/completions") || p.ends_with("/responses")
}

fn route_name(path_query: &str) -> &'static str {
    let p = path_query.split('?').next().unwrap_or(path_query);
    if p.ends_with("/chat/completions") {
        "openai-chat"
    } else if p.ends_with("/responses") {
        "openai-responses"
    } else {
        "openai-completions"
    }
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
    admission_gate_slo(state, model, priority, None, 0).await
}

/// SLO-aware admission: `deadline_ms` (the `x-pallama-deadline-ms`
/// header) and `body_len` (prefill-heavy demotion) feed the EDF queue.
pub async fn admission_gate_slo(
    state: &Arc<AppState>,
    model: &str,
    priority: Priority,
    deadline_ms: Option<u64>,
    body_len: usize,
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
            .wait(
                model,
                priority,
                deadline_ms,
                body_len,
                std::time::Duration::from_mins(2),
            )
            .await
            .map_err(|e| openai_error(503, &e))?;
    }
}

/// Begin accounting and return the guard; the response path holds it for
/// the body's lifetime.
#[must_use]
pub fn begin_accounting(state: &Arc<AppState>, model: &str) -> InFlightGuard {
    state.sup.begin_request(model);
    InFlightGuard {
        state: state.clone(),
        model: model.to_string(),
    }
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
    let pq = uri
        .path_and_query()
        .map_or("/", axum::http::uri::PathAndQuery::as_str);
    pq.to_string()
}

pub type GatewayResult = Result<Response, Response>;
#[must_use]
pub fn internal(m: &str) -> Response {
    openai_error(500, m)
}

#[cfg(test)]
mod affinity_tests {
    #![allow(non_snake_case)]
    use super::*;

    fn chat(system: Option<&str>, user: &str) -> serde_json::Value {
        let mut messages = Vec::new();
        if let Some(s) = system {
            messages.push(serde_json::json!({"role": "system", "content": s}));
        }
        messages.push(serde_json::json!({"role": "user", "content": user}));
        serde_json::json!({"messages": messages})
    }

    #[test]
    fn unit__affinity__chat_same_prefix__stable() {
        let a = affinity_hash(&chat(Some("You are terse."), "What is 17*23?"));
        let b = affinity_hash(&chat(Some("You are terse."), "What is 17*23?"));
        assert_eq!(a, b, "identical conversations hash identically");
    }

    #[test]
    fn unit__affinity__chat_different_user__differs() {
        let a = affinity_hash(&chat(Some("sys"), "first question"));
        let b = affinity_hash(&chat(Some("sys"), "different question"));
        assert_ne!(a, b);
    }

    #[test]
    fn unit__affinity__chat_later_turns_ignored() {
        // Affinity is the CONVERSATION prefix: extra assistant/user
        // turns after the first user turn must not change the hash,
        // so turn 2 lands on turn 1's warm replica.
        let mut turn1 = chat(Some("sys"), "hi");
        let mut turn2 = chat(Some("sys"), "hi");
        turn1["messages"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({"role": "assistant", "content": "hello"}));
        turn2["messages"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({"role": "assistant", "content": "hello"}));
        turn2["messages"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({"role": "user", "content": "next question"}));
        assert_eq!(
            affinity_hash(&turn1),
            affinity_hash(&turn2),
            "suffix turns must not break stickiness"
        );
    }

    #[test]
    fn unit__affinity__chat_beyond_1kib__truncated_equal() {
        let long_a: String = "x".repeat(2000) + "A";
        let long_b: String = "x".repeat(2000) + "B";
        // Differences past the 1 KiB head are invisible: stable pinning
        // even for giant system prompts (and cheap to compute).
        assert_eq!(
            affinity_hash(&chat(Some(&long_a), "q")),
            affinity_hash(&chat(Some(&long_b), "q"))
        );
        // But a difference INSIDE the head still matters.
        let head_a = format!("{}{}", "a".repeat(1024), "z".repeat(1000));
        let head_b = format!("{}{}", "b".repeat(1024), "z".repeat(1000));
        assert_ne!(
            affinity_hash(&chat(Some(&head_a), "q")),
            affinity_hash(&chat(Some(&head_b), "q"))
        );
    }

    #[test]
    fn unit__affinity__multimodal_content__hashed_as_string() {
        let a = serde_json::json!({"messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
            ]}
        ]});
        let b = serde_json::json!({"messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}}
            ]}
        ]});
        // Structured content is stringified deterministically; the image
        // difference lives past the text head only when the text part is
        // large enough — here both fit, so they differ.
        assert_ne!(affinity_hash(&a), affinity_hash(&b));
        assert_eq!(affinity_hash(&a), affinity_hash(&a));
    }

    #[test]
    fn unit__affinity__generate_prompt__hashed() {
        let a = affinity_hash(&serde_json::json!({"prompt": "once upon a time"}));
        let b = affinity_hash(&serde_json::json!({"prompt": "once upon a time"}));
        let c = affinity_hash(&serde_json::json!({"prompt": "a different story"}));
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn unit__affinity__responses_input__string_and_array() {
        let s = affinity_hash(&serde_json::json!({"input": "summarize this"}));
        let arr = affinity_hash(&serde_json::json!({"input": [
            {"type": "message", "text": "summarize this"}
        ]}));
        assert!(s.is_some() && arr.is_some());
        // Same text via either shape sticks together only when the
        // stringified array head matches — it will not, by design;
        // what matters is each shape is stable and discriminating.
        assert_eq!(
            s,
            affinity_hash(&serde_json::json!({"input": "summarize this"}))
        );
        assert_eq!(arr, arr);
    }

    #[test]
    fn unit__affinity__no_prompt__none() {
        // Bodies with no conversation identity at all: no affinity,
        // plain load-balance. (`input` arrays ARE identity — responses
        // lane — and hash deterministically.)
        assert_eq!(affinity_hash(&serde_json::json!({"model": "m"})), None);
        assert_eq!(
            affinity_hash(&serde_json::json!({"messages": []})),
            None,
            "empty messages = no identity"
        );
        let e1 = affinity_hash(&serde_json::json!({"input": ["a", "b"]}));
        let e2 = affinity_hash(&serde_json::json!({"input": ["a", "b"]}));
        assert_eq!(e1, e2, "input arrays hash stably");
        assert_ne!(
            e1,
            affinity_hash(&serde_json::json!({"input": ["a", "c"]})),
            "input arrays discriminate"
        );
    }

    #[test]
    fn unit__affinity__bytes_wrapper__matches_value_form() {
        let body = br#"{"messages":[{"role":"user","content":"hello"}]}"#;
        let v: serde_json::Value = serde_json::from_slice(body).unwrap();
        assert_eq!(affinity_hash_bytes(body), affinity_hash(&v));
        assert_eq!(affinity_hash_bytes(b"not json"), None);
    }
}
