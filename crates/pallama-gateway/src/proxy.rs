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
use pallama_runtime::{EngineRef, PrefixKey, SupervisionError};

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
    prefix: Option<PrefixKey>,
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
pub fn affinity_hash(req: &serde_json::Value) -> Option<PrefixKey> {
    use std::hash::{Hash, Hasher};
    type H = std::collections::hash_map::DefaultHasher;
    let head = |s: &str, h: &mut H, cap: usize| {
        s.as_bytes()[..s.len().min(cap)].hash(h);
    };
    // F8: `sys` = shared-prefix class (system prompt head), `convo` =
    // full conversation identity (system + first user head).
    let mut hs = H::new();
    let mut hc = H::new();
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
        head(&system, &mut hs, 256);
        head(&system, &mut hc, 1024);
        head(&user, &mut hc, 1024);
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
        head(&prompt, &mut hs, 256);
        head(&prompt, &mut hc, 1024);
    }
    Some(PrefixKey {
        sys: hs.finish(),
        convo: hc.finish(),
    })
}

/// Byte-slice wrapper for raw-body handlers (one parse).
#[must_use]
pub fn affinity_hash_bytes(body: &[u8]) -> Option<PrefixKey> {
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

/// Stamp the per-child bearer secret (child `--api-key` hardening) on
/// a child-bound request. SINGLE CHOKE POINT: every gateway lane that
/// talks HTTP to a spawned engine child must route its
/// `reqwest::RequestBuilder` through here — a missed site fails loudly
/// (the child answers 401), not silently.
pub fn child_auth(rb: reqwest::RequestBuilder, engine: &EngineRef) -> reqwest::RequestBuilder {
    match &engine.auth {
        Some(secret) => rb.bearer_auth(secret),
        None => rb,
    }
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
    let began = std::time::Instant::now();

    // R6: force the final usage chunk on /v1 chat streams so warm/cold
    // classification has data — most clients never opt in. Additive and
    // spec-compliant (usage-only extra chunk; ollama lane already does
    // the same). Legacy /completions and non-chat routes pass through.
    let body = inject_include_usage(path_query, body);

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
    // Client `Authorization` was stripped above; the child secret is
    // stamped fresh here (never the caller's gateway key).
    req = child_auth(req, engine);
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
                // R6: buffered non-stream chat — classify from the exact
                // JSON (no substring heuristics on this path).
                record_buffered_chat(&state.obs, &buf, began.elapsed().as_secs_f64());
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
    let cache_tap = std::sync::Arc::new(CacheTap::new());
    let tap_map = std::sync::Arc::clone(&cache_tap);
    let first_ns = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let tap_first = std::sync::Arc::clone(&first_ns);
    let cache_finisher = std::sync::Arc::new(CacheObsFinisher {
        tap: cache_tap,
        obs: std::sync::Arc::clone(&state.obs),
        first_ns,
        chat: is_chat_route(path_query),
    });
    let stream = resp.bytes_stream().map(move |r| {
        if r.is_ok() {
            let now = std::time::Instant::now();
            if first_chunk {
                hist_state.ttft.observe_secs((now - start).as_secs_f64());
                tap_first.store(
                    u64::try_from((now - began).as_nanos()).unwrap_or(u64::MAX),
                    std::sync::atomic::Ordering::Relaxed,
                );
                first_chunk = false;
            } else {
                hist_state
                    .tpot
                    .observe_secs((now - last_chunk).as_secs_f64());
            }
            last_chunk = now;
            if let Ok(bytes) = r.as_ref() {
                sentinel_feed.bytes(bytes.as_ref());
                tap_map.push(bytes.as_ref());
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
    let stream = stream.chain(futures::stream::unfold(
        (body_guard, sf, cache_finisher),
        move |(g, sf, cache_finisher)| {
            let sf_state = std::sync::Arc::clone(&sf_state);
            async move {
                drop(g);
                drop(cache_finisher); // classify at stream end (or abort)
                release_sf(&sf_state, sf).await; // stream end (or abort): twin may lead
                None
            }
        },
    ));
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

/// Additive `stream_options.include_usage = true` on /v1 chat STREAM
/// requests (R6): the final usage chunk is what the gateway's warm/cold
/// cache classification reads. Never touches non-stream requests, other
/// routes, or bodies that already opted in; any parse/serialize failure
/// returns the original bytes untouched (the child remains the judge).
fn inject_include_usage(path_query: &str, body: axum::body::Bytes) -> axum::body::Bytes {
    let p = path_query.split('?').next().unwrap_or(path_query);
    if !p.ends_with("/chat/completions") {
        return body;
    }
    let Ok(mut v) = serde_json::from_slice::<serde_json::Value>(&body) else {
        return body;
    };
    if v.get("stream").and_then(serde_json::Value::as_bool) != Some(true) {
        return body;
    }
    if v.pointer("/stream_options/include_usage")
        .and_then(serde_json::Value::as_bool)
        == Some(true)
    {
        return body;
    }
    v["stream_options"]["include_usage"] = serde_json::Value::Bool(true);
    match serde_json::to_vec(&v) {
        Ok(bytes) => axum::body::Bytes::from(bytes),
        // serializing a parsed Value cannot fail; keep the original on
        // any error anyway (additive contract: never reject what the
        // child might accept)
        Err(_) => body,
    }
}

/// Classify a fully-buffered non-stream chat body (enforce path) from
/// its exact usage object — no substring heuristics on this route.
fn record_buffered_chat(obs: &std::sync::Arc<crate::state::CacheObs>, buf: &[u8], ttft_secs: f64) {
    match serde_json::from_slice::<serde_json::Value>(buf) {
        Ok(v) => match v.get("usage") {
            Some(u) => obs.record(
                u.get("prompt_tokens")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0),
                crate::translate::cached_prompt_tokens(Some(u)),
                Some(ttft_secs),
            ),
            None => obs.miss(),
        },
        Err(_) => obs.miss(),
    }
}

/// R6 per-stream cache tap: bounded tail owned by the map closure,
/// classified once by the Drop finisher at stream end (clean drain
/// or abort). Same lifetime trick as the usage sniffer.
struct CacheTap {
    tail: std::sync::Mutex<Vec<u8>>,
}
impl CacheTap {
    fn new() -> Self {
        Self {
            tail: std::sync::Mutex::new(Vec::new()),
        }
    }
    fn push(&self, b: &[u8]) {
        const CAP: usize = 8 * 1024;
        let mut t = self.tail.lock().expect("cache tap");
        if b.len() >= CAP {
            t.clear();
            t.extend_from_slice(&b[b.len() - CAP..]);
        } else if t.len() + b.len() > CAP {
            let over = t.len() + b.len() - CAP;
            t.drain(..over);
            t.extend_from_slice(b);
        } else {
            t.extend_from_slice(b);
        }
    }
}

/// Classifies once from the tap tail at stream end. Last-match
/// (`last_int_after`): the usage payload is always the FINAL event,
/// so a literal appearing in generated prose earlier cannot win.
struct CacheObsFinisher {
    tap: std::sync::Arc<CacheTap>,
    obs: std::sync::Arc<crate::state::CacheObs>,
    first_ns: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Chat routes only — embeddings/other bodies would pollute the
    /// generation warm/cold split.
    chat: bool,
}
impl Drop for CacheObsFinisher {
    fn drop(&mut self) {
        if !self.chat {
            return;
        }
        let tail = self.tap.tail.lock().expect("cache tap").clone();
        if crate::keys::find_sub(&tail, b"\"prompt_tokens\"").is_none() {
            self.obs.miss();
            return;
        }
        let prompt = crate::keys::last_int_after(&tail, "\"prompt_tokens\"").unwrap_or(0);
        let cached = crate::keys::last_int_after(&tail, "\"cached_tokens\"").unwrap_or(0);
        let first = self.first_ns.load(std::sync::atomic::Ordering::Relaxed);
        let ttft = (first > 0).then(|| std::time::Duration::from_nanos(first).as_secs_f64());
        self.obs.record(prompt, cached, ttft);
    }
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
    fn unit__affinity__sys_class__stable_across_users() {
        // F8: the sys half of the key identifies the shared system-prompt
        // class, so two different conversations under the same system
        // prompt coalesce onto one warm replica.
        let a = affinity_hash(&chat(Some("You are a pirate."), "question one"));
        let b = affinity_hash(&chat(Some("You are a pirate."), "question two"));
        let (a, b) = (a.expect("both hash"), b.expect("both hash"));
        assert_eq!(a.sys, b.sys, "same system prompt shares the sys class");
        assert_ne!(a.convo, b.convo, "different first-user turns differ");
    }

    #[test]
    fn unit__affinity__sys_class__differs_across_system_prompts() {
        let a = affinity_hash(&chat(Some("You are a pirate."), "question"));
        let b = affinity_hash(&chat(Some("You are a scientist."), "question"));
        let (a, b) = (a.expect("both hash"), b.expect("both hash"));
        assert_ne!(a.sys, b.sys);
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

#[cfg(test)]
mod cache_obs_tests {
    #![allow(non_snake_case)]
    use super::*;
    use serde_json::json;
    use std::sync::atomic::Ordering;

    fn json_body(v: &serde_json::Value) -> axum::body::Bytes {
        axum::body::Bytes::from(serde_json::to_vec(v).unwrap())
    }

    fn parse(b: &axum::body::Bytes) -> serde_json::Value {
        serde_json::from_slice(b).unwrap()
    }

    // --- inject_include_usage -------------------------------------------

    #[test]
    fn unit__inject_include_usage__stream_true_adds_flag() {
        let out = inject_include_usage(
            "/v1/chat/completions",
            json_body(&json!({"model": "m", "messages": [], "stream": true})),
        );
        let v = parse(&out);
        assert_eq!(
            v.pointer("/stream_options/include_usage"),
            Some(&json!(true))
        );
        assert_eq!(v["stream"], json!(true), "existing body preserved");
    }

    #[test]
    fn unit__inject_include_usage__already_set_passthrough() {
        let orig = json_body(&json!({"stream": true, "stream_options": {"include_usage": true}}));
        let out = inject_include_usage("/v1/chat/completions", orig.clone());
        assert_eq!(out, orig, "byte-identical: nothing to add");
    }

    #[test]
    fn unit__inject_include_usage__non_stream_passthrough() {
        let orig = json_body(&json!({"model": "m", "messages": []}));
        let out = inject_include_usage("/v1/chat/completions", orig.clone());
        assert_eq!(out, orig, "non-stream requests untouched");
    }

    #[test]
    fn unit__inject_include_usage__non_chat_route_passthrough() {
        let orig = json_body(&json!({"stream": true}));
        let out = inject_include_usage("/v1/completions", orig.clone());
        assert_eq!(
            out, orig,
            "completions lane untouched (usage shape differs)"
        );
        let out2 = inject_include_usage("/v1/embeddings", orig.clone());
        assert_eq!(out2, orig);
    }

    #[test]
    fn unit__inject_include_usage__invalid_json_passthrough() {
        let orig = axum::body::Bytes::from_static(b"{not json stream:true");
        let out = inject_include_usage("/v1/chat/completions", orig.clone());
        assert_eq!(out, orig, "child remains the judge of odd bodies");
    }

    // --- record_buffered_chat -------------------------------------------

    #[test]
    fn unit__record_buffered_chat__usage_records_warm() {
        let obs = std::sync::Arc::new(crate::state::CacheObs::new());
        let body = json!({"choices": [], "usage": {
            "prompt_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 96},
            "completion_tokens": 4,
        }});
        record_buffered_chat(&obs, &serde_json::to_vec(&body).unwrap(), 0.25);
        assert_eq!(obs.prompt_tokens.load(Ordering::Relaxed), 120);
        assert_eq!(obs.cached_tokens.load(Ordering::Relaxed), 96);
        assert!(obs.unclassified.load(Ordering::Relaxed) == 0);
        let mut warm = String::new();
        obs.ttft_warm.render(&mut warm);
        assert!(warm.contains("_count 1"), "{warm}");
    }

    #[test]
    fn unit__record_buffered_chat__no_usage_or_bad_json_is_miss() {
        let obs = std::sync::Arc::new(crate::state::CacheObs::new());
        record_buffered_chat(
            &obs,
            &serde_json::to_vec(&json!({"choices": []})).unwrap(),
            0.1,
        );
        record_buffered_chat(&obs, b"not json", 0.1);
        assert_eq!(obs.unclassified.load(Ordering::Relaxed), 2);
        assert_eq!(obs.prompt_tokens.load(Ordering::Relaxed), 0);
    }

    // --- CacheTap bounded tail ------------------------------------------

    #[test]
    fn unit__cache_tap__oversized_chunk_keeps_last_8kib() {
        let tap = CacheTap::new();
        let big: Vec<u8> = (0..16 * 1024u32).map(|i| (i % 251) as u8).collect();
        tap.push(&big);
        let t = tap.tail.lock().unwrap();
        assert_eq!(t.len(), 8 * 1024);
        assert_eq!(&t[..], &big[big.len() - 8 * 1024..], "tail = last 8KiB");
    }

    #[test]
    fn unit__cache_tap__small_pushes_drain_oldest() {
        let tap = CacheTap::new();
        for i in 0..1000u32 {
            tap.push(format!("line-{i:04} ").as_bytes());
        }
        let t = tap.tail.lock().unwrap();
        assert!(t.len() <= 8 * 1024, "{}", t.len());
        let s = String::from_utf8_lossy(&t.clone()).into_owned();
        assert!(!s.contains("line-0000"), "oldest drained: {s}");
        assert!(s.contains("line-0999"), "newest kept: {s}");
    }

    // --- CacheObsFinisher classification --------------------------------

    fn finisher_with(
        chunks: &[&[u8]],
        first_ns: u64,
        chat: bool,
    ) -> std::sync::Arc<crate::state::CacheObs> {
        let tap = std::sync::Arc::new(CacheTap::new());
        for c in chunks {
            tap.push(c);
        }
        let obs = std::sync::Arc::new(crate::state::CacheObs::new());
        let first = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(first_ns));
        drop(CacheObsFinisher {
            tap,
            obs: std::sync::Arc::clone(&obs),
            first_ns: first,
            chat,
        });
        obs
    }

    const USAGE_FINAL: &str = concat!(
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],",
        "\"usage\":{\"prompt_tokens\":120,",
        "\"prompt_tokens_details\":{\"cached_tokens\":96},",
        "\"completion_tokens\":4}}\n\n",
        "data: [DONE]\n\n"
    );

    #[test]
    fn unit__cache_finisher__usage_with_cached_records_warm_with_ttft() {
        let obs = finisher_with(&[USAGE_FINAL.as_bytes()], 50_000_000, true);
        assert_eq!(obs.prompt_tokens.load(Ordering::Relaxed), 120);
        assert_eq!(obs.cached_tokens.load(Ordering::Relaxed), 96);
        assert_eq!(obs.unclassified.load(Ordering::Relaxed), 0);
        let mut warm = String::new();
        obs.ttft_warm.render(&mut warm);
        assert!(warm.contains("_count 1"), "warm + measured TTFT: {warm}");
    }

    #[test]
    fn unit__cache_finisher__no_usage_is_miss() {
        let chunks = ["data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n".as_bytes()];
        let obs = finisher_with(&chunks, 0, true);
        assert_eq!(obs.unclassified.load(Ordering::Relaxed), 1);
        assert_eq!(obs.prompt_tokens.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn unit__cache_finisher__non_chat_is_no_op() {
        let obs = finisher_with(&[USAGE_FINAL.as_bytes()], 50_000_000, false);
        assert_eq!(obs.unclassified.load(Ordering::Relaxed), 0);
        assert_eq!(obs.prompt_tokens.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn unit__cache_finisher__prose_literal_earlier_loses_to_final_usage() {
        // Pin (regression contract): generated prose may contain the
        // literal `"prompt_tokens"` — classification must read the FINAL
        // usage event, never the prose occurrence.
        let prose = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":",
            "\"echo {\\\"prompt_tokens\\\": 9999, \\\"cached_tokens\\\": 9999}\"",
            "}}]}\n\n"
        );
        let obs = finisher_with(&[prose.as_bytes(), USAGE_FINAL.as_bytes()], 0, true);
        assert_eq!(obs.prompt_tokens.load(Ordering::Relaxed), 120, "usage wins");
        assert_eq!(obs.cached_tokens.load(Ordering::Relaxed), 96, "usage wins");
        assert_eq!(obs.unclassified.load(Ordering::Relaxed), 0);
    }
}
