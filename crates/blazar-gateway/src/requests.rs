//! Request lifecycle API: in-flight generation cards + programmatic
//! cancel/interrupt (`/v1/requests`). v0.15 reliability layer.
//!
//! Client-disconnect cancellation has always worked implicitly (dropping
//! the handler future drops the upstream stream and the engine frees the
//! slot). This module makes the SAME mechanism addressable by id from a
//! SECOND connection — what IDE agents and orchestration tools need when
//! a generation runs long: list what is in flight, cancel it, keep the
//! partial output (`interrupt`), all without killing the client socket.
//!
//! Design laws (mirrors [`crate::whisper::AudioJobs`]):
//! - The registry is a [`dashmap::DashMap`] of live entries plus a bounded
//!   terminal ring; every method is short and synchronous — never a lock
//!   across an await.
//! - Cancellation travels over a `tokio::sync::watch` bool. The buffered
//!   path races the handler future with the token (dropping the future is
//!   exactly the disconnect semantics the proxy already guarantees); the
//!   streaming path races each body chunk and ends the body cleanly.
//! - The cancel ROUTE only records intent and fires the token; the party
//!   holding the request (select! or body wrapper) performs the stop and
//!   the terminal transition — one finisher, no races.
//! - SSE bodies get the protocol-standard `data: [DONE]` sentinel on
//!   truncation (it carries no content — no fabricated tokens). ndjson
//!   and raw bodies just end; clients detect EOF. We never synthesize a
//!   fake final record for any dialect.
//! - Honesty over completeness: `model` comes from a bounded body sniff,
//!   `engine` is omitted until attribution lands with `/api/capacity`,
//!   `tokens_generated` is only set when a buffered usage tail was
//!   parsed — never estimated.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU16, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::{Path, RawQuery, State};
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use dashmap::DashMap;
use futures::stream::{StreamExt, unfold};
use tokio::sync::watch;

use crate::keys::KeyCtx;
use crate::state::AppState;

/// Bounded partial-output capture per request (bytes, not tokens — chunks
/// may batch tokens). Enough to show the agent what was generated before
/// the cut; the full text is the client's own stream.
const PARTIAL_CAP_BYTES: usize = 64 * 1024;
/// Terminal cards kept for post-mortem listing (bounded ring).
pub const TERMINAL_RING_CAP: usize = 256;
/// Model sniff ceiling: bodies larger than this skip the sniff (model
/// stays unknown) — parsing multi-MiB multimodal payloads per request
/// would tax the hot path for a nicety.
const MODEL_SNIFF_CAP_BYTES: usize = 8 * 1024 * 1024;
/// Buffered usage-tail parse ceiling (JSON responses only).
const USAGE_SNIFF_CAP_BYTES: usize = 256 * 1024;

/// Card state machine (stored as an atomic so every sharer sees flips).
/// `disconnected` is the honest terminal for streams the CLIENT abandoned
/// (middleware body dropped before natural end).
const STATE_QUEUED: u8 = 0;
const STATE_RUNNING: u8 = 1;
const STATE_DONE: u8 = 2;
const STATE_CANCELLED: u8 = 3;
const STATE_INTERRUPTED: u8 = 4;
const STATE_DISCONNECTED: u8 = 5;

/// Stop intent, recorded by the cancel routes ahead of the actual stop.
const STOP_NONE: u8 = 0;
const STOP_CANCEL: u8 = 1;
const STOP_INTERRUPT: u8 = 2;

fn state_name(s: u8) -> &'static str {
    match s {
        STATE_QUEUED => "queued",
        STATE_RUNNING => "running",
        STATE_DONE => "done",
        STATE_CANCELLED => "cancelled",
        STATE_INTERRUPTED => "interrupted",
        STATE_DISCONNECTED => "disconnected",
        _ => "unknown",
    }
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

fn boot_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos())
}

/// One tracked request. Fully interior-mutable: the middleware, the body
/// wrapper, and the cancel routes all share a single live card with no
/// `&mut` paths — every field is an atomic or a short-lived mutex.
pub struct RequestCard {
    pub id: String,
    pub trace_id: String,
    pub path: String,
    model: Mutex<Option<String>>,
    key: Mutex<Option<String>>,
    state: AtomicU8,
    stop: AtomicU8,
    /// HTTP status once a response exists (0 until then).
    status: AtomicU16,
    created_ms: u64,
    started_ms: AtomicU64,
    finished_ms: AtomicU64,
    /// Buffered usage total when known (0 = unknown, never estimated).
    tokens: AtomicU64,
    /// One-shot latch: the first `retire` files the card in the ring.
    retired: AtomicBool,
    partial: Mutex<Vec<u8>>,
}

impl RequestCard {
    fn new(id: String, trace_id: String, path: String, model: Option<String>) -> Self {
        Self {
            id,
            trace_id,
            path,
            model: Mutex::new(model),
            key: Mutex::new(None),
            state: AtomicU8::new(STATE_QUEUED),
            stop: AtomicU8::new(STOP_NONE),
            status: AtomicU16::new(0),
            created_ms: now_unix_ms(),
            started_ms: AtomicU64::new(0),
            finished_ms: AtomicU64::new(0),
            tokens: AtomicU64::new(0),
            retired: AtomicBool::new(false),
            partial: Mutex::new(Vec::new()),
        }
    }

    pub fn set_model(&self, model: String) {
        *self.model.lock().expect("card model lock") = Some(model);
    }

    pub fn set_key(&self, key: String) {
        *self.key.lock().expect("card key lock") = Some(key);
    }

    fn mark_running(&self) {
        self.started_ms.store(now_unix_ms(), Ordering::Release);
        self.state.store(STATE_RUNNING, Ordering::Release);
    }

    fn is_terminal(&self) -> bool {
        self.state.load(Ordering::Acquire) >= STATE_DONE
    }

    /// Terminal state the recorded stop intent asks for (used by whichever
    /// party performs the stop; defaults to `cancelled`).
    fn stop_terminal(&self) -> u8 {
        match self.stop.load(Ordering::Acquire) {
            STOP_INTERRUPT => STATE_INTERRUPTED,
            _ => STATE_CANCELLED,
        }
    }

    fn capture_partial(&self, bytes: &[u8]) {
        let mut partial = self.partial.lock().expect("card partial lock");
        if partial.len() >= PARTIAL_CAP_BYTES {
            return;
        }
        let room = PARTIAL_CAP_BYTES - partial.len();
        let take = room.min(bytes.len());
        partial.extend_from_slice(&bytes[..take]);
    }

    fn payload(&self, live: bool) -> serde_json::Value {
        let state = state_name(self.state.load(Ordering::Acquire));
        let model = self.model.lock().expect("card model lock").clone();
        let key = self.key.lock().expect("card key lock").clone();
        let tokens = self.tokens.load(Ordering::Acquire);
        let partial_len = self.partial.lock().expect("card partial lock").len();
        let mut v = serde_json::json!({
            "id": self.id,
            "object": "blazar.request",
            "trace_id": self.trace_id,
            "state": state,
            "path": self.path,
            "created_at": self.created_ms / 1000,
        });
        let obj = v.as_object_mut().expect("literal object");
        if let Some(m) = model {
            obj.insert("model".into(), m.into());
        }
        if let Some(k) = key {
            obj.insert("key".into(), k.into());
        }
        let started = self.started_ms.load(Ordering::Acquire);
        if started > 0 {
            obj.insert("queue_time_ms".into(), (started - self.created_ms).into());
        }
        let finished = self.finished_ms.load(Ordering::Acquire);
        if started > 0 && finished >= started {
            obj.insert("compute_time_ms".into(), (finished - started).into());
            obj.insert("finished_at".into(), (finished / 1000).into());
        }
        let status = self.status.load(Ordering::Acquire);
        if status > 0 {
            obj.insert("status".into(), status.into());
        }
        if tokens > 0 {
            obj.insert("tokens_generated".into(), tokens.into());
        }
        if partial_len > 0 {
            obj.insert("partial_bytes".into(), partial_len.into());
            let preview = {
                let partial = self.partial.lock().expect("card partial lock");
                String::from_utf8_lossy(&partial[..partial.len().min(512)]).into_owned()
            };
            obj.insert("partial_preview".into(), preview.into());
        }
        if live {
            obj.insert(
                "cancel_url".into(),
                format!("/v1/requests/{}/cancel", self.id).into(),
            );
            obj.insert(
                "interrupt_url".into(),
                format!("/v1/requests/{}/interrupt", self.id).into(),
            );
        }
        v
    }
}

/// Live entry: the card plus the cancellation sender. The sender dies with
/// the entry (removed at finish), which is exactly the request's lifetime.
struct LiveEntry {
    card: Arc<RequestCard>,
    tx: watch::Sender<bool>,
}

#[derive(Default)]
pub struct RequestRuntime {
    inflight: DashMap<String, LiveEntry>,
    terminal: Mutex<VecDeque<Arc<RequestCard>>>,
    seq: AtomicU64,
    boot_nanos: u128,
}

fn card_payload(card: &RequestCard) -> serde_json::Value {
    card.payload(!card.is_terminal())
}

impl RequestRuntime {
    #[must_use]
    pub fn new() -> Self {
        Self {
            inflight: DashMap::new(),
            terminal: Mutex::new(VecDeque::with_capacity(TERMINAL_RING_CAP)),
            seq: AtomicU64::new(1),
            boot_nanos: boot_nanos(),
        }
    }

    /// Register a card; returns the shared card + the cancel token.
    fn register(
        &self,
        trace_id: String,
        path: &str,
        model: Option<String>,
    ) -> (Arc<RequestCard>, watch::Receiver<bool>) {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let id = format!("req_{:x}_{}", self.boot_nanos, seq);
        let card = Arc::new(RequestCard::new(id, trace_id, path.to_string(), model));
        let (tx, rx) = watch::channel(false);
        self.inflight.insert(
            card.id.clone(),
            LiveEntry {
                card: Arc::clone(&card),
                tx,
            },
        );
        (card, rx)
    }

    /// Record stop intent and fire the token. The party holding the
    /// request (buffered select! or stream wrapper) performs the stop;
    /// this never mutates state or evicts the entry itself.
    fn request_stop(&self, id: &str, interrupt: bool) -> StopOutcome {
        let Some(entry) = self.inflight.get(id) else {
            return self
                .terminal_payload(id)
                .map_or(StopOutcome::NotFound, StopOutcome::AlreadyFinished);
        };
        if entry.card.is_terminal() {
            return StopOutcome::AlreadyFinished(card_payload(&entry.card));
        }
        entry.card.stop.store(
            if interrupt {
                STOP_INTERRUPT
            } else {
                STOP_CANCEL
            },
            Ordering::Release,
        );
        let _ = entry.tx.send(true);
        StopOutcome::Live(card_payload(&entry.card))
    }

    /// Retire a card the holder has already flipped terminal: remove the
    /// live entry (dropping the cancel sender with it) and file the card
    /// in the bounded terminal ring. Idempotent — the natural-end path and
    /// the drop guard may both land here; only the first filing sticks.
    fn retire(&self, card: &Arc<RequestCard>) {
        if !card.retired.swap(true, Ordering::AcqRel) {
            self.inflight.remove(&card.id);
            let mut ring = self.terminal.lock().expect("terminal ring lock");
            if ring.len() == TERMINAL_RING_CAP {
                ring.pop_front();
            }
            ring.push_back(Arc::clone(card));
        }
    }

    fn terminal_payload(&self, id: &str) -> Option<serde_json::Value> {
        let ring = self.terminal.lock().expect("terminal ring lock");
        ring.iter().find(|c| c.id == id).map(|c| c.payload(false))
    }

    /// Listing: live cards (oldest first) + terminal ring (newest first).
    fn list(&self, key_filter: Option<&str>, model_filter: Option<&str>) -> Vec<serde_json::Value> {
        let mut live: Vec<serde_json::Value> = self
            .inflight
            .iter()
            .filter(|e| matches_key(&e.card, key_filter) && matches_model(&e.card, model_filter))
            .map(|e| e.card.payload(true))
            .collect();
        live.sort_by(|a, b| {
            a["id"]
                .as_str()
                .cmp(&b["id"].as_str())
                .then_with(|| a["created_at"].as_u64().cmp(&b["created_at"].as_u64()))
        });
        let terminal: Vec<serde_json::Value> = {
            let ring = self.terminal.lock().expect("terminal ring lock");
            ring.iter()
                .rev()
                .filter(|c| matches_key(c, key_filter) && matches_model(c, model_filter))
                .map(|c| c.payload(false))
                .collect()
        };
        live.into_iter().chain(terminal).collect()
    }

    fn get(&self, id: &str) -> Option<serde_json::Value> {
        if let Some(entry) = self.inflight.get(id) {
            return Some(card_payload(&entry.card));
        }
        self.terminal_payload(id)
    }

    pub fn inflight_count(&self) -> usize {
        self.inflight.len()
    }
}

/// Terminal ring filing works on shared handles, so the card carries its
/// own one-shot `retired` latch (see [`RequestRuntime::retire`]).
fn matches_key(card: &RequestCard, filter: Option<&str>) -> bool {
    match filter {
        None => true,
        Some(want) => card
            .key
            .lock()
            .expect("card key lock")
            .as_deref()
            .is_some_and(|k| k == want),
    }
}

fn matches_model(card: &RequestCard, filter: Option<&str>) -> bool {
    match filter {
        None => true,
        Some(want) => card
            .model
            .lock()
            .expect("card model lock")
            .as_deref()
            .is_some_and(|m| m == want),
    }
}

#[derive(Debug)]
enum StopOutcome {
    NotFound,
    AlreadyFinished(serde_json::Value),
    Live(serde_json::Value),
}

/// Generation lanes the lifecycle middleware tracks. Fast JSON lanes
/// (embeddings, rerank, tokenize) and read surfaces stay out — cards are
/// for requests that can run long enough to want cancelling.
const LIFECYCLE_PATHS: [&str; 7] = [
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
    "/responses",
    "/v1/messages",
    "/api/chat",
    "/api/generate",
];

/// The request lifecycle middleware. Mounted inside auth (key identity is
/// on the request extensions) and inside the global body limit. Non-tracked
/// paths are a verbatim `next.run` passthrough.
#[allow(clippy::too_many_lines)] // single middleware pass: register -> guard -> forward -> stamp -> wrap, in request order
pub async fn lifecycle(
    State(state): State<Arc<AppState>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    if method != axum::http::Method::POST || !LIFECYCLE_PATHS.contains(&path.as_str()) {
        return next.run(req).await;
    }

    let trace_id = req
        .extensions()
        .get::<crate::TraceId>()
        .map(|t| t.0.clone())
        .unwrap_or_default();
    let key_name = req.extensions().get::<KeyCtx>().map(|k| k.name.clone());

    // Bounded body sniff for `model` (the audit lane's proven pattern):
    // buffer, extract, hand the bytes back untouched. Only attempted when
    // the declared length fits the cap — an over-cap body stays untouched
    // (model stays unknown) rather than being consumed and destroyed.
    let mut req = req;
    let mut model = None;
    let content_len = req
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    if content_len.is_some_and(|n| n <= MODEL_SNIFF_CAP_BYTES as u64) {
        let (parts, body) = req.into_parts();
        match axum::body::to_bytes(body, MODEL_SNIFF_CAP_BYTES).await {
            Ok(bytes) => {
                model = crate::audit::extract_model(&bytes);
                req = Request::from_parts(parts, Body::from(bytes));
            }
            Err(_) => {
                // Declared length lied mid-collect: the stream is broken.
                // Fail loud rather than hand the handler a truncated body.
                return crate::error_response(
                    400,
                    "request body did not match its declared content-length",
                );
            }
        }
    }

    let (card, rx) = state.requests.register(trace_id, &path, model);
    if let Some(k) = &key_name {
        card.set_key(k.clone());
    }
    card.mark_running();

    // Buffered phase: race the handler with the cancel token. Dropping
    // the handler future here is byte-identical to a client disconnect
    // (the proxy frees the engine slot on drop). `biased` with the
    // response first: a completed response beats a pending cancel.
    let mut rx_buf = rx.clone();
    let resp = tokio::select! {
        biased;
        r = next.run(req) => r,
        _ = rx_buf.changed() => {
            let terminal = card.stop_terminal();
            card.state.store(terminal, Ordering::Release);
            card.finished_ms.store(now_unix_ms(), Ordering::Release);
            state.requests.retire(&card);
            return (
                StatusCode::from_u16(499).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                axum::Json(serde_json::json!({
                    "error": {
                        "message": format!(
                            "request {} was {} before completion — the engine slot was released",
                            card.id,
                            if terminal == STATE_INTERRUPTED { "interrupted" } else { "cancelled" },
                        ),
                        "type": if terminal == STATE_INTERRUPTED { "request_interrupted" } else { "request_cancelled" },
                        "request_id": card.id,
                    }
                })),
            )
                .into_response();
        }
    };

    let status = resp.status().as_u16();
    card.status.store(status, Ordering::Release);
    if let Some(k) = resp.extensions().get::<KeyCtx>().map(|k| k.name.clone()) {
        card.set_key(k);
    }
    let mut resp = resp;
    if let Ok(v) = axum::http::HeaderValue::from_str(&card.id) {
        resp.headers_mut().insert("x-blazar-request-id", v);
    }

    // Buffered usage sniff: JSON responses within the cap get their token
    // total recorded (exact, never estimated). Streams keep tokens at 0.
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    let declared = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    let buffered_json = content_type.contains("application/json")
        && declared.is_some_and(|n| n <= USAGE_SNIFF_CAP_BYTES as u64);
    let (parts, raw_body) = resp.into_parts();
    let body = if buffered_json {
        match axum::body::to_bytes(raw_body, USAGE_SNIFF_CAP_BYTES).await {
            Ok(bytes) => {
                if let Some(total) = sniff_tokens(&bytes) {
                    card.tokens.store(total, Ordering::Release);
                }
                Body::from(bytes)
            }
            // Declared length lied mid-collect; the stream is unusable.
            Err(_) => Body::empty(),
        }
    } else {
        raw_body
    };

    let sse = content_type.contains("text/event-stream");
    let wrapped = wrap_body(state.clone(), card, rx, body, sse);
    Response::from_parts(parts, wrapped)
}

/// SSE-only terminal sentinel: `data: [DONE]` is protocol standard and
/// carries no model output. Every other dialect just ends (clients see
/// EOF; we never fabricate a closing record).
fn terminal_frame(sse: bool) -> axum::body::Bytes {
    if sse {
        axum::body::Bytes::from_static(b"data: [DONE]\n\n")
    } else {
        axum::body::Bytes::new()
    }
}

/// Wrap the response body: forwards chunks, captures a bounded partial,
/// and ends the body when the cancel token fires. The stream's drop guard
/// finishes the card as `disconnected` if the client walks away first.
struct Fin {
    state: Arc<AppState>,
    card: Arc<RequestCard>,
}
impl Drop for Fin {
    fn drop(&mut self) {
        // Natural end / cancel paths retire the card first; a Drop
        // that still finds it non-terminal means the body was dropped
        // early = the client went away.
        if !self.card.is_terminal() {
            self.card.state.store(STATE_DISCONNECTED, Ordering::Release);
            self.card
                .finished_ms
                .store(now_unix_ms(), Ordering::Release);
        }
        self.state.requests.retire(&self.card);
    }
}

fn wrap_body(
    state: Arc<AppState>,
    card: Arc<RequestCard>,
    rx: watch::Receiver<bool>,
    body: Body,
    sse: bool,
) -> Body {
    let fin = Fin {
        state: state.clone(),
        card: card.clone(),
    };
    let st = WrapSt {
        body: body.into_data_stream(),
        rx,
        state,
        card,
        fin: Some(fin),
        sse,
        ended: false,
    };
    let stream = unfold(st, |mut st| async move {
        if st.ended {
            st.fin.take();
            return None;
        }
        tokio::select! {
            biased;
            _ = st.rx.changed() => {
                let terminal = st.card.stop_terminal();
                st.card.state.store(terminal, Ordering::Release);
                st.card.finished_ms.store(now_unix_ms(), Ordering::Release);
                st.state.requests.retire(&st.card);
                st.ended = true;
                st.fin.take();
                Some((Ok(terminal_frame(st.sse)), st))
            }
            chunk = st.body.next() => match chunk {
                Some(Ok(bytes)) => {
                    st.card.capture_partial(&bytes);
                    Some((Ok(bytes), st))
                }
                Some(Err(e)) => {
                    if !st.card.is_terminal() {
                        st.card.state.store(STATE_DONE, Ordering::Release);
                        st.card.finished_ms.store(now_unix_ms(), Ordering::Release);
                    }
                    st.state.requests.retire(&st.card);
                    st.ended = true;
                    st.fin.take();
                    Some((Err(std::io::Error::other(e)), st))
                }
                None => {
                    if !st.card.is_terminal() {
                        st.card.state.store(STATE_DONE, Ordering::Release);
                        st.card.finished_ms.store(now_unix_ms(), Ordering::Release);
                    }
                    st.state.requests.retire(&st.card);
                    st.ended = true;
                    st.fin.take();
                    None
                }
            },
        }
    });
    Body::from_stream(stream)
}

struct WrapSt {
    body: axum::body::BodyDataStream,
    rx: watch::Receiver<bool>,
    state: Arc<AppState>,
    card: Arc<RequestCard>,
    fin: Option<Fin>,
    sse: bool,
    ended: bool,
}

/// Extract the token total from a buffered JSON response tail. Handles
/// the `OpenAI` shape (`usage.total_tokens` / `prompt+completion_tokens`)
/// and the Anthropic shape (`usage.input_tokens + output_tokens`).
fn sniff_tokens(body: &[u8]) -> Option<u64> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    let usage = v.get("usage")?;
    if let Some(t) = usage
        .get("total_tokens")
        .and_then(serde_json::Value::as_u64)
    {
        return Some(t);
    }
    let prompt = usage
        .get("prompt_tokens")
        .or_else(|| usage.get("input_tokens"))
        .and_then(serde_json::Value::as_u64);
    let completion = usage
        .get("completion_tokens")
        .or_else(|| usage.get("output_tokens"))
        .and_then(serde_json::Value::as_u64);
    match (prompt, completion) {
        (Some(p), Some(c)) => Some(p + c),
        (None, Some(c)) => Some(c),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

/// GET /v1/requests?model=&state= — live cards then terminal ring.
pub async fn requests_list(
    State(state): State<Arc<AppState>>,
    key: Option<axum::Extension<KeyCtx>>,
    RawQuery(q): RawQuery,
) -> Response {
    let (model_filter, state_filter) = parse_filters(q.as_deref());
    let key_filter = key.map(|k| k.0.name);
    let mut data = state
        .requests
        .list(key_filter.as_deref(), model_filter.as_deref());
    if let Some(want) = state_filter.as_deref() {
        data.retain(|c| c["state"].as_str() == Some(want));
    }
    (
        StatusCode::OK,
        axum::Json(serde_json::json!({
            "object": "blazar.request.list",
            "inflight": state.requests.inflight_count(),
            "count": data.len(),
            "data": data,
        })),
    )
        .into_response()
}

/// GET /v1/requests/{id}
pub async fn requests_get(
    State(state): State<Arc<AppState>>,
    key: Option<axum::Extension<KeyCtx>>,
    Path(id): Path<String>,
) -> Response {
    if !valid_id(&id) {
        return crate::error_response(400, "invalid request id");
    }
    match state.requests.get(&id) {
        Some(card) if visible(&state, &card, key.as_ref().map(|e| &e.0)) => {
            (StatusCode::OK, axum::Json(card)).into_response()
        }
        Some(_) => crate::error_response(
            404,
            "request exists but belongs to another key — cards are key-scoped when keys are configured",
        ),
        None => crate::error_response(
            404,
            "no such request — live cards die with the gateway process; \
             terminal cards live in a bounded 256-entry ring per boot",
        ),
    }
}

/// POST /v1/requests/{id}/cancel — stop the generation, discard output.
pub async fn requests_cancel(
    State(state): State<Arc<AppState>>,
    key: Option<axum::Extension<KeyCtx>>,
    Path(id): Path<String>,
) -> Response {
    request_stop(&state, key.as_ref(), &id, false)
}

/// POST /v1/requests/{id}/interrupt — stop the generation, keep the
/// partial output (streams end cleanly; the card keeps the capture).
pub async fn requests_interrupt(
    State(state): State<Arc<AppState>>,
    key: Option<axum::Extension<KeyCtx>>,
    Path(id): Path<String>,
) -> Response {
    request_stop(&state, key.as_ref(), &id, true)
}

fn request_stop(
    state: &Arc<AppState>,
    key: Option<&axum::Extension<crate::keys::KeyCtx>>,
    id: &str,
    interrupt: bool,
) -> Response {
    if !valid_id(id) {
        return crate::error_response(400, "invalid request id");
    }
    // Key scoping BEFORE mutating: another key's card is indistinguishable
    // from a missing one (no existence leak).
    match state.requests.get(id) {
        Some(card) if visible(state, &card, key.map(|e| &e.0)) => {}
        _ => {
            return crate::error_response(
                404,
                "no such request visible to this key — live cards die with the \
                 gateway process; terminal cards live in a bounded 256-entry ring",
            );
        }
    }
    match state.requests.request_stop(id, interrupt) {
        StopOutcome::NotFound => crate::error_response(404, "no such request"),
        StopOutcome::AlreadyFinished(card) => (
            StatusCode::OK,
            axum::Json(serde_json::json!({
                "cancelled": false,
                "reason": "already finished",
                "request": card,
            })),
        )
            .into_response(),
        StopOutcome::Live(card) => (
            StatusCode::OK,
            axum::Json(serde_json::json!({
                "cancelled": true,
                "interrupt": interrupt,
                "request": card,
            })),
        )
            .into_response(),
    }
}

fn visible(state: &AppState, card: &serde_json::Value, key: Option<&KeyCtx>) -> bool {
    match key {
        None => true,
        Some(ctx) => card["key"].as_str() == Some(&ctx.name) || state.keys.is_empty(),
    }
}

fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 64 && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn parse_filters(q: Option<&str>) -> (Option<String>, Option<String>) {
    let mut model = None;
    let mut state = None;
    for pair in q.iter().flat_map(|q| q.split('&')) {
        if let Some(v) = pair.strip_prefix("model=") {
            model = Some(v.to_string());
        } else if let Some(v) = pair.strip_prefix("state=") {
            state = Some(v.to_string());
        }
    }
    (model, state)
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;

    fn card_with(state: u8, key: Option<&str>, model: Option<&str>) -> RequestCard {
        let c = RequestCard::new(
            "req_t_1".into(),
            "plm-x-1".into(),
            "/api/chat".into(),
            model.map(str::to_string),
        );
        if let Some(k) = key {
            c.set_key(k.to_string());
        }
        c.state.store(state, Ordering::Release);
        c
    }

    #[test]
    fn unit__payload__omits_unknowns_and_adds_live_urls() {
        let c = card_with(STATE_RUNNING, Some("k1"), Some("m1"));
        let v = c.payload(true);
        assert_eq!(v["state"], "running");
        assert_eq!(v["model"], "m1");
        assert_eq!(v["key"], "k1");
        assert!(v.get("engine").is_none());
        assert!(v.get("tokens_generated").is_none());
        assert_eq!(v["cancel_url"], "/v1/requests/req_t_1/cancel");
        assert_eq!(v["interrupt_url"], "/v1/requests/req_t_1/interrupt");

        let term = card_with(STATE_DONE, None, None);
        let v = term.payload(false);
        assert_eq!(v["state"], "done");
        assert!(v.get("cancel_url").is_none());
        assert!(v.get("model").is_none());
    }

    #[test]
    fn unit__payload__partial_status_tokens_surface() {
        let c = card_with(STATE_INTERRUPTED, None, None);
        c.status.store(200, Ordering::Release);
        c.tokens.store(42, Ordering::Release);
        c.capture_partial(b"hello world");
        let v = c.payload(false);
        assert_eq!(v["partial_bytes"], 11);
        assert_eq!(v["partial_preview"], "hello world");
        assert_eq!(v["status"], 200);
        assert_eq!(v["tokens_generated"], 42);
    }

    #[test]
    fn unit__partial_capture__bounded() {
        let c = card_with(STATE_RUNNING, None, None);
        c.capture_partial(&vec![b'a'; PARTIAL_CAP_BYTES + 10]);
        c.capture_partial(b"more");
        assert_eq!(c.partial.lock().expect("partial").len(), PARTIAL_CAP_BYTES);
    }

    #[tokio::test]
    async fn unit__runtime__register_stop_retire_ring() {
        let rt = RequestRuntime::new();
        let (card, mut rx) = rt.register("plm-1".into(), "/api/chat", Some("m".into()));
        assert!(rt.get(&card.id).is_some());
        assert_eq!(rt.inflight_count(), 1);
        card.mark_running();

        match rt.request_stop(&card.id, false) {
            StopOutcome::Live(v) => assert_eq!(v["state"], "running"),
            other => panic!("expected live, got {other:?}"),
        }
        assert!(rx.changed().await.is_ok());

        // retire performs the terminal flip + ring insert
        card.state.store(card.stop_terminal(), Ordering::Release);
        rt.retire(&card);
        assert_eq!(rt.inflight_count(), 0);
        let got = rt.get(&card.id).expect("terminal card");
        assert_eq!(got["state"], "cancelled");

        // unknown id
        assert!(rt.get("req_nope").is_none());
        match rt.request_stop("req_nope", true) {
            StopOutcome::NotFound => {}
            other => panic!("expected notfound, got {other:?}"),
        }
    }

    #[test]
    fn unit__runtime__ring_cap_evicts_oldest() {
        let rt = RequestRuntime::new();
        for _ in 0..=TERMINAL_RING_CAP {
            let (card, _) = rt.register("t".into(), "/api/chat", None);
            rt.retire(&card);
        }
        assert_eq!(rt.list(None, None).len(), TERMINAL_RING_CAP);
    }

    #[test]
    fn unit__list__filters_by_key_and_model() {
        let rt = RequestRuntime::new();
        let (a, _) = rt.register("t1".into(), "/api/chat", Some("m1".into()));
        a.set_key("alice".into());
        let (b, _) = rt.register("t2".into(), "/api/generate", Some("m2".into()));
        b.set_key("bob".into());
        rt.retire(&b);

        assert_eq!(rt.list(None, None).len(), 2);
        assert_eq!(rt.list(Some("alice"), None).len(), 1);
        assert_eq!(rt.list(None, Some("m2")).len(), 1);
        assert_eq!(rt.list(Some("alice"), Some("m2")).len(), 0);
    }

    #[test]
    fn unit__sniff_tokens__openai_and_anthropic_shapes() {
        assert_eq!(sniff_tokens(br#"{"usage":{"total_tokens":33}}"#), Some(33));
        assert_eq!(
            sniff_tokens(br#"{"usage":{"prompt_tokens":10,"completion_tokens":7}}"#),
            Some(17)
        );
        assert_eq!(
            sniff_tokens(br#"{"usage":{"input_tokens":4,"output_tokens":6}}"#),
            Some(10)
        );
        assert_eq!(sniff_tokens(br#"{"nope":1}"#), None);
        assert_eq!(sniff_tokens(b"not json"), None);
    }

    #[test]
    fn unit__terminal_frame__sse_only() {
        assert_eq!(terminal_frame(true).as_ref(), b"data: [DONE]\n\n");
        assert!(terminal_frame(false).is_empty());
    }

    #[test]
    fn unit__valid_id__rejects_traversal() {
        assert!(valid_id("req_abc_123"));
        assert!(!valid_id(""));
        assert!(!valid_id("../etc/passwd"));
        assert!(!valid_id(&"x".repeat(65)));
    }
}
