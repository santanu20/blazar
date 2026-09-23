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
use futures::StreamExt;

use blazar_core::store::Store;
use blazar_core::ModelRow;

use crate::queue::Priority;
use crate::sentinel;
use crate::state::AppState;
use blazar_runtime::{EngineRef, PrefixKey, SupervisionError};

/// Headers never forwarded client->child (hop-by-hop / blazar-internal).
pub(crate) const STRIP_REQUEST: &[&str] = &[
    "host",
    "authorization",
    "connection",
    "content-length",
    "transfer-encoding",
    "x-blazar-priority",
    "x-blazar-num-ctx",
    "x-blazar-enforce",
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
pub fn child_base(ep: &blazar_core::Endpoint) -> String {
    match ep {
        blazar_core::Endpoint::Tcp { host, port } => format!("http://{host}:{port}"),
        blazar_core::Endpoint::Unix { .. } => String::new(),
    }
}

/// Resolve a user-visible model name to the stored row: exact match
/// (case-insensitive) → ollama-migrant `model:tag` `:`→`-` swap onto an
/// existing row (the shared `Store::resolve_model_name` rule the CLI and
/// supervisor use) → unique bare prefix → levenshtein-3 suggestion.
/// The tag is never discarded: a colon form that swaps onto no row falls
/// to the prefix ladder, which reports ambiguity instead of guessing.
/// An on-demand `LoRA` variant suffix (`model+adapter`) resolves the BASE
/// row here; the stem rides the caller's request string through to the
/// supervisor's variant lane (see `ensure_with_admission`).
pub fn resolve_model(store: &Store, requested: &str) -> Result<ModelRow, String> {
    let (base, _lora) = blazar_core::catalog::split_lora_suffix(requested);
    let canonical = store.resolve_model_name(&base.to_lowercase());
    if let Ok(Some(m)) = store.get_model(&canonical) {
        return Ok(m);
    }
    let colonless = base.split(':').next().unwrap_or(base).to_lowercase();
    let models = store.list_models().map_err(|e| e.to_string())?;
    if let Some(m) = models.iter().find(|m| m.name == colonless) {
        return Ok(m.clone());
    }
    let prefixed: Vec<&ModelRow> = models
        .iter()
        .filter(|m| m.name.starts_with(&colonless))
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
        .filter(|m| blazar_core::catalog::levenshtein(&m.name, &colonless) <= 3)
        .map(|m| m.name.as_str())
        .collect();
    if near.is_empty() {
        Err(format!("model {requested:?} not found; try `blazar list`"))
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

/// Run a supervisor load on a detached task. Dropping the request future
/// (client disconnect mid-spawn) would otherwise cancel an in-flight
/// spawn — leaking a half-started child and stranding the loader
/// protocol (live-repro'd wedge: one timed-out generate wedged every
/// later generate on that model). A dropped `JoinHandle` detaches, so the
/// load always runs to completion; only this request stops waiting.
pub(crate) async fn ensure_detached(
    sup: &std::sync::Arc<blazar_runtime::Supervisor>,
    name: &str,
    prefix: Option<PrefixKey>,
) -> Result<EngineRef, SupervisionError> {
    let sup = std::sync::Arc::clone(sup);
    let name = name.to_string();
    tokio::spawn(async move { sup.ensure_routed(&name, prefix).await })
        .await
        .unwrap_or_else(|e| {
            Err(SupervisionError::Internal(anyhow::anyhow!(
                "load task panicked: {e}"
            )))
        })
}

/// Vision-aware variant of [`ensure_detached`]: routes through
/// `Supervisor::ensure_vision` so a `mmproj = "lazy"` spawn respawns
/// WITH the projector when the request carries images. Text requests
/// keep the plain path — zero overhead, zero behavior change.
pub(crate) async fn ensure_vision_detached(
    sup: &std::sync::Arc<blazar_runtime::Supervisor>,
    name: &str,
    prefix: Option<PrefixKey>,
) -> Result<EngineRef, SupervisionError> {
    let sup = std::sync::Arc::clone(sup);
    let name = name.to_string();
    tokio::spawn(async move { sup.ensure_vision(&name, prefix).await })
        .await
        .unwrap_or_else(|e| {
            Err(SupervisionError::Internal(anyhow::anyhow!(
                "vision load task panicked: {e}"
            )))
        })
}

/// Captive variant of [`ensure_detached`]: the load refuses under
/// admission pressure instead of evicting resident models. Same detach
/// contract — the spawn (or its refusal) always runs to completion.
pub(crate) async fn ensure_detached_captive(
    sup: &std::sync::Arc<blazar_runtime::Supervisor>,
    name: &str,
) -> Result<EngineRef, SupervisionError> {
    let sup = std::sync::Arc::clone(sup);
    let name = name.to_string();
    tokio::spawn(async move { sup.ensure_routed_captive(&name).await })
        .await
        .unwrap_or_else(|e| {
            Err(SupervisionError::Internal(anyhow::anyhow!(
                "captive load task panicked: {e}"
            )))
        })
}

/// Does this parsed chat body carry images? Shapes covered:
///
/// - `OpenAI` chat: `messages[].content[]` items with an `image`-prefixed
///   [`type`] (or a bare `image_url` key — some clients omit the tag)
/// - `OpenAI` responses: `input[]` items with an `image`-prefixed [`type`]
/// - Anthropic messages: `messages[].content[]` items `type: "image"`
///   (the prefix check covers it)
/// - Ollama chat/generate: `messages[].images` non-empty or a
///   top-level `images` array (generate shape)
///
/// Pure inspection of the single-parsed body — no re-parse on lanes
/// that already hold one; the responses lane parses once here (it
/// parses again downstream for translation — one bounded extra parse,
/// noted, not silently hot-pathed).
#[must_use]
pub fn body_needs_vision(parsed: &serde_json::Value, ollama_shape: bool) -> bool {
    fn image_item(it: &serde_json::Value) -> bool {
        it.get("type")
            .and_then(|t| t.as_str())
            .is_some_and(|t| t.starts_with("image"))
            || it.get("image_url").is_some()
    }
    if ollama_shape {
        let msgs = parsed
            .get("messages")
            .and_then(|m| m.as_array())
            .is_some_and(|ms| {
                ms.iter().any(|msg| {
                    msg.get("images")
                        .and_then(|i| i.as_array())
                        .is_some_and(|a| !a.is_empty())
                })
            });
        let top = parsed
            .get("images")
            .and_then(|i| i.as_array())
            .is_some_and(|a| !a.is_empty());
        return msgs || top;
    }
    let msgs = parsed
        .get("messages")
        .and_then(|m| m.as_array())
        .is_some_and(|ms| {
            ms.iter().any(|msg| {
                msg.get("content")
                    .and_then(|c| c.as_array())
                    .is_some_and(|items| items.iter().any(image_item))
            })
        });
    let input = parsed
        .get("input")
        .and_then(|i| i.as_array())
        .is_some_and(|items| items.iter().any(image_item));
    msgs || input
}

/// Router-mode variant: plain ensure on the router key, same detach
/// contract ([`ensure_detached`]).
pub(crate) async fn ensure_router_detached(
    sup: &std::sync::Arc<blazar_runtime::Supervisor>,
) -> Result<EngineRef, SupervisionError> {
    let sup = std::sync::Arc::clone(sup);
    tokio::spawn(async move { sup.ensure(blazar_runtime::ROUTER_KEY).await })
        .await
        .unwrap_or_else(|e| {
            Err(SupervisionError::Internal(anyhow::anyhow!(
                "router task panicked: {e}"
            )))
        })
}

/// Ensure the model is running (priority-aware admission) and hand back
/// the engine reference. Errors map to typed HTTP statuses. `prefix`
/// carries the prompt-affinity hash (B1): chat-family callers pass it
/// so repeat conversations land on their warm replica; everything else
/// passes `None`. `needs_vision` routes the first ensure through the
/// lazy-attach path (projector respawn) — chat callers derive it from
/// the single-parsed body via [`body_needs_vision`]; every other lane
/// passes `false`. `allow_diffusion` mirrors [`crate::images::images_gate`]:
/// the images handlers pass `true` (component sets are their cargo);
/// every text/embedding surface passes `false` and gets a teaching 400
/// instead of an sd-server child that 404s every chat-shaped route.
#[allow(clippy::duration_suboptimal_units)] // 120s admission bound per plan
#[allow(clippy::too_many_arguments)] // mirrors proxy_request: flat call params, one seam
pub async fn ensure_with_admission(
    state: &Arc<AppState>,
    model: &str,
    priority: Priority,
    class: crate::queue::WorkClass,
    prefix: Option<PrefixKey>,
    needs_vision: bool,
    allow_diffusion: bool,
    no_evict: bool,
) -> Result<(EngineRef, u128), Box<Response>> {
    let started = Instant::now();
    // Model resolution is the only store need; it completes inside the
    // cached-connection visit (sync, guard never crosses an await).
    let row = state
        .with_store(|s| resolve_model(s, model))
        .ok_or_else(|| Box::new(openai_error(500, "store unavailable")))?
        .map_err(|e| match e.as_str() {
            msg if msg.contains("not found") || msg.contains("ambiguous") => {
                Box::new(openai_error(StatusCode::NOT_FOUND.as_u16(), msg))
            }
            msg => Box::new(openai_error(500, msg)),
        })?;
    // Domain mirror of images::images_gate: a diffusion component set on a
    // text/embedding surface must teach the images lane up front. Spawning
    // sd-server for it would "succeed" and then 404 every chat-shaped
    // route the caller could possibly use.
    if row.has_component_set() && !allow_diffusion {
        return Err(Box::new(openai_error(
            StatusCode::BAD_REQUEST.as_u16(),
            &crate::images::diffusion_text_refusal(&row.name),
        )));
    }
    // On-demand LoRA variant (`model+adapter`): re-attach the stem to
    // the CANONICAL base row so the supervisor spawns/looks up the
    // variant lane (`base+adapter`) regardless of how the caller spelled
    // the base (prefix, colon tag, case).
    let lane = match blazar_core::catalog::split_lora_suffix(model).1 {
        Some(stem) => format!("{}+{}", row.name, stem),
        None => row.name.clone(),
    };
    let lane = lane.as_str();

    let ensure_first = |needs: bool| -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<EngineRef, SupervisionError>> + Send>,
    > {
        if needs {
            Box::pin(ensure_vision_detached(&state.sup, lane, prefix))
        } else if no_evict {
            Box::pin(ensure_detached_captive(&state.sup, lane))
        } else {
            Box::pin(ensure_detached(&state.sup, lane, prefix))
        }
    };
    let first = ensure_first(needs_vision).await;
    // (Bank restore happens inside the supervisor at spawn-readiness.)
    let engine = match first {
        Ok(ep) => ep,
        Err(SupervisionError::AllSlotsBusy) if no_evict => {
            // Captive loads never queue for capacity: waiting two
            // minutes for the right to evict a generation model is
            // exactly the side-effect a cache probe must not have.
            // Fail fast — callers treat this as "side load skipped".
            return Err(Box::new(openai_error(
                503,
                "load refused: model would need to evict resident models \
                 (no-evict admission); retry when capacity frees up",
            )));
        }
        Err(SupervisionError::AllSlotsBusy) => {
            // Capacity exhausted: queue at our priority, bounded wait.
            // Report the pressure — sustained queueing is the demand
            // signal that drives adaptive slot adoption. Enter/leave
            // bracketing keeps it a gauge (a level the reaper can see
            // on every tick, not a one-shot event).
            state.sup.note_slot_pressure(&row.name);
            state.bus.publish(blazar_runtime::BlazarEvent::QueueDepth {
                n: state.queue.depth() + 1,
            });
            let waited = state
                .queue
                .wait(
                    &row.name,
                    priority,
                    class,
                    None,
                    0,
                    std::time::Duration::from_mins(2),
                    None,
                )
                .await;
            state.sup.note_slot_pressure_release(&row.name);
            waited.map_err(|e| Box::new(openai_error(503, &e)))?;
            ensure_first(needs_vision)
                .await
                .map_err(|e| Box::new(supervision_error(&e)))?
        }
        Err(e) => return Err(Box::new(supervision_error(&e))),
    };
    Ok((engine, started.elapsed().as_millis()))
}

/// Prompt-prefix affinity hash (B1) from a parsed request body, stable
/// across conversation turns and blind to samplers/options on purpose:
/// chat/completions → system + first user turn; generate → `prompt`;
/// responses → `input` (string or parts array). Sys-half hashes the
/// system prompt at 256 B (F33: comment previously claimed 1 KiB);
/// convo-half hashes system + first user turn at 1 KiB each. `None`
/// when no recognizable prompt (embeddings, tools) — no affinity,
/// plain load-balance.
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
        // Wrong-lane model (safetensors dir on llama.cpp, GGUF on
        // sglang): the message carries the engine-kind remedy.
        SupervisionError::UnsupportedModel(m) => openai_error(400, m),
        SupervisionError::ModelLoadTimeout(m) => openai_error(
            503,
            &format!("model {m} failed to become healthy (model_load_timeout); check `blazar ps`"),
        ),
        SupervisionError::CircuitOpen(m) => openai_error(
            503,
            &format!("circuit open for {m}: engine keeps crashing; run `blazar ps --reset`"),
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
        "error": {"message": message, "type": "blazar_error", "code": status}
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
/// Child-bound send with the crash-window retry the text-lane forward
/// has: a transport failure means the child died between the health
/// gate and this send (capacity eviction, warm-up crash under VRAM
/// pressure). Reap, respawn the EXACT lane detached, retry ONCE
/// in-band (H16); the circuit breaker inside `ensure_key` bounds
/// crash-looping. `build` receives the (possibly fresh) engine ref so
/// bodied/multipart requests are rebuilt per attempt — reqwest
/// builders are single-use. Rides [`child_send`], so the header-phase
/// bound (and its synchronous wedged-child eviction) covers the image
/// lanes exactly as it covers every text lane.
pub(crate) async fn send_with_child_retry(
    state: &AppState,
    engine: &EngineRef,
    build: impl Fn(&EngineRef) -> reqwest::RequestBuilder,
) -> Result<reqwest::Response, String> {
    let first = match child_send(state, engine, build(engine).send()).await {
        Ok(r) => return Ok(r),
        Err(e) => e,
    };
    tracing::warn!(
        model = %engine.key,
        "child send failure ({first}); reaping and retrying once on a respawned lane"
    );
    state.sup.reap_dead_children().await;
    // A WEDGED child — process alive, HTTP listener dead (observed
    // live: transport-refused while the reaper sees the process
    // running) — survives the reap, and `ensure_key`'s fast path would
    // hand the retry the SAME dead endpoint. Evict the key outright:
    // state flips to Evicted, terminate_group (TERM, then KILL) cleans
    // the process, the map entry drops. A header-stall child is
    // already gone (`child_send` evicted it synchronously) — this is
    // the idempotent no-op side of the same guard. An already-reaped
    // key evicts as a no-op; a healthy replica that took the slot
    // meanwhile is ptr-guarded and survives.
    let _ = state.sup.evict(&engine.key).await;
    match ensure_key_detached(&state.sup, &engine.key).await {
        Ok(fresh) => match child_send(state, &fresh, build(&fresh).send()).await {
            Ok(r) => Ok(r),
            Err(e2) => Err(format!(
                "engine request failed: {first}; retry on respawned child: {e2}"
            )),
        },
        Err(re) => Err(format!("engine request failed: {first}; respawn: {re:#}")),
    }
}

/// Reconnect after a failed child send: reap the dead, then respawn the
/// exact instance lane. The failing child is already gone by the time
/// this runs — `child_send` evicts header-stalls synchronously, and
/// transport errors mean the process died — so this is purely
/// re-establishment, shared by every in-band retry arm.
pub(crate) async fn respawn_lane(
    state: &AppState,
    key: &str,
) -> Result<EngineRef, SupervisionError> {
    state.sup.reap_dead_children().await;
    ensure_key_detached(&state.sup, key).await
}

/// Crash-window variant of [`ensure_detached`]: re-ensures the exact
/// INSTANCE lane that died (`model`, replica `model#N`, projector
/// `model@vision`) instead of re-resolving the model name, which
/// auto-routing could hand to a sibling replica. Rides the same detach
/// contract — the respawn must survive a dropped request future.
pub(crate) async fn ensure_key_detached(
    sup: &std::sync::Arc<blazar_runtime::Supervisor>,
    key: &str,
) -> Result<EngineRef, SupervisionError> {
    let sup = std::sync::Arc::clone(sup);
    let key = key.to_string();
    tokio::spawn(async move { sup.ensure_key(&key).await })
        .await
        .unwrap_or_else(|e| {
            Err(SupervisionError::Internal(anyhow::anyhow!(
                "respawn task panicked: {e}"
            )))
        })
}

/// Child-bound send failure classes: transport errors mean the child is
/// gone (crash-window semantics, 502); a header-phase stall means the
/// child is ALIVE but wedged (504) — [`child_send`] has already evicted
/// it synchronously by the time this reaches a caller.
pub(crate) enum ChildSendError {
    Transport(reqwest::Error),
    HeaderTimeout { secs: u64 },
}

impl ChildSendError {
    /// Terminal status for the caller's error response.
    pub(crate) fn status_u16(&self) -> u16 {
        match self {
            Self::Transport(_) => 502,
            Self::HeaderTimeout { .. } => 504,
        }
    }
}

impl std::fmt::Display for ChildSendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transport(e) => write!(f, "{e:#}"),
            Self::HeaderTimeout { secs } => {
                write!(f, "no response headers from child within {secs}s")
            }
        }
    }
}

/// Bounded child-bound send — the transport choke point for EVERY
/// user-facing child request (proxy lanes AND the ollama dialect's own
/// sends). The header phase (request written -> first response byte)
/// must not ride the blanket transport ceiling: a live-wedged child
/// (observed live: a slot restore raced traffic and one /api/chat
/// parked 300s+ pre-first-byte while the child served every later
/// request) parks the client invisibly instead. On expiry the wedged
/// child is evicted SYNCHRONOUSLY here — state flip, TERM then KILL,
/// map entry drop — because every caller that retries in-band needs the
/// respawn to be guaranteed fresh: `ensure_key`'s fast path would hand
/// the retry the same wedged endpoint (the async J5 eviction lane
/// debounces 1/min per model, far too slow to gate a retry). 0 disables
/// the bound (legacy ceiling only).
pub(crate) async fn child_send(
    state: &AppState,
    engine: &EngineRef,
    send: impl std::future::Future<Output = Result<reqwest::Response, reqwest::Error>>,
) -> Result<reqwest::Response, ChildSendError> {
    match state.config.child_header_timeout_secs {
        0 => send.await.map_err(ChildSendError::Transport),
        secs => match tokio::time::timeout(std::time::Duration::from_secs(secs), send).await {
            Ok(result) => result.map_err(ChildSendError::Transport),
            Err(_elapsed) => {
                tracing::warn!(
                    target: "blazar::proxy",
                    model = %engine.key,
                    "child produced no response headers in {secs}s — evicting synchronously (child_header_timeout_secs)"
                );
                let _ = state.sup.evict(&engine.key).await;
                Err(ChildSendError::HeaderTimeout { secs })
            }
        },
    }
}

/// One child-engine transport attempt: child auth, hop-by-hop header
/// filtering, send. The hot path and the crash-window retry both ride
/// it, so a retry carries identical auth and header hygiene.
async fn forward_once(
    state: &Arc<AppState>,
    engine: &EngineRef,
    method: &axum::http::Method,
    url: &str,
    headers: &HeaderMap,
    body: axum::body::Bytes,
) -> Result<reqwest::Response, ChildSendError> {
    let mut req = state.http.request(method.clone(), url);
    // Client `Authorization` was stripped above; the child secret is
    // stamped fresh here (never the caller's gateway key).
    req = child_auth(req, engine);
    for (name, value) in headers {
        if !STRIP_REQUEST.contains(&name.as_str()) {
            req = req.header(name, value);
        }
    }
    let send = req
        .body(reqwest::Body::wrap_stream(futures::stream::once(
            async move { Ok::<_, std::io::Error>(body) },
        )))
        .send();
    child_send(state, engine, send).await
}

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

// The forwarding core: every per-request concern (auth, tracing, model
// rewrite, singleflight, sentinel) is deliberately threaded through this
// one signature rather than hidden in shared state.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
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
    // Hot-lane pre-parse of `body` (one parse, many consumers). `None`
    // = caller had no parse; consumers that need JSON fall back to
    // parsing `body` themselves (legacy behavior).
    parsed: Option<serde_json::Value>,
    // Pre-admission single-flight phase (chat lanes with a parsed body
    // acquire BEFORE slot admission so duplicates wait guard-free).
    sf_gate: SfGate,
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
    // Both consumers reuse the hot lane's single parse.
    let body = inject_include_usage(path_query, body, parsed.as_ref());
    let body = rewrite_child_model(engine, body, parsed.as_ref());

    // Single-flight (B8, FIX2): identical NON-STREAM chat requests
    // coalesce AT THE CHILD-CALL BOUNDARY. Two acquisition phases:
    // chat lanes that pre-acquired in `sf_gate_before_admission` (held
    // BEFORE slot admission — duplicates wait without occupying
    // in_flight capacity) arrive as `Held`/`TimedOut` and skip this
    // block; everyone else takes the legacy in-function acquire after
    // admission (queued duplicates are not serialized behind queue
    // waits). Stream detection rides the pre-parsed Value — prompt
    // text containing `{"stream":true}` cannot fool it (real JSON, not
    // a sniff). Neither mutation above touches the `stream` field, so
    // the pre-parse stays authoritative for it. Bounded wait: after 5s
    // the twin proceeds uncoalesced (long generations never serialize
    // their duplicates indefinitely).
    let mut sf: Option<SingleFlight> = None;
    match sf_gate {
        SfGate::Held(g) => sf = Some(g),
        // The bounded wait already ran pre-admission; proceeding
        // uncoalesced matches the legacy timeout outcome.
        SfGate::TimedOut => {}
        SfGate::Ineligible => {
            if state.config.singleflight && is_chat_route(path_query) && body.len() <= 32 * 1024 {
                let asks_stream = parsed
                    .as_ref()
                    .and_then(|v| v.get("stream").and_then(serde_json::Value::as_bool))
                    .unwrap_or(false);
                if !asks_stream {
                    let key = sentinel::singleflight_key(model, &body, false);
                    let lock =
                        {
                            let mut map = state
                                .singleflight
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner);
                            if map.len() > 256 {
                                map.clear(); // bounded; a cleared key elects a new leader
                            }
                            std::sync::Arc::clone(map.entry(key).or_insert_with(|| {
                                std::sync::Arc::new(tokio::sync::Mutex::new(()))
                            }))
                        };
                    if let Ok(guard) = tokio::time::timeout(
                        std::time::Duration::from_secs(5),
                        lock.clone().lock_owned(),
                    )
                    .await
                    {
                        sf = Some(SingleFlight {
                            key,
                            map: std::sync::Arc::clone(&state.singleflight),
                            _guard: guard,
                        });
                    }
                }
            }
        }
    }

    // Bytes clone = refcount bump: the sentinel request-side parse reads
    // this snapshot after `body` moved into the upstream stream. The
    // crash-window retry below also rebuilds from it.
    let mut body_snapshot = body.clone();
    let upstream = forward_once(state, engine, method, &url, headers, body).await;
    let resp = match upstream {
        Ok(r) => r,
        Err(e) => {
            // Terminal status reflects the dominant failure: a wedged
            // LIVE child is a gateway timeout (504); a dead child that
            // could not be revived is a bad gateway (502).
            let terminal = match &e {
                ChildSendError::HeaderTimeout { secs } => {
                    tracing::warn!(
                        target: "blazar::proxy",
                        model,
                        trace = ?trace,
                        "child produced no response headers in {secs}s — evicting synchronously, retrying once in-band (child_header_timeout_secs)"
                    );
                    StatusCode::GATEWAY_TIMEOUT
                }
                ChildSendError::Transport(_) => {
                    tracing::error!(model, "proxy {path_query}: {e}");
                    StatusCode::BAD_GATEWAY
                }
            };
            // The child died in the crash window between the health gate
            // and this forward. Reap, respawn the exact lane detached,
            // and retry ONCE in-band — single-shot clients (run
            // --verbose) otherwise eat a 502 for a child they never
            // got to talk to. The circuit breaker inside `ensure_key`
            // bounds crash-looping; exactly one retry (H16). A WEDGED
            // child — process alive, HTTP listener silent — was already
            // evicted synchronously inside `child_send` before this
            // error surfaced; the respawn is guaranteed a fresh lane.
            match respawn_lane(state, &engine.key).await {
                Ok(fresh) => {
                    let fresh_url = format!("{}{path_query}", child_base(&fresh.endpoint));
                    tracing::warn!(
                        model,
                        "proxy {path_query}: child died mid-request; retrying once on respawned lane {}",
                        fresh.key
                    );
                    let retry_body =
                        rewrite_child_model(&fresh, body_snapshot.clone(), parsed.as_ref());
                    body_snapshot = retry_body.clone();
                    match forward_once(state, &fresh, method, &fresh_url, headers, retry_body).await
                    {
                        Ok(r) => r,
                        Err(e2) => {
                            let status = match e2 {
                                ChildSendError::HeaderTimeout { .. } => StatusCode::GATEWAY_TIMEOUT,
                                ChildSendError::Transport(_) => terminal,
                            };
                            drop(sf); // F31: Drop removes the singleflight entry
                            return openai_error(
                                status.as_u16(),
                                &format!(
                                    "engine request failed: {e}; retry on respawned child: {e2}"
                                ),
                            );
                        }
                    }
                }
                Err(re) => {
                    drop(sf); // F31: Drop removes the singleflight entry
                    return openai_error(
                        terminal.as_u16(),
                        &format!("engine request failed: {e}; respawn: {re:#}"),
                    );
                }
            }
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
        builder = builder.header("x-blazar-status", "loading");
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
    // buffering costs no extra round trip. F62: bodies whose declared
    // content-length already exceeds ENFORCE_BODY_CAP skip the
    // buffering branch (streamed through) — the cap bounds judging,
    // never the read. Streaming stays warn-only (bytes on the wire).
    let enforce_oversized = resp
        .content_length()
        .is_some_and(|cl| cl > u64::try_from(sentinel::ENFORCE_BODY_CAP).unwrap_or(u64::MAX));
    if enforce_oversized {
        tracing::warn!(
            target: "blazar::sentinel",
            trace = ?trace,
            "enforce skipped: declared body exceeds cap (streamed, not buffered)"
        );
    }
    if state.config.sentinel
        && !sse
        && status.is_success()
        && is_chat_route(path_query)
        && sentinel::enforce_enabled(&state.config, headers)
        && !enforce_oversized
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
            builder = builder.header("x-blazar-warnings", warnings.join(","));
        }
        match resp.bytes().await {
            // F30: the buffered read failed — finish the sniffer (nothing
            // chargeable was generated) and release the single-flight
            // entry before returning; previously both leaked on this arm.
            Err(e) => {
                if let Some(s) = sniffer.take() {
                    s.finish(&state.keys);
                }
                drop(sf); // F31: Drop removes the singleflight entry
                return openai_error(502, &format!("engine body: {e}"));
            }
            Ok(buf) => {
                if buf.len() > sentinel::ENFORCE_BODY_CAP {
                    tracing::warn!(
                        target: "blazar::sentinel",
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
                        // F30: a fully-generated enforce-rejected response
                        // still consumed child tokens — charge the key's
                        // budgets and release the single-flight entry.
                        if let Some(mut s) = sniffer.take() {
                            s.push(&buf);
                            s.finish(&state.keys);
                        }
                        drop(sf); // F31: Drop removes the singleflight entry
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
                record_buffered_chat(&state.obs, model, &buf, began.elapsed().as_secs_f64());
                drop(sf); // F31: Drop removes the singleflight entry
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
        builder = builder.header("x-blazar-warnings", sentinel_warnings.join(","));
    }
    // Evidence loop: measure TTFT (first body byte) and inter-chunk cadence
    // for this generation. Observing is infallible (atomics only); the
    // stream's data/errors pass through untouched. The in-flight guard
    // chain below still owns the body lifetime.
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
        model: model.to_string(),
    });
    let model_owned = model.to_string();
    let path_owned = path_query.to_string();
    let stream = resp.bytes_stream().flat_map(move |r| {
        match r {
            Ok(bytes) => {
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
                sentinel_feed.bytes(bytes.clone());
                tap_map.push(bytes.as_ref());
                if let Some(s) = sniffer_finisher.lock().expect("sniffer").0.as_mut() {
                    s.push(bytes.as_ref());
                }
                futures::stream::iter(vec![Ok(bytes)])
            }
            // Mid-body upstream failure (sentinel-evicted wedged child,
            // conn churn, respawn race): once headers are committed the
            // only legal SSE close is a semantic one — an Err item here
            // aborts the HTTP body and the client eats a protocol error
            // instead of a readable terminal event.
            Err(e) => {
                tracing::warn!(
                    model = model_owned.as_str(),
                    "proxy {path_owned}: upstream body failed: {e:#}",
                );
                if sse {
                    let frame = serde_json::json!({
                        "error": {
                            "code": "stream_truncated",
                            "message": e.to_string(),
                        }
                    });
                    futures::stream::iter(vec![
                        Ok(axum::body::Bytes::from(format!("data: {frame}\n\n"))),
                        Ok(axum::body::Bytes::from_static(b"data: [DONE]\n\n")),
                    ])
                } else {
                    futures::stream::iter(vec![Err(std::io::Error::other(e.to_string()))])
                }
            }
        }
    });
    // Hold in-flight accounting for the body's lifetime: the guard drops
    // when the client drains (or aborts) the stream.
    let stream = stream.chain(futures::stream::unfold(
        (body_guard, sf, cache_finisher),
        move |(g, sf, cache_finisher)| {
            async move {
                drop(g);
                drop(cache_finisher); // classify at stream end (or abort)
                drop(sf); // F31: Drop releases singleflight at stream end or abort
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
pub struct SingleFlight {
    key: u64,
    map: std::sync::Arc<
        std::sync::Mutex<std::collections::HashMap<u64, std::sync::Arc<tokio::sync::Mutex<()>>>>,
    >,
    _guard: tokio::sync::OwnedMutexGuard<()>,
}

/// Result of the pre-admission single-flight phase (see
/// `sf_gate_before_admission`). Handed straight to `proxy_request`,
/// which skips its internal acquire for `Held`/`TimedOut` — re-locking
/// the same entry from the same task would deadlock (tokio mutexes are
/// not reentrant), and a timed-out waiter must not pay a second 5s.
pub enum SfGate {
    /// Not eligible (config off / non-chat route / stream / oversized
    /// body): `proxy_request`'s legacy internal block stays the owner.
    Ineligible,
    /// Waited (bounded) and now holding the single-flight: the caller
    /// may proceed to slot admission; duplicates coalesce behind this.
    Held(SingleFlight),
    /// Eligible but the bounded wait expired while the leader was still
    /// generating: proceed uncoalesced, exactly as the legacy in-function
    /// path does after its own 5s timeout.
    TimedOut,
}

/// Single-flight BEFORE slot admission: the bounded coalescing wait
/// runs while the caller holds NO `InFlightGuard`, so identical
/// duplicates do not occupy gateway slots (`in_flight` capacity, queue
/// headroom) while merely waiting for the leader. Eligibility mirrors
/// `proxy_request`'s internal block exactly — same route check, size
/// cap, stream bit, and key material — so the two phases always agree
/// on ownership. The key folds the pre-mutation body here (the handler
/// runs before `inject_include_usage`/`rewrite_child_model`); the
/// legacy block folds the post-mutation body. Twins hashing identically
/// within their own phase is what matters — the two key spaces never
/// mix for one request because `Held`/`TimedOut` bypass the internal
/// acquire.
pub(crate) async fn sf_gate_before_admission(
    state: &Arc<AppState>,
    model: &str,
    path_query: &str,
    body: &[u8],
    parsed: Option<&serde_json::Value>,
) -> SfGate {
    if !(state.config.singleflight && is_chat_route(path_query) && body.len() <= 32 * 1024) {
        return SfGate::Ineligible;
    }
    let asks_stream = parsed
        .and_then(|v| v.get("stream").and_then(serde_json::Value::as_bool))
        .unwrap_or(false);
    if asks_stream {
        return SfGate::Ineligible;
    }
    let key = sentinel::singleflight_key(model, body, false);
    let lock = {
        let mut map = state
            .singleflight
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if map.len() > 256 {
            map.clear(); // bounded; a cleared key elects a new leader
        }
        std::sync::Arc::clone(
            map.entry(key)
                .or_insert_with(|| std::sync::Arc::new(tokio::sync::Mutex::new(()))),
        )
    };
    match tokio::time::timeout(std::time::Duration::from_secs(5), lock.lock_owned()).await {
        Ok(guard) => SfGate::Held(SingleFlight {
            key,
            map: std::sync::Arc::clone(&state.singleflight),
            _guard: guard,
        }),
        Err(_) => SfGate::TimedOut,
    }
}

// F31: Drop-safe release — client aborts drop the unfold future before
// any explicit cleanup ran, leaking the map entry until the 256-clear.
impl Drop for SingleFlight {
    fn drop(&mut self) {
        self.map
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.key);
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
/// mistral.rs children register models under derived ids ("default" +
/// the staging dir path — verified against v0.9.3), not Blazar names;
/// their CLI has no `--alias` equivalent. The gateway owns the facade,
/// so the outbound `model` field is rewritten to the child's stable
/// `default` id for mistral.rs engines. llama-server keeps receiving
/// Blazar names (its `--alias` lane matches them natively). Non-JSON
/// bodies and store failures pass through untouched — the child then
/// answers its own clear not-found error.
/// mistral.rs derives /v1 model ids from the `-f` path (no --alias
/// equivalent); Blazar spawns one model per child, so their stable
/// `default` id is the unambiguous target. Shared by every child-bound
/// body site (proxy lane + ollama translation lanes).
pub(crate) fn child_model_default(engine: &EngineRef) -> bool {
    engine.kind == blazar_core::engine_kind::EngineKind::MistralRs
}

/// Per-model serving resolution for sites that must answer BEFORE a
/// spawn exists (request translation + model listings). `tag` is the
/// engine row that WOULD serve this model right now — the routed lane's
/// tag, or the global active row's tag when manual mode (or a pin)
/// resolves to the global engine. `None` from the fn = store
/// unavailable; `tag: None` inside = nothing can serve this model.
pub(crate) struct LaneResolution {
    pub tag: Option<String>,
    pub kind: blazar_core::engine_kind::EngineKind,
}

/// Kind of the engine the router will use for `model` — the ROUTED lane,
/// not the global active row: since auto-routing the active tag is a
/// coincidence for anything but manual mode, so surface gates keyed on it
/// (llama-only endpoints, slot checkpoints) wrongly 400 models that route
/// to a llamacpp child. Unknown model: the global active kind (`None`
/// when no engines, mirroring the old gate-skip). Err lanes resolve to
/// the global kind inside [`resolve_serving`] — the spawn delivers the
/// real teaching error.
pub(crate) fn routed_kind_for(
    state: &Arc<AppState>,
    model: &str,
) -> Option<blazar_core::engine_kind::EngineKind> {
    let row = state
        .with_store(|s| s.get_model(model).ok().flatten())
        .flatten()?;
    resolve_serving(state, model, &row.path)
        .map(|lane| lane.kind)
        .or_else(|| {
            state
                .with_store(|s| s.active_engine().ok().flatten())
                .flatten()
                .map(|e| e.kind)
        })
}

pub(crate) fn resolve_serving(
    state: &Arc<AppState>,
    model_name: &str,
    model_path: &str,
) -> Option<LaneResolution> {
    use blazar_core::engine_kind::{self, EngineKind};
    let overlay = state.config.overlay_for(model_name);
    state
        .with_store(|s| {
            let rows = s.list_engines().unwrap_or_default();
            // Lane class carries fork provenance (core `serving_lane`
            // prefers mainstream builds for overlapping formats).
            #[allow(clippy::type_complexity)]
            let installed: Vec<(String, EngineKind, engine_kind::LaneClass)> = rows
                .iter()
                .map(|r| (r.tag.clone(), r.kind, r.lane_class()))
                .collect();
            let global_row = s.active_engine().ok().flatten();
            let global = global_row.as_ref().map_or(EngineKind::LlamaCpp, |r| r.kind);
            // Capability-rescue preview: when the picked lane provably
            // cannot load the model's GGUF arch and another installed
            // lane advertises it, spawn re-routes after one bounded
            // crash — predict that end-state lane here so listings and
            // routing can never disagree. Never fires under a user pin
            // (the spawn rescue has the same gate) or when the picked
            // lane's arch set is unknown (honest unknown).
            let model_row = s.get_model(model_name).ok().flatten();
            let arch = model_row.as_ref().and_then(|r| r.arch.clone());
            let rescue_preview = |tag: &str, kind: EngineKind| -> Option<String> {
                blazar_runtime::predicted_rescue_lane(
                    &rows,
                    arch.as_deref(),
                    overlay.engine.as_deref(),
                    tag,
                    kind,
                )
            };
            let lane = engine_kind::serving_lane(
                state.config.engine_routing.mode,
                state.config.engine_routing.policy,
                overlay.engine.as_deref(),
                model_row
                    .as_ref()
                    .is_some_and(blazar_core::ModelRow::has_component_set),
                std::path::Path::new(model_path).is_dir(),
                blazar_core::store::quantized_safetensors_signal(model_name, "", model_path),
                global,
                &installed,
            );
            match lane {
                // Routed lane: its row is the answer, unless the
                // rescue preview sees a provably-unservable arch.
                Ok(Some((tag, kind))) => Some(LaneResolution {
                    tag: rescue_preview(&tag, kind).or(Some(tag)),
                    kind,
                }),
                // Manual mode (or a pin resolving to the global engine): the
                // GLOBAL engine serves — same contract the pre-routing
                // daemon-lifetime cache had (rescue preview applies here
                // too: spawn's re-route is not mode-gated).
                Ok(None) => {
                    let tag = global_row.as_ref().and_then(|g| {
                        rescue_preview(&g.tag, g.kind).or_else(|| Some(g.tag.clone()))
                    });
                    Some(LaneResolution { tag, kind: global })
                }
                // Nothing can serve (or a bad pin): no tag to advertise; the
                // spawn path delivers the teaching error on use.
                Err(_) => Some(LaneResolution {
                    tag: None,
                    kind: global,
                }),
            }
        })
        .flatten()
}

/// Pre-spawn prediction of [`child_model_default`] for sites that mutate
/// the request BEFORE the engine exists (ollama chat translation).
/// Consults the SAME core `serving_lane` the supervisor routes by, so
/// the prediction and the actual spawn can never disagree — the
/// daemon-global kind cache this replaces was wrong under
/// `[engine_routing]` (routed mistral.rs children kept the caller's
/// model name and answered `model ... was not found`).
pub(crate) fn child_model_default_predicted(
    state: &Arc<AppState>,
    model_name: &str,
    model_path: &str,
) -> bool {
    use blazar_core::engine_kind::EngineKind;
    // `tag: None` = nothing can serve (or a bad pin) — the spawn path
    // delivers the teaching error, so no rewrite fires; matching the
    // pre-refactor contract where every Err predicted `false`.
    resolve_serving(state, model_name, model_path)
        .is_some_and(|lane| lane.tag.is_some() && lane.kind == EngineKind::MistralRs)
}

pub(crate) fn set_child_model_default(v: &mut serde_json::Value) {
    if v.get("model").and_then(serde_json::Value::as_str).is_some() {
        v["model"] = serde_json::Value::String("default".to_string());
    }
}

/// `parsed` = the request body pre-parsed by the hot lane (one parse,
/// many consumers); `None` = caller had no parse (cold lanes fall back
/// to parsing here, exactly the old behavior).
fn rewrite_child_model(
    engine: &EngineRef,
    body: axum::body::Bytes,
    parsed: Option<&serde_json::Value>,
) -> axum::body::Bytes {
    if body.is_empty() || !child_model_default(engine) {
        return body;
    }
    let maybe_owned = parsed.cloned().map_or_else(
        || serde_json::from_slice::<serde_json::Value>(&body).ok(),
        Some,
    );
    let Some(v) = maybe_owned else {
        return body;
    };
    let mut v = v.clone();
    set_child_model_default(&mut v);
    match serde_json::to_vec(&v) {
        Ok(bytes) => bytes.into(),
        Err(_) => body,
    }
}

fn inject_include_usage(
    path_query: &str,
    body: axum::body::Bytes,
    parsed: Option<&serde_json::Value>,
) -> axum::body::Bytes {
    let p = path_query.split('?').next().unwrap_or(path_query);
    if !p.ends_with("/chat/completions") {
        return body;
    }
    let maybe_owned = parsed.cloned().map_or_else(
        || serde_json::from_slice::<serde_json::Value>(&body).ok(),
        Some,
    );
    let Some(v) = maybe_owned else {
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
    let mut v = v; // owned already — F32: the clone re-copied the body
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
fn record_buffered_chat(
    obs: &std::sync::Arc<crate::state::CacheObs>,
    model: &str,
    buf: &[u8],
    ttft_secs: f64,
) {
    match serde_json::from_slice::<serde_json::Value>(buf) {
        Ok(v) => match v.get("usage") {
            Some(u) => obs.record(
                model,
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
    /// Per-model attribution for the cache-hit surfaces.
    model: String,
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
        self.obs.record(&self.model, prompt, cached, ttft);
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

/// Minimum samples in a TTFT histogram before its p90 is trusted for
/// predictive rejection (below this, admit and measure instead).
const PREDICTIVE_MIN_SAMPLES: u64 = 20;
/// Reject only when the OPTIMISTIC p90 estimate exceeds this multiple of
/// the caller's explicit deadline — a wide margin so we only early-reject
/// requests that are near-certain to blow their SLO anyway.
const PREDICTIVE_MARGIN: f64 = 2.0;

/// Predictive admission (#28): when the caller pinned an explicit
/// deadline and the gateway's TTFT evidence says even the OPTIMISTIC
/// p90 (best of warm/cold, whichever has enough samples) overshoots the
/// deadline by a wide margin, reject fast with a `Retry-After` hint
/// instead of burning queue time on a request that cannot land in time.
/// Returns `Some(retry_after_secs)` when rejection is warranted.
/// Pure decision — unit-tested in isolation from histogram state.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "p90 latencies are small positive seconds; Retry-After wants whole seconds"
)]
pub fn predictive_retry(
    deadline_ms: u64,
    warm_p90_ms: Option<(f64, u64)>,
    cold_p90_ms: Option<(f64, u64)>,
) -> Option<u64> {
    // Optimistic estimate: the best p90 among sufficiently-sampled
    // histograms ((p90_ms, sample_count) pairs; None = no data).
    let est_ms = [warm_p90_ms, cold_p90_ms]
        .into_iter()
        .flatten()
        .filter(|&(_, n)| n >= PREDICTIVE_MIN_SAMPLES)
        .map(|(p90, _)| p90)
        .reduce(f64::min)?;
    let deadline_ms_f = deadline_ms as f64;
    if est_ms > PREDICTIVE_MARGIN * deadline_ms_f && deadline_ms > 0 {
        // Retry-After is whole seconds; ceil the optimistic p90.
        Some((est_ms / 1000.0).ceil().max(1.0) as u64)
    } else {
        None
    }
}

/// Slots-aware admission: when the instance already has `slots` requests
/// in flight (1 by default), WAIT at the caller's priority instead of
/// colliding with the engine's single slot (instant rejects). Bounded by
/// the queue timeout -> 503.
/// SLO-aware admission: `deadline_ms` (the `x-blazar-deadline-ms`
/// header) and `body_len` (prefill-heavy demotion) feed the EDF queue;
/// `wfq` = (API key name, weight) enables weighted fair queuing among
/// same-tier waiters under contention.
#[allow(clippy::too_many_arguments)]
pub async fn admission_gate_slo(
    state: &Arc<AppState>,
    model: &str,
    priority: Priority,
    class: crate::queue::WorkClass,
    deadline_ms: Option<u64>,
    body_len: usize,
    wfq: Option<(&str, u32)>,
) -> Result<InFlightGuard, Box<Response>> {
    let max_inflight: i64 = state.sup.slot_cap(model);
    // Predictive early-reject (#28): an explicit deadline that measured
    // TTFT p90 says we cannot possibly meet -> fail fast with numbers.
    if let Some(ms) = deadline_ms {
        let warm = (
            state.obs.ttft_warm.quantile(0.9) * 1e3,
            state.obs.ttft_warm.count(),
        );
        let cold = (
            state.obs.ttft_cold.quantile(0.9) * 1e3,
            state.obs.ttft_cold.count(),
        );
        if let Some(retry) = predictive_retry(ms, Some(warm), Some(cold)) {
            let mut resp = openai_error(
                429,
                &format!(
                    "predicted TTFT p90 ~{retry}s exceeds 2x your explicit \
                     deadline {ms}ms; retry after the hint or raise the deadline"
                ),
            );
            resp.headers_mut().insert(
                "retry-after",
                axum::http::HeaderValue::from_str(&retry.to_string())
                    .unwrap_or(axum::http::HeaderValue::from_static("1")),
            );
            return Err(Box::new(resp));
        }
    }
    loop {
        let busy = state
            .sup
            .ps()
            .into_iter()
            .find(|p| p.name == model)
            .map_or(0, |p| p.in_flight);
        // Reserved capacity (MT4): the last ceil(N/4) slots belong to
        // interactive/tool work. RawLong may not take one while any
        // non-raw waiter is queued for this model (borrowing keeps the
        // reservation from idling when nobody needs it). On N = 1 the
        // clamp degenerates to "no isolation possible" — a single-slot
        // child cannot separate classes (documented tradeoff).
        let ceiling = match class {
            crate::queue::WorkClass::Interactive | crate::queue::WorkClass::Tool => max_inflight,
            crate::queue::WorkClass::RawLong => {
                crate::queue::raw_ceiling(max_inflight, state.queue.has_non_raw_waiter(model))
            }
        };
        if busy < ceiling {
            return Ok(begin_accounting(state, model));
        }
        // The REAL same-model park: this request waits for a slot on the
        // admission queue. Bracket the wait with the slot-pressure gauge
        // so the adaptive reshaper sees currently-parked demand every
        // reaper tick (the ensure-path AllSlotsBusy arm only covers
        // multi-model spawn contention — live-proven 2026-09-12: 12
        // streams on one loaded model queued here for 37s with the
        // gauge reading zero the whole time).
        state.sup.note_slot_pressure(model);
        let waited = state
            .queue
            .wait(
                model,
                priority,
                class,
                deadline_ms,
                body_len,
                std::time::Duration::from_mins(2),
                wfq,
            )
            .await;
        state.sup.note_slot_pressure_release(model);
        waited.map_err(|e| Box::new(openai_error(503, &e)))?;
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

/// Hold in-flight accounting for an already-built response BODY's
/// lifetime (F29): the guard rides inside the body stream, so clean
/// drains, buffered one-shot bodies, AND client aborts (handler future
/// dropped before the body is consumed) all end accounting exactly
/// once. Replaces the old `with_accounting` future-bracket, which
/// released at headers-ready for streams and never on aborts.
#[must_use]
pub fn hold_body(guard: InFlightGuard, resp: Response) -> Response {
    let status = resp.status();
    let headers = resp.headers().clone();
    let body = futures::StreamExt::map(Body::into_data_stream(resp.into_body()), |r| {
        r.map_err(|e| std::io::Error::other(e.to_string()))
    })
    .chain(futures::stream::unfold(guard, |g| async move {
        drop(g);
        None
    }));
    let mut builder = Response::builder().status(status);
    if let Some(h) = builder.headers_mut() {
        *h = headers;
    }
    builder
        .body(Body::from_stream(body))
        .unwrap_or_else(|e| internal(&format!("hold_body: {e}")))
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

    // --- predictive admission (#28) -------------------------------------

    #[test]
    fn unit__predictive_retry__fires_only_past_double_margin() {
        // est 5s vs deadline 2s: 5 > 2*2 -> reject, hint ceil(5s).
        assert_eq!(predictive_retry(2_000, Some((5_000.0, 100)), None), Some(5));
        // est exactly 2x deadline: within margin -> serve and measure.
        assert_eq!(
            predictive_retry(2_000, Some((4_000.0, 100)), None),
            None,
            "boundary est == 2x deadline must NOT reject"
        );
        // Comfortably under: admit.
        assert_eq!(predictive_retry(2_000, Some((1_500.0, 100)), None), None);
    }

    #[test]
    fn unit__predictive_retry__insufficient_samples_never_rejects() {
        // 19 samples = below the trust gate on BOTH histograms -> None.
        assert_eq!(
            predictive_retry(2_000, Some((9_000.0, 19)), Some((9_500.0, 5))),
            None,
            "no histogram has enough samples -> admit and measure"
        );
        // One histogram reaching 20 samples is enough for its p90.
        assert_eq!(predictive_retry(2_000, Some((9_000.0, 20)), None), Some(9));
    }

    #[test]
    fn unit__predictive_retry__optimistic_estimate_uses_min_p90() {
        // Warm 5s, cold 9s: optimistic est = 5s.
        assert_eq!(
            predictive_retry(2_000, Some((5_000.0, 100)), Some((9_000.0, 100))),
            Some(5)
        );
        // Cold alone (warm unsampled): under margin -> admit.
        assert_eq!(predictive_retry(2_000, None, Some((3_000.0, 100))), None);
        // No data at all: never reject.
        assert_eq!(predictive_retry(2_000, None, None), None);
    }

    #[test]
    fn unit__predictive_retry__fractional_est_ceils_to_seconds() {
        // 4.1s est -> 5s retry hint (Retry-After is whole seconds).
        assert_eq!(predictive_retry(2_000, Some((4_100.0, 100)), None), Some(5));
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
            None,
        );
        let v = parse(&out);
        assert_eq!(
            v.pointer("/stream_options/include_usage"),
            Some(&json!(true))
        );
        assert_eq!(v["stream"], json!(true), "existing body preserved");
    }

    #[test]
    fn unit__inject_include_usage__pre_parsed_matches_fallback_exactly() {
        // The hot lane hands the pre-parsed body in; the byte-identical
        // contract must hold against the None fallback (parse-here)
        // path for EVERY branch: mutate, already-set, non-stream.
        let cases = [
            json!({"model": "m", "messages": [], "stream": true}),
            json!({"stream": true, "stream_options": {"include_usage": true}}),
            json!({"model": "m", "messages": []}),
        ];
        for body in &cases {
            let bytes = json_body(body);
            let pre = serde_json::from_slice::<serde_json::Value>(&bytes).ok();
            let a = inject_include_usage("/v1/chat/completions", bytes.clone(), pre.as_ref());
            let b = inject_include_usage("/v1/chat/completions", bytes.clone(), None);
            assert_eq!(a, b, "pre-parsed lane must equal fallback lane: {body}");
        }
    }

    #[test]
    fn unit__inject_include_usage__already_set_passthrough() {
        let orig = json_body(&json!({"stream": true, "stream_options": {"include_usage": true}}));
        let out = inject_include_usage("/v1/chat/completions", orig.clone(), None);
        assert_eq!(out, orig, "byte-identical: nothing to add");
    }

    #[test]
    fn unit__inject_include_usage__non_stream_passthrough() {
        let orig = json_body(&json!({"model": "m", "messages": []}));
        let out = inject_include_usage("/v1/chat/completions", orig.clone(), None);
        assert_eq!(out, orig, "non-stream requests untouched");
    }

    #[test]
    fn unit__inject_include_usage__non_chat_route_passthrough() {
        let orig = json_body(&json!({"stream": true}));
        let out = inject_include_usage("/v1/completions", orig.clone(), None);
        assert_eq!(
            out, orig,
            "completions lane untouched (usage shape differs)"
        );
        let out2 = inject_include_usage("/v1/embeddings", orig.clone(), None);
        assert_eq!(out2, orig);
    }

    #[test]
    fn unit__inject_include_usage__invalid_json_passthrough() {
        let orig = axum::body::Bytes::from_static(b"{not json stream:true");
        let out = inject_include_usage("/v1/chat/completions", orig.clone(), None);
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
        record_buffered_chat(&obs, "m", &serde_json::to_vec(&body).unwrap(), 0.25);
        assert_eq!(obs.prompt_tokens.load(Ordering::Relaxed), 120);
        assert_eq!(obs.cached_tokens.load(Ordering::Relaxed), 96);
        assert_eq!(obs.unclassified.load(Ordering::Relaxed), 0);
        let mut warm = String::new();
        obs.ttft_warm.render(&mut warm);
        assert!(warm.contains("_count 1"), "{warm}");
    }

    #[test]
    fn unit__record_buffered_chat__no_usage_or_bad_json_is_miss() {
        let obs = std::sync::Arc::new(crate::state::CacheObs::new());
        record_buffered_chat(
            &obs,
            "m",
            &serde_json::to_vec(&json!({"choices": []})).unwrap(),
            0.1,
        );
        record_buffered_chat(&obs, "m", b"not json", 0.1);
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
            model: "m".to_string(),
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

#[cfg(test)]
#[allow(non_snake_case)]
mod resolve_model_tests {
    use super::*;
    use blazar_core::{BlazarDirs, ModelRow, Store};

    fn row(name: &str) -> ModelRow {
        ModelRow {
            name: name.to_string(),
            repo: format!("registry.ollama.ai/library/{name}"),
            quant: "q4_k_m".to_string(),
            path: format!("/models/{name}"),
            bytes: 1,
            sha256: None,
            mmproj_path: None,
            components: vec![],
            shards: 1,
            arch: None,
            params: None,
            ctx_train: None,
            pulled_at: 0,
        }
    }

    fn store_with(names: &[&str]) -> (tempfile::TempDir, Store) {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        let store = Store::open(&dirs).unwrap();
        for n in names {
            store.upsert_model(&row(n)).unwrap();
        }
        (tmp, store)
    }

    #[test]
    fn unit__resolve_model__colon_tag_swaps_onto_exact_row() {
        // Regression pin (2026-09-14 validate run.colon): the live store
        // qwen2.5-0.5b + qwen2.5-0.5b-instruct + qwen2.5-1.5b-instruct
        // made `qwen2.5:0.5b` 404 as "ambiguous" — the tag was discarded
        // and the bare prefix matched three rows. The colon form must
        // resolve through the shared exact/swap rule like the CLI does.
        let (_tmp, store) = store_with(&[
            "qwen2.5-0.5b",
            "qwen2.5-0.5b-instruct",
            "qwen2.5-1.5b-instruct",
        ]);
        let m = resolve_model(&store, "qwen2.5:0.5b").expect("colon tag resolves");
        assert_eq!(m.name, "qwen2.5-0.5b");
    }

    #[test]
    fn unit__resolve_model__lora_variant_resolves_base_row() {
        let (_tmp, store) = store_with(&["qwen2.5-0.5b", "qwen2.5-1.5b-instruct"]);
        // The stem is not part of model resolution: every spelling of
        // the base keeps working with the suffix attached.
        let m = resolve_model(&store, "qwen2.5-0.5b+rsmma").expect("variant resolves base");
        assert_eq!(m.name, "qwen2.5-0.5b");
        let m = resolve_model(&store, "qwen2.5:0.5b+rsmma").expect("colon+stem resolves base");
        assert_eq!(m.name, "qwen2.5-0.5b");
        let m = resolve_model(&store, "qwen2.5-0.5+rsmma").expect("prefix+stem resolves base");
        assert_eq!(m.name, "qwen2.5-0.5b");
        // Degenerate suffixes pass through and fail with the user's own
        // spelling (never a silent wrong-model match).
        let err = resolve_model(&store, "qwen2.5-0.5b+").unwrap_err();
        assert!(err.contains("qwen2.5-0.5b+"), "{err}");
    }

    #[test]
    fn unit__resolve_model__wrong_tag_stays_ambiguous_teaching() {
        // A tag that swaps onto no row must fail loud with the candidate
        // rows — never silently resolve a sibling.
        let (_tmp, store) = store_with(&[
            "qwen2.5-0.5b",
            "qwen2.5-0.5b-instruct",
            "qwen2.5-1.5b-instruct",
        ]);
        let err = resolve_model(&store, "qwen2.5:7b").unwrap_err();
        assert!(err.contains("ambiguous"), "err: {err}");
        assert!(err.contains("qwen2.5-0.5b-instruct"), "err: {err}");
        assert!(err.contains("qwen2.5-1.5b-instruct"), "err: {err}");
    }

    #[test]
    fn unit__resolve_model__colonless_exact_case_insensitive() {
        let (_tmp, store) = store_with(&["qwen2.5-0.5b", "qwen2.5-0.5b-instruct"]);
        let m = resolve_model(&store, "Qwen2.5-0.5b").expect("case-insensitive exact");
        assert_eq!(m.name, "qwen2.5-0.5b");
    }

    #[test]
    fn unit__resolve_model__unique_prefix_still_resolves() {
        let (_tmp, store) =
            store_with(&["qwen2.5-0.5b", "qwen2.5-0.5b-instruct", "nanbeige4.2-3b"]);
        let m = resolve_model(&store, "nanbeige").expect("unique prefix");
        assert_eq!(m.name, "nanbeige4.2-3b");
    }

    #[test]
    fn unit__resolve_model__typo_keeps_levenshtein_suggestion() {
        let (_tmp, store) = store_with(&["qwen2.5-0.5b"]);
        let err = resolve_model(&store, "qwen2.5-0.5c").unwrap_err();
        assert!(err.contains("did you mean"), "err: {err}");
        assert!(err.contains("qwen2.5-0.5b"), "err: {err}");
    }
}
