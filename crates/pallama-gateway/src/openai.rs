//! OpenAI-compatible surface: byte-faithful proxy to the child engine.
//! Pallama never parses or rewrites these bodies — tool calls, structured
//! output, logprobs, `stream_options` ride through untouched (complaint #1
//! fidelity argument). Two deliberate exceptions, both additive and
//! user-explicit-wins: top-level `reasoning_effort` is bridged into
//! `chat_template_kwargs` on /chat/completions (llama-server reads the
//! kwarg, not the `OpenAI` field), and `stream_options.include_usage` is
//! injected on streams so token accounting survives translation.

use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::proxy::{
    admission_gate_slo, affinity_hash_bytes, ensure_with_admission, openai_error, path_and_query,
    proxy_request,
};
use crate::queue::Priority;
use crate::state::AppState;
use crate::TraceId;

use axum::Extension;

/// GET /v1/models — synthesized from the local store (any pulled model is
/// servable; the engine is hot-swapped underneath).
/// Admission classification from the RAW request bytes: tool-bearing
/// bodies carry a `"tools"` member. A substring scan (no re-parse of an
/// already-validated body — the hot path parsed it once inside the
/// preflight block); a prompt that merely MENTIONS `"tools"` only
/// tightens its own admission class, never correctness.
fn body_has_tools(bytes: &[u8]) -> bool {
    bytes.windows(7).any(|w| w == b"\"tools\"")
}

pub async fn models(State(state): State<Arc<AppState>>) -> Response {
    let list = match state.with_store(pallama_core::Store::list_models) {
        Some(Ok(l)) => l,
        Some(Err(e)) => return openai_error(500, &e.to_string()),
        None => return openai_error(500, "store unavailable"),
    };
    let data: Vec<serde_json::Value> = list
        .iter()
        .map(|m| {
            // Which engine row WOULD serve this model right now (routed
            // lane, or the global active row in manual mode). null =
            // nothing can serve it — the spawn path teaches on use.
            let engine =
                crate::proxy::resolve_serving(&state, &m.name, &m.path).and_then(|lane| lane.tag);
            json!({
                "id": format!("{}:{}", m.name, m.quant.to_lowercase()),
                "object": "model",
                "owned_by": "pallama",
                "created": m.pulled_at,
                "engine": engine,
            })
        })
        .collect();
    axum::Json(json!({"object": "list", "data": data})).into_response()
}

/// POST /v1/embeddings — byte-proxied for normal models; gateway-
/// terminated (R1 late chunking) for models opted in via
/// `[model_overrides.<name>] late_chunking = true` (their child runs
/// `--pooling none`, which the OAI-compatible child route rejects, so the
/// gateway must pool). Input may be a string, an array of strings, or an
/// array of pre-tokenized int arrays.
pub async fn embeddings(
    state: State<Arc<AppState>>,
    trace_ext: Option<Extension<TraceId>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    uri: Uri,
    method: Method,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // Resolve the model's registry row for the override lookup; on any
    // resolve miss fall through to the proxy path (it owns 404 shaping).
    let requested = extract_model(&body);
    let late = requested.as_deref().is_some_and(|m| {
        state
            .with_store(|s| crate::proxy::resolve_model(s, m).ok())
            .flatten()
            .is_some_and(|row| state.config.effective_late_chunking(&row.name))
    });
    if !late {
        return openai_proxy(state, trace_ext, key_ext, uri, method, headers, body).await;
    }
    let model = requested.unwrap_or_default();
    // Per-key admission mirrors the proxy path (scope + rpm + count).
    if let Some(Extension(k)) = &key_ext {
        if let Some(entry) = state.keys.entry(&k.name) {
            if let Err(rej) = state.keys.check(&entry, &model) {
                return rej.to_response();
            }
            state.keys.charge_request(&k.name);
        }
    }
    let req: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return openai_error(400, &format!("invalid JSON: {e}")),
    };
    // Shape-check before admission so malformed input never loads a model.
    if !matches!(&req["input"], serde_json::Value::String(_))
        && !matches!(&req["input"], serde_json::Value::Array(a) if !a.is_empty())
    {
        return openai_error(400, "\"input\" must be a string or a non-empty array");
    }
    // Admission is shape-independent: one call serves all three input lanes.
    let (engine, _) = match crate::proxy::ensure_with_admission(
        &state,
        &model,
        crate::queue::Priority::Normal,
        crate::queue::WorkClass::Interactive,
        None,
        false, // embeddings: text-only
    )
    .await
    {
        Ok(ok) => ok,
        Err(resp) => return *resp,
    };
    let outcome = match &req["input"] {
        serde_json::Value::String(s) => {
            crate::latechunk::late_embed(&state, &engine, &model, std::slice::from_ref(s)).await
        }
        serde_json::Value::Array(items) => {
            let all_strings: Option<Vec<String>> = items
                .iter()
                .map(|v| v.as_str().map(str::to_string))
                .collect();
            if let Some(strings) = all_strings {
                crate::latechunk::late_embed(&state, &engine, &model, &strings).await
            } else {
                let all_int_arrays: Option<Vec<Vec<u64>>> = items
                    .iter()
                    .map(|v| {
                        v.as_array().map(|a| {
                            a.iter()
                                .filter_map(serde_json::Value::as_u64)
                                .collect::<Vec<u64>>()
                        })
                    })
                    .collect();
                match all_int_arrays {
                    Some(chunk_ids) => {
                        crate::latechunk::late_embed_pretokenized(&state, &engine, &model, &chunk_ids)
                            .await
                    }
                    None => {
                        return openai_error(
                            400,
                            "late chunking: \"input\" must be a string, an array of strings, or an array of token-id arrays",
                        )
                    }
                }
            }
        }
        _ => unreachable!("shape checked above"),
    };
    match outcome {
        Ok(out) => {
            if let Some(Extension(k)) = &key_ext {
                state.keys.charge_tokens(&k.name, out.total_tokens);
            }
            late_openai_response(&model, out)
        }
        Err((code, msg)) => openai_error(code, &msg),
    }
}

/// Shape a late-chunking result into the `OpenAI` embeddings response body.
fn late_openai_response(model: &str, out: crate::latechunk::LateChunkOutput) -> Response {
    let data: Vec<serde_json::Value> = out
        .embeddings
        .into_iter()
        .enumerate()
        .map(|(i, e)| {
            json!({
                "object": "embedding",
                "index": i,
                "embedding": e,
            })
        })
        .collect();
    axum::Json(json!({
        "object": "list",
        "data": data,
        "model": model,
        "usage": {
            "prompt_tokens": out.total_tokens,
            "total_tokens": out.total_tokens,
        },
    }))
    .into_response()
}

/// Bridge 's top-level `reasoning_effort` into llama-server's
/// template dialect (`chat_template_kwargs.reasoning_effort`) — the child
/// ignores the `OpenAI` field, templates that expose an effort knob read it
/// from the kwargs. Returns true when the kwarg was injected. Explicit
/// user kwargs always win; non-string/empty values ride verbatim.
fn inject_reasoning_effort_kwarg(body: &mut serde_json::Value) -> bool {
    let Some(effort) = body
        .get("reasoning_effort")
        .and_then(|v| v.as_str())
        .filter(|e| !e.is_empty())
        .map(str::to_string)
    else {
        return false;
    };
    if let Some(map) = body
        .get_mut("chat_template_kwargs")
        .and_then(|v| v.as_object_mut())
    {
        if map.contains_key("reasoning_effort") {
            return false;
        }
        map.insert(
            "reasoning_effort".to_string(),
            serde_json::Value::String(effort.clone()),
        );
        return true;
    }
    body["chat_template_kwargs"] = json!({"reasoning_effort": effort});
    true
}

/// All POST /v1/* traffic: one handler, one proxy path, zero body
/// rewriting. `X-Pallama-Priority` orders admission under load.
#[allow(clippy::too_many_lines)] // one cohesive admission + forwarding path
pub async fn openai_proxy(
    State(state): State<Arc<AppState>>,
    trace_ext: Option<Extension<TraceId>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    uri: Uri,
    method: Method,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // ONE parse of the JSON body serves every downstream consumer on
    // this lane (model extraction, usage-flag injection, model-id
    // rewrite, single-flight stream detection); previously each
    // re-parsed the same bytes. Multipart bodies skip JSON entirely.
    let mut parsed_body: Option<serde_json::Value> = if headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.starts_with("multipart/form-data"))
    {
        None
    } else {
        serde_json::from_slice(&body).ok()
    };
    let model = parsed_body
        .as_ref()
        .and_then(|v| v.get("model")?.as_str().map(str::to_string))
        .or_else(|| {
            let ct = headers
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("");
            if ct.starts_with("multipart/form-data") {
                extract_model_multipart(&body, ct)
            } else {
                None
            }
        });
    // Routed-lane surface gate: needs the model to know which engine
    // will serve it (GGUF routes llamacpp regardless of the active row).
    if let Some(resp) = llamacpp_only_gate(&state, &uri, model.as_deref()) {
        return resp;
    }
    let Some(model) = model else {
        return openai_error(400, "missing `model` field in request body");
    };
    // Per-key admission: model scope + rate limits + request accounting.
    if let Some(key) = key_ext.as_ref().map(|Extension(k)| k) {
        if let Some(entry) = state.keys.entry(&key.name) {
            if let Err(rej) = state.keys.check(&entry, &model) {
                return rej.to_response();
            }
            state.keys.charge_request(&key.name);
        }
    }
    // Remote routing: `<remote-name>:<model>` never loads locally.
    if crate::remotes::split_remote(&model, &state.config).is_some() {
        return crate::remotes::forward_with_health(
            &state,
            &model,
            &method,
            &path_and_query(&uri),
            &headers,
            body,
        )
        .await;
    }
    // Reasoning-effort bridge (chat lane only — /completions and
    // /responses are unverified child surfaces for the kwarg): rewrite
    // the parsed body BEFORE every downstream consumer so lint, affinity
    // hashing, and the forwarded bytes all see the same final body.
    let mut body = body;
    if uri.path().ends_with("/chat/completions") {
        if let Some(v) = parsed_body.as_mut() {
            if inject_reasoning_effort_kwarg(v) {
                body = Bytes::from(serde_json::to_vec(v).unwrap_or_default());
            }
        }
    }
    // Strict tool-def lint: catch broken definitions before the model
    // burns a turn (chat/responses lanes only). Prompt-fit preflight:
    // refuse what the engine would silently truncate. FIX8: the LIVE
    // instance ctx wins over config (tuned/overridden instances).
    let chat_family = uri.path().ends_with("/chat/completions")
        || uri.path().ends_with("/completions")
        || uri.path().ends_with("/responses");
    if chat_family {
        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&body) {
            if let Some(err) = state.sentinel.strict_tool_def_error_cached(&v) {
                return openai_error(400, &format!("invalid tools: {err}"));
            }
            let eff = state
                .sup
                .ps()
                .into_iter()
                .find(|p| p.name == model)
                .map_or_else(|| state.config.effective_ctx(&model), |p| p.ctx);
            if let Err(resp) = crate::preflight::enforce_prompt_fits(&state, &model, &v, eff).await
            {
                return *resp;
            }
        }
    }
    let priority = Priority::from_header(
        headers
            .get("x-pallama-priority")
            .and_then(|v| v.to_str().ok()),
    );

    // X-Pallama-Num-Ctx: per-request ctx on the OpenAI path — the
    // protocol has no such field, the extension header fills the gap
    // (same restart-once semantics as options.num_ctx on the ollama API).
    if let Some(raw) = headers
        .get("x-pallama-num-ctx")
        .and_then(|v| v.to_str().ok())
    {
        let want: i64 = match raw.parse() {
            Ok(w) => w,
            Err(_) => {
                return openai_error(400, &format!("X-Pallama-Num-Ctx: not a number: {raw:?}"))
            }
        };
        if want <= 0 {
            return openai_error(400, "X-Pallama-Num-Ctx must be positive");
        }
        if let Err(resp) = crate::ollama::apply_num_ctx(&state, &model, want).await {
            return *resp;
        }
    }

    // X-Pallama-Spec: per-request spec mode on the OpenAI path —
    // protocol has no such field, the extension header fills the gap
    // (same semantics as options.spec on the ollama API).
    if let Some(raw) = headers.get("x-pallama-spec").and_then(|v| v.to_str().ok()) {
        if let Err(resp) = crate::ollama::apply_spec(&state, &model, raw).await {
            return *resp;
        }
    }

    let prefix = if chat_family {
        affinity_hash_bytes(&body)
    } else {
        None
    };
    let (engine, load_ms) = match ensure_with_admission(
        &state,
        &model,
        priority,
        crate::queue::classify_work(body_has_tools(&body), false),
        prefix,
        parsed_body
            .as_ref()
            .is_some_and(|b| crate::proxy::body_needs_vision(b, false)),
    )
    .await
    {
        Ok(ok) => ok,
        Err(resp) => return *resp,
    };
    let model_name = engine.name.clone();
    // Prefix heat (FIX3): chat-family traffic warms the RESOLVED name —
    // aliases would never match the instance key and the eviction bias
    // would silently no-op.
    if chat_family {
        state.sup.note_prefix_hit(&model_name);
    }
    // Cache-bust telemetry (chat completions only: the messages+tools
    // shape the detector fingerprints). Strictly advisory — never
    // mutates parsed_body, never blocks the forward.
    if uri.path().ends_with("/chat/completions") {
        if let Some(req) = parsed_body.as_ref() {
            crate::cache_bust::note_request(
                &state.sentinel,
                &state.cache_bust,
                &model_name,
                req,
                "v1/chat/completions",
                &trace_ext
                    .as_ref()
                    .map(|Extension(t)| t.0.clone())
                    .unwrap_or_default(),
            );
        }
    }

    // SLO: explicit deadline header + prefill-heavy body demotion feed
    // the EDF queue (same-priority shorts beat giant prefills).
    let deadline_ms = headers
        .get("x-pallama-deadline-ms")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    let guard = match admission_gate_slo(
        &state,
        &model_name,
        priority,
        crate::queue::classify_work(body_has_tools(&body), false),
        deadline_ms,
        body.len(),
        wfq_of(key_ext.as_ref()),
    )
    .await
    {
        Ok(g) => g,
        Err(resp) => return *resp,
    };
    proxy_request(
        &state,
        &engine,
        &model_name,
        &method,
        &path_and_query(&uri),
        &headers,
        body,
        load_ms,
        trace_ext.map(|Extension(t)| t.0),
        Some(guard),
        key_ext.map(|Extension(k)| k),
        parsed_body,
    )
    .await
}

/// `GET /v1/responses/{id}` — retrieve a stored (chained) response:
/// `OpenAI` retrieval surface over the gateway registry. Expired or
/// streamed (never stored) ids 404 with the same teaching message as
/// the chaining path.
pub async fn responses_get(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    let found = state
        .responses
        .lock()
        .expect("responses registry")
        .get(&id)
        .map(|s| {
            serde_json::json!({
                "id": id,
                "model": s.model,
                "input": s.input_items,
                "output": s.output_items,
                "usage": {
                    "input_tokens": s.input_tokens,
                    "output_tokens": s.output_tokens,
                },
                "stored_at": s.ts,
            })
        });
    match found {
        Some(v) => (StatusCode::OK, axum::Json(v)).into_response(),
        None => openai_error(
            404,
            "response not found — expired (24h / 256-entry LRU), evicted, streamed, or store:false",
        ),
    }
}

fn extract_model(body: &[u8]) -> Option<String> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    v.get("model")?.as_str().map(str::to_string)
}

/// Child surfaces only llama-server implements. A non-llamacpp child
/// (mistralrs, sglang) would answer these with a bare 404/HTML error;
/// Pallama teaches instead (fail fast, name the limitation, name the
/// switch).
const LLAMACPP_ONLY_PATHS: &[&str] = &[
    "/tokenize",
    "/detokenize",
    "/apply-template",
    "/infill",
    "/v1/chat/completions/control",
    "/v1/chat/completions/input_tokens",
    "/v1/responses/input_tokens",
    "/responses/input_tokens",
    "/v1/rerank",
    "/v1/reranking",
    "/props",
    "/slots",
    "/v1/stream",
    "/v1/streams/lookup",
];

/// `Some(teaching 400)` when the path is llama-server-only AND the engine
/// the router will use for `model` is not llamacpp (mistralrs and sglang
/// children both lack these surfaces). Keys on the ROUTED lane, not the
/// global active row — auto-routing serves GGUF models on a llamacpp
/// child regardless of which engine is globally active. Path matching
/// covers sub-paths (`/slots/{id}`). `model: None` (no resolvable
/// target) falls back to the global active kind, matching the
/// pre-routing estimate.
fn llamacpp_only_gate(state: &Arc<AppState>, uri: &Uri, model: Option<&str>) -> Option<Response> {
    let path = uri.path();
    if !LLAMACPP_ONLY_PATHS
        .iter()
        .any(|p| path == *p || path.starts_with(&format!("{p}/")))
    {
        return None;
    }
    let kind = model
        .and_then(|m| crate::proxy::routed_kind_for(state, m))
        .or_else(|| {
            state
                .with_store(|s| s.active_engine().ok().flatten())
                .flatten()
                .map(|r| r.kind)
        })?;
    if kind == pallama_core::engine_kind::EngineKind::LlamaCpp {
        return None;
    }
    Some(openai_error(
        400,
        &format!(
            "this endpoint is llama-server-only; the {kind} lane that serves this \
             request does not implement it — switch with `pallama engine use <tag>` \
             or a model_overrides engine pin (see `pallama engine list`)",
        ),
    ))
}

/// Engine-scoped upstream surfaces whose body carries no `model` field:
/// `/props`, `/slots`, `/slots/{id}`, `/v1/stream`, `/v1/streams/lookup`.
/// Model resolution order: `X-Pallama-Model` header > `?model=` query >
/// the single hot child (only when exactly one is loaded). Ambiguity is
/// a teaching 400 — never a silent guess across children. POST bodies on
/// `/v1/streams/lookup` may carry `model` themselves (chat-shaped) and
/// are honored first. Everything else mirrors `openai_proxy`: same key
/// admission, remote forwarding, queue admission, byte-faithful proxy.
#[allow(clippy::too_many_lines)] // one cohesive resolution + forwarding path
pub async fn scoped_proxy(
    State(state): State<Arc<AppState>>,
    trace_ext: Option<Extension<TraceId>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    uri: Uri,
    method: Method,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // Model-carrying bodies: /v1/streams/lookup (chat-shaped) plus the
    // llama-only surfaces whose POST bodies name their model
    // (/tokenize, /detokenize, /apply-template) — the gate needs the
    // target to know which lane will serve it.
    let body_model = if matches!(uri.path(), "/tokenize" | "/detokenize" | "/apply-template")
        || uri.path() == "/v1/streams/lookup"
    {
        extract_model(&body)
    } else {
        None
    };
    if let Some(resp) = llamacpp_only_gate(&state, &uri, body_model.as_deref()) {
        return resp;
    }
    let header_model = headers
        .get("x-pallama-model")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let query_model = uri.query().and_then(|q| {
        q.split('&').find_map(|kv| {
            let (k, v) = kv.split_once('=')?;
            (k == "model").then(|| v.to_string())
        })
    });
    let hot = state.sup.ps();
    let single = if hot.len() == 1 {
        hot.first().map(|p| p.name.clone())
    } else {
        None
    };
    let model = body_model.or(header_model).or(query_model).or(single);
    let Some(model) = model else {
        return openai_error(
            400,
            "model-scoped surface needs a target: set X-Pallama-Model, ?model=, or have exactly one loaded model (GET /api/ps lists candidates)",
        );
    };
    // Per-key admission (same contract as openai_proxy).
    if let Some(key) = key_ext.as_ref().map(|Extension(k)| k) {
        if let Some(entry) = state.keys.entry(&key.name) {
            if let Err(rej) = state.keys.check(&entry, &model) {
                return rej.to_response();
            }
            state.keys.charge_request(&key.name);
        }
    }
    if crate::remotes::split_remote(&model, &state.config).is_some() {
        return crate::remotes::forward_with_health(
            &state,
            &model,
            &method,
            &path_and_query(&uri),
            &headers,
            body,
        )
        .await;
    }
    let priority = Priority::from_header(
        headers
            .get("x-pallama-priority")
            .and_then(|v| v.to_str().ok()),
    );
    let class = crate::queue::classify_work(body_has_tools(&body), false);
    let (engine, load_ms) =
        match ensure_with_admission(&state, &model, priority, class, None, false).await {
            Ok(ok) => ok,
            Err(resp) => return *resp,
        };
    let model_name = engine.name.clone();
    let deadline_ms = headers
        .get("x-pallama-deadline-ms")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    let guard = match admission_gate_slo(
        &state,
        &model_name,
        priority,
        crate::queue::classify_work(body_has_tools(&body), false),
        deadline_ms,
        body.len(),
        wfq_of(key_ext.as_ref()),
    )
    .await
    {
        Ok(g) => g,
        Err(resp) => return *resp,
    };
    proxy_request(
        &state,
        &engine,
        &model_name,
        &method,
        &path_and_query(&uri),
        &headers,
        body,
        load_ms,
        trace_ext.map(|Extension(t)| t.0),
        Some(guard),
        key_ext.map(|Extension(k)| k),
        None,
    )
    .await
}

/// POST /v1/responses — proxied WITH gateway conversation state:
/// `previous_response_id` chaining and `store` semantics that upstream
/// llama-server does not implement (no storage server-side, verified).
/// Non-streaming responses are stored (bounded LRU) and re-id'd so
/// clients can chain; streaming responses consume stored context but
/// are not themselves stored (warned via `x-pallama-warnings`).
#[allow(clippy::too_many_lines)]
pub async fn responses_api(
    State(state): State<Arc<AppState>>,
    trace_ext: Option<Extension<TraceId>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    uri: Uri,
    method: Method,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let model = extract_model(&body)
        .or_else(|| extract_model_multipart(&body, "multipart/form-data"))
        .unwrap_or_default();
    if let Some(key) = key_ext.as_ref().map(|Extension(k)| k) {
        if let Some(entry) = state.keys.entry(&key.name) {
            if let Err(rej) = state.keys.check(&entry, &model) {
                return rej.to_response();
            }
            state.keys.charge_request(&key.name);
        }
    }
    // Remote routing first (registry chaining is local-only today).
    if crate::remotes::split_remote(&model, &state.config).is_some() {
        return crate::remotes::forward_with_health(
            &state,
            &model,
            &method,
            &path_and_query(&uri),
            &headers,
            body,
        )
        .await;
    }
    // Parse-failure passthrough: an unparseable body still deserves the
    // engine's own error, byte-faithful.
    let mut parsed: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => {
            return openai_proxy(State(state), trace_ext, key_ext, uri, method, headers, body)
                .await;
        }
    };
    let stream = parsed
        .get("stream")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let store = parsed
        .get("store")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true);
    let previous = parsed
        .get("previous_response_id")
        .and_then(|p| p.as_str())
        .map(str::to_string);

    // Chaining: resolve the stored prefix, rebuild `input`, drop the
    // field (upstream rejects unknown fields less gracefully than we
    // reject stale ids).
    let mut inherited_model = None;
    if let Some(pid) = previous {
        let found = {
            let mut reg = state.responses.lock().expect("responses registry");
            reg.get(&pid).map(|s| {
                (
                    crate::responses::ResponsesRegistry::chain_input(s, &parsed["input"]),
                    s.model.clone(),
                )
            })
        };
        let Some((chained, prev_model)) = found else {
            return openai_error(
                404,
                &format!(
                    "previous_response_id {pid:?} not found — expired (24h / 256-entry LRU), \
                     evicted, or the response streamed (streamed responses are not stored)"
                ),
            );
        };
        parsed["input"] = chained;
        inherited_model = Some(prev_model);
        parsed
            .as_object_mut()
            .map(|o| o.remove("previous_response_id"));
    }
    // Model: explicit > inherited-from-chain > named-400 (every OpenAI
    // route's contract on a missing model).
    let model = parsed
        .get("model")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .or(inherited_model)
        .unwrap_or_default();
    if model.is_empty() {
        return openai_error(400, "missing `model` field in request body");
    }
    parsed
        .as_object_mut()
        .map(|o| o.insert("model".into(), serde_json::json!(model)));
    let new_body = serde_json::to_vec(&parsed).unwrap_or_else(|_| body.to_vec());
    let priority = Priority::from_header(
        headers
            .get("x-pallama-priority")
            .and_then(|v| v.to_str().ok()),
    );
    let (engine, load_ms) = match ensure_with_admission(
        &state,
        &model,
        priority,
        crate::queue::classify_work(body_has_tools(&body), false),
        affinity_hash_bytes(&body),
        serde_json::from_slice::<serde_json::Value>(&body)
            .is_ok_and(|b| crate::proxy::body_needs_vision(&b, false)),
    )
    .await
    {
        Ok(ok) => ok,
        Err(resp) => return *resp,
    };
    let model_name = engine.name.clone();
    let guard = match admission_gate_slo(
        &state,
        &model_name,
        priority,
        crate::queue::classify_work(body_has_tools(&body), false),
        None,
        0,
        wfq_of(key_ext.as_ref()),
    )
    .await
    {
        Ok(g) => g,
        Err(resp) => return *resp,
    };

    if stream || !store {
        // Streaming + unstored: pure passthrough. A streaming response
        // asked to be stored gets a named warning header instead of
        // silent loss (bytes are already on the wire — storage is
        // physically impossible after the first token). Response headers
        // are still mutable here: nothing has shipped yet.
        let mut r = proxy_request(
            &state,
            &engine,
            &model_name,
            &method,
            &path_and_query(&uri),
            &headers,
            Bytes::from(new_body),
            load_ms,
            trace_ext.map(|Extension(t)| t.0),
            Some(guard),
            key_ext.map(|Extension(k)| k),
            None,
        )
        .await;
        if stream && store {
            r.headers_mut().insert(
                "x-pallama-warnings",
                "responses_stream_not_stored".parse().unwrap_or(
                    axum::http::HeaderValue::from_static("responses_stream_not_stored"),
                ),
            );
        }
        return r;
    }

    // Non-stream + store: buffered forward, store, re-id, return.
    let url = format!(
        "{}/v1/responses",
        crate::proxy::child_base(&engine.endpoint)
    );
    let upstream = crate::proxy::child_auth(
        state
            .http
            .post(&url)
            .header("content-type", "application/json"),
        &engine,
    )
    .body(new_body.clone())
    .send()
    .await;
    let resp = match upstream {
        Ok(r) => r,
        Err(e) => {
            state.sup.reap_dead_children().await;
            return openai_error(502, &format!("engine request failed: {e:#}"));
        }
    };
    let status = resp.status().as_u16();
    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        return openai_error(status, &format!("engine error: {text}"));
    }
    let mut out: serde_json::Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => return openai_error(502, &format!("bad engine response: {e}")),
    };
    // Sentinel observation: the buffered branch bypasses proxy_request,
    // so the feed is driven here (same as the ollama translate path).
    {
        let (feed, _) = crate::sentinel::begin_chat_observation(
            &state,
            "openai-responses",
            &model_name,
            &new_body,
            trace_ext.clone().map(|Extension(t)| t.0),
            status,
            false,
        );
        feed.value(out.clone());
        drop(feed); // Drop fires End: analyzer finalizes off the path
    }
    // Store under OUR id and rewrite the body's id so the client chains
    // through the gateway (the engine's own id carries no storage).
    let id = crate::responses::new_response_id();
    let usage = out.get("usage").cloned().unwrap_or(serde_json::Value::Null);
    let in_tok = usage["input_tokens"].as_u64();
    let out_tok = usage["output_tokens"].as_u64();
    // FIX6: this buffered branch bypasses proxy_request's sniffer —
    // charge token budgets directly from the parsed usage.
    if let Some(Extension(k)) = key_ext.as_ref() {
        let total = in_tok.unwrap_or(0).saturating_add(out_tok.unwrap_or(0));
        state.keys.charge_tokens(&k.name, total);
    }
    let input_items = parsed
        .get("input")
        .cloned()
        .unwrap_or(serde_json::Value::Array(vec![]));
    let output_items = out
        .get("output")
        .cloned()
        .unwrap_or(serde_json::Value::Array(vec![]));
    if let Some(obj) = out.as_object_mut() {
        obj.insert("id".into(), serde_json::json!(id));
    }
    state.responses.lock().expect("responses registry").put(
        id,
        crate::responses::StoredResponse {
            model: model_name.clone(),
            input_items,
            output_items,
            input_tokens: in_tok,
            output_tokens: out_tok,
            ts: crate::responses::unix_now(),
        },
    );
    drop(guard);
    let mut builder = Response::builder().status(status);
    if load_ms > 100 {
        builder = builder.header("x-pallama-status", "loading");
    }
    builder
        .body(axum::body::Body::from(
            serde_json::to_vec(&out).unwrap_or_default(),
        ))
        .map_or_else(
            |e| openai_error(500, &format!("response build: {e}")).into_response(),
            |mut r| {
                r.headers_mut().insert(
                    axum::http::header::CONTENT_TYPE,
                    axum::http::HeaderValue::from_static("application/json"),
                );
                r
            },
        )
}

/// Extract the `model` field from a multipart/form-data body (audio
/// transcriptions upload audio bytes; the JSON path cannot apply).
/// Boundary-aware: only scans part HEADERS (bounded window after each
/// boundary), never audio payload bytes.
fn extract_model_multipart(body: &[u8], content_type: &str) -> Option<String> {
    let boundary = content_type
        .split(';')
        .map(str::trim)
        .find_map(|p| p.strip_prefix("boundary="))?
        .trim_matches('"');
    let delim = format!("--{boundary}");
    let mut pos = 0;
    while let Some(rel) = find_sub(&body[pos..], delim.as_bytes()) {
        let start = pos + rel + delim.len();
        // Part header ends at CRLFCRLF; scan only within it.
        let header_end = find_sub(&body[start..], b"\r\n\r\n")? + start;
        let headers = &body[start..header_end];
        let is_field = !headers.windows(9).any(|w| w == b"filename=");
        if is_field && find_sub(headers, b"name=\"model\"").is_some() {
            // value starts after the blank line, ends at next CRLF
            let vstart = header_end + 4;
            let vend = find_sub(&body[vstart..], b"\r\n")? + vstart;
            if vend > vstart {
                return Some(String::from_utf8_lossy(&body[vstart..vend]).into_owned());
            }
        }
        pos = header_end;
    }
    None
}

/// WFQ identity for the admission queue: (key name, weight). Absent for
/// unauthenticated (open) gateways — those waiters share one bucket.
fn wfq_of(key_ext: Option<&Extension<crate::keys::KeyCtx>>) -> Option<(&str, u32)> {
    key_ext.map(|Extension(k)| (k.name.as_str(), k.weight))
}

fn find_sub(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Model-less /v1/adapters target resolution: (name, state, `idle_secs`) of
/// the live engines + the first pulled store row as cold fallback.
/// Rank 1 = most-recently-used READY engine (a model-less "list my
/// adapters" means the engine I am using); rank 2 = first store row.
/// Without the live preference, an arbitrary store-first model hijacks
/// the request — on boxes with a draft-only model sorted first that
/// spawns a doomed child and 502s instead of listing the running
/// engine's adapters. Pure so the ranking is unit-testable.
fn adapters_target<'a>(
    live: &'a [(String, String, u64)],
    store_first: Option<&'a str>,
) -> Option<&'a str> {
    live.iter()
        .filter(|(_, state, _)| state == "ready")
        .min_by_key(|(name, _, idle)| (*idle, name.clone()))
        .map(|(name, _, _)| name.as_str())
        .or(store_first)
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
    let store_first = match state.with_store(|s| s.list_models().ok()) {
        Some(list) => list.and_then(|v| v.into_iter().next().map(|m| m.name)),
        None => return openai_error(500, "store unavailable"),
    };
    let live: Vec<(String, String, u64)> = state
        .sup
        .ps()
        .iter()
        .map(|p| (p.name.clone(), p.state.to_string(), p.idle_secs))
        .collect();
    let Some(m) = adapters_target(&live, store_first.as_deref()) else {
        return openai_error(404, "no models pulled; adapters apply to a running engine");
    };
    let (engine, load_ms) = match ensure_with_admission(
        &state,
        m,
        Priority::Normal,
        crate::queue::WorkClass::Interactive,
        None,
        false,
    )
    .await
    {
        Ok(ok) => ok,
        Err(resp) => return *resp,
    };
    let url = format!(
        "/lora-adapters{}",
        path_and_query(&uri).trim_start_matches("/v1/adapters")
    );
    let name = engine.name.clone();
    let guard = admission_gate_slo(
        &state,
        &name,
        Priority::Normal,
        crate::queue::WorkClass::Interactive,
        None,
        0,
        None,
    )
    .await;
    match guard {
        Ok(g) => {
            proxy_request(
                &state,
                &engine,
                &name,
                &method,
                &url,
                &headers,
                body,
                load_ms,
                None,
                Some(g),
                None,
                None,
            )
            .await
        }
        Err(resp) => *resp,
    }
}

/// GET /healthz — liveness of the gateway itself (children not required).
pub async fn healthz() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

/// Type alias to satisfy the router signature expectations.
pub type BodyBytes = Bytes;
pub type StreamBody = Body;

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    const CT: &str = "multipart/form-data; boundary=pallama-b";

    #[test]
    fn unit__multipart_extract__model_field_after_file_part() {
        let body = b"--pallama-b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF--pallama-b-fake-bytes\r\n--pallama-b\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nm1\r\n--pallama-b--\r\n";
        assert_eq!(extract_model_multipart(body, CT).as_deref(), Some("m1"));
    }

    #[test]
    fn unit__multipart_extract__model_field_before_file_part() {
        let body = b"--pallama-b\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nqwen3-0.6b\r\n--pallama-b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"b.mp3\"\r\n\r\nID3-bytes\r\n--pallama-b--\r\n";
        assert_eq!(
            extract_model_multipart(body, CT).as_deref(),
            Some("qwen3-0.6b")
        );
    }

    #[test]
    fn unit__multipart_extract__binary_bytes_never_misread_as_model() {
        // hostile/odd audio that contains a decoy header-looking sequence
        let body = b"--pallama-b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.wav\"\r\n\r\nname=\"model\"\r\n\r\ndecoy\r\n--pallama-b--\r\n";
        assert_eq!(extract_model_multipart(body, CT), None);
    }

    #[test]
    fn unit__multipart_extract__no_boundary_header_is_none() {
        assert_eq!(
            extract_model_multipart(b"whatever", "application/json"),
            None
        );
    }

    fn row(name: &str, state: &str, idle: u64) -> (String, String, u64) {
        (name.into(), state.into(), idle)
    }

    #[test]
    fn unit__adapters_target__prefers_mru_ready_engine_over_store_first() {
        // store-first would hijack with the draft-only model (live bug:
        // doomed spawn + 502); the engine actually in use must win.
        let live = vec![row("qwen2.5-0.5b-instruct", "ready", 5)];
        assert_eq!(
            adapters_target(&live, Some("dflash-qwen3-8b-q8_0")),
            Some("qwen2.5-0.5b-instruct")
        );
    }

    #[test]
    fn unit__adapters_target__ignores_non_ready_rows() {
        let live = vec![
            row("loading-model", "loading", 0),
            row("stopping-model", "stopping", 0),
        ];
        assert_eq!(
            adapters_target(&live, Some("store-first")),
            Some("store-first")
        );
    }

    #[test]
    fn unit__adapters_target__tie_breaks_on_name_deterministically() {
        let live = vec![row("b-model", "ready", 7), row("a-model", "ready", 7)];
        assert_eq!(adapters_target(&live, Some("s")), Some("a-model"));
    }

    #[test]
    fn unit__adapters_target__no_live_no_store_is_none() {
        assert_eq!(adapters_target(&[], None), None);
    }

    #[test]
    fn unit__adapters_target__cold_gateway_falls_back_to_store_first() {
        assert_eq!(
            adapters_target(&[], Some("only-pulled")),
            Some("only-pulled")
        );
    }

    #[test]
    fn unit__inject_reasoning_effort_kwarg__bridges_top_level_field() {
        let mut v = json!({
            "model": "m1",
            "reasoning_effort": "low",
            "messages": [],
        });
        assert!(inject_reasoning_effort_kwarg(&mut v));
        assert_eq!(v["chat_template_kwargs"]["reasoning_effort"], "low");
        // Top-level field stays (children that DO read it still can).
        assert_eq!(v["reasoning_effort"], "low");
    }

    #[test]
    fn unit__inject_reasoning_effort_kwarg__explicit_user_kwarg_wins() {
        let mut v = json!({
            "reasoning_effort": "low",
            "chat_template_kwargs": {"reasoning_effort": "max"},
        });
        assert!(!inject_reasoning_effort_kwarg(&mut v));
        assert_eq!(v["chat_template_kwargs"]["reasoning_effort"], "max");
        // Existing kwargs object keeps its siblings untouched.
        let mut v2 = json!({
            "reasoning_effort": "low",
            "chat_template_kwargs": {"enable_thinking": true},
        });
        assert!(inject_reasoning_effort_kwarg(&mut v2));
        assert_eq!(v2["chat_template_kwargs"]["enable_thinking"], true);
        assert_eq!(v2["chat_template_kwargs"]["reasoning_effort"], "low");
    }

    #[test]
    fn unit__inject_reasoning_effort_kwarg__non_string_and_empty_ignored() {
        let mut num = json!({"reasoning_effort": 3});
        assert!(!inject_reasoning_effort_kwarg(&mut num));
        assert!(num.get("chat_template_kwargs").is_none());
        let mut empty = json!({"reasoning_effort": ""});
        assert!(!inject_reasoning_effort_kwarg(&mut empty));
        let mut absent = json!({"model": "m1"});
        assert!(!inject_reasoning_effort_kwarg(&mut absent));
    }
}
