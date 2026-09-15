//! Pallama gateway: `OpenAI` + ollama-compat HTTP surface over the
//! supervisor. Router assembly + bearer auth.

pub mod anthropic;
pub mod audit;
pub mod batch;
pub mod histogram;
pub mod keys;
pub mod latechunk;
pub mod ollama;
pub mod openai;
pub mod otlp;
pub mod preflight;
pub mod proxy;
pub mod queue;
pub mod remotes;
pub mod responses;
pub mod scrub;
pub mod semcache;
pub mod sentinel;
pub mod sessions;
pub mod state;
pub mod translate;
pub mod whisper;

use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Extension, State};
use axum::http::{Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::serve::ListenerExt;
use axum::Router;

use pallama_core::ApiKey;
use state::AppState;

/// Request-scoped trace id, inserted by `request_log` (the outermost
/// layer) so inner handlers can correlate sentinel records with the
/// response header clients see.
#[derive(Clone)]
pub struct TraceId(pub String);

/// Bearer auth: active iff any `[[keys]]` exist (children stay loopback
/// and unauthenticated). Resolves the presented secret to its key entry
/// and tags the request with `KeyCtx` for scope/rate/accounting.
/// Secret sources (in priority order): `Authorization: Bearer <secret>`
/// (`OpenAI` convention) and `x-api-key: <secret>` (Anthropic convention —
/// Claude Code & friends never send Authorization).
async fn auth(State(state): State<Arc<AppState>>, req: Request<Body>, next: Next) -> Response {
    if !state.keys.is_empty() {
        let path = req.uri().path();
        let open = path == "/healthz" || path == "/health";
        if !open {
            let presented = req
                .headers()
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer ").map(str::to_string))
                .or_else(|| {
                    req.headers()
                        .get("x-api-key")
                        .and_then(|v| v.to_str().ok())
                        .map(str::to_string)
                });
            match presented.as_deref().and_then(|p| state.keys.resolve(p)) {
                Some(k) => {
                    // Concurrency cap is the cheapest gate, so it runs
                    // before scope/rpm/tpm: a capped key cannot even
                    // probe scopes. The lease rides the response body —
                    // freed when the client drains (or drops) it.
                    let lease = match state.keys.acquire(&k.name, k.max_concurrent) {
                        Ok(l) => l,
                        Err(rej) => return rej.to_response(),
                    };
                    let mut req = req;
                    req.extensions_mut().insert(keys::KeyCtx {
                        name: k.name.clone(),
                        weight: k.effective_weight(),
                    });
                    let resp = next.run(req).await;
                    let mut resp = keys::guard_response(resp, lease);
                    // F64: request_log now runs OUTSIDE auth and can no
                    // longer read KeyCtx off the request — stamp it on
                    // the response extensions so the audit/access line
                    // still carries the key name.
                    resp.extensions_mut().insert(keys::KeyCtx {
                        name: k.name.clone(),
                        weight: k.effective_weight(),
                    });
                    return resp;
                }
                None => {
                    return (
                        StatusCode::UNAUTHORIZED,
                        axum::Json(serde_json::json!({"error": {"message": "missing or invalid API key (Authorization: Bearer or x-api-key)", "type": "pallama_error", "code": 401}})),
                    )
                        .into_response();
                }
            }
        }
    }
    next.run(req).await
}

/// Per-request structured log + trace id (debuggability complaint):
/// method, path, status, duration, priority, trace id — echoed back as
/// `x-pallama-trace-id` so clients can correlate.
#[allow(clippy::too_many_lines)] // trace+access+otlp+audit pipeline in one middleware
async fn request_log(
    State(state): State<Arc<AppState>>,
    req: Request<Body>,
    next: Next,
) -> Response {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(1);
    let trace = format!(
        "plm-{:x}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    );
    let mut req = req;
    req.extensions_mut().insert(TraceId(trace.clone()));
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let priority = req
        .headers()
        .get("x-pallama-priority")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("normal")
        .to_string();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    // E4 audit: buffer the body ONLY when audit is on AND this is an
    // audited generation lane — extract `model`, hand the bytes back
    // untouched. Audit off = zero request-path changes.
    let audit_on = state.audit_tx.is_some();
    let key = req
        .extensions()
        .get::<keys::KeyCtx>()
        .map(|k| k.name.clone());
    let mut model = None;
    if audit_on {
        let content_len = req
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok());
        if crate::audit::should_sniff(&method, &path, content_len) {
            let (parts, body) = req.into_parts();
            match axum::body::to_bytes(body, crate::audit::AUDIT_BODY_SNIFF_LIMIT).await {
                Ok(bytes) => {
                    model = crate::audit::extract_model(&bytes);
                    req = Request::from_parts(parts, Body::from(bytes));
                }
                Err(_) => {
                    // Over the sniff cap == over the gateway body limit; the
                    // outer DefaultBodyLimit would reject it identically.
                    return (
                        StatusCode::PAYLOAD_TOO_LARGE,
                        "request body exceeds the gateway limit",
                    )
                        .into_response();
                }
            }
        }
    }
    let started = std::time::Instant::now();
    let span_start_nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let mut resp = next.run(req).await;
    let ms = started.elapsed().as_millis();
    let status = resp.status().as_u16();
    if let Ok(v) = axum::http::HeaderValue::from_str(&trace) {
        resp.headers_mut().insert("x-pallama-trace-id", v);
    }
    // F64: auth runs INSIDE this layer now; the key arrives on the
    // response extensions (stamped by `auth`), not the request.
    let key = resp
        .extensions()
        .get::<keys::KeyCtx>()
        .map(|k| k.name.clone())
        .or(key);
    // E4 audit line: identity + outcome, never content. try_send — a
    // saturated writer drops (and counts) rather than stalling the
    // response.
    if let Some(tx) = &state.audit_tx {
        if crate::audit::AUDITED_PATHS.contains(&path.as_str()) {
            let line = crate::audit::AuditLine {
                ts,
                trace: trace.clone(),
                key,
                method: method.to_string(),
                path: path.clone(),
                model,
                status,
                ms,
                priority: priority.clone(),
                queue_depth: state.queue.depth(),
            };
            if tx.try_send(line).is_err() {
                state.audit_dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    if state.otlp.enabled() {
        let (trace_hex, span_hex) = otlp::new_ids();
        state.otlp.record(otlp::OtlpSpan {
            trace_id_hex: trace_hex,
            span_id_hex: span_hex,
            name: format!("{method} {path}"),
            start_unix_nano: span_start_nanos,
            end_unix_nano: span_start_nanos + ms * 1_000_000,
            http_status: status,
            attributes: vec![
                ("pallama.trace_id".to_string(), trace.clone()),
                ("http.request.method".to_string(), method.to_string()),
                ("url.path".to_string(), path.clone()),
                ("http.response.status_code".to_string(), status.to_string()),
                (
                    "pallama.queue_depth".to_string(),
                    state.queue.depth().to_string(),
                ),
            ],
        });
    }
    tracing::info!(
        target: "pallama::access",
        trace = %trace,
        method = %method,
        path = %path,
        status,
        ms,
        priority = %priority,
        queue_depth = state.queue.depth(),
        "request"
    );
    resp
}

// Flat route table; length is the surface itself, not complexity.
#[allow(clippy::too_many_lines)]
pub fn router(state: Arc<AppState>) -> Router {
    let openai_any = Router::new()
        .route("/v1/chat/completions", post(openai::openai_proxy))
        .route("/v1/completions", post(openai::openai_proxy))
        .route("/v1/embeddings", post(openai::embeddings))
        .route("/v1/rerank", post(openai::openai_proxy))
        .route("/v1/messages", post(anthropic::messages))
        // Full upstream API surface (routes verified in upstream server.cpp):
        // Responses API (current-gen OpenAI clients), audio transcriptions
        // (multipart; MTMD audio-in models), FIM infill, control-vector
        // steering, token counting, and the tokenize/apply-template dev
        // tools. All byte-stream proxied with model-affinity routing.
        .route("/v1/responses", post(openai::responses_api))
        .route(
            "/v1/batches",
            post(batch::create_batch).get(batch::list_batches),
        )
        .route("/v1/batches/{id}", get(batch::get_batch))
        .route("/v1/batches/{id}/cancel", post(batch::cancel_batch))
        .route("/v1/files", post(batch::upload_file))
        .route("/v1/files/{id}", get(batch::get_file))
        .route("/v1/files/{id}/content", get(batch::get_file_content))
        .route("/responses", post(openai::responses_api))
        .route("/v1/responses/{id}", get(openai::responses_get))
        .route(
            "/v1/audio/transcriptions",
            post(whisper::audio_transcriptions),
        )
        .route("/audio/transcriptions", post(whisper::audio_transcriptions))
        .route("/infill", post(openai::openai_proxy))
        .route("/v1/chat/completions/control", post(openai::openai_proxy))
        .route(
            "/v1/chat/completions/input_tokens",
            post(openai::openai_proxy),
        )
        .route("/v1/responses/input_tokens", post(openai::openai_proxy))
        .route("/responses/input_tokens", post(openai::openai_proxy))
        .route("/v1/messages/count_tokens", post(anthropic::count_tokens))
        .route("/tokenize", post(openai::openai_proxy))
        .route("/detokenize", post(openai::openai_proxy))
        .route("/apply-template", post(openai::openai_proxy))
        // Jina-style rerank alias (upstream serves both spellings).
        .route("/v1/reranking", post(openai::openai_proxy))
        // Engine-scoped upstream surfaces (no `model` in the body): slot
        // inspection/control, props, and the disconnected-stream family.
        // Model comes from X-Pallama-Model > ?model= > single hot child.
        .route(
            "/props",
            get(openai::scoped_proxy).post(openai::scoped_proxy),
        )
        .route("/slots", get(openai::scoped_proxy))
        .route("/slots/{id}", post(openai::scoped_proxy))
        .route(
            "/v1/stream",
            get(openai::scoped_proxy).delete(openai::scoped_proxy),
        )
        .route("/v1/streams/lookup", post(openai::scoped_proxy));

    let api = Router::new()
        .route("/api/version", get(ollama::version))
        .route("/api/tags", get(ollama::tags))
        .route("/api/show", post(ollama::show))
        .route("/api/delete", post(ollama::delete))
        .route("/api/ps", get(ollama::ps))
        .route("/api/pull", post(ollama::pull))
        .route("/api/chat", post(ollama::chat))
        .route("/api/embeddings", post(ollama::embeddings))
        .route("/api/embed", post(ollama::embed))
        .route("/api/rerank", post(ollama::rerank))
        .route("/api/generate", post(ollama::generate))
        .route("/api/evict", post(ollama::evict))
        .route(
            "/api/session",
            post(ollama::session).get(ollama::session_list),
        )
        .route("/api/sessions", get(sessions::list))
        .route("/api/why", get(ollama::why))
        .route("/api/watch", get(ollama::watch))
        .route(
            "/api/keys",
            get(keys_list).post(keys_add).delete(keys_remove),
        )
        .route("/api/keys/rotate", post(keys_rotate))
        .route("/.well-known/pallama", get(well_known));

    Router::new()
        .route("/healthz", get(openai::healthz))
        // Ecosystem probes (k8s, uptime checkers, OpenAI-compatible
        // clients) hit GET /health by convention; answer it the same way
        // as /healthz instead of 404.
        .route("/health", get(openai::healthz))
        .route("/metrics", get(ollama::metrics))
        .route("/api/events", get(ollama::events))
        .route("/v1/models", get(openai::models))
        .route(
            "/v1/adapters",
            get(openai::lora_adapters).post(openai::lora_adapters),
        )
        .merge(openai_any)
        .merge(api)
        // Session pins (R3): innermost layer — auth/CORS/logging/body
        // limit have already run; only header-carrying requests buffer.
        .layer(middleware::from_fn_with_state(
            state.clone(),
            sessions::pin_mw,
        ))
        // Hardening: 50 MiB request ceiling (audio uploads fit; nothing
        // legit is larger locally).
        .layer(axum::extract::DefaultBodyLimit::max(50 * 1024 * 1024))
        // CORS: opt-in per configured origins (empty = none, matching
        // pre-CORS behavior exactly; ["*"] = any). Non-browser clients
        // are unaffected either way. Sits OUTSIDE auth (F64): browser
        // preflights carry no bearer and must get a CORS answer, not a
        // bare 401.
        .layer(
            tower_http::cors::CorsLayer::new()
                .allow_origin(cors_allow_origin(&state.config.cors_origins))
                .allow_methods([
                    axum::http::Method::GET,
                    axum::http::Method::POST,
                    axum::http::Method::DELETE,
                ])
                .allow_headers(tower_http::cors::Any)
                .expose_headers([axum::http::header::HeaderName::from_static(
                    "x-pallama-trace-id",
                )]),
        )
        // Auth INSIDE request_log + CORS (F64): rejected secrets are
        // still access-logged and traced — brute-force probing no
        // longer flies blind.
        .layer(middleware::from_fn_with_state(state.clone(), auth))
        // request_log: outermost — every request, including 401s and
        // preflights, gets a trace id and an access-log line.
        .layer(middleware::from_fn_with_state(state.clone(), request_log))
        .with_state(state)
}

/// Translate the config's origin list into tower-http's `AllowOrigin`.
/// Empty list -> empty allow-list (no ACAO header is ever emitted —
/// exactly the pre-CORS behavior).
fn cors_allow_origin(origins: &[String]) -> tower_http::cors::AllowOrigin {
    if origins.iter().any(|o| o == "*") {
        tower_http::cors::AllowOrigin::any()
    } else {
        tower_http::cors::AllowOrigin::list(
            origins
                .iter()
                .filter_map(|o| o.parse::<axum::http::HeaderValue>().ok()),
        )
    }
}

/// Persist the live key set back to config.toml ([[keys]] tables).
/// Full-file round-trip (the established `create`/`config set` pattern):
/// parse -> swap keys -> validate -> atomic write (temp + rename, F67).
/// A process-wide mutex excludes the other config.toml writers in this
/// daemon (SIGHUP reload reads are lock-free but writes serialize), so
/// two racing mutations can't clobber each other with a stale read.
/// Returns an error response on failure (the live registry is already
/// updated; a failed persist is loud, never silent — the next daemon
/// restart loses the change).
fn persist_keys(state: &AppState, entries: &[ApiKey]) -> Response {
    static PERSIST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = PERSIST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let path = state.dirs.config_file();
    let mut cfg = if path.exists() {
        match std::fs::read_to_string(&path)
            .map_err(|e| e.to_string())
            .and_then(|raw| pallama_core::Config::from_toml(&raw).map_err(|e| e.to_string()))
        {
            Ok(c) => c,
            Err(e) => return error_response(500, &format!("config.toml unreadable: {e}")),
        }
    } else {
        pallama_core::Config::default()
    };
    cfg.keys = entries.to_vec();
    if let Err(e) = cfg.validate() {
        return error_response(400, &format!("generated config invalid: {e}"));
    }
    // F67: atomic write (temp+rename) + timestamped .bak via the shared
    // core helper — a crash mid-write can never truncate config.toml and
    // an overwrite is always recoverable.
    let write = cfg
        .to_toml()
        .map_err(|e| e.to_string())
        .and_then(|t| pallama_core::persist_config(&path, &t).map_err(|e| e.to_string()));
    match write {
        Ok(()) => (StatusCode::OK, axum::Json(serde_json::json!({"ok": true}))).into_response(),
        Err(e) => error_response(500, &format!("write {}: {e}", path.display())),
    }
}

fn error_response(code: u16, msg: &str) -> Response {
    (
        StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        axum::Json(
            serde_json::json!({"error": {"message": msg, "type": "pallama_error", "code": code}}),
        ),
    )
        .into_response()
}

/// Redacted secret for listings: enough to recognize, not enough to use.
fn redact(secret: &str) -> String {
    let prefix: String = secret.chars().take(8).collect();
    format!("{prefix}…")
}

/// Management guard: when keys exist, only UNSCOPED keys (empty model
/// list = admin semantics) may manage the key set. When no keys exist,
/// the gateway is authless and management is refused — bootstrap the
/// first key via config.toml `[[keys]]` or `PALLAMA_KEYS` (never let an
/// open gateway mint its own credentials).
#[allow(clippy::result_large_err)] // Response is the handlers' currency here
fn management_authorized(state: &AppState, key: Option<&keys::KeyCtx>) -> Result<(), Response> {
    if state.keys.is_empty() {
        return Err(error_response(
            403,
            "no keys configured: add the first [[keys]] entry in config.toml (or PALLAMA_KEYS env) and restart",
        ));
    }
    match key {
        None => Err(error_response(401, "missing bearer token")),
        Some(k) => {
            let admin = state
                .keys
                .entries()
                .into_iter()
                .find(|e| e.name == k.name)
                .is_some_and(|e| e.models.is_empty());
            if admin {
                Ok(())
            } else {
                Err(error_response(
                    403,
                    &format!(
                        "key {:?} is model-scoped; only unscoped (admin) keys manage keys",
                        k.name
                    ),
                ))
            }
        }
    }
}

async fn keys_list(
    State(state): State<Arc<AppState>>,
    key: Option<Extension<keys::KeyCtx>>,
) -> Response {
    if let Err(resp) = management_authorized(&state, key.as_ref().map(|e| &e.0)) {
        return resp;
    }
    let usage: std::collections::HashMap<String, (String, u64, u64)> = state
        .keys
        .usage_snapshot()
        .into_iter()
        .map(|(name, day, req, tok)| (name, (day, req, tok)))
        .collect();
    let keys: Vec<serde_json::Value> = state
        .keys
        .entries()
        .into_iter()
        .map(|k| {
            let (day, req, tok) = usage
                .get(&k.name)
                .cloned()
                .unwrap_or_else(|| ("-".to_string(), 0, 0));
            serde_json::json!({
                "name": k.name,
                "key": redact(&k.key),
                "models": k.models,
                "rpm": k.rpm,
                "tpm": k.tpm,
                "daily_tokens": k.daily_tokens,
                "max_concurrent": k.max_concurrent,
                "weight": k.effective_weight(),
                "usage": {"day": day, "requests": req, "tokens": tok},
            })
        })
        .collect();
    (
        StatusCode::OK,
        axum::Json(serde_json::json!({
            "keys": keys,
            "token_accounting": "exact on chat-family routes with usage payloads",
        })),
    )
        .into_response()
}

async fn keys_add(
    State(state): State<Arc<AppState>>,
    key: Option<Extension<keys::KeyCtx>>,
    axum::Json(body): axum::Json<serde_json::Value>,
) -> Response {
    if let Err(resp) = management_authorized(&state, key.as_ref().map(|e| &e.0)) {
        return resp;
    }
    let Some(name) = body
        .get("name")
        .and_then(|v| v.as_str())
        .map(str::to_string)
    else {
        return error_response(400, "body needs a string `name`");
    };
    if name.trim().is_empty() {
        return error_response(400, "name must not be empty");
    }
    // F68: remove/rotate address keys via `?name=` — restrict the
    // charset at creation so every name stays manageable over HTTP.
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
    {
        return error_response(
            400,
            "key name may only contain [A-Za-z0-9._-] (it is addressed via ?name= queries)",
        );
    }
    if state.keys.entries().iter().any(|k| k.name == name) {
        return error_response(
            409,
            &format!("key {name:?} already exists (delete it first)"),
        );
    }
    let models: Vec<String> = body
        .get("models")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|m| m.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    #[allow(clippy::result_large_err)]
    let num = |field: &str, default: u64| -> Result<u64, Response> {
        match body.get(field) {
            None | Some(serde_json::Value::Null) => Ok(default),
            Some(v) => v.as_u64().ok_or_else(|| {
                error_response(400, &format!("{field} must be a non-negative integer"))
            }),
        }
    };
    // F65: the teaching 400s above are answers, not decoration — an
    // invalid rpm/tpm/daily_tokens must be REJECTED, not silently
    // coerced to 0 (= unlimited).
    #[allow(clippy::result_large_err)] // the 400 Response IS the error type here
    let nums = || -> Result<(u64, u64, u64, u64, u64), Response> {
        Ok((
            num("rpm", 0)?,
            num("tpm", 0)?,
            num("daily_tokens", 0)?,
            num("max_concurrent", 0)?,
            num("weight", 1)?,
        ))
    };
    let (rpm, tpm, daily_tokens, max_concurrent, weight) = match nums() {
        Ok(v) => v,
        Err(r) => return r,
    };
    let entry = ApiKey {
        key: format!("plm_{}", rand_secret()),
        name,
        models,
        rpm: u32::try_from(rpm).unwrap_or(u32::MAX),
        tpm,
        daily_tokens,
        max_concurrent: u32::try_from(max_concurrent).unwrap_or(u32::MAX),
        weight: u32::try_from(weight).unwrap_or(1),
    };
    state.keys.upsert(entry.clone());
    let entries = state.keys.entries();
    let persist = persist_keys(&state, &entries);
    if persist.status() != StatusCode::OK {
        // Roll the registry back: file is the source of truth.
        state.keys.remove(&entry.name);
        return persist;
    }
    (
        StatusCode::CREATED,
        // The secret appears exactly once, here — listings only ever show
        // the redacted prefix.
        axum::Json(serde_json::json!({
            "created": entry.name,
            "key": entry.key,
            "models": entry.models,
            "rpm": entry.rpm,
            "tpm": entry.tpm,
            "daily_tokens": entry.daily_tokens,
            "max_concurrent": entry.max_concurrent,
        })),
    )
        .into_response()
}

async fn keys_remove(
    State(state): State<Arc<AppState>>,
    key: Option<Extension<keys::KeyCtx>>,
    axum::extract::RawQuery(q): axum::extract::RawQuery,
) -> Response {
    if let Err(resp) = management_authorized(&state, key.as_ref().map(|e| &e.0)) {
        return resp;
    }
    let Some(name) = q
        .as_deref()
        .and_then(|query| query.split('&').find_map(|p| p.strip_prefix("name=")))
        .map(str::to_string)
    else {
        return error_response(400, "query needs ?name=<key name>");
    };
    if state.keys.entries().len() == 1 {
        return error_response(
            409,
            "refusing to remove the last key — that would leave the gateway authless; add a replacement first",
        );
    }
    if !state.keys.remove(&name) {
        return error_response(404, &format!("no key named {name:?}"));
    }
    let entries = state.keys.entries();
    let persist = persist_keys(&state, &entries);
    if persist.status() != StatusCode::OK {
        // Nothing to roll back (remove already ran) — the error response
        // tells the operator the file write failed.
        return persist;
    }
    (
        StatusCode::OK,
        axum::Json(serde_json::json!({"removed": name})),
    )
        .into_response()
}

/// POST /api/keys/rotate?name=N — mint a fresh secret for a key,
/// keeping its name/scopes/limits and usage history. The old secret
/// dies immediately; the new one is shown exactly once.
async fn keys_rotate(
    State(state): State<Arc<AppState>>,
    key: Option<Extension<keys::KeyCtx>>,
    axum::extract::RawQuery(q): axum::extract::RawQuery,
) -> Response {
    if let Err(resp) = management_authorized(&state, key.as_ref().map(|e| &e.0)) {
        return resp;
    }
    let Some(name) = q
        .as_deref()
        .and_then(|query| query.split('&').find_map(|p| p.strip_prefix("name=")))
        .map(str::to_string)
    else {
        return error_response(400, "query needs ?name=<key name>");
    };
    let entries = state.keys.entries();
    let Some(entry) = entries.iter().find(|e| e.name == name).cloned() else {
        return error_response(404, &format!("no key named {name:?}"));
    };
    let rotated = ApiKey {
        key: format!("plm_{}", rand_secret()),
        ..entry.clone()
    };
    state.keys.upsert(rotated.clone());
    let persist = persist_keys(&state, &state.keys.entries());
    if persist.status() != StatusCode::OK {
        // F66: mirror keys_add's rollback — the file is the source of
        // truth; without this the shown-once new secret dies silently
        // at the next restart when the old file entry wins.
        state.keys.upsert(entry);
        return persist;
    }
    (
        StatusCode::OK,
        axum::Json(serde_json::json!({
            "rotated": name,
            "key": rotated.key, // shown exactly once
        })),
    )
        .into_response()
}

/// GET /.well-known/pallama — capability discovery: routes, limits,
/// engine identity. Clients introspect instead of probing.
async fn well_known(State(state): State<Arc<AppState>>) -> Response {
    let engine = state
        .with_store(|s| s.active_engine().ok().flatten())
        .flatten()
        .map_or(
            serde_json::Value::Null,
            |e| serde_json::json!({"tag": e.tag, "asset": e.asset, "kind": e.kind.as_str()}),
        );
    axum::Json(serde_json::json!({
        "name": "pallama",
        "version": env!("CARGO_PKG_VERSION"),
        "engine": engine,
        // F52: the messages lane TRANSLATES and refuses passthrough —
        // the label must not advertise passthrough semantics.
        "apis": ["openai", "ollama", "anthropic"],
        // F69: full route census (verified against the router table).
        "endpoints": {
            "openai": ["/v1/chat/completions", "/v1/chat/completions/control",
                       "/v1/completions", "/v1/embeddings", "/v1/rerank", "/v1/reranking",
                       "/v1/responses", "/v1/responses/{id}", "/v1/responses/input_tokens",
                       "/v1/messages", "/v1/messages/count_tokens", "/v1/models",
                       "/v1/adapters", "/v1/batches", "/v1/batches/{id}",
                       "/v1/batches/{id}/cancel", "/v1/files", "/v1/files/{id}",
                       "/v1/files/{id}/content", "/v1/streams/lookup",
                       "/v1/audio/transcriptions", "/infill", "/tokenize", "/detokenize",
                       "/apply-template", "/slots", "/slots/{id}", "/responses",
                       "/responses/input_tokens"],
            "ollama": ["/api/chat", "/api/generate", "/api/tags", "/api/ps", "/api/show",
                       "/api/embeddings", "/api/embed", "/api/rerank", "/api/pull",
                       "/api/delete", "/api/events", "/api/version"],
            "pallama": ["/api/evict", "/api/session", "/api/sessions", "/api/why",
                        "/api/watch", "/api/keys", "/api/keys/rotate",
                        "/.well-known/pallama", "/metrics", "/healthz", "/health"],
        },
        "headers": ["x-pallama-num-ctx", "x-pallama-deadline-ms", "x-pallama-priority",
                    "x-pallama-enforce", "x-pallama-trace-id", "x-pallama-status",
                    "x-pallama-warnings", "x-pallama-session",
                    "x-pallama-cache", "x-pallama-cache-ttl", "x-pallama-cache-threshold"],
        "features": {
            "keys": !state.keys.is_empty(),
            "tls": !state.config.tls_cert.is_empty(),
            "otlp": state.otlp.enabled(),
            "remotes": state.config.remotes.iter().map(|r| r.name.clone()).collect::<Vec<_>>(),
            "singleflight": state.config.singleflight,
            "session_bank": state.config.session_bank,
            "prompt_preflight": state.config.prompt_preflight,
            "session_pins": state.config.session_keep_secs > 0,
            "semantic_cache": state.config.semantic_cache.enabled
                && state.config.semantic_cache.model.is_some(),
        },
    }))
    .into_response()
}

/// 32 hex chars of OS entropy (128 bits) for `plm_` secrets.
fn rand_secret() -> String {
    use std::fmt::Write as _;
    let mut buf = [0u8; 16];
    getrandom::fill(&mut buf).expect("OS entropy source");
    let mut out = String::with_capacity(32);
    for b in buf {
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// Last value of a bare Prometheus counter (`name<space>value`) in a
/// /metrics text body. Labeled lines (`name{...} value`) fail the value
/// parse and are skipped; `None` when absent (e.g. spec counters on a
/// dense child).
fn counter(text: &str, name: &str) -> Option<u64> {
    text.lines()
        .filter_map(|line| line.strip_prefix(name).map(str::trim))
        .filter_map(|v| v.parse::<u64>().ok())
        .next_back()
}

/// Serve the gateway until `shutdown` resolves (SIGTERM/SIGINT), then
/// drain: stop accepting, stop children, exit clean. TLS when
/// `tls_cert`/`tls_key` are configured, plain HTTP otherwise.
#[allow(clippy::too_many_lines)]
/// Serve returned because the configured listen address is already
/// taken (another server — typically ollama — owns the port). The CLI
/// maps this to `EXIT_BIND_CONFLICT` and the systemd unit pins
/// `RestartPreventExitStatus=` to it: a hard port conflict must fail
/// loudly once, not crash-loop `Restart=always` into the journal.
#[derive(Debug)]
pub struct BindConflict(pub std::io::Error);

impl std::fmt::Display for BindConflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "listen address already in use: {}", self.0)
    }
}

impl std::error::Error for BindConflict {}

/// `pallama serve` exit code for an unrecoverable bind conflict.
pub const EXIT_BIND_CONFLICT: i32 = 3;

// One wiring function: listener, TLS, flushers, teardown — all the
// serve-lifetime setup in a single readable sequence.
#[allow(clippy::too_many_lines)]
pub async fn serve(
    state: Arc<AppState>,
    host: &str,
    port: u16,
    shutdown: std::pin::Pin<Box<dyn futures::Future<Output = ()> + Send>>,
) -> anyhow::Result<()> {
    let app = router(state.clone());
    let addr = format!("{host}:{port}");
    // Prompt-cache hit-rate + spec-accept poller (A16/G3): every 60 s,
    // sum the children's Prometheus counters, compute the window delta
    // rate and EWMA it into the supervisor's CacheHints — the adaptive
    // --cache-ram clamp's input and the spec gauge's source. Bounded
    // (2 s) fetches; failures just skip a window.
    let hint_task = {
        let state = Arc::clone(&state);
        tokio::spawn(async move {
            let mut prev: (u64, u64) = (0, 0); // (Σ cache_n, Σ prompt_n)
            let mut prev_spec: (u64, u64) = (0, 0); // (Σ accepted, Σ drafted)
            let mut ewma: Option<f64> = None;
            let mut ewma_spec: Option<f64> = None;
            let mut interval = tokio::time::interval(std::time::Duration::from_mins(1));
            interval.tick().await; // immediate first tick: skip (empty)
            loop {
                interval.tick().await;
                let mut totals = (0u64, 0u64);
                let mut spec_totals = (0u64, 0u64);
                for e in state.sup.live_http_endpoints() {
                    let pallama_core::profile::Endpoint::Tcp { host, port } = &e.endpoint else {
                        continue;
                    };
                    // Child Prometheus text: aggregate counters (the /slots
                    // per-slot stats are short-lived and unreliable — the
                    // metrics counters are the durable truth).
                    let url = format!("http://{host}:{port}/metrics");
                    let Ok(resp) = crate::proxy::child_auth(
                        state
                            .http
                            .get(&url)
                            .timeout(std::time::Duration::from_secs(2)),
                        &e,
                    )
                    .send()
                    .await
                    else {
                        continue;
                    };
                    let Ok(text) = resp.text().await else {
                        continue;
                    };
                    if let (Some(c), Some(p)) = (
                        counter(&text, "llamacpp:prompt_tokens_cached_total"),
                        counter(&text, "llamacpp:prompt_tokens_total"),
                    ) {
                        totals.0 += c;
                        totals.1 += p;
                    }
                    // G3: spec-decoding acceptance (only present when a
                    // spec pair is live on that child).
                    if let (Some(a), Some(d)) = (
                        counter(&text, "llamacpp:spec_decode_num_accepted_tokens_total"),
                        counter(&text, "llamacpp:spec_decode_num_draft_tokens_total"),
                    ) {
                        spec_totals.0 += a;
                        spec_totals.1 += d;
                    }
                }
                let d_cache = totals.0.saturating_sub(prev.0);
                let d_prompt = totals.1.saturating_sub(prev.1);
                prev = totals;
                let denom = d_cache + d_prompt;
                if denom > 0 {
                    // Token counts far below 2^52: the precision cast is
                    // exact for any realistic window.
                    #[allow(clippy::cast_precision_loss)]
                    let rate = d_cache as f64 / denom as f64;
                    ewma = Some(match ewma {
                        Some(e) => e * 0.7 + rate * 0.3,
                        None => rate,
                    });
                    if let Some(e) = ewma {
                        state.sup.cache_hint.set(e);
                    }
                }
                let d_acc = spec_totals.0.saturating_sub(prev_spec.0);
                let d_draft = spec_totals.1.saturating_sub(prev_spec.1);
                prev_spec = spec_totals;
                if d_draft > 0 {
                    #[allow(clippy::cast_precision_loss)]
                    let rate = d_acc.min(d_draft) as f64 / d_draft as f64;
                    ewma_spec = Some(match ewma_spec {
                        Some(e) => e * 0.7 + rate * 0.3,
                        None => rate,
                    });
                    if let Some(e) = ewma_spec {
                        state.sup.spec_accept.set(e);
                    }
                }
            }
        })
    };
    // OTLP flusher (no-op task when the endpoint is unset).
    let otlp_task = {
        let otlp = Arc::clone(&state.otlp);
        // F81: bounded POSTs — a black-hole collector used to wedge the
        // flusher forever (the 5s select only bounds the first-span wait).
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap_or_default();
        tokio::spawn(async move { otlp.run(http).await })
    };
    if state.config.tls_cert.is_empty() {
        let listener = tokio::net::TcpListener::bind(&addr).await.map_err(|e| {
            // Typed so the CLI can exit 3 and the systemd unit's
            // RestartPreventExitStatus=3 stops the Restart=always loop.
            anyhow::Error::new(BindConflict(e)).context(format!(
                "bind {addr} — another server (ollama?) on this port? stop it or set PALLAMA_PORT"
            ))
        })?;
        // Record the real bound address for the batch worker (config port
        // may be 0 = ephemeral; fold wildcard binds to loopback-reachable).
        if let Ok(sa) = listener.local_addr() {
            let h = sa.ip().to_string();
            let h = if h == "0.0.0.0" || h == "::" {
                "127.0.0.1".to_string()
            } else {
                h
            };
            let _ = state.http_addr.set((h, sa.port()));
        }
        tracing::info!("pallama listening on http://{addr} (OpenAI + ollama APIs)");
        tracing::info!("powered by upstream llama.cpp, mistral.rs and SGLang — unmodified engines");
        // TCP_NODELAY on every accepted socket: NDJSON/SSE streams are strings
        // of small writes, and with Nagle the first content packet waits for
        // the client's delayed ACK of the headers/role packet — a flat
        // +40-60ms on first-token latency for every streaming response
        // (live-measured: child's first content SSE at ~77ms reached the
        // client at ~135ms; Go net/http, ollama's stack, sets nodelay too).
        let listener = listener.tap_io(|stream| {
            let _ = stream.set_nodelay(true);
        });
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown)
            .await
            .map_err(|e| anyhow::anyhow!("server: {e}"))?;
    } else {
        // Feature unification across the dep graph can leave rustls with
        // both providers enabled; pick ours deterministically (idempotent
        // if already installed).
        let _ = rustls::crypto::ring::default_provider().install_default();
        let server_config =
            rustls_server_config(&state.config).map_err(|e| anyhow::anyhow!("TLS config: {e}"))?;
        let rustls_config =
            axum_server::tls_rustls::RustlsConfig::from_config(std::sync::Arc::new(server_config));
        let bind_addr: std::net::SocketAddr = addr
            .parse()
            .map_err(|e| anyhow::anyhow!("bind address {addr}: {e}"))?;
        // F69: record the bound address on the TLS branch too — the
        // batch worker's loopback replay must not guess config.port
        // (invalid when 0 = ephemeral). Binding with tokio first gives
        // us the real port the same way the plain-HTTP branch gets it.
        let listener = tokio::net::TcpListener::bind(bind_addr)
            .await
            .map_err(|e| anyhow::anyhow!("bind {addr}: {e}"))?;
        if let Ok(sa) = listener.local_addr() {
            let h = sa.ip().to_string();
            let h = if h == "0.0.0.0" || h == "::" {
                "127.0.0.1".to_string()
            } else {
                h
            };
            let _ = state.http_addr.set((h, sa.port()));
        }
        let handle = axum_server::Handle::new();
        let h2 = handle.clone();
        tokio::spawn(async move {
            shutdown.await;
            h2.graceful_shutdown(Some(std::time::Duration::from_secs(10)));
        });
        // axum-server takes a blocking std listener (its serve loop
        // re-asserts non-blocking mode itself); tokio's into_std hands
        // over the already-bound socket without a rebind race.
        let std_listener = listener
            .into_std()
            .map_err(|e| anyhow::anyhow!("tls listener handoff: {e}"))?;
        axum_server::from_tcp_rustls(std_listener, rustls_config)
            .handle(handle)
            .serve(app.into_make_service())
            .await
            .map_err(|e| anyhow::anyhow!("tls server: {e}"))?;
    }
    otlp_task.abort();
    hint_task.abort();
    // H8: the lazy whisper-server child (if any request spawned one).
    state.whisper.shutdown().await;
    state.sup.shutdown_all().await?;
    Ok(())
}

/// Build the rustls server config from the configured PEM files.
fn rustls_server_config(
    config: &pallama_core::Config,
) -> Result<rustls::ServerConfig, Box<dyn std::error::Error + Send + Sync>> {
    use rustls::pki_types::CertificateDer;
    let certs: Vec<CertificateDer> = rustls_pemfile::certs(&mut std::io::BufReader::new(
        std::fs::File::open(&config.tls_cert)?,
    ))
    .collect::<Result<_, _>>()?;
    if certs.is_empty() {
        return Err("tls_cert contains no certificates".into());
    }
    let key = rustls_pemfile::private_key(&mut std::io::BufReader::new(std::fs::File::open(
        &config.tls_key,
    )?))?
    .ok_or("tls_key contains no private key")?;
    Ok(rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)?)
}

#[cfg(test)]
mod tests {
    use super::counter;

    #[test]
    #[allow(non_snake_case)]
    fn unit__counter__last_bare_value_and_labeled_skipped() {
        let body = "# HELP x y\n# TYPE x counter\n\
            llamacpp:prompt_tokens_total 10\n\
            llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"} 1 2\n\
            llamacpp:prompt_tokens_total 42\n";
        assert_eq!(counter(body, "llamacpp:prompt_tokens_total"), Some(42));
        // Labeled histogram lines never match the bare-name lookup.
        assert_eq!(
            counter(
                body,
                "llamacpp:spec_decode_num_accepted_tokens_per_pos_total"
            ),
            None
        );
        assert_eq!(counter(body, "llamacpp:missing_total"), None);
    }

    #[tokio::test]
    #[allow(non_snake_case)]
    async fn unit__supervision_error__unsupported_model_maps_to_400_with_remedy() {
        use crate::proxy::supervision_error;
        use pallama_runtime::SupervisionError;
        let resp = supervision_error(&SupervisionError::UnsupportedModel(
            "safetensors model /m.d is not servable by the llama.cpp engine \
             — run: pallama engine install --kind sglang"
                .to_string(),
        ));
        assert_eq!(resp.status(), 400);
        let body = axum::body::to_bytes(
            resp.into_body(),
            usize::MAX,
        )
        .await
        .expect("read body");
        let v: serde_json::Value = serde_json::from_slice(&body).expect("json body");
        assert_eq!(
            v["error"]["message"]
                .as_str()
                .expect("message")
                .contains("engine install --kind sglang"),
            true,
            "teaching remedy must survive the mapping"
        );
    }
}

#[cfg(test)]
#[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
mod bind_conflict_tests {
    use super::BindConflict;

    #[test]
    fn unit__bind_conflict__downcast_survives_context_chain() {
        // The CLI exit-3 path relies on anyhow::downcast_ref finding the
        // typed cause through a .context() wrapper — pin that mechanic.
        let e = anyhow::Error::new(BindConflict(std::io::Error::other(
            "Address already in use",
        )))
        .context("bind 127.0.0.1:11434 — another server?");
        assert!(e.downcast_ref::<BindConflict>().is_some());
        assert_eq!(super::EXIT_BIND_CONFLICT, 3);
    }
}
