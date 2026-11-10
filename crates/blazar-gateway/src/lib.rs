//! Blazar gateway: `OpenAI` + ollama-compat HTTP surface over the
//! supervisor. Router assembly + bearer auth.

pub mod anthropic;
pub mod audit;
pub mod batch;
pub mod benchmarks;
pub mod bestof;
pub mod cache_bust;
pub mod capacity;
pub mod cascade;
pub mod classify;
pub mod config_api;
pub mod console;
pub mod engine_recommend;
pub mod error_codes;
pub mod explain;
pub mod fabric;
pub mod federation;
pub mod fleet;
pub mod histogram;
pub mod host_guard;
pub mod http_pool;
pub mod images;
pub mod jobs;
pub mod model_doctor;
pub mod quantiles;
pub mod requests;
pub mod usage;
pub mod voice_tee;

pub mod failover;
pub mod keys;
pub mod latechunk;
pub mod mcp;
pub mod metadata_card;
pub mod ollama;
pub mod openai;
pub mod otlp;
pub mod preflight;
pub mod prompt_recipe;
pub mod proxy;
pub mod queue;
pub mod realtime;
pub mod remotes;
pub mod responses;
pub mod scrub;
pub mod semcache;
pub mod sentinel;
pub mod sessions;
pub mod state;
pub mod translate;
pub mod vram;

pub mod tts;
pub mod whisper;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract::{Extension, State};
use axum::http::{Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::serve::ListenerExt;

use blazar_core::ApiKey;
use state::AppState;

/// Request-scoped trace id, inserted by `request_log` (the outermost
/// layer) so inner handlers can correlate sentinel records with the
/// response header clients see.
#[derive(Clone)]
pub struct TraceId(pub String);

/// Per-response usage accumulator the metered body stream folds into:
/// the trailing byte window (last `usage::TAIL_LIMIT` bytes hold the
/// final usage chunk across dialects) plus first/last chunk stamps for
/// TTFT and generation-duration accounting.
#[derive(Default, Clone)]
struct UsageTap {
    tail: Vec<u8>,
    first: Option<std::time::Instant>,
    last: Option<std::time::Instant>,
}

/// Bearer auth: active iff any `[[keys]]` exist (children stay loopback
/// and unauthenticated). Resolves the presented secret to its key entry
/// and tags the request with `KeyCtx` for scope/rate/accounting.
/// Secret sources (in priority order): `Authorization: Bearer <secret>`
/// (`OpenAI` convention) and `x-api-key: <secret>` (Anthropic convention -
/// Claude Code & friends never send Authorization).
async fn auth(State(state): State<Arc<AppState>>, req: Request<Body>, next: Next) -> Response {
    if !state.keys.is_empty() {
        let path = req.uri().path();
        // Fleet membership traffic rides its OWN credentials - the
        // shared fleet token, invite codes, worker session tokens -
        // never API keys, so those routes self-gate instead of the
        // key middleware (F-GP11a-1: keys-mode must not 401 the join
        // handshake). `/api/fleet/workers` is NOT exempt: it is
        // management-class topology and stays key-authenticated.
        let fleet_self_gated = path == "/api/fleet/join"
            || path == "/api/fleet/heartbeat"
            || path == "/api/fleet/leave"
            || path == "/api/fleet/invite";
        let open = path == "/healthz" || path == "/health" || fleet_self_gated;
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
                    // probe scopes. The lease rides the response body -
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
                    // longer read KeyCtx off the request - stamp it on
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
                        axum::Json(serde_json::json!({"error": {"message": "missing or invalid API key (Authorization: Bearer or x-api-key)", "type": "blazar_error", "code": 401, "blazar_code": "UNAUTHORIZED"}})),
                    )
                        .into_response();
                }
            }
        }
    }
    next.run(req).await
}

/// Buffer a small non-2xx response body and return a truncated excerpt
/// for the access log (F-S4-3): the refusing gate's teaching message
/// otherwise never reaches any log. Gate on the body size hint (not the
/// content-length header, which hyper only stamps at send time):
/// gateway-generated errors are fixed-size and announce themselves;
/// unknown-size or over-cap error streams pass through untouched, and a
/// stream that violates its own announcement is rebuilt from the empty
/// prefix so the client still receives a well-formed error. 2xx never
/// buffers.
async fn error_body_excerpt(resp: &mut Response) -> Option<String> {
    use http_body::Body as _;
    const ERR_BODY_LOG_CAP: usize = 8 * 1024;
    const ERR_EXCERPT_CHARS: usize = 300;
    if resp.status().as_u16() < 400 {
        return None;
    }
    let hint = resp.body().size_hint();
    let small = hint
        .exact()
        .or(hint.upper())
        .is_some_and(|len| len <= ERR_BODY_LOG_CAP as u64);
    if !small {
        return None;
    }
    let taken = std::mem::take(resp);
    let (parts, body) = taken.into_parts();
    if let Ok(bytes) = axum::body::to_bytes(body, ERR_BODY_LOG_CAP).await {
        let text = String::from_utf8_lossy(&bytes);
        let cut = text
            .char_indices()
            .nth(ERR_EXCERPT_CHARS)
            .map_or(text.len(), |(i, _)| i);
        let excerpt = text[..cut].to_string();
        *resp = Response::from_parts(parts, Body::from(bytes));
        Some(excerpt)
    } else {
        *resp = Response::from_parts(parts, Body::from(axum::body::Bytes::default()));
        None
    }
}

/// Per-request structured log + trace id (debuggability complaint):
/// method, path, status, duration, priority, trace id - echoed back as
/// `x-blazar-trace-id` so clients can correlate.
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
        .get("x-blazar-priority")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("normal")
        .to_string();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    // E4 audit + F4 usage: buffer the body ONLY when audit is on or
    // this is a metered generation lane - extract `model`, hand the
    // bytes back untouched. Neither on = zero request-path changes.
    let audit_on = state.audit_tx.is_some();
    let usage_lane = crate::usage::USAGE_PATHS.contains(&path.as_str());
    let key = req
        .extensions()
        .get::<keys::KeyCtx>()
        .map(|k| k.name.clone());
    let mut model = None;
    if audit_on || usage_lane {
        let content_len = req
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok());
        let sniff = method == axum::http::Method::POST
            && content_len.is_none_or(|n| n <= crate::audit::AUDIT_BODY_SNIFF_LIMIT as u64)
            && (usage_lane || crate::audit::AUDITED_PATHS.contains(&path.as_str()));
        if sniff {
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
        resp.headers_mut().insert("x-blazar-trace-id", v);
    }
    // F64: auth runs INSIDE this layer now; the key arrives on the
    // response extensions (stamped by `auth`), not the request.
    let key = resp
        .extensions()
        .get::<keys::KeyCtx>()
        .map(|k| k.name.clone())
        .or(key);
    // E4 audit line: identity + outcome, never content. try_send - a
    // saturated writer drops (and counts) rather than stalling the
    // response.
    // The audit line takes ownership of `model`; the usage tap needs
    // the name past that point, so lift it out first.
    let usage_model = model.clone();
    if let Some(tx) = &state.audit_tx
        && crate::audit::AUDITED_PATHS.contains(&path.as_str())
    {
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
                ("blazar.trace_id".to_string(), trace.clone()),
                ("http.request.method".to_string(), method.to_string()),
                ("url.path".to_string(), path.clone()),
                ("http.response.status_code".to_string(), status.to_string()),
                (
                    "blazar.queue_depth".to_string(),
                    state.queue.depth().to_string(),
                ),
            ],
        });
    }
    // F-S4-3: attach the refusing gate's teaching excerpt to the
    // access line so a bare "status=400" is never the only evidence.
    let err_excerpt = error_body_excerpt(&mut resp).await;
    if let Some(err) = &err_excerpt {
        tracing::info!(
            target: "blazar::access",
            trace = %trace,
            method = %method,
            path = %path,
            status,
            ms,
            priority = %priority,
            queue_depth = state.queue.depth(),
            err = %err,
            "request"
        );
    } else {
        tracing::info!(
            target: "blazar::access",
            trace = %trace,
            method = %method,
            path = %path,
            status,
            ms,
            priority = %priority,
            queue_depth = state.queue.depth(),
            "request"
        );
    }
    // F4 usage: re-stream successful generation-lane responses through
    // a tap that keeps the last 8 KiB (usage JSON rides the final
    // chunk in every dialect) plus first/last chunk timings. The meter
    // records at stream end - one hash bump, no store I/O on this path
    // (flushing is the periodic task's job). No model or non-2xx =
    // nothing metered.
    if usage_lane
        && (200..300).contains(&status)
        && let Some(model_name) = usage_model
    {
        let (parts, body) = resp.into_parts();
        let tap = std::sync::Arc::new(std::sync::Mutex::new(UsageTap::default()));
        let tap_map = std::sync::Arc::clone(&tap);
        let metered = futures::StreamExt::map(body.into_data_stream(), move |chunk| {
            if let Ok(bytes) = chunk.as_ref() {
                let mut t = tap_map.lock().expect("usage tap poisoned");
                let now = std::time::Instant::now();
                t.first.get_or_insert(now);
                t.last = Some(now);
                // Keep the trailing window; the usage object is in the
                // final SSE/NDJSON line, never the head.
                t.tail.extend_from_slice(bytes);
                let overflow = t.tail.len().saturating_sub(crate::usage::TAIL_LIMIT);
                if overflow > 0 {
                    t.tail.drain(..overflow);
                }
            }
            chunk
        });
        let tap_end = std::sync::Arc::clone(&tap);
        let meter = std::sync::Arc::clone(&state.usage);
        let finalized = futures::StreamExt::chain(
            metered,
            futures::stream::once(async move {
                let tap = tap_end.lock().expect("usage tap poisoned").clone();
                let (in_tok, out_tok, cached_tok) = crate::usage::sniff_usage_full(&tap.tail);
                let gen_secs = match (tap.first, tap.last) {
                    (Some(f), Some(l)) => l.saturating_duration_since(f).as_secs_f64(),
                    _ => 0.0,
                };
                // TTFT is measured from request entry (before the
                // handler chain), the user-felt first-token latency.
                let ttft_ms = tap
                    .first
                    .map(|f| f.duration_since(started).as_secs_f64() * 1000.0);
                meter.record(
                    &model_name,
                    in_tok.unwrap_or(0),
                    out_tok.unwrap_or(0),
                    cached_tok.unwrap_or(0),
                    gen_secs,
                    ttft_ms,
                );
                Ok::<axum::body::Bytes, axum::Error>(axum::body::Bytes::new())
            }),
        );
        resp = Response::from_parts(parts, Body::from_stream(finalized));
    }
    resp
}

// Flat route table; length is the surface itself, not complexity.
#[allow(clippy::too_many_lines)]
pub fn router(state: Arc<AppState>) -> Router {
    let openai_any = Router::new()
        .route("/v1/chat/completions", post(openai::openai_proxy))
        .route(
            "/v1/chat/completions/{id}",
            post(openai::chat_completion_update),
        )
        .route("/v1/completions", post(openai::openai_proxy))
        .route("/v1/embeddings", post(openai::embeddings))
        .route("/v1/rerank", post(openai::openai_proxy))
        // sglang-native scoring + decisions endpoints (0.5.21+, verified
        // against the installed engine's http_server.py): score ranks
        // query-item pairs in one request; decisions answers typed
        // choice/score/yes-no questions via single-token label scoring.
        .route("/v1/score", post(openai::openai_proxy))
        .route("/v1/decisions", post(openai::openai_proxy))
        // Upstream-native bare rerank paths: llama-server documents POST
        // /reranking as canonical with /rerank as its alias, so clients
        // speaking the raw upstream dialect hit these without the /v1
        // prefix. Same passthrough handler as the prefixed forms.
        .route("/rerank", post(openai::openai_proxy))
        .route("/reranking", post(openai::openai_proxy))
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
        .route("/v1/conversations/{id}", get(openai::conversations_get))
        .route(
            "/v1/conversations/{id}",
            delete(openai::conversations_delete),
        )
        .route(
            "/v1/audio/transcriptions",
            post(whisper::audio_transcriptions),
        )
        .route("/v1/audio/translations", post(whisper::audio_translations))
        .route("/v1/audio/speech", post(tts::audio_speech))
        .route("/v1/audio/voices", get(tts::voices))
        .route("/v1/audio/jobs/{id}", get(whisper::audio_jobs_get))
        .route(
            "/v1/audio/jobs/{id}/cancel",
            post(whisper::audio_jobs_cancel),
        )
        // Unified durable job plane: every async lane (audio tasks,
        // image/video child jobs) writes through to the SQLite ledger;
        // these routes read/cancel across all of them and survive
        // restarts. Creation stays on the native async surfaces.
        .route("/v1/jobs", get(jobs::jobs_list))
        .route("/v1/jobs/{id}", get(jobs::jobs_get))
        .route("/v1/jobs/{id}/cancel", post(jobs::jobs_cancel))
        .route("/v1/jobs/{id}/events", get(jobs::jobs_events))
        .route("/v1/jobs/{id}/artifact", get(jobs::jobs_artifact))
        // Request lifecycle (v0.15): in-flight generation cards +
        // programmatic cancel/interrupt. Cards are per-boot memory;
        // the durable history plane is /v1/jobs above.
        .route("/v1/requests", get(requests::requests_list))
        .route("/v1/requests/{id}", get(requests::requests_get))
        .route("/v1/requests/{id}/cancel", post(requests::requests_cancel))
        .route(
            "/v1/requests/{id}/interrupt",
            post(requests::requests_interrupt),
        )
        .route("/v1/audio/capabilities", get(whisper::audio_capabilities))
        .route("/v1/images/generations", post(images::generations))
        .route("/v1/images/edits", post(images::edits))
        .route("/v1/images/variations", post(images::variations))
        .route("/v1/images/upscale", post(images::upscale))
        .route("/v1/images/jobs/{id}", get(images::jobs_get))
        .route("/v1/images/jobs/{id}/cancel", post(images::jobs_cancel))
        .route("/v1/images/capabilities", get(images::capabilities))
        .route("/v1/videos/generations", post(images::video_generations))
        // Sora verbs: POST /v1/videos is the Sora-native create (same
        // handler), GET lists the durable video rows, {id}/content
        // serves bytes once completed, DELETE cancels.
        .route(
            "/v1/videos",
            get(images::videos_list).post(images::video_generations),
        )
        .route(
            "/v1/videos/{id}",
            get(images::video_get).delete(images::video_delete),
        )
        .route("/v1/videos/{id}/content", get(images::video_content))
        // Jobs and capabilities are surface-agnostic upstream (one
        // queue, `kind` distinguishes) - same handlers on both mounts.
        .route("/v1/videos/jobs/{id}", get(images::jobs_get))
        .route("/v1/videos/jobs/{id}/cancel", post(images::jobs_cancel))
        .route("/v1/videos/capabilities", get(images::capabilities))
        .route("/audio/transcriptions", post(whisper::audio_transcriptions))
        // Non-`/v1` alias for older OpenAI clients (docs promise the
        // same alias symmetry as transcriptions).
        .route("/audio/translations", post(whisper::audio_translations))
        .route("/infill", post(openai::openai_proxy))
        // Native llama-server dialect (no /v1 prefix, no `model` field
        // in the body upstream): /completion is the raw prompt surface,
        // /embeddings the token-level batch one (pooling=none). Same
        // handler - model resolution mirrors scoped_proxy for them.
        .route("/completion", post(openai::openai_proxy))
        .route("/embeddings", post(openai::openai_proxy))
        .route("/v1/chat/completions/control", post(openai::openai_proxy))
        .route(
            "/v1/chat/completions/input_tokens",
            post(openai::openai_proxy),
        )
        .route("/v1/responses/input_tokens", post(openai::openai_proxy))
        .route("/v1/systemone", post(openai::openai_proxy))
        .route("/v1/classify", post(classify::classify))
        .route("/responses/input_tokens", post(openai::openai_proxy))
        // Realtime voice lane: WS upgrade, one utterance pipeline per
        // commit (STT -> chat -> TTS through the gateway's own lanes).
        .route("/v1/realtime", get(realtime::realtime_session))
        .route("/v1/messages/count_tokens", post(anthropic::count_tokens))
        // Anthropic Message Batches: inline-request batch dialect. The
        // worker replays each params body through our own /v1/messages.
        .route(
            "/v1/messages/batches",
            post(batch::anthropic_batches_create).get(batch::anthropic_batches_list),
        )
        .route(
            "/v1/messages/batches/{id}",
            get(batch::anthropic_batches_get).delete(batch::anthropic_batches_delete),
        )
        .route(
            "/v1/messages/batches/{id}/cancel",
            post(batch::anthropic_batches_cancel),
        )
        .route(
            "/v1/messages/batches/{id}/results",
            get(batch::anthropic_batches_results),
        )
        .route("/tokenize", post(openai::openai_proxy))
        .route("/detokenize", post(openai::openai_proxy))
        .route("/apply-template", post(openai::openai_proxy))
        // Jina-style rerank alias (upstream serves both spellings).
        .route("/v1/reranking", post(openai::openai_proxy))
        // Engine-scoped upstream surfaces (no `model` in the body): slot
        // inspection/control, props, and the disconnected-stream family.
        // Model comes from X-Blazar-Model > ?model= > single hot child.
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
        .route("/api/delete", post(ollama::delete).delete(ollama::delete))
        .route("/api/ps", get(ollama::ps))
        // Per-GPU capacity: device census + per-resident VRAM
        // attribution joined from the supervisor's live rows.
        .route("/api/capacity", get(capacity::capacity))
        .route("/api/usage", get(crate::usage::usage_api))
        .route("/api/failover", get(failover::list))
        .route("/api/failover/{chain}", get(failover::detail))
        .route("/api/failover/{chain}/pin", post(failover::pin))
        .route("/api/failover/{chain}/unpin", post(failover::unpin))
        .route("/api/mcp", get(mcp::mcp_status))
        .route("/api/fabric", get(fabric::fabric))
        .route("/api/quantiles", get(quantiles::quantiles))
        .route("/api/benchmarks", get(benchmarks::benchmarks))
        .route("/ui", get(console::ui))
        .route("/api/replicate", post(federation::replicate))
        .route("/api/route/{model}", get(federation::route))
        .route(
            "/api/engine/recommend",
            get(engine_recommend::engine_recommend),
        )
        .route("/api/explain/{model}", get(explain::explain))
        // Per-model capability certificate: bounded probes through the
        // real gateway path, stored as a doctor job + model_caps row.
        .route("/api/model-doctor", post(model_doctor::run))
        .route("/api/model-doctor/{model}", get(model_doctor::cert))
        .route("/api/pull", post(ollama::pull))
        .route("/api/chat", post(ollama::chat))
        .route("/api/embeddings", post(ollama::embeddings))
        .route("/api/embed", post(ollama::embed))
        .route("/api/rerank", post(ollama::rerank))
        .route("/api/generate", post(ollama::generate))
        .route("/api/evict", post(ollama::evict))
        .route("/api/warm", post(federation::warm))
        .route("/api/lora/sync", post(federation::lora_sync))
        .route(
            "/api/session",
            post(ollama::session).get(ollama::session_list),
        )
        .route("/api/sessions", get(sessions::list))
        // Automatic fleet membership (blazar rpc worker --join writes
        // here; spawn planning merges the registry into --rpc).
        .route("/api/fleet/join", post(fleet::join))
        .route("/api/fleet/heartbeat", post(fleet::heartbeat))
        .route("/api/fleet/leave", post(fleet::leave))
        .route("/api/fleet/workers", get(fleet::workers))
        .route("/api/fleet/invite", post(fleet::invite))
        .route("/api/config/reload", post(config_api::reload))
        .route("/api/why", get(ollama::why))
        .route("/api/watch", get(ollama::watch))
        // Machine-readable stable error-code catalog (self-describing
        // API contract; codes are additive-only once shipped).
        .route("/api/errors", get(error_codes::errors_catalog))
        .route(
            "/api/keys",
            get(keys_list).post(keys_add).delete(keys_remove),
        )
        .route("/api/keys/rotate", post(keys_rotate))
        .route("/.well-known/blazar", get(well_known));

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
        // Unknown routes answer in the standard error envelope with the
        // offending method + path and a pointer at the route census,
        // instead of axum's empty-body 404. Inside the layer stack, so
        // auth/CORS/access-log still apply to 404s.
        .fallback(unknown_route)
        // Session pins (R3): innermost layer - auth/CORS/logging/body
        // limit have already run; only header-carrying requests buffer.
        .layer(middleware::from_fn_with_state(
            state.clone(),
            sessions::pin_mw,
        ))
        // Request lifecycle (v0.15): tracked generation lanes get a card
        // stamped `x-blazar-request-id` and become cancelable by id from
        // a second connection. Inside auth (key identity is on the
        // request extensions) and inside the body limit (the model sniff
        // is bounded under it). Non-tracked paths: verbatim passthrough.
        .layer(middleware::from_fn_with_state(
            state.clone(),
            requests::lifecycle,
        ))
        // Hardening: 50 MiB request ceiling (audio uploads fit; nothing
        // legit is larger locally).
        .layer(axum::extract::DefaultBodyLimit::max(50 * 1024 * 1024))
        // Auth inside request_log (F64): rejected secrets are still
        // access-logged and traced - brute-force probing no longer
        // flies blind. Sits INSIDE CORS: browser preflights carry no
        // bearer, so the CORS layer answers them before auth runs, and
        // auth's 401s flow back out through CORS, letting permitted
        // origins read the error body instead of an opaque network
        // failure.
        .layer(middleware::from_fn_with_state(state.clone(), auth))
        // CORS: browser origins are gated to the local set by default
        // (localhost/127.0.0.1/[::1]/0.0.0.0 on any port plus desktop
        // webview schemes), so a page on some other site cannot read
        // gateway responses. `cors_origins` ADDS explicit origins
        // (additive, e.g. a hosted UI), and ["*"] restores any-origin.
        // Headers stay `Any` on purpose: the origin gate already limits
        // callers to the local user's own pages, and SDK header
        // families (x-stainless-*) evolve independently of us.
        // Sits OUTSIDE auth: preflights carry no bearer token and must
        // get a CORS answer (BLZ-5), not a bare 401 from auth.
        .layer(
            tower_http::cors::CorsLayer::new()
                .allow_origin(cors_allow_origin(&state.cfg().cors_origins))
                .allow_methods([
                    axum::http::Method::GET,
                    axum::http::Method::POST,
                    axum::http::Method::DELETE,
                ])
                .allow_headers(tower_http::cors::Any)
                .expose_headers([axum::http::header::HeaderName::from_static(
                    "x-blazar-trace-id",
                )]),
        )
        // Host guard: DNS-rebinding defense (ollama CVE-2024-28224
        // class). Armed only on loopback binds - a non-loopback bind
        // is a deliberate LAN exposure decision. Preflights (OPTIONS)
        // and health probes pass through so CORS and orchestrators
        // keep working; anything else with a non-local Host is 403.
        // Inside request_log so blocks are visible in the access log.
        .layer(middleware::from_fn_with_state(
            (host_guard::guard_scope(&state.cfg().host),),
            host_guard::host_guard_mw,
        ))
        // Status enrichment: axum's own rejections (405 method
        // mismatch, 415 content-type mismatch) leave the router as
        // bare empty-body responses - no blazar envelope, no teaching.
        // Wrapped into the catalog's shape here so clients get the
        // same actionable envelope every other error path emits;
        // headers axum already set (notably `Allow` on 405) are
        // preserved. Sits inside request_log so the access log sees
        // the enriched status.
        .layer(middleware::map_response(status_enrich_mw))
        // request_log: outermost - every request, including 401s and
        // preflights, gets a trace id and an access-log line.
        .layer(middleware::from_fn_with_state(state.clone(), request_log))
        .with_state(state)
}

/// Translate the config's origin list into tower-http's `AllowOrigin`.
/// Default: the local browser set (localhost/127.0.0.1/IPv6 loopback/0.0.0.0 on
/// any port + desktop webview schemes) - local UIs work with zero
/// config, other sites get nothing. Configured entries are ADDITIVE on
/// top of that set; the sole entry "*" restores any-origin.
fn cors_allow_origin(origins: &[String]) -> tower_http::cors::AllowOrigin {
    if origins.iter().any(|o| o == "*") {
        tower_http::cors::AllowOrigin::any()
    } else {
        let explicit: Vec<String> = origins
            .iter()
            .map(|o| o.trim().to_ascii_lowercase())
            .collect();
        tower_http::cors::AllowOrigin::predicate(move |origin, _| {
            let Ok(value) = origin.to_str() else {
                return false;
            };
            let lowered = value.trim().to_ascii_lowercase();
            explicit.iter().any(|e| e == &lowered)
                || host_guard::is_default_browser_origin(&lowered)
        })
    }
}

/// Enrich axum-native 405/415 rejections with the blazar error
/// envelope. Only touches responses whose body is empty (a handler
/// that already produced a body wins); re-attaches the `Allow` header
/// axum sets on 405 so method discovery keeps working.
async fn status_enrich_mw(resp: axum::response::Response) -> axum::response::Response {
    use axum::http::{
        StatusCode,
        header::{ALLOW, CONTENT_TYPE},
    };
    use http_body::Body as _;
    let status = resp.status();
    // Enrich only framework-generated rejections, never a handler payload:
    // - 405s from axum's method router carry an empty body (Allow says the rest);
    // - 415s from the `Json` extractor carry a plain-text body, so anything
    //   already speaking JSON is a real handler response and stays untouched.
    let framework_rejection = match status {
        StatusCode::METHOD_NOT_ALLOWED => resp.body().size_hint().exact() == Some(0),
        StatusCode::UNSUPPORTED_MEDIA_TYPE => resp
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_none_or(|ct| !ct.contains("json")),
        _ => false,
    };
    if !framework_rejection {
        return resp;
    }
    let code = match status {
        StatusCode::METHOD_NOT_ALLOWED => Some((
            error_codes::BlazarCode::MethodNotAllowed,
            "the route exists but not for this HTTP method - the Allow header lists what it accepts",
        )),
        StatusCode::UNSUPPORTED_MEDIA_TYPE => Some((
            error_codes::BlazarCode::UnsupportedMediaType,
            "this endpoint expects an application/json body",
        )),
        _ => None,
    };
    let Some((code, message)) = code else {
        return resp;
    };
    let allow = resp.headers().get(ALLOW).cloned();
    let mut out = error_codes::error_response(code, message);
    if let Some(allow) = allow {
        out.headers_mut().insert(ALLOW, allow);
    }
    out
}

/// Persist the live key set back to config.toml ([[keys]] tables).
/// Full-file round-trip (the established `create`/`config set` pattern):
/// parse -> swap keys -> validate -> atomic write (temp + rename, F67).
/// A process-wide mutex excludes the other config.toml writers in this
/// daemon (SIGHUP reload reads are lock-free but writes serialize), so
/// two racing mutations can't clobber each other with a stale read.
/// Returns an error response on failure (the live registry is already
/// updated; a failed persist is loud, never silent - the next daemon
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
            .and_then(|raw| blazar_core::Config::from_toml(&raw).map_err(|e| e.to_string()))
        {
            Ok(c) => c,
            Err(e) => return error_response(500, &format!("config.toml unreadable: {e}")),
        }
    } else {
        blazar_core::Config::default()
    };
    cfg.keys = entries.to_vec();
    if let Err(e) = cfg.validate() {
        return error_response(400, &format!("generated config invalid: {e}"));
    }
    // F67: atomic write (temp+rename) + timestamped .bak via the shared
    // core helper - a crash mid-write can never truncate config.toml and
    // an overwrite is always recoverable.
    let write = cfg
        .to_toml()
        .map_err(|e| e.to_string())
        .and_then(|t| blazar_core::persist_config(&path, &t).map_err(|e| e.to_string()));
    match write {
        Ok(()) => (StatusCode::OK, axum::Json(serde_json::json!({"ok": true}))).into_response(),
        Err(e) => error_response(500, &format!("write {}: {e}", path.display())),
    }
}

fn error_response(code: u16, msg: &str) -> Response {
    (
        StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        axum::Json(
            serde_json::json!({"error": {"message": msg, "type": "blazar_error", "code": code,
                "blazar_code": error_codes::generic_for_status(code).as_str()}}),
        ),
    )
        .into_response()
}

/// Router fallback: every unmatched method+path lands here. The message
/// names what was asked and where the full mounted-route census lives -
/// `GET /.well-known/blazar` is kept in lockstep with the router table.
async fn unknown_route(method: axum::http::Method, uri: axum::http::Uri) -> Response {
    error_response(
        404,
        &format!(
            "unknown route: {} {} - GET /.well-known/blazar lists every mounted route",
            method,
            uri.path()
        ),
    )
}

/// Redacted secret for listings: enough to recognize, not enough to use.
fn redact(secret: &str) -> String {
    let prefix: String = secret.chars().take(8).collect();
    format!("{prefix}...")
}

/// Management guard: when keys exist, only UNSCOPED keys (empty model
/// list = admin semantics) may manage the key set. When no keys exist,
/// the gateway is authless and management is refused - bootstrap the
/// first key via config.toml `[[keys]]` or `BLAZAR_KEYS` (never let an
/// open gateway mint its own credentials).
#[allow(clippy::result_large_err)] // Response is the handlers' currency here
pub(crate) fn management_authorized(
    state: &AppState,
    key: Option<&keys::KeyCtx>,
) -> Result<(), Response> {
    if state.keys.is_empty() {
        return Err(error_response(
            403,
            "no keys configured: add the first [[keys]] entry in config.toml (or BLAZAR_KEYS env) and restart",
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
    // F68: remove/rotate address keys via `?name=` - restrict the
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
    // F65: the teaching 400s above are answers, not decoration - an
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
        // The secret appears exactly once, here - listings only ever show
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
            "refusing to remove the last key - that would leave the gateway authless; add a replacement first",
        );
    }
    if !state.keys.remove(&name) {
        return error_response(404, &format!("no key named {name:?}"));
    }
    let entries = state.keys.entries();
    let persist = persist_keys(&state, &entries);
    if persist.status() != StatusCode::OK {
        // Nothing to roll back (remove already ran) - the error response
        // tells the operator the file write failed.
        return persist;
    }
    (
        StatusCode::OK,
        axum::Json(serde_json::json!({"removed": name})),
    )
        .into_response()
}

/// POST /api/keys/rotate?name=N - mint a fresh secret for a key,
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
        // F66: mirror keys_add's rollback - the file is the source of
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

/// GET /.well-known/blazar - capability discovery: routes, limits,
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
        "name": "blazar",
        "version": env!("CARGO_PKG_VERSION"),
        "engine": engine,
        // F52: the messages lane TRANSLATES and refuses passthrough -
        // the label must not advertise passthrough semantics.
        "apis": ["openai", "ollama", "anthropic"],
        // F69: full route census (verified against the router table).
        // Audit MM15: extended to the media generation lanes and the
        // engine-scoped surfaces - the census is the machine-readable
        // discovery contract and must not lag the router.
        "endpoints": {
            "openai": ["/v1/chat/completions", "/v1/chat/completions/{id}",
                       "/v1/chat/completions/control",
                       "/v1/chat/completions/input_tokens",
                       "/v1/completions", "/v1/embeddings", "/v1/rerank", "/v1/reranking",
                       "/rerank", "/reranking",
                       "/v1/responses", "/v1/responses/{id}", "/v1/conversations/{id}",
                       "/v1/responses/input_tokens",
                       "/v1/messages", "/v1/messages/count_tokens", "/v1/models",
                       "/v1/systemone",
                       "/v1/adapters", "/v1/batches", "/v1/batches/{id}",
                       "/v1/batches/{id}/cancel", "/v1/files", "/v1/files/{id}",
                       "/v1/files/{id}/content", "/v1/streams/lookup", "/v1/stream",
                        "/props", "/infill", "/completion", "/embeddings",
                        "/tokenize", "/detokenize",
                        "/apply-template", "/slots", "/slots/{id}", "/responses",
                       "/responses/input_tokens",
                        "/v1/images/generations", "/v1/images/edits",
                        "/v1/images/variations", "/v1/images/upscale",
                        "/v1/images/jobs/{id}", "/v1/images/jobs/{id}/cancel",
                        "/v1/images/capabilities",
                        "/v1/videos/generations", "/v1/videos",
                        "/v1/videos/{id}", "/v1/videos/{id}/content",
                        "/v1/videos/jobs/{id}",
                        "/v1/videos/jobs/{id}/cancel", "/v1/videos/capabilities",
                         "/v1/audio/transcriptions", "/v1/audio/translations",
                         "/v1/audio/speech", "/v1/audio/voices",
                         "/v1/audio/jobs/{id}",
                         "/v1/audio/jobs/{id}/cancel", "/v1/audio/capabilities",
                        "/v1/jobs", "/v1/jobs/{id}", "/v1/jobs/{id}/cancel",
                        "/v1/jobs/{id}/events", "/v1/jobs/{id}/artifact",
                        "/v1/requests", "/v1/requests/{id}",
                        "/v1/requests/{id}/cancel", "/v1/requests/{id}/interrupt",
                        "/audio/transcriptions", "/audio/translations"],
            "ollama": ["/api/chat", "/api/generate", "/api/tags", "/api/ps", "/api/capacity", "/api/explain/{model}", "/api/show",
                       "/api/usage",
                       "/api/replicate", "/api/route/{model}",
                       "/api/model-doctor", "/api/model-doctor/{model}",
                       "/api/embeddings", "/api/embed", "/api/rerank", "/api/pull",
                       "/api/delete", "/api/events", "/api/version"],
            "blazar": ["/api/evict", "/api/warm", "/api/lora/sync", "/api/session", "/api/sessions", "/api/why",
                        "/api/watch", "/api/keys", "/api/keys/rotate",
                        "/.well-known/blazar", "/metrics", "/healthz", "/health"],
        },
        "headers": ["x-blazar-num-ctx", "x-blazar-deadline-ms", "x-blazar-priority",
                    "x-blazar-enforce", "x-blazar-trace-id", "x-blazar-status",
                    "x-blazar-warnings", "x-blazar-session",
                    "x-blazar-cache", "x-blazar-cache-ttl", "x-blazar-cache-threshold"],
        "features": {
            "keys": !state.keys.is_empty(),
            "tls": !state.cfg().tls_cert.is_empty(),
            "otlp": state.otlp.enabled(),
            "remotes": state.cfg().remotes.iter().map(|r| r.name.clone()).collect::<Vec<_>>(),
            "singleflight": state.cfg().singleflight,
            "session_bank": state.cfg().session_bank,
            "prompt_preflight": state.cfg().prompt_preflight,
            "session_pins": state.cfg().session_keep_secs > 0,
            "semantic_cache": state.cfg().semantic_cache.enabled
                && state.cfg().semantic_cache.model.is_some(),
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

/// EWMA step shared by the merged-global and per-model hint windows
/// (0.7 history / 0.3 new - the A16/G3 smoothing constant). Also the
/// remote time-to-head hint (v0.16 federation) - one constant, one bias.
pub(crate) fn hint_ewma(prev: Option<f64>, rate: f64) -> f64 {
    match prev {
        Some(e) => e * 0.7 + rate * 0.3,
        None => rate,
    }
}

/// Serve the gateway until `shutdown` resolves (SIGTERM/SIGINT), then
/// drain: stop accepting, stop children, exit clean. TLS when
/// `tls_cert`/`tls_key` are configured, plain HTTP otherwise.
#[allow(clippy::too_many_lines)]
/// Serve returned because the configured listen address is already
/// taken (another server - typically ollama - owns the port). The CLI
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

/// `blazar serve` exit code for an unrecoverable bind conflict.
pub const EXIT_BIND_CONFLICT: i32 = 3;

// One wiring function: listener, TLS, flushers, teardown - all the
// serve-lifetime setup in a single readable sequence.
#[allow(clippy::too_many_lines)]
pub async fn serve(
    state: Arc<AppState>,
    host: &str,
    port: u16,
    shutdown: std::pin::Pin<Box<dyn futures::Future<Output = ()> + Send>>,
) -> anyhow::Result<()> {
    let app = router(state.clone());
    // Durable jobs boot hygiene, detached: sweep latency is not boot
    // latency. Abandons in-flight rows from a previous gateway process
    // (grace-window protected) and prunes terminal rows older than 7
    // days - the ledger must not grow unbounded.
    {
        let state = Arc::clone(&state);
        tokio::spawn(async move { state.jobs.boot_sweep(&state) });
    }
    let addr = format!("{host}:{port}");
    // Prompt-cache hit-rate + spec-accept poller (A16/G3): every 60 s,
    // sum the children's Prometheus counters, compute the window delta
    // rate and EWMA it into the supervisor's CacheHints - the adaptive
    // --cache-ram clamp's input and the spec gauge's source. Bounded
    // (2 s) fetches; failures just skip a window.
    // Wake admission waiters when a child swap changes slot capacity:
    // a completed adaptive reshape (SlotsReshaped) or an engine
    // update/rollback replaced the child while requests were parked on
    // the admission queue - only stream completions signal otherwise,
    // so these waiters would sit until the 2-min park timeout.
    let reshape_wake_task = {
        let state = Arc::clone(&state);
        tokio::spawn(async move {
            let mut rx = state.bus.subscribe();
            loop {
                match rx.recv().await {
                    Ok(
                        blazar_runtime::events::BlazarEvent::SlotsReshaped { .. }
                        | blazar_runtime::events::BlazarEvent::EngineUpdated { .. }
                        | blazar_runtime::events::BlazarEvent::EngineRolledBack { .. },
                    ) => state.queue.signal_free(),
                    Ok(_) => {}
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(skipped = n, "reshape wake subscriber lagged");
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
        })
    };
    let hint_task = {
        let state = Arc::clone(&state);
        tokio::spawn(async move {
            let mut prev: (u64, u64) = (0, 0); // (Σ cache_n, Σ prompt_n)
            let mut prev_spec: (u64, u64) = (0, 0); // (Σ accepted, Σ drafted)
            let mut ewma: Option<f64> = None;
            let mut ewma_spec: Option<f64> = None;
            // C1 per-model split: same windowed rates keyed by the
            // engine-reported model name, so the adaptive --cache-ram
            // clamp steers per model instead of on the all-children blur.
            // (prev counters, (cache ewma, accept ewma)) per model.
            let mut per_prev: std::collections::HashMap<String, (u64, u64, u64, u64)> =
                std::collections::HashMap::new();
            let mut per_ewma: std::collections::HashMap<String, (Option<f64>, Option<f64>)> =
                std::collections::HashMap::new();
            let mut interval = tokio::time::interval(std::time::Duration::from_mins(1));
            interval.tick().await; // immediate first tick: skip (empty)
            loop {
                interval.tick().await;
                let mut totals = (0u64, 0u64);
                let mut spec_totals = (0u64, 0u64);
                let mut by_model: std::collections::HashMap<String, (u64, u64, u64, u64)> =
                    std::collections::HashMap::new();
                for e in state.sup.live_http_endpoints() {
                    // Child Prometheus text: aggregate counters (the /slots
                    // per-slot stats are short-lived and unreliable - the
                    // metrics counters are the durable truth). Both
                    // transports serve /metrics; child_client picks the
                    // socket-pinned client for unix children.
                    let url = format!("{}/metrics", crate::proxy::child_base(&e.endpoint));
                    let Ok(resp) = crate::proxy::child_auth(
                        crate::state::child_client(&state, &e.endpoint)
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
                    // Replicas of one model share the entry - same
                    // weights, same KV economics.
                    let slot = by_model.entry(e.name.clone()).or_insert((0, 0, 0, 0));
                    if let (Some(c), Some(p)) = (
                        counter(&text, "llamacpp:prompt_tokens_cached_total"),
                        counter(&text, "llamacpp:prompt_tokens_total"),
                    ) {
                        totals.0 += c;
                        totals.1 += p;
                        slot.0 += c;
                        slot.1 += p;
                    }
                    // G3: spec-decoding acceptance (only present when a
                    // spec pair is live on that child).
                    if let (Some(a), Some(d)) = (
                        counter(&text, "llamacpp:spec_decode_num_accepted_tokens_total"),
                        counter(&text, "llamacpp:spec_decode_num_draft_tokens_total"),
                    ) {
                        spec_totals.0 += a;
                        spec_totals.1 += d;
                        slot.2 += a;
                        slot.3 += d;
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
                    ewma = Some(hint_ewma(ewma, rate));
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
                    ewma_spec = Some(hint_ewma(ewma_spec, rate));
                    if let Some(e) = ewma_spec {
                        state.sup.spec_accept.set(e);
                    }
                }
                // Per-model windows: identical math on per-model sums.
                // A model absent this window keeps its last measured
                // value (no zeroing on idle); counter resets (child
                // swap) yield 0-deltas and skip, same as the global.
                for (model, cur) in &by_model {
                    let mp = per_prev.entry(model.clone()).or_insert((0, 0, 0, 0));
                    let (d_cache, d_prompt) =
                        (cur.0.saturating_sub(mp.0), cur.1.saturating_sub(mp.1));
                    let (d_acc, d_draft) = (cur.2.saturating_sub(mp.2), cur.3.saturating_sub(mp.3));
                    *mp = *cur;
                    let (mut c_ewma, mut a_ewma) =
                        per_ewma.get(model).copied().unwrap_or((None, None));
                    let mut cache_rate = None;
                    if d_cache + d_prompt > 0 {
                        #[allow(clippy::cast_precision_loss)]
                        let rate = d_cache as f64 / (d_cache + d_prompt) as f64;
                        c_ewma = Some(hint_ewma(c_ewma, rate));
                        cache_rate = c_ewma;
                    }
                    let mut accept_rate = None;
                    if d_draft > 0 {
                        #[allow(clippy::cast_precision_loss)]
                        let rate = d_acc.min(d_draft) as f64 / d_draft as f64;
                        a_ewma = Some(hint_ewma(a_ewma, rate));
                        accept_rate = a_ewma;
                    }
                    per_ewma.insert(model.clone(), (c_ewma, a_ewma));
                    state.sup.set_model_hints(model, cache_rate, accept_rate);
                    // B2: the same window feeds the per-model spec
                    // governor (inert unless spec_auto_manage is on).
                    state.sup.spec_governor_window(model, accept_rate);
                    // D1: and the prefill-shaping governor (inert
                    // unless ubatch_auto is on).
                    state.sup.ubatch_governor_window(model);
                }
            }
        })
    };
    // F4 usage rollup flusher: the request tap records into the meter's
    // in-memory buckets (zero store I/O on the hot path); this task
    // drains them into `usage_daily` every minute - a crash loses at
    // most one window, and `/api/usage` + `blazar usage` stay current.
    let usage_flush_task = {
        let meter = std::sync::Arc::clone(&state.usage);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_mins(1));
            interval.tick().await; // immediate first tick: skip (empty)
            loop {
                interval.tick().await;
                if let Err(e) = meter.flush() {
                    tracing::warn!("usage rollup flush failed: {e}");
                }
            }
        })
    };
    // OTLP flusher (no-op task when the endpoint is unset).
    let otlp_task = {
        let otlp = Arc::clone(&state.otlp);
        // F81: bounded POSTs - a black-hole collector used to wedge the
        // flusher forever (the 5s select only bounds the first-span wait).
        let http = http_pool::tuned(reqwest::Client::builder())
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap_or_default();
        tokio::spawn(async move { otlp.run(http).await })
    };
    if state.cfg().tls_cert.is_empty() {
        let listener = tokio::net::TcpListener::bind(&addr).await.map_err(|e| {
            // Typed so the CLI can exit 3 and the systemd unit's
            // RestartPreventExitStatus=3 stops the Restart=always loop.
            anyhow::Error::new(BindConflict(e)).context(format!(
                "bind {addr} - another server (ollama?) on this port? stop it or set BLAZAR_PORT"
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
        tracing::info!("blazar listening on http://{addr} (OpenAI + ollama APIs)");
        tracing::info!("powered by upstream llama.cpp, mistral.rs and SGLang - unmodified engines");
        // F1 startup preload: listed models spawn now, one at a time,
        // while the listener already answers - healthz stays fast.
        {
            let sup = Arc::clone(&state.sup);
            tokio::spawn(async move { sup.preload_listed().await });
        }
        // Wave L warm-on-pull: one bus subscription turns every future
        // pull completion into an opt-in background spawn (gated on AC
        // power; the knob itself is off by default).
        {
            let sup = Arc::clone(&state.sup);
            tokio::spawn(async move { sup.listen_warm_on_pull().await });
        }
        // Hot-reload watcher: one reload attempt per config-file change
        // (fingerprint debounce inside reload_config_now); a missing file
        // is never reloaded - Config::load would rewrite it with defaults.
        {
            let sup = Arc::clone(&state.sup);
            let keys = Arc::clone(&state.keys);
            tokio::spawn(async move {
                let mut last = sup.config_file_stamp();
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    let stamp = sup.config_file_stamp();
                    if stamp.is_some() && stamp != last {
                        last = stamp;
                        match sup.reload_config_now() {
                            Ok(outcome) => {
                                if outcome.preload_changed {
                                    let s2 = Arc::clone(&sup);
                                    tokio::spawn(async move { s2.preload_listed().await });
                                }
                                if outcome.keys_changed {
                                    keys.replace_all(sup.cfg().keys.clone());
                                    tracing::info!(
                                        "api keys hot-reloaded - {} live key(s)",
                                        keys.entries().len()
                                    );
                                }
                            }
                            Err(reason) => {
                                tracing::warn!("{reason}");
                            }
                        }
                    } else {
                        last = stamp;
                    }
                }
            });
        }
        // LAN autodiscovery: opt-in UDP responder naming this gateway
        // to `rpc worker --discover` seekers. Spawned unconditionally -
        // the lifecycle task itself binds/unbinds as discovery_enabled
        // (and discovery_port) flip through config hot-reloads, so the
        // flag is fully hot in both directions without a restart.
        {
            let sup = Arc::clone(&state.sup);
            let port = state
                .http_addr
                .get()
                .map_or_else(|| sup.cfg().port, |&(_, p)| p);
            tokio::spawn(blazar_runtime::discovery::run_discovery_responder(
                sup, port,
            ));
        }
        // TCP_NODELAY on every accepted socket: NDJSON/SSE streams are strings
        // of small writes, and with Nagle the first content packet waits for the
        // client's delayed ACK of the headers/role packet - a flat
        // +40-60ms on first-token latency for every streaming response
        // (live-measured: child's first content SSE at ~77ms reached the
        // client at ~135ms; Go net/http, ollama's stack, sets nodelay too).
        let listener = listener.tap_io(|stream| {
            let _ = stream.set_nodelay(true);
        });
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(|e| anyhow::anyhow!("server: {e}"))?;
    } else {
        // Feature unification across the dep graph can leave rustls with
        // both providers enabled; pick ours deterministically (idempotent
        // if already installed).
        let _ = rustls::crypto::ring::default_provider().install_default();
        let server_config =
            rustls_server_config(&state.cfg()).map_err(|e| anyhow::anyhow!("TLS config: {e}"))?;
        let rustls_config =
            axum_server::tls_rustls::RustlsConfig::from_config(std::sync::Arc::new(server_config));
        let bind_addr: std::net::SocketAddr = addr
            .parse()
            .map_err(|e| anyhow::anyhow!("bind address {addr}: {e}"))?;
        // F69: record the bound address on the TLS branch too - the
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
        // F1 startup preload on the TLS branch too - same contract as
        // the plain-HTTP branch above.
        {
            let sup = Arc::clone(&state.sup);
            tokio::spawn(async move { sup.preload_listed().await });
        }
        // Wave L warm-on-pull on the TLS branch too.
        {
            let sup = Arc::clone(&state.sup);
            tokio::spawn(async move { sup.listen_warm_on_pull().await });
        }
        // Hot-reload watcher on the TLS branch too - same contract as
        // the plain-HTTP branch above.
        {
            let sup = Arc::clone(&state.sup);
            let keys = Arc::clone(&state.keys);
            tokio::spawn(async move {
                let mut last = sup.config_file_stamp();
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    let stamp = sup.config_file_stamp();
                    if stamp.is_some() && stamp != last {
                        last = stamp;
                        match sup.reload_config_now() {
                            Ok(outcome) => {
                                if outcome.preload_changed {
                                    let s2 = Arc::clone(&sup);
                                    tokio::spawn(async move { s2.preload_listed().await });
                                }
                                if outcome.keys_changed {
                                    keys.replace_all(sup.cfg().keys.clone());
                                    tracing::info!(
                                        "api keys hot-reloaded - {} live key(s)",
                                        keys.entries().len()
                                    );
                                }
                            }
                            Err(reason) => {
                                tracing::warn!("{reason}");
                            }
                        }
                    } else {
                        last = stamp;
                    }
                }
            });
        }
        // LAN autodiscovery on the TLS branch too - unconditional
        // lifecycle task, same contract as the plain-HTTP branch.
        {
            let sup = Arc::clone(&state.sup);
            let port = state
                .http_addr
                .get()
                .map_or_else(|| sup.cfg().port, |&(_, p)| p);
            tokio::spawn(blazar_runtime::discovery::run_discovery_responder(
                sup, port,
            ));
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
        // axum-server 0.8 made from_tcp_rustls fallible (its internal
        // acceptor wiring can reject the listener state up front).
        axum_server::from_tcp_rustls(std_listener, rustls_config)
            .map_err(|e| anyhow::anyhow!("tls acceptor init: {e}"))?
            .handle(handle)
            .serve(app.into_make_service_with_connect_info::<SocketAddr>())
            .await
            .map_err(|e| anyhow::anyhow!("tls server: {e}"))?;
    }
    otlp_task.abort();
    hint_task.abort();
    reshape_wake_task.abort();
    usage_flush_task.abort();
    // Final usage drain after the flusher stops: everything recorded
    // since the last tick lands in the store before exit.
    if let Err(e) = state.usage.flush() {
        tracing::warn!("usage rollup final flush failed: {e}");
    }
    // H8: the lazy whisper-server child (if any request spawned one).
    state.whisper.shutdown().await;
    // MCP stdio children die before the engines (fast, independent).
    state.mcp.shutdown_all().await;
    state.sup.shutdown_all().await?;
    Ok(())
}

/// Build the rustls server config from the configured PEM files.
fn rustls_server_config(
    config: &blazar_core::Config,
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
        use blazar_runtime::SupervisionError;
        let resp = supervision_error(&SupervisionError::UnsupportedModel(
            "safetensors model /m.d is not servable by the llama.cpp engine \
             - run: blazar engine install --kind sglang"
                .to_string(),
        ));
        assert_eq!(resp.status(), 400);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("read body");
        let v: serde_json::Value = serde_json::from_slice(&body).expect("json body");
        assert!(
            v["error"]["message"]
                .as_str()
                .expect("message")
                .contains("engine install --kind sglang"),
            "teaching remedy must survive the mapping"
        );
    }

    #[tokio::test]
    #[allow(non_snake_case)]
    async fn unit__supervision_error__model_too_large_maps_to_503_teaching() {
        use crate::proxy::supervision_error;
        use blazar_runtime::SupervisionError;
        let resp = supervision_error(&SupervisionError::ModelTooLarge(
            "qwen: admission floor 17.2 GiB exceeds every GPU budget (largest card 7.6 GiB) \
             even with the box empty - waiting for a slot can never change this. Use a smaller \
             quant (`blazar fit`), lower the ctx/slots, or serve on a larger card"
                .to_string(),
        ));
        assert_eq!(resp.status(), 503);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("read body");
        let v: serde_json::Value = serde_json::from_slice(&body).expect("json body");
        let msg = v["error"]["message"].as_str().expect("message");
        assert!(
            msg.contains("can never change this") && msg.contains("blazar fit"),
            "physical-refusal teaching must survive the mapping: {msg}"
        );
    }

    #[tokio::test]
    #[allow(non_snake_case)]
    async fn unit__supervision_error__insufficient_memory_maps_to_507_teaching() {
        use crate::proxy::supervision_error;
        use blazar_runtime::SupervisionError;
        // Live shape of the 18 GiB bf16 refusal (MemAvailable < spawn
        // floor): honest and teaching, but it surfaced as a bare 500 -
        // 507 says "resource exhausted", the levers stay in the body.
        let resp = supervision_error(&SupervisionError::InsufficientMemory(
            "insufficient memory to load \"qwen3.5-9b-bf16\": MemAvailable 1612 MiB < \
             floor 9437 MiB (model 18432 MiB). Stop co-resident engines (blazar ps / \
             ollama stop) or free RAM; disable this guard with spawn_mem_guard = false"
                .to_string(),
        ));
        assert_eq!(resp.status(), 507);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("read body");
        let v: serde_json::Value = serde_json::from_slice(&body).expect("json body");
        let msg = v["error"]["message"].as_str().expect("message");
        assert!(
            msg.contains("MemAvailable") && msg.contains("spawn_mem_guard"),
            "memory-floor teaching must survive the mapping: {msg}"
        );
    }

    #[tokio::test]
    #[allow(non_snake_case)]
    async fn unit__supervision_error__rpc_endpoints_dead_maps_to_503_teaching() {
        use crate::proxy::supervision_error;
        use blazar_runtime::SupervisionError;
        // A dead rpc fleet member is infrastructure-down, not an internal
        // bug: the refusal fired before any child existed, so the honest
        // status is 503 with a precise code clients can branch on.
        let resp = supervision_error(&SupervisionError::RpcEndpointsUnreachable(
            "--rpc endpoint(s) unreachable: 127.0.0.1:51099 (unreachable) - llama-server \
             aborts at startup when an RPC backend is down; start the RPC worker(s) or \
             fix `rpc_servers` in the config"
                .to_string(),
        ));
        assert_eq!(resp.status(), 503);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("read body");
        let v: serde_json::Value = serde_json::from_slice(&body).expect("json body");
        assert_eq!(
            v["error"]["blazar_code"].as_str().expect("code"),
            "RPC_ENDPOINT_UNREACHABLE"
        );
        let msg = v["error"]["message"].as_str().expect("message");
        assert!(
            msg.contains("unreachable") && msg.contains("rpc_servers"),
            "fleet teaching must survive the mapping: {msg}"
        );
    }

    #[tokio::test]
    #[allow(non_snake_case)]
    async fn unit__error_body_excerpt__fixed_size_400_captured_and_body_preserved() {
        use super::error_body_excerpt;
        let mut r = axum::response::Response::builder()
            .status(400)
            .body(axum::body::Body::from(
                r#"{"error":{"message":"profile: num_ctx 3072 cannot fit","type":"blazar_error"}}"#,
            ))
            .unwrap();
        let excerpt = error_body_excerpt(&mut r).await;
        let excerpt = excerpt.expect("small fixed 400 must be captured");
        assert!(
            excerpt.contains("num_ctx 3072 cannot fit"),
            "teaching message in excerpt: {excerpt}"
        );
        // The client-facing body must survive the peek byte-for-byte.
        let bytes = axum::body::to_bytes(r.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(bytes.windows(6).any(|w| w == b"blazar"));
    }

    #[tokio::test]
    #[allow(non_snake_case)]
    async fn unit__error_body_excerpt__truncates_long_teaching_to_bounded_excerpt() {
        use super::error_body_excerpt;
        let long = format!(r#"{{"error":{{"message":"{}"}}}}"#, "x".repeat(5_000));
        let mut r = axum::response::Response::builder()
            .status(422)
            .body(axum::body::Body::from(long.clone()))
            .unwrap();
        let excerpt = error_body_excerpt(&mut r).await.expect("captured");
        assert!(excerpt.chars().count() <= 300, "bounded excerpt");
        // Over-cap fixed-size bodies (exact hint > 8 KiB) pass through
        // unlogged rather than being buffered-then-dropped.
        let mut huge = br#"{"error":{"message":"#.to_vec();
        huge.extend(std::iter::repeat_n(b'x', 8 * 1024 + 512));
        let mut r = axum::response::Response::builder()
            .status(507)
            .body(axum::body::Body::from(huge.clone()))
            .unwrap();
        assert!(
            error_body_excerpt(&mut r).await.is_none(),
            "over-cap body must not be buffered"
        );
        let bytes = axum::body::to_bytes(r.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(bytes.len(), huge.len(), "body intact");
    }

    #[tokio::test]
    #[allow(non_snake_case)]
    async fn unit__error_body_excerpt__success_and_unknown_size_stream_untouched() {
        use super::error_body_excerpt;
        // 2xx never buffers.
        let mut ok = axum::response::Response::builder()
            .status(200)
            .body(axum::body::Body::from(r#"{"choices":[]}"#))
            .unwrap();
        assert!(error_body_excerpt(&mut ok).await.is_none());
        // Unknown-size streamed 400 (proxied child error): no exact or
        // upper hint -> passthrough, body intact.
        let stream = futures::stream::iter(
            [Ok::<_, std::convert::Infallible>(
                axum::body::Bytes::from_static(br#"{"error":{"message":"child oom"}}"#),
            )]
            .into_iter(),
        );
        let mut streamed = axum::response::Response::builder()
            .status(500)
            .body(axum::body::Body::from_stream(stream))
            .unwrap();
        assert!(
            error_body_excerpt(&mut streamed).await.is_none(),
            "unknown-size stream must not be buffered"
        );
        let bytes = axum::body::to_bytes(streamed.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(bytes.windows(9).any(|w| w == b"child oom"));
    }
}

#[cfg(test)]
#[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
mod bind_conflict_tests {
    use super::BindConflict;

    #[test]
    fn unit__bind_conflict__downcast_survives_context_chain() {
        // The CLI exit-3 path relies on anyhow::downcast_ref finding the
        // typed cause through a .context() wrapper - pin that mechanic.
        let e = anyhow::Error::new(BindConflict(std::io::Error::other(
            "Address already in use",
        )))
        .context("bind 127.0.0.1:11434 - another server?");
        assert!(e.downcast_ref::<BindConflict>().is_some());
        assert_eq!(super::EXIT_BIND_CONFLICT, 3);
    }
}
