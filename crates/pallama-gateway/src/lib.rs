//! Pallama gateway: `OpenAI` + ollama-compat HTTP surface over the
//! supervisor. Router assembly + bearer auth.

pub mod histogram;
pub mod ollama;
pub mod openai;
pub mod proxy;
pub mod queue;
pub mod sentinel;
pub mod state;
pub mod translate;

use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;

use state::AppState;

/// Request-scoped trace id, inserted by `request_log` (the outermost
/// layer) so inner handlers can correlate sentinel records with the
/// response header clients see.
#[derive(Clone)]
pub struct TraceId(pub String);

/// Bearer auth: active iff `config.api_keys` is non-empty (children stay
/// loopback and unauthenticated).
async fn auth(State(state): State<Arc<AppState>>, req: Request<Body>, next: Next) -> Response {
    if !state.config.api_keys.is_empty() {
        let path = req.uri().path();
        let open = path == "/healthz";
        if !open {
            let presented = req
                .headers()
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer ").map(str::to_string));
            let ok = presented.is_some_and(|p| state.config.api_keys.contains(&p));
            if !ok {
                return (
                    StatusCode::UNAUTHORIZED,
                    axum::Json(serde_json::json!({"error": {"message": "missing or invalid bearer token", "type": "pallama_error", "code": 401}})),
                )
                    .into_response();
            }
        }
    }
    next.run(req).await
}

/// Per-request structured log + trace id (debuggability complaint):
/// method, path, status, duration, priority, trace id — echoed back as
/// `x-pallama-trace-id` so clients can correlate.
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
    let started = std::time::Instant::now();
    let mut resp = next.run(req).await;
    let ms = started.elapsed().as_millis();
    let status = resp.status().as_u16();
    if let Ok(v) = axum::http::HeaderValue::from_str(&trace) {
        resp.headers_mut().insert("x-pallama-trace-id", v);
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

pub fn router(state: Arc<AppState>) -> Router {
    let openai_any = Router::new()
        .route(
            "/v1/chat/completions",
            post(openai::openai_proxy),
        )
        .route("/v1/completions", post(openai::openai_proxy))
        .route("/v1/embeddings", post(openai::openai_proxy))
        .route("/v1/rerank", post(openai::openai_proxy))
        .route("/v1/messages", post(openai::openai_proxy))
        // Full upstream API surface (routes verified in upstream server.cpp):
        // Responses API (current-gen OpenAI clients), audio transcriptions
        // (multipart; MTMD audio-in models), FIM infill, control-vector
        // steering, token counting, and the tokenize/apply-template dev
        // tools. All byte-stream proxied with model-affinity routing.
        .route("/v1/responses", post(openai::openai_proxy))
        .route("/responses", post(openai::openai_proxy))
        .route("/v1/audio/transcriptions", post(openai::openai_proxy))
        .route("/audio/transcriptions", post(openai::openai_proxy))
        .route("/infill", post(openai::openai_proxy))
        .route("/v1/chat/completions/control", post(openai::openai_proxy))
        .route("/v1/chat/completions/input_tokens", post(openai::openai_proxy))
        .route("/v1/responses/input_tokens", post(openai::openai_proxy))
        .route("/responses/input_tokens", post(openai::openai_proxy))
        .route("/v1/messages/count_tokens", post(openai::openai_proxy))
        .route("/tokenize", post(openai::openai_proxy))
        .route("/detokenize", post(openai::openai_proxy))
        .route("/apply-template", post(openai::openai_proxy));

    let api = Router::new()
        .route("/api/version", get(ollama::version))
        .route("/api/tags", get(ollama::tags))
        .route("/api/show", post(ollama::show))
        .route("/api/delete", post(ollama::delete))
        .route("/api/ps", get(ollama::ps))
        .route("/api/pull", post(ollama::pull))
        .route("/api/chat", post(ollama::chat))
        .route("/api/embeddings", post(ollama::embeddings))
        .route("/api/generate", post(ollama::generate))
        .route("/api/evict", post(ollama::evict))
        .route("/api/session", post(ollama::session).get(ollama::session_list))
        .route("/api/why", get(ollama::why));

    Router::new()
        .route("/healthz", get(openai::healthz))
        .route("/metrics", get(ollama::metrics))
        .route("/api/events", get(ollama::events))
        .route("/v1/models", get(openai::models))
        .route("/v1/adapters", get(openai::lora_adapters).post(openai::lora_adapters))
        .merge(openai_any)
        .merge(api)
        .layer(middleware::from_fn_with_state(state.clone(), request_log))
        .layer(middleware::from_fn_with_state(state.clone(), auth))
        .with_state(state)
}

/// Serve the gateway until `shutdown` resolves (SIGTERM/SIGINT), then
/// drain: stop accepting, stop children, exit clean.
pub async fn serve(
    state: Arc<AppState>,
    host: &str,
    port: u16,
    shutdown: std::pin::Pin<Box<dyn futures::Future<Output = ()> + Send>>,
) -> anyhow::Result<()> {
    let app = router(state.clone());
    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| anyhow::anyhow!("bind {addr}: {e} — another server (ollama?) on this port? stop it or set PALLAMA_PORT"))?;
    tracing::info!("pallama listening on http://{addr} (OpenAI + ollama APIs)");
    tracing::info!("powered by llama.cpp / ggml / ggerganov — https://github.com/ggml-org/llama.cpp");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(|e| anyhow::anyhow!("server: {e}"))?;
    state.sup.shutdown_all().await?;
    Ok(())
}

