//! stub-llama-server: test-double for the real llama-server child.
//!
//! Mirrors the observable surface Pallama depends on (verified against
//! upstream master `references/llama.cpp-master`):
//!   - `--list-devices` prints PLAIN TEXT (not JSON):
//!       "Available devices:\n  NAME: DESC (TOTAL MiB, FREE MiB free)\n"
//!   - `--version` prints "version: NNNN (commit)" style lines
//!   - `/health` returns {"status":"ok"}
//!   - `/v1/models`, `/v1/chat/completions` (stream + non-stream), `/metrics`,
//!     `/slots` (slot activity for cancellation tests)
//!
//! Env knobs (so one binary covers many test shapes):
//!   STUB_ARGV_FILE   write the full argv as a JSON array before anything else
//!   STUB_DEVICES     device lines to print for --list-devices (default: 1 GPU)
//!   STUB_BUILD       build number for --version (default 9999)
//!   STUB_DELAY_MS    extra latency before answering each request (default 0)

use axum::extract::State;
use axum::response::IntoResponse;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if let Ok(path) = std::env::var("STUB_ARGV_FILE") {
        let json = serde_json::to_string_pretty(&args).expect("argv is serializable");
        if let Some(dir) = std::path::Path::new(&path).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        std::fs::write(&path, json).expect("write STUB_ARGV_FILE");
    }

    if args.iter().any(|a| a == "--version" || a == "-v") {
        let build = std::env::var("STUB_BUILD").unwrap_or_else(|_| "9999".into());
        println!("version: {build} (deadbeef0000deadbeef0000deadbeef0000dead)");
        println!("built with cc (Linux) for Linux x64");
        return;
    }

    if args.iter().any(|a| a == "--list-devices") {
        let devices = std::env::var("STUB_DEVICES")
            .unwrap_or_else(|_| "STUB0: stub-gpu (8192 MiB, 8192 MiB free)".into());
        println!("Available devices:");
        if devices.trim().is_empty() {
            println!("  (none)");
        } else {
            for line in devices.split(';') {
                println!("  {line}");
            }
        }
        return;
    }

    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let host = flag("--host").unwrap_or_else(|| "127.0.0.1".into());
    let port: u16 = flag("--port").unwrap_or_else(|| "8080".into()).parse().expect("--port must be numeric");
    let alias = flag("--alias").unwrap_or_else(|| "stub-model".into());

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    rt.block_on(serve(host, port, alias));
}

async fn serve(host: String, port: u16, alias: String) {
    use axum::routing::{get, post};
    let alias_for_routes = alias.clone();
    let app = axum::Router::new()
        .route("/health", get(|| async { axum::Json(serde_json::json!({"status": "ok"})) }))
        .route("/v1/models", get(move || {
            let alias = alias_for_routes.clone();
            async move {
                axum::Json(serde_json::json!({
                    "object": "list",
                    "data": [{
                        "id": alias,
                        "object": "model",
                        "owned_by": "pallama-stub",
                        "created": 0_u64,
                    }],
                }))
            }
        }))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/metrics", get(metrics))
        .route("/slots", get(slots))
        .with_state(alias.clone());

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("stub-llama-server: bind {addr}: {e}"));
    eprintln!("stub-llama-server: listening on {addr} (alias {alias})");
    axum::serve(listener, app)
        .await
        .expect("stub server error");
}

type AppState = String;

#[derive(serde::Deserialize)]
struct ChatRequest {
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<serde_json::Value>,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(serde::Deserialize, Clone)]
struct ChatMessage {
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub content: serde_json::Value,
}

fn content_text(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Deterministic reply: `stub:<alias>:<last-user-text>`. Split into two
/// content chunks when streaming so SSE delta handling is exercised.
fn reply_text(state: &AppState, messages: &[ChatMessage]) -> String {
    let last_user = messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| content_text(&m.content))
        .unwrap_or_default();
    format!("stub:{state}:{last_user}")
}

fn count_tokens(s: &str) -> i64 {
    s.split_whitespace().count().max(1) as i64
}

async fn chat_completions(
    State(state): State<AppState>,
    axum::Json(req): axum::Json<ChatRequest>,
) -> axum::response::Response {
    let delay_ms: u64 = std::env::var("STUB_DELAY_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    if delay_ms > 0 {
        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
    }
    let text = reply_text(&state, &req.messages);
    let model = req.model.clone().unwrap_or_else(|| state.clone());
    let prompt_tokens = count_tokens(
        &req.messages.iter().map(|m| content_text(&m.content)).collect::<Vec<_>>().join(" "),
    );
    let completion_tokens = count_tokens(&text);

    if !req.stream {
        let body = serde_json::json!({
            "id": format!("chatcmpl-stub-{}", std::process::id()),
            "object": "chat.completion",
            "created": 0_u64,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        });
        return axum::Json(body).into_response();
    }

    let include_usage = req
        .stream_options
        .as_ref()
        .and_then(|o| o.get("include_usage"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let model_stream = model.clone();
    let stream = async_stream_sse(text, model_stream, include_usage, prompt_tokens);
    axum::response::Response::builder()
        .status(200)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(axum::body::Body::from_stream(stream))
        .unwrap()
}

fn async_stream_sse(
    text: String,
    model: String,
    include_usage: bool,
    prompt_tokens: i64,
) -> impl futures::Stream<Item = Result<bytes::Bytes, std::io::Error>> {
    use futures::stream::iter;
    let (first, second) = split_in_half(&text);
    let completion_tokens = count_tokens(&text);
    let mut events: Vec<String> = Vec::new();
    let mut chunk = |delta: serde_json::Value| {
        let payload = serde_json::json!({
            "id": format!("chatcmpl-stub-{}", std::process::id()),
            "object": "chat.completion.chunk",
            "created": 0_u64,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": null}],
        });
        format!("data: {payload}\n\n")
    };
    events.push(chunk(serde_json::json!({"role": "assistant"})));
    events.push(chunk(serde_json::json!({"content": first})));
    events.push(chunk(serde_json::json!({"content": second})));
    events.push(format!(
        "data: {}\n\n",
        serde_json::json!({
            "id": format!("chatcmpl-stub-{}", std::process::id()),
            "object": "chat.completion.chunk",
            "created": 0_u64,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        })
    ));
    if include_usage {
        events.push(format!(
            "data: {}\n\n",
            serde_json::json!({
                "id": format!("chatcmpl-stub-{}", std::process::id()),
                "object": "chat.completion.chunk",
                "created": 0_u64,
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            })
        ));
    }
    events.push("data: [DONE]\n\n".to_string());
    iter(events.into_iter().map(|e| Ok(bytes::Bytes::from(e))))
}

fn split_in_half(s: &str) -> (String, String) {
    if s.is_empty() {
        return (String::new(), String::new());
    }
    let mid = s.len() / 2;
    // Respect char boundaries.
    let mut cut = mid;
    while cut < s.len() && !s.is_char_boundary(cut) {
        cut += 1;
    }
    (s[..cut].to_string(), s[cut..].to_string())
}

async fn metrics() -> axum::response::Response {
    let body = format!(
        "# HELP stub_requests_total Total requests\n# TYPE stub_requests_total counter\nstub_requests_total 0\n# HELP stub_build Build number\n# TYPE stub_build gauge\nstub_build {}\n",
        std::env::var("STUB_BUILD").unwrap_or_else(|_| "9999".into())
    );
    (
        [(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        body,
    )
        .into_response()
}

async fn slots() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({
        "slots": [
            {"id": 0, "is_processing": false, "prompt": "", "n_ctx": 4096},
        ],
    }))
}

