//! stub-llama-server: test-double for the real llama-server child.
//!
//! Mirrors the observable surface Pallama depends on (verified against
//! upstream master `references/llama.cpp-master`):
//!   - `--list-devices` prints PLAIN TEXT (not JSON):
//!     "Available devices:\n  NAME: DESC (TOTAL MiB, FREE MiB free)\n"
//!   - `--version` prints "version: NNNN (commit)" style lines
//!   - `/health` returns {"status":"ok"}
//!   - `/v1/models`, `/v1/chat/completions` (stream + non-stream), `/metrics`,
//!     `/slots` (slot activity for cancellation tests)
//!
//! Env knobs (so one binary covers many test shapes):
//!   `STUB_ARGV_FILE`   write the full argv as a JSON array before anything else
//!   `STUB_DEVICES`     device lines to print for --list-devices (default: 1 GPU)
//!   `STUB_BUILD`       build number for --version (default 9999)
//!   `STUB_DELAY_MS`    extra latency before answering each request (default 0)

use axum::extract::State;
use axum::response::IntoResponse;

/// Serving ctx from our argv (`-c`/`--ctx-size`): lets a truncating stub
/// report realistic usage (see `trunc_usage`).
static SERVING_CTX: std::sync::OnceLock<i64> = std::sync::OnceLock::new();

/// A REAL ctx-ceiling hit reports prompt+completion at the window edge;
/// tiny counts would mean a client `max_tokens` cap instead — the exact
/// distinction the sentinel's truncation gate checks. Counted usage that
/// already sits at the edge is kept as-is (it IS the real story).
fn trunc_usage(prompt_tokens: i64, completion_tokens: i64) -> (i64, i64) {
    let c = SERVING_CTX.get().copied().unwrap_or(4096);
    if (prompt_tokens + completion_tokens) * 10 >= c * 9 {
        (prompt_tokens, completion_tokens)
    } else {
        (c - 4, 4)
    }
}

#[allow(clippy::too_many_lines)] // test fixture: argv parsing + setup in one place
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let serving = {
        let mut ctx: i64 = 4096;
        let mut it = args.iter().peekable();
        while let Some(a) = it.next() {
            if a == "-c" || a == "--ctx-size" {
                if let Some(v) = it.peek().and_then(|s| s.parse::<i64>().ok()) {
                    ctx = v;
                }
            } else if let Some(v) = a.strip_prefix("--ctx-size=") {
                if let Ok(v) = v.parse::<i64>() {
                    ctx = v;
                }
            }
        }
        ctx
    };
    let _ = SERVING_CTX.set(serving);

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

    if args.iter().any(|a| a == "--help" || a == "-h") {
        // llama-server-like option table: exercises the manifest parser.
        println!("usage: llama-server [options]");
        println!();
        println!("options:");
        println!("  -h, --help            show this help message and exit");
        println!("  -v, --version         show version information and exit");
        println!("  --list-devices        print list of available devices and exit");
        println!("  -m, --model FNAME     model path to load");
        println!("  -a, --alias STRING    set model name aliases");
        println!("  --host HOST           ip to listen");
        println!("  --port PORT           port to listen");
        println!("  -c, --ctx-size N      size of the prompt context");
        println!("  -t, --threads N       number of CPU threads");
        println!("  -ngl, --gpu-layers N  layers in VRAM (auto)");
        println!("  -fa, --flash-attn [on|off|auto]");
        println!("  --jinja               use jinja templates");
        println!("  --metrics             prometheus metrics endpoint");
        println!("  --cache-reuse N       KV-shift chunk reuse");
        println!("  -ctk, --cache-type-k TYPE");
        println!("  -ctv, --cache-type-v TYPE");
        println!("  -cmoe, --cpu-moe      keep MoE on CPU");
        println!("  --sleep-idle-seconds SECONDS");
        println!("  -np, --parallel N     server slots");
        println!("  -cb, --cont-batching  continuous batching");
        println!("  --rpc SERVERS         rpc servers");
        println!("  --lora FNAME          lora adapter");
        println!("  --lora-scaled FNAME:SCALE");
        println!("  -cram, --cache-ram N  cache size in MiB");
        println!("  -mm, --mmproj FILE    multimodal projector");
        println!("  -fit, --fit [on|off]  adjust args to fit device memory");
        println!("  --spec-type none,draft-simple,draft-eagle3,draft-mtp,ngram-simple types of speculative decoding");
        println!("  --spec-draft-model FNAME");
        println!("  --spec-draft-n-max N");
        println!("  --slots               slots endpoint");
        println!("  --cpu-range lo-hi     pin threads");
        println!("  --poll N              busy poll");
        println!("  --ubatch-size N       micro batch");
        println!("  --reasoning-format F");
        println!("  --lookup-cache-dynamic FNAME");
        println!("  --slot-save-path PATH");
        println!("  --rope-scaling NONE|LINEAR|YARN");
        println!("  --rope-scale N");
        println!("  --n-cpu-moe N");
        println!("  --override-tensor PATTERN=DEV");
        println!("  --agent               tools + mcp proxy");
        println!("  --models-dir PATH     router server models dir");
        println!("  --models-preset PATH  router server model presets (INI)");
        println!("  --models-max N        router max simultaneously loaded");
        println!("  --slot-prompt-similarity SIM");
        println!("  --api-key KEY          endpoint api key (bearer auth)");
        println!("  --api-key-file FNAME   file containing the api key");
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
    let port: u16 = flag("--port")
        .unwrap_or_else(|| "8080".into())
        .parse()
        .expect("--port must be numeric");
    let alias = flag("--alias").unwrap_or_else(|| "stub-model".into());
    // Child-auth hardening: --api-key wins, else --api-key-file content
    // (trimmed) — mirrors upstream's two intake paths.
    let api_key = flag("--api-key").or_else(|| {
        flag("--api-key-file")
            .and_then(|p| std::fs::read_to_string(p).ok())
            .map(|s| s.trim().to_string())
    });
    if let Some(dir) = flag("--slot-save-path") {
        let _ = std::fs::create_dir_all(&dir);
        SLOT_SAVE_PATH.set(dir).expect("slot path once");
    }
    // Router mode: --models-preset INI instead of -m. Parse section names;
    // requests route by body `model`, mirroring upstream proxy_post.
    if let Some(ini_path) = flag("--models-preset") {
        let raw = std::fs::read_to_string(&ini_path).unwrap_or_default();
        let mut names: Vec<String> = Vec::new();
        let mut slot_dirs: Vec<(String, String)> = Vec::new();
        let mut section: Option<String> = None;
        for line in raw.lines() {
            let t = line.trim();
            if t.starts_with('[') && t.ends_with(']') {
                let name = t[1..t.len() - 1].to_string();
                if name != "*" {
                    names.push(name.clone());
                }
                section = Some(name);
            } else if let Some((k, v)) = t.split_once('=') {
                if k.trim() == "slot-save-path" && section.as_deref() != Some("*") {
                    if let Some(s) = &section {
                        slot_dirs.push((s.clone(), v.trim().to_string()));
                    }
                }
            }
        }
        ROUTER_MODELS.set(names).expect("router models once");
        ROUTER_SLOT_DIRS
            .set(slot_dirs)
            .expect("router slot dirs once");
    }

    // Test knob: die N ms after startup (fail-fast/liveness-race tests).
    if let Some(ms) = std::env::var("STUB_DIE_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    {
        eprintln!("stub: simulated crash in {ms}ms");
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(ms));
            std::process::exit(3);
        });
    }

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    rt.block_on(serve(host, port, alias, api_key));
}

async fn serve(host: String, port: u16, alias: String, api_key: Option<String>) {
    use axum::routing::{get, post};
    let alias_for_routes = alias.clone();
    let app = axum::Router::new()
        .route(
            "/health",
            get(|| async {
                if std::env::var("STUB_HEALTH_NEVER").as_deref() == Ok("1") {
                    return (
                        axum::http::StatusCode::SERVICE_UNAVAILABLE,
                        axum::Json(serde_json::json!({"status": "loading"})),
                    )
                        .into_response();
                }
                axum::Json(serde_json::json!({"status": "ok"})).into_response()
            }),
        )
        .route(
            "/v1/models",
            get(move || {
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
            }),
        )
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/embeddings", post(embeddings))
        .route("/lora-adapters", get(lora_adapters))
        .route("/metrics", get(metrics))
        .route("/slots", get(slots))
        // New-wave surfaces: Responses API, audio transcriptions, FIM
        // infill, control vectors, tokenize, and slot save/restore/erase
        // (session checkpoints round-trip against --slot-save-path).
        .route("/v1/responses", post(responses_api))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/messages/count_tokens", post(anthropic_count_tokens))
        .route("/v1/audio/transcriptions", post(transcriptions))
        .route("/infill", post(infill))
        .route("/v1/chat/completions/control", post(control_vectors))
        .route("/tokenize", post(tokenize))
        .route("/embedding", post(embedding_legacy))
        .route("/slots/{id_slot}", post(slots_action))
        // Model-scoped upstream surfaces forwarded by the gateway's
        // scoped_proxy: props settings, slot-save streams, stream lookup,
        // and the Jina reranking alias.
        .route("/props", get(props).post(props))
        .route("/v1/stream", get(stream_list).delete(stream_delete))
        .route("/v1/streams/lookup", post(streams_lookup))
        .route("/v1/reranking", post(reranking))
        .route("/models", get(router_models))
        .route("/models/unload", post(models_unload))
        .with_state(alias.clone())
        // Child-auth middleware (upstream server-http.cpp contract):
        // every route requires the secret EXCEPT the public set
        // (/health). Authorization: Bearer or X-Api-Key both accepted.
        .layer(axum::middleware::from_fn(
            move |req: axum::http::Request<axum::body::Body>, next: axum::middleware::Next| {
                let secret = api_key.clone();
                async move {
                    if let Some(sec) = secret {
                        let path = req.uri().path();
                        if path != "/health" && path != "/v1/health" {
                            let ok = req
                                .headers()
                                .get("authorization")
                                .and_then(|v| v.to_str().ok())
                                .and_then(|v| v.strip_prefix("Bearer "))
                                .is_some_and(|v| v == sec)
                                || req
                                    .headers()
                                    .get("x-api-key")
                                    .and_then(|v| v.to_str().ok())
                                    .is_some_and(|v| v == sec);
                            if !ok {
                                return (
                                    axum::http::StatusCode::UNAUTHORIZED,
                                    "invalid or missing API key",
                                )
                                    .into_response();
                            }
                        }
                    }
                    next.run(req).await
                }
            },
        ));

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("stub-llama-server: bind {addr}: {e}"));
    eprintln!("stub-llama-server: listening on {addr} (alias {alias})");
    axum::serve(listener, app).await.expect("stub server error");
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
    // Token counts are tiny; wrap is unreachable in practice.
    #[allow(clippy::cast_possible_wrap)]
    {
        s.split_whitespace().count().max(1) as i64
    }
}

async fn chat_completions(
    State(state): State<AppState>,
    axum::Json(req): axum::Json<ChatRequest>,
) -> axum::response::Response {
    let delay_ms: u64 = std::env::var("STUB_DELAY_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if delay_ms > 0 {
        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
    }
    let model = req.model.clone().unwrap_or_else(|| state.clone());
    // Router mode: unknown body model -> upstream's error shape.
    if let Some(false) = router_model_ok(&model) {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            format!("model '{model}' not found"),
        )
            .into_response();
    }
    // Sentinel test knobs: shape the response semantics, never the route.
    let text = if env_flag("STUB_EMPTY") {
        String::new()
    } else if env_flag("STUB_SCHEMA_VIOLATION") {
        "not json at all".to_string()
    } else {
        reply_text(&state, &req.messages)
    };
    let finish = std::env::var("STUB_FINISH").unwrap_or_else(|_| "stop".into());
    let prompt_tokens = count_tokens(
        &req.messages
            .iter()
            .map(|m| content_text(&m.content))
            .collect::<Vec<_>>()
            .join(" "),
    );
    let completion_tokens = count_tokens(&text);
    let (prompt_tokens, completion_tokens) = if finish == "length" {
        trunc_usage(prompt_tokens, completion_tokens)
    } else {
        (prompt_tokens, completion_tokens)
    };

    if !req.stream {
        let body = serde_json::json!({
            "id": format!("chatcmpl-stub-{}", std::process::id()),
            "object": "chat.completion",
            "created": 0_u64,
            "model": model,
            "choices": [{
                "index": 0,
                "message": if env_flag("STUB_BAD_TOOL_ARGS") {
                    serde_json::json!({"role": "assistant", "content": "", "tool_calls": [
                        {"id": "t1", "type": "function", "function": {"name": "echo", "arguments": "{\"broken"}}
                    ]})
                } else {
                    serde_json::json!({"role": "assistant", "content": text})
                },
                "finish_reason": finish,
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
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let chunk_delay_ms: u64 = std::env::var("STUB_DELAY_CHUNK_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let events = sse_events(&text, &model, include_usage, prompt_tokens, &finish);
    let base_iter = futures::stream::iter(
        events
            .into_iter()
            .map(Ok::<axum::body::Bytes, std::io::Error>),
    );
    let stream: std::pin::Pin<
        Box<dyn futures::Stream<Item = Result<axum::body::Bytes, std::io::Error>> + Send>,
    > = if chunk_delay_ms > 0 {
        use futures::StreamExt as _;
        Box::pin(base_iter.then(move |b| async move {
            tokio::time::sleep(std::time::Duration::from_millis(chunk_delay_ms)).await;
            b
        }))
    } else {
        Box::pin(base_iter)
    };
    axum::response::Response::builder()
        .status(200)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(axum::body::Body::from_stream(stream))
        .unwrap()
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).as_deref() == Ok("1")
}

fn sse_events(
    text: &str,
    model: &str,
    include_usage: bool,
    prompt_tokens: i64,
    finish: &str,
) -> Vec<axum::body::Bytes> {
    let (first, second) = split_in_half(text);
    let completion_tokens = if finish == "length" {
        trunc_usage(prompt_tokens, count_tokens(text)).1
    } else {
        count_tokens(text)
    };
    let mut events: Vec<String> = Vec::new();
    let chunk = |delta: serde_json::Value| {
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
    if env_flag("STUB_BAD_TOOL_ARGS") {
        events.push(chunk(serde_json::json!({"tool_calls": [{"index": 0, "function": {"name": "echo", "arguments": "{\"bro"}}]})));
        events.push(chunk(
            serde_json::json!({"tool_calls": [{"index": 0, "function": {"arguments": "ken\""}}]}),
        ));
    }
    events.push(format!(
        "data: {}\n\n",
        serde_json::json!({
            "id": format!("chatcmpl-stub-{}", std::process::id()),
            "object": "chat.completion.chunk",
            "created": 0_u64,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
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
    events.into_iter().map(axum::body::Bytes::from).collect()
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

#[derive(serde::Deserialize)]
struct CompletionRequest {
    #[serde(default)]
    pub prompt: serde_json::Value,
    #[serde(default)]
    pub model: Option<String>,
}

async fn completions(
    State(state): State<AppState>,
    axum::Json(req): axum::Json<CompletionRequest>,
) -> axum::Json<serde_json::Value> {
    let prompt = match &req.prompt {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    let text = format!("stub:{state}:{prompt}");
    let tokens = i64::try_from(text.split_whitespace().count().max(1)).unwrap_or(1);
    axum::Json(serde_json::json!({
        "id": format!("cmpl-stub-{}", std::process::id()),
        "object": "text_completion",
        "created": 0_u64,
        "model": req.model.unwrap_or_else(|| state.clone()),
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": tokens,
            "completion_tokens": tokens,
            "total_tokens": tokens * 2,
        },
    }))
}

#[derive(serde::Deserialize)]
struct EmbeddingsRequest {
    #[serde(default)]
    pub input: serde_json::Value,
    #[serde(default)]
    pub model: Option<String>,
}

async fn embeddings(
    State(state): State<AppState>,
    axum::Json(req): axum::Json<EmbeddingsRequest>,
) -> axum::Json<serde_json::Value> {
    let text = match &req.input {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    // Deterministic pseudo-embedding from char codes.
    let vec: Vec<f64> = text.bytes().take(8).map(|b| f64::from(b) / 255.0).collect();
    axum::Json(serde_json::json!({
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": vec}],
        "model": req.model.unwrap_or_else(|| state.clone()),
    }))
}

async fn lora_adapters() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({"adapters": []}))
}

async fn metrics() -> axum::response::Response {
    let body = format!(
        "# HELP stub_requests_total Total requests\n# TYPE stub_requests_total counter\nstub_requests_total 0\n# HELP stub_build Build number\n# TYPE stub_build gauge\nstub_build {}\n",
        std::env::var("STUB_BUILD").unwrap_or_else(|_| "9999".into())
    );
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        body,
    )
        .into_response()
}

static SLOT_SAVE_PATH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
static ROUTER_MODELS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
static ROUTER_SLOT_DIRS: std::sync::OnceLock<Vec<(String, String)>> = std::sync::OnceLock::new();

/// Router-mode membership check: Some(name) when routing and the model is
/// known; None when not in router mode; Err-ish empty when unknown.
fn router_model_ok(model: &str) -> Option<bool> {
    ROUTER_MODELS.get().map(|m| m.iter().any(|n| n == model))
}

/// Anthropic messages shape (upstream llama-server serves /v1/messages
/// natively; the stub mirrors it for client-compat tests).
async fn anthropic_messages(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> axum::response::Response {
    let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
    let model: String = parsed["model"]
        .as_str()
        .map_or_else(|| state.clone(), str::to_string);
    let stream = parsed["stream"].as_bool().unwrap_or(false);
    if stream {
        // SSE: message_start -> content_block_delta -> message_stop.
        let sse = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_stub\"}}\n\n\
                   event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"stub\"}}\n\n\
                   event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        return axum::response::Response::builder()
            .status(200)
            .header("content-type", "text/event-stream")
            .body(axum::body::Body::from(sse))
            .unwrap_or_else(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR.into_response());
    }
    axum::Json(serde_json::json!({
        "id": "msg_stub",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "stub anthropic reply"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 4, "output_tokens": 3},
    }))
    .into_response()
}

async fn anthropic_count_tokens() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({"input_tokens": 4}))
}

async fn responses_api(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> axum::response::Response {
    let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
    let model: String = parsed["model"]
        .as_str()
        .map_or_else(|| state.clone(), str::to_string);
    let stream = parsed["stream"].as_bool().unwrap_or(false);
    // Sentinel test knobs (same semantics as the chat route).
    let incomplete = std::env::var("STUB_FINISH").as_deref() == Ok("length");
    let empty = env_flag("STUB_EMPTY");
    let bad_tool = env_flag("STUB_BAD_TOOL_ARGS");
    let status = if incomplete {
        "incomplete"
    } else {
        "completed"
    };
    let output: Vec<serde_json::Value> = if empty {
        Vec::new()
    } else if bad_tool {
        vec![serde_json::json!({
            "type": "function_call", "call_id": "c1",
            "name": "echo", "arguments": "{\"broken"
        })]
    } else {
        vec![serde_json::json!({"type": "message", "content": "stub response"})]
    };
    let usage = if incomplete {
        let (p, c) = trunc_usage(3, 2);
        serde_json::json!({"input_tokens": p, "output_tokens": c})
    } else {
        serde_json::json!({"input_tokens": 3, "output_tokens": 2})
    };
    let mut resp_obj = serde_json::json!({
        "id": "resp_stub",
        "object": "response",
        "status": status,
        "model": model,
        "output": output,
        "usage": usage,
        // Test observability: the gateway's chained reconstruction is
        // verified through this echo (never read by real clients).
        "debug_input_items": parsed["input"].clone(),
    });
    if incomplete {
        resp_obj["incomplete_details"] = serde_json::json!({"reason": "max_output_tokens"});
    }
    if !stream {
        return axum::Json(resp_obj).into_response();
    }
    let mut events: Vec<String> = Vec::new();
    events.push(format!(
        "data: {}\n\n",
        serde_json::json!({"type": "response.created", "response": {"id": "resp_stub"}})
    ));
    if bad_tool {
        events.push(format!(
            "data: {}\n\n",
            serde_json::json!({"type": "response.output_item.added", "output_index": 0,
                "item": {"type": "function_call", "call_id": "c1", "name": "echo", "arguments": ""}})
        ));
        events.push(format!(
            "data: {}\n\n",
            serde_json::json!({"type": "response.function_call_arguments.delta", "delta": "{\"bro"})
        ));
        events.push(format!(
            "data: {}\n\n",
            serde_json::json!({"type": "response.function_call_arguments.delta", "delta": "ken\""})
        ));
    } else if !empty {
        events.push(format!(
            "data: {}\n\n",
            serde_json::json!({"type": "response.output_item.added", "output_index": 0,
                "item": {"type": "message"}})
        ));
        events.push(format!(
            "data: {}\n\n",
            serde_json::json!({"type": "response.output_text.delta", "delta": "stub response"})
        ));
    }
    events.push(format!(
        "data: {}\n\n",
        serde_json::json!({
            "type": if incomplete { "response.incomplete" } else { "response.completed" },
            "response": resp_obj,
        })
    ));
    events.push("data: [DONE]\n\n".into());
    let stream = futures::stream::iter(
        events
            .into_iter()
            .map(|s| Ok::<axum::body::Bytes, std::io::Error>(axum::body::Bytes::from(s))),
    );
    axum::response::Response::builder()
        .status(200)
        .header("content-type", "text/event-stream")
        .body(axum::body::Body::from_stream(stream))
        .unwrap()
}

async fn transcriptions(
    State(state): State<AppState>,
    _body: axum::body::Bytes,
) -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({
        "text": format!("stub transcription for {state}"),
    }))
}

async fn infill(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> axum::Json<serde_json::Value> {
    let model: String = serde_json::from_slice::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| v["model"].as_str().map(str::to_string))
        .unwrap_or_else(|| state.clone());
    axum::Json(serde_json::json!({
        "model": model,
        "content": "stub infill",
    }))
}

async fn control_vectors(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> axum::Json<serde_json::Value> {
    let n = serde_json::from_slice::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| v["control_vectors"].as_array().map(Vec::len))
        .unwrap_or(0);
    axum::Json(serde_json::json!({
        "model": state,
        "control_vectors_applied": n,
    }))
}

async fn tokenize(body: axum::body::Bytes) -> axum::Json<serde_json::Value> {
    let text = serde_json::from_slice::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| v["content"].as_str().map(str::to_string))
        .unwrap_or_default();
    let tokens: Vec<u64> = text
        .split_whitespace()
        .map(|w| w.len() as u64 + 1)
        .collect();
    axum::Json(serde_json::json!({ "tokens": tokens }))
}

/// Legacy single-doc embedding route (upstream shape): accepts
/// `{"content": [token ids]}` or a plain string, returns a per-token
/// matrix `[{index, embedding: [[f64]]}]`. Deterministic 16-dim
/// multiplicative-hash vectors per id — identical prompts embed
/// identically, different id sets land in quasi-random directions.
async fn embedding_legacy(body: axum::body::Bytes) -> axum::Json<serde_json::Value> {
    let req: serde_json::Value = serde_json::from_slice(&body).unwrap_or_default();
    let ids: Vec<u64> = match &req["content"] {
        serde_json::Value::Array(a) => a.iter().filter_map(serde_json::Value::as_u64).collect(),
        serde_json::Value::String(s) => s.split_whitespace().map(|w| w.len() as u64 + 1).collect(),
        _ => Vec::new(),
    };
    let matrix: Vec<Vec<f64>> = ids
        .iter()
        .map(|id| {
            (0..16u32)
                .map(|d| {
                    let h = id
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(u64::from(d).wrapping_mul(0xD1B5_4A32_D192_ED03));
                    f64::from((h >> 33) as u32 % 1009) / 1009.0
                })
                .collect()
        })
        .collect();
    axum::Json(serde_json::json!([{ "index": 0, "embedding": matrix }]))
}

async fn slots_action(
    axum::extract::Path(id): axum::extract::Path<u32>,
    axum::extract::RawQuery(q): axum::extract::RawQuery,
    body: axum::body::Bytes,
) -> axum::response::Response {
    let query = q.unwrap_or_default();
    let action = query
        .split('&')
        .find_map(|p| p.strip_prefix("action="))
        .unwrap_or_default()
        .to_string();
    let parsed: Option<serde_json::Value> = serde_json::from_slice(&body).ok();
    let filename: String = parsed
        .as_ref()
        .and_then(|v| v["filename"].as_str().map(str::to_string))
        .unwrap_or_default();
    // Router mode: per-model subdirectory (mirrors upstream per-model
    // children each with their own --slot-save-path).
    let model_subdir: Option<String> = parsed
        .as_ref()
        .and_then(|v| v["model"].as_str().map(str::to_string));
    if filename
        .chars()
        .any(|c| !(c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-'))
        || filename.is_empty()
    {
        return (axum::http::StatusCode::BAD_REQUEST, "invalid filename").into_response();
    }
    // Dir resolution mirrors the real engine: router mode reads the
    // per-model slot-save-path from the preset INI; classic mode uses the
    // child's own --slot-save-path (already per-model).
    let base: std::path::PathBuf =
        if let (Some(m), Some(dirs)) = (&model_subdir, ROUTER_SLOT_DIRS.get()) {
            match dirs.iter().find(|(n, _)| n == m) {
                Some((_, dir)) => dir.clone().into(),
                None => {
                    return (
                        axum::http::StatusCode::NOT_IMPLEMENTED,
                        format!("router model '{m}' has no slot-save-path preset"),
                    )
                        .into_response();
                }
            }
        } else {
            match SLOT_SAVE_PATH.get() {
                Some(dir) => dir.clone().into(),
                None => {
                    return (
                        axum::http::StatusCode::NOT_IMPLEMENTED,
                        "stub started without --slot-save-path",
                    )
                        .into_response();
                }
            }
        };
    let _ = std::fs::create_dir_all(&base);
    let path = base.join(&filename);
    match action.as_str() {
        "save" => {
            let blob = format!("stub-kv:{filename}");
            std::fs::write(&path, blob).expect("stub slot save");
            axum::Json(serde_json::json!({"status": "ok", "filename": filename, "slot": id}))
                .into_response()
        }
        "restore" => match std::fs::read(&path) {
            Ok(bytes) => axum::Json(serde_json::json!({
                "status": "ok", "filename": filename,
                "restored_bytes": bytes.len(),
            }))
            .into_response(),
            Err(_) => (
                axum::http::StatusCode::NOT_FOUND,
                format!("no such checkpoint: {filename}"),
            )
                .into_response(),
        },
        "erase" => {
            let _ = std::fs::remove_file(&path);
            axum::Json(serde_json::json!({"status": "ok"})).into_response()
        }
        _ => (axum::http::StatusCode::BAD_REQUEST, "invalid action").into_response(),
    }
}

async fn router_models() -> axum::Json<serde_json::Value> {
    let data: Vec<serde_json::Value> = ROUTER_MODELS
        .get()
        .map(|names| {
            names
                .iter()
                .map(|n| {
                    serde_json::json!({
                        "id": n,
                        "object": "model",
                        "status": {"value": "loaded"},
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    axum::Json(serde_json::json!({"data": data, "object": "list"}))
}

async fn models_unload(body: axum::body::Bytes) -> axum::response::Response {
    let model: Option<String> = serde_json::from_slice::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| v["model"].as_str().map(str::to_string));
    match model {
        Some(m) if router_model_ok(&m) == Some(true) => {
            axum::Json(serde_json::json!({"success": true})).into_response()
        }
        Some(m) => (
            axum::http::StatusCode::BAD_REQUEST,
            format!("model '{m}' not found"),
        )
            .into_response(),
        None => (axum::http::StatusCode::BAD_REQUEST, "missing model").into_response(),
    }
}

async fn slots() -> axum::Json<serde_json::Value> {
    // Upstream GET /slots answers a bare array (not wrapped in an object).
    axum::Json(serde_json::json!([
        {"id": 0, "is_processing": false, "prompt": "", "n_ctx": 4096},
    ]))
}

/// GET/POST /props — echoes the alias so the e2e can prove per-child
/// routing, and merges any settings from a `POST` back into the response.
async fn props(
    State(state): State<AppState>,
    method: axum::http::Method,
    body: axum::body::Bytes,
) -> axum::Json<serde_json::Value> {
    let mut props = serde_json::json!({
        "model_alias": state,
        "default_props": {"temperature": 0.8, "top_k": 40},
    });
    if method == axum::http::Method::POST {
        if let Ok(parsed) = serde_json::from_slice::<serde_json::Value>(&body) {
            props["user_props"] = parsed;
        }
    }
    axum::Json(props)
}

/// GET /v1/stream — slot-save stream listing; `DELETE /v1/stream?stream_id=`
/// removes it (upstream novel streaming surface).
async fn stream_list(
    axum::extract::RawQuery(q): axum::extract::RawQuery,
) -> axum::Json<serde_json::Value> {
    let id = q
        .and_then(|q| {
            q.split('&')
                .find_map(|p| p.strip_prefix("stream_id=").map(str::to_string))
        })
        .unwrap_or_else(|| "s0".into());
    axum::Json(serde_json::json!({"streams": [{"stream_id": id, "slot_id": 0}]}))
}

async fn stream_delete(
    axum::extract::RawQuery(q): axum::extract::RawQuery,
) -> axum::response::Response {
    match q.and_then(|q| {
        q.split('&')
            .find_map(|p| p.strip_prefix("stream_id=").map(str::to_string))
    }) {
        Some(id) => (axum::http::StatusCode::OK, format!("deleted {id}")).into_response(),
        None => (axum::http::StatusCode::BAD_REQUEST, "missing stream_id").into_response(),
    }
}

/// POST /v1/streams/lookup — chat-shaped body; finds a stream by prompt
/// prefix. Stub answers deterministically so the e2e can pin the shape.
async fn streams_lookup(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> axum::response::Response {
    let parsed: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => {
            return (axum::http::StatusCode::BAD_REQUEST, "bad json").into_response();
        }
    };
    let Some(prompt) = parsed["prompt"].as_str() else {
        return (axum::http::StatusCode::BAD_REQUEST, "missing prompt").into_response();
    };
    axum::Json(serde_json::json!({
        "stream_id": "s0",
        "matched_prompt": prompt,
        "model_alias": state,
    }))
    .into_response()
}

/// POST /v1/reranking — Jina-style alias; ranks by naive doc length so
/// the e2e can verify pass-through ordering.
#[allow(clippy::cast_precision_loss)] // doc length -> score is a stub heuristic
async fn reranking(body: axum::body::Bytes) -> axum::response::Response {
    let parsed: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => {
            return (axum::http::StatusCode::BAD_REQUEST, "bad json").into_response();
        }
    };
    let Some(docs) = parsed["documents"].as_array() else {
        return (axum::http::StatusCode::BAD_REQUEST, "missing documents").into_response();
    };
    let mut results: Vec<(usize, f64)> = docs
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let len = d.as_str().map_or(0, str::len) as f64;
            (i, len / 100.0)
        })
        .collect();
    results.sort_by(|a, b| b.1.total_cmp(&a.1));
    let results: Vec<serde_json::Value> = results
        .into_iter()
        .map(|(i, s)| serde_json::json!({"index": i, "relevance_score": s}))
        .collect();
    axum::Json(serde_json::json!({"results": results})).into_response()
}
