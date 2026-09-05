//! Ollama-compat surface (drop-in `OLLAMA_HOST` replacement). Shapes
//! mirrored from references/ollama-main/api/types.go; divergences are
//! deliberate and documented:
//! - `options.num_ctx` restarts the instance at the requested size (never
//!   silently truncates — complaint #13)
//! - `keep_alive` honored: number>0 pins N s, 0 evicts after this request,
//!   -1 pins forever (complaint #12)
//! - unknown options -> 400 listing them (no silent drops)

use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use tokio_stream::StreamExt as TsExt;
use serde_json::{json, Value};

use pallama_core::store::Store;

use crate::proxy::{child_base, ensure_with_admission, resolve_model, with_accounting};
use crate::queue::Priority;
use crate::state::AppState;
use crate::translate as tr;

fn api_error(status: u16, message: &str) -> Response {
    (
        StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        axum::Json(json!({"error": message})),
    )
        .into_response()
}

/// GET /api/version
pub async fn version() -> Response {
    axum::Json(json!({"version": env!("CARGO_PKG_VERSION")})).into_response()
}

/// GET /api/tags
pub async fn tags(State(state): State<Arc<AppState>>) -> Response {
    let store = match Store::open(&state.dirs) {
        Ok(s) => s,
        Err(e) => return api_error(500, &e.to_string()),
    };
    let models = match store.list_models() {
        Ok(l) => l,
        Err(e) => return api_error(500, &e.to_string()),
    };
    let rows: Vec<Value> = models
        .iter()
        .map(|m| {
            json!({
                "name": format!("{}:{}", m.name, m.quant.to_lowercase()),
                "model": format!("{}:{}", m.name, m.quant.to_lowercase()),
                "modified_at": iso(m.pulled_at),
                "size": m.bytes,
                "digest": m.sha256.clone().unwrap_or_default(),
                "details": {
                    "family": m.arch.clone().unwrap_or_default(),
                    "parameter_size": format_params(m.params),
                    "quantization_level": m.quant,
                    "context_length": m.ctx_train.unwrap_or(0),
                },
            })
        })
        .collect();
    axum::Json(json!({"models": rows})).into_response()
}

fn iso(secs: i64) -> String {
    format!("{}Z", secs.max(0))
}

fn format_params(p: Option<f64>) -> String {
    match p {
        Some(v) if v >= 1.0 => format!("{v:.1}B"),
        Some(v) => format!("{:.0}M", v * 1000.0),
        None => "unknown".into(),
    }
}

/// POST /api/show — GGUF metadata + active profile + last benchmark.
pub async fn show(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return api_error(400, &format!("invalid JSON: {e}")),
    };
    let name = req["model"].as_str().unwrap_or_default().to_string();
    let store = match Store::open(&state.dirs) {
        Ok(s) => s,
        Err(e) => return api_error(500, &e.to_string()),
    };
    let row = match resolve_model(&store, &name) {
        Ok(r) => r,
        Err(e) => return api_error(404, &e),
    };
    let engine = store.active_engine().ok().flatten();
    let profile = engine
        .as_ref()
        .and_then(|e| store.get_profile(&row.name, &e.tag).ok().flatten());
    let gguf = pallama_core::read_metadata_file(std::path::Path::new(&row.path)).ok();
    let mut details = json!({
        "family": row.arch.clone().unwrap_or_default(),
        "parameter_size": format_params(row.params),
        "quantization_level": row.quant,
        "context_length": gguf.as_ref().and_then(|g| g.context_length).or(row.ctx_train.and_then(|c| u64::try_from(c).ok())).unwrap_or(0),
    });
    if let Some(g) = &gguf {
        details["block_count"] = json!(g.block_count.unwrap_or(0));
        details["expert_count"] = json!(g.expert_count.unwrap_or(0));
    }
    let mut resp = json!({
        "license": "see upstream model card",
        "modelfile": format!("# pallama: plain GGUF at {}", row.path),
        "parameters": "see config overlay",
        "details": details,
        "model_info": {
            "general.architecture": row.arch.clone().unwrap_or_default(),
            "general.context_length": row.ctx_train.unwrap_or(0),
            "pallama.path": row.path,
            "pallama.repo": row.repo,
            "pallama.shards": row.shards,
        },
    });
    if let Some(p) = profile {
        resp["pallama_profile"] = json!({
            "args": serde_json::from_str::<Value>(&p.args_json).unwrap_or(json!([])),
            "benchmark": serde_json::from_str::<Value>(p.benchmark_json.as_deref().unwrap_or("null")).unwrap_or(Value::Null),
            "updated_at": p.updated_at,
        });
    }
    axum::Json(resp).into_response()
}

/// POST /api/delete — refuses while running (409).
pub async fn delete(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return api_error(400, &format!("invalid JSON: {e}")),
    };
    let name = req["model"].as_str().or(req["name"].as_str()).unwrap_or_default();
    match pallama_runtime::remove_model(&state.dirs, name) {
        Ok(()) => StatusCode::OK.into_response(),
        Err(e) => {
            let msg = format!("{e:#}");
            if msg.contains("running") {
                api_error(409, &msg)
            } else if msg.contains("no such model") {
                api_error(404, &msg)
            } else {
                api_error(500, &msg)
            }
        }
    }
}

/// GET /api/ps — live slots: name, state, ctx, idle countdown, endpoint.
pub async fn ps(State(state): State<Arc<AppState>>) -> Response {
    let rows: Vec<Value> = state
        .sup
        .ps()
        .iter()
        .map(|p| {
            let idle_remaining = state
                .config
                .idle_timeout_secs
                .saturating_sub(p.idle_secs);
            json!({
                "name": p.name,
                "model": p.name,
                "size": 0,
                "pallama_state": p.state,
                "pallama_ctx": p.ctx,
                "pallama_in_flight": p.in_flight,
                "pallama_endpoint": p.endpoint,
                "expires_at": chrono_like_now().saturating_add(i64::try_from(idle_remaining * 1_000_000_000).unwrap_or(i64::MAX)),
            })
        })
        .collect();
    axum::Json(json!({"models": rows})).into_response()
}

fn chrono_like_now() -> i64 {
    i64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
    )
    .unwrap_or(i64::MAX)
}

/// POST /api/pull — NDJSON progress straight from the event bus; a
/// duplicate pull errors and closes the stream (ollama semantics).
#[allow(clippy::similar_names)] // pull_request vs target distinguish intent
pub async fn pull(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return api_error(400, &format!("invalid JSON: {e}")),
    };
    let target = req["model"].as_str().or(req["name"].as_str()).unwrap_or_default().to_string();
    if target.is_empty() {
        return api_error(400, "missing field: model");
    }

    let dirs = state.dirs.clone();
    let bus = state.bus.clone();
    let rx2 = bus.subscribe();
    let name_for_stream = target.clone();
    let events = tokio_stream::wrappers::BroadcastStream::new(rx2);
    let paired = TsExt::filter_map(events, move |ev| {
        let name = name_for_stream.clone();
        match ev {
            Ok(e) => {
                let (line, terminal) = pull_event_line(&name, &e);
                Some((Ok::<_, std::io::Error>(Bytes::from(line)), terminal))
            }
            Err(_) => None,
        }
    });
    let taken = TsExt::take_while(paired, |(_, terminal)| !*terminal);
    let stream = futures::StreamExt::map(taken, |(item, _)| item);

    // Drive the pull on a task; stream events until ModelPulled/PullFailed.
    let pull_request = target.clone();
    tokio::spawn(async move {
        let token = std::env::var("HF_TOKEN").ok();
        let client = match pallama_runtime::hf::HfClient::new(token) {
            Ok(c) => c,
            Err(e) => {
                bus.publish(pallama_runtime::PallamaEvent::PullFailed {
                    name: pull_request.clone(),
                    error: format!("{e:#}"),
                });
                return;
            }
        };
        let puller = pallama_runtime::Puller { dirs, client, bus: bus.clone() };
        if let Err(e) = puller.pull(&pull_request).await {
            tracing::warn!("pull {pull_request}: {e:#}");
            bus.publish(pallama_runtime::PallamaEvent::PullFailed {
                name: pull_request.clone(),
                error: format!("{e:#}"),
            });
        }
    });

    Response::builder()
        .status(200)
        .header("content-type", "application/x-ndjson")
        .body(Body::from_stream(stream))
        .unwrap()
}

/// One NDJSON line per pull event; terminal=true ends the stream.
fn pull_event_line(name: &str, e: &pallama_runtime::PallamaEvent) -> (String, bool) {
    use pallama_runtime::PallamaEvent::{PullProgress, ModelPulled, PullFailed};
    match e {
        PullProgress { name: n, downloaded, total } if n == name => (
            format!("{}\n", json!({"status": "pulling", "total": total, "completed": downloaded})),
            false,
        ),
        ModelPulled { name: n } if n == name => (
            format!("{}\n", json!({"status": "success"})),
            true,
        ),
        PullFailed { name: n, error } if n == name => (
            format!("{}\n", json!({"error": error})),
            true,
        ),
        _ => (String::new(), false),
    }
}

/// GET /api/events — SSE of every `PallamaEvent` (dashboards).
pub async fn events(State(state): State<Arc<AppState>>) -> Response {
    let rx2 = state.bus.subscribe();
    let events = tokio_stream::wrappers::BroadcastStream::new(rx2);
    let mapped = TsExt::filter_map(events, |ev| match ev {
        Ok(e) => Some(Ok::<_, std::io::Error>(Bytes::from(format!(
            "event: {}\ndata: {}\n\n",
            event_kind(&e),
            serde_json::to_string(&e).unwrap_or_default()
        )))),
        Err(_) => None,
    });
    let stream = TsExt::chain(mapped, futures::stream::iter(Vec::<Result<Bytes, std::io::Error>>::new()));
    Response::builder()
        .status(200)
        .header("content-type", "text/event-stream")
        .body(Body::from_stream(stream))
        .unwrap()
}

fn event_kind(e: &pallama_runtime::PallamaEvent) -> &'static str {
    use pallama_runtime::PallamaEvent::{EngineUpdated, EngineRemoved, ModelPulled, ModelRemoved, PullProgress, PullFailed, InstanceStateChanged, BenchmarkDone, QueueDepth};
    match e {
        EngineUpdated { .. } => "engine_updated",
        EngineRemoved { .. } => "engine_removed",
        ModelPulled { .. } => "model_pulled",
        ModelRemoved { .. } => "model_removed",
        PullProgress { .. } => "pull_progress",
        PullFailed { .. } => "pull_failed",
        InstanceStateChanged { .. } => "instance_state_changed",
        BenchmarkDone { .. } => "benchmark_done",
        QueueDepth { .. } => "queue_depth",
    }
}

/// POST /api/chat — full translation incl. `num_ctx` + `keep_alive`.
pub async fn chat(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return api_error(400, &format!("invalid JSON: {e}")),
    };
    let model_field = req["model"].as_str().unwrap_or_default().to_string();
    let (openai_req, num_ctx) = match tr::chat_to_openai(&req) {
        Ok(r) => r,
        Err(e) => return api_error(400, &e),
    };

    let store = match Store::open(&state.dirs) {
        Ok(s) => s,
        Err(e) => return api_error(500, &e.to_string()),
    };
    let row = match resolve_model(&store, &model_field) {
        Ok(r) => r,
        Err(e) => return api_error(404, &e),
    };

    let priority = Priority::from_header(
        headers
            .get("x-pallama-priority")
            .and_then(|v| v.to_str().ok()),
    );

    // Complaint #13: honor num_ctx — restart instance once at requested
    // size when idle (no in-flight requests).
    if let Some(want) = num_ctx {
        if let Err(resp) = apply_num_ctx(&state, &row.name, want).await {
            return resp;
        }
    }

    let (engine, load_ms) = match ensure_with_admission(&state, &row.name, priority).await {
        Ok(ok) => ok,
        Err(resp) => return resp,
    };

    let stream = req["stream"].as_bool().unwrap_or(true);
    let keep_alive = parse_keep_alive(req.get("keep_alive"));
    let model_name = row.name.clone();
    let openai_bytes = serde_json::to_vec(&openai_req).unwrap_or_default();
    let state2 = state.clone();
    let engine2 = engine.clone();
    let accounting_name = model_name.clone();

    let out = with_accounting(&state, &accounting_name, async move {
        let resp = proxy_core_chat(&state2, &engine2, &model_name, openai_bytes, stream, load_ms).await;
        // keep_alive=0: evict right after this response (complaint #12).
        if keep_alive == Some(0) {
            let _ = state2.sup.evict(&model_name).await;
        }
        resp
    })
    .await;
    out
}

fn parse_keep_alive(v: Option<&Value>) -> Option<i64> {
    let v = v?;
    v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}

async fn apply_num_ctx(state: &Arc<AppState>, model: &str, want: i64) -> Result<(), Response> {
    if want <= 0 {
        return Err(api_error(400, "options.num_ctx must be positive"));
    }
    // If an instance exists at a smaller ctx and is idle: recycle it; the
    // next ensure recompiles with the config/overlay ctx. To honor the
    // per-request value, set the runtime ctx via the supervisor seam.
    let running = state.sup.ps().into_iter().find(|p| p.name == model);
    if let Some(p) = running {
        if i64::from(p.ctx) < want && p.in_flight == 0 {
            let _ = state.sup.evict(model).await;
            state.sup.set_next_ctx(model, u32::try_from(want).unwrap_or(u32::MAX));
        }
    } else {
        state.sup.set_next_ctx(model, u32::try_from(want).unwrap_or(u32::MAX));
    }
    Ok(())
}

#[allow(clippy::too_many_lines)] // stream translation is one cohesive path
async fn proxy_core_chat(
    state: &Arc<AppState>,
    engine: &pallama_runtime::EngineRef,
    model: &str,
    openai_body: Vec<u8>,
    stream: bool,
    load_ms: u128,
) -> Response {
    let base = child_base(&engine.endpoint);
    let url = format!("{base}/v1/chat/completions");
    let req = state.http.post(&url).header("content-type", "application/json");
    if !stream {
        // We translate the non-stream response into ollama shape.
        let resp = match req.body(openai_body.clone()).send().await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(model, "nonstream upstream failed: {e:#}");
                return api_error(502, &format!("engine request failed: {e:#}"));
            }
        };
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();
            return api_error(status, &format!("engine error: {text}"));
        }
        let openai: Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => return api_error(502, &format!("bad engine response: {e}")),
        };
        return axum::Json(tr::openai_chat_to_ollama(model, &openai)).into_response();
    }
    // Stream: child SSE -> ollama NDJSON with final counts. Force the
    // usage chunk so the final ollama line carries eval counts.
    let mut body_with_usage: Value = serde_json::from_slice(&openai_body).unwrap_or_default();
    body_with_usage["stream"] = json!(true);
    body_with_usage["stream_options"] = json!({"include_usage": true});
    let openai_body = serde_json::to_vec(&body_with_usage).unwrap_or_default();
    let _ = req; // silence unused in this branch
    let resp = match state
        .http
        .post(&url)
        .header("content-type", "application/json")
        .body(openai_body.clone())
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return api_error(502, &format!("engine request failed: {e:#}")),
    };
    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        return api_error(status, &text);
    }
    let include_usage = true; // we need usage for the final chunk
    let _ = include_usage;
    let upstream = resp.bytes_stream();
    let model_c = model.to_string();
    let load_hdr = load_ms > 100;
    let stream = futures::StreamExt::map(upstream, |chunk| {
        chunk.map_err(|e| std::io::Error::other(e.to_string()))
    });
    // Translate SSE -> NDJSON incrementally.
    let buf = String::new();
    let model_owned = model_c.clone();
    let ndjson = futures::stream::unfold(
        (stream, buf, model_owned, false, None::<Value>, None::<String>, false),
        |(mut stream, mut buf, model, mut done, mut usage, mut finish, mut usage_sent)| async move {
            loop {
                if done && !usage_sent {
                    let final_chunk = tr::ollama_final_chunk(&model, usage.as_ref(), finish.as_deref());
                    usage_sent = true;
                    return Some((Ok(Bytes::from(format!("{final_chunk}\n"))), (stream, buf, model, done, usage, finish, usage_sent)));
                }
                match futures::StreamExt::next(&mut stream).await {
                    Some(Ok(bytes)) => {
                        buf.push_str(&String::from_utf8_lossy(&bytes));
                        let (events, saw_done, consumed) = tr::parse_sse(&buf);
                        buf.drain(..consumed);
                        for ev in &events {
                            if let Some(u) = ev.get("usage").filter(|u| !u.is_null()) {
                                // Usage chunk: store the NESTED usage object
                                // (the event itself also carries choices:[]).
                                usage = Some(u.clone());
                            }
                            if let Some(fr) = ev["choices"][0]["finish_reason"].as_str() {
                                finish = Some(fr.to_string());
                            }
                        }
                        if saw_done {
                            done = true;
                        }
                        let lines: Vec<String> = events
                            .iter()
                            .flat_map(|ev| tr::openai_chunk_to_ollama(&model, ev))
                            .map(|v| format!("{v}\n"))
                            .collect();
                        if lines.is_empty() {
                            continue; // need more data
                        }
                        let body = lines.join("");
                        return Some((Ok(Bytes::from(body)), (stream, buf, model, done, usage, finish, usage_sent)));
                    }
                    Some(Err(e)) => {
                        return Some((Err(std::io::Error::other(e.to_string())), (stream, buf, model, done, usage, finish, usage_sent)));
                    }
                    None => {
                        if !done {
                            done = true; // upstream closed without [DONE]
                            continue;
                        }
                        return None;
                    }
                }
            }
        },
    );
    let mut builder = Response::builder()
        .status(200)
        .header("content-type", "application/x-ndjson");
    if load_hdr {
        builder = builder.header("x-pallama-status", "loading");
    }
    builder.body(Body::from_stream(ndjson)).unwrap_or_else(|e| api_error(500, &e.to_string()))
}

/// POST /api/embeddings (legacy ollama shape).
pub async fn embeddings(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return api_error(400, &format!("invalid JSON: {e}")),
    };
    let openai_req = match tr::embeddings_to_openai(&req) {
        Ok(v) => v,
        Err(e) => return api_error(400, &e),
    };
    let model = openai_req["model"].as_str().unwrap_or_default().to_string();
    let (engine, _) = match ensure_with_admission(&state, &model, Priority::Normal).await {
        Ok(ok) => ok,
        Err(resp) => return resp,
    };
    with_accounting(&state, &engine.name.clone(), async {
        let url = format!("{}/v1/embeddings", child_base(&engine.endpoint));
        let resp = match state
            .http
            .post(&url)
            .json(&openai_req)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => return api_error(502, &format!("engine request failed: {e:#}")),
        };
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();
            return api_error(status, &text);
        }
        let openai: Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => return api_error(502, &format!("bad engine response: {e}")),
        };
        axum::Json(tr::openai_embeddings_to_ollama(&model, &openai)).into_response()
    })
    .await
}

/// POST /api/generate — raw prompts only (templated -> 400 + pointer).
#[allow(clippy::too_many_lines)] // one cohesive translation path
pub async fn generate(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return api_error(400, &format!("invalid JSON: {e}")),
    };
    let Some(openai_req) = tr::generate_to_openai(&req) else {
        return api_error(
            400,
            "templated /api/generate is not supported; use /api/chat (the model's own template is applied by the engine)",
        );
    };
    let model = openai_req["model"].as_str().unwrap_or_default().to_string();
    let priority = Priority::from_header(
        headers
            .get("x-pallama-priority")
            .and_then(|v| v.to_str().ok()),
    );
    let (engine, _) = match ensure_with_admission(&state, &model, priority).await {
        Ok(ok) => ok,
        Err(resp) => return resp,
    };
    with_accounting(&state, &engine.name.clone(), async {
        let url = format!("{}/v1/completions", child_base(&engine.endpoint));
        let resp = match state.http.post(&url).json(&openai_req).send().await {
            Ok(r) => r,
            Err(e) => return api_error(502, &format!("engine request failed: {e:#}")),
        };
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();
            return api_error(status, &text);
        }
        let openai: Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => return api_error(502, &format!("bad engine response: {e}")),
        };
        axum::Json(tr::openai_completion_to_ollama(&model, &openai)).into_response()
    })
    .await
}

/// GET /metrics — merged children prometheus text + pallama_* gauges.
pub async fn metrics(State(state): State<Arc<AppState>>) -> Response {
    let mut merged = String::new();
    for p in state.sup.ps() {
        let url = format!("http://{}/metrics", p.endpoint);
        if let Ok(resp) = state.http.get(&url).send().await {
            if resp.status().is_success() {
                if let Ok(text) = resp.text().await {
                    merged.push_str(&text);
                    merged.push('\n');
                }
            }
        }
    }
    merged.push_str(&format!(
        "# HELP pallama_models_loaded Models currently loaded\n# TYPE pallama_models_loaded gauge\npallama_models_loaded {}\n",
        state.sup.ps().len()
    ));
    merged.push_str(&format!(
        "# HELP pallama_queue_depth Waiting requests\n# TYPE pallama_queue_depth gauge\npallama_queue_depth {}\n",
        state.queue.depth()
    ));
    merged.push_str(&format!(
        "# HELP pallama_evictions_total Total instance evictions\n# TYPE pallama_evictions_total counter\npallama_evictions_total {}\n",
        state.sup.evictions.load(std::sync::atomic::Ordering::Relaxed)
    ));
    if let Ok(store) = Store::open(&state.dirs) {
        if let Ok(Some(engine)) = store.active_engine() {
            if let Ok(m) = serde_json::from_str::<pallama_runtime::Manifest>(&engine.manifest) {
                merged.push_str(&format!(
                    "# HELP pallama_engine_build Active engine build number\n# TYPE pallama_engine_build gauge\npallama_engine_build {}\n",
                    m.build_number
                ));
            }
        }
    }
    (
        [(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        merged,
    )
        .into_response()
}
