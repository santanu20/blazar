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
use axum::http::{HeaderMap, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::Extension;
use serde_json::{json, Value};
use std::fmt::Write as _;
use tokio_stream::StreamExt as TsExt;

use pallama_core::store::Store;

use crate::proxy::{
    affinity_hash, child_base, ensure_with_admission, resolve_model, with_accounting,
};
use crate::queue::Priority;
use crate::sentinel;
use crate::state::AppState;
use crate::translate as tr;
use crate::TraceId;

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

/// GET /api/watch — live SSE tail of sentinel records (`pallama watch`).
/// Each committed record arrives as `data: {record json}`; a lagged
/// consumer gets a resync note and keeps streaming. History is `why`'s
/// job; this is strictly live.
pub async fn watch(State(state): State<Arc<AppState>>) -> Response {
    let scrub = state.config.pii_scrub;
    let rx = state.sentinel.watch();
    let stream = futures::stream::unfold((rx, scrub), |(mut rx, scrub)| async move {
        match rx.recv().await {
            Ok(record) => {
                let mut j = record.to_json();
                if scrub {
                    j = crate::scrub::scrub_value(&j);
                }
                Some((
                    Ok::<Bytes, std::io::Error>(Bytes::from(format!("data: {j}\n\n"))),
                    (rx, scrub),
                ))
            }
            Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => Some((
                Ok(Bytes::from(format!(
                    "data: {{\"note\":\"watch lagged, {n} records skipped\"}}\n\n"
                ))),
                (rx, scrub),
            )),
            Err(tokio::sync::broadcast::error::RecvError::Closed) => None,
        }
    });
    Response::builder()
        .status(200)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(Body::from_stream(stream))
        .unwrap_or_else(|e| api_error(500, &format!("watch body: {e}")))
}

/// GET /api/why[?trace=...] — the sentinel ring: what the model returned,
/// what was wrong with it, which knob fixes it. Powers `pallama why`.
/// `sentinel: false` reports the kill-switch state instead of an error:
/// silent-empty is the exact failure mode this exists to expose.
pub async fn why(State(state): State<Arc<AppState>>, uri: Uri) -> Response {
    let q = uri.query().unwrap_or_default();
    let mut trace = None;
    let mut model = None;
    let mut code = None;
    for pair in q.split('&') {
        if let Some(v) = pair.strip_prefix("trace=") {
            trace = Some(v.to_string());
        } else if let Some(v) = pair.strip_prefix("model=") {
            model = Some(v.to_string());
        } else if let Some(v) = pair.strip_prefix("code=") {
            code = Some(v.to_string());
        }
    }
    let records: Vec<_> = state
        .sentinel
        .why(trace.as_deref(), 100)
        .into_iter()
        .filter(|r| model.as_ref().is_none_or(|m| r.model.contains(m.as_str())))
        .filter(|r| {
            code.as_ref().is_none_or(|c| {
                r.detections
                    .iter()
                    .any(|d| d.code.as_str().contains(c.as_str()))
            })
        })
        .take(10)
        .collect();
    let records_json: Vec<_> = records
        .iter()
        .map(sentinel::SentinelRecord::to_json)
        .collect::<Vec<_>>();
    let records_json = if state.config.pii_scrub {
        records_json.iter().map(crate::scrub::scrub_value).collect()
    } else {
        records_json
    };
    axum::Json(json!({
        "sentinel": state.config.sentinel,
        "records": records_json,
    }))
    .into_response()
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
            // One logical model = LLM + projector: the reported size is the
            // full on-disk footprint (multimodal models ship both).
            let mm_bytes = m
                .mmproj_path
                .as_ref()
                .and_then(|p| std::fs::metadata(p).ok())
                .map_or(0, |md| i64::try_from(md.len()).unwrap_or(i64::MAX));
            let details_vision = m.mmproj_path.as_ref().map(|p| {
                json!({
                    "mmproj": p,
                    "mmproj_bytes": mm_bytes,
                })
            });
            json!({
                "name": format!("{}:{}", m.name, m.quant.to_lowercase()),
                "model": format!("{}:{}", m.name, m.quant.to_lowercase()),
                "modified_at": iso(m.pulled_at),
                "size": m.bytes.saturating_add(mm_bytes),
                "vision": details_vision,
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
    let name = req["model"]
        .as_str()
        .or(req["name"].as_str())
        .unwrap_or_default();
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
/// Router mode: translate the child's GET /models (per-model load state
/// from the engine's own supervisor) instead of pallama instance rows.
pub async fn ps(State(state): State<Arc<AppState>>) -> Response {
    if state.config.router {
        return ps_router(&state).await;
    }
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
                "size": p.bytes,
                "pallama_replica": p.replica,
                "pallama_state": p.state,
                "pallama_ctx": p.ctx,
                "pallama_gpu": p.gpu,
                "pallama_in_flight": p.in_flight,
                "pallama_endpoint": p.endpoint,
                "pallama_heat": p.heat,
                "expires_at": chrono_like_now().saturating_add(i64::try_from(idle_remaining * 1_000_000_000).unwrap_or(i64::MAX)),
            })
        })
        .collect();
    // Remotes: probed concurrently (3s cap each) so `ps` stays fast
    // even with a dead remote on the list.
    let remotes: Vec<Value> = if state.config.remotes.is_empty() {
        Vec::new()
    } else {
        let state_for_probe = Arc::clone(&state);
        let futs: Vec<_> = state
            .config
            .remotes
            .iter()
            .map(|r| {
                let s = &state_for_probe;
                async move { (r.name.clone(), crate::remotes::probe(s, r).await) }
            })
            .collect();
        let mut out = Vec::new();
        for f in futs {
            let (name, (ok, note)) = f.await;
            out.push(json!({"name": name, "ok": ok, "note": note}));
        }
        out
    };
    axum::Json(json!({"models": rows, "remotes": remotes})).into_response()
}

/// Router-mode ps: one process, N engine-managed models. The child's
/// GET /models reports per-model status (downloading/downloaded/unloaded/
/// loading/loaded/sleeping); `loaded_info` merges child fields when running.
async fn ps_router(state: &Arc<AppState>) -> Response {
    let engine = match state.sup.ensure(pallama_runtime::ROUTER_KEY).await {
        Ok(e) => e,
        Err(e) => return api_error(503, &e.to_string()),
    };
    let url = format!("{}/models", child_base(&engine.endpoint));
    let resp = state.http.get(&url).send().await;
    let Ok(resp) = resp else {
        return api_error(502, "router /models unreachable");
    };
    if !resp.status().is_success() {
        return api_error(502, "router /models non-success");
    }
    let v: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => return api_error(502, &format!("router /models parse: {e}")),
    };
    let idle_remaining = state.config.idle_timeout_secs;
    let rows: Vec<Value> = v["data"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .filter_map(|m| {
            let name = m["id"].as_str()?.to_string();
            let status = m["status"]["value"].as_str().unwrap_or("unknown");
            // Only engine-managed load states surface as rows; unloaded
            // presets are store rows, not live ones.
            if matches!(status, "unloaded" | "downloaded" | "downloading") {
                return None;
            }
            let pallama_state = match status {
                "loading" => "loading",
                "sleeping" => "sleeping",
                _ => "ready",
            };
            Some(json!({
                "name": name,
                "model": name,
                "size": 0,
                "pallama_state": pallama_state,
                "pallama_ctx": 0,
                "pallama_in_flight": 0,
                "pallama_endpoint": "router",
                "expires_at": chrono_like_now()
                    .saturating_add(i64::try_from(idle_remaining * 1_000_000_000).unwrap_or(i64::MAX)),
            }))
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
    let target = req["model"]
        .as_str()
        .or(req["name"].as_str())
        .unwrap_or_default()
        .to_string();
    if target.is_empty() {
        return api_error(400, "missing field: model");
    }

    let dirs = state.dirs.clone();
    let bus = state.bus.clone();
    let rx2 = bus.subscribe();
    let name_for_stream = target.clone();
    let events = tokio_stream::wrappers::BroadcastStream::new(rx2);
    // Emit-then-STOP: the terminal line (success/error) must reach the
    // client AND the stream must end right after it. `take_while` drops
    // the item that trips the predicate (silencing bogus-repo failures
    // into empty 200s); a bare filter_map never ends the body. The
    // unfold state machine gives both guarantees.
    let stream = futures::stream::unfold(
        (events, false, name_for_stream),
        |(mut events, done, name)| async move {
            if done {
                return None;
            }
            loop {
                match events.next().await {
                    Some(Ok(e)) => {
                        let (line, terminal) = pull_event_line(&name, &e);
                        return Some((
                            Ok::<_, std::io::Error>(Bytes::from(line)),
                            (events, terminal, name),
                        ));
                    }
                    // Broadcast lag yields Err: skip it, keep streaming.
                    Some(Err(_)) => {}
                    None => return None,
                }
            }
        },
    );

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
        let puller = pallama_runtime::Puller {
            dirs,
            client,
            bus: bus.clone(),
        };
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
    use pallama_runtime::PallamaEvent::{ModelPulled, PullFailed, PullProgress};
    match e {
        PullProgress {
            name: n,
            downloaded,
            total,
        } if n == name => (
            format!(
                "{}\n",
                json!({"status": "pulling", "total": total, "completed": downloaded})
            ),
            false,
        ),
        ModelPulled { name: n, warning } if n == name => {
            // ollama NDJSON parity + the health warning when the GGUF
            // header did not parse (quant alternatives included).
            let mut payload = json!({"status": "success"});
            if let Some(w) = warning {
                payload["warning"] = json!(w);
            }
            (format!("{payload}\n"), true)
        }
        PullFailed { name: n, error } if n == name => {
            (format!("{}\n", json!({"error": error})), true)
        }
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
    let stream = TsExt::chain(
        mapped,
        futures::stream::iter(Vec::<Result<Bytes, std::io::Error>>::new()),
    );
    Response::builder()
        .status(200)
        .header("content-type", "text/event-stream")
        .body(Body::from_stream(stream))
        .unwrap()
}

fn event_kind(e: &pallama_runtime::PallamaEvent) -> &'static str {
    use pallama_runtime::PallamaEvent::{
        BenchmarkDone, EngineRemoved, EngineRolledBack, EngineUpdated, InstanceStateChanged,
        ModelPreloaded, ModelPulled, ModelRemoved, PullFailed, PullProgress, QueueDepth,
        SlotsAutoAdopted,
    };
    match e {
        EngineUpdated { .. } => "engine_updated",
        EngineRemoved { .. } => "engine_removed",
        EngineRolledBack { .. } => "engine_rolled_back",
        ModelPreloaded { .. } => "model_preloaded",
        SlotsAutoAdopted { .. } => "slots_auto_adopted",
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
#[allow(clippy::too_many_lines)] // one cohesive translation + admission path
pub async fn chat(
    State(state): State<Arc<AppState>>,
    trace_ext: Option<Extension<TraceId>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return api_error(400, &format!("invalid JSON: {e}")),
    };
    let model_field = req["model"].as_str().unwrap_or_default().to_string();
    // Remote routing: `<remote>:<model>` on the ollama API too.
    if let Some((remote, remote_model)) = crate::remotes::split_remote(&model_field, &state.config)
    {
        let remote = remote.clone();
        return crate::remotes::ollama_chat_remote(&state, &remote, remote_model, &req).await;
    }
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

    // Per-key admission (scope + rate + request count), then token
    // accounting on the outgoing NDJSON (final line carries eval counts).
    let key_entry = key_ext
        .as_ref()
        .map(|Extension(k)| k.name.clone())
        .and_then(|name| state.keys.entry(&name).map(|e| (name, e)));
    if let Some((name, entry)) = &key_entry {
        if let Err(rej) = state.keys.check(entry, &row.name) {
            return rej.to_response();
        }
        state.keys.charge_request(name);
    }
    // Strict tool-def lint (tools arrive in OpenAI shape after translate).
    if let Some(err) = crate::sentinel::strict_tool_def_error(&req) {
        return api_error(400, &format!("invalid tools: {err}"));
    }
    // Prompt-fit preflight (num_ctx request override counts).
    {
        let eff = match req
            .pointer("/options/num_ctx")
            .and_then(serde_json::Value::as_u64)
        {
            Some(v) => v,
            None => u64::from(state.config.effective_ctx(&row.name)),
        };
        if let Err(resp) = crate::preflight::enforce_prompt_fits(
            &state,
            &row.name,
            &req,
            u32::try_from(eff).unwrap_or(u32::MAX),
        )
        .await
        {
            return resp;
        }
    }

    let priority = Priority::from_header(
        headers
            .get("x-pallama-priority")
            .and_then(|v| v.to_str().ok()),
    );

    // Complaint #13: honor num_ctx — restart instance once at requested
    // size when idle (no in-flight requests). Router mode has one shared
    // child: per-request restarts are impossible — fail fast with the
    // overlay fix instead of silently ignoring (H1).
    if let Some(want) = num_ctx {
        if state.config.router {
            return api_error(
                400,
                &format!(
                    "router mode serves all models from one child; per-request num_ctx is not available — set [model_overrides.{}] ctx = {} in config.toml and restart the daemon",
                    row.name, want
                ),
            );
        }
        if let Err(resp) = apply_num_ctx(&state, &row.name, want).await {
            return resp;
        }
    }

    let (engine, load_ms) =
        match ensure_with_admission(&state, &row.name, priority, affinity_hash(&req)).await {
            Ok(ok) => ok,
            Err(resp) => return resp,
        };
    // Heat under the INSTANCE key (model or model#N): replica victim
    // choice and idle tracking are key-scoped (B1).
    state.sup.note_prefix_hit(&engine.name);

    let stream = req["stream"].as_bool().unwrap_or(true);
    let keep_alive = parse_keep_alive(req.get("keep_alive"));
    let model_name = row.name.clone();
    let openai_bytes = serde_json::to_vec(&openai_req).unwrap_or_default();
    let state2 = state.clone();
    let engine2 = engine.clone();
    let accounting_name = engine.name.clone();

    let mut out = with_accounting(&state, &accounting_name, async move {
        let enforce = state2.config.sentinel && sentinel::enforce_enabled(&state2.config, &headers);
        let resp = proxy_core_chat(
            &state2,
            &engine2,
            &model_name,
            openai_bytes,
            stream,
            load_ms,
            trace_ext.map(|Extension(t)| t.0),
            enforce,
        )
        .await;
        // keep_alive=0: evict right after this response (complaint #12),
        // banking the KV checkpoint first.
        if keep_alive == Some(0) {
            let _ = state2.sup.evict_model(&model_name).await;
        }
        resp
    })
    .await;
    // Token accounting rides the outgoing ollama body (the final NDJSON
    // line / JSON carries eval counts); constructed only for keys with
    // token budgets.
    if let Some((name, _entry)) = key_entry.filter(|(_, e)| e.tpm > 0 || e.daily_tokens > 0) {
        out = crate::keys::charge_outgoing(out, &name, Arc::clone(&state.keys));
    }
    out
}

fn parse_keep_alive(v: Option<&Value>) -> Option<i64> {
    let v = v?;
    v.as_i64()
        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}

pub(crate) async fn apply_num_ctx(
    state: &Arc<AppState>,
    model: &str,
    want: i64,
) -> Result<(), Response> {
    if want <= 0 {
        return Err(api_error(400, "options.num_ctx must be positive"));
    }
    // VRAM preflight (I5): refuse a ctx that cannot fit even on an EMPTY
    // GPU — weights + f16 KV at the target ctx. Conservative by design
    // (no live-free query exists across backends), zero false positives:
    // anything that passes here still gets the engine's own fit juggling.
    {
        let store = pallama_core::Store::open(&state.dirs).ok();
        let row = store
            .as_ref()
            .and_then(|s| s.get_model(model).ok().flatten());
        if let Some(row) = row {
            if let Ok(meta) = pallama_core::read_metadata_file(std::path::Path::new(&row.path)) {
                let total_vram = state.sup.hardware.total_vram_mib();
                if total_vram > 0 {
                    let weights =
                        u64::try_from(row.bytes.max(0)).unwrap_or(u64::MAX) / (1024 * 1024);
                    // KV (MiB, f16): 2 (K+V) * layers * kv_heads * head_dim * ctx * 2B
                    let kv = crate::preflight::kv_f16_mib(
                        &meta,
                        u64::try_from(want).unwrap_or(u64::MAX),
                    );
                    if weights.saturating_add(kv) > total_vram {
                        return Err(api_error(
                            400,
                            &format!(
                            "num_ctx {want} needs ~{kv} MiB KV on top of {weights} MiB weights — \
                             over the {total_vram} MiB GPU. Lower num_ctx, pull a smaller quant, \
                             or set cache_type = \"q8_0\""
                        ),
                        ));
                    }
                }
            }
        }
    }
    // If an instance exists at a smaller ctx and is idle: recycle it; the
    // next ensure recompiles with the config/overlay ctx. To honor the
    // per-request value, set the runtime ctx via the supervisor seam.
    let running = state.sup.ps().into_iter().find(|p| p.name == model);
    if let Some(p) = running {
        if i64::from(p.ctx) < want && p.in_flight == 0 {
            let _ = state.sup.evict_model(model).await;
            state
                .sup
                .set_next_ctx(model, u32::try_from(want).unwrap_or(u32::MAX));
        }
    } else {
        state
            .sup
            .set_next_ctx(model, u32::try_from(want).unwrap_or(u32::MAX));
    }
    Ok(())
}

#[allow(clippy::too_many_lines, clippy::too_many_arguments)] // one cohesive translation path
async fn proxy_core_chat(
    state: &Arc<AppState>,
    engine: &pallama_runtime::EngineRef,
    model: &str,
    openai_body: Vec<u8>,
    stream: bool,
    load_ms: u128,
    trace: Option<String>,
    enforce: bool,
) -> Response {
    let base = child_base(&engine.endpoint);
    let url = format!("{base}/v1/chat/completions");
    let req = state
        .http
        .post(&url)
        .header("content-type", "application/json");
    if !stream {
        // We translate the non-stream response into ollama shape.
        let t0 = std::time::Instant::now();
        let resp = match req.body(openai_body.clone()).send().await {
            Ok(r) => {
                state.ttft.observe_secs(t0.elapsed().as_secs_f64());
                r
            }
            Err(e) => {
                tracing::warn!(model, "nonstream upstream failed: {e:#}");
                // Reap now so the NEXT request respawns instead of
                // 502-looping until the periodic reaper notices (~10s).
                state.sup.reap_dead_children().await;
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
        // Non-stream + enforce: judge before translation (streaming stays
        // warn-only — bytes are already on the wire).
        if enforce {
            let (ctx, _) =
                sentinel::request_ctx(state, "ollama-chat", model, &openai_body, trace, false);
            let hard =
                state
                    .sentinel
                    .judge(&ctx, &serde_json::to_vec(&openai).unwrap_or_default(), 200);
            if !hard.is_empty() {
                let detail = hard
                    .iter()
                    .map(|d| format!("[{}] {}", d.code.as_str(), d.detail))
                    .collect::<Vec<_>>()
                    .join("; ");
                return api_error(422, &format!("sentinel enforce: {detail}"));
            }
        } else {
            let (feed, _) = sentinel::begin_chat_observation(
                state,
                "ollama-chat",
                model,
                &openai_body,
                trace,
                200,
                false,
            );
            feed.value(openai.clone());
            // Drop fires End: the analyzer finalizes off the response path.
            drop(feed);
        }
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
        Err(e) => {
            // Same crash-recovery contract as the OpenAI proxy path.
            state.sup.reap_dead_children().await;
            return api_error(502, &format!("engine request failed: {e:#}"));
        }
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
    // Sentinel: same side-channel tap as the OpenAI path — bytes cloned
    // in the observation closure, analyzer parses off the hot path.
    let (sentinel_feed, _) = sentinel::begin_chat_observation(
        state,
        "ollama-chat",
        model,
        &openai_body,
        trace,
        200,
        true,
    );
    // Evidence loop: TTFT on first upstream byte, inter-chunk cadence after.
    let t0 = std::time::Instant::now();
    let mut first = true;
    let mut last = t0;
    let hist_state = std::sync::Arc::clone(state);
    let stream = futures::StreamExt::map(upstream, move |chunk| {
        let now = std::time::Instant::now();
        if first {
            hist_state.ttft.observe_secs((now - t0).as_secs_f64());
            first = false;
        } else {
            hist_state.tpot.observe_secs((now - last).as_secs_f64());
        }
        last = now;
        if let Ok(bytes) = chunk.as_ref() {
            sentinel_feed.bytes(bytes.as_ref());
        }
        chunk.map_err(|e| std::io::Error::other(e.to_string()))
    });
    // Translate SSE -> NDJSON incrementally.
    let buf = String::new();
    let model_owned = model_c.clone();
    let ndjson = futures::stream::unfold(
        (
            stream,
            buf,
            model_owned,
            false,
            None::<Value>,
            None::<String>,
            false,
        ),
        |(mut stream, mut buf, model, mut done, mut usage, mut finish, mut usage_sent)| async move {
            loop {
                if done && !usage_sent {
                    let final_chunk =
                        tr::ollama_final_chunk(&model, usage.as_ref(), finish.as_deref());
                    usage_sent = true;
                    return Some((
                        Ok(Bytes::from(format!("{final_chunk}\n"))),
                        (stream, buf, model, done, usage, finish, usage_sent),
                    ));
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
                        return Some((
                            Ok(Bytes::from(body)),
                            (stream, buf, model, done, usage, finish, usage_sent),
                        ));
                    }
                    Some(Err(e)) => {
                        return Some((
                            Err(std::io::Error::other(e.to_string())),
                            (stream, buf, model, done, usage, finish, usage_sent),
                        ));
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
    builder
        .body(Body::from_stream(ndjson))
        .unwrap_or_else(|e| api_error(500, &e.to_string()))
}

/// POST /api/embeddings (legacy ollama shape).
pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    body: Bytes,
) -> Response {
    let req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return api_error(400, &format!("invalid JSON: {e}")),
    };
    let openai_req = match tr::embeddings_to_openai(&req) {
        Ok(v) => v,
        Err(e) => return api_error(400, &e),
    };
    let model = openai_req["model"].as_str().unwrap_or_default().to_string();
    // Scope + request count (embeddings carry no token usage; budgets
    // apply on request counts only for this route).
    if let Some(Extension(k)) = &key_ext {
        if let Some(entry) = state.keys.entry(&k.name) {
            if let Err(rej) = state.keys.check(&entry, &model) {
                return rej.to_response();
            }
            state.keys.charge_request(&k.name);
        }
    }
    let (engine, _) = match ensure_with_admission(&state, &model, Priority::Normal, None).await {
        Ok(ok) => ok,
        Err(resp) => return resp,
    };
    with_accounting(&state, &engine.name.clone(), async {
        let url = format!("{}/v1/embeddings", child_base(&engine.endpoint));
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
        axum::Json(tr::openai_embeddings_to_ollama(&model, &openai)).into_response()
    })
    .await
}

/// POST /api/embed — ollama new-style embeddings (input: str | [str]);
/// same engine lane as /api/embeddings, array-friendly shape.
pub async fn embed(
    State(state): State<Arc<AppState>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    body: Bytes,
) -> Response {
    let req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return api_error(400, &format!("invalid JSON: {e}")),
    };
    let model = req["model"].as_str().unwrap_or_default().to_string();
    if model.is_empty() {
        return api_error(400, "missing \"model\"");
    }
    // Normalize input to the OpenAI array shape (str -> [str]).
    let inputs: Vec<Value> = match &req["input"] {
        Value::String(s) => vec![Value::String(s.clone())],
        Value::Array(a) if !a.is_empty() => a.clone(),
        _ => {
            return api_error(
                400,
                "\"input\" must be a string or a non-empty array of strings",
            )
        }
    };
    if let Some(Extension(k)) = &key_ext {
        if let Some(entry) = state.keys.entry(&k.name) {
            if let Err(rej) = state.keys.check(&entry, &model) {
                return rej.to_response();
            }
            state.keys.charge_request(&k.name);
        }
    }
    let (engine, _) = match ensure_with_admission(&state, &model, Priority::Normal, None).await {
        Ok(ok) => ok,
        Err(resp) => return resp,
    };
    with_accounting(&state, &engine.name.clone(), async {
        let url = format!("{}/v1/embeddings", child_base(&engine.endpoint));
        let openai_req = json!({"model": model, "input": inputs});
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
        // ollama /api/embed shape: {model, embeddings: [[f32]]}
        let embeddings: Vec<Value> = openai["data"]
            .as_array()
            .map(|d| d.iter().map(|e| e["embedding"].clone()).collect())
            .unwrap_or_default();
        axum::Json(json!({"model": model, "embeddings": embeddings})).into_response()
    })
    .await
}

/// POST /api/rerank — ollama-lane rerank: forwards to the child's
/// /v1/rerank. Accepts `documents` as strings or {text} objects
/// (normalizes to the `OpenAI` string shape); response passes through
/// verbatim (results + usage).
pub async fn rerank(
    State(state): State<Arc<AppState>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    body: Bytes,
) -> Response {
    let req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return api_error(400, &format!("invalid JSON: {e}")),
    };
    let model = req["model"].as_str().unwrap_or_default().to_string();
    if model.is_empty() {
        return api_error(400, "missing \"model\"");
    }
    if req
        .get("query")
        .and_then(Value::as_str)
        .unwrap_or("")
        .is_empty()
    {
        return api_error(400, "missing \"query\"");
    }
    let Some(docs) = req.get("documents").and_then(Value::as_array) else {
        return api_error(400, "\"documents\" must be an array");
    };
    // {text: "..."} objects -> plain strings.
    let normalized: Vec<Value> = docs
        .iter()
        .map(|d| match d {
            Value::String(s) => Value::String(s.clone()),
            Value::Object(_) => Value::String(d["text"].as_str().unwrap_or_default().to_string()),
            other => other.clone(),
        })
        .collect();
    let forward = json!({
        "model": model,
        "query": req["query"],
        "documents": normalized,
        "top_n": req.get("top_n").cloned().unwrap_or(json!(docs.len())),
    });
    if let Some(Extension(k)) = &key_ext {
        if let Some(entry) = state.keys.entry(&k.name) {
            if let Err(rej) = state.keys.check(&entry, &model) {
                return rej.to_response();
            }
            state.keys.charge_request(&k.name);
        }
    }
    let (engine, _) = match ensure_with_admission(&state, &model, Priority::Normal, None).await {
        Ok(ok) => ok,
        Err(resp) => return resp,
    };
    with_accounting(&state, &engine.name.clone(), async {
        let url = format!("{}/v1/rerank", child_base(&engine.endpoint));
        let resp = match state.http.post(&url).json(&forward).send().await {
            Ok(r) => r,
            Err(e) => return api_error(502, &format!("engine request failed: {e:#}")),
        };
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        // Engine error text (e.g. non-rerank model) passes through with
        // its status — teaching, not masking.
        if status == 200 {
            match serde_json::from_str::<Value>(&text) {
                Ok(v) => return axum::Json(v).into_response(),
                Err(e) => return api_error(502, &format!("bad engine response: {e}")),
            }
        }
        api_error(status, &text)
    })
    .await
}

/// POST /api/generate — raw prompts only (templated -> 400 + pointer).
#[allow(clippy::too_many_lines)] // one cohesive translation path
pub async fn generate(
    State(state): State<Arc<AppState>>,
    trace_ext: Option<Extension<TraceId>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
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
    let key_entry = key_ext
        .as_ref()
        .map(|Extension(k)| k.name.clone())
        .and_then(|name| state.keys.entry(&name).map(|e| (name, e)));
    if let Some((name, entry)) = &key_entry {
        if let Err(rej) = state.keys.check(entry, &model) {
            return rej.to_response();
        }
        state.keys.charge_request(name);
    }
    let priority = Priority::from_header(
        headers
            .get("x-pallama-priority")
            .and_then(|v| v.to_str().ok()),
    );
    let (engine, _) =
        match ensure_with_admission(&state, &model, priority, affinity_hash(&req)).await {
            Ok(ok) => ok,
            Err(resp) => return resp,
        };
    let mut out = with_accounting(&state, &engine.name.clone(), async {
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
        let (feed, _) = sentinel::begin_chat_observation(
            &state,
            "ollama-generate",
            &model,
            &serde_json::to_vec(&openai_req).unwrap_or_default(),
            trace_ext.map(|Extension(t)| t.0),
            200,
            false,
        );
        feed.value(openai.clone());
        drop(feed);
        axum::Json(tr::openai_completion_to_ollama(&model, &openai)).into_response()
    })
    .await;
    // /api/generate responses are always whole JSON with usage inside.
    if let Some((name, _entry)) = key_entry.filter(|(_, e)| e.tpm > 0 || e.daily_tokens > 0) {
        out = crate::keys::charge_outgoing(out, &name, Arc::clone(&state.keys));
    }
    out
}

/// POST /api/evict {"model": name} — pallama-internal (ollama has no REST
/// equivalent): unload a model now. Powers `pallama stop <model>`.
pub async fn evict(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let v: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return api_error(400, &e.to_string()),
    };
    let Some(model) = v["model"].as_str() else {
        return api_error(400, "missing 'model'");
    };
    // Router mode: the engine owns per-model lifecycle — forward the
    // unload to the router child (models stay preset-listed, just unloaded).
    if state.config.router {
        let engine = match state.sup.ensure(pallama_runtime::ROUTER_KEY).await {
            Ok(e) => e,
            Err(e) => return api_error(503, &e.to_string()),
        };
        let url = format!("{}/models/unload", child_base(&engine.endpoint));
        return match state
            .http
            .post(&url)
            .json(&json!({"model": model}))
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => axum::Json(json!({"status": "ok"})).into_response(),
            Ok(r) => {
                let code = r.status().as_u16();
                let text = r.text().await.unwrap_or_default();
                api_error(code, &text)
            }
            Err(e) => api_error(502, &format!("router unload: {e}")),
        };
    }
    match state.sup.evict_model(model).await {
        Ok(()) => axum::Json(json!({"status": "ok"})).into_response(),
        Err(e) => api_error(404, &e.to_string()),
    }
}

/// A session checkpoint filename: alphanumerics, dot, underscore, dash.
/// Mirrors upstream `fs_validate_filename` — no path separators, ever.
fn valid_session_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-')
        && !name.starts_with('.')
}

/// POST /api/session {"model", "action": "save|restore|erase",
/// "filename", "slot": 0} — pallama-internal: slot KV-cache
/// checkpoints via upstream `--slot-save-path` + `POST /slots/{id}`.
/// Ensures the model is loaded first (save needs live slot state).
pub async fn session(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let v: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return api_error(400, &format!("invalid JSON: {e}")),
    };
    let Some(model) = v["model"].as_str() else {
        return api_error(400, "missing 'model'");
    };
    let Some(action) = v["action"].as_str() else {
        return api_error(400, "missing 'action' (save | restore | erase)");
    };
    if !matches!(action, "save" | "restore" | "erase") {
        return api_error(400, "action must be save, restore or erase");
    }
    let Some(filename) = v["filename"].as_str() else {
        return api_error(400, "missing 'filename'");
    };
    if !valid_session_name(filename) {
        return api_error(
            400,
            "filename must be [A-Za-z0-9._-], not start with '.', max 128 chars",
        );
    }
    let slot = v["slot"].as_u64().unwrap_or(0);
    let slot = u32::try_from(slot).unwrap_or(0);

    // Erase is a FILE deletion, owned by the daemon (it owns the sessions
    // dir). Upstream's slot-erase action clears live KV state, not the
    // checkpoint file — forwarding would report success and delete nothing.
    if action == "erase" {
        let path = state
            .dirs
            .sessions_dir()
            .join(pallama_core::profile::path_safe(model))
            .join(filename);
        return match std::fs::remove_file(&path) {
            Ok(()) => axum::Json(json!({"status": "ok", "filename": filename})).into_response(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                api_error(404, &format!("no such checkpoint: {filename}"))
            }
            Err(e) => api_error(500, &format!("delete {filename}: {e}")),
        };
    }

    let (engine, _load_ms) =
        match ensure_with_admission(&state, model, Priority::Normal, None).await {
            Ok(ok) => ok,
            Err(resp) => return resp,
        };
    with_accounting(&state, &engine.name.clone(), async {
        let url = format!(
            "{}/slots/{}?action={action}&filename={filename}",
            child_base(&engine.endpoint),
            slot
        );
        // `model` rides along for router mode (upstream proxy_post routes
        // by body model); classic children already run per-model dirs and
        // never see the extra key.
        let body_json = if state.config.router {
            json!({"filename": filename, "model": model})
        } else {
            json!({"filename": filename})
        };
        let resp = state.http.post(&url).json(&body_json).send().await;
        match resp {
            Ok(r) => {
                let status =
                    StatusCode::from_u16(r.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
                let body = r.text().await.unwrap_or_default();
                (status, body).into_response()
            }
            Err(e) => api_error(502, &format!("child slot action failed: {e}")),
        }
    })
    .await
}

/// GET /api/session?model=NAME — list checkpoint files for a model from
/// the sessions dir the daemon owns (no child required).
pub async fn session_list(
    State(state): State<Arc<AppState>>,
    axum::extract::RawQuery(query): axum::extract::RawQuery,
) -> Response {
    let model = query
        .as_deref()
        .and_then(|q| q.split('&').find_map(|p| p.strip_prefix("model=")))
        .unwrap_or_default()
        .to_string();
    if model.is_empty() {
        return api_error(400, "missing ?model=");
    }
    let dir = state
        .dirs
        .sessions_dir()
        .join(pallama_core::profile::path_safe(&model));
    let mut files: Vec<Value> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for entry in rd.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let size = entry.metadata().map_or(0, |m| m.len());
            files.push(json!({"filename": name, "bytes": size}));
        }
    }
    files.sort_by(|a, b| a["filename"].as_str().cmp(&b["filename"].as_str()));
    axum::Json(json!({"model": model, "sessions": files})).into_response()
}

/// GET /metrics — merged children prometheus text + pallama_* gauges.
/// Emit a poller-fed gauge iff it has been measured at least once.
fn hint_gauge(out: &mut String, rate: Option<f64>, name: &str, help: &str) {
    if let Some(rate) = rate {
        let _ = write!(
            out,
            "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {rate}\n"
        );
    }
}

/// Active engine build gauge; absent when the store is not readable.
fn engine_build_gauge(state: &AppState, out: &mut String) {
    if let Ok(store) = Store::open(&state.dirs) {
        if let Ok(Some(engine)) = store.active_engine() {
            if let Ok(m) = serde_json::from_str::<pallama_runtime::Manifest>(&engine.manifest) {
                let _ = write!(
                    out,
                    "# HELP pallama_engine_build Active engine build number\n# TYPE pallama_engine_build gauge\npallama_engine_build {}\n",
                    m.build_number
                );
            }
        }
    }
}

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
    let _ = write!(
        merged,
        "# HELP pallama_models_loaded Models currently loaded\n# TYPE pallama_models_loaded gauge\npallama_models_loaded {}\n",
        state.sup.ps().len()
    );
    let _ = write!(
        merged,
        "# HELP pallama_queue_depth Waiting requests\n# TYPE pallama_queue_depth gauge\npallama_queue_depth {}\n",
        state.queue.depth()
    );
    let _ = write!(
        merged,
        "# HELP pallama_evictions_total Total instance evictions\n# TYPE pallama_evictions_total counter\npallama_evictions_total {}\n",
        state.sup.evictions.load(std::sync::atomic::Ordering::Relaxed)
    );
    // Per-key usage today (the [[keys]] tier's live accounting).
    let _ = write!(
        merged,
        "# HELP pallama_key_usage_requests Requests today per key\n# TYPE pallama_key_usage_requests counter\n"
    );
    for (name, _, req, _) in state.keys.usage_snapshot() {
        let _ = writeln!(merged, "pallama_key_usage_requests{{key=\"{name}\"}} {req}");
    }
    let _ = write!(
        merged,
        "# HELP pallama_key_usage_tokens Tokens today per key\n# TYPE pallama_key_usage_tokens counter\n"
    );
    for (name, _, _, tok) in state.keys.usage_snapshot() {
        let _ = writeln!(merged, "pallama_key_usage_tokens{{key=\"{name}\"}} {tok}");
    }
    // Spec-under-saturation: speculative drafting costs compute the
    // batch needs; flag it so operators flip spec=off under load.
    {
        let spec_on = !state.config.spec.is_empty() && state.config.spec != "off";
        let saturated = state.queue.depth() > 0;
        let _ = write!(
            merged,
            "# HELP pallama_spec_saturation 1 = spec decoding active while requests are queued (drafting competes with the batch)\n# TYPE pallama_spec_saturation gauge\npallama_spec_saturation {}\n",
            u8::from(spec_on && saturated)
        );
    }
    // Prefix heat per loaded model (why a capacity eviction chose its
    // victim — hot caches survive).
    let _ = write!(
        merged,
        "# HELP pallama_model_prefix_heat Recency-weighted prefix heat (capacity-eviction bias)\n# TYPE pallama_model_prefix_heat gauge\n"
    );
    for p in state.sup.ps() {
        let _ = writeln!(
            merged,
            "pallama_model_prefix_heat{{model=\"{}\"}} {}",
            p.name, p.heat
        );
    }
    state.ttft.render(&mut merged);
    state.tpot.render(&mut merged);
    // SLO burn (F4): waiters admitted past their deadline class.
    let _ = write!(
        merged,
        "# HELP pallama_slo_deadline_exceeded_total Requests admitted AFTER their SLO deadline expired (served late: queue burn)\n# TYPE pallama_slo_deadline_exceeded_total counter\npallama_slo_deadline_exceeded_total {}\n",
        state.queue.slo_deadline_exceeded()
    );
    // Measured prompt-cache hit rate (A16): drives the adaptive
    // --cache-ram clamp. Absent until the poller has a full window.
    hint_gauge(
        &mut merged,
        state.sup.cache_hint.get(),
        "pallama_prefix_cache_hit_rate",
        "Windowed prompt-cache hit rate (cache_n / (cache_n + prompt_n))",
    );
    // Speculative-decoding acceptance (G3): accepted / drafted tokens,
    // EWMA'd. Absent while no spec-decoding child is live.
    hint_gauge(
        &mut merged,
        state.sup.spec_accept.get(),
        "pallama_spec_accept_rate",
        "Windowed speculative-decoding acceptance rate (accepted / drafted tokens)",
    );
    engine_build_gauge(&state, &mut merged);
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        merged,
    )
        .into_response()
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use pallama_runtime::PallamaEvent;

    #[test]
    fn unit__pull_event_line__warning_rides_success_ndjson() {
        let (line, done) = pull_event_line(
            "m1",
            &PallamaEvent::ModelPulled {
                name: "m1".into(),
                warning: Some("GGUF metadata unreadable: try Q8_0".into()),
            },
        );
        assert!(done);
        let v: Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(v["status"], "success");
        assert_eq!(v["warning"], "GGUF metadata unreadable: try Q8_0");
    }

    #[test]
    fn unit__pull_event_line__clean_pull_has_no_warning_field() {
        let (line, done) = pull_event_line(
            "m1",
            &PallamaEvent::ModelPulled {
                name: "m1".into(),
                warning: None,
            },
        );
        assert!(done);
        let v: Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(v["status"], "success");
        assert!(v.get("warning").is_none());
    }
}
