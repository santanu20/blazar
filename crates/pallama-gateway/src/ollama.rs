//! Ollama-compat surface (drop-in `OLLAMA_HOST` replacement). Shapes
//! mirrored from ollama's upstream `api/types.go`; divergences are
//! deliberate and documented:
//! - `options.num_ctx` restarts the instance at the requested size (never
//!   silently truncates — complaint #13)
//! - `keep_alive` honored: number>0 pins N s, 0 evicts after this request,
//!   -1 pins forever (complaint #12)
//! - unknown options -> 400 listing them (no silent drops)

use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::Extension;
use serde_json::{json, Value};
use std::fmt::Write as _;
use tokio_stream::StreamExt as TsExt;

use crate::proxy::{affinity_hash, child_auth, child_base, ensure_with_admission, resolve_model};
use crate::queue::Priority;
use crate::semcache;
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

/// GET /api/why — the sentinel ring: what the model returned,
/// what was wrong with it, which knob fixes it. Powers `pallama why`.
/// `sentinel: false` reports the kill-switch state instead of an error:
/// silent-empty is the exact failure mode this exists to expose.
/// Params: `trace=<id>` (exact match), `model=<substr>`,
/// `code=<substr>` (detection code), `flagged=1` (detection-carrying
/// records only), `limit=<n>` (default 10, capped at the ring size) —
/// filters apply before the newest-first cut so flagged records older
/// than the latest clean batch stay reachable.
/// Minimal percent-decoding for hand-parsed query values (F19: model
/// names like `Qwen3%2F0.6` must round-trip; `+` is left alone since
/// path-shaped values never use the form-encoding space convention).
fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            let hi = (b[i + 1] as char).to_digit(16);
            let lo = (b[i + 2] as char).to_digit(16);
            if let (Some(hi), Some(lo)) = (hi, lo) {
                // to_digit(16) caps each at 0xF, so the sum is always <= 255.
                out.push(u8::try_from(hi * 16 + lo).expect("hex digit pair <= 0xFF"));
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

pub async fn why(State(state): State<Arc<AppState>>, uri: Uri) -> Response {
    let q = uri.query().unwrap_or_default();
    let mut trace = None;
    let mut model = None;
    let mut code = None;
    let mut flagged_only = false;
    let mut limit = 10usize;
    for pair in q.split('&') {
        if let Some(v) = pair.strip_prefix("trace=") {
            trace = Some(percent_decode(v));
        } else if let Some(v) = pair.strip_prefix("model=") {
            model = Some(percent_decode(v));
        } else if let Some(v) = pair.strip_prefix("code=") {
            code = Some(percent_decode(v));
        } else if pair == "flagged=1" {
            flagged_only = true;
        } else if let Some(v) = pair.strip_prefix("limit=") {
            limit = v.parse().unwrap_or(10);
        }
    }
    // Scan the full ring (not a fixed 100) so filtered views reach older
    // records; the cap + take live in filter_why_records. `limit` above
    // RING_CAP would just return the whole ring — clamp for honesty.
    let records = sentinel::filter_why_records(
        state.sentinel.why(trace.as_deref(), sentinel::RING_CAP),
        model.as_deref(),
        code.as_deref(),
        flagged_only,
        limit.min(sentinel::RING_CAP),
    );
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
    let models = match state.with_store(pallama_core::Store::list_models) {
        Some(Ok(l)) => l,
        Some(Err(e)) => return api_error(500, &e.to_string()),
        None => return api_error(500, "store unavailable"),
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
            // Which engine row WOULD serve this model right now (routed
            // lane, or the global active row in manual mode). null =
            // nothing can serve it — the spawn path teaches on use.
            let engine =
                crate::proxy::resolve_serving(&state, &m.name, &m.path).and_then(|lane| lane.tag);
            json!({
                "name": format!("{}:{}", m.name, m.quant.to_lowercase()),
                "model": format!("{}:{}", m.name, m.quant.to_lowercase()),
                "modified_at": iso(m.pulled_at),
                "size": m.bytes.saturating_add(mm_bytes),
                "vision": details_vision,
                "digest": m.sha256.clone().unwrap_or_default(),
                "engine": engine,
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

/// Civil date from days-since-epoch (Howard Hinnant's `civil_from_days`,
/// proleptic Gregorian; same integer math as `keys::utc_day`). Feeds
/// [`iso`].
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// RFC3339 UTC stamp from epoch seconds (ollama emits full date-time
/// strings, not bare epoch ints — F16).
fn iso(secs: i64) -> String {
    let secs = secs.max(0);
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem / 60) % 60,
        rem % 60
    )
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
    // One cached-connection visit gathers row + profile: the guard lives
    // and drops inside this sync block (never across await).
    let Some((row, profile)) = state.with_store(|s| {
        let row = resolve_model(s, &name);
        let profile = match &row {
            Ok(r) => s
                .active_engine()
                .ok()
                .flatten()
                .and_then(|e| s.get_profile(&r.name, &e.tag).ok().flatten()),
            Err(_) => None,
        };
        (row, profile)
    }) else {
        return api_error(500, "store unavailable");
    };
    let row = match row {
        Ok(r) => r,
        Err(e) => return api_error(404, &e),
    };
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
    // ollama-parity capability discovery (vision clients read this):
    // vision iff an mmproj projector is attached to the model —
    // evidence from the store, never a filename guess.
    let mut capabilities = vec!["completion"];
    if row.mmproj_path.is_some() {
        capabilities.push("vision");
    }
    // Thinking capability is template-evidenced — the same marker sniff
    // the think:true request gate uses. Downstream tooling reads this
    // to decide whether the `think` request key is legal; an
    // unreadable row never claims the capability (fail-closed).
    if gguf
        .as_ref()
        .and_then(|g| g.chat_template.as_deref())
        .is_some_and(template_supports_thinking)
    {
        capabilities.push("thinking");
    }
    let mut resp = json!({
        "license": "see upstream model card",
        "modelfile": format!("# pallama: plain GGUF at {}", row.path),
        "parameters": "see config overlay",
        "capabilities": capabilities,
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
    // F22: classify from typed state, not error-string substrings — an
    // unrelated io error whose text contains "running" must not 409.
    if pallama_runtime::instance_running(&state.dirs, name) {
        return api_error(
            409,
            &format!(
                "model {name} is currently running; stop it first (`pallama stop {name}` or wait for eviction)"
            ),
        );
    }
    // unwrap_or(false): store unavailable != model-missing — never
    // fabricate a 404; remove_model below surfaces the real failure.
    if state
        .with_store(|s| s.get_model(name).ok().flatten().is_none())
        .unwrap_or(false)
    {
        return api_error(404, &format!("no such model: {name}"));
    }
    match pallama_runtime::remove_model(&state.dirs, name) {
        Ok(()) => StatusCode::OK.into_response(),
        // State may shift between pre-checks and removal (race) —
        // classify from live state again, never from the message text.
        Err(e) => {
            if pallama_runtime::instance_running(&state.dirs, name) {
                api_error(409, &format!("{e:#}"))
            } else if state
                .with_store(|s| s.get_model(name).ok().flatten().is_none())
                .unwrap_or(false)
            {
                api_error(404, &format!("{e:#}"))
            } else {
                api_error(500, &format!("{e:#}"))
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
            // keep_alive pin extends the visible countdown past the idle
            // timeout (F12 + README "ps shows the countdown").
            let remaining = idle_remaining.max(p.keep_alive_secs.unwrap_or(0));
            let now_secs = i64::try_from(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            )
            .unwrap_or(i64::MAX);
            json!({
                "name": p.name,
                "model": p.name,
                "size": p.bytes,
                "pallama_replica": p.replica,
                "pallama_engine": p.engine,
                "pallama_state": p.state,
                "pallama_ctx": p.ctx,
                "pallama_gpu": p.gpu,
                "pallama_device": p.device,
                "pallama_warnings": p.warnings,
                "pallama_spec": p.spec_mode,
                "pallama_draft": p.draft,
                "pallama_cache_hit": state.obs.hit_ratio(&p.name),
                "pallama_in_flight": p.in_flight,
                "pallama_endpoint": p.endpoint,
                "pallama_heat": p.heat,
                "expires_at": iso(now_secs.saturating_add(i64::try_from(remaining).unwrap_or(i64::MAX))),
            })
        })
        .collect();
    // Remotes: probed concurrently (3s cap each) so `ps` stays fast
    // even with a dead remote on the list (F15: collected futures are
    // NOT concurrent under a sequential await loop — join them).
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
        for (name, (ok, note)) in futures::future::join_all(futs).await {
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
    let engine = match crate::proxy::ensure_router_detached(&state.sup).await {
        Ok(e) => e,
        Err(e) => return api_error(503, &e.to_string()),
    };
    let url = format!("{}/models", child_base(&engine.endpoint));
    let resp = child_auth(state.http.get(&url), &engine).send().await;
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
            let now_secs = i64::try_from(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            )
            .unwrap_or(i64::MAX);
            Some(json!({
                "name": name,
                "model": name,
                "size": 0,
                "pallama_state": pallama_state,
                "pallama_ctx": 0,
                "pallama_spec": "",
                "pallama_draft": null,
                "pallama_in_flight": 0,
                "pallama_endpoint": "router",
                "expires_at": iso(now_secs.saturating_add(i64::try_from(idle_remaining).unwrap_or(i64::MAX))),
            }))
        })
        .collect();
    axum::Json(json!({"models": rows})).into_response()
}

/// Names a /api/pull stream must match. The puller publishes under the
/// normalized registry name ("ggml-org/Qwen3-0.6B-GGUF:Q4_K_M" ->
/// "qwen3-0.6b"), while early failures (parse/lock/client) are published
/// by the handler under the raw request target. Match both so every
/// terminal path reaches the client.
fn pull_stream_names(target: &str) -> Vec<String> {
    let mut names = vec![target.to_string()];
    if let Ok(parsed) = pallama_runtime::hf::parse_pull_target(target) {
        names.push(pallama_runtime::hf::registry_name(&parsed.repo));
    }
    names
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
    let stream_names = pull_stream_names(&target);
    let events = tokio_stream::wrappers::BroadcastStream::new(rx2);
    // Emit-then-STOP: the terminal line (success/error) must reach the
    // client AND the stream must end right after it. `take_while` drops
    // the item that trips the predicate (silencing bogus-repo failures
    // into empty 200s); a bare filter_map never ends the body. The
    // unfold state machine gives both guarantees.
    let stream = futures::stream::unfold(
        (events, false, stream_names),
        |(mut events, done, names)| async move {
            if done {
                return None;
            }
            loop {
                match events.next().await {
                    Some(Ok(e)) => {
                        let (line, terminal) = pull_event_line(&names, &e);
                        // F17: non-matching bus events produce no frame —
                        // skip them instead of yielding empty DATA chunks.
                        if line.is_empty() && !terminal {
                            continue;
                        }
                        return Some((
                            Ok::<_, std::io::Error>(Bytes::from(line)),
                            (events, terminal, names),
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
    let dl_conns = state.config.download_connections;
    tokio::spawn(async move {
        let token = std::env::var("HF_TOKEN").ok();
        let client = match pallama_runtime::hf::HfClient::new(token)
            .map(|c| c.with_download_connections(dl_conns))
        {
            Ok(c) => c,
            Err(e) => {
                bus.publish(pallama_runtime::PallamaEvent::PullFailed {
                    name: pull_request.clone(),
                    error: format!("{e:#}"),
                });
                return;
            }
        };
        // API pulls never force: a format flip needs the CLI's explicit
        // --force (the refusal message teaches it).
        let puller = pallama_runtime::Puller {
            dirs,
            client,
            bus: bus.clone(),
            force: false,
        };
        match puller.route_pull(&pull_request).await {
            Ok(outcome) => tracing::info!(
                model = %outcome.row.name,
                already_present = outcome.already_present,
                "pull finished"
            ),
            Err(e) => {
                tracing::warn!("pull {pull_request}: {e:#}");
                bus.publish(pallama_runtime::PallamaEvent::PullFailed {
                    name: pull_request.clone(),
                    error: format!("{e:#}"),
                });
            }
        }
    });

    Response::builder()
        .status(200)
        .header("content-type", "application/x-ndjson")
        .body(Body::from_stream(stream))
        .unwrap()
}

/// One NDJSON line per pull event; terminal=true ends the stream.
/// `names` holds both the raw request target and the normalized registry
/// name (see `pull_stream_names`).
fn pull_event_line(names: &[String], e: &pallama_runtime::PallamaEvent) -> (String, bool) {
    use pallama_runtime::PallamaEvent::{ModelPulled, PullFailed, PullProgress};
    let matches = |n: &str| names.iter().any(|nm| n == nm);
    match e {
        PullProgress {
            name: n,
            downloaded,
            total,
        } if matches(n) => (
            format!(
                "{}\n",
                json!({"status": "pulling", "total": total, "completed": downloaded})
            ),
            false,
        ),
        ModelPulled { name: n, warning } if matches(n) => {
            // ollama NDJSON parity + the health warning when the GGUF
            // header did not parse (quant alternatives included).
            let mut payload = json!({"status": "success"});
            if let Some(w) = warning {
                payload["warning"] = json!(w);
            }
            (format!("{payload}\n"), true)
        }
        PullFailed { name: n, error } if matches(n) => {
            (format!("{}\n", json!({"error": error})), true)
        }
        _ => (String::new(), false),
    }
}

/// GET /api/events — SSE of every `PallamaEvent` (dashboards).
pub async fn events(State(state): State<Arc<AppState>>) -> Response {
    let rx2 = state.bus.subscribe();
    // F18: scrub parity with /api/watch — event payloads today carry
    // names/counts only, but error strings can surface remote URLs.
    let scrub = state.config.pii_scrub;
    let events = tokio_stream::wrappers::BroadcastStream::new(rx2);
    let mapped = TsExt::filter_map(events, move |ev| match ev {
        Ok(e) => {
            let body = serde_json::to_value(&e).unwrap_or_default();
            let body = if scrub {
                crate::scrub::scrub_value(&body)
            } else {
                body
            };
            Some(Ok::<_, std::io::Error>(Bytes::from(format!(
                "event: {}\ndata: {body}\n\n",
                event_kind(&e),
            ))))
        }
        Err(_) => None,
    });
    Response::builder()
        .status(200)
        .header("content-type", "text/event-stream")
        .body(Body::from_stream(mapped))
        .unwrap()
}

fn event_kind(e: &pallama_runtime::PallamaEvent) -> &'static str {
    use pallama_runtime::PallamaEvent::{
        BenchmarkDone, EngineRemoved, EngineRolledBack, EngineUpdated, InstanceStateChanged,
        ModelPreloaded, ModelPulled, ModelRemoved, PullFailed, PullProgress, QueueDepth,
        SlotsAutoAdopted, SlotsCtxAutoFit,
    };
    match e {
        EngineUpdated { .. } => "engine_updated",
        EngineRemoved { .. } => "engine_removed",
        EngineRolledBack { .. } => "engine_rolled_back",
        ModelPreloaded { .. } => "model_preloaded",
        SlotsAutoAdopted { .. } => "slots_auto_adopted",
        SlotsCtxAutoFit { .. } => "slots_ctx_auto_fit",
        ModelPulled { .. } => "model_pulled",
        ModelRemoved { .. } => "model_removed",
        PullProgress { .. } => "pull_progress",
        PullFailed { .. } => "pull_failed",
        InstanceStateChanged { .. } => "instance_state_changed",
        BenchmarkDone { .. } => "benchmark_done",
        QueueDepth { .. } => "queue_depth",
    }
}

/// ollama unload idiom: `{model, keep_alive: 0}` with no inference
/// payload is a load-state ping, not a request — real ollama answers
/// 200 (harnesses release models with exactly this shape; a 400 here
/// broke `_unload_all`-style sweeps). Evict-if-running, session pins
/// still win (R3, same contract as the serve-then-evict path), then a
/// minimal lane-shaped done body.
async fn unload_ping(state: &Arc<AppState>, model: &str, chat_shape: bool) -> Response {
    let row = match state.with_store(|s| resolve_model(s, model)) {
        Some(Ok(r)) => r,
        Some(Err(e)) => return api_error(404, &e),
        None => return api_error(500, "store unavailable"),
    };
    let pinned = state
        .sup
        .sessions
        .pins(
            &row.name,
            std::time::Duration::from_secs(state.config.session_keep_secs),
        )
        .live;
    if !pinned {
        let _ = state.sup.evict_model(&row.name).await;
    }
    let body = if chat_shape {
        serde_json::json!({
            "model": row.name,
            "created_at": tr::iso_now(),
            "message": {"role": "assistant", "content": ""},
            "done": true,
        })
    } else {
        serde_json::json!({
            "model": row.name,
            "created_at": tr::iso_now(),
            "response": "",
            "done": true,
        })
    };
    (
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
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
    let mut req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return api_error(400, &format!("invalid JSON: {e}")),
    };
    let model_field = req["model"].as_str().unwrap_or_default().to_string();
    // Key admission resolved ONCE, before any lane decision (F11): the
    // remote branch below needs it too — previously `remote:<model>`
    // requests bypassed scope/rate/request-count entirely.
    let key_entry = key_ext
        .as_ref()
        .map(|Extension(k)| k.name.clone())
        .and_then(|name| state.keys.entry(&name).map(|e| (name, e)));
    // Remote routing: `<remote>:<model>` on the ollama API too —
    // POOL-aware (C4): select among same-named remotes with prefix
    // stickiness + circuit health, not just the first configured entry.
    if crate::remotes::split_remote(&model_field, &state.config).is_some() {
        // F11: remote lanes pay key admission against the full
        // `remote:model` string (same semantics as the OpenAI lane's
        // split: a scope entry like `my-remote:*` admits, anything
        // else rejects).
        if let Some((name, entry)) = &key_entry {
            if let Err(rej) = state.keys.check(entry, &model_field) {
                return rej.to_response();
            }
            state.keys.charge_request(name);
        }
        let prefix = crate::proxy::affinity_hash(&req);
        let mut out = match crate::remotes::select_remote(&state, &model_field, prefix.as_ref()) {
            Ok((remote, remote_model, _lease, akey)) => {
                let remote = remote.clone();
                let resp =
                    crate::remotes::ollama_chat_remote(&state, &remote, remote_model, &req).await;
                crate::remotes::tag_remote_result(&state, &remote, akey, resp)
            }
            Err(resp) => resp,
        };
        // Token budgets ride the remote NDJSON exactly like the local
        // lane (final line carries eval counts).
        if let Some((name, _entry)) = key_entry.filter(|(_, e)| e.tpm > 0 || e.daily_tokens > 0) {
            out = crate::keys::charge_outgoing(out, &name, Arc::clone(&state.keys));
        }
        return out;
    }
    // ollama unload idiom: no messages + keep_alive 0 = release ping.
    if parse_keep_alive(req.get("keep_alive")) == Some(0)
        && req
            .get("messages")
            .and_then(Value::as_array)
            .is_none_or(|m: &Vec<Value>| m.is_empty())
    {
        return unload_ping(&state, &model_field, true).await;
    }
    let row = match state.with_store(|s| resolve_model(s, &model_field)) {
        Some(Ok(r)) => r,
        Some(Err(e)) => return api_error(404, &e),
        None => return api_error(500, "store unavailable"),
    };

    // Key admission already ran above (F11 hoist) — scope check against
    // the RESOLVED row name + request-count charge for the local lane.
    if let Some((name, entry)) = &key_entry {
        if let Err(rej) = state.keys.check(entry, &row.name) {
            return rej.to_response();
        }
        state.keys.charge_request(name);
    }
    // Agent-client cache teaching (advisory, never mutates `req`): flag
    // a conversation whose system-prompt/tools fingerprint churns turn
    // over turn — that client silently re-prefills from scratch every
    // request. Runs on the local lane only (remote hops manage their
    // own cache).
    crate::cache_bust::note_request(
        &state.sentinel,
        &state.cache_bust,
        &row.name,
        &req,
        "api/chat",
        &trace_ext
            .as_ref()
            .map(|Extension(t)| t.0.clone())
            .unwrap_or_default(),
    );
    // Think-capability gate (ollama parity): refuse `think: true` on a
    // provably non-thinking template instead of silently ignoring it.
    // Both think gates run BEFORE translation — the translator maps
    // `think` to engine kwargs, so a post-translate mutation would never
    // reach the child (live-caught: injected `think:false` was a no-op
    // while explicit `think:false` worked).
    if let Some(resp) = refuse_unsupported_think(&row, &req) {
        return resp;
    }
    // Ollama parity for the absent toggle: thinking-capable templates
    // default OFF when the caller did not say `think` (local lane only —
    // remote forwards carry the caller's own bytes to the remote's policy).
    default_think_off(&row, &mut req);
    let (mut openai_req, num_ctx) = match tr::chat_to_openai(&req) {
        Ok(r) => r,
        Err(e) => return api_error(400, &e),
    };
    // mistral.rs children register models as `default` (see proxy.rs).
    // Pre-spawn: predict the routed lane (engine not resolved yet).
    if crate::proxy::child_model_default_predicted(&state, &row.name, &row.path) {
        crate::proxy::set_child_model_default(&mut openai_req);
    }
    // Strict tool-def lint (tools arrive in OpenAI shape after translate).
    if let Some(err) = state.sentinel.strict_tool_def_error_cached(&req) {
        return api_error(400, &format!("invalid tools: {err}"));
    }
    // Structured-output lint (R9): malformed format/grammar fails fast
    // before admission instead of dying at the child (or being ignored).
    if let Some(err) = state.sentinel.structured_output_error_cached(&req) {
        return api_error(400, &format!("invalid structured output: {err}"));
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
            return *resp;
        }
    }

    // F27: validate num_ctx BEFORE the semantic cache — a hit must not
    // skip the validation a miss enforces (miss→400, hit→200 asymmetry).
    // The restart itself stays on the miss path (a hit needs no model).
    if let Some(want) = num_ctx {
        if want <= 0 {
            return api_error(400, "options.num_ctx must be positive");
        }
    }
    // R4 semantic cache (ollama lane, non-stream only). A hit skips model
    // admission entirely; embed failures bypass (never fail the request);
    // malformed override headers fail fast (400). Key-scoped keys whose
    // scope excludes the embed model bypass silently (the embed admission
    // would be rejected anyway — no point failing the chat for it).
    // Structured-output requests (grammar or any non-null format) bypass
    // in BOTH directions: a constraint narrows the valid response space,
    // so serving a cached unconstrained answer — or storing a constrained
    // one for later unconstrained reuse — violates the constraint
    // (live-pinned 11499: a malformed-grammar request was served a
    // semantic-cache 200 instead of the child's 400).
    let constrained = req
        .get("grammar")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|s| !s.is_empty())
        || req.get("format").is_some_and(|v| !v.is_null());
    let mut sem_ctx: Option<semcache::SemCtx> = None;
    if !req["stream"].as_bool().unwrap_or(true) && !constrained {
        let sc = &state.config.semantic_cache;
        match semcache::directive(
            sc.enabled,
            sc.model.is_some(),
            sc.ttl_secs,
            sc.threshold,
            |h| {
                headers
                    .get(h)
                    .and_then(|v| v.to_str().ok().map(str::to_string))
            },
        ) {
            Err(msg) => return api_error(400, &msg),
            Ok(Some(directive)) => {
                let embed_model = sc.model.clone().unwrap_or_default();
                let scope_ok = key_entry
                    .as_ref()
                    .is_none_or(|(_, e)| e.models.is_empty() || e.models.contains(&embed_model));
                if scope_ok {
                    let prompt_text = emb_prompt_text(&req);
                    match semcache::embed_prompt(&state, &embed_model, &prompt_text).await {
                        Ok(emb) => {
                            let key_name = key_entry.as_ref().map(|(n, _)| n.clone());
                            if let Some((id, sim, cached)) = state.semcache.lookup(
                                semcache::LANE_OLLAMA,
                                &row.name,
                                key_name.as_deref(),
                                &emb,
                                directive.threshold,
                            ) {
                                state
                                    .sem
                                    .hits
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                return semcache_hit_response(&row.name, &cached, id, sim);
                            }
                            state
                                .sem
                                .misses
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            sem_ctx = Some(semcache::SemCtx {
                                emb,
                                directive,
                                key: key_name,
                            });
                        }
                        Err(_) => {
                            state
                                .sem
                                .embed_failures
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                }
            }
            Ok(None) => {}
        }
    }

    let priority = Priority::from_header(
        headers
            .get("x-pallama-priority")
            .and_then(|v| v.to_str().ok()),
    );
    let keep_alive = parse_keep_alive(req.get("keep_alive"));

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
            return *resp;
        }
    }
    // Per-request spec mode (`options.spec`): same restart-once shape
    // as num_ctx, and router mode refuses it for the same reason — one
    // shared child has no per-model spawn shape to override.
    if let Some(mode) = req
        .pointer("/options/spec")
        .and_then(serde_json::Value::as_str)
    {
        if state.config.router {
            return api_error(
                400,
                &format!(
                    "router mode serves all models from one child; per-request spec is not available — set [model_overrides.{}] spec = \"{mode}\" in config.toml and restart the daemon",
                    row.name
                ),
            );
        }
        if let Err(resp) = apply_spec(&state, &row.name, mode).await {
            return *resp;
        }
    }

    // WorkClass: raw-lane classification follows the LANE the request
    // will actually take — image requests fall back to the child lane
    // even under ollama_compat, so they keep their interactive class
    // (and are exactly the mixed traffic that makes reserved capacity
    // reachable on a compat model).
    let class = crate::queue::classify_work(
        req.get("tools").is_some_and(serde_json::Value::is_array),
        state.config.effective_prompt_recipe(&model_field) == crate::prompt_recipe::OLLAMA_COMPAT
            && !crate::prompt_recipe::has_images(&req),
    );
    let (engine, load_ms) = match ensure_with_admission(
        &state,
        &model_field,
        priority,
        class,
        affinity_hash(&req),
        crate::proxy::body_needs_vision(&req, true),
    )
    .await
    {
        Ok(ok) => ok,
        Err(resp) => return *resp,
    };
    // F12: full keep_alive contract — >0 pins the instance for N s,
    // -1 pins "forever", 0 clears any prior pin (the evict-after-response
    // path below then owns teardown). Applied at admission so the pin
    // covers the generation itself.
    if let Some(secs) = keep_alive {
        state.sup.set_keep_alive(&engine.name, secs);
    }
    // Heat under the INSTANCE key (model or model#N): replica victim
    // choice and idle tracking are key-scoped (B1).
    state.sup.note_prefix_hit(&engine.name);

    let stream = req["stream"].as_bool().unwrap_or(true);
    let model_name = row.name.clone();
    let openai_bytes = serde_json::to_vec(&openai_req).unwrap_or_default();
    let state2 = state.clone();
    let engine2 = engine.clone();
    let accounting_name = engine.name.clone();

    // F29: accounting rides the response BODY (begin at admission,
    // release when the body drains or the client aborts) — not the
    // handler future, which resolves at headers-ready for streams and
    // can be dropped mid-flight on client disconnects.
    let guard = crate::proxy::begin_accounting(&state, &accounting_name);
    let mut out = crate::proxy::hold_body(
        guard,
        (async move {
            let enforce =
                state2.config.sentinel && sentinel::enforce_enabled(&state2.config, &headers);
            let resp = proxy_core_chat(
                &state2,
                &engine2,
                &model_name,
                openai_bytes,
                stream,
                load_ms,
                trace_ext.map(|Extension(t)| t.0),
                enforce,
                sem_ctx,
                OutputShape::Chat,
            )
            .await;
            // keep_alive=0: evict right after this response (complaint #12),
            // banking the KV checkpoint first. A live session pin wins (R3):
            // the client asked to keep the model for its session — the pin's
            // TTL (or an explicit close / force stop) ends the protection.
            if keep_alive == Some(0)
                && !state2
                    .sup
                    .sessions
                    .pins(
                        &model_name,
                        std::time::Duration::from_secs(state2.config.session_keep_secs),
                    )
                    .live
            {
                let _ = state2.sup.evict_model(&model_name).await;
            }
            resp
        })
        .await,
    );
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
        .or_else(|| v.as_str().and_then(parse_keep_alive_str))
}

/// Ollama duration strings: Go format (`"5m"`, `"1h30m"`, `"45s"`,
/// bare `"300"`). Malformed → `None` (request then follows the daemon's
/// default idle policy — same silent fallback as before, now reachable
/// only for genuinely unparseable input).
fn parse_keep_alive_str(s: &str) -> Option<i64> {
    let s = s.trim();
    if let Ok(n) = s.parse::<i64>() {
        return Some(n);
    }
    let mut total: i64 = 0;
    let mut rest = s;
    let mut matched = false;
    while !rest.is_empty() {
        let digits = rest
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(rest.len());
        if digits == 0 {
            return None; // unit without a number, or stray char
        }
        let (num, tail) = rest.split_at(digits);
        let n: i64 = num.parse().ok()?;
        let unit_len = tail.chars().next().map_or(0, char::len_utf8);
        let mult = match &tail[..unit_len] {
            "s" => 1,
            "m" => 60,
            "h" => 3_600,
            "d" => 86_400,
            "w" => 604_800,
            _ => return None,
        };
        total = total.saturating_add(n.saturating_mul(mult));
        rest = &tail[unit_len..];
        matched = true;
    }
    matched.then_some(total)
}

/// Text fed to the embed model for the semantic cache: role-tagged
/// message contents, so "write code: X" and "review code: X" embed
/// differently. System + user + assistant history all participate.
fn emb_prompt_text(req: &Value) -> String {
    let mut out = String::new();
    if let Some(msgs) = req["messages"].as_array() {
        for m in msgs {
            let role = m["role"].as_str().unwrap_or("user");
            if let Some(content) = m["content"].as_str() {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(role);
                out.push_str(": ");
                out.push_str(content);
            }
        }
    }
    out
}

/// Serve a semantic-cache hit: translate the stored `OpenAI` shape back to
/// ollama (same code path as a live response), attach `cache_debug` and
/// F36: `remote:` routing exists only on /api/chat — on every other
/// ollama lane the prefix would fall through to model resolution and
/// return a confusing 404. Teaching 400 instead.
fn refuse_remote_prefix(state: &AppState, model: &str) -> Option<Response> {
    if model
        .split_once(':')
        .is_some_and(|(name, _)| state.config.remotes.iter().any(|r| r.name == name))
    {
        return Some(api_error(
            400,
            "`remote:` routing is only supported on /api/chat; use the OpenAI-compatible /v1 lanes for other verbs on remote models",
        ));
    }
    None
}

/// Whether a GGUF chat template carries thinking machinery. Template
/// dialects vary (`enable_thinking` switch, `<think>` tags, `reasoning`
/// blocks); any marker counts. Same evidence class the engine itself
/// uses when deciding whether to emit reasoning content.
fn template_supports_thinking(template: &str) -> bool {
    template.contains("enable_thinking")
        || template.contains("<think>")
        || template.contains("reasoning")
}

/// Think-capability gate (ollama parity): `think: true` on a model whose
/// chat template provably lacks thinking markers is a teaching 400, not
/// a silent no-op. Fail-open on unreadable/absent templates — safetensors
/// lanes and legacy GGUFs without `tokenizer.chat_template` keep today's
/// behavior; refuse only on evidence.
/// Ollama thinking parity for the *absent* `think` key: qwen3-dialect
/// templates render thinking ON when the kwarg is missing, so an unbounded
/// reasoning trace eats the whole `num_predict` budget and the answer never
/// arrives (70k-char `reasoning_no_answer` spirals observed live). Ollama's
/// documented default for thinking-capable models is OFF unless `think` is
/// set — inject the explicit off-toggle when the template carries a thinking
/// switch. Evidence rule matches the refuse gate: no readable template =
/// no injection (fail-open). Explicit user `chat_template_kwargs` still win
/// — the translate layer fills only missing keys.
fn default_think_off(row: &pallama_core::ModelRow, req: &mut Value) {
    if !req.get("think").is_none_or(Value::is_null) {
        return;
    }
    if template_supports_thinking_cached(row) {
        req["think"] = Value::Bool(false);
    }
}

/// Per-path cache for `template_supports_thinking`, validated by file
/// length + mtime. The think gates run on every request that omits
/// `think`, and a full GGUF metadata parse walks the whole tokenizer
/// vocabulary — live-measured ~60ms on a 5.7 GiB model, the single
/// largest first-token latency component on the ollama lane (parity
/// probes: explicit `think:false` 74-80ms vs absent key 133-141ms).
/// Model files are immutable once pulled; the stat guard keeps the
/// cache honest if a path is ever replaced on disk.
static TEMPLATE_SUPPORT_CACHE: std::sync::OnceLock<
    dashmap::DashMap<String, (u64, std::time::SystemTime, bool)>,
> = std::sync::OnceLock::new();

fn template_supports_thinking_cached(row: &pallama_core::ModelRow) -> bool {
    let cache = TEMPLATE_SUPPORT_CACHE.get_or_init(dashmap::DashMap::new);
    let Ok(meta) = std::fs::metadata(&row.path) else {
        return false; // fail-open, same as an unreadable file below
    };
    let Some(mtime) = meta.modified().ok() else {
        return false;
    };
    if let Some(entry) = cache.get(&row.path) {
        let (len, seen_mtime, supports) = *entry;
        if len == meta.len() && seen_mtime == mtime {
            return supports;
        }
    }
    let template = pallama_core::read_metadata_file(std::path::Path::new(&row.path))
        .ok()
        .and_then(|gguf| gguf.chat_template)
        .unwrap_or_default();
    let supports = template_supports_thinking(&template);
    cache.insert(row.path.clone(), (meta.len(), mtime, supports));
    supports
}

fn refuse_unsupported_think(row: &pallama_core::ModelRow, req: &Value) -> Option<Response> {
    if req["think"].as_bool() != Some(true) {
        return None;
    }
    if template_supports_thinking_cached(row) {
        return None;
    }
    // Distinguish "read the template and it lacks markers" (refuse) from
    // "could not read" (fail-open): an empty template never reaches the
    // cache with `true`, so re-derive emptiness only on the slow path.
    let template = pallama_core::gguf::read_metadata_file(std::path::Path::new(&row.path))
        .ok()
        .and_then(|meta| meta.chat_template)
        .unwrap_or_default();
    if template.is_empty() {
        return None;
    }
    Some(api_error(
        400,
        &format!(
            "model {} does not support thinking: its chat template has no thinking markers (enable_thinking/<think>/reasoning); omit \"think\" or use a thinking-capable model",
            row.name
        ),
    ))
}

/// A child can answer HTTP 200 with an empty `choices` array and an
/// `error` object attached (live-proven: mistral.rs returns
/// `service_unavailable` this way when a prompt exceeds the
/// paged-attention batch step). Mapping that to an empty 200 hides the
/// failure from clients; the sentinel logs it but the caller still sees
/// "success". Surface it as a 502 instead.
fn child_error_body(v: &Value) -> Option<String> {
    let choices_empty = match v.get("choices").and_then(Value::as_array) {
        Some(a) => a.is_empty(),
        None => true,
    };
    if !choices_empty {
        return None;
    }
    v.get("error").map(|e| match e {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    })
}

/// hit headers. Eval counts come from the original generation.
fn semcache_hit_response(model: &str, cached: &Value, id: u64, sim: f32) -> Response {
    let mut ollama = tr::openai_chat_to_ollama(model, cached);
    // Entries cached before raw-think suppression may carry think
    // blocks in content; stripping again is idempotent for clean ones.
    tr::suppress_raw_think_response(&mut ollama);
    ollama["cache_debug"] = serde_json::json!({
        "cache_hit": true,
        "hit_type": "semantic",
        "similarity": sim,
        "cache_id": id,
    });
    let mut resp = axum::Json(ollama).into_response();
    let headers = resp.headers_mut();
    if let Ok(v) = HeaderValue::from_str(&format!("hit; similarity={sim:.3}")) {
        headers.insert(semcache::HDR_CACHE, v);
    }
    resp
}

/// Scale the f16 KV estimate by the effective `cache_type` quant the
/// spawn will actually use (mirrors the compiler's kv ladder), so the
/// preflight judges the buffer the child allocates — the 400's own
/// teaching (`cache_type = "q8_0" halves KV`) must not be a dead end
/// when the user applies it.
fn scale_kv_by_cache_type(kv_bytes: u64, cache_type: Option<&str>) -> u64 {
    match cache_type {
        Some("q8_0") => kv_bytes / 2,
        Some("q4_0") => kv_bytes / 4,
        Some("q4_1") => kv_bytes * 9 / 20,
        Some("q5_0") => kv_bytes * 11 / 32,
        Some("q5_1") => kv_bytes * 3 / 8,
        _ => kv_bytes,
    }
}

pub(crate) async fn apply_num_ctx(
    state: &Arc<AppState>,
    model: &str,
    want: i64,
) -> Result<(), Box<Response>> {
    if want <= 0 {
        return Err(Box::new(api_error(400, "options.num_ctx must be positive")));
    }
    // ctx preflight (I5): refuse a num_ctx no spawn could host, BEFORE
    // the evict below can take a healthy instance down. One placement
    // model for both lanes (device truth): the KV pool is device-backed
    // under `--kv-unified` too, so both judge empty-GPU VRAM through
    // the same shared 2b verdict. Conservative by design (no live-free
    // query exists across backends): anything that passes still gets
    // the engine's own fit juggling at spawn.
    {
        let row = state
            .with_store(|s| s.get_model(model).ok().flatten())
            .flatten();
        if let Some(row) = row {
            if let Ok(meta) = pallama_core::read_metadata_file(std::path::Path::new(&row.path)) {
                let total_vram = state.sup.hardware.total_vram_mib();
                if total_vram > 0 {
                    // KV must be judged at the quant the spawn will run.
                    // An EXPLICIT config/overlay cache_type is sovereign
                    // (single shot, mirroring the compiler's
                    // explicit-beats-ladder doctrine); an unpinned one
                    // LADDERS f16 -> q8_0 -> q4_0 exactly like the
                    // spawn compiler's kv_quant_ladder, so the preflight
                    // never refuses a pin the spawn itself would host
                    // (split-brain observed live: a 65536 vision pin
                    // refused at f16 math while the spawn laddered to
                    // q8_0 happily).
                    let effective = state.config.effective_cache_type(model);
                    let ladder: Vec<Option<&str>> = if effective.is_empty() {
                        vec![None, Some("q8_0"), Some("q4_0")]
                    } else {
                        vec![Some(effective)]
                    };
                    let kv_f16 = crate::preflight::kv_f16_mib(
                        &meta,
                        u64::try_from(want).unwrap_or(u64::MAX),
                    ) * 1024
                        * 1024;
                    let mut refuse: Option<String> = None;
                    for quant in ladder {
                        let kv_bytes = scale_kv_by_cache_type(kv_f16, quant);
                        match pallama_core::profile::unified_ctx_verdict(
                            u64::try_from(row.bytes.max(0)).unwrap_or(u64::MAX),
                            kv_bytes,
                            total_vram * 1024 * 1024,
                            u32::try_from(want).unwrap_or(u32::MAX),
                        ) {
                            pallama_core::profile::UnifiedCtxVerdict::Fit => {
                                refuse = None;
                                break;
                            }
                            pallama_core::profile::UnifiedCtxVerdict::Refuse(msg) => {
                                refuse = Some(msg);
                            }
                        }
                    }
                    if let Some(msg) = refuse {
                        return Err(Box::new(api_error(400, &msg)));
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

/// Per-request spec-mode override (`options.spec`, `X-Pallama-Spec`,
/// and the REPL's `--no-draft`): queue the mode for the model's NEXT
/// spawn; a live instance spawned under a DIFFERENT mode is recycled
/// first when idle (same restart-once shape as `apply_num_ctx`). Same
/// shape = no-op, so a REPL re-sending `spec: "off"` every turn never
/// churns a matching instance. Draft availability/manifest gates stay
/// in `profile::compile` — queueing `eagle3` without the draft pulled
/// fails at spawn exactly like config-pinned eagle3 does.
pub(crate) async fn apply_spec(
    state: &Arc<AppState>,
    model: &str,
    want: &str,
) -> Result<(), Box<Response>> {
    if !pallama_core::is_valid_spec_mode(want) {
        return Err(Box::new(api_error(
            400,
            &format!(
                "options.spec must be one of \"off\", \"auto\", \"ngram\", \"ngram-map-k\", \
                 \"ngram-map-k4v\", \"ngram-mod\", \"ngram-cache\", \"mtp\", \"eagle3\", \
                 \"dflash\" or \"dspark\", got {want:?} (see `spec` in config.toml)"
            ),
        )));
    }
    if let Some(p) = state.sup.ps().into_iter().find(|p| p.name == model) {
        if p.spec_mode != want && p.in_flight == 0 {
            let _ = state.sup.evict_model(model).await;
            state.sup.set_next_spec(model, want);
        }
    } else {
        state.sup.set_next_spec(model, want);
    }
    Ok(())
}

/// Output wire-shape for the unified chat-bus pipeline: /api/chat emits
/// `message` objects, /api/generate emits `response` objects. Everything
/// else (sentinel, TTFT/TPOT, admission, usage accounting) is shared.
#[derive(Clone, Copy, PartialEq, Eq)]
enum OutputShape {
    Chat,
    Generate,
}

impl OutputShape {
    fn lane(self) -> &'static str {
        match self {
            Self::Chat => "ollama-chat",
            Self::Generate => "ollama-generate",
        }
    }
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
    sem: Option<semcache::SemCtx>,
    shape: OutputShape,
) -> Response {
    let base = child_base(&engine.endpoint);
    // Pre-render recipe lane: the gateway owns the prompt (chatml wrap
    // + JSON tool-call grammar) and drives the child through the raw
    // completion lane, adapting the response back to chat shape at the
    // two translation seams below. Image requests always ride the
    // child's native multimodal template lane.
    let ollama_compat = {
        let parsed: Option<serde_json::Value> = serde_json::from_slice(&openai_body).ok();
        state.config.effective_prompt_recipe(model) == crate::prompt_recipe::OLLAMA_COMPAT
            && parsed
                .as_ref()
                .is_some_and(|v| !crate::prompt_recipe::has_images(v))
    };
    let openai_body: Vec<u8> = if ollama_compat {
        let parsed = serde_json::from_slice::<serde_json::Value>(&openai_body)
            .unwrap_or(serde_json::Value::Null);
        let strict = state.config.effective_decode_policy(model) == "strict";
        serde_json::to_vec(&crate::prompt_recipe::to_completion_request(
            &parsed,
            stream,
            state.config.effective_raw_lane_max_tokens(&engine.name),
            strict,
        ))
        .unwrap_or_default()
    } else {
        openai_body
    };
    let url = if ollama_compat {
        format!("{base}/v1/completions")
    } else {
        format!("{base}/v1/chat/completions")
    };
    let req = child_auth(
        state
            .http
            .post(&url)
            .header("content-type", "application/json"),
        engine,
    );
    if !stream {
        // We translate the non-stream response into ollama shape.
        let t0 = std::time::Instant::now();
        let ttft_secs;
        let resp =
            match crate::proxy::child_send(state, engine, req.body(openai_body.clone()).send())
                .await
            {
                Ok(r) => {
                    let s = t0.elapsed().as_secs_f64();
                    state.ttft.observe_secs(s);
                    ttft_secs = Some(s);
                    r
                }
                Err(e) => {
                    tracing::warn!(model, "nonstream upstream failed: {e}");
                    // Reap now so the NEXT request respawns instead of
                    // 502-looping until the periodic reaper notices (~10s).
                    state.sup.reap_dead_children().await;
                    return api_error(e.status_u16(), &format!("engine request failed: {e}"));
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
        // Raw-completion lane: adapt the text completion into the chat
        // shape before every downstream concern (cache observability,
        // enforce, translation) — they see a native chat response.
        let openai = if ollama_compat {
            crate::prompt_recipe::raw_json_to_chat(&openai)
        } else {
            openai
        };
        // Cache observability (R6): classify this completed response
        // warm/cold from its usage object before translation.
        match openai.get("usage") {
            Some(u) => state.obs.record(
                model,
                u.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0),
                tr::cached_prompt_tokens(openai.get("usage")),
                ttft_secs,
            ),
            None => state.obs.miss(),
        }
        // Non-stream + enforce: judge before translation (streaming stays
        // warn-only — bytes are already on the wire).
        if enforce {
            let (ctx, _) =
                sentinel::request_ctx(state, shape.lane(), model, &openai_body, trace, false);
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
                shape.lane(),
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
        if let Some(msg) = child_error_body(&openai) {
            return api_error(502, &format!("engine error: {msg}"));
        }
        let mut ollama = match shape {
            OutputShape::Chat => tr::openai_chat_to_ollama(model, &openai),
            OutputShape::Generate => tr::openai_chat_to_generate(model, &openai),
        };
        // Templates that ignore the thinking-off kwargs still emit raw
        // <think> blocks; hide them when the request did not ask for
        // thinking (already-split content is unaffected — idempotent).
        if !tr::request_think_on(&openai_body) {
            tr::suppress_raw_think_response(&mut ollama);
        }
        // Cold-load wall (only when a spawn actually happened) for ollama
        // parity: clients read load_duration after first requests.
        if load_ms > 100 {
            ollama["load_duration"] = json!(u64::try_from(load_ms).unwrap_or(u64::MAX) * 1_000_000);
        }
        // R4: file the response for future semantic hits (miss path from
        // chat()). Decorates with cache_debug + miss header so clients can
        // distinguish "computed now" from "served from cache".
        if let Some(sem) = sem {
            let id = state.semcache.store(
                semcache::LANE_OLLAMA,
                model,
                sem.key.as_deref(),
                sem.emb,
                openai,
                sem.directive.ttl,
                state.config.semantic_cache.max_entries,
            );
            state
                .sem
                .stores
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            ollama["cache_debug"] = serde_json::json!({"cache_hit": false, "cache_id": id});
            let mut resp = axum::Json(ollama).into_response();
            resp.headers_mut()
                .insert(semcache::HDR_CACHE, HeaderValue::from_static("miss"));
            return resp;
        }
        return axum::Json(ollama).into_response();
    }
    // Stream: child SSE -> ollama NDJSON with final counts. Force the
    // usage chunk so the final ollama line carries eval counts.
    let mut body_with_usage: Value = serde_json::from_slice(&openai_body).unwrap_or_default();
    body_with_usage["stream"] = json!(true);
    body_with_usage["stream_options"] = json!({"include_usage": true});
    let openai_body = serde_json::to_vec(&body_with_usage).unwrap_or_default();
    let resp = match crate::proxy::child_send(
        state,
        engine,
        child_auth(
            state
                .http
                .post(&url)
                .header("content-type", "application/json"),
            engine,
        )
        .body(openai_body.clone())
        .send(),
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            // Same crash-recovery contract as the OpenAI proxy path.
            state.sup.reap_dead_children().await;
            return api_error(e.status_u16(), &format!("engine request failed: {e}"));
        }
    };
    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        return api_error(status, &text);
    }
    let upstream: std::pin::Pin<
        Box<dyn futures::Stream<Item = std::io::Result<axum::body::Bytes>> + Send>,
    > = if ollama_compat {
        Box::pin(crate::prompt_recipe::raw_sse_to_chat_sse(
            resp.bytes_stream()
                .map(|chunk| chunk.map_err(|e| std::io::Error::other(e.to_string()))),
        ))
    } else {
        Box::pin(
            resp.bytes_stream()
                .map(|chunk| chunk.map_err(|e| std::io::Error::other(e.to_string()))),
        )
    };
    let model_c = model.to_string();
    let load_hdr = load_ms > 100;
    // Sentinel: same side-channel tap as the OpenAI path — bytes cloned
    // in the observation closure, analyzer parses off the hot path.
    let (sentinel_feed, _) = sentinel::begin_chat_observation(
        state,
        shape.lane(),
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
    // Shared clock: the map closure stamps first/last-byte offsets, the
    // unfold closure reads them for the final NDJSON line (streaming
    // children report no timings — durations are gateway-measured).
    let clock = std::sync::Arc::new(std::sync::Mutex::new((None::<u64>, 0u64)));
    let clock_map = std::sync::Arc::clone(&clock);
    let stream = futures::StreamExt::map(upstream, move |chunk| {
        let now = std::time::Instant::now();
        if first {
            hist_state.ttft.observe_secs((now - t0).as_secs_f64());
            first = false;
        } else {
            hist_state.tpot.observe_secs((now - last).as_secs_f64());
        }
        last = now;
        {
            let mut c = clock_map.lock().unwrap();
            let elapsed = u64::try_from((now - t0).as_nanos()).unwrap_or(u64::MAX);
            if c.0.is_none() {
                c.0 = Some(elapsed);
            }
            c.1 = elapsed;
        }
        if let Ok(bytes) = chunk.as_ref() {
            sentinel_feed.bytes(bytes.as_ref());
        }
        chunk.map_err(|e| std::io::Error::other(e.to_string()))
    });
    // Translate SSE -> NDJSON incrementally. F70: bytes accumulate in a
    // boundary-safe LineBuffer — a multi-byte UTF-8 char split across
    // chunks never decodes to twin U+FFFD.
    let buf = String::new();
    let lines = tr::LineBuffer::new();
    let model_owned = model_c.clone();
    let ndjson = futures::stream::unfold(
        (
            stream,
            buf,
            lines,
            model_owned,
            false,
            None::<Value>,
            None::<String>,
            false,
            std::sync::Arc::clone(&clock),
            std::sync::Arc::clone(&state.obs),
            shape,
            tr::ToolCallAccum::default(),
            (!tr::request_think_on(&openai_body)).then(tr::ThinkSplitter::new),
        ),
        |(
            mut stream,
            mut buf,
            mut lines,
            model,
            mut done,
            mut usage,
            mut finish,
            mut usage_sent,
            clock,
            obs,
            shape,
            mut tool_accum,
            mut think_split,
        )| async move {
            loop {
                if done && !usage_sent {
                    // Decode window = last - first byte; total = full wall.
                    let (eval_ns, total_ns) = {
                        let c = clock.lock().unwrap();
                        match c.0 {
                            Some(f) => (Some(c.1.saturating_sub(f)), Some(c.1)),
                            None => (None, (c.1 > 0).then_some(c.1)),
                        }
                    };
                    // Cache observability (R6): classify exactly once at
                    // completion — first-byte time is the TTFT evidence.
                    let ttft_secs = {
                        let c = clock.lock().unwrap();
                        c.0.map(|ns| std::time::Duration::from_nanos(ns).as_secs_f64())
                    };
                    match usage.as_ref() {
                        Some(u) => obs.record(
                            &model,
                            u.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0),
                            tr::cached_prompt_tokens(usage.as_ref()),
                            ttft_secs,
                        ),
                        None => obs.miss(),
                    }
                    let final_chunk = match shape {
                        OutputShape::Chat => tr::ollama_final_chunk(
                            &model,
                            usage.as_ref(),
                            finish.as_deref(),
                            eval_ns,
                            total_ns,
                        ),
                        OutputShape::Generate => tr::ollama_generate_final_chunk(
                            &model,
                            usage.as_ref(),
                            finish.as_deref(),
                            eval_ns,
                            total_ns,
                        ),
                    };
                    usage_sent = true;
                    // A stream that ended mid-call (no post-fragment chunk)
                    // still owes the client its merged tool_calls line.
                    let flush_prefix = tool_accum.flush_line(&model).unwrap_or_default();
                    // The think suppressor may still hold a literal tail
                    // (an unterminated marker prefix is plain text); ship
                    // it as one last content line before the final chunk.
                    let think_tail = match think_split.as_mut() {
                        Some(splitter) => {
                            let tail = splitter.finish();
                            if tail.is_empty() {
                                String::new()
                            } else {
                                let ev = json!({"choices": [{"delta": {"content": tail}}]});
                                let lines = match shape {
                                    OutputShape::Chat => {
                                        tr::openai_chunk_to_ollama(&mut tool_accum, &model, &ev)
                                    }
                                    OutputShape::Generate => {
                                        tr::openai_chunk_to_generate(&model, &ev)
                                    }
                                };
                                let mut rendered = String::new();
                                for line in lines {
                                    rendered.push_str(&line.to_string());
                                    rendered.push('\n');
                                }
                                rendered
                            }
                        }
                        None => String::new(),
                    };
                    return Some((
                        Ok(Bytes::from(format!(
                            "{flush_prefix}{think_tail}{final_chunk}\n"
                        ))),
                        (
                            stream,
                            buf,
                            lines,
                            model,
                            done,
                            usage,
                            finish,
                            usage_sent,
                            clock,
                            obs,
                            shape,
                            tool_accum,
                            think_split,
                        ),
                    ));
                }
                match futures::StreamExt::next(&mut stream).await {
                    Some(Ok(bytes)) => {
                        buf.push_str(&lines.feed(&bytes));
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
                        // Raw <think> suppression: rewrite the content
                        // delta in place; an emptied delta renders no
                        // ollama line (both translators skip empty
                        // content).
                        let mut events = events;
                        if let Some(splitter) = think_split.as_mut() {
                            for ev in &mut events {
                                let Some(delta) = ev
                                    .get_mut("choices")
                                    .and_then(Value::as_array_mut)
                                    .and_then(|c| c.first_mut())
                                    .and_then(|c| c.get_mut("delta"))
                                    .and_then(|d| d.get_mut("content"))
                                else {
                                    continue;
                                };
                                if let Value::String(content) = delta {
                                    *content = splitter.feed(content.as_str());
                                }
                            }
                        }
                        let ndjson_lines: Vec<String> = events
                            .iter()
                            .flat_map(|ev| match shape {
                                OutputShape::Chat => {
                                    tr::openai_chunk_to_ollama(&mut tool_accum, &model, ev)
                                }
                                OutputShape::Generate => tr::openai_chunk_to_generate(&model, ev),
                            })
                            .map(|v| format!("{v}\n"))
                            .collect();
                        if ndjson_lines.is_empty() {
                            continue; // need more data
                        }
                        let body = ndjson_lines.join("");
                        return Some((
                            Ok(Bytes::from(body)),
                            (
                                stream,
                                buf,
                                lines,
                                model,
                                done,
                                usage,
                                finish,
                                usage_sent,
                                clock,
                                obs,
                                shape,
                                tool_accum,
                                think_split,
                            ),
                        ));
                    }
                    Some(Err(e)) => {
                        // Mid-body upstream failure (wedged child evicted
                        // by the sentinel stall, conn churn, respawn
                        // race): once headers are committed the only
                        // legal close is a clean one. Surface the
                        // truncation as a semantic terminal error line,
                        // then let the done lane finish the stream —
                        // an Err item here aborts the HTTP body and the
                        // client eats a RemoteProtocolError instead.
                        let line = serde_json::json!({
                            "model": model,
                            "error": {
                                "code": "stream_truncated",
                                "message": e.to_string(),
                            },
                        });
                        return Some((
                            Ok::<axum::body::Bytes, std::io::Error>(axum::body::Bytes::from(
                                line.to_string(),
                            )),
                            (
                                stream,
                                buf,
                                lines,
                                model,
                                true, // done: next poll emits the final chunk
                                usage,
                                finish,
                                usage_sent,
                                clock,
                                obs,
                                shape,
                                tool_accum,
                                think_split,
                            ),
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
    let mut openai_req = match tr::embeddings_to_openai(&req) {
        Ok(v) => v,
        Err(e) => return api_error(400, &e),
    };
    let model = openai_req["model"].as_str().unwrap_or_default().to_string();
    // F14: resolve first, scope-check the RESOLVED name — alias/quant-tag
    // parity with /api/chat.
    if let Some(resp) = refuse_remote_prefix(&state, &model) {
        return resp;
    }
    let row = match state.with_store(|s| resolve_model(s, &model)) {
        Some(Ok(r)) => r,
        Some(Err(e)) => return api_error(404, &e),
        None => return api_error(500, "store unavailable"),
    };
    // Scope + request count (plain embeds carry no token usage, so budgets
    // apply on request counts; late-chunking embeds additionally charge
    // their exact token count below).
    if let Some(Extension(k)) = &key_ext {
        if let Some(entry) = state.keys.entry(&k.name) {
            if let Err(rej) = state.keys.check(&entry, &row.name) {
                return rej.to_response();
            }
            state.keys.charge_request(&k.name);
        }
    }
    let (engine, _) = match ensure_with_admission(
        &state,
        &model,
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
    crate::proxy::hold_body(
        crate::proxy::begin_accounting(&state, &engine.name),
        (async {
            // R1 late chunking: gateway-terminated embed for opted-in models.
            if state.config.effective_late_chunking(&engine.name) {
                let prompt = req["prompt"].as_str().unwrap_or_default().to_string();
                return match crate::latechunk::late_embed(&state, &engine, &model, &[prompt]).await
                {
                    Ok(out) => {
                        if let Some(Extension(k)) = &key_ext {
                            state.keys.charge_tokens(&k.name, out.total_tokens);
                        }
                        axum::Json(json!({
                            "model": model,
                            "embedding": out.embeddings.into_iter().next().unwrap_or_default(),
                            "prompt_eval_count": out.total_tokens,
                            "late_chunking": true,
                        }))
                        .into_response()
                    }
                    Err((code, msg)) => api_error(code, &msg),
                };
            }
            let url = format!("{}/v1/embeddings", child_base(&engine.endpoint));
            // mistral.rs children register models as `default` (see proxy.rs).
            if crate::proxy::child_model_default(&engine) {
                crate::proxy::set_child_model_default(&mut openai_req);
            }
            let resp = match crate::proxy::child_send(
                &state,
                &engine,
                child_auth(state.http.post(&url), &engine)
                    .json(&openai_req)
                    .send(),
            )
            .await
            {
                Ok(r) => r,
                Err(e) => return api_error(e.status_u16(), &format!("engine request failed: {e}")),
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
        .await,
    )
}

/// POST /api/embed — ollama new-style embeddings (input: str | [str]);
/// same engine lane as /api/embeddings, array-friendly shape.
// ollama-embed batch fan-out: one request shape, one handler. Splitting the
// batch loop out is queued with the parallel gateway wave.
#[allow(clippy::too_many_lines)]
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
    // F14: resolve first, scope-check the RESOLVED name (chat parity).
    if let Some(resp) = refuse_remote_prefix(&state, &model) {
        return resp;
    }
    let row = match state.with_store(|s| resolve_model(s, &model)) {
        Some(Ok(r)) => r,
        Some(Err(e)) => return api_error(404, &e),
        None => return api_error(500, "store unavailable"),
    };
    if let Some(Extension(k)) = &key_ext {
        if let Some(entry) = state.keys.entry(&k.name) {
            if let Err(rej) = state.keys.check(&entry, &row.name) {
                return rej.to_response();
            }
            state.keys.charge_request(&k.name);
        }
    }
    let (engine, _) = match ensure_with_admission(
        &state,
        &model,
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
    crate::proxy::hold_body(
        crate::proxy::begin_accounting(&state, &engine.name),
        (async {
            // R1 late chunking: embed the joined document once (per-token
            // matrix from the pooling=none child), mean-pool per chunk span.
            if state.config.effective_late_chunking(&engine.name) {
                let strings: Option<Vec<String>> = inputs
                    .iter()
                    .map(|v| v.as_str().map(str::to_string))
                    .collect();
                let Some(strings) = strings else {
                    return api_error(
                        400,
                        "late chunking: \"input\" array must contain only strings",
                    );
                };
                return match crate::latechunk::late_embed(&state, &engine, &model, &strings).await {
                    Ok(out) => {
                        if let Some(Extension(k)) = &key_ext {
                            state.keys.charge_tokens(&k.name, out.total_tokens);
                        }
                        axum::Json(json!({
                            "model": model,
                            "embeddings": out.embeddings,
                            "prompt_eval_count": out.total_tokens,
                            "total_tokens": out.total_tokens,
                            "late_chunking": true,
                        }))
                        .into_response()
                    }
                    Err((code, msg)) => api_error(code, &msg),
                };
            }
            let url = format!("{}/v1/embeddings", child_base(&engine.endpoint));
            let mut openai_req = json!({"model": model, "input": inputs});
            // mistral.rs children register models as `default` (see proxy.rs).
            if crate::proxy::child_model_default(&engine) {
                crate::proxy::set_child_model_default(&mut openai_req);
            }
            let resp = match crate::proxy::child_send(
                &state,
                &engine,
                child_auth(state.http.post(&url), &engine)
                    .json(&openai_req)
                    .send(),
            )
            .await
            {
                Ok(r) => r,
                Err(e) => return api_error(e.status_u16(), &format!("engine request failed: {e}")),
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
        .await,
    )
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
    // F14: resolve first, scope-check the RESOLVED name (chat parity).
    if let Some(resp) = refuse_remote_prefix(&state, &model) {
        return resp;
    }
    let row = match state.with_store(|s| resolve_model(s, &model)) {
        Some(Ok(r)) => r,
        Some(Err(e)) => return api_error(404, &e),
        None => return api_error(500, "store unavailable"),
    };
    if let Some(Extension(k)) = &key_ext {
        if let Some(entry) = state.keys.entry(&k.name) {
            if let Err(rej) = state.keys.check(&entry, &row.name) {
                return rej.to_response();
            }
            state.keys.charge_request(&k.name);
        }
    }
    let (engine, _) = match ensure_with_admission(
        &state,
        &model,
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
    let mut forward = forward;
    crate::proxy::hold_body(
        crate::proxy::begin_accounting(&state, &engine.name),
        (async {
            let url = format!("{}/v1/rerank", child_base(&engine.endpoint));
            // mistral.rs children register models as `default` (see
            // proxy.rs) — F28: rerank lane now rewrites like every other.
            if crate::proxy::child_model_default(&engine) {
                crate::proxy::set_child_model_default(&mut forward);
            }
            let resp = match crate::proxy::child_send(
                &state,
                &engine,
                child_auth(state.http.post(&url), &engine)
                    .json(&forward)
                    .send(),
            )
            .await
            {
                Ok(r) => r,
                Err(e) => return api_error(e.status_u16(), &format!("engine request failed: {e}")),
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
        .await,
    )
}

/// POST /api/generate — chat-bus translation (system/prompt/images/think
/// map onto /v1/chat/completions exactly like ollama's templated
/// generate); template/suffix stay rejected (400 + pointer).
#[allow(clippy::too_many_lines)] // one cohesive translation path
pub async fn generate(
    State(state): State<Arc<AppState>>,
    trace_ext: Option<Extension<TraceId>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let mut req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return api_error(400, &format!("invalid JSON: {e}")),
    };
    // ollama unload idiom: no/empty prompt + keep_alive 0 = release
    // ping — must short-circuit before the translator's required-field
    // 400 (real ollama is an idempotent 200).
    if parse_keep_alive(req.get("keep_alive")) == Some(0)
        && req
            .get("prompt")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
    {
        let ping_model = req["model"].as_str().unwrap_or_default().to_string();
        return unload_ping(&state, &ping_model, false).await;
    }
    let model = req["model"].as_str().unwrap_or_default().to_string();
    let key_entry = key_ext
        .as_ref()
        .map(|Extension(k)| k.name.clone())
        .and_then(|name| state.keys.entry(&name).map(|e| (name, e)));
    // F14: resolve FIRST, scope-check the RESOLVED name — alias and
    // quant-tag requests behave identically to /api/chat.
    if let Some(resp) = refuse_remote_prefix(&state, &model) {
        return resp;
    }
    let row = match state.with_store(|s| resolve_model(s, &model)) {
        Some(Ok(r)) => r,
        Some(Err(e)) => return api_error(404, &e),
        None => return api_error(500, "store unavailable"),
    };
    if let Some((name, entry)) = &key_entry {
        if let Err(rej) = state.keys.check(entry, &row.name) {
            return rej.to_response();
        }
        state.keys.charge_request(name);
    }
    // Think-capability gate (ollama parity), same as /api/chat. Both
    // think gates run BEFORE translation — the translator maps `think`
    // to engine kwargs, so a post-translate mutation would never reach
    // the child (live-caught on /api/chat; same ordering here).
    if let Some(resp) = refuse_unsupported_think(&row, &req) {
        return resp;
    }
    // Ollama parity for the absent toggle (local lane), same as /api/chat.
    default_think_off(&row, &mut req);
    let mut openai_req = match tr::generate_to_openai(&req) {
        Ok(Some(v)) => v,
        Ok(None) => {
            return api_error(
                400,
                "template/suffix in /api/generate are not supported; the engine applies the model's own template (use /api/chat for full message control)",
            );
        }
        Err(msg) => return api_error(400, &msg),
    };
    let priority = Priority::from_header(
        headers
            .get("x-pallama-priority")
            .and_then(|v| v.to_str().ok()),
    );
    let keep_alive = parse_keep_alive(req.get("keep_alive"));
    // F13: prompt-fit preflight (num_ctx request override counts) — the
    // same pre-hoc contract /api/chat enforces.
    {
        let eff = req
            .pointer("/options/num_ctx")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(u64::from(state.config.effective_ctx(&row.name)));
        if let Err(resp) = crate::preflight::enforce_prompt_fits(
            &state,
            &row.name,
            &req,
            u32::try_from(eff).unwrap_or(u32::MAX),
        )
        .await
        {
            return *resp;
        }
    }
    // F13: honor options.num_ctx — restart once at the requested size;
    // router mode cannot (one shared child), fail fast with the overlay
    // fix instead of silently ignoring (complaint #13, chat parity).
    if let Some(want) = req
        .pointer("/options/num_ctx")
        .and_then(serde_json::Value::as_u64)
        .map(|v| u32::try_from(v.min(u64::from(u32::MAX))).unwrap_or(u32::MAX))
    {
        if state.config.router {
            return api_error(
                400,
                &format!(
                    "router mode serves all models from one child; per-request num_ctx is not available — set [model_overrides.{}] ctx = {} in config.toml and restart the daemon",
                    row.name, want
                ),
            );
        }
        if let Err(resp) = apply_num_ctx(&state, &row.name, i64::from(want)).await {
            return *resp;
        }
    }
    // options.spec parity on /api/generate (chat above explains the
    // router refusal).
    if let Some(mode) = req
        .pointer("/options/spec")
        .and_then(serde_json::Value::as_str)
    {
        if state.config.router {
            return api_error(
                400,
                &format!(
                    "router mode serves all models from one child; per-request spec is not available — set [model_overrides.{}] spec = \"{mode}\" in config.toml and restart the daemon",
                    row.name
                ),
            );
        }
        if let Err(resp) = apply_spec(&state, &row.name, mode).await {
            return *resp;
        }
    }
    let class = crate::queue::classify_work(
        req.get("tools").is_some_and(serde_json::Value::is_array),
        state.config.effective_prompt_recipe(&model) == crate::prompt_recipe::OLLAMA_COMPAT,
    );
    let (engine, load_ms) = match ensure_with_admission(
        &state,
        &model,
        priority,
        class,
        affinity_hash(&req),
        crate::proxy::body_needs_vision(&req, true),
    )
    .await
    {
        Ok(ok) => ok,
        Err(resp) => return *resp,
    };
    // F12: keep_alive parity with /api/chat — pin window at admission,
    // explicit evict after the response on 0 (session pins still win).
    if let Some(secs) = keep_alive {
        state.sup.set_keep_alive(&engine.name, secs);
    }
    state.sup.note_prefix_hit(&engine.name);
    let model_name = row.name.clone();
    // mistral.rs children register models as `default` (see proxy.rs).
    if crate::proxy::child_model_default(&engine) {
        crate::proxy::set_child_model_default(&mut openai_req);
    }
    let openai_bytes = serde_json::to_vec(&openai_req).unwrap_or_default();
    // ollama defaults stream=true on generate; the chat bus mirrors it.
    let stream = req["stream"].as_bool().unwrap_or(true);
    let state_ej = state.clone();
    let mut out = crate::proxy::hold_body(
        crate::proxy::begin_accounting(&state, &engine.name),
        (async move {
            let enforce =
                state_ej.config.sentinel && sentinel::enforce_enabled(&state_ej.config, &headers);
            let resp = proxy_core_chat(
                &state_ej,
                &engine,
                &model_name,
                openai_bytes,
                stream,
                load_ms,
                trace_ext.map(|Extension(t)| t.0),
                enforce,
                None, // semantic cache is chat-lane only (response-shape keyed)
                OutputShape::Generate,
            )
            .await;
            // keep_alive=0: evict right after this response (complaint
            // #12), banking the KV checkpoint first. A live session pin
            // wins (R3) — identical to the /api/chat lane.
            if keep_alive == Some(0)
                && !state_ej
                    .sup
                    .sessions
                    .pins(
                        &model_name,
                        std::time::Duration::from_secs(state_ej.config.session_keep_secs),
                    )
                    .live
            {
                let _ = state_ej.sup.evict_model(&model_name).await;
            }
            resp
        })
        .await,
    );
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
        let engine = match crate::proxy::ensure_router_detached(&state.sup).await {
            Ok(e) => e,
            Err(e) => return api_error(503, &e.to_string()),
        };
        let url = format!("{}/models/unload", child_base(&engine.endpoint));
        return match child_auth(state.http.post(&url), &engine)
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
#[allow(clippy::too_many_lines)] // save/restore/erase/list + identity pre-flight; precedent: chat()
pub async fn session(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let v: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return api_error(400, &format!("invalid JSON: {e}")),
    };
    let Some(action) = v["action"].as_str() else {
        return api_error(400, "missing 'action' (save | restore | erase | close)");
    };
    if !matches!(action, "save" | "restore" | "erase" | "close") {
        return api_error(400, "action must be save, restore, erase or close");
    }
    // Slot KV checkpoints ride llama-server's --slot-save-path + POST
    // /slots/{id}; non-llamacpp children (mistralrs, sglang) have no slot
    // surface. Keyed on the ROUTED lane for this model, not the global
    // active row — auto-routing serves GGUF models on a llamacpp child
    // regardless of which engine is globally active. `close` stays open —
    // it only releases a gateway-side session pin.
    let engine_kind = v["model"]
        .as_str()
        .and_then(|m| crate::proxy::routed_kind_for(&state, m))
        .or_else(|| {
            state
                .with_store(|s| s.active_engine().ok().flatten())
                .flatten()
                .map(|e| e.kind)
        });
    if let Some(kind) = engine_kind {
        if action != "close" && kind != pallama_core::engine_kind::EngineKind::LlamaCpp {
            return api_error(
                400,
                &format!(
                    "slot KV checkpoints are llama-server-only; the {kind} lane that \
                     serves this request does not implement /slots — switch with \
                     `pallama engine use <tag>` or a model_overrides engine pin",
                ),
            );
        }
    }
    // Close releases a session PIN (R3) — no model, slot or checkpoint
    // involved; safe to run while children are asleep or absent.
    if action == "close" {
        let Some(session) = v["session"].as_str() else {
            return api_error(
                400,
                "missing 'session' (the x-pallama-session name to close)",
            );
        };
        if !valid_session_name(session) {
            return api_error(
                400,
                "session must be [A-Za-z0-9._-], not start with '.', max 128 chars",
            );
        }
        let released = state.sup.sessions.release(session);
        return axum::Json(json!({
            "status": "ok", "session": session, "released": released
        }))
        .into_response();
    }
    let Some(model) = v["model"].as_str() else {
        return api_error(400, "missing 'model'");
    };
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
    // F21: a slot beyond u32 range is a client bug (often a typo'd
    // 4_294_967_296) — silently clamping to slot 0 would clobber the
    // wrong slot's checkpoint.
    let Ok(slot) = u32::try_from(slot) else {
        return api_error(400, "slot out of range (max u32)");
    };

    // Erase is a FILE deletion, owned by the daemon (it owns the sessions
    // dir). Upstream's slot-erase action clears live KV state, not the
    // checkpoint file — forwarding would report success and delete nothing.
    if action == "erase" {
        let path = state
            .dirs
            .sessions_dir()
            .join(pallama_core::profile::path_safe(model))
            .join(filename);
        // Sibling identity manifest dies with the checkpoint (H19: no
        // orphaned metadata).
        let _ = std::fs::remove_file(pallama_core::session_identity::manifest_path(&path));
        return match std::fs::remove_file(&path) {
            Ok(()) => axum::Json(json!({"status": "ok", "filename": filename})).into_response(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                api_error(404, &format!("no such checkpoint: {filename}"))
            }
            Err(e) => api_error(500, &format!("delete {filename}: {e}")),
        };
    }

    let (engine, _load_ms) = match ensure_with_admission(
        &state,
        model,
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
    // #20 identity: the LIVE instance ctx wins over config (tuned or
    // overridden instances — same precedence as the prompt-fit gate).
    let live_ctx = state
        .sup
        .ps()
        .into_iter()
        .find(|p| p.name == engine.name)
        .map_or_else(|| state.config.effective_ctx(&engine.name), |p| p.ctx);
    let ckpt_dir = state
        .dirs
        .sessions_dir()
        .join(pallama_core::profile::path_safe(model));
    // RESTORE pre-flight: a checkpoint from a different runtime shape is
    // silent garbage, not a warm start. Missing manifest = unverifiable
    // (pre-#20 checkpoint): warn and let the caller decide to proceed.
    if action == "restore" {
        let ckpt = ckpt_dir.join(filename);
        match pallama_core::session_identity::read_manifest(&ckpt) {
            Some(saved) => {
                let current =
                    pallama_core::session_identity::build(&state.dirs, &state.config, &engine.name)
                        .map(|mut id| {
                            id.ctx = live_ctx;
                            id
                        });
                if let Some(cur) = current {
                    let diffs = pallama_core::session_identity::verify(&saved, &cur);
                    if !diffs.is_empty() {
                        return api_error(
                            400,
                            &format!(
                                "checkpoint {filename} was saved under a different runtime \
                                 shape; restoring it would inject stale KV. Diff: {}. \
                                 Re-save the session under the current shape first",
                                diffs.join("; ")
                            ),
                        );
                    }
                }
            }
            None => {
                tracing::warn!(
                    target: "pallama::session",
                    model = %engine.name,
                    "restoring checkpoint {filename} with no identity manifest (pre-#20?) — unverified"
                );
            }
        }
    }
    crate::proxy::hold_body(
        crate::proxy::begin_accounting(&state, &engine.name),
        (async {
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
        let resp = child_auth(state.http.post(&url), &engine)
            .json(&body_json)
            .send()
            .await;
        match resp {
            Ok(r) => {
                let status =
                    StatusCode::from_u16(r.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
                let body = r.text().await.unwrap_or_default();
                // SAVE post-success: stamp the identity manifest beside
                // the checkpoint so future restores can verify shape.
                if status.is_success() && action == "save" {
                    let ckpt = ckpt_dir.join(filename);
                    match pallama_core::session_identity::build(
                        &state.dirs,
                        &state.config,
                        &engine.name,
                    )
                    .map(|mut id| {
                        id.ctx = live_ctx;
                        id
                    }) {
                        Some(id) => {
                            if let Err(e) =
                                pallama_core::session_identity::write_manifest(&ckpt, &id)
                            {
                                tracing::warn!(
                                    target: "pallama::session",
                                    "identity manifest write failed: {e}"
                                );
                            }
                        }
                        None => {
                            tracing::warn!(
                                target: "pallama::session",
                                "could not determine runtime shape — checkpoint saved WITHOUT identity manifest"
                            );
                        }
                    }
                }
                (status, body).into_response()
            }
            Err(e) => api_error(502, &format!("child slot action failed: {e}")),
        }
    }).await,
    )
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
        .map(percent_decode)
        .unwrap_or_default();
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
            // F26: `.identity.json` sidecars (session-identity manifests)
            // are bookkeeping, not checkpoints — don't list them.
            if name.ends_with(".identity.json") {
                continue;
            }
            let size = entry.metadata().map_or(0, |m| m.len());
            files.push(json!({"filename": name, "bytes": size}));
        }
    }
    files.sort_by(|a, b| a["filename"].as_str().cmp(&b["filename"].as_str()));
    axum::Json(json!({"model": model, "sessions": files})).into_response()
}

/// Prometheus text-format label escaping (F25): key names are admin
/// free-text — backslash, quote and newline must not break the scrape.
fn escape_label(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
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

/// Sum one `llamacpp:<name>` counter across the merged child scrapes.
/// Sample-value lines only (`name <v>`); `_bucket`/`_sum` histogram
/// suffixes cannot collide with the trailing space.
fn sum_child_counter(text: &str, name: &str) -> u64 {
    let prefix = format!("llamacpp:{name} ");
    text.lines()
        .filter_map(|l| l.strip_prefix(&prefix))
        .filter_map(|v| v.split(' ').next())
        .filter_map(|v| v.parse::<u64>().ok())
        .sum()
}

/// Prompt-cache hit ratio from scrape-summed counters. `prompt_processed`
/// EXCLUDES cached tokens upstream, so the denominator is the sum; returns
/// `None` when no prompt traffic has been seen yet.
#[allow(clippy::cast_precision_loss)] // integer counters -> ratio
fn cache_hit_ratio(cached: u64, prompt_processed: u64) -> Option<f64> {
    let seen = cached + prompt_processed;
    (seen > 0).then(|| cached as f64 / seen as f64)
}

/// Prompt-cache economics block (R6/R7-lite): scrape-summed child
/// counters as the authoritative hit ratio + the gateway-observed
/// per-response split (counters + warm/cold TTFT histograms).
fn cache_metrics(state: &AppState, merged: &mut String) {
    let cached_total = sum_child_counter(merged, "prompt_tokens_cached_total");
    // Upstream `prompt_tokens_total` counts tokens PROCESSED, excluding cached
    // ones (server-task.cpp metric help), so the hit ratio denominator is the
    // sum — cached/processed alone can exceed 1.
    let prompt_processed = sum_child_counter(merged, "prompt_tokens_total");
    if let Some(ratio) = cache_hit_ratio(cached_total, prompt_processed) {
        let _ = write!(
            merged,
            "# HELP pallama_cache_hit_ratio Reused prompt tokens / all prompt tokens seen (cached + processed), summed from live engine scrapes (authoritative)\n# TYPE pallama_cache_hit_ratio gauge\npallama_cache_hit_ratio {ratio:.4}\n"
        );
    }
    let _ = write!(
        merged,
        "# HELP pallama_prompt_cached_tokens_observed_total Prompt tokens reported REUSED by completed responses observed at the gateway (per-response usage sum, not the scrape)\n# TYPE pallama_prompt_cached_tokens_observed_total counter\npallama_prompt_cached_tokens_observed_total {}\n# HELP pallama_prompt_tokens_observed_total Prompt tokens reported by completed responses observed at the gateway\n# TYPE pallama_prompt_tokens_observed_total counter\npallama_prompt_tokens_observed_total {}\n# HELP pallama_ttft_unclassified_total Completed responses whose usage never surfaced (aborted before usage chunk or upstream omission) - excluded from the warm/cold split\n# TYPE pallama_ttft_unclassified_total counter\npallama_ttft_unclassified_total {}\n",
        state.obs.cached_tokens.load(std::sync::atomic::Ordering::Relaxed),
        state.obs.prompt_tokens.load(std::sync::atomic::Ordering::Relaxed),
        state.obs.unclassified.load(std::sync::atomic::Ordering::Relaxed),
    );
    state.obs.ttft_warm.render(merged);
    state.obs.ttft_cold.render(merged);
    state.obs.render_per_model(merged);
    let live_sessions = state.sup.sessions.live_len(std::time::Duration::from_secs(
        state.config.session_keep_secs,
    ));
    let _ = write!(
        merged,
        "# HELP pallama_sessions_live Sessions with an unexpired eviction pin (x-pallama-session)\n# TYPE pallama_sessions_live gauge\npallama_sessions_live {live_sessions}\n"
    );
    let _ = write!(
        merged,
        "# HELP pallama_semantic_cache_hits_total Responses served from the semantic cache (x-pallama-cache)\n# TYPE pallama_semantic_cache_hits_total counter\npallama_semantic_cache_hits_total {}\n# HELP pallama_semantic_cache_misses_total Cache-eligible requests that missed (and were stored)\n# TYPE pallama_semantic_cache_misses_total counter\npallama_semantic_cache_misses_total {}\n# HELP pallama_semantic_cache_stores_total Responses filed into the semantic cache\n# TYPE pallama_semantic_cache_stores_total counter\npallama_semantic_cache_stores_total {}\n# HELP pallama_semantic_cache_embed_failures_total Embed-model failures that bypassed the cache (request still served live)\n# TYPE pallama_semantic_cache_embed_failures_total counter\npallama_semantic_cache_embed_failures_total {}\n# HELP pallama_semantic_cache_entries Live (non-expired) semantic cache entries\n# TYPE pallama_semantic_cache_entries gauge\npallama_semantic_cache_entries {}\n",
        state.sem.hits.load(std::sync::atomic::Ordering::Relaxed),
        state.sem.misses.load(std::sync::atomic::Ordering::Relaxed),
        state.sem.stores.load(std::sync::atomic::Ordering::Relaxed),
        state
            .sem
            .embed_failures
            .load(std::sync::atomic::Ordering::Relaxed),
        state.semcache.live_len(),
    );
}

/// Active engine build gauge; absent when the store is not readable.
fn engine_build_gauge(state: &AppState, out: &mut String) {
    if let Some(Ok(Some(engine))) = state.with_store(pallama_core::Store::active_engine) {
        if let Ok(m) = serde_json::from_str::<pallama_runtime::Manifest>(&engine.manifest) {
            let _ = write!(
                out,
                "# HELP pallama_engine_build Active engine build number\n# TYPE pallama_engine_build gauge\npallama_engine_build {}\n",
                m.build_number
            );
        }
    }
}

// exported metric families in one responder
// Prometheus exposition: one flat text buffer write per metric family.
// Extracting families into helpers is queued with the parallel gateway wave.
#[allow(clippy::too_many_lines)]
pub async fn metrics(State(state): State<Arc<AppState>>) -> Response {
    let mut merged = String::new();
    for e in state.sup.live_http_endpoints() {
        let pallama_core::profile::Endpoint::Tcp { host, port } = &e.endpoint else {
            continue;
        };
        let url = format!("http://{host}:{port}/metrics");
        if let Ok(resp) = child_auth(state.http.get(&url), &e).send().await {
            if resp.status().is_success() {
                if let Ok(text) = resp.text().await {
                    merged.push_str(&text);
                    merged.push('\n');
                }
            }
        }
    }
    // Prompt-cache economics (R6/R7-lite): authoritative child-side
    // counters summed across live engines (scrape time = truth), plus
    // the gateway-observed warm/cold split.
    cache_metrics(&state, &mut merged);
    // F25: router mode serves N models through ONE pallama instance —
    // count the router child's loaded models, not supervisor rows.
    let models_loaded = if state.config.router {
        match crate::proxy::ensure_router_detached(&state.sup).await {
            Ok(engine) => {
                let url = format!("{}/models", child_base(&engine.endpoint));
                match child_auth(state.http.get(&url), &engine).send().await {
                    Ok(r) if r.status().is_success() => match r.json::<Value>().await {
                        Ok(v) => v["data"].as_array().map_or_else(
                            || state.sup.ps().len(),
                            |a| {
                                a.iter()
                                    .filter(|m| {
                                        !matches!(
                                            m["status"]["value"].as_str().unwrap_or(""),
                                            "unloaded" | "downloaded" | "downloading"
                                        )
                                    })
                                    .count()
                            },
                        ),
                        Err(_) => state.sup.ps().len(),
                    },
                    _ => state.sup.ps().len(),
                }
            }
            Err(_) => state.sup.ps().len(),
        }
    } else {
        state.sup.ps().len()
    };
    let _ = write!(
        merged,
        "# HELP pallama_models_loaded Models currently loaded\n# TYPE pallama_models_loaded gauge\npallama_models_loaded {models_loaded}\n"
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
    // E4: audit lines dropped (writer saturated / IO failure). Non-zero
    // while `audit_log = true` means the trail has gaps.
    let _ = write!(
        merged,
        "# HELP pallama_audit_dropped_total Audit lines dropped (writer saturated or IO failure)\n# TYPE pallama_audit_dropped_total counter\npallama_audit_dropped_total {}\n",
        state
            .audit_dropped
            .load(std::sync::atomic::Ordering::Relaxed)
    );
    // F81: spans never exported (channel saturated or POST failed) —
    // same honesty contract as the audit drop counter.
    let _ = write!(
        merged,
        "# HELP pallama_otlp_dropped_total OTLP spans dropped (channel saturated or export failure)\n# TYPE pallama_otlp_dropped_total counter\npallama_otlp_dropped_total {}\n",
        state
            .otlp
            .dropped
            .load(std::sync::atomic::Ordering::Relaxed)
    );
    // Per-key usage today (the [[keys]] tier's live accounting).
    let _ = write!(
        merged,
        "# HELP pallama_key_usage_requests Requests today per key\n# TYPE pallama_key_usage_requests counter\n"
    );
    for (name, _, req, _) in state.keys.usage_snapshot() {
        let _ = writeln!(
            merged,
            "pallama_key_usage_requests{{key=\"{}\"}} {req}",
            escape_label(&name)
        );
    }
    let _ = write!(
        merged,
        "# HELP pallama_key_usage_tokens Tokens today per key\n# TYPE pallama_key_usage_tokens counter\n"
    );
    for (name, _, _, tok) in state.keys.usage_snapshot() {
        let _ = writeln!(
            merged,
            "pallama_key_usage_tokens{{key=\"{}\"}} {tok}",
            escape_label(&name)
        );
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
// `unit__name__scenario` triple-underscore test names violate the
// consecutive-underscores snake_case lint by design (audit naming spec).
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use pallama_runtime::PallamaEvent;

    /// Minimal GGUF with `general.architecture` + optional
    /// `tokenizer.chat_template` — just enough header for
    /// `read_metadata_file` to reach the template field.
    fn write_gguf_with_template(path: &std::path::Path, template: Option<&str>) {
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        let mut kvs: Vec<(&str, String)> = vec![("general.architecture", "qwen3".into())];
        if let Some(t) = template {
            kvs.push(("tokenizer.chat_template", t.to_string()));
        }
        b.extend_from_slice(&u64::try_from(kvs.len()).unwrap().to_le_bytes());
        for (k, v) in kvs {
            b.extend_from_slice(&u64::try_from(k.len()).unwrap().to_le_bytes());
            b.extend_from_slice(k.as_bytes());
            b.extend_from_slice(&8u32.to_le_bytes()); // GgufValue::String
            b.extend_from_slice(&u64::try_from(v.len()).unwrap().to_le_bytes());
            b.extend_from_slice(v.as_bytes());
        }
        std::fs::write(path, b).unwrap();
    }

    fn row_with_path(path: &str) -> pallama_core::ModelRow {
        pallama_core::ModelRow {
            name: "m1".into(),
            repo: "registry.ollama.ai/library/m1".into(),
            quant: "Q4_K_M".into(),
            path: path.into(),
            bytes: 500,
            sha256: None,
            mmproj_path: None,
            shards: 1,
            arch: None,
            params: None,
            ctx_train: Some(40960),
            pulled_at: 0,
        }
    }

    fn think_req(think: Option<bool>) -> Value {
        let mut v = serde_json::json!({"model": "m1", "messages": []});
        if let Some(t) = think {
            v["think"] = serde_json::Value::Bool(t);
        }
        v
    }

    #[test]
    fn unit__template_supports_thinking__marker_dialects() {
        assert!(template_supports_thinking(
            "{%- if enable_thinking -%}<think>"
        ));
        assert!(template_supports_thinking("{{- '<think>' -}}"));
        assert!(template_supports_thinking("{{ reasoning }}"));
        assert!(!template_supports_thinking(
            "You are a helpful assistant.<|im_end|>"
        ));
        assert!(!template_supports_thinking(""));
    }

    #[test]
    fn unit__child_error_body__empty_choices_with_error_surfaced() {
        // The exact live failure shape: mistral.rs 200s with an empty
        // choices array and a service_unavailable error object when a
        // prompt exceeds the paged-attention batch step.
        let body = serde_json::json!({
            "id": "x",
            "choices": [],
            "error": {"message": "service_unavailable", "code": 503}
        });
        let msg = child_error_body(&body).expect("empty-choices error must surface");
        assert!(msg.contains("service_unavailable"));
        // String-typed error works too.
        let str_err = serde_json::json!({"choices": [], "error": "boom"});
        assert_eq!(child_error_body(&str_err).as_deref(), Some("boom"));
        // Normal responses and error-free empties stay None.
        assert!(child_error_body(&serde_json::json!({"choices": [{"i": 0}]})).is_none());
        assert!(child_error_body(&serde_json::json!({"choices": []})).is_none());
        assert!(child_error_body(&serde_json::json!({})).is_none());
    }

    #[test]
    fn unit__refuse_unsupported_think__gates_on_evidence_fail_open_without() {
        let tmp = tempfile::tempdir().unwrap();
        // Plain template + think:true -> teaching 400.
        let plain = tmp.path().join("plain.gguf");
        write_gguf_with_template(&plain, Some("You are a helpful assistant."));
        let resp = refuse_unsupported_think(
            &row_with_path(plain.to_str().unwrap()),
            &think_req(Some(true)),
        )
        .expect("plain template + think:true must refuse");
        assert_eq!(resp.status(), axum::http::StatusCode::BAD_REQUEST);
        // Thinking template + think:true -> pass.
        let thinker = tmp.path().join("thinker.gguf");
        write_gguf_with_template(&thinker, Some("{%- if enable_thinking -%}"));
        assert!(refuse_unsupported_think(
            &row_with_path(thinker.to_str().unwrap()),
            &think_req(Some(true))
        )
        .is_none());
        // Template absent -> fail-open (legacy GGUFs).
        let bare = tmp.path().join("bare.gguf");
        write_gguf_with_template(&bare, None);
        assert!(refuse_unsupported_think(
            &row_with_path(bare.to_str().unwrap()),
            &think_req(Some(true))
        )
        .is_none());
        // Unreadable path -> fail-open (safetensors lanes).
        assert!(refuse_unsupported_think(
            &row_with_path("/nonexistent/m1.gguf"),
            &think_req(Some(true))
        )
        .is_none());
        // Gate dormant without an explicit think:true.
        assert!(refuse_unsupported_think(
            &row_with_path(plain.to_str().unwrap()),
            &think_req(None)
        )
        .is_none());
        assert!(refuse_unsupported_think(
            &row_with_path(plain.to_str().unwrap()),
            &think_req(Some(false))
        )
        .is_none());
    }

    #[test]
    fn unit__default_think_off__injects_false_only_for_thinking_templates() {
        let tmp = tempfile::tempdir().unwrap();
        let thinker = tmp.path().join("thinker.gguf");
        write_gguf_with_template(&thinker, Some("{%- if enable_thinking -%}"));
        let plain = tmp.path().join("plain.gguf");
        write_gguf_with_template(&plain, Some("You are a helpful assistant."));
        let bare = tmp.path().join("bare.gguf");
        write_gguf_with_template(&bare, None);

        // Thinking template + absent think -> injected false (ollama parity).
        let mut req = think_req(None);
        default_think_off(&row_with_path(thinker.to_str().unwrap()), &mut req);
        assert_eq!(req["think"], serde_json::Value::Bool(false));
        // null think is treated as absent -> injected false.
        let mut req: Value = serde_json::json!({"model": "m1", "messages": [], "think": null});
        default_think_off(&row_with_path(thinker.to_str().unwrap()), &mut req);
        assert_eq!(req["think"], serde_json::Value::Bool(false));
        // Explicit user toggles are never touched.
        let mut req = think_req(Some(true));
        default_think_off(&row_with_path(thinker.to_str().unwrap()), &mut req);
        assert_eq!(req["think"], serde_json::Value::Bool(true));
        let mut req = think_req(Some(false));
        default_think_off(&row_with_path(thinker.to_str().unwrap()), &mut req);
        assert_eq!(req["think"], serde_json::Value::Bool(false));
        // Non-thinking template: template default (off) already correct -> untouched.
        let mut req = think_req(None);
        default_think_off(&row_with_path(plain.to_str().unwrap()), &mut req);
        assert!(req.get("think").is_none());
        // Template-less GGUF -> fail-open, no injection.
        let mut req = think_req(None);
        default_think_off(&row_with_path(bare.to_str().unwrap()), &mut req);
        assert!(req.get("think").is_none());
        // Unreadable path -> fail-open (safetensors lanes).
        let mut req = think_req(None);
        default_think_off(&row_with_path("/nonexistent/m1.gguf"), &mut req);
        assert!(req.get("think").is_none());
    }

    #[test]
    fn unit__pull_stream_names__raw_target_includes_registry_name() {
        // Regression pin (2026-09-09 /api/pull stream starvation): events
        // publish under the registry name, the client asked by raw target.
        let names = pull_stream_names("ggml-org/Qwen3-0.6B-GGUF:Q4_K_M");
        assert!(names.contains(&"ggml-org/Qwen3-0.6B-GGUF:Q4_K_M".to_string()));
        assert!(names.contains(&"qwen3-0.6b".to_string()));
    }

    #[test]
    fn unit__parse_keep_alive_str__go_durations_and_rejects() {
        // F12/F20: ollama clients send Go-duration keep_alive strings.
        assert_eq!(parse_keep_alive_str("0"), Some(0));
        assert_eq!(parse_keep_alive_str("300"), Some(300));
        assert_eq!(parse_keep_alive_str("-1"), Some(-1));
        assert_eq!(parse_keep_alive_str("10m"), Some(600));
        assert_eq!(parse_keep_alive_str("1h30m"), Some(5400));
        assert_eq!(parse_keep_alive_str("2d"), Some(172_800));
        assert_eq!(parse_keep_alive_str("1w"), Some(604_800));
        assert_eq!(parse_keep_alive_str("abc"), None);
        assert_eq!(parse_keep_alive_str("m10"), None);
        assert_eq!(parse_keep_alive_str("1x"), None);
        assert_eq!(parse_keep_alive_str(""), None);
    }

    #[test]
    fn unit__iso__rfc3339_utc_shape() {
        // F16: ollama clients expect RFC3339 strings, not epoch ints.
        assert_eq!(iso(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso(1_700_000_000), "2023-11-14T22:13:20Z");
        // day/month/year rollups land on real calendar boundaries
        assert_eq!(iso(86_399), "1970-01-01T23:59:59Z");
        assert_eq!(iso(86_400), "1970-01-02T00:00:00Z");
    }

    #[test]
    fn unit__pull_stream_names__bogus_target_keeps_raw() {
        let names = pull_stream_names("::no such repo::");
        assert_eq!(names, vec!["::no such repo::".to_string()]);
    }

    #[test]
    fn unit__pull_event_line__normalized_event_matches_raw_request_filter() {
        // The exact lane that hung: request by raw HF coordinate, events
        // arrive under the normalized registry name -> must terminate.
        let names = pull_stream_names("ggml-org/Qwen3-0.6B-GGUF:Q4_K_M");
        let (line, progress) = pull_event_line(
            &names,
            &PallamaEvent::PullProgress {
                name: "qwen3-0.6b".into(),
                downloaded: 1,
                total: 10,
            },
        );
        assert!(!progress);
        assert!(
            line.contains("\"status\":\"pulling\"") || line.contains("\"status\": \"pulling\"")
        );
        let (line, done) = pull_event_line(
            &names,
            &PallamaEvent::ModelPulled {
                name: "qwen3-0.6b".into(),
                warning: None,
            },
        );
        assert!(done);
        let v: Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(v["status"], "success");
    }

    #[test]
    fn unit__pull_event_line__handler_failure_path_matches_raw_target() {
        // Parse/lock failures publish PullFailed under the RAW target.
        let names = pull_stream_names("ggml-org/Qwen3-0.6B-GGUF:Q4_K_M");
        let (line, done) = pull_event_line(
            &names,
            &PallamaEvent::PullFailed {
                name: "ggml-org/Qwen3-0.6B-GGUF:Q4_K_M".into(),
                error: "pull already in flight".into(),
            },
        );
        assert!(done);
        assert!(line.contains("already in flight"));
    }

    #[test]
    fn unit__pull_event_line__warning_rides_success_ndjson() {
        let (line, done) = pull_event_line(
            &["m1".to_string()],
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
            &["m1".to_string()],
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

    #[test]
    fn unit__sum_child_counter__sums_samples_and_ignores_collisions() {
        // Two children merged into one scrape: sample values sum.
        let scrape = "llamacpp:prompt_tokens_cached_total 96\n\
                      # HELP llamacpp:prompt_tokens_cached_total tokens\n\
                      llamacpp:prompt_tokens_cached_total 24\n\
                      llamacpp:prompt_tokens_total 240\n\
                      llamacpp:prompt_tokens_cached_total_bucket{le=\"1\"} 7\n";
        assert_eq!(sum_child_counter(scrape, "prompt_tokens_cached_total"), 120);
        assert_eq!(sum_child_counter(scrape, "prompt_tokens_total"), 240);
        // Distinct name must not match the longer counter's lines.
        assert_eq!(
            sum_child_counter(
                "llamacpp:prompt_tokens_cached_total 5\n",
                "prompt_tokens_total"
            ),
            0
        );
        // Missing name, non-numeric sample, empty text: zero.
        assert_eq!(sum_child_counter("", "prompt_tokens_total"), 0);
        assert_eq!(sum_child_counter("llamacpp:x nan\n", "x"), 0);
    }

    #[test]
    fn unit__cache_hit_ratio__denominator_includes_cached() {
        // Pin (live 11499, 2026-09-09): upstream `prompt_tokens_total` counts
        // processed tokens EXCLUDING cached ones, so cached/processed hit
        // 2.27 > 1 on a warm sandbox. The ratio must be cached/(cached+processed).
        let ratio = cache_hit_ratio(157, 69).expect("traffic seen -> ratio");
        assert!((ratio - 157.0 / 226.0).abs() < 1e-12, "got {ratio}");
        // All-cached and all-processed extremes stay inside [0, 1].
        assert_eq!(cache_hit_ratio(100, 0), Some(1.0));
        assert_eq!(cache_hit_ratio(0, 100), Some(0.0));
        // No prompt traffic yet: no gauge.
        assert_eq!(cache_hit_ratio(0, 0), None);
    }
}
