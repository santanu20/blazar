//! v0.16 completion: replication + scheduling-visibility surfaces.
//!
//! Three operators on the same federation machinery `remotes.rs` already
//! runs for routing:
//!
//! * `POST /api/warm` — make a model resident on THIS gateway (the
//!   replication primitive a peer calls).
//! * `POST /api/replicate` — fan a warm out to named peers so several
//!   nodes hold the model warm (remote replication).
//! * `GET /api/route/{model}` — the cross-node scheduling decision an
//!   incoming request would take, with the signals behind it (local
//!   residency, per-peer tier / wait estimate / free VRAM).
//!
//! Honesty contracts:
//! * warm loads the TEXT lane (`needs_vision = false`): a vision model
//!   warms its base child; the projector lane attaches on the first
//!   vision-carrying request. Warm does not pin residency — the idle
//!   TTL governs the child afterwards exactly as if a chat had loaded
//!   it.
//! * replicate reports per-peer outcomes verbatim, including peers
//!   that have no `/api/warm` (non-Blazar remotes).
//! * route never fabricates a decision: unknown-everywhere is a valid
//!   answer (`target: "none"` with the teaching reason).

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use blazar_core::config::Remote;
use std::sync::Arc;
use std::time::Duration;

use crate::AppState;
use crate::ollama::api_error;
use crate::proxy::{ensure_with_admission, resolve_model};
use crate::queue::{Priority, WorkClass};

/// Default per-peer bound for a replicate warm call. Generous on
/// purpose: a peer loading a multi-GiB model from cold disk legally
/// takes tens of seconds; the clamp ceiling (`900`) bounds the worst
/// case while the request override lets scripted fleets tune it.
const REPLICATE_TIMEOUT_DEFAULT_SECS: u64 = 300;
const REPLICATE_TIMEOUT_MAX_SECS: u64 = 900;

/// `PeerScore::tier` numbering → the name the API and CLI print.
#[must_use]
pub fn tier_name(tier: u8) -> &'static str {
    match tier {
        0 => "warm",
        1 => "unknown",
        _ => "cold",
    }
}

/// Per-peer timeout clamp: absent → default; present → bounded
/// `[1, 900]` seconds. Pure so the clamp is unit-pinnable.
#[must_use]
pub fn clamp_timeout(secs: Option<u64>) -> Duration {
    Duration::from_secs(
        secs.unwrap_or(REPLICATE_TIMEOUT_DEFAULT_SECS)
            .clamp(1, REPLICATE_TIMEOUT_MAX_SECS),
    )
}

/// Select the remotes a replicate call targets. `None` = every
/// configured remote. An unknown name is a teaching error listing what
/// IS configured — silent best-effort matching here would hide typos.
/// Pure over the config slice.
pub fn filter_remotes<'a>(
    remotes: &'a [Remote],
    names: Option<&[String]>,
) -> Result<Vec<&'a Remote>, String> {
    let Some(names) = names else {
        return Ok(remotes.iter().collect());
    };
    if names.is_empty() {
        return Err(
            "peers list is empty — pass peer names or omit the field for all remotes".into(),
        );
    }
    let mut selected = Vec::new();
    for name in names {
        let Some(remote) = remotes.iter().find(|r| &r.name == name) else {
            let configured = remotes
                .iter()
                .map(|r| r.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "unknown peer '{name}' — configured remotes: {}",
                if configured.is_empty() {
                    "(none)"
                } else {
                    &configured
                }
            ));
        };
        selected.push(remote);
    }
    Ok(selected)
}

/// Convert a boxed openai-shaped error response (the admission/ensure
/// path's error currency) into the ollama-lane error shape this API
/// family speaks. Same extraction pattern as `admission_gate_ollama`.
async fn boxed_to_api_error(boxed: Box<Response>) -> Response {
    let status = boxed.status().as_u16();
    let msg = axum::body::to_bytes(boxed.into_body(), 64 * 1024)
        .await
        .ok()
        .and_then(|bytes| {
            serde_json::from_slice::<serde_json::Value>(&bytes)
                .ok()
                .and_then(|v| {
                    v.pointer("/error/message")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_string)
                })
        })
        .unwrap_or_else(|| "warm failed".to_string());
    api_error(status, &msg)
}

/// `POST /api/warm {"model", "wait"?}` — make a model resident on this
/// gateway. Two modes, one endpoint:
///
/// * notify (default, the deployed `blazar pull` contract): detached
///   `warm_on_pull_if_enabled` — policy-gated (knob + AC power +
///   admission belts, warn-not-fail), answers `{"status": "ok"}`
///   immediately. The spawn outlives the request; tying it to the
///   handler once let a client disconnect cancel it mid-load.
/// * `wait: true` (replication + `blazar warm` CLI): synchronous
///   ensure through the standard admission path (low priority,
///   interactive class — never starves real traffic; concurrent loads
///   of the same model coalesce), then reports the live child state.
pub async fn warm(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let v: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return api_error(400, &e.to_string()),
    };
    let Some(model) = v["model"].as_str().map(str::to_string) else {
        return api_error(400, "missing 'model'");
    };
    if !v["wait"].as_bool().unwrap_or(false) {
        // Notify mode: byte-identical to the long-standing pull-notify
        // contract — detached, policy-gated, always "ok".
        let sup = Arc::clone(&state.sup);
        tokio::spawn(async move {
            sup.warm_on_pull_if_enabled(&model).await;
        });
        return axum::Json(serde_json::json!({ "status": "ok" })).into_response();
    }
    let model = model.trim();
    if model.is_empty() {
        return api_error(400, "missing 'model'");
    }
    // Canonical name first: the post-load ps() lookup keys on it, and a
    // tag-form alias (`qwen3-1.7b:bf16`) must resolve like chat does.
    let resolved = state
        .with_store(|s| resolve_model(s, model))
        .and_then(std::result::Result::ok);
    let Some(row) = resolved else {
        return api_error(
            StatusCode::NOT_FOUND.as_u16(),
            &format!(
                "unknown model: {model} — `blazar ls` lists the store, `blazar pull` adds one"
            ),
        );
    };
    // Low priority + interactive class: warm is background work that
    // must never jump the request queue, but still needs a slot to
    // load. No prefix pin, text lane, no diffusion, no eviction hold —
    // the idle TTL governs residency after this returns, same as a
    // completed chat.
    match ensure_with_admission(
        &state,
        model,
        Priority::Low,
        WorkClass::Interactive,
        None,
        false,
        false,
        false,
    )
    .await
    {
        Ok((engine_ref, load_ms)) => {
            let resident = state
                .sup
                .ps()
                .into_iter()
                .find(|r| r.name == row.name)
                .map(|r| {
                    serde_json::json!({
                        "engine": r.engine,
                        "state": r.state,
                        "ctx": r.ctx,
                        "slots": r.slots,
                        "slots_configured": r.slots_configured,
                        "in_flight": r.in_flight,
                    })
                });
            axum::Json(serde_json::json!({
                "object": "blazar.warm",
                "model": row.name,
                "lane": engine_ref.name,
                "engine": engine_ref.kind.as_str(),
                "load_ms": load_ms,
                "resident": resident,
                "note": "warm loads the text lane and holds no pin — the idle TTL governs residency; a vision projector attaches on the first vision request",
            }))
            .into_response()
        }
        Err(boxed) => boxed_to_api_error(boxed).await,
    }
}

/// `POST /api/replicate {"model", "peers"?, "timeout_secs"?}` — warm a
/// model on the selected peers concurrently. This is remote
/// replication: after it, several nodes hold the model warm and the
/// capacity-aware fallback spreads load across them.
///
/// Per-peer results are reported verbatim; a non-Blazar remote answers
/// 404/405 on `/api/warm` and is called out as such, never silently
/// dropped.
pub async fn replicate(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let v: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return api_error(400, &format!("invalid JSON: {e}")),
    };
    let Some(model) = v["model"].as_str().map(str::trim).filter(|m| !m.is_empty()) else {
        return api_error(400, "missing 'model' — the model to replicate");
    };
    if state.config.remotes.is_empty() {
        return api_error(
            400,
            "no remotes configured — add a [[remotes]] entry before replicating",
        );
    }
    let names: Option<Vec<String>> = v["peers"].as_array().map(|a| {
        a.iter()
            .filter_map(|n| n.as_str().map(str::to_string))
            .collect()
    });
    let peers = match filter_remotes(&state.config.remotes, names.as_deref()) {
        Ok(p) => p,
        Err(teach) => return api_error(StatusCode::NOT_FOUND.as_u16(), &teach),
    };
    let timeout = clamp_timeout(v["timeout_secs"].as_u64());
    // Concurrent fan-out: peer loads are independent; the slowest peer
    // sets the wall time, not the sum.
    let results = futures::future::join_all(peers.iter().map(|remote| {
        let url = format!("{}/api/warm", remote.url.trim_end_matches('/'));
        let mut req = state
            .http
            .post(&url)
            .json(&serde_json::json!({ "model": model, "wait": true }));
        if !remote.key.is_empty() {
            req = req.bearer_auth(&remote.key);
        }
        let remote = *remote;
        async move {
            match req.timeout(timeout).send().await {
                Ok(resp) => {
                    let status = resp.status();
                    // Read the body whatever the status: a peer's warm
                    // error carries its own teaching message.
                    let body = resp
                        .json::<serde_json::Value>()
                        .await
                        .unwrap_or(serde_json::Value::Null);
                    if status.is_success() {
                        serde_json::json!({
                            "peer": remote.name,
                            "url": remote.url,
                            "ok": true,
                            "state": body.pointer("/resident/state").cloned(),
                            "load_ms": body.get("load_ms").cloned(),
                        })
                    } else if status.as_u16() == 404 || status.as_u16() == 405 {
                        serde_json::json!({
                            "peer": remote.name,
                            "url": remote.url,
                            "ok": false,
                            "note": "peer has no /api/warm — remote replication needs a Blazar gateway peer",
                        })
                    } else {
                        serde_json::json!({
                            "peer": remote.name,
                            "url": remote.url,
                            "ok": false,
                            "error": format!(
                                "peer answered {status}: {}",
                                body.pointer("/error").cloned().unwrap_or(serde_json::Value::Null)
                            ),
                        })
                    }
                }
                Err(e) => serde_json::json!({
                    "peer": remote.name,
                    "url": remote.url,
                    "ok": false,
                    "error": format!("unreachable within the timeout: {e}"),
                }),
            }
        }
    }))
    .await;
    // Refresh the capacity cache for the touched peers so /api/ps and
    // the next routing decision see the new residents immediately
    // instead of after the presence TTL.
    let refreshes = futures::future::join_all(
        peers
            .iter()
            .map(|remote| crate::remotes::refresh_capacity_cache(&state, remote)),
    )
    .await;
    let _ = refreshes;
    let ok_count = results
        .iter()
        .filter(|r| r["ok"].as_bool().unwrap_or(false))
        .count();
    axum::Json(serde_json::json!({
        "object": "blazar.replicate",
        "model": model,
        "warmed": ok_count,
        "of": peers.len(),
        "results": results,
    }))
    .into_response()
}

/// `GET /api/route/{model}` — the scheduling decision an incoming
/// request for this model would take, with every signal behind it.
/// Read-only; forces a presence refresh only when the model is not
/// local (the fallback path would pay that probe anyway).
// One linear decision tree (local → peers → none) with inline JSON
// assembly per branch; splitting it would hide the decision order it exists to show.
#[allow(clippy::too_many_lines)]
pub async fn route(State(state): State<Arc<AppState>>, Path(model): Path<String>) -> Response {
    let model = model.trim().to_string();
    if model.is_empty() {
        return api_error(400, "missing model name in the path");
    }
    let local_row = state
        .with_store(|s| resolve_model(s, &model))
        .and_then(std::result::Result::ok);
    // Snapshot the cached peer signals once (R4: clones before any
    // comparison, no lock held during scoring).
    let healths = state
        .remote_health
        .lock()
        .expect("remote_health lock poisoned")
        .clone();
    let caps = state
        .remote_capacity
        .lock()
        .expect("remote_capacity lock poisoned")
        .clone();
    let peer_lines = |serving: &[&Remote]| -> Vec<serde_json::Value> {
        state
            .config
            .remotes
            .iter()
            .map(|remote| {
                let hkey = crate::remotes::health_key(remote);
                let h = healths.get(&hkey).copied().unwrap_or_default();
                let score = crate::remotes::score_peer(&model, &h, caps.get(&hkey));
                serde_json::json!({
                    "name": remote.name,
                    "url": remote.url,
                    "serves_model": serving.iter().any(|s| s.name == remote.name),
                    "tier": tier_name(score.tier),
                    "est_wait_ms": score.est_wait_ms,
                    "free_vram_bytes": score.free_vram.0,
                    "leases": h.in_flight,
                    "marked_down": h.down_until.is_some(),
                })
            })
            .collect()
    };
    if let Some(row) = local_row {
        // Local ownership wins: chat resolves this model here and never
        // falls back. Peers stay informational from cache only — no
        // probe cost on the hot path.
        let resident = state
            .sup
            .ps()
            .into_iter()
            .find(|r| r.name == row.name)
            .map(|r| {
                serde_json::json!({
                    "engine": r.engine,
                    "state": r.state,
                    "ctx": r.ctx,
                    "slots": r.slots,
                    "in_flight": r.in_flight,
                })
            });
        let reason = if resident.is_some() {
            "model is owned and resident locally — requests are served by this gateway"
        } else {
            "model is owned locally — the next request loads it here (federation is not consulted for local models)"
        };
        let serving: Vec<&Remote> = Vec::new();
        return axum::Json(serde_json::json!({
            "object": "blazar.route",
            "model": row.name,
            "local": { "resident": resident },
            "peers": peer_lines(&serving),
            "decision": { "target": "local", "reason": reason },
        }))
        .into_response();
    }
    // Not local: this is exactly the fallback path — refresh presence
    // (throttled force-probe) and rank whoever serves it.
    let serving = crate::remotes::peers_serving_with_refresh(&state, &model).await;
    let peers = peer_lines(&serving);
    if let Some(best) = crate::remotes::rank_peers_by_capacity(&state, &model, &serving)
        .into_iter()
        .next()
    {
        let hkey = crate::remotes::health_key(best);
        let h = healths.get(&hkey).copied().unwrap_or_default();
        let score = crate::remotes::score_peer(&model, &h, caps.get(&hkey));
        return axum::Json(serde_json::json!({
            "object": "blazar.route",
            "model": model,
            "local": { "resident": serde_json::Value::Null },
            "peers": peers,
            "decision": {
                "target": format!("remote:{}", best.name),
                "reason": format!(
                    "model is not in the local store — best-ranked serving peer (tier {}, est wait {} ms)",
                    tier_name(score.tier),
                    score.est_wait_ms
                ),
            },
        }))
        .into_response();
    }
    axum::Json(serde_json::json!({
        "object": "blazar.route",
        "model": model,
        "local": { "resident": serde_json::Value::Null },
        "peers": peers,
        "decision": {
            "target": "none",
            "reason": "unknown model: not in the local store and no peer lists it — `blazar ls` shows local models, `blazar ps` shows peers",
        },
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use super::*;

    fn remote(name: &str) -> Remote {
        Remote {
            name: name.into(),
            url: format!("http://{name}.local:11434"),
            key: String::new(),
            allow_insecure_http: false,
        }
    }

    #[test]
    fn unit__tier_name__maps_score_numbering() {
        assert_eq!(tier_name(0), "warm");
        assert_eq!(tier_name(1), "unknown");
        assert_eq!(tier_name(2), "cold");
        // Defensive: out-of-range tiers still name honestly as cold.
        assert_eq!(tier_name(9), "cold");
    }

    #[test]
    fn unit__clamp_timeout__default_and_bounds() {
        assert_eq!(clamp_timeout(None), Duration::from_secs(300));
        assert_eq!(clamp_timeout(Some(1)), Duration::from_secs(1));
        assert_eq!(clamp_timeout(Some(900)), Duration::from_mins(15));
        assert_eq!(clamp_timeout(Some(0)), Duration::from_secs(1));
        assert_eq!(clamp_timeout(Some(9_999)), Duration::from_mins(15));
    }

    #[test]
    fn unit__filter_remotes__all_when_unset_teaches_on_unknown() {
        let remotes = vec![remote("a"), remote("b")];
        let all = filter_remotes(&remotes, None).expect("all selects every remote");
        assert_eq!(all.len(), 2);
        let one = filter_remotes(&remotes, Some(&["b".to_string()])).expect("named selects");
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].name, "b");
        let err = filter_remotes(&remotes, Some(&["c".to_string()])).expect_err("unknown teaches");
        assert!(err.contains("unknown peer 'c'"), "{err}");
        assert!(err.contains("a, b"), "{err}");
        let empty = filter_remotes(&remotes, Some(&[])).expect_err("empty list teaches");
        assert!(empty.contains("peers list is empty"), "{empty}");
        let none_cfg = filter_remotes(&[], None).expect("no remotes selects nothing");
        assert!(none_cfg.is_empty(), "no remotes must select nothing");
    }
}
