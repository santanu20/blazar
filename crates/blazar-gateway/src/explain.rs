//! `blazar explain <model>` / `GET /api/explain/{model}` — the effective
//! configuration card, with provenance for every value it shows. Answers
//! the ollama#18229 class of confusion ("loaded context length: 32768 —
//! but WHERE did 32768 come from?") before the user has to go digging:
//! each line names its source (overlay pin, config default, or the live
//! child's argv) and admits `unknown` when the value is only decided at
//! spawn time.
//!
//! Data sources are all existing surfaces: the store (model row, engine
//! rows), `Config::effective_*` resolution (overlay > global), the
//! supervisor census (`ps()` — what the live child actually holds), and
//! the spec-mode tiering shared with the supervisor. Nothing here spawns
//! a child or compiles a profile: explain is read-only and honest about
//! what only a spawn can decide.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{json, Value};

use crate::state::AppState;
use blazar_core::config::{RoutingMode, RoutingPolicy};

/// Reason the routing lane resolved the way it did, stated as facts about
/// the inputs (never invented internals of `serving_lane`). Pure so the
/// wording is pinnable without a daemon.
#[must_use]
pub fn lane_reason(
    mode: RoutingMode,
    pin: Option<&str>,
    diffusion: bool,
    safetensors: bool,
    quantized: bool,
    policy: RoutingPolicy,
) -> String {
    if matches!(mode, RoutingMode::Manual) {
        return "engine_routing.mode = manual — the active engine serves every model".into();
    }
    if pin.is_some() {
        return "model_overrides.<model>.engine pins the lane".into();
    }
    if diffusion {
        return "diffusion component set routes to the sdcpp lane".into();
    }
    if safetensors && quantized {
        return format!(
            "quantized safetensors dir under engine_routing.policy = {policy:?} picks the lane"
        );
    }
    if safetensors {
        return format!(
            "HF safetensors dir under engine_routing.policy = {policy:?} picks the lane"
        );
    }
    "GGUF file with no overlay pin — the global active lane serves it".into()
}

/// Where the REQUESTED ctx number came from (overlay pin vs config
/// default). Pure for pinning.
#[must_use]
pub fn requested_ctx_source(overlay_ctx_set: bool, model: &str, default_ctx: u32) -> String {
    if overlay_ctx_set {
        format!("model_overrides.{model}.ctx")
    } else {
        format!("config default_ctx = {default_ctx}")
    }
}

/// Honesty line for the EFFECTIVE ctx when no live child holds the model:
/// the spawn-time planner (auto-fit, VRAM budget) may still shrink it.
/// Pure for pinning.
#[must_use]
pub fn cold_ctx_note() -> &'static str {
    "config estimate — model not resident; the spawn-time planner may shrink it to fit VRAM"
}

/// `GET /api/explain/{model}` — the effective-config card.
/// Store lookup for the routing lane: (global tag, lane resolution).
type LaneLookup = (
    Option<String>,
    Result<Option<(String, blazar_core::engine_kind::EngineKind)>, String>,
);

#[allow(clippy::too_many_lines)] // the card IS one document: model facts, lane, context, slots, speculation, cache, residents
pub async fn explain(State(state): State<Arc<AppState>>, Path(model): Path<String>) -> Response {
    // Store resolution first: unknown models fail with teaching, never a
    // half-empty card that looks authoritative.
    // Canonical gateway ladder (exact → colon-swap → colonless strip →
    // prefix → suggestion) so the `name:quant` display form /api/tags emits
    // resolves here exactly like it does on /api/chat.
    let resolved = match state.with_store(|s| crate::proxy::resolve_model(s, &model)) {
        Some(Ok(row)) => row.name,
        Some(Err(teach)) => return crate::error_response(404, &teach),
        None => {
            return crate::error_response(
                404,
                "store unavailable — start the daemon with a writable data dir",
            )
        }
    };
    let Some(Some(row)) = state.with_store(|s| s.get_model(&resolved).ok().flatten()) else {
        return crate::error_response(
            404,
            &format!(
                "unknown model: {model} — `blazar ls` lists the store, `blazar pull` adds one"
            ),
        );
    };

    let overlay = state.config.overlay_for(&resolved);
    let safetensors = std::path::Path::new(&row.path).is_dir();
    let diffusion = row.has_component_set();
    let quantized = blazar_core::store::quantized_safetensors_signal(&resolved, "", &row.path);
    // MLX dirs are safetensors-shaped; the token marker plus the dir
    // gate is the discriminator the router uses.
    let mlx = row.is_mlx() && safetensors;

    // Lane + kind via the same decision the CLI/router make. The reason
    // string is derived from the INPUTS (facts), not from inside
    // serving_lane. The global tag rides along so the no-lane case can
    // still name the engine that WOULD serve (mirrors routed_engine_lane).
    let lane: Option<LaneLookup> = state.with_store(|s| {
        let engine_rows = s.list_engines().unwrap_or_default();
        let global = engine_rows
            .iter()
            .find(|r| r.active)
            .map(|r| (r.tag.clone(), r.kind));
        let Some((g_tag, g_kind)) = global else {
            return (
                None,
                Err("no engine installed — blazar engine install --kind <kind>".into()),
            );
        };
        let installed: Vec<(
            String,
            blazar_core::engine_kind::EngineKind,
            blazar_core::engine_kind::LaneClass,
        )> = engine_rows
            .iter()
            .map(|r| (r.tag.clone(), r.kind, r.lane_class()))
            .collect();
        let verdict = blazar_core::engine_kind::serving_lane(
            state.config.engine_routing.mode,
            state.config.engine_routing.policy,
            overlay.engine.as_deref(),
            diffusion,
            safetensors,
            quantized,
            mlx,
            g_kind,
            &installed,
        );
        (Some(g_tag), verdict)
    });
    let engine = match lane {
        Some((_g_tag, Ok(Some((tag, kind))))) => json!({
            "tag": tag,
            "kind": format!("{kind:?}").to_lowercase(),
            "source": if overlay.engine.is_some() { "pin" } else { "auto" },
            "reason": lane_reason(
                state.config.engine_routing.mode,
                overlay.engine.as_deref(),
                diffusion,
                safetensors,
                quantized,
                state.config.engine_routing.policy,
            ),
        }),
        Some((g_tag, Ok(None))) => json!({
            "tag": g_tag,
            "kind": Value::Null,
            "source": "global",
            "reason": "no per-model lane resolved — the global active engine serves it",
        }),
        Some((_, Err(teach))) => json!({
            "tag": Value::Null,
            "kind": Value::Null,
            "source": "error",
            "reason": teach,
        }),
        // with_store returned None: the store was open for the model row
        // moments ago and closed under us — surface, never guess.
        None => json!({
            "tag": Value::Null,
            "kind": Value::Null,
            "source": "error",
            "reason": "store unavailable — engine rows unreadable",
        }),
    };

    // Context: requested (config precedence) vs effective (what a live
    // child actually holds — admission_ctx's discipline, stated with
    // sources).
    let requested = state.config.effective_ctx(&resolved);
    let resident_ctx = state
        .sup
        .ps()
        .into_iter()
        .find(|p| p.name == resolved && p.ctx > 0)
        .map(|p| p.ctx);
    let (effective, effective_source, limitation) = match resident_ctx {
        Some(c) => {
            let lim = if c < requested {
                "live child holds less ctx than requested (auto-fit divided it across slots, or a VRAM-budget respawn)"
            } else {
                "none observed"
            };
            (c, "live child argv", lim)
        }
        None => (requested, cold_ctx_note(), "unknown until first spawn"),
    };
    let context = json!({
        "requested": requested,
        "requested_source": requested_ctx_source(overlay.ctx.is_some(), &resolved, state.config.default_ctx),
        "effective": effective,
        "effective_source": effective_source,
        "limitation": limitation,
    });

    // Slots: live census when resident; honest "sized at spawn" otherwise.
    // PsRow slot fields are Option (mid-reshape / slot-less engines) — the
    // card passes them through rather than guessing.
    let residents: Vec<Value> = state
        .sup
        .ps()
        .into_iter()
        .filter(|p| p.name == resolved)
        .map(|p| {
            json!({
                "engine": p.engine,
                "state": p.state,
                "slots": p.slots,
                "slots_configured": p.slots_configured,
                "in_flight": p.in_flight,
                "ctx": p.ctx,
                "gpu": p.gpu,
                "device": p.device,
                "pid": p.pid,
                "spec_mode": p.spec_mode,
                "draft": p.draft,
                "warnings": p.warnings,
            })
        })
        .collect();
    let live = residents.iter().filter_map(|r| r["slots"].as_u64()).max();
    let live_inflight: i64 = residents
        .iter()
        .filter_map(|r| r["in_flight"].as_i64())
        .sum();
    let slots = json!({
        "live": live,
        "live_in_flight": live_inflight,
        "adaptive_slots": state.config.adaptive_slots,
        "note": if live.is_some() { "" } else { "slots = 0 (auto) is sized at spawn from device capacity" },
    });

    // Speculation: a live spawn's receipts when resident (its own
    // spec_mode + draft), else the STATIC tier (overlay > config). The
    // per-request override and the spec governor sit above both — say so,
    // never pretend the static answer is the whole truth.
    let live_spec = residents
        .iter()
        .find(|r| !r["spec_mode"].as_str().unwrap_or("").is_empty())
        .map(|r| (r["spec_mode"].clone(), r["draft"].clone()));
    let static_mode =
        blazar_runtime::resolve_spec_mode(None, overlay.spec.clone(), &state.config.spec);
    let static_draft = state
        .with_store(|s| blazar_runtime::resolve_draft_path(s, &resolved, &static_mode))
        .flatten();
    let (mode, draft, source) = match &live_spec {
        Some((m, d)) => (m.clone(), d.clone(), "live child".to_string()),
        None => (
            static_mode.clone().into(),
            static_draft.clone().map_or(Value::Null, Value::String),
            if overlay.spec.is_some() {
                format!("model_overrides.{resolved}.spec")
            } else {
                format!("config spec = {}", state.config.spec)
            },
        ),
    };
    let seeks_draft = matches!(static_mode.as_str(), "auto" | "eagle3" | "eagle");
    let speculation = json!({
        "mode": mode,
        "source": source,
        "draft_model": draft,
        "note": if seeks_draft && static_draft.is_none() && live_spec.is_none() {
            "mode seeks an external draft model but none is pulled (catalog pair missing or not pulled)"
        } else {
            "per-request options.spec / X-Blazar-Spec and the spec governor can override this at spawn"
        },
    });

    // Cache: KV quant grades (empty = the engine's auto ladder decides)
    // and the semantic cache config surface.
    let (kv_k, kv_v) = state.config.effective_cache_type_kv(&resolved);
    let cache = json!({
        "kv_k": if kv_k.is_empty() { "auto ladder".into() } else { kv_k },
        "kv_v": if kv_v.is_empty() { "auto ladder".into() } else { kv_v },
        "semantic_cache": {
            "enabled": state.config.semantic_cache.enabled,
            "model": state.config.semantic_cache.model,
            "threshold": state.config.semantic_cache.threshold,
            "ttl_secs": state.config.semantic_cache.ttl_secs,
        },
    });

    let admission = json!({
        "gateway_requests_inflight": state.requests.inflight_count(),
    });

    let card = json!({
        "object": "blazar.explain",
        "model": {
            "name": row.name,
            "repo": row.repo,
            "quant": row.quant,
            "bytes": row.bytes,
            "arch": row.arch,
            "params_b": row.params,
            "ctx_train": row.ctx_train,
            "shards": row.shards,
            "format": if safetensors { "safetensors-dir" } else { "gguf" },
            "mmproj": row.mmproj_path.is_some(),
            "path": row.path,
        },
        "engine": engine,
        "context": context,
        "slots": slots,
        "speculation": speculation,
        "cache": cache,
        "admission": admission,
        "residents": residents,
    });
    (StatusCode::OK, axum::Json(card)).into_response()
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;

    #[test]
    fn unit__lane_reason__states_input_facts_per_branch() {
        use blazar_core::config::{RoutingMode, RoutingPolicy as P};
        let r = |mode, pin, d, s, q, p| lane_reason(mode, pin, d, s, q, p);
        assert_eq!(
            r(RoutingMode::Manual, None, false, false, false, P::Quality),
            "engine_routing.mode = manual — the active engine serves every model"
        );
        assert_eq!(
            r(
                RoutingMode::Auto,
                Some("x"),
                false,
                false,
                false,
                P::Quality
            ),
            "model_overrides.<model>.engine pins the lane"
        );
        assert_eq!(
            r(RoutingMode::Auto, None, true, false, false, P::Quality),
            "diffusion component set routes to the sdcpp lane"
        );
        let s = r(RoutingMode::Auto, None, false, true, true, P::Quality);
        assert!(s.starts_with("quantized safetensors dir under"));
        let s = r(RoutingMode::Auto, None, false, true, false, P::Latency);
        assert!(s.starts_with("HF safetensors dir under"));
        assert_eq!(
            r(RoutingMode::Auto, None, false, false, false, P::Quality),
            "GGUF file with no overlay pin — the global active lane serves it"
        );
    }

    #[test]
    fn unit__ctx_sources__overlay_vs_default_and_cold_note() {
        assert_eq!(
            requested_ctx_source(true, "m", 8192),
            "model_overrides.m.ctx"
        );
        assert_eq!(
            requested_ctx_source(false, "m", 8192),
            "config default_ctx = 8192"
        );
        assert!(cold_ctx_note().contains("spawn-time planner"));
    }
}
