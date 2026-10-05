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
use serde_json::{Value, json};

use crate::state::AppState;
use blazar_core::config::{RoutingMode, RoutingPolicy};
use blazar_core::engine_kind::ShardFormat;

/// Reason the routing lane resolved the way it did, stated as facts about
/// the inputs (never invented internals of `serving_lane`). Pure so the
/// wording is pinnable without a daemon.
#[must_use]
pub fn lane_reason(
    mode: RoutingMode,
    pin: Option<&str>,
    shape: blazar_core::engine_kind::FormatShape,
    policy: RoutingPolicy,
) -> String {
    if matches!(mode, RoutingMode::Manual) {
        return "engine_routing.mode = manual — the active engine serves every model".into();
    }
    if pin.is_some() {
        return "model_overrides.<model>.engine pins the lane".into();
    }
    if shape.diffusion {
        return "diffusion component set routes to the sdcpp lane".into();
    }
    if shape.shards == ShardFormat::MlxLayout {
        // Checked before the quantized-safetensors arm, mirroring
        // route_format: only mlx-lm decodes MLX dirs, so the format
        // forces the lane ahead of any policy preference.
        return "MLX quant dir — the mlx lane is the only decoder, format-forced ahead of policy"
            .into();
    }
    if shape.shards == ShardFormat::QuantizedSafetensors {
        return format!(
            "quantized safetensors dir under engine_routing.policy = {policy:?} picks the lane"
        );
    }
    if shape.shards.is_dir() {
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

/// The honest quant label for an MLX dir: the quantization block in
/// config.json carries the real `bits/group_size`. The store's `quant`
/// column only records tensor storage dtype (e.g. BF16), which says
/// nothing about the MLX quant strength shown to the user.
#[must_use]
pub fn mlx_quant_display(path: &str) -> String {
    let block = std::fs::read(std::path::Path::new(path).join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|c| c.get("quantization").cloned());
    match block {
        Some(q) => {
            let bits = q.get("bits").and_then(serde_json::Value::as_u64);
            let group = q.get("group_size").and_then(serde_json::Value::as_u64);
            match (bits, group) {
                (Some(b), Some(g)) => format!("MLX {b}-bit (group {g})"),
                (Some(b), None) => format!("MLX {b}-bit"),
                _ => "MLX (config.json quantization block unrecognized)".into(),
            }
        }
        None => "MLX (unquantized f16/bf16 weights)".into(),
    }
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
            );
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
    // MLX dirs are safetensors-shaped; the token marker plus the
    // dir gate is the discriminator the router uses (folded inside
    // detect, which keeps the MLX-beats-quantized cascade in one place).
    let shape = blazar_core::engine_kind::FormatShape {
        diffusion: row.has_component_set(),
        shards: blazar_core::engine_kind::ShardFormat::detect(
            safetensors,
            blazar_core::store::quantized_safetensors_signal(&resolved, "", &row.path),
            row.is_mlx(),
        ),
    };

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
            shape,
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
                shape,
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
    let context = if shape.shards == ShardFormat::MlxLayout {
        // mlx_lm owns the context window: the gateway passes no ctx flag
        // (compile_mlx ships ctx 0) — the shrink-to-fit ladder is a
        // llama.cpp concept that does not govern this lane.
        json!({
            "requested": "runtime-managed",
            "requested_source": "mlx_lm owns the context window — no gateway ctx knob for this lane",
            "effective": "runtime-managed",
            "effective_source": "mlx_lm",
            "limitation": "kv-bits / kv-group-size / prompt-cache flags pass through model_overrides argv",
        })
    } else {
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
        json!({
            "requested": requested,
            "requested_source": requested_ctx_source(overlay.ctx.is_some(), &resolved, state.config.default_ctx),
            "effective": effective,
            "effective_source": effective_source,
            "limitation": limitation,
        })
    };

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
    // The KV ladder is a llama.cpp concept; on the mlx lane the runtime
    // manages KV and shaping flags ride argv passthrough.
    let (kv_key_label, kv_value_label) = if shape.shards == ShardFormat::MlxLayout {
        (
            "mlx runtime".to_string(),
            "kv-bits / kv-group-size via model_overrides argv".to_string(),
        )
    } else if kv_k.is_empty() {
        ("auto ladder".to_string(), "auto ladder".to_string())
    } else {
        (kv_k.clone(), kv_v.clone())
    };
    let cache = json!({
        "kv_k": kv_key_label,
        "kv_v": kv_value_label,
        "kv_advisory": kv_advisory(shape, kv_k.as_str(), requested, resident_ctx),
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
            "quant": if shape.shards == ShardFormat::MlxLayout { mlx_quant_display(&row.path) } else { row.quant },
            "bytes": row.bytes,
            "arch": row.arch,
            "params_b": row.params,
            "ctx_train": row.ctx_train,
            "shards": row.shards,
            "format": match shape.shards {
                ShardFormat::MlxLayout => "mlx-dir",
                ShardFormat::GgufFile => "gguf",
                ShardFormat::SafetensorsDir | ShardFormat::QuantizedSafetensors => {
                    "safetensors-dir"
                }
            },
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

/// C3 per-phase KV-quant advisory for the explain card. llama.cpp
/// documents the ladder as f16 (lossless, largest KV footprint) >
/// `q8_0` (near-lossless, roughly half the footprint) > `q4_0`
/// (measurable quality impact, quarter footprint). The advisory says
/// which tier the config sits at and when one step down pays; it never
/// invents byte counts (per-token KV size depends on the model's
/// layer/head geometry, which this card does not read).
fn kv_advisory(
    shape: blazar_core::engine_kind::FormatShape,
    kv_k: &str,
    requested_ctx: u32,
    resident_ctx: Option<u32>,
) -> Value {
    if shape.shards == blazar_core::engine_kind::ShardFormat::MlxLayout {
        return json!({
            "applies": false,
            "reason": "mlx runtime owns KV quantization (kv-bits / kv-group-size via argv)",
        });
    }
    let ctx_in_play = resident_ctx.unwrap_or(requested_ctx);
    let long_ctx = ctx_in_play >= 32_768;
    let (grade, next_step, why) = if kv_k.is_empty() {
        (
            "f16 (engine auto ladder)",
            Some("cache_type_k = \"q8_0\" + cache_type_v = \"q8_0\""),
            if long_ctx {
                "effective ctx >= 32k: q8_0 KV roughly halves the KV footprint with \
                 near-lossless quality — the standard first step when ctx or VRAM is tight"
            } else {
                "KV stays f16 unless ctx or VRAM pressure appears (q8_0 is the first \
                 step when it does)"
            },
        )
    } else if kv_k.eq_ignore_ascii_case("q8_0") {
        (
            "q8_0",
            Some("cache_type_* = \"q4_0\" (only when VRAM-bound and quality-tolerant)"),
            "already at the near-lossless tier; q4_0 trades measurable quality for a \
             quarter of the f16 footprint",
        )
    } else if kv_k.eq_ignore_ascii_case("q4_0") {
        (
            "q4_0",
            None,
            "most aggressive KV tier; stepping back up to q8_0 restores quality if \
             generations degrade",
        )
    } else {
        (
            kv_k,
            None,
            "custom cache_type_k grade; ladder advice does not apply",
        )
    };
    json!({
        "applies": true,
        "current": grade,
        "effective_ctx": ctx_in_play,
        "next_step": next_step,
        "why": why,
    })
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;

    use blazar_core::engine_kind::ShardFormat;

    /// Positional sugar for the routing-axis struct: domain gate
    /// first, shard shape second.
    fn fs(diffusion: bool, shards: ShardFormat) -> blazar_core::engine_kind::FormatShape {
        blazar_core::engine_kind::FormatShape { diffusion, shards }
    }

    #[test]
    fn unit__mlx_quant_display__reads_the_quantization_block() {
        let dir = std::env::temp_dir().join(format!("mlxq-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // quantized dir: bits + group_size
        std::fs::write(
            dir.join("config.json"),
            r#"{"quantization": {"bits": 4, "group_size": 64, "version": 2}}"#,
        )
        .unwrap();
        assert_eq!(
            mlx_quant_display(dir.to_str().unwrap()),
            "MLX 4-bit (group 64)"
        );
        // unquantized dir: no quantization block
        std::fs::write(dir.join("config.json"), r#"{"model_type": "qwen2"}"#).unwrap();
        assert_eq!(
            mlx_quant_display(dir.to_str().unwrap()),
            "MLX (unquantized f16/bf16 weights)"
        );
        // missing config.json entirely
        std::fs::remove_file(dir.join("config.json")).unwrap();
        assert_eq!(
            mlx_quant_display(dir.to_str().unwrap()),
            "MLX (unquantized f16/bf16 weights)"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn unit__lane_reason__mlx_dir_is_format_forced_before_policy() {
        use blazar_core::config::{RoutingMode, RoutingPolicy as P};
        // MLX arm must fire even when the quantized-safetensors shape would
        // otherwise claim the row for policy-based sglang routing — the
        // router checks the mlx axis first and so does this string.
        assert_eq!(
            lane_reason(
                RoutingMode::Auto,
                None,
                fs(false, ShardFormat::MlxLayout),
                P::Quality
            ),
            "MLX quant dir — the mlx lane is the only decoder, format-forced ahead of policy"
        );
    }

    #[test]
    fn unit__lane_reason__states_input_facts_per_branch() {
        use blazar_core::config::{RoutingMode, RoutingPolicy as P};
        let r = |mode, pin, d, f, p| lane_reason(mode, pin, fs(d, f), p);
        assert_eq!(
            r(
                RoutingMode::Manual,
                None,
                false,
                ShardFormat::GgufFile,
                P::Quality
            ),
            "engine_routing.mode = manual — the active engine serves every model"
        );
        assert_eq!(
            r(
                RoutingMode::Auto,
                Some("x"),
                false,
                ShardFormat::GgufFile,
                P::Quality
            ),
            "model_overrides.<model>.engine pins the lane"
        );
        assert_eq!(
            r(
                RoutingMode::Auto,
                None,
                true,
                ShardFormat::GgufFile,
                P::Quality
            ),
            "diffusion component set routes to the sdcpp lane"
        );
        let s = r(
            RoutingMode::Auto,
            None,
            false,
            ShardFormat::QuantizedSafetensors,
            P::Quality,
        );
        assert!(s.starts_with("quantized safetensors dir under"));
        let s = r(
            RoutingMode::Auto,
            None,
            false,
            ShardFormat::SafetensorsDir,
            P::Latency,
        );
        assert!(s.starts_with("HF safetensors dir under"));
        assert_eq!(
            r(
                RoutingMode::Auto,
                None,
                false,
                ShardFormat::GgufFile,
                P::Quality
            ),
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
