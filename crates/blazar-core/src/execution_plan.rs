//! Canonical execution planning - the typed, explainable decision layer
//! over the lane router.
//!
//! [`crate::engine_kind::serving_lane_typed`] answers "which installed
//! lane serves this model's format shape under this routing policy".
//! That verdict is correct but thin: it carries no context size, no
//! slot posture, no KV/spec feature decisions, no confidence, and no
//! record of the alternatives it rejected. Everything downstream
//! (`blazar plan`, the explain card, fit previews) re-derives those
//! facets ad hoc, so they can drift from the spawn path.
//!
//! This module is the single place that composes the FULL decision:
//! selection comes from the same `serving_lane_typed` the supervisor
//! consults (planner and spawn can never disagree), feature decisions
//! come from the capability matrix evidence statuses, and every facet
//! carries a reason plus a confidence class. The plan is a model of
//! the decision - callers compile it to engine argv separately.
//!
//! Purity contract: inputs are caller-resolved (store snapshot, probed
//! hardware, effective config) exactly like [`crate::optimize`]; the
//! builder has no clock, no filesystem walks, no spawn.

use crate::capability_matrix::{CapabilityEvidence, lane_capabilities};
use crate::config::{RoutingMode, RoutingPolicy};
use crate::engine_kind::{EngineKind, FormatShape, LaneClass, serving_lane_typed};
use crate::store::ModelRow;
use serde::Serialize;

/// Hardware budget facts the planner reasons about. Caller probes
/// (`blazar_runtime::probe_hardware`) and hands the summary in; core
/// stays free of runtime dependencies. `None` fields mean "not
/// probed" - the plan states the absence, never guesses a number.
#[derive(Debug, Clone, Default)]
pub struct HardwareBudget {
    /// Live accelerator name, when a probe saw one.
    pub gpu_name: Option<String>,
    /// Total VRAM (or unified memory) in bytes.
    pub vram_bytes: Option<u64>,
    /// System RAM in bytes.
    pub ram_bytes: Option<u64>,
}

/// Everything the planner needs, resolved by the caller. Mirrors the
/// explicit-inputs convention of [`crate::optimize::OptimizeInputs`].
pub struct ExecutionPlanInputs<'a> {
    pub row: &'a ModelRow,
    /// Installed lanes: `(tag, kind, lane_class)` - the same roster
    /// shape `serving_lane_typed` consumes.
    pub installed: &'a [(String, EngineKind, LaneClass)],
    /// The active (global) engine kind - what manual mode falls back to.
    pub global_kind: EngineKind,
    /// The active engine row's tag, when one exists - the lane that
    /// serves both manual mode and auto-routes that land on the global
    /// kind (`serving_lane` returns `Ok(None)` for those).
    pub global_tag: Option<&'a str>,
    pub mode: RoutingMode,
    pub policy: RoutingPolicy,
    /// `model_overrides.<model>.engine` pin, when set.
    pub pin: Option<&'a str>,
    pub shape: FormatShape,
    pub hardware: HardwareBudget,
    /// Effective `resources.default_ctx` from config (or a tuned
    /// profile's pinned `-c`), already resolved by the caller.
    pub config_ctx: u32,
    /// Effective `slots` (0 = auto-sized at spawn).
    pub config_slots: u32,
    /// Effective `spec` mode literal ("auto", "off", "ngram", ...).
    pub spec_mode: &'a str,
    /// Effective KV `cache_type` literal ("" = auto ladder).
    pub cache_type: &'a str,
    /// Caller-resolved: a bench receipt exists for THIS model on the
    /// lane the router picks (presence is the confidence axis; age and
    /// perf parsing belong to `optimize::build_plan`).
    pub has_bench_receipt: bool,
    /// Epoch seconds (caller-supplied; the builder has no clock).
    pub now: i64,
}

/// How the selected lane was chosen - the explain axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanSource {
    /// A `model_overrides.<model>.engine` pin won.
    UserPin,
    /// Auto mode routed on format shape + routing policy.
    RoutedPolicy,
    /// Manual mode (or policy fell back) - the global engine serves.
    GlobalEngine,
}

/// Slot posture for the plan.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanSlots {
    /// `slots = 0`: the spawn path capacity-sizes against live VRAM;
    /// `hint` is the planner's coarse predicted range, labeled as a
    /// prediction, never a promise.
    Auto {
        hint: String,
    },
    Pinned(u32),
}

/// KV-cache quantization decision.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanKv {
    /// `cache_type = ""`: the ladder picks at spawn from live memory
    /// pressure; eligibility comes from the capability matrix.
    AutoLadder { eligible: bool, datum: String },
    /// User/model pinned a concrete type pair (e.g. `q8_0`).
    Pinned { cache_type: String, eligible: bool },
}

/// Speculative decoding decision.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanSpec {
    /// `spec = "auto"`: family selection deferred to receipts/governor.
    Auto {
        supported: bool,
        datum: String,
    },
    Off,
    Pinned {
        mode: String,
        supported: bool,
    },
}

/// Evidence-backed confidence in the whole plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanConfidence {
    /// Pin or routed lane with a current bench receipt for this model.
    High,
    /// Routed on format truth, performance unmeasured (predicted).
    Medium,
    /// Manual/global fallback, or nothing can serve the shape.
    Low,
}

/// One installed lane that was considered and not selected.
#[derive(Debug, Clone, Serialize)]
pub struct RejectedLane {
    pub tag: String,
    pub kind: EngineKind,
    /// Plain sentence: format-unservable, policy-preferred another
    /// lane, or manual mode kept the global engine.
    pub why: String,
}

/// A complete, explainable execution decision for one model.
#[derive(Debug, Serialize)]
pub struct ExecutionPlan {
    pub model: String,
    /// Selected lane; `tag = None` means nothing installed can serve
    /// the shape (the teaching error travels in `warnings`).
    pub tag: Option<String>,
    pub engine: EngineKind,
    pub source: PlanSource,
    /// Planned context size (already clamped to the model's train
    /// window when both are known).
    pub context_size: u32,
    pub slots: PlanSlots,
    pub kv: PlanKv,
    pub speculation: PlanSpec,
    pub confidence: PlanConfidence,
    /// Datum sentences behind the decisions, in decision order.
    pub reasons: Vec<String>,
    /// Problems the operator should know about (clamps, missing
    /// hardware facts, unservable shapes, unproven features).
    pub warnings: Vec<String>,
    /// Other installed lanes that were considered.
    pub alternatives: Vec<RejectedLane>,
}

/// Family lookup helper - the capability matrix rows carry `family`
/// strings; missing rows read as "no evidence" rather than failing.
fn lane_evidence(kind: EngineKind, family: &str) -> Option<CapabilityEvidence> {
    lane_capabilities(kind)
        .iter()
        .find(|row| row.family == family)
        .map(|row| row.evidence)
}

// one flat decision pass per facet: splitting it would scatter the
// planning policy across fns without shrinking any facet
#[allow(clippy::too_many_lines)]
/// Build the canonical plan. Deterministic over the inputs; every
/// evidence gap degrades to an honest warning, never to a fabricated
/// default.
pub fn plan_execution(input: &ExecutionPlanInputs<'_>) -> Result<ExecutionPlan, String> {
    let mut reasons: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    // ---- lane selection: the SAME router the spawn consults --------
    let lane = serving_lane_typed(
        input.mode,
        input.policy,
        input.pin,
        input.shape,
        input.global_kind,
        input.installed,
    )
    .map_err(|e| e.to_string())?;

    let (tag, kind, source) = if let Some((tag, kind)) = &lane {
        if let Some(pin) = input.pin {
            reasons.push(format!(
                "engine pin '{pin}' resolved to lane {tag} ({kind:?})"
            ));
            (Some(tag.clone()), *kind, PlanSource::UserPin)
        } else {
            reasons.push(format!(
                "auto routing on {:?} under policy {:?} selected lane {tag} ({kind:?})",
                input.shape.shards, input.policy
            ));
            (Some(tag.clone()), *kind, PlanSource::RoutedPolicy)
        }
    } else {
        // `Ok(None)` covers two cases the daemon treats identically
        // (the global engine serves): manual mode, and auto routing
        // landing on the global kind. The reason line keeps them
        // distinguishable for the operator.
        if input.mode == RoutingMode::Auto {
            reasons.push(format!(
                "auto routing on {:?} landed on the global engine ({:?}) - no adapter lane needed",
                input.shape.shards, input.global_kind
            ));
        } else {
            reasons.push(format!(
                "manual mode keeps the global engine ({:?})",
                input.global_kind
            ));
        }
        (
            input.global_tag.map(str::to_string),
            input.global_kind,
            PlanSource::GlobalEngine,
        )
    };

    // ---- alternatives: every other installed lane, with a reason ---
    let mut alternatives: Vec<RejectedLane> = Vec::new();
    for (other_tag, other_kind, _class) in input.installed {
        if Some(other_tag.as_str()) == tag.as_deref() {
            continue;
        }
        let servable =
            EngineKind::route_format(input.shape, &[*other_kind], input.policy).is_some();
        let why = if !servable {
            format!("cannot decode {:?} shards", input.shape.shards)
        } else if source == PlanSource::UserPin {
            "could serve this shape, but the engine pin wins".to_string()
        } else if source == PlanSource::GlobalEngine {
            "could serve this shape, but manual mode keeps the global engine".to_string()
        } else {
            format!(
                "could serve this shape; policy {:?} prefers the selected lane",
                input.policy
            )
        };
        alternatives.push(RejectedLane {
            tag: other_tag.clone(),
            kind: *other_kind,
            why,
        });
    }

    // ---- context: effective config clamped to the train window -----
    let mut context_size = input.config_ctx;
    let train = input.row.ctx_train.unwrap_or(0);
    if train > 0 && i64::from(context_size) > train {
        context_size = u32::try_from(train).unwrap_or(u32::MAX);
        warnings.push(format!(
            "requested context {} exceeds the model's train window {train} - clamped",
            input.config_ctx
        ));
    }
    if context_size == 0 {
        warnings.push(
            "context size unresolved (no config default, no train window) - spawn decides"
                .to_string(),
        );
    }
    reasons.push(format!(
        "context {context_size} (effective config, clamped to train window)"
    ));

    // ---- slots ------------------------------------------------------
    let slots = if input.config_slots > 0 {
        reasons.push(format!(
            "slots {} pinned by config/profile - capacity autosizing stays off",
            input.config_slots
        ));
        PlanSlots::Pinned(input.config_slots)
    } else {
        let hint = match (
            input.hardware.vram_bytes,
            u64::try_from(input.row.bytes).ok(),
        ) {
            (Some(vram), Some(bytes)) if vram > 0 && bytes > 0 => {
                // Coarse VRAM headroom split: weights resident plus a
                // per-slot KV allowance. Labeled a prediction - the
                // spawn path's capacity math owns the real number.
                let headroom = vram.saturating_sub(bytes);
                let per_slot = bytes.max(64 * 1024 * 1024);
                let n = (headroom / per_slot).clamp(1, 4);
                format!("predicted {n} (VRAM headroom split, spawn re-sizes)")
            }
            _ => "unknown - hardware not probed; spawn capacity-sizes".to_string(),
        };
        reasons
            .push("slots auto (slots = 0): spawn path capacity-sizes against live memory".into());
        PlanSlots::Auto { hint }
    };

    // ---- KV cache quantization --------------------------------------
    let kv_ev = lane_evidence(kind, "kv_cache_quantization");
    let eligible = matches!(
        kv_ev,
        Some(CapabilityEvidence::LiveCertified | CapabilityEvidence::FirstClass)
    );
    let kv = if input.cache_type.is_empty() {
        let datum = match kv_ev {
            Some(e) => format!("lane capability: {e:?}"),
            None => "no capability row for this lane - treated as ineligible".to_string(),
        };
        if !eligible && kv_ev.is_some() {
            warnings.push(format!(
                "KV quant ladder requested (cache_type = \"\") but lane {kind:?} evidence is {kv_ev:?} - ladder stays on f16"
            ));
        }
        reasons.push("KV cache type auto: spawn ladder picks from live memory pressure".into());
        PlanKv::AutoLadder { eligible, datum }
    } else {
        let ct = input.cache_type.to_string();
        reasons.push(format!("KV cache type pinned to {ct} by config/profile"));
        if !eligible {
            warnings.push(format!(
                "KV cache type {ct} pinned but lane {kind:?} evidence is {kv_ev:?} - spawn may reject or ignore it"
            ));
        }
        PlanKv::Pinned {
            cache_type: ct,
            eligible,
        }
    };

    // ---- speculative decoding ---------------------------------------
    let spec_ev = lane_evidence(kind, "speculative_decoding");
    let supported = !matches!(spec_ev, Some(CapabilityEvidence::Unsupported) | None);
    let speculation = if input.spec_mode.eq_ignore_ascii_case("off") {
        reasons.push("speculation off by config".into());
        PlanSpec::Off
    } else {
        let datum = match spec_ev {
            Some(CapabilityEvidence::Unsupported) => "lane declares speculation unsupported".into(),
            Some(e) => format!("lane capability: {e:?}"),
            None => "no capability row - treated as unsupported".to_string(),
        };
        if !supported {
            warnings.push(format!(
                "spec = '{}' but lane {kind:?} has no supported speculation family - running dense",
                input.spec_mode
            ));
        }
        let auto = input.spec_mode.eq_ignore_ascii_case("auto");
        reasons.push(if auto {
            "spec auto: family selection defers to receipts and the live governor".into()
        } else {
            format!("spec mode '{}' from config/profile", input.spec_mode)
        });
        if auto {
            PlanSpec::Auto { supported, datum }
        } else {
            PlanSpec::Pinned {
                mode: input.spec_mode.to_string(),
                supported,
            }
        }
    };

    // ---- confidence --------------------------------------------------
    // Auto-routing that lands on the global kind is still format-proven
    // (`route_format` returned the global kind); manual mode is not.
    let routed_to_global = lane.is_none() && input.mode == RoutingMode::Auto;
    let confidence = match (source, routed_to_global, input.has_bench_receipt) {
        (PlanSource::UserPin, _, _) | (_, true, true) | (PlanSource::RoutedPolicy, _, true) => {
            PlanConfidence::High
        }
        (_, true, false) | (PlanSource::RoutedPolicy, _, false) => PlanConfidence::Medium,
        (PlanSource::GlobalEngine, false, _) => PlanConfidence::Low,
    };
    match confidence {
        PlanConfidence::High => reasons.push(
            "confidence high: selection is pinned or backed by a bench receipt for this model on this lane"
                .into(),
        ),
        PlanConfidence::Medium => reasons.push(
            "confidence medium: routed on format truth, no bench receipt yet - performance is predicted"
                .into(),
        ),
        PlanConfidence::Low => reasons.push(
            "confidence low: manual/global fallback (or nothing installed can serve the shape)"
                .into(),
        ),
    }
    if source == PlanSource::GlobalEngine && tag.is_none() {
        warnings.push(
            "no active engine row - nothing is installed to serve this model; install one first"
                .into(),
        );
    }

    Ok(ExecutionPlan {
        model: input.row.name.clone(),
        tag,
        engine: kind,
        source,
        context_size,
        slots,
        kv,
        speculation,
        confidence,
        reasons,
        warnings,
        alternatives,
    })
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::engine_kind::ShardFormat;
    use crate::store::ModelRow;

    fn row(name: &str) -> ModelRow {
        ModelRow {
            name: name.into(),
            repo: "test/org".into(),
            quant: "Q4_K_M".into(),
            path: format!("/models/{name}.gguf"),
            bytes: 400 * 1024 * 1024,
            sha256: None,
            mmproj_path: None,
            components: Vec::new(),
            shards: 1,
            arch: Some("qwen".into()),
            params: Some(0.5),
            ctx_train: Some(32_768),
            pulled_at: 0,
            last_used_at: 0,

            license: None,
        }
    }

    fn installed() -> Vec<(String, EngineKind, LaneClass)> {
        vec![
            (
                "b1-cuda".into(),
                EngineKind::LlamaCpp,
                LaneClass::Mainstream,
            ),
            ("mlx-1".into(), EngineKind::Mlx, LaneClass::Mainstream),
        ]
    }

    fn inputs_for<'a>(
        row: &'a ModelRow,
        installed: &'a [(String, EngineKind, LaneClass)],
    ) -> ExecutionPlanInputs<'a> {
        ExecutionPlanInputs {
            row,
            installed,
            global_kind: EngineKind::LlamaCpp,
            global_tag: Some("b1-cuda"),
            mode: RoutingMode::Auto,
            policy: RoutingPolicy::Quality,
            pin: None,
            shape: FormatShape {
                diffusion: false,
                shards: ShardFormat::GgufFile,
            },
            hardware: HardwareBudget::default(),
            config_ctx: 8_192,
            config_slots: 0,
            spec_mode: "auto",
            cache_type: "",
            has_bench_receipt: false,
            now: 0,
        }
    }

    #[test]
    fn unit__plan_execution__gguf_routes_to_llamacpp_with_reasons() {
        let r = row("m");
        let lanes = installed();
        let plan = plan_execution(&inputs_for(&r, &lanes)).unwrap();
        assert_eq!(plan.engine, EngineKind::LlamaCpp);
        // GGUF + global llamacpp: the router needs no adapter lane, so
        // the GLOBAL row serves - tag present, source GlobalEngine,
        // still format-proven (Medium confidence without a receipt).
        assert_eq!(plan.tag.as_deref(), Some("b1-cuda"));
        assert_eq!(plan.source, PlanSource::GlobalEngine);
        assert_eq!(plan.context_size, 8_192);
        assert!(matches!(plan.slots, PlanSlots::Auto { .. }));
        assert!(plan.reasons.iter().any(|s| s.contains("global engine")));
        assert_eq!(plan.confidence, PlanConfidence::Medium);
    }

    #[test]
    fn unit__plan_execution__pin_wins_and_names_the_pin() {
        let r = row("m");
        let lanes = installed();
        let mut inp = inputs_for(&r, &lanes);
        inp.pin = Some("mlx-1");
        let plan = plan_execution(&inp).unwrap();
        assert_eq!(plan.engine, EngineKind::Mlx);
        assert_eq!(plan.source, PlanSource::UserPin);
        assert_eq!(plan.confidence, PlanConfidence::High);
    }

    #[test]
    fn unit__plan_execution__ctx_clamped_to_train_window_and_warns() {
        let r = row("m");
        let lanes = installed();
        let mut inp = inputs_for(&r, &lanes);
        inp.config_ctx = 131_072;
        let plan = plan_execution(&inp).unwrap();
        assert_eq!(plan.context_size, 32_768);
        assert!(plan.warnings.iter().any(|w| w.contains("clamped")));
    }

    #[test]
    fn unit__plan_execution__kv_ladder_reads_capability_matrix() {
        let r = row("m");
        let lanes = installed();
        let plan = plan_execution(&inputs_for(&r, &lanes)).unwrap();
        // llamacpp kv_cache_quantization is LiveCertified in the matrix.
        match &plan.kv {
            PlanKv::AutoLadder { eligible, .. } => assert!(eligible),
            PlanKv::Pinned { .. } => panic!("expected AutoLadder, got Pinned"),
        }
        // mlx pin: kv evidence exists (LiveCertified) too.
        let mut inp = inputs_for(&r, &lanes);
        inp.pin = Some("mlx-1");
        let plan = plan_execution(&inp).unwrap();
        assert!(matches!(
            &plan.kv,
            PlanKv::AutoLadder { eligible: true, .. }
        ));
    }

    #[test]
    fn unit__plan_execution__spec_unsupported_lane_warns_and_runs_dense() {
        let r = row("m");
        // vllm declares speculation Unsupported in the matrix.
        let lanes = vec![(
            "vllm-1".to_string(),
            EngineKind::Vllm,
            LaneClass::Mainstream,
        )];
        let mut inp = inputs_for(&r, &lanes);
        inp.shape = FormatShape {
            diffusion: false,
            shards: ShardFormat::SafetensorsDir,
        };
        inp.spec_mode = "auto";
        let plan = plan_execution(&inp).unwrap();
        assert_eq!(plan.engine, EngineKind::Vllm);
        match &plan.speculation {
            PlanSpec::Auto { supported, .. } => assert!(!supported),
            other => panic!("expected Auto, got {other:?}"),
        }
        assert!(plan.warnings.iter().any(|w| w.contains("running dense")));
    }

    #[test]
    fn unit__plan_execution__alternatives_explain_rejections() {
        let r = row("m");
        let lanes = installed();
        let plan = plan_execution(&inputs_for(&r, &lanes)).unwrap();
        let mlx = plan
            .alternatives
            .iter()
            .find(|a| a.kind == EngineKind::Mlx)
            .expect("mlx listed as alternative");
        assert!(mlx.why.contains("cannot decode"));
    }

    #[test]
    fn unit__plan_execution__unservable_shape_is_err_with_teaching() {
        let r = row("m");
        let lanes = vec![("mlx-1".to_string(), EngineKind::Mlx, LaneClass::Mainstream)];
        let mut inp = inputs_for(&r, &lanes);
        inp.shape = FormatShape {
            diffusion: false,
            shards: ShardFormat::GgufFile,
        };
        let err = plan_execution(&inp).unwrap_err();
        assert_ne!(err, "");
    }

    #[test]
    fn unit__plan_execution__slots_pin_and_vram_hint() {
        let r = row("m");
        let lanes = installed();
        let mut inp = inputs_for(&r, &lanes);
        inp.config_slots = 3;
        let plan = plan_execution(&inp).unwrap();
        assert!(matches!(plan.slots, PlanSlots::Pinned(3)));

        let mut inp = inputs_for(&r, &lanes);
        inp.config_slots = 0;
        inp.hardware = HardwareBudget {
            gpu_name: Some("RTX 4070".into()),
            vram_bytes: Some(8 * 1024 * 1024 * 1024),
            ram_bytes: None,
        };
        let plan = plan_execution(&inp).unwrap();
        match &plan.slots {
            PlanSlots::Auto { hint } => assert!(hint.contains("predicted")),
            PlanSlots::Pinned(_) => panic!("expected Auto, got Pinned"),
        }
    }
}
