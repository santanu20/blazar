//! `blazar optimize --plan` — the deterministic DECIDE stage of the
//! optimization architecture (docs/research/2026-10-31 §7.6). Reads the
//! evidence store, names every gap it can prove, and orders the work
//! that would close it. Pure over the store: no spawns, no writes, no
//! clocks (the caller passes `now`), so the same builder serves the
//! read-only adviser today and the Wave-2 experiment runner later.
//!
//! Scope discipline: `autopilot` already covers fleet-wide bench/tune/
//! certificate *staleness*; this module goes deeper per model — receipt
//! provenance (engine tag, hardware fingerprint), the optimization
//! ledger's adoption pointer, KV-cache quality eligibility at the
//! serving context, and speculation-receipt qualification under the
//! same rules the supervisor applies at selection time.

use crate::CoreResult;
use crate::kv_quality;
use crate::store::{EngineRow, ModelRow, Store};

/// Finding severity. `Ok` rows prove the evidence is current; `Warn`
/// rows name a gap a recommendation can close; `Info` rows carry
/// context the operator should know but that needs no command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Ok,
    Info,
    Warn,
}

impl Severity {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Severity::Ok => "ok",
            Severity::Info => "info",
            Severity::Warn => "warn",
        }
    }
}

/// One evidence verdict: what was checked, what the store proved.
#[derive(Debug, Clone)]
pub struct OptimizeFinding {
    pub severity: Severity,
    /// Evidence domain: engine, bench, profile, kv, spec, cert, workload.
    pub area: &'static str,
    /// Plain sentence stating the verdict.
    pub text: String,
    /// The stored datum the verdict rests on (receipt age, tag, ratio…).
    pub datum: String,
}

/// One recommended command, in priority order (the Vec's order IS the
/// priority — no separate ranking axis to keep honest).
#[derive(Debug, Clone)]
pub struct OptimizeRec {
    /// The exact command the operator could type.
    pub cmd: String,
    /// Why this closes a named finding.
    pub why: String,
    /// The datum that made this the next move.
    pub datum: String,
    /// How many benchmark-bearing sessions this command will run
    /// (0 for non-benchmark work like a certificate probe).
    pub benchmark_sessions: u32,
}

/// A complete advise-only plan for one model.
#[derive(Debug, Clone)]
pub struct OptimizePlan {
    pub model: String,
    /// Active engine tag serving the lane, if any.
    pub engine_tag: Option<String>,
    /// Current hardware fingerprint derived from the engine manifest.
    pub hw_current: Option<String>,
    /// Context the findings were judged against (profile `-c`, else
    /// the config default).
    pub serving_ctx: u32,
    pub findings: Vec<OptimizeFinding>,
    pub recs: Vec<OptimizeRec>,
}

impl OptimizePlan {
    /// Sum of benchmark-bearing sessions across the recommendations —
    /// the operator's cost preview. Never estimates runs the commands
    /// themselves do not account for.
    #[must_use]
    pub fn benchmark_budget(&self) -> u32 {
        self.recs.iter().map(|r| r.benchmark_sessions).sum()
    }
}

/// Evidence inputs the CLI resolves before building the plan. Kept
/// explicit (rather than re-read inside) so unit tests seed stores and
/// pass exact values — and so the builder stays pure.
pub struct OptimizeInputs<'a> {
    pub row: &'a ModelRow,
    pub engine: Option<&'a EngineRow>,
    /// `resources.default_ctx` from the effective config — the serving
    /// context used when no tuned profile pins one.
    pub config_ctx: u32,
    /// Epoch seconds (caller-supplied; the builder has no clock).
    pub now: i64,
    /// Workload-verification floor: recent requests below this count
    /// make workload-representative verification (Wave 2) ineligible.
    pub workload_min_requests: i64,
}

/// Evidence-sufficiency threshold used by the CLI. Below this many
/// requests in the last 30 days the usage telemetry cannot stand for a
/// workload — the adviser says so instead of pretending otherwise.
pub const WORKLOAD_MIN_REQUESTS: i64 = 20;

/// Build the advise-only plan. Every rule is deterministic over the
/// store snapshot; unreadable evidence degrades to an honest `warn`
/// finding naming the error, never to silence.
pub fn build_plan(store: &Store, input: &OptimizeInputs<'_>) -> CoreResult<OptimizePlan> {
    let mut findings: Vec<OptimizeFinding> = Vec::new();
    let mut recs: Vec<OptimizeRec> = Vec::new();
    let row = input.row;
    let name = row.name.as_str();

    let (engine_tag, hw_current, tuneable) = match input.engine {
        None => {
            findings.push(OptimizeFinding {
                severity: Severity::Warn,
                area: "engine",
                text: "no active engine lane — tuning evidence cannot be produced or judged".into(),
                datum: "engines table holds no active row".into(),
            });
            recs.push(OptimizeRec {
                cmd: "blazar engine list".into(),
                why: "pick (or pull) an engine before benchmarking".into(),
                datum: "no active lane".into(),
                benchmark_sessions: 0,
            });
            (None, None, false)
        }
        Some(e) => {
            // Tuning/bench/spec surfaces drive llama-bench tooling —
            // only the llamacpp lane carries it. Other lanes get an
            // honest scope note, not an empty plan pretending success.
            let tuneable = e.kind == crate::engine_kind::EngineKind::LlamaCpp;
            if !tuneable {
                findings.push(OptimizeFinding {
                    severity: Severity::Info,
                    area: "engine",
                    text: format!(
                        "active lane {:?} has no tuning surface — optimize advises on llamacpp lanes only",
                        e.kind
                    ),
                    datum: format!("engine {} ({:?})", e.tag, e.kind),
                });
            }
            let hw = crate::store::EvidenceAxes::hw_fingerprint_from_manifest(&e.manifest);
            (Some(e.tag.clone()), hw, tuneable && is_gguf(row))
        }
    };

    // ---- bench receipt provenance ------------------------------------
    let mut bench_rec_pending = false;
    if tuneable {
        match store.latest_bench_result_with_axes(name, None) {
            Ok(None) => {
                findings.push(OptimizeFinding {
                    severity: Severity::Warn,
                    area: "bench",
                    text: "no benchmark receipt on record".into(),
                    datum: "bench_results empty for this model".into(),
                });
                bench_rec_pending = true;
            }
            Ok(Some((tag, _payload, at, axes))) => {
                let age_days = (input.now - at).max(0) / 86_400;
                let mut staleness: Option<String> = None;
                if Some(tag.as_str()) != engine_tag.as_deref() {
                    staleness = Some(format!(
                        "receipt from lane '{tag}' but '{opt}' is active — engine moved under the evidence",
                        opt = engine_tag.clone().unwrap_or_default()
                    ));
                } else if axes.hw_fingerprint.as_deref() != hw_current.as_deref() {
                    // `None` (pre-v13 row) is also a mismatch: the
                    // receipt cannot be attributed to THIS hardware.
                    staleness = Some(if axes.hw_fingerprint.is_none() {
                        "receipt predates hardware fingerprinting — cannot be attributed to this machine".into()
                    } else {
                        "receipt was measured on different hardware".into()
                    });
                }
                match staleness {
                    Some(why) => {
                        findings.push(OptimizeFinding {
                            severity: Severity::Warn,
                            area: "bench",
                            text: format!("benchmark receipt is stale: {why}"),
                            datum: format!("lane {tag}, {age_days}d old"),
                        });
                        bench_rec_pending = true;
                    }
                    None => {
                        findings.push(OptimizeFinding {
                            severity: Severity::Ok,
                            area: "bench",
                            text: "benchmark receipt current (lane + hardware attributed)".into(),
                            datum: format!("lane {tag}, {age_days}d old"),
                        });
                    }
                }
            }
            Err(e) => findings.push(unreadable("bench", e)),
        }
    }

    // ---- tuned profile + ledger adoption pointer ---------------------
    let mut serving_ctx = input.config_ctx;
    if tuneable && let Some(tag) = engine_tag.as_deref() {
        match store.get_profile(name, tag) {
            Ok(None) => {
                findings.push(OptimizeFinding {
                    severity: Severity::Warn,
                    area: "profile",
                    text: "no tuned launch profile for the active lane".into(),
                    datum: format!("profiles empty for ({name}, {tag})"),
                });
                recs.push(OptimizeRec {
                    cmd: format!("blazar tune {name} --search"),
                    why:
                        "grid-search the launch profile (KV grade, flash-attention, threads, batch)"
                            .into(),
                    datum: "no profile row".into(),
                    benchmark_sessions: 1,
                });
            }
            Ok(Some(p)) => {
                let argv = parse_argv_json(&p.args_json);
                if let Some(ctx) = argv_value(&argv, "-c").and_then(|v| v.parse().ok()) {
                    serving_ctx = ctx;
                }
                let age_days = (input.now - p.updated_at).max(0) / 86_400;
                match store.active_profile_of(name, tag) {
                    Ok(Some((exp_id, at))) => {
                        let adopt_days = (input.now - at).max(0) / 86_400;
                        findings.push(OptimizeFinding {
                            severity: Severity::Ok,
                            area: "profile",
                            text: format!(
                                "tuned profile adopted via optimization ledger experiment #{exp_id}"
                            ),
                            datum: format!(
                                "profile {age_days}d old, adopted {adopt_days}d ago, hash {}",
                                p.args_hash
                            ),
                        });
                    }
                    Ok(None) => findings.push(OptimizeFinding {
                        severity: Severity::Info,
                        area: "profile",
                        text: "tuned profile predates the v19 optimization ledger — its adoption is unpointed".into(),
                        datum: format!("profile {age_days}d old, hash {}", p.args_hash),
                    }),
                    Err(e) => findings.push(unreadable("profile", e)),
                }
            }
            Err(e) => findings.push(unreadable("profile", e)),
        }
    }

    // ---- KV-cache quality eligibility at the serving context ---------
    if tuneable {
        let grades = ["q8_0", "q5_0", "q4_0"];
        let evidenced: Vec<&str> = grades
            .iter()
            .copied()
            .filter(|g| kv_quality::grade_evidenced_at_ctx(name, g, serving_ctx))
            .collect();
        // The profile's own KV flags: compiled argv omits them when the
        // grade is the f16 default — absence IS the f16 statement.
        let argv_probe = engine_tag
            .as_deref()
            .and_then(|tag| store.get_profile(name, tag).ok().flatten())
            .map(|p| parse_argv_json(&p.args_json));
        let active_kv = argv_probe.as_deref().map(|argv| {
            (
                argv_value(argv, "-ctk").unwrap_or_else(|| "f16".into()),
                argv_value(argv, "-ctv").unwrap_or_else(|| "f16".into()),
            )
        });
        let unevidenced_active = active_kv.as_ref().is_some_and(|(k, v)| {
            [k, v]
                .iter()
                .any(|g| *g != "f16" && !kv_quality::grade_evidenced_at_ctx(name, g, serving_ctx))
        });
        if unevidenced_active {
            let (k, v) = active_kv.clone().unwrap_or_default();
            findings.push(OptimizeFinding {
                severity: Severity::Warn,
                area: "kv",
                text: format!(
                    "active profile serves quantized KV without quality evidence at ctx {serving_ctx} — the tune gate now excludes such grades, this profile predates it"
                ),
                datum: format!("kv {k}/{v} vs registry evidence at ctx {serving_ctx}"),
            });
        }
        if evidenced.is_empty() {
            findings.push(OptimizeFinding {
                severity: Severity::Info,
                area: "kv",
                text: format!(
                    "no quantized-KV quality evidence for this model — the tune gate keeps every search on f16 at ctx {serving_ctx}"
                ),
                datum: format!("registry has no receipts for {name} at ctx >= {serving_ctx}"),
            });
        } else {
            findings.push(OptimizeFinding {
                severity: Severity::Ok,
                area: "kv",
                text: format!(
                    "quantized-KV grades {} quality-evidenced at ctx {serving_ctx} — eligible to win a gated tune search",
                    evidenced.join(", ")
                ),
                datum: format!("ppl registry receipts for {name}"),
            });
        }
    }

    // ---- speculation receipt qualification ---------------------------
    if tuneable && let Some(tag) = engine_tag.as_deref() {
        match store.list_spec_receipts(name, tag) {
            Ok(rows) if !rows.is_empty() => {
                let dense = rows.iter().any(|r| r.mode == "dense");
                // Same qualification rules the supervisor applies at
                // selection time (best_measured_spec_mode): ratio floor
                // 1.05 and — for ngram — a tuned receipt. An untuned
                // ngram row can never be the adviser's "winner".
                let qualified: Vec<_> = rows
                    .iter()
                    .filter(|r| {
                        r.mode != "dense" && r.ratio >= 1.05 && (r.mode != "ngram" || r.tuned)
                    })
                    .collect();
                let untuned_ngram = rows
                    .iter()
                    .filter(|r| r.mode == "ngram" && !r.tuned)
                    .max_by(|a, b| a.ratio.total_cmp(&b.ratio));
                if let Some(best) = qualified.iter().max_by(|a, b| a.ratio.total_cmp(&b.ratio)) {
                    findings.push(OptimizeFinding {
                        severity: Severity::Ok,
                        area: "spec",
                        text: format!(
                            "speculation mode '{}' qualified for auto-selection ({:.2}x dense)",
                            best.mode, best.ratio
                        ),
                        datum: format!(
                            "{:.1} tok/s vs dense {:.1} tok/s",
                            best.tok_per_s, best.dense_tps
                        ),
                    });
                } else {
                    if let Some(r) = untuned_ngram {
                        findings.push(OptimizeFinding {
                            severity: Severity::Warn,
                            area: "spec",
                            text: format!(
                                "ngram receipt at {:.2}x dense is EXCLUDED from selection — raced engine defaults, never tuned knobs",
                                r.ratio
                            ),
                            datum: format!(
                                "mode ngram, tuned=false, {:.1} tok/s vs dense {:.1} tok/s",
                                r.tok_per_s, r.dense_tps
                            ),
                        });
                        recs.push(OptimizeRec {
                            cmd: format!("blazar tune {name} --ngram"),
                            why: "refine the ngram shape so the race measures a real candidate"
                                .into(),
                            datum: format!("untuned receipt at {:.2}x", r.ratio),
                            benchmark_sessions: 1,
                        });
                    }
                    findings.push(OptimizeFinding {
                        severity: if dense {
                            Severity::Info
                        } else {
                            Severity::Warn
                        },
                        area: "spec",
                        text: if dense {
                            "no speculation mode has qualified against dense (>= 1.05x, tuned)"
                                .into()
                        } else {
                            "speculation race never ran on this lane — no dense baseline either"
                                .into()
                        },
                        datum: format!(
                            "{} receipt(s) on lane {tag}, best qualified ratio {:.2}",
                            rows.len(),
                            qualified.iter().map(|r| r.ratio).fold(0.0_f64, f64::max)
                        ),
                    });
                    recs.push(OptimizeRec {
                        cmd: format!("blazar tune {name} --spec-race"),
                        why: "race dense/ngram/dflash under identical knobs and persist qualified receipts".into(),
                        datum: "no qualified spec receipt".into(),
                        benchmark_sessions: 1,
                    });
                }
            }
            Ok(_) => {
                findings.push(OptimizeFinding {
                    severity: Severity::Warn,
                    area: "spec",
                    text: "no speculation receipts — the lane has never raced its spec modes"
                        .into(),
                    datum: format!("spec_receipts empty for ({name}, {tag})"),
                });
                recs.push(OptimizeRec {
                    cmd: format!("blazar tune {name} --ngram"),
                    why: "tune the ngram shape first so its race arm is a real candidate".into(),
                    datum: "no receipts at all".into(),
                    benchmark_sessions: 1,
                });
                recs.push(OptimizeRec {
                    cmd: format!("blazar tune {name} --spec-race"),
                    why: "then race the modes and persist provenance-carrying receipts".into(),
                    datum: "no receipts at all".into(),
                    benchmark_sessions: 1,
                });
            }
            Err(e) => findings.push(unreadable("spec", e)),
        }
    }

    // If the bench receipt was flagged stale, the cheapest first move
    // is re-benching before any search reuses stale evidence.
    if bench_rec_pending {
        recs.insert(
            0,
            OptimizeRec {
                cmd: format!("blazar bench {name}"),
                why: "re-measure on the current lane + hardware so later tuning judges fresh evidence".into(),
                datum: "stale or missing bench receipt".into(),
                benchmark_sessions: 1,
            },
        );
    }

    // ---- capability certificate (absence only — staleness is
    // autopilot's lane, thresholds must not fork between advisers) ----
    match store.get_model_caps_dated(name) {
        Ok(None) => {
            findings.push(OptimizeFinding {
                severity: Severity::Warn,
                area: "cert",
                text: "no capability certificate — admission gates run on unverified caps".into(),
                datum: "model_caps empty for this model".into(),
            });
            recs.push(OptimizeRec {
                cmd: format!("blazar model-doctor {name}"),
                why: "verify the capability certificate end-to-end".into(),
                datum: "no certificate".into(),
                benchmark_sessions: 0,
            });
        }
        Ok(Some((tag, tested, _))) => {
            let age_days = (input.now - tested).max(0) / 86_400;
            findings.push(OptimizeFinding {
                severity: Severity::Ok,
                area: "cert",
                text: "capability certificate present".into(),
                datum: format!("verified {age_days}d ago on lane {tag}"),
            });
        }
        Err(e) => findings.push(unreadable("cert", e)),
    }

    // ---- workload sufficiency ----------------------------------------
    match store.list_usage_days(30) {
        Ok(days) => {
            let requests: i64 = days
                .iter()
                .filter(|d| d.model == name)
                .map(|d| d.requests)
                .sum();
            if requests < input.workload_min_requests {
                findings.push(OptimizeFinding {
                    severity: Severity::Info,
                    area: "workload",
                    text: format!(
                        "workload evidence thin ({requests} requests in 30d) — workload-representative verification needs real traffic shape"
                    ),
                    datum: format!(
                        "usage_daily: {requests} of {} request floor",
                        input.workload_min_requests
                    ),
                });
            } else {
                findings.push(OptimizeFinding {
                    severity: Severity::Ok,
                    area: "workload",
                    text: format!("{requests} requests in 30d — enough shape for workload-representative verification"),
                    datum: "usage_daily 30d rollup".into(),
                });
            }
        }
        Err(e) => findings.push(unreadable("workload", e)),
    }

    Ok(OptimizePlan {
        model: name.to_string(),
        engine_tag,
        hw_current,
        serving_ctx,
        findings,
        recs,
    })
}

fn unreadable(area: &'static str, e: crate::CoreError) -> OptimizeFinding {
    OptimizeFinding {
        severity: Severity::Warn,
        area,
        text: format!("evidence unreadable: {e}"),
        datum: "store read failed — fix the store before trusting any plan".into(),
    }
}

/// GGUF check mirrors autopilot's benchability gate: llama-bench reads
/// GGUF files only.
fn is_gguf(row: &ModelRow) -> bool {
    std::path::Path::new(&row.path)
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
}

/// Parse a profile's argv JSON array leniently — a malformed payload
/// yields an empty argv (every probe then reports the honest default)
/// rather than failing the whole plan.
fn parse_argv_json(args_json: &str) -> Vec<String> {
    serde_json::from_str::<Vec<String>>(args_json).unwrap_or_default()
}

/// Value following `flag` in an argv (`["-c", "4096", …]`). Returns
/// `None` when the flag is absent or its value looks like another flag
/// (malformed pair — treated as absent, never guessed).
fn argv_value(argv: &[String], flag: &str) -> Option<String> {
    argv.iter()
        .position(|a| a == flag)
        .and_then(|i| argv.get(i + 1))
        .filter(|v| !v.starts_with('-'))
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dirs::BlazarDirs,
        engine_kind::EngineKind,
        store::{EvidenceAxes, ProfileRow, SpecProvenance},
    };

    fn test_dirs() -> (tempfile::TempDir, Store) {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        dirs.ensure().unwrap();
        (tmp, Store::open(&dirs).unwrap())
    }

    fn model_row(name: &str, path: &str) -> ModelRow {
        ModelRow {
            name: name.into(),
            repo: format!("adopted:{path}"),
            quant: "Q4_K_M".into(),
            path: path.into(),
            bytes: 1,
            sha256: None,
            mmproj_path: None,
            components: Vec::new(),
            shards: 1,
            arch: Some("qwen3".into()),
            params: Some(0.5),
            ctx_train: Some(32_768),
            pulled_at: 0,
            last_used_at: 0,
        }
    }

    fn engine(tag: &str, hw: Option<&str>) -> EngineRow {
        // Manifest shaped like the real engine rows: devices[0] carries
        // the name hw_fingerprint_from_manifest derives from.
        let manifest = hw.map_or_else(
            || "{}".into(),
            |h| serde_json::json!({ "devices": [{ "name": h }] }).to_string(),
        );
        EngineRow {
            tag: tag.into(),
            asset: "test".into(),
            sha256: String::new(),
            installed_at: 0,
            active: true,
            manifest,
            kind: EngineKind::LlamaCpp,
        }
    }

    /// The fingerprint `engine(tag, Some(h))` produces — derived through
    /// the same fn the builder uses, so tests never hand-roll a string
    /// that could drift from the real derivation.
    fn expected_hw(h: &str) -> Option<String> {
        crate::store::EvidenceAxes::hw_fingerprint(Some(h), None, std::env::consts::ARCH)
    }

    fn inputs<'a>(row: &'a ModelRow, eng: Option<&'a EngineRow>) -> OptimizeInputs<'a> {
        OptimizeInputs {
            row,
            engine: eng,
            config_ctx: 4096,
            now: 10 * 86_400,
            workload_min_requests: WORKLOAD_MIN_REQUESTS,
        }
    }

    fn finding<'p>(plan: &'p OptimizePlan, area: &str) -> &'p OptimizeFinding {
        plan.findings
            .iter()
            .find(|f| f.area == area)
            .unwrap_or_else(|| panic!("no {area} finding: {:?}", plan.findings))
    }

    fn rec_cmds(plan: &OptimizePlan) -> Vec<String> {
        plan.recs.iter().map(|r| r.cmd.clone()).collect()
    }

    #[test]
    fn unit__build_plan__no_engine_names_the_gap() {
        let (_t, s) = test_dirs();
        let row = model_row("m", "/models/m.gguf");
        let plan = build_plan(&s, &inputs(&row, None)).unwrap();
        assert_eq!(finding(&plan, "engine").severity, Severity::Warn);
        assert!(rec_cmds(&plan).contains(&"blazar engine list".to_string()));
        // No lane → no bench/profile/spec verdicts to fake.
        assert!(plan.findings.iter().all(|f| f.area != "bench"));
    }

    #[test]
    fn unit__build_plan__fresh_model_recommends_the_evidence_ladder() {
        let (_t, s) = test_dirs();
        let row = model_row("m", "/models/m.gguf");
        let e = engine("b1-cuda", Some("gpu=X"));
        s.upsert_engine(&e).unwrap();
        let plan = build_plan(&s, &inputs(&row, Some(&e))).unwrap();
        // Order is priority: bench → tune search → ngram → spec-race →
        // model-doctor, with the cert last (non-benchmark).
        assert_eq!(
            rec_cmds(&plan),
            vec![
                "blazar bench m",
                "blazar tune m --search",
                "blazar tune m --ngram",
                "blazar tune m --spec-race",
                "blazar model-doctor m",
            ]
        );
        assert_eq!(plan.benchmark_budget(), 4);
        assert_eq!(plan.serving_ctx, 4096, "no profile → config default ctx");
    }

    #[test]
    fn unit__build_plan__bench_staleness_by_lane_and_hardware() {
        let (_t, s) = test_dirs();
        let row = model_row("m", "/models/m.gguf");
        let e = engine("b2-cuda", Some("X"));
        s.upsert_engine(&e).unwrap();
        let fp = expected_hw("X").unwrap();
        let axes = EvidenceAxes {
            quant: Some("Q4_K_M".into()),
            hw_fingerprint: Some(fp),
            workload: None,
            profile_hash: None,
        };
        // Receipt from a different lane: stale even with matching hw.
        s.put_bench_result_with_axes("m", "b1-cuda", "[]", &axes)
            .unwrap();
        let plan = build_plan(&s, &inputs(&row, Some(&e))).unwrap();
        assert_eq!(finding(&plan, "bench").severity, Severity::Warn);
        assert!(finding(&plan, "bench").text.contains("lane 'b1-cuda'"));
        // Same lane, pre-fingerprint receipt: unattributable.
        s.put_bench_result_with_axes("m", "b2-cuda", "[]", &EvidenceAxes::default())
            .unwrap();
        let plan = build_plan(&s, &inputs(&row, Some(&e))).unwrap();
        assert_eq!(finding(&plan, "bench").severity, Severity::Warn);
        assert!(
            finding(&plan, "bench")
                .text
                .contains("predates hardware fingerprinting")
        );
        // Same lane + attributed: current.
        s.put_bench_result_with_axes("m", "b2-cuda", "[]", &axes)
            .unwrap();
        let plan = build_plan(&s, &inputs(&row, Some(&e))).unwrap();
        assert_eq!(finding(&plan, "bench").severity, Severity::Ok);
    }

    #[test]
    fn unit__build_plan__ledger_pointer_cited_and_ctx_from_profile() {
        let (_t, s) = test_dirs();
        let row = model_row("m", "/models/m.gguf");
        let e = engine("b1-cuda", Some("X"));
        s.upsert_engine(&e).unwrap();
        let fp = expected_hw("X").unwrap();
        s.put_bench_result_with_axes(
            "m",
            "b1-cuda",
            "[]",
            &EvidenceAxes {
                quant: Some("Q4_K_M".into()),
                hw_fingerprint: Some(fp),
                workload: None,
                profile_hash: None,
            },
        )
        .unwrap();
        s.upsert_profile(&ProfileRow {
            model_name: "m".into(),
            engine_tag: "b1-cuda".into(),
            args_hash: "h1".into(),
            args_json: r#"["llama-server","-c","16384"]"#.to_string(),
            benchmark_json: None,
            updated_at: 9 * 86_400,
        })
        .unwrap();
        let exp = s
            .insert_experiment(
                "m",
                "b1-cuda",
                None,
                Some("Q4_K_M"),
                "[]",
                "h1",
                "Throughput",
            )
            .unwrap();
        s.set_active_profile("m", "b1-cuda", exp).unwrap();
        let plan = build_plan(&s, &inputs(&row, Some(&e))).unwrap();
        let f = finding(&plan, "profile");
        assert_eq!(f.severity, Severity::Ok);
        assert!(f.text.contains(&format!("experiment #{exp}")), "{:?}", f);
        assert_eq!(plan.serving_ctx, 16_384, "ctx from profile argv -c");
    }

    #[test]
    fn unit__build_plan__untuned_ngram_flagged_and_excluded() {
        let (_t, s) = test_dirs();
        let row = model_row("m", "/models/m.gguf");
        let e = engine("b1-cuda", Some("X"));
        s.upsert_engine(&e).unwrap();
        let put = |mode: &str, tps: f64, dense: f64, tuned: bool| {
            s.put_spec_receipt(
                "m",
                "b1-cuda",
                mode,
                tps,
                dense,
                &SpecProvenance {
                    hw_fingerprint: None,
                    spec_params: None,
                    tuned,
                    reps: 1,
                },
            )
            .unwrap();
        };
        put("dense", 100.0, 100.0, true);
        // Untuned ngram at a gaudy 1.5x must NOT qualify.
        put("ngram", 150.0, 100.0, false);
        let plan = build_plan(&s, &inputs(&row, Some(&e))).unwrap();
        let f = finding(&plan, "spec");
        assert_eq!(f.severity, Severity::Warn);
        assert!(f.text.contains("EXCLUDED"), "{:?}", f);
        assert!(rec_cmds(&plan).contains(&"blazar tune m --ngram".to_string()));
        // Same race, tuned: qualifies, no ngram rec.
        put("ngram", 150.0, 100.0, true);
        let plan = build_plan(&s, &inputs(&row, Some(&e))).unwrap();
        assert_eq!(finding(&plan, "spec").severity, Severity::Ok);
        assert!(finding(&plan, "spec").text.contains("qualified"));
        assert!(!rec_cmds(&plan).iter().any(|c| c.contains("--ngram")));
    }

    #[test]
    fn unit__build_plan__kv_evidence_named_by_grade() {
        let (_t, s) = test_dirs();
        let e = engine("b1-cuda", Some("X"));
        s.upsert_engine(&e).unwrap();
        // Unevidenced model: info row, no eligibility claim.
        let row = model_row("fresh-model", "/models/fresh.gguf");
        let plan = build_plan(&s, &inputs(&row, Some(&e))).unwrap();
        assert_eq!(finding(&plan, "kv").severity, Severity::Info);
        assert!(
            finding(&plan, "kv")
                .text
                .contains("keeps every search on f16")
        );
        // Registry-evidenced model at a covered ctx: ok row naming grades.
        let row = model_row("qwen3.5-9b", "/models/qwen.gguf");
        let mut inp = inputs(&row, Some(&e));
        inp.config_ctx = 2048;
        let plan = build_plan(&s, &inp).unwrap();
        let f = finding(&plan, "kv");
        assert_eq!(f.severity, Severity::Ok, "{f:?}");
        assert!(f.text.contains("q8_0") && f.text.contains("q4_0"), "{f:?}");
    }

    #[test]
    fn unit__build_plan__unevidenced_active_kv_profile_warns() {
        let (_t, s) = test_dirs();
        let e = engine("b1-cuda", Some("gpu=X"));
        s.upsert_engine(&e).unwrap();
        let row = model_row("fresh-model", "/models/fresh.gguf");
        s.upsert_profile(&ProfileRow {
            model_name: "fresh-model".into(),
            engine_tag: "b1-cuda".into(),
            args_hash: "h".into(),
            args_json: r#"["llama-server","-ctk","q8_0","-ctv","q8_0"]"#.to_string(),
            benchmark_json: None,
            updated_at: 0,
        })
        .unwrap();
        let plan = build_plan(&s, &inputs(&row, Some(&e))).unwrap();
        // kv area holds BOTH the unevidenced-active warn and the
        // no-evidence info; the warn must exist.
        let warned = plan.findings.iter().any(|f| {
            f.area == "kv"
                && f.severity == Severity::Warn
                && f.text.contains("without quality evidence")
        });
        assert!(warned, "{:?}", plan.findings);
    }

    #[test]
    fn unit__build_plan__workload_threshold_and_cert() {
        let (_t, s) = test_dirs();
        let row = model_row("m", "/models/m.gguf");
        let e = engine("b1-cuda", Some("gpu=X"));
        s.upsert_engine(&e).unwrap();
        s.add_usage_delta(
            20_261_101,
            "m",
            &crate::store::UsageDelta {
                in_tokens: 1,
                out_tokens: 1,
                cached_tokens: 0,
                requests: 25,
                gen_secs: 1.0,
                ttft_count: 1,
                ttft_ms_total: 1.0,
            },
        )
        .unwrap();
        let plan = build_plan(&s, &inputs(&row, Some(&e))).unwrap();
        assert_eq!(finding(&plan, "workload").severity, Severity::Ok);
        assert_eq!(finding(&plan, "cert").severity, Severity::Warn);
        assert!(rec_cmds(&plan).contains(&"blazar model-doctor m".to_string()));
    }

    #[test]
    fn unit__build_plan__non_llamacpp_lane_scopes_honestly() {
        let (_t, s) = test_dirs();
        let row = model_row("m", "/models/m.gguf");
        let mut e = engine("mlx-1", Some("gpu=X"));
        e.kind = EngineKind::Mlx;
        s.upsert_engine(&e).unwrap();
        let plan = build_plan(&s, &inputs(&row, Some(&e))).unwrap();
        assert_eq!(finding(&plan, "engine").severity, Severity::Info);
        assert!(finding(&plan, "engine").text.contains("no tuning surface"));
        // No bench/tune/spec recs fabricated for a lane that cannot run
        // them — only cert/workload still apply.
        assert!(
            plan.recs
                .iter()
                .all(|r| !r.cmd.contains("--search") && !r.cmd.contains("spec-race"))
        );
    }

    #[test]
    fn unit__argv_value__pairs_and_malformed() {
        let argv: Vec<String> = ["-c", "16384", "-ctk"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(argv_value(&argv, "-c").as_deref(), Some("16384"));
        assert_eq!(argv_value(&argv, "-ctk"), None, "flag-like value = absent");
        assert_eq!(argv_value(&argv, "-missing"), None);
    }
}
