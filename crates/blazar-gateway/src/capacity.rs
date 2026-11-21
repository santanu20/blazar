//! Per-GPU capacity observability (`GET /api/capacity`). v0.15.
//!
//! The planner already knows placement decisions; llama.cpp#26129-class
//! tooling wants the LIVE numbers as an API, not logs. This endpoint
//! joins three sources, each honest about its own limits:
//!
//! - `nvidia-smi` device census (index, name, total/free VRAM,
//!   utilization) - absent on non-NVIDIA boxes, reported as an empty
//!   device list with a note, never guessed.
//! - `nvidia-smi --query-compute-apps` (pid + used VRAM) - the ONLY
//!   per-process memory authority on the box.
//! - the supervisor's live instance rows (`Supervisor::ps`) - model,
//!   engine, slots, state, placement. Joining by pid attributes VRAM to
//!   Blazar residents; unmatched compute apps surface as `external`
//!   (another process eating the GPU - ollama users' "where did my VRAM
//!   go" question, answered with data).
//!
//! Weights/KV/compute SPLIT per resident is deliberately not claimed:
//! that split lives inside each engine child and would be an estimate
//! here. Per-process totals are measured; the split lands when engines
//! expose it natively.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::state::AppState;

/// Bounded external-process run: kill past the deadline, drain both pipes
/// through threads (mirrors the runtime probe helper's fork-pressure
/// discipline - a wedged probe must never wedge the endpoint).
pub(crate) fn bounded_output(
    cmd: &mut std::process::Command,
    secs: u64,
) -> Option<std::process::Output> {
    use std::io::Read;
    use std::process::Stdio;
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().ok()?;
    let mut stdout = child.stdout.take().map(|mut p| {
        std::thread::spawn(move || {
            let mut b = Vec::new();
            p.read_to_end(&mut b).ok();
            b
        })
    });
    let mut stderr = child.stderr.take().map(|mut p| {
        std::thread::spawn(move || {
            let mut b = Vec::new();
            p.read_to_end(&mut b).ok();
            b
        })
    });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(secs);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            Err(_) => return None,
        }
    };
    let stdout = stdout.take().map(|h| h.join().unwrap_or_default());
    let stderr = stderr.take().map(|h| h.join().unwrap_or_default());
    Some(std::process::Output {
        status,
        stdout: stdout.unwrap_or_default(),
        stderr: stderr.unwrap_or_default(),
    })
}

#[derive(Clone)]
pub(crate) struct DeviceRow {
    pub(crate) index: u32,
    pub(crate) name: String,
    pub(crate) total_mib: u64,
    pub(crate) free_mib: u64,
    pub(crate) util_pct: u64,
}

/// `index, name, total MiB, free MiB, util %` per line
/// (`--format=csv,noheader,nounits`). Rows with unparseable fields drop
/// silently-unknown rather than half-guessed.
pub(crate) fn parse_devices(text: &str) -> Vec<DeviceRow> {
    text.lines()
        .filter_map(|ln| {
            let mut parts = ln.split(',').map(str::trim);
            let index: u32 = parts.next()?.parse().ok()?;
            let name = parts.next()?.to_string();
            if name.is_empty() {
                return None;
            }
            Some(DeviceRow {
                index,
                name,
                total_mib: parts.next()?.parse().ok()?,
                free_mib: parts.next()?.parse().ok()?,
                util_pct: parts.next().unwrap_or("0").parse().unwrap_or(0),
            })
        })
        .collect()
}

/// `pid, used MiB` per line. Drivers without the number print `[N/A]` -
/// those rows report the pid with an unknown footprint (0), never a guess.
pub(crate) fn parse_compute_apps(text: &str) -> Vec<(u32, u64)> {
    text.lines()
        .filter_map(|ln| {
            let mut parts = ln.split(',').map(str::trim);
            let pid: u32 = parts.next()?.parse().ok()?;
            let mib: u64 = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
            Some((pid, mib))
        })
        .collect()
}

/// GET /api/capacity - device census + per-resident VRAM attribution.
#[allow(clippy::too_many_lines)] // census + join + honest notes: one response body assembled in reading order
pub async fn capacity(State(state): State<Arc<AppState>>) -> Response {
    // Two nvidia-smi spawns (~30-60 ms each) + pipe polling: blocking
    // work, off the async runtime threads.
    let (devices_raw, apps_raw) = tokio::task::spawn_blocking(|| {
        let devices = bounded_output(
            std::process::Command::new("nvidia-smi").args([
                "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]),
            5,
        )
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned());
        let apps = bounded_output(
            std::process::Command::new("nvidia-smi").args([
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]),
            5,
        )
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned());
        (devices, apps)
    })
    .await
    .unwrap_or((None, None));

    let mut notes = Vec::new();
    let devices: Vec<serde_json::Value> = if let Some(text) = devices_raw.as_deref() {
        parse_devices(text)
            .into_iter()
            .map(|d| {
                serde_json::json!({
                    "id": format!("cuda:{}", d.index),
                    "index": d.index,
                    "name": d.name,
                    "total_vram_bytes": d.total_mib * 1024 * 1024,
                    "free_vram_bytes": d.free_mib * 1024 * 1024,
                    "utilization_percent": d.util_pct,
                })
            })
            .collect()
    } else {
        notes.push(
            "no nvidia-smi census on this box - non-NVIDIA GPU boxes report an empty \
             device list (residents below still come from the supervisor)",
        );
        Vec::new()
    };

    // Supervisor rows are the resident truth; compute-apps attributes VRAM.
    let ps_rows = state.sup.ps();
    let apps = apps_raw.as_deref().map(parse_compute_apps);
    let mut external: Vec<serde_json::Value> = Vec::new();
    let residents: Vec<serde_json::Value> = ps_rows
        .iter()
        .map(|p| {
            let mut v = serde_json::json!({
                "model": p.name,
                "engine": p.engine,
                "pid": p.pid,
                "state": p.state,
                "slots": p.slots,
                "slots_configured": p.slots_configured,
                "in_flight": p.in_flight,
                "device": p.device,
                "device_id": p.device_id,
                "model_bytes": p.bytes,
            });
            if let Some(apps) = &apps {
                match apps.iter().find(|(pid, _)| *pid == p.pid) {
                    Some((_, mib)) => {
                        v["vram_bytes"] = serde_json::json!(mib * 1024 * 1024);
                    }
                    // Resident but not (yet) holding GPU memory - CPU
                    // instances and engine kinds without compute apps.
                    None => {
                        v["vram_bytes"] = serde_json::Value::Null;
                    }
                }
            }
            v["host_rss_mib"] = serde_json::json!(p.host_rss_mib);
            v["host_charge_bytes"] = serde_json::json!(p.host_charge_bytes);
            v
        })
        .collect();

    if let Some(apps) = &apps {
        let known: std::collections::HashSet<u32> = ps_rows.iter().map(|p| p.pid).collect();
        for (pid, mib) in apps {
            if !known.contains(pid) {
                external.push(serde_json::json!({
                    "pid": pid,
                    "vram_bytes": mib * 1024 * 1024,
                    "note": "external process (not a Blazar child)",
                }));
            }
        }
    }
    notes.push(
        "per-resident vram is the process total from nvidia-smi compute-apps; \
         the weights/KV/compute split lives inside each engine child and is \
         not estimated here",
    );

    // Host ledger: the RAM twin of the device census above. `committed`
    // is what the spawn guard subtracts from MemAvailable (resident
    // file-backed shares + in-flight spawn reservations); `reserved` is
    // just the in-flight slice of that.
    let snapshot = blazar_runtime::probe::host_mem_snapshot();
    let host = if let Some(s) = snapshot {
        let committed = state.sup.host_committed_bytes();
        serde_json::json!({
            "total_mib": s.total_mib,
            "available_mib": s.available_mib,
            "committed_bytes": committed,
            "reserved_bytes": state.sup.host_reserved_bytes(),
            "effective_available_mib": s.available_mib.saturating_sub(committed / (1024 * 1024)),
            "pressure": state.sup.host_pressure_label(),
        })
    } else {
        notes.push(
            "host memory snapshot unavailable on this platform - the spawn \
             guard still refuses loads below its floor, only this report \
             is blind",
        );
        serde_json::json!({
            "total_mib": serde_json::Value::Null,
            "available_mib": serde_json::Value::Null,
            "committed_bytes": state.sup.host_committed_bytes(),
            "reserved_bytes": state.sup.host_reserved_bytes(),
            "effective_available_mib": serde_json::Value::Null,
            "pressure": state.sup.host_pressure_label(),
        })
    };

    (
        StatusCode::OK,
        axum::Json(serde_json::json!({
            "object": "blazar.capacity",
            "devices": devices,
            "host": host,
            "residents": residents,
            "external": external,
            "notes": notes,
        })),
    )
        .into_response()
}

/// Mirror of `profile::UNIFIED_SPAWN_OVERHEAD_BYTES` (private upstream) so
/// `/api/fit` demand matches the spawn guard exactly.
/// `unit__fit__overhead_mirror_matches_admission_floor` pins this against
/// the public `admission_floor_bytes` - if the upstream constant ever
/// changes, that test fails and forces a sync here.
const FIT_SPAWN_OVERHEAD_MIB: u64 = 700;

/// Halve `ctx` (floor 1024) until `weights + kv(ctx) + overhead` fits
/// `free_mib` - the same ladder `blazar fit` walks. Returns the largest
/// fitting ctx, or 0 when even 1024 does not fit.
fn ctx_ladder(weights_mib: u64, kv_mib: impl Fn(u64) -> u64, free_mib: u64, start_ctx: u64) -> u64 {
    let mut ctx = start_ctx.max(1024);
    loop {
        if weights_mib + kv_mib(ctx) + FIT_SPAWN_OVERHEAD_MIB <= free_mib {
            return ctx;
        }
        if ctx <= 1024 {
            return 0;
        }
        ctx = (ctx / 2).max(1024);
    }
}

#[derive(serde::Deserialize)]
pub(crate) struct FitRequest {
    model: String,
    #[serde(default)]
    ctx: Option<u32>,
}

/// Per-device fit facts for one demand. Pure so the multi-GPU math is
/// unit-pinnable without a live census: one JSON row per device
/// (`fits` / `max_ctx` / `max_ctx_q8` / `max_parallel`) plus the index of
/// the recommended device - the first card where the demand fits. `None`
/// when it fits nowhere (or there is no census at the call site).
fn per_device_fit(
    weights_mib: u64,
    kv: impl Fn(u64) -> u64,
    ctx_evaluated: u64,
    demand_mib: u64,
    devs: &[DeviceRow],
) -> (Vec<serde_json::Value>, Option<usize>) {
    let mut recommended = None;
    let rows = devs
        .iter()
        .enumerate()
        .map(|(idx, dev)| {
            let fits = demand_mib <= dev.free_mib;
            if fits && recommended.is_none() {
                recommended = Some(idx);
            }
            let max_ctx = ctx_ladder(weights_mib, &kv, dev.free_mib, ctx_evaluated);
            let max_ctx_q8 = ctx_ladder(
                weights_mib,
                |c| kv(c).div_ceil(2),
                dev.free_mib,
                ctx_evaluated,
            );
            // Whole instances of this demand the card holds at free time.
            let max_parallel = dev.free_mib / demand_mib.max(1);
            serde_json::json!({
                "id": format!("cuda:{}", dev.index),
                "name": dev.name,
                "free_mib": dev.free_mib,
                "fits": fits,
                "max_ctx": max_ctx,
                "max_ctx_q8": max_ctx_q8,
                "max_parallel": max_parallel,
            })
        })
        .collect();
    (rows, recommended)
}

/// POST /api/fit - pre-flight capacity math for one model at a context
/// size: "will `<model>` load at `<ctx>` on the primary GPU right now?"
///
/// Composes the same primitives the runtime uses - `admission_floor_bytes`
/// spawn-overhead semantics, `preflight::kv_f16_mib` KV math, the
/// `parse_devices` census - so the answer cannot drift from what a spawn
/// attempt would actually do. `max_ctx` / `max_ctx_q8` walk the same
/// halve-until-fits ladder as `blazar fit` (floor 1024; q8 halves KV).
/// Multi-GPU boxes additionally get one row per card (placement
/// recommendation = first fitting card) and the measured-vs-predicted
/// calibration from `/api/admission-accuracy` when a sample exists.
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]
pub(crate) async fn fit(
    State(state): State<Arc<AppState>>,
    axum::Json(body): axum::Json<FitRequest>,
) -> Response {
    let Some(row) = state.with_store(|s| crate::proxy::resolve_model(s, &body.model).ok()) else {
        return crate::ollama::api_error(500, "store unavailable");
    };
    let Some(row) = row else {
        return crate::ollama::api_error(404, &format!("model '{}' not found", body.model));
    };

    let weights_mib = std::fs::metadata(&row.path).map_or(0, |m| m.len() / (1024 * 1024));
    let meta = match blazar_core::read_metadata_file(std::path::Path::new(&row.path)) {
        Ok(meta) => meta,
        Err(e) => {
            return crate::ollama::api_error(500, &format!("model file metadata unreadable: {e}"));
        }
    };
    let ctx_evaluated = body.ctx.map_or_else(
        || u64::from(crate::preflight::admission_ctx(&state, &body.model)),
        u64::from,
    );

    // Same census command as `capacity()`; the primary GPU is device 0.
    let census = tokio::task::spawn_blocking(|| {
        bounded_output(
            std::process::Command::new("nvidia-smi").args([
                "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]),
            5,
        )
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
    })
    .await
    .unwrap_or(None);

    let kv = |ctx: u64| crate::preflight::kv_f16_mib(&meta, ctx);
    let kv_mib = kv(ctx_evaluated);
    let demand_mib = weights_mib + kv_mib + FIT_SPAWN_OVERHEAD_MIB;

    let mut notes = Vec::new();
    let census_rows = census.as_deref().map(parse_devices).unwrap_or_default();
    let (devices, recommended) =
        per_device_fit(weights_mib, kv, ctx_evaluated, demand_mib, &census_rows);
    let primary = census_rows.first().cloned();
    let pick = recommended
        .and_then(|idx| census_rows.get(idx).cloned())
        .or(primary.clone());
    // Captured before the destructure below moves `pick`: the
    // parallelism advisor wants the picked card's free VRAM.
    let pick_free_mib = pick.as_ref().map(|d| d.free_mib);
    let (device, free_mib, fits, max_ctx, max_ctx_q8, max_parallel) = if let Some(dev) = pick {
        let fits = demand_mib <= dev.free_mib;
        let max_ctx = ctx_ladder(weights_mib, kv, dev.free_mib, ctx_evaluated);
        let max_ctx_q8 = ctx_ladder(
            weights_mib,
            |c| kv(c).div_ceil(2),
            dev.free_mib,
            ctx_evaluated,
        );
        let max_parallel = dev.free_mib / demand_mib.max(1);
        (
            format!("cuda:{} {}", dev.index, dev.name),
            dev.free_mib,
            Some(fits),
            Some(max_ctx),
            Some(max_ctx_q8),
            Some(max_parallel),
        )
    } else {
        notes.push(
            "no nvidia-smi census on this box - fit math reported, fits/max_ctx null \
             (non-NVIDIA boxes: consult the engine's own loader)",
        );
        ("unknown".to_string(), 0, None, None, None, None)
    };
    if max_ctx == Some(0) {
        notes.push("even ctx 1024 does not fit free VRAM - smaller quant or free capacity first");
    }
    let recommended_device = recommended.map(|idx| format!("cuda:{}", census_rows[idx].index));

    // Measured-vs-predicted calibration for this exact model, when a load
    // has been observed (ties the fit answer to `/api/admission-accuracy`
    // instead of trusting the floor blindly).
    let calibration = {
        let model = row.name.as_str();
        state
            .admission
            .snapshot()
            .into_iter()
            .find(|r| r.model == model)
            .map(|r| {
                serde_json::json!({
                    "predicted_mib": r.predicted_mib,
                    "actual_mib": r.actual_mib,
                    "delta_mib": r.actual_mib.map(|a| a as i64 - r.predicted_mib as i64),
                    "error_pct": r.actual_mib.filter(|_| r.predicted_mib > 0).map(|a| {
                        ((a as f64 - r.predicted_mib as f64) / r.predicted_mib as f64) * 100.0
                    }),
                    "sampled_unix": r.sampled_unix,
                })
            })
    };
    let confidence = fit_confidence(!census_rows.is_empty(), calibration.as_ref());

    // Media-job pressure from the durable ledger: running jobs already
    // hold VRAM (the census free numbers include them), queued ones
    // will contend when they spawn - the fit math never subtracts a
    // reservation it cannot verify, so the pressure rides as a block
    // plus notes instead.
    let (media_jobs, media_notes) = {
        let (running, queued) = state
            .with_store(|s| {
                (
                    s.list_jobs(Some("running"), None, 1000, None)
                        .unwrap_or_default(),
                    s.list_jobs(Some("queued"), None, 1000, None)
                        .unwrap_or_default(),
                )
            })
            .unwrap_or_default();
        let mut media_notes = Vec::new();
        let block = media_jobs_block(&running, &queued);
        if block["running"].as_u64().is_some_and(|n| n > 0) {
            media_notes
                .push("running media job(s) already hold VRAM the census reflects in free_mib");
        }
        if block["queued"].as_u64().is_some_and(|n| n > 0) {
            media_notes.push("queued media job(s) will contend for VRAM when they spawn");
        }
        (block, media_notes)
    };
    notes.extend(media_notes);

    // Parallelism advisory keyed off the lane that would actually
    // serve this model (same shape resolution as /api/explain): its
    // newest installed row's probed flag vocabulary gates every axis
    // reported, so no combo is suggested the child would reject.
    let parallelism = {
        let serving = state.with_store(|s| {
            let engines = s.list_engines().unwrap_or_default();
            let kinds: Vec<blazar_core::engine_kind::EngineKind> =
                engines.iter().map(|r| r.kind).collect();
            let shape = blazar_core::engine_kind::FormatShape {
                diffusion: row.has_component_set(),
                shards: blazar_core::engine_kind::ShardFormat::detect(
                    std::path::Path::new(&row.path).is_dir(),
                    blazar_core::store::quantized_safetensors_signal(&row.name, "", &row.path),
                    row.is_mlx(),
                    row.is_litert_bundle(),
                ),
            };
            let policy = state.cfg().engine_routing.policy;
            let kind = blazar_core::engine_kind::EngineKind::route_format(shape, &kinds, policy);
            (kind, engines)
        });
        let flags = serving.and_then(|(kind, engines)| {
            kind.and_then(|k| {
                engines
                    .iter()
                    .find(|r| r.kind == k)
                    .and_then(|r| {
                        serde_json::from_str::<blazar_runtime::Manifest>(&r.manifest).ok()
                    })
                    .map(|m| m.flags)
            })
        });
        flags.and_then(|flags| {
            parallelism_advisor(
                &flags,
                census_rows.len(),
                weights_mib,
                kv_mib,
                pick_free_mib,
            )
        })
    };

    // PD-disaggregation advisory: multi-GPU + sglang lane + long ctx.
    let sglang_installed = state
        .with_store(|s| {
            s.list_engines().is_ok_and(|rows| {
                rows.iter()
                    .any(|r| r.kind == blazar_core::engine_kind::EngineKind::Sglang)
            })
        })
        .unwrap_or(false);
    let pd = pd_advisory(census_rows.len(), ctx_evaluated, sglang_installed);
    if pd.is_some() {
        notes.push(
            "pd advisory: long-context multi-GPU prefill/decode disaggregation candidate - see pd_advisory",
        );
    }

    (
        StatusCode::OK,
        axum::Json(serde_json::json!({
            "object": "blazar.fit",
            "model": row.name,
            "ctx_evaluated": ctx_evaluated,
            "weights_mib": weights_mib,
            "kv_mib": kv_mib,
            "spawn_overhead_mib": FIT_SPAWN_OVERHEAD_MIB,
            "demand_mib": demand_mib,
            "device": device,
            "free_mib": free_mib,
            "fits": fits,
            "max_ctx": max_ctx,
            "max_ctx_q8": max_ctx_q8,
            "max_parallel": max_parallel,
            "devices": devices,
            "recommended_device": recommended_device,
            "calibration": calibration,
            "confidence": confidence,
            "media_jobs": media_jobs,
            "parallelism": parallelism,
            "pd_advisory": pd,
            "notes": notes,
        })),
    )
        .into_response()
}

/// How much to trust this fit card: `measured` when this exact model
/// has an observed load in the admission calibration, `estimated` when
/// only the live census backs the math, `unavailable` when not even
/// the census runs (non-NVIDIA boxes keep the arithmetic but carry no
/// device truth).
fn fit_confidence(census_present: bool, calibration: Option<&serde_json::Value>) -> &'static str {
    match (census_present, calibration.is_some()) {
        (_, true) => "measured",
        (true, false) => "estimated",
        (false, false) => "unavailable",
    }
}

/// Job kinds that render on the GPU box outside the text lanes. The
/// ledger also carries `doctor` / `responses` jobs - CPU-bound or
/// gateway-local, they never contend for VRAM, so they stay out.
const MEDIA_JOB_KINDS: [&str; 3] = ["image", "video", "audio"];

/// Active media-job pressure for the fit card: running jobs already
/// hold VRAM (the census `free_mib` reflects them), queued ones will
/// contend when they spawn. Kinds are unique+sorted so the JSON is
/// stable regardless of ledger row order.
fn media_jobs_block(
    running: &[blazar_core::store::JobRow],
    queued: &[blazar_core::store::JobRow],
) -> serde_json::Value {
    fn media(rows: &[blazar_core::store::JobRow]) -> Vec<&blazar_core::store::JobRow> {
        rows.iter()
            .filter(|j| MEDIA_JOB_KINDS.contains(&j.kind.as_str()))
            .collect()
    }
    let (run_m, queued_m) = (media(running), media(queued));
    let mut kinds: Vec<&str> = run_m
        .iter()
        .chain(queued_m.iter())
        .map(|j| j.kind.as_str())
        .collect();
    kinds.sort_unstable();
    kinds.dedup();
    serde_json::json!({
        "running": run_m.len(),
        "queued": queued_m.len(),
        "kinds": kinds,
    })
}

/// One parallelism axis the engine's own `--help` vocabulary verified.
struct ParallelAxis {
    key: &'static str,
    flag: &'static str,
}

/// The axes the card reports, in the order the capability card scans
/// them (same names, same `--` prefix - one vocabulary everywhere).
const PARALLELISM_AXES: [ParallelAxis; 6] = [
    ParallelAxis {
        key: "tensor",
        flag: "--tensor-parallel-size",
    },
    ParallelAxis {
        key: "pipeline",
        flag: "--pipeline-parallel-size",
    },
    ParallelAxis {
        key: "data",
        flag: "--data-parallel-size",
    },
    ParallelAxis {
        key: "expert",
        flag: "--expert-parallel-size",
    },
    ParallelAxis {
        key: "context",
        flag: "--context-parallel-size",
    },
    ParallelAxis {
        key: "decode_context",
        flag: "--decode-context-parallel-size",
    },
];

/// Flag-verified parallelism advisory for the lane that would serve
/// this model: which axes its build actually accepts, plus the
/// even-split TP shard counts that would fit one card. `None` when
/// the serving lane's flag vocabulary carries no parallelism axis at
/// all (llama.cpp lanes today) - the card stays honestly absent
/// instead of enumerating combos the child would reject at argv time.
/// Estimates are even-split approximations; interconnect/communication
/// cost is NOT modeled (no portable probe), and the note says so.
fn parallelism_advisor(
    flags: &std::collections::BTreeSet<String>,
    n_devices: usize,
    weights_mib: u64,
    kv_mib: u64,
    min_free_mib: Option<u64>,
) -> Option<serde_json::Value> {
    let supported: Vec<&ParallelAxis> = PARALLELISM_AXES
        .iter()
        .filter(|a| flags.contains(a.flag))
        .collect();
    if supported.is_empty() {
        return None;
    }
    let axes: serde_json::Value = serde_json::Map::from_iter(
        supported
            .iter()
            .map(|a| {
                (
                    a.key.to_string(),
                    serde_json::json!({
                        "flag": a.flag,
                        "max_shards": n_devices,
                    }),
                )
            })
            .collect::<Vec<_>>(),
    )
    .into();
    // Viable even-split TP shard counts: weights/tp + kv/tp + spawn
    // overhead must fit one card's free VRAM. No census -> honest
    // null (a guessed free number would drive real spawn decisions).
    let viable_tp: Option<Vec<u64>> = min_free_mib.map(|free| {
        (2..=n_devices as u64)
            .filter(|&tp| weights_mib / tp + kv_mib / tp + FIT_SPAWN_OVERHEAD_MIB <= free)
            .collect()
    });
    Some(serde_json::json!({
        "axes": axes,
        "viable_tp": viable_tp,
        "note": "flag-verified only; even-split approximation; interconnect cost not modeled",
    }))
}

/// Prefill/decode disaggregation advisory for long-context shapes on
/// multi-GPU boxes with the sglang lane installed. Advisory ONLY -
/// the supervised router owns room allocation once enabled (the
/// gateway never invents rooms; pd.rs boundary), and real KV transfer
/// needs the mooncake/nixl wheel. Single-GPU or short-context boxes
/// get an honest null, not a hedge.
fn pd_advisory(n_devices: usize, ctx: u64, sglang_installed: bool) -> Option<serde_json::Value> {
    if n_devices < 2 || !sglang_installed || ctx < 8192 {
        return None;
    }
    Some(serde_json::json!({
        "candidate": true,
        "gpus": n_devices,
        "ctx": ctx,
        "why": "long-context prefill on a multi-GPU box with the sglang lane installed",
        "config": "sglang.disagg_pair = true (supervised router; KV transfer needs the mooncake/nixl wheel)",
        "caveat": "advisory only - consult /api/route before enabling",
    }))
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]
    use super::*;

    #[test]
    fn unit__parse_devices__census_rows_and_drops_malformed() {
        let rows = parse_devices(
            "0, NVIDIA GeForce RTX 4070, 8192, 4096, 37\n\
             1, NVIDIA GeForce RTX 3090, 24576, 24575, 0\n\
             garbage line\n",
        );
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].name, "NVIDIA GeForce RTX 4070");
        assert_eq!(rows[0].total_mib, 8192);
        assert_eq!(rows[0].free_mib, 4096);
        assert_eq!(rows[0].util_pct, 37);
        assert_eq!(rows[1].index, 1);
    }

    #[test]
    fn unit__parse_compute_apps__na_footprint_is_zero_not_guessed() {
        let apps = parse_compute_apps("12345, 2048\n67890, [N/A]\nnope\n");
        assert_eq!(apps.len(), 2);
        assert_eq!(apps[0], (12345, 2048));
        assert_eq!(apps[1], (67890, 0), "[N/A] footprint stays unknown(0)");
    }

    #[test]
    fn unit__fit__overhead_mirror_matches_admission_floor() {
        // admission_floor_bytes = weights + KV floor (512) + spawn overhead.
        // If this ever fails, UNIFIED_SPAWN_OVERHEAD_BYTES changed upstream
        // and FIT_SPAWN_OVERHEAD_MIB must be resynced.
        let mib = |m: u64| m * 1024 * 1024;
        let floor = blazar_core::profile::admission_floor_bytes(mib(100), 0) / (1024 * 1024);
        assert_eq!(floor, 100 + 512 + FIT_SPAWN_OVERHEAD_MIB);
    }

    #[test]
    fn unit__fit__ctx_ladder_halves_until_fit() {
        // weights 5000 + overhead 700 = 5700; kv = ctx/10 MiB; free 6000:
        // ctx 8192 -> 5700+819 no; 4096 -> 5700+409 no; 2048 -> 5700+204 fits.
        assert_eq!(ctx_ladder(5000, |c| c / 10, 6000, 8192), 2048);
        // Room for the full request.
        assert_eq!(ctx_ladder(5000, |c| c / 10, 10_000, 8192), 8192);
        // Even 1024 (kv ~102) does not fit.
        assert_eq!(ctx_ladder(6000, |c| c / 10, 6000, 8192), 0);
    }

    fn dev_row(index: u32, name: &str, free_mib: u64) -> DeviceRow {
        DeviceRow {
            index,
            name: name.to_string(),
            total_mib: free_mib,
            free_mib,
            util_pct: 0,
        }
    }

    #[test]
    fn unit__per_device_fit__recommends_first_fitting_card_with_parallelism() {
        // Small primary + roomy secondary: the recommendation skips the
        // full card 0 and lands on card 1, whose free MiB holds four
        // whole instances of the demand.
        let devs = [dev_row(0, "RTX 4070", 4096), dev_row(1, "RTX 3090", 24_575)];
        let (rows, rec) = per_device_fit(5000, |c| c / 10, 8192, 5700, &devs);
        assert_eq!(rec, Some(1));
        assert_eq!(rows[0]["fits"], serde_json::json!(false));
        assert_eq!(rows[0]["max_parallel"], serde_json::json!(0));
        assert_eq!(rows[0]["max_ctx"], serde_json::json!(0));
        assert_eq!(rows[1]["fits"], serde_json::json!(true));
        assert_eq!(rows[1]["max_parallel"], serde_json::json!(4));
        assert_eq!(rows[1]["max_ctx"], serde_json::json!(8192));
        assert_eq!(rows[1]["id"], serde_json::json!("cuda:1"));
    }

    #[test]
    fn unit__per_device_fit__fits_nowhere_recommends_none() {
        let devs = [dev_row(0, "a", 1000), dev_row(1, "b", 2000)];
        let (rows, rec) = per_device_fit(6000, |c| c / 10, 8192, 6700, &devs);
        assert_eq!(rec, None);
        assert!(rows.iter().all(|r| r["fits"] == serde_json::json!(false)));
        assert!(rows.iter().all(|r| r["max_ctx"] == serde_json::json!(0)));
    }

    #[test]
    fn unit__fit_confidence__tiers() {
        // Calibration sample outranks census: a measured prediction
        // error exists for this model.
        assert_eq!(
            fit_confidence(true, Some(&serde_json::json!({"error_pct": 4.2}))),
            "measured"
        );
        // Census without calibration: honest estimate.
        assert_eq!(fit_confidence(true, None), "estimated");
        // No census at all: nothing was measured on this box.
        assert_eq!(fit_confidence(false, None), "unavailable");
        assert_eq!(
            fit_confidence(false, Some(&serde_json::json!({}))),
            "measured"
        );
    }

    fn job_row(kind: &str, state: &str) -> blazar_core::store::JobRow {
        blazar_core::store::JobRow {
            id: format!("j-{kind}-{state}"),
            kind: kind.to_string(),
            model: None,
            state: state.to_string(),
            request_json: String::new(),
            result_json: None,
            error: None,
            artifact_path: None,
            owner: None,
            created_at: 0,
            updated_at: 0,
        }
    }

    #[test]
    fn unit__media_jobs_block__counts_media_kinds_only() {
        // doctor/responses jobs are gateway bookkeeping, not media
        // pressure - they must not inflate the VRAM-contention counts.
        let running = [
            job_row("image", "running"),
            job_row("video", "running"),
            job_row("doctor", "running"),
        ];
        let queued = [job_row("audio", "queued"), job_row("responses", "queued")];
        let block = media_jobs_block(&running, &queued);
        assert_eq!(block["running"], serde_json::json!(2));
        assert_eq!(block["queued"], serde_json::json!(1));
        assert_eq!(
            block["kinds"],
            serde_json::json!(["audio", "image", "video"])
        );
        // Nothing active: zero counts, empty kinds - not nulls (the
        // ledger WAS consulted).
        let empty = media_jobs_block(&[], &[]);
        assert_eq!(empty["running"], serde_json::json!(0));
        assert_eq!(empty["kinds"], serde_json::json!([]));
    }

    #[test]
    fn unit__parallelism_advisor__flag_gated_axes_and_viable_tp() {
        let mut flags = std::collections::BTreeSet::new();
        flags.insert("--tensor-parallel-size".to_string());
        flags.insert("--context-parallel-size".to_string());
        // weights 10000, kv 0, free 5800: tp2 5000+700 fits, tp3
        // 3333+700 fits, tp4 2500+700 fits -> [2, 3, 4].
        let card = parallelism_advisor(&flags, 4, 10_000, 0, Some(5800)).unwrap();
        assert_eq!(
            card["axes"].as_object().unwrap().keys().collect::<Vec<_>>(),
            vec!["context", "tensor"],
            "only flag-verified axes appear, scanned order"
        );
        assert_eq!(card["axes"]["tensor"]["flag"], "--tensor-parallel-size");
        assert_eq!(card["axes"]["tensor"]["max_shards"], serde_json::json!(4));
        assert_eq!(card["viable_tp"], serde_json::json!([2, 3, 4]));
        // No census: axes still reported, viable_tp honestly null.
        let no_census = parallelism_advisor(&flags, 4, 10_000, 0, None).unwrap();
        assert_eq!(no_census["viable_tp"], serde_json::json!(null));
        // Tight card: only tp4's shard fits.
        let tight = parallelism_advisor(&flags, 4, 10_000, 0, Some(3300)).unwrap();
        assert_eq!(tight["viable_tp"], serde_json::json!([4]));
        // A lane with no parallelism flag vocabulary (llama.cpp today)
        // gets no card at all - never a guessed axis list.
        let mut none = std::collections::BTreeSet::new();
        none.insert("--ctx-size".to_string());
        assert!(parallelism_advisor(&none, 4, 10_000, 0, Some(5800)).is_none());
    }

    #[test]
    fn unit__pd_advisory__requires_multi_gpu_long_ctx_sglang() {
        // All three gates must hold: multi-GPU, long context, sglang.
        assert!(pd_advisory(1, 16_384, true).is_none(), "single GPU");
        assert!(pd_advisory(4, 4096, true).is_none(), "short context");
        assert!(pd_advisory(4, 16_384, false).is_none(), "no sglang lane");
        let advisory = pd_advisory(4, 16_384, true).unwrap();
        assert_eq!(advisory["candidate"], serde_json::json!(true));
        assert_eq!(advisory["gpus"], serde_json::json!(4));
        assert_eq!(advisory["ctx"], serde_json::json!(16_384));
        let config = advisory["config"].as_str().unwrap();
        assert!(config.contains("disagg_pair"), "{config}");
        assert!(
            advisory["caveat"]
                .as_str()
                .unwrap()
                .contains("advisory only")
        );
    }
}
