//! `OpenAI` images lane: diffusion component-set models (`DiT` plus
//! `VAE` plus text encoder) served by the sdcpp engine's `sd-server`
//! child. The child's OpenAI-compat routes accept `{model, prompt,
//! size "WxH", steps, n, ...}` and answer `{created, data:
//! [{b64_json}], output_format}` — verified live against
//! master-890-74988b2. The gateway owns key admission, the diffusion
//! domain gate and spawn; the child owns generation shaping. Requests
//! ride the shared 10-minute HTTP budget: one 512x512x20 image
//! measures ~67s on this class of box, and the state client already
//! covers long generations.
//!
//! Two dialects, one route. Plain `OpenAI` requests (the verified
//! `{model, prompt, size, steps, n}` set) forward byte-identical to
//! the child's compat route. Anything richer — cache modes, `LoRA`
//! weights, sampling subtrees, guidance — rides the child's native
//! async job dialect (`/sdcpp/v1/img_gen` + `/sdcpp/v1/jobs/{id}`)
//! through a shallow translator, because the compat route accepts
//! unknown fields without honoring them (probe 2026-09-22: `steps`
//! changes timing, `negative_prompt` does not surface). `"async":
//! true` returns the job handle immediately; `"stream": true` relays
//! progress as `SSE`; both set → stream wins. Job poll/cancel routes
//! resolve the LIVE child only — a job dies with its child, and these
//! routes must never boot a new one.

use axum::body::{Body, Bytes};
use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap};
use axum::response::{IntoResponse, Response};
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use crate::proxy::{child_auth, child_base, ensure_with_admission, openai_error, resolve_model};
use crate::queue::{Priority, WorkClass};
use crate::state::AppState;

/// How a generations request rides the child.
#[derive(Debug, PartialEq, Eq)]
enum DeliveryMode {
    /// Verified plain `OpenAI` shape: compat route, byte-identical.
    SyncPlain,
    /// Rich dialect, caller wants the answer now: native submit +
    /// gateway-side wait, answer mapped back to the `OpenAI` shape.
    SyncNative,
    /// `"async": true`: native submit, job handle returned at once.
    AsyncNative,
    /// `"stream": true`: native submit + `SSE` progress relay.
    /// Wins when both `async` and `stream` are set.
    StreamNative,
}

/// Keys the compat route is verified to honor. Everything else in a
/// request means the native dialect, where the field is documented —
/// and actually applied — rather than silently swallowed.
const PLAIN_GENERATION_KEYS: [&str; 8] = [
    "model", "prompt", "size", "steps", "n", "user", "async", "stream",
];

/// One `SSE` poll per second. A single failed poll is a transport
/// hiccup, not child death — three CONSECUTIVE failures (≈3s) declare
/// it, matching the evict-and-retry-once tolerance of the header lane.
const JOB_POLL_INTERVAL_MS: u64 = 1_000;
const JOB_POLL_TRANSPORT_RETRIES: u32 = 3;

fn is_plain_generation_request(v: &serde_json::Value) -> bool {
    v.as_object().is_some_and(|o| {
        o.keys()
            .all(|k| PLAIN_GENERATION_KEYS.contains(&k.as_str()))
    })
}

// ---- video scratch budget (submit-time VRAM gate) ---------------------
//
// Calibrated 2026-09-23 on a quiet 8 GiB box (wan2.1 1.3B bf16 trio,
// --offload-to-cpu, steps=8, warm child): child peak minus warm baseline
// is FLAT ≈ 3.0 GiB from 5 to 33 frames at 320x320 (DiT token count
// 800 -> 3600) and ≈ 3.5 GiB at 512x512 — the constant is the umt5-xxl
// staging pass plus the VAE working set, not attention matrices. A 13
// frame 512x512 request dies deterministically upstream at that same
// 3.0 GiB ("generate_video returned no results", child alive) — NOT a
// VRAM signature, so the gate lets it through and the child's own
// failure answers. Beyond the 33-frame calibration horizon the floor
// scales linearly with frames: conservative on purpose, and the 400 it
// produces names every lever.

/// Measured warm-child scratch floor at or below the 320x320 reference.
const VIDEO_SCRATCH_FLOOR_MIB: u64 = 3_050;
/// Extra scratch per pixel of frame area beyond the reference — fit
/// from the 512x512 point (+510 MiB over 161k px²).
const VIDEO_SCRATCH_AREA_MIB_PER_PX2: f64 = 3.17e-3;
const VIDEO_SCRATCH_REF_AREA_PX2: u64 = 320 * 320;
/// Unmodeled working set (driver, allocator granularity, first-touch).
const VIDEO_SCRATCH_HEDGE_MIB: u64 = 192;
/// The measured numbers are pass/fail lower bounds; the gate sells
/// estimates, so it never undersells the measured failure edge.
const VIDEO_SCRATCH_SAFETY: f64 = 1.25;
/// Highest frame count with a measured pass; the floor scales linearly
/// past it instead of pretending flatness holds forever.
const VIDEO_SCRATCH_CALIBRATED_FRAMES: u64 = 33;
/// Warm wan-trio residency (child idle, weights staged) — charged only
/// when no warm child exists yet, because a cold spawn lands on the
/// same card the scratch will claim.
const VIDEO_CHILD_WEIGHTS_STAGED_MIB: u64 = 2_800;
/// Leave the driver and desktop a slice; foreign tenants grow too.
const VIDEO_VRAM_HEADROOM_FRACTION: f64 = 0.95;
/// Default frame geometry when the request names none — the child's own
/// 512x512 default, which is also the conservative estimate.
const VIDEO_DEFAULT_SIZE: (u64, u64) = (512, 512);

/// Wan's temporal grid: the engine aligns DOWN to 4k+1 frames (1, 5,
/// 9, 13, ...) before generating — verified live: a `video_frames:4`
/// request renders one frame and a 16-frame duration render delivered
/// 13. The gate prices what the child will actually render.
fn align_wan_frames(frames: u64) -> u64 {
    if frames <= 1 {
        1
    } else {
        4 * ((frames - 1) / 4) + 1
    }
}

struct VideoScratchEstimate {
    estimate_mib: u64,
    aligned_frames: u64,
}

// MiB-scale calibrated math: u64->f64 precision loss starts above
// 2^52 MiB and f64->u64 truncation never lands mid-MiB on values this
// small, so the plain casts are exact for every reachable input.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn estimate_video_scratch(
    frames: u64,
    width: u64,
    height: u64,
    cold_child: bool,
) -> VideoScratchEstimate {
    let aligned = align_wan_frames(frames);
    let frame_scale = (aligned as f64 / VIDEO_SCRATCH_CALIBRATED_FRAMES as f64).max(1.0);
    let area = width * height;
    let area_term = if area > VIDEO_SCRATCH_REF_AREA_PX2 {
        ((area - VIDEO_SCRATCH_REF_AREA_PX2) as f64) * VIDEO_SCRATCH_AREA_MIB_PER_PX2
    } else {
        0.0
    };
    // Weights residency is deterministic (known model files), so it is added
    // linearly; only the measured-scratch terms carry the uncertainty safety
    // multiplier — stacking safety on weights over-rejects cold spawns that
    // demonstrably fit (live 5f@320 on an 8 GiB box).
    let weights = if cold_child {
        VIDEO_CHILD_WEIGHTS_STAGED_MIB
    } else {
        0
    };
    let raw =
        VIDEO_SCRATCH_FLOOR_MIB as f64 * frame_scale + area_term + VIDEO_SCRATCH_HEDGE_MIB as f64;
    VideoScratchEstimate {
        estimate_mib: (raw * VIDEO_SCRATCH_SAFETY).ceil() as u64 + weights,
        aligned_frames: aligned,
    }
}

/// Parse `size` (``WxH``) for the gate; absent or unparseable sizes take
/// the child's 512x512 default — the conservative side of the estimate.
fn gate_size(v: &serde_json::Value) -> (u64, u64) {
    v.get("size")
        .and_then(|s| s.as_str())
        .and_then(|s| s.split_once('x'))
        .and_then(|(w, h)| {
            let w = w.trim().parse::<u64>().ok()?;
            let h = h.trim().parse::<u64>().ok()?;
            Some((w, h))
        })
        .filter(|(w, h)| *w > 0 && *h > 0)
        .unwrap_or(VIDEO_DEFAULT_SIZE)
}

/// Frame count the gate prices: the request's own `video_frames` when
/// it names one (`frames`/`num_frames`/`duration`×fps were canonicalized
/// onto it upstream), else the calibration horizon. The child's default
/// for a fully unspecified `vid_gen` is not contracted anywhere — its CLI
/// flag says 1, but bare live renders have delivered multi-frame video —
/// and a safety gate must not undersell the one dimension it cannot
/// know. Naming frames explicitly prices exactly that count.
fn gate_frames(v: &serde_json::Value) -> u64 {
    v.get("video_frames")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(VIDEO_SCRATCH_CALIBRATED_FRAMES)
}

/// Submit-time gate: refuse video requests whose measured scratch
/// envelope cannot fit the card's FREE memory right now (a warm child's
/// weights are already inside "used", a cold spawn's are not). `Ok(())`
/// when the request may proceed — including on boxes without an
/// nvidia-smi census, where the runtime's spawn heuristic still guards
/// weights and the request rides through.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn video_scratch_gate(
    parsed: &serde_json::Value,
    model: &str,
    state: &AppState,
) -> Result<(), String> {
    if parsed
        .get("vram_overcommit")
        .and_then(serde_json::Value::as_bool)
        == Some(true)
    {
        // Operator's explicit call: attempt it anyway; the child's own
        // failure (or success) answers. No silent path — the audit log
        // carries the request that used the lever.
        return Ok(());
    }
    let Some(free) = crate::vram::free_vram_mib(std::time::Duration::from_secs(5)) else {
        return Ok(());
    };
    let (w, h) = gate_size(parsed);
    let frames = gate_frames(parsed);
    let cold = live_sdcpp_children(state, Some(model)).is_empty();
    let est = estimate_video_scratch(frames, w, h, cold);
    let budget = (free as f64 * VIDEO_VRAM_HEADROOM_FRACTION) as u64;
    if est.estimate_mib > budget {
        let spawn_note = if cold {
            format!(" + ~{VIDEO_CHILD_WEIGHTS_STAGED_MIB} MiB weights (cold spawn)")
        } else {
            String::new()
        };
        return Err(format!(
            "video request would exceed free VRAM: scratch ≈ {} MiB ({} aligned frames at \
             {w}x{h}{spawn_note}, safety ×{VIDEO_SCRATCH_SAFETY}) vs {budget} MiB free of \
             {free} (95% headroom rule). Levers: fewer frames (video_frames/duration), \
             smaller size, free GPU memory held by other processes, or \
             \"vram_overcommit\": true to force the attempt",
            est.estimate_mib, est.aligned_frames,
        ));
    }
    Ok(())
}

/// Canonicalize the video frame-count vocabulary onto the child's
/// native knob. sd-server's `vid_gen` field is `video_frames`; clients
/// raised on OpenAI-ish surfaces reach for `frames` or `num_frames`,
/// and the verbatim passthrough below would let those ride past the
/// child unheard (one-frame videos, no error — verified live against
/// sd-server master-890). `duration` seconds (the REPL's `/duration`
/// knob) joins the vocabulary as `duration` × `fps` frames — sd-server
/// has no duration field, so an untranslated knob is the same silent
/// one-frame no-op. Contradictory specs fail loud instead of picking a
/// silent winner.
/// Canonicalize the frame-count vocabulary (`frames`/`num_frames`/
/// `duration`) onto sd-server's native `video_frames` before relay.
/// Conflicting spellings are named in the error rather than racing.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn canonicalize_video_frames(v: &mut serde_json::Value) -> Result<(), String> {
    const NATIVE: &str = "video_frames";
    const SYNONYMS: [&str; 2] = ["frames", "num_frames"];
    // sd-server's vid_gen default sample rate: every live job answered
    // `fps: 16` and the server's own defaults catalog carries 16.
    const DEFAULT_FPS: u64 = 16;

    let Some(obj) = v.as_object_mut() else {
        return Ok(());
    };

    let mut specs: Vec<(&str, u64)> = Vec::new();
    for key in SYNONYMS.iter().copied().chain(std::iter::once(NATIVE)) {
        if let Some(val) = obj.get(key).cloned() {
            let n = val
                .as_u64()
                .filter(|n| *n >= 1)
                .ok_or_else(|| format!("{key} must be a positive integer (got {val})"))?;
            specs.push((key, n));
        }
    }

    // `duration` seconds (the REPL's `/duration` knob): sd-server has no
    // duration field, so the knob translates to `duration` x `fps`
    // frames here — an untranslated ride past the child is the same
    // silent one-frame no-op the synonyms above would suffer.
    if let Some(val) = obj.get("duration").cloned() {
        let fps = obj
            .get("fps")
            .and_then(|f| f.as_u64().filter(|f| *f >= 1))
            .unwrap_or(DEFAULT_FPS);
        let dur_secs = val
            .as_f64()
            .filter(|s| s.is_finite() && *s > 0.0)
            .ok_or_else(|| format!("duration must be a positive number of seconds (got {val})"))?;
        obj.remove("duration");
        let dur_frames = ((dur_secs * fps as f64).round() as u64).max(1);
        specs.push(("duration", dur_frames));
    }

    if specs.is_empty() {
        return Ok(());
    }
    let values: std::collections::BTreeSet<u64> = specs.iter().map(|s| s.1).collect();
    if values.len() > 1 {
        let named = specs
            .iter()
            .map(|(k, n)| format!("{k}={n}"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "conflicting frame specs ({named}): send one — the native field is {NATIVE}"
        ));
    }
    let n = specs[0].1;
    for key in SYNONYMS.iter().copied().chain(std::iter::once(NATIVE)) {
        obj.remove(key);
    }
    obj.insert(NATIVE.to_string(), serde_json::json!(n));
    Ok(())
}

/// `n`→`batch_count`, `size "WxH"`→`width`/`height`, `steps`→
/// `sample_params.sample_steps` (merged into an existing subtree);
/// control keys drop; every other key — including whole subtrees like
/// `guidance`, `cache`, `lora`, `vae_tiling_params` — rides verbatim,
/// so upstream-additive fields need no translator change.
fn translate_to_native(v: &serde_json::Value, video: bool) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    let Some(obj) = v.as_object() else {
        return v.clone();
    };
    for (k, val) in obj {
        match k.as_str() {
            "model" | "user" | "async" | "stream" => {}
            // `img_gen` batches through `batch_count`; the `vid_gen`
            // shape has no such field, so the key rides untouched.
            "n" if !video => {
                out.insert("batch_count".into(), val.clone());
            }
            "steps" => {
                let sp = out
                    .entry("sample_params".to_string())
                    .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
                if let Some(sp) = sp.as_object_mut() {
                    sp.insert("sample_steps".into(), val.clone());
                }
            }
            "size" => {
                let dims = val
                    .as_str()
                    .and_then(|s| s.split_once('x'))
                    .and_then(|(w, h)| {
                        Some((w.trim().parse::<u32>().ok()?, h.trim().parse::<u32>().ok()?))
                    });
                if let Some((w, h)) = dims {
                    out.insert("width".into(), serde_json::json!(w));
                    out.insert("height".into(), serde_json::json!(h));
                } else {
                    // Unparseable size passes through untouched — the
                    // child's own 400 teaches better than we can.
                    out.insert(k.clone(), val.clone());
                }
            }
            _ => {
                out.insert(k.clone(), val.clone());
            }
        }
    }
    serde_json::Value::Object(out)
}

fn delivery_mode(v: &serde_json::Value) -> DeliveryMode {
    let stream = v.get("stream").and_then(serde_json::Value::as_bool) == Some(true);
    let asynch = v.get("async").and_then(serde_json::Value::as_bool) == Some(true);
    if stream {
        DeliveryMode::StreamNative
    } else if asynch {
        DeliveryMode::AsyncNative
    } else if !is_plain_generation_request(v) {
        // Rich dialect, no async flag: the caller still holds OpenAI
        // sync semantics — wait and answer in the OpenAI shape.
        DeliveryMode::SyncNative
    } else {
        DeliveryMode::SyncPlain
    }
}

/// Domain gate for the images routes. `row: None` (unknown model) is NOT
/// a rejection — `ensure_with_admission` owns that 404 — it only means
/// the gate has nothing to check.
pub(crate) enum ImagesGate {
    Serve,
    Reject(String),
}

/// Which diffusion surface a request arrived on. The family table
/// decides the match before any child exists: image families teach the
/// videos route and vice versa, so a caller never boots a model that
/// would 404 one hop later.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Surface {
    ImgGen,
    ImgEdits,
    VidGen,
}

/// Family mode of a stored row: `false` when the repo is not a curated
/// video family (image family or unknown — both serve image surfaces).
fn row_is_video(row: &blazar_core::store::ModelRow) -> bool {
    matches!(
        blazar_runtime::diffusion::family_mode(&row.repo),
        Some(blazar_runtime::diffusion::FamilyMode::Vid)
    )
}

/// Teaching text for a diffusion component set hit through a text or
/// embedding surface (chat/completions, generate, embeddings, messages,
/// rerank...). Mirror of the `images_gate` rejection above: one gate per
/// direction, same vocabulary both ways.
pub(crate) fn diffusion_text_refusal(name: &str) -> String {
    format!(
        "\"{name}\" is a diffusion component set — text endpoints cannot drive it; \
         image generation serves POST /v1/images/generations \
         (instruction edits: POST /v1/images/edits)"
    )
}

/// Only diffusion component rows reach an sdcpp child: a chat model
/// here would boot a text engine that 404s the forward one hop later.
/// Edits additionally need the vision-encoder companion (`--llm_vision`
/// rides the spawn argv only when the row carries it).
pub(crate) fn images_gate(
    row: Option<&blazar_core::store::ModelRow>,
    surface: Surface,
) -> ImagesGate {
    let Some(row) = row else {
        return ImagesGate::Serve;
    };
    if !row.has_component_set() {
        return ImagesGate::Reject(format!(
            "\"{}\" is not a diffusion model — /v1/images and /v1/videos serve diffusion \
             component sets (DiT + VAE + text encoder, sdcpp lane); chat models serve \
             /v1/chat/completions",
            row.name
        ));
    }
    match surface {
        Surface::VidGen => {
            if !row_is_video(row) {
                return ImagesGate::Reject(format!(
                    "\"{}\" is an image diffusion family — video generation serves \
                     /v1/videos/generations on video families (Wan)",
                    row.name
                ));
            }
        }
        Surface::ImgGen => {
            if row_is_video(row) {
                return ImagesGate::Reject(format!(
                    "\"{}\" is a video diffusion family — image generation serves \
                     /v1/images/generations; video serves POST /v1/videos/generations",
                    row.name
                ));
            }
        }
        Surface::ImgEdits => {}
    }
    if surface == Surface::ImgEdits && !row.serves_image_edits() {
        // Family-aware refusal: a vision-less family (FLUX) would loop
        // forever on re-pull teaching — its set is already complete.
        if blazar_runtime::diffusion::family_supports_edits(&row.repo) == Some(false) {
            return ImagesGate::Reject(format!(
                "\"{}\" belongs to a diffusion family without a vision encoder — \
                 instruction edits are not supported; use /v1/images/generations",
                row.name
            ));
        }
        return ImagesGate::Reject(format!(
            "image edits need the vision-encoder companion — \"{}\" was pulled without one; \
             re-pull it to fetch the set: blazar pull {}",
            row.name, row.name
        ));
    }
    ImagesGate::Serve
}

/// Forward the (already gated) request bytes to the child's same-named
/// OpenAI-compat route and relay the JSON answer. Errors mirror the
/// responses-lane shapes: transport failure reaps, non-2xx relays the
/// engine text, a body decode failure is a 502.
async fn forward_images(
    state: &AppState,
    engine: &blazar_runtime::EngineRef,
    path: &str,
    content_type: Option<&str>,
    body: &[u8],
    load_ms: u128,
) -> Response {
    // Crash-window retry: a dead child (capacity eviction, warm-up
    // crash) surfaces here as a transport error — respawn + one retry
    // instead of a 502 the client never caused. Same contract as the
    // proxy text-lane forward.
    let upstream = crate::proxy::send_with_child_retry(state, engine, |eng| {
        // Media client: this forward holds the request open for the
        // whole render — a total timeout here is the binding ceiling
        // long before the child's own lanes fire.
        let mut rb = crate::state::media_child_client(state, &eng.endpoint)
            .post(format!("{}{path}", child_base(&eng.endpoint)));
        if let Some(ct) = content_type {
            rb = rb.header(header::CONTENT_TYPE, ct);
        }
        child_auth(rb, eng).body(body.to_vec())
    })
    .await;
    let resp = match upstream {
        Ok(r) => r,
        Err(msg) => return openai_error(502, &msg),
    };
    let status = resp.status().as_u16();
    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        return openai_error(status, &format!("engine error: {text}"));
    }
    let bytes = match resp.bytes().await {
        Ok(b) => b,
        Err(e) => return openai_error(502, &format!("bad engine response: {e}")),
    };
    let mut builder = Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json");
    if load_ms > 100 {
        builder = builder.header("x-blazar-status", "loading");
    }
    builder
        .body(Body::from(bytes))
        .unwrap_or_else(|e| openai_error(500, &format!("response build: {e}")).into_response())
}

/// POST /v1/images/generations — text-to-image on a diffusion set.
pub async fn generations(
    State(state): State<Arc<AppState>>,
    key_ext: Option<axum::Extension<crate::keys::KeyCtx>>,
    body: Bytes,
) -> Response {
    let Some(model) = crate::audit::extract_model(&body) else {
        return openai_error(400, "\"model\" is required (diffusion model name)");
    };
    if let Err(resp) = state.admit_or_respond(key_ext.as_ref(), &model) {
        return *resp;
    }
    let parsed: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return openai_error(400, &format!("invalid JSON: {e}")),
    };
    if parsed
        .get("prompt")
        .and_then(serde_json::Value::as_str)
        .is_none_or(str::is_empty)
    {
        return openai_error(400, "\"prompt\" must be a non-empty string");
    }
    let row = state
        .with_store(|s| resolve_model(s, &model).ok())
        .flatten();
    match images_gate(row.as_ref(), Surface::ImgGen) {
        ImagesGate::Serve => {}
        ImagesGate::Reject(msg) => return openai_error(400, &msg),
    }
    let (engine, load_ms) = match ensure_with_admission(
        &state,
        &model,
        Priority::Normal,
        WorkClass::Interactive,
        None,
        false, // images: no mmproj lane — the vision encoder rides argv
        true,  // images lane: component sets are its cargo
        false,
    )
    .await
    {
        Ok(ok) => ok,
        Err(resp) => return *resp,
    };
    match delivery_mode(&parsed) {
        DeliveryMode::SyncPlain => {
            // Verified dialect: byte-identical compat forward.
            forward_images(
                &state,
                &engine,
                "/v1/images/generations",
                Some("application/json"),
                &body,
                load_ms,
            )
            .await
        }
        mode => {
            deliver_native(
                state,
                engine,
                &parsed,
                mode,
                "/sdcpp/v1/img_gen",
                false,
                load_ms,
            )
            .await
        }
    }
}

/// Native-dialect delivery shared by the three non-plain modes:
/// submit once, then hand back an async handle, a mapped sync answer,
/// or an `SSE` progress relay. Admission was charged at submit.
async fn deliver_native(
    state: Arc<AppState>,
    engine: blazar_runtime::EngineRef,
    parsed: &serde_json::Value,
    mode: DeliveryMode,
    native_path: &str,
    video: bool,
    load_ms: u128,
) -> Response {
    let submitted = match submit_native_job(
        &state,
        &engine,
        native_path,
        &translate_to_native(parsed, video),
    )
    .await
    {
        Ok(v) => v,
        Err(resp) => return *resp,
    };
    let Some(job_id) = submitted.get("id").and_then(serde_json::Value::as_str) else {
        return openai_error(502, "engine accepted the job but returned no id");
    };
    match mode {
        DeliveryMode::AsyncNative => async_handle_response(submitted, load_ms),
        DeliveryMode::StreamNative => stream_native_job(state, engine, job_id.to_string()),
        DeliveryMode::SyncNative => {
            // Keep the child non-idle for the whole await (audit MM5):
            // the poll loop below bypasses `ensure`, so this bracket is
            // the only thing telling the reaper a render is in flight.
            let _activity =
                ChildActivityGuard::begin(std::sync::Arc::clone(&state.sup), &engine.name);
            let job = match await_native_job(&state, &engine, job_id).await {
                Ok(j) => j,
                Err(resp) => return *resp,
            };
            if job.get("status").and_then(serde_json::Value::as_str) != Some("completed") {
                let msg = job
                    .pointer("/error/message")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("generation did not complete");
                return openai_error(502, msg);
            }
            let mut builder = Response::builder()
                .status(200)
                .header(header::CONTENT_TYPE, "application/json");
            if load_ms > 100 {
                builder = builder.header("x-blazar-status", "loading");
            }
            builder
                .body(Body::from(
                    serde_json::to_vec(&native_job_to_openai(&job)).unwrap_or_default(),
                ))
                .unwrap_or_else(|e| {
                    openai_error(500, &format!("response build: {e}")).into_response()
                })
        }
        // Plain requests never reach this helper: they ride the
        // byte-identical compat forward in `generations`.
        DeliveryMode::SyncPlain => unreachable!("plain requests ride the compat forward"),
    }
}

/// POST /v1/videos/generations — text-to-video on a video family
/// (Wan). Same contract as image generations with two differences:
/// the gate demands a curated video family, and there is no compat
/// forward — sd-server's OpenAI-compat route serves images only, so
/// every request rides the native `vid_gen` dialect.
pub async fn video_generations(
    State(state): State<Arc<AppState>>,
    key_ext: Option<axum::Extension<crate::keys::KeyCtx>>,
    body: Bytes,
) -> Response {
    let Some(model) = crate::audit::extract_model(&body) else {
        return openai_error(400, "\"model\" is required (video diffusion model name)");
    };
    if let Err(resp) = state.admit_or_respond(key_ext.as_ref(), &model) {
        return *resp;
    }
    let mut parsed: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return openai_error(400, &format!("invalid JSON: {e}")),
    };
    if parsed
        .get("prompt")
        .and_then(serde_json::Value::as_str)
        .is_none_or(str::is_empty)
    {
        return openai_error(400, "\"prompt\" must be a non-empty string");
    }
    if let Err(msg) = canonicalize_video_frames(&mut parsed) {
        return openai_error(400, &msg);
    }
    // Family gate BEFORE the VRAM scratch gate (audit MM10): a wrong-
    // family request (image set on the video route) must hear the lane
    // teaching first — complying with VRAM levers (fewer frames, smaller
    // size) could never fix a family mismatch, so that error order lied.
    let row = state
        .with_store(|s| resolve_model(s, &model).ok())
        .flatten();
    match images_gate(row.as_ref(), Surface::VidGen) {
        ImagesGate::Serve => {}
        ImagesGate::Reject(msg) => return openai_error(400, &msg),
    }
    if let Err(msg) = video_scratch_gate(&parsed, &model, &state) {
        return openai_error(400, &msg);
    }
    // The lever is gateway-only vocabulary; it never rides to the child.
    if let Some(obj) = parsed.as_object_mut() {
        obj.remove("vram_overcommit");
    }
    let (engine, load_ms) = match ensure_with_admission(
        &state,
        &model,
        Priority::Normal,
        WorkClass::Interactive,
        None,
        false,
        true, // videos lane: component sets are its cargo
        false,
    )
    .await
    {
        Ok(ok) => ok,
        Err(resp) => return *resp,
    };
    let mode = match delivery_mode(&parsed) {
        // No compat route exists for video — plain requests also ride
        // the native dialect and come back mapped OpenAI-style.
        DeliveryMode::SyncPlain | DeliveryMode::SyncNative => DeliveryMode::SyncNative,
        mode => mode,
    };
    deliver_native(
        state,
        engine,
        &parsed,
        mode,
        "/sdcpp/v1/vid_gen",
        true,
        load_ms,
    )
    .await
}

/// POST /v1/images/edits — image-to-image; multipart body forwarded
/// byte-for-byte (the boundary is the child's contract, never rebuilt).
pub async fn edits(
    State(state): State<Arc<AppState>>,
    key_ext: Option<axum::Extension<crate::keys::KeyCtx>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let Some(model) = content_type
        .as_deref()
        .and_then(|ct| crate::openai::extract_model_multipart(&body, ct))
    else {
        return openai_error(
            400,
            "\"model\" form field is required (diffusion model name)",
        );
    };
    if let Err(resp) = state.admit_or_respond(key_ext.as_ref(), &model) {
        return *resp;
    }
    let row = state
        .with_store(|s| resolve_model(s, &model).ok())
        .flatten();
    match images_gate(row.as_ref(), Surface::ImgEdits) {
        ImagesGate::Serve => {}
        ImagesGate::Reject(msg) => return openai_error(400, &msg),
    }
    let (engine, load_ms) = match ensure_with_admission(
        &state,
        &model,
        Priority::Normal,
        WorkClass::Interactive,
        None,
        false,
        true, // images lane: component sets are its cargo
        false,
    )
    .await
    {
        Ok(ok) => ok,
        Err(resp) => return *resp,
    };
    forward_images(
        &state,
        &engine,
        "/v1/images/edits",
        content_type.as_deref(),
        &body,
        load_ms,
    )
    .await
}

// ---- native job dialect bridge ------------------------------------

/// Live `sd-server` children, optionally narrowed to one model. The
/// job routes resolve against this snapshot ONLY — they never spawn:
/// a job belongs to the child that accepted it and dies with it.
fn live_sdcpp_children(state: &AppState, model: Option<&str>) -> Vec<blazar_runtime::EngineRef> {
    state
        .sup
        .live_http_endpoints()
        .into_iter()
        .filter(|e| e.kind == blazar_core::engine_kind::EngineKind::SdCpp)
        .filter(|e| model.is_none_or(|m| e.name == m))
        .collect()
}

/// Submit a translated request to the child's native job dialect.
/// `Err` is a ready 502/relay response.
async fn submit_native_job(
    state: &AppState,
    engine: &blazar_runtime::EngineRef,
    native_path: &str,
    native: &serde_json::Value,
) -> Result<serde_json::Value, Box<Response>> {
    // Crash-window retry — same contract as forward_images above: a
    // dead child surfaces as transport error, respawn + one retry.
    let resp = match crate::proxy::send_with_child_retry(state, engine, |eng| {
        let url = format!("{}{native_path}", child_base(&eng.endpoint));
        child_auth(
            crate::state::child_client(state, &eng.endpoint)
                .post(&url)
                .json(native),
            eng,
        )
    })
    .await
    {
        Ok(r) => r,
        Err(msg) => return Err(Box::new(openai_error(502, &msg))),
    };
    let status = resp.status().as_u16();
    let body_text = resp.text().await.unwrap_or_default();
    if !(200..300).contains(&status) {
        return Err(Box::new(openai_error(
            status,
            &format!("engine error: {body_text}"),
        )));
    }
    serde_json::from_str(&body_text)
        .map_err(|e| Box::new(openai_error(502, &format!("bad engine response: {e}"))))
}

/// Holds supervision activity for a native job's poll lifetime. Sync
/// and stream renders poll the child DIRECTLY (not through `ensure`),
/// so the supervisor's idle clock never sees them: without this bracket
/// the reaper can evict the child mid-render once `media_job_wait_secs`
/// outruns `idle_timeout_secs` (audit MM5). Same begin/end bracket the
/// text lanes hold via `InFlightGuard` — begin marks in-flight (the
/// reaper never evicts under load) and both ends refresh `last_used`.
struct ChildActivityGuard {
    sup: std::sync::Arc<blazar_runtime::Supervisor>,
    name: String,
}

impl ChildActivityGuard {
    fn begin(sup: std::sync::Arc<blazar_runtime::Supervisor>, name: &str) -> Self {
        sup.begin_request(name);
        Self {
            sup,
            name: name.to_string(),
        }
    }
}

impl Drop for ChildActivityGuard {
    fn drop(&mut self) {
        self.sup.end_request(&self.name);
    }
}

/// One poll of the child's job state; `Err` = transport/decode
/// failure (child died mid-poll), `Ok(None)` = HTTP 404 from the
/// child (job unknown there).
async fn poll_child_job(
    state: &AppState,
    engine: &blazar_runtime::EngineRef,
    job_id: &str,
) -> Result<Option<serde_json::Value>, ()> {
    let url = format!("{}/sdcpp/v1/jobs/{job_id}", child_base(&engine.endpoint));
    let resp = child_auth(
        crate::state::child_client(state, &engine.endpoint).get(&url),
        engine,
    )
    .send()
    .await
    .map_err(|_| ())?;
    if resp.status() == axum::http::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    let status = resp.status().as_u16();
    let text = resp.text().await.map_err(|_| ())?;
    if !(200..300).contains(&status) {
        return Err(());
    }
    serde_json::from_str(&text).map_err(|_| ())
}

/// Best-effort child-side cancel (stream disconnect). The upstream
/// endpoint resets the connection after the cancel lands (probe
/// 2026-09-22, see `jobs_cancel`) — any transport failure here is
/// treated as taken-effect, never worth surfacing.
async fn cancel_child_job(state: &AppState, engine: &blazar_runtime::EngineRef, job_id: &str) {
    let url = format!(
        "{}/sdcpp/v1/jobs/{job_id}/cancel",
        child_base(&engine.endpoint)
    );
    let _ = child_auth(
        crate::state::child_client(state, &engine.endpoint).post(&url),
        engine,
    )
    .send()
    .await;
}

/// Map a completed native job to the `OpenAI` images shape the sync
/// generations contract promises: `{created, data: [{b64_json}...],
/// output_format}`.
fn native_job_to_openai(job: &serde_json::Value) -> serde_json::Value {
    let images = job
        .pointer("/result/images")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    // vid_gen answers with a FLAT payload (b64_json + fps + frame_count +
    // mime_type at the result root — verified against sd-server master-890);
    // img_gen nests per-image objects under images[]. Map the flat video
    // bytes into the same data[] envelope so clients read one shape.
    let data = if images.is_empty() {
        job.pointer("/result/b64_json")
            .filter(|v| !v.is_null())
            .map(|b64| {
                vec![serde_json::json!({
                    "b64_json": b64,
                    "fps": job.pointer("/result/fps").cloned().unwrap_or(serde_json::Value::Null),
                    "frame_count": job.pointer("/result/frame_count").cloned().unwrap_or(serde_json::Value::Null),
                    "mime_type": job.pointer("/result/mime_type").cloned().unwrap_or(serde_json::Value::Null),
                })]
            })
            .unwrap_or_default()
    } else {
        images
    };
    let output_format = job
        .pointer("/result/output_format")
        .cloned()
        .unwrap_or_else(|| serde_json::json!("png"));
    serde_json::json!({
        "created": job.get("created").cloned().unwrap_or_else(|| serde_json::json!(0)),
        "data": data,
        "output_format": output_format,
    })
}

/// Wait for a submitted job until terminal (completed/failed/
/// cancelled) or the configured wait budget (`media_job_wait_secs`;
/// 0 = no gateway cap) is spent. `Ok(job)` = terminal job; `Err` =
/// child died or budget exhausted (ready 502/504 response).
async fn await_native_job(
    state: &AppState,
    engine: &blazar_runtime::EngineRef,
    job_id: &str,
) -> Result<serde_json::Value, Box<Response>> {
    let budget_secs = state.config.media_job_wait_secs;
    let mut polled = 0_u64;
    let mut transport_fails = 0_u32;
    loop {
        if budget_secs != 0 && polled >= budget_secs {
            // The client walk away at the budget, but the child keeps
            // rendering — cancel it instead of burning the GPU on an
            // answer nobody will read (best-effort; upstream reset
            // probe 2026-09-22).
            cancel_child_job(state, engine, job_id).await;
            return Err(Box::new(openai_error(
                504,
                &format!(
                    "generation did not finish within {budget_secs}s — raise media_job_wait_secs \
                     or use \"async\": true and poll the job handle"
                ),
            )));
        }
        tokio::time::sleep(std::time::Duration::from_millis(JOB_POLL_INTERVAL_MS)).await;
        polled += 1;
        match poll_child_job(state, engine, job_id).await {
            Ok(Some(job)) => {
                transport_fails = 0;
                let status = job.get("status").and_then(serde_json::Value::as_str);
                if matches!(status, Some("completed" | "failed" | "cancelled")) {
                    return Ok(job);
                }
            }
            Ok(None) => {
                return Err(Box::new(openai_error(
                    502,
                    "job vanished from the engine child mid-generation",
                )));
            }
            Err(()) => {
                transport_fails += 1;
                if transport_fails >= JOB_POLL_TRANSPORT_RETRIES {
                    state.sup.reap_dead_children().await;
                    return Err(Box::new(openai_error(
                        502,
                        "engine child died mid-generation",
                    )));
                }
                // One failed poll is a hiccup, not death — keep polling.
            }
        }
    }
}

/// `SSE` relay: progress events every poll, one terminal event, then
/// close. The spawned task stops on client disconnect (send fails —
/// and cancels the child job so nobody pays GPU for an uncollected
/// render), child death, or the `media_job_wait_secs` budget.
fn stream_native_job(
    state: Arc<AppState>,
    engine: blazar_runtime::EngineRef,
    job_id: String,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(16);
    tokio::spawn(async move {
        // Same bracket as the sync arm (audit MM5): the relay's poll
        // loop must read as activity for as long as it runs, or the
        // reaper can idle-evict the child under a long stream.
        let _activity = ChildActivityGuard::begin(std::sync::Arc::clone(&state.sup), &engine.name);
        let budget_secs = state.config.media_job_wait_secs;
        let mut polled = 0_u64;
        let mut transport_fails = 0_u32;
        loop {
            if budget_secs != 0 && polled >= budget_secs {
                // Same policy as the sync await path: the client is
                // gone at budget expiry, so stop the render instead of
                // burning the GPU to completion (best-effort cancel).
                cancel_child_job(&state, &engine, &job_id).await;
                let msg = format!(
                    "generation did not finish within {budget_secs}s — raise \
                     media_job_wait_secs or use \"async\": true"
                );
                let frame = format!(
                    "event: error\ndata: {}\n\n",
                    serde_json::json!({"status": "error", "error": {"message": msg}})
                );
                let _ = tx.send(Ok(Bytes::from(frame))).await;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(JOB_POLL_INTERVAL_MS)).await;
            polled += 1;
            let job = match poll_child_job(&state, &engine, &job_id).await {
                Ok(Some(job)) => {
                    transport_fails = 0;
                    job
                }
                Ok(None) => {
                    let frame = format!(
                        "event: error\ndata: {}\n\n",
                        serde_json::json!({
                            "status": "error",
                            "error": {"message": "job vanished from the engine child mid-generation"}
                        })
                    );
                    let _ = tx.send(Ok(Bytes::from(frame))).await;
                    break;
                }
                Err(()) => {
                    transport_fails += 1;
                    if transport_fails >= JOB_POLL_TRANSPORT_RETRIES {
                        let frame = format!(
                            "event: error\ndata: {}\n\n",
                            serde_json::json!({
                                "status": "error",
                                "error": {"message": "engine child died mid-generation"}
                            })
                        );
                        let _ = tx.send(Ok(Bytes::from(frame))).await;
                        break;
                    }
                    // One failed poll is a hiccup, not death — keep polling.
                    continue;
                }
            };
            let status = job.get("status").and_then(serde_json::Value::as_str);
            let terminal = matches!(status, Some("completed" | "failed" | "cancelled"));
            let (event, payload) = if terminal {
                match status {
                    Some("completed") => ("completed", job.clone()),
                    _ => ("error", job.clone()),
                }
            } else {
                (
                    "progress",
                    serde_json::json!({
                        "status": status,
                        "progress": job.get("progress").cloned().unwrap_or(serde_json::json!(null)),
                        "queue_position": job.get("queue_position").cloned().unwrap_or(serde_json::json!(null)),
                    }),
                )
            };
            let frame = format!("event: {event}\ndata: {payload}\n\n");
            if tx.send(Ok(Bytes::from(frame))).await.is_err() {
                // Client went away mid-render: stop paying GPU for a
                // clip nobody will collect.
                cancel_child_job(&state, &engine, &job_id).await;
                break;
            }
            if terminal {
                break;
            }
        }
    });
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .unwrap_or_else(|e| openai_error(500, &format!("stream build: {e}")).into_response())
}

/// Optional job-route query: `?model=NAME` narrows child resolution.
#[derive(serde::Deserialize)]
pub struct JobQuery {
    model: Option<String>,
}

/// Job-handle response for `"async": true`: the upstream submit body
/// with gateway-owned URLs (the child's `poll_url` addresses its own
/// dialect; clients hold gateway addresses).
fn async_handle_response(submitted: serde_json::Value, load_ms: u128) -> Response {
    let id = submitted
        .get("id")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();
    let mut out = submitted;
    if let Some(obj) = out.as_object_mut() {
        obj.insert(
            "poll_url".into(),
            serde_json::json!(format!("/v1/images/jobs/{id}")),
        );
        obj.insert(
            "cancel_url".into(),
            serde_json::json!(format!("/v1/images/jobs/{id}/cancel")),
        );
    }
    let mut builder = Response::builder()
        .status(200)
        .header(header::CONTENT_TYPE, "application/json");
    if load_ms > 100 {
        builder = builder.header("x-blazar-status", "loading");
    }
    builder
        .body(Body::from(serde_json::to_vec(&out).unwrap_or_default()))
        .unwrap_or_else(|e| openai_error(500, &format!("response build: {e}")).into_response())
}

/// GET /v1/images/jobs/{id}?model=NAME — poll a native job on the
/// LIVE child. Never spawns; jobs die with their child.
pub async fn jobs_get(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
    Query(params): Query<JobQuery>,
) -> Response {
    if !job_id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return openai_error(400, "invalid job id");
    }
    let children = live_sdcpp_children(&state, params.model.as_deref());
    if children.is_empty() {
        return openai_error(
            404,
            "no live diffusion child serves jobs — POST /v1/images/generations boots one",
        );
    }
    for engine in &children {
        // Every client poll counts as activity (audit MM5): async jobs
        // have no gateway-side bracket, so an actively-polled job keeps
        // its child warm; once the client stops polling, normal idle
        // eviction applies (the documented die-with-child contract).
        state.sup.begin_request(&engine.name);
        let polled = poll_child_job(&state, engine, &job_id).await;
        state.sup.end_request(&engine.name);
        if let Ok(Some(job)) = polled {
            return Response::builder()
                .status(200)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&job).unwrap_or_default()))
                .unwrap_or_else(|e| {
                    openai_error(500, &format!("response build: {e}")).into_response()
                });
        }
        // Not this child's job, or the child flapped mid-poll: either
        // way, try the siblings.
    }
    openai_error(
        404,
        "job not found on any live diffusion child — jobs die with their engine child \
         (eviction, crash or restart); submit a new generation",
    )
}

/// POST /v1/images/jobs/{id}/cancel — cancel on the LIVE child. The
/// upstream cancel endpoint resets the connection after taking effect
/// (probe 2026-09-22): a transport failure here is treated as
/// likely-success and followed by one state poll whose verdict is the
/// response.
pub async fn jobs_cancel(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
    Query(params): Query<JobQuery>,
) -> Response {
    if !job_id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return openai_error(400, "invalid job id");
    }
    let children = live_sdcpp_children(&state, params.model.as_deref());
    if children.is_empty() {
        return openai_error(
            404,
            "no live diffusion child serves jobs — POST /v1/images/generations boots one",
        );
    }
    let mut last_known: Option<(serde_json::Value, blazar_runtime::EngineRef)> = None;
    for engine in &children {
        // Locate the owning child first — cancel on a non-owner would
        // 404 or reset pointlessly.
        if let Ok(Some(job)) = poll_child_job(&state, engine, &job_id).await {
            last_known = Some((job, engine.clone()));
            break;
        }
    }
    let Some((job, engine)) = last_known else {
        return openai_error(
            404,
            "job not found on any live diffusion child — jobs die with their engine child",
        );
    };
    if job.get("completed").and_then(serde_json::Value::as_bool) == Some(true) {
        return Response::builder()
            .status(200)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_vec(&job).unwrap_or_default()))
            .unwrap_or_else(|e| {
                openai_error(500, &format!("response build: {e}")).into_response()
            });
    }
    let url = format!(
        "{}/sdcpp/v1/jobs/{job_id}/cancel",
        child_base(&engine.endpoint)
    );
    // Upstream cancel answers OR resets the connection after the cancel
    // lands — both are "probably cancelled"; the poll decides. One real
    // refusal exists though (live-verified on master-890): HTTP 409
    // "job is currently generating and cannot be interrupted yet" — this
    // build only cancels queued jobs. Relay that truth instead of
    // answering a fake-optimistic 200.
    let sent = child_auth(
        crate::state::child_client(&state, &engine.endpoint).post(&url),
        &engine,
    )
    .json(&serde_json::json!({}))
    .send()
    .await;
    if let Ok(resp) = sent {
        if resp.status().as_u16() == 409 {
            let upstream_msg = resp
                .json::<serde_json::Value>()
                .await
                .ok()
                .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(str::to_string))
                .unwrap_or_else(|| "cancel refused".to_string());
            return openai_error(
                409,
                &format!(
                    "engine refused the cancel: {upstream_msg} — this build only cancels \
                     queued jobs; generating ones run to completion"
                ),
            )
            .into_response();
        }
    }
    match poll_child_job(&state, &engine, &job_id).await {
        Ok(Some(verdict)) => Response::builder()
            .status(200)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_vec(&verdict).unwrap_or_default()))
            .unwrap_or_else(|e| openai_error(500, &format!("response build: {e}")).into_response()),
        _ => openai_error(
            502,
            "cancel was sent but the engine child did not confirm job state afterwards",
        ),
    }
}

/// GET /v1/images/capabilities?model=NAME — the live child's sampler/
/// cache/LoRA menu. Read-only: boots nothing; no live child teaches
/// instead. `?model=` narrows to that family's child — on a multi-family
/// box the first live child is otherwise an arbitrary pick (audit MM3).
pub async fn capabilities(
    State(state): State<Arc<AppState>>,
    Query(params): Query<JobQuery>,
) -> Response {
    let children = live_sdcpp_children(&state, params.model.as_deref());
    let Some(engine) = children.first() else {
        return openai_error(
            400,
            "no live diffusion child — POST /v1/images/generations boots one, \
             then capabilities serve (pass ?model=NAME to pick a family's child)",
        );
    };
    let url = format!("{}/sdcpp/v1/capabilities", child_base(&engine.endpoint));
    match child_auth(
        crate::state::child_client(&state, &engine.endpoint).get(&url),
        engine,
    )
    .send()
    .await
    {
        Ok(resp) if resp.status().is_success() => match resp.bytes().await {
            Ok(bytes) => Response::builder()
                .status(200)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(bytes))
                .unwrap_or_else(|e| {
                    openai_error(500, &format!("response build: {e}")).into_response()
                }),
            Err(e) => openai_error(502, &format!("bad engine response: {e}")),
        },
        Ok(resp) => openai_error(
            resp.status().as_u16(),
            &format!("engine error: {}", resp.text().await.unwrap_or_default()),
        ),
        Err(e) => openai_error(502, &format!("engine request failed: {e:#}")),
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    fn row(vae: Option<&str>, vision: Option<&str>) -> blazar_core::store::ModelRow {
        // vae=None models a TEXT row: no components at all. With the
        // component set present (--vae), the Qwen-Image pair follows.
        let mut components = Vec::new();
        if let Some(vae) = vae {
            components.push(blazar_core::store::ComponentFile::new("--vae", vae));
            components.push(blazar_core::store::ComponentFile::new("--llm", "te.gguf"));
        }
        if let Some(vision) = vision {
            components.push(blazar_core::store::ComponentFile::new(
                "--llm_vision",
                vision,
            ));
        }
        blazar_core::store::ModelRow {
            name: "qwen-image-2.1".into(),
            repo: "abenzerps/Qwen-Image-2.1-GGUF".into(),
            quant: "Q4_K_M".into(),
            path: "dit.gguf".into(),
            bytes: 1,
            sha256: None,
            mmproj_path: None,
            components,
            shards: 1,
            arch: None,
            params: None,
            ctx_train: None,
            pulled_at: 1,
        }
    }

    /// FLUX row: full component set, no vision encoder in the family.
    fn flux_row() -> blazar_core::store::ModelRow {
        let mut row = row(Some("ae.safetensors"), None);
        row.name = "flux.1-dev".into();
        row.repo = "city96/FLUX.1-dev-gguf".into();
        row.components = vec![
            blazar_core::store::ComponentFile::new("--vae", "ae.safetensors"),
            blazar_core::store::ComponentFile::new("--t5xxl", "t5.gguf"),
            blazar_core::store::ComponentFile::new("--clip_l", "clip_l.safetensors"),
        ];
        row
    }

    #[test]
    fn unit__images_gate__text_model_teaches_chat_route() {
        let gate = images_gate(Some(&row(None, None)), Surface::ImgGen);
        let ImagesGate::Reject(msg) = gate else {
            panic!("text model must be rejected");
        };
        assert!(msg.contains("not a diffusion model"));
        assert!(msg.contains("/v1/chat/completions"), "{msg}");
    }

    #[test]
    fn unit__diffusion_text_refusal__teaches_images_api_on_text_surfaces() {
        let msg = diffusion_text_refusal("qwen-image-2.1");
        assert!(msg.contains("\"qwen-image-2.1\""), "{msg}");
        assert!(msg.contains("text endpoints cannot drive"), "{msg}");
        assert!(msg.contains("/v1/images/generations"), "{msg}");
        assert!(msg.contains("/v1/images/edits"), "{msg}");
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__images_gate__video_surface_partition_teaches_both_directions() {
        let mut wan = row(Some("wan_2.1_vae.safetensors"), None);
        wan.repo = "Comfy-Org/Wan_2.1_ComfyUI_repackaged".into();
        wan.components.push(blazar_core::store::ComponentFile::new(
            "--t5xxl",
            "/models/umt5-xxl-encoder-Q4_K_M.gguf",
        ));
        // Wan on the image surface teaches the videos route.
        let ImagesGate::Reject(msg) = images_gate(Some(&wan), Surface::ImgGen) else {
            panic!("video family on the image surface must be rejected");
        };
        assert!(msg.contains("/v1/videos/generations"), "{msg}");
        // Wan serves the video surface.
        assert!(matches!(
            images_gate(Some(&wan), Surface::VidGen),
            ImagesGate::Serve
        ));
        // An image family on the video surface is refused with the
        // image route named.
        let qwen = row(Some("vae.safetensors"), None);
        let ImagesGate::Reject(msg) = images_gate(Some(&qwen), Surface::VidGen) else {
            panic!("image family on the video surface must be rejected");
        };
        assert!(msg.contains("image diffusion family"), "{msg}");
        assert!(msg.contains("/v1/videos/generations"), "{msg}");
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__canonicalize_video_frames__synonyms_map_to_native() {
        let mut v = serde_json::json!({"model": "wan", "frames": 33, "fps": 16});
        canonicalize_video_frames(&mut v).unwrap();
        assert_eq!(v["video_frames"], serde_json::json!(33));
        assert!(v.get("frames").is_none());

        let mut v = serde_json::json!({"num_frames": 16});
        canonicalize_video_frames(&mut v).unwrap();
        assert_eq!(v["video_frames"], serde_json::json!(16));
        assert!(v.get("num_frames").is_none());

        // Agreeing duplicates collapse; disagreeing ones never get here.
        let mut v = serde_json::json!({"frames": 8, "num_frames": 8});
        canonicalize_video_frames(&mut v).unwrap();
        assert_eq!(v["video_frames"], serde_json::json!(8));
        assert!(v.get("frames").is_none() && v.get("num_frames").is_none());

        // The REPL's /duration knob: seconds x fps -> video_frames. The
        // child's default fps is 16 when the request carries none.
        let mut v = serde_json::json!({"duration": 2});
        canonicalize_video_frames(&mut v).unwrap();
        assert_eq!(v["video_frames"], serde_json::json!(32));
        assert!(v.get("duration").is_none());

        let mut v = serde_json::json!({"duration": 2, "fps": 8});
        canonicalize_video_frames(&mut v).unwrap();
        assert_eq!(v["video_frames"], serde_json::json!(16));
        assert_eq!(
            v["fps"],
            serde_json::json!(8),
            "fps is a real child field — it rides"
        );

        // Fractional seconds round; sub-frame durations clamp to one.
        let mut v = serde_json::json!({"duration": 0.5});
        canonicalize_video_frames(&mut v).unwrap();
        assert_eq!(v["video_frames"], serde_json::json!(8));
        let mut v = serde_json::json!({"duration": 0.01});
        canonicalize_video_frames(&mut v).unwrap();
        assert_eq!(v["video_frames"], serde_json::json!(1));

        // A duration that agrees with an explicit spec collapses.
        let mut v = serde_json::json!({"frames": 32, "duration": 2});
        canonicalize_video_frames(&mut v).unwrap();
        assert_eq!(v["video_frames"], serde_json::json!(32));
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__canonicalize_video_frames__native_verbatim_and_absent_ok() {
        let mut v = serde_json::json!({"video_frames": 33});
        canonicalize_video_frames(&mut v).unwrap();
        assert_eq!(v["video_frames"], serde_json::json!(33));

        let mut v = serde_json::json!({"model": "wan", "prompt": "p"});
        canonicalize_video_frames(&mut v).unwrap();
        assert!(v.get("video_frames").is_none());

        let mut v = serde_json::json!(["not", "an", "object"]);
        canonicalize_video_frames(&mut v).unwrap();
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__canonicalize_video_frames__conflict_and_type_errors_teach() {
        let mut v = serde_json::json!({"frames": 33, "video_frames": 16});
        let err = canonicalize_video_frames(&mut v).unwrap_err();
        assert!(
            err.contains("frames=33") && err.contains("video_frames=16"),
            "{err}"
        );

        let mut v = serde_json::json!({"frames": "33"});
        assert!(canonicalize_video_frames(&mut v)
            .unwrap_err()
            .contains("positive integer"));

        let mut v = serde_json::json!({"frames": 0});
        assert!(canonicalize_video_frames(&mut v)
            .unwrap_err()
            .contains("positive integer"));

        // duration joins the conflict vocabulary; fps itself stays.
        let mut v = serde_json::json!({"frames": 33, "duration": 1});
        let err = canonicalize_video_frames(&mut v).unwrap_err();
        assert!(
            err.contains("frames=33") && err.contains("duration=16"),
            "{err}"
        );

        let mut v = serde_json::json!({"duration": -1});
        assert!(canonicalize_video_frames(&mut v)
            .unwrap_err()
            .contains("positive number of seconds"));
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__align_wan_frames__temporal_grid() {
        // 4k+1 grid, aligned DOWN (live: 4 renders 1; a 16-frame
        // duration render delivered 13).
        for (asked, aligned) in [
            (1, 1),
            (2, 1),
            (4, 1),
            (5, 5),
            (13, 13),
            (16, 13),
            (33, 33),
            (34, 33),
            (960, 957),
        ] {
            assert_eq!(align_wan_frames(asked), aligned, "asked={asked}");
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__estimate_video_scratch__matches_measured_envelope() {
        // Warm 320x320 points (5/13/33 frames) all measured ≈ 3017 MiB
        // scratch; the gate's estimate must sit above the pass floor and
        // below the 5.3 GiB that was free when they passed.
        let warm_320 = estimate_video_scratch(33, 320, 320, false);
        assert_eq!(warm_320.aligned_frames, 33);
        assert!(
            (3_900..=4_200).contains(&warm_320.estimate_mib),
            "got {}",
            warm_320.estimate_mib
        );

        // Warm 512x512: 5-frame pass measured 3527 MiB; 13f@512 is an
        // upstream no-results bug at the same memory, not VRAM — the
        // estimate must stay under a quiet box's ~5.3 GiB so the gate
        // does not steal blame from the child's own failure.
        let warm_512 = estimate_video_scratch(13, 512, 512, false);
        assert!(
            (4_400..=4_900).contains(&warm_512.estimate_mib),
            "got {}",
            warm_512.estimate_mib
        );

        // The original hard-OOM repro lane: 16 frames at 512 against
        // ~3.8 GiB free (a foreign tenant holding 4.1 of 8 GiB). Aligned
        // down to 13 the estimate must still exceed that budget.
        let sixteen = estimate_video_scratch(16, 512, 512, false);
        assert_eq!(sixteen.aligned_frames, 13);
        assert!(sixteen.estimate_mib > 3_800, "got {}", sixteen.estimate_mib);

        // Beyond the 33-frame calibration horizon the floor scales: a
        // 60-second /duration default-fps request (961 aligned frames)
        // must price out of any 8 GiB card, cold or warm.
        let long = estimate_video_scratch(960, 320, 320, false); // 957 aligned after floor
        assert!(long.estimate_mib > 100_000, "got {}", long.estimate_mib);

        // A cold spawn charges the staged weights on top — exactly the
        // deterministic weight size, unscaled by the scratch safety factor.
        let cold = estimate_video_scratch(5, 320, 320, true);
        let warm = estimate_video_scratch(5, 320, 320, false);
        let delta = cold.estimate_mib - warm.estimate_mib;
        assert!((2_700..=2_900).contains(&delta), "cold delta {delta}");
        // And the live regression that motivated the split: 5f@320 cold
        // must fit a quiet 8 GiB card's 95% budget (~7.4 GiB).
        assert!(cold.estimate_mib <= 7_400, "got {}", cold.estimate_mib);
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__gate_size__parses_and_defaults_conservatively() {
        assert_eq!(
            gate_size(&serde_json::json!({"size": "320x320"})),
            (320, 320)
        );
        assert_eq!(
            gate_size(&serde_json::json!({"size": "640x480"})),
            (640, 480)
        );
        // Absent / malformed / zero dims fall back to the child's own
        // default — the bigger, conservative side.
        assert_eq!(gate_size(&serde_json::json!({})), (512, 512));
        assert_eq!(gate_size(&serde_json::json!({"size": "big"})), (512, 512));
        assert_eq!(gate_size(&serde_json::json!({"size": "0x0"})), (512, 512));
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__gate_frames__absent_priced_at_calibration_horizon() {
        // The child's default for an unspecified vid_gen is not
        // contracted (CLI flag says 1, live bare renders delivered
        // multi-frame) — a safety gate must not undersell what it
        // cannot know. Explicit counts price exactly what was asked.
        assert_eq!(gate_frames(&serde_json::json!({})), 33);
        assert_eq!(
            gate_frames(&serde_json::json!({"video_frames": 1})),
            1,
            "explicit 1 is the user's call, priced as asked"
        );
        assert_eq!(gate_frames(&serde_json::json!({"video_frames": 81})), 81);
        assert_eq!(
            gate_frames(&serde_json::json!({"video_frames": "33"})),
            33,
            "string numerals are not u64 — priced at the horizon"
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__translate_to_native__video_keeps_n_verbatim() {
        let req = serde_json::json!({"model": "wan", "prompt": "p", "n": 2, "steps": 4});
        let native = translate_to_native(&req, true);
        assert_eq!(native["n"], serde_json::json!(2));
        assert!(native.get("batch_count").is_none());
        assert_eq!(
            native["sample_params"]["sample_steps"],
            serde_json::json!(4)
        );
        // Image dialect still batches.
        let img = translate_to_native(&req, false);
        assert_eq!(img["batch_count"], serde_json::json!(2));
        assert!(img.get("n").is_none());
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__images_gate__diffusion_row_serves_generations() {
        assert!(matches!(
            images_gate(Some(&row(Some("vae.safetensors"), None)), Surface::ImgGen),
            ImagesGate::Serve
        ));
    }

    #[test]
    fn unit__images_gate__edits_without_vision_encoder_teaches_repull() {
        let gate = images_gate(Some(&row(Some("vae.safetensors"), None)), Surface::ImgEdits);
        let ImagesGate::Reject(msg) = gate else {
            panic!("edits without the vision encoder must be rejected");
        };
        assert!(msg.contains("vision-encoder"), "{msg}");
        assert!(msg.contains("blazar pull"), "{msg}");
    }

    #[test]
    fn unit__images_gate__visionless_family_edits_refuse_without_repull_loop() {
        // FLUX has no vision encoder to re-pull: the teaching must say
        // edits are unsupported, not send the user in a re-pull circle.
        let gate = images_gate(Some(&flux_row()), Surface::ImgEdits);
        let ImagesGate::Reject(msg) = gate else {
            panic!("edits on a vision-less family must be rejected");
        };
        assert!(msg.contains("without a vision encoder"), "{msg}");
        assert!(
            !msg.contains("blazar pull"),
            "re-pull teaching on a complete set loops forever: {msg}"
        );
        // Generations still serve for the same row.
        assert!(matches!(
            images_gate(Some(&flux_row()), Surface::ImgGen),
            ImagesGate::Serve
        ));
    }

    #[test]
    fn unit__images_gate__edits_with_vision_encoder_serves() {
        assert!(matches!(
            images_gate(
                Some(&row(Some("vae.safetensors"), Some("mmproj.gguf"))),
                Surface::ImgEdits
            ),
            ImagesGate::Serve
        ));
    }

    #[test]
    fn unit__images_gate__unknown_row_defers_to_ensure() {
        // Unknown model is not the gate's call — ensure owns the 404.
        assert!(matches!(
            images_gate(None, Surface::ImgEdits),
            ImagesGate::Serve
        ));
    }

    #[test]
    fn unit__is_plain_generation_request__verified_set_rides_compat() {
        let plain = serde_json::json!({
            "model": "m", "prompt": "p", "size": "512x512", "steps": 20, "n": 1
        });
        assert!(is_plain_generation_request(&plain));
        // Control keys never force the native dialect alone.
        let with_async = serde_json::json!({ "model": "m", "prompt": "p", "async": true });
        assert!(is_plain_generation_request(&with_async));
        for rich in [
            serde_json::json!({ "model": "m", "prompt": "p", "negative_prompt": "fog" }),
            serde_json::json!({ "model": "m", "prompt": "p", "cache": {"mode": "spectrum"} }),
            serde_json::json!({ "model": "m", "prompt": "p", "lora": [{"path": "a.safetensors"}] }),
        ] {
            assert!(
                !is_plain_generation_request(&rich),
                "rich request must ride the native dialect: {rich}"
            );
        }
    }

    #[test]
    fn unit__translate_to_native__shallow_renames_and_verbatim_subtrees() {
        let guidance = serde_json::json!({
            "txt_cfg": 4.0,
            "slg": {"layers": [7, 8], "start": 0.01, "end": 0.2, "scale": 0.2}
        });
        let cache = serde_json::json!({"mode": "spectrum", "option": 0});
        let req = serde_json::json!({
            "model": "qwen-image-2.1",
            "user": "u",
            "async": true,
            "prompt": "a cat",
            "n": 2,
            "size": "512x768",
            "steps": 30,
            "seed": 7,
            "guidance": guidance.clone(),
            "cache": cache.clone(),
        });
        let native = translate_to_native(&req, false);
        assert_eq!(native["batch_count"], serde_json::json!(2));
        assert_eq!(native["width"], serde_json::json!(512));
        assert_eq!(native["height"], serde_json::json!(768));
        assert_eq!(
            native["sample_params"]["sample_steps"],
            serde_json::json!(30)
        );
        assert_eq!(native["prompt"], serde_json::json!("a cat"));
        assert_eq!(native["seed"], serde_json::json!(7));
        // Subtrees ride verbatim — no re-typing, upstream-additive safe.
        assert_eq!(native["guidance"], guidance);
        assert_eq!(native["cache"], cache);
        // Control keys and renamed keys never leak under old names.
        for gone in ["model", "user", "async", "n", "size", "steps"] {
            assert!(native.get(gone).is_none(), "{gone} leaked: {native}");
        }
    }

    #[test]
    fn unit__translate_to_native__existing_sample_params_merges_steps() {
        let req = serde_json::json!({
            "model": "m", "prompt": "p",
            "steps": 8,
            "sample_params": {"sample_method": "euler"}
        });
        let native = translate_to_native(&req, false);
        assert_eq!(
            native["sample_params"]["sample_method"],
            serde_json::json!("euler")
        );
        assert_eq!(
            native["sample_params"]["sample_steps"],
            serde_json::json!(8)
        );
    }

    #[test]
    fn unit__translate_to_native__unparseable_size_passes_through() {
        let req = serde_json::json!({ "model": "m", "prompt": "p", "size": "big" });
        let native = translate_to_native(&req, false);
        assert_eq!(native["size"], serde_json::json!("big"));
        assert!(native.get("width").is_none());
    }

    #[test]
    fn unit__delivery_mode__stream_wins_and_plain_stays_compat() {
        let plain = serde_json::json!({"model": "m", "prompt": "p", "size": "512x512"});
        assert_eq!(delivery_mode(&plain), DeliveryMode::SyncPlain);
        assert_eq!(
            delivery_mode(&serde_json::json!({"model": "m", "prompt": "p", "async": true})),
            DeliveryMode::AsyncNative
        );
        // Rich without async keeps sync semantics via the native path.
        assert_eq!(
            delivery_mode(
                &serde_json::json!({"model": "m", "prompt": "p", "negative_prompt": "x"})
            ),
            DeliveryMode::SyncNative
        );
        // Both set → stream wins (pinned revision).
        assert_eq!(
            delivery_mode(&serde_json::json!({
                "model": "m", "prompt": "p", "async": true, "stream": true
            })),
            DeliveryMode::StreamNative
        );
        assert_eq!(
            delivery_mode(&serde_json::json!({"model": "m", "prompt": "p", "stream": true})),
            DeliveryMode::StreamNative
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__native_job_to_openai__flat_video_payload_maps_to_data() {
        // vid_gen answers with a FLAT result (b64_json + fps + frame_count +
        // mime_type at the root, no images[]); the video bytes must land in
        // data[0] with their metadata, never silently dropped.
        let job = serde_json::json!({
            "created": 1_690_000_000i64,
            "status": "completed",
            "result": {
                "b64_json": "R2tY", // "GkX" EBML magic, base64
                "fps": 16,
                "frame_count": 5,
                "mime_type": "video/webm",
                "output_format": "webm"
            }
        });
        let out = native_job_to_openai(&job);
        let data = out.get("data").and_then(|d| d.as_array()).unwrap();
        assert_eq!(
            data.len(),
            1,
            "flat video payload must map to one data entry: {out}"
        );
        assert_eq!(data[0]["b64_json"], "R2tY");
        assert_eq!(data[0]["fps"], 16);
        assert_eq!(data[0]["frame_count"], 5);
        assert_eq!(data[0]["mime_type"], "video/webm");
        assert_eq!(out["output_format"], "webm");

        // img_gen keeps its nested images[] shape untouched.
        let img = serde_json::json!({
            "created": 1_690_000_000i64,
            "result": {"images": [{"b64_json": "AAAA", "index": 0}], "output_format": "png"}
        });
        let out = native_job_to_openai(&img);
        let data = out.get("data").and_then(|d| d.as_array()).unwrap();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0]["b64_json"], "AAAA");
        assert!(
            data[0].get("fps").is_none(),
            "image entries carry no video metadata"
        );

        // A failed/empty job never invents data.
        let failed =
            serde_json::json!({"created": 1i64, "status": "failed", "error": {"message": "x"}});
        let out = native_job_to_openai(&failed);
        assert!(out["data"].as_array().unwrap().is_empty());
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__native_job_to_openai__maps_result_shape() {
        let job = serde_json::json!({
            "id": "job_1", "status": "completed", "created": 123,
            "result": {
                "images": [{"b64_json": "AAA", "index": 0}],
                "output_format": "png"
            }
        });
        let out = native_job_to_openai(&job);
        assert_eq!(out["created"], serde_json::json!(123));
        assert_eq!(out["output_format"], serde_json::json!("png"));
        assert_eq!(out["data"][0]["b64_json"], serde_json::json!("AAA"));
        // Missing result → empty data array, never a null.
        let failed = serde_json::json!({"status": "failed"});
        assert!(native_job_to_openai(&failed)["data"]
            .as_array()
            .is_some_and(std::vec::Vec::is_empty));
    }
}
