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
//! `{model, prompt, size, steps, n, output_format, output_compression}`
//! set) forward byte-identical to the child's compat route. Anything
//! richer — cache modes, `LoRA` weights, sampling subtrees, guidance —
//! rides the child's native async job dialect (`/sdcpp/v1/img_gen` +
//! `/sdcpp/v1/jobs/{id}`) through a shallow translator, because the
//! compat route accepts unknown fields without honoring them (probe
//! 2026-09-22: `steps` changes timing, `negative_prompt` does not
//! surface). `OpenAI`'s `quality` and `style` knobs are consumed here,
//! resolved against the child's per-model `capabilities` defaults so a
//! tier is a budget RELATIVE to the loaded model, never an absolute
//! step count. Engine-unsupported `OpenAI` fields (`response_format:
//! "url"`, `moderation`, `partial_images`, `background`,
//! `input_fidelity`) answer teaching 400s instead of being swallowed.
//! `"async":
//! true` returns the job handle immediately; `"stream": true` relays
//! progress as `SSE`; both set → stream wins. Job poll/cancel routes
//! resolve the LIVE child only — a job dies with its child, and these
//! routes must never boot a new one.

use axum::body::{Body, Bytes};
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, header};
use axum::response::{IntoResponse, Response};
use std::sync::Arc;
use std::time::{Duration, Instant};
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

/// Keys the compat route is verified to honor (api.md: prompt, n,
/// size, `output_format`, `output_compression`). Everything else in a
/// request means the native dialect, where the field is documented —
/// and actually applied — rather than silently swallowed.
const PLAIN_GENERATION_KEYS: [&str; 10] = [
    "model",
    "prompt",
    "size",
    "steps",
    "n",
    "user",
    "async",
    "stream",
    "output_format",
    "output_compression",
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

/// Refusal when a request pulls the overcommit lever while the operator
/// has centrally disabled it. Names the env var so the operator reading
/// a client's error can find their own switch.
const VRAM_OVERCOMMIT_DISABLED_TEACH: &str = "operator disabled the per-request VRAM overcommit \
     lever (BLAZAR_VRAM_OVERCOMMIT=0) — remaining levers: fewer frames (video_frames/duration), \
     smaller size, or freeing GPU memory held by other processes";

/// Operator kill-switch for the per-request VRAM overcommit lever. The
/// lever stays enabled by default (the scratch refusal teaches it);
/// `BLAZAR_VRAM_OVERCOMMIT=0` makes the gateway reject requests that use
/// it, so safety posture can be enforced centrally instead of per client.
static VRAM_OVERCOMMIT_ALLOWED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    vram_overcommit_enabled(std::env::var("BLAZAR_VRAM_OVERCOMMIT").ok().as_deref())
});

/// Pure env-value parse so the policy is unit-testable without touching
/// the process environment. Anything but an explicit off-value leaves
/// the lever on — an unset or garbage value must not silently flip
/// safety posture.
fn vram_overcommit_enabled(raw: Option<&str>) -> bool {
    let off = ["0", "false", "off", "no"];
    !matches!(
        raw,
        Some(v) if off.contains(&v.trim().to_ascii_lowercase().as_str())
    )
}

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

/// Parse the frame geometry the gate prices. Two dialects name it:
/// `size` (``WxH`` string, translated onto width/height for the child in
/// `translate_to_native`) and the documented top-level `width`/`height`
/// pair, which rides to the child verbatim — the gate must price BOTH,
/// or a 320x320 request that fits gets billed the 512x512 default and
/// over-rejected. Absent or unparseable geometry still takes the
/// child's 512x512 default — the conservative side of the estimate.
fn gate_size(v: &serde_json::Value) -> (u64, u64) {
    if let Some((w, h)) = v.get("size").and_then(|s| s.as_str()).and_then(|s| {
        s.split_once('x').and_then(|(w, h)| {
            let w = w.trim().parse::<u64>().ok()?;
            let h = h.trim().parse::<u64>().ok()?;
            Some((w, h))
        })
    }) && w > 0
        && h > 0
    {
        return (w, h);
    }
    if let (Some(w), Some(h)) = (
        v.get("width").and_then(serde_json::Value::as_u64),
        v.get("height").and_then(serde_json::Value::as_u64),
    ) && w > 0
        && h > 0
    {
        return (w, h);
    }
    VIDEO_DEFAULT_SIZE
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
        // carries the request that used the lever. A centrally disabled
        // lever (BLAZAR_VRAM_OVERCOMMIT=0) turns the same request away
        // instead of honoring it.
        if !*VRAM_OVERCOMMIT_ALLOWED {
            return Err(VRAM_OVERCOMMIT_DISABLED_TEACH.to_string());
        }
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
        // The overcommit lever is only taught when it can actually be
        // pulled — pointing a client at a disabled switch would be a lie.
        let overcommit_hint = if *VRAM_OVERCOMMIT_ALLOWED {
            ", or \\\"vram_overcommit\\\": true to force the attempt"
        } else {
            " (the vram_overcommit lever is operator-disabled)"
        };
        return Err(format!(
            "video request would exceed free VRAM: scratch ≈ {} MiB ({} aligned frames at \
             {w}x{h}{spawn_note}, safety ×{VIDEO_SCRATCH_SAFETY}) vs {budget} MiB free of \
             {free} (95% headroom rule). Levers: fewer frames (video_frames/duration), \
             smaller size, free GPU memory held by other processes{overcommit_hint}",
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

    // `seconds` (the Sora dialect): the same dial as `duration`, except
    // the Sora SDK ships it as a STRING ("4" | "8" | "12") — number or
    // clean numeric string both parse; anything else answers by name.
    // Agreeing duration+seconds collapse; disagreeing ones hit the
    // conflict check below like any other duplicate spelling.
    if let Some(val) = obj.get("seconds").cloned() {
        let fps = obj
            .get("fps")
            .and_then(|f| f.as_u64().filter(|f| *f >= 1))
            .unwrap_or(DEFAULT_FPS);
        let seconds_value = val
            .as_f64()
            .or_else(|| val.as_str().and_then(|s| s.trim().parse::<f64>().ok()))
            .filter(|s| s.is_finite() && *s > 0.0)
            .ok_or_else(|| {
                format!(
                    "seconds must be a positive number of seconds, number or numeric \
                     string (got {val})"
                )
            })?;
        obj.remove("seconds");
        let secs_frames = ((seconds_value * fps as f64).round() as u64).max(1);
        specs.push(("seconds", secs_frames));
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

// ---- OpenAI surface knobs: quality / style / response_format -------
//
// OpenAI's generation-quality vocabulary is RELATIVE ("high" spends
// more compute than "low"). The engine's step budget is per-model, so
// a tier resolves against the child's own `capabilities` defaults —
// the multiplier is the curated fact, the base never is.

/// Output containers the child validates per mode (api.md
/// `output_formats_by_mode`). Gateway-side first so a typo answers a
/// teaching instead of a child 400 that names neither the set nor lane.
const OUTPUT_FORMATS_IMG: [&str; 3] = ["png", "jpeg", "webp"];
const OUTPUT_FORMATS_VID: [&str; 3] = ["webm", "webp", "avi"];

/// Families with a curated `style` preset. `txt_cfg` shifts are only
/// honest where the family's guidance behavior is known; elsewhere the
/// request teaches to set `sample_params.guidance.txt_cfg` directly.
const STYLE_FAMILY_TOKENS: [&str; 1] = ["qwen-image"];

/// Step-budget multipliers for `quality`. `Ok(None)` = engine default
/// (`OpenAI`'s `auto`, or the knob absent). `standard`/`hd` are the
/// dall-e-2 spellings of medium/high.
fn quality_step_multiplier(quality: &str) -> Result<Option<f64>, String> {
    let mult = match quality {
        "auto" => return Ok(None),
        "low" | "standard" => 0.5,
        "medium" => 0.75,
        "high" | "hd" => 1.0,
        "xhigh" => 1.25,
        "max" => 1.5,
        other => {
            return Err(format!(
                "unknown quality \"{other}\" — supported: auto, low, medium, high, \
                 xhigh, max (dall-e-2 synonyms: standard, hd)"
            ));
        }
    };
    Ok(Some(mult))
}

/// `txt_cfg` preset for `style`, relative to the model default so the
/// family's own guidance scale stays the anchor.
fn style_preset(style: &str) -> Option<f64> {
    match style {
        "vivid" => Some(1.25),
        "natural" => Some(0.75),
        _ => None,
    }
}

/// Teaching gate for `OpenAI` fields this gateway does not serve.
/// Returns the first teaching message, `None` when the surface is
/// clean. `repo: None` (unknown model) counts as an uncurated family:
/// `style` cannot be verified against a family it does not know.
fn openai_surface_teaching(
    v: &serde_json::Value,
    video: bool,
    repo: Option<&str>,
) -> Option<String> {
    let get = |k: &str| v.get(k).filter(|val| !val.is_null());
    if get("response_format").and_then(serde_json::Value::as_str) == Some("url") {
        return Some(
            "response_format \"url\" is not served — blazar returns base64 directly \
             (data[].b64_json); omit response_format or send \"b64_json\""
                .into(),
        );
    }
    if let Some(val) = get("moderation") {
        return Some(format!(
            "moderation ({val}) is a hosted-OpenAI filter — local generation has no \
             moderation service to configure; drop the field"
        ));
    }
    if let Some(val) = get("partial_images") {
        return Some(format!(
            "partial_images ({val}) is not implemented — \"stream\": true relays job \
             progress as SSE events instead"
        ));
    }
    if let Some(val) = get("background") {
        return Some(format!(
            "background \"{val}\" is not served — local diffusion has no \
             alpha-controlled generation; PNG alpha only rides init/edit images"
        ));
    }
    if let Some(val) = get("input_fidelity") {
        return Some(format!(
            "input_fidelity ({val}) is not served — edits approximate fidelity through \
             the native \"strength\" knob (0..1 denoise)"
        ));
    }
    if let Some(style) = get("style").and_then(serde_json::Value::as_str) {
        if style_preset(style).is_none() {
            return Some(format!(
                "unknown style \"{style}\" — supported: vivid, natural"
            ));
        }
        let curated = repo.is_some_and(|r| {
            let lower = r.to_lowercase();
            STYLE_FAMILY_TOKENS.iter().any(|t| lower.contains(t))
        });
        if !curated {
            return Some(format!(
                "style presets are curated per family — this model{} has none; set \
                 sample_params.guidance.txt_cfg directly (vivid ≈ higher, natural ≈ lower)",
                repo.map_or_else(String::new, |r| format!(" ({r})"))
            ));
        }
    }
    if let Some(quality) = get("quality").and_then(serde_json::Value::as_str)
        && let Err(msg) = quality_step_multiplier(quality)
    {
        return Some(msg);
    }
    if let Some(fmt) = get("output_format").and_then(serde_json::Value::as_str) {
        let set: &[&str] = if video {
            &OUTPUT_FORMATS_VID
        } else {
            &OUTPUT_FORMATS_IMG
        };
        if !set.contains(&fmt) {
            return Some(format!(
                "output_format \"{fmt}\" is not one of {} on the {} lane",
                set.join("|"),
                if video { "video" } else { "image" }
            ));
        }
    }
    None
}

/// Per-model defaults from the child's capabilities, memoized per
/// engine for [`CHILD_DEFAULTS_TTL`]. One model rides one child, so the
/// key is the engine name; the TTL covers a re-pull swapping quants
/// under the same name. A failed fetch is NOT cached — the next request
/// retries.
const CHILD_DEFAULTS_TTL: Duration = Duration::from_secs(600);
type ChildDefaultsMap = std::collections::HashMap<String, (Instant, Arc<serde_json::Value>)>;
static CHILD_DEFAULTS: std::sync::LazyLock<tokio::sync::RwLock<ChildDefaultsMap>> =
    std::sync::LazyLock::new(|| tokio::sync::RwLock::new(ChildDefaultsMap::new()));

async fn child_mode_defaults(
    state: &AppState,
    engine: &blazar_runtime::EngineRef,
    mode: &str,
) -> Result<Arc<serde_json::Value>, String> {
    if let Some(hit) = CHILD_DEFAULTS
        .read()
        .await
        .get(&engine.name)
        .filter(|(at, _)| at.elapsed() < CHILD_DEFAULTS_TTL)
    {
        return Ok(Arc::clone(&hit.1));
    }
    let resp = crate::proxy::send_with_child_retry(state, engine, |eng| {
        let mut rb = crate::state::media_child_client(state, &eng.endpoint).get(format!(
            "{}/sdcpp/v1/capabilities",
            child_base(&eng.endpoint)
        ));
        rb = child_auth(rb, eng);
        rb
    })
    .await
    .map_err(|msg| format!("engine capabilities unavailable: {msg}"))?;
    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("engine capabilities answered {status}: {text}"));
    }
    let caps: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("engine capabilities body unreadable: {e}"))?;
    let defaults = caps
        .pointer(&format!("/defaults_by_mode/{mode}"))
        .cloned()
        .ok_or_else(|| format!("engine capabilities lack defaults_by_mode.{mode}"))?;
    let wrapped = Arc::new(defaults);
    CHILD_DEFAULTS
        .write()
        .await
        .insert(engine.name.clone(), (Instant::now(), Arc::clone(&wrapped)));
    Ok(wrapped)
}

/// Resolve the gateway-consumed `OpenAI` knobs (`quality`, `style`)
/// against the child's own defaults, rewriting `parsed` into plain
/// native vocabulary before translation. Explicit `steps` or
/// `sample_params.sample_steps` always wins over a quality tier.
// Casting policy: step/cfg math runs in f64 (multipliers are fractional)
// and lands back on u64 knobs the child validates — the trio below is
// deliberate, mirroring `canonicalize_video_frames`.
#[allow(
    clippy::result_large_err,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
async fn apply_generation_semantics(
    state: &AppState,
    engine: &blazar_runtime::EngineRef,
    parsed: &mut serde_json::Value,
    video: bool,
) -> Result<(), Response> {
    let mode = if video { "vid_gen" } else { "img_gen" };
    let wants_quality = parsed.get("quality").is_some();
    let wants_style = parsed.get("style").is_some();
    if !wants_quality && !wants_style {
        return Ok(());
    }
    let has_explicit_steps = parsed.get("steps").is_some()
        || parsed
            .pointer("/sample_params/sample_steps")
            .is_some_and(|s| !s.is_null());
    // A fetch is only owed when a knob actually resolves to a value:
    // quality=auto needs no defaults, style always does.
    let needs_defaults = wants_style
        || parsed
            .get("quality")
            .and_then(serde_json::Value::as_str)
            .map(quality_step_multiplier)
            .is_some_and(|m| matches!(m, Ok(Some(_))));
    let defaults = if needs_defaults {
        Some(
            child_mode_defaults(state, engine, mode)
                .await
                .map_err(|msg| openai_error(502, &msg))?,
        )
    } else {
        None
    };
    if wants_quality {
        let quality = parsed
            .get("quality")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        // Auto (and a null) means engine default: nothing to resolve.
        if let Ok(Some(mult)) = quality_step_multiplier(quality) {
            if has_explicit_steps {
                return Err(openai_error(
                    400,
                    "\"steps\" and \"quality\" are two dials for the same budget — send one",
                ));
            }
            let base = defaults
                .as_deref()
                .and_then(|d| d.pointer("/sample_params/sample_steps"))
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| openai_error(502,
                    "engine capabilities carry no sample_steps default — set \"steps\" explicitly",
                ))?;
            let steps = ((base as f64) * mult).round().max(1.0) as u64;
            if let Some(obj) = parsed.as_object_mut() {
                obj.remove("quality");
                obj.insert("steps".into(), serde_json::json!(steps));
            }
        } else if let Some(obj) = parsed.as_object_mut() {
            obj.remove("quality");
        }
    }
    if wants_style {
        let style = parsed
            .get("style")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if let Some(preset) = style_preset(style) {
            let base = defaults
                .as_deref()
                .and_then(|d| d.pointer("/sample_params/guidance/txt_cfg"))
                .and_then(serde_json::Value::as_f64);
            let target = base.map(|b| (b * preset).max(1.0)).ok_or_else(|| {
                openai_error(
                    502,
                    "engine capabilities carry no txt_cfg default — set \
                     sample_params.guidance.txt_cfg explicitly",
                )
            })?;
            // Same merge path as "steps": plant the subtrees when absent
            // so the translator's verbatim arm carries the preset down.
            if let Some(sp) = parsed
                .as_object_mut()
                .map(|o| o.entry("sample_params".to_string()))
                .map(|e| e.or_insert_with(|| serde_json::Value::Object(serde_json::Map::new())))
                .and_then(|v| v.as_object_mut())
            {
                let guidance = sp
                    .entry("guidance".to_string())
                    .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
                if let Some(g) = guidance.as_object_mut() {
                    g.insert("txt_cfg".into(), serde_json::json!(target));
                }
            }
        }
        if let Some(obj) = parsed.as_object_mut() {
            obj.remove("style");
        }
    }
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
            // Consumed gateway-side (see `apply_generation_semantics`
            // and the surface teachings); dropped defensively so a
            // stale copy never rides to the child as junk. `seconds`
            // is the Sora ghost-echo the video route plants for the
            // ledger — never a child field.
            "model" | "user" | "async" | "stream" | "quality" | "style" | "response_format"
            | "seconds" => {}
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
    let row = state
        .with_store(|s| resolve_model(s, &model).ok())
        .flatten();
    // Federation fallback: a diffusion model no local row owns serves
    // from a live peer that lists it — byte-forward the generation
    // request, the peer's own gate owns capability checks. Local-only
    // teaching below stays untouched when no peer claims it.
    if row.is_none() {
        let mut fwd_headers = axum::http::HeaderMap::new();
        fwd_headers.insert(
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static("application/json"),
        );
        if let Some(resp) = crate::remotes::try_fallback_forward(
            &state,
            &model,
            crate::remotes::FallbackLane::OpenAi {
                method: &axum::http::Method::POST,
                path: "/v1/images/generations",
                headers: &fwd_headers,
                body: body.clone(),
            },
        )
        .await
        {
            return resp;
        }
    }
    match images_gate(row.as_ref(), Surface::ImgGen) {
        ImagesGate::Serve => {}
        ImagesGate::Reject(msg) => return openai_error(400, &msg),
    }
    if let Some(msg) =
        openai_surface_teaching(&parsed, false, row.as_ref().map(|r| r.repo.as_str()))
    {
        return openai_error(400, &msg);
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
    if let Err(resp) = apply_generation_semantics(&state, &engine, &mut parsed, false).await {
        return resp;
    }
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

/// Ledger echo of the client-facing request shape (`prompt`, `size`,
/// seconds, frame budget). The Sora video plane reads it back from the
/// durable row; the child never sees any of it.
fn request_echo(v: &serde_json::Value) -> serde_json::Value {
    let mut echo = serde_json::Map::new();
    if let Some(p) = v.get("prompt").and_then(serde_json::Value::as_str) {
        echo.insert("prompt".into(), serde_json::json!(p));
    }
    for key in ["size", "n", "fps", "video_frames", "seconds"] {
        if let Some(val) = v.get(key).filter(|x| !x.is_null()) {
            echo.insert(key.into(), val.clone());
        }
    }
    serde_json::Value::Object(echo)
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
    // Durable ledger: every native submission gets a row (async handles,
    // sync awaits and SSE relays alike) so `/v1/jobs` shows the full
    // render history and a completed result survives a restart. The
    // child accepted the job, so it lands straight in running.
    state.jobs.record_created(
        &state,
        job_id,
        if video { "video" } else { "image" },
        parsed.get("model").and_then(serde_json::Value::as_str),
        serde_json::json!({
            "engine": engine.name,
            "native_path": native_path,
            "load_ms": load_ms,
            "request": request_echo(parsed),
        }),
    );
    state.jobs.record_running(&state, job_id);
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

/// Video timing pass, after semantics: dynamic-fps correction plus the
/// ghost-seconds echo for the Sora plane's ledger `request` block. The
/// canonicalizer converted `seconds` with an assumed 16 fps; the loaded
/// model's own capabilities default says otherwise — recompute the frame
/// count and pin fps explicitly so the rendered duration matches the
/// seconds asked. Unreachable capabilities keep the 16-fps conversion
/// (today's behavior) — the request proceeds either way. The ghost
/// `seconds` is gateway vocabulary: the translator drops it, it never
/// rides to the child.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
async fn apply_video_timing(
    state: &AppState,
    engine: &blazar_runtime::EngineRef,
    parsed: &mut serde_json::Value,
    seconds_based: Option<f64>,
    had_explicit_fps: bool,
) {
    if let (Some(secs), false) = (seconds_based, had_explicit_fps)
        && let Ok(defaults) = child_mode_defaults(state, engine, "vid_gen").await
        && let Some(fps) = defaults
            .pointer("/fps")
            .and_then(serde_json::Value::as_f64)
            .filter(|f| *f >= 1.0)
        && (fps as u64) != 16
        && let Some(obj) = parsed.as_object_mut()
    {
        let frames = ((secs * fps).round() as u64).max(1);
        obj.insert("video_frames".into(), serde_json::json!(frames));
        obj.insert("fps".into(), serde_json::json!(fps as u64));
    }
    if let Some(secs) = seconds_based
        && let Some(obj) = parsed.as_object_mut()
    {
        obj.insert("seconds".into(), serde_json::json!(secs));
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
    if parsed.get("remixed_from_video_id").is_some() {
        return openai_error(
            400,
            "video remix (remixed_from_video_id) has no local engine primitive — sdcpp \
             vid_gen is text-to-video (plus first/last-frame init images on some families); \
             compose a continuation manually: an image edit, then a new video",
        );
    }
    // Capture the seconds dial BEFORE canonicalization consumes it: the
    // pure converter must assume the 16 fps default, but the honest
    // frame count derives from the model's own default — corrected once
    // the engine is known (below).
    let seconds_based = parsed
        .get("seconds")
        .or_else(|| parsed.get("duration"))
        .and_then(|v| {
            v.as_f64()
                .or_else(|| v.as_str().and_then(|s| s.trim().parse::<f64>().ok()))
                .filter(|s| s.is_finite() && *s > 0.0)
        });
    let had_explicit_fps = parsed
        .get("fps")
        .and_then(serde_json::Value::as_u64)
        .is_some();
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
    if let Some(msg) = openai_surface_teaching(&parsed, true, row.as_ref().map(|r| r.repo.as_str()))
    {
        return openai_error(400, &msg);
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
    if let Err(resp) = apply_generation_semantics(&state, &engine, &mut parsed, true).await {
        return resp;
    }
    apply_video_timing(
        &state,
        &engine,
        &mut parsed,
        seconds_based,
        had_explicit_fps,
    )
    .await;
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

/// Standard-alphabet base64 (RFC 4648, padded). The workspace ships no
/// base64 crate — every other lane hands the child raw bytes or
/// pre-encoded payloads — so the media lane owns this tiny codec for the
/// spots where the gateway itself must transcode (variations upload
/// encode here; video content decode in the Sora verbs).
const B64_ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn b64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let n = (u32::from(chunk[0]) << 16)
            | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
            | u32::from(*chunk.get(2).unwrap_or(&0));
        out.push(B64_ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(B64_ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            B64_ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            B64_ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// Scalar form fields that keep JSON number/bool types on the ride to
/// the child; everything else stays a string (a prompt of "2024" must
/// not silently become the number 2024).
const VARIATIONS_INT_FIELDS: [&str; 3] = ["n", "output_compression", "clip_skip"];
const VARIATIONS_SEED_FIELDS: [&str; 1] = ["seed"];
const VARIATIONS_FLOAT_FIELDS: [&str; 3] = ["strength", "control_strength", "ip_adapter_strength"];
const VARIATIONS_BOOL_FIELDS: [&str; 2] = ["async", "stream"];

/// Assemble the native variations payload from parsed multipart parts.
/// The uploaded image becomes `init_image` (base64 img2img); `OpenAI`'s
/// contract sends no prompt, so a neutral empty one rides by default and
/// the seed is random (-1) unless the client sets it.
fn variations_payload(parts: &[crate::whisper::Part]) -> Result<serde_json::Value, String> {
    let mut obj = serde_json::Map::new();
    let mut image: Option<&crate::whisper::Part> = None;
    for part in parts {
        if part.name == "image" {
            if image.is_some() {
                return Err("multiple \"image\" parts — send exactly one image".into());
            }
            image = Some(part);
        } else if part.filename.is_some() {
            return Err(format!(
                "unexpected file part \"{}\" — variations takes one \"image\" file; \
                 mask/instruction editing is /v1/images/edits",
                part.name
            ));
        } else if part.name == "model" || part.name == "user" {
            // model is consumed by admission; user never rides (same
            // strip the JSON generations lane performs).
        } else {
            let text = String::from_utf8_lossy(&part.data).trim().to_string();
            let value = if VARIATIONS_SEED_FIELDS.contains(&part.name.as_str()) {
                match text.parse::<i64>() {
                    Ok(n) => serde_json::Value::from(n),
                    Err(_) => return Err("seed must be an integer".into()),
                }
            } else if VARIATIONS_INT_FIELDS.contains(&part.name.as_str()) {
                match text.parse::<u64>() {
                    Ok(n) => serde_json::Value::from(n),
                    Err(_) => {
                        return Err(format!("\"{}\" must be a non-negative integer", part.name));
                    }
                }
            } else if VARIATIONS_FLOAT_FIELDS.contains(&part.name.as_str()) {
                match text.parse::<f64>() {
                    Ok(f) => serde_json::Value::from(f),
                    Err(_) => return Err(format!("\"{}\" must be a number", part.name)),
                }
            } else if VARIATIONS_BOOL_FIELDS.contains(&part.name.as_str()) {
                match text.to_ascii_lowercase().as_str() {
                    "true" => serde_json::Value::Bool(true),
                    "false" => serde_json::Value::Bool(false),
                    _ => return Err(format!("\"{}\" must be true or false", part.name)),
                }
            } else {
                serde_json::Value::from(text)
            };
            obj.insert(part.name.clone(), value);
        }
    }
    let image = image.ok_or("an \"image\" file part is required (png/jpeg/webp bytes)")?;
    obj.insert(
        "init_image".into(),
        serde_json::Value::from(b64_encode(&image.data)),
    );
    obj.entry("prompt")
        .or_insert_with(|| serde_json::Value::from(""));
    obj.entry("seed")
        .or_insert_with(|| serde_json::Value::from(-1_i64));
    Ok(serde_json::Value::Object(obj))
}

/// POST /v1/images/variations — the `OpenAI` variation verb, served
/// natively: the uploaded image rides as `init_image` (img2img with a
/// neutral empty prompt — the `OpenAI` contract sends none), random seed
/// unless set. Answers in the same `{created, data[].b64_json}` shape
/// generations uses; the native dialect (strength, `negative_prompt`,
/// `sample_params`, ...) rides as documented extensions, and the
/// async/stream job modes are the same lanes generations offers.
pub async fn variations(
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
    let Some(parts) = content_type
        .as_deref()
        .and_then(|ct| crate::whisper::parse_multipart(&body, ct))
    else {
        return openai_error(400, "multipart/form-data body with a boundary is required");
    };
    let mut payload = match variations_payload(&parts) {
        Ok(v) => v,
        Err(msg) => return openai_error(400, &msg),
    };
    let row = state
        .with_store(|s| resolve_model(s, &model).ok())
        .flatten();
    match images_gate(row.as_ref(), Surface::ImgGen) {
        ImagesGate::Serve => {}
        ImagesGate::Reject(msg) => return openai_error(400, &msg),
    }
    if let Some(msg) =
        openai_surface_teaching(&payload, false, row.as_ref().map(|r| r.repo.as_str()))
    {
        return openai_error(400, &msg);
    }
    let (engine, load_ms) = match ensure_with_admission(
        &state,
        &model,
        Priority::Normal,
        WorkClass::Interactive,
        None,
        false, // img2img needs no vision-encoder sidecar (that is edits cargo)
        true,  // images lane: component sets are its cargo
        false,
    )
    .await
    {
        Ok(ok) => ok,
        Err(resp) => return *resp,
    };
    if let Err(resp) = apply_generation_semantics(&state, &engine, &mut payload, false).await {
        return resp;
    }
    // `init_image` keeps this off the plain compat lane by construction —
    // every variation is a native submit with an OpenAI-shaped answer.
    deliver_native(
        state,
        engine,
        &payload,
        delivery_mode(&payload),
        "/sdcpp/v1/img_gen",
        false,
        load_ms,
    )
    .await
}

// ---- native job dialect bridge ------------------------------------

/// Query-dialect match for child narrowing: `?model=` accepts both the
/// canonical child name (engine registration, e.g.
/// `wan_2.1_comfyui_repackaged`) and any caller spelling the store
/// resolves onto that row (`wan_2.1`).
fn child_matches_model(query: &str, canonical: Option<&str>, child: &str) -> bool {
    child == query || canonical == Some(child)
}

/// Live `sd-server` children, optionally narrowed to one model. The
/// job routes resolve against this snapshot ONLY — they never spawn:
/// a job belongs to the child that accepted it and dies with it.
/// `model` is caller-spelled (body name or `?model=`); children
/// register under the canonical row name, so the store resolves the
/// spelling before narrowing — the same resolution the spawn path
/// (`ensure_with_admission` -> `lane = row.name`) performs.
fn live_sdcpp_children(state: &AppState, model: Option<&str>) -> Vec<blazar_runtime::EngineRef> {
    let canonical = model.and_then(|m| {
        state
            .with_store(|s| resolve_model(s, m).ok())
            .flatten()
            .map(|row| row.name)
    });
    state
        .sup
        .live_http_endpoints()
        .into_iter()
        .filter(|e| e.kind == blazar_core::engine_kind::EngineKind::SdCpp)
        .filter(|e| model.is_none_or(|m| child_matches_model(m, canonical.as_deref(), &e.name)))
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

/// Poll the live children for a job, first owner wins. Shared by the
/// lane's own `jobs_get` and the unified `/v1/jobs/{id}` read plane;
/// every poll keeps its MM5 activity bracket so an actively-polled job
/// holds its child warm exactly as before.
pub(crate) async fn poll_live_child(
    state: &AppState,
    model: Option<&str>,
    job_id: &str,
) -> Option<serde_json::Value> {
    for engine in live_sdcpp_children(state, model) {
        state.sup.begin_request(&engine.name);
        let polled = poll_child_job(state, &engine, job_id).await;
        state.sup.end_request(&engine.name);
        if let Ok(Some(job)) = polled {
            return Some(job);
        }
        // Not this child's job, or the child flapped mid-poll: either
        // way, try the siblings.
    }
    None
}

/// Locate the owning child and fire a cancel at it — the unified
/// `/v1/jobs/{id}/cancel` path, which has no `?model=` hint and relies
/// on the engine name recorded at submit time.
pub(crate) async fn cancel_child_best_effort(state: &AppState, engine: Option<&str>, job_id: &str) {
    for child in live_sdcpp_children(state, engine) {
        if poll_child_job(state, &child, job_id)
            .await
            .is_ok_and(|found| found.is_some())
        {
            cancel_child_job(state, &child, job_id).await;
            return;
        }
    }
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
            state.jobs.record_cancelled(state, job_id);
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
                    crate::jobs::mirror_child_terminal(state, job_id, &job);
                    return Ok(job);
                }
            }
            Ok(None) => {
                state.jobs.record_failed(
                    state,
                    job_id,
                    "job vanished from the engine child mid-generation",
                );
                return Err(Box::new(openai_error(
                    502,
                    "job vanished from the engine child mid-generation",
                )));
            }
            Err(()) => {
                transport_fails += 1;
                if transport_fails >= JOB_POLL_TRANSPORT_RETRIES {
                    state.sup.reap_dead_children().await;
                    state
                        .jobs
                        .record_failed(state, job_id, "engine child died mid-generation");
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
                state.jobs.record_cancelled(&state, &job_id);
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
                    state.jobs.record_failed(
                        &state,
                        &job_id,
                        "job vanished from the engine child mid-generation",
                    );
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
                        state.jobs.record_failed(
                            &state,
                            &job_id,
                            "engine child died mid-generation",
                        );
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
/// LIVE child first (never spawns); the durable ledger answers when the
/// child is gone, so completed results survive eviction and restarts.
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
    let children_empty = live_sdcpp_children(&state, params.model.as_deref()).is_empty();
    if children_empty {
        return openai_error(
            404,
            "no live diffusion child serves jobs — POST /v1/images/generations boots one",
        );
    }
    if let Some(job) = poll_live_child(&state, params.model.as_deref(), &job_id).await {
        crate::jobs::mirror_child_terminal(&state, &job_id, &job);
        return Response::builder()
            .status(200)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_vec(&job).unwrap_or_default()))
            .unwrap_or_else(|e| {
                openai_error(500, &format!("response build: {e}")).into_response()
            });
    }
    // No live child owns the job: the ledger is the afterlife. A
    // non-terminal row means the child died mid-render — close it
    // honestly instead of leaving a forever-running zombie row.
    let row = state
        .with_store(|s| s.get_job(&job_id).ok().flatten())
        .flatten();
    if let Some(row) = row {
        if matches!(row.state.as_str(), "queued" | "running") {
            state.jobs.record_failed(
                &state,
                &job_id,
                "job's engine child is gone (eviction, crash or restart) — \
                 resubmit the generation",
            );
        }
        let fresh = state
            .with_store(|s| s.get_job(&job_id).ok().flatten())
            .flatten();
        if let Some(fresh) = fresh {
            return axum::Json(crate::jobs::row_payload(&fresh)).into_response();
        }
    }
    openai_error(
        404,
        "job not found on any live diffusion child — jobs die with their engine child \
         (eviction, crash or restart); completed results live on at /v1/jobs/{id}",
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
        // No live child owns it: the durable ledger decides. A terminal
        // row answers as-is; an in-flight row belongs to a dead child,
        // so the cancel intent is satisfied by closing it.
        let row = state
            .with_store(|s| s.get_job(&job_id).ok().flatten())
            .flatten();
        if let Some(row) = row {
            if matches!(row.state.as_str(), "queued" | "running") {
                state.jobs.record_cancelled(&state, &job_id);
            }
            let fresh = state
                .with_store(|s| s.get_job(&job_id).ok().flatten())
                .flatten();
            if let Some(fresh) = fresh {
                return axum::Json(crate::jobs::row_payload(&fresh)).into_response();
            }
        }
        return openai_error(
            404,
            "job not found on any live diffusion child — jobs die with their engine child; \
             completed records live on at /v1/jobs/{id}",
        );
    };
    if job.get("completed").and_then(serde_json::Value::as_bool) == Some(true) {
        crate::jobs::mirror_child_terminal(&state, &job_id, &job);
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
    if let Ok(resp) = sent
        && resp.status().as_u16() == 409
    {
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
    match poll_child_job(&state, &engine, &job_id).await {
        Ok(Some(verdict)) => {
            crate::jobs::mirror_child_terminal(&state, &job_id, &verdict);
            Response::builder()
                .status(200)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&verdict).unwrap_or_default()))
                .unwrap_or_else(|e| {
                    openai_error(500, &format!("response build: {e}")).into_response()
                })
        }
        _ => openai_error(
            502,
            "cancel was sent but the engine child did not confirm job state afterwards",
        ),
    }
}

// ---- Sora video plane: /v1/videos -----------------------------------
//
// The Sora dialect reads the SAME durable job ledger the video lane
// writes (`kind: "video"` rows) — no parallel job subsystem. The
// differences are shape only: `object: "video"`, `in_progress`
// status, a `seconds` echo, content delivery as raw bytes
// (`GET /{id}/content`) and DELETE-as-cancel. `cancelled` stays a
// blazar-superset status (Sora's enum has no cancelled) — an honest
// superset beats a lying `failed`.

/// Strict RFC 4648 decoder — the inverse of [`b64_encode`], fed by the
/// child's own padded standard-base64 output. Table-built once; any
/// non-alphabet byte, misplaced `=` or ragged length is an error by
/// name rather than a silent partial decode.
// Table indices are 0..=63 by construction (alphabet length), so the
// narrowing casts below are total, not lossy.
#[allow(clippy::cast_possible_truncation)]
static B64_DECODE_TABLE: std::sync::LazyLock<[u8; 256]> = std::sync::LazyLock::new(|| {
    let mut table = [255u8; 256];
    for (i, &c) in B64_ALPHABET.iter().enumerate() {
        table[c as usize] = i as u8;
    }
    table
});

// Shifts below stay inside u32 then narrow to u8 — values are ≤ 0xFF by
// construction (three payload bytes per quad).
#[allow(clippy::cast_possible_truncation)]
fn b64_decode(s: &str) -> Result<Vec<u8>, String> {
    let bytes = s.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "base64 length {} is not a multiple of 4",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for (i, chunk) in bytes.chunks(4).enumerate() {
        let mut vals = [0u32; 4];
        let mut pad = 0usize;
        for (j, &b) in chunk.iter().enumerate() {
            if b == b'=' {
                // '=' only closes the tail: positions 2/3, never data
                // after a pad.
                if j < 2 || (j == 2 && chunk[3] != b'=') || pad > 0 && j == 3 && pad == 2 {
                    return Err(format!("misplaced '=' at byte {}", i * 4 + j));
                }
                pad += 1;
                continue;
            }
            if pad > 0 {
                return Err(format!("data after '=' at byte {}", i * 4 + j));
            }
            let v = B64_DECODE_TABLE[b as usize];
            if v == 255 {
                return Err(format!(
                    "invalid base64 byte {:?} at {}",
                    b as char,
                    i * 4 + j
                ));
            }
            vals[j] = u32::from(v);
        }
        let n = (vals[0] << 18) | (vals[1] << 12) | (vals[2] << 6) | vals[3];
        out.push((n >> 16) as u8);
        if pad < 2 {
            out.push((n >> 8) as u8);
        }
        if pad < 1 {
            out.push(n as u8);
        }
    }
    Ok(out)
}

#[derive(serde::Deserialize)]
pub struct VideoListQuery {
    pub limit: Option<u64>,
}

/// Map a ledger row (plus optional live child snapshot for fresh
/// progress) to the Sora video object. `seconds` prefers the completed
/// result's `frame_count`/`fps` — the engine normalizes the frame
/// count (4n+1), so the render's own numbers are the truth over the
/// request echo.
// Progress arrives as an f64 fraction from the child; the OpenAI shape
// wants an integer percentage — truncation toward 0 is the safe side.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn sora_video_object(
    row: &blazar_core::store::JobRow,
    fresh: Option<&serde_json::Value>,
) -> serde_json::Value {
    let status = match row.state.as_str() {
        "queued" => "queued",
        "running" => "in_progress",
        "completed" => "completed",
        "failed" => "failed",
        other => other, // "cancelled" — documented superset
    };
    let mut obj = serde_json::json!({
        "id": row.id,
        "object": "video",
        "status": status,
        "created_at": row.created_at,
    });
    if let Some(progress) = fresh
        .and_then(|j| j.get("progress"))
        .and_then(serde_json::Value::as_f64)
        .map(|p| p.clamp(0.0, 100.0) as u64)
    {
        obj["progress"] = serde_json::json!(progress);
    } else if row.state == "queued" {
        obj["progress"] = serde_json::json!(0);
    } else if row.state == "completed" {
        obj["progress"] = serde_json::json!(100);
    }
    let request = row
        .request_json
        .parse::<serde_json::Value>()
        .ok()
        .and_then(|r| r.get("request").cloned());
    let echo = |key: &str| request.as_ref().and_then(|r| r.get(key));
    if let Some(p) = echo("prompt") {
        obj["prompt"] = p.clone();
    }
    if let Some(s) = echo("size") {
        obj["size"] = s.clone();
    }
    if row.state == "completed" {
        obj["completed_at"] = serde_json::json!(row.updated_at);
        let result = row
            .result_json
            .as_deref()
            .and_then(|r| serde_json::from_str::<serde_json::Value>(r).ok());
        let frames = result
            .as_ref()
            .and_then(|r| r.pointer("/result/frame_count"))
            .and_then(serde_json::Value::as_f64);
        let fps = result
            .as_ref()
            .and_then(|r| r.pointer("/result/fps"))
            .and_then(serde_json::Value::as_f64);
        if let (Some(frames), Some(fps)) = (frames, fps)
            && fps > 0.0
        {
            obj["seconds"] = serde_json::json!((frames / fps * 1000.0).round() / 1000.0);
        } else if let Some(secs) = echo("seconds").and_then(serde_json::Value::as_f64) {
            obj["seconds"] = serde_json::json!(secs);
        }
    } else if let Some(secs) = echo("seconds").and_then(serde_json::Value::as_f64) {
        obj["seconds"] = serde_json::json!(secs);
    }
    if let Some(err) = &row.error {
        obj["error"] = serde_json::json!(err);
    }
    obj
}

/// Why `/content` cannot serve a completed row's bytes.
#[derive(Debug)]
enum VideoContentError {
    NotCompleted,
    Gone,
    Unreadable(String),
}

/// Decode a completed video row into raw bytes + the child-reported
/// media type. Inline results decode straight from `result_json`;
/// artifact-spilled ones re-read the spilled file (it holds the same
/// child job JSON); a dropped artifact answers `Gone`.
fn completed_video_bytes(
    row: &blazar_core::store::JobRow,
) -> Result<(Vec<u8>, String), VideoContentError> {
    if row.state != "completed" {
        return Err(VideoContentError::NotCompleted);
    }
    let inline = row
        .result_json
        .as_deref()
        .and_then(|r| serde_json::from_str::<serde_json::Value>(r).ok());
    let job = match inline {
        Some(v) if v.get("artifact") == Some(&serde_json::json!("dropped")) => {
            return Err(VideoContentError::Gone);
        }
        Some(v) if v.get("artifact") == Some(&serde_json::json!(true)) => {
            let Some(path) = row.artifact_path.as_deref() else {
                return Err(VideoContentError::Gone);
            };
            let spilled = std::fs::read(path)
                .map_err(|e| VideoContentError::Unreadable(format!("artifact read {path}: {e}")))?;
            serde_json::from_slice::<serde_json::Value>(&spilled)
                .map_err(|e| VideoContentError::Unreadable(format!("artifact JSON {path}: {e}")))?
        }
        Some(v) => v,
        None => {
            return Err(VideoContentError::Unreadable(
                "completed row carries no parseable result".into(),
            ));
        }
    };
    let Some(b64) = job
        .pointer("/result/b64_json")
        .and_then(serde_json::Value::as_str)
    else {
        return Err(VideoContentError::Unreadable(
            "engine result carries no b64_json".into(),
        ));
    };
    let bytes = b64_decode(b64).map_err(VideoContentError::Unreadable)?;
    let mime = job
        .pointer("/result/mime_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("application/octet-stream")
        .to_string();
    Ok((bytes, mime))
}

/// Poll the live child for an in-flight row and mirror any terminal
/// verdict into the ledger. Returns the fresh child snapshot (progress
/// carrier) or `None` when the row is terminal or its child is gone.
async fn freshen_video_row(
    state: &AppState,
    row: &blazar_core::store::JobRow,
) -> Option<serde_json::Value> {
    if !matches!(row.state.as_str(), "queued" | "running") {
        return None;
    }
    let fresh = poll_live_child(state, None, &row.id).await?;
    crate::jobs::mirror_child_terminal(state, &row.id, &fresh);
    Some(fresh)
}

fn video_row(state: &AppState, id: &str) -> Option<blazar_core::store::JobRow> {
    state
        .with_store(|s| s.get_job(id).ok().flatten())
        .flatten()
        .filter(|row| row.kind == "video")
}

/// GET /v1/videos — Sora's list verb over the durable video rows,
/// newest first (Sora default limit 10, max 100).
pub async fn videos_list(
    State(state): State<Arc<AppState>>,
    Query(q): Query<VideoListQuery>,
) -> Response {
    let limit = q.limit.unwrap_or(10).clamp(1, 100);
    let rows = state
        .with_store(|s| {
            s.list_jobs(None, Some("video"), limit + 1)
                .unwrap_or_default()
        })
        .unwrap_or_default();
    let has_more = rows.len() as u64 > limit;
    let mut data: Vec<serde_json::Value> = Vec::new();
    for row in rows.iter().take(limit as usize) {
        let fresh = freshen_video_row(&state, row).await;
        let refetched = state
            .with_store(|s| s.get_job(&row.id).ok().flatten())
            .unwrap_or_default();
        let current = refetched.as_ref().unwrap_or(row);
        data.push(sora_video_object(current, fresh.as_ref()));
    }
    let mut list = serde_json::json!({
        "object": "list",
        "data": data,
        "has_more": has_more,
    });
    if let Some(first) = list
        .pointer("/data/0/id")
        .and_then(serde_json::Value::as_str)
    {
        list["first_id"] = serde_json::json!(first);
    }
    if let Some(idx) = data.len().checked_sub(1)
        && let Some(last) = data[idx].get("id").and_then(serde_json::Value::as_str)
    {
        list["last_id"] = serde_json::json!(last);
    }
    axum::Json(list).into_response()
}

/// GET /v1/videos/{id} — the Sora video object, live-polled while
/// in flight.
pub async fn video_get(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return openai_error(400, "invalid video id");
    }
    let Some(row) = video_row(&state, &id) else {
        return openai_error(404, &format!("video {id} not found"));
    };
    let fresh = freshen_video_row(&state, &row).await;
    let refetched = state
        .with_store(|s| s.get_job(&id).ok().flatten())
        .unwrap_or_default();
    let current = refetched.as_ref().unwrap_or(&row);
    axum::Json(sora_video_object(current, fresh.as_ref())).into_response()
}

/// GET /v1/videos/{id}/content — the rendered bytes. Sora semantics:
/// 404 until the render is completed, then the container the engine
/// produced (webm today; webp/avi when asked via `output_format`).
pub async fn video_content(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return openai_error(400, "invalid video id");
    }
    let Some(row) = video_row(&state, &id) else {
        return openai_error(404, &format!("video {id} not found"));
    };
    match completed_video_bytes(&row) {
        Ok((bytes, mime)) => Response::builder()
            .status(200)
            .header(header::CONTENT_TYPE, mime)
            .body(Body::from(bytes))
            .unwrap_or_else(|e| openai_error(500, &format!("content build: {e}")).into_response()),
        Err(VideoContentError::NotCompleted) => {
            if row.state == "cancelled" {
                openai_error(
                    404,
                    &format!(
                        "video {id} was cancelled or deleted — its bytes are gone; \
                         submit a new generation"
                    ),
                )
            } else {
                openai_error(
                    404,
                    &format!(
                        "video {id} is not completed yet — poll GET /v1/videos/{id} until \
                         status is \"completed\""
                    ),
                )
            }
        }
        Err(VideoContentError::Gone) => openai_error(
            410,
            &format!(
                "video {id}'s rendered bytes outgrew the inline ledger, spilled to disk \
                 and the spill was lost — regenerate"
            ),
        ),
        Err(VideoContentError::Unreadable(msg)) => openai_error(500, &msg),
    }
}

/// DELETE /v1/videos/{id} — Sora's deprecated cancel verb. Open rows
/// (queued/running) get a best-effort child cancel (this engine build
/// only interrupts queued jobs) and close `cancelled`. Terminal rows
/// (completed/failed) have their render deleted — the stored result is
/// purged so `/content` 404s afterward, matching the `video.deleted`
/// claim instead of leaving a zombie download behind. Idempotent on
/// every state.
pub async fn video_delete(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return openai_error(400, "invalid video id");
    }
    let Some(row) = video_row(&state, &id) else {
        return openai_error(404, &format!("video {id} not found"));
    };
    match row.state.as_str() {
        "queued" | "running" => {
            let engine = row
                .request_json
                .parse::<serde_json::Value>()
                .ok()
                .and_then(|r| {
                    r.get("engine")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_string)
                });
            cancel_child_best_effort(&state, engine.as_deref(), &id).await;
            state.jobs.record_cancelled(&state, &id);
        }
        "completed" | "failed" => {
            state
                .jobs
                .record_deleted(&state, &id, row.artifact_path.as_deref());
        }
        _ => {} // already cancelled/deleted — idempotent
    }
    axum::Json(serde_json::json!({
        "id": id,
        "object": "video.deleted",
        "deleted": true,
    }))
    .into_response()
}

/// POST /v1/images/upscale — standalone ESRGAN upscale over HTTP
/// (upstream PR2026, sd-server master-929+). Sync, no diffusion model
/// load, no job machinery: upscaler weights come from the engine's
/// `--hires-upscalers-dir`. Serves from a live diffusion child; pass
/// `?model=NAME` to boot one on demand.
pub async fn upscale(
    State(state): State<Arc<AppState>>,
    key_ext: Option<axum::Extension<crate::keys::KeyCtx>>,
    Query(params): Query<JobQuery>,
    body: Bytes,
) -> Response {
    let parsed: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return openai_error(400, &format!("invalid JSON: {e}")),
    };
    if parsed
        .get("image")
        .and_then(serde_json::Value::as_str)
        .is_none_or(str::is_empty)
    {
        return openai_error(
            400,
            "\"image\" is required (base64 or data-URL of the image to upscale)",
        );
    }
    // Admission rides a synthetic model label: no model row is loaded,
    // but rate/key gates must still see the request.
    if let Err(resp) = state.admit_or_respond(key_ext.as_ref(), "upscale") {
        return *resp;
    }
    // Route gate: the standalone endpoint shipped in master-929; on an
    // older engine the child would answer a confusing 404 — teach
    // instead. Unparseable/missing manifests forward (the child speaks).
    let too_old = state
        .with_store(|s| {
            s.list_engines()
                .ok()
                .and_then(|rows| {
                    rows.into_iter()
                        .find(|r| r.kind == blazar_core::engine_kind::EngineKind::SdCpp && r.active)
                        .and_then(|r| {
                            serde_json::from_str::<blazar_runtime::Manifest>(&r.manifest).ok()
                        })
                })
                .is_some_and(|m| m.build_number < 929)
        })
        .unwrap_or(false);
    if too_old {
        return openai_error(
            400,
            "upscale needs sd-server build \u{2265} 929 (standalone /sdcpp/v1/upscale shipped in \
             master-929) \u{2014} run: blazar engine update",
        );
    }
    let children = live_sdcpp_children(&state, params.model.as_deref());
    let (engine, load_ms) = if let Some(engine) = children.first() {
        (engine.clone(), 0)
    } else if let Some(model) = params.model.as_deref() {
        match ensure_with_admission(
            &state,
            model,
            Priority::Normal,
            WorkClass::Interactive,
            None,
            false, // no mmproj lane
            true,  // component sets are the diffusion lane's cargo
            false,
        )
        .await
        {
            Ok(ok) => ok,
            Err(resp) => return *resp,
        }
    } else {
        return openai_error(
            400,
            "no live diffusion child \u{2014} POST /v1/images/generations boots one, or pass \
             ?model=NAME to boot a family's child (the upscaler itself needs no model load)",
        );
    };
    forward_images(
        &state,
        &engine,
        "/sdcpp/v1/upscale",
        Some("application/json"),
        &body,
        load_ms,
    )
    .await
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
             then capabilities serve (pass ?model=NAME — store name or \
             canonical engine name — to pick a family's child)",
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

    #[test]
    fn unit__child_matches_model__accepts_store_alias_via_canonical_row() {
        let child = "wan_2.1_comfyui_repackaged";
        // Canonical spelling: exact match, no store roundtrip needed.
        assert!(child_matches_model(child, None, child));
        // Caller-spelled store alias: matches once the store resolves it
        // onto the canonical row name.
        assert!(child_matches_model("wan_2.1", Some(child), child));
        // Different family: never matches, resolved or not.
        assert!(!child_matches_model(
            "qwen-image-2.1",
            Some("qwen_image21"),
            child
        ));
        assert!(!child_matches_model("wan_2.1", None, child)); // unresolved alias
        // Store resolution landing elsewhere must not widen the match.
        assert!(!child_matches_model("wan_2.1", Some("other_family"), child));
    }

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
            last_used_at: 1,
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
        assert!(
            canonicalize_video_frames(&mut v)
                .unwrap_err()
                .contains("positive integer")
        );

        let mut v = serde_json::json!({"frames": 0});
        assert!(
            canonicalize_video_frames(&mut v)
                .unwrap_err()
                .contains("positive integer")
        );

        // duration joins the conflict vocabulary; fps itself stays.
        let mut v = serde_json::json!({"frames": 33, "duration": 1});
        let err = canonicalize_video_frames(&mut v).unwrap_err();
        assert!(
            err.contains("frames=33") && err.contains("duration=16"),
            "{err}"
        );

        let mut v = serde_json::json!({"duration": -1});
        assert!(
            canonicalize_video_frames(&mut v)
                .unwrap_err()
                .contains("positive number of seconds")
        );
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
    fn unit__gate_size__reads_documented_width_height_dialect() {
        // The API documents top-level width/height and the child takes
        // them verbatim; a 320x320 request must not be billed the 512x512
        // default (over-reject of a render that fits).
        assert_eq!(
            gate_size(&serde_json::json!({"width": 320, "height": 320})),
            (320, 320)
        );
        assert_eq!(
            gate_size(&serde_json::json!({"width": 768, "height": 432})),
            (768, 432)
        );
        // Zero/partial/malformed numeric dims are not a geometry — the
        // conservative default still wins.
        assert_eq!(
            gate_size(&serde_json::json!({"width": 0, "height": 320})),
            (512, 512)
        );
        assert_eq!(gate_size(&serde_json::json!({"width": 320})), (512, 512));
        assert_eq!(
            gate_size(&serde_json::json!({"width": "320", "height": "320"})),
            (512, 512)
        );
        // The explicit size dialect outranks stray numeric fields.
        assert_eq!(
            gate_size(&serde_json::json!({"size": "320x320", "width": 1024, "height": 1024})),
            (320, 320)
        );
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
    fn unit__vram_overcommit_enabled__off_values_only_disable() {
        // Unset (default), enabled spellings and garbage leave the lever on.
        for raw in [
            None,
            Some("1"),
            Some("true"),
            Some("on"),
            Some("yes"),
            Some("junk"),
        ] {
            assert!(vram_overcommit_enabled(raw), "enabled for {raw:?}");
        }
        // Explicit off-spellings (case/whitespace tolerant) disable it.
        for raw in ["0", "false", "OFF", " no "] {
            assert!(!vram_overcommit_enabled(Some(raw)), "disabled for {raw:?}");
        }
    }

    #[test]
    fn unit__vram_overcommit_disabled_teach__names_the_switch() {
        // The refusal must tell the operator where their own switch is.
        assert!(VRAM_OVERCOMMIT_DISABLED_TEACH.contains("BLAZAR_VRAM_OVERCOMMIT=0"));
    }

    #[test]
    fn unit__quality_step_multiplier__tiers_synonyms_auto_and_teaching() {
        assert_eq!(quality_step_multiplier("low"), Ok(Some(0.5)));
        assert_eq!(quality_step_multiplier("standard"), Ok(Some(0.5)));
        assert_eq!(quality_step_multiplier("medium"), Ok(Some(0.75)));
        assert_eq!(quality_step_multiplier("high"), Ok(Some(1.0)));
        assert_eq!(quality_step_multiplier("hd"), Ok(Some(1.0)));
        assert_eq!(quality_step_multiplier("xhigh"), Ok(Some(1.25)));
        assert_eq!(quality_step_multiplier("max"), Ok(Some(1.5)));
        // Auto hands the budget back to the engine default.
        assert_eq!(quality_step_multiplier("auto"), Ok(None));
        let err = quality_step_multiplier("ultra").unwrap_err();
        assert!(err.contains("unknown quality \"ultra\""), "{err}");
        assert!(err.contains("xhigh, max"), "{err}");
    }

    #[test]
    fn unit__style_preset__vivid_natural_only() {
        assert_eq!(style_preset("vivid"), Some(1.25));
        assert_eq!(style_preset("natural"), Some(0.75));
        assert_eq!(style_preset("cinematic"), None);
    }

    #[test]
    fn unit__openai_surface_teaching__covers_every_unserved_knob() {
        let clean = serde_json::json!({"model": "m", "prompt": "p"});
        assert!(openai_surface_teaching(&clean, false, None).is_none());

        let url = serde_json::json!({"model": "m", "response_format": "url"});
        let msg = openai_surface_teaching(&url, false, None).unwrap();
        assert!(msg.contains("b64_json"), "{msg}");

        // b64_json is the served shape — no teaching.
        let b64 = serde_json::json!({"model": "m", "response_format": "b64_json"});
        assert!(openai_surface_teaching(&b64, false, None).is_none());

        for key in [
            "moderation",
            "partial_images",
            "background",
            "input_fidelity",
        ] {
            let req = serde_json::json!({"model": "m", key: "high"});
            let msg = openai_surface_teaching(&req, false, None)
                .unwrap_or_else(|| panic!("{key} must teach"));
            assert!(msg.contains(key), "{key}: {msg}");
        }

        // Style: curated family passes, everything else teaches.
        let styled = serde_json::json!({"model": "m", "style": "vivid"});
        assert!(
            openai_surface_teaching(&styled, false, Some("city96/Qwen-Image-2.1-GGUF")).is_none()
        );
        let flux = openai_surface_teaching(&styled, false, Some("city96/FLUX.1-dev-gguf")).unwrap();
        assert!(flux.contains("curated per family"), "{flux}");
        let unknown = openai_surface_teaching(&styled, false, None).unwrap();
        assert!(unknown.contains("txt_cfg"), "{unknown}");
        let bad = serde_json::json!({"model": "m", "style": "noir"});
        assert!(openai_surface_teaching(&bad, false, Some("city96/Qwen-Image-2.1-GGUF")).is_some());

        // Quality vocabulary teaches on the unknown value.
        let q = serde_json::json!({"model": "m", "quality": "ultra"});
        assert!(openai_surface_teaching(&q, false, None).is_some());
        assert!(
            openai_surface_teaching(
                &serde_json::json!({"model": "m", "quality": "max"}),
                false,
                None
            )
            .is_none()
        );

        // output_format is mode-scoped: webm teaches on the image lane,
        // jpeg teaches on the video lane, native values pass.
        let img = serde_json::json!({"model": "m", "output_format": "webm"});
        assert!(openai_surface_teaching(&img, false, None).is_some());
        let vid = serde_json::json!({"model": "m", "output_format": "jpeg"});
        assert!(openai_surface_teaching(&vid, true, None).is_some());
        assert!(
            openai_surface_teaching(
                &serde_json::json!({"model": "m", "output_format": "avi"}),
                true,
                None
            )
            .is_none()
        );
    }

    #[test]
    fn unit__translate_to_native__consumed_openai_knobs_never_ride() {
        let req = serde_json::json!({
            "model": "m", "prompt": "p",
            "quality": "high", "style": "vivid", "response_format": "b64_json",
            "output_format": "webp"
        });
        let native = translate_to_native(&req, false);
        for gone in ["quality", "style", "response_format"] {
            assert!(native.get(gone).is_none(), "{gone} rode: {native}");
        }
        // The child-honored output field still rides.
        assert_eq!(native["output_format"], serde_json::json!("webp"));
    }

    #[test]
    fn unit__translate_to_native__unparseable_size_passes_through() {
        let req = serde_json::json!({ "model": "m", "prompt": "p", "size": "big" });
        let native = translate_to_native(&req, false);
        assert_eq!(native["size"], serde_json::json!("big"));
        assert!(native.get("width").is_none());
    }

    fn multipart_part(name: &str, filename: Option<&str>, data: &[u8]) -> crate::whisper::Part {
        crate::whisper::Part {
            name: name.to_string(),
            filename: filename.map(str::to_string),
            content_type: None,
            data: data.to_vec(),
        }
    }

    #[test]
    fn unit__b64_encode__rfc4648_vectors() {
        assert_eq!(b64_encode(b""), "");
        assert_eq!(b64_encode(b"f"), "Zg==");
        assert_eq!(b64_encode(b"fo"), "Zm8=");
        assert_eq!(b64_encode(b"foo"), "Zm9v");
        assert_eq!(b64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(b64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(b64_encode(b"foobar"), "Zm9vYmFy");
        // The pair the vid_gen fixture already pins repo-wide.
        assert_eq!(b64_encode(b"GkX"), "R2tY");
    }

    #[test]
    fn unit__variations_payload__init_image_neutral_prompt_random_seed() {
        let parts = [
            multipart_part("model", None, b"flux-dev"),
            multipart_part("image", Some("cat.png"), b"GkX"),
            multipart_part("n", None, b"2"),
            multipart_part("size", None, b"512x512"),
            multipart_part("prompt", None, b""),
            multipart_part("user", None, b"tenant-7"),
        ];
        let v = variations_payload(&parts).expect("assembles");
        assert_eq!(v["init_image"], "R2tY");
        assert_eq!(v["prompt"], "");
        assert_eq!(v["seed"], -1);
        assert_eq!(v["n"], 2);
        assert_eq!(v["size"], "512x512");
        assert!(v.get("model").is_none(), "model never rides");
        assert!(v.get("user").is_none(), "user never rides");
    }

    #[test]
    fn unit__variations_payload__typed_fields_and_digit_prompts_survive() {
        let parts = [
            multipart_part("image", Some("i.png"), b"x"),
            multipart_part("prompt", None, b"2024 a cat"),
            multipart_part("seed", None, b"42"),
            multipart_part("strength", None, b"0.5"),
            multipart_part("async", None, b"true"),
        ];
        let v = variations_payload(&parts).expect("assembles");
        assert_eq!(
            v["prompt"], "2024 a cat",
            "digit-only prefixes stay strings"
        );
        assert_eq!(v["seed"], 42);
        assert_eq!(v["strength"], 0.5);
        assert_eq!(v["async"], true);
    }

    #[test]
    fn unit__variations_payload__missing_duplicate_and_stray_files_teach() {
        let err =
            variations_payload(&[multipart_part("model", None, b"m")]).expect_err("image required");
        assert!(err.contains("image"), "{err}");
        let two = [
            multipart_part("image", Some("a.png"), b"a"),
            multipart_part("image", Some("b.png"), b"b"),
        ];
        let err = variations_payload(&two).expect_err("one image only");
        assert!(err.contains("exactly one"), "{err}");
        let stray = [
            multipart_part("image", Some("a.png"), b"a"),
            multipart_part("mask", Some("m.png"), b"m"),
        ];
        let err = variations_payload(&stray).expect_err("mask is edits cargo");
        assert!(err.contains("unexpected file part"), "{err}");
        let bad_seed = [
            multipart_part("image", Some("a.png"), b"a"),
            multipart_part("seed", None, b"soon"),
        ];
        let err = variations_payload(&bad_seed).expect_err("seed must be an integer");
        assert!(err.contains("seed"), "{err}");
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
        assert_eq!(out["data"].as_array().unwrap().len(), 0);
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
        assert!(
            native_job_to_openai(&failed)["data"]
                .as_array()
                .is_some_and(std::vec::Vec::is_empty)
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__b64_decode__rfc4648_vectors_and_rejections() {
        // RFC 4648 test vectors.
        assert_eq!(b64_decode("").unwrap(), b"");
        assert_eq!(b64_decode("Zg==").unwrap(), b"f");
        assert_eq!(b64_decode("Zm8=").unwrap(), b"fo");
        assert_eq!(b64_decode("Zm9v").unwrap(), b"foo");
        assert_eq!(b64_decode("Zm9vYg==").unwrap(), b"foob");
        assert_eq!(b64_decode("Zm9vYmE=").unwrap(), b"fooba");
        assert_eq!(b64_decode("Zm9vYmFy").unwrap(), b"foobar");
        // Roundtrip against the encoder over binary with all byte values.
        let data: Vec<u8> = (0..=u8::MAX).collect();
        let encoded = b64_encode(&data);
        assert_eq!(b64_decode(&encoded).unwrap(), data);
        // Rejections: ragged length, bad alphabet, misplaced padding.
        assert!(b64_decode("Zm9").is_err());
        assert!(b64_decode("Zm9*").is_err());
        assert!(b64_decode("Z=9v").is_err());
        assert!(b64_decode("Zm9v=Y==").is_err());
        assert!(b64_decode("A===").is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__canonicalize_video_frames__seconds_synonym_string_or_number() {
        // Sora ships seconds as a STRING ("4"|"8"|"12"); both dialects
        // convert with the assumed 16 fps default.
        let mut v = serde_json::json!({"seconds": "8"});
        canonicalize_video_frames(&mut v).unwrap();
        assert_eq!(v["video_frames"], serde_json::json!(128));
        assert!(v.get("seconds").is_none());

        let mut v = serde_json::json!({"seconds": 4});
        canonicalize_video_frames(&mut v).unwrap();
        assert_eq!(v["video_frames"], serde_json::json!(64));

        // Client fps participates: 2s x 12fps = 24 frames.
        let mut v = serde_json::json!({"seconds": "2", "fps": 12});
        canonicalize_video_frames(&mut v).unwrap();
        assert_eq!(v["video_frames"], serde_json::json!(24));

        // duration + seconds are the same dial: agreeing values collapse.
        let mut v = serde_json::json!({"duration": 2, "seconds": "2"});
        canonicalize_video_frames(&mut v).unwrap();
        assert_eq!(v["video_frames"], serde_json::json!(32));

        // Disagreeing values name both spellings in the conflict.
        let mut v = serde_json::json!({"seconds": "4", "frames": 33});
        let err = canonicalize_video_frames(&mut v).unwrap_err();
        assert!(
            err.contains("seconds=64") && err.contains("frames=33"),
            "{err}"
        );

        // Junk answers by name.
        let mut v = serde_json::json!({"seconds": "four"});
        assert!(
            canonicalize_video_frames(&mut v)
                .unwrap_err()
                .contains("seconds must be a positive")
        );
        let mut v = serde_json::json!({"seconds": 0});
        assert!(canonicalize_video_frames(&mut v).is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__request_echo__collects_client_facing_fields() {
        let v = serde_json::json!({
            "model": "wan", "prompt": "a cat", "size": "832x480",
            "seconds": 8.0, "video_frames": 129, "fps": 16, "n": 1,
            "negative_prompt": "blurry", "sample_params": {"sample_steps": 20}
        });
        let echo = request_echo(&v);
        assert_eq!(echo["prompt"], serde_json::json!("a cat"));
        assert_eq!(echo["size"], serde_json::json!("832x480"));
        assert_eq!(echo["seconds"], serde_json::json!(8.0));
        assert_eq!(echo["video_frames"], serde_json::json!(129));
        assert_eq!(echo["fps"], serde_json::json!(16));
        assert_eq!(echo["n"], serde_json::json!(1));
        assert!(echo.get("negative_prompt").is_none());
        assert!(echo.get("sample_params").is_none());
        assert!(echo.get("model").is_none());
    }

    fn video_row_fixture(
        state: &str,
        request_json: &str,
        result_json: Option<&str>,
    ) -> blazar_core::store::JobRow {
        blazar_core::store::JobRow {
            id: "job_video1".into(),
            kind: "video".into(),
            model: Some("wan".into()),
            state: state.into(),
            request_json: request_json.into(),
            result_json: result_json.map(str::to_string),
            error: None,
            artifact_path: None,
            created_at: 1000,
            updated_at: 1005,
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__sora_video_object__statuses_progress_and_seconds_echo() {
        let request = r#"{"engine":"sdcpp","request":{"prompt":"a cat","size":"832x480","seconds":8.0,"fps":16,"video_frames":129}}"#;
        // Queued: status maps, progress 0, request echo surfaces.
        let row = video_row_fixture("queued", request, None);
        let obj = sora_video_object(&row, None);
        assert_eq!(obj["object"], serde_json::json!("video"));
        assert_eq!(obj["status"], serde_json::json!("queued"));
        assert_eq!(obj["progress"], serde_json::json!(0));
        assert_eq!(obj["prompt"], serde_json::json!("a cat"));
        assert_eq!(obj["size"], serde_json::json!("832x480"));
        assert_eq!(obj["seconds"], serde_json::json!(8.0));
        assert!(obj.get("completed_at").is_none());

        // Running + fresh child progress: the live number wins over any
        // state-derived guess.
        let row = video_row_fixture("running", request, None);
        let fresh = serde_json::json!({"status": "generating", "progress": 42});
        let obj = sora_video_object(&row, Some(&fresh));
        assert_eq!(obj["status"], serde_json::json!("in_progress"));
        assert_eq!(obj["progress"], serde_json::json!(42));

        // Completed inline: seconds from the render's own frame_count/
        // fps (the engine normalized 129 = 4n+1), progress 100,
        // completed_at stamped.
        let result = r#"{"status":"completed","result":{"b64_json":"Zm9vYmFy","mime_type":"video/webm","fps":24.0,"frame_count":129}}"#;
        let row = video_row_fixture("completed", request, Some(result));
        let obj = sora_video_object(&row, None);
        assert_eq!(obj["status"], serde_json::json!("completed"));
        assert_eq!(obj["progress"], serde_json::json!(100));
        assert_eq!(obj["completed_at"], serde_json::json!(1005));
        assert_eq!(obj["seconds"], serde_json::json!(5.375));

        // Result without frame truth falls back to the request echo.
        let thin = r#"{"status":"completed","result":{"b64_json":"Zm9vYmFy"}}"#;
        let row = video_row_fixture("completed", request, Some(thin));
        assert_eq!(
            sora_video_object(&row, None)["seconds"],
            serde_json::json!(8.0)
        );

        // Cancelled stays an honest superset status.
        let row = video_row_fixture("cancelled", request, None);
        assert_eq!(
            sora_video_object(&row, None)["status"],
            serde_json::json!("cancelled")
        );

        // Failed carries the ledger error.
        let row = blazar_core::store::JobRow {
            state: "failed".into(),
            error: Some("engine exploded".into()),
            ..video_row_fixture("failed", request, None)
        };
        assert_eq!(
            sora_video_object(&row, None)["error"],
            serde_json::json!("engine exploded")
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__completed_video_bytes__inline_artifact_and_gates() {
        // In-flight rows never serve content (Sora's 404-until-done).
        let request = r#"{"engine":"sdcpp","request":{"seconds":8.0}}"#;
        let row = video_row_fixture("running", request, None);
        assert!(matches!(
            completed_video_bytes(&row),
            Err(VideoContentError::NotCompleted)
        ));

        // Inline completed result decodes to bytes + child mime type.
        let result =
            r#"{"status":"completed","result":{"b64_json":"Zm9vYmFy","mime_type":"video/webm"}}"#;
        let row = video_row_fixture("completed", request, Some(result));
        let (bytes, mime) = completed_video_bytes(&row).unwrap();
        assert_eq!(bytes, b"foobar");
        assert_eq!(mime, "video/webm");

        // Artifact-spilled: the spill file holds the same child job JSON.
        let dir = std::env::temp_dir().join(format!(
            "blazar-video-artifact-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("result");
        std::fs::write(&path, result.as_bytes()).unwrap();
        let row = blazar_core::store::JobRow {
            result_json: Some(r#"{"artifact":true,"bytes":9001}"#.into()),
            artifact_path: Some(path.to_string_lossy().into_owned()),
            ..video_row_fixture("completed", request, None)
        };
        let (bytes, mime) = completed_video_bytes(&row).unwrap();
        assert_eq!(
            (bytes.as_slice(), mime.as_str()),
            (b"foobar".as_slice(), "video/webm")
        );
        std::fs::remove_dir_all(&dir).ok();

        // Dropped spill answers Gone.
        let row = blazar_core::store::JobRow {
            result_json: Some(r#"{"artifact":"dropped","bytes":9001}"#.into()),
            ..video_row_fixture("completed", request, None)
        };
        assert!(matches!(
            completed_video_bytes(&row),
            Err(VideoContentError::Gone)
        ));
    }
}
