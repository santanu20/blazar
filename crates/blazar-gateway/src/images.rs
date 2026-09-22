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

/// One `SSE` poll per second; 900 polls ≈ 15 minutes of relay before
/// the gateway gives up (the state client bounds each poll itself).
const JOB_POLL_INTERVAL_MS: u64 = 1_000;
const JOB_POLL_MAX: u32 = 900;

fn is_plain_generation_request(v: &serde_json::Value) -> bool {
    v.as_object().is_some_and(|o| {
        o.keys()
            .all(|k| PLAIN_GENERATION_KEYS.contains(&k.as_str()))
    })
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

/// Per-key admission + shape checks shared by both routes; `Err` is a
/// ready response. Shape gates run BEFORE admission so malformed input
/// never loads a model.
fn admit_or_respond(
    state: &AppState,
    key_ext: Option<&axum::Extension<crate::keys::KeyCtx>>,
    model: &str,
) -> Result<(), Box<Response>> {
    if let Some(axum::Extension(k)) = key_ext {
        if let Some(entry) = state.keys.entry(&k.name) {
            if let Err(rej) = state.keys.check(&entry, model) {
                return Err(Box::new(rej.to_response()));
            }
            state.keys.charge_request(&k.name);
        }
    }
    Ok(())
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
    let url = format!("{}{path}", child_base(&engine.endpoint));
    let mut rb = state.http.post(&url);
    if let Some(ct) = content_type {
        rb = rb.header(header::CONTENT_TYPE, ct);
    }
    let upstream = child_auth(rb, engine).body(body.to_vec()).send().await;
    let resp = match upstream {
        Ok(r) => r,
        Err(e) => {
            state.sup.reap_dead_children().await;
            return openai_error(502, &format!("engine request failed: {e:#}"));
        }
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
    if let Err(resp) = admit_or_respond(&state, key_ext.as_ref(), &model) {
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
    if let Err(resp) = admit_or_respond(&state, key_ext.as_ref(), &model) {
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
    match images_gate(row.as_ref(), Surface::VidGen) {
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
        true, // videos lane: component sets are its cargo
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
    if let Err(resp) = admit_or_respond(&state, key_ext.as_ref(), &model) {
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
    let url = format!("{}{native_path}", child_base(&engine.endpoint));
    let rb = state.http.post(&url).json(native);
    let resp = match child_auth(rb, engine).send().await {
        Ok(r) => r,
        Err(e) => {
            state.sup.reap_dead_children().await;
            return Err(Box::new(openai_error(
                502,
                &format!("engine request failed: {e:#}"),
            )));
        }
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

/// One poll of the child's job state; `Err` = transport/decode
/// failure (child died mid-poll), `Ok(None)` = HTTP 404 from the
/// child (job unknown there).
async fn poll_child_job(
    state: &AppState,
    engine: &blazar_runtime::EngineRef,
    job_id: &str,
) -> Result<Option<serde_json::Value>, ()> {
    let url = format!("{}/sdcpp/v1/jobs/{job_id}", child_base(&engine.endpoint));
    let resp = child_auth(state.http.get(&url), engine)
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
/// cancelled) or the poll budget is spent. `Ok(job)` = terminal job;
/// `Err` = child died or budget exhausted (ready 502/504 response).
async fn await_native_job(
    state: &AppState,
    engine: &blazar_runtime::EngineRef,
    job_id: &str,
) -> Result<serde_json::Value, Box<Response>> {
    for _ in 0..JOB_POLL_MAX {
        tokio::time::sleep(std::time::Duration::from_millis(JOB_POLL_INTERVAL_MS)).await;
        match poll_child_job(state, engine, job_id).await {
            Ok(Some(job)) => {
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
                state.sup.reap_dead_children().await;
                return Err(Box::new(openai_error(
                    502,
                    "engine child died mid-generation",
                )));
            }
        }
    }
    Err(Box::new(openai_error(
        504,
        &format!("generation did not finish within {JOB_POLL_MAX} polls"),
    )))
}

/// `SSE` relay: progress events every poll, one terminal event, then
/// close. The spawned task stops on client disconnect (send fails)
/// or child death.
fn stream_native_job(
    state: Arc<AppState>,
    engine: blazar_runtime::EngineRef,
    job_id: String,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(16);
    tokio::spawn(async move {
        for _ in 0..JOB_POLL_MAX {
            tokio::time::sleep(std::time::Duration::from_millis(JOB_POLL_INTERVAL_MS)).await;
            let Ok(Some(job)) = poll_child_job(&state, &engine, &job_id).await else {
                break;
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
                break; // client went away
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
        if let Ok(Some(job)) = poll_child_job(&state, engine, &job_id).await {
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
    let sent = child_auth(state.http.post(&url), &engine)
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

/// GET /v1/images/capabilities — the live child's sampler/cache/LoRA
/// menu. Read-only: boots nothing; no live child teaches instead.
pub async fn capabilities(State(state): State<Arc<AppState>>) -> Response {
    let children = live_sdcpp_children(&state, None);
    let Some(engine) = children.first() else {
        return openai_error(
            400,
            "no live diffusion child — POST /v1/images/generations boots one, \
             then capabilities serve",
        );
    };
    let url = format!("{}/sdcpp/v1/capabilities", child_base(&engine.endpoint));
    match child_auth(state.http.get(&url), engine).send().await {
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
