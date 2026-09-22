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

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{header, HeaderMap};
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

use crate::proxy::{child_auth, child_base, ensure_with_admission, openai_error, resolve_model};
use crate::queue::{Priority, WorkClass};
use crate::state::AppState;

/// Domain gate for the images routes. `row: None` (unknown model) is NOT
/// a rejection — `ensure_with_admission` owns that 404 — it only means
/// the gate has nothing to check.
pub(crate) enum ImagesGate {
    Serve,
    Reject(String),
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
pub(crate) fn images_gate(row: Option<&blazar_core::store::ModelRow>, edits: bool) -> ImagesGate {
    let Some(row) = row else {
        return ImagesGate::Serve;
    };
    if !row.has_component_set() {
        return ImagesGate::Reject(format!(
            "\"{}\" is not a diffusion model — /v1/images serves diffusion component sets \
             (DiT + VAE + text encoder, sdcpp lane); chat models serve /v1/chat/completions",
            row.name
        ));
    }
    if edits && !row.serves_image_edits() {
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
    match images_gate(row.as_ref(), false) {
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
    match images_gate(row.as_ref(), true) {
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

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    fn row(vae: Option<&str>, vision: Option<&str>) -> blazar_core::store::ModelRow {
        let mut components = Vec::new();
        if let Some(vae) = vae {
            components.push(blazar_core::store::ComponentFile::new("--vae", vae));
        }
        components.push(blazar_core::store::ComponentFile::new("--llm", "te.gguf"));
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
        let gate = images_gate(Some(&row(None, None)), false);
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
    fn unit__images_gate__diffusion_row_serves_generations() {
        assert!(matches!(
            images_gate(Some(&row(Some("vae.safetensors"), None)), false),
            ImagesGate::Serve
        ));
    }

    #[test]
    fn unit__images_gate__edits_without_vision_encoder_teaches_repull() {
        let gate = images_gate(Some(&row(Some("vae.safetensors"), None)), true);
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
        let gate = images_gate(Some(&flux_row()), true);
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
            images_gate(Some(&flux_row()), false),
            ImagesGate::Serve
        ));
    }

    #[test]
    fn unit__images_gate__edits_with_vision_encoder_serves() {
        assert!(matches!(
            images_gate(
                Some(&row(Some("vae.safetensors"), Some("mmproj.gguf"))),
                true
            ),
            ImagesGate::Serve
        ));
    }

    #[test]
    fn unit__images_gate__unknown_row_defers_to_ensure() {
        // Unknown model is not the gate's call — ensure owns the 404.
        assert!(matches!(images_gate(None, true), ImagesGate::Serve));
    }
}
