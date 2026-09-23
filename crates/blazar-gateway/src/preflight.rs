//! Load preflights: named, fail-fast checks that turn silent swap-death
//! and VRAM overcommit into teaching errors. Pure math — no side effects.

use axum::response::IntoResponse as _;
use blazar_core::GgufMeta;

/// Flat per-image token estimate (mmproj clip ≈ pixels/750; typical
/// 512-1024px images land ~400-1400 tokens). F82: image data used to
/// walk as raw base64 bytes/4 into the estimator (a 1 MiB image
/// estimated ~262K tokens and false-tripped the 90% gate), while the
/// exact path counted multimodal arrays as empty text (~0 tokens and a
/// silent bypass). Both directions now carry this flat figure.
const IMAGE_TOKEN_EST: u64 = 1500;

fn walk_strings(v: &serde_json::Value, bytes: &mut u64) {
    match v {
        serde_json::Value::String(s) => *bytes += s.len() as u64,
        serde_json::Value::Array(a) => a.iter().for_each(|x| walk_strings(x, bytes)),
        serde_json::Value::Object(o) => {
            if o.contains_key("image_url") {
                *bytes += IMAGE_TOKEN_EST * 4;
                return;
            }
            o.values().for_each(|x| walk_strings(x, bytes));
        }
        _ => {}
    }
}

/// Rough token estimate for a chat-family body: total string content
/// across messages + system + tools serialized, bytes/4 (conservative
/// for English/code; CJK inflates ~2x and the exact check catches it).
#[must_use]
pub fn prompt_tokens_est(body: &serde_json::Value) -> u64 {
    let mut bytes = 0u64;
    walk_strings(body, &mut bytes);
    (bytes / 4).max(1)
}

/// Context bound for prompt-fit admission: the LIVE per-slot ctx of a
/// resident instance when one exists (auto-fit or explicit slots may
/// have traded the configured depth for width — e.g. 16K configured,
/// 4x4096 compiled — and preflight must bound against what a single
/// slot actually holds), the configured effective ctx otherwise
/// (cold-spawn estimate). Rows whose profile has not compiled yet
/// (ctx == 0) fall back to the config estimate.
pub fn admission_ctx(state: &crate::state::AppState, model: &str) -> u32 {
    state
        .sup
        .ps()
        .into_iter()
        .find(|p| p.name == model && p.ctx > 0)
        .map_or_else(|| state.config.effective_ctx(model), |p| p.ctx)
}

/// Prompt-fit admission (K1): refuse requests that cannot fit the
/// effective context BEFORE the kernel silently truncates them (the
/// sentinel's most common post-hoc detection, moved to pre-hoc).
/// Byte-estimate first (hot path, ~free); exact `/tokenize` only when
/// the estimate crosses the 90% threshold AND the child is running.
pub async fn enforce_prompt_fits(
    state: &crate::state::AppState,
    model: &str,
    body: &serde_json::Value,
    effective_ctx: u32,
) -> Result<(), Box<axum::response::Response>> {
    if !state.config.prompt_preflight || effective_ctx == 0 {
        return Ok(());
    }
    let est = prompt_tokens_est(body);
    // Cheap path: under 90% of ctx — the kernel fits it.
    if est < u64::from(effective_ctx) * 9 / 10 {
        return Ok(());
    }
    // Over threshold: try the exact count against a RUNNING child.
    let running = state
        .sup
        .live_http_endpoints()
        .into_iter()
        .find(|e| e.name == model)
        .map(|e| (crate::proxy::child_base(&e.endpoint), e));
    let exact = match running {
        Some((base, engine)) => {
            // F82: multimodal bodies — collect text fields AND count
            // image parts (arrays used to tokenize as empty text and
            // bypass the fit check entirely).
            let (text, images) = body
                .pointer("/messages")
                .and_then(|m| m.as_array())
                .map(|a| {
                    let mut parts: Vec<String> = Vec::new();
                    let mut images = 0u64;
                    for m in a {
                        match m.get("content") {
                            Some(serde_json::Value::String(s)) => parts.push(s.clone()),
                            Some(serde_json::Value::Array(blocks)) => {
                                for b in blocks {
                                    match b.get("type").and_then(|v| v.as_str()) {
                                        Some("text") => {
                                            if let Some(t) = b.get("text").and_then(|v| v.as_str())
                                            {
                                                parts.push(t.to_string());
                                            }
                                        }
                                        Some("image_url" | "image") => images += 1,
                                        _ => {}
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    (parts.join("\n"), images)
                })
                // /api/generate bodies carry the prompt under "prompt"
                // (F13 — the exact count must not silently see "").
                .or_else(|| {
                    body.get("prompt")
                        .and_then(|p| p.as_str())
                        .map(str::to_string)
                        .map(|p| (p, 0u64))
                })
                .unwrap_or_default();
            match crate::proxy::child_auth(state.http.post(format!("{base}/tokenize")), &engine)
                .json(&serde_json::json!({"content": text}))
                .send()
                .await
            {
                Ok(r) => r.json::<serde_json::Value>().await.ok().and_then(|v| {
                    v.get("tokens")
                        .and_then(|t| t.as_array())
                        .map(std::vec::Vec::len)
                        .map(|n| n as u64)
                        .map(|exact| {
                            // Tools ride as a bytes/4 estimate on top (the
                            // engine serializes tool defs into the prompt).
                            let tools_est = body
                                .get("tools")
                                .map_or(0, |t| u64::try_from(t.to_string().len()).unwrap_or(0) / 4);
                            exact
                                .saturating_add(tools_est)
                                .saturating_add(images.saturating_mul(IMAGE_TOKEN_EST))
                        })
                }),
                Err(_) => None,
            }
        }
        None => None,
    };
    let tokens = exact.unwrap_or(est);
    if tokens >= u64::from(effective_ctx) {
        let how = if exact.is_some() {
            "exact"
        } else {
            "estimated"
        };
        Err(Box::new(
            crate::proxy::openai_error(
                400,
                &format!(
                    "prompt {tokens} tokens ({how}) exceeds the {effective_ctx}-token context for \
                     {model:?} — the engine would silently truncate it. Shorten the prompt, raise \
                     ctx (config/[model_overrides] ctx or X-Blazar-Num-Ctx), or set \
                     prompt_preflight = false to allow truncation",
                ),
            )
            .into_response(),
        ))
    } else {
        Ok(())
    }
}

/// f16 KV cache size in MiB for a target ctx. 2 (K+V) * layers *
/// `kv_heads` * `head_dim` * ctx * 2 bytes. Conservative: no quantized-KV
/// credit (the engine may still fit it via the cache-type ladder).
#[must_use]
pub fn kv_f16_mib(meta: &GgufMeta, ctx: u64) -> u64 {
    blazar_core::coreside::kv_f16_mib(meta, ctx)
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    fn meta(layers: u64, heads: u64, kv: u64, emb: u64) -> GgufMeta {
        GgufMeta {
            architecture: "qwen3".into(),
            block_count: Some(layers),
            head_count: Some(heads),
            head_count_kv: Some(kv),
            embedding_length: Some(emb),
            ..GgufMeta::default()
        }
    }

    #[test]
    fn unit__kv_f16_mib__known_shape() {
        // 32 layers, 8 kv heads, 128 head dim, 32k ctx:
        // 2*32*8*128*32768*2 = 4 GiB exactly.
        let m = meta(32, 32, 8, 4096);
        assert_eq!(kv_f16_mib(&m, 32_768), 4096);
        // Linear in ctx.
        assert_eq!(kv_f16_mib(&m, 8_192), 1024);
        // Missing head_dim derives from emb/heads (4096/32 = 128).
        let mut m2 = m;
        m2.head_dim = None;
        assert_eq!(kv_f16_mib(&m2, 32_768), 4096);
    }
}
