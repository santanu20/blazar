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
/// F82: multimodal bodies — collect text fields AND count image parts
/// (arrays used to tokenize as empty text and bypass the fit check
/// entirely). `/api/generate` bodies carry the prompt under `prompt`
/// (F13 — the exact count must not silently see "").
fn prompt_text_and_images(body: &serde_json::Value) -> (String, u64) {
    body.pointer("/messages")
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
                                    if let Some(t) = b.get("text").and_then(|v| v.as_str()) {
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
        .or_else(|| {
            body.get("prompt")
                .and_then(|p| p.as_str())
                .map(str::to_string)
                .map(|p| (p, 0u64))
        })
        .unwrap_or_default()
}

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
            let (text, images) = prompt_text_and_images(body);
            match crate::proxy::child_auth(
                crate::state::child_client(state, &engine.endpoint)
                    .post(format!("{base}/tokenize")),
                &engine,
            )
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

/// What a request strictly asks of the model, extracted from its body —
/// the only features the capability certificate can vouch for. Strict
/// shapes only: a false positive would refuse legitimate plain chat,
/// while a false negative merely fails open (the sentinel still watches
/// the response live).
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub struct CapabilityNeeds {
    pub tools: bool,
    pub vision: bool,
    pub json_mode: bool,
}

impl CapabilityNeeds {
    #[must_use]
    pub fn any(self) -> bool {
        self.tools || self.vision || self.json_mode
    }
}

/// Extract [`CapabilityNeeds`] from a parsed chat body. `ollama_shape`
/// selects the ollama dialect (`images` arrays, `format` field) over
/// the `OpenAI` dialect (content blocks, `response_format`).
#[must_use]
pub fn capability_needs(parsed: &serde_json::Value, ollama_shape: bool) -> CapabilityNeeds {
    let non_empty_arr =
        |v: Option<&serde_json::Value>| v.and_then(|t| t.as_array()).is_some_and(|a| !a.is_empty());
    let tools = non_empty_arr(parsed.get("tools")) || non_empty_arr(parsed.get("functions"));
    let vision = crate::proxy::body_needs_vision(parsed, ollama_shape);
    let json_mode = if ollama_shape {
        parsed.get("format").is_some_and(|f| !f.is_null())
    } else {
        parsed
            .pointer("/response_format/type")
            .and_then(|t| t.as_str())
            .is_some_and(|t| t == "json_object" || t == "json_schema")
    };
    CapabilityNeeds {
        tools,
        vision,
        json_mode,
    }
}

/// Certificate gate (pure): `Some(teaching message)` when a needed
/// capability is VERIFIED-FAILED. Absent probes, `N/A` verdicts, and
/// unknown-or-mismatched engine kinds all pass — a stale or partial
/// certificate must never refuse traffic, only a fresh matching FAIL
/// on the SAME engine kind does (a build bump within the kind keeps
/// the verdict; `blazar model-doctor` refresh advice covers drift).
#[must_use]
pub fn capability_refusal(
    model: &str,
    cert_engine_tag: &str,
    cert_kind: Option<&str>,
    serving_kind: Option<&str>,
    tested_at: i64,
    caps: &serde_json::Value,
    needs: CapabilityNeeds,
) -> Option<String> {
    match (cert_kind, serving_kind) {
        (Some(ck), Some(sk)) if ck == sk => {}
        _ => return None,
    }
    let status = |probe: &str| {
        caps.pointer(&format!("/caps/{probe}/status"))
            .and_then(|s| s.as_str())
            .map(str::to_string)
    };
    let verified = blazar_core::store::epoch_to_utc_date(tested_at);
    for (needed, probe, what) in [
        (needs.tools, "tools", "tool calling"),
        (needs.vision, "vision", "image input"),
        (needs.json_mode, "json", "structured JSON output"),
    ] {
        if needed && status(probe).as_deref() == Some("FAIL") {
            return Some(format!(
                "model {model:?} failed its verified {what} probe (certificate \
{verified}, engine {cert_engine_tag}) — the request would burn a turn on \
garbage output. Plain chat still works; for {what} pick a sibling whose \
certificate shows {probe} = PASS (`blazar scorecard {model}` lists them), \
or re-verify after an engine change: `blazar model-doctor {model}`",
            ));
        }
    }
    None
}

/// Capability-certificate admission: read the stored model-doctor
/// certificate once, then gate the request's strict needs against it.
/// `None` = proceed (no certificate, no matching need, or fail-open
/// staleness); `Some(msg)` = teaching refusal the lane wraps in its own
/// error shape.
pub fn capability_cert_refusal(
    state: &std::sync::Arc<crate::state::AppState>,
    model: &str,
    needs: CapabilityNeeds,
) -> Option<String> {
    if !needs.any() {
        return None;
    }
    let cert = state.with_store(|s| s.get_model_caps_dated(model).ok().flatten())??;
    let caps: serde_json::Value = serde_json::from_str(&cert.2).ok()?;
    // Two SEPARATE store borrows: routed_kind_for takes the store mutex
    // itself, so nesting it inside a with_store closure would deadlock
    // the (non-reentrant) mutex and freeze every store-reading request.
    let cert_kind = state.with_store(|s| s.engine_kind_of_tag(&cert.0).ok().flatten())?;
    let serving_kind = crate::proxy::routed_kind_for(state, model);
    capability_refusal(
        model,
        &cert.0,
        cert_kind.as_ref().map(|k| k.as_str()),
        serving_kind.as_ref().map(|k| k.as_str()),
        cert.1,
        &caps,
        needs,
    )
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    fn cert(tools: &str, vision: &str, json: &str) -> serde_json::Value {
        serde_json::json!({
            "object": "blazar.model-doctor",
            "engine_tag": "llamacpp-b6019",
            "tested_at": 1_790_985_600_i64,
            "caps": {
                "tools": {"status": tools},
                "vision": {"status": vision},
                "json": {"status": json},
            }
        })
    }

    #[test]
    fn unit__capability_needs__strict_shapes_only() {
        // OpenAI dialect: non-empty tools, legacy functions, image
        // content blocks, JSON response_format.
        let oa = serde_json::json!({
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}
            ]}],
            "response_format": {"type": "json_schema"}
        });
        let n = capability_needs(&oa, false);
        assert!(n.tools && n.vision && n.json_mode && n.any());

        // Strictness: EMPTY tools array and plain-text-only bodies ask
        // for nothing — a false positive here would refuse legit chat.
        let plain = serde_json::json!({
            "tools": [],
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": {"type": "text"}
        });
        assert_eq!(capability_needs(&plain, false), CapabilityNeeds::default());

        // Ollama dialect: `images` + `format`, not content blocks.
        let om = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi", "images": ["base64"]}],
            "format": "json",
            "tools": [{"name": "f"}]
        });
        let n = capability_needs(&om, true);
        assert!(n.tools && n.vision && n.json_mode);
        // Ollama plain chat with an images-free body asks nothing.
        let oplain =
            serde_json::json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
        assert_eq!(capability_needs(&oplain, true), CapabilityNeeds::default());
    }

    #[test]
    #[allow(clippy::too_many_lines)] // one assert block per refusal rule
    fn unit__capability_refusal__fresh_matching_fail_only() {
        let needs = CapabilityNeeds {
            tools: true,
            ..CapabilityNeeds::default()
        };
        // FAIL probe + matching need → teaching refusal naming the
        // probe, the verification date, and the refresh command.
        let msg = capability_refusal(
            "m:8b",
            "b11370-cuda",
            Some("llamacpp"),
            Some("llamacpp"),
            1_790_985_600,
            &cert("FAIL", "N/A", "PASS"),
            needs,
        )
        .expect("verified FAIL must refuse");
        assert!(msg.contains("\"m:8b\""), "{msg}");
        assert!(msg.contains("tool calling"), "{msg}");
        assert!(msg.contains("2026-10-03"), "{msg}");
        assert!(msg.contains("blazar model-doctor m:8b"), "{msg}");

        // N/A and absent probes never refuse (a gap is not a verdict).
        assert!(
            capability_refusal(
                "m",
                "b11370-cuda",
                Some("llamacpp"),
                Some("llamacpp"),
                0,
                &cert("N/A", "-", "PASS"),
                needs,
            )
            .is_none()
        );
        assert!(
            capability_refusal(
                "m",
                "b11370-cuda",
                Some("llamacpp"),
                Some("llamacpp"),
                0,
                &serde_json::json!({"caps": {}}),
                needs,
            )
            .is_none()
        );

        // FAIL on an UNneeded probe stays silent.
        let vision_needs = CapabilityNeeds {
            vision: true,
            ..CapabilityNeeds::default()
        };
        assert!(
            capability_refusal(
                "m",
                "b11370-cuda",
                Some("llamacpp"),
                Some("llamacpp"),
                0,
                &cert("FAIL", "PASS", "PASS"),
                vision_needs,
            )
            .is_none()
        );

        // Kind mismatch (lane switched since certification) fails open.
        assert!(
            capability_refusal(
                "m",
                "b11370-cuda",
                Some("llamacpp"),
                Some("mistralrs"),
                0,
                &cert("FAIL", "FAIL", "FAIL"),
                CapabilityNeeds {
                    tools: true,
                    vision: true,
                    json_mode: true
                },
            )
            .is_none()
        );

        // Retired certificate engine (kind unresolvable) fails open —
        // the engines row is gone, so the verdict cannot be matched to
        // a serving lane.
        assert!(
            capability_refusal(
                "m",
                "deleted-build",
                None,
                Some("llamacpp"),
                0,
                &cert("FAIL", "FAIL", "FAIL"),
                CapabilityNeeds {
                    tools: true,
                    vision: true,
                    json_mode: true
                },
            )
            .is_none()
        );

        // json_mode gates on the json probe.
        let json_needs = CapabilityNeeds {
            json_mode: true,
            ..CapabilityNeeds::default()
        };
        assert!(
            capability_refusal(
                "m",
                "b11370-cuda",
                Some("llamacpp"),
                Some("llamacpp"),
                0,
                &cert("PASS", "PASS", "FAIL"),
                json_needs,
            )
            .is_some()
        );
    }

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
