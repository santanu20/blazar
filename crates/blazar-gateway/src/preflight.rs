//! Load preflights: named, fail-fast checks that turn silent swap-death
//! and VRAM overcommit into teaching errors. Pure math - no side effects.

use axum::response::IntoResponse as _;
use blazar_core::GgufMeta;

/// Flat per-image token estimate (mmproj clip ~ pixels/750; typical
/// 512-1024px images land ~400-1400 tokens). F82: image data used to
/// walk as raw base64 bytes/4 into the estimator (a 1 MiB image
/// estimated ~262K tokens and false-tripped the 90% gate), while the
/// exact path counted multimodal arrays as empty text (~0 tokens and a
/// silent bypass). Both directions now carry this flat figure.
const IMAGE_TOKEN_EST: u64 = 1500;

/// Character classes of the token estimate, indexed like
/// [`CLASS_PRIORS`].
const CLASS_COUNT: usize = 10;
const HEX: usize = 0;
const ALNUM: usize = 1;
const SPACE: usize = 2;
const PUNCT: usize = 3;
const LATIN2: usize = 4;
const RTL2: usize = 5;
const CJK: usize = 6;
const HANGUL: usize = 7;
const THREE_BYTE: usize = 8;
const FOUR_BYTE: usize = 9;

/// Tokens per byte of each class, as `[low, high]` thousandths. Fitted
/// (2026-11-24) so that every row of a live calibration sits inside its
/// bracket - eleven text classes tokenized by a Qwen2.5 and a Mistral-7B
/// tokenizer through the engine's `/tokenize`
/// (`tests/token_bounds.rs` holds the rows) - then widened where only two
/// vocabularies were measured. A flat bytes/4 sat OUTSIDE the truth for
/// four of the eleven classes on the Qwen tokenizer (hex blobs 3.5x more
/// tokens, Devanagari 1.5x, emoji 1.3x, code 1.2x).
const CLASS_PRIORS: [[u64; 2]; CLASS_COUNT] = [
    [600, 1000], // hex runs
    [140, 400],  // ASCII letters and digits
    [50, 250],   // ASCII whitespace
    [100, 1300], // ASCII punctuation
    [150, 600],  // two-byte Latin, Greek, Cyrillic
    [100, 550],  // two-byte Arabic, Hebrew
    [120, 400],  // CJK ideographs and kana
    [150, 600],  // Hangul
    [100, 500],  // other three-byte scripts
    [200, 1000], // four-byte emoji and symbols
];

type ClassBytes = [u64; CLASS_COUNT];

/// Shortest run of hex digits that reads as a blob (a hash, a GUID)
/// rather than as a word or a plain number.
const HEX_RUN_MIN: usize = 8;

fn class_of(c: char) -> usize {
    let code = c as u32;
    match c {
        ' ' | '\t' | '\n' | '\r' | '\u{c}' => SPACE,
        c if c.is_ascii_alphanumeric() => ALNUM,
        c if c.is_ascii() => PUNCT,
        _ if code < 0x530 => LATIN2,
        _ if code < 0x800 => RTL2,
        _ if (0x1100..=0x11ff).contains(&code) || (0xac00..=0xd7af).contains(&code) => HANGUL,
        _ if (0x2e80..=0x9fff).contains(&code)
            || (0xf900..=0xfaff).contains(&code)
            || (0xff00..=0xffef).contains(&code) =>
        {
            CJK
        }
        _ if code < 0x10000 => THREE_BYTE,
        _ => FOUR_BYTE,
    }
}

/// Byte count of `text` per character class. Runs of hex digits that
/// carry a hex letter count as a blob; plain digit runs stay ordinary
/// digits (tokenizers group those in twos and threes, far below the
/// blob rate).
fn count_text(text: &str, bytes: &mut ClassBytes) {
    let raw = text.as_bytes();
    let mut i = 0;
    while i < raw.len() {
        let b = raw[i];
        if !b.is_ascii() {
            // `i` is a char boundary: every step below advances whole chars.
            let c = text[i..].chars().next().unwrap_or('\u{fffd}');
            bytes[class_of(c)] += c.len_utf8() as u64;
            i += c.len_utf8();
        } else if b.is_ascii_hexdigit() {
            let start = i;
            while i < raw.len() && raw[i].is_ascii_hexdigit() {
                i += 1;
            }
            let run = &raw[start..i];
            let blob = run.len() >= HEX_RUN_MIN
                && run
                    .iter()
                    .any(|d| d.is_ascii_hexdigit() && !d.is_ascii_digit());
            bytes[if blob { HEX } else { ALNUM }] += run.len() as u64;
        } else {
            bytes[if matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0c) {
                SPACE
            } else {
                PUNCT
            }] += 1;
            i += 1;
        }
    }
}

/// `[low, high]` bracket on the token count of `text`, from the class
/// priors: the exact count of every calibrated tokenizer lies inside.
#[must_use]
pub fn token_bounds(text: &str) -> (u64, u64) {
    let mut bytes = [0u64; CLASS_COUNT];
    count_text(text, &mut bytes);
    finish(&bytes, 0)
}

fn finish(bytes: &ClassBytes, images: u64) -> (u64, u64) {
    let low_milli = bytes
        .iter()
        .zip(CLASS_PRIORS)
        .map(|(count, [low, _])| count * low)
        .sum::<u64>();
    let high_milli = bytes
        .iter()
        .zip(CLASS_PRIORS)
        .map(|(count, [_, high])| count * high)
        .sum::<u64>();
    let flat = images * IMAGE_TOKEN_EST;
    (low_milli / 1000 + flat, high_milli.div_ceil(1000) + flat)
}

fn walk_body(v: &serde_json::Value, bytes: &mut ClassBytes, images: &mut u64) {
    match v {
        serde_json::Value::String(s) => count_text(s, bytes),
        serde_json::Value::Array(a) => a.iter().for_each(|x| walk_body(x, bytes, images)),
        serde_json::Value::Object(o) => {
            if o.contains_key("image_url") {
                *images += 1;
                return;
            }
            o.iter().for_each(|(k, x)| {
                if k == "images"
                    && x.as_array()
                        .is_some_and(|a| a.iter().all(|i| i.as_str().is_some()))
                {
                    // Ollama multimodal dialect: sibling `images` arrays
                    // carry base64 payloads - walking them as raw text
                    // re-creates F82's ~262K-token false trip (a 1 MiB
                    // image must estimate flat, like image_url objects).
                    *images += x.as_array().map_or(0, Vec::len) as u64;
                } else {
                    walk_body(x, bytes, images);
                }
            });
        }
        _ => {}
    }
}

/// Token bracket for a chat-family body: every string in it (messages,
/// system, tools) by character class, plus the flat per-image figure.
#[must_use]
pub fn prompt_token_bounds(body: &serde_json::Value) -> (u64, u64) {
    let mut bytes = [0u64; CLASS_COUNT];
    let mut images = 0u64;
    walk_body(body, &mut bytes, &mut images);
    finish(&bytes, images)
}

/// Context bound for prompt-fit admission: the LIVE limit of a resident
/// instance when one exists (auto-fit or explicit slots may have traded
/// the configured depth for width, e.g. 16K configured, 4x4096 compiled,
/// and a unified-KV child widens a single prompt's limit back to the
/// whole pool, see `Supervisor::prompt_ctx`), the configured effective
/// ctx otherwise (cold-spawn estimate). Rows whose profile has not
/// compiled yet (ctx == 0) fall back to the config estimate.
pub fn admission_ctx(state: &crate::state::AppState, model: &str) -> u32 {
    state
        .sup
        .prompt_ctx(model)
        .map_or_else(|| state.cfg().effective_ctx(model), u32::from)
}

fn prompt_text_and_images(body: &serde_json::Value) -> (String, u64) {
    body.pointer("/messages")
        .and_then(|m| m.as_array())
        .map(|a| {
            let mut parts: Vec<String> = Vec::new();
            let mut images = 0u64;
            for m in a {
                // Ollama multimodal dialect: base64 vision parts ride a
                // sibling `images` array on the message (never inside
                // `content`) - each non-empty entry is one image.
                images += m.get("images").and_then(|v| v.as_array()).map_or(0, |arr| {
                    arr.iter()
                        .filter(|i| i.as_str().is_some_and(|s| !s.is_empty()))
                        .count() as u64
                });
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
            body.get("prompt").and_then(|p| p.as_str()).map(|p| {
                // Generate dialect: vision parts ride a top-level
                // `images` array (proxy's body_needs_vision sees the
                // same shape) - count them, never the base64 text.
                let imgs = body
                    .get("images")
                    .and_then(|v| v.as_array())
                    .map_or(0, |arr| {
                        arr.iter()
                            .filter(|i| i.as_str().is_some_and(|s| !s.is_empty()))
                            .count() as u64
                    });
                (p.to_string(), imgs)
            })
        })
        .unwrap_or_default()
}

/// How a prompt-fit verdict was reached, in metric-label order: the
/// bracket cleared the limit, the bracket straddled it with no child
/// up to count, a running child's tokenizer counted it and it fits,
/// that count refused it, and the lower bound alone refused it.
const VERDICT_LABELS: [&str; 5] = [
    "fits",
    "uncertain",
    "exact_fit",
    "exact_refuse",
    "estimated_refuse",
];
const VERDICT_FITS: usize = 0;
const VERDICT_UNCERTAIN: usize = 1;
const VERDICT_EXACT_FIT: usize = 2;
const VERDICT_EXACT_REFUSE: usize = 3;
const VERDICT_ESTIMATED_REFUSE: usize = 4;
static VERDICTS: [std::sync::atomic::AtomicU64; VERDICT_LABELS.len()] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

fn note_verdict(verdict: usize) {
    VERDICTS[verdict].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// `/metrics` text for the verdict counters: every label, zeros
/// included, so a dashboard keeps a stable series.
#[must_use]
pub fn verdict_metrics() -> String {
    let mut text = String::from(
        "# HELP blazar_prompt_preflight_verdicts_total Prompt-fit verdicts, by how they were \
         reached\n# TYPE blazar_prompt_preflight_verdicts_total counter\n",
    );
    for (label, count) in VERDICT_LABELS.iter().zip(VERDICTS.iter()) {
        // Written straight into the buffer: no throwaway `String` per label.
        let _ = std::fmt::Write::write_fmt(
            &mut text,
            format_args!(
                "blazar_prompt_preflight_verdicts_total{{verdict=\"{label}\"}} {count}\n",
                count = count.load(std::sync::atomic::Ordering::Relaxed)
            ),
        );
    }
    text
}

/// Prompt-fit admission (K1): refuse requests that cannot fit the
/// effective context BEFORE the engine rejects or truncates them (the
/// sentinel's most common post-hoc detection, moved to pre-hoc).
/// Estimate first (hot path, ~free); the exact `/tokenize` count from a
/// RUNNING child only when the estimate's bracket reaches 90% of the
/// limit. Returns `Ok(true)` when that bracket straddles the limit and
/// no child is up to count exactly: the caller then re-checks through
/// [`recheck_prompt_fits`] once the model is loaded.
/// F82: multimodal bodies - collect text fields AND count image parts
/// (arrays used to tokenize as empty text and bypass the fit check
/// entirely). `/api/generate` bodies carry the prompt under `prompt`
/// (F13 - the exact count must not silently see "").
pub async fn enforce_prompt_fits(
    state: &crate::state::AppState,
    model: &str,
    body: &serde_json::Value,
    effective_ctx: u32,
) -> Result<bool, Box<axum::response::Response>> {
    if !state.cfg().prompt_preflight || effective_ctx == 0 {
        return Ok(false);
    }
    let ctx = u64::from(effective_ctx);
    let (low, high) = prompt_token_bounds(body);
    // Cheap path: even the upper bound sits under 90% of ctx.
    if high.saturating_mul(10) < ctx * 9 {
        note_verdict(VERDICT_FITS);
        return Ok(false);
    }
    // Near or past the limit: a running child gives the exact count.
    if let Some(engine) = crate::tokenize::resident(state, model)
        && let Some(verdict) = exact_verdict(state, model, &engine, body, ctx).await
    {
        note_verdict(if verdict.is_ok() {
            VERDICT_EXACT_FIT
        } else {
            VERDICT_EXACT_REFUSE
        });
        return verdict.map(|()| false);
    }
    // No child, or a lane that cannot count: only a prompt that cannot
    // fit on any reading is refused here - never pay a model load for it.
    if low >= ctx {
        note_verdict(VERDICT_ESTIMATED_REFUSE);
        return Err(Box::new(
            prompt_too_long(state, model, low, "estimated", ctx).into_response(),
        ));
    }
    note_verdict(VERDICT_UNCERTAIN);
    Ok(true)
}

/// Second half of prompt-fit admission for the cold path: `engine` is
/// the child a request with a straddling estimate just saw started. The
/// live tokenizer is available now, so a prompt over the limit is
/// refused before it burns a prefill. Lanes without a tokenize route
/// and failed calls pass: the engine reports the overflow itself.
pub async fn recheck_prompt_fits(
    state: &crate::state::AppState,
    model: &str,
    body: &serde_json::Value,
    engine: &blazar_runtime::EngineRef,
) -> Result<(), Box<axum::response::Response>> {
    if !state.cfg().prompt_preflight {
        return Ok(());
    }
    let ctx = u64::from(admission_ctx(state, model));
    if ctx == 0 {
        return Ok(());
    }
    let (_, high) = prompt_token_bounds(body);
    if high.saturating_mul(10) < ctx * 9 {
        return Ok(());
    }
    match exact_verdict(state, model, engine, body, ctx).await {
        Some(Err(response)) => {
            note_verdict(VERDICT_EXACT_REFUSE);
            Err(response)
        }
        Some(Ok(())) => {
            note_verdict(VERDICT_EXACT_FIT);
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Exact verdict of a running child: `Some(Err(response))` = over the
/// limit, `Some(Ok(()))` = fits, `None` = the lane cannot count.
async fn exact_verdict(
    state: &crate::state::AppState,
    model: &str,
    engine: &blazar_runtime::EngineRef,
    body: &serde_json::Value,
    ctx: u64,
) -> Option<Result<(), Box<axum::response::Response>>> {
    let (text, images) = prompt_text_and_images(body);
    let text_tokens = match crate::tokenize::count_prompt(state, engine, &text).await {
        Ok(count) => count,
        Err(error) => {
            tracing::debug!(target: "blazar::preflight", model, "exact token count unavailable: {error}");
            return None;
        }
    };
    // Tool definitions ride into the prompt as text the engine
    // serializes: their own bracket, not a second engine call. Only the
    // LOWER bound refuses - a schema's upper bound must never reject a
    // request the engine would have served.
    let (tools_low, _) = body
        .get("tools")
        .map_or((0, 0), |tools| token_bounds(&tools.to_string()));
    let tokens = text_tokens + tools_low + images * IMAGE_TOKEN_EST;
    Some(if tokens >= ctx {
        Err(Box::new(
            prompt_too_long(state, model, tokens, "exact", ctx).into_response(),
        ))
    } else {
        Ok(())
    })
}

/// The refusal: the count, the real limit, and what the engine would
/// have done with the prompt.
fn prompt_too_long(
    state: &crate::state::AppState,
    model: &str,
    tokens: u64,
    how: &str,
    ctx: u64,
) -> axum::response::Response {
    let engine_side = if state.cfg().context_shift {
        "context shift is on, so the engine would silently drop the oldest tokens to make room"
    } else {
        "context shift is off, so the engine would reject it (nothing is truncated silently)"
    };
    crate::proxy::openai_error_code(
        crate::error_codes::BlazarCode::PromptTooLong,
        &format!(
            "prompt {tokens} tokens ({how}) exceeds the {ctx}-token context for {model:?} - the \
             engine cannot serve it as sent: {engine_side}. Shorten the prompt, raise ctx \
             (config/[model_overrides] ctx or X-Blazar-Num-Ctx), or set prompt_preflight = false \
             to let the engine answer it"
        ),
    )
    .into_response()
}

/// f16 KV cache size in MiB for a target ctx. 2 (K+V) * layers *
/// `kv_heads` * `head_dim` * ctx * 2 bytes. Conservative: no quantized-KV
/// credit (the engine may still fit it via the cache-type ladder).
#[must_use]
pub fn kv_f16_mib(meta: &GgufMeta, ctx: u64) -> u64 {
    blazar_core::coreside::kv_f16_mib(meta, ctx)
}

/// What a request strictly asks of the model, extracted from its body -
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

/// Teaching refusal for vision requests against a model with no
/// projector sidecar. Without this the request reaches the lane and
/// dies inside the engine with a buried 500 hint; the store row
/// already knows the truth, so admission teaches instead.
#[must_use]
pub fn vision_missing_refusal(name: &str) -> String {
    format!(
        "\"{name}\" has no projector sidecar attached - vision requests cannot be served. \
         Attach one while the model is stopped: `blazar mmproj {name} <mmproj.gguf path>` \
         (the next spawn picks it up). Plain text chat works unchanged."
    )
}

/// Certificate gate (pure): `Some(teaching message)` when a needed
/// capability is VERIFIED-FAILED. Absent probes, `N/A` verdicts, and
/// unknown-or-mismatched engine kinds all pass - a stale or partial
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
{verified}, engine {cert_engine_tag}) - the request would burn a turn on \
garbage output. Plain chat still works; for {what} pick a sibling whose \
certificate shows {probe} = PASS (`blazar scorecard {model}` lists them), \
or re-verify after an engine change: `blazar model-doctor {model}`",
            ));
        }
    }
    None
}

/// The prober is the one actor the certificate gate must never block:
/// a model whose stored verdict is FAIL would otherwise 400 the very
/// probe sent to re-measure it - a self-heal deadlock (the probe
/// receipt would record the gate's refusal, not the model's behavior).
/// `model_doctor` tags its in-process probe calls with this header; both
/// gate call sites stand down for it.
pub const PROBE_HEADER: &str = "x-blazar-model-doctor";

#[must_use]
pub fn is_capability_probe(headers: &axum::http::HeaderMap) -> bool {
    headers.contains_key(PROBE_HEADER)
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
    let cert = state.read_store_or_absent("models.caps", |s| s.get_model_caps_dated(model))?;
    let caps: serde_json::Value = serde_json::from_str(&cert.2).ok()?;
    // Two SEPARATE store borrows: routed_kind_for takes the store mutex
    // itself, so nesting it inside a with_store closure would deadlock
    // the (non-reentrant) mutex and freeze every store-reading request.
    let cert_kind = state.read_store_or_absent("engines.kind_of_tag", |s| {
        s.engine_kind_of_tag(&cert.0).map(Some)
    })?;
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
    #[test]
    #[allow(non_snake_case)]
    fn unit__verdict_metrics__renders_every_verdict_label() {
        note_verdict(VERDICT_UNCERTAIN);
        let text = verdict_metrics();
        for label in VERDICT_LABELS {
            let prefix = format!("blazar_prompt_preflight_verdicts_total{{verdict=\"{label}\"}} ");
            let line = text
                .lines()
                .find(|line| line.starts_with(&prefix))
                .unwrap_or_else(|| panic!("no series for verdict {label}:\n{text}"));
            let value = line.rsplit(' ').next().unwrap_or_default();
            value
                .parse::<u64>()
                .unwrap_or_else(|e| panic!("{line}: {e}"));
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__prompt_token_bounds__ollama_images_estimate_flat_not_base64() {
        // Ollama dialect: a 1 MiB base64 image inside messages[].images
        // must estimate ~1 image, not ~262K tokens of raw base64/4 (the
        // F82 false-trip, reborn on the sibling-array shape).
        let big = "x".repeat(1024 * 1024);
        let body = serde_json::json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "what is this?", "images": [big]}
            ]
        });
        let (low, high) = prompt_token_bounds(&body);
        assert!(
            high < 10_000,
            "1 image + tiny text must not estimate as raw base64: {high}"
        );
        assert!(low >= 1_500, "the per-image figure is present: {low}");
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__prompt_text_and_images__counts_ollama_shapes_both_dialects() {
        // Chat dialect: sibling messages[].images arrays are vision
        // parts (OpenAI content blocks keep their own arm).
        let chat = serde_json::json!({
            "messages": [
                {"role": "user", "content": "hi", "images": ["b64", "b64", ""]}
            ]
        });
        let (text, images) = prompt_text_and_images(&chat);
        assert_eq!(text, "hi");
        assert_eq!(images, 2, "two non-empty base64 entries, empty ignored");
        // Generate dialect: top-level `images` beside `prompt`.
        let generate = serde_json::json!({"prompt": "describe", "images": ["b64"]});
        let (text, images) = prompt_text_and_images(&generate);
        assert_eq!(text, "describe");
        assert_eq!(images, 1);
        // Mixed non-string `images` arrays (not the ollama shape) are
        // not counted as images anywhere.
        let mixed = serde_json::json!({"prompt": "p", "images": [1, 2]});
        let (_, images) = prompt_text_and_images(&mixed);
        assert_eq!(images, 0);
    }

    use super::*;

    #[test]
    fn unit__is_capability_probe__header_marks_the_prober() {
        // The self-heal loop depends on this: a stored FAIL verdict must
        // never block the probe that re-measures it.
        let mut headers = axum::http::HeaderMap::new();
        assert!(!is_capability_probe(&headers));
        headers.insert(PROBE_HEADER, axum::http::HeaderValue::from_static("1"));
        assert!(is_capability_probe(&headers));
    }

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
        // for nothing - a false positive here would refuse legit chat.
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
    #[allow(non_snake_case)]
    fn unit__vision_missing_refusal__teaches_the_fix_command() {
        let msg = vision_missing_refusal("qwen:8b");
        assert!(msg.contains("\"qwen:8b\""), "{msg}");
        assert!(msg.contains("no projector sidecar attached"), "{msg}");
        // The exact attach command, including the stop-first contract.
        assert!(msg.contains("blazar mmproj qwen:8b"), "{msg}");
        assert!(msg.contains("while the model is stopped"), "{msg}");
        assert!(msg.contains("Plain text chat works unchanged"), "{msg}");
    }

    #[test]
    #[allow(clippy::too_many_lines)] // one assert block per refusal rule
    fn unit__capability_refusal__fresh_matching_fail_only() {
        let needs = CapabilityNeeds {
            tools: true,
            ..CapabilityNeeds::default()
        };
        // FAIL probe + matching need -> teaching refusal naming the
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

        // Retired certificate engine (kind unresolvable) fails open -
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

    #[test]
    #[allow(non_snake_case)]
    fn unit__token_bounds__each_class_carries_its_own_prior() {
        // Pure single-class strings: the bracket is that class's prior
        // times its byte count.
        assert_eq!(token_bounds("aaaa"), (0, 2));
        assert_eq!(token_bounds("abcd ef01"), (1, 4));
        assert_eq!(token_bounds(""), (0, 0));
        assert_eq!(token_bounds("\u{65e5}\u{672c}"), (0, 3));
        assert_eq!(token_bounds("😀😀"), (1, 8));
        // A hex run with a letter is a blob: near one token per character.
        assert_eq!(token_bounds("deadbeef00"), (6, 10));
        // A plain digit run is not: low like any other digits.
        assert_eq!(token_bounds("1234567890"), (1, 4));
        assert_eq!(token_bounds("1234567"), (0, 3));
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__token_bounds__hex_blobs_no_longer_under_count_like_a_flat_ratio() {
        // Priors only - the measured tokenizer rows live in
        // tests/token_bounds.rs. A hash-like blob starts well above the
        // bytes/4 figure that used to wave it through.
        let blob = "b4924f81e0c37a5d".repeat(7500);
        let (low, high) = token_bounds(&blob);
        assert_eq!((low, high), (72_000, 120_000));
        assert!(low > (blob.len() as u64) / 4);
        // The same text inside a chat body keeps that bracket (the
        // envelope's own strings only add to it).
        let body = serde_json::json!({"messages": [{"role": "user", "content": blob}]});
        let (body_low, body_high) = prompt_token_bounds(&body);
        assert!(body_low >= low && body_high >= high);
        // Images stay a flat figure on top.
        let vision = serde_json::json!({
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}
            ]}]
        });
        let (low, high) = prompt_token_bounds(&vision);
        assert!(low >= 1_500 && high < 2_000, "({low}, {high})");
    }

    fn meta(layers: u64, heads: u64, kv: u64, emb: u64) -> GgufMeta {
        GgufMeta {
            base_model_id: None,
            file_type: None,
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
