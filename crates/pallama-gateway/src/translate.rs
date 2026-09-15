//! Pure ollama <-> `OpenAI` translation functions. No I/O — every function
//! is table-testable, including stream-chunk splitting across boundaries.

use serde_json::{json, Value};

/// llama-server's default sampler chain (verified live via /props on
/// b10896). Used when `adaptive_p` is enabled without an explicit
/// `samplers` list — the engine then appends the adaptive sampler itself.
const DEFAULT_SAMPLER_CHAIN: &[&str] = &[
    "penalties",
    "dry",
    "top_n_sigma",
    "top_k",
    "typ_p",
    "top_p",
    "min_p",
    "xtc",
    "temperature",
];

/// Ollama sampling options -> `OpenAI` request fields. Unknown option keys
/// are returned so the caller can 400 listing them (fail fast, complaint
/// #15's sibling: never silently drop what the user asked for).
pub fn apply_ollama_options(openai_req: &mut Value, options: &Value) -> Vec<String> {
    let mut unknown = Vec::new();
    // Recorded during the pass, applied after it: with serde_json's
    // alphabetical object order "adaptive_p" runs before "samplers", so
    // an explicit samplers list would clobber the merged chain.
    let mut adaptive_p = false;
    let Some(map) = options.as_object() else {
        return unknown;
    };
    for (k, v) in map {
        match k.as_str() {
            "temperature" => openai_req["temperature"] = v.clone(),
            "top_p" => openai_req["top_p"] = v.clone(),
            "top_k" => {
                // No direct OpenAI equiv; llama-server accepts top_k
                // natively, keep it as an extension field.
                openai_req["top_k"] = v.clone();
            }
            "min_p" => openai_req["min_p"] = v.clone(),
            "seed" => openai_req["seed"] = v.clone(),
            "num_predict" => openai_req["max_tokens"] = v.clone(),
            "stop" => openai_req["stop"] = v.clone(),
            "repeat_penalty" => openai_req["repeat_penalty"] = v.clone(),
            "repeat_last_n" => openai_req["repeat_last_n"] = v.clone(),
            "presence_penalty" => openai_req["presence_penalty"] = v.clone(),
            "frequency_penalty" => openai_req["frequency_penalty"] = v.clone(),
            // llama-server native sampler fields (verified against the
            // b10896 request schema): pass through 1:1 like top_k/min_p.
            "xtc_probability" => openai_req["xtc_probability"] = v.clone(),
            "xtc_threshold" => openai_req["xtc_threshold"] = v.clone(),
            "top_n_sigma" => openai_req["top_n_sigma"] = v.clone(),
            "logit_bias" => openai_req["logit_bias"] = v.clone(),
            "dry_multiplier" => openai_req["dry_multiplier"] = v.clone(),
            "dry_base" => openai_req["dry_base"] = v.clone(),
            "dry_allowed_length" => openai_req["dry_allowed_length"] = v.clone(),
            "dry_penalty_last_n" => openai_req["dry_penalty_last_n"] = v.clone(),
            "dry_sequence_breakers" => {
                // Upstream asserts a non-empty array of strings; a bad
                // shape must 400 here, not die in the child.
                let ok = v
                    .as_array()
                    .is_some_and(|a| !a.is_empty() && a.iter().all(Value::is_string));
                if ok {
                    openai_req["dry_sequence_breakers"] = v.clone();
                } else {
                    unknown.push(
                        "dry_sequence_breakers (must be a non-empty array of strings)".into(),
                    );
                }
            }
            "mirostat" => openai_req["mirostat"] = v.clone(),
            "mirostat_tau" => openai_req["mirostat_tau"] = v.clone(),
            "mirostat_eta" => openai_req["mirostat_eta"] = v.clone(),
            "dynatemp_range" => openai_req["dynatemp_range"] = v.clone(),
            "dynatemp_exponent" => openai_req["dynatemp_exponent"] = v.clone(),
            "adaptive_target" => openai_req["adaptive_target"] = v.clone(),
            "adaptive_decay" => openai_req["adaptive_decay"] = v.clone(),
            "samplers" => openai_req["samplers"] = v.clone(),
            "adaptive_p" => {
                // No bool field upstream: adaptive_p activates by joining
                // the samplers chain (engine appends it at chain end).
                // Applied post-loop — see the flag comment above.
                if v.as_bool() == Some(true) {
                    adaptive_p = true;
                }
            }
            "num_ctx" | "num_batch" | "num_gpu" | "num_thread" | "num_keep" | "numa" => {
                // Runner options: num_ctx is handled by the caller
                // (instance restart); the rest are accepted-and-ignored
                // only for num_ctx's siblings — explicitly listed, not
                // silently swallowed.
                if k != "num_ctx" {
                    unknown.push(format!(
                        "{k} (not settable per-request in pallama; use config)"
                    ));
                }
            }
            _ => unknown.push(k.clone()),
        }
    }
    if adaptive_p {
        let mut chain = openai_req
            .get("samplers")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_else(|| DEFAULT_SAMPLER_CHAIN.iter().map(|s| json!(s)).collect());
        if !chain.iter().any(|s| s == "adaptive_p") {
            chain.push(json!("adaptive_p"));
        }
        openai_req["samplers"] = Value::Array(chain);
    }
    unknown
}

/// ollama `format` -> `OpenAI` `response_format`.
#[must_use]
pub fn translate_format(format: &Value) -> Option<Value> {
    match format {
        Value::String(s) if s == "json" => Some(json!({"type": "json_object"})),
        // Structured schema: ollama accepts a raw JSON schema in format.
        Value::Object(_) => Some(json!({"type": "json_schema", "json_schema": {"schema": format}})),
        _ => None,
    }
}

/// Standard-alphabet base64 symbol -> 6-bit value.
fn b64_val(c: u8) -> Option<u8> {
    match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

/// Decode the first `n` bytes of a standard-alphabet base64 string —
/// enough for magic-byte sniffing without a full decoder (or a new
/// dependency). Short/garbage input yields fewer bytes than asked.
fn b64_prefix_bytes(b64: &str, n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    let bytes = b64.as_bytes();
    for g in 0..bytes.len() / 4 {
        let Some(v) = b64_val(bytes[4 * g])
            .zip(b64_val(bytes[4 * g + 1]))
            .and_then(|(a, b)| {
                b64_val(bytes[4 * g + 2])
                    .zip(b64_val(bytes[4 * g + 3]))
                    .map(|(c, d)| (a, b, c, d))
            })
        else {
            break;
        };
        out.extend_from_slice(&[v.0 << 2 | v.1 >> 4, v.1 << 4 | v.2 >> 2, v.2 << 6 | v.3]);
        if out.len() >= n {
            break;
        }
    }
    out.truncate(n);
    out
}

/// Sniff the image MIME type from decoded magic bytes.
fn sniff_image_mime(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG") {
        Some("image/png")
    } else if bytes.starts_with(b"\xff\xd8\xff") {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF8") {
        Some("image/gif")
    } else if bytes.starts_with(b"RIFF") && bytes.len() >= 12 && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else {
        None
    }
}

/// ollama base64 image -> `data:` URL for `OpenAI` `image_url` parts.
/// 12 decoded bytes cover every supported magic; an unrecognized (or
/// undecodable) prefix fails fast instead of guessing a MIME (H1).
fn image_data_url(b64: &str) -> Result<String, String> {
    let head = b64_prefix_bytes(b64, 12);
    let mime = sniff_image_mime(&head)
        .ok_or_else(|| "unsupported image format (supported: png, jpeg, gif, webp)".to_string())?;
    Ok(format!("data:{mime};base64,{b64}"))
}

/// ollama messages -> `OpenAI` messages: any message carrying ollama-style
/// `images: [<base64>]` becomes multimodal content parts with `image_url`
/// data-URLs; messages without images pass through verbatim.
fn translate_message_images(messages: &Value) -> Result<Value, String> {
    let Some(list) = messages.as_array() else {
        return Ok(messages.clone());
    };
    let mut out = Vec::with_capacity(list.len());
    for msg in list {
        let Some(images) = msg.get("images").filter(|i| i.is_array()) else {
            out.push(msg.clone());
            continue;
        };
        let mut parts = Vec::new();
        if let Some(text) = msg.get("content").and_then(Value::as_str) {
            if !text.is_empty() {
                parts.push(json!({"type": "text", "text": text}));
            }
        }
        if let Some(arr) = images.as_array() {
            for img in arr {
                let b64 = img
                    .as_str()
                    .ok_or("message images must be base64 strings")?;
                parts
                    .push(json!({"type": "image_url", "image_url": {"url": image_data_url(b64)?}}));
            }
        }
        let mut m = msg.clone();
        if let Some(obj) = m.as_object_mut() {
            obj.remove("images");
        }
        m["content"] = Value::Array(parts);
        out.push(m);
    }
    Ok(Value::Array(out))
}

/// ollama assistant `tool_calls` entries may omit the `type` field
/// (ollama's own wire shape); llama-server rejects them with
/// "Missing tool call type". Fill `type: "function"` where absent so
/// multi-turn tool transcripts replay 1:1. Non-array messages and
/// entries that already carry a string `type` pass through verbatim.
fn normalize_tool_call_types(messages: &Value) -> Value {
    let Some(list) = messages.as_array() else {
        return messages.clone();
    };
    let mut out = Vec::with_capacity(list.len());
    for msg in list {
        let Some(calls) = msg.get("tool_calls").filter(|c| c.is_array()) else {
            out.push(msg.clone());
            continue;
        };
        let mut m = msg.clone();
        let mut fixed = Vec::with_capacity(calls.as_array().unwrap().len());
        for call in calls.as_array().unwrap() {
            let has_type = call
                .get("type")
                .is_some_and(|t| t.as_str().is_some_and(|s| !s.is_empty()));
            if has_type {
                fixed.push(call.clone());
            } else {
                let mut c = call.clone();
                if let Some(obj) = c.as_object_mut() {
                    // Insertion order: type first, then the function body.
                    // The stale (empty) type entry, if any, must not
                    // overwrite the filled one during the merge.
                    let old = std::mem::take(obj);
                    let mut entry = serde_json::Map::new();
                    entry.insert("type".into(), json!("function"));
                    for (k, v) in old {
                        if k != "type" {
                            entry.insert(k, v);
                        }
                    }
                    *obj = entry;
                }
                fixed.push(c);
            }
        }
        // ollama's dialect sends `arguments` as a PARSED object; the child
        // demands a JSON STRING. Clients that capture our ollama-shape
        // responses replay them verbatim, so stringify object arguments
        // on the way in.
        for call in &mut fixed {
            if let Some(args) = call
                .get_mut("function")
                .and_then(|f| f.get_mut("arguments"))
            {
                if args.is_object() {
                    *args = Value::String(args.to_string());
                }
            }
        }
        m["tool_calls"] = Value::Array(fixed);
        out.push(m);
    }
    Value::Array(out)
}

/// /api/chat request -> /v1/chat/completions body.
pub fn chat_to_openai(req: &Value) -> Result<(Value, Option<i64>), String> {
    let model = req["model"].as_str().unwrap_or_default().to_string();
    if model.is_empty() {
        return Err("missing field: model".into());
    }
    if !req["messages"].is_array() {
        return Err("missing field: messages".into());
    }
    let mut out = json!({
        "model": model,
        // ollama-style message images[] become multimodal content parts
        // (children ignore the raw field — this was silent vision loss),
        // and ollama tool_calls entries gain the `type` the child demands.
        "messages": normalize_tool_call_types(&translate_message_images(&req["messages"])?),
    });
    // ollama defaults stream=true; OpenAI defaults false — mirror the
    // caller's explicit choice only.
    if let Some(stream) = req["stream"].as_bool() {
        out["stream"] = json!(stream);
    }
    if let Some(tools) = req
        .get("tools")
        .filter(|t| t.is_array() && !t.as_array().unwrap().is_empty())
    {
        out["tools"] = tools.clone();
    }
    if let Some(rf) = translate_format(&req["format"]) {
        out["response_format"] = rf;
    }
    // Raw GBNF passthrough (R9): an explicit grammar is forwarded 1:1 —
    // the child skips schema conversion entirely for it.
    if let Some(g) = req.get("grammar").and_then(Value::as_str) {
        if !g.is_empty() {
            out["grammar"] = json!(g);
        }
    }
    // Confidence scoring (R2): ollama clients asking for logprobs get
    // them passed through 1:1 — the sentinel turns them into response
    // confidence in `why`. Absent = unchanged request.
    if let Some(lp) = req.get("logprobs").and_then(Value::as_bool) {
        out["logprobs"] = json!(lp);
        if let Some(n) = req.get("top_logprobs").and_then(Value::as_u64) {
            out["top_logprobs"] = json!(n);
        }
    }
    // User-supplied `chat_template_kwargs` forwards 1:1 (audit H1: the
    // field used to be silently dropped — power users coming from
    // llama-server/vllm/sglang lost their template switches with no
    // error). Copied BEFORE the think injection so explicit user keys
    // beat pallama's derived pair.
    if let Some(kw) = req.get("chat_template_kwargs").filter(|k| k.is_object()) {
        let mut merged = kw.clone();
        if let Some(dst) = out
            .get_mut("chat_template_kwargs")
            .and_then(Value::as_object_mut)
        {
            for (k, v) in dst.iter() {
                merged
                    .as_object_mut()
                    .expect("checked object")
                    .entry(k.clone())
                    .or_insert(v.clone());
            }
        }
        out["chat_template_kwargs"] = merged;
    }
    // ollama `think` toggle → template-level switch. No OpenAI
    // equivalent field exists; llama-server consumes
    // `chat_template_kwargs`, whose variable name varies by model
    // family (qwen3: `enable_thinking`, others: `thinking`) — set both;
    // a template only reads the var it knows, so the extra is inert.
    // Absent = template default (unchanged behavior). mistral.rs
    // children ignore unknown body fields; the toggle is llama-lane
    // effective and harmless elsewhere. Keys already present (copied
    // from the user's own chat_template_kwargs above) keep the user's
    // value — explicit beats derived.
    if let Some(think) = req.get("think").and_then(Value::as_bool) {
        if out.get("chat_template_kwargs").is_none() {
            out["chat_template_kwargs"] = json!({});
        }
        if let Some(kw) = out
            .get_mut("chat_template_kwargs")
            .and_then(Value::as_object_mut)
        {
            // Fill-only: a user-supplied key (copied above) keeps its
            // value — explicit beats derived.
            kw.entry("thinking".to_string()).or_insert(json!(think));
            kw.entry("enable_thinking".to_string())
                .or_insert(json!(think));
        }
    }
    let mut num_ctx = None;
    if let Some(opts) = req.get("options").filter(|o| o.is_object()) {
        if let Some(nc) = opts.get("num_ctx").and_then(Value::as_i64) {
            num_ctx = Some(nc);
        }
        let unknown = apply_ollama_options(&mut out, opts);
        if !unknown.is_empty() {
            return Err(format!("unsupported options: {}", unknown.join(", ")));
        }
    }
    Ok((out, num_ctx))
}

/// Child `timings` (llama-server non-stream shape, milliseconds) ->
/// ollama nanosecond duration. Returns `None` when the child omitted
/// the field so callers skip the key instead of emitting a lie.
// Durations are non-negative by construction; ns precision past the u64
// range (~584 years) cannot occur with real requests.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn timing_ns(openai: &Value, key: &str) -> Option<u64> {
    openai
        .pointer(&format!("/timings/{key}"))
        .and_then(Value::as_f64)
        .map(|ms| (ms * 1e6).max(0.0) as u64)
}

/// Merge child `timings` into an ollama response object: prompt/eval
/// durations plus an honest `total_duration` (prompt+eval, queue time
/// excluded — the child never saw it).
fn merge_timing_fields(v: &mut Value, openai: &Value) {
    let prompt_ns = timing_ns(openai, "prompt_ms");
    let eval_ns = timing_ns(openai, "predicted_ms");
    if let Some(p) = prompt_ns {
        v["prompt_eval_duration"] = json!(p);
    }
    if let Some(e) = eval_ns {
        v["eval_duration"] = json!(e);
    }
    if let (Some(p), Some(e)) = (prompt_ns, eval_ns) {
        v["total_duration"] = json!(p.saturating_add(e));
    }
}

/// Extract the cached prompt-token count from an `OpenAI` usage object
/// (`prompt_tokens_details` -> `cached_tokens`; llama-server emits it on
/// stream usage chunks and non-stream chat). 0 when absent — callers
/// treat absent as "cold, unmeasured".
#[must_use]
pub fn cached_prompt_tokens(usage: Option<&Value>) -> u64 {
    usage
        .and_then(|u| {
            u.pointer("/prompt_tokens_details/cached_tokens")
                .and_then(Value::as_u64)
        })
        .unwrap_or(0)
}

/// `OpenAI` non-stream chat response -> ollama `ChatResponse` (done:true with
/// counts from usage).
#[must_use]
pub fn openai_chat_to_ollama(model: &str, openai: &Value) -> Value {
    let choice = &openai["choices"][0];
    let message = &choice["message"];
    let mut msg = json!({"role": "assistant", "content": message["content"].clone()});
    if let Some(tc) = message.get("tool_calls").and_then(Value::as_array) {
        // Ollama dialect: `index` inside `function`, `arguments` as a
        // PARSED object (the child returns a JSON string). Clients replay
        // these verbatim, so the shape must match ollama's exactly.
        let calls = tc
            .iter()
            .enumerate()
            .map(|(i, call)| {
                let f = &call["function"];
                let raw = f["arguments"].as_str().unwrap_or("");
                let args =
                    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()));
                json!({
                    "id": call["id"].clone(),
                    "function": {"index": i, "name": f["name"].clone(), "arguments": args},
                })
            })
            .collect::<Vec<_>>();
        msg["tool_calls"] = Value::Array(calls);
    }
    if let Some(reasoning) = message.get("reasoning_content") {
        msg["thinking"] = reasoning.clone();
    }
    let mut v = json!({
        "model": model,
        "created_at": iso_now(),
        "message": msg,
        "done_reason": choice["finish_reason"].clone(),
        "done": true,
        "total_duration": 0,
        "prompt_eval_count": openai["usage"]["prompt_tokens"].clone(),
        "eval_count": openai["usage"]["completion_tokens"].clone(),
    });
    // Cache-hit transparency (A9): surface reused prompt tokens under the
    // same ollama-shaped key the gateway's /metrics uses.
    let cached = cached_prompt_tokens(openai.get("usage"));
    if cached > 0 {
        v["prompt_eval_cached_count"] = json!(cached);
    }
    // Confidence scoring (R2): map the child's token logprobs back into
    // ollama's native top-level array (field names align 1:1 — pure clone).
    if let Some(lp) = choice
        .get("logprobs")
        .and_then(|l| l.get("content"))
        .filter(|c| c.as_array().is_some_and(|a| !a.is_empty()))
    {
        v["logprobs"] = lp.clone();
    }
    merge_timing_fields(&mut v, openai);
    v
}

/// Accumulates `OpenAI` streaming `tool_call` fragments into complete calls.
#[derive(Default)]
pub struct ToolCallAccum {
    calls: Vec<Value>,
}

impl ToolCallAccum {
    /// Merge one delta's `tool_calls` array (keyed by `index`).
    fn absorb(&mut self, fragments: &[Value]) {
        for frag in fragments {
            let idx = frag
                .get("index")
                .and_then(Value::as_u64)
                .unwrap_or(0)
                .try_into()
                .unwrap_or(usize::MAX);
            while self.calls.len() <= idx {
                let i = self.calls.len();
                self.calls.push(json!({
                    "id": null,
                    "function": {"index": i, "name": null, "arguments": ""},
                }));
            }
            let slot = &mut self.calls[idx];
            if let Some(id) = frag.get("id").and_then(Value::as_str) {
                slot["id"] = Value::String(id.to_string());
            }
            if let Some(f) = frag.get("function") {
                if let Some(n) = f.get("name").and_then(Value::as_str) {
                    slot["function"]["name"] = Value::String(n.to_string());
                }
                if let Some(a) = f.get("arguments").and_then(Value::as_str) {
                    let joined = format!(
                        "{}{a}",
                        slot["function"]["arguments"].as_str().unwrap_or("")
                    );
                    slot["function"]["arguments"] = Value::String(joined);
                }
            }
        }
    }

    /// True while merged calls await their flush.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }

    /// Flush merged calls as a ready-to-write NDJSON line (with newline).
    #[must_use]
    pub fn flush_line(&mut self, model: &str) -> Option<String> {
        self.flush()
            .map(|calls| format!("{}\n", tool_calls_line(model, &calls)))
    }

    /// Drain merged calls as one ollama-dialect `tool_calls` array.
    /// Malformed accumulated arguments (broken model output) fall back to
    /// the raw string instead of failing the stream.
    fn flush(&mut self) -> Option<Value> {
        if self.calls.is_empty() {
            return None;
        }
        let calls = std::mem::take(&mut self.calls);
        let mut out = Vec::with_capacity(calls.len());
        for mut call in calls {
            let raw = call["function"]["arguments"]
                .as_str()
                .unwrap_or("")
                .to_string();
            call["function"]["arguments"] = match serde_json::from_str(&raw) {
                Ok(v) => v,
                Err(_) => Value::String(raw),
            };
            out.push(call);
        }
        Some(Value::Array(out))
    }
}

fn tool_calls_line(model: &str, calls: &Value) -> Value {
    json!({
        "model": model,
        "created_at": iso_now(),
        "message": {"role": "assistant", "tool_calls": calls},
        "done": false,
    })
}

/// One `OpenAI` SSE chunk -> zero or more ollama NDJSON lines.
///
/// Tool-call deltas are MERGED across chunks: ollama-native streaming
/// emits each call exactly once, complete, in ollama dialect
/// (`{id, function: {index, name, arguments: <parsed object>}}`), while
/// the child streams fragments (id+name first, argument shards after).
/// Clients built for the ollama wire (e.g. geokit's compare harness)
/// `extend()` fragments verbatim and replay them, which the child rejects
/// ("Missing tool call name") — merging at the edge restores the contract.
#[must_use]
pub fn openai_chunk_to_ollama(accum: &mut ToolCallAccum, model: &str, chunk: &Value) -> Vec<Value> {
    let mut out = Vec::new();
    if let Some(choices) = chunk["choices"].as_array() {
        for choice in choices {
            let delta = &choice["delta"];
            let has_content = delta
                .get("content")
                .is_some_and(|c| c.as_str().is_some_and(|s| !s.is_empty()));
            let has_thinking = delta
                .get("reasoning_content")
                .is_some_and(|c| c.as_str().is_some_and(|s| !s.is_empty()));
            let fragments = delta
                .get("tool_calls")
                .and_then(Value::as_array)
                .filter(|a| !a.is_empty());
            // A non-tool delta after fragments means the call is complete —
            // flush the merged calls as their own line first.
            if fragments.is_none() && !accum.is_empty() {
                if let Some(merged) = accum.flush() {
                    out.push(tool_calls_line(model, &merged));
                }
            }
            if let Some(frags) = fragments {
                accum.absorb(frags);
                continue;
            }
            if has_content || has_thinking {
                let mut msg = json!({"role": "assistant"});
                if let Some(c) = delta.get("content") {
                    if c.as_str().is_some_and(|s| !s.is_empty()) {
                        msg["content"] = c.clone();
                    }
                }
                if let Some(rc) = delta.get("reasoning_content") {
                    msg["thinking"] = rc.clone();
                }
                let mut line = json!({
                    "model": model,
                    "created_at": iso_now(),
                    "message": msg,
                    "done": false,
                });
                // R2: streaming logprobs ride the line top-level, same
                // ollama-native shape as the non-stream path.
                if let Some(lp) = choice
                    .get("logprobs")
                    .and_then(|l| l.get("content"))
                    .filter(|c| c.as_array().is_some_and(|a| !a.is_empty()))
                {
                    line["logprobs"] = lp.clone();
                }
                out.push(line);
            }
        }
    }
    out
}

/// Final ollama chunk from the usage/finish information. `eval_ns` /
/// `total_ns` are the gateway-measured decode window and full wall
/// (streaming children do not report timings); prompt duration is
/// omitted there rather than inflated with queue wait.
#[must_use]
pub fn ollama_final_chunk(
    model: &str,
    usage: Option<&Value>,
    finish: Option<&str>,
    eval_ns: Option<u64>,
    total_ns: Option<u64>,
) -> Value {
    let mut v = json!({
        "model": model,
        "created_at": iso_now(),
        "message": {"role": "assistant", "content": ""},
        "done_reason": finish.unwrap_or("stop"),
        "done": true,
        "total_duration": 0,
        "prompt_eval_count": usage.and_then(|u| u["prompt_tokens"].as_i64()).unwrap_or(0),
        "eval_count": usage.and_then(|u| u["completion_tokens"].as_i64()).unwrap_or(0),
    });
    // Cache-hit transparency (A9): stream usage chunks carry the same
    // prompt_tokens_details.cached_tokens as non-stream bodies.
    let cached = cached_prompt_tokens(usage);
    if cached > 0 {
        v["prompt_eval_cached_count"] = json!(cached);
    }
    if let Some(e) = eval_ns {
        v["eval_duration"] = json!(e);
    }
    if let Some(t) = total_ns {
        v["total_duration"] = json!(t);
    }
    v
}

/// Chunk-boundary-safe text accumulator for streaming decodes (F70):
/// buffers RAW BYTES and only lossily-decodes the region up to the last
/// complete `\n`, so a multi-byte UTF-8 char split across TCP chunk
/// boundaries never decodes into twin U+FFFD. Feed every chunk; push
/// the returned text into the lane's existing line/parsing buffer.
#[derive(Default)]
pub struct LineBuffer {
    buf: Vec<u8>,
}

impl LineBuffer {
    #[must_use]
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Decode everything up to and including the last complete newline;
    /// a trailing partial line (possibly a split UTF-8 char) stays
    /// buffered until the next chunk completes it.
    pub fn feed(&mut self, chunk: &[u8]) -> String {
        self.buf.extend_from_slice(chunk);
        let Some(nl) = self.buf.iter().rposition(|&b| b == b'\n') else {
            return String::new();
        };
        let complete: Vec<u8> = self.buf.drain(..=nl).collect();
        String::from_utf8_lossy(&complete).into_owned()
    }
}

/// Parse `data: {...}` SSE lines from a byte buffer; returns (events, rest).
/// Handles events split across chunk boundaries and the [DONE] sentinel.
#[must_use]
pub fn parse_sse(buf: &str) -> (Vec<Value>, bool /*done*/, usize /*consumed*/) {
    let mut events = Vec::new();
    let mut done = false;
    let mut lines = buf.split_inclusive('\n').peekable();
    let mut complete_line_count = 0;
    for line in lines.by_ref() {
        if !line.ends_with('\n') {
            break; // partial line stays in buffer
        }
        complete_line_count += 1;
        let trimmed = line.trim_end();
        if let Some(payload) = trimmed.strip_prefix("data: ") {
            if payload == "[DONE]" {
                done = true;
            } else if let Ok(v) = serde_json::from_str::<Value>(payload) {
                events.push(v);
            }
        }
    }
    let consumed: usize = buf
        .split_inclusive('\n')
        .take(complete_line_count)
        .map(str::len)
        .sum();
    (events, done, consumed)
}

/// /api/embeddings request -> /v1/embeddings body.
pub fn embeddings_to_openai(req: &Value) -> Result<Value, String> {
    let model = req["model"].as_str().unwrap_or_default();
    let prompt = req["prompt"].as_str().unwrap_or_default();
    if model.is_empty() {
        return Err("missing field: model".into());
    }
    Ok(json!({"model": model, "input": prompt}))
}

/// `OpenAI` /v1/embeddings response -> ollama embeddings response.
#[must_use]
pub fn openai_embeddings_to_ollama(model: &str, openai: &Value) -> Value {
    json!({
        "model": model,
        "embedding": openai["data"][0]["embedding"].clone(),
    })
}

/// /api/generate request -> /v1/chat/completions body (chat-bus unified
/// translation): `system` -> messages[0], `prompt` -> user text part,
/// `images` -> `image_url` data-URL parts, `think` ->
/// `chat_template_kwargs` — the same machinery as /api/chat, so the
/// engine's own template shapes the request exactly like ollama's
/// (templated) generate. `template`/`suffix` stay unsupported (the
/// engine owns the template) -> `Ok(None)` for the caller's teaching
/// 400. Unknown sampling options are an `Err` — the generate lane must
/// fail fast exactly like the chat lane, never silently drop what was
/// asked for.
pub fn generate_to_openai(req: &Value) -> Result<Option<Value>, String> {
    let templated = ["template", "suffix"]
        .iter()
        .any(|k| req.get(*k).is_some_and(|v| !v.is_null()));
    if templated {
        return Ok(None);
    }
    let model = req["model"].as_str().ok_or("missing field: model")?;
    let prompt = req["prompt"].as_str().ok_or("missing field: prompt")?;
    let mut messages = Vec::new();
    if let Some(system) = req.get("system").and_then(Value::as_str) {
        if !system.is_empty() {
            messages.push(json!({"role": "system", "content": system}));
        }
    }
    let mut user = json!({"role": "user", "content": prompt});
    if let Some(images) = req
        .get("images")
        .filter(|i| i.is_array() && !i.as_array().unwrap().is_empty())
    {
        let mut parts = vec![json!({"type": "text", "text": prompt})];
        for img in images.as_array().unwrap() {
            let b64 = img.as_str().ok_or("images must be base64 strings")?;
            parts.push(json!({"type": "image_url", "image_url": {"url": image_data_url(b64)?}}));
        }
        user["content"] = Value::Array(parts);
    }
    messages.push(user);
    let mut out = json!({"model": model, "messages": messages});
    if let Some(stream) = req["stream"].as_bool() {
        out["stream"] = json!(stream);
    }
    // ollama `think` toggle — identical mapping to the chat lane (both
    // template variable names set; a template reads only the one it knows).
    if let Some(think) = req.get("think").and_then(Value::as_bool) {
        // Fill only missing keys: user-supplied chat_template_kwargs
        // (forwarded below) beats the derived pair — same precedence
        // as the chat lane.
        if out.get("chat_template_kwargs").is_none() {
            out["chat_template_kwargs"] = json!({});
        }
        let kw = out
            .get_mut("chat_template_kwargs")
            .and_then(Value::as_object_mut)
            .expect("just inserted");
        kw.entry("thinking".to_string()).or_insert(json!(think));
        kw.entry("enable_thinking".to_string())
            .or_insert(json!(think));
    }
    // Same forwarding as the chat lane (audit H1: silently dropped).
    if let Some(kw) = req.get("chat_template_kwargs").filter(|k| k.is_object()) {
        let mut merged = kw.clone();
        if let Some(dst) = out
            .get_mut("chat_template_kwargs")
            .and_then(Value::as_object_mut)
        {
            for (k, v) in dst.iter() {
                merged
                    .as_object_mut()
                    .expect("checked object")
                    .entry(k.clone())
                    .or_insert(v.clone());
            }
        }
        out["chat_template_kwargs"] = merged;
    }
    if let Some(opts) = req.get("options").filter(|o| o.is_object()) {
        let unknown = apply_ollama_options(&mut out, opts);
        if !unknown.is_empty() {
            return Err(format!("unsupported options: {}", unknown.join(", ")));
        }
    }
    Ok(Some(out))
}

/// `OpenAI` non-stream chat response -> ollama `GenerateResponse`
/// (chat-bus generate: the response is the assistant message content).
#[must_use]
pub fn openai_chat_to_generate(model: &str, openai: &Value) -> Value {
    let choice = &openai["choices"][0];
    let message = &choice["message"];
    let mut v = json!({
        "model": model,
        "created_at": iso_now(),
        "response": message["content"].clone(),
        "done_reason": choice["finish_reason"].clone(),
        "done": true,
        "total_duration": 0,
        "prompt_eval_count": openai["usage"]["prompt_tokens"].clone(),
        "eval_count": openai["usage"]["completion_tokens"].clone(),
    });
    if let Some(reasoning) = message.get("reasoning_content") {
        v["thinking"] = reasoning.clone();
    }
    let cached = cached_prompt_tokens(openai.get("usage"));
    if cached > 0 {
        v["prompt_eval_cached_count"] = json!(cached);
    }
    merge_timing_fields(&mut v, openai);
    v
}

/// One `OpenAI` SSE chunk -> ollama generate NDJSON lines (`response`
/// deltas; empty deltas emit nothing, mirroring the chat mapper).
#[must_use]
pub fn openai_chunk_to_generate(model: &str, chunk: &Value) -> Vec<Value> {
    let mut out = Vec::new();
    if let Some(choices) = chunk["choices"].as_array() {
        for choice in choices {
            let delta = &choice["delta"];
            let has_content = delta
                .get("content")
                .is_some_and(|c| c.as_str().is_some_and(|s| !s.is_empty()));
            let has_thinking = delta
                .get("reasoning_content")
                .is_some_and(|c| c.as_str().is_some_and(|s| !s.is_empty()));
            if has_content || has_thinking {
                let mut v = json!({"model": model, "created_at": iso_now(), "done": false});
                if has_content {
                    v["response"] = delta["content"].clone();
                }
                if has_thinking {
                    v["thinking"] = delta["reasoning_content"].clone();
                }
                out.push(v);
            }
        }
    }
    out
}

/// Final ollama generate line from usage/finish (stream path — gateway-
/// measured durations, same contract as `ollama_final_chunk`).
#[must_use]
pub fn ollama_generate_final_chunk(
    model: &str,
    usage: Option<&Value>,
    finish: Option<&str>,
    eval_ns: Option<u64>,
    total_ns: Option<u64>,
) -> Value {
    let mut v = json!({
        "model": model,
        "created_at": iso_now(),
        "response": "",
        "done_reason": finish.unwrap_or("stop"),
        "done": true,
        "total_duration": 0,
        "prompt_eval_count": usage.and_then(|u| u["prompt_tokens"].as_i64()).unwrap_or(0),
        "eval_count": usage.and_then(|u| u["completion_tokens"].as_i64()).unwrap_or(0),
    });
    let cached = cached_prompt_tokens(usage);
    if cached > 0 {
        v["prompt_eval_cached_count"] = json!(cached);
    }
    if let Some(e) = eval_ns {
        v["eval_duration"] = json!(e);
    }
    if let Some(t) = total_ns {
        v["total_duration"] = json!(t);
    }
    v
}

pub(crate) fn iso_now() -> String {
    // Cheap UTC timestamp without pulling chrono.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{secs}Z")
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    #[test]
    fn unit__line_buffer__split_utf8_char_across_chunks() {
        // F70 pin: a multi-byte char split across TCP chunk boundaries
        // must reassemble, never decode into twin U+FFFD.
        let mut lb = super::LineBuffer::new();
        let text = "data: {\"delta\":\"\u{4f60}\u{597d}\"}\n\n";
        let bytes = text.as_bytes();
        let cut = text.find("\u{597d}").expect("char present") + 1; // inside 好's bytes
        let a = lb.feed(&bytes[..cut]);
        assert!(a.is_empty(), "partial line held back, got {a:?}");
        let b = lb.feed(&bytes[cut..]);
        let joined = format!("{a}{b}");
        assert_eq!(joined, text);
        assert!(!joined.contains('\u{fffd}'), "replacement char leaked");
    }

    #[test]
    fn unit__line_buffer__complete_lines_pass_through() {
        let mut lb = super::LineBuffer::new();
        assert_eq!(lb.feed(b"hello\nworld"), "hello\n");
        assert_eq!(lb.feed(b"!\n"), "world!\n");
        assert_eq!(lb.feed(b"tail"), "");
    }

    use super::*;

    #[test]
    fn unit__sampler_options__native_passthrough_table() {
        // Every llama-server-native sampler field passes through 1:1
        // (field names verified against the b10896 request schema).
        let cases: &[(&str, serde_json::Value)] = &[
            ("xtc_probability", json!(0.5)),
            ("xtc_threshold", json!(0.1)),
            ("top_n_sigma", json!(2)),
            ("logit_bias", json!({"12834": -3.0})),
            ("dry_multiplier", json!(0.8)),
            ("dry_base", json!(1.75)),
            ("dry_allowed_length", json!(2)),
            ("dry_penalty_last_n", json!(256)),
            ("mirostat", json!(2)),
            ("mirostat_tau", json!(5.0)),
            ("mirostat_eta", json!(0.1)),
            ("dynatemp_range", json!(1.5)),
            ("dynatemp_exponent", json!(1.0)),
            ("adaptive_target", json!(0.9)),
            ("adaptive_decay", json!(0.9)),
            ("samplers", json!(["top_k", "temperature"])),
        ];
        for (key, val) in cases {
            let mut out = json!({});
            let unknown = apply_ollama_options(&mut out, &json!({*key: val.clone()}));
            assert!(unknown.is_empty(), "{key} flagged unknown: {unknown:?}");
            assert_eq!(&out[*key], val, "{key} passthrough");
        }
    }

    #[test]
    fn unit__sampler_options__dry_sequence_breakers_shape_checked() {
        let mut out = json!({});
        let unknown = apply_ollama_options(
            &mut out,
            &json!({"dry_sequence_breakers": ["\\n", ":", "\""]}),
        );
        assert!(unknown.is_empty());
        assert_eq!(
            out["dry_sequence_breakers"].as_array().map(Vec::len),
            Some(3)
        );
        // Empty array / non-strings must 400, not die in the child.
        for bad in [json!([]), json!("\\n"), json!([1, 2])] {
            let mut out2 = json!({});
            let unknown = apply_ollama_options(&mut out2, &json!({"dry_sequence_breakers": bad}));
            assert!(
                unknown.len() == 1 && unknown[0].contains("non-empty array"),
                "bad shape {bad} must be flagged: {unknown:?}"
            );
        }
    }

    #[test]
    fn unit__sampler_options__adaptive_p_joins_chain() {
        // Without an explicit samplers list: default chain + adaptive_p.
        let mut out = json!({});
        apply_ollama_options(&mut out, &json!({"adaptive_p": true}));
        let chain = out["samplers"].as_array().expect("chain built");
        assert_eq!(chain.last(), Some(&json!("adaptive_p")));
        assert!(chain.len() == DEFAULT_SAMPLER_CHAIN.len() + 1);
        // With an explicit list (processed AFTER adaptive_p alphabetically):
        // the user chain is honored and adaptive_p is appended, not lost.
        let mut out2 = json!({});
        apply_ollama_options(
            &mut out2,
            &json!({"adaptive_p": true, "samplers": ["top_k"]}),
        );
        assert_eq!(out2["samplers"], json!(["top_k", "adaptive_p"]));
        // false / absent: never touched.
        let mut out3 = json!({});
        apply_ollama_options(&mut out3, &json!({"adaptive_p": false}));
        assert!(out3.get("samplers").is_none());
    }

    #[test]
    fn unit__chat_to_openai__sampling_and_schema() {
        let req = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false,
            "format": {"type": "object", "properties": {"x": {"type": "number"}}},
            "options": {"temperature": 0.7, "num_predict": 100, "stop": ["\n"], "seed": 42},
        });
        let (out, num_ctx) = chat_to_openai(&req).unwrap();
        assert_eq!(num_ctx, None);
        assert_eq!(out["messages"][0]["content"], "hi");
        assert_eq!(out["temperature"], 0.7);
        assert_eq!(out["max_tokens"], 100);
        assert_eq!(out["stop"][0], "\n");
        assert_eq!(out["seed"], 42);
        assert_eq!(out["response_format"]["type"], "json_schema");
        assert_eq!(out["stream"], false);
    }

    #[test]
    fn unit__chat_to_openai__think_maps_to_template_kwargs() {
        // think=false is the ollama "answer directly" switch: both known
        // template vars carry it; absent think leaves no kwargs key.
        let req = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "think": false,
        });
        let (out, _) = chat_to_openai(&req).unwrap();
        assert_eq!(out["chat_template_kwargs"]["thinking"], false);
        assert_eq!(out["chat_template_kwargs"]["enable_thinking"], false);

        let req_on = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "think": true,
        });
        let (out_on, _) = chat_to_openai(&req_on).unwrap();
        assert_eq!(out_on["chat_template_kwargs"]["enable_thinking"], true);

        let req_none = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        });
        let (out_none, _) = chat_to_openai(&req_none).unwrap();
        assert!(out_none.get("chat_template_kwargs").is_none());
    }

    #[test]
    fn unit__chat_to_openai__num_ctx_extracted() {
        let req = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"num_ctx": 32768, "temperature": 0.1},
        });
        let (out, num_ctx) = chat_to_openai(&req).unwrap();
        assert_eq!(num_ctx, Some(32768));
        assert!(
            out.get("num_ctx").is_none(),
            "num_ctx never forwarded per-request"
        );
    }

    #[test]
    fn unit__chat_to_openai__unknown_option__lists_all() {
        let req = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"temperature": 1, "vram_magic": true, "num_gpu": 5},
        });
        let err = chat_to_openai(&req).unwrap_err();
        assert!(
            err.contains("vram_magic") && err.contains("num_gpu"),
            "{err}"
        );
    }

    #[test]
    fn unit__chat_to_openai__format_json_string() {
        let req = json!({"model": "m", "messages": [], "format": "json"});
        let (out, _) = chat_to_openai(&req).unwrap();
        assert_eq!(out["response_format"]["type"], "json_object");
    }

    #[test]
    fn unit__chat_to_openai__missing_model__error() {
        assert!(chat_to_openai(&json!({"messages": []})).is_err());
    }

    #[test]
    fn unit__openai_chat_to_ollama__counts_from_usage() {
        let openai = json!({
            "choices": [{"message": {"role": "assistant", "content": "hi there"},
                          "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        });
        let o = openai_chat_to_ollama("m", &openai);
        assert_eq!(o["message"]["content"], "hi there");
        assert_eq!(o["done"], true);
        assert_eq!(o["prompt_eval_count"], 5);
        assert_eq!(o["eval_count"], 2);
    }

    #[test]
    fn unit__openai_chunk_to_ollama__content_and_tool_deltas() {
        let c1 = json!({"choices": [{"delta": {"content": "he"}}]});
        let c2 = json!({"choices": [{"delta": {"tool_calls": [{"id": "t1", "function": {"name": "f", "arguments": "{\"a"}}]}}]});
        let c2b = json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\":1}"}}]}}]});
        let c3 = json!({"choices": [{"delta": {}}]});
        let o1 = openai_chunk_to_ollama(&mut ToolCallAccum::default(), "m", &c1);
        assert_eq!(o1.len(), 1);
        assert_eq!(o1[0]["message"]["content"], "he");
        assert_eq!(o1[0]["done"], false);
        // Fragments MERGE: the id+name delta absorbs, the argument shard
        // appends, and the empty delta after them flushes ONE complete
        // ollama-dialect call (index inside function, parsed arguments).
        let mut acc = ToolCallAccum::default();
        assert!(openai_chunk_to_ollama(&mut acc, "m", &c2).is_empty());
        assert!(openai_chunk_to_ollama(&mut acc, "m", &c2b).is_empty());
        let o2 = openai_chunk_to_ollama(&mut acc, "m", &c3);
        assert_eq!(o2.len(), 1);
        let call = &o2[0]["message"]["tool_calls"][0];
        assert_eq!(call["id"], "t1");
        assert_eq!(call["function"]["name"], "f");
        assert_eq!(call["function"]["index"], 0);
        assert_eq!(call["function"]["arguments"], json!({"a": 1}));
        assert_eq!(o2[0]["done"], false);
        // Thinking deltas (reasoning_content) map to message.thinking.
        let c4 = json!({"choices": [{"delta": {"reasoning_content": "hmm"}}]});
        let o4 = openai_chunk_to_ollama(&mut ToolCallAccum::default(), "m", &c4);
        assert_eq!(o4.len(), 1);
        assert_eq!(o4[0]["message"]["thinking"], "hmm");
        assert!(openai_chunk_to_ollama(&mut ToolCallAccum::default(), "m", &c3).is_empty());
    }

    #[test]
    fn unit__tool_call_accum__parallel_indexes_and_malformed_fallback() {
        let mut acc = ToolCallAccum::default();
        acc.absorb(&[
            json!({"index": 0, "id": "a", "function": {"name": "fa", "arguments": "{\"x\":"}}),
        ]);
        acc.absorb(&[
            json!({"index": 1, "id": "b", "function": {"name": "fb", "arguments": "not-json"}}),
        ]);
        acc.absorb(&[json!({"index": 0, "function": {"arguments": "1}"}})]);
        let line = acc.flush_line("m").unwrap();
        let v: Value = serde_json::from_str(line.trim()).unwrap();
        let calls = v["message"]["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["function"]["name"], "fa");
        assert_eq!(calls[0]["function"]["arguments"], json!({"x": 1}));
        // Malformed accumulated arguments fall back to the raw string.
        assert_eq!(calls[1]["function"]["arguments"], "not-json");
        assert_eq!(calls[1]["function"]["index"], 1);
        assert!(acc.is_empty());
        assert!(acc.flush_line("m").is_none());
    }

    #[test]
    fn unit__parse_sse__split_across_chunks() {
        // Full events.
        let (ev, done, used) = parse_sse("data: {\"a\":1}\n\ndata: [DONE]\n\n");
        assert_eq!(ev.len(), 1);
        assert!(done);
        assert_eq!(used, 29);
        // Event split mid-payload: nothing consumed.
        let (ev, done, used) = parse_sse("data: {\"a\":");
        assert!(ev.is_empty());
        assert!(!done);
        assert_eq!(used, 0);
        // First line complete, second partial.
        let (ev, _, used) = parse_sse("data: {\"a\":1}\ndata: {\"b\"");
        assert_eq!(ev.len(), 1);
        assert_eq!(used, 14);
    }

    #[test]
    fn unit__embeddings_translation() {
        let req = json!({"model": "m", "prompt": "hello"});
        let oai = embeddings_to_openai(&req).unwrap();
        assert_eq!(oai["input"], "hello");
        let back = openai_embeddings_to_ollama("m", &json!({"data": [{"embedding": [0.1, 0.2]}]}));
        assert_eq!(back["embedding"][1], 0.2);
    }

    #[test]
    fn unit__generate_raw_and_templated() {
        // Raw: user message carries the prompt, engine template applies.
        let raw = json!({"model": "m", "prompt": "say x", "stream": false});
        let oai = generate_to_openai(&raw).unwrap().unwrap();
        assert_eq!(oai["messages"][0]["role"], "user");
        assert_eq!(oai["messages"][0]["content"], "say x");
        assert_eq!(oai["stream"], false);
        assert!(
            oai.get("prompt").is_none(),
            "chat bus carries no raw prompt"
        );
        // template/suffix stay rejected (engine owns the template).
        let templated = json!({"model": "m", "prompt": "x", "template": "..."});
        assert!(generate_to_openai(&templated).unwrap().is_none());
        let suffixed = json!({"model": "m", "prompt": "x", "suffix": "..."});
        assert!(generate_to_openai(&suffixed).unwrap().is_none());
    }

    #[test]
    fn unit__chat_to_openai__user_template_kwargs_beat_derived_think() {
        // Audit H1 pin: user chat_template_kwargs forward 1:1 and WIN
        // over the think-derived pair (explicit beats derived).
        let req = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "think": true,
            "chat_template_kwargs": {"enable_thinking": false}
        });
        let (out, _) = chat_to_openai(&req).expect("translate");
        let kw = &out["chat_template_kwargs"];
        assert_eq!(kw["enable_thinking"], json!(false), "user key must win");
        assert_eq!(kw["thinking"], json!(true), "missing key filled from think");

        // Without user kwargs the derived pair lands whole.
        let req2 = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "think": false
        });
        let (out2, _) = chat_to_openai(&req2).expect("translate2");
        assert_eq!(out2["chat_template_kwargs"]["thinking"], json!(false));
        assert_eq!(
            out2["chat_template_kwargs"]["enable_thinking"],
            json!(false)
        );
    }

    #[test]
    fn unit__generate_system_images_think_options() {
        // The full geokit pdf_ocr shape: system + prompt + images + options.
        // "iVBORw0KGgo" = 12-byte PNG magic prefix.
        let req = json!({
            "model": "m",
            "prompt": "describe",
            "system": "be terse",
            "images": ["iVBORw0KGgo"],
            "think": true,
            "stream": false,
            "options": {"temperature": 0.2, "num_predict": 64, "num_ctx": 8192}
        });
        let oai = generate_to_openai(&req).unwrap().unwrap();
        assert_eq!(oai["messages"].as_array().unwrap().len(), 2);
        assert_eq!(oai["messages"][0]["role"], "system");
        assert_eq!(oai["messages"][0]["content"], "be terse");
        let user = &oai["messages"][1];
        assert_eq!(user["role"], "user");
        let parts = user["content"].as_array().unwrap();
        assert_eq!(parts.len(), 2, "text + image: {parts:?}");
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "describe");
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(
            parts[1]["image_url"]["url"],
            "data:image/png;base64,iVBORw0KGgo"
        );
        assert_eq!(oai["chat_template_kwargs"]["thinking"], true);
        assert_eq!(oai["chat_template_kwargs"]["enable_thinking"], true);
        assert_eq!(oai["temperature"], 0.2);
        assert_eq!(oai["max_tokens"], 64);
        // Empty system is dropped, not an empty system message.
        let no_sys = json!({"model": "m", "prompt": "x", "system": ""});
        let oai2 = generate_to_openai(&no_sys).unwrap().unwrap();
        assert_eq!(oai2["messages"].as_array().unwrap().len(), 1);
        // think:false still forwards the explicit toggle (chat-lane parity).
        let off = json!({"model": "m", "prompt": "x", "think": false});
        let oai3 = generate_to_openai(&off).unwrap().unwrap();
        assert_eq!(oai3["chat_template_kwargs"]["thinking"], false);
        // No think key: nothing injected.
        let bare = json!({"model": "m", "prompt": "x"});
        let oai4 = generate_to_openai(&bare).unwrap().unwrap();
        assert!(oai4.get("chat_template_kwargs").is_none());
    }

    #[test]
    fn unit__generate_image_mime_sniffing() {
        let mimes = [
            ("iVBORw0KGgo", "image/png"),       // \x89PNG
            ("/9j/4AAQ", "image/jpeg"),         // \xff\xd8\xff
            ("R0lGODlh", "image/gif"),          // GIF8
            ("UklGRgAAAABXRUJQ", "image/webp"), // RIFF....WEBP
        ];
        for (b64, mime) in mimes {
            let req = json!({"model": "m", "prompt": "x", "images": [b64]});
            let oai = generate_to_openai(&req)
                .unwrap()
                .unwrap_or_else(|| panic!("{b64} should map to {mime}"));
            let url = oai["messages"][0]["content"][1]["image_url"]["url"]
                .as_str()
                .unwrap();
            assert!(url.starts_with(&format!("data:{mime};base64,")), "{url}");
        }
        // Unrecognized magic fails fast, never guesses a MIME.
        let bad = json!({"model": "m", "prompt": "x", "images": ["AAAAAAAA"]});
        let err = generate_to_openai(&bad).unwrap_err();
        assert!(err.contains("unsupported image format"), "{err}");
        // Garbage (non-base64) fails fast too.
        let garbage = json!({"model": "m", "prompt": "x", "images": ["!!not-b64!!"]});
        assert!(generate_to_openai(&garbage).is_err());
        // Non-string image entries are a shape error.
        let shaped = json!({"model": "m", "prompt": "x", "images": [42]});
        let err2 = generate_to_openai(&shaped).unwrap_err();
        assert!(err2.contains("base64 strings"), "{err2}");
    }

    #[test]
    fn unit__assistant_tool_calls_gain_type_function() {
        // ollama wire shape omits `type` on tool_calls entries; the child
        // rejects them with "Missing tool call type" (live-repro'd via the
        // geokit server-compare harness: every MULTITURN replay 500'd).
        let req = json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "Compute Mg# for Fo=90"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": "compute_geochem", "arguments": {"query": "Mg# Fo=90"}}}
                ]},
                {"role": "tool", "name": "compute_geochem", "content": "Mg# = 90.9"}
            ]
        });
        let (out, _) = chat_to_openai(&req).unwrap();
        let call = &out["messages"][1]["tool_calls"][0];
        assert_eq!(call["type"], "function", "type filled");
        assert_eq!(call["function"]["name"], "compute_geochem");
        // ollama sends `arguments` as a parsed object; the child demands a
        // JSON string — the request translator stringifies it.
        let want_args = json!({"query": "Mg# Fo=90"}).to_string();
        assert_eq!(
            call["function"]["arguments"], want_args,
            "arguments stringified for the child"
        );
        // Explicit type is preserved verbatim (never overwritten), and
        // empty-string type is treated as absent (children reject "").
        let typed = json!({
            "model": "m",
            "messages": [
                {"role": "assistant", "tool_calls": [
                    {"type": "custom", "function": {"name": "f", "arguments": {}}}
                ]},
                {"role": "assistant", "tool_calls": [
                    {"type": "", "function": {"name": "g", "arguments": {}}}
                ]}
            ]
        });
        let (out2, _) = chat_to_openai(&typed).unwrap();
        assert_eq!(out2["messages"][0]["tool_calls"][0]["type"], "custom");
        assert_eq!(out2["messages"][1]["tool_calls"][0]["type"], "function");
        // Messages without tool_calls ride through untouched.
        let (out3, _) = chat_to_openai(&json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        assert_eq!(out3["messages"][0]["content"], "hi");
    }

    #[test]
    fn unit__chat_message_images_become_parts() {
        // ollama-style message images[] -> multimodal content parts.
        let req = json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "what is this?", "images": ["iVBORw0KGgo"]},
                {"role": "assistant", "content": "a png"},
                {"role": "user", "content": "plain"}
            ]
        });
        let (out, _) = chat_to_openai(&req).unwrap();
        let msgs = out["messages"].as_array().unwrap();
        let parts = msgs[0]["content"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(
            parts[1]["image_url"]["url"],
            "data:image/png;base64,iVBORw0KGgo"
        );
        assert!(msgs[0].get("images").is_none(), "ollama field removed");
        assert_eq!(msgs[1]["content"], "a png", "untouched message verbatim");
        assert_eq!(msgs[2]["content"], "plain");
        // Images-only message (empty content): no empty text part.
        let img_only = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "", "images": ["iVBORw0KGgo"]}]
        });
        let (out2, _) = chat_to_openai(&img_only).unwrap();
        let parts2 = out2["messages"][0]["content"].as_array().unwrap();
        assert_eq!(parts2.len(), 1);
        assert_eq!(parts2[0]["type"], "image_url");
        // Bad image in a message: hard error, not silent text-only.
        let bad = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "x", "images": ["zzz"]}]
        });
        assert!(chat_to_openai(&bad).is_err());
    }

    #[test]
    fn unit__logprobs_mapped_back_in_chat_response() {
        // geokit verifier shape: top-level logprobs array with
        // token/logprob/top_logprobs, cloned 1:1 from the child.
        let openai = json!({
            "choices": [{
                "message": {"content": "hi"},
                "finish_reason": "stop",
                "logprobs": {"content": [
                    {"token": "h", "logprob": -0.1, "top_logprobs": [
                        {"token": "h", "logprob": -0.1}, {"token": "i", "logprob": -2.0}
                    ]}
                ]}
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1}
        });
        let v = openai_chat_to_ollama("m", &openai);
        let lp = v["logprobs"].as_array().unwrap();
        assert_eq!(lp.len(), 1);
        assert_eq!(lp[0]["token"], "h");
        assert_eq!(lp[0]["logprob"], -0.1);
        assert_eq!(lp[0]["top_logprobs"].as_array().unwrap().len(), 2);
        // Absent logprobs: key stays absent (native parity).
        let bare = json!({
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1}
        });
        assert!(openai_chat_to_ollama("m", &bare).get("logprobs").is_none());
    }

    #[test]
    fn unit__logprobs_mapped_back_in_stream_chunks() {
        let chunk = json!({
            "choices": [{
                "delta": {"content": "h"},
                "logprobs": {"content": [
                    {"token": "h", "logprob": -0.1, "top_logprobs": [
                        {"token": "h", "logprob": -0.1}
                    ]}
                ]}
            }]
        });
        let lines = openai_chunk_to_ollama(&mut ToolCallAccum::default(), "m", &chunk);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["logprobs"][0]["token"], "h");
        assert_eq!(lines[0]["logprobs"][0]["logprob"], -0.1);
        // No logprobs on the chunk: line stays lean.
        let bare = json!({"choices": [{"delta": {"content": "h"}}]});
        let lines2 = openai_chunk_to_ollama(&mut ToolCallAccum::default(), "m", &bare);
        assert_eq!(lines2.len(), 1);
        assert!(lines2[0].get("logprobs").is_none());
    }

    #[test]
    fn unit__generate_response_mappers() {
        // Non-stream: response = assistant content, thinking passthrough.
        let openai = json!({
            "choices": [{
                "message": {"content": "42", "reasoning_content": "thinking..."},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            "timings": {"prompt_ms": 100.0, "predicted_ms": 200.0}
        });
        let g = openai_chat_to_generate("m", &openai);
        assert_eq!(g["response"], "42");
        assert_eq!(g["thinking"], "thinking...");
        assert_eq!(g["done"], true);
        assert_eq!(g["done_reason"], "stop");
        assert_eq!(g["prompt_eval_count"], 10);
        assert_eq!(g["eval_count"], 2);
        assert_eq!(g["prompt_eval_duration"], 100_000_000);
        assert_eq!(g["eval_duration"], 200_000_000);
        assert_eq!(g["total_duration"], 300_000_000);
        // Stream chunks: response deltas, empty deltas dropped.
        let chunks = json!({"choices": [{"delta": {"content": "he"}}]});
        let lines = openai_chunk_to_generate("m", &chunks);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["response"], "he");
        assert_eq!(lines[0]["done"], false);
        let think_only = json!({"choices": [{"delta": {"reasoning_content": "hm"}}]});
        let lines2 = openai_chunk_to_generate("m", &think_only);
        assert_eq!(lines2.len(), 1);
        assert_eq!(lines2[0]["thinking"], "hm");
        let empty = json!({"choices": [{"delta": {"content": ""}}]});
        assert!(openai_chunk_to_generate("m", &empty).is_empty());
        // Final line: counts + done_reason + measured durations.
        let usage = json!({"prompt_tokens": 10, "completion_tokens": 2});
        let fin = ollama_generate_final_chunk("m", Some(&usage), Some("length"), Some(1), Some(2));
        assert_eq!(fin["response"], "");
        assert_eq!(fin["done"], true);
        assert_eq!(fin["done_reason"], "length");
        assert_eq!(fin["eval_count"], 2);
        assert_eq!(fin["eval_duration"], 1);
        assert_eq!(fin["total_duration"], 2);
    }

    #[test]
    fn unit__generate_unknown_option__fails_fast() {
        let bad = json!({"model": "m", "prompt": "x", "options": {"num_batch": 512}});
        let err = generate_to_openai(&bad).unwrap_err();
        assert!(
            err.contains("num_batch"),
            "error names the unknown option: {err}"
        );
        // Supported options still pass through untouched.
        let ok = json!({"model": "m", "prompt": "x", "options": {"temperature": 0.7, "seed": 42}});
        let oai = generate_to_openai(&ok).unwrap().unwrap();
        assert_eq!(oai["temperature"], 0.7);
        assert_eq!(oai["seed"], 42);
    }

    #[test]
    fn unit__ollama_final_chunk__usage_optional() {
        let c = ollama_final_chunk("m", None, Some("length"), None, None);
        assert_eq!(c["done"], true);
        assert_eq!(c["done_reason"], "length");
        assert_eq!(c["eval_count"], 0);
        assert_eq!(c["total_duration"], 0);
        // Measured stream timings ride the final line.
        let timed = ollama_final_chunk("m", None, None, Some(1_500_000), Some(2_500_000));
        assert_eq!(timed["eval_duration"], 1_500_000);
        assert_eq!(timed["total_duration"], 2_500_000);
    }

    #[test]
    fn unit__durations_from_child_timings() {
        // ollama parity: t/s displays read eval_duration/prompt_eval_duration.
        let openai = json!({
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            "timings": {"prompt_ms": 100.5, "predicted_ms": 200.25}
        });
        let c = openai_chat_to_ollama("m", &openai);
        assert_eq!(c["eval_count"], 2);
        assert_eq!(c["prompt_eval_duration"], 100_500_000);
        assert_eq!(c["eval_duration"], 200_250_000);
        assert_eq!(c["total_duration"], 300_750_000);
        let g = openai_chat_to_generate("m", &openai);
        assert_eq!(g["response"], "hi");
        assert_eq!(g["eval_duration"], 200_250_000);
        assert_eq!(g["total_duration"], 300_750_000);
        // No timings: keys stay absent, total stays the legacy 0.
        let bare = openai_chat_to_ollama(
            "m",
            &json!({
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}
            }),
        );
        assert!(bare.get("eval_duration").is_none());
        assert_eq!(bare["total_duration"], 0);
    }

    #[test]
    fn unit__keep_alive_values__accepted_shapes() {
        // keep_alive parsing: number (secs), "-1" (forever), "0" (evict).
        for (raw, expect) in [("-1", -1i64), ("0", 0), ("300", 300)] {
            let v: Value = serde_json::from_str(raw).unwrap();
            let parsed = v
                .as_i64()
                .or_else(|| v.as_str().and_then(|s| s.parse::<i64>().ok()));
            assert_eq!(parsed, Some(expect));
        }
    }

    #[test]
    fn unit__cached_prompt_tokens__present_absent_nested() {
        let full = json!({"prompt_tokens": 120, "prompt_tokens_details": {"cached_tokens": 96}});
        assert_eq!(cached_prompt_tokens(Some(&full)), 96);
        // Absent details / None usage / non-numeric: 0, never a panic.
        assert_eq!(cached_prompt_tokens(Some(&json!({"prompt_tokens": 5}))), 0);
        assert_eq!(cached_prompt_tokens(None), 0);
        assert_eq!(
            cached_prompt_tokens(Some(
                &json!({"prompt_tokens_details": {"cached_tokens": "x"}})
            )),
            0
        );
    }

    #[test]
    fn unit__ollama_final_chunk__cached_count_emitted_only_when_nonzero() {
        let warm = json!({"prompt_tokens": 120, "prompt_tokens_details": {"cached_tokens": 96}, "completion_tokens": 4});
        let v = ollama_final_chunk("m", Some(&warm), Some("stop"), None, None);
        assert_eq!(v["prompt_eval_cached_count"], json!(96));
        assert_eq!(v["prompt_eval_count"], json!(120));
        let cold = json!({"prompt_tokens": 40, "completion_tokens": 4});
        let v2 = ollama_final_chunk("m", Some(&cold), Some("stop"), None, None);
        assert!(v2.get("prompt_eval_cached_count").is_none(), "{v2}");
        let v3 = ollama_final_chunk("m", None, None, None, None);
        assert!(v3.get("prompt_eval_cached_count").is_none(), "{v3}");
    }

    #[test]
    fn unit__openai_chat_to_ollama__cached_count_passthrough() {
        let openai = json!({
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 120, "prompt_tokens_details": {"cached_tokens": 96}, "completion_tokens": 4},
        });
        let v = openai_chat_to_ollama("m", &openai);
        assert_eq!(v["prompt_eval_cached_count"], json!(96));
        assert_eq!(v["prompt_eval_count"], json!(120));
        let cold = json!({
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 4},
        });
        assert!(openai_chat_to_ollama("m", &cold)
            .get("prompt_eval_cached_count")
            .is_none());
    }

    #[test]
    fn unit__chat_to_openai__logprobs_passthrough() {
        let req = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": true,
            "top_logprobs": 5,
        });
        let (out, _) = chat_to_openai(&req).unwrap();
        assert_eq!(out["logprobs"], json!(true));
        assert_eq!(out["top_logprobs"], json!(5));
        // top_logprobs without logprobs=true is NOT forwarded (OpenAI
        // requires the flag; forwarding it alone would 400 upstream).
        let req2 = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "top_logprobs": 5,
        });
        let (out2, _) = chat_to_openai(&req2).unwrap();
        assert!(out2.get("logprobs").is_none());
        assert!(out2.get("top_logprobs").is_none());
        // Plain request: nothing added.
        let (out3, _) = chat_to_openai(&json!({
            "model": "m", "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        assert!(out3.get("logprobs").is_none());
    }

    #[test]
    fn unit__chat_to_openai__grammar_passthrough() {
        let gbnf = "root ::= \"yes\" | \"no\"";
        let req = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "grammar": gbnf,
        });
        let (out, _) = chat_to_openai(&req).unwrap();
        assert_eq!(out["grammar"], json!(gbnf));
        // Empty grammar = absent (nothing forwarded).
        let req2 = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "grammar": "",
        });
        let (out2, _) = chat_to_openai(&req2).unwrap();
        assert!(out2.get("grammar").is_none());
        // Non-string grammar never forwarded (the sentinel lint 400s it
        // before translate; translate stays defensive).
        let req3 = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "grammar": 42,
        });
        let (out3, _) = chat_to_openai(&req3).unwrap();
        assert!(out3.get("grammar").is_none());
        // Plain request: nothing added.
        let (out4, _) = chat_to_openai(&json!({
            "model": "m", "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        assert!(out4.get("grammar").is_none());
    }
}
