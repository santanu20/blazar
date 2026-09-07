//! Pure ollama <-> `OpenAI` translation functions. No I/O — every function
//! is table-testable, including stream-chunk splitting across boundaries.

use serde_json::{json, Value};

/// Ollama sampling options -> `OpenAI` request fields. Unknown option keys
/// are returned so the caller can 400 listing them (fail fast, complaint
/// #15's sibling: never silently drop what the user asked for).
pub fn apply_ollama_options(openai_req: &mut Value, options: &Value) -> Vec<String> {
    let mut unknown = Vec::new();
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
        "messages": req["messages"],
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

/// `OpenAI` non-stream chat response -> ollama `ChatResponse` (done:true with
/// counts from usage).
#[must_use]
pub fn openai_chat_to_ollama(model: &str, openai: &Value) -> Value {
    let choice = &openai["choices"][0];
    let message = &choice["message"];
    let mut msg = json!({"role": "assistant", "content": message["content"].clone()});
    if let Some(tc) = message.get("tool_calls") {
        msg["tool_calls"] = tc.clone();
    }
    if let Some(reasoning) = message.get("reasoning_content") {
        msg["thinking"] = reasoning.clone();
    }
    json!({
        "model": model,
        "created_at": iso_now(),
        "message": msg,
        "done_reason": choice["finish_reason"].clone(),
        "done": true,
        "total_duration": 0,
        "prompt_eval_count": openai["usage"]["prompt_tokens"].clone(),
        "eval_count": openai["usage"]["completion_tokens"].clone(),
    })
}

/// One `OpenAI` SSE chunk -> zero or more ollama NDJSON lines.
#[must_use]
pub fn openai_chunk_to_ollama(model: &str, chunk: &Value) -> Vec<Value> {
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
            if has_content || has_thinking || delta.get("tool_calls").is_some() {
                let mut msg = json!({"role": "assistant"});
                if let Some(c) = delta.get("content") {
                    if c.as_str().is_some_and(|s| !s.is_empty()) {
                        msg["content"] = c.clone();
                    }
                }
                if let Some(rc) = delta.get("reasoning_content") {
                    msg["thinking"] = rc.clone();
                }
                if let Some(tc) = delta.get("tool_calls") {
                    msg["tool_calls"] = tc.clone();
                }
                out.push(json!({
                    "model": model,
                    "created_at": iso_now(),
                    "message": msg,
                    "done": false,
                }));
            }
        }
    }
    out
}

/// Final ollama chunk from the usage/finish information.
#[must_use]
pub fn ollama_final_chunk(model: &str, usage: Option<&Value>, finish: Option<&str>) -> Value {
    json!({
        "model": model,
        "created_at": iso_now(),
        "message": {"role": "assistant", "content": ""},
        "done_reason": finish.unwrap_or("stop"),
        "done": true,
        "total_duration": 0,
        "prompt_eval_count": usage.and_then(|u| u["prompt_tokens"].as_i64()).unwrap_or(0),
        "eval_count": usage.and_then(|u| u["completion_tokens"].as_i64()).unwrap_or(0),
    })
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

/// /api/generate raw prompt -> /v1/completions body. Returns None when
/// the request uses templated features (documented divergence: use
/// /api/chat, which passes the model's own template through --jinja).
#[must_use]
pub fn generate_to_openai(req: &Value) -> Option<Value> {
    let templated = ["system", "template", "suffix", "images"]
        .iter()
        .any(|k| req.get(*k).is_some_and(|v| !v.is_null()));
    if templated {
        return None;
    }
    let model = req["model"].as_str()?;
    let prompt = req["prompt"].as_str()?;
    let mut out = json!({"model": model, "prompt": prompt});
    if let Some(stream) = req["stream"].as_bool() {
        out["stream"] = json!(stream);
    }
    if let Some(opts) = req.get("options").filter(|o| o.is_object()) {
        let _ = apply_ollama_options(&mut out, opts);
    }
    Some(out)
}

/// `OpenAI` completion response -> ollama `GenerateResponse`.
#[must_use]
pub fn openai_completion_to_ollama(model: &str, openai: &Value) -> Value {
    json!({
        "model": model,
        "created_at": iso_now(),
        "response": openai["choices"][0]["text"].clone(),
        "done": true,
        "done_reason": openai["choices"][0]["finish_reason"].clone(),
        "prompt_eval_count": openai["usage"]["prompt_tokens"].clone(),
        "eval_count": openai["usage"]["completion_tokens"].clone(),
    })
}

fn iso_now() -> String {
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
    use super::*;

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
        let c2 = json!({"choices": [{"delta": {"tool_calls": [{"id": "t1", "function": {"name": "f", "arguments": "{\"a\""}}]}}]});
        let c3 = json!({"choices": [{"delta": {}}]});
        let o1 = openai_chunk_to_ollama("m", &c1);
        assert_eq!(o1.len(), 1);
        assert_eq!(o1[0]["message"]["content"], "he");
        assert_eq!(o1[0]["done"], false);
        let o2 = openai_chunk_to_ollama("m", &c2);
        assert_eq!(o2.len(), 1);
        assert!(o2[0]["message"]["tool_calls"].is_array());
        // Thinking deltas (reasoning_content) map to message.thinking.
        let c4 = json!({"choices": [{"delta": {"reasoning_content": "hmm"}}]});
        let o4 = openai_chunk_to_ollama("m", &c4);
        assert_eq!(o4.len(), 1);
        assert_eq!(o4[0]["message"]["thinking"], "hmm");
        assert!(openai_chunk_to_ollama("m", &c3).is_empty());
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
        let raw = json!({"model": "m", "prompt": "say x", "stream": false});
        let oai = generate_to_openai(&raw).unwrap();
        assert_eq!(oai["prompt"], "say x");
        let templated = json!({"model": "m", "prompt": "x", "system": "you are y"});
        assert!(generate_to_openai(&templated).is_none());
        let with_images = json!({"model": "m", "prompt": "x", "images": [""]});
        assert!(generate_to_openai(&with_images).is_none());
    }

    #[test]
    fn unit__ollama_final_chunk__usage_optional() {
        let c = ollama_final_chunk("m", None, Some("length"));
        assert_eq!(c["done"], true);
        assert_eq!(c["done_reason"], "length");
        assert_eq!(c["eval_count"], 0);
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
}
