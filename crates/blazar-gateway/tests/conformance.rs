//! Cross-dialect conformance: one semantic request, four spellings, one
//! semantic answer. Every case in `CASES` is sent through the `OpenAI` chat,
//! Anthropic messages, Ollama chat, and Ollama generate surfaces and must
//! come back with the SAME completion text and the SAME token accounting —
//! the dialect-specific spellings (`usage.prompt_tokens` vs
//! `usage.input_tokens` vs `prompt_eval_count`) are translation, never
//! divergence. The streaming spellings must carry the same text in deltas
//! and terminate cleanly. The Responses surface is a different output
//! class (output items, not chat choices), so it carries a shape+status
//! smoke rather than text parity.

mod support;

use blazar_core::config::Config;
use serde_json::{Value, json};
use support::{client, start};

struct Case {
    name: &'static str,
    user_text: &'static str,
}

const MAX_TOKENS: u64 = 16;

const CASES: &[Case] = &[
    Case {
        name: "short_command",
        user_text: "Reply with exactly: OK",
    },
    Case {
        name: "longer_prompt",
        user_text: "Count from one to five in words, then stop.",
    },
];

/// `OpenAI` spelling: returns (text, `prompt_tokens`, `completion_tokens`).
async fn openai_chat(base: &str, user_text: &str) -> (String, u64, u64) {
    let resp = client()
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": "m1",
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": user_text}]
        }))
        .send()
        .await
        .expect("openai chat request");
    assert_eq!(resp.status(), 200, "openai chat status");
    let body: Value = resp.json().await.expect("openai chat json");
    let text = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_else(|| panic!("openai content missing: {body}"))
        .to_string();
    let prompt = body["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
    let completion = body["usage"]["completion_tokens"].as_u64().unwrap_or(0);
    (text, prompt, completion)
}

/// Anthropic spelling: returns (text, `input_tokens`, `output_tokens`).
async fn anthropic_messages(base: &str, user_text: &str) -> (String, u64, u64) {
    let resp = client()
        .post(format!("{base}/v1/messages"))
        .json(&json!({
            "model": "m1",
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": user_text}]
        }))
        .send()
        .await
        .expect("anthropic messages request");
    assert_eq!(resp.status(), 200, "anthropic messages status");
    let body: Value = resp.json().await.expect("anthropic messages json");
    let text = body["content"][0]["text"]
        .as_str()
        .unwrap_or_else(|| panic!("anthropic text missing: {body}"))
        .to_string();
    let prompt = body["usage"]["input_tokens"].as_u64().unwrap_or(0);
    let completion = body["usage"]["output_tokens"].as_u64().unwrap_or(0);
    (text, prompt, completion)
}

/// Ollama chat spelling: returns (text, `prompt_eval_count`, `eval_count`).
async fn ollama_chat(base: &str, user_text: &str) -> (String, u64, u64) {
    let resp = client()
        .post(format!("{base}/api/chat"))
        .json(&json!({
            "model": "m1",
            "stream": false,
            "options": {"num_predict": MAX_TOKENS},
            "messages": [{"role": "user", "content": user_text}]
        }))
        .send()
        .await
        .expect("ollama chat request");
    assert_eq!(resp.status(), 200, "ollama chat status");
    let body: Value = resp.json().await.expect("ollama chat json");
    let text = body["message"]["content"]
        .as_str()
        .unwrap_or_else(|| panic!("ollama chat content missing: {body}"))
        .to_string();
    let prompt = body["prompt_eval_count"].as_u64().unwrap_or(0);
    let completion = body["eval_count"].as_u64().unwrap_or(0);
    (text, prompt, completion)
}

/// Ollama generate spelling: returns (text, `prompt_eval_count`, `eval_count`).
async fn ollama_generate(base: &str, user_text: &str) -> (String, u64, u64) {
    let resp = client()
        .post(format!("{base}/api/generate"))
        .json(&json!({
            "model": "m1",
            "prompt": user_text,
            "stream": false,
            "options": {"num_predict": MAX_TOKENS}
        }))
        .send()
        .await
        .expect("ollama generate request");
    assert_eq!(resp.status(), 200, "ollama generate status");
    let body: Value = resp.json().await.expect("ollama generate json");
    let text = body["response"]
        .as_str()
        .unwrap_or_else(|| panic!("ollama generate response missing: {body}"))
        .to_string();
    let prompt = body["prompt_eval_count"].as_u64().unwrap_or(0);
    let completion = body["eval_count"].as_u64().unwrap_or(0);
    (text, prompt, completion)
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__conformance__one_request_four_spellings_one_answer() {
    let server = start(Config::default()).await;
    for case in CASES {
        let (openai_text, openai_prompt, openai_completion) =
            openai_chat(&server.base, case.user_text).await;
        assert!(
            !openai_text.is_empty(),
            "case {}: openai returned empty content",
            case.name
        );

        let (anthropic_text, anthropic_prompt, anthropic_completion) =
            anthropic_messages(&server.base, case.user_text).await;
        assert_eq!(
            anthropic_text, openai_text,
            "case {}: anthropic text diverged from openai",
            case.name
        );
        assert_eq!(
            anthropic_prompt, openai_prompt,
            "case {}: anthropic input_tokens diverged from usage.prompt_tokens",
            case.name
        );
        assert_eq!(
            anthropic_completion, openai_completion,
            "case {}: anthropic output_tokens diverged from usage.completion_tokens",
            case.name
        );

        let (chat_text, chat_prompt, chat_completion) =
            ollama_chat(&server.base, case.user_text).await;
        assert_eq!(
            chat_text, openai_text,
            "case {}: ollama chat text diverged from openai",
            case.name
        );
        assert_eq!(
            chat_prompt, openai_prompt,
            "case {}: ollama chat prompt_eval_count diverged",
            case.name
        );
        assert_eq!(
            chat_completion, openai_completion,
            "case {}: ollama chat eval_count diverged",
            case.name
        );

        let (gen_text, gen_prompt, gen_completion) =
            ollama_generate(&server.base, case.user_text).await;
        assert_eq!(
            gen_text, openai_text,
            "case {}: ollama generate text diverged from openai",
            case.name
        );
        assert_eq!(
            gen_prompt, openai_prompt,
            "case {}: ollama generate prompt_eval_count diverged",
            case.name
        );
        assert_eq!(
            gen_completion, openai_completion,
            "case {}: ollama generate eval_count diverged",
            case.name
        );
    }
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__conformance__streaming_spellings_carry_same_text_and_terminate() {
    let server = start(Config::default()).await;
    let base = server.base.clone();
    let user_text = CASES[0].user_text;

    // Ground truth from the non-streaming spelling.
    let (expected_text, _, _) = openai_chat(&server.base, user_text).await;

    // OpenAI SSE: assemble delta.content frames, require [DONE].
    let body = client()
        .post(format!("{base}/v1/chat/completions"))
        .json(&json!({
            "model": "m1",
            "stream": true,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": user_text}]
        }))
        .send()
        .await
        .expect("openai stream request")
        .text()
        .await
        .expect("openai stream body");
    let mut assembled = String::new();
    let mut saw_done = false;
    for line in body.lines().filter(|l| l.starts_with("data: ")) {
        let payload = &line["data: ".len()..];
        if payload == "[DONE]" {
            saw_done = true;
            break;
        }
        if let Ok(frame) = serde_json::from_str::<Value>(payload)
            && let Some(delta) = frame["choices"][0]["delta"]["content"].as_str()
        {
            assembled.push_str(delta);
        }
    }
    assert!(saw_done, "openai stream missing [DONE] terminator");
    assert_eq!(
        assembled, expected_text,
        "openai stream deltas diverged from non-stream text"
    );

    // Ollama NDJSON: assemble message.content lines, require final done.
    let body = client()
        .post(format!("{base}/api/chat"))
        .json(&json!({
            "model": "m1",
            "stream": true,
            "options": {"num_predict": MAX_TOKENS},
            "messages": [{"role": "user", "content": user_text}]
        }))
        .send()
        .await
        .expect("ollama stream request")
        .text()
        .await
        .expect("ollama stream body");
    let mut assembled = String::new();
    let mut saw_done = false;
    for line in body.lines() {
        let Ok(frame) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if let Some(delta) = frame["message"]["content"].as_str() {
            assembled.push_str(delta);
        }
        if frame["done"].as_bool() == Some(true) {
            saw_done = true;
        }
    }
    assert!(saw_done, "ollama stream missing done:true frame");
    assert_eq!(
        assembled, expected_text,
        "ollama stream deltas diverged from non-stream text"
    );

    // Anthropic SSE: the documented event sequence must all appear.
    let body = client()
        .post(format!("{base}/v1/messages"))
        .json(&json!({
            "model": "m1",
            "stream": true,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": user_text}]
        }))
        .send()
        .await
        .expect("anthropic stream request")
        .text()
        .await
        .expect("anthropic stream body");
    for event in [
        "event: message_start",
        "event: content_block_start",
        "event: content_block_delta",
        "event: content_block_stop",
        "event: message_delta",
        "event: message_stop",
    ] {
        assert!(body.contains(event), "anthropic stream missing {event}");
    }
}

#[tokio::test]
#[allow(non_snake_case)]
async fn e2e__conformance__responses_surface_shape_smoke() {
    let server = start(Config::default()).await;
    let base = server.base.clone();
    let resp = client()
        .post(format!("{base}/v1/responses"))
        .json(&json!({
            "model": "m1",
            "max_output_tokens": MAX_TOKENS,
            "input": CASES[0].user_text
        }))
        .send()
        .await
        .expect("responses request");
    assert_eq!(resp.status(), 200, "responses status");
    let body: Value = resp.json().await.expect("responses json");
    assert!(
        body["id"].as_str().is_some_and(|s| !s.is_empty()),
        "responses id missing: {body}"
    );
    assert!(
        body["output"].as_array().is_some_and(|a| !a.is_empty()),
        "responses output items missing: {body}"
    );
}
