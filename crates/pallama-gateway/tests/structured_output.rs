//! R9 structured-output pre-validation — integration tests over the real
//! router: malformed `format`/`grammar`/`response_format` fail fast with
//! 400 BEFORE model admission (no child spawn), format+grammar mutual
//! exclusion, and the clean-lane passthrough (stub child 200s).

#![allow(non_snake_case)]

mod support;

use pallama_core::Config;
use support::{client, start};

fn chat_body() -> serde_json::Value {
    serde_json::json!({
        "model": "m1",
        "stream": false,
        "messages": [{"role": "user", "content": "hi"}]
    })
}

async fn post_chat(ts: &support::TestServer, body: &serde_json::Value) -> reqwest::Response {
    client()
        .post(format!("{}/api/chat", ts.base))
        .json(body)
        .send()
        .await
        .unwrap()
}

#[tokio::test]
async fn integration__structured_output__malformed_format_400_pre_admission() {
    let ts = start(Config::default()).await;
    for fmt in [
        serde_json::json!("nonsense"),
        serde_json::json!(5),
        serde_json::json!(true),
        serde_json::json!(["object"]),
    ] {
        let mut body = chat_body();
        body["format"] = fmt.clone();
        let resp = post_chat(&ts, &body).await;
        assert_eq!(resp.status(), 400, "format {fmt} should 400");
        let text = resp.text().await.unwrap();
        assert!(text.contains("invalid structured output"), "{text}");
        // Teaching error names the field.
        assert!(text.contains("format"), "{text}");
    }
    // Malformed schema shape: properties non-object.
    let mut body = chat_body();
    body["format"] = serde_json::json!({"properties": 5});
    let resp = post_chat(&ts, &body).await;
    assert_eq!(resp.status(), 400);
    let text = resp.text().await.unwrap();
    assert!(text.contains("properties"), "{text}");
}

#[tokio::test]
async fn integration__structured_output__format_and_grammar_both_400() {
    let ts = start(Config::default()).await;
    let mut body = chat_body();
    body["format"] = serde_json::json!({"type": "object"});
    body["grammar"] = serde_json::json!("root ::= \"x\"");
    let resp = post_chat(&ts, &body).await;
    assert_eq!(resp.status(), 400);
    let text = resp.text().await.unwrap();
    assert!(text.contains("both"), "{text}");
}

#[tokio::test]
async fn integration__structured_output__grammar_only_flows_to_child() {
    // Clean GBNF passes the lint and reaches the stub child (200).
    let ts = start(Config::default()).await;
    let mut body = chat_body();
    body["grammar"] = serde_json::json!("root ::= \"yes\" | \"no\"");
    let resp = post_chat(&ts, &body).await;
    assert_eq!(resp.status(), 200);
    // Over-cap grammar 400s without admission.
    let mut big = chat_body();
    big["grammar"] = serde_json::Value::String("a".repeat(256 * 1024 + 1));
    let resp = post_chat(&ts, &big).await;
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn integration__structured_output__malformed_client_response_format_400() {
    let ts = start(Config::default()).await;
    // Unknown non-empty type (mirrors child set: json_object/json_schema).
    let mut body = chat_body();
    body["response_format"] = serde_json::json!({"type": "yaml"});
    let resp = post_chat(&ts, &body).await;
    assert_eq!(resp.status(), 400);
    let text = resp.text().await.unwrap();
    assert!(text.contains("response_format"), "{text}");
    // json_schema.schema present-non-object.
    let mut body = chat_body();
    body["response_format"] =
        serde_json::json!({"type": "json_schema", "json_schema": {"schema": "x"}});
    let resp = post_chat(&ts, &body).await;
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn integration__structured_output__valid_format_still_works() {
    // "json" string and a clean schema object flow through unchanged.
    let ts = start(Config::default()).await;
    let mut body = chat_body();
    body["format"] = serde_json::json!("json");
    let resp = post_chat(&ts, &body).await;
    assert_eq!(resp.status(), 200);
    let mut body = chat_body();
    body["format"] = serde_json::json!({
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
    });
    let resp = post_chat(&ts, &body).await;
    assert_eq!(resp.status(), 200);
}
