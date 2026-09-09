//! Client-compatibility pins: the exact request shapes real agent CLIs
//! send. Each test pins a named client's dialect so a gateway change
//! that breaks one fails loudly with the client's name in the test id.
//! Served by the stub engine (no live model needed).

#![allow(non_snake_case)] // client__dialect__behavior naming

mod support;

use support::{client, start};

#[tokio::test]
async fn client__claude_code__anthropic_dialect_via_x_api_key() {
    // Claude Code: POST /v1/messages, `x-api-key` (never Authorization),
    // `anthropic-version` header, system as a top-level array, streaming
    // by default; also non-stream JSON responses.
    let cfg = support::config_with_keys();
    let ts = start(cfg).await;
    let c = client();
    let url = format!("{}/v1/messages", ts.base);

    // No auth -> 401 (auth middleware must understand x-api-key).
    let no_auth = c
        .post(&url)
        .header("anthropic-version", "2023-06-01")
        .json(&serde_json::json!({"model": "m1", "max_tokens": 8, "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(no_auth.status(), 401);

    let r: serde_json::Value = c
        .post(&url)
        .header("x-api-key", "plm_admin")
        .header("anthropic-version", "2023-06-01")
        .json(&serde_json::json!({
            "model": "m1", "max_tokens": 16, "stream": false,
            "system": [{"type": "text", "text": "be brief"}],
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    // The child speaks the anthropic messages shape on this route.
    assert!(
        r.get("content").is_some() || r.get("id").is_some(),
        "anthropic-shaped response: {r}"
    );
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
async fn client__codex__responses_strict_tools_and_chaining() {
    // Codex-CLI class: /v1/responses with strict function definitions
    // (internally tagged, strict-by-default per OpenAI docs) and
    // previous_response_id chaining through the gateway registry.
    let ts = start(support::config_with_keys()).await;
    let c = client();
    let url = format!("{}/v1/responses", ts.base);
    let body = serde_json::json!({
        "model": "m1", "stream": false,
        "instructions": "call the tool",
        "input": [{"role": "user", "content": "weather?"}],
        "tools": [{
            "type": "function",
            "name": "get_weather",           // internal tagging (Responses)
            "strict": true,
            "parameters": {"type": "object", "properties": {}, "additionalProperties": false},
        }],
    });
    let r1: serde_json::Value = c
        .post(&url)
        .bearer_auth("plm_admin")
        .json(&body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let id = r1["id"].as_str().unwrap_or_default().to_string();
    assert!(id.starts_with("resp_"), "gateway-issued id: {id}");

    let chained = serde_json::json!({
        "model": "m1", "stream": false,
        "previous_response_id": id,
        "input": [{"role": "user", "content": "again"}],
    });
    let r2: serde_json::Value = c
        .post(&url)
        .bearer_auth("plm_admin")
        .json(&chained)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        r2["debug_input_items"]
            .as_array()
            .is_some_and(|a| a.len() == 3),
        "chained history reached the child: {r2}"
    );
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
async fn client__continue_vscode__chat_stream_and_usage_chunk() {
    // Continue/Cline class: /v1/chat/completions SSE with
    // stream_options.include_usage (final chunk carries usage for
    // accounting), model field, standard OpenAI headers.
    let ts = start(support::config_with_keys()).await;
    let c = client();
    let resp = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .bearer_auth("plm_admin")
        .header("accept", "text/event-stream")
        .json(&serde_json::json!({
            "model": "m1", "stream": true,
            "stream_options": {"include_usage": true},
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap();
    assert!(resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.contains("text/event-stream")));
    let body = resp.text().await.unwrap();
    assert!(body.contains("data:"), "SSE framing: {body}");
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
async fn client__remote_instance__routes_by_prefix() {
    // A second stub-llama-server plays the remote (vLLM/MLX/another
    // pallama would look the same: OpenAI-compatible).
    let stub = support::spawn_remote_stub().await;
    let cfg = support::config_with_keys();
    let cfg = support::with_remote(cfg, "far", &stub.base);
    let ts = start(cfg).await;
    let c = client();

    // OpenAI lane: model "far:m1" hits the remote, model rewritten.
    let r: serde_json::Value = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .bearer_auth("plm_admin")
        .json(&serde_json::json!({
            "model": "far:m1", "stream": false,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r["model"], "m1", "prefix stripped for the remote: {r}");
    assert!(r["choices"].as_array().is_some());

    // ollama lane: /api/chat with far:m1 translates through the remote.
    let o: serde_json::Value = c
        .post(format!("{}/api/chat", ts.base))
        .bearer_auth("plm_admin")
        .json(&serde_json::json!({
            "model": "far:m1", "stream": false,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(o["done"], true);
    assert!(o["message"]["content"].as_str().is_some());

    // Unknown prefix stays a local 404 (never silently routed).
    let miss = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .bearer_auth("plm_admin")
        .json(&serde_json::json!({"model": "ghost:x", "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(miss.status(), 404);

    // ps carries the remote probe.
    let ps: serde_json::Value = c
        .get(format!("{}/api/ps", ts.base))
        .bearer_auth("plm_admin")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let rem = ps["remotes"].as_array().cloned().unwrap_or_default();
    assert_eq!(rem.len(), 1);
    assert_eq!(rem[0]["name"], "far");
    assert_eq!(rem[0]["ok"], true);
    stub.shutdown().await;
    ts.state.sup.shutdown_all().await.unwrap();
}

/// C4: same-named remotes form a POOL; a conversation prefix sticks to
/// the member that served it (its KV holds the prefix), surfaced via
/// `x-pallama-remote`.
#[tokio::test]
async fn client__remote_pool__prefix_sticky_and_remote_header() {
    let a = support::spawn_remote_stub().await;
    let b = support::spawn_remote_stub().await;
    let cfg = support::config_with_keys();
    let cfg = support::with_remote(cfg, "far", &a.base);
    let cfg = support::with_remote(cfg, "far", &b.base);
    let ts = start(cfg).await;
    let c = client();
    let url = format!("{}/v1/chat/completions", ts.base);
    let convo = serde_json::json!({
        "model": "far:m1", "stream": false,
        "messages": [
            {"role": "system", "content": "You are a terse oracle."},
            {"role": "user", "content": "raven facts"}
        ],
    });
    let r1 = c
        .post(&url)
        .bearer_auth("plm_admin")
        .json(&convo)
        .send()
        .await
        .unwrap();
    assert_eq!(r1.status(), 200);
    let who1 = r1
        .headers()
        .get("x-pallama-remote")
        .expect("serving member named")
        .to_str()
        .unwrap()
        .to_string();
    assert!(who1.starts_with("far|"), "health-key shape: {who1}");

    // Same prefix (system + first-user head identical): sticky.
    let r2 = c
        .post(&url)
        .bearer_auth("plm_admin")
        .json(&convo)
        .send()
        .await
        .unwrap();
    assert_eq!(r2.status(), 200);
    let who2 = r2
        .headers()
        .get("x-pallama-remote")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert_eq!(who1, who2, "same conversation prefix sticks to its member");

    // A DIFFERENT prefix: still served, header still present.
    let other = serde_json::json!({
        "model": "far:m1", "stream": false,
        "messages": [{"role": "user", "content": "completely different topic"}],
    });
    let r3 = c
        .post(&url)
        .bearer_auth("plm_admin")
        .json(&other)
        .send()
        .await
        .unwrap();
    assert_eq!(r3.status(), 200);
    assert!(r3.headers().get("x-pallama-remote").is_some());
    a.shutdown().await;
    b.shutdown().await;
    ts.state.sup.shutdown_all().await.unwrap();
}

/// E4: `audit_log = true` appends one JSON line per generation request
/// with identity + outcome (key, model, status) — never content.
#[tokio::test]
async fn client__audit_log__generation_lines_written() {
    let mut cfg = support::config_with_keys();
    cfg.audit_log = true;
    let ts = support::start(cfg).await;
    let c = client();
    let r = c
        .post(format!("{}/api/chat", ts.base))
        .bearer_auth("plm_ci")
        .json(
            &serde_json::json!({"model": "m1", "stream": false, "messages": [
                {"role": "user", "content": "hi"}
            ]}),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let trace = r
        .headers()
        .get("x-pallama-trace-id")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();

    // Writer is async: poll for the line (bounded).
    let audit_path = ts.dirs.data_dir.join("log").join("audit.jsonl");
    let mut line = String::new();
    for _ in 0..40 {
        if let Ok(raw) = std::fs::read_to_string(&audit_path) {
            if let Some(l) = raw.lines().find(|l| l.contains(&trace)) {
                line = l.to_string();
                break;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(!line.is_empty(), "audit line for {trace} in {audit_path:?}");
    let v: serde_json::Value = serde_json::from_str(&line).unwrap();
    assert_eq!(v["key"], "ci", "key name recorded: {line}");
    assert_eq!(v["model"], "m1", "model recorded: {line}");
    assert_eq!(v["status"], 200);
    assert_eq!(v["path"], "/api/chat");
    assert!(v["ms"].as_u64().is_some(), "latency recorded");
    assert!(v["ts"].as_u64().is_some(), "timestamp recorded");
    // Content is never audited.
    assert!(!line.contains("hi"), "prompt bytes must not appear: {line}");
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
async fn client__ollama_native__env_dialect_untouched() {
    // ollama-native CLIs (OLLAMA_HOST) speak /api/*; auth still applies
    // when keys exist, via EITHER header style.
    let ts = start(support::config_with_keys()).await;
    let c = client();
    let r = c
        .post(format!("{}/api/chat", ts.base))
        .header("x-api-key", "plm_admin")
        .json(&serde_json::json!({
            "model": "m1", "stream": false,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let v: serde_json::Value = r.json().await.unwrap();
    assert_eq!(v["done"], true);
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
async fn client__well_known__capability_discovery() {
    let cfg = support::config_with_keys();
    let ts = start(cfg).await;
    let v: serde_json::Value = reqwest::Client::new()
        .get(format!("{}/.well-known/pallama", ts.base))
        .bearer_auth("plm_admin")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(v["name"], "pallama");
    assert_eq!(v["features"]["keys"], true);
    assert_eq!(v["features"]["singleflight"], true);
    assert!(v["endpoints"]["openai"]
        .as_array()
        .unwrap()
        .iter()
        .any(|e| e == "/v1/chat/completions"));
    assert!(v["headers"]
        .as_array()
        .unwrap()
        .iter()
        .any(|h| h == "x-pallama-num-ctx"));
    // Auth applies: anonymous discovery is refused when keys exist.
    let anon = reqwest::Client::new()
        .get(format!("{}/.well-known/pallama", ts.base))
        .send()
        .await
        .unwrap();
    assert_eq!(anon.status(), 401);
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
async fn client__keys_rotate__old_secret_dies_new_works() {
    let ts = start(support::config_with_keys()).await;
    let c = support::client();
    let created: serde_json::Value = c
        .post(format!("{}/api/keys", ts.base))
        .bearer_auth("plm_admin")
        .json(&serde_json::json!({"name": "rot-me", "models": ["m1"]}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let old = created["key"].as_str().unwrap().to_string();
    let rotated: serde_json::Value = c
        .post(format!("{}/api/keys/rotate?name=rot-me", ts.base))
        .bearer_auth("plm_admin")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let new = rotated["key"].as_str().unwrap().to_string();
    assert_ne!(old, new);
    assert!(new.starts_with("plm_"));
    assert_eq!(
        c.get(format!("{}/api/tags", ts.base))
            .bearer_auth(old)
            .send()
            .await
            .unwrap()
            .status(),
        401,
        "old secret revoked"
    );
    assert_eq!(
        c.get(format!("{}/api/tags", ts.base))
            .bearer_auth(&new)
            .send()
            .await
            .unwrap()
            .status(),
        200,
        "new secret valid"
    );
    ts.state.sup.shutdown_all().await.unwrap();
}
