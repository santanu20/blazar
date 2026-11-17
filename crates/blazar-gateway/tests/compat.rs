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
    assert!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.contains("text/event-stream"))
    );
    let body = resp.text().await.unwrap();
    assert!(body.contains("data:"), "SSE framing: {body}");
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
async fn client__remote_instance__routes_by_prefix() {
    // A second stub-llama-server plays the remote (vLLM/MLX/another
    // blazar would look the same: OpenAI-compatible).
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

#[tokio::test]
async fn client__remote_ollama_lane__scoped_key_rejected() {
    // F11 pin: the ollama /api/chat remote lane pays key admission - a
    // key scoped to `m1` must NOT reach `far:m1` (pre-fix the remote
    // split returned before keys.check entirely).
    let stub = support::spawn_remote_stub().await;
    let cfg = support::config_with_keys();
    let cfg = support::with_remote(cfg, "far", &stub.base);
    let ts = start(cfg).await;
    let c = client();
    let resp = c
        .post(format!("{}/api/chat", ts.base))
        .bearer_auth("plm_ci")
        .json(&serde_json::json!({
            "model": "far:m1", "stream": false,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 403, "scoped key must not reach the remote");
    let resp = c
        .post(format!("{}/api/chat", ts.base))
        .bearer_auth("plm_admin")
        .json(&serde_json::json!({
            "model": "far:m1", "stream": false,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "unscoped key still reaches the remote");
    ts.state.sup.shutdown_all().await.unwrap();
    stub.shutdown().await;
}

/// Remote-lane token accounting: a streamed response that never carries
/// a usage payload (client did not ask `stream_options.include_usage`)
/// still settles the key's budget from the observed token frames -
/// role + 2 content + finish chunks count, `[DONE]` does not. Also
/// pins that the remote forward now charges key budgets at all (the
/// `OpenAI` remote lane previously returned the forward uncharged).
#[tokio::test]
async fn client__remote_stream_without_usage__charges_token_frames() {
    let stub = support::spawn_remote_stub().await;
    let cfg = blazar_core::Config {
        keys: vec![
            blazar_core::ApiKey {
                name: "admin".into(),
                key: "plm_admin".into(),
                ..blazar_core::ApiKey::default()
            },
            blazar_core::ApiKey {
                name: "frames".into(),
                key: "plm_frames".into(),
                daily_tokens: 1000,
                ..blazar_core::ApiKey::default()
            },
        ],
        ..blazar_core::Config::default()
    };
    let cfg = support::with_remote(cfg, "far", &stub.base);
    let ts = start(cfg).await;
    let c = client();

    let resp = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .bearer_auth("plm_frames")
        .json(&serde_json::json!({
            "model": "far:m1", "stream": true,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    // Drain the FULL stream so the accounting wrapper's Drop runs.
    let body = resp.text().await.unwrap();
    assert!(body.contains("data: "), "SSE stream forwarded: {body}");
    assert!(
        !body.contains("\"usage\""),
        "stub omits the usage chunk without include_usage: {body}"
    );

    let usage: serde_json::Value = c
        .get(format!("{}/api/keys", ts.base))
        .bearer_auth("plm_admin")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let frames = usage["keys"]
        .as_array()
        .unwrap()
        .iter()
        .find(|k| k["name"] == "frames")
        .expect("frames key listed")
        .clone();
    assert_eq!(
        frames["usage"]["tokens"], 4,
        "role + 2 content + finish chunks counted; [DONE] not: {frames}"
    );
    ts.state.sup.shutdown_all().await.unwrap();
    stub.shutdown().await;
}

/// Model pools (`[[model_pools]]`): a pool name routes to the
/// strategy-picked member. The stub 404s unknown model ids, so a 200
/// on a pool ask proves the body rewrite reached the member; scope
/// semantics follow the expanded member string (a key scoped to the
/// pool name does not bypass the member's scope - same posture as
/// failover aliases); `/api/route` shows the pool table; the
/// Anthropic translate lane refuses remote picks.
// one sandbox, one scenario: five observations of one pool
#[allow(clippy::too_many_lines)]
#[tokio::test]
async fn client__model_pool__routes_member_scope_and_route_table() {
    let stub = support::spawn_remote_stub().await;
    let cfg = blazar_core::Config {
        keys: vec![
            blazar_core::ApiKey {
                name: "admin".into(),
                key: "plm_admin".into(),
                ..blazar_core::ApiKey::default()
            },
            blazar_core::ApiKey {
                name: "member_scoped".into(),
                key: "plm_member".into(),
                models: vec!["far:m1".into()],
                ..blazar_core::ApiKey::default()
            },
            blazar_core::ApiKey {
                name: "pool_scoped".into(),
                key: "plm_pool".into(),
                models: vec!["svc".into()],
                ..blazar_core::ApiKey::default()
            },
        ],
        ..blazar_core::Config::default()
    };
    let cfg = support::with_remote(cfg, "far", &stub.base);
    let cfg = support::with_pool(
        cfg,
        "svc",
        blazar_core::config::PoolStrategy::Ordered,
        &[("far", "m1")],
    );
    let ts = start(cfg).await;
    let c = client();

    // Admin ask through the pool: 200 proves the member rewrite landed
    // (the stub rejects unknown model ids).
    let resp = c
        .post(format!("{}/api/chat", ts.base))
        .bearer_auth("plm_admin")
        .json(&serde_json::json!({
            "model": "svc", "stream": false,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "pool ask must reach the member");
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(
        body["message"]["content"]
            .as_str()
            .is_some_and(|t| !t.is_empty()),
        "stub answer translated: {body}"
    );

    // Member-scoped key: admitted (scope covers the expanded member).
    let resp = c
        .post(format!("{}/api/chat", ts.base))
        .bearer_auth("plm_member")
        .json(&serde_json::json!({
            "model": "svc", "stream": false,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "member-scoped key serves the pool");

    // Pool-name-scoped key: the scope check runs on the expanded
    // member string, so it refuses (no bypass through pool identity).
    let resp = c
        .post(format!("{}/api/chat", ts.base))
        .bearer_auth("plm_pool")
        .json(&serde_json::json!({
            "model": "svc", "stream": false,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp.status(),
        403,
        "pool-name scope does not bypass member scope"
    );

    // Route introspection: pool envelope with the member table and the
    // decision target.
    let route: serde_json::Value = c
        .get(format!("{}/api/route/svc", ts.base))
        .bearer_auth("plm_admin")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(route["object"], "blazar.route");
    assert_eq!(route["decision"]["target"], "far:m1", "route: {route}");
    assert_eq!(
        route["pool"]["members"].as_array().map(Vec::len),
        Some(1),
        "member table listed: {route}"
    );

    // Anthropic translate lane: a remote pick refuses with the pool
    // named (local-only lane contract).
    let resp = c
        .post(format!("{}/v1/messages", ts.base))
        .bearer_auth("plm_admin")
        .json(&serde_json::json!({
            "model": "svc", "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400, "translate lane refuses remote picks");
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("svc") && m.contains("far:m1")),
        "refusal names pool and member: {body}"
    );

    ts.state.sup.shutdown_all().await.unwrap();
    stub.shutdown().await;
}

/// Local member service EWMA: completing one chat through a
/// local-member pool stamps the per-model EWMA, and the route table
/// reflects `service_measured: true` for that member (the local twin
/// of the remote lane's per-peer EWMA).
#[tokio::test]
async fn client__model_pool__local_member_stamps_service_ewma() {
    let cfg = blazar_core::Config {
        keys: vec![blazar_core::ApiKey {
            name: "admin".into(),
            key: "plm_admin".into(),
            ..blazar_core::ApiKey::default()
        }],
        ..blazar_core::Config::default()
    };
    let cfg = support::with_pool(
        cfg,
        "svc_local",
        blazar_core::config::PoolStrategy::Adaptive,
        &[("local", "m1")],
    );
    let ts = start(cfg).await;
    let c = client();

    // Before any traffic: the member carries the declared default and
    // no measurement.
    let route: serde_json::Value = c
        .get(format!("{}/api/route/svc_local", ts.base))
        .bearer_auth("plm_admin")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        route["pool"]["members"][0]["service_measured"], false,
        "{route}"
    );

    // One completed generation through the pool (expansion rewrites
    // the model to the local member "m1").
    let resp = c
        .post(format!("{}/api/chat", ts.base))
        .bearer_auth("plm_admin")
        .json(&serde_json::json!({
            "model": "svc_local", "stream": false,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "local member serves the pool ask");

    let route: serde_json::Value = c
        .get(format!("{}/api/route/svc_local", ts.base))
        .bearer_auth("plm_admin")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        route["pool"]["members"][0]["service_measured"], true,
        "route table after one completed chat: {route}"
    );
    assert!(
        route["pool"]["members"][0]["service_ms"]
            .as_u64()
            .is_some_and(|ms| ms > 0),
        "measured service time is positive: {route}"
    );

    ts.state.sup.shutdown_all().await.unwrap();
}

/// C4: same-named remotes form a POOL; a conversation prefix sticks to
/// the member that served it (its KV holds the prefix), surfaced via
/// `x-blazar-remote`.
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
        .get("x-blazar-remote")
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
        .get("x-blazar-remote")
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
    assert!(r3.headers().get("x-blazar-remote").is_some());
    a.shutdown().await;
    b.shutdown().await;
    ts.state.sup.shutdown_all().await.unwrap();
}

/// E4: `audit_log = true` appends one JSON line per generation request
/// with identity + outcome (key, model, status) - never content.
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
        .get("x-blazar-trace-id")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();

    // Writer is async: poll for the line (bounded).
    let audit_path = ts.dirs.data_dir.join("log").join("audit.jsonl");
    let mut line = String::new();
    for _ in 0..40 {
        if let Ok(raw) = std::fs::read_to_string(&audit_path)
            && let Some(l) = raw.lines().find(|l| l.contains(&trace))
        {
            line = l.to_string();
            break;
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
// Capability discovery asserts one thing per surface; splitting it
// would scatter the census contract this test pins in one place.
#[allow(clippy::too_many_lines)]
async fn client__well_known__capability_discovery() {
    let cfg = support::config_with_keys();
    let ts = start(cfg).await;
    blazar_core::tls::ensure_tls_provider();
    let v: serde_json::Value = reqwest::Client::new()
        .get(format!("{}/.well-known/blazar", ts.base))
        .bearer_auth("plm_admin")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(v["name"], "blazar");
    assert_eq!(v["features"]["keys"], true);
    // Reports the runtime switch, not feature presence: single-flight
    // is opt-in since the C>=2 serialization finding (default false).
    assert_eq!(v["features"]["singleflight"], false);
    assert!(
        v["endpoints"]["openai"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e == "/v1/chat/completions")
    );
    // Audit MM15: the census must advertise EVERY media generation
    // route (plus the engine-scoped surfaces) - a client discovering
    // capabilities needs them as much as the text lanes.
    let openai_eps: Vec<&str> = v["endpoints"]["openai"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|e| e.as_str())
        .collect();
    for path in [
        "/v1/images/generations",
        "/v1/images/edits",
        "/v1/images/jobs/{id}",
        "/v1/images/jobs/{id}/cancel",
        "/v1/images/capabilities",
        "/v1/videos/generations",
        "/v1/videos/jobs/{id}",
        "/v1/videos/jobs/{id}/cancel",
        "/v1/videos/capabilities",
        "/v1/audio/transcriptions",
        "/v1/audio/translations",
        "/v1/audio/speech",
        "/v1/audio/jobs/{id}",
        "/v1/audio/jobs/{id}/cancel",
        "/v1/audio/capabilities",
        "/audio/transcriptions",
        "/audio/translations",
        "/props",
        "/v1/stream",
        "/v1/chat/completions/input_tokens",
        "/v1/systemone",
        "/completion",
        "/embeddings",
        "/rerank",
        "/reranking",
        "/v1/chat/completions/{id}",
    ] {
        assert!(openai_eps.contains(&path), "census missing {path}");
    }
    // Audit MM16: the non-/v1 audio aliases are real routes, not just
    // census entries - they must route (auth refusal, not 404).
    for path in ["/audio/transcriptions", "/audio/translations"] {
        blazar_core::tls::ensure_tls_provider();
        let r = reqwest::Client::new()
            .post(format!("{}{}", ts.base, path))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 401, "{path} must be a mounted route");
    }
    assert!(
        v["headers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|h| h == "x-blazar-num-ctx")
    );
    // Auth applies: anonymous discovery is refused when keys exist.
    blazar_core::tls::ensure_tls_provider();
    let anon = reqwest::Client::new()
        .get(format!("{}/.well-known/blazar", ts.base))
        .send()
        .await
        .unwrap();
    assert_eq!(anon.status(), 401);
    // F4 census rule: /api/usage is advertised in the ollama lane AND
    // is a mounted route serving the rollup shape (empty is honest -
    // no traffic has flown through this test server).
    let ollama_eps: Vec<&str> = v["endpoints"]["ollama"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|e| e.as_str())
        .collect();
    assert!(
        ollama_eps.contains(&"/api/usage"),
        "census missing /api/usage"
    );
    blazar_core::tls::ensure_tls_provider();
    let usage: serde_json::Value = reqwest::Client::new()
        .get(format!("{}/api/usage", ts.base))
        .bearer_auth("plm_admin")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(usage["object"], "blazar.usage.daily");
    assert_eq!(usage["days_window"], 7);
    assert!(
        usage["days"].as_array().is_some(),
        "days must be an array even when empty"
    );
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

#[tokio::test]
async fn client__ollama_options_spec__invalid_mode_teaches_vocabulary() {
    // `options.spec` is the per-request spec-mode override (REPL
    // --no-draft rides it). A typo must 400 with the full valid-mode
    // list - NOT fall through to the generic catalog pair, which would
    // silently serve the wrong spawn shape.
    let ts = start(support::config_with_keys()).await;
    let c = client();
    let r: serde_json::Value = c
        .post(format!("{}/api/chat", ts.base))
        .bearer_auth("plm_admin")
        .json(&serde_json::json!({
            "model": "m1", "stream": false,
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"spec": "eagle"},
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let err = r["error"].as_str().unwrap_or_default();
    assert!(
        err.contains("options.spec must be one of") && err.contains("\"eagle3\""),
        "teaching 400 must list the vocabulary: {r}"
    );
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
async fn client__ollama_options_spec__off_shapes_the_spawn_and_ps_reports_it() {
    // A valid `options.spec` rides the request: the instance spawns
    // dense for it, and /api/ps surfaces the effective mode so the
    // shape is never a guess.
    let ts = start(support::config_with_keys()).await;
    let c = client();
    let r: serde_json::Value = c
        .post(format!("{}/api/chat", ts.base))
        .bearer_auth("plm_admin")
        .json(&serde_json::json!({
            "model": "m1", "stream": false,
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"spec": "off"},
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(r.get("message").is_some(), "chat answered: {r}");
    let ps: serde_json::Value = c
        .get(format!("{}/api/ps", ts.base))
        .bearer_auth("plm_admin")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let row = ps["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["name"].as_str() == Some("m1"))
        .expect("m1 instance row");
    assert_eq!(row["blazar_spec"].as_str(), Some("off"));
    // Cache-hit surface: the key exists on every row (null until the
    // gateway has classified a completed response for the model -
    // the earlier chat in this test may or may not have carried
    // usage, so only the key shape is pinned here).
    assert!(row.get("blazar_cache_hit").is_some());
    ts.state.sup.shutdown_all().await.unwrap();
}

#[tokio::test]
async fn client__audio_speech__teaches_install_and_unknown_formats() {
    // Hermetic harness dir carries no piper install: the route must
    // name the exact remedy. Lossy formats are first-class (ffmpeg
    // lane), so an accepted format rides through to the same install
    // teaching, while a genuinely unknown format must teach the full
    // vocabulary instead of silently falling back to WAV.
    let ts = start(support::config_with_keys()).await;
    let c = client();
    let resp = c
        .post(format!("{}/v1/audio/speech", ts.base))
        .bearer_auth("plm_admin")
        .json(&serde_json::json!({
            "model": "en_US-amy-medium", "input": "hello"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let body = resp.text().await.unwrap();
    assert!(
        body.contains("blazar tts --install"),
        "missing-lane teaching: {body}"
    );
    let resp = c
        .post(format!("{}/v1/audio/speech", ts.base))
        .bearer_auth("plm_admin")
        .json(&serde_json::json!({
            "model": "en_US-amy-medium", "input": "hi",
            "response_format": "mp3"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let body = resp.text().await.unwrap();
    assert!(
        body.contains("blazar tts --install"),
        "accepted lossy format reaches the install teaching, not a format rejection: {body}"
    );
    let resp = c
        .post(format!("{}/v1/audio/speech", ts.base))
        .bearer_auth("plm_admin")
        .json(&serde_json::json!({
            "model": "en_US-amy-medium", "input": "hi",
            "response_format": "wma"
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body = resp.text().await.unwrap();
    assert!(
        body.contains("wav") && body.contains("mp3") && body.contains("pcm16"),
        "unknown-format teaching lists the full vocabulary: {body}"
    );
    let resp = c
        .post(format!("{}/v1/audio/speech", ts.base))
        .bearer_auth("plm_admin")
        .json(&serde_json::json!({
            "model": "en_US-amy-medium", "input": "hi", "speed": 9.0
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body = resp.text().await.unwrap();
    assert!(body.contains("0.25..=4.0"), "speed range teaching: {body}");
}

/// Native llama-server surface parity (upstream audit 2026-10-05):
/// /infill, /v1/systemone, /completion, and /embeddings bodies carry no
/// `model` field upstream - direct-server clients post them bare.
/// Blazar must resolve a target (X-Blazar-Model > ?model= > single hot
/// child) and forward, not answer the OpenAI-dialect hard 400.
#[tokio::test]
async fn client__native_modelless_surfaces__header_target_forwards() {
    let stub = support::spawn_remote_stub().await;
    let cfg = support::config_with_keys();
    let cfg = support::with_remote(cfg, "far", &stub.base);
    let ts = start(cfg).await;
    let c = client();
    for path in ["/infill", "/v1/systemone", "/completion", "/embeddings"] {
        let r = c
            .post(format!("{}{path}", ts.base))
            .bearer_auth("plm_admin")
            .header("x-blazar-model", "far:m1")
            .json(&serde_json::json!({"input_prefix": "fn main() {"}))
            .send()
            .await
            .unwrap();
        assert_eq!(
            r.status(),
            200,
            "{path}: model-less native body must forward via X-Blazar-Model"
        );
    }
    ts.state.sup.shutdown_all().await.unwrap();
    stub.shutdown().await;
}

/// Same native surfaces, `?model=` query resolution - the second rung
/// of the `scoped_proxy` chain.
#[tokio::test]
async fn client__native_modelless_surfaces__query_target_forwards() {
    let stub = support::spawn_remote_stub().await;
    let cfg = support::config_with_keys();
    let cfg = support::with_remote(cfg, "far", &stub.base);
    let ts = start(cfg).await;
    let c = client();
    let r = c
        .post(format!("{}/completion?model=far:m1", ts.base))
        .bearer_auth("plm_admin")
        .json(&serde_json::json!({"prompt": "the quick brown"}))
        .send()
        .await
        .unwrap();
    assert_eq!(
        r.status(),
        200,
        "?model= must resolve the native /completion target"
    );
    ts.state.sup.shutdown_all().await.unwrap();
    stub.shutdown().await;
}

/// llama-server treats /rerank, /reranking, /v1/rerank and /v1/reranking
/// as ONE endpoint; every spelling must forward to the child (rerank
/// bodies carry the model, so resolution is the normal body-model path).
#[tokio::test]
async fn client__rerank_alias_spellings__forward_to_child() {
    let stub = support::spawn_remote_stub().await;
    let cfg = support::config_with_keys();
    let cfg = support::with_remote(cfg, "far", &stub.base);
    let ts = start(cfg).await;
    let c = client();
    for path in ["/rerank", "/reranking", "/v1/rerank", "/v1/reranking"] {
        let r = c
            .post(format!("{}{path}", ts.base))
            .bearer_auth("plm_admin")
            .json(&serde_json::json!({
                "model": "far:m1",
                "query": "what is a panda?",
                "documents": ["hi", "it is a bear"],
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(
            r.status(),
            200,
            "{path}: upstream rerank spelling must forward"
        );
    }
    ts.state.sup.shutdown_all().await.unwrap();
    stub.shutdown().await;
}

/// Pin the OTHER side of the audit fix: OpenAI-dialect paths have no
/// fallback - their clients always name the model, and a silent
/// fallback would route a typo to the wrong lane.
#[tokio::test]
async fn client__openai_dialect__missing_model_still_400() {
    let ts = start(support::config_with_keys()).await;
    let c = client();
    for path in ["/v1/chat/completions", "/v1/completions"] {
        let r = c
            .post(format!("{}{path}", ts.base))
            .bearer_auth("plm_admin")
            .json(&serde_json::json!({"prompt": "hi"}))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 400, "{path}: missing model stays a hard 400");
        let body = r.text().await.unwrap();
        assert!(
            body.contains("missing `model` field"),
            "{path}: teaching text preserved: {body}"
        );
    }
    ts.state.sup.shutdown_all().await.unwrap();
}
