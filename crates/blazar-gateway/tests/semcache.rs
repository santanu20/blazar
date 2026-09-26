//! R4 semantic cache — integration tests over the real router + stub
//! child (which serves `/tokenize` + legacy `/embedding`): full-stack
//! miss→store→hit flow, header bypass, malformed overrides, disabled
//! default, stream bypass, and the metrics surface.

#![allow(non_snake_case)]

mod support;

use blazar_core::{Config, SemanticCacheConfig};
use support::{client, start};

fn sem_config() -> Config {
    Config {
        semantic_cache: SemanticCacheConfig {
            enabled: true,
            model: Some("m1".into()),
            ..SemanticCacheConfig::default()
        },
        ..Config::default()
    }
}

fn chat_body(content: &str) -> serde_json::Value {
    serde_json::json!({
        "model": "m1",
        "stream": false,
        "messages": [{"role": "user", "content": content}]
    })
}

#[tokio::test]
async fn integration__semcache__miss_store_then_hit_full_stack() {
    let ts = start(sem_config()).await;
    let c = client();

    // 1st request: miss (computed live, stored, decorated).
    let r: serde_json::Value = c
        .post(format!("{}/api/chat", ts.base))
        .json(&chat_body("hello semantic cache world"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r["cache_debug"]["cache_hit"], false);
    assert!(r["cache_debug"]["cache_id"].as_u64().is_some());
    assert_eq!(
        ts.state
            .sem
            .misses
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );

    // Identical request: hit with similarity 1.0 (deterministic stub embed).
    let resp = c
        .post(format!("{}/api/chat", ts.base))
        .json(&chat_body("hello semantic cache world"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers()
            .get("x-blazar-cache")
            .and_then(|v| v.to_str().ok()),
        Some("hit; similarity=1.000")
    );
    let r: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(r["cache_debug"]["cache_hit"], true);
    assert_eq!(r["cache_debug"]["hit_type"], "semantic");
    assert!(
        (r["cache_debug"]["similarity"].as_f64().unwrap() - 1.0).abs() < 1e-6,
        "similarity: {}",
        r["cache_debug"]["similarity"]
    );
    assert_eq!(
        ts.state.sem.hits.load(std::sync::atomic::Ordering::Relaxed),
        1
    );

    // Different prompt: below threshold (orthogonal stub embeds) → miss.
    let r: serde_json::Value = c
        .post(format!("{}/api/chat", ts.base))
        .json(&chat_body("completely unrelated different words here"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r["cache_debug"]["cache_hit"], false);
}

#[tokio::test]
async fn integration__semcache__same_prompt_different_sampling_class_misses() {
    let ts = start(sem_config()).await;
    let c = client();
    let mut cold = chat_body("hello semantic cache world");
    cold["options"] = serde_json::json!({"temperature": 0.1});
    let r: serde_json::Value = c
        .post(format!("{}/api/chat", ts.base))
        .json(&cold)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r["cache_debug"]["cache_hit"], false);

    // Same prompt text, different generation problem (temperature): the
    // similarity is 1.0 but the class key differs — must be served live.
    let mut hot = chat_body("hello semantic cache world");
    hot["options"] = serde_json::json!({"temperature": 0.9});
    let resp = c
        .post(format!("{}/api/chat", ts.base))
        .json(&hot)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers()
            .get("x-blazar-cache")
            .and_then(|v| v.to_str().ok()),
        Some("miss")
    );
    let r: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(r["cache_debug"]["cache_hit"], false);
    assert_eq!(
        ts.state
            .sem
            .misses
            .load(std::sync::atomic::Ordering::Relaxed),
        2
    );
    assert_eq!(
        ts.state.sem.hits.load(std::sync::atomic::Ordering::Relaxed),
        0
    );

    // Class isolation is bidirectional: the original class still hits
    // over the now-two stored entries of identical prompt text.
    let resp = c
        .post(format!("{}/api/chat", ts.base))
        .json(&cold)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers()
            .get("x-blazar-cache")
            .and_then(|v| v.to_str().ok()),
        Some("hit; similarity=1.000")
    );
}

#[tokio::test]
async fn integration__semcache__header_off_bypasses() {
    let ts = start(sem_config()).await;
    let c = client();
    // Prime the cache with one request.
    let _: serde_json::Value = c
        .post(format!("{}/api/chat", ts.base))
        .json(&chat_body("hello semantic cache world"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    // Same prompt with x-blazar-cache: off — served live, no cache_debug.
    let resp = c
        .post(format!("{}/api/chat", ts.base))
        .header("x-blazar-cache", "off")
        .json(&chat_body("hello semantic cache world"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert!(resp.headers().get("x-blazar-cache").is_none());
    let r: serde_json::Value = resp.json().await.unwrap();
    assert!(r.get("cache_debug").is_none());
}

#[tokio::test]
async fn integration__semcache__stream_bypasses_even_with_config_on() {
    let ts = start(sem_config()).await;
    let c = client();
    let body = serde_json::json!({
        "model": "m1",
        "stream": true,
        "messages": [{"role": "user", "content": "hello semantic cache world"}]
    });
    let resp = c
        .post(format!("{}/api/chat", ts.base))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    // No cache activity at all for streams.
    assert_eq!(
        ts.state
            .sem
            .misses
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    assert_eq!(
        ts.state
            .sem
            .stores
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
}

#[tokio::test]
async fn integration__semcache__malformed_override_headers_400() {
    let ts = start(sem_config()).await;
    let c = client();
    for (h, v) in [
        ("x-blazar-cache", "maybe"),
        ("x-blazar-cache-ttl", "0"),
        ("x-blazar-cache-threshold", "1.5"),
    ] {
        let resp = c
            .post(format!("{}/api/chat", ts.base))
            .header(h, v)
            .json(&chat_body("hello semantic cache world"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 400, "header {h}={v} should 400");
    }
}

#[tokio::test]
async fn integration__semcache__disabled_default_and_missing_model_bypass() {
    // Default config: disabled, no embed model — request flows unchanged.
    let ts = start(Config::default()).await;
    let c = client();
    let resp = c
        .post(format!("{}/api/chat", ts.base))
        .json(&chat_body("hello semantic cache world"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert!(resp.headers().get("x-blazar-cache").is_none());
    let r: serde_json::Value = resp.json().await.unwrap();
    assert!(r.get("cache_debug").is_none());

    // Header on but no embed model configured → silent bypass, not 400.
    let resp = c
        .post(format!("{}/api/chat", ts.base))
        .header("x-blazar-cache", "on")
        .json(&chat_body("hello semantic cache world"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn integration__semcache__structured_output_bypasses() {
    // R9 interaction pin: grammar/format constrain the response space,
    // so constrained requests never consult or feed the semantic cache
    // (live bug 2026-09-09: malformed-grammar request got a cached 200
    // instead of the child's 400).
    let ts = start(sem_config()).await;
    let c = client();

    // Prime the cache with an unconstrained request.
    let _: serde_json::Value = c
        .post(format!("{}/api/chat", ts.base))
        .json(&chat_body("hello semantic cache world"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        ts.state
            .sem
            .misses
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );

    // Same prompt + grammar: no lookup (misses stay 1), no cache_debug,
    // and the response is computed live — nothing stored either.
    let mut body = chat_body("hello semantic cache world");
    body["grammar"] = serde_json::json!("root ::= \"yes\" | \"no\"");
    let resp = c
        .post(format!("{}/api/chat", ts.base))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert!(resp.headers().get("x-blazar-cache").is_none());
    let r: serde_json::Value = resp.json().await.unwrap();
    assert!(r.get("cache_debug").is_none());
    assert_eq!(
        ts.state
            .sem
            .misses
            .load(std::sync::atomic::Ordering::Relaxed),
        1,
        "constrained request must not count as a cache miss"
    );
    assert_eq!(
        ts.state
            .sem
            .stores
            .load(std::sync::atomic::Ordering::Relaxed),
        1,
        "constrained response must not be stored"
    );

    // Same prompt + format schema: same bypass.
    let mut body = chat_body("hello semantic cache world");
    body["format"] = serde_json::json!({"type": "object"});
    let resp = c
        .post(format!("{}/api/chat", ts.base))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let r: serde_json::Value = resp.json().await.unwrap();
    assert!(r.get("cache_debug").is_none());
    assert_eq!(
        ts.state
            .sem
            .misses
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );

    // Unconstrained repeat still hits the primed entry (cache intact).
    let resp = c
        .post(format!("{}/api/chat", ts.base))
        .json(&chat_body("hello semantic cache world"))
        .send()
        .await
        .unwrap();
    let r: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(r["cache_debug"]["cache_hit"], true);
}

#[tokio::test]
async fn integration__semcache__metrics_surface() {
    let ts = start(sem_config()).await;
    let c = client();
    let _: serde_json::Value = c
        .post(format!("{}/api/chat", ts.base))
        .json(&chat_body("metrics probe one"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let m = c
        .get(format!("{}/metrics", ts.base))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(m.contains("blazar_semantic_cache_misses_total 1"), "{m}");
    assert!(m.contains("blazar_semantic_cache_stores_total 1"), "{m}");
    assert!(m.contains("blazar_semantic_cache_entries 1"), "{m}");
    assert!(m.contains("blazar_semantic_cache_hits_total 0"), "{m}");
}
