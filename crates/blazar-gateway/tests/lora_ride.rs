//! llama.cpp LoRA-variant ride: `base+adapter` generation requests ride
//! the resident base child with the per-request completion-schema
//! `lora` field, instead of a dedicated variant child re-loading the
//! base weights. The stub reflects the stamp it received, so the
//! round-trip is proven at the child boundary.

#![allow(non_snake_case)]

mod support;

#[tokio::test]
async fn gateway__llamacpp_lora_variant__rides_dense_child_with_request_stamp() {
    let ts = support::start(support::config_with_keys()).await;
    // Adapter file + store row BEFORE first spawn: the child comes up
    // with --lora in argv, and the stub seeds its /lora-adapters state
    // from that argv (server.cpp fidelity).
    let adapter = ts.dirs.data_dir.join("anonymizer.gguf");
    std::fs::write(&adapter, b"gguf").unwrap();
    let store = blazar_core::Store::open(&ts.dirs).unwrap();
    store
        .add_lora("m1", &adapter.display().to_string(), 1.0)
        .unwrap();

    let c = support::client();
    // Base request first: one dense resident.
    let r = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .bearer_auth("plm_admin")
        .json(&serde_json::json!({
            "model": "m1",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let _: serde_json::Value = r.json().await.unwrap();

    // Variant request rides the SAME child: the stub reflects the
    // per-request lora stamp it received (id 0 = argv order, scale 1.0).
    let r = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .bearer_auth("plm_admin")
        .json(&serde_json::json!({
            "model": "m1+anonymizer",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let v: serde_json::Value = r.json().await.unwrap();
    let content = v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();
    assert!(
        content.contains("[lora-reflect 0@1]"),
        "the per-request lora stamp must reach the child: {content}"
    );

    // Exactly one resident served both requests - no dedicated variant
    // child, no second weights load.
    let ps = ts.state.sup.ps();
    assert_eq!(ps.len(), 1, "variant must ride the dense child: {ps:?}");
    assert_eq!(ps[0].name, "m1");
}

#[tokio::test]
#[allow(non_snake_case)]
async fn gateway__llamacpp_lora_variant__missing_adapter_teaches_not_silent_base() {
    let ts = support::start(support::config_with_keys()).await;
    // No adapter attached: `m1+ghost` must teach, never silently serve
    // base weights.
    let c = support::client();
    let r = c
        .post(format!("{}/v1/chat/completions", ts.base))
        .bearer_auth("plm_admin")
        .json(&serde_json::json!({
            "model": "m1+ghost",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false
        }))
        .send()
        .await
        .unwrap();
    let status = r.status();
    let body: serde_json::Value = r.json().await.unwrap_or(serde_json::Value::Null);
    assert!(
        status == 404 || status == 400,
        "missing adapter must teach: {status} {body}"
    );
    let msg = body["error"]["message"]
        .as_str()
        .or_else(|| body["error"].as_str())
        .unwrap_or_default();
    assert!(
        msg.contains("blazar lora add") && msg.contains("m1+ghost"),
        "teaching lost the remedy: {msg}"
    );
}
