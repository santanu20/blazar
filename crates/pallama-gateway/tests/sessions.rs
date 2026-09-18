//! R3 session pins — integration tests over the real router + stub child:
//! header touch through the middleware, pass-through without the header,
//! idempotent close, and the /api/sessions listing.

#![allow(non_snake_case)]
#![allow(clippy::duration_suboptimal_units)]

mod support;

use std::time::Duration;

use support::{client, start};

#[tokio::test]
async fn integration__pin_mw__header_pins_model_full_stack() {
    let ts = start(pallama_core::Config::default()).await;
    let c = client();
    let body = serde_json::json!({"model": "m1", "messages": [
        {"role": "user", "content": "hi"}
    ]});
    let r = c
        .post(format!("{}/api/chat", ts.base))
        .header("x-pallama-session", "agent1")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    // Canonical model pinned through the whole stack (auth → … → pin → handler).
    assert!(
        ts.state
            .sup
            .sessions
            .pins("m1", Duration::from_secs(15 * 60))
            .live
    );

    // The listing reflects it with shape and TTL bookkeeping.
    let l: serde_json::Value = c
        .get(format!("{}/api/sessions", ts.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(l["sessions"].as_array().map(Vec::len), Some(1));
    assert_eq!(l["sessions"][0]["session"], "agent1");
    assert_eq!(l["sessions"][0]["model"], "m1");
    assert_eq!(l["keep_secs"], 900);

    // And the gauge follows the registry.
    let m = c
        .get(format!("{}/metrics", ts.base))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(m.contains("pallama_sessions_live 1"), "gauge in:\n{m}");
}

#[tokio::test]
async fn integration__pin_mw__absent_header_no_pin() {
    let ts = start(pallama_core::Config::default()).await;
    let c = client();
    let body = serde_json::json!({"model": "m1", "messages": [
        {"role": "user", "content": "hi"}
    ]});
    let r = c
        .post(format!("{}/api/chat", ts.base))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert!(ts
        .state
        .sup
        .sessions
        .list(Duration::from_secs(15 * 60))
        .is_empty());
}

#[tokio::test]
async fn integration__session_close__releases_pin_idempotently() {
    let ts = start(pallama_core::Config::default()).await;
    let c = client();
    let body = serde_json::json!({"model": "m1", "messages": [
        {"role": "user", "content": "hi"}
    ]});
    let r = c
        .post(format!("{}/api/chat", ts.base))
        .header("x-pallama-session", "agent1")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert!(
        ts.state
            .sup
            .sessions
            .pins("m1", Duration::from_secs(15 * 60))
            .live
    );

    // Close: no model, no filename — the pin is enough.
    let r: serde_json::Value = c
        .post(format!("{}/api/session", ts.base))
        .json(&serde_json::json!({"action": "close", "session": "agent1"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r["status"], "ok");
    assert_eq!(r["released"], true);
    assert!(ts
        .state
        .sup
        .sessions
        .list(Duration::from_secs(15 * 60))
        .is_empty());

    // Idempotent second close.
    let r: serde_json::Value = c
        .post(format!("{}/api/session", ts.base))
        .json(&serde_json::json!({"action": "close", "session": "agent1"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(r["released"], false);

    // Invalid names are rejected outright (same policy as checkpoints).
    let r = c
        .post(format!("{}/api/session", ts.base))
        .json(&serde_json::json!({"action": "close", "session": "../evil"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);
}

/// #20 identity manifests: save stamps a sibling manifest; restore
/// verifies the runtime shape and 400s on mismatch with a teaching
/// error; erase removes checkpoint AND manifest (no orphans).
#[tokio::test]
async fn integration__session_identity__manifest_written_tamper_blocks_restore() {
    let ts = start(pallama_core::Config::default()).await;
    let c = client();
    // Warm the child (save needs a live slot).
    let r = c
        .post(format!("{}/api/chat", ts.base))
        .json(&serde_json::json!({"model": "m1", "messages": [
            {"role": "user", "content": "hi"}
        ]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);

    // Save: 200 + manifest beside the checkpoint.
    let r = c
        .post(format!("{}/api/session", ts.base))
        .json(&serde_json::json!({"model": "m1", "action": "save", "filename": "conv1"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "stub slot save works");
    let ckpt = ts
        .dirs
        .sessions_dir()
        .join(pallama_core::profile::path_safe("m1"))
        .join("conv1");
    assert!(ckpt.exists(), "checkpoint file exists");
    let id = pallama_core::session_identity::read_manifest(&ckpt)
        .expect("identity manifest written and parses");
    assert!(!id.pallama_version.is_empty());

    // Clean restore: shape matches -> passes the gate.
    let r = c
        .post(format!("{}/api/session", ts.base))
        .json(&serde_json::json!({"model": "m1", "action": "restore", "filename": "conv1"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "matching identity restores");

    // Tampered shape: restore refused with a teaching 400.
    let mut bad = id.clone();
    bad.ctx += 1024;
    pallama_core::session_identity::write_manifest(&ckpt, &bad).unwrap();
    let r = c
        .post(format!("{}/api/session", ts.base))
        .json(&serde_json::json!({"model": "m1", "action": "restore", "filename": "conv1"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400);
    let text = r.text().await.unwrap();
    assert!(
        text.contains("different runtime shape") && text.contains("ctx"),
        "teaching error names the diff: {text}"
    );

    // Erase removes checkpoint AND manifest (H19: no orphaned metadata).
    let r = c
        .post(format!("{}/api/session", ts.base))
        .json(&serde_json::json!({"model": "m1", "action": "erase", "filename": "conv1"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert!(!ckpt.exists());
    assert!(!pallama_core::session_identity::manifest_path(&ckpt).exists());
}
