//! Guided-onboarding pins: one-use invite codes authorize a single join
//! (even on tokened gateways), and the per-worker token minted at join
//! carries the session afterwards.

use blazar_core::Config;
use serde_json::json;

mod support;

#[tokio::test]
#[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
async fn integration__invite_code__one_join_then_consumed() {
    let config = Config {
        // A tokened gateway refuses bare joins from ANY peer; invite codes
        // are the guided-onboarding bootstrap around exactly that.
        fleet_token: "master-tok".into(),
        ..Config::default()
    };
    let ts = support::start(config).await;
    let client = reqwest::Client::new();

    // Without token or code: refused.
    let refused = client
        .post(format!("{}/api/fleet/join", ts.base))
        .json(&json!({"worker_id": "w1", "endpoints": ["127.0.0.1:59901"]}))
        .send()
        .await
        .unwrap();
    assert_eq!(refused.status(), 401, "tokened gateway refuses bare join");

    // Mint a code (the adopter on the gateway box is loopback-trusted).
    let minted = client
        .post(format!("{}/api/fleet/invite", ts.base))
        .json(&json!({"ttl_secs": 300, "token": "master-tok"}))
        .send()
        .await
        .unwrap();
    assert_eq!(minted.status(), 200);
    let code = minted.json::<serde_json::Value>().await.unwrap();
    let code = code["invite_code"].as_str().unwrap().to_string();
    assert_eq!(code.len(), 32, "32-hex invite code");

    // First join with the code: passes, and the ack carries a worker token.
    let joined = client
        .post(format!("{}/api/fleet/join", ts.base))
        .json(&json!({
            "worker_id": "w1",
            "endpoints": ["127.0.0.1:59901"],
            "token": code,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(joined.status(), 200, "invite code authorizes one join");
    let ack = joined.json::<serde_json::Value>().await.unwrap();
    let worker_token = ack["worker_token"].as_str().unwrap().to_string();
    assert_eq!(worker_token.len(), 32, "per-worker session token minted");

    // The SAME code is dead: one-use.
    let replay = client
        .post(format!("{}/api/fleet/join", ts.base))
        .json(&json!({
            "worker_id": "w2",
            "endpoints": ["127.0.0.1:59902"],
            "token": code,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(replay.status(), 401, "invite code consumed by first join");

    // The worker token carries the session: heartbeat passes with it,
    // fails with garbage.
    let beat = client
        .post(format!("{}/api/fleet/heartbeat", ts.base))
        .json(&json!({"worker_id": "w1", "token": worker_token}))
        .send()
        .await
        .unwrap();
    assert_eq!(beat.status(), 200, "worker token authorizes heartbeat");
    let bad = client
        .post(format!("{}/api/fleet/heartbeat", ts.base))
        .json(&json!({"worker_id": "w1", "token": "0".repeat(32)}))
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status(), 401, "wrong worker token refused");
}
