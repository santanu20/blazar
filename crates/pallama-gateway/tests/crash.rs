//! Crash-respawn contract at the GATEWAY boundary: a crashed engine
//! child must be reaped on demand by the first failed request, not on
//! the 10s reaper tick. The harness disables the tick (1h interval), so
//! any recovery observed here is the on-demand reap path alone.
#![allow(unsafe_code)] // SAFETY: only libc::kill against our own child pid below

mod support;

use std::time::{Duration, Instant};

use pallama_core::Config;

fn chat_body(n: u32) -> serde_json::Value {
    serde_json::json!({
        "model": "m1",
        "max_tokens": 8,
        "stream": false,
        "messages": [{"role": "user", "content": format!("Say ok. ({n})")}],
    })
}

async fn chat(base: &str, n: u32) -> u16 {
    support::client()
        .post(format!("{base}/v1/chat/completions"))
        .json(&chat_body(n))
        .send()
        .await
        .expect("request sent")
        .status()
        .as_u16()
}

#[tokio::test]
#[allow(non_snake_case)]
async fn integration__crash__on_demand_reap_respawns_within_one_retry() {
    let server = support::start(Config::default()).await;
    // Warm: child up and serving.
    assert_eq!(chat(&server.base, 0).await, 200);

    // Kill -9 the engine child by exact pid (simulates an engine crash).
    let pid = server.state.sup.ps()[0].pid;
    unsafe { libc::kill(i32::try_from(pid).unwrap_or(-1), libc::SIGKILL) };
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Retry loop: request 1 may hit the stale entry (502, its error path
    // reaps the corpse); the NEXT ensure() must respawn and serve. The
    // reaper tick is 1h here — no recovery within budget = the on-demand
    // contract regressed to periodic-only.
    let t0 = Instant::now();
    let mut timeline: Vec<(u128, u16)> = Vec::new();
    let recovered = loop {
        let st = chat(
            &server.base,
            u32::try_from(timeline.len() + 1).unwrap_or(u32::MAX),
        )
        .await;
        timeline.push((t0.elapsed().as_millis(), st));
        if st == 200 {
            break true;
        }
        if t0.elapsed() > Duration::from_secs(8) {
            break false;
        }
        tokio::time::sleep(Duration::from_millis(150)).await;
    };
    let fifties = timeline.iter().filter(|(_, s)| *s == 502).count();
    let new_pid = server.state.sup.ps().first().map(|r| r.pid);

    assert!(recovered, "no recovery in 8s (tick disabled): {timeline:?}");
    assert!(
        (1..=3).contains(&fifties),
        "expected 1-3 bridging 502s, got {fifties}: {timeline:?}"
    );
    assert_ne!(new_pid, Some(pid), "must serve from a respawned child");
}
