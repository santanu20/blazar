//! Crash-respawn contract at the GATEWAY boundary: a request whose
//! forward dies in the crash window (child killed between health gate
//! and send) must respawn the exact lane and retry ITSELF in-band —
//! single-shot clients never observe the 502. The harness disables the
//! reaper tick (1h interval), so any recovery observed here is the
//! on-demand reap + in-band retry path alone.
#![allow(unsafe_code)] // SAFETY: only libc::kill against our own child pid below (unix)

mod support;

use std::time::{Duration, Instant};

use blazar_core::Config;

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
async fn integration__crash__in_band_respawn_retry_serves_first_post_crash_request() {
    let server = support::start(Config::default()).await;
    // Warm: child up and serving.
    assert_eq!(chat(&server.base, 0).await, 200);

    // Kill -9 the engine child by exact pid (simulates an engine crash).
    // taskkill /T on Windows: TerminateProcess has no posix-signal
    // equivalent, and the pid is owned by the server, not this test.
    let pid = server.state.sup.ps()[0].pid;
    #[cfg(unix)]
    unsafe {
        libc::kill(i32::try_from(pid).unwrap_or(-1), libc::SIGKILL)
    };
    #[cfg(windows)]
    let _ = std::process::Command::new("taskkill")
        .args(["/F", "/T", "/PID", &pid.to_string()])
        .status();
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Contract (upgraded from bridging-502s): the FIRST request after
    // the crash self-heals — the failed forward reaps the corpse,
    // respawns the exact instance lane, and retries once in-band. A 502
    // here means the crash-window retry regressed to next-request-only
    // recovery. The reaper tick is 1h here, so nothing else can save it.
    let t0 = Instant::now();
    let st = chat(&server.base, 1).await;
    let took = t0.elapsed();
    let new_pid = server.state.sup.ps().first().map(|r| r.pid);

    assert_eq!(st, 200, "first post-crash request must self-heal, got {st}");
    assert!(
        took < Duration::from_secs(8),
        "in-band respawn+retry too slow: {took:?}"
    );
    assert_ne!(new_pid, Some(pid), "must serve from a respawned child");

    // And the lane stays healthy for the next client (no half-state).
    assert_eq!(chat(&server.base, 2).await, 200);
}
