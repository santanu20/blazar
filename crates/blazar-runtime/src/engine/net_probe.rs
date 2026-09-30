//! Bounded retry for upstream metadata probes (GitHub API, `PyPI`).
//!
//! Consumer ISPs intermittently reset or blackhole TLS connections to
//! these endpoints — observed live: back-to-back curls to pypi.org where
//! one connection died mid-TLS and the next returned 200 instantly;
//! api.github.com unreachable at a 20 s cap, then serving normally
//! minutes later. Single-shot probes turn that transient into a spurious
//! "cannot check upstream" failure.
//!
//! Policy (bounded, explicit): 3 attempts, flat 500 ms backoff — flat
//! because each probe is a lone caller re-issuing one small idempotent
//! GET, no herd to de-synchronize (same reasoning as the spawn retry in
//! `probe.rs`). Only transport-class failures re-issue: send errors,
//! the per-attempt deadline, 5xx, and body decodes interrupted
//! mid-stream. Deterministic answers (4xx: rate limits, missing
//! repos/tags) fail fast — re-issuing past a 403 is exactly what the
//! rate limit asks us not to do.
//!
//! Probes wrapped here self-bound (attempts × cap + backoff, ~31.5 s
//! worst case, one attempt on a healthy network): do NOT stack another
//! `tokio::time::timeout` on top — an outer cap would silently cut the
//! retry budget down to its own deadline. Asset downloads are out of
//! scope: they are 30-400 MB streams with their own resume logic, not
//! idempotent metadata GETs.

use std::future::Future;
use std::time::Duration;

use anyhow::Result;

/// Re-issues per probe: one fresh try plus two bounded retries.
pub(crate) const PROBE_ATTEMPTS: u32 = 3;
/// Deadline for a single attempt, covering send through body decode.
pub(crate) const PROBE_ATTEMPT_CAP: Duration = Duration::from_secs(10);
/// Pause between attempts (see module docs for why flat).
pub(crate) const PROBE_BACKOFF: Duration = Duration::from_millis(500);

/// One probe attempt's outcome: `Done` is a final answer (success or a
/// deterministic failure such as a rate limit or 404), `Retry` marks a
/// transient worth one more attempt.
pub(crate) enum Attempt<T> {
    Done(Result<T>),
    Retry(anyhow::Error),
}

/// Run `attempt` under the bounded probe policy: on a healthy network
/// the first attempt settles it with zero added latency; transients are
/// re-issued after [`PROBE_BACKOFF`] until [`PROBE_ATTEMPTS`] run out,
/// and the final transient error surfaces as-is.
pub(crate) async fn retry_probe<T, F, Fut>(mut attempt: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Attempt<T>>,
{
    let mut last_err: Option<anyhow::Error> = None;
    for n in 1..=PROBE_ATTEMPTS {
        match attempt().await {
            Attempt::Done(res) => return res,
            Attempt::Retry(err) => {
                if n < PROBE_ATTEMPTS {
                    tokio::time::sleep(PROBE_BACKOFF).await;
                }
                last_err = Some(err);
            }
        }
    }
    Err(last_err.expect("probe loop runs at least once"))
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use anyhow::anyhow;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn counter() -> AtomicU32 {
        AtomicU32::new(0)
    }

    #[tokio::test(start_paused = true)]
    async fn unit__retry_probe__first_attempt_success__single_call() {
        let calls = counter();
        let res: Result<u32> = retry_probe(|| {
            calls.fetch_add(1, Ordering::SeqCst);
            async { Attempt::Done(Ok(7)) }
        })
        .await;
        assert_eq!(res.unwrap(), 7);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn unit__retry_probe__transient_then_success__recovers() {
        let calls = counter();
        let res: Result<u32> = retry_probe(|| {
            let n = calls.fetch_add(1, Ordering::SeqCst);
            async move {
                if n < 2 {
                    Attempt::Retry(anyhow!("connection reset"))
                } else {
                    Attempt::Done(Ok(42))
                }
            }
        })
        .await;
        assert_eq!(res.unwrap(), 42);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test(start_paused = true)]
    async fn unit__retry_probe__all_transient__last_error_after_budget() {
        let calls = counter();
        let res: Result<()> = retry_probe(|| {
            let n = calls.fetch_add(1, Ordering::SeqCst);
            async move { Attempt::Retry(anyhow!("reset #{n}")) }
        })
        .await;
        assert_eq!(calls.load(Ordering::SeqCst), PROBE_ATTEMPTS);
        assert_eq!(format!("{:#}", res.unwrap_err()), "reset #2");
    }

    #[tokio::test(start_paused = true)]
    async fn unit__retry_probe__deterministic_failure__no_reissue() {
        let calls = counter();
        let res: Result<()> = retry_probe(|| {
            calls.fetch_add(1, Ordering::SeqCst);
            async { Attempt::Done(Err(anyhow!("GitHub API rate limited (403)"))) }
        })
        .await;
        assert_eq!(
            format!("{:#}", res.unwrap_err()),
            "GitHub API rate limited (403)"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
