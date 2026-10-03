//! Opt-in download speed cap (token bucket, shared across a pull's
//! connections). Ollama #2006 complaint class: a model pull saturating
//! a shared link. `0` (the default) constructs no bucket at all, so the
//! uncapped path stays byte-identical — pacing only exists when asked
//! for via `download_speed_limit_mb`.

use std::time::Duration;
use tokio::sync::Mutex;

/// One second of full-rate bytes: the burst allowance. Small enough
/// that a cap holds on interactive links, large enough that per-read
/// jitter never stalls a fast LAN pull that happens to be capped high.
const BURST_SECS: f64 = 1.0;

struct BucketState {
    tokens: f64,
    updated: std::time::Instant,
}

/// Shared bytes/sec limiter. Clone the `Arc`, call [`Throttle::acquire`]
/// after each successful read of `bytes` (read-then-sleep: data has
/// landed, the pause throttles the NEXT request window, and transport
/// buffers absorb it instead of erroring).
///
/// The bucket guard is held ACROSS the deficit sleep, so parallel
/// download workers serialize at the bucket: deficits carry as shared
/// negative debt, and each waiter's elapsed-time refill pays down that
/// debt (capped at one burst) before its own bytes are charged. Sleeping
/// outside the lock instead let every concurrent worker refill to a
/// fresh burst and run the pull at roughly `cap × active workers` — the
/// aggregate cap is the whole point, so the lock has to own the wait.
pub struct Throttle {
    rate_bytes_per_sec: f64,
    state: Mutex<BucketState>,
}

/// Pure delay math, split out so the contract is unit-testable without
/// a clock: given the pre-refill state, how long must this read wait?
///
/// * refill `tokens` for the elapsed time (capped at one full burst),
/// * spend `bytes` from the bucket,
/// * a deficit carries over as negative tokens and divides by the rate
///   into a wait — the debt is shared, so concurrent waiters cannot
///   ride each other's wait windows.
#[must_use]
pub fn bucket_delay(
    tokens: f64,
    elapsed: Duration,
    bytes: f64,
    rate_bytes_per_sec: f64,
) -> (f64, Duration) {
    debug_assert!(rate_bytes_per_sec.is_finite() && rate_bytes_per_sec > 0.0);
    let capacity = rate_bytes_per_sec * BURST_SECS;
    let refill = elapsed.as_secs_f64() * rate_bytes_per_sec;
    let tokens = (tokens + refill).min(capacity);
    let remaining = tokens - bytes;
    if remaining >= 0.0 {
        (remaining, Duration::ZERO)
    } else {
        // Carry the deficit as NEGATIVE tokens: the next acquirer must
        // wait out its share of the same debt. Clamping to 0 instead let
        // a sleeping worker's wait window regenerate a fresh allowance
        // for the next worker while the sleeper's own bytes still flowed
        // at wake-up — every window carried two reads and the aggregate
        // rate ran at cap x (1 + freeloaders).
        let wait = (-remaining) / rate_bytes_per_sec;
        (remaining, Duration::from_secs_f64(wait))
    }
}

impl Throttle {
    /// `limit_bytes_per_sec <= 0` or non-finite means "no cap": callers
    /// store `None` and skip pacing entirely.
    #[must_use]
    pub fn shared(limit_bytes_per_sec: f64) -> Option<std::sync::Arc<Self>> {
        if !limit_bytes_per_sec.is_finite() || limit_bytes_per_sec <= 0.0 {
            return None;
        }
        let rate = limit_bytes_per_sec;
        Some(std::sync::Arc::new(Self {
            rate_bytes_per_sec: rate,
            state: Mutex::new(BucketState {
                tokens: rate * BURST_SECS,
                updated: std::time::Instant::now(),
            }),
        }))
    }

    /// Paced sleep for a read of `bytes`. Holds the bucket guard across
    /// the sleep (see the struct docs for why); uncontended cost is one
    /// lock round-trip.
    // refill + deficit + sleep accounting read best as one audited block
    #[allow(clippy::too_many_lines)]
    pub async fn acquire(&self, bytes: u64) {
        let mut st = self.state.lock().await;
        let elapsed = st.updated.elapsed();
        // byte counts are bounded far below 2^52; f64 is the bucket unit
        #[allow(clippy::cast_precision_loss)]
        let requested = bytes as f64;
        let (tokens, wait) = bucket_delay(st.tokens, elapsed, requested, self.rate_bytes_per_sec);
        // Stamp BEFORE sleeping: the wait window is what earns the spent
        // tokens back for the next waiter, so elapsed-time refill in the
        // next acquire lands on the deficit just paid.
        st.tokens = tokens;
        st.updated = std::time::Instant::now();
        if !wait.is_zero() {
            tokio::time::sleep(wait).await;
        }
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // tiny consts
mod tests {
    use super::*;

    #[test]
    fn unit__bucket_delay__first_read_within_burst_is_free() {
        // Fresh bucket holds one full second of bytes: a smaller first
        // read must not wait.
        let rate = 1000.0;
        let (tokens, wait) = bucket_delay(rate, Duration::ZERO, 400.0, rate);
        assert_eq!(wait, Duration::ZERO);
        assert!((tokens - 600.0).abs() < f64::EPSILON, "spent exactly");
    }

    #[test]
    fn unit__bucket_delay__deficit_waits_deficit_over_rate() {
        let rate = 1000.0; // bytes/sec
        let (_, wait) = bucket_delay(0.0, Duration::ZERO, 2500.0, rate);
        assert_eq!(wait, Duration::from_millis(2500));
    }

    #[test]
    fn unit__bucket_delay__elapsed_refill_is_capped_at_one_burst() {
        // Hours of idle credit at most one second of bytes.
        let rate = 100.0;
        let (tokens, wait) = bucket_delay(0.0, Duration::from_secs(3600), 0.0, rate);
        assert_eq!(wait, Duration::ZERO);
        assert!((tokens - 100.0).abs() < f64::EPSILON, "no infinite burst");
    }

    #[test]
    fn unit__bucket_delay__refill_after_spend_is_gradual() {
        let rate = 1000.0;
        // Spent down to 0; 250ms elapsed refills 250 bytes.
        let (tokens, wait) = bucket_delay(0.0, Duration::from_millis(250), 0.0, rate);
        assert_eq!(wait, Duration::ZERO);
        assert!((tokens - 250.0).abs() < f64::EPSILON);
    }

    #[test]
    fn unit__throttle_shared__rejects_nonpositive_and_nonfinite() {
        assert!(Throttle::shared(0.0).is_none());
        assert!(Throttle::shared(-5.0).is_none());
        assert!(Throttle::shared(f64::NAN).is_none());
        assert!(Throttle::shared(f64::INFINITY).is_none());
        assert!(Throttle::shared(1.0).is_some());
    }

    #[tokio::test]
    async fn unit__acquire__paced_reads_sleep_the_deficit() {
        let t = Throttle::shared(1_000_000.0).expect("valid rate"); // 1 MB/s
        // Three immediate 500 KiB reads against a 1 MiB burst: the first
        // two drain the allowance, the third owes its full 500 KiB ->
        // ~0.5 s (CI jitter margin in the assert).
        t.acquire(500_000).await;
        t.acquire(500_000).await;
        let t0 = std::time::Instant::now();
        t.acquire(500_000).await;
        assert!(
            t0.elapsed() >= Duration::from_millis(400),
            "third read paced, took {:?}",
            t0.elapsed()
        );
    }

    // Regression pin for the parallel-lane defect the live e2e caught:
    // when each worker slept its deficit OUTSIDE the lock, the bucket
    // refilled to a fresh burst for every waiter and a capped pull ran
    // at roughly cap x active workers. Holding the guard across the
    // sleep, N workers each charging B bytes must take at least
    // (N x B - burst) / rate of wall clock.
    #[tokio::test]
    async fn unit__acquire__concurrent_reads_enforce_the_aggregate_cap() {
        const WORKERS: u64 = 8;
        const BYTES_EACH: u64 = 250_000;
        let rate = 1_000_000.0; // 1 MB/s, burst = 1 MB
        let t = Throttle::shared(rate).expect("valid rate");
        let start = std::time::Instant::now();
        let mut handles = Vec::new();
        for _ in 0..WORKERS {
            let t = std::sync::Arc::clone(&t);
            handles.push(tokio::spawn(async move {
                t.acquire(BYTES_EACH).await;
            }));
        }
        for h in handles {
            h.await.expect("worker finished");
        }
        let total = f64::from(WORKERS as u32) * BYTES_EACH as f64;
        let floor_secs = ((total - rate * BURST_SECS) / rate).max(0.0);
        assert!(
            start.elapsed() >= Duration::from_secs_f64(floor_secs * 0.8),
            "aggregate paced: floor {:?}, took {:?}",
            Duration::from_secs_f64(floor_secs),
            start.elapsed()
        );
    }
}
