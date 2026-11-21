//! Per-request stage timing: where non-model time goes.
//!
//! The gateway already records end-to-end TTFT/TPOT quantiles and the
//! contended-only admission wait, but gateway-vs-direct overhead cannot
//! be attributed to a stage from those alone. This module stamps
//! boundary crossings on the request hot path and folds them into six
//! stage histograms:
//!
//! | Stage | Span | Contains |
//! |-------|------|----------|
//! | `auth` | received -> key resolved | bearer/x-api-key resolution + concurrency lease |
//! | `body` | auth -> model resolved | body transport + JSON parse + store resolve |
//! | `admission` | model resolved -> engine granted | supervisor ensure + priority-queue wait + cold spawn |
//! | `dispatch` | engine granted -> child headers | child send up to response headers |
//! | `model_ttft` | engine granted -> first stream byte | engine-side time-to-first-token |
//! | `stream` | first byte -> last byte | decode stream duration |
//!
//! Boundaries travel on a tokio task-local so the shared chokepoints
//! (`auth` middleware, `proxy::ensure_with_admission`, `proxy::child_send`,
//! the usage tap) stamp them without signature changes. The first four
//! spans sum to the gateway's own overhead; the last two are engine
//! time. A boundary that never fired leaves its stage unobserved for
//! that request - durations are never extrapolated across a gap.
//!
//! Stamps are one relaxed atomic each (~ns); observation happens once at
//! response completion, off the streaming path's per-chunk cost beyond
//! the two token stamps the tap already timed.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::histogram::Histogram;

tokio::task_local! {
    /// Boundary stamps for the request being served on this task. Set by
    /// the `request_log` middleware around the handler chain; absent on
    /// non-HTTP callers, where every mark is a no-op.
    static STAGE_TRACE: Arc<StageTrace>;
}

/// Sentinel for "boundary never fired" (a real stamp stores at least 1).
const UNSET: u64 = 0;

/// Boundary stamps for one request, elapsed-nanos since receipt. Every
/// slot is a relaxed atomic: stamps race nothing (set-once via CAS, or
/// last-wins for the final token), and observation reads each slot once
/// after the stream ends.
#[derive(Debug)]
pub struct StageTrace {
    received: Instant,
    /// Key resolution done (end of the `auth` middleware's gate work).
    auth_done: AtomicU64,
    /// Model row resolved (inside `ensure_with_admission`, after the
    /// store lookup and lane gates; everything since `auth_done` was
    /// body transport, parse and handler pre-work).
    routed: AtomicU64,
    /// Engine reference granted (ensure + any priority-queue wait done).
    admitted: AtomicU64,
    /// Child produced response headers (`child_send` succeeded).
    dispatched: AtomicU64,
    /// First body byte of the generation stream reached the tap.
    first_token: AtomicU64,
    /// Last body byte seen by the tap (plain store - last wins).
    last_token: AtomicU64,
}

impl Default for StageTrace {
    fn default() -> Self {
        Self::new()
    }
}

impl StageTrace {
    #[must_use]
    pub fn new() -> Self {
        Self {
            received: Instant::now(),
            auth_done: AtomicU64::new(UNSET),
            routed: AtomicU64::new(UNSET),
            admitted: AtomicU64::new(UNSET),
            dispatched: AtomicU64::new(UNSET),
            first_token: AtomicU64::new(UNSET),
            last_token: AtomicU64::new(UNSET),
        }
    }

    /// Run `f` with this trace as the task's current stage context -
    /// the `request_log` middleware wraps the whole handler chain in it.
    pub async fn scope<R>(self: Arc<Self>, f: impl std::future::Future<Output = R>) -> R {
        STAGE_TRACE.scope(self, f).await
    }

    /// Stamp a set-once boundary: the first crossing wins, so gateway
    /// retries (child resend, failover) never move an earlier stage.
    fn mark_once(slot: &AtomicU64, received: &Instant) {
        let ns = u64::try_from(received.elapsed().as_nanos()).unwrap_or(u64::MAX);
        _ = slot.compare_exchange(UNSET, ns.max(1), Ordering::Relaxed, Ordering::Relaxed);
    }

    /// Stamp the last-token boundary: every observed chunk stores, the
    /// final value is the stream's true end.
    fn mark_last(&self, at: Instant) {
        let ns = u64::try_from(at.duration_since(self.received).as_nanos()).unwrap_or(u64::MAX);
        self.last_token.store(ns.max(1), Ordering::Relaxed);
    }

    fn mark_auth(&self) {
        Self::mark_once(&self.auth_done, &self.received);
    }

    fn mark_routed(&self) {
        Self::mark_once(&self.routed, &self.received);
    }

    fn mark_admitted(&self) {
        Self::mark_once(&self.admitted, &self.received);
    }

    fn mark_dispatched(&self) {
        Self::mark_once(&self.dispatched, &self.received);
    }

    /// First-token stamp takes the OBSERVED chunk instant (the tap's
    /// `first`), not call time - the connection task may poll the tap
    /// after further buffering.
    fn mark_first_token(&self, at: Instant) {
        let ns = u64::try_from(at.duration_since(self.received).as_nanos()).unwrap_or(u64::MAX);
        _ = self.first_token.compare_exchange(
            UNSET,
            ns.max(1),
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
    }
}

/// Stamp helpers for the shared chokepoints. No-ops off the HTTP task
/// (bench harnesses, tests, direct calls) - a missing boundary just
/// leaves that stage unobserved, never fabricated.
pub(crate) fn mark_auth_done() {
    _ = STAGE_TRACE.try_with(|t| t.mark_auth());
}

pub(crate) fn mark_model_resolved() {
    _ = STAGE_TRACE.try_with(|t| t.mark_routed());
}

pub(crate) fn mark_admitted() {
    _ = STAGE_TRACE.try_with(|t| t.mark_admitted());
}

pub(crate) fn mark_child_dispatched() {
    _ = STAGE_TRACE.try_with(|t| t.mark_dispatched());
}

/// Token-boundary stamp used by the usage tap in `request_log`: stream
/// chunks are polled on the connection's task, past the task-local's
/// scope, so the tap calls this on its captured trace handle directly.
pub(crate) fn note_chunk(trace: &StageTrace, first_seen: Option<Instant>, chunk_at: Instant) {
    if let Some(at) = first_seen {
        trace.mark_first_token(at);
    }
    trace.mark_last(chunk_at);
}

/// Sub-millisecond headroom for the gateway-owned stages, wide enough
/// that a pathological store read still lands inside a finite bucket.
const GATEWAY_STAGE_BOUNDS: &[f64] = &[
    0.000_05, 0.000_1, 0.000_25, 0.000_5, 0.001, 0.002_5, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
    1.0, 2.5, 5.0,
];

/// Admission spans cold spawns (a child boot is seconds-to-minutes), so
/// its buckets extend to the 2-minute queue ceiling like `queue_wait`.
const ADMISSION_STAGE_BOUNDS: &[f64] = &[
    0.000_5, 0.001, 0.002_5, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
    60.0, 120.0,
];

/// Engine-side stages mirror the TTFT histogram's bucket ladder.
const ENGINE_STAGE_BOUNDS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
];

/// The six stage histograms. Owned by `AppState`; rendered into
/// `/metrics` (Prometheus) and `/api/quantiles` (JSON) like the
/// TTFT/TPOT families.
#[derive(Debug)]
pub struct StageTimings {
    auth: Histogram,
    body: Histogram,
    admission: Histogram,
    dispatch: Histogram,
    model_ttft: Histogram,
    stream: Histogram,
}

impl Default for StageTimings {
    fn default() -> Self {
        Self::new()
    }
}

impl StageTimings {
    #[must_use]
    pub fn new() -> Self {
        Self {
            auth: Histogram::new(
                "blazar_stage_auth_seconds",
                "Gateway stage: request received to API key resolved (bearer/x-api-key + concurrency lease)",
                GATEWAY_STAGE_BOUNDS,
            ),
            body: Histogram::new(
                "blazar_stage_body_seconds",
                "Gateway stage: key resolved to model row resolved (body transport + parse + store lookup)",
                GATEWAY_STAGE_BOUNDS,
            ),
            admission: Histogram::new(
                "blazar_stage_admission_seconds",
                "Gateway stage: model resolved to engine granted (supervisor ensure + priority-queue wait + cold spawn)",
                ADMISSION_STAGE_BOUNDS,
            ),
            dispatch: Histogram::new(
                "blazar_stage_dispatch_seconds",
                "Gateway stage: engine granted to child response headers (child send)",
                GATEWAY_STAGE_BOUNDS,
            ),
            model_ttft: Histogram::new(
                "blazar_stage_model_ttft_seconds",
                "Gateway stage: engine granted to first stream byte (engine-side time-to-first-token)",
                ENGINE_STAGE_BOUNDS,
            ),
            stream: Histogram::new(
                "blazar_stage_stream_seconds",
                "Gateway stage: first stream byte to last stream byte (decode duration)",
                ENGINE_STAGE_BOUNDS,
            ),
        }
    }

    /// Fold one finished request's boundaries into the histograms. A
    /// stage whose end boundary never fired is skipped - durations are
    /// never measured across a gap.
    pub fn observe(&self, trace: &StageTrace) {
        let auth = trace.auth_done.load(Ordering::Relaxed);
        let routed = trace.routed.load(Ordering::Relaxed);
        let admitted = trace.admitted.load(Ordering::Relaxed);
        let dispatched = trace.dispatched.load(Ordering::Relaxed);
        let first = trace.first_token.load(Ordering::Relaxed);
        let last = trace.last_token.load(Ordering::Relaxed);
        // `received` is the zero baseline: every stamp is elapsed-nanos
        // since it, so a span is a plain difference.
        if auth != UNSET {
            self.auth.observe_ns(auth);
        }
        if auth != UNSET && routed != UNSET {
            self.body.observe_ns(routed - auth);
        }
        if routed != UNSET && admitted != UNSET {
            self.admission.observe_ns(admitted - routed);
        }
        if admitted != UNSET && dispatched != UNSET {
            self.dispatch.observe_ns(dispatched - admitted);
        }
        if admitted != UNSET && first != UNSET {
            self.model_ttft.observe_ns(first - admitted);
        }
        if first != UNSET && last != UNSET && last >= first {
            self.stream.observe_ns(last - first);
        }
    }

    /// Prometheus exposition for all six stages (per-histogram
    /// `_bucket`/`_sum`/`_count` plus the p50/p99/p999 gauges the other
    /// latency families render).
    pub fn render(&self, out: &mut String) {
        self.auth.render(out);
        self.body.render(out);
        self.admission.render(out);
        self.dispatch.render(out);
        self.model_ttft.render(out);
        self.stream.render(out);
    }

    /// JSON block for `/api/quantiles` - same shape as the TTFT/TPOT
    /// latency blocks (count + p50/p95/p99 in milliseconds).
    #[must_use]
    pub fn quantile_blocks(&self) -> serde_json::Value {
        use serde_json::json;
        fn block(h: &Histogram) -> serde_json::Value {
            if h.count() == 0 {
                return json!({"count": 0, "p50_ms": null, "p95_ms": null, "p99_ms": null});
            }
            json!({
                "count": h.count(),
                "p50_ms": (h.quantile(0.50) * 1000.0).round(),
                "p95_ms": (h.quantile(0.95) * 1000.0).round(),
                "p99_ms": (h.quantile(0.99) * 1000.0).round(),
            })
        }
        json!({
            "auth_ms": block(&self.auth),
            "body_ms": block(&self.body),
            "admission_ms": block(&self.admission),
            "dispatch_ms": block(&self.dispatch),
            "model_ttft_ms": block(&self.model_ttft),
            "stream_ms": block(&self.stream),
        })
    }
}

#[cfg(test)]
#[allow(non_snake_case)] // pin names read as scenario sentences (house pattern)
mod tests {
    use super::*;

    /// Stamp helper: auth boundary at `micros` after receipt (stored
    /// directly as elapsed-nanos, the way a real stamp would land).
    fn stamped(micros: u64) -> StageTrace {
        let t = StageTrace::new();
        t.auth_done.store(micros * 1_000, Ordering::Relaxed);
        t
    }

    #[test]
    fn unit__stage_trace__set_once_boundaries_keep_first_stamp() {
        let t = StageTrace::new();
        t.mark_auth();
        let first = t.auth_done.load(Ordering::Relaxed);
        assert!(first >= 1, "a fired boundary is never UNSET");
        std::thread::sleep(std::time::Duration::from_micros(50));
        t.mark_auth();
        assert_eq!(
            t.auth_done.load(Ordering::Relaxed),
            first,
            "second crossing must not move a set-once boundary"
        );
    }

    #[test]
    fn unit__stage_timings__observe_skips_stages_with_unfired_boundaries() {
        let timings = StageTimings::new();
        // Only auth fired: every other stage's end boundary is UNSET.
        let t = stamped(100);
        timings.observe(&t);
        let mut out = String::new();
        timings.render(&mut out);
        assert!(out.contains("blazar_stage_auth_seconds_count 1"), "{out}");
        for name in [
            "blazar_stage_body_seconds_count",
            "blazar_stage_admission_seconds_count",
            "blazar_stage_dispatch_seconds_count",
            "blazar_stage_model_ttft_seconds_count",
            "blazar_stage_stream_seconds_count",
        ] {
            assert!(
                out.contains(&format!("{name} 0")),
                "stage without both boundaries must stay unobserved: {out}"
            );
        }
    }

    #[test]
    fn unit__stage_timings__observe_full_trace_counts_every_stage() {
        let timings = StageTimings::new();
        let t = StageTrace::new();
        let at = |us: u64| t.received + std::time::Duration::from_micros(us);
        // Monotonic boundary ladder, stored directly as elapsed-nanos.
        for (slot, us) in [
            (&t.auth_done, 10),
            (&t.routed, 20),
            (&t.admitted, 30),
            (&t.dispatched, 40),
            (&t.first_token, 50),
        ] {
            slot.store(
                u64::try_from((at(us) - t.received).as_nanos()).unwrap_or(u64::MAX),
                Ordering::Relaxed,
            );
        }
        t.mark_last(at(90));
        timings.observe(&t);
        let mut out = String::new();
        timings.render(&mut out);
        for name in [
            "blazar_stage_auth_seconds_count 1",
            "blazar_stage_body_seconds_count 1",
            "blazar_stage_admission_seconds_count 1",
            "blazar_stage_dispatch_seconds_count 1",
            "blazar_stage_model_ttft_seconds_count 1",
            "blazar_stage_stream_seconds_count 1",
        ] {
            assert!(
                out.contains(name),
                "full trace must count every stage ({name}): {out}"
            );
        }
    }

    #[test]
    fn unit__stage_timings__empty_quantile_block_renders_nulls() {
        let timings = StageTimings::new();
        let v = timings.quantile_blocks();
        assert_eq!(v["auth_ms"]["count"], 0, "{v}");
        assert!(
            v["stream_ms"]["p99_ms"].is_null(),
            "0-obs must be null: {v}"
        );
    }

    #[tokio::test]
    async fn unit__stage_trace__task_local_scope_carries_marks_to_chokepoints() {
        let trace = Arc::new(StageTrace::new());
        let probe = Arc::clone(&trace);
        trace
            .scope(async move {
                mark_auth_done();
                mark_model_resolved();
                mark_admitted();
                mark_child_dispatched();
            })
            .await;
        assert!(probe.auth_done.load(Ordering::Relaxed) != UNSET);
        assert!(probe.routed.load(Ordering::Relaxed) != UNSET);
        assert!(probe.admitted.load(Ordering::Relaxed) != UNSET);
        assert!(probe.dispatched.load(Ordering::Relaxed) != UNSET);
    }

    #[tokio::test]
    async fn unit__stage_trace__marks_outside_scope_are_noops() {
        // No scope installed: the helpers must not panic (bench
        // harnesses and tests call the chokepoints bare).
        mark_auth_done();
        mark_model_resolved();
        mark_admitted();
        mark_child_dispatched();
        // Tap-path helper on a bare trace (no scope needed by design).
        let trace = StageTrace::new();
        note_chunk(
            &trace,
            Some(std::time::Instant::now()),
            std::time::Instant::now(),
        );
        assert!(trace.first_token.load(Ordering::Relaxed) != UNSET);
        assert!(trace.last_token.load(Ordering::Relaxed) != UNSET);
    }
}
