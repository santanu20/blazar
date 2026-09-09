//! Lock-free latency histograms for the gateway evidence loop.
//!
//! vLLM/SGLang ship TTFT/TPOT histograms as their most-used observability;
//! pallama measures the same things at the proxy — it is the only component
//! that sees every byte of every stream. Hand-rolled on `AtomicU64`
//! (nanoseconds internally): no metrics crate, no locks on the streaming
//! hot path, no label cardinality (global histograms — per-model split when
//! someone asks with a reason).

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};

/// Cumulative-bucket histogram rendering in Prometheus text format.
pub struct Histogram {
    name: &'static str,
    help: &'static str,
    /// Upper bounds in seconds per bucket, ascending, exclusive of +Inf.
    bounds: &'static [f64],
    buckets: Vec<AtomicU64>,
    sum_ns: AtomicU64,
    count: AtomicU64,
}

impl Histogram {
    #[must_use]
    pub fn new(name: &'static str, help: &'static str, bounds: &'static [f64]) -> Self {
        assert!(!bounds.is_empty(), "histogram needs at least one bound");
        assert!(
            bounds.windows(2).all(|w| w[0] < w[1]),
            "histogram bounds must be ascending"
        );
        Self {
            name,
            help,
            bounds,
            buckets: bounds.iter().map(|_| AtomicU64::new(0)).collect(),
            sum_ns: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    /// Record one observation in seconds. Infallible by construction:
    /// saturating atomics only, never panics on the stream hot path.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "ns conversion is clamp-guarded; 52-bit mantissa covers centuries of latency sums"
    )]
    pub fn observe_secs(&self, secs: f64) {
        if !(secs.is_finite() && secs >= 0.0) {
            return;
        }
        let ns = (secs * 1e9).clamp(0.0, u64::MAX as f64) as u64;
        if let Some(i) = self.bounds.iter().position(|b| secs <= *b) {
            self.buckets[i].fetch_add(1, Ordering::Relaxed);
        }
        // Above every bound: counted in _sum/_count (and therefore +Inf)
        // only — finite buckets must stay true upper bounds.
        self.sum_ns.fetch_add(ns, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// Append `# HELP/# TYPE`, cumulative `_bucket{le=…}` lines, `_sum`,
    /// `_count` in Prometheus exposition format.
    #[allow(
        clippy::cast_precision_loss,
        reason = "ns -> s display rounding is irrelevant"
    )]
    pub fn render(&self, out: &mut String) {
        let count = self.count.load(Ordering::Relaxed);
        let _ = writeln!(
            out,
            "# HELP {} {}\n# TYPE {} histogram",
            self.name, self.help, self.name
        );
        let mut cumulative = 0u64;
        for (b, c) in self.bounds.iter().zip(&self.buckets) {
            cumulative += c.load(Ordering::Relaxed);
            let _ = writeln!(
                out,
                "{}_bucket{{le=\"{}\"}} {}",
                self.name,
                format_f64(*b),
                cumulative
            );
        }
        let _ = writeln!(out, "{}_bucket{{le=\"+Inf\"}} {}", self.name, count);
        let _ = writeln!(
            out,
            "{}_sum {}",
            self.name,
            format_f64(self.sum_ns.load(Ordering::Relaxed) as f64 / 1e9)
        );
        let _ = writeln!(out, "{}_count {}", self.name, count);
        // Quantile gauges (p50/p99/p999) so `doctor` and dashboards read
        // directly without a PromQL histogram_quantile. Linear
        // interpolation inside the containing bucket — approximate by
        // construction (documented in the HELP line above).
        for (label, q) in [("p50", 0.5), ("p99", 0.99), ("p999", 0.999)] {
            let _ = writeln!(
                out,
                "{}_{}_seconds {}",
                self.name,
                label,
                format_f64(self.quantile(q))
            );
        }
    }

    /// Approximate quantile from cumulative buckets (linear
    /// interpolation within the containing bucket). 0 observations = 0.
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "target rank is ceil()'d and count-clamped; bucket math is integer after the rank cast"
    )]
    pub fn quantile(&self, q: f64) -> f64 {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        let target = (f64::from(u32::try_from(count).unwrap_or(u32::MAX)) * q).ceil();
        let target = target.max(1.0) as u64;
        let mut cumulative = 0u64;
        let mut prev_bound = 0.0f64;
        for (b, c) in self.bounds.iter().zip(&self.buckets) {
            let bucket = c.load(Ordering::Relaxed);
            // Buckets above a zero-count run don't move the cumulative.
            if bucket == 0 {
                prev_bound = *b;
                continue;
            }
            let prev_cum = cumulative;
            cumulative += bucket;
            if cumulative >= target {
                let frac = if cumulative > prev_cum {
                    (target - prev_cum) as f64 / bucket as f64
                } else {
                    1.0
                };
                return prev_bound + (*b - prev_bound) * frac.clamp(0.0, 1.0);
            }
            prev_bound = *b;
        }
        // Above every finite bound: report the top bound (a floor, not an
        // extrapolation — never invent latency beyond what was measured).
        *self.bounds.last().unwrap_or(&0.0)
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "guarded by the integral check above"
)]
fn format_f64(v: f64) -> String {
    // Prometheus accepts plain decimal; render compactly without trailing
    // zeros for values like 0.005 or 3.
    if (v - v.trunc()).abs() < f64::EPSILON {
        format!("{}", v as u64)
    } else {
        format!("{v}")
    }
}

/// Time-to-first-token of the response stream (time to first body byte).
#[must_use]
pub fn ttft() -> Histogram {
    Histogram::new(
        "pallama_ttft_seconds",
        "Gateway time-to-first-token: request start to first body byte of a generation (vLLM-style evidence loop)",
        &[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )
}

/// Inter-chunk cadence of streamed generations. NOTE: SSE chunks may carry
/// more than one token per flush — this is chunk cadence, an approximation
/// of time-per-output-token, not an exact TPOT.
#[must_use]
pub fn tpot() -> Histogram {
    Histogram::new(
        "pallama_tpot_seconds",
        "Gateway inter-chunk cadence for streamed generations (approximate time-per-output-token; chunks may batch tokens)",
        &[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0],
    )
}

/// TTFT of responses that reused prompt cache (cached prompt tokens > 0).
#[must_use]
pub fn ttft_warm() -> Histogram {
    Histogram::new(
        "pallama_ttft_warm_seconds",
        "Time-to-first-token for warm generations (usage reported cached prompt tokens > 0)",
        &[
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
        ],
    )
}

/// TTFT of responses that processed the whole prompt (no cached tokens).
#[must_use]
pub fn ttft_cold() -> Histogram {
    Histogram::new(
        "pallama_ttft_cold_seconds",
        "Time-to-first-token for cold generations (usage reported 0 cached prompt tokens)",
        &[
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
        ],
    )
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn unit__histogram__buckets_are_cumulative_and_sorted() {
        let h = Histogram::new("t", "test", &[0.1, 1.0]);
        for v in [0.05, 0.2, 0.9, 1.0, 7.0] {
            h.observe_secs(v);
        }
        let mut out = String::new();
        h.render(&mut out);
        assert!(out.contains("t_bucket{le=\"0.1\"} 1"), "{out}");
        assert!(out.contains("t_bucket{le=\"1\"} 4"), "{out}"); // 0.05,0.2,0.9,1.0
        assert!(out.contains("t_bucket{le=\"+Inf\"} 5"), "{out}");
        assert!(out.contains("t_count 5"), "{out}");
        assert!(out.contains("t_sum 9.15"), "{out}"); // 0.05+0.2+0.9+1.0+7.0
    }

    #[test]
    fn unit__histogram__rejects_nonfinite_and_negative() {
        let h = Histogram::new("t", "test", &[1.0]);
        h.observe_secs(f64::NAN);
        h.observe_secs(f64::NEG_INFINITY);
        h.observe_secs(-1.0);
        let mut out = String::new();
        h.render(&mut out);
        assert!(out.contains("t_count 0"), "{out}");
    }

    #[test]
    fn unit__histogram__bound_edges_inclusive() {
        let h = Histogram::new("t", "test", &[0.1]);
        h.observe_secs(0.1); // exactly on the bound -> first bucket
        let mut out = String::new();
        h.render(&mut out);
        assert!(out.contains("t_bucket{le=\"0.1\"} 1"), "{out}");
    }

    #[test]
    fn unit__histogram__quantiles_and_render_gauges() {
        let h = Histogram::new("t", "test", &[0.1, 0.5, 1.0, 5.0]);
        for v in [0.05, 0.05, 0.05, 0.4, 0.9, 4.0] {
            h.observe_secs(v);
        }
        let mut out = String::new();
        h.render(&mut out);
        assert!(out.contains("t_p50_seconds"), "{out}");
        assert!(out.contains("t_p99_seconds"), "{out}");
        assert!(out.contains("t_p999_seconds"), "{out}");
        // p50 lands in the first bucket (3 of 6 obs <= 0.1).
        assert!(h.quantile(0.5) <= 0.1, "{}", h.quantile(0.5));
        // p99 is inside the top finite bucket (4.0 obs).
        assert!(h.quantile(0.99) <= 5.0 && h.quantile(0.99) > 1.0);
        // Empty histogram: zero, not NaN.
        let empty = Histogram::new("e", "test", &[1.0]);
        assert!((empty.quantile(0.999) - 0.0).abs() < f64::EPSILON);
    }
}
