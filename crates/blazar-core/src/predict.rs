//! Decode-throughput prediction from local benchmark history (v1).
//!
//! Physics: single-stream decode is memory-bound — every generated token
//! reads essentially the whole weight file. So a measured sample's
//! `t/s × bytes` estimates a lane's effective memory bandwidth; the
//! median across measured neighbors calibrates the lane, and
//! `median_bandwidth / target_bytes` predicts an unmeasured model on it.
//!
//! Every prediction is labeled with its basis and meant as a starting
//! point, never a substitute for measuring. Limitations (v1): dense
//! models assumed (`MoE` active-expert bytes are not modeled), context/KV
//! pressure ignored, and hardware grouping arrives with the evidence
//! axes (a single-machine store makes `engine_tag` a sufficient scope
//! until then).

use serde::Serialize;

/// One calibration sample: a measured `(model, t/s)` pair plus the
/// model's on-disk size in bytes.
#[derive(Debug, Clone)]
pub struct CalibSample {
    pub model: String,
    pub tg_tokens_per_sec: f64,
    pub bytes: u64,
}

/// A prediction with its provenance. Callers must render the method and
/// sample count alongside the number so it can never pose as a
/// measurement.
#[derive(Debug, Clone, Serialize)]
pub struct Prediction {
    pub decode_tps: f64,
    pub method: &'static str,
    pub n_samples: usize,
}

/// Median effective bandwidth over `samples`, divided by the target's
/// size.
///
/// Returns `None` — never a fabricated number — when no usable sample
/// exists, or when the result strays outside `[0.05, 20]×` the fastest
/// measured sample (such an outlier means the calibration does not
/// transfer to this size class). Rows for the target model itself are
/// excluded: self-prediction would launder a stale measurement in as a
/// fresh prediction.
pub fn predict_decode_tps(
    samples: &[CalibSample],
    target_bytes: u64,
    target_model: &str,
) -> Option<Prediction> {
    // Byte counts exceed f64's exact-integer range only past 4.5 PiB —
    // no real model file is anywhere near that.
    #![allow(clippy::cast_precision_loss)]
    if target_bytes == 0 {
        return None;
    }
    let usable: Vec<&CalibSample> = samples
        .iter()
        .filter(|s| s.model != target_model && s.bytes > 0 && s.tg_tokens_per_sec > 0.0)
        .collect();
    if usable.is_empty() {
        return None;
    }
    let mut bandwidths: Vec<f64> = usable
        .iter()
        .map(|s| s.tg_tokens_per_sec * s.bytes as f64)
        .collect();
    bandwidths.sort_by(f64::total_cmp);
    let mid = bandwidths.len() / 2;
    // Even counts average the middle pair — deterministic and unbiased.
    let median_bw = if bandwidths.len() % 2 == 1 {
        bandwidths[mid]
    } else {
        bandwidths[mid - 1].midpoint(bandwidths[mid])
    };
    let predicted = median_bw / target_bytes as f64;
    let max_sample_tps = usable
        .iter()
        .map(|s| s.tg_tokens_per_sec)
        .fold(f64::NEG_INFINITY, f64::max);
    if !(predicted >= max_sample_tps * 0.05 && predicted <= max_sample_tps * 20.0) {
        return None;
    }
    Some(Prediction {
        decode_tps: predicted,
        method: "median-effective-bandwidth",
        n_samples: usable.len(),
    })
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use super::*;

    fn sample(model: &str, tps: f64, bytes: u64) -> CalibSample {
        CalibSample {
            model: model.to_string(),
            tg_tokens_per_sec: tps,
            bytes,
        }
    }

    #[test]
    fn unit__predict__exact_median_math_odd_and_even_counts() {
        // Bandwidths: 40*2 = 80, 20*4 = 80, 10*8 = 80 GB/s-equivalent —
        // any target size divides the shared median exactly.
        let s = [sample("a", 40.0, 2), sample("b", 20.0, 4)];
        let p = predict_decode_tps(&s, 4, "target").expect("even count predicts");
        assert!((p.decode_tps - 20.0).abs() < 1e-9);
        assert_eq!(p.n_samples, 2);
        assert_eq!(p.method, "median-effective-bandwidth");

        let s = [
            sample("a", 40.0, 2),
            sample("b", 20.0, 4),
            sample("c", 80.0, 1),
        ];
        // Bandwidths 80, 80, 80 → median 80.
        let p = predict_decode_tps(&s, 8, "target").expect("odd count predicts");
        assert!((p.decode_tps - 10.0).abs() < 1e-9);
    }

    #[test]
    fn unit__predict__same_size_as_sample_reproduces_its_tps() {
        // Self-consistency: predicting a target the same size as one
        // measured sample yields that sample's t/s (bytes cancel).
        let s = [sample("a", 41.5, 5_000_000_000)];
        let p = predict_decode_tps(&s, 5_000_000_000, "target").expect("predicts");
        assert!((p.decode_tps - 41.5).abs() < 1e-6);
    }

    #[test]
    fn unit__predict__target_row_excluded_and_garbage_skipped() {
        // A stale history row for the target itself is not evidence, and
        // zero-byte / zero-tps rows contribute nothing.
        let s = [
            sample("target", 999.0, 100),
            sample("gone", 0.0, 100),
            sample("empty", 50.0, 0),
            sample("real", 20.0, 4),
        ];
        let p = predict_decode_tps(&s, 8, "target").expect("only real row usable");
        assert_eq!(p.n_samples, 1);
        assert!((p.decode_tps - 10.0).abs() < 1e-9);
    }

    #[test]
    fn unit__predict__clamp_rejects_far_transfers() {
        // Bandwidth 20×1e6 = 2e7 → 40 KB target predicts 500 t/s, which
        // exceeds the 20× cap over the 20 t/s max sample (400).
        let s = [sample("a", 20.0, 1_000_000)];
        assert!(predict_decode_tps(&s, 40_000, "t").is_none());
        // 0.05× floor: cannot predict below 1 t/s either.
        assert!(predict_decode_tps(&s, 800_000_000, "t").is_none()); // 0.025 t/s
    }

    #[test]
    fn unit__predict__no_samples_or_zero_target_is_none() {
        assert!(predict_decode_tps(&[], 1_000, "t").is_none());
        assert!(predict_decode_tps(&[sample("a", 10.0, 1_000)], 0, "t").is_none());
        // Only the target's own rows → still None.
        assert!(predict_decode_tps(&[sample("t", 10.0, 1_000)], 1_000, "t").is_none());
    }
}
