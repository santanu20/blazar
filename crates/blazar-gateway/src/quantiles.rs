//! `/api/quantiles` — latency + cache-effectiveness summary as JSON.
//!
//! The same histograms `/metrics` renders as Prometheus text, in the
//! shape dashboards and the console want: per-family p50/p95/p99 in
//! milliseconds with observation counts, plus the prompt-cache and
//! semantic-cache tallies. Empty histograms render `null` quantiles —
//! a zero would read as a measured latency, which it never is.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use axum::Json;
use axum::extract::State;
use serde_json::{Value, json};

use crate::histogram::Histogram;
use crate::state::AppState;

/// One latency family in milliseconds. `count == 0` → null quantiles.
fn latency_block(h: &Histogram) -> Value {
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

pub async fn quantiles(State(state): State<Arc<AppState>>) -> Json<Value> {
    let obs = &state.obs;
    let prompt = obs.prompt_tokens.load(Ordering::Relaxed);
    let cached = obs.cached_tokens.load(Ordering::Relaxed);
    let sem = &state.sem;
    // Cache hit rate stays null until tokens exist — 0.0 would read as
    // "measured and terrible", which an idle daemon has not measured.
    // Token counters are request-scale, far below f64's 2^52 exact
    // integer range, so the lossy cast is exact in practice.
    #[allow(clippy::cast_precision_loss)]
    let hit_rate = (prompt > 0)
        .then(|| json!(((cached as f64 / prompt as f64) * 10_000.0).round() / 10_000.0));
    Json(json!({
        "object": "blazar.quantiles",
        "ttft_ms": latency_block(&state.ttft),
        "tpot_ms": latency_block(&state.tpot),
        "ttft_warm_ms": latency_block(&obs.ttft_warm),
        "ttft_cold_ms": latency_block(&obs.ttft_cold),
        "prompt_cache": {
            "prompt_tokens": prompt,
            "cached_tokens": cached,
            "hit_rate": hit_rate,
            "unclassified_responses": obs.unclassified.load(Ordering::Relaxed),
        },
        "semantic_cache": {
            "hits": sem.hits.load(Ordering::Relaxed),
            "misses": sem.misses.load(Ordering::Relaxed),
            "stores": sem.stores.load(Ordering::Relaxed),
            "embed_failures": sem.embed_failures.load(Ordering::Relaxed),
            "tool_bypasses": sem.tool_bypasses.load(Ordering::Relaxed),
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::histogram;

    #[test]
    #[allow(non_snake_case)] // pin names read as scenario sentences (house pattern)
    fn unit__latency_block__empty_histogram_renders_null_not_zero() {
        let block = latency_block(&histogram::ttft());
        assert_eq!(block["count"], 0, "{block}");
        assert!(block["p50_ms"].is_null(), "0-obs must be null: {block}");
        assert!(block["p95_ms"].is_null());
        assert!(block["p99_ms"].is_null());
    }

    #[test]
    #[allow(non_snake_case)] // pin names read as scenario sentences (house pattern)
    fn unit__latency_block__observations_render_milliseconds() {
        // Prometheus histograms interpolate inside buckets, so the pin
        // asserts the ms conversion + ordering, not an exact sample.
        let h = Histogram::new("t", "help", &[0.2]);
        h.observe_secs(0.15);
        h.observe_secs(0.15);
        let block = latency_block(&h);
        assert_eq!(block["count"], 2);
        let p50 = block["p50_ms"].as_f64().expect("observed → numeric");
        let p99 = block["p99_ms"].as_f64().expect("observed → numeric");
        // 0.15 s samples in a [0, 0.2] bucket interpolate well above
        // 50 ms; a forgotten secs→ms conversion would stay below 1.0.
        assert!(p50 > 50.0 && p50 <= 200.0, "p50 out of bucket: {p50}");
        assert!(p99 >= p50 && p99 <= 200.0, "p99 out of bucket: {p99}");
    }
}
