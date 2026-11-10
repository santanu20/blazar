//! Measured KV-cache quantization quality receipts - the evidence base
//! the tuner consults before a quantized KV grade may win a profile.
//!
//! The registry (`registry/kv-quant-quality.json`, compiled in with
//! `include_str!`) holds llama-perplexity A/B receipts: fixed corpus,
//! identical ctx/chunks/ngl, only the cache type varied. Per the
//! registry's own contract, `model` is the store name - the same string
//! `/api/explain/{model}` resolves - so eligibility lookups match
//! EXACTLY. A basename or alias match would borrow one checkpoint's
//! quality proof for another: fabrication, not evidence.
//!
//! The gate rule: `f16` is the baseline and always eligible; any other
//! grade is eligible only when a receipt for THIS model measured a rung
//! containing that grade at a context window at least as long as the
//! profile being tuned (`grade_evidenced_at_ctx`). A ppl receipt at
//! ctx 2048 says nothing about 32k retrieval - hence the ctx floor.

use serde_json::Value;
use std::sync::OnceLock;

const KV_QUALITY_RECEIPTS_JSON: &str = include_str!("../registry/kv-quant-quality.json");

/// Parsed registry, once per process. The file ships inside the crate
/// root (cargo publish tarballs only the package directory), so the
/// include path stays crate-relative.
pub fn receipts() -> &'static Value {
    static PARSED: OnceLock<Value> = OnceLock::new();
    PARSED.get_or_init(|| serde_json::from_str(KV_QUALITY_RECEIPTS_JSON).unwrap_or(Value::Null))
}

/// Whether `grade` (a KV cache type such as `q8_0`) has measured
/// quality evidence for `model` at a context window of at least
/// `min_ctx`. Exact store-name match only - see the module docs. Rung
/// keys are either a single grade (`f16`) or a `k/v` pair
/// (`q8_0/q8_0`); a grade evidenced on either side of the pair counts.
#[must_use]
pub fn grade_evidenced_at_ctx(model: &str, grade: &str, min_ctx: u32) -> bool {
    // f16 is the reference every receipt is compared against - the
    // baseline needs no receipt of its own.
    if grade == "f16" {
        return true;
    }
    receipts()["receipts"].as_array().is_some_and(|receipts| {
        receipts.iter().any(|receipt| {
            receipt["model"].as_str() == Some(model)
                && receipt["ctx"].as_u64().unwrap_or(0) >= u64::from(min_ctx)
                && receipt["rungs"].as_array().is_some_and(|rungs| {
                    rungs.iter().any(|rung| {
                        rung["kv"]
                            .as_str()
                            .is_some_and(|kv| kv.split('/').any(|side| side == grade))
                    })
                })
        })
    })
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // unit__<fn>__<scenario> is the repo pin-name convention
    use super::*;

    /// Pins the truth table against the real compiled-in registry:
    /// qwen3.5-9b has a `q8_0` rung at ctx 2048 AND 32768 (eligible at
    /// both floors), a `q4_0` rung at both ctxs, no `q5_0` rung anywhere,
    /// and an unknown model has no evidence at all. f16 is always
    /// eligible regardless.
    #[test]
    fn unit__grade_evidenced_at_ctx__truth_table_from_real_registry() {
        // q8_0: measured at 2048 and 32768.
        assert!(grade_evidenced_at_ctx("qwen3.5-9b", "q8_0", 2048));
        assert!(grade_evidenced_at_ctx("qwen3.5-9b", "q8_0", 32768));
        // The 32k receipt floors eligibility there; nothing covers 64k.
        assert!(!grade_evidenced_at_ctx("qwen3.5-9b", "q8_0", 65536));
        // q4_0: measured at both windows too.
        assert!(grade_evidenced_at_ctx("qwen3.5-9b", "q4_0", 32768));
        // turbo grades: fork-lane receipts at ctx 2048 (f16 A/B) and
        // 16384 (turbo3-vs-turbo4); the 32k attempt died on the 8GiB
        // card, so the evidence floor stops at 16k.
        assert!(grade_evidenced_at_ctx("qwen3.5-9b", "turbo3", 2048));
        assert!(grade_evidenced_at_ctx("qwen3.5-9b", "turbo3", 16384));
        assert!(grade_evidenced_at_ctx("qwen3.5-9b", "turbo4", 16384));
        assert!(!grade_evidenced_at_ctx("qwen3.5-9b", "turbo3", 32768));
        assert!(!grade_evidenced_at_ctx("qwen3.5-9b", "turbo4", 32768));
        // turbo2 was never measured - no receipt, no borrowing.
        assert!(!grade_evidenced_at_ctx("qwen3.5-9b", "turbo2", 2048));
        // q5_0: never measured - throughput alone can never adopt it.
        assert!(!grade_evidenced_at_ctx("qwen3.5-9b", "q5_0", 2048));
        // Exact-name contract: case or basename variants do not borrow
        // the receipt.
        assert!(!grade_evidenced_at_ctx("Qwen3.5-9B", "q8_0", 2048));
        assert!(!grade_evidenced_at_ctx("qwen3.5-9b-q4_k_m", "q8_0", 2048));
        // Unknown model: honest no.
        assert!(!grade_evidenced_at_ctx("qwen3-0.6b", "q8_0", 2048));
        // Baseline is unconditional.
        assert!(grade_evidenced_at_ctx("qwen3-0.6b", "f16", 131_072));
    }

    /// The registry must parse and carry the schema marker - a
    /// malformed file would silently make every grade ineligible,
    /// which is safe for quality but hides the breakage. Pin the parse.
    #[test]
    fn unit__receipts__registry_parses_with_schema_marker() {
        assert_eq!(receipts()["schema"].as_i64(), Some(1));
        assert!(
            receipts()["receipts"]
                .as_array()
                .is_some_and(|r| !r.is_empty())
        );
    }
}
