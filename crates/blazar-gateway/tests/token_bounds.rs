//! Live-calibration bracket for the prompt token estimate.
//!
//! Every row of `fixtures/token_calibration.json` was tokenized by a
//! real engine (`/tokenize` on llama.cpp b12606, 2026-11-24). The class
//! priors in `preflight` must keep each exact count inside the
//! estimate's `[low, high]`: a count above the bracket would let an
//! oversized prompt through, one below it would refuse a prompt the
//! engine serves.

use std::collections::{BTreeMap, BTreeSet};

use blazar_gateway::preflight::token_bounds;
use serde_json::Value;

#[test]
#[allow(non_snake_case)]
fn integration__token_bounds__bracket_every_live_calibration_row() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/token_calibration.json")).unwrap();
    let units: BTreeMap<String, String> = serde_json::from_value(fixture["units"].clone()).unwrap();
    let repeats: BTreeMap<String, u64> =
        serde_json::from_value(fixture["repeats"].clone()).unwrap();
    let rows = fixture["rows"].as_array().expect("rows");
    assert!(rows.len() >= 22, "calibration rows: {}", rows.len());

    let mut tokenizers = BTreeSet::new();
    for row in rows {
        let tokenizer = row["tokenizer"].as_str().expect("tokenizer");
        let unit = row["unit"].as_str().expect("unit");
        let exact = row["tokens"].as_u64().expect("tokens");
        tokenizers.insert(tokenizer);
        let times = usize::try_from(repeats[unit]).expect("repeat count");
        let text = units[unit].repeat(times);
        assert_eq!(
            text.len() as u64,
            row["bytes"].as_u64().expect("bytes"),
            "{unit}: fixture byte count"
        );
        let (low, high) = token_bounds(&text);
        assert!(
            low <= exact,
            "{tokenizer} {unit}: {exact} tokens sits below the bracket [{low}, {high}]"
        );
        assert!(
            exact <= high,
            "{tokenizer} {unit}: {exact} tokens sits above the bracket [{low}, {high}]"
        );
    }
    assert!(tokenizers.len() >= 2, "{tokenizers:?}");
}
