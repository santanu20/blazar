//! `/api/benchmarks` — the stored measurement record, read-only.
//!
//! One row per (model, engine lane): the plain-bench history from
//! `blazar bench` (`bench_results` table) merged with the tuned launch
//! profile (`profiles` table) so the console shows measured vs tuned
//! side by side. Nothing here re-runs a benchmark; a missing side
//! renders as `null`, which the console prints as an honest gap.

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde_json::{Value, json};

use crate::state::AppState;
use blazar_core::store::{BenchResultRow, ProfileRow};

/// Test names from a stored payload. `bench_results` rows are a JSON
/// array of measurement rows; tuned profiles carry `{score, rows}`.
/// Raw llama-bench rows carry an EMPTY `test` field — the real name is
/// derived from the shape (`pp512`, `tg128`, `pp512 @ tg32`), exactly
/// the way `BenchRow::test_name` derives it everywhere else. Rows that
/// derive to nothing are dropped; unparseable payloads yield an empty
/// list — the row still shows its dates, the tests column just renders
/// empty.
fn payload_test_names(payload: &str) -> Vec<String> {
    use blazar_runtime::bench::BenchRow;
    let rows: Vec<BenchRow> = match serde_json::from_str::<Value>(payload) {
        // bench_results shape: a bare array of measurement rows.
        Ok(rows @ Value::Array(_)) => serde_json::from_value(rows).unwrap_or_default(),
        // Tuned-profile shape: {score, rows: [...]}.
        Ok(mut v) => {
            serde_json::from_value(v.get_mut("rows").map(std::mem::take).unwrap_or_default())
                .unwrap_or_default()
        }
        Err(_) => Vec::new(),
    };
    let mut names: Vec<String> = rows.iter().map(BenchRow::test_name).collect();
    names.retain(|n| !n.is_empty());
    names.sort();
    names.dedup();
    names
}

/// Tuned-profile composite score when the profile carries one
/// (`benchmark_json.score`, written by `blazar tune`).
fn profile_score(profile: &ProfileRow) -> Option<f64> {
    let payload: Value = profile
        .benchmark_json
        .as_deref()
        .and_then(|p| serde_json::from_str(p).ok())?;
    payload["score"].as_f64()
}

/// Pure assembly: merge bench history + tuned profiles into one row per
/// (model, lane), sorted by model then lane. Each side is optional —
/// `measured_at` without `tuned_at` is a benched-but-untuned model and
/// vice versa.
fn benchmarks_rows(benches: &[BenchResultRow], profiles: &[ProfileRow]) -> Vec<Value> {
    #[derive(Default)]
    struct Merged {
        measured_at: Option<i64>,
        tests: Vec<String>,
        score: Option<f64>,
        tuned_at: Option<i64>,
    }
    let mut merged: BTreeMap<(String, String), Merged> = BTreeMap::new();
    for bench in benches {
        let entry = merged
            .entry((bench.model.clone(), bench.engine_tag.clone()))
            .or_default();
        entry.measured_at = Some(bench.updated_at);
        entry.tests = payload_test_names(&bench.payload_json);
    }
    for profile in profiles {
        let entry = merged
            .entry((profile.model_name.clone(), profile.engine_tag.clone()))
            .or_default();
        entry.score = profile_score(profile);
        entry.tuned_at = Some(profile.updated_at);
    }
    merged
        .into_iter()
        .map(|((model, engine_tag), m)| {
            json!({
                "model": model,
                "engine_tag": engine_tag,
                "measured_at": m.measured_at,
                "tests": m.tests,
                "score": m.score,
                "tuned_at": m.tuned_at,
            })
        })
        .collect()
}

/// `GET /api/benchmarks` — every stored benchmark + tuning record.
pub async fn benchmarks(State(state): State<Arc<AppState>>) -> Json<Value> {
    // Flat, short store borrows (never nested inside another with_store).
    let benches = state
        .with_store(blazar_core::Store::list_bench_results)
        .and_then(std::result::Result::ok)
        .unwrap_or_default();
    let profiles = state
        .with_store(blazar_core::Store::list_profiles)
        .and_then(std::result::Result::ok)
        .unwrap_or_default();
    Json(json!({
        "object": "blazar.benchmarks",
        "rows": benchmarks_rows(&benches, &profiles),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bench_row_payload(name: &str) -> String {
        format!(r#"[{{"test":"{name}","ts":41.2,"n_ctx":8192}}]"#)
    }

    #[test]
    #[allow(non_snake_case)] // pin names read as scenario sentences (house pattern)
    fn unit__payload_test_names__accepts_both_stored_shapes() {
        let plain = payload_test_names(&bench_row_payload("pp512"));
        assert_eq!(plain, vec!["pp512".to_string()]);

        let tuned =
            payload_test_names(r#"{"score":41.7,"rows":[{"test":"tg128"},{"test":"tg128"}]}"#);
        assert_eq!(tuned, vec!["tg128".to_string()], "deduped, sorted");

        // Real stored bench rows carry an EMPTY test column — the name
        // must derive from the shape exactly like the CLI/scorecard do.
        let stored = payload_test_names(
            r#"[{"test":"","t/s":21082.3,"n_prompt":512,"n_gen":0},
                 {"test":"","t/s":390.1,"n_prompt":0,"n_gen":128},
                 {"test":"","t/s":11.2,"n_prompt":512,"n_gen":32},
                 {"test":"","t/s":9.9}]"#,
        );
        assert_eq!(
            stored,
            vec!["pp512", "pp512 @ tg", "tg128"],
            "derived, underivable row dropped"
        );

        assert!(payload_test_names("not json").is_empty());
        assert!(payload_test_names("42").is_empty());
    }

    #[test]
    #[allow(non_snake_case)] // pin names read as scenario sentences (house pattern)
    fn unit__benchmarks_rows__merges_by_model_and_lane_with_optional_sides() {
        let benches = vec![BenchResultRow {
            model: "m1".into(),
            engine_tag: "e1".into(),
            payload_json: bench_row_payload("pp512"),
            updated_at: 100,
        }];
        // Same (model, lane) profile with a tuned score + a profile-only
        // lane that has never been plain-benched.
        let profile = ProfileRow {
            model_name: "m1".into(),
            engine_tag: "e1".into(),
            args_hash: "h".into(),
            args_json: "{}".into(),
            benchmark_json: Some(r#"{"score":41.7,"rows":[]}"#.into()),
            updated_at: 200,
        };
        let other_lane = ProfileRow {
            model_name: "m2".into(),
            engine_tag: "e2".into(),
            args_hash: "h".into(),
            args_json: "{}".into(),
            benchmark_json: None,
            updated_at: 300,
        };
        let rows = benchmarks_rows(&benches, &[profile, other_lane]);
        assert_eq!(rows.len(), 2, "m1/e1 merged + m2/e2 profile-only");
        assert_eq!(rows[0]["model"], "m1");
        assert_eq!(rows[0]["measured_at"], 100);
        assert_eq!(rows[0]["tuned_at"], 200);
        assert_eq!(rows[0]["score"], 41.7);
        assert_eq!(
            rows[0]["tests"],
            json!(["pp512"]),
            "bench payload wins for tests on the merged row"
        );
        assert_eq!(rows[1]["model"], "m2");
        assert_eq!(rows[1]["measured_at"], Value::Null, "never benched");
        assert_eq!(rows[1]["tuned_at"], 300);
        assert_eq!(rows[1]["score"], Value::Null, "no benchmark_json");
    }
}
