//! Double-ended pin on the committed first-class flag registry.
//!
//! `scripts/first_class_flags.json` is generated from live emission:
//! `cargo run -p blazar-runtime --example knob_registry > scripts/first_class_flags.json`.
//! This test recomputes the registry and demands set equality with the
//! committed file in BOTH directions - emission added without
//! regenerating the file fails (stale registry), and a hand-edited file
//! without matching emission fails (fabricated registry). The version-2
//! `capabilities` section gets the same treatment against
//! `blazar-core::capability_matrix`, and the version-3 `kv_matrix`
//! section against `blazar_runtime::kv::certification` - disposition
//! or interop-verdict drift without a regen fails, and a hand-edited
//! matrix fails.
#![allow(non_snake_case)] // house test-naming: integration__subject__behavior

use std::collections::BTreeSet;

const REGEN: &str = "regenerate: cargo run -p blazar-runtime --example knob_registry > scripts/first_class_flags.json";

#[test]
fn integration__knob_registry__committed_json_matches_live_emission() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../scripts/first_class_flags.json"
    );
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let doc: serde_json::Value =
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("registry JSON parses: {e}"));
    assert_eq!(doc["version"], 3, "registry schema version must be pinned");
    let committed = doc["lanes"]
        .as_object()
        .unwrap_or_else(|| panic!("registry JSON lanes must be an object"));
    let live = blazar_runtime::knob_registry::lane_flags();

    let committed_keys: BTreeSet<&str> = committed.keys().map(String::as_str).collect();
    let live_keys: BTreeSet<&str> = live.keys().copied().collect();
    assert_eq!(
        committed_keys, live_keys,
        "lane key sets diverged ({REGEN})"
    );

    for (lane, flags) in &live {
        assert!(
            !flags.is_empty(),
            "{lane} registry came back empty - fixture matrix broke"
        );
        let committed_flags: BTreeSet<&str> = committed[*lane]
            .as_array()
            .unwrap_or_else(|| panic!("lane {lane} must be a flag array"))
            .iter()
            .map(|v| {
                v.as_str()
                    .unwrap_or_else(|| panic!("{lane} entry not a string"))
            })
            .collect();
        let live_flags: BTreeSet<&str> = flags.iter().map(String::as_str).collect();
        let missing: Vec<_> = live_flags.difference(&committed_flags).collect();
        let extra: Vec<_> = committed_flags.difference(&live_flags).collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "{lane} registry drift - emission not in file: {missing:?}, file not in emission: \
             {extra:?}; {REGEN}"
        );
    }
}

#[test]
fn integration__knob_registry__committed_capabilities_match_live_matrix() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../scripts/first_class_flags.json"
    );
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let doc: serde_json::Value =
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("registry JSON parses: {e}"));
    assert_eq!(doc["version"], 3, "capabilities ride schema version 3");
    let committed = doc["capabilities"]
        .as_object()
        .unwrap_or_else(|| panic!("registry JSON capabilities must be an object ({REGEN})"));
    let live = serde_json::to_value(blazar_core::capability_matrix::capabilities_doc())
        .expect("capability matrix serializes");
    let live_obj = live
        .as_object()
        .expect("capabilities_doc is a map keyed by lane");

    let committed_keys: BTreeSet<&str> = committed.keys().map(String::as_str).collect();
    let live_keys: BTreeSet<&str> = live_obj.keys().map(String::as_str).collect();
    assert_eq!(
        committed_keys, live_keys,
        "capability lane key sets diverged ({REGEN})"
    );

    for lane in live_keys {
        let committed_rows = committed.get(lane).unwrap_or_else(|| {
            panic!("capability lane {lane} in live matrix but not committed ({REGEN})")
        });
        let live_rows = live_obj
            .get(lane)
            .unwrap_or_else(|| panic!("capability lane {lane} missing from live matrix"));
        assert_eq!(
            committed_rows, live_rows,
            "{lane} capability matrix drift ({REGEN})"
        );
    }
}

#[test]
fn integration__knob_registry__piper_lane_covers_option_and_output_flags() {
    let lanes = blazar_runtime::knob_registry::lane_flags();
    let piper: BTreeSet<&str> = lanes["piper"].iter().map(String::as_str).collect();
    for flag in [
        "--model",
        "--espeak_data",
        "--output_file",
        "--output_raw",
        "--length_scale",
        "--speaker",
        "--noise_scale",
        "--noise_w",
        "--sentence_silence",
    ] {
        assert!(piper.contains(flag), "piper lane missing {flag}");
    }
}

#[test]
fn integration__knob_registry__committed_kv_matrix_matches_live_certification() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../scripts/first_class_flags.json"
    );
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let doc: serde_json::Value =
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("registry JSON parses: {e}"));
    let committed = doc["kv_matrix"]
        .as_array()
        .unwrap_or_else(|| panic!("kv_matrix must be an array ({REGEN})"));
    let live: Vec<serde_json::Value> = blazar_runtime::kv::certification::kv_certification_matrix()
        .iter()
        .map(|row| {
            serde_json::json!({
                "pair": row.pair,
                "status": row.status.label(),
                "receipt": row.receipt,
                "note": row.note,
            })
        })
        .collect();
    assert_eq!(committed, &live, "kv certification matrix drift ({REGEN})");
}
