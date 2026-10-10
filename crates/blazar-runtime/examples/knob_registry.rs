//! Print the emission-truth first-class flag registry as JSON on stdout.
//!
//! The output is committed at `scripts/first_class_flags.json` (the
//! single source of truth the capability auditor reads) and pinned by
//! `tests/knob_registry.rs`. Regenerate after any emission change:
//!
//! ```text
//! cargo run -p blazar-runtime --example knob_registry > scripts/first_class_flags.json
//! ```

use std::collections::BTreeMap;

fn main() {
    let lanes: BTreeMap<&str, Vec<String>> = blazar_runtime::knob_registry::lane_flags();
    let doc = serde_json::json!({
        "version": 1,
        "lanes": lanes,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&doc).expect("registry JSON is always serializable")
    );
}
