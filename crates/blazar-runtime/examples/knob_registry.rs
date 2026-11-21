//! Print the emission-truth first-class flag registry as JSON on stdout.
//!
//! The output is committed at `scripts/first_class_flags.json` (the
//! single source of truth the capability auditor reads) and pinned by
//! `tests/knob_registry.rs`. Regenerate after any emission change:
//!
//! ```text
//! cargo run -p blazar-runtime --example knob_registry > scripts/first_class_flags.json
//! ```
//!
//! Version 2 added the `capabilities` section (per-lane capability
//! matrix from `blazar-core::capability_matrix`). Version 3 adds the
//! `kv_matrix` section - the cross-engine KV interoperability
//! certification matrix from `blazar_runtime::kv::certification` - so
//! the committed JSON carries emission truth (lanes), disposition
//! truth (capabilities), and interop evidence truth (`kv_matrix`) in one
//! auditable document.

use std::collections::BTreeMap;

fn main() {
    let lanes: BTreeMap<&str, Vec<String>> = blazar_runtime::knob_registry::lane_flags();
    let capabilities = blazar_core::capability_matrix::capabilities_doc();
    let kv_matrix: Vec<serde_json::Value> =
        blazar_runtime::kv::certification::kv_certification_matrix()
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
    let doc = serde_json::json!({
        "version": 3,
        "lanes": lanes,
        "capabilities": capabilities,
        "kv_matrix": kv_matrix,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&doc).expect("registry JSON is always serializable")
    );
}
