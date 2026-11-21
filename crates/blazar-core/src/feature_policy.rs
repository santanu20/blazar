//! Centralized automatic-feature policy - one table stating what every
//! auto-managed knob means, which evidence activates it, and who wins
//! when several sources disagree.
//!
//! Blazar's automatic behaviours grew one switch at a time (`slots = 0`,
//! `cache_type = ""`, `spec = "auto"`, `adaptive_slots`, ...). Each one
//! is documented where it lives, but no single place answers "what
//! policy governs THIS feature, and what outranks it". This module is
//! that place. It changes no runtime behaviour - the spawn paths keep
//! their logic; the table gives the planner (`execution_plan`), the
//! explain card, and `docs/7.SETUP.md` one authoritative description.
//!
//! Precedence contract (strongest first):
//!
//! 1. explicit user configuration (config file / CLI pin)
//! 2. verified model-specific profile (tuned, receipt-backed)
//! 3. measured automatic policy (receipts/governor say it helps)
//! 4. safe default
//!
//! The resolver encodes that ladder; autopilot-style writers must call
//! it before overwriting a knob so a deliberate operator choice can
//! never be silently undone by automation.

use serde::Serialize;

/// How a feature may activate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaturePolicy {
    /// Never activates automatically (command-gated opt-in only).
    Disabled,
    /// Activates on cheap, local safety checks (memory arithmetic,
    /// compatibility gates) with no measurement required.
    SafeAuto,
    /// Activates only when receipts/telemetry for THIS setup say it
    /// helps (spec-race receipts, governor acceptance rates).
    MeasuredAuto,
    /// Off by default until enough live measurements exist; opt-in.
    Experimental,
    /// An explicit operator setting won - automation must not touch it.
    UserPinned,
}

/// One row of the feature policy table.
#[derive(Debug, Serialize)]
pub struct FeaturePolicyRow {
    /// Stable feature id (`snake_case`, matches explain output).
    pub feature: &'static str,
    /// The config knob literal that selects the behaviour (the exact
    /// spelling `docs/7.SETUP.md` documents, e.g. `slots = 0`).
    pub knob: &'static str,
    /// Policy the feature runs under when the knob is at its auto
    /// value.
    pub default_policy: FeaturePolicy,
    /// The evidence that justifies activation - named, not implied.
    pub evidence: &'static str,
}

/// The complete policy table. Order is presentation order (matching
/// the SETUP doc's auto-settings table); lookups scan it - it is tiny
/// and cold.
#[must_use]
pub fn feature_policy_table() -> &'static [FeaturePolicyRow] {
    const TABLE: &[FeaturePolicyRow] = &[
        FeaturePolicyRow {
            feature: "slot_sizing",
            knob: "slots = 0",
            default_policy: FeaturePolicy::SafeAuto,
            evidence: "capacity estimate against live VRAM at spawn",
        },
        FeaturePolicyRow {
            feature: "max_loaded_models",
            knob: "max_loaded_models = 0",
            default_policy: FeaturePolicy::SafeAuto,
            evidence: "hardware probe (VRAM/RAM) sizes the resident set",
        },
        FeaturePolicyRow {
            feature: "kv_quantization_ladder",
            knob: "cache_type = \"\"",
            default_policy: FeaturePolicy::SafeAuto,
            evidence: "memory pressure + lane capability evidence; quality grades from the KV registry",
        },
        FeaturePolicyRow {
            feature: "spec_family_selection",
            knob: "spec = \"auto\"",
            default_policy: FeaturePolicy::MeasuredAuto,
            evidence: "spec-race receipts per model/lane; dense fallback when none qualify",
        },
        FeaturePolicyRow {
            feature: "ngram_fallback",
            knob: "spec_auto_ngram = true",
            default_policy: FeaturePolicy::MeasuredAuto,
            evidence: "patterned-workload receipt (+92% patterned / -3% prose lanes)",
        },
        FeaturePolicyRow {
            feature: "spec_governor",
            knob: "spec_auto_manage = true",
            default_policy: FeaturePolicy::MeasuredAuto,
            evidence: "live acceptance-rate governor disables unproductive speculation",
        },
        FeaturePolicyRow {
            feature: "adaptive_slot_expansion",
            knob: "adaptive_slots = true",
            default_policy: FeaturePolicy::MeasuredAuto,
            evidence: "sustained queue pressure + decode scaling receipts",
        },
        FeaturePolicyRow {
            feature: "adaptive_decode_gate",
            knob: "adaptive_decode_gate = true",
            default_policy: FeaturePolicy::MeasuredAuto,
            evidence: "decode queue saturation telemetry (avoids unproductive slot growth)",
        },
        FeaturePolicyRow {
            feature: "adaptive_ubatch",
            knob: "ubatch_auto = false",
            default_policy: FeaturePolicy::Experimental,
            evidence: "memory safety + throughput A/Bs still being collected",
        },
        FeaturePolicyRow {
            feature: "cross_engine_kv_import",
            knob: "(command-gated: kv import/export)",
            default_policy: FeaturePolicy::Disabled,
            evidence: "strict compatibility proof per carrier pair; never auto-applied",
        },
    ];
    TABLE
}

/// Resolve the EFFECTIVE policy for one feature under the precedence
/// ladder. `user_set` = an explicit config/CLI value exists;
/// `profile_set` = a verified model profile pins it; `measured_ok` =
/// the receipts the `MeasuredAuto` tier requires are present.
///
/// A feature whose default is Experimental or Disabled never activates
/// through the measured tier alone - those tiers exist precisely
/// because measurement is not sufficient proof for them.
#[must_use]
pub fn resolve_effective(
    feature: &str,
    user_set: bool,
    profile_set: bool,
    measured_ok: bool,
) -> FeaturePolicy {
    let default = feature_policy_table()
        .iter()
        .find(|r| r.feature == feature)
        .map_or(FeaturePolicy::Disabled, |r| r.default_policy);
    if user_set || profile_set {
        return FeaturePolicy::UserPinned;
    }
    match default {
        FeaturePolicy::SafeAuto => FeaturePolicy::SafeAuto,
        FeaturePolicy::MeasuredAuto if measured_ok => FeaturePolicy::MeasuredAuto,
        FeaturePolicy::Experimental
        | FeaturePolicy::MeasuredAuto
        | FeaturePolicy::Disabled
        | FeaturePolicy::UserPinned => FeaturePolicy::Disabled,
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn unit__feature_policy_table__covers_the_auto_surface() {
        let t = feature_policy_table();
        assert_eq!(t.len(), 10, "one row per documented auto knob");
        for expected in [
            "slot_sizing",
            "max_loaded_models",
            "kv_quantization_ladder",
            "spec_family_selection",
            "ngram_fallback",
            "spec_governor",
            "adaptive_slot_expansion",
            "adaptive_decode_gate",
            "adaptive_ubatch",
            "cross_engine_kv_import",
        ] {
            assert!(
                t.iter().any(|r| r.feature == expected),
                "missing feature {expected}"
            );
        }
        // Knob literals are the documented spellings (7.SETUP table).
        assert!(t.iter().any(|r| r.knob == "slots = 0"));
        assert!(t.iter().any(|r| r.knob == "spec = \"auto\""));
        // Every row names its evidence - no blank justifications.
        assert!(t.iter().all(|r| !r.evidence.is_empty()));
    }

    #[test]
    fn unit__resolve_effective__user_pin_beats_everything() {
        assert_eq!(
            resolve_effective("slot_sizing", true, false, false),
            FeaturePolicy::UserPinned
        );
        assert_eq!(
            resolve_effective("adaptive_ubatch", true, true, true),
            FeaturePolicy::UserPinned
        );
    }

    #[test]
    fn unit__resolve_effective__profile_outranks_measurement() {
        // A verified model profile pins the knob even when receipts
        // would also justify the auto tier.
        assert_eq!(
            resolve_effective("ngram_fallback", false, true, true),
            FeaturePolicy::UserPinned
        );
    }

    #[test]
    fn unit__resolve_effective__measured_tiers_require_evidence() {
        assert_eq!(
            resolve_effective("ngram_fallback", false, false, true),
            FeaturePolicy::MeasuredAuto
        );
        assert_eq!(
            resolve_effective("ngram_fallback", false, false, false),
            FeaturePolicy::Disabled,
            "no receipts -> the feature stays off, never guessed on"
        );
    }

    #[test]
    fn unit__resolve_effective__safe_auto_needs_no_measurement() {
        assert_eq!(
            resolve_effective("slot_sizing", false, false, false),
            FeaturePolicy::SafeAuto
        );
    }

    #[test]
    fn unit__resolve_effective__experimental_stays_off_without_a_pin() {
        // Measurement alone must NOT flip an Experimental feature on.
        assert_eq!(
            resolve_effective("adaptive_ubatch", false, false, true),
            FeaturePolicy::Disabled
        );
    }

    #[test]
    fn unit__resolve_effective__unknown_feature_is_disabled() {
        assert_eq!(
            resolve_effective("no_such_feature", false, false, true),
            FeaturePolicy::Disabled
        );
    }
}
