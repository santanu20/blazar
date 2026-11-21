//! Workload-derived routing policy (the dynamic half of
//! `[engine_routing]`).
//!
//! `engine_routing.policy` names a STATIC preference order for lanes
//! that overlap on a format. That order cannot know what a model is
//! actually USED for: the same safetensors model serving an agent's
//! tool loop wants the latency lane, while the same weights behind a
//! batch-extraction job want the throughput lane. The gateway already
//! classifies every request into a [`WorkloadKind`] at admission and
//! (v24+) persists the per-model mix into `usage_workload`; this
//! module folds that mix into a policy decision at spawn time.
//!
//! Contract (the same three invariants everywhere in Blazar's
//! adaptive surfaces):
//!
//! 1. **Evidence floor** - fewer than [`WORKLOAD_POLICY_MIN_REQUESTS`]
//!    classified requests in the window means NO derivation: the
//!    configured policy stands. A guessed mix is worse than a static
//!    one.
//! 2. **Dominance hysteresis** - a class must hold at least
//!    [`WORKLOAD_POLICY_DOMINANCE`] of classified traffic before it
//!    moves the policy; anything less is a mixed workload where the
//!    configured default (quality) is the honest pick. This is
//!    anti-flap: traffic that hovers near the boundary must not
//!    oscillate lanes across restarts.
//! 3. **Escape hatch** - `engine_routing.workload_aware = false`
//!    restores pure static-policy behavior for every model at once;
//!    a per-model `engine` pin in `[[model_overrides]]` still wins
//!    over any policy (that precedence lives in `serving_lane`,
//!    upstream of this module).
//!
//! The decision is computed by BOTH consumers that must never
//! disagree: the supervisor's spawn path and the core execution-plan
//! explainer - one function, two call sites.

use crate::config::RoutingPolicy;
use crate::store::{Store, UsageWorkloadRow};

use crate::optimize::WORKLOAD_MIN_REQUESTS;

/// Classified requests in the window before a mix may move policy.
/// Deliberately equal to the optimize ladder's
/// [`crate::optimize::WORKLOAD_MIN_REQUESTS`]: one notion of "enough
/// traffic shape" across every adaptive surface.
pub const WORKLOAD_POLICY_MIN_REQUESTS: i64 = WORKLOAD_MIN_REQUESTS;

/// Share of classified traffic a class must hold to own the policy
/// (0.6 = 60%). Below it the workload is mixed and the configured
/// policy stands.
pub const WORKLOAD_POLICY_DOMINANCE: f64 = 0.6;

/// How many days of workload mix feed the derivation. 30d mirrors the
/// optimize ladder's usage window.
pub const WORKLOAD_POLICY_WINDOW_DAYS: i64 = 30;

/// Class groups mapping workload kinds onto routing signals. Kinds
/// with no routing signal yet (media generation) stay out of every
/// group - they count toward the floor but never move the policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WorkloadMix {
    /// Interactive chat turns - latency-sensitive.
    pub interactive: i64,
    /// Tool-execution chat - latency-sensitive (agent loops pay TTFT
    /// per hop).
    pub tool: i64,
    /// Raw-lane long prefills - throughput-shaped.
    pub long_prefill: i64,
    /// Embeddings/reranking - throughput-shaped batch work.
    pub batch: i64,
    /// Media generation kinds - no policy signal yet.
    pub media: i64,
}

impl WorkloadMix {
    /// Fold window rows for ONE model into the class groups. Rows for
    /// other models are ignored (the mix is per-model by contract).
    #[must_use]
    pub fn from_rows<'a>(
        rows: impl IntoIterator<Item = &'a UsageWorkloadRow>,
        model: &str,
    ) -> Self {
        let mut mix = Self::default();
        for row in rows {
            if row.model != model {
                continue;
            }
            match row.kind.as_str() {
                "interactive_chat" => mix.interactive += row.requests,
                "tool_execution" => mix.tool += row.requests,
                "long_prefill" => mix.long_prefill += row.requests,
                "embedding" | "reranking" => mix.batch += row.requests,
                // Media surfaces and any future kind: counted for the
                // floor, inert for the policy.
                _ => mix.media += row.requests,
            }
        }
        mix
    }

    /// Total classified requests in the window (every kind counts).
    #[must_use]
    pub fn total(&self) -> i64 {
        self.interactive + self.tool + self.long_prefill + self.batch + self.media
    }

    /// The latency-sensitive share (interactive + tool).
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // request counts live far below 2^53
    fn latency_share(&self) -> f64 {
        let signalled = self.interactive + self.tool + self.long_prefill + self.batch;
        if signalled == 0 {
            return 0.0;
        }
        (self.interactive + self.tool) as f64 / signalled as f64
    }

    /// The throughput-shaped share (long prefill + batch).
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // request counts live far below 2^53
    fn throughput_share(&self) -> f64 {
        let signalled = self.interactive + self.tool + self.long_prefill + self.batch;
        if signalled == 0 {
            return 0.0;
        }
        (self.long_prefill + self.batch) as f64 / signalled as f64
    }
}

/// The policy decision handed to `serving_lane`: which policy applies
/// and WHY (the reason is operator-facing - it lands in spawn logs and
/// the execution plan, so it must read like an audit line, not debug
/// noise).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyDecision {
    pub policy: RoutingPolicy,
    /// `false` = the configured policy stands (thin evidence, mixed
    /// workload, or the feature off).
    pub derived: bool,
    pub reason: String,
}

/// Derive the effective policy for one model's spawn.
///
/// `configured` is `[engine_routing] policy` verbatim - it stands
/// unless the workload mix clears both the floor and the dominance
/// threshold. `aware` is `[engine_routing] workload_aware`.
#[must_use]
pub fn derive_policy(configured: RoutingPolicy, aware: bool, mix: &WorkloadMix) -> PolicyDecision {
    if !aware {
        return PolicyDecision {
            policy: configured,
            derived: false,
            reason: "workload-aware routing off (engine_routing.workload_aware = false)".into(),
        };
    }
    let total = mix.total();
    if total < WORKLOAD_POLICY_MIN_REQUESTS {
        return PolicyDecision {
            policy: configured,
            derived: false,
            reason: format!(
                "workload evidence thin ({total} classified requests in \
                 {WORKLOAD_POLICY_WINDOW_DAYS}d < {WORKLOAD_POLICY_MIN_REQUESTS}) - \
                 configured policy stands"
            ),
        };
    }
    // Dominance over SIGNALLED traffic only: a box serving 40% media +
    // 35% interactive + 25% batch derives latency from a 35/60 = 58%
    // interactive share of signalled traffic... which misses the
    // threshold - mixed is mixed, whatever the media floor did.
    if mix.latency_share() >= WORKLOAD_POLICY_DOMINANCE {
        return PolicyDecision {
            policy: RoutingPolicy::Latency,
            derived: true,
            reason: format!(
                "workload mix: {}/{} signalled requests latency-shaped \
                 (interactive {} + tool {} vs long_prefill {} + batch {}) - \
                 latency policy",
                mix.interactive + mix.tool,
                mix.interactive + mix.tool + mix.long_prefill + mix.batch,
                mix.interactive,
                mix.tool,
                mix.long_prefill,
                mix.batch
            ),
        };
    }
    if mix.throughput_share() >= WORKLOAD_POLICY_DOMINANCE {
        return PolicyDecision {
            policy: RoutingPolicy::Throughput,
            derived: true,
            reason: format!(
                "workload mix: {}/{} signalled requests throughput-shaped \
                 (long_prefill {} + batch {}) vs interactive {} + tool {}) - \
                 throughput policy",
                mix.long_prefill + mix.batch,
                mix.interactive + mix.tool + mix.long_prefill + mix.batch,
                mix.long_prefill,
                mix.batch,
                mix.interactive,
                mix.tool
            ),
        };
    }
    PolicyDecision {
        policy: configured,
        derived: false,
        reason: format!(
            "workload mixed (latency {:.0}% / throughput {:.0}% of {} signalled) - \
             configured policy stands",
            mix.latency_share() * 100.0,
            mix.throughput_share() * 100.0,
            mix.interactive + mix.tool + mix.long_prefill + mix.batch
        ),
    }
}

/// Read-side convenience: open the store, pull the window, fold the
/// mix for one model, derive. Store-unreadable degrades to the
/// configured policy with the failure named in the reason - routing
/// must never hard-fail on a telemetry read.
#[must_use]
pub fn derive_policy_from_store(
    store: &Store,
    configured: RoutingPolicy,
    aware: bool,
    model: &str,
) -> PolicyDecision {
    match store.list_usage_workload(WORKLOAD_POLICY_WINDOW_DAYS) {
        Ok(rows) => derive_policy(configured, aware, &WorkloadMix::from_rows(&rows, model)),
        Err(e) => PolicyDecision {
            policy: configured,
            derived: false,
            reason: format!("usage_workload unreadable ({e}) - configured policy stands"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::UsageWorkloadRow;

    fn row(model: &str, kind: &str, requests: i64) -> UsageWorkloadRow {
        UsageWorkloadRow {
            day: 20_261_101,
            model: model.into(),
            kind: kind.into(),
            requests,
        }
    }

    fn mix(interactive: i64, tool: i64, long_prefill: i64, batch: i64, media: i64) -> WorkloadMix {
        WorkloadMix {
            interactive,
            tool,
            long_prefill,
            batch,
            media,
        }
    }

    /// Feature off: the configured policy stands verbatim, reason
    /// names the knob (the operator must be able to see WHY nothing
    /// moved).
    #[test]
    #[allow(non_snake_case)]
    fn unit__derive_policy__feature_off_keeps_configured() {
        let d = derive_policy(RoutingPolicy::Quality, false, &mix(100, 0, 0, 0, 0));
        assert_eq!(d.policy, RoutingPolicy::Quality);
        assert!(!d.derived);
        assert!(d.reason.contains("workload_aware"));
    }

    /// Thin evidence: a floor of classified requests gates derivation
    /// even at a 100% dominant class - 19 requests of pure throughput
    /// shape stay on the configured policy.
    #[test]
    #[allow(non_snake_case)]
    fn unit__derive_policy__thin_evidence_never_derives() {
        let d = derive_policy(RoutingPolicy::Quality, true, &mix(0, 0, 19, 0, 0));
        assert_eq!(d.policy, RoutingPolicy::Quality);
        assert!(!d.derived);
        assert!(d.reason.contains("thin"));
    }

    /// Latency dominance: interactive+tool >= 60% of signalled traffic
    /// derives the latency policy, and the reason names the counts.
    #[test]
    #[allow(non_snake_case)]
    fn unit__derive_policy__latency_dominance_derives() {
        let d = derive_policy(RoutingPolicy::Quality, true, &mix(50, 10, 20, 10, 5));
        assert_eq!(d.policy, RoutingPolicy::Latency);
        assert!(d.derived);
        assert!(d.reason.contains("interactive 50"));
    }

    /// Throughput dominance: `long_prefill+batch` >= 60% derives the
    /// throughput policy.
    #[test]
    #[allow(non_snake_case)]
    fn unit__derive_policy__throughput_dominance_derives() {
        let d = derive_policy(RoutingPolicy::Quality, true, &mix(10, 5, 40, 15, 30));
        assert_eq!(d.policy, RoutingPolicy::Throughput);
        assert!(d.derived);
        assert!(d.reason.contains("throughput-shaped"));
    }

    /// Mixed workload: 55/45 splits derive nothing - the anti-flap
    /// hysteresis is the whole point of the dominance threshold.
    #[test]
    #[allow(non_snake_case)]
    fn unit__derive_policy__mixed_workload_stands() {
        let d = derive_policy(RoutingPolicy::Quality, true, &mix(44, 0, 36, 0, 200));
        assert_eq!(d.policy, RoutingPolicy::Quality);
        assert!(!d.derived);
        assert!(d.reason.contains("mixed"));
    }

    /// Media traffic counts toward the floor but never moves policy:
    /// pure media over the floor is "mixed" (nothing signalled), not a
    /// derivation.
    #[test]
    #[allow(non_snake_case)]
    fn unit__derive_policy__media_counts_for_floor_only() {
        let d = derive_policy(RoutingPolicy::Quality, true, &mix(0, 0, 0, 0, 100));
        assert!(!d.derived);
        assert!(d.reason.contains("mixed"));
    }

    /// Row folding: per-model isolation + kind grouping, unknown kinds
    /// land in media (floor-only), never panic, never dropped.
    #[test]
    #[allow(non_snake_case)]
    fn unit__workload_mix__folds_rows_per_model() {
        let rows = vec![
            row("m", "interactive_chat", 10),
            row("m", "tool_execution", 5),
            row("m", "long_prefill", 20),
            row("m", "embedding", 7),
            row("m", "reranking", 3),
            row("m", "video_generation", 9),
            row("m", "future_kind", 2),
            row("other", "interactive_chat", 999),
        ];
        let m = WorkloadMix::from_rows(&rows, "m");
        assert_eq!(
            (m.interactive, m.tool, m.long_prefill, m.batch, m.media),
            (10, 5, 20, 10, 11)
        );
        assert_eq!(m.total(), 56);
    }

    /// End to end over a real store: rows land, window filters, the
    /// derivation reads them back and derives.
    #[test]
    #[allow(non_snake_case)]
    fn integration__derive_policy_from_store__reads_mix_and_derives() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = crate::dirs::BlazarDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        let store = Store::open(&dirs).unwrap();
        // Day key from SQLite's own clock so the seed always sits
        // inside the window the read-back query compares against.
        let day: i64 = store
            .conn()
            .query_row(
                "SELECT CAST(strftime('%Y%m%d','now') AS INTEGER)",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let batch: Vec<(i64, String, String, i64)> = (0..30)
            .map(|i| {
                (
                    day,
                    "m".into(),
                    "long_prefill".into(),
                    if i == 0 { 3 } else { 1 },
                )
            })
            .collect();
        store.add_usage_workload_deltas(&batch).unwrap();
        // Retried flush adds, never overwrites (accumulate contract).
        store
            .add_usage_workload_deltas(&[(day, "m".into(), "long_prefill".into(), 5)])
            .unwrap();
        let d = derive_policy_from_store(&store, RoutingPolicy::Quality, true, "m");
        assert_eq!(d.policy, RoutingPolicy::Throughput);
        assert!(d.derived, "reason: {}", d.reason);
        // Per-model isolation: a model with no rows keeps the policy.
        let d2 = derive_policy_from_store(&store, RoutingPolicy::Quality, true, "other");
        assert_eq!(d2.policy, RoutingPolicy::Quality);
        assert!(!d2.derived);
    }
}
