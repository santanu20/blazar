//! Adaptive model pools (design: `docs/research/2026-11-16-replica-pools-design.md`).
//!
//! A pool gives a model a logical service identity served by multiple
//! replicas - the local instance plus `[[remotes]]` peers - and routes
//! each request to the member the strategy picks. The pool is strictly
//! an enumeration + scoring layer ABOVE the two existing lanes: it
//! rewrites the requested model to the chosen member's concrete string
//! (`model` for the local lane, `remote:model` for a peer) and every
//! downstream behavior - scope checks, admission, reservation,
//! forwarding, occupancy, charging - is today's code, untouched.
//!
//! Defaults that keep routing trustworthy are preserved verbatim:
//! direct model names never consult pools (local ownership wins, pins
//! authoritative), and escalation-on-failure stays with `[[failover]]`
//! chains (a pool member may itself be a chain alias).

use axum::body::Bytes;
use blazar_core::config::{Config, ModelPool, PoolStrategy};
use std::collections::HashMap;
use std::time::Instant;

use crate::remotes::{self, PeerCapacity, PeerScore, RemoteHealth};
use crate::state::AppState;

/// Neutral time-to-head guess for the local member before any success
/// teaches a better number - mirrors `remotes::REMOTE_TTFT_DEFAULT_MS`
/// so an unscored local member and an unscored peer tie honestly and
/// declaration order decides.
const LOCAL_TTFT_DEFAULT_MS: f64 = remotes::REMOTE_TTFT_DEFAULT_MS;

/// Declared service-time estimate (milliseconds) for members with no
/// measured EWMA yet - local members (no per-model local service lane
/// exists today) and peers that have not cleanly drained a stream.
/// Keeping it equal to the wait default makes an unmeasured member's
/// `adaptive` total collapse to `wait + default`, which preserves
/// today's ordering posture; the honest gap (local service is never
/// measured) is documented in the strategy table.
// REMOTE_TTFT_DEFAULT_MS is the positive exact constant 750.0.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
const SERVICE_DEFAULT_MS: u64 = remotes::REMOTE_TTFT_DEFAULT_MS as u64;

/// How long a sticky `adaptive` pick stays authoritative before the
/// strategy reconsiders from scratch (anti-thrash window).
const POOL_STICKY_WINDOW: std::time::Duration = std::time::Duration::from_secs(10);

/// The last `adaptive` decision for one pool, consumed as hysteresis:
/// a healthy sticky member within the margin of the best estimate is
/// retained so bursty load cannot flip-flop members request to request.
#[derive(Debug, Clone)]
pub struct StickyPick {
    /// The member's effective routing string (unique per member).
    pub effective: String,
    /// When this pick was made (`expand`, the request path).
    pub at: Instant,
}

/// A member with its synthesized score and effective routing string.
#[derive(Debug, Clone)]
pub(crate) struct MemberScore {
    /// Index into `pool.members` (declaration order survives ties).
    pub idx: usize,
    /// What the request model becomes: `model` (local) or `peer:model`.
    pub effective: String,
    /// Same ingredients and priority as remote peer ranking.
    pub score: PeerScore,
    /// Expected service time (milliseconds): the peer's measured
    /// `service_ewma_ms` when one exists, the declared default
    /// otherwise. `adaptive` adds it to `est_wait_ms` to estimate
    /// completion; every other strategy ignores it.
    pub service_ms: u64,
    /// True when `service_ms` came from a measured EWMA - drives the
    /// no-signal parity contract (`adaptive` equals `warm_first` until
    /// at least one member carries a real service measurement).
    pub measured: bool,
    /// True when the member's peer is currently marked down (skipped
    /// by `ordered`; score-aware strategies already rank it last via
    /// the unknown/cold tier plus its stale signals).
    pub down: bool,
}

/// Find the pool a request name addresses, if any. Exact match only -
/// pool names never shadow prefixes (`peer:model` split stays first)
/// and partial matches keep today's not-found teaching.
#[must_use]
pub fn find<'a>(cfg: &'a Config, requested: &str) -> Option<&'a ModelPool> {
    let pools = &cfg.model_pools;
    if pools.is_empty() {
        return None;
    }
    pools.iter().find(|p| p.name == requested)
}

/// Score every member with the same signals remote ranking uses.
/// Pure over its signal snapshots so every ordering rule is
/// unit-pinnable without a live state, exactly like
/// [`remotes::score_peer`]. Local members synthesize a `PeerScore`
/// from supervisor signals; remote members score verbatim against the
/// member's concrete model (not the pool name).
// wave counts (< 2^52 by construction) and ceiling results are exact in f64;
// the local-member arm mirrors remotes::score_peer's float waves x ttft math
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)] // in_flight clamped >= 0; wave counts exact in f64
#[allow(clippy::too_many_arguments)] // signal snapshot inputs, one per telemetry lane
pub(crate) fn score_members_core(
    pool: &ModelPool,
    cfg: &Config,
    queue_depth: u64,
    ps_rows: &[blazar_runtime::supervisor::PsRow],
    healths: &HashMap<String, RemoteHealth>,
    caps: &HashMap<String, PeerCapacity>,
    local_service: &HashMap<String, f64>,
    local_names: &HashMap<String, String>,
    now: Instant,
) -> Vec<MemberScore> {
    pool.members
        .iter()
        .enumerate()
        .map(|(idx, member)| {
            if member.peer == "local" {
                // A qualified member ("name:quant") serves under its
                // resolved row name - the same canonicalization the
                // serving lanes apply - so tier/service lookups key by
                // the row name, never the qualified spelling.
                let row_name = local_names
                    .get(&member.model)
                    .cloned()
                    .unwrap_or_else(|| member.model.clone());
                let resident: Vec<_> = ps_rows.iter().filter(|r| r.name == row_name).collect();
                let (tier, slots, busy) = if resident.is_empty() {
                    (remotes::REMOTE_TIER_COLD, 1u64, 0u64)
                } else {
                    let busy: u64 = resident.iter().map(|r| r.in_flight.max(0) as u64).sum();
                    (remotes::REMOTE_TIER_WARM, resident.len() as u64, busy)
                };
                let queue = queue_depth.saturating_add(busy);
                let waves = queue.div_ceil(slots);
                let (service_ms, measured) = match local_service.get(&row_name) {
                    Some(ewma) => (ewma.ceil() as u64, true),
                    None => (SERVICE_DEFAULT_MS, false),
                };
                MemberScore {
                    idx,
                    effective: member.model.clone(),
                    score: PeerScore {
                        tier,
                        est_wait_ms: (waves as f64 * LOCAL_TTFT_DEFAULT_MS).ceil() as u64,
                        free_vram: std::cmp::Reverse(0),
                    },
                    service_ms,
                    measured,
                    down: false,
                }
            } else {
                let remote = cfg
                    .remotes
                    .iter()
                    .find(|r| r.name == member.peer)
                    .expect("config validation refuses unknown pool member peers");
                let hkey = remotes::health_key(remote);
                let h = healths.get(&hkey).copied().unwrap_or_default();
                let cap = caps.get(&hkey);
                let down = h.down_until.is_some_and(|t| now < t);
                let (service_ms, measured) = match h.service_ewma_ms {
                    Some(ewma) => (ewma.ceil() as u64, true),
                    None => (SERVICE_DEFAULT_MS, false),
                };
                MemberScore {
                    idx,
                    effective: format!("{}:{}", remote.name, member.model),
                    score: remotes::score_peer(&member.model, &h, cap),
                    service_ms,
                    measured,
                    down,
                }
            }
        })
        .collect()
}

/// State-facing snapshot wrapper: one queue-depth read, one `ps()`, one
/// clone of each signal cache (no lock held during scoring - R4).
pub(crate) fn score_members(state: &AppState, pool: &ModelPool) -> Vec<MemberScore> {
    let queue_depth = u64::try_from(state.queue.depth()).unwrap_or(u64::MAX);
    let ps_rows = state.sup.ps();
    let healths = state
        .remote_health
        .lock()
        .map(|h| h.clone())
        .unwrap_or_default();
    let caps = state
        .remote_capacity
        .lock()
        .map(|c| c.clone())
        .unwrap_or_default();
    let local_service = state
        .service_ewma_local
        .lock()
        .map(|m| m.clone())
        .unwrap_or_default();
    // Resolve each local member to its serving row name (qualified
    // spellings like "name:quant" canonicalize exactly as the serving
    // lanes do), so tier and service lookups match the names the
    // runtime and the EWMA stamps actually use.
    let local_names: HashMap<String, String> = pool
        .members
        .iter()
        .filter(|m| m.peer == "local")
        .map(|m| {
            let row_name = state
                .with_store(|s| {
                    crate::proxy::resolve_model(s, &m.model)
                        .ok()
                        .map(|row| row.name)
                })
                .flatten()
                .unwrap_or_else(|| m.model.clone());
            (m.model.clone(), row_name)
        })
        .collect();
    score_members_core(
        pool,
        &state.cfg(),
        queue_depth,
        &ps_rows,
        &healths,
        &caps,
        &local_service,
        &local_names,
        Instant::now(),
    )
}

/// Pick the serving member per the pool strategy. Ties break to
/// declaration order (stable min over `idx`). When every remote
/// member is down and no other member exists, the least-bad member is
/// still returned so the request surfaces the peer's real error
/// instead of a confusing not-found on the pool name.
///
/// `sticky` is the pool's last adaptive pick (hysteresis): a healthy
/// member within [`POOL_STICKY_MARGIN`] of the best expected
/// completion and inside [`POOL_STICKY_WINDOW`] is retained. Other
/// strategies ignore it.
pub(crate) fn pick<'a>(
    pool: &ModelPool,
    members: &'a [MemberScore],
    sticky: Option<&StickyPick>,
) -> Option<&'a MemberScore> {
    match pool.strategy {
        // Failover posture: first member whose peer is not marked
        // down. The local member is always eligible.
        PoolStrategy::Ordered => members.iter().find(|m| !m.down),
        PoolStrategy::LeastWait => members.iter().min_by_key(|m| (m.score.est_wait_ms, m.idx)),
        PoolStrategy::WarmFirst => members.iter().min_by_key(|m| (m.score, m.idx)),
        // Least expected completion: wait + service time, warm tier as
        // the tie-break. Until at least one member carries a measured
        // service EWMA the service term is the declared default for
        // everyone and the completion order collapses onto the
        // warm-first order (documented parity, pinned by test).
        PoolStrategy::Adaptive => {
            if !members.iter().any(|m| m.measured) {
                return members.iter().min_by_key(|m| (m.score, m.idx));
            }
            let best_total = members
                .iter()
                .filter(|m| !m.down)
                .map(|m| m.score.est_wait_ms.saturating_add(m.service_ms))
                .min()?;
            let best = members.iter().filter(|m| !m.down).min_by_key(|m| {
                let total = m.score.est_wait_ms.saturating_add(m.service_ms);
                (total, m.score.tier, m.idx)
            })?;
            // Hysteresis: keep the previous pick when it is still
            // healthy, fresh, and within the margin of the best
            // estimate - small estimate noise must not reshuffle
            // members (and KV caches) request to request. The margin
            // stays in integer math: best + best/10 = +10%.
            if let Some(s) = sticky
                && s.at.elapsed() < POOL_STICKY_WINDOW
                && let Some(current) = members
                    .iter()
                    .find(|m| m.effective == s.effective && !m.down)
            {
                let total = current.score.est_wait_ms.saturating_add(current.service_ms);
                let threshold = best_total.saturating_add(best_total / 10);
                if total <= threshold {
                    return Some(current);
                }
            }
            Some(best)
        }
    }
    .or_else(|| members.first())
}

/// Resolve a requested model through pools: `Some(effective)` when the
/// name addresses a pool and a member is selectable, `None` when the
/// name is not a pool (the untouched legacy path). Adaptive pools also
/// record their pick as hysteresis state for the next request.
pub fn expand(state: &AppState, requested: &str) -> Option<String> {
    let cfg = state.cfg();
    let pool = find(&cfg, requested)?;
    let members = score_members(state, pool);
    let sticky = state
        .pools_sticky
        .lock()
        .ok()
        .and_then(|m| m.get(&pool.name).cloned());
    let picked = pick(pool, &members, sticky.as_ref())?;
    if pool.strategy == PoolStrategy::Adaptive
        && let Ok(mut m) = state.pools_sticky.lock()
    {
        m.insert(
            pool.name.clone(),
            StickyPick {
                effective: picked.effective.clone(),
                at: Instant::now(),
            },
        );
    }
    Some(picked.effective.clone())
}

/// [`expand`] plus the request-body rewrite: the forwarded body must
/// carry the member's concrete model, not the pool name (remote peers
/// do not know pool identities; the local lane stamps its own model
/// id regardless). Unparseable bodies ride unchanged - the engine
/// reports its own error, same fail-open contract as
/// [`remotes::rewrite_model`], which does the swap.
pub fn expand_request(state: &AppState, requested: &str, body: &Bytes) -> Option<(String, Bytes)> {
    let effective = expand(state, requested)?;
    let rewritten = remotes::rewrite_model(body.clone(), &effective);
    Some((effective, rewritten))
}

/// Introspection for `GET /api/route/{name}` (admin gate applies at
/// the route): the full member table with scores and the decision,
/// same envelope as the existing route payload so `blazar route`
/// renders pools without a new client path.
pub fn route_json(state: &AppState, pool: &ModelPool) -> serde_json::Value {
    let members = score_members(state, pool);
    // Introspection reads (never writes) the sticky state so the
    // reported decision matches what the request path would do.
    let sticky = state
        .pools_sticky
        .lock()
        .ok()
        .and_then(|m| m.get(&pool.name).cloned());
    let chosen = pick(pool, &members, sticky.as_ref());
    let member_lines: Vec<serde_json::Value> = members
        .iter()
        .map(|m| {
            serde_json::json!({
                "peer": pool.members[m.idx].peer,
                "model": pool.members[m.idx].model,
                "effective": m.effective,
                "tier": crate::federation::tier_name(m.score.tier),
                "est_wait_ms": m.score.est_wait_ms,
                "service_ms": m.service_ms,
                "service_measured": m.measured,
                "free_vram": m.score.free_vram.0,
                "down": m.down,
            })
        })
        .collect();
    let decision = match chosen {
        Some(m) => {
            let sticky_held = sticky
                .is_some_and(|s| s.at.elapsed() < POOL_STICKY_WINDOW && s.effective == m.effective);
            serde_json::json!({
                "target": m.effective,
                "member": format!("{}:{}", pool.members[m.idx].peer, pool.members[m.idx].model),
                "strategy": format!("{:?}", pool.strategy),
                "reason": format!(
                    "pool '{}' picked member {} under {:?} (tier {}, est wait {} ms, service {} ms{})",
                    pool.name,
                    m.effective,
                    pool.strategy,
                    crate::federation::tier_name(m.score.tier),
                    m.score.est_wait_ms,
                    m.service_ms,
                    if sticky_held { ", sticky" } else { "" }
                ),
            })
        }
        None => serde_json::json!({
            "target": "none",
            "reason": format!("pool '{}' has no members", pool.name),
        }),
    };
    serde_json::json!({
        "object": "blazar.route",
        "model": pool.name,
        "pool": {
            "strategy": format!("{:?}", pool.strategy),
            "members": member_lines,
        },
        "decision": decision,
    })
}

/// Boot-time guard: a pool name colliding with a local model row
/// would be unreachable (local ownership wins before pools are
/// consulted), so refuse it loudly instead of silently shadowing the
/// pool.
pub fn validate_boot(state: &AppState) -> Result<(), String> {
    for pool in &state.cfg().model_pools {
        let collision = state
            .with_store(|s| crate::proxy::resolve_model(s, &pool.name).ok())
            .flatten();
        if collision.is_some() {
            return Err(format!(
                "model pool name '{}' collides with a local model row - rename the pool; pool names and model names share one request namespace",
                pool.name
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use super::*;
    use blazar_core::config::{PoolMember, Remote};
    use blazar_runtime::supervisor::PsRow;

    fn member(peer: &str, model: &str) -> PoolMember {
        PoolMember {
            peer: peer.into(),
            model: model.into(),
        }
    }

    fn scored(
        idx: usize,
        effective: &str,
        tier: u8,
        wait: u64,
        vram: u64,
        down: bool,
    ) -> MemberScore {
        MemberScore {
            idx,
            effective: effective.into(),
            score: PeerScore {
                tier,
                est_wait_ms: wait,
                free_vram: std::cmp::Reverse(vram),
            },
            service_ms: SERVICE_DEFAULT_MS,
            measured: false,
            down,
        }
    }

    /// Same as [`scored`] but with an explicit service-time estimate
    /// (the `adaptive` signal): `measured = true` marks a real EWMA.
    fn scored_service(
        idx: usize,
        effective: &str,
        tier: u8,
        wait: u64,
        service: u64,
        measured: bool,
    ) -> MemberScore {
        MemberScore {
            service_ms: service,
            measured,
            ..scored(idx, effective, tier, wait, 0, false)
        }
    }

    fn pool(strategy: PoolStrategy) -> ModelPool {
        ModelPool {
            name: "svc".into(),
            strategy,
            members: vec![member("local", "m1"), member("far", "m1")],
        }
    }

    fn cfg_with(remote: Option<Remote>) -> Config {
        let mut cfg = Config::default();
        if let Some(r) = remote {
            cfg.remotes.push(r);
        }
        cfg
    }

    fn far_remote() -> Remote {
        Remote {
            name: "far".into(),
            url: "http://127.0.0.1:1".into(),
            key: String::new(),
            allow_insecure_http: false,
        }
    }

    /// Ordered is failover posture: skip downed peers, keep declaration
    /// order, local always eligible.
    #[test]
    fn unit__pick__ordered_skips_down_keeps_declaration_order() {
        let p = pool(PoolStrategy::Ordered);
        let members = vec![
            scored(0, "m1", 0, 900, 0, false),
            scored(1, "far:m1", 0, 10, 0, true),
        ];
        let picked = pick(&p, &members, None).expect("local member eligible");
        assert_eq!(picked.idx, 0);

        let members = vec![
            scored(0, "far:m1", 0, 10, 0, true),
            scored(1, "m1", 0, 900, 0, false),
        ];
        let picked = pick(&p, &members, None).expect("second member eligible");
        assert_eq!(picked.idx, 1);
    }

    /// All members down still returns one (least-bad first) so the
    /// request surfaces the peer's real error, not a pool not-found.
    #[test]
    fn unit__pick__all_down_returns_least_bad_member() {
        let p = pool(PoolStrategy::Ordered);
        let members = vec![scored(0, "far:m1", 0, 10, 0, true)];
        let picked = pick(&p, &members, None).expect("degenerate pool still picks");
        assert_eq!(picked.effective, "far:m1");
    }

    /// `LeastWait` ignores tier entirely - a cold-but-idle member beats
    /// a warm-but-queued one, exactly as documented.
    #[test]
    fn unit__pick__least_wait_ignores_tier() {
        let p = pool(PoolStrategy::LeastWait);
        let members = vec![
            scored(0, "m1", 0, 1500, 0, false),
            scored(1, "far:m1", 2, 100, 0, false),
        ];
        let picked = pick(&p, &members, None).expect("two members");
        assert_eq!(picked.idx, 1, "cold peer with the smaller wait must win");
    }

    /// WarmFirst/Adaptive rank by the full `PeerScore` order: tier first,
    /// then wait, then VRAM; ties break to declaration order.
    #[test]
    fn unit__pick__warm_first_ranks_tier_then_wait() {
        let p = pool(PoolStrategy::WarmFirst);
        let members = vec![
            scored(0, "far:m1", 2, 10, 9999, false),
            scored(1, "m1", 0, 800, 0, false),
        ];
        let picked = pick(&p, &members, None).expect("two members");
        assert_eq!(picked.idx, 1, "warm member wins despite larger wait");

        let members = vec![
            scored(0, "m1", 0, 750, 5, false),
            scored(1, "far:m1", 0, 750, 5, false),
        ];
        let picked = pick(&p, &members, None).expect("two members");
        assert_eq!(picked.idx, 0, "full tie breaks to declaration order");
    }

    /// Adaptive without any measured service signal equals warm-first:
    /// the service term is the declared default for every member, so
    /// the completion order collapses onto the tier-first order
    /// (graceful degradation, pinned).
    #[test]
    fn unit__pick__adaptive_equals_warm_first_without_service_signals() {
        let members = vec![
            scored(0, "far:m1", 1, 100, 0, false),
            scored(1, "m1", 2, 50, 0, false),
        ];
        let warm = pick(&pool(PoolStrategy::WarmFirst), &members, None);
        let adaptive = pick(&pool(PoolStrategy::Adaptive), &members, None);
        assert_eq!(
            warm.map(|m| m.idx),
            adaptive.map(|m| m.idx),
            "no measured signal: adaptive must degrade to warm-first order"
        );
    }

    /// With a measured service EWMA, adaptive diverges from warm-first:
    /// expected completion (wait + service) outranks the warm tier.
    #[test]
    fn unit__pick__adaptive_diverges_from_warm_first_with_service_signal() {
        // Warm local: wait 800, unmeasured service default 750.
        // Cold remote: wait 100, MEASURED service 200 -> total 300.
        let members = vec![
            scored_service(0, "m1", 0, 800, SERVICE_DEFAULT_MS, false),
            scored_service(1, "far:m1", 2, 100, 200, true),
        ];
        let warm = pick(&pool(PoolStrategy::WarmFirst), &members, None);
        let adaptive = pick(&pool(PoolStrategy::Adaptive), &members, None);
        assert_eq!(
            warm.unwrap().idx,
            0,
            "warm-first still prefers the warm tier"
        );
        assert_eq!(
            adaptive.unwrap().idx,
            1,
            "adaptive picks least expected completion despite the cold tier"
        );
    }

    /// Hysteresis: a fresh, healthy sticky pick within the margin of
    /// the best estimate is retained - small estimate noise must not
    /// reshuffle members request to request.
    #[test]
    fn unit__pick__adaptive_sticky_member_retained_within_margin() {
        // Best: total 1000 (wait 900 + service 100). Sticky member idx 0:
        // total 1050 <= 1000 * 1.10 = 1100 -> retained.
        let members = vec![
            scored_service(0, "m1", 0, 950, 100, true),
            scored_service(1, "far:m1", 0, 900, 100, true),
        ];
        let sticky = StickyPick {
            effective: "m1".into(),
            at: Instant::now(),
        };
        let picked = pick(&pool(PoolStrategy::Adaptive), &members, Some(&sticky)).unwrap();
        assert_eq!(picked.idx, 0, "sticky member within 10% margin is kept");

        // Challenger is >10% better (best 1000, sticky total 1200):
        // the sticky hold releases and the challenger wins.
        let members = vec![
            scored_service(0, "m1", 0, 1100, 100, true),
            scored_service(1, "far:m1", 0, 900, 100, true),
        ];
        let picked = pick(&pool(PoolStrategy::Adaptive), &members, Some(&sticky)).unwrap();
        assert_eq!(picked.idx, 1, "materially better challenger takes over");
    }

    /// Hysteresis expiry: a stale sticky pick (outside the window) and
    /// a sticky pick whose member went down are both ignored.
    #[test]
    fn unit__pick__adaptive_sticky_ignored_when_stale_or_down() {
        // Stale: `at` older than the window -> pure best estimate wins.
        let members = vec![
            scored_service(0, "m1", 0, 950, 100, true),
            scored_service(1, "far:m1", 0, 900, 100, true),
        ];
        let stale = StickyPick {
            effective: "m1".into(),
            at: Instant::now()
                .checked_sub(POOL_STICKY_WINDOW + std::time::Duration::from_secs(1))
                .expect("sticky window well inside Instant range"),
        };
        let picked = pick(&pool(PoolStrategy::Adaptive), &members, Some(&stale)).unwrap();
        assert_eq!(
            picked.idx, 1,
            "expired sticky pick loses to the best estimate"
        );

        // Down: sticky member's peer is marked down -> skipped even
        // though it is fresh and would be within margin.
        let members = vec![
            scored_service(0, "m1", 0, 900, 100, true),
            scored_service(1, "far:m1", 0, 900, 100, true),
        ];
        let members: Vec<MemberScore> = members
            .into_iter()
            .map(|mut m| {
                if m.effective == "m1" {
                    m.down = true;
                }
                m
            })
            .collect();
        let sticky = StickyPick {
            effective: "m1".into(),
            at: Instant::now(),
        };
        let picked = pick(&pool(PoolStrategy::Adaptive), &members, Some(&sticky)).unwrap();
        assert_eq!(picked.idx, 1, "down sticky member is never retained");
    }

    fn resident_ps_row(model: &str, in_flight: i64) -> PsRow {
        PsRow {
            name: model.to_string(),
            engine: "llamacpp".to_string(),
            replica: None,
            state: "running",
            endpoint: "http://127.0.0.1:1".to_string(),
            idle_secs: 0,
            in_flight,
            ctx: 0,
            capacity: None,
            slots: None,
            slots_configured: None,
            slots_source: None,
            gpu: "full".to_string(),
            device: None,
            device_id: None,
            warnings: Vec::new(),
            spec_mode: "off".to_string(),
            draft: None,
            pid: 1,
            bytes: 1024,
            heat: 0,
            keep_alive_secs: None,
        }
    }

    /// Remote members score through `score_peer` verbatim (warm resident
    /// with slots + busy queue), local members synthesize the same
    /// shape - parity is what lets them compete in one list.
    #[test]
    fn unit__score_members_core__local_and_remote_same_shape() {
        let cfg = cfg_with(Some(far_remote()));
        let mut healths = HashMap::new();
        healths.insert(
            remotes::health_key(&cfg.remotes[0]),
            RemoteHealth {
                consec_failures: 0,
                down_until: None,
                in_flight: 2,
                ttft_ewma_ms: Some(250.0),
                service_ewma_ms: None,
            },
        );
        let mut caps = HashMap::new();
        caps.insert(
            remotes::health_key(&cfg.remotes[0]),
            PeerCapacity {
                fetched: Instant::now(),
                devices: vec![remotes::PeerDevice {
                    name: "gpu0".into(),
                    total_vram_bytes: 16,
                    free_vram_bytes: 8,
                }],
                residents: vec![remotes::PeerResident {
                    model: "m1".into(),
                    engine: "llamacpp".into(),
                    state: "running".into(),
                    slots: Some(4),
                    slots_configured: None,
                    in_flight: 2,
                }],
            },
        );

        let members = score_members_core(
            &pool(PoolStrategy::Adaptive),
            &cfg,
            0,
            &[],
            &healths,
            &caps,
            &HashMap::new(),
            &HashMap::new(),
            Instant::now(),
        );
        assert_eq!(members.len(), 2);
        // Local: cold (nothing resident), empty queue = immediate
        // service (0 waves), same wait semantics as an idle peer.
        assert_eq!(members[0].score.tier, remotes::REMOTE_TIER_COLD);
        assert_eq!(members[0].score.est_wait_ms, 0);
        assert_eq!(members[0].effective, "m1");
        // Remote: warm resident, (2+2)/4 = 1 wave * 250 ms EWMA.
        assert_eq!(members[1].score.tier, remotes::REMOTE_TIER_WARM);
        assert_eq!(members[1].score.est_wait_ms, 250);
        assert_eq!(members[1].score.free_vram.0, 8);
        assert_eq!(members[1].effective, "far:m1");
        assert!(!members[1].down);
    }

    /// A resident local member scores warm with its replicas as slots
    /// and in-flight requests folded into the wait wave.
    #[test]
    fn unit__score_members_core__resident_local_warm_with_replica_slots() {
        // Local-only pool: the remote-member fixture lane is covered by
        // the parity test above.
        let cfg = cfg_with(None);
        let local_pool = ModelPool {
            name: "svc".into(),
            strategy: PoolStrategy::Ordered,
            members: vec![member("local", "m1")],
        };
        let ps_rows = vec![
            resident_ps_row("m1", 1),
            resident_ps_row("m1", 3), // second replica: 2 slots total
        ];
        let members = score_members_core(
            &local_pool,
            &cfg,
            2, // gateway queue depth
            &ps_rows,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            Instant::now(),
        );
        assert_eq!(members[0].score.tier, remotes::REMOTE_TIER_WARM);
        // (2 queued + 4 busy) / 2 slots = 3 waves * 750 ms.
        assert_eq!(members[0].score.est_wait_ms, 2250);
    }

    /// A local member's service estimate comes from the per-model EWMA
    /// map (stamped by `keys::LocalServiceStamp` at local stream
    /// completion); absent entries keep the declared default and stay
    /// unmeasured.
    #[test]
    fn unit__score_members_core__local_member_consumes_service_ewma_map() {
        let cfg = cfg_with(None);
        let local_pool = ModelPool {
            name: "svc".into(),
            strategy: PoolStrategy::Adaptive,
            members: vec![member("local", "m1"), member("local", "m2")],
        };
        let mut local_service = HashMap::new();
        local_service.insert("m1".to_string(), 4000.0);
        let members = score_members_core(
            &local_pool,
            &cfg,
            0,
            &[],
            &HashMap::new(),
            &HashMap::new(),
            &local_service,
            &HashMap::new(),
            Instant::now(),
        );
        assert_eq!(members[0].service_ms, 4000);
        assert!(members[0].measured, "m1 carries a stamped EWMA");
        assert_eq!(members[1].service_ms, SERVICE_DEFAULT_MS);
        assert!(!members[1].measured, "m2 has no observation yet");
    }

    /// A pool member whose peer is marked down carries `down = true`.
    #[test]
    fn unit__score_members_core__down_peer_flagged() {
        let cfg = cfg_with(Some(far_remote()));
        let mut healths = HashMap::new();
        healths.insert(
            remotes::health_key(&cfg.remotes[0]),
            RemoteHealth {
                consec_failures: 9,
                down_until: Some(Instant::now() + std::time::Duration::from_secs(30)),
                in_flight: 0,
                ttft_ewma_ms: None,
                service_ewma_ms: None,
            },
        );
        let members = score_members_core(
            &pool(PoolStrategy::Ordered),
            &cfg,
            0,
            &[],
            &healths,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            Instant::now(),
        );
        assert!(members[1].down, "remote member inside the down window");
    }

    /// `find()` is exact-match and prefix-safe.
    #[test]
    fn unit__find__exact_match_only() {
        let mut cfg = cfg_with(None);
        cfg.model_pools = vec![pool(PoolStrategy::Ordered)];
        assert!(find(&cfg, "svc").is_some());
        assert!(find(&cfg, "svc2").is_none());
        assert!(find(&cfg, "far:m1").is_none());
        assert!(find(&cfg, "m1").is_none());
    }

    /// The body rewrite (`remotes::rewrite_model`, shared with the
    /// failover lane) carries the member model; unparseable bodies
    /// ride unchanged (fail-open, the engine reports its own error).
    #[test]
    fn unit__rewrite_body_model__swaps_field_keeps_rest() {
        let body = Bytes::from(r#"{"model":"svc","messages":[1]}"#);
        let out = remotes::rewrite_model(body.clone(), "m1");
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["model"], "m1");
        assert_eq!(v["messages"], serde_json::json!([1]));

        let raw = Bytes::from_static(b"not json");
        assert_eq!(remotes::rewrite_model(raw.clone(), "m1"), raw);
    }
}
