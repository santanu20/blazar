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

/// How long a service-time EWMA stays authoritative before it has
/// fully decayed back to the declared default. Idle-sleep evicts
/// idle models on this order, so an estimate older than this
/// describes an engine state that no longer exists.
const SERVICE_STALE_AFTER: std::time::Duration = std::time::Duration::from_secs(300);

/// Cache-affinity discount: one default TTFT (the prefill the
/// affinity member likely skips because its KV cache still holds the
/// prompt prefix). Full while the entry is [`AFFINITY_FRESH`] old,
/// then decays linearly to zero at [`AFFINITY_TTL`] - past that the
/// prefix is assumed evicted and the affinity entry is ignored.
const AFFINITY_FULL_MS: u64 = 750;
const AFFINITY_FRESH: std::time::Duration = std::time::Duration::from_secs(30);
const AFFINITY_TTL: std::time::Duration = std::time::Duration::from_secs(120);
/// Rolling window for the steering hit-rate (`hit_rate_pct` in the
/// route table): events older than this stop counting, so the rate
/// reflects current traffic instead of the whole session.
const STEER_WINDOW: std::time::Duration = std::time::Duration::from_secs(300);
/// Hard cap on retained window events per pool (memory bound under
/// sustained traffic far above what a 5-minute window needs).
const STEER_WINDOW_CAP: usize = 1024;
/// Sparkline resolution: the steering window is sliced into this many
/// equal buckets (300s / 15 = 20s each) for the route table's history.
const STEER_HISTORY_BUCKETS: u64 = 15;

/// Record one event in a rolling window map and report its live size:
/// expired entries are pruned from the front on both insert and read,
/// and the cap sheds the oldest events first.
pub(crate) fn window_push(
    map: &mut std::collections::HashMap<String, std::collections::VecDeque<Instant>>,
    pool: &str,
    now: Instant,
) {
    let q = map.entry(pool.to_string()).or_default();
    q.push_back(now);
    while let Some(front) = q.front()
        && now.duration_since(*front) >= STEER_WINDOW
    {
        q.pop_front();
    }
    while q.len() > STEER_WINDOW_CAP {
        q.pop_front();
    }
}

/// Live event count for a pool's rolling window (prunes first, so the
/// number always reflects the current window, never stale entries).
pub(crate) fn window_count(
    map: &mut std::collections::HashMap<String, std::collections::VecDeque<Instant>>,
    pool: &str,
    now: Instant,
) -> u64 {
    let Some(q) = map.get_mut(pool) else { return 0 };
    while let Some(front) = q.front()
        && now.duration_since(*front) >= STEER_WINDOW
    {
        q.pop_front();
    }
    q.len() as u64
}

/// Record one member-tagged steering hit (same pruning and cap as
/// [`window_push`]; the tag enables per-member trend splits).
pub(crate) fn window_push_tagged(
    map: &mut std::collections::HashMap<String, std::collections::VecDeque<(Instant, String)>>,
    pool: &str,
    member: &str,
    now: Instant,
) {
    let q = map.entry(pool.to_string()).or_default();
    q.push_back((now, member.to_string()));
    while let Some((front, _)) = q.front()
        && now.duration_since(*front) >= STEER_WINDOW
    {
        q.pop_front();
    }
    while q.len() > STEER_WINDOW_CAP {
        q.pop_front();
    }
}

/// Live tagged-event count for a pool's rolling window (prunes by the
/// event timestamp, ignores the member tag).
pub(crate) fn window_count_tagged(
    map: &mut std::collections::HashMap<String, std::collections::VecDeque<(Instant, String)>>,
    pool: &str,
    now: Instant,
) -> u64 {
    let Some(q) = map.get_mut(pool) else { return 0 };
    while let Some((front, _)) = q.front()
        && now.duration_since(*front) >= STEER_WINDOW
    {
        q.pop_front();
    }
    q.len() as u64
}

/// Per-member steering hit history: the same 15-bucket slicing as
/// [`window_history`] but counting only events steered to `member`.
/// Unknown members (or none yet) read as all zeros.
#[allow(clippy::cast_possible_truncation)] // bucket index math, always < 15
pub(crate) fn member_hit_history(
    hit_map: &mut std::collections::HashMap<String, std::collections::VecDeque<(Instant, String)>>,
    pool: &str,
    member: &str,
    now: Instant,
) -> Vec<u64> {
    let bucket_secs = STEER_WINDOW.as_secs() / STEER_HISTORY_BUCKETS;
    let mut out = vec![0u64; STEER_HISTORY_BUCKETS as usize];
    let Some(q) = hit_map.get_mut(pool) else {
        return out;
    };
    while let Some((front, _)) = q.front()
        && now.duration_since(*front) >= STEER_WINDOW
    {
        q.pop_front();
    }
    for (t, m) in q.iter() {
        if m != member {
            continue;
        }
        let elapsed = now.duration_since(*t).as_secs();
        if let Some(idx) = (STEER_HISTORY_BUCKETS - 1).checked_sub(elapsed / bucket_secs.max(1))
            && let Some(slot) = out.get_mut(idx as usize)
        {
            *slot += 1;
        }
    }
    out
}

/// Per-bucket steering history for one pool: the rolling window sliced
/// into [`STEER_HISTORY_BUCKETS`] equal spans (20s each), oldest
/// bucket first, as `(requests, hits)` counts. This is what the CLI
/// renders as a sparkline - no traffic in a bucket reads as (0, 0),
/// which the renderer shows as a gap instead of a zero rate.
#[allow(clippy::cast_possible_truncation)] // bucket index math, always < 15
pub(crate) fn window_history(
    req_map: &mut std::collections::HashMap<String, std::collections::VecDeque<Instant>>,
    hit_map: &mut std::collections::HashMap<String, std::collections::VecDeque<(Instant, String)>>,
    pool: &str,
    now: Instant,
) -> Vec<(u64, u64)> {
    let bucket_secs = STEER_WINDOW.as_secs() / STEER_HISTORY_BUCKETS;
    let mut out = vec![(0u64, 0u64); STEER_HISTORY_BUCKETS as usize];
    {
        if let Some(q) = req_map.get_mut(pool) {
            while let Some(front) = q.front()
                && now.duration_since(*front) >= STEER_WINDOW
            {
                q.pop_front();
            }
            for t in q.iter() {
                let elapsed = now.duration_since(*t).as_secs();
                // Newest events land in the LAST bucket; anything older
                // than the window (only possible for a caller-built queue,
                // pruning keeps real ones clean) is skipped, not wrapped.
                if let Some(idx) =
                    (STEER_HISTORY_BUCKETS - 1).checked_sub(elapsed / bucket_secs.max(1))
                    && let Some(slot) = out.get_mut(idx as usize)
                {
                    slot.0 += 1;
                }
            }
        }
    }
    bucketize_hits(hit_map, pool, now, bucket_secs, &mut out);
    out
}

/// Hit half of [`window_history`]: member-tagged events, counted
/// member-agnostically so the pool trend stays the sum of its parts.
#[allow(clippy::cast_possible_truncation)] // bucket index math, always < 15
fn bucketize_hits(
    hit_map: &mut std::collections::HashMap<String, std::collections::VecDeque<(Instant, String)>>,
    pool: &str,
    now: Instant,
    bucket_secs: u64,
    out: &mut [(u64, u64)],
) {
    let Some(q) = hit_map.get_mut(pool) else {
        return;
    };
    while let Some((front, _)) = q.front()
        && now.duration_since(*front) >= STEER_WINDOW
    {
        q.pop_front();
    }
    for (t, _) in q.iter() {
        let elapsed = now.duration_since(*t).as_secs();
        if let Some(idx) = (STEER_HISTORY_BUCKETS - 1).checked_sub(elapsed / bucket_secs.max(1))
            && let Some(slot) = out.get_mut(idx as usize)
        {
            slot.1 += 1;
        }
    }
}
/// Bound on tracked prompt prefixes per gateway (memory hygiene);
/// expired entries are dropped first, a still-full map is cleared.
const AFFINITY_CAP: usize = 256;

/// Response header carrying the member a pool request actually served
/// (`local:model` or `peer:model`). Set by the five pool-aware lanes
/// when expansion fires; the usage meter reads it to attribute the
/// request to the serving member alongside the requested alias, and
/// clients may inspect it for transparency.
pub const EFFECTIVE_MODEL_HEADER: &str = "x-blazar-effective-model";

/// The affinity discount for an entry of this age (see
/// [`AFFINITY_FULL_MS`]). Pure integer math, unit-pinned.
fn affinity_discount(age: std::time::Duration) -> u64 {
    if age <= AFFINITY_FRESH {
        return AFFINITY_FULL_MS;
    }
    if age >= AFFINITY_TTL {
        return 0;
    }
    let span = AFFINITY_TTL
        .checked_sub(AFFINITY_FRESH)
        .expect("TTL outlives FRESH by construction")
        .as_secs();
    let left = AFFINITY_TTL
        .checked_sub(age)
        .expect("age < TTL guarded above")
        .as_secs();
    // ratio in basis points avoids float math: left/span in [0,1).
    AFFINITY_FULL_MS * left / span
}

/// Stable-ish key for one conversation prefix: the first message's
/// text (string content, or the first `text` part of multimodal
/// content), truncated to 512 chars. Only the OPENING of the prompt
/// decides affinity - the prefix whose KV reuse matters is the one
/// both turns share. `None` when the body has no textual opener.
fn affinity_key(pool: &str, body: &serde_json::Value) -> Option<String> {
    use std::hash::{Hash as _, Hasher as _};
    let first = body.get("messages")?.as_array()?.first()?;
    let content = match first.get("content") {
        Some(serde_json::Value::String(s)) => s.as_str(),
        Some(serde_json::Value::Array(parts)) => parts
            .iter()
            .find_map(|p| p.get("text").and_then(|t| t.as_str()))
            .unwrap_or_default(),
        _ => "",
    };
    let prefix: String = content.chars().take(512).collect();
    if prefix.is_empty() {
        return None;
    }
    let mut h = std::collections::hash_map::DefaultHasher::new();
    prefix.hash(&mut h);
    Some(format!("{pool}:{:016x}", h.finish()))
}

/// Blend a measured service EWMA toward the declared default as it
/// ages: fresh observations pass through verbatim, observations older
/// than [`SERVICE_STALE_AFTER`] are indistinguishable from never
/// having been measured (the no-signal `warm_first` parity contract),
/// and ages in between interpolate linearly. Returns the effective
/// milliseconds and whether a measurement still informs the value.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)] // SERVICE_DEFAULT_MS is the exact constant 750; blend math is advisory
pub(crate) fn effective_service(measured: f64, age: std::time::Duration) -> (u64, bool) {
    let (blended, live) = decay_toward(
        measured,
        SERVICE_DEFAULT_MS as f64,
        age,
        SERVICE_STALE_AFTER,
    );
    (blended.ceil() as u64, live)
}

/// Local-member TTFT counterpart of [`effective_service`]: a measured
/// first-token EWMA blended toward the declared default as it ages.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn effective_ttft(measured: f64, age: std::time::Duration) -> (u64, bool) {
    let (blended, live) = decay_toward(measured, LOCAL_TTFT_DEFAULT_MS, age, SERVICE_STALE_AFTER);
    (blended.ceil() as u64, live)
}

/// Shared staleness curve: fresh observations pass through verbatim,
/// observations older than `horizon` fully decay to `target`, and
/// ages in between interpolate linearly. Returns the blended value
/// and whether a measurement still informs it.
#[allow(clippy::cast_precision_loss)]
fn decay_toward(
    measured: f64,
    target: f64,
    age: std::time::Duration,
    horizon: std::time::Duration,
) -> (f64, bool) {
    let factor = (age.as_secs_f64() / horizon.as_secs_f64()).clamp(0.0, 1.0);
    (measured + (target - measured) * factor, factor < 1.0)
}

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
// A scoring row: strategy inputs plus operator-state flags. Bools
// read clearer than a bitfield here (each names one exact override).
#[allow(clippy::struct_excessive_bools)]
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
    /// When the breaker has this member down: seconds until the next
    /// retry attempt (`down_until` minus now). `None` while healthy.
    pub retry_in_secs: Option<u64>,
    /// Operator drain (`POST /api/pools/{name}/drain`): the member
    /// receives no new picks while any alternative exists. Never
    /// persists across restarts.
    pub drained: bool,
    /// Remaining seconds of a TTL drain (None = no deadline set or
    /// not drained) - what `blazar pool watch` counts down during a
    /// maintenance window.
    pub drained_for_secs: Option<u64>,
    /// Operator pin (`POST /api/pools/{name}/pin`): outranks every
    /// strategy while the member's peer is healthy.
    pub pinned: bool,
}

/// Runtime operator override for one pool member, keyed
/// `"{pool}:{effective}"` in [`AppState::pools_ops`]. Drain optionally
/// carries a deadline (maintenance window) after which the member
/// quietly rejoins the rotation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemberOps {
    pub drained: bool,
    pub drained_until: Option<Instant>,
    pub pinned: bool,
}

impl MemberOps {
    /// Drain is active while the flag is set and the optional deadline
    /// (a maintenance-window TTL) has not passed.
    #[must_use]
    pub fn is_drained(&self, now: Instant) -> bool {
        self.drained
            && match self.drained_until {
                None => true,
                Some(deadline) => now < deadline,
            }
    }
}

/// Rehydrate operator overrides from persisted rows (store tuples:
/// `(key, drained, drained_until_unix, pinned)`). Deadlines persist
/// as absolute unix seconds so a restart converts wall-clock back to
/// the monotonic clock: `now + remaining`. Expired rows are dropped
/// (and reported by key so the caller can clear them from the store);
/// a drained member never rehydrates its pin - drain beats pin at
/// read exactly as the live handler enforces.
#[must_use]
pub fn member_ops_from_persisted(
    rows: &[(String, bool, Option<i64>, bool)],
    now_unix: i64,
) -> (std::collections::HashMap<String, MemberOps>, Vec<String>) {
    let mut live = std::collections::HashMap::new();
    let mut expired = Vec::new();
    for (key, drained, drained_until_unix, pinned) in rows {
        let drained = *drained;
        let mut ops = MemberOps {
            drained,
            drained_until: None,
            pinned: *pinned && !drained,
        };
        if let (true, Some(deadline)) = (drained, drained_until_unix) {
            let remaining = deadline - now_unix;
            if remaining <= 0 {
                expired.push(key.clone());
                continue;
            }
            ops.drained_until =
                Some(Instant::now() + std::time::Duration::from_secs(remaining.cast_unsigned()));
        }
        live.insert(key.clone(), ops);
    }
    (live, expired)
}

/// Restore persisted cache-affinity rows into the live map. Rows
/// older than [`AFFINITY_TTL`] are dead by definition - they are
/// skipped and reported by key so the caller clears them from the
/// store. `at_unix` converts back to the monotonic clock as
/// "now minus age"; a wall clock older than the process (or an
/// unreadable instant) is treated as stale rather than guessed.
#[must_use]
pub fn affinity_from_persisted(
    rows: &[(String, String, i64)],
    now_unix: i64,
) -> (HashMap<String, (String, Instant)>, Vec<String>) {
    let mut live = HashMap::new();
    let mut stale = Vec::new();
    let now = Instant::now();
    for (key, effective, at_unix) in rows {
        let age_secs = now_unix.saturating_sub(*at_unix).max(0);
        if age_secs >= AFFINITY_TTL.as_secs().cast_signed() {
            stale.push(key.clone());
            continue;
        }
        // Rebuild the monotonic stamp: fresh entries keep most of
        // their discount window across the restart.
        let stamped = now
            .checked_sub(std::time::Duration::from_secs(age_secs.cast_unsigned()))
            .unwrap_or(now);
        live.insert(key.clone(), (effective.clone(), stamped));
    }
    (live, stale)
}
/// Rebuild the rolling steering-rate windows from persisted event
/// rows (`pool`, `kind`, `member`, `at_unix` - the shape
/// [`crate::blazar_core` store's `load_pool_windows`] returns).
/// Events older than the window are dropped (a restart longer than
/// the window starts clean, exactly like an idle daemon), each
/// surviving event is re-anchored onto the monotonic clock, and the
/// per-pool cap is enforced by shedding the oldest entries first.
/// Rows arrive ordered oldest-first; the rebuilt deques preserve
/// that order so their backs are the newest events.
#[must_use]
#[allow(clippy::type_complexity)] // mirrors AppState's window map shapes verbatim
pub fn windows_from_persisted(
    rows: &[(String, String, Option<String>, i64)],
    now_unix: i64,
    now: Instant,
) -> (
    HashMap<String, std::collections::VecDeque<Instant>>,
    HashMap<String, std::collections::VecDeque<(Instant, String)>>,
) {
    let window_secs = STEER_WINDOW.as_secs().cast_signed();
    let mut req: HashMap<String, std::collections::VecDeque<Instant>> = HashMap::new();
    let mut hits: HashMap<String, std::collections::VecDeque<(Instant, String)>> = HashMap::new();
    for (pool, kind, member, at_unix) in rows {
        let age = now_unix.saturating_sub(*at_unix).max(0);
        if age >= window_secs {
            continue;
        }
        let stamped = now
            .checked_sub(std::time::Duration::from_secs(age.cast_unsigned()))
            .unwrap_or(now);
        match kind.as_str() {
            "req" => {
                let lane = req.entry(pool.clone()).or_default();
                lane.push_back(stamped);
                while lane.len() > STEER_WINDOW_CAP {
                    lane.pop_front();
                }
            }
            "hit" => {
                let Some(member) = member.as_deref() else {
                    continue;
                };
                let lane = hits.entry(pool.clone()).or_default();
                lane.push_back((stamped, member.to_string()));
                while lane.len() > STEER_WINDOW_CAP {
                    lane.pop_front();
                }
            }
            _ => {}
        }
    }
    (req, hits)
}

/// Local-member model strings across every pool (deduped) - the
/// candidates whose affinity entries die when that model's row
/// unloads. Remote members are out of scope: a peer's eviction is
/// invisible here, and the TTL (120 s) reaps those entries anyway.
#[must_use]
pub fn local_member_strings(pools: &[ModelPool]) -> Vec<String> {
    let mut out: Vec<String> = pools
        .iter()
        .flat_map(|p| p.members.iter().filter(|m| m.peer == "local"))
        .map(|m| m.model.clone())
        .collect();
    out.sort();
    out.dedup();
    out
}

/// Drop affinity openers whose recorded member is one of `owned`
/// (local member strings that resolved to the evicted row). Pure
/// filter over a cloned snapshot - callers pass the lock's contents.
#[must_use]
#[allow(clippy::implicit_hasher)] // mirrors the AppState map type verbatim
pub fn affinity_keys_dropping_for(
    entries: &HashMap<String, (String, std::time::Instant)>,
    owned: &[String],
) -> Vec<String> {
    entries
        .iter()
        .filter(|(_, (eff, _))| owned.iter().any(|m| m == eff))
        .map(|(k, _)| k.clone())
        .collect()
}

/// Notify pool affinity that a local model row just unloaded: every
/// opener steered toward a member of that row loses its bonus - the
/// KV cache that made the member preferable is gone. Store resolution
/// runs BEFORE taking the affinity lock (no lock inversion).
pub fn on_model_evicted(state: &crate::state::AppState, row: &str) {
    let candidates = local_member_strings(&state.cfg().model_pools);
    let owned: Vec<String> = state
        .with_store(|s| {
            candidates
                .iter()
                .filter(|m| crate::proxy::resolve_model(s, m).is_ok_and(|r| r.name == row))
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    if owned.is_empty() {
        return;
    }
    if let Ok(mut aff) = state.pools_affinity.lock() {
        let drop_keys = affinity_keys_dropping_for(&aff, &owned);
        for k in drop_keys {
            aff.remove(&k);
        }
    }
}

/// Key under which a member's [`MemberOps`] live.
#[must_use]
pub fn ops_key(pool: &str, effective: &str) -> String {
    format!("{pool}:{effective}")
}

/// Whether a persisted ops key belongs to a still-configured pool.
/// Keys are `{pool}:{effective}` and effective strings may contain
/// colons themselves (`peer:model`), so the test is an exact
/// `{pool-name}:` prefix against every configured pool - a key
/// matching no pool is inert (nothing reads it) and drops safely at
/// boot when pool config has changed since it was written.
#[must_use]
pub fn ops_key_pool_known(pools: &[ModelPool], key: &str) -> bool {
    pools
        .iter()
        .any(|p| key.starts_with(&format!("{}:", p.name)))
}

/// Rolling-maintenance plan: draining a whole pool staggers the
/// members' return windows in declaration order - member `i`
/// re-enters rotation after `ttl + i * stagger` seconds, so replicas
/// come back one at a time instead of thundering back together.
/// Saturating arithmetic: an absurd stagger plan drains everyone for
/// "as long as possible" rather than overflowing.
#[must_use]
pub fn stagger_ttls(count: usize, ttl_secs: u64, stagger_secs: u64) -> Vec<u64> {
    (0..count)
        .map(|i| ttl_secs.saturating_add(stagger_secs.saturating_mul(i as u64)))
        .collect()
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
    local_service: &HashMap<String, (f64, Instant)>,
    local_ttft: &HashMap<String, (f64, Instant)>,
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
                    Some((ewma, at)) => effective_service(*ewma, now.duration_since(*at)),
                    None => (SERVICE_DEFAULT_MS, false),
                };
                // Queue waves multiply the MEASURED local TTFT when the
                // tap lane has one (staleness-decayed toward the
                // declared default), mirroring how remote members
                // scale by their per-peer `ttft_ewma_ms`.
                let ttft_ms = local_ttft
                    .get(&row_name)
                    .or_else(|| local_ttft.get(&member.model))
                    .map_or(LOCAL_TTFT_DEFAULT_MS as u64, |(v, at)| {
                        effective_ttft(*v, now.duration_since(*at)).0
                    });
                MemberScore {
                    idx,
                    effective: member.model.clone(),
                    score: PeerScore {
                        tier,
                        est_wait_ms: waves.saturating_mul(ttft_ms),
                        free_vram: std::cmp::Reverse(0),
                    },
                    service_ms,
                    measured,
                    down: false,
                    retry_in_secs: None,
                    drained: false,
                    drained_for_secs: None,
                    pinned: false,
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
                let retry_in_secs = if down {
                    h.down_until
                        .and_then(|t| t.checked_duration_since(now))
                        .map(|left| left.as_secs())
                } else {
                    None
                };
                let (service_ms, measured) = match (h.service_ewma_ms, h.last_service_at) {
                    (Some(ewma), Some(at)) => effective_service(ewma, now.duration_since(at)),
                    _ => (SERVICE_DEFAULT_MS, false),
                };
                MemberScore {
                    idx,
                    effective: format!("{}:{}", remote.name, member.model),
                    score: remotes::score_peer(&member.model, &h, cap),
                    service_ms,
                    measured,
                    down,
                    retry_in_secs,
                    drained: false,
                    drained_for_secs: None,
                    pinned: false,
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
    let local_ttft = state
        .ttft_ewma_local
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
    let now = Instant::now();
    let mut members = score_members_core(
        pool,
        &state.cfg(),
        queue_depth,
        &ps_rows,
        &healths,
        &caps,
        &local_service,
        &local_ttft,
        &local_names,
        now,
    );
    // Operator overrides (drain/pin) stamp on top of the scored view;
    // lazy deadline expiry keeps the map honest without a sweeper.
    if let Ok(ops) = state.pools_ops.lock() {
        for m in &mut members {
            if let Some(op) = ops.get(&ops_key(&pool.name, &m.effective)).copied() {
                m.drained = op.is_drained(now);
                m.drained_for_secs = op
                    .drained_until
                    .filter(|_| m.drained)
                    .and_then(|deadline| deadline.checked_duration_since(now))
                    .map(|left| left.as_secs());
                // A drained member can never be pinned (the API refuses
                // the combination; expired drains release the pin slot).
                m.pinned = op.pinned && !m.drained;
            }
        }
    }
    members
}

/// Pick the serving member per the pool strategy. Ties break to
/// declaration order (stable min over `idx`).
///
/// Eligibility first: members whose peer is marked down or that the
/// operator drained are excluded from every strategy. When EVERY
/// member is excluded the strategy runs over the raw list instead -
/// pools never hard-fail on health or drains alone, so the request
/// surfaces the real state (the peer's error, or the drained member
/// itself) rather than a confusing not-found on the pool name.
///
/// A pinned member returns before any strategy arm: pinning is an
/// operator override, not a strategy signal, and operator intent
/// outranks strategy choice - including `ordered`'s declaration
/// determinism (the operator IS reordering, deliberately). A pinned
/// member that goes down or gets drained re-enters the normal
/// eligibility rules automatically.
///
/// `sticky` is the pool's last adaptive pick (hysteresis): a healthy
/// member within [`POOL_STICKY_MARGIN`] of the best expected
/// completion and inside [`POOL_STICKY_WINDOW`] is retained. Other
/// strategies ignore it.
pub(crate) fn pick<'a>(
    pool: &ModelPool,
    members: &'a [MemberScore],
    sticky: Option<&StickyPick>,
    affinity: Option<(&str, u64)>,
) -> Option<&'a MemberScore> {
    let view: Vec<&MemberScore> = members.iter().filter(|m| !m.down && !m.drained).collect();
    let all: Vec<&MemberScore> = members.iter().collect();
    let list: &[&MemberScore] = if view.is_empty() { &all } else { &view };
    if let Some(pinned) = list.iter().copied().find(|m| m.pinned) {
        return Some(pinned);
    }
    match pool.strategy {
        // Failover posture: first eligible member in declaration
        // order (drains and breaker state skip ahead).
        PoolStrategy::Ordered => list.first().copied(),
        // Wait-first posture: the affinity member's re-prefill cost
        // drops out of its wait estimate exactly as it does for
        // `adaptive` - a member holding the prefix KV serves this
        // opener sooner than its raw queue waves suggest.
        PoolStrategy::LeastWait => {
            let wait = |m: &MemberScore| match affinity {
                Some((eff, discount)) if m.effective == eff => {
                    m.score.est_wait_ms.saturating_sub(discount)
                }
                _ => m.score.est_wait_ms,
            };
            list.iter().copied().min_by_key(|m| (wait(m), m.idx))
        }
        // Warm-first: the PeerScore order (tier, wait, free VRAM) with
        // the same affinity discount the other scoring strategies
        // apply - among equally warm members, the one holding this
        // opener's prefix KV genuinely serves sooner.
        PoolStrategy::WarmFirst => {
            let rank = |m: &MemberScore| {
                let wait = match affinity {
                    Some((eff, discount)) if m.effective == eff => {
                        m.score.est_wait_ms.saturating_sub(discount)
                    }
                    _ => m.score.est_wait_ms,
                };
                (m.score.tier, wait, m.score.free_vram)
            };
            list.iter().copied().min_by_key(|m| (rank(m), m.idx))
        }
        // Least expected completion: wait + service time, warm tier as
        // the tie-break. Until at least one ELIGIBLE member carries a
        // measured service EWMA the service term is the declared
        // default for everyone and the completion order collapses onto
        // the warm-first order (documented parity, pinned by test).
        PoolStrategy::Adaptive => {
            if !list.iter().any(|m| m.measured) {
                return list.iter().copied().min_by_key(|m| (m.score, m.idx));
            }
            // Cache affinity: the member that recently served this
            // prompt prefix likely still holds its KV cache, so its
            // re-prefill cost (about one TTFT) drops out of the
            // completion estimate. The discount is request-scoped -
            // introspection (`route_json`) passes `None` and reports
            // the affinity-free estimate.
            let total = |m: &MemberScore| {
                let base = m.score.est_wait_ms.saturating_add(m.service_ms);
                match affinity {
                    Some((eff, discount)) if m.effective == eff => base.saturating_sub(discount),
                    _ => base,
                }
            };
            let best_total = list.iter().copied().map(total).min()?;
            let best = list
                .iter()
                .copied()
                .min_by_key(|m| (total(m), m.score.tier, m.idx))?;
            // Hysteresis: keep the previous pick when it is still
            // healthy, fresh, and within the margin of the best
            // estimate - small estimate noise must not reshuffle
            // members (and KV caches) request to request. The margin
            // stays in integer math: best + best/10 = +10%.
            if let Some(s) = sticky
                && s.at.elapsed() < POOL_STICKY_WINDOW
                && let Some(current) = list.iter().copied().find(|m| m.effective == s.effective)
            {
                let current_total = total(current);
                let threshold = best_total.saturating_add(best_total / 10);
                if current_total <= threshold {
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
    expand_with(state, requested, None)
}

/// [`expand`] with an optional request-scoped cache-affinity hint
/// (effective member + discount) for the adaptive arm.
fn expand_with(state: &AppState, requested: &str, affinity: Option<(&str, u64)>) -> Option<String> {
    let cfg = state.cfg();
    let pool = find(&cfg, requested)?;
    let members = score_members(state, pool);
    let sticky = state
        .pools_sticky
        .lock()
        .ok()
        .and_then(|m| m.get(&pool.name).cloned());
    let picked = pick(pool, &members, sticky.as_ref(), affinity)?;
    // Denominator for the steering hit-rate: every successful
    // expansion routes one pool request, whatever lane initiated it
    // (chat, generate, responses, count-tokens).
    if let Ok(mut reqs) = state.pools_requests.lock() {
        *reqs.entry(pool.name.clone()).or_insert(0) += 1;
    }
    if let Ok(mut win) = state.pools_req_window.lock() {
        window_push(&mut win, &pool.name, Instant::now());
    }
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
///
/// Adaptive pools additionally consult and record per-prefix cache
/// affinity: the member that served the same opening prompt recently
/// gets its re-prefill cost discounted (see [`AFFINITY_FULL_MS`]).
pub fn expand_request(state: &AppState, requested: &str, body: &Bytes) -> Option<(String, Bytes)> {
    let cfg = state.cfg();
    let pool = find(&cfg, requested)?;
    let parsed = serde_json::from_slice::<serde_json::Value>(body).ok();
    let key = parsed.as_ref().and_then(|v| affinity_key(&pool.name, v));
    let affinity = key.as_ref().and_then(|k| {
        state
            .pools_affinity
            .lock()
            .ok()
            .and_then(|m| m.get(k).cloned())
            .and_then(|(effective, at)| {
                let age = Instant::now().duration_since(at);
                let discount = affinity_discount(age);
                (discount > 0).then_some((effective, discount))
            })
    });
    let effective = expand_with(
        state,
        requested,
        affinity.as_ref().map(|(e, d)| (e.as_str(), *d)),
    )?;
    if let Some(k) = key
        && let Ok(mut m) = state.pools_affinity.lock()
    {
        // Hygiene: drop expired entries at the cap, clear if still
        // full (a burst of unique prefixes is not worth unbounded
        // memory; the next request re-seeds).
        if m.len() >= AFFINITY_CAP {
            m.retain(|_, (_, at)| at.elapsed() < AFFINITY_TTL);
            if m.len() >= AFFINITY_CAP {
                m.clear();
            }
        }
        // Persistence rides the operator-override flag: write only
        // when this opener is NEW or switched members, so steady
        // same-opener traffic never touches the store. Cap-hygiene
        // evictions are not mirrored - stale rows self-clear at boot.
        let changed = m.get(&k).is_none_or(|(prev, _)| *prev != effective);
        m.insert(k.clone(), (effective.clone(), Instant::now()));
        if changed && cfg.pools_ops_persist {
            let at_unix = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs().cast_signed());
            let _ = state.with_store(|s| s.put_pool_affinity(&k, &effective, at_unix));
        }
    }
    // Steering telemetry: this request HAD a live affinity hint and
    // landed on the member it pointed at - the opener's prefix KV is
    // warm where it arrived. Session-only counter for the route table.
    if let Some((member, _)) = affinity.as_ref()
        && member == &effective
    {
        if let Ok(mut hits) = state.pools_affinity_hits.lock() {
            *hits.entry(ops_key(&pool.name, &effective)).or_insert(0) += 1;
        }
        if let Ok(mut win) = state.pools_hit_window.lock() {
            window_push_tagged(&mut win, &pool.name, &effective, Instant::now());
        }
    }
    let rewritten = remotes::rewrite_model(body.clone(), &effective);
    Some((effective, rewritten))
}

/// Introspection for `GET /api/route/{name}` (admin gate applies at
/// the route): the full member table with scores and the decision,
/// same envelope as the existing route payload so `blazar route`
/// renders pools without a new client path.
#[allow(clippy::too_many_lines)] // introspection payload assembly, one member table
pub fn route_json(state: &AppState, pool: &ModelPool) -> serde_json::Value {
    let members = score_members(state, pool);
    // Introspection reads (never writes) the sticky state so the
    // reported decision matches what the request path would do.
    let sticky = state
        .pools_sticky
        .lock()
        .ok()
        .and_then(|m| m.get(&pool.name).cloned());
    let chosen = pick(pool, &members, sticky.as_ref(), None);
    // Live affinity keys for THIS pool: bounded map, "{pool}:{hash}"
    // keys, each holding the member that served the opener last.
    // Read-only introspection - the request path records.
    let affinity_now = std::time::Instant::now();
    let affinity_entries: Vec<(String, std::time::Instant)> = state
        .pools_affinity
        .lock()
        .map(|m| {
            m.iter()
                .filter(|(k, _)| k.strip_prefix(&format!("{}:", pool.name)).is_some())
                .map(|(_, (eff, at))| (eff.clone(), *at))
                .collect()
        })
        .unwrap_or_default();
    let affinity_live: Vec<&(String, std::time::Instant)> = affinity_entries
        .iter()
        .filter(|(_, at)| affinity_now.duration_since(*at) < AFFINITY_TTL)
        .collect();
    // Steering counters ride the same shape ("{pool}:{effective}").
    let affinity_hits: std::collections::HashMap<String, u64> = state
        .pools_affinity_hits
        .lock()
        .map(|h| {
            h.iter()
                .filter(|(k, _)| k.strip_prefix(&format!("{}:", pool.name)).is_some())
                .map(|(k, v)| (k.clone(), *v))
                .collect()
        })
        .unwrap_or_default();
    // Pool-level steering rate: hits against every routed request for
    // this pool (session counters both). Percent with one decimal so
    // small fleets read honestly (1 of 3 = 33.3, not "33").
    let pool_requests = state
        .pools_requests
        .lock()
        .ok()
        .and_then(|r| r.get(&pool.name).copied())
        .unwrap_or(0);
    // Rolling rate: windowed events on both sides, so current traffic
    // speaks - a burst that steered 10 minutes ago does not pin the
    // percentage forever. Lifetime totals stay visible alongside.
    let steer_now = Instant::now();
    let window_req = state
        .pools_req_window
        .lock()
        .map_or(0, |mut w| window_count(&mut w, &pool.name, steer_now));
    let window_hits = state.pools_hit_window.lock().map_or(0, |mut w| {
        window_count_tagged(&mut w, &pool.name, steer_now)
    });
    let steer_rate = if window_req > 0 {
        #[allow(clippy::cast_precision_loss)] // advisory percent, 0.1 resolution
        let pct = window_hits as f64 / window_req as f64 * 100.0;
        Some((pct * 10.0).round() / 10.0)
    } else {
        None
    };
    // Sparkline history: the same rolling window sliced into equal
    // buckets (oldest first). Fixed lock order (requests, then hits)
    // - this is the only site that holds both guards. The per-member
    // hit trends come out of the same dual-lock so each member's
    // `hit_history` slices only its own steering events.
    let (history, member_trends): (
        Vec<serde_json::Value>,
        std::collections::HashMap<String, Vec<u64>>,
    ) = match (state.pools_req_window.lock(), state.pools_hit_window.lock()) {
        (Ok(mut reqs), Ok(mut hits)) => {
            let history = window_history(&mut reqs, &mut hits, &pool.name, steer_now)
                .into_iter()
                .map(|(reqs, hits)| serde_json::json!({ "reqs": reqs, "hits": hits }))
                .collect();
            let trends = members
                .iter()
                .map(|m| {
                    (
                        m.effective.clone(),
                        member_hit_history(&mut hits, &pool.name, &m.effective, steer_now),
                    )
                })
                .collect();
            (history, trends)
        }
        _ => (Vec::new(), std::collections::HashMap::new()),
    };
    let member_lines: Vec<serde_json::Value> = members
        .iter()
        .map(|m| {
            // Per-member trend ships only when this member actually
            // steered something in the window - silent members keep
            // the payload lean.
            let hit_history = member_trends
                .get(&m.effective)
                .filter(|h| h.iter().any(|&n| n > 0));
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
                "retry_in_secs": m.retry_in_secs,
                "drained": m.drained,
                "drained_for_secs": m.drained_for_secs,
                "pinned": m.pinned,
                "affinity_keys": affinity_live
                    .iter()
                    .filter(|(eff, _)| *eff == m.effective)
                    .count(),
                "affinity_hits": affinity_hits
                    .get(&ops_key(&pool.name, &m.effective))
                    .copied()
                    .unwrap_or(0),
                "hit_history": hit_history,
                "affinity_freshest_ms": affinity_live
                    .iter()
                    .filter(|(eff, _)| *eff == m.effective)
                    .map(|(_, at)| {
                        u64::try_from(affinity_now.duration_since(*at).as_millis())
                            .unwrap_or(u64::MAX)
                    })
                    .min(),
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
                    "pool '{}' picked member {} under {:?} (tier {}, est wait {} ms, service {} ms{}{})",
                    pool.name,
                    m.effective,
                    pool.strategy,
                    crate::federation::tier_name(m.score.tier),
                    m.score.est_wait_ms,
                    m.service_ms,
                    if m.pinned { ", operator pinned" } else { "" },
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
            "affinity": {
                "live_keys": affinity_live.len(),
                "ttl_secs": AFFINITY_TTL.as_secs(),
                "requests": pool_requests,
                "hit_rate_pct": steer_rate,
                "hit_rate_window_secs": STEER_WINDOW.as_secs(),
                "history_bucket_secs": STEER_WINDOW.as_secs() / STEER_HISTORY_BUCKETS,
                "history": history,
            },
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
            retry_in_secs: None,
            drained: false,
            drained_for_secs: None,
            pinned: false,
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
        let picked = pick(&p, &members, None, None).expect("local member eligible");
        assert_eq!(picked.idx, 0);

        let members = vec![
            scored(0, "far:m1", 0, 10, 0, true),
            scored(1, "m1", 0, 900, 0, false),
        ];
        let picked = pick(&p, &members, None, None).expect("second member eligible");
        assert_eq!(picked.idx, 1);
    }

    /// All members down still returns one (least-bad first) so the
    /// request surfaces the peer's real error, not a pool not-found.
    #[test]
    fn unit__pick__all_down_returns_least_bad_member() {
        let p = pool(PoolStrategy::Ordered);
        let members = vec![scored(0, "far:m1", 0, 10, 0, true)];
        let picked = pick(&p, &members, None, None).expect("degenerate pool still picks");
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
        let picked = pick(&p, &members, None, None).expect("two members");
        assert_eq!(picked.idx, 1, "cold peer with the smaller wait must win");
    }

    /// The affinity member's discount applies to `least_wait` too: on
    /// an otherwise equal wait the member holding the prefix KV wins,
    /// and it can even close a raw-wait gap up to the discount.
    #[test]
    fn unit__pick__least_wait_honors_affinity_discount() {
        let p = pool(PoolStrategy::LeastWait);
        let members = vec![
            scored(0, "m1", 0, 1000, 0, false),
            scored(1, "far:m1", 2, 1400, 0, false),
        ];
        // No affinity: raw wait order wins.
        let picked = pick(&p, &members, None, None).expect("two members");
        assert_eq!(picked.idx, 0);
        // Affinity for the peer discounts 750 (one full TTFT):
        // 1400 - 750 = 650 < 1000, so the KV-holding peer wins.
        let picked = pick(&p, &members, None, Some(("far:m1", 750))).expect("two members");
        assert_eq!(picked.idx, 1);
        // A discount smaller than the gap (300 < 400) must NOT flip.
        let picked = pick(&p, &members, None, Some(("far:m1", 300))).expect("two members");
        assert_eq!(picked.idx, 0);
    }

    /// A down peer never wins `least_wait` on a stale-low wait; the
    /// healthy member serves and the all-down case degrades to the
    /// raw minimum so the request still surfaces a real answer.
    #[test]
    fn unit__pick__least_wait_skips_down_members() {
        let p = pool(PoolStrategy::LeastWait);
        let members = vec![
            scored(0, "m1", 0, 1500, 0, false),
            scored(1, "far:m1", 2, 100, 0, true),
        ];
        let picked = pick(&p, &members, None, None).expect("two members");
        assert_eq!(picked.idx, 0, "down peer must not win on a stale wait");
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
        let picked = pick(&p, &members, None, None).expect("two members");
        assert_eq!(picked.idx, 1, "warm member wins despite larger wait");

        let members = vec![
            scored(0, "m1", 0, 750, 5, false),
            scored(1, "far:m1", 0, 750, 5, false),
        ];
        let picked = pick(&p, &members, None, None).expect("two members");
        assert_eq!(picked.idx, 0, "full tie breaks to declaration order");
    }

    /// The affinity discount reaches `warm_first` too: among equally
    /// warm members the one holding this opener's prefix KV wins, even
    /// across a raw-wait gap up to the discount - the KV holder
    /// genuinely serves sooner.
    #[test]
    fn unit__pick__warm_first_honors_affinity_discount() {
        let p = pool(PoolStrategy::WarmFirst);
        let members = vec![
            scored(0, "m1", 0, 1000, 5, false),
            scored(1, "far:m1", 0, 1400, 5, false),
        ];
        let picked = pick(&p, &members, None, None).expect("two members");
        assert_eq!(picked.idx, 0, "no affinity: smaller wait wins the tier tie");
        // 1400 - 750 = 650 < 1000: the KV-holding peer wins the tier.
        let picked = pick(&p, &members, None, Some(("far:m1", 750))).expect("two members");
        assert_eq!(picked.idx, 1);
        // 300 < 400 gap: the discount must NOT flip the tie.
        let picked = pick(&p, &members, None, Some(("far:m1", 300))).expect("two members");
        assert_eq!(picked.idx, 0);
    }

    /// A down peer never wins `warm_first` on a stale score; the
    /// all-down case degrades to the raw order so the request still
    /// surfaces a real answer from the member it would try anyway.
    #[test]
    fn unit__pick__warm_first_skips_down_members() {
        let p = pool(PoolStrategy::WarmFirst);
        let members = vec![
            scored(0, "m1", 0, 10, 9999, true),
            scored(1, "far:m1", 2, 800, 0, false),
        ];
        let picked = pick(&p, &members, None, None).expect("two members");
        assert_eq!(picked.idx, 1, "down member must not win on a stale score");
        let all_down = vec![
            scored(0, "m1", 0, 10, 9999, true),
            scored(1, "far:m1", 2, 800, 0, true),
        ];
        let picked = pick(&p, &all_down, None, None).expect("all-down still picks");
        assert_eq!(picked.idx, 0, "all-down degrades to the raw order");
    }

    /// `ordered` stays pure declaration order regardless of affinity:
    /// determinism is that strategy's contract - operators pin a
    /// member sequence exactly because nothing reorders it.
    #[test]
    fn unit__pick__ordered_ignores_affinity() {
        let p = pool(PoolStrategy::Ordered);
        let members = vec![
            scored(0, "m1", 0, 1500, 0, false),
            scored(1, "far:m1", 0, 10, 9999, false),
        ];
        let picked = pick(&p, &members, None, Some(("far:m1", 750))).expect("two members");
        assert_eq!(picked.idx, 0, "ordered keeps declaration order");
    }

    /// `MemberOps::is_drained` truth table: an undrained entry never
    /// counts, a deadline-free drain lasts forever, a future deadline
    /// holds and an expired one self-restores the member.
    /// Persisted rows convert back into live ops: wall-clock
    /// deadlines become monotonic ones, expired rows are dropped and
    /// reported for store cleanup, and a row that is both drained and
    /// pinned keeps only the drain (drain-beats-pin at read, same as
    /// at write).
    #[test]
    fn unit__member_ops_from_persisted__conversion_truth_table() {
        let now_unix = 1_800_000_000_i64;
        let rows = vec![
            // Fresh drain with a deadline: live, deadline in the future.
            ("svc:m1".to_string(), true, Some(now_unix + 300), false),
            // Expired drain: dropped and reported.
            ("svc:old".to_string(), true, Some(now_unix - 5), false),
            // Permanent drain (no deadline) passes through.
            ("svc:perm".to_string(), true, None, false),
            // Drain beats pin on the same row.
            ("svc:both".to_string(), true, Some(now_unix + 60), true),
            // Pure pin survives.
            ("svc:pin".to_string(), false, None, true),
        ];
        let (live, expired) = member_ops_from_persisted(&rows, now_unix);
        assert_eq!(expired, vec!["svc:old".to_string()]);
        assert_eq!(live.len(), 4, "expired row dropped, rest live");
        let m1 = &live["svc:m1"];
        assert!(m1.drained);
        let deadline = m1.drained_until.expect("deadline mapped");
        let after = Instant::now();
        assert!(deadline > after, "deadline is in the future");
        assert!(
            deadline < after + std::time::Duration::from_secs(300),
            "deadline is the stored remaining window"
        );
        assert!(live["svc:perm"].drained_until.is_none());
        assert!(live["svc:both"].drained);
        assert!(!live["svc:both"].pinned, "drain beats pin at read");
        assert!(!live["svc:pin"].drained);
        assert!(live["svc:pin"].pinned);
    }

    /// Persisted affinity rows: fresh openers rehydrate with their
    /// monotonic stamp inside the TTL window, stale rows (older than
    /// `AFFINITY_TTL`) drop and report for store cleanup.
    #[test]
    fn unit__affinity_from_persisted__conversion_truth_table() {
        let now_unix = 1_800_000_000_i64;
        let rows = vec![
            // Fresh opener: 10s old, well inside the 120s TTL.
            ("svc:abc".to_string(), "m1".to_string(), now_unix - 10),
            // Boundary: exactly the TTL reads as stale (a discount of
            // zero carries no routing weight - nothing to restore).
            ("svc:edge".to_string(), "m1".to_string(), now_unix - 120),
            // Long dead: dropped and reported.
            ("svc:old".to_string(), "far:m1".to_string(), now_unix - 900),
        ];
        let (live, stale) = affinity_from_persisted(&rows, now_unix);
        assert_eq!(stale, vec!["svc:edge".to_string(), "svc:old".to_string()]);
        assert_eq!(live.len(), 1, "only the fresh opener rehydrates");
        let (effective, stamped) = &live["svc:abc"];
        assert_eq!(effective, "m1");
        // The rebuilt monotonic stamp is ~10s in the past: newer than
        // now-11s, older than now.
        let before = Instant::now()
            .checked_sub(std::time::Duration::from_secs(11))
            .expect("11s is always in the past on test hosts");
        let after = Instant::now();
        assert!(stamped > &before, "stamp reflects the stored age");
        assert!(stamped <= &after, "stamp is not in the future");
    }

    /// Persisted window events rebuild into the live deques: fresh
    /// req/hit events land in their maps (hits tagged), expired events
    /// of both kinds drop, a hit without a member tag is skipped, the
    /// cap sheds the oldest events, and deques stay chronological
    /// (back = newest) so bucket math keeps working after a restart.
    #[test]
    fn unit__windows_from_persisted__conversion_truth_table() {
        let now_unix = 1_800_000_000_i64;
        let now = Instant::now();
        let fresh = |secs: i64| now_unix - secs;
        let rows = vec![
            // Fresh req events for two pools.
            ("svc".into(), "req".into(), None, fresh(5)),
            ("svc".into(), "req".into(), None, fresh(3)),
            ("other".into(), "req".into(), None, fresh(1)),
            // Fresh tagged hit events.
            ("svc".into(), "hit".into(), Some("m1".into()), fresh(4)),
            ("svc".into(), "hit".into(), Some("far:m1".into()), fresh(2)),
            // Expired events of both kinds drop (>= 300s window).
            ("svc".into(), "req".into(), None, fresh(300)),
            ("svc".into(), "hit".into(), Some("m1".into()), fresh(301)),
            // A hit row without its member tag cannot steer anyone.
            ("svc".into(), "hit".into(), None, fresh(1)),
            // Unknown kinds are ignored outright.
            ("svc".into(), "bogus".into(), None, fresh(1)),
        ];
        let (req, hits) = windows_from_persisted(&rows, now_unix, now);
        assert_eq!(req["svc"].len(), 2);
        assert_eq!(req["other"].len(), 1);
        assert_eq!(hits["svc"].len(), 2);
        // Chronological: the back of each deque is the newest event.
        let ages: Vec<std::time::Duration> =
            req["svc"].iter().map(|t| now.duration_since(*t)).collect();
        assert!(ages[0] > ages[1], "older events sit at the front");
        assert!(hits["svc"].iter().any(|(_, m)| m == "far:m1"));

        // Cap: more fresh events than the cap (packed inside the
        // window - many events can share a second) keeps only the
        // newest STEER_WINDOW_CAP events.
        #[allow(clippy::cast_possible_wrap)]
        // test data: tiny indices reinterpreted as second offsets
        let long: Vec<(String, String, Option<String>, i64)> = (0..STEER_WINDOW_CAP + 30)
            .map(|i| ("svc".into(), "req".into(), None, fresh((i % 30) as i64)))
            .collect();
        let (req, _) = windows_from_persisted(&long, now_unix, now);
        assert_eq!(req["svc"].len(), STEER_WINDOW_CAP);
        let oldest = req["svc"].front().expect("cap leaves events");
        // Everything shed was pushed before the survivors: the front
        // is at most one 30s sweep old.
        assert!(
            now.duration_since(*oldest) < std::time::Duration::from_secs(30 + 2),
            "front is one of the newest events"
        );
    }

    /// Persisted-key pool matching: an exact `{pool}:` prefix keeps a
    /// row alive through config edits; keys for removed pools (and
    /// prefix lookalikes like `ab:` vs pool `a`) drop at boot.
    #[test]
    fn unit__ops_key_pool_known__prefix_truth_table() {
        let mk = |name: &str| ModelPool {
            name: name.into(),
            strategy: PoolStrategy::Ordered,
            members: vec![],
        };
        let pools = vec![mk("svc"), mk("a:b")];
        assert!(ops_key_pool_known(&pools, "svc:m1"));
        // Remote effectives carry their own colon - still the svc row.
        assert!(ops_key_pool_known(&pools, "svc:far:m1"));
        // A pool name containing a colon matches on its full prefix.
        assert!(ops_key_pool_known(&pools, "a:b:m1"));
        // Prefix lookalike must NOT match pool "svc"/"a:b".
        assert!(!ops_key_pool_known(&pools, "sv:m1"));
        assert!(!ops_key_pool_known(&pools, "a:bm1"));
        // A pool deleted from config: its rows become GC candidates.
        assert!(!ops_key_pool_known(&pools, "gone:m1"));
    }

    #[test]
    fn unit__member_ops__drain_deadline_truth_table() {
        let now = Instant::now();
        let undrained = MemberOps {
            drained: false,
            drained_until: Some(now + std::time::Duration::from_secs(60)),
            pinned: false,
        };
        assert!(!undrained.is_drained(now));
        let forever = MemberOps {
            drained: true,
            drained_until: None,
            pinned: false,
        };
        assert!(forever.is_drained(now));
        let ttl = MemberOps {
            drained: true,
            drained_until: Some(now + std::time::Duration::from_secs(60)),
            pinned: false,
        };
        assert!(ttl.is_drained(now));
        assert!(
            !ttl.is_drained(now + std::time::Duration::from_secs(61)),
            "an expired drain self-restores"
        );
    }

    /// A drained member is skipped while an alternative exists - even
    /// by `ordered`, where declaration order otherwise wins.
    #[test]
    fn unit__pick__drained_member_skipped_while_alternative_exists() {
        let p = pool(PoolStrategy::Ordered);
        let members = vec![
            MemberScore {
                drained: true,
                ..scored(0, "m1", 0, 10, 9999, false)
            },
            scored(1, "far:m1", 2, 1500, 0, false),
        ];
        let picked = pick(&p, &members, None, None).expect("two members");
        assert_eq!(picked.idx, 1, "drained first-declared member is skipped");
        // The exclusion reaches every strategy, not just ordered.
        let p = pool(PoolStrategy::LeastWait);
        let members = vec![
            MemberScore {
                drained: true,
                ..scored(0, "m1", 0, 10, 9999, false)
            },
            scored(1, "far:m1", 2, 1500, 0, false),
        ];
        let picked = pick(&p, &members, None, None).expect("two members");
        assert_eq!(
            picked.idx, 1,
            "drained member loses least_wait despite lower wait"
        );
    }

    /// When every member is down or drained the pool still serves
    /// through the least-bad one instead of failing the request -
    /// the unified fallback contract across all strategies.
    #[test]
    fn unit__pick__all_excluded_falls_back_to_raw_order() {
        for strategy in [
            PoolStrategy::Ordered,
            PoolStrategy::LeastWait,
            PoolStrategy::WarmFirst,
            PoolStrategy::Adaptive,
        ] {
            let p = pool(strategy);
            let members = vec![
                MemberScore {
                    drained: true,
                    ..scored(0, "m1", 0, 10, 9999, false)
                },
                scored(1, "far:m1", 2, 1500, 0, true),
            ];
            let picked = pick(&p, &members, None, None)
                .unwrap_or_else(|| panic!("{strategy:?} still picks"));
            assert_eq!(picked.idx, 0, "{strategy:?} falls back to the raw order");
        }
    }

    /// A pinned member outranks the strategy while healthy - even
    /// `ordered`'s declaration order (the operator IS the reorder) -
    /// and the pin auto-voids when the member is down or drained.
    #[test]
    fn unit__pick__pinned_member_outranks_strategy_until_unhealthy() {
        let p = pool(PoolStrategy::Ordered);
        let members = vec![
            scored(0, "m1", 0, 10, 9999, false),
            MemberScore {
                pinned: true,
                ..scored(1, "far:m1", 2, 1500, 0, false)
            },
        ];
        let picked = pick(&p, &members, None, None).expect("two members");
        assert_eq!(picked.idx, 1, "pin outranks ordered declaration order");
        let p = pool(PoolStrategy::LeastWait);
        let picked = pick(&p, &members, None, None).expect("two members");
        assert_eq!(picked.idx, 1, "pin outranks least_wait's smaller wait");
        // Down or drained voids the pin: no eligible member carries it.
        let down_pinned = vec![
            scored(0, "m1", 0, 10, 9999, false),
            MemberScore {
                pinned: true,
                ..scored(1, "far:m1", 2, 1500, 0, true)
            },
        ];
        let picked = pick(&p, &down_pinned, None, None).expect("two members");
        assert_eq!(picked.idx, 0, "a down pinned member loses its pin");
    }

    /// A drained-but-pinned member is excluded, not pinned: drain is
    /// the stronger operator statement (maintenance wins).
    #[test]
    fn unit__pick__drain_beats_pin() {
        let p = pool(PoolStrategy::Ordered);
        let members = vec![
            scored(0, "m1", 0, 10, 9999, false),
            MemberScore {
                drained: true,
                pinned: true,
                ..scored(1, "far:m1", 2, 1500, 0, false)
            },
        ];
        let picked = pick(&p, &members, None, None).expect("two members");
        assert_eq!(picked.idx, 0, "drained member excluded despite pin");
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
        let warm = pick(&pool(PoolStrategy::WarmFirst), &members, None, None);
        let adaptive = pick(&pool(PoolStrategy::Adaptive), &members, None, None);
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
        let warm = pick(&pool(PoolStrategy::WarmFirst), &members, None, None);
        let adaptive = pick(&pool(PoolStrategy::Adaptive), &members, None, None);
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
        let picked = pick(&pool(PoolStrategy::Adaptive), &members, Some(&sticky), None).unwrap();
        assert_eq!(picked.idx, 0, "sticky member within 10% margin is kept");

        // Challenger is >10% better (best 1000, sticky total 1200):
        // the sticky hold releases and the challenger wins.
        let members = vec![
            scored_service(0, "m1", 0, 1100, 100, true),
            scored_service(1, "far:m1", 0, 900, 100, true),
        ];
        let picked = pick(&pool(PoolStrategy::Adaptive), &members, Some(&sticky), None).unwrap();
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
        let picked = pick(&pool(PoolStrategy::Adaptive), &members, Some(&stale), None).unwrap();
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
        let picked = pick(&pool(PoolStrategy::Adaptive), &members, Some(&sticky), None).unwrap();
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
                last_service_at: None,
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
        local_service.insert("m1".to_string(), (4000.0, Instant::now()));
        let members = score_members_core(
            &local_pool,
            &cfg,
            0,
            &[],
            &HashMap::new(),
            &HashMap::new(),
            &local_service,
            &HashMap::new(),
            &HashMap::new(),
            Instant::now(),
        );
        assert_eq!(members[0].service_ms, 4000);
        assert!(members[0].measured, "m1 carries a stamped EWMA");
        assert_eq!(members[1].service_ms, SERVICE_DEFAULT_MS);
        assert!(!members[1].measured, "m2 has no observation yet");
    }

    /// Rolling-maintenance stagger: member i returns after
    /// `ttl + i * stagger` seconds, saturating instead of overflowing
    /// even for absurd inputs.
    #[test]
    fn unit__stagger_ttls__spaces_return_windows_in_declaration_order() {
        assert_eq!(stagger_ttls(3, 60, 30), vec![60, 90, 120]);
        assert_eq!(stagger_ttls(2, 60, 0), vec![60, 60]);
        // u64::MAX stagger must saturate, not wrap.
        assert_eq!(stagger_ttls(2, 10, u64::MAX), vec![10, u64::MAX]);
    }

    /// Staleness decay: a service EWMA older than the stale horizon is
    /// indistinguishable from never having been measured (default
    /// value, `measured = false`), restoring the no-signal parity
    /// contract for idle members.
    #[test]
    fn unit__score_members_core__stale_local_service_decays_to_default() {
        let cfg = cfg_with(None);
        let local_pool = ModelPool {
            name: "svc".into(),
            strategy: PoolStrategy::Adaptive,
            members: vec![member("local", "m1")],
        };
        let mut local_service = HashMap::new();
        let ancient = Instant::now()
            .checked_sub(std::time::Duration::from_secs(600))
            .expect("10 min is always in the past on test hosts");
        local_service.insert("m1".to_string(), (4000.0, ancient));
        let members = score_members_core(
            &local_pool,
            &cfg,
            0,
            &[],
            &HashMap::new(),
            &HashMap::new(),
            &local_service,
            &HashMap::new(),
            &HashMap::new(),
            Instant::now(),
        );
        assert_eq!(members[0].service_ms, SERVICE_DEFAULT_MS);
        assert!(!members[0].measured, "a stale observation no longer counts");
    }

    /// Local members scale queue waves by the MEASURED TTFT EWMA when
    /// the tap lane has one (mirroring remote `ttft_ewma_ms` scaling),
    /// and a fully stale TTFT falls back to the declared default.
    #[test]
    fn unit__score_members_core__local_est_wait_uses_measured_ttft() {
        let cfg = cfg_with(None);
        let local_pool = ModelPool {
            name: "svc".into(),
            strategy: PoolStrategy::LeastWait,
            members: vec![member("local", "m1"), member("local", "m2")],
        };
        let mut ttft = HashMap::new();
        ttft.insert("m1".to_string(), (3000.0, Instant::now()));
        let members = score_members_core(
            &local_pool,
            &cfg,
            2, // gateway queue depth -> 2 waves per member
            &[],
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &ttft,
            &HashMap::new(),
            Instant::now(),
        );
        assert_eq!(
            members[0].score.est_wait_ms, 6000,
            "measured 3000 ms TTFT x 2 queue waves"
        );
        assert_eq!(
            members[1].score.est_wait_ms, 1500,
            "unmeasured member keeps the 750 ms default x 2 waves"
        );

        // Stale measurement: 10 min old -> decayed to the default, so
        // both members wait identically again.
        let ancient = Instant::now()
            .checked_sub(std::time::Duration::from_secs(600))
            .expect("10 min is always in the past on test hosts");
        ttft.insert("m1".to_string(), (3000.0, ancient));
        let members = score_members_core(
            &local_pool,
            &cfg,
            2,
            &[],
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &ttft,
            &HashMap::new(),
            Instant::now(),
        );
        assert_eq!(members[0].score.est_wait_ms, 1500);
    }

    /// Local-member collection: dedupes across pools and ignores
    /// remote members - only local rows can be eviction-matched.
    #[test]
    fn unit__local_member_strings__local_only_and_deduped() {
        let pools = vec![
            ModelPool {
                name: "a".into(),
                strategy: PoolStrategy::Ordered,
                members: vec![member("local", "m1"), member("far", "m1")],
            },
            ModelPool {
                name: "b".into(),
                strategy: PoolStrategy::Ordered,
                members: vec![member("local", "m1"), member("local", "m2")],
            },
        ];
        let got = local_member_strings(&pools);
        assert_eq!(got, vec!["m1".to_string(), "m2".to_string()]);
    }

    /// The eviction filter: only openers recorded against an owned
    /// member string drop; everything else (other members, unowned
    /// keys) survives untouched.
    #[test]
    fn unit__affinity_keys_dropping_for__owned_only() {
        let now = Instant::now();
        let mut entries = HashMap::new();
        entries.insert("svc:h1".to_string(), ("m1".to_string(), now));
        entries.insert("svc:h2".to_string(), ("far:m1".to_string(), now));
        entries.insert("other:h3".to_string(), ("m1".to_string(), now));
        let owned = vec!["m1".to_string()];
        let mut drop_keys = affinity_keys_dropping_for(&entries, &owned);
        drop_keys.sort();
        assert_eq!(
            drop_keys,
            vec!["other:h3".to_string(), "svc:h1".to_string()]
        );
    }

    /// The decay blend itself: fresh observations pass through, the
    /// halfway age interpolates, and the horizon boundary flips to the
    /// unmeasured posture.
    #[test]
    fn unit__effective_service__blends_toward_default_with_age() {
        let (fresh, measured) = effective_service(4000.0, std::time::Duration::from_secs(0));
        assert_eq!((fresh, measured), (4000, true));

        let (half, measured) = effective_service(4000.0, std::time::Duration::from_secs(150));
        assert!(measured, "halfway still carries measurement weight");
        // 4000 + (750 - 4000) * 0.5 = 2375 exactly.
        assert_eq!(half, 2375);

        let (stale, measured) = effective_service(4000.0, std::time::Duration::from_secs(301));
        assert_eq!((stale, measured), (SERVICE_DEFAULT_MS, false));

        // A measurement BELOW the default blends upward the same way.
        let (rising, measured) = effective_service(100.0, std::time::Duration::from_secs(150));
        assert!(measured);
        assert_eq!(rising, 425); // 100 + (750 - 100) * 0.5
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
                last_service_at: None,
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
            &HashMap::new(),
            Instant::now(),
        );
        assert!(members[1].down, "remote member inside the down window");
        assert!(
            members[1]
                .retry_in_secs
                .is_some_and(|left| left > 0 && left <= 30),
            "breaker countdown mirrors the remaining down window"
        );
        assert_eq!(
            members[0].retry_in_secs, None,
            "healthy member has no countdown"
        );
    }

    /// Rolling steering-rate windows: expired events drop out on read,
    /// fresh ones count, and the cap bounds memory under sustained
    /// traffic.
    #[test]
    fn unit__window_count__prunes_expired_and_caps_live() {
        let now = Instant::now();
        let old_event = now
            .checked_sub(STEER_WINDOW + std::time::Duration::from_secs(100))
            .expect("window + 100s is always in the past on test hosts");
        let mut map = std::collections::HashMap::new();
        let mut q = std::collections::VecDeque::new();
        q.push_back(old_event);
        q.push_back(now);
        q.push_back(now);
        map.insert("svc".to_string(), q);
        assert_eq!(
            window_count(&mut map, "svc", now),
            2,
            "expired event dropped"
        );
        assert_eq!(
            map["svc"].front(),
            Some(&now),
            "prune happened in place, front is fresh"
        );
        assert_eq!(
            window_count(&mut map, "other", now),
            0,
            "unknown pool reads zero"
        );

        // Cap: sustained traffic sheds the OLDEST events, keeping the
        // most recent window slice.
        for _ in 0..(STEER_WINDOW_CAP + 30) {
            window_push(&mut map, "svc", now);
        }
        assert_eq!(window_count(&mut map, "svc", now), STEER_WINDOW_CAP as u64);
    }

    /// The sparkline history buckets events by age: newest in the last
    /// slot, boundary ages land in the right bucket, expired events
    /// never appear, and an unknown pool reads as silence.
    #[test]
    #[allow(clippy::cast_possible_truncation)] // const 15, always fits
    fn unit__window_history__buckets_by_age_and_drops_expired() {
        let now = Instant::now();
        let age = |secs: u64| {
            now.checked_sub(std::time::Duration::from_secs(secs))
                .expect("test ages are always in the past on test hosts")
        };
        let mut reqs = std::collections::HashMap::new();
        let mut hits = std::collections::HashMap::new();
        // Chronological order like the real window keeps (oldest first).
        reqs.insert(
            "svc".to_string(),
            [age(STEER_WINDOW.as_secs() + 50), age(25), age(5), age(5)]
                .into_iter()
                .collect(),
        );
        hits.insert(
            "svc".to_string(),
            [
                (age(STEER_WINDOW.as_secs() + 50), "m1".to_string()),
                (age(25), "far:m1".to_string()),
            ]
            .into_iter()
            .collect(),
        );
        let h = window_history(&mut reqs, &mut hits, "svc", now);
        assert_eq!(h.len(), STEER_HISTORY_BUCKETS as usize);
        let last = h.len() - 1;
        let prev = last - 1;
        assert_eq!(h[last], (2, 0), "newest bucket holds both age-5s requests");
        assert_eq!(
            h[prev],
            (1, 1),
            "age-25s lands one bucket back, hit and all"
        );
        assert!(
            h[..prev].iter().all(|&(r, x)| r == 0 && x == 0),
            "older buckets are silent"
        );
        // Unknown pool: fifteen silent buckets, no panic.
        let silent = window_history(&mut reqs, &mut hits, "nonesuch", now);
        assert!(silent.iter().all(|&(r, x)| r == 0 && x == 0));

        // Per-member split: the same events bucket per member, so the
        // pool trend is exactly the sum of its member trends.
        let m1 = member_hit_history(&mut hits, "svc", "m1", now);
        let far = member_hit_history(&mut hits, "svc", "far:m1", now);
        assert!(m1.iter().all(|&n| n == 0), "m1 steered nothing here");
        assert_eq!(far[prev], 1, "the age-25 hit belongs to far:m1 alone");
        let unknown = member_hit_history(&mut hits, "svc", "ghost", now);
        assert!(unknown.iter().all(|&n| n == 0));
    }

    /// `find()` is exact-match and prefix-safe.
    /// The affinity discount math: full while fresh, linear decay,
    /// zero at (and past) the TTL.
    #[test]
    fn unit__affinity_discount__full_fresh_linear_decay_zero_stale() {
        assert_eq!(affinity_discount(std::time::Duration::from_secs(0)), 750);
        assert_eq!(affinity_discount(std::time::Duration::from_secs(30)), 750);
        // 750 * (120-75)/(120-30) = 750*45/90 = 375.
        assert_eq!(affinity_discount(std::time::Duration::from_secs(75)), 375);
        assert_eq!(affinity_discount(std::time::Duration::from_secs(120)), 0);
        assert_eq!(affinity_discount(std::time::Duration::from_secs(600)), 0);
    }

    /// Two equally-measured members tie to declaration order; a
    /// request-scoped affinity discount for the second member flips
    /// the adaptive pick to it (its KV cache skips the re-prefill).
    #[test]
    fn unit__pick_adaptive__affinity_discount_flips_tie() {
        let members = vec![
            scored_service(0, "m1", 0, 100, 750, true),
            scored_service(1, "far:m2", 0, 100, 750, true),
        ];
        let p = pool(PoolStrategy::Adaptive);
        let tie = pick(&p, &members, None, None).unwrap();
        assert_eq!(tie.effective, "m1", "equal estimates tie to idx");
        let pinned = pick(&p, &members, None, Some(("far:m2", 750))).unwrap();
        assert_eq!(
            pinned.effective, "far:m2",
            "discounted affinity member wins"
        );
        // With a 200 ms wait gap a partial discount cannot bridge, the
        // cheaper member keeps the request; one that covers the gap
        // flips it.
        let gap = vec![
            scored_service(0, "m1", 0, 100, 750, true),
            scored_service(1, "far:m2", 0, 300, 750, true),
        ];
        let weak = pick(&p, &gap, None, Some(("far:m2", 49))).unwrap();
        assert_eq!(weak.effective, "m1", "49 ms discount < 200 ms gap");
        let strong = pick(&p, &gap, None, Some(("far:m2", 250))).unwrap();
        assert_eq!(strong.effective, "far:m2", "250 ms discount > 200 ms gap");
    }

    /// The affinity key derives from the conversation opener only:
    /// same opener across turns -> same key; different opener or
    /// empty/textless body -> None (no affinity signal).
    #[test]
    fn unit__affinity_key__opener_prefix_and_misses() {
        let base = serde_json::json!({"model": "svc", "messages": [
            {"role": "system", "content": "you are terse"},
            {"role": "user", "content": "summarize this"},
        ]});
        let k1 = affinity_key("svc", &base).expect("string opener keys");
        let k2 = affinity_key(
            "svc",
            &serde_json::json!({"messages": [
                {"role": "system", "content": "you are terse"},
                {"role": "user", "content": "unrelated follow-up"},
            ]}),
        )
        .expect("second turn keys");
        assert_eq!(k1, k2, "shared opener -> shared affinity key");
        assert!(k1.starts_with("svc:"));
        let parts = serde_json::json!({"messages": [
            {"role": "user", "content": [{"type": "text", "text": "describe"}]}
        ]});
        assert!(
            affinity_key("svc", &parts).is_some(),
            "multimodal text part keys"
        );
        assert_eq!(
            affinity_key(
                "svc",
                &serde_json::json!({"messages": [
            {"role": "user", "content": ""}]})
            ),
            None,
            "empty opener never keys"
        );
        assert_eq!(affinity_key("svc", &serde_json::json!({})), None);
    }

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
