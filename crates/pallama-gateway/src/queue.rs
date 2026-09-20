//! Priority admission queue: when capacity is exhausted, waiters are
//! ordered by (priority desc, deadline, arrival); queued (never running)
//! low-pri waiters yield to high-pri. Within one (priority, SLO-tier)
//! group, weighted fair queuing interleaves API keys by `weight` — a
//! weight-3 key earns ~3x the slots of a weight-1 key under contention.
//! Explicit `x-pallama-deadline-ms` waiters keep strict EDF (never
//! WFQ-delayed past their deadline by a later-deadline peer). Bounded
//! wait -> 503 `all_slots_busy`.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use tokio::sync::oneshot;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low,
    Normal,
    High,
}

impl Priority {
    pub fn from_header(value: Option<&str>) -> Self {
        match value.map(str::to_ascii_lowercase).as_deref() {
            Some("high") => Self::High,
            Some("low") => Self::Low,
            _ => Self::Normal,
        }
    }
}

/// Semantic workload class riding ALONGSIDE priority: it selects the
/// admission deadline tier and the reserved-capacity ceiling, never the
/// EDF/WFQ mechanics below. `Tool` = tool-bearing chat (latency-critical:
/// the user is waiting on a call round-trip); `RawLong` = raw completion
/// lane (`ollama_compat` recipe) whose 2048-class generations hold a slot
/// ~26 s on a single-slot child; `Interactive` = everything else.
/// Background is deliberately absent: the gateway has no background
/// traffic lane today (H24 — no dead values).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkClass {
    Interactive,
    Tool,
    RawLong,
}

/// Admission ceiling for a `RawLong` request on a lane with
/// `max_inflight` slots: while any Tool/Interactive waiter is parked,
/// raw work cannot consume the reserved share, `max(1, ceil(N/4))`;
/// with no protected waiter it borrows the full lane. N = 1 degenerates
/// to no isolation (a single slot cannot separate classes).
#[must_use]
pub fn raw_ceiling(max_inflight: i64, has_non_raw_waiter: bool) -> i64 {
    if !has_non_raw_waiter {
        return max_inflight;
    }
    let reserved = i64::try_from(interactive_reserved(
        usize::try_from(max_inflight.max(0)).unwrap_or(usize::MAX),
    ))
    .unwrap_or(i64::MAX);
    max_inflight.saturating_sub(reserved).max(1)
}

impl WorkClass {
    /// Ordering rank inside a priority tier: tool work jumps ahead of raw
    /// long work at equal deadline (the MT4 shape: a 26 s raw generation
    /// must not park a tool round-trip behind it).
    #[must_use]
    fn rank(self) -> u8 {
        match self {
            Self::Tool => 0,
            Self::Interactive => 1,
            Self::RawLong => 2,
        }
    }
}

/// Pure classification from request shape. `has_tools` = a non-empty
/// `tools` array on the body; `raw_lane` = the `ollama_compat` recipe was
/// taken (raw /v1/completions generation). Raw lane wins over tools: a
/// raw completion is long by construction.
#[must_use]
pub fn classify_work(has_tools: bool, raw_lane: bool) -> WorkClass {
    if raw_lane {
        WorkClass::RawLong
    } else if has_tools {
        WorkClass::Tool
    } else {
        WorkClass::Interactive
    }
}

/// Reserved-capacity arithmetic: interactive/tool work may always take
/// the last `ceil(N/4)` slots; raw long work may not (unless it borrows
/// while no interactive waiter is queued). `max(1, ..)` keeps the
/// reservation meaningful on tiny shapes; on N = 1 the clamp makes the
/// reservation degenerate — single-slot children cannot isolate classes
/// (documented tradeoff, see docs/2.ARCHITECTURE.md admission section).
#[must_use]
pub fn interactive_reserved(slots: usize) -> usize {
    slots.div_ceil(4).max(1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct WaitKey(u8, u8, u128, u64);
// (inverted priority rank, deadline nanos, arrival seq): BTreeMap's
// `next()` pops the SMALLEST key = highest priority, then EARLIEST
// deadline (EDF within a class — SLO tiers), then earliest arrival (FIFO).
// No deadline (bulk) sorts last via u128::MAX.

#[derive(Debug)]
struct Waiter {
    #[allow(dead_code)]
    model: String,
    class: WorkClass,
    /// WFQ bucket: API key name. `None` (unauthenticated waiters) shares
    /// the anonymous `""` bucket with weight 1.
    wfq_name: Option<String>,
    weight: u32,
    /// SLO tier the deadline was derived from; `None` = explicit
    /// `x-pallama-deadline-ms` (strict EDF, never WFQ-delayed).
    tier_secs: Option<u64>,
}

#[derive(Debug)]
pub struct PriorityQueue {
    inner: Mutex<QueueInner>,
    notify: tokio::sync::Notify,
    /// Fixed instant deadline keys are measured from (stable ordering).
    epoch: std::time::Instant,
    /// Waiters admitted AFTER their SLO deadline expired (the "burn"
    /// signal — the request was served, but too late for its class).
    deadline_exceeded: std::sync::atomic::AtomicU64,
}

impl Default for PriorityQueue {
    fn default() -> Self {
        Self {
            inner: Mutex::new(QueueInner::default()),
            notify: tokio::sync::Notify::new(),
            epoch: std::time::Instant::now(),
            deadline_exceeded: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

#[derive(Debug, Default)]
struct QueueInner {
    seq: u64,
    waiters: BTreeMap<WaitKey, oneshot::Sender<()>>,
    meta: BTreeMap<WaitKey, Waiter>,
    /// Virtual-runtime credit per WFQ bucket: `credit[bucket] += 1/weight`
    /// on each admission; the candidate with the lowest credit goes next
    /// (the 1/weight increment makes the race weight-proportional).
    /// Reset wholesale when too many buckets accumulate.
    credits: HashMap<String, f64>,
}

/// SLO class defaults (seconds until admission). Explicit
/// `x-pallama-deadline-ms` overrides; prefill-heavy bodies (large
/// prompts) drop one class so interactive shorts jump ahead of a
/// multi-thousand-token prefill hogging the next slot.
/// Tool-class admission budget: a tool round-trip waits at most this
/// long before its SLO is burned (tighter than NORMAL because the caller
/// is mid-workflow; looser than HIGH which stays explicit-priority).
const SLO_TOOL_SECS: u64 = 10;
const SLO_HIGH_SECS: u64 = 2;
const SLO_NORMAL_SECS: u64 = 30;
const SLO_LOW_SECS: u64 = 120;
/// >= this many request bytes (~16k tokens at 4B/token) counts as a
/// > prefill-heavy request for class demotion.
pub const PREFILL_HEAVY_BYTES: usize = 64 * 1024;
/// WFQ credit buckets are stateless-vs-time; when this many distinct
/// key buckets accumulate, drop all credits and start the race even.
const WFQ_CREDIT_BUCKETS_MAX: usize = 256;

/// Exact wait-duration rendering for the 503 message: whole seconds as
/// `Ns`, anything fractional as millis (F60: a 1500 ms timeout used to
/// print "1s" and hide the sub-second budget).
fn fmt_wait(t: Duration) -> String {
    let ms = t.as_millis();
    if ms.is_multiple_of(1000) {
        format!("{}s", t.as_secs())
    } else {
        format!("{ms}ms")
    }
}

impl PriorityQueue {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // Seven parameters is the honest shape of one admission decision;
    // bundling them (AdmissionKey) is the queued roadmap refinement.
    #[allow(clippy::too_many_arguments)]
    /// Wait for admission at `priority`, at most `timeout`. Returns Err on
    /// timeout (maps to 503) or when the queue shuts down. `deadline_ms`
    /// (from `x-pallama-deadline-ms`) or class defaults order waiters
    /// EDF-within-priority; `body_len` demotes prefill-heavy requests one
    /// class so short interactive requests are not stuck behind a giant
    /// prefill at the same priority. `wfq` = (key name, weight) enables
    /// weighted fair queuing among same-tier waiters of the same priority.
    pub async fn wait(
        &self,
        model: &str,
        priority: Priority,
        class: WorkClass,
        deadline_ms: Option<u64>,
        body_len: usize,
        timeout: Duration,
        wfq: Option<(&str, u32)>,
    ) -> Result<(), String> {
        let (tx, rx) = oneshot::channel();
        let key = {
            let mut q = self.inner.lock().expect("queue lock");
            q.seq += 1;
            // Inverted rank: smallest key pops first via `next()`.
            let (rank, class_secs) = match priority {
                Priority::High => (0u8, SLO_HIGH_SECS),
                Priority::Normal => (1u8, SLO_NORMAL_SECS),
                Priority::Low => (2u8, SLO_LOW_SECS),
            };
            // Class-aware urgency (explicit deadline stays sovereign
            // below): tool work tightens to the tool budget, raw long
            // work relaxes to the LOW floor — its class is structurally
            // patient. Applies before prefill demotion.
            let class_secs = match class {
                WorkClass::Tool => class_secs.min(SLO_TOOL_SECS),
                WorkClass::Interactive => class_secs,
                WorkClass::RawLong => class_secs.max(SLO_LOW_SECS),
            };
            // Prefill-heavy: demote the deadline class one tier (the
            // request KEEPS its priority for ordering vs other classes —
            // only its within-class urgency relaxes).
            let tier_secs = if body_len >= PREFILL_HEAVY_BYTES {
                // Demote one tier; Low cannot go lower.
                match priority {
                    Priority::High => SLO_NORMAL_SECS,
                    Priority::Normal | Priority::Low => SLO_LOW_SECS,
                }
            } else {
                class_secs
            };
            let now = std::time::Instant::now();
            let (deadline, tier_secs) = if let Some(ms) = deadline_ms {
                (now + Duration::from_millis(ms.max(1)), None)
            } else {
                (now + Duration::from_secs(tier_secs), Some(tier_secs))
            };
            // Deadline as nanos-from-epoch: later deadline = larger key =
            // admitted later among same-priority waiters (EDF).
            let deadline_key = deadline.saturating_duration_since(self.epoch).as_nanos();
            let key = WaitKey(rank, class.rank(), deadline_key, q.seq);
            q.waiters.insert(key, tx);
            q.meta.insert(
                key,
                Waiter {
                    model: model.to_string(),
                    class,
                    wfq_name: wfq.map(|(name, _)| name.to_string()),
                    weight: wfq.map_or(1, |(_, w)| w.max(1)),
                    tier_secs,
                },
            );
            key
        };
        self.notify.notify_waiters();
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(_)) => Err("queue shut down".into()),
            Err(_) => {
                // F59: remove our own entry — on an idle system no future
                // `signal_free` arrives to lazily clean a timed-out
                // waiter, and `depth()` would report phantom saturation
                // (doctor row, /metrics, 503 body) forever.
                let mut q = self.inner.lock().expect("queue lock");
                q.waiters.remove(&key);
                q.meta.remove(&key);
                drop(q);
                Err(format!(
                    "waited longer than {} for a slot (all_slots_busy)",
                    fmt_wait(timeout)
                ))
            }
        }
    }

    /// A slot freed: admit the single best waiter, if any. Ordering:
    /// highest priority rank first; explicit-deadline waiters keep strict
    /// EDF; same-tier waiters interleave by WFQ credit/weight — but never
    /// past an explicit-deadline peer (the barrier rule).
    pub fn signal_free(&self) {
        let next = {
            let mut q = self.inner.lock().expect("queue lock");
            loop {
                let Some(head_key) = q.waiters.keys().next().copied() else {
                    break None;
                };
                let Some(head) = q.meta.get(&head_key) else {
                    // Orphaned channel (timed-out waiter mid-removal): drop.
                    q.waiters.remove(&head_key);
                    continue;
                };
                let pick = if head.tier_secs.is_none() {
                    // Explicit deadline: strict EDF, admit the head.
                    head_key
                } else {
                    // Barrier: earliest explicit deadline in this rank
                    // group. Tier waiters sorting at/after the barrier
                    // must not jump the explicit waiter via WFQ.
                    let same_rank = q.waiters.iter().take_while(|(k, _)| k.0 == head_key.0);
                    let barrier = same_rank
                        .filter(|(k, _)| q.meta.get(*k).is_some_and(|m| m.tier_secs.is_none()))
                        .map(|(k, _)| k.2)
                        .min()
                        .unwrap_or(u128::MAX);
                    // WFQ candidates: same rank + same tier + class-tier
                    // deadline (not explicit) + not past the barrier.
                    // Compared by accumulated credit: each admission adds
                    // 1/weight to its bucket, so weight already shapes
                    // the race (steady state = weight-proportional).
                    let mut best: Option<(f64, u64, WaitKey)> = None;
                    for (k, _) in q.waiters.iter().take_while(|(k, _)| k.0 == head_key.0) {
                        if k.2 > barrier {
                            break; // deadline-sorted: nothing legal beyond.
                        }
                        let Some(m) = q.meta.get(k) else {
                            continue;
                        };
                        if m.tier_secs != head.tier_secs {
                            continue; // other tiers keep pure EDF order.
                        }
                        let bucket = m.wfq_name.clone().unwrap_or_default();
                        let credit = q.credits.get(&bucket).copied().unwrap_or(0.0);
                        let better = best.is_none_or(|(br, bseq, _)| (credit, k.3) < (br, bseq));
                        if better {
                            best = Some((credit, k.3, *k));
                        }
                    }
                    match best {
                        Some((_, _, k)) => k,
                        None => head_key, // defensive: head always qualifies.
                    }
                };
                let Some(tx) = q.waiters.remove(&pick) else {
                    continue;
                };
                let admitted = q.meta.remove(&pick);
                if tx.is_closed() {
                    continue; // timed out while queued
                }
                // Charge the WFQ bucket of the admitted waiter.
                if let Some(m) = admitted {
                    if m.tier_secs.is_some() {
                        let bucket = m.wfq_name.unwrap_or_default();
                        let w = f64::from(m.weight.max(1));
                        let credit = q.credits.entry(bucket).or_insert(0.0);
                        *credit += 1.0 / w;
                        if q.credits.len() > WFQ_CREDIT_BUCKETS_MAX {
                            q.credits.clear();
                        }
                    }
                }
                break Some((pick, tx));
            }
        };
        if let Some((key, tx)) = next {
            // SLO burn: admitted, but past the waiter's deadline.
            let now_ns: u128 = std::time::Instant::now()
                .saturating_duration_since(self.epoch)
                .as_nanos();
            if now_ns > key.2 {
                self.deadline_exceeded
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            let _ = tx.send(());
        }
        // One free slot admits exactly one waiter.
    }

    /// Total waiters admitted after their SLO deadline. Monotonic counter for `/metrics` and the doctor SLO row.
    #[must_use]
    pub fn slo_deadline_exceeded(&self) -> u64 {
        self.deadline_exceeded
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Borrowing check for reserved capacity: is any Tool/Interactive
    /// waiter parked for this model right now? While one is, `RawLong`
    /// admission is capped below the reserved slots.
    pub fn has_non_raw_waiter(&self, model: &str) -> bool {
        let q = self.inner.lock().expect("queue lock");
        q.meta
            .values()
            .any(|w| w.model == model && w.class != WorkClass::RawLong)
    }

    pub fn depth(&self) -> usize {
        self.inner.lock().expect("queue lock").waiters.len()
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn unit__queue__slo_deadline_counter_counts_late_admission() {
        let q = PriorityQueue::default();
        assert_eq!(q.slo_deadline_exceeded(), 0);
        // A waiter whose deadline already passed (nanos-from-epoch 0 =
        // the past) must count as burn when admitted.
        let _rx = {
            let mut inner = q.inner.lock().expect("queue lock");
            inner.seq += 1;
            let key = WaitKey(0, 1, 0u128, inner.seq);
            let (tx, rx) = oneshot::channel();
            inner.waiters.insert(key, tx);
            inner.meta.insert(
                key,
                Waiter {
                    model: "m".to_string(),
                    class: WorkClass::Interactive,
                    wfq_name: None,
                    weight: 1,
                    tier_secs: None,
                },
            );
            rx
        };
        std::thread::sleep(Duration::from_millis(5));
        q.signal_free();
        assert_eq!(q.slo_deadline_exceeded(), 1, "late admission counted");
    }

    #[test]
    fn unit__queue__on_time_admission_does_not_burn() {
        let q = PriorityQueue::default();
        let key_deadline_far_future = u128::MAX - 1;
        {
            let mut inner = q.inner.lock().expect("queue lock");
            inner.seq += 1;
            let key = WaitKey(0, 1, key_deadline_far_future, inner.seq);
            let (tx, _rx) = oneshot::channel();
            inner.waiters.insert(key, tx);
            inner.meta.insert(
                key,
                Waiter {
                    model: "m".to_string(),
                    class: WorkClass::Interactive,
                    wfq_name: None,
                    weight: 1,
                    tier_secs: Some(SLO_NORMAL_SECS),
                },
            );
        }
        q.signal_free();
        assert_eq!(q.slo_deadline_exceeded(), 0, "on-time admission is free");
    }

    #[tokio::test]
    async fn unit__queue__high_priority_admitted_first() {
        let q = std::sync::Arc::new(PriorityQueue::new());
        let a = tokio::spawn({
            let q = q.clone();
            async move {
                q.wait(
                    "m",
                    Priority::Low,
                    WorkClass::Interactive,
                    None,
                    0,
                    Duration::from_secs(5),
                    None,
                )
                .await
            }
        });
        let b = tokio::spawn({
            let q = q.clone();
            async move {
                q.wait(
                    "m",
                    Priority::High,
                    WorkClass::Interactive,
                    None,
                    0,
                    Duration::from_secs(5),
                    None,
                )
                .await
            }
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(q.depth(), 2);

        q.signal_free();
        // High admitted first.
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(b.is_finished(), "high-priority waiter goes first");
        assert!(!a.is_finished(), "low-priority still waiting");
        q.signal_free();
        let (ra, rb) = tokio::join!(a, b);
        assert!(ra.unwrap().is_ok());
        assert!(rb.unwrap().is_ok());
    }

    #[tokio::test]
    async fn unit__queue__edf_same_priority_earlier_deadline_first() {
        // Two Normal waiters: the one with the tight deadline (or the
        // prefill-light one whose class budget is tighter) is admitted
        // first, regardless of arrival order.
        let q = std::sync::Arc::new(PriorityQueue::new());
        let bulk = tokio::spawn({
            let q = q.clone();
            // Prefill-heavy body demotes its class deadline (120s tier).
            async move {
                q.wait(
                    "m",
                    Priority::Normal,
                    WorkClass::Interactive,
                    None,
                    PREFILL_HEAVY_BYTES + 1,
                    Duration::from_secs(5),
                    None,
                )
                .await
            }
        });
        let interactive = tokio::spawn({
            let q = q.clone();
            async move {
                q.wait(
                    "m",
                    Priority::Normal,
                    WorkClass::Interactive,
                    None,
                    64,
                    Duration::from_secs(5),
                    None,
                )
                .await
            }
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(q.depth(), 2);
        q.signal_free();
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            interactive.is_finished(),
            "interactive (tight class) admitted first"
        );
        assert!(!bulk.is_finished(), "prefill-heavy still waiting");
        // Explicit deadline beats class defaults.
        let tight = tokio::spawn({
            let q = q.clone();
            async move {
                q.wait(
                    "m",
                    Priority::Normal,
                    WorkClass::Interactive,
                    Some(50),
                    0,
                    Duration::from_secs(5),
                    None,
                )
                .await
            }
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        q.signal_free();
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(tight.is_finished(), "explicit tight deadline wins");
        let _ = tokio::join!(bulk, tight);
    }

    #[tokio::test]
    async fn unit__queue__timeout_maps_to_error() {
        let q = PriorityQueue::new();
        let start = std::time::Instant::now();
        let r = q
            .wait(
                "m",
                Priority::Normal,
                WorkClass::Interactive,
                None,
                0,
                Duration::from_millis(200),
                None,
            )
            .await;
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("all_slots_busy"));
        assert!(start.elapsed() >= Duration::from_millis(200));
    }

    #[tokio::test]
    async fn unit__queue__wfq_interleaves_buckets_by_weight() {
        // Weight-3 bucket earns ~3x the admissions of weight-1 under
        // contention: over 8 admissions expect 6 vs 2 (ties fall to the
        // earlier arrival, spawned first here).
        let q = std::sync::Arc::new(PriorityQueue::new());
        let mut handles = Vec::new();
        let mut tags = Vec::new();
        for _ in 0..8 {
            let q2 = q.clone();
            tags.push("a");
            handles.push(tokio::spawn(async move {
                let _ = q2
                    .wait(
                        "m",
                        Priority::Normal,
                        WorkClass::Interactive,
                        None,
                        0,
                        Duration::from_secs(10),
                        Some(("a", 3)),
                    )
                    .await;
            }));
        }
        for _ in 0..8 {
            let q2 = q.clone();
            tags.push("b");
            handles.push(tokio::spawn(async move {
                let _ = q2
                    .wait(
                        "m",
                        Priority::Normal,
                        WorkClass::Interactive,
                        None,
                        0,
                        Duration::from_secs(10),
                        Some(("b", 1)),
                    )
                    .await;
            }));
        }
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert_eq!(q.depth(), 16, "all waiters queued");

        let mut order = Vec::new();
        let mut seen = Vec::new();
        for _ in 0..8 {
            q.signal_free();
            tokio::time::sleep(Duration::from_millis(40)).await;
            for (i, h) in handles.iter().enumerate() {
                if h.is_finished() && !seen.contains(&i) {
                    seen.push(i);
                    order.push(tags[i]);
                    break;
                }
            }
        }
        // Drain the rest so all tasks complete.
        for _ in 0..8 {
            q.signal_free();
        }
        for h in handles {
            let _ = h.await;
        }
        let a_count = order.iter().filter(|t| **t == "a").count();
        let b_count = order.iter().filter(|t| **t == "b").count();
        assert_eq!(
            (a_count, b_count),
            (6, 2),
            "weight 3:1 interleave, got {order:?}"
        );
    }

    #[tokio::test]
    async fn unit__queue__wfq_explicit_deadline_barrier_holds() {
        // A later-arriving same-tier waiter must never jump an explicit
        // `x-pallama-deadline-ms` waiter via a fresh WFQ bucket, even
        // when the head bucket carries credit.
        let q = std::sync::Arc::new(PriorityQueue::new());
        // Sacrificial waiter from bucket "h" charges it: credit 1.0.
        let sacrifice = tokio::spawn({
            let q = q.clone();
            async move {
                let _ = q
                    .wait(
                        "m",
                        Priority::Normal,
                        WorkClass::Interactive,
                        Some(60_000),
                        0,
                        Duration::from_secs(10),
                        Some(("h", 1)),
                    )
                    .await;
            }
        });
        tokio::time::sleep(Duration::from_millis(60)).await;
        q.signal_free();
        let _ = sacrifice.await;

        // Head: bucket "h" (tier 30s). Explicit waiter E at ~31s. Bucket
        // "b" waiter arrives later so its tier deadline (arrival+30s)
        // lands just past E's explicit deadline -> excluded by barrier.
        let head = tokio::spawn({
            let q = q.clone();
            async move {
                let _ = q
                    .wait(
                        "m",
                        Priority::Normal,
                        WorkClass::Interactive,
                        None,
                        0,
                        Duration::from_secs(10),
                        Some(("h", 1)),
                    )
                    .await;
            }
        });
        tokio::time::sleep(Duration::from_millis(10)).await;
        let explicit = tokio::spawn({
            let q = q.clone();
            async move {
                let _ = q
                    .wait(
                        "m",
                        Priority::Normal,
                        WorkClass::Interactive,
                        Some(31_000),
                        0,
                        Duration::from_secs(10),
                        None,
                    )
                    .await;
            }
        });
        tokio::time::sleep(Duration::from_millis(1_100)).await;
        let late = tokio::spawn({
            let q = q.clone();
            async move {
                let _ = q
                    .wait(
                        "m",
                        Priority::Normal,
                        WorkClass::Interactive,
                        None,
                        0,
                        Duration::from_secs(10),
                        Some(("b", 1)),
                    )
                    .await;
            }
        });
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert_eq!(q.depth(), 3);

        let mut order = Vec::new();
        let checks = [("head", &head), ("explicit", &explicit), ("late", &late)];
        for _ in &checks {
            q.signal_free();
            tokio::time::sleep(Duration::from_millis(40)).await;
            for (t, h) in &checks {
                if h.is_finished() && !order.contains(t) {
                    order.push(*t);
                    break;
                }
            }
        }
        let _ = tokio::join!(head, explicit, late);
        assert_eq!(
            order,
            vec!["head", "explicit", "late"],
            "explicit deadline is a barrier: fresh bucket must not jump it"
        );
    }

    #[tokio::test]
    async fn unit__queue__uniform_anonymous_waiters_stay_fifo() {
        // No WFQ identity + equal everything: strict FIFO by arrival.
        let q = std::sync::Arc::new(PriorityQueue::new());
        let mut handles = Vec::new();
        for i in 0..4 {
            let q2 = q.clone();
            handles.push(tokio::spawn(async move {
                let _ = q2
                    .wait(
                        &format!("m{i}"),
                        Priority::Normal,
                        WorkClass::Interactive,
                        None,
                        0,
                        Duration::from_secs(10),
                        None,
                    )
                    .await;
            }));
        }
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert_eq!(q.depth(), 4);
        let mut finished: Vec<usize> = Vec::new();
        for _ in 0..4 {
            q.signal_free();
            tokio::time::sleep(Duration::from_millis(40)).await;
            for (i, h) in handles.iter().enumerate() {
                if h.is_finished() && !finished.contains(&i) {
                    finished.push(i);
                    break;
                }
            }
        }
        assert_eq!(finished, vec![0, 1, 2, 3], "uniform waiters stay FIFO");
        for h in handles {
            let _ = h.await;
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__interactive_reserved__arithmetic_table() {
        // max(1, ceil(N/4)): the reservation floor keeps one protected
        // slot even where quartering rounds to zero.
        assert_eq!(interactive_reserved(1), 1);
        assert_eq!(interactive_reserved(2), 1);
        assert_eq!(interactive_reserved(4), 1);
        assert_eq!(interactive_reserved(5), 2);
        assert_eq!(interactive_reserved(8), 2);
        assert_eq!(interactive_reserved(16), 4);
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__workclass__tool_outranks_raw_long_at_equal_deadline() {
        // Class rank sits before the deadline key: a Tool waiter jumps
        // an equal-deadline RawLong waiter in the admission order.
        let tool = WaitKey(0, WorkClass::Tool.rank(), 100, 1);
        let raw = WaitKey(0, WorkClass::RawLong.rank(), 100, 2);
        assert!(tool < raw);
        // Interactive sits between: tools first, raw last.
        let inter = WaitKey(0, WorkClass::Interactive.rank(), 100, 3);
        assert!(tool < inter && inter < raw);
        // Explicit deadlines still dominate class at the tier boundary:
        // an earlier RawLong deadline beats a later Tool one only when
        // priorities differ — same priority, earlier deadline, raw still
        // second (class first). Documented tradeoff, pinned.
        let raw_early = WaitKey(0, WorkClass::RawLong.rank(), 50, 4);
        assert!(tool < raw_early);
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__raw_ceiling__reservation_and_borrow_table() {
        // Borrow when idle: raw sees the full lane.
        assert_eq!(raw_ceiling(4, false), 4);
        assert_eq!(raw_ceiling(1, false), 1);
        // Protected waiter parked: raw cannot consume the reserved
        // quarter (min 1 raw slot stays admissible — starvation-free).
        assert_eq!(raw_ceiling(4, true), 3);
        assert_eq!(raw_ceiling(8, true), 6);
        assert_eq!(raw_ceiling(2, true), 1);
        // N = 1 degenerates to no isolation (documented tradeoff).
        assert_eq!(raw_ceiling(1, true), 1);
    }
}
