//! Priority admission queue: when capacity is exhausted, waiters are
//! ordered by (priority desc, arrival); queued (never running) low-pri
//! waiters yield to high-pri. Bounded wait -> 503 `all_slots_busy`.

use std::collections::BTreeMap;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct WaitKey(u8, u128, u64);
// (inverted priority rank, deadline nanos, arrival seq): BTreeMap's
// `next()` pops the SMALLEST key = highest priority, then EARLIEST
// deadline (EDF within a class — SLO tiers), then earliest arrival (FIFO).
// No deadline (bulk) sorts last via u128::MAX.

#[derive(Debug)]
struct Waiter {
    #[allow(dead_code)]
    model: String,
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
    names: BTreeMap<WaitKey, String>,
}

/// SLO class defaults (seconds until admission). Explicit
/// `x-pallama-deadline-ms` overrides; prefill-heavy bodies (large
/// prompts) drop one class so interactive shorts jump ahead of a
/// multi-thousand-token prefill hogging the next slot.
const SLO_HIGH_SECS: u64 = 2;
const SLO_NORMAL_SECS: u64 = 30;
const SLO_LOW_SECS: u64 = 120;
/// >= this many request bytes (~16k tokens at 4B/token) counts as a
/// > prefill-heavy request for class demotion.
pub const PREFILL_HEAVY_BYTES: usize = 64 * 1024;

impl PriorityQueue {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Wait for admission at `priority`, at most `timeout`. Returns Err on
    /// timeout (maps to 503) or when the queue shuts down. `deadline_ms`
    /// (from `x-pallama-deadline-ms`) or class defaults order waiters
    /// EDF-within-priority; `body_len` demotes prefill-heavy requests one
    /// class so short interactive requests are not stuck behind a giant
    /// prefill at the same priority.
    pub async fn wait(
        &self,
        model: &str,
        priority: Priority,
        deadline_ms: Option<u64>,
        body_len: usize,
        timeout: Duration,
    ) -> Result<(), String> {
        let (tx, rx) = oneshot::channel();
        {
            let mut q = self.inner.lock().expect("queue lock");
            q.seq += 1;
            // Inverted rank: smallest key pops first via `next()`.
            let (rank, class_secs) = match priority {
                Priority::High => (0u8, SLO_HIGH_SECS),
                Priority::Normal => (1u8, SLO_NORMAL_SECS),
                Priority::Low => (2u8, SLO_LOW_SECS),
            };
            // Prefill-heavy: demote the deadline class one tier (the
            // request KEEPS its priority for ordering vs other classes —
            // only its within-class urgency relaxes).
            let class_secs = if body_len >= PREFILL_HEAVY_BYTES {
                // Demote one tier; Low cannot go lower.
                match priority {
                    Priority::High => SLO_NORMAL_SECS,
                    Priority::Normal | Priority::Low => SLO_LOW_SECS,
                }
            } else {
                class_secs
            };
            let now = std::time::Instant::now();
            let deadline = deadline_ms.map_or(now + Duration::from_secs(class_secs), |ms| {
                now + Duration::from_millis(ms.max(1))
            });
            // Deadline as nanos-from-epoch: later deadline = larger key =
            // admitted later among same-priority waiters (EDF).
            let deadline_key = deadline.saturating_duration_since(self.epoch).as_nanos();
            let key = WaitKey(rank, deadline_key, q.seq);
            q.waiters.insert(key, tx);
            q.names.insert(
                key,
                Waiter {
                    model: model.to_string(),
                }
                .model
                .clone(),
            );
        }
        self.notify.notify_waiters();
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(_)) => Err("queue shut down".into()),
            Err(_) => Err(format!(
                "waited longer than {}s for a slot (all_slots_busy)",
                timeout.as_secs()
            )),
        }
    }

    /// A slot freed: admit the single highest-priority waiter, if any.
    pub fn signal_free(&self) {
        let next = {
            let mut q = self.inner.lock().expect("queue lock");
            loop {
                let Some(key) = q.waiters.keys().next().copied() else {
                    break None;
                };
                let Some(tx) = q.waiters.remove(&key) else {
                    continue;
                };
                q.names.remove(&key);
                if tx.is_closed() {
                    continue; // timed out while queued
                }
                break Some((key, tx));
            }
        };
        if let Some((key, tx)) = next {
            // SLO burn: admitted, but past the waiter's deadline.
            let now_ns: u128 = std::time::Instant::now()
                .saturating_duration_since(self.epoch)
                .as_nanos();
            if now_ns > key.1 {
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
            let key = WaitKey(0, 0u128, inner.seq);
            let (tx, rx) = oneshot::channel();
            inner.waiters.insert(key, tx);
            inner.names.insert(key, "m".to_string());
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
            let key = WaitKey(0, key_deadline_far_future, inner.seq);
            let (tx, _rx) = oneshot::channel();
            inner.waiters.insert(key, tx);
            inner.names.insert(key, "m".to_string());
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
                q.wait("m", Priority::Low, None, 0, Duration::from_secs(5))
                    .await
            }
        });
        let b = tokio::spawn({
            let q = q.clone();
            async move {
                q.wait("m", Priority::High, None, 0, Duration::from_secs(5))
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
                    None,
                    PREFILL_HEAVY_BYTES + 1,
                    Duration::from_secs(5),
                )
                .await
            }
        });
        let interactive = tokio::spawn({
            let q = q.clone();
            async move {
                q.wait("m", Priority::Normal, None, 64, Duration::from_secs(5))
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
                q.wait("m", Priority::Normal, Some(50), 0, Duration::from_secs(5))
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
            .wait("m", Priority::Normal, None, 0, Duration::from_millis(200))
            .await;
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("all_slots_busy"));
        assert!(start.elapsed() >= Duration::from_millis(200));
    }
}
