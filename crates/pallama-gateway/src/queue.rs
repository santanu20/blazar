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
struct WaitKey(u8, u64); // (priority rank, seq) — max-heap pops high first

#[derive(Debug)]
struct Waiter {
    #[allow(dead_code)]
    model: String,
}

#[derive(Debug, Default)]
pub struct PriorityQueue {
    inner: Mutex<QueueInner>,
    notify: tokio::sync::Notify,
}

#[derive(Debug, Default)]
struct QueueInner {
    seq: u64,
    // BTreeMap<WaitKey(_)> iterates rev() = highest priority, earliest.
    waiters: BTreeMap<WaitKey, oneshot::Sender<()>>,
    names: BTreeMap<WaitKey, String>,
}

impl PriorityQueue {
    #[must_use] 
    pub fn new() -> Self {
        Self::default()
    }

    /// Wait for admission at `priority`, at most `timeout`. Returns Err on
    /// timeout (maps to 503) or when the queue shuts down.
    pub async fn wait(&self, model: &str, priority: Priority, timeout: Duration) -> Result<(), String> {
        let (tx, rx) = oneshot::channel();
        {
            let mut q = self.inner.lock().expect("queue lock");
            q.seq += 1;
            let rank = match priority {
                Priority::High => 2,
                Priority::Normal => 1,
                Priority::Low => 0,
            };
            let key = WaitKey(rank, q.seq);
            q.waiters.insert(key, tx);
            q.names.insert(key, Waiter { model: model.to_string() }.model.clone());
        }
        self.notify.notify_waiters();
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(_)) => Err("queue shut down".into()),
            Err(_) => {
                Err(format!("waited longer than {}s for a slot (all_slots_busy)", timeout.as_secs()))
            }
        }
    }

    /// A slot freed: admit the single highest-priority waiter, if any.
    pub fn signal_free(&self) {
        let next = {
            let mut q = self.inner.lock().expect("queue lock");
            loop {
                let Some(key) = q.waiters.keys().next_back().copied() else {
                    break None;
                };
                let Some(tx) = q.waiters.remove(&key) else {
                    continue;
                };
                q.names.remove(&key);
                if tx.is_closed() {
                    continue; // timed out while queued
                }
                break Some(tx);
            }
        };
        if let Some(tx) = next {
            let _ = tx.send(());
        }
        // One free slot admits exactly one waiter.
    }

    pub fn depth(&self) -> usize {
        self.inner.lock().expect("queue lock").waiters.len()
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn unit__queue__high_priority_admitted_first() {
        let q = std::sync::Arc::new(PriorityQueue::new());
        let a = tokio::spawn({
            let q = q.clone();
            async move { q.wait("m", Priority::Low, Duration::from_secs(5)).await }
        });
        let b = tokio::spawn({
            let q = q.clone();
            async move { q.wait("m", Priority::High, Duration::from_secs(5)).await }
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
    async fn unit__queue__timeout_maps_to_error() {
        let q = PriorityQueue::new();
        let start = std::time::Instant::now();
        let r = q.wait("m", Priority::Normal, Duration::from_millis(200)).await;
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("all_slots_busy"));
        assert!(start.elapsed() >= Duration::from_millis(200));
    }
}
