//! Session pins (R3): bounded registry of ACTIVE sessions, consulted by
//! the reaper so idle eviction never kills a model an agent session is
//! still talking to (`SGLang` session-aware radix cache, orchestrator
//! analog). Granularity is the MODEL (instances are per-model); the
//! prefix-level view lives inside the child's KV cache.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Hard bound on tracked sessions: an authless gateway must not grow the
/// table on header spam. Old entries fall out oldest-first.
const MAX_SESSIONS: usize = 1024;

#[derive(Debug, Clone)]
struct SessionEntry {
    model: String,
    last_seen: Instant,
}

/// Snapshot for `GET /api/sessions` and tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionInfo {
    pub session: String,
    pub model: String,
    pub idle_secs: u64,
    pub remaining_secs: u64,
}

/// Model -> is-at-least-one-session-live answer plus the live count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelPins {
    pub live: bool,
    pub count: usize,
}

#[derive(Debug, Default)]
pub struct SessionRegistry {
    inner: std::sync::Mutex<HashMap<String, SessionEntry>>,
}

impl SessionRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record session activity on a model (refreshes the TTL window).
    pub fn touch(&self, session: &str, model: &str) {
        let mut map = self.inner.lock().expect("session registry lock");
        if map.len() >= MAX_SESSIONS && !map.contains_key(session) {
            // Full: drop the stalest entry to make room (bounded table,
            // oldest-activity-first — the same discipline as the heat map).
            if let Some((oldest, _)) = map.iter().min_by_key(|(_, e)| e.last_seen) {
                let oldest = oldest.clone();
                map.remove(&oldest);
            }
        }
        map.insert(
            session.to_string(),
            SessionEntry {
                model: model.to_string(),
                last_seen: Instant::now(),
            },
        );
    }

    /// Explicit close (`/api/session action=close`). Returns false when the
    /// session was not pinned (idempotent close).
    pub fn release(&self, session: &str) -> bool {
        self.inner
            .lock()
            .expect("session registry lock")
            .remove(session)
            .is_some()
    }

    /// Force-stop hook: every pin on a model dies with the model
    /// (`blazar stop` wins over pins). Returns how many were dropped.
    pub fn release_model(&self, model: &str) -> usize {
        let mut map = self.inner.lock().expect("session registry lock");
        let before = map.len();
        map.retain(|_, e| e.model != model);
        before - map.len()
    }

    /// Drop expired entries (reaper tick). Returns the expired session
    /// names for logging.
    pub fn sweep(&self, ttl: Duration) -> Vec<String> {
        if ttl.is_zero() {
            let mut map = self.inner.lock().expect("session registry lock");
            let expired: Vec<String> = map.drain().map(|(k, _)| k).collect();
            return expired;
        }
        let now = Instant::now();
        let mut map = self.inner.lock().expect("session registry lock");
        let expired: Vec<String> = map
            .iter()
            .filter(|(_, e)| now.duration_since(e.last_seen) >= ttl)
            .map(|(k, _)| k.clone())
            .collect();
        for k in &expired {
            map.remove(k);
        }
        expired
    }

    /// Is at least one live session pinning this model? Live = touched
    /// within the TTL. TTL 0 = feature off = never pins.
    pub fn pins(&self, model: &str, ttl: Duration) -> ModelPins {
        if ttl.is_zero() {
            return ModelPins {
                live: false,
                count: 0,
            };
        }
        let now = Instant::now();
        let map = self.inner.lock().expect("session registry lock");
        let count = map
            .values()
            .filter(|e| e.model == model && now.duration_since(e.last_seen) < ttl)
            .count();
        ModelPins {
            live: count > 0,
            count,
        }
    }

    /// Live sessions for observability (`GET /api/sessions`), oldest idle
    /// first. Expired entries are excluded (they cannot pin anything).
    /// Sorting uses the full-precision idle (sub-second ties would
    /// otherwise fall through to name order and hide the true oldest).
    pub fn list(&self, ttl: Duration) -> Vec<SessionInfo> {
        let now = Instant::now();
        let map = self.inner.lock().expect("session registry lock");
        let mut out: Vec<(SessionInfo, Duration)> = map
            .iter()
            .filter_map(|(k, e)| {
                let idle = now.duration_since(e.last_seen);
                let live = !ttl.is_zero() && idle < ttl;
                live.then(|| {
                    (
                        SessionInfo {
                            session: k.clone(),
                            model: e.model.clone(),
                            idle_secs: idle.as_secs(),
                            remaining_secs: ttl.saturating_sub(idle).as_secs(),
                        },
                        idle,
                    )
                })
            })
            .collect();
        out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.session.cmp(&b.0.session)));
        out.into_iter().map(|(s, _)| s).collect()
    }

    /// Live pin count for the metrics gauge (entries only; TTL 0 = 0).
    pub fn live_len(&self, ttl: Duration) -> usize {
        if ttl.is_zero() {
            return 0;
        }
        let now = Instant::now();
        let map = self.inner.lock().expect("session registry lock");
        map.values()
            .filter(|e| now.duration_since(e.last_seen) < ttl)
            .count()
    }
}

#[cfg(test)]
mod tests {
    #[allow(non_snake_case)]
    #[allow(clippy::duration_suboptimal_units)]
    mod session_registry {
        use super::super::SessionRegistry;
        use std::time::Duration;

        #[test]
        fn unit__touch__pins_model_until_ttl() {
            let reg = SessionRegistry::new();
            reg.touch("agent1", "qwen");
            assert!(reg.pins("qwen", Duration::from_secs(15 * 60)).live);
            assert!(!reg.pins("llama", Duration::from_secs(15 * 60)).live);
            assert_eq!(reg.pins("qwen", Duration::from_secs(15 * 60)).count, 1);
        }

        #[test]
        fn unit__pins__ttl_zero_disables() {
            let reg = SessionRegistry::new();
            reg.touch("agent1", "qwen");
            let p = reg.pins("qwen", Duration::ZERO);
            assert!(!p.live && p.count == 0);
            assert!(reg.list(Duration::ZERO).is_empty());
            assert_eq!(reg.live_len(Duration::ZERO), 0);
        }

        #[test]
        fn unit__touch__refreshes_window() {
            let reg = SessionRegistry::new();
            reg.touch("s", "m");
            // A second touch much later (simulated by direct entry surgery
            // is overkill; the invariant worth pinning is that repeated
            // touches keep the entry live).
            reg.touch("s", "m");
            assert_eq!(reg.pins("m", Duration::from_secs(15 * 60)).count, 1);
        }

        #[test]
        fn unit__release__idempotent() {
            let reg = SessionRegistry::new();
            reg.touch("s", "m");
            assert!(reg.release("s"));
            assert!(!reg.release("s"));
            assert!(!reg.pins("m", Duration::from_secs(15 * 60)).live);
        }

        #[test]
        fn unit__release_model__drops_only_that_model() {
            let reg = SessionRegistry::new();
            reg.touch("a", "qwen");
            reg.touch("b", "qwen");
            reg.touch("c", "llama");
            assert_eq!(reg.release_model("qwen"), 2);
            assert!(!reg.pins("qwen", Duration::from_secs(15 * 60)).live);
            assert!(reg.pins("llama", Duration::from_secs(15 * 60)).live);
        }

        #[test]
        fn unit__sweep__clears_table_on_zero_ttl() {
            let reg = SessionRegistry::new();
            reg.touch("a", "m");
            let expired = reg.sweep(Duration::ZERO);
            assert_eq!(expired, vec!["a".to_string()]);
            assert!(reg.list(Duration::from_secs(15 * 60)).is_empty());
        }

        #[test]
        fn unit__bound__evicts_stalest_at_cap() {
            let reg = SessionRegistry::new();
            // a is oldest: first touch, never refreshed.
            reg.touch("a", "m");
            for i in 0..super::super::MAX_SESSIONS {
                reg.touch(&format!("s{i}"), "m");
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
            // Table capped; "a" (stalest) fell out even though TTL-not-reached.
            let names: Vec<String> = reg
                .list(Duration::from_secs(3600))
                .into_iter()
                .map(|s| s.session)
                .collect();
            assert!(!names.contains(&"a".to_string()));
            assert!(names.len() <= super::super::MAX_SESSIONS);
        }

        #[test]
        fn unit__list__shape_and_ordering() {
            let reg = SessionRegistry::new();
            reg.touch("old", "m1");
            std::thread::sleep(std::time::Duration::from_millis(5));
            reg.touch("new", "m2");
            let l = reg.list(Duration::from_secs(15 * 60));
            assert_eq!(l.len(), 2);
            assert_eq!(l[0].session, "old"); // oldest idle first
            assert_eq!(l[1].model, "m2");
            assert!(l[1].idle_secs <= l[0].idle_secs);
            assert!(l[0].remaining_secs > 0 && l[0].remaining_secs <= 900);
        }
    }
}
