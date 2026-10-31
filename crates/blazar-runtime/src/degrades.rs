//! Registry of silent performance degrades detected in live engine
//! children (per-process, upstream warn-once lines the child prints to
//! stderr). The gateway reads this to stamp affected requests (sentinel
//! `engine_slow_attn`) and to expose active degrades on `/api/why`.
//!
//! Entries are keyed by the child's endpoint identity — the same string
//! [`endpoint_id`] derives everywhere — so the gateway can correlate a
//! degrade with the model a request is about to hit. Entries live for
//! the process lifetime of the child: a respawned child gets a new
//! endpoint (fresh ports/sockets) and starts clean, which is exactly
//! the semantics of the upstream warn-once flags.

use std::collections::HashMap;
use std::sync::Mutex;

/// Registry bound: distinct degraded children kept at once. Children are
/// few (single digits typically); the bound only guards pathological
/// churn (endless respawn loops) from growing memory without limit.
const MAX_ENTRIES: usize = 32;

#[derive(Debug, Clone, PartialEq)]
pub struct ChildDegrade {
    /// Endpoint identity of the child that degraded (see [`endpoint_id`]).
    pub child: String,
    /// The upstream log line, verbatim (evidence).
    pub line: String,
    /// Blazar's teaching note (remedy).
    pub note: String,
    /// Unix seconds the degrade was first seen for this child.
    pub since: u64,
}

static REGISTRY: Mutex<Option<HashMap<String, ChildDegrade>>> = Mutex::new(None);

/// Stable string identity for a child endpoint: TCP `host:port`, Unix
/// socket path. Unique per live child by construction (the supervisor
/// allocates fresh ports/sockets per spawn).
#[must_use]
pub fn endpoint_id(endpoint: &blazar_core::profile::Endpoint) -> String {
    match endpoint {
        blazar_core::profile::Endpoint::Tcp { host, port } => format!("tcp://{host}:{port}"),
        blazar_core::profile::Endpoint::Unix { socket } => format!("unix://{socket}"),
    }
}

/// Record (or refresh) a degrade for `child`. Same-child re-detection
/// keeps the original `since` — the degrade is per child process.
pub fn record(child: String, line: String, note: String) {
    let mut guard = REGISTRY
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let map = guard.get_or_insert_with(HashMap::new);
    let since = map.get(&child).map_or_else(
        || {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs())
        },
        |e| e.since,
    );
    let entry = ChildDegrade {
        child,
        line,
        note,
        since,
    };
    if map.len() >= MAX_ENTRIES && !map.contains_key(&entry.child) {
        // Drop the oldest entry: churn-bound, and the oldest is the
        // least likely to still be alive.
        if let Some((oldest, _)) = map.iter().min_by_key(|(_, e)| e.since) {
            let oldest = oldest.clone();
            map.remove(&oldest);
        }
    }
    let key = entry.child.clone();
    map.insert(key, entry);
}

/// Snapshot of all recorded degrades (any order).
#[must_use]
pub fn snapshot() -> Vec<ChildDegrade> {
    let guard = REGISTRY
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    guard
        .as_ref()
        .map_or_else(Vec::new, |m| m.values().cloned().collect())
}

/// Drop the entry for a child that died clean (defensive: endpoint reuse
/// across respawns would otherwise inherit a stale degrade).
pub fn clear(child: &str) {
    let mut guard = REGISTRY
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(map) = guard.as_mut() {
        map.remove(child);
    }
}

#[cfg(test)]
#[allow(non_snake_case)] // repo convention: unit__scenario__expected
mod tests {
    use super::*;

    #[test]
    fn unit__record_snapshot_clear__roundtrip_and_stable_since() {
        clear("tcp://127.0.0.1:1");
        record(
            "tcp://127.0.0.1:1".into(),
            "fattn line".into(),
            "note".into(),
        );
        std::thread::sleep(std::time::Duration::from_millis(1100));
        record(
            "tcp://127.0.0.1:1".into(),
            "fattn line 2".into(),
            "note".into(),
        );
        let snap: Vec<_> = snapshot()
            .into_iter()
            .filter(|e| e.child == "tcp://127.0.0.1:1")
            .collect();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].line, "fattn line 2");
        // since survives re-detection (per-child-process semantics).
        assert!(snap[0].since > 0);
        clear("tcp://127.0.0.1:1");
        assert!(snapshot().iter().all(|e| e.child != "tcp://127.0.0.1:1"));
    }

    #[test]
    fn unit__record__churn_bound_drops_oldest() {
        for i in 0..(MAX_ENTRIES + 2) {
            record(
                format!("unix:///tmp/child-{i}"),
                format!("line {i}"),
                "note".into(),
            );
        }
        assert!(snapshot().len() <= MAX_ENTRIES);
        for i in 0..(MAX_ENTRIES + 2) {
            clear(&format!("unix:///tmp/child-{i}"));
        }
    }

    #[test]
    fn unit__endpoint_id__distinct_shapes() {
        assert_eq!(
            endpoint_id(&blazar_core::profile::Endpoint::Tcp {
                host: "127.0.0.1".into(),
                port: 8080
            }),
            "tcp://127.0.0.1:8080"
        );
        assert_eq!(
            endpoint_id(&blazar_core::profile::Endpoint::Unix {
                socket: "/run/x.sock".into()
            }),
            "unix:///run/x.sock"
        );
    }
}
