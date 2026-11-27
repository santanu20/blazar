//! Automatic fleet membership: the live worker registry.
//!
//! `blazar rpc worker --join` announces itself to the gateway, which
//! records it here. Spawn planning merges the announced endpoints into
//! the static `rpc_servers` list (`--rpc` argv), so a machine that
//! joins the fleet becomes usable capacity without any config edit or
//! daemon restart. Membership is ephemeral by design: a worker that
//! stops heartbeating is evicted after the TTL window, and the engine
//! preflight remains the loud gate at spawn - the registry only feeds
//! planning, it never silently resurrects a dead endpoint.

use std::time::{Duration, Instant};

use dashmap::DashMap;
use serde::Serialize;

use crate::events::EventBus;
use crate::rpc_fleet::RpcDeviceMemory;

/// Registry capacity. A join beyond this is refused loudly rather than
/// evicting someone else's worker - an unbounded map is a trivial
/// memory-exhaustion surface for anyone who can reach the join route.
pub const FLEET_MAX_WORKERS: usize = 64;

/// Endpoints per worker. One RPC server process listens on exactly one
/// port; a small list covers multi-NIC boxes, anything larger is a
/// misconfiguration, not a fleet.
pub const FLEET_MAX_ENDPOINTS: usize = 4;

/// Registry failures. Each maps to a distinct HTTP status at the
/// gateway; the messages are user-facing teaching, not debug noise.
#[derive(Debug, thiserror::Error)]
pub enum FleetError {
    #[error(
        "worker_id must be 1..=64 chars of [a-zA-Z0-9_-] (it names the worker in logs and /api/fleet/workers), got {0:?}"
    )]
    BadWorkerId(String),
    #[error(
        "endpoints must be host:port with a nonempty host (<= 253 chars) and a valid port, got {0:?}"
    )]
    BadEndpoint(String),
    #[error(
        "at most {FLEET_MAX_ENDPOINTS} endpoints per worker - one RPC server process is one endpoint"
    )]
    TooManyEndpoints,
    #[error(
        "fleet registry is full ({FLEET_MAX_WORKERS} workers) - remove a stale worker before joining another"
    )]
    RegistryFull,
    #[error("unknown worker_id {0:?} - join before heartbeating")]
    UnknownWorker(String),
}

/// A live fleet worker record. `devices` is informational (reported by
/// the worker's self-probe heartbeat): planning probes endpoints live
/// anyway, so a stale device list can never misroute a spawn.
#[derive(Debug, Clone)]
pub struct FleetWorker {
    pub worker_id: String,
    /// RPC endpoints in announce order; the first is the primary and
    /// rides the join/depart events.
    pub endpoints: Vec<String>,
    pub hostname: String,
    /// Last accepted device self-report (heartbeat payload).
    pub devices: Vec<RpcDeviceMemory>,
    /// Scoped session credential minted at join: the worker presents it
    /// on heartbeat/leave when it holds no master `fleet_token` (it
    /// joined via a one-use invite code). Revoked with the record -
    /// leave, TTL prune, or replacement re-join.
    pub worker_token: String,
    pub joined_at: Instant,
    pub last_seen: Instant,
}

impl FleetWorker {
    /// The endpoint used in events and status summaries.
    #[must_use]
    pub fn primary_endpoint(&self) -> &str {
        self.endpoints.first().map_or("", String::as_str)
    }
}

/// API view of a worker: wall-clock-free (ages relative to `now`) so
/// the list is stable under clock adjustments and trivially testable.
#[derive(Debug, Clone, Serialize)]
pub struct FleetWorkerView {
    pub worker_id: String,
    pub endpoints: Vec<String>,
    pub hostname: String,
    pub devices: Vec<RpcDeviceMemory>,
    pub joined_secs_ago: u64,
    pub last_seen_secs_ago: u64,
    /// 0 = already past the TTL (only visible mid-prune).
    pub expires_in_secs: u64,
}

/// The registry itself. Sharded like the rest of the supervisor's
/// maps; TTL pruning is lazy (on read) so a quiet fleet costs nothing.
#[derive(Debug, Default)]
pub struct FleetRegistry {
    workers: DashMap<String, FleetWorker>,
}

/// Validate a worker id: 1..=64 chars of `[A-Za-z0-9_-]`. Announced by
/// the worker (`--name`), echoed in logs/events - it must be greppable
/// and injection-free.
#[must_use]
pub fn valid_worker_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// Mint a 32-hex session credential from 16 OS-entropy bytes. Scoped to
/// one registry record; never derived from the master fleet token.
fn mint_worker_token() -> String {
    use std::fmt::Write as _;
    let mut raw = [0u8; 16];
    getrandom::fill(&mut raw).expect("OS entropy source");
    let mut out = String::with_capacity(32);
    for b in raw {
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// Validate one `host:port` endpoint (host nonempty, <= 253 chars,
/// port parses as u16). IPv6 literals in brackets (`[::1]:50052`)
/// parse naturally via rsplit.
#[must_use]
pub fn valid_endpoint(ep: &str) -> bool {
    let Some((host, port)) = ep.rsplit_once(':') else {
        return false;
    };
    !host.is_empty() && host.len() <= 253 && port.parse::<u16>().is_ok()
}

impl FleetRegistry {
    /// Record a join (or re-join). A fresh id under an EXISTING one
    /// replaces the record and emits a `replaced` departure for the old
    /// identity - the worker restarted with a new announce, the fleet
    /// keeps exactly one record per live worker. Returns the freshly
    /// minted worker token for the ack.
    pub fn join(
        &self,
        worker_id: &str,
        endpoints: Vec<String>,
        hostname: &str,
        devices: Vec<RpcDeviceMemory>,
        now: Instant,
        bus: &EventBus,
    ) -> Result<String, FleetError> {
        if !valid_worker_id(worker_id) {
            return Err(FleetError::BadWorkerId(worker_id.to_string()));
        }
        if endpoints.is_empty() || endpoints.len() > FLEET_MAX_ENDPOINTS {
            return Err(FleetError::TooManyEndpoints);
        }
        for ep in &endpoints {
            if !valid_endpoint(ep) {
                return Err(FleetError::BadEndpoint(ep.clone()));
            }
        }
        let worker_token = mint_worker_token();
        // Replace semantics first: an existing record never grows the
        // registry, so only genuinely-new ids check the cap.
        if let Some(entry) = self.workers.get(worker_id) {
            let old = entry.value().clone();
            drop(entry);
            let endpoint = endpoints.first().cloned().unwrap_or_default();
            self.workers.insert(
                worker_id.to_string(),
                FleetWorker {
                    worker_id: worker_id.to_string(),
                    endpoints,
                    hostname: hostname.to_string(),
                    devices,
                    worker_token: worker_token.clone(),
                    joined_at: now,
                    last_seen: now,
                },
            );
            let _ = bus.publish(crate::events::BlazarEvent::FleetWorkerJoined {
                worker_id: worker_id.to_string(),
                endpoint,
            });
            drop_replaced(bus, &old);
            return Ok(worker_token);
        }
        if self.workers.len() >= FLEET_MAX_WORKERS {
            return Err(FleetError::RegistryFull);
        }
        let record = FleetWorker {
            worker_id: worker_id.to_string(),
            endpoints,
            hostname: hostname.to_string(),
            devices,
            worker_token: worker_token.clone(),
            joined_at: now,
            last_seen: now,
        };
        let endpoint = record.primary_endpoint().to_string();
        self.workers.insert(worker_id.to_string(), record);
        let _ = bus.publish(crate::events::BlazarEvent::FleetWorkerJoined {
            worker_id: worker_id.to_string(),
            endpoint,
        });
        Ok(worker_token)
    }

    /// Constant-time-ish match of a presented token against the
    /// worker's session credential (unknown ids and empty presentations
    /// simply fail).
    #[must_use]
    pub fn worker_token_matches(&self, worker_id: &str, presented: &str) -> bool {
        if presented.is_empty() {
            return false;
        }
        match self.workers.get(worker_id) {
            Some(w) => {
                let a = w.worker_token.as_bytes();
                let b = presented.as_bytes();
                a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
            }
            None => false,
        }
    }

    /// Refresh a worker's liveness (and device report). Unknown ids are
    /// an error, not a silent re-join: a worker that lost its registry
    /// slot (TTL, restart) must re-announce to restore identity state.
    pub fn heartbeat(
        &self,
        worker_id: &str,
        devices: Vec<RpcDeviceMemory>,
        now: Instant,
    ) -> Result<(), FleetError> {
        match self.workers.get_mut(worker_id) {
            Some(mut w) => {
                w.last_seen = now;
                w.devices = devices;
                Ok(())
            }
            None => Err(FleetError::UnknownWorker(worker_id.to_string())),
        }
    }

    /// Explicit departure (`/api/fleet/leave`): remove and emit.
    #[must_use]
    pub fn leave(&self, worker_id: &str, bus: &EventBus) -> Option<FleetWorker> {
        let (_, w) = self.workers.remove(worker_id)?;
        let _ = bus.publish(crate::events::BlazarEvent::FleetWorkerDeparted {
            worker_id: w.worker_id.clone(),
            endpoint: w.primary_endpoint().to_string(),
            reason: "leave".to_string(),
        });
        Some(w)
    }

    /// Drop every worker whose last heartbeat is older than `ttl`.
    /// Lazy: called by the read paths (snapshot, workers list), so a
    /// fleet nobody looks at still prunes the moment something does.
    pub fn prune(&self, now: Instant, ttl: Duration, bus: &EventBus) {
        let expired: Vec<FleetWorker> = self
            .workers
            .iter()
            .filter(|entry| now.duration_since(entry.value().last_seen) > ttl)
            .map(|entry| entry.value().clone())
            .collect();
        for w in expired {
            if let Some((_, removed)) = self.workers.remove(&w.worker_id) {
                let _ = bus.publish(crate::events::BlazarEvent::FleetWorkerDeparted {
                    worker_id: removed.worker_id.clone(),
                    endpoint: removed.primary_endpoint().to_string(),
                    reason: "expired".to_string(),
                });
                tracing::warn!(
                    target: "blazar::fleet",
                    worker_id = %removed.worker_id,
                    endpoint = %removed.primary_endpoint(),
                    "fleet worker missed its TTL window - evicted from spawn planning (restart it with `blazar rpc worker --join` to rejoin)"
                );
            }
        }
    }

    /// One consistent endpoint snapshot for spawn planning: prunes
    /// first, then reads once. The caller feeds THIS exact list to both
    /// the budget probe and the argv merge - never two reads.
    #[must_use]
    pub fn snapshot_targets(&self, now: Instant, ttl: Duration, bus: &EventBus) -> Vec<String> {
        self.prune(now, ttl, bus);
        let mut out = Vec::with_capacity(self.workers.len());
        for entry in &self.workers {
            out.extend(entry.value().endpoints.iter().cloned());
        }
        out
    }

    /// Pruned list for `/api/fleet/workers`.
    #[must_use]
    pub fn workers(&self, now: Instant, ttl: Duration, bus: &EventBus) -> Vec<FleetWorkerView> {
        self.prune(now, ttl, bus);
        self.workers
            .iter()
            .map(|entry| {
                let w = entry.value();
                let age = now.duration_since(w.last_seen);
                FleetWorkerView {
                    worker_id: w.worker_id.clone(),
                    endpoints: w.endpoints.clone(),
                    hostname: w.hostname.clone(),
                    devices: w.devices.clone(),
                    joined_secs_ago: now.duration_since(w.joined_at).as_secs(),
                    last_seen_secs_ago: age.as_secs(),
                    expires_in_secs: ttl.saturating_sub(age).as_secs(),
                }
            })
            .collect()
    }

    /// Live worker count (post-prune) for status surfaces.
    #[must_use]
    pub fn len(&self, now: Instant, ttl: Duration, bus: &EventBus) -> usize {
        self.prune(now, ttl, bus);
        self.workers.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.workers.is_empty()
    }
}

fn drop_replaced(bus: &EventBus, old: &FleetWorker) {
    let _ = bus.publish(crate::events::BlazarEvent::FleetWorkerDeparted {
        worker_id: old.worker_id.clone(),
        endpoint: old.primary_endpoint().to_string(),
        reason: "replaced".to_string(),
    });
}

/// Merge announced fleet endpoints into a compiled engine argv.
///
/// Pure and total: an existing `--rpc host:port,host:port` value gains
/// any announced endpoint it lacks (static list stays first - it is
/// the configured, operator-ordered placement); an argv without
/// `--rpc` gains the flag with the announced list (the zero-static
/// config case - join alone makes the fleet usable); empty `announced`
/// leaves argv byte-identical (the static-only path must never drift).
pub fn merge_fleet_argv(argv: &mut Vec<String>, announced: &[String]) {
    if announced.is_empty() {
        return;
    }
    if let Some(i) = argv.iter().position(|flag| flag == "--rpc") {
        let mut parts: Vec<String> = argv
            .get(i + 1)
            .map(|v| {
                v.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default();
        for ep in announced {
            if !parts.contains(ep) {
                parts.push(ep.clone());
            }
        }
        if i + 1 < argv.len() {
            argv[i + 1] = parts.join(",");
        } else {
            argv.push(parts.join(","));
        }
    } else {
        argv.push("--rpc".to_string());
        argv.push(announced.join(","));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bus() -> EventBus {
        EventBus::new(64)
    }

    fn devices() -> Vec<RpcDeviceMemory> {
        vec![RpcDeviceMemory {
            index: 0,
            free_bytes: 1,
            total_bytes: 2,
        }]
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__join_heartbeat_leave__full_membership_cycle() {
        let reg = FleetRegistry::default();
        let bus = bus();
        let now = Instant::now();
        reg.join(
            "w1",
            vec!["127.0.0.1:50052".into()],
            "box-a",
            devices(),
            now,
            &bus,
        )
        .unwrap();
        reg.heartbeat("w1", devices(), now + Duration::from_secs(5))
            .unwrap();
        let targets =
            reg.snapshot_targets(now + Duration::from_secs(6), Duration::from_secs(45), &bus);
        assert_eq!(targets, vec!["127.0.0.1:50052".to_string()]);
        let left = reg.leave("w1", &bus);
        assert!(left.is_some());
        assert!(reg.is_empty());
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__ttl_prune__silent_worker_evicted_after_window() {
        let reg = FleetRegistry::default();
        let bus = bus();
        let now = Instant::now();
        reg.join(
            "w1",
            vec!["127.0.0.1:50052".into()],
            "box-a",
            devices(),
            now,
            &bus,
        )
        .unwrap();
        // No heartbeats: past the TTL the read paths prune it.
        let targets =
            reg.snapshot_targets(now + Duration::from_secs(46), Duration::from_secs(45), &bus);
        assert!(targets.is_empty(), "expired worker must leave planning");
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__ttl_prune__live_worker_survives() {
        let reg = FleetRegistry::default();
        let bus = bus();
        let now = Instant::now();
        reg.join(
            "w1",
            vec!["127.0.0.1:50052".into()],
            "box-a",
            devices(),
            now,
            &bus,
        )
        .unwrap();
        reg.heartbeat("w1", devices(), now + Duration::from_secs(30))
            .unwrap();
        let targets =
            reg.snapshot_targets(now + Duration::from_secs(31), Duration::from_secs(45), &bus);
        assert_eq!(targets.len(), 1);
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__join__replaces_same_id_without_growing() {
        let reg = FleetRegistry::default();
        let bus = bus();
        let now = Instant::now();
        reg.join(
            "w1",
            vec!["127.0.0.1:5001".into()],
            "box-a",
            devices(),
            now,
            &bus,
        )
        .unwrap();
        reg.join(
            "w1",
            vec!["127.0.0.1:5002".into()],
            "box-a",
            devices(),
            now,
            &bus,
        )
        .unwrap();
        assert_eq!(reg.len(now, Duration::from_secs(45), &bus), 1);
        let targets = reg.snapshot_targets(now, Duration::from_secs(45), &bus);
        assert_eq!(targets, vec!["127.0.0.1:5002".to_string()]);
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__join__bad_ids_and_endpoints_refused() {
        let reg = FleetRegistry::default();
        let bus = bus();
        let now = Instant::now();
        assert!(matches!(
            reg.join("", vec!["127.0.0.1:1".into()], "h", vec![], now, &bus),
            Err(FleetError::BadWorkerId(_))
        ));
        assert!(matches!(
            reg.join("bad id", vec!["127.0.0.1:1".into()], "h", vec![], now, &bus),
            Err(FleetError::BadWorkerId(_))
        ));
        assert!(matches!(
            reg.join("w", vec!["no-port".into()], "h", vec![], now, &bus),
            Err(FleetError::BadEndpoint(_))
        ));
        assert!(matches!(
            reg.join("w", vec![], "h", vec![], now, &bus),
            Err(FleetError::TooManyEndpoints)
        ));
        let many: Vec<String> = (0..5).map(|i| format!("127.0.0.1:{i}")).collect();
        assert!(matches!(
            reg.join("w", many, "h", vec![], now, &bus),
            Err(FleetError::TooManyEndpoints)
        ));
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__registry_cap__refuses_beyond_64() {
        let reg = FleetRegistry::default();
        let bus = bus();
        let now = Instant::now();
        for i in 0..FLEET_MAX_WORKERS {
            reg.join(
                &format!("w{i}"),
                vec![format!("127.0.0.1:{}", 10_000 + i)],
                "h",
                vec![],
                now,
                &bus,
            )
            .unwrap();
        }
        assert!(matches!(
            reg.join(
                "overflow",
                vec!["127.0.0.1:1".into()],
                "h",
                vec![],
                now,
                &bus
            ),
            Err(FleetError::RegistryFull)
        ));
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__heartbeat__unknown_worker_must_rejoin() {
        let reg = FleetRegistry::default();
        let err = reg.heartbeat("ghost", vec![], Instant::now());
        assert!(matches!(err, Err(FleetError::UnknownWorker(_))));
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__events__join_and_depart_ride_the_bus() {
        let reg = FleetRegistry::default();
        let bus = EventBus::new(64);
        let mut rx = bus.subscribe();
        let now = Instant::now();
        reg.join(
            "w1",
            vec!["10.0.0.4:50052".into()],
            "box-a",
            vec![],
            now,
            &bus,
        )
        .unwrap();
        reg.leave("w1", &bus).expect("live worker departs on leave");
        let mut kinds = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            kinds.push(ev);
        }
        assert_eq!(kinds.len(), 2, "join + depart, exactly");
        assert!(matches!(
            kinds[0],
            crate::events::BlazarEvent::FleetWorkerJoined { .. }
        ));
        assert!(matches!(
            kinds[1],
            crate::events::BlazarEvent::FleetWorkerDeparted { .. }
        ));
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__endpoint_validation__bracketed_ipv6_accepted() {
        assert!(valid_endpoint("[::1]:50052"));
        assert!(!valid_endpoint("just-host"));
        assert!(!valid_endpoint("host:99999"));
        assert!(valid_endpoint("127.0.0.1:1"));
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__merge_fleet_argv__existing_rpc_list_appends_dedup() {
        let mut argv = vec![
            "llama-server".to_string(),
            "--rpc".to_string(),
            "10.0.0.2:50052,10.0.0.3:50052".to_string(),
        ];
        merge_fleet_argv(
            &mut argv,
            &["10.0.0.4:50052".to_string(), "10.0.0.2:50052".to_string()],
        );
        assert_eq!(argv[2], "10.0.0.2:50052,10.0.0.3:50052,10.0.0.4:50052");
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__merge_fleet_argv__absent_rpc_flag_inserts() {
        let mut argv = vec![
            "llama-server".to_string(),
            "-m".to_string(),
            "x".to_string(),
        ];
        merge_fleet_argv(&mut argv, &["10.0.0.4:50052".to_string()]);
        assert_eq!(argv[argv.len() - 2], "--rpc");
        assert_eq!(argv[argv.len() - 1], "10.0.0.4:50052");
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__merge_fleet_argv__empty_announced_leaves_argv_identical() {
        let mut argv = vec![
            "llama-server".to_string(),
            "--rpc".to_string(),
            "10.0.0.2:1".to_string(),
        ];
        let before = argv.clone();
        merge_fleet_argv(&mut argv, &[]);
        assert_eq!(argv, before);
    }
}
