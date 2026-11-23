//! Broadcast event bus. Every state transition in Blazar is observable:
//! consumers are structured logs, `/api/events` SSE, and the metrics
//! exporter. No silent anything.

use std::collections::VecDeque;

use serde::Serialize;
use tokio::sync::broadcast;

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BlazarEvent {
    EngineUpdated {
        tag: String,
    },
    EngineRemoved {
        tag: String,
    },
    /// J2 self-healing: the supervisor rolled the active engine back to
    /// the previous install after spawn failures. Loud + reversible
    /// (`blazar engine use <tag>` switches back).
    EngineRolledBack {
        from: String,
        to: String,
        reason: String,
    },
    /// LC1 predictive pre-loading: the supervisor pre-spawned `model`
    /// because it historically follows `from` - the switch will be warm.
    ModelPreloaded {
        model: String,
        from: String,
    },
    /// LC4 adaptive capacity: sustained concurrent load bumped the
    /// model's effective slots (in-memory; restart resets, `tune --slots`
    /// persists).
    SlotsAutoAdopted {
        model: String,
        from: u32,
        to: u32,
    },
    /// Slot auto-fit at spawn: the default ctx was re-spent as N shallower
    /// parallel slots (identical total-ctx capacity budget) because the
    /// full-depth ctx only fit one concurrent slot on this card.
    SlotsCtxAutoFit {
        model: String,
        per_slot_ctx: u32,
        slots: u32,
        total_ctx: u32,
    },
    /// LC4 adaptive capacity: a queued reshape (adoption or decay) landed -
    /// the child respawned with the new slot count and parked admission
    /// waiters were woken. Emitted when the drain completes, not when the
    /// adoption fires (adoption only queues; the drain performs it).
    SlotsReshaped {
        model: String,
        slots: u32,
    },
    /// D2 pinned teach: sustained admission pressure on a model excluded
    /// from adaptation ONLY by a manual `slots` pin. The pin stays
    /// authoritative (no adoption happens) - the event makes the
    /// silent queueing floor visible; the remedy is
    /// `model_overrides.<model>.adaptive_override on`.
    SlotsPinnedPressure {
        model: String,
        /// The pinned slot count (argv-resolved).
        slots: u32,
        /// Parked-waiter gauge level at the teach tick.
        pressure: u32,
    },
    /// D3 starvation bound: the reshape drain had parked admissions past
    /// the drain budget (90s) and switched to complete-and-cutover -
    /// admissions reopened into the old shape and the reshape now waits
    /// for a natural quiesce instead of starving waiters.
    SlotsDrainCutover {
        model: String,
        waited_secs: u64,
    },
    /// D5 decode-scaling awareness: fresh solo + loaded throughput
    /// evidence showed the model's aggregate decode barely scales with
    /// concurrency (GPU-bound), so a slot adoption was suppressed - the
    /// reshape would pay its transition for queue relief at best. The
    /// lane is taught instead (`blazar tune --slots` for an explicit
    /// shape, or `adaptive_decode_gate = false` to restore always-adopt).
    SlotsDecodeBound {
        model: String,
        /// Solo-class decode EWMA (tok/s).
        solo_tps: f64,
        /// Loaded-class per-stream decode EWMA (tok/s).
        loaded_tps: f64,
        /// Loaded-class average concurrency (streams).
        loaded_conc: f64,
        /// `loaded_tps * loaded_conc / solo_tps` - the aggregate multiple.
        ratio: f64,
    },
    /// Replica pre-warm cutover: the reshape landed through a warm
    /// standby - the new-shape child reached Ready under a `#warm` key,
    /// routing swapped to it atomically, and the old child retired (its
    /// in-flight streams finish undisturbed). Zero admission gap vs the
    /// evict-then-respawn path.
    SlotsPrewarmCutover {
        model: String,
        slots: u32,
    },
    /// B2 spec governor: sustained speculation overhead (queue
    /// saturation or a collapsed draft-acceptance rate) parked
    /// speculation for `model` - the child reshapes dense; a long quiet
    /// streak restores it (`SpecGovernorRecovered`). The `reason` is
    /// "saturation" or "acceptance".
    SpecGovernorOff {
        model: String,
        reason: String,
    },
    /// B2 spec governor: the quiet streak was satisfied - speculation
    /// is restored for `model` and the override cleared (the natural
    /// overlay/config resolution resumes).
    SpecGovernorRecovered {
        model: String,
    },
    /// `autotune = "advise"` periodic scan found optimization
    /// opportunities for `model` (read-only by construction: the
    /// operator closes the gaps by hand). `next` is the
    /// highest-priority recommended command, `warnings`/`recommendations`
    /// count the plan's warn findings and recs.
    AutotuneAdvice {
        model: String,
        warnings: u32,
        recommendations: u32,
        next: Option<String>,
    },
    /// Scheduled autotune (`autotune = "scheduled"`) finished a spec-mode
    /// race on an idle lane and landed receipts: `arms` raced, of which
    /// `parity_failures` diverged from the dense reference under greedy
    /// parity probes (those arms land non-adoption-grade evidence). The
    /// winning mode and its gain over dense are advisory context - actual
    /// adoption happens later, at spawn, via the receipt resolver.
    AutotuneRace {
        model: String,
        arms: u32,
        parity_failures: u32,
        best_mode: Option<String>,
        /// (best tok/s / dense tok/s - 1) * 100, rounded; None when dense
        /// did not run or measured zero.
        best_gain_pct: Option<i64>,
    },
    ModelPulled {
        name: String,
        /// Post-download GGUF health check: set when the header did not
        /// parse (engine will likely refuse to load; carries quant
        /// alternatives from the same repo).
        #[serde(default)]
        warning: Option<String>,
    },
    ModelRemoved {
        name: String,
    },
    PullProgress {
        name: String,
        downloaded: u64,
        total: u64,
    },
    PullFailed {
        name: String,
        error: String,
    },
    InstanceStateChanged {
        name: String,
        state: InstanceState,
    },
    BenchmarkDone {
        name: String,
        tok_s: f64,
    },
    QueueDepth {
        n: usize,
    },
    /// A failover chain moved to a different target: the previous hop
    /// failed (unknown model, load/spawn error, 5xx) and the chain
    /// benched it for `min_residence_secs`. Visible in `/api/events`
    /// and the chain's history via `/api/failover/<name>`.
    FailoverSwitched {
        chain: String,
        from: usize,
        to: usize,
        reason: String,
    },
    /// Automatic fleet membership: a `blazar rpc worker --join` announced
    /// itself and is now part of spawn planning. `endpoint` is the
    /// worker's primary (first) RPC endpoint.
    FleetWorkerJoined {
        worker_id: String,
        endpoint: String,
    },
    /// Automatic fleet membership: a worker left the registry - it
    /// announced a shutdown (`leave`), missed its TTL window
    /// (`expired`), or was displaced by a re-join with a fresh identity
    /// (`replaced`).
    FleetWorkerDeparted {
        worker_id: String,
        endpoint: String,
        reason: String,
    },
    /// Hot reload applied a new config generation. `cold_only` names the
    /// knobs that were NOT applied because they are bind-time or baked
    /// into long-lived objects (see `Config::cold_only_diff`); empty
    /// means the whole file took effect live.
    ConfigReloaded {
        cold_only: Vec<String>,
    },
    /// Hot reload rejected the on-disk config (parse/validation error)
    /// and kept the previous generation running.
    ConfigReloadFailed {
        reason: String,
    },
}

/// Lifecycle state of a model instance.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InstanceState {
    Loading,
    Ready,
    Sleeping,
    Evicted,
    Crashed,
}

impl InstanceState {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Loading => "loading",
            Self::Ready => "ready",
            Self::Sleeping => "sleeping",
            Self::Evicted => "evicted",
            Self::Crashed => "crashed",
        }
    }

    /// The lifecycle contract as an executable table. Every state flip in
    /// the supervisor must be a legal edge here; the full matrix is pinned
    /// by `unit__instance_state__transition_matrix_is_the_documented_contract`.
    ///
    /// Legal edges (8) and their owners:
    ///
    /// | from | to | owner / trigger |
    /// |---|---|---|
    /// | Loading | Ready | spawn path after the health check passes |
    /// | Loading | Crashed | child died before first health (counts toward the breaker) |
    /// | Ready | Sleeping | engine check observes the child self-slept (`--sleep-idle-seconds`) |
    /// | Ready | Evicted | idle ladder, capacity admission, user stop, header-timeout wedge |
    /// | Ready | Crashed | reap path: child died while serving |
    /// | Sleeping | Ready | request touch wakes the child (`was_sleeping` in the ensure path) |
    /// | Sleeping | Evicted | same eviction funnels as Ready |
    /// | Sleeping | Crashed | reap path: sleeper died |
    ///
    /// Everything else is illegal, notably: no edge back INTO Loading (a
    /// respawn is a fresh instance, never a state flip), and no edges out
    /// of Evicted/Crashed (the map entry is removed; `Evicted` is a
    /// posthumous marker so `blazar ps` can attribute the vanished child).
    #[must_use]
    pub fn can_transition(from: Self, to: Self) -> bool {
        matches!(
            (from, to),
            (Self::Loading, Self::Ready | Self::Crashed)
                | (Self::Ready, Self::Sleeping | Self::Evicted | Self::Crashed)
                | (Self::Sleeping, Self::Ready | Self::Evicted | Self::Crashed)
        )
    }

    /// Terminal lifecycle states: the instance is gone from the map (the
    /// state is only retained as an attribution marker).
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Evicted | Self::Crashed)
    }
}

#[derive(Debug, Clone)]
pub struct EventBus {
    tx: broadcast::Sender<BlazarEvent>,
    /// Bounded retention of recent events (observability replay for
    /// consumers that connect late - SSE dashboards refreshing a
    /// browser lose their stream on every reload). Arc so the derived
    /// Clone keeps bus semantics: a cloned bus shares one ring.
    /// Plain mutex: the critical section is a push + trim.
    ring: std::sync::Arc<std::sync::Mutex<VecDeque<BlazarEvent>>>,
}

/// Replay depth: enough that a dashboard refresh after a burst (pull
/// progress, failover ladder) still shows what happened; small enough
/// that retention stays a rounding error next to the broadcast buffer.
const REPLAY_DEPTH: usize = 256;

impl Default for EventBus {
    fn default() -> Self {
        Self::new(1024)
    }
}

impl EventBus {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self {
            tx,
            ring: std::sync::Arc::new(std::sync::Mutex::new(VecDeque::new())),
        }
    }

    /// Publish to all subscribers. Returns the receiver count; publishing
    /// with zero subscribers is not an error (fire-and-forget observability).
    /// The event is recorded for replay BEFORE broadcasting: a consumer
    /// that subscribes and then snapshots [`recent`] can therefore see a
    /// rare duplicate at the boundary (recorded + received live) but
    /// never a gap - missed events are the failure mode dashboards
    /// actually care about.
    #[allow(clippy::must_use_candidate)] // observability fire-and-forget
    pub fn publish(&self, event: BlazarEvent) -> usize {
        {
            let mut ring = self
                .ring
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if ring.len() == REPLAY_DEPTH {
                ring.pop_front();
            }
            ring.push_back(event.clone());
        }
        self.tx.send(event).unwrap_or(0)
    }

    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<BlazarEvent> {
        self.tx.subscribe()
    }

    /// Recent events, oldest first, up to [`REPLAY_DEPTH`]. Replay
    /// consumers subscribe FIRST, then snapshot this, then stream live:
    /// replay-then-live covers every event exactly once except a
    /// microsecond boundary duplicate (see [`publish`]).
    #[must_use]
    pub fn recent(&self) -> Vec<BlazarEvent> {
        self.ring
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .cloned()
            .collect()
    }
}

/// Resolves when the process receives an interrupt-class signal (SIGINT,
/// SIGTERM, SIGHUP on unix; Ctrl+C on Windows). Use inside `tokio::select!`
/// to cancel long-running futures gracefully: cancellation drops the
/// future's locals, which releases file locks and keeps `.part` files
/// for resume. A second signal hits the default disposition (hard kill).
pub async fn interrupted() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut sigint = signal(SignalKind::interrupt()).expect("install SIGINT handler");
        let mut sigterm = signal(SignalKind::terminate()).expect("install SIGTERM handler");
        let mut sighup = signal(SignalKind::hangup()).expect("install SIGHUP handler");
        tokio::select! {
            _ = sigint.recv() => {}
            _ = sigterm.recv() => {}
            _ = sighup.recv() => {}
        }
    }
    #[cfg(windows)]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn unit__bus__publish_reaches_subscriber() {
        let bus = EventBus::default();
        let mut rx = bus.subscribe();
        bus.publish(BlazarEvent::ModelPulled {
            name: "m".into(),
            warning: None,
        });
        let got = rx.recv().await.unwrap();
        assert_eq!(
            got,
            BlazarEvent::ModelPulled {
                name: "m".into(),
                warning: None
            }
        );
    }

    #[tokio::test]
    async fn unit__bus__json_shape__tagged_snake_case() {
        let e = BlazarEvent::InstanceStateChanged {
            name: "m".into(),
            state: InstanceState::Ready,
        };
        let j = serde_json::to_value(&e).unwrap();
        assert_eq!(j["type"], "instance_state_changed");
        assert_eq!(j["state"], "ready");
        assert_eq!(InstanceState::Sleeping.as_str(), "sleeping");
    }

    #[tokio::test]
    async fn unit__bus__no_subscriber__not_an_error() {
        let bus = EventBus::default();
        assert_eq!(bus.publish(BlazarEvent::QueueDepth { n: 1 }), 0);
    }

    #[test]
    fn unit__bus__recent_replays_published_events_oldest_first() {
        let bus = EventBus::default();
        bus.publish(BlazarEvent::QueueDepth { n: 1 });
        bus.publish(BlazarEvent::QueueDepth { n: 2 });
        // A late consumer (dashboard refresh) sees the history it missed.
        assert_eq!(
            bus.recent(),
            vec![
                BlazarEvent::QueueDepth { n: 1 },
                BlazarEvent::QueueDepth { n: 2 }
            ]
        );
    }

    #[test]
    fn unit__bus__recent_is_bounded__oldest_evicted_first() {
        let bus = EventBus::default();
        for n in 0..(REPLAY_DEPTH + 10) {
            bus.publish(BlazarEvent::QueueDepth { n });
        }
        let recent = bus.recent();
        assert_eq!(recent.len(), REPLAY_DEPTH);
        assert_eq!(
            recent.first(),
            Some(&BlazarEvent::QueueDepth { n: 10 }),
            "the oldest events fell off, the newest REPLAY_DEPTH stayed"
        );
        assert_eq!(
            recent.last(),
            Some(&BlazarEvent::QueueDepth {
                n: REPLAY_DEPTH + 9
            })
        );
    }

    /// Pins the complete 5x5 transition matrix: exactly the 8 documented
    /// edges are legal, all 17 others (incl. every edge into Loading and
    /// every edge out of Evicted/Crashed) are refused. If a new edge is
    /// ever needed, change `can_transition` and this test together - the
    /// doc table in its comment is the same contract.
    #[test]
    fn unit__instance_state__transition_matrix_is_the_documented_contract() {
        use InstanceState as S;
        const ALL: [S; 5] = [S::Loading, S::Ready, S::Sleeping, S::Evicted, S::Crashed];
        let legal = [
            (S::Loading, S::Ready),
            (S::Loading, S::Crashed),
            (S::Ready, S::Sleeping),
            (S::Ready, S::Evicted),
            (S::Ready, S::Crashed),
            (S::Sleeping, S::Ready),
            (S::Sleeping, S::Evicted),
            (S::Sleeping, S::Crashed),
        ];
        let mut legal_count = 0;
        for from in ALL {
            for to in ALL {
                let allowed = S::can_transition(from, to);
                assert_eq!(
                    allowed,
                    legal.contains(&(from, to)),
                    "edge {from:?} -> {to:?} disagrees with the documented contract"
                );
                legal_count += usize::from(allowed);
            }
        }
        assert_eq!(legal_count, 8);

        // Terminal semantics: posthumous markers, nothing follows them.
        assert!(S::Evicted.is_terminal() && S::Crashed.is_terminal());
        assert!(!S::Loading.is_terminal());
        assert!(!S::Ready.is_terminal());
        assert!(!S::Sleeping.is_terminal());
    }
}
