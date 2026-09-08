//! Broadcast event bus. Every state transition in Pallama is observable:
//! consumers are structured logs, `/api/events` SSE, and the metrics
//! exporter. No silent anything.

use serde::Serialize;
use tokio::sync::broadcast;

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PallamaEvent {
    EngineUpdated {
        tag: String,
    },
    EngineRemoved {
        tag: String,
    },
    /// J2 self-healing: the supervisor rolled the active engine back to
    /// the previous install after spawn failures. Loud + reversible
    /// (`pallama engine use <tag>` switches back).
    EngineRolledBack {
        from: String,
        to: String,
        reason: String,
    },
    /// LC1 predictive pre-loading: the supervisor pre-spawned `model`
    /// because it historically follows `from` — the switch will be warm.
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
}

#[derive(Debug, Clone)]
pub struct EventBus {
    tx: broadcast::Sender<PallamaEvent>,
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new(1024)
    }
}

impl EventBus {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    /// Publish to all subscribers. Returns the receiver count; publishing
    /// with zero subscribers is not an error (fire-and-forget observability).
    #[allow(clippy::must_use_candidate)] // observability fire-and-forget
    pub fn publish(&self, event: PallamaEvent) -> usize {
        self.tx.send(event).unwrap_or(0)
    }

    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<PallamaEvent> {
        self.tx.subscribe()
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
        bus.publish(PallamaEvent::ModelPulled {
            name: "m".into(),
            warning: None,
        });
        let got = rx.recv().await.unwrap();
        assert_eq!(
            got,
            PallamaEvent::ModelPulled {
                name: "m".into(),
                warning: None
            }
        );
    }

    #[tokio::test]
    async fn unit__bus__json_shape__tagged_snake_case() {
        let e = PallamaEvent::InstanceStateChanged {
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
        assert_eq!(bus.publish(PallamaEvent::QueueDepth { n: 1 }), 0);
    }
}
