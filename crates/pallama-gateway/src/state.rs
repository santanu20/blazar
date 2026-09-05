//! Shared gateway state.

use std::sync::Arc;

use pallama_core::{Config, PallamaDirs};
use pallama_runtime::{EventBus, Supervisor};

use crate::histogram::Histogram;
use crate::queue::PriorityQueue;
use crate::sentinel::Sentinel;

pub struct AppState {
    pub dirs: PallamaDirs,
    pub config: Config,
    pub sup: Arc<Supervisor>,
    pub bus: EventBus,
    pub queue: Arc<PriorityQueue>,
    /// Loopback client for child traffic + metrics scrape.
    pub http: reqwest::Client,
    /// vLLM-style evidence loop: measured at the proxy, owned by the gateway.
    pub ttft: Histogram,
    pub tpot: Histogram,
    /// Warn-only response-semantics observation layer (`pallama why`).
    pub sentinel: Arc<Sentinel>,
}

impl AppState {
    #[must_use] 
    #[allow(clippy::duration_suboptimal_units)] // 10-minute ceiling mirrors long generations
    pub fn new(
        dirs: PallamaDirs,
        config: Config,
        sup: Arc<Supervisor>,
        bus: EventBus,
    ) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10 * 60))
            .build()
            .expect("gateway http client");
        let run_dir = dirs.run_dir();
        let sentinel = Sentinel::new(
            config.sentinel,
            config.sentinel_stall_secs,
            Some(&run_dir),
        );
        Self {
            dirs,
            config,
            sup,
            bus,
            queue: Arc::new(PriorityQueue::new()),
            http,
            ttft: crate::histogram::ttft(),
            tpot: crate::histogram::tpot(),
            sentinel,
        }
    }
}
