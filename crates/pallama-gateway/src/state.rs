//! Shared gateway state.

use std::sync::Arc;

use pallama_core::{Config, PallamaDirs};
use pallama_runtime::{EventBus, Supervisor};

use crate::queue::PriorityQueue;

pub struct AppState {
    pub dirs: PallamaDirs,
    pub config: Config,
    pub sup: Arc<Supervisor>,
    pub bus: EventBus,
    pub queue: Arc<PriorityQueue>,
    /// Loopback client for child traffic + metrics scrape.
    pub http: reqwest::Client,
}

impl AppState {
    #[must_use] 
    pub fn new(
        dirs: PallamaDirs,
        config: Config,
        sup: Arc<Supervisor>,
        bus: EventBus,
    ) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()
            .expect("gateway http client");
        Self {
            dirs,
            config,
            sup,
            bus,
            queue: Arc::new(PriorityQueue::new()),
            http,
        }
    }
}
