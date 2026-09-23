//! Shared HTTP client tuning for the gateway's reqwest builders.
//!
//! Every long-lived client (child transport, batch fan-out, OTLP flusher)
//! gets the same explicit pool policy: idle connections are kept warm for
//! reuse but bounded in number so a burst of requests cannot leave an
//! unbounded set of open sockets behind, and TCP keepalive probes dead
//! peers so a stalled connection is detected instead of hanging until the
//! request timeout fires.

use std::time::Duration;

/// Idle-connection lifetime before the pool reaps a kept-alive socket.
const POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(90);

/// Upper bound on idle connections the pool retains per host.
const POOL_MAX_IDLE_PER_HOST: usize = 16;

/// TCP keepalive interval so half-open connections are detected early.
const TCP_KEEPALIVE: Duration = Duration::from_secs(30);

/// Apply the shared pool policy to a client builder.
pub fn tuned(builder: reqwest::ClientBuilder) -> reqwest::ClientBuilder {
    builder
        .pool_idle_timeout(POOL_IDLE_TIMEOUT)
        .pool_max_idle_per_host(POOL_MAX_IDLE_PER_HOST)
        .tcp_keepalive(TCP_KEEPALIVE)
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn unit__tuned__builds_a_usable_client() {
        let client = tuned(reqwest::Client::builder()).build();
        assert!(client.is_ok(), "tuned pool policy must yield a client");
    }
}
