//! LAN autodiscovery for fleet membership: a UDP responder on a
//! dedicated port (default 11439) that answers a fixed magic packet
//! with the gateway's HTTP port and join requirements. A worker with
//! `blazar rpc worker --discover` broadcasts the magic, takes the
//! first reply's SOURCE address as the gateway, and joins through the
//! normal `/api/fleet/join` flow - no addresses typed by hand.
//!
//! Security posture: the responder is OFF unless `discovery_enabled`
//! is set, it leaks no secrets (the reply names the HTTP port and
//! whether a token is required - nothing else), and a global reply
//! budget caps discovery-sweep amplification.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::net::UdpSocket;

use crate::supervisor::Supervisor;

/// The exact packet a seeker sends; anything else is ignored.
pub const DISCOVER_MAGIC: &[u8] = b"BLAZAR_DISCOVER_V1";

/// Reply budget per one-second window across ALL seekers: discovery
/// sweeps are rare, so a flat global cap is enough (a per-IP map would
/// be state an attacker could grow).
const REPLY_BUDGET_PER_SEC: u64 = 64;

/// Bind the responder socket. Public so a gateway can host discovery
/// from its own supervision if ever needed; the daemon simply spawns
/// [`run_discovery_responder`].
pub async fn bind_responder(port: u16) -> std::io::Result<UdpSocket> {
    UdpSocket::bind(("0.0.0.0", port)).await
}

/// One-shot reply payload. `hostname` is informational only - the
/// seeker uses the SOURCE address of this reply as the gateway host,
/// so a wrong hostname never breaks the join.
fn reply_payload(http_port: u16, token_required: bool, version: &str) -> String {
    serde_json::json!({
        "http_port": http_port,
        "fleet_token_required": token_required,
        "blazar_version": version,
        "hostname": std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".into()),
    })
    .to_string()
}

/// Beacon lifecycle owner. Spawned UNCONDITIONALLY by the gateway and
/// kept for the daemon's life: it binds or unbinds the UDP responder
/// whenever a `ConfigReloaded` event (or the initial reconcile) says
/// `discovery_enabled` / `discovery_port` changed. This is what makes
/// the flag fully hot in BOTH directions - a daemon booted with the
/// beacon off starts answering the moment the config turns it on, and
/// one booted with it on goes silent (socket released) the moment it
/// turns off - no restart involved. Bind failure is a WARN and the
/// lifecycle stays alive to retry on the next reload event; HTTP
/// serving is never coupled to the discovery port.
pub async fn run_discovery_responder(sup: Arc<Supervisor>, http_port: u16) {
    let mut events = sup.bus.subscribe();
    let mut bound: Option<(u16, tokio::task::JoinHandle<()>)> = None;
    loop {
        // Reconcile: bind/unbind to match the CURRENT config.
        let cfg = sup.cfg();
        let desired = if cfg.discovery_enabled {
            Some(cfg.discovery_port)
        } else {
            None
        };
        let key_changed = bound.as_ref().map(|(p, _)| *p) != desired;
        if key_changed {
            if let Some((old_port, task)) = bound.take() {
                task.abort(); // dropping the socket unbinds the port
                tracing::info!("discovery beacon unbound from udp://{old_port}");
            }
            if let Some(port) = desired {
                match bind_responder(port).await {
                    Ok(socket) => {
                        tracing::info!(
                            "discovery beacon bound on udp://{port} (magic {DISCOVER_MAGIC:?})"
                        );
                        let sup2 = sup.clone();
                        bound = Some((
                            port,
                            tokio::spawn(async move {
                                serve_beacon(socket, sup2, http_port).await;
                            }),
                        ));
                    }
                    Err(e) => {
                        tracing::warn!(
                            "discovery beacon not bound on udp://{port}: {e} - retrying on the next config reload"
                        );
                    }
                }
            }
        }
        // Sleep until the config world might have changed; any bus event
        // is a cheap enough trigger for one reconcile pass.
        let _ = events.recv().await;
    }
}

/// Packet loop for one bound socket. The per-probe
/// `discovery_enabled` gate is defense-in-depth: the lifecycle owner
/// above already unbinds on disable, but a probe racing the reconcile
/// must also find silence.
async fn serve_beacon(socket: UdpSocket, sup: Arc<Supervisor>, http_port: u16) {
    let window_start = std::sync::Mutex::new(std::time::Instant::now());
    let replies = AtomicU64::new(0);
    let mut buf = [0u8; 64];
    loop {
        let (len, peer) = match socket.recv_from(&mut buf).await {
            Ok(x) => x,
            Err(e) => {
                tracing::warn!("discovery recv error: {e} - responder restarting");
                tokio::time::sleep(Duration::from_secs(1)).await;
                continue;
            }
        };
        if &buf[..len] != DISCOVER_MAGIC {
            continue;
        }
        // Hot gate: a discovery_enabled=false flip (config hot-reload) must
        // silence the beacon immediately - the lifecycle owner unbinds the
        // socket, but a probe racing that reconcile still gets no reply.
        if !sup.cfg().discovery_enabled {
            continue;
        }
        // Global rate budget: reset the counter each second, drop
        // replies past the cap (a sweep should never swamp the LAN).
        {
            let mut w = window_start.lock().expect("discovery window lock");
            if w.elapsed() >= Duration::from_secs(1) {
                *w = std::time::Instant::now();
                replies.store(0, Ordering::Relaxed);
            }
        }
        if replies.fetch_add(1, Ordering::Relaxed) >= REPLY_BUDGET_PER_SEC {
            continue;
        }
        let cfg = sup.cfg();
        let token_required = !cfg.fleet_token.is_empty();
        let payload = reply_payload(http_port, token_required, env!("CARGO_PKG_VERSION"));
        if let Err(e) = socket.send_to(payload.as_bytes(), peer).await {
            tracing::debug!("discovery reply to {peer} failed: {e}");
        }
    }
}

/// Seeker side: broadcast + same-host unicast the magic, collect
/// replies for up to `wait`, and return the first gateway as
/// `http://<source-ip>:<http_port>` plus whether a fleet token is
/// required. Same-host unicast guarantees a hit when gateway and
/// worker share a machine (broadcast loops are not guaranteed).
pub async fn seek_gateway(port: u16, wait: Duration) -> std::io::Result<Option<(String, bool)>> {
    let mut all = seek_gateways(port, wait).await?;
    Ok(all.drain(..).next())
}

/// Multi-gateway variant for the interactive `blazar join` picker:
/// collects EVERY distinct gateway that answers within `wait`, deduped
/// by final base URL (a gateway reachable over several interfaces, or
/// via both the broadcast and the same-host unicast, is one choice,
/// not three). Ordered by first reply.
pub async fn seek_gateways(port: u16, wait: Duration) -> std::io::Result<Vec<(String, bool)>> {
    let socket = UdpSocket::bind(("0.0.0.0", 0)).await?;
    socket.set_broadcast(true)?;
    socket
        .send_to(DISCOVER_MAGIC, ("255.255.255.255", port))
        .await
        .ok();
    socket
        .send_to(DISCOVER_MAGIC, ("127.0.0.1", port))
        .await
        .ok();
    let mut seen: Vec<SocketAddr> = Vec::new();
    let mut found: Vec<(String, bool)> = Vec::new();
    let deadline = tokio::time::Instant::now() + wait;
    loop {
        let now = tokio::time::Instant::now();
        if now >= deadline {
            return Ok(found);
        }
        let mut buf = [0u8; 512];
        match tokio::time::timeout_at(deadline, socket.recv_from(&mut buf)).await {
            Ok(Ok((len, peer))) => {
                if seen.contains(&peer) {
                    continue;
                }
                seen.push(peer);
                let Ok(v) = serde_json::from_slice::<serde_json::Value>(&buf[..len]) else {
                    continue;
                };
                let Some(http_port) = v.get("http_port").and_then(|p| p.as_u64()) else {
                    continue;
                };
                let token_required = v
                    .get("fleet_token_required")
                    .and_then(|t| t.as_bool())
                    .unwrap_or(false);
                let host = peer.ip();
                let base = format!("http://{host}:{http_port}");
                // Dedup by base: same gateway seen over two paths is one
                // picker entry (first reply wins the position).
                if found.iter().any(|(b, _)| *b == base) {
                    continue;
                }
                found.push((base, token_required));
            }
            _ => return Ok(found),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit__reply_payload__names_port_and_token_flag_only() {
        let v: serde_json::Value =
            serde_json::from_str(&reply_payload(11438, true, "0.29.0")).unwrap();
        assert_eq!(v["http_port"], 11438);
        assert_eq!(v["fleet_token_required"], true);
        assert_eq!(v["blazar_version"], "0.29.0");
        // No secrets: the payload has exactly these four keys.
        let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "blazar_version",
                "fleet_token_required",
                "hostname",
                "http_port"
            ]
        );
    }

    #[tokio::test]
    async fn unit__discovery_roundtrip__seeker_gets_gateway_from_reply_source() {
        // Bind a responder-like socket on an ephemeral port, run the
        // seeker against it, and prove the reply's SOURCE address
        // (127.0.0.1) and payload port become the gateway base.
        let responder = UdpSocket::bind(("127.0.0.1", 0)).await.unwrap();
        let port = responder.local_addr().unwrap().port();
        let task = tokio::spawn(async move {
            let mut buf = [0u8; 64];
            let (len, peer) = responder.recv_from(&mut buf).await.unwrap();
            assert_eq!(&buf[..len], DISCOVER_MAGIC);
            responder
                .send_to(reply_payload(11438, false, "test").as_bytes(), peer)
                .await
                .unwrap();
        });
        let found = seek_gateway(port, Duration::from_secs(3)).await.unwrap();
        task.await.unwrap();
        let (base, token_required) = found.expect("seeker must find the responder");
        assert_eq!(base, "http://127.0.0.1:11438");
        assert!(!token_required);
    }

    #[tokio::test]
    async fn unit__seek_gateway__silence_times_out_with_none() {
        // Port with no listener: broadcast may go nowhere, unicast to
        // 127.0.0.1 gets ECONNREFUSED surfaced as a timeout -> None.
        let dead = bind_responder(0).await.unwrap();
        let port = dead.local_addr().unwrap().port();
        drop(dead);
        let found = seek_gateway(port, Duration::from_millis(700))
            .await
            .unwrap();
        assert!(found.is_none());
    }
}
