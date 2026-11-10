//! Donor-side adoption agent for guided cluster onboarding.
//!
//! The agent is the DONOR half of `blazar adopt`: a foreground process
//! (`blazar agent run`) that (a) answers `BLAZAR_ADOPT_V1` UDP probes
//! so gateways can list this box, and (b) accepts TCP invitations from
//! gateways on the same port number. An invitation is approved either
//! by policy (`agent_policy = "token"`: the invite carries this box's
//! own `fleet_token`) or by the owner (`ask-owner`: the invite queues
//! until `blazar agent approve` says yes). Approval hands the gateway
//! base to the local worker cascade - the joining itself is the same
//! live-proven `rpc worker` loop, nothing new on the wire.
//!
//! Nothing secret ever leaves this box in a beacon or a reply: the
//! fleet token only travels back over the LOCAL channel to the worker
//! this agent starts, never to the inviting gateway.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use blazar_core::{AgentPolicy, BlazarDirs, Config};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream, UdpSocket};
use tokio::sync::mpsc;

/// Probe magic for the adopt beacon (UDP). Bumped per wire change.
pub const ADOPT_MAGIC: &[u8] = b"BLAZAR_ADOPT_V1";
/// Global beacon reply budget - one box must never become a LAN echo
/// service under a probe sweep.
const REPLY_BUDGET_PER_SEC: u64 = 32;
/// An invite payload larger than this is not an invite - drop it before
/// parsing so a stray connection cannot balloon memory.
const INVITE_MAX_BYTES: usize = 4096;

/// What a gateway's invitation looks like on the wire.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentInvite {
    pub gateway_name: String,
    /// Base URL to join, e.g. `http://192.168.1.10:11438`.
    pub gateway_base: String,
    /// Gateway invite code (from `/api/fleet/invite`) or `fleet_token`
    /// when the gateway runs token policy - which one it is depends on
    /// the gateway's config, the agent only matches it.
    #[serde(default)]
    pub code: Option<String>,
}

/// Wire reply to the inviting gateway. Carries the verdict only.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AgentReply {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl AgentReply {
    fn denied(reason: impl Into<String>) -> Self {
        Self {
            status: "denied".into(),
            reason: Some(reason.into()),
        }
    }
}

/// LOCAL approval event: everything the worker cascade needs. Never
/// serialized onto the network.
#[derive(Debug, Clone)]
pub struct AgentApproval {
    pub gateway_base: String,
    /// This donor's fleet token when one is configured (the worker
    /// presents it when joining), else `None`.
    pub fleet_token: Option<String>,
}

/// Constant-time-ish comparison of a presented invite code against the
/// configured fleet token (same shape as the fleet registry's worker
/// token match): equal-length XOR fold, so a mismatch reveals no early
/// exit. Empty presentations simply fail.
#[must_use]
fn constant_time_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Policy decision for one invitation. `approval` is set exactly when
/// the reply is `approved`.
#[must_use]
pub fn decide_invite(
    policy: AgentPolicy,
    donor_token: &str,
    invite: &AgentInvite,
) -> (AgentReply, Option<AgentApproval>) {
    match policy {
        AgentPolicy::Token => {
            if donor_token.is_empty() {
                return (
                    AgentReply::denied(
                        "this agent runs token policy but no fleet_token is set in its config",
                    ),
                    None,
                );
            }
            if invite
                .code
                .as_deref()
                .is_some_and(|code| constant_time_eq(code, donor_token))
            {
                (
                    AgentReply {
                        status: "approved".into(),
                        reason: None,
                    },
                    Some(AgentApproval {
                        gateway_base: invite.gateway_base.clone(),
                        fleet_token: Some(donor_token.to_string()),
                    }),
                )
            } else {
                (
                    AgentReply::denied("invite carried no matching fleet token"),
                    None,
                )
            }
        }
        AgentPolicy::AskOwner => (
            AgentReply {
                status: "queued".into(),
                reason: Some(
                    "owner approval required - run `blazar agent list` on the donor".into(),
                ),
            },
            None,
        ),
    }
}

/// Beacon payload: what a scanning gateway learns about this box.
/// Exactly these keys - no identifiers beyond the hostname, no secrets.
#[must_use]
pub fn beacon_payload(policy: AgentPolicy, gpu_name: Option<&str>, free_vram_mib: u64) -> String {
    let hostname = std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".into());
    serde_json::json!({
        "hostname": hostname,
        "gpu_name": gpu_name,
        "free_vram_mib": free_vram_mib,
        "policy": if matches!(policy, AgentPolicy::Token) { "token" } else { "ask-owner" },
        "agent_version": env!("CARGO_PKG_VERSION"),
    })
    .to_string()
}

/// A queued invitation the owner has not decided on yet.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PendingInvite {
    pub invite: AgentInvite,
    /// Unix millis when the invite arrived (for aging and display).
    pub received_ms: u64,
}

fn pending_dir(dirs: &BlazarDirs) -> PathBuf {
    dirs.data_dir.join("agent").join("pending")
}

/// Queue an invitation for the owner. One file per invite so `list` and
/// `approve` are plain directory reads - no daemon state to corrupt.
/// The queue is bounded: entries older than a day are dropped on write,
/// and a full queue drops the OLDEST survivors - an agent can always
/// re-ask, an unattended donor must not accumulate forever.
pub fn save_pending(dirs: &BlazarDirs, pending: &PendingInvite) -> std::io::Result<PathBuf> {
    const PENDING_TTL_MS: u64 = 24 * 60 * 60 * 1000;
    const PENDING_MAX: usize = 64;
    let dir = pending_dir(dirs);
    std::fs::create_dir_all(&dir)?;
    let now_ms = u64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis()),
    )
    .unwrap_or(u64::MAX);
    let mut live = list_pending(dirs);
    // Expired first, then oldest-survivors until the new invite fits.
    live.retain(|(_, p)| now_ms.saturating_sub(p.received_ms) <= PENDING_TTL_MS);
    while live.len() >= PENDING_MAX {
        let (path, _) = live.remove(0);
        let _ = std::fs::remove_file(&path);
    }
    // Same-millisecond arrivals are real (two agents, one sweep): give
    // the newcomer a distinct stem instead of overwriting its peer.
    let mut seq = 0u32;
    let mut path = dir.join(format!("{}.json", pending.received_ms));
    while path.exists() {
        seq += 1;
        path = dir.join(format!("{}-{seq}.json", pending.received_ms));
    }
    let body =
        serde_json::to_string_pretty(pending).map_err(|e| std::io::Error::other(e.to_string()))?;
    std::fs::write(&path, body)
        .and(Ok(path.clone()))
        .inspect_err(|_e| {
            let _ = std::fs::remove_file(&path);
        })
}

/// Every queued invitation, oldest first.
#[must_use]
pub fn list_pending(dirs: &BlazarDirs) -> Vec<(PathBuf, PendingInvite)> {
    let Ok(entries) = std::fs::read_dir(pending_dir(dirs)) else {
        return Vec::new();
    };
    let mut out: Vec<(PathBuf, PendingInvite)> = entries
        .flatten()
        .filter_map(|e| {
            let path = e.path();
            let body = std::fs::read_to_string(&path).ok()?;
            let pending = serde_json::from_str(&body).ok()?;
            Some((path, pending))
        })
        .collect();
    out.sort_by_key(|(_, p)| p.received_ms);
    out
}

/// Owner said yes: drop the queue entry and hand the cascade its
/// approval. The join credential is the invite's own code when it
/// carried one (one-use: the gateway redeems it at join, then the
/// worker's minted session token takes over); a code-less invite
/// falls back to this box's configured `fleet_token`.
pub fn approve_pending(
    path: &std::path::Path,
    fleet_token: Option<String>,
) -> std::io::Result<AgentApproval> {
    let body = std::fs::read_to_string(path)?;
    let pending: PendingInvite = serde_json::from_str(&body)
        .map_err(|e| std::io::Error::other(format!("unreadable pending invite: {e}")))?;
    std::fs::remove_file(path)?;
    let credential = pending
        .invite
        .code
        .clone()
        .filter(|c| !c.is_empty())
        .or(fleet_token);
    Ok(AgentApproval {
        gateway_base: pending.invite.gateway_base,
        fleet_token: credential,
    })
}

/// Owner said no: drop the queue entry.
pub fn deny_pending(dirs: &BlazarDirs, path: &std::path::Path) -> std::io::Result<()> {
    let _ = dirs;
    std::fs::remove_file(path)
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        // u128 millis overflow u64 only past year ~584M; saturate, never truncate
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

async fn handle_invite_conn(
    mut stream: TcpStream,
    policy: AgentPolicy,
    donor_token: String,
    dirs: &BlazarDirs,
    approved_tx: &mpsc::Sender<AgentApproval>,
) {
    // Cap before parse: an oversized frame is hostile or confused, and
    // either way it is not an invitation. The frame ends at the newline
    // delimiter: the adopter keeps its write side open while waiting for
    // the reply, so waiting for EOF would deadlock both peers.
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 1024];
    loop {
        match stream.read(&mut chunk).await {
            Ok(0) => break,
            Ok(n) => {
                if let Some(pos) = chunk[..n].iter().position(|&b| b == b'\n') {
                    buf.extend_from_slice(&chunk[..pos]);
                    break;
                }
                buf.extend_from_slice(&chunk[..n]);
                if buf.len() > INVITE_MAX_BYTES {
                    let reply = AgentReply::denied("invite payload too large");
                    let _ = stream
                        .write_all(serde_json::to_string(&reply).unwrap_or_default().as_bytes())
                        .await;
                    return;
                }
            }
            Err(_) => return,
        }
    }
    let Ok(invite) = serde_json::from_slice::<AgentInvite>(&buf) else {
        let reply = AgentReply::denied("payload was not a valid invite");
        let _ = stream
            .write_all(serde_json::to_string(&reply).unwrap_or_default().as_bytes())
            .await;
        return;
    };
    // A gateway base with no host part cannot be joined; refuse before
    // anything is queued.
    if !invite.gateway_base.starts_with("http://") && !invite.gateway_base.starts_with("https://") {
        let reply = AgentReply::denied("gateway_base must be an http(s) URL");
        let _ = stream
            .write_all(serde_json::to_string(&reply).unwrap_or_default().as_bytes())
            .await;
        return;
    }
    let (reply, approval) = decide_invite(policy, &donor_token, &invite);
    if reply.status == "queued"
        && let Err(e) = save_pending(
            dirs,
            &PendingInvite {
                invite,
                received_ms: now_ms(),
            },
        )
    {
        tracing::warn!("could not queue invite: {e}");
    }
    let _ = stream
        .write_all(serde_json::to_string(&reply).unwrap_or_default().as_bytes())
        .await;
    if let Some(approval) = approval {
        // Send after the reply: the gateway learns the verdict first,
        // the local cascade starts second.
        let _ = approved_tx.send(approval).await;
    }
}

/// Read one beacon UDP frame into `buf` and reply if it matches the
/// magic. Split out so the loop stays readable.
async fn answer_beacon(
    socket: &UdpSocket,
    len: usize,
    peer: SocketAddr,
    buf: &[u8],
    payload: &str,
    window: &Mutex<std::time::Instant>,
    replies: &AtomicU64,
) {
    if &buf[..len] != ADOPT_MAGIC {
        return;
    }
    {
        let mut w = window.lock().expect("adopt window lock");
        if w.elapsed() >= Duration::from_secs(1) {
            *w = std::time::Instant::now();
            replies.store(0, Ordering::Relaxed);
        }
    }
    if replies.fetch_add(1, Ordering::Relaxed) >= REPLY_BUDGET_PER_SEC {
        return;
    }
    let _ = socket.send_to(payload.as_bytes(), peer).await;
}

/// One donor box seen by `blazar adopt`. `ip` is the address the
/// beacon reply came FROM - the same address the invite goes TO.
#[derive(Debug, Clone)]
pub struct AgentSighting {
    pub ip: std::net::IpAddr,
    pub hostname: String,
    pub gpu_name: Option<String>,
    pub free_vram_mib: u64,
    pub policy: String,
}

/// Gateway side of adoption: broadcast + same-host unicast the adopt
/// magic and collect every agent that answers within `wait`, deduped
/// by source address. Same shape as `discovery::seek_gateways`.
pub async fn scan_agents(port: u16, wait: Duration) -> std::io::Result<Vec<AgentSighting>> {
    let socket = tokio::net::UdpSocket::bind(("0.0.0.0", 0)).await?;
    socket.set_broadcast(true)?;
    socket
        .send_to(ADOPT_MAGIC, ("255.255.255.255", port))
        .await
        .ok();
    socket.send_to(ADOPT_MAGIC, ("127.0.0.1", port)).await.ok();
    let mut seen: Vec<std::net::SocketAddr> = Vec::new();
    let mut found: Vec<AgentSighting> = Vec::new();
    let deadline = tokio::time::Instant::now() + wait;
    loop {
        if tokio::time::Instant::now() >= deadline {
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
                let Some(hostname) = v.get("hostname").and_then(|h| h.as_str()) else {
                    continue;
                };
                let gpu_name = v
                    .get("gpu_name")
                    .and_then(|g| g.as_str())
                    .filter(|g| !g.is_empty())
                    .map(str::to_owned);
                let free_vram_mib = v
                    .get("free_vram_mib")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                let policy = v
                    .get("policy")
                    .and_then(|p| p.as_str())
                    .unwrap_or("ask-owner")
                    .to_owned();
                if found.iter().any(|s| s.ip == peer.ip()) {
                    continue;
                }
                found.push(AgentSighting {
                    ip: peer.ip(),
                    hostname: hostname.to_owned(),
                    gpu_name,
                    free_vram_mib,
                    policy,
                });
            }
            _ => return Ok(found),
        }
    }
}

/// The agent main loop: beacon + invite listener on `agent_port`.
/// Approvals flow to `approved_tx` (the CLI cascade); every other
/// condition teaches and continues. Returns on `shutdown`.
#[allow(clippy::too_many_arguments)]
pub async fn run_agent(
    config: Arc<Config>,
    dirs: BlazarDirs,
    policy: AgentPolicy,
    gpu_name: Option<String>,
    free_vram_mib: u64,
    approved_tx: mpsc::Sender<AgentApproval>,
    shutdown: tokio::sync::watch::Receiver<bool>,
) -> std::io::Result<()> {
    let port = config.agent_port;
    let udp = match UdpSocket::bind(("0.0.0.0", port)).await {
        Ok(s) => Some(s),
        Err(e) => {
            tracing::warn!(
                "adopt beacon not started on udp/{port}: {e} - invites still work over tcp/{port}"
            );
            None
        }
    };
    let tcp = TcpListener::bind(("0.0.0.0", port)).await.map_err(|e| {
        std::io::Error::other(format!(
            "agent invite listener tcp/{port} is unavailable: {e} - another agent, or a port conflict; fix agent_port in the config"
        ))
    })?;
    tracing::info!("agent listening on tcp+udp/{port} (policy {config:?})");
    let beacon = beacon_payload(policy, gpu_name.as_deref(), free_vram_mib);
    let window = Mutex::new(std::time::Instant::now());
    let replies = AtomicU64::new(0);
    let mut udp_buf = [0u8; 64];
    let donor_token = config.fleet_token.clone();
    loop {
        let mut shutdown_rx = shutdown.clone();
        tokio::select! {
            _ = shutdown_rx.changed() => {
                tracing::info!("agent shutting down");
                return Ok(());
            }
            res = tcp.accept() => {
                let Ok((stream, _peer)) = res else { continue };
                let tx = approved_tx.clone();
                let dirs = dirs.clone();
                let token = donor_token.clone();
                tokio::spawn(async move {
                    handle_invite_conn(stream, policy, token, &dirs, &tx).await;
                });
            }
            res = async {
                match udp.as_ref() {
                    Some(s) => s.recv_from(&mut udp_buf).await,
                    // No beacon socket: park forever so select is not
                    // spinning on a None.
                    None => std::future::pending().await,
                }
            } => {
                let Ok((len, peer)) = res else { continue };
                if let Some(s) = udp.as_ref() {
                    answer_beacon(s, len, peer, &udp_buf, &beacon, &window, &replies).await;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blazar_core::BlazarDirs;

    fn dirs() -> (BlazarDirs, tempfile::TempDir) {
        let root = tempfile::TempDir::new().unwrap();
        (
            BlazarDirs {
                config_dir: root.path().join("cfg"),
                data_dir: root.path().join("data"),
            },
            root,
        )
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__beacon_payload__lists_hardware_and_policy_only() {
        let body = beacon_payload(AgentPolicy::AskOwner, Some("RTX 4070"), 7_100);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "agent_version",
                "free_vram_mib",
                "gpu_name",
                "hostname",
                "policy"
            ]
        );
        assert_eq!(v["policy"], "ask-owner");
        assert_eq!(v["free_vram_mib"], 7_100);
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__decide_invite__token_policy_matrix() {
        let invite = AgentInvite {
            gateway_name: "gw".into(),
            gateway_base: "http://127.0.0.1:11438".into(),
            code: Some("sekrit".into()),
        };
        let (reply, approval) = decide_invite(AgentPolicy::Token, "sekrit", &invite);
        assert_eq!(reply.status, "approved");
        assert_eq!(approval.unwrap().fleet_token.as_deref(), Some("sekrit"));

        let (reply, approval) = decide_invite(AgentPolicy::Token, "other", &invite);
        assert_eq!(reply.status, "denied");
        assert!(approval.is_none());

        let no_token = AgentInvite {
            code: None,
            ..invite.clone()
        };
        let (reply, _) = decide_invite(AgentPolicy::Token, "sekrit", &no_token);
        assert_eq!(reply.status, "denied");

        let (reply, _) = decide_invite(AgentPolicy::Token, "", &invite);
        assert_eq!(reply.status, "denied");
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__decide_invite__ask_owner_queues_for_owner() {
        let invite = AgentInvite {
            gateway_name: "gw".into(),
            gateway_base: "http://127.0.0.1:11438".into(),
            code: Some("anything".into()),
        };
        let (reply, approval) = decide_invite(AgentPolicy::AskOwner, "sekrit", &invite);
        assert_eq!(reply.status, "queued");
        assert!(approval.is_none(), "ask-owner NEVER auto-approves");
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__pending_store__queue_list_approve_roundtrip() {
        let (d, _root) = dirs();
        let invite = AgentInvite {
            gateway_name: "gw".into(),
            gateway_base: "http://10.0.0.2:11435".into(),
            code: None,
        };
        let path = save_pending(
            &d,
            &PendingInvite {
                invite: invite.clone(),
                received_ms: 42,
            },
        )
        .unwrap();
        let listed = list_pending(&d);
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].1.invite.gateway_base, invite.gateway_base);
        let approval = approve_pending(&path, Some("sekrit".into())).unwrap();
        assert_eq!(approval.gateway_base, "http://10.0.0.2:11435");
        assert_eq!(approval.fleet_token.as_deref(), Some("sekrit"));
        assert!(list_pending(&d).is_empty(), "approve consumes the entry");

        // Deny path consumes too.
        let path = save_pending(
            &d,
            &PendingInvite {
                invite,
                received_ms: 43,
            },
        )
        .unwrap();
        deny_pending(&d, &path).unwrap();
        assert!(list_pending(&d).is_empty());
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__pending_store__code_bearing_invite_approves_with_the_code() {
        // Ask-owner approval must hand the cascade a credential the
        // gateway will actually accept: the invite's one-use code, not
        // this donor's (unrelated) fleet_token fallback.
        let root = tempfile::TempDir::new().unwrap();
        let d = BlazarDirs {
            config_dir: root.path().join("cfg"),
            data_dir: root.path().join("data"),
        };
        let path = save_pending(
            &d,
            &PendingInvite {
                invite: AgentInvite {
                    gateway_name: "gw".into(),
                    gateway_base: "http://10.0.0.9:11435".into(),
                    code: Some("abcdef0123456789".into()),
                },
                received_ms: 7,
            },
        )
        .unwrap();
        let approval = approve_pending(&path, Some("donor-master-tok".into())).unwrap();
        assert_eq!(
            approval.fleet_token.as_deref(),
            Some("abcdef0123456789"),
            "invite code wins over the donor's own token"
        );
        assert_eq!(approval.gateway_base, "http://10.0.0.9:11435");
    }

    #[tokio::test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    async fn unit__invite_conn__oversized_frame_refused_before_parse() {
        let (d, _root) = dirs();
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, _rx) = mpsc::channel(1);
        let dirs_clone = d.clone();
        let task = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            handle_invite_conn(
                stream,
                AgentPolicy::AskOwner,
                String::new(),
                &dirs_clone,
                &tx,
            )
            .await;
        });
        let mut client = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        client
            .write_all(&vec![b'a'; INVITE_MAX_BYTES + 1])
            .await
            .unwrap();
        client.shutdown().await.unwrap();
        task.await.unwrap();
        assert!(
            list_pending(&d).is_empty(),
            "nothing queued from a hostile frame"
        );
    }

    #[tokio::test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    async fn unit__invite_conn__newline_terminated_invite_answers_without_eof() {
        // The adopter keeps its write side OPEN while waiting for the reply
        // (it never half-closes), so the agent must decide on the newline
        // delimiter, not on EOF - waiting for EOF deadlocks both peers.
        let (d, _root) = dirs();
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, _rx) = mpsc::channel(1);
        let dirs_clone = d.clone();
        let task = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            handle_invite_conn(
                stream,
                AgentPolicy::AskOwner,
                String::new(),
                &dirs_clone,
                &tx,
            )
            .await;
        });
        let mut client = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        let invite = serde_json::to_string(&AgentInvite {
            gateway_name: "gateway".into(),
            gateway_base: "http://127.0.0.1:11443".into(),
            code: None,
        })
        .unwrap();
        client
            .write_all(format!("{invite}\n").as_bytes())
            .await
            .unwrap();
        // NO shutdown: reply must arrive while our write side stays open.
        let reply = tokio::time::timeout(Duration::from_secs(3), async {
            let mut buf = Vec::new();
            client.read_to_end(&mut buf).await.map(|_| buf)
        })
        .await
        .expect("reply must not wait for the client to half-close")
        .unwrap();
        task.await.unwrap();
        let reply: AgentReply = serde_json::from_slice(&reply).unwrap();
        assert_eq!(reply.status, "queued");
        assert_eq!(list_pending(&d).len(), 1, "ask-owner queues the invite");
    }
}
