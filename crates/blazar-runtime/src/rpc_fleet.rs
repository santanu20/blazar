//! Read-only ggml-RPC wire client for fleet introspection.
//!
//! Speaks just enough of the worker protocol (upstream
//! `tools/rpc/rpc-server.cpp`, protocol v7 at engine b11429) to answer
//! "which workers are out there, and how much memory do they have
//! free": the mandatory HELLO handshake, `RPC_CMD_DEVICE_COUNT` and
//! per-device `RPC_CMD_GET_DEVICE_MEMORY`. Nothing here allocates
//! buffers or ships tensors — a status probe must be safe to point at
//! any worker, including a hostile one.
//!
//! Wire framing (little-endian; upstream memcpy's native structs, and
//! every supported deployment target is little-endian):
//!
//! ```text
//! request  = u8 cmd || u64 input_len || input bytes
//! response = u64 output_len || output bytes
//! ```
//!
//! Every connection MUST open with HELLO — the server drops any
//! client whose first command is something else. On any later
//! protocol error the server closes the socket WITHOUT a reply, so an
//! EOF mid-conversation is a distinct failure from a timeout.
//!
//! Byte-level constants are pinned by `unit__query_worker__*` tests
//! against the documented layout; the live oracle is the installed
//! engine's own ggml-rpc-server (validated per release wave).

use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// `RPC_CMD_GET_DEVICE_MEMORY` (enum position 11 at protocol v7).
const CMD_GET_DEVICE_MEMORY: u8 = 11;
/// `RPC_CMD_HELLO` — upstream `static_assert` pins it at 14 forever.
const CMD_HELLO: u8 = 14;
/// `RPC_CMD_DEVICE_COUNT` (enum position 15 at protocol v7).
const CMD_DEVICE_COUNT: u8 = 15;

/// Protocol family this client speaks. A worker answering with a
/// different MAJOR (or a higher MINOR) is refused with a teaching
/// error instead of silently misread fields.
const PROTO_MAJOR: u8 = 7;
const PROTO_MINOR: u8 = 0;

/// HELLO request: `conn_caps[24]`, all zero = plain TCP, none of the
/// optional transport accelerations negotiated.
const HELLO_REQ_LEN: usize = 24;
/// HELLO response: `{u8 major, u8 minor, u8 patch, u8 pad, caps[24]}`.
const HELLO_RSP_LEN: usize = 28;
const DEVICE_COUNT_RSP_LEN: usize = 4;
const DEVICE_MEM_REQ_LEN: usize = 4;
const DEVICE_MEM_RSP_LEN: usize = 16;

/// Connect budget, matching the spawn preflight's 2s TCP probe so
/// `rpc status` and `doctor` agree on what "unreachable" means.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);
/// Whole-conversation budget once connected: a worker that accepts but
/// never answers must not hang the CLI. Generous enough for a
/// many-device box queried sequentially over a slow link.
const CONVERSATION_TIMEOUT: Duration = Duration::from_secs(10);

/// A device-count reply above this is treated as a protocol fault,
/// not a fleet — the per-device loop would otherwise amplify a
/// misdecoded length into thousands of bogus queries.
const MAX_SANITY_DEVICES: u32 = 64;

/// One worker device as the fleet sees it. Byte counts, not MiB:
/// callers format (and the JSON surface rounds) at the edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RpcDeviceMemory {
    pub index: u32,
    pub free_bytes: u64,
    pub total_bytes: u64,
}

/// What a healthy worker told us: its protocol version and one entry
/// per exposed device (GPU(s), plus the CPU device on CPU-only
/// workers).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RpcWorkerInfo {
    pub proto_major: u8,
    pub proto_minor: u8,
    pub proto_patch: u8,
    pub devices: Vec<RpcDeviceMemory>,
}

/// Query one `host:port` worker: HELLO, count devices, fetch each
/// device's free/total memory. The connection is closed by drop after
/// the last reply — every command here is read-only, so there is
/// nothing to drain.
///
/// Errors are user-facing sentences (endpoint-prefixed, same shape as
/// the TCP preflight's `host:port (reason)` strings).
pub async fn query_worker(endpoint: &str) -> Result<RpcWorkerInfo, String> {
    let (host, port) = match endpoint.rsplit_once(':') {
        Some((h, p)) if !h.is_empty() => match p.parse::<u16>() {
            Ok(p) => (h.to_string(), p),
            Err(_) => return Err(format!("{endpoint} (malformed port)")),
        },
        _ => return Err(format!("{endpoint} (malformed host:port)")),
    };
    tokio::time::timeout(CONVERSATION_TIMEOUT, async {
        let sock = tokio::time::timeout(CONNECT_TIMEOUT, TcpStream::connect((host.as_str(), port)))
            .await
            .map_err(|_| format!("{endpoint} (unreachable: no answer within 2s)"))?
            .map_err(|e| format!("{endpoint} (unreachable: {e})"))?;
        query_worker_on(endpoint, sock).await
    })
    .await
    .map_err(|_| format!("{endpoint} (no reply within 10s — worker accepted but went silent)"))?
}

/// Conversation body once connected: HELLO → device count → memory.
async fn query_worker_on(endpoint: &str, sock: TcpStream) -> Result<RpcWorkerInfo, String> {
    let (mut r, mut w) = sock.into_split();

    send_cmd(&mut w, CMD_HELLO, &[0u8; HELLO_REQ_LEN])
        .await
        .map_err(|e| format!("{endpoint} (send failed: {e})"))?;
    let hello = recv_rsp(endpoint, &mut r, HELLO_RSP_LEN, "HELLO reply").await?;
    let (proto_major, proto_minor, proto_patch) = (hello[0], hello[1], hello[2]);
    if proto_major != PROTO_MAJOR || proto_minor > PROTO_MINOR {
        return Err(format!(
            "{endpoint} (worker speaks rpc protocol v{proto_major}.{proto_minor}.{proto_patch}, \
             this blazar speaks v{PROTO_MAJOR}.{PROTO_MINOR}.0 — update the engine lane \
             (`blazar engine update`) or point at a v{PROTO_MAJOR}.x worker)"
        ));
    }

    send_cmd(&mut w, CMD_DEVICE_COUNT, &[])
        .await
        .map_err(|e| format!("{endpoint} (send failed: {e})"))?;
    let count_rsp = recv_rsp(endpoint, &mut r, DEVICE_COUNT_RSP_LEN, "device count").await?;
    let count = u32::from_le_bytes(count_rsp[..4].try_into().unwrap());
    if count > MAX_SANITY_DEVICES {
        return Err(format!(
            "{endpoint} (worker reported {count} devices — protocol fault, expected <= {MAX_SANITY_DEVICES})"
        ));
    }

    let mut devices = Vec::with_capacity(count as usize);
    for index in 0..count {
        send_cmd(
            &mut w,
            CMD_GET_DEVICE_MEMORY,
            &index.to_le_bytes()[..DEVICE_MEM_REQ_LEN],
        )
        .await
        .map_err(|e| format!("{endpoint} (send failed: {e})"))?;
        let mem = recv_rsp(endpoint, &mut r, DEVICE_MEM_RSP_LEN, "device memory").await?;
        devices.push(RpcDeviceMemory {
            index,
            free_bytes: u64::from_le_bytes(mem[..8].try_into().unwrap()),
            total_bytes: u64::from_le_bytes(mem[8..].try_into().unwrap()),
        });
    }
    Ok(RpcWorkerInfo {
        proto_major,
        proto_minor,
        proto_patch,
        devices,
    })
}

/// Frame and send one command.
async fn send_cmd(
    w: &mut tokio::net::tcp::OwnedWriteHalf,
    cmd: u8,
    input: &[u8],
) -> std::io::Result<()> {
    let mut buf = Vec::with_capacity(9 + input.len());
    buf.push(cmd);
    buf.extend_from_slice(&(input.len() as u64).to_le_bytes());
    buf.extend_from_slice(input);
    w.write_all(&buf).await
}

/// Read one framed reply. The announced length MUST equal `expected`:
/// a mismatch (or an absurd value) is a version fault, and the length
/// is never trusted for allocation — exactly `expected` bytes are
/// read, no more.
async fn recv_rsp(
    endpoint: &str,
    r: &mut tokio::net::tcp::OwnedReadHalf,
    expected: usize,
    what: &str,
) -> Result<Vec<u8>, String> {
    let mut len_buf = [0u8; 8];
    r.read_exact(&mut len_buf)
        .await
        .map_err(|_| format!("{endpoint} (worker closed the connection while awaiting {what})"))?;
    let announced = u64::from_le_bytes(len_buf);
    if announced != expected as u64 {
        return Err(format!(
            "{endpoint} ({what} announced {announced} bytes, expected {expected} — \
             rpc protocol version mismatch?)"
        ));
    }
    let mut body = vec![0u8; expected];
    r.read_exact(&mut body)
        .await
        .map_err(|_| format!("{endpoint} (worker closed the connection mid-{what})"))?;
    Ok(body)
}

/// Real-wire b11429 worker double, shared by the `rpc_fleet` and
/// supervisor test suites so every fleet-aware decision is pinned
/// against the actual protocol, not a mock of it (H7). Binds an
/// ephemeral loopback port; answers the three commands this crate
/// sends; `received` captures every request byte for framing pins.
#[cfg(test)]
pub(crate) mod test_support {
    use super::{CMD_DEVICE_COUNT, CMD_GET_DEVICE_MEMORY, CMD_HELLO};
    use std::sync::Arc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    pub(crate) struct FakeWorker {
        pub(crate) endpoint: String,
        pub(crate) received: Arc<std::sync::Mutex<Vec<u8>>>,
    }

    pub(crate) enum FakeBehavior {
        Normal,
        DropOnMemory,
    }

    /// Parameterized for fault injection: protocol version to
    /// advertise, per-device `(free, total)` memory in bytes, and
    /// whether to drop the connection mid-conversation.
    pub(crate) async fn spawn_fake(
        major: u8,
        minor: u8,
        patch: u8,
        devices: Vec<(u64, u64)>,
        behavior: FakeBehavior,
    ) -> FakeWorker {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let endpoint = format!("127.0.0.1:{}", listener.local_addr().unwrap().port());
        let received = Arc::new(std::sync::Mutex::new(Vec::new()));
        let rx = received.clone();
        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            loop {
                let mut cmd = [0u8; 1];
                if sock.read_exact(&mut cmd).await.is_err() {
                    return;
                }
                let mut len_buf = [0u8; 8];
                if sock.read_exact(&mut len_buf).await.is_err() {
                    return;
                }
                let len = usize::try_from(u64::from_le_bytes(len_buf)).unwrap();
                let mut input = vec![0u8; len];
                if sock.read_exact(&mut input).await.is_err() {
                    return;
                }
                {
                    let mut rx = rx.lock().unwrap();
                    rx.push(cmd[0]);
                    rx.extend_from_slice(&len_buf);
                    rx.extend_from_slice(&input);
                }
                match cmd[0] {
                    CMD_HELLO => {
                        let mut rsp = vec![major, minor, patch, 0u8];
                        rsp.extend_from_slice(&[0u8; 24]);
                        write_rsp(&mut sock, &rsp).await;
                    }
                    CMD_DEVICE_COUNT => {
                        write_rsp(
                            &mut sock,
                            &u32::try_from(devices.len()).unwrap().to_le_bytes(),
                        )
                        .await;
                    }
                    CMD_GET_DEVICE_MEMORY => {
                        let dev = u32::from_le_bytes(input[..4].try_into().unwrap()) as usize;
                        match behavior {
                            FakeBehavior::Normal => {
                                let (free, total) = devices[dev];
                                let mut rsp = Vec::with_capacity(16);
                                rsp.extend_from_slice(&free.to_le_bytes());
                                rsp.extend_from_slice(&total.to_le_bytes());
                                write_rsp(&mut sock, &rsp).await;
                            }
                            FakeBehavior::DropOnMemory => return,
                        }
                    }
                    _ => return,
                }
            }
        });
        FakeWorker { endpoint, received }
    }

    async fn write_rsp(sock: &mut TcpStream, body: &[u8]) {
        let mut rsp = (body.len() as u64).to_le_bytes().to_vec();
        rsp.extend_from_slice(body);
        let _ = sock.write_all(&rsp).await;
    }
}

#[cfg(test)]
#[allow(non_snake_case)] // repo convention: unit__scenario__expected
mod tests {
    use super::*;
    use crate::rpc_fleet::test_support::{FakeBehavior, spawn_fake};

    #[tokio::test]
    async fn unit__query_worker__hello_count_and_memory_roundtrip() {
        let fake = spawn_fake(
            7,
            0,
            0,
            vec![(5 << 20, 8 << 20), (1 << 20, 2 << 20)],
            FakeBehavior::Normal,
        )
        .await;
        let info = query_worker(&fake.endpoint).await.unwrap();
        assert_eq!(
            (info.proto_major, info.proto_minor, info.proto_patch),
            (7, 0, 0)
        );
        assert_eq!(
            info.devices,
            vec![
                RpcDeviceMemory {
                    index: 0,
                    free_bytes: 5 << 20,
                    total_bytes: 8 << 20
                },
                RpcDeviceMemory {
                    index: 1,
                    free_bytes: 1 << 20,
                    total_bytes: 2 << 20
                },
            ]
        );
    }

    #[tokio::test]
    async fn unit__query_worker__hello_request_bytes_pin_the_wire_format() {
        let fake = spawn_fake(7, 0, 0, vec![(1, 1)], FakeBehavior::Normal).await;
        query_worker(&fake.endpoint).await.unwrap();
        let rx = fake.received.lock().unwrap().clone();
        // First command on the wire: HELLO(14) || u64le(24) || 24 zero caps.
        let mut expect = vec![CMD_HELLO];
        expect.extend_from_slice(&24u64.to_le_bytes());
        expect.extend_from_slice(&[0u8; 24]);
        // Then DEVICE_COUNT(15) || u64le(0).
        expect.push(CMD_DEVICE_COUNT);
        expect.extend_from_slice(&0u64.to_le_bytes());
        // Then GET_DEVICE_MEMORY(11) || u64le(4) || u32le(0).
        expect.push(CMD_GET_DEVICE_MEMORY);
        expect.extend_from_slice(&4u64.to_le_bytes());
        expect.extend_from_slice(&0u32.to_le_bytes());
        assert_eq!(
            &rx[..expect.len()],
            &expect[..],
            "wire bytes drifted from protocol v7"
        );
    }

    #[tokio::test]
    async fn unit__query_worker__version_mismatch_teaches_instead_of_misreading() {
        let fake = spawn_fake(8, 1, 0, vec![(1, 1)], FakeBehavior::Normal).await;
        let err = query_worker(&fake.endpoint).await.unwrap_err();
        assert!(
            err.contains("v8.1.0"),
            "error names the worker version: {err}"
        );
        assert!(err.contains("v7.0"), "error names our version: {err}");
        assert!(
            err.contains("engine update"),
            "error carries the fix hint: {err}"
        );
    }

    #[tokio::test]
    async fn unit__query_worker__mid_conversation_close_is_distinct_from_hang() {
        let drop = spawn_fake(7, 0, 0, vec![(1, 1)], FakeBehavior::DropOnMemory).await;
        let err = query_worker(&drop.endpoint).await.unwrap_err();
        assert!(err.contains("closed the connection"), "{err}");
        // CONVERSATION_TIMEOUT is 10s — the test budget proves the EOF
        // path wins long before any timer fires.
    }

    #[tokio::test]
    async fn unit__query_worker__malformed_endpoints_reported_without_connect() {
        assert!(
            query_worker("nohost")
                .await
                .unwrap_err()
                .contains("malformed host:port")
        );
        assert!(
            query_worker("host:notaport")
                .await
                .unwrap_err()
                .contains("malformed port")
        );
    }
}
