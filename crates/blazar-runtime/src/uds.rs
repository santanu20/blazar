//! Unix-domain-socket child transport (`child_transport = "unix"`).
//!
//! llama-server (b11147+, #28690) accepts `.sock` paths via `--host`;
//! the gateway dials children over the socket with reqwest's
//! `unix_socket` connector (one client per socket path — the connector
//! pins a single path per client). TCP remains the default and
//! byte-identical; unix is opt-in per daemon.
//!
//! Socket files live in `run_dir()` next to the pidfiles and die with
//! their instance (evict unlinks under the same ptr-identity guard as
//! the pidfile/apikey teardown); a boot sweep reaps sockets orphaned by
//! an unclean daemon exit.

use std::collections::HashMap;
use std::path::PathBuf;

/// Filesystem-socket path budget. `sockaddr_un.sun_path` is 108 bytes
/// on Linux (104 on macOS); 100 leaves headroom for the longest common
/// kernel limit so a spawn fails with a teaching error at config time
/// instead of a runtime `bind(ENGLISH)`-style failure in the child.
pub const MAX_SOCKET_PATH_BYTES: usize = 100;

/// Bound on cached per-socket clients. One entry per live unix child;
/// the cap only guards a leak (reaped children must not accumulate
/// entries). On overflow the cache clears — connections rebuild lazily.
const MAX_CACHED_CLIENTS: usize = 32;

/// Reject socket paths that cannot fit `sun_path` at spawn time.
pub fn validate_socket_path(socket: &str) -> Result<(), String> {
    #[cfg(unix)]
    {
        let bytes = socket.len();
        if bytes > MAX_SOCKET_PATH_BYTES {
            return Err(format!(
                "unix socket path is {bytes} bytes, over the {MAX_SOCKET_PATH_BYTES}-byte \
                 sockaddr_un.sun_path budget — move the blazar data dir shallower or use \
                 child_transport = \"tcp\""
            ));
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        let _ = socket;
        Err("child_transport = \"unix\" requires a unix platform".to_string())
    }
}

/// Build one client pinned to `socket`. All requests through it dial
/// the socket regardless of URL host; the URL scheme still decides
/// plain-vs-TLS, so bases stay `http://…`.
#[must_use]
pub fn client(socket: &str) -> reqwest::Client {
    #[cfg(unix)]
    {
        reqwest::Client::builder()
            .unix_socket(PathBuf::from(socket))
            .build()
            .unwrap_or_default()
    }
    #[cfg(not(unix))]
    {
        // Unreachable in practice: config validation rejects
        // child_transport = "unix" on non-unix platforms before any
        // endpoint can carry a socket. Loud, not silent: a default
        // client cannot dial the socket and requests fail visibly.
        tracing::error!(socket, "unix sockets unavailable on this platform");
        reqwest::Client::new()
    }
}

/// Bounded cache of per-socket clients for hot lanes (the proxy dials
/// the same child repeatedly; a fresh client per request would pay
/// pool setup each time). Cold lanes (bank save/restore, warm peg,
/// health probes) use [`client`] directly — they run once per spawn.
#[derive(Default)]
pub struct UdsClients {
    inner: std::sync::Mutex<HashMap<String, reqwest::Client>>,
}

impl UdsClients {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get-or-build the client for `socket`.
    pub fn get(&self, socket: &str) -> reqwest::Client {
        let mut guard = self.inner.lock().expect("uds client cache lock");
        if guard.len() >= MAX_CACHED_CLIENTS && !guard.contains_key(socket) {
            tracing::warn!(
                cached = guard.len(),
                "unix client cache over budget — clearing (leaked child sockets?)"
            );
            guard.clear();
        }
        guard
            .entry(socket.to_string())
            .or_insert_with(|| client(socket))
            .clone()
    }
}

/// Reap sockets orphaned by an unclean daemon exit: a `.sock` in the
/// run dir that no process accepts connections on is stale by
/// definition (a live daemon's children accept). Runs once at daemon
/// start, before any spawn reuses the deterministic per-name path.
pub fn sweep_stale(run_dir: &std::path::Path) {
    #[cfg(unix)]
    {
        let Ok(entries) = std::fs::read_dir(run_dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_none_or(|e| e != "sock") {
                continue;
            }
            let stale = std::os::unix::net::UnixStream::connect(&path).is_err();
            if stale {
                match std::fs::remove_file(&path) {
                    Ok(()) => tracing::info!(
                        socket = %path.display(),
                        "swept stale child socket from previous run"
                    ),
                    Err(e) => tracing::warn!(
                        socket = %path.display(),
                        error = %e,
                        "failed to remove stale child socket"
                    ),
                }
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = run_dir;
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn unit__validate_socket_path__short_path_accepted_long_rejected() {
        assert!(validate_socket_path("/run/blazar/llava.sock").is_ok());
        let long = format!("/tmp/{}.sock", "x".repeat(200));
        let err = validate_socket_path(&long).unwrap_err();
        assert!(err.contains("sun_path"), "teaching error: {err}");
    }

    #[test]
    fn unit__client__cache_dedupes_per_socket() {
        let cache = UdsClients::new();
        let _ = cache.get("/tmp/blazar-test-a.sock");
        let _ = cache.get("/tmp/blazar-test-a.sock");
        let _ = cache.get("/tmp/blazar-test-b.sock");
        let guard = cache.inner.lock().expect("cache lock");
        assert_eq!(guard.len(), 2, "one entry per distinct socket path");
    }

    #[test]
    fn unit__sweep_stale__removes_dead_socket_keeps_live() {
        let tmp = tempfile::tempdir().unwrap();
        let dead = tmp.path().join("dead.sock");
        let live = tmp.path().join("live.sock");
        let _listener = std::os::unix::net::UnixListener::bind(&live).unwrap();
        std::fs::write(&dead, b"").unwrap();
        sweep_stale(tmp.path());
        assert!(!dead.exists(), "dead socket swept");
        assert!(live.exists(), "live socket kept");
    }
}
