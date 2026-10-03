//! DNS-rebinding guard for the browser-facing surface.
//!
//! Loopback-bound servers are reachable from any web page the local user
//! visits: a public site can rebind a DNS name to `127.0.0.1` and issue
//! requests that look locally originated (the ollama CVE-2024-28224
//! class). The middleware in this module rejects requests whose `Host`
//! header cannot belong to a local caller, which breaks the rebinding
//! trick: a rebound name arrives carrying the attacker's host, not ours.
//!
//! Scope follows ollama's semantics: the guard is armed only when the
//! gateway itself is bound to loopback. A non-loopback bind is a
//! deliberate LAN/remote exposure decision made by the operator, and
//! remote callers legitimately address the machine by other names.

use axum::extract::State;
use axum::http::Request;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

/// Paths that must answer regardless of `Host`: orchestrators and
/// container health probes address the daemon by network-internal names.
const ALWAYS_OPEN_PATHS: [&str; 2] = ["/healthz", "/health"];

/// Decide whether the guard applies, from the configured bind host.
///
/// Loopback IPs and the literal name `localhost` arm the guard; wildcard
/// (`0.0.0.0`, `::`) and LAN addresses are deliberate broader exposure
/// and skip it, matching ollama's behavior.
#[must_use]
pub fn guard_scope(bind_host: &str) -> bool {
    match bind_host
        .trim_matches(['[', ']'])
        .parse::<std::net::IpAddr>()
    {
        Ok(ip) => ip.is_loopback(),
        Err(_) => bind_host.eq_ignore_ascii_case("localhost"),
    }
}

/// Strip an optional port from a `Host`/origin authority, IPv6-bracket
/// aware: `127.0.0.1:11435` -> `127.0.0.1`, `[::1]:80` -> `::1`.
fn host_without_port(host: &str) -> &str {
    if let Some(rest) = host.strip_prefix('[') {
        return rest.split(']').next().unwrap_or(rest);
    }
    // A bare (unbracketed) IPv6 literal carries several colons and no
    // port; `host:port` has exactly one.
    if host.matches(':').count() != 1 {
        return host;
    }
    match host.rfind(':') {
        Some(i)
            if !host[i + 1..].is_empty() && host[i + 1..].chars().all(|c| c.is_ascii_digit()) =>
        {
            &host[..i]
        }
        _ => host,
    }
}

/// Is this `Host` value one a local caller could legitimately send?
///
/// Allowed: loopback/private/link-local/unspecified IP literals (the
/// LAN ranges cover port-forwarded and docker-bridge setups where the
/// request still originates on this machine), `localhost` and its
/// subdomains, `.local`/`.internal` names, and this machine's own
/// hostname.
#[must_use]
pub fn host_is_allowed(host: &str) -> bool {
    let bare = host_without_port(host.trim());
    if bare.is_empty() {
        return false;
    }
    if let Ok(ip) = bare.parse::<std::net::IpAddr>() {
        return match ip {
            std::net::IpAddr::V4(v4) => {
                v4.is_loopback() || v4.is_private() || v4.is_link_local() || v4.is_unspecified()
            }
            std::net::IpAddr::V6(v6) => {
                let seg0 = v6.segments()[0];
                let unique_local = (seg0 & 0xfe00) == 0xfc00; // fc00::/7
                let link_local = (seg0 & 0xffc0) == 0xfe80; // fe80::/10
                v6.is_loopback() || v6.is_unspecified() || unique_local || link_local
            }
        };
    }
    let lower = bare.to_ascii_lowercase();
    if lower == "localhost" {
        return true;
    }
    for suffix in [".localhost", ".local", ".internal"] {
        if lower.ends_with(suffix) {
            return true;
        }
    }
    machine_hostnames()
        .iter()
        .any(|h| h.eq_ignore_ascii_case(&lower))
}

/// The local machine's hostname(s), best effort: the process environment
/// (`HOSTNAME` on unix daemons is often unset; `COMPUTERNAME` is the
/// Windows convention) plus the short form before any domain component.
fn machine_hostnames() -> Vec<String> {
    let raw = std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_default();
    if raw.is_empty() {
        return Vec::new();
    }
    let mut names = vec![raw.clone()];
    if let Some(short) = raw.split('.').next() {
        names.push(short.to_string());
    }
    names
}

/// Is this `Origin` one the default browser policy should serve?
///
/// The default set keeps local web UIs zero-config — any port on
/// `localhost`, `127.0.0.1`, `[::1]`, `0.0.0.0` — plus the custom
/// schemes used by desktop/webview shells (`app://`, `file://`,
/// `tauri://`, `vscode-webview://`, `vscode-file://`). Everything else
/// (public websites) needs an explicit `cors_origins` entry.
#[must_use]
pub fn is_default_browser_origin(origin: &str) -> bool {
    let trimmed = origin.trim();
    if let Some(rest) = trimmed.strip_prefix("https://") {
        authority_is_default_host(rest)
    } else if let Some(rest) = trimmed.strip_prefix("http://") {
        authority_is_default_host(rest)
    } else {
        matches!(
            trimmed,
            "app://*" | "file://*" | "tauri://*" | "vscode-webview://*" | "vscode-file://*"
        ) || trimmed.starts_with("app://")
            || trimmed.starts_with("file://")
            || trimmed.starts_with("tauri://")
            || trimmed.starts_with("vscode-webview://")
            || trimmed.starts_with("vscode-file://")
    }
}

fn authority_is_default_host(authority: &str) -> bool {
    let host = host_without_port(authority.trim_end_matches('/'));
    matches!(
        host.to_ascii_lowercase().as_str(),
        "localhost" | "127.0.0.1" | "::1" | "0.0.0.0"
    )
}

/// Middleware body: reject requests whose `Host` cannot be local when
/// the guard is armed. Preflights (`OPTIONS`) must reach the CORS layer
/// to stay answerable, and the health paths stay open.
pub async fn host_guard_mw(
    State((armed,)): State<(bool,)>,
    req: Request<axum::body::Body>,
    next: Next,
) -> Response {
    if !armed
        || req.method() == axum::http::Method::OPTIONS
        || ALWAYS_OPEN_PATHS.contains(&req.uri().path())
    {
        return next.run(req).await;
    }
    let host = req
        .headers()
        .get(axum::http::header::HOST)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    if host_is_allowed(host) {
        return next.run(req).await;
    }
    tracing::warn!("host guard: rejected Host {host:?} on loopback bind (possible DNS rebinding)");
    (
        axum::http::StatusCode::FORBIDDEN,
        axum::Json(serde_json::json!({"error": {"message": format!("unrecognized Host header {host:?} for a loopback server (possible DNS rebinding); bind blazar to your LAN address or set host accordingly"), "type": "blazar_error", "code": 403}})),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    // Test names use the project's unit__area__behavior convention.
    #![allow(non_snake_case)]

    use super::*;

    #[test]
    fn unit__host_without_port__strips_ports_bracket_aware() {
        assert_eq!(host_without_port("127.0.0.1:11435"), "127.0.0.1");
        assert_eq!(host_without_port("[::1]:8080"), "::1");
        assert_eq!(host_without_port("[::1]"), "::1");
        assert_eq!(host_without_port("localhost:3000"), "localhost");
        assert_eq!(host_without_port("::1"), "::1");
        assert_eq!(host_without_port("example.com"), "example.com");
    }

    #[test]
    fn unit__host_is_allowed__accepts_local_ip_literals() {
        for good in [
            "127.0.0.1",
            "127.0.0.1:11435",
            "[::1]:80",
            "::1",
            "0.0.0.0",
            "10.0.0.5",
            "192.168.1.20:8080",
            "172.16.0.1",
            "169.254.7.7",
            "fd00::1",
            "fe80::1",
        ] {
            assert!(host_is_allowed(good), "should allow {good}");
        }
    }

    #[test]
    fn unit__host_is_allowed__accepts_local_names() {
        for good in [
            "localhost",
            "LOCALHOST:11435",
            "app.localhost",
            "mybox.local",
            "scanner.internal",
        ] {
            assert!(host_is_allowed(good), "should allow {good}");
        }
    }

    #[test]
    fn unit__host_is_allowed__rejects_public_hosts_and_empty() {
        for bad in [
            "",
            "evil.com",
            "EVIL.com:443",
            "sub.evil.com",
            "127.0.0.1.evil.com",
            "8.8.8.8",
            "2001:db8::1",
        ] {
            assert!(!host_is_allowed(bad), "should reject {bad}");
        }
    }

    #[test]
    fn unit__guard_scope__arms_only_for_loopback_binds() {
        assert!(guard_scope("127.0.0.1"));
        assert!(guard_scope("::1"));
        assert!(guard_scope("localhost"));
        assert!(!guard_scope("0.0.0.0"));
        assert!(!guard_scope("192.168.1.10"));
        assert!(!guard_scope("mybox.local"));
    }

    #[test]
    fn unit__is_default_browser_origin__allows_localhost_any_port_and_webviews() {
        for good in [
            "http://localhost:3000",
            "https://localhost",
            "http://127.0.0.1:8080",
            "http://[::1]:5173",
            "https://0.0.0.0:9000",
            "app://blazar",
            "file://",
            "tauri://localhost",
            "vscode-webview://16c0b4c0-1f0f-1f0f-1f0f-16c0b4c0a1b2",
            "vscode-file://vscode-app",
        ] {
            assert!(is_default_browser_origin(good), "should allow {good}");
        }
    }

    #[test]
    fn unit__is_default_browser_origin__rejects_public_sites() {
        for bad in [
            "https://evil.com",
            "http://evil.com:8080",
            "https://sub.localhost.evil.com",
            "chrome-extension://abc",
            "null",
            "",
        ] {
            assert!(!is_default_browser_origin(bad), "should reject {bad}");
        }
    }
}
