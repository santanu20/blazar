//! Network-environment diagnostics: the egress truths that decide
//! whether engine updates can work at all. Three bounded probes answer
//! the questions a support conversation always ends up asking:
//!
//! - `PyPI` (`pypi.org`, dual-stack): is there any egress, and which
//!   address family carried it?
//! - GitHub API (`api.github.com`, IPv4-only - no AAAA, verified
//!   2026-10-31): can this box reach the host every engine lane's
//!   update check needs?
//! - Captive-portal check (`connectivitycheck.gstatic.com/generate_204`
//!   over plain HTTP): 204 = open internet; anything else that answers
//!   = a portal is intercepting.
//!
//! The probes ride the standard proxy environment (`HTTPS_PROXY`/
//! `ALL_PROXY`) through reqwest's defaults, so a proxied box reports
//! the proxy's egress - which is the truth that matters for updates.

use std::net::IpAddr;

/// How long each probe may take before it counts as unreachable. The
/// diagnosis runs on error paths and in `doctor`, never in a hot loop.
const PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);

/// Endpoints of the three probes. The GitHub row is deliberately the
/// API host (not github.com): it is IPv4-only, so it is the one probe
/// that cannot succeed over an IPv6-only path.
const PYPI_PROBE: &str = "https://pypi.org/pypi/pip/json";
const GITHUB_PROBE: &str = "https://api.github.com/zen";
const PORTAL_PROBE: &str = "http://connectivitycheck.gstatic.com/generate_204";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IpFamily {
    V4,
    V6,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ProbeResult {
    pub status: Option<u16>,
    pub family: Option<IpFamily>,
    pub err: Option<String>,
}

impl ProbeResult {
    fn ok(status: u16, family: Option<IpFamily>) -> Self {
        Self {
            status: Some(status),
            family,
            err: None,
        }
    }
    fn failed(err: impl Into<String>) -> Self {
        Self {
            status: None,
            family: None,
            err: Some(err.into()),
        }
    }
    #[must_use]
    pub fn reachable(&self) -> bool {
        self.status.is_some()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NetSnapshot {
    /// Dual-stack endpoint (PyPI): any-egress + family truth.
    pub any: ProbeResult,
    /// IPv4-only endpoint (GitHub API): engine-update reachability.
    pub github: ProbeResult,
    /// Captive-portal verdict from the 204 check.
    pub portal: ProbeResult,
}

/// One bounded GET. Errors are compressed to their display string -
/// the diagnosis names conditions, not transport trivia.
async fn probe(client: &reqwest::Client, url: &str) -> ProbeResult {
    match client.get(url).send().await {
        Ok(resp) => {
            let status = resp.status().as_u16();
            let family = resp.remote_addr().map(|sa| match sa.ip() {
                IpAddr::V4(_) => IpFamily::V4,
                IpAddr::V6(_) => IpFamily::V6,
            });
            ProbeResult::ok(status, family)
        }
        Err(e) => ProbeResult::failed(e.to_string()),
    }
}

/// Run the three probes concurrently. Each is independently bounded by
/// [`PROBE_TIMEOUT`]; the snapshot returns even when every probe fails
/// - a dead network is itself the diagnosis.
pub async fn snapshot() -> NetSnapshot {
    blazar_core::tls::ensure_tls_provider();
    let Ok(client) = reqwest::Client::builder()
        .timeout(PROBE_TIMEOUT)
        .connect_timeout(PROBE_TIMEOUT)
        .build()
    else {
        return NetSnapshot::default();
    };
    let (any, github, portal) = tokio::join!(
        probe(&client, PYPI_PROBE),
        probe(&client, GITHUB_PROBE),
        probe(&client, PORTAL_PROBE)
    );
    NetSnapshot {
        any,
        github,
        portal,
    }
}

/// Human teaching for a snapshot. `None` = everything healthy, nothing
/// to say. Pure - the unit pins cover every branch.
#[must_use]
pub fn diagnosis(snap: &NetSnapshot) -> Option<String> {
    let github_ok = snap.github.reachable();
    let any_ok = snap.any.reachable();
    // A portal is only "the story" when it intercepts: the 204 check
    // answering with anything-but-204 means a captive gateway rewrote
    // the request.
    let portal_active = matches!(snap.portal.status, Some(s) if s != 204);

    if github_ok && any_ok {
        return None; // healthy: no noise
    }
    if !any_ok && !github_ok {
        return Some(
            "no egress: pypi.org and api.github.com both unreachable - check the \
             network/Wi-Fi, or set HTTPS_PROXY to a reachable proxy"
                .to_string(),
        );
    }
    if portal_active {
        return Some(
            "captive portal active (connectivity check answered without a 204) - \
             complete the portal login in a browser, then retry"
                .to_string(),
        );
    }
    if !github_ok && snap.any.family == Some(IpFamily::V6) {
        return Some(
            "IPv6-only egress: pypi.org answers over IPv6 but GitHub's API and \
             release CDNs are IPv4-only (no AAAA) - engine updates are blocked \
             until the network provides IPv4 (portal auth / different network) or \
             HTTPS_PROXY points at a reachable proxy; model pulls keep working \
             (huggingface.co is dual-stack)"
                .to_string(),
        );
    }
    if !github_ok {
        return Some(
            "GitHub API unreachable while the general internet answers - GitHub-side \
             block or throttle (rate limit/abuse flag): wait it out or set GH_TOKEN"
                .to_string(),
        );
    }
    // GitHub answers but the dual-stack probe does not: an odd partial
    // network - say so without over-explaining.
    Some(
        "partial egress: api.github.com answers but pypi.org does not - pip lanes \
         (sglang/mlx) will report stale; check HTTPS_PROXY or the network path"
            .to_string(),
    )
}

/// Memoized [`diagnosis`] for error paths: `engine update --all` walks
/// four lanes and each failure wants the same suffix - probe once,
/// cache for half a minute, never add probe latency on a healthy box
/// (healthy snapshots memoize `None` too).
pub async fn diagnosis_suffix() -> String {
    type CacheCell = Option<(std::time::Instant, Option<String>)>;
    static CACHE: std::sync::OnceLock<tokio::sync::RwLock<CacheCell>> = std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| tokio::sync::RwLock::new(None));
    if let Some((at, hit)) = cache.read().await.as_ref()
        && at.elapsed() < std::time::Duration::from_secs(30)
    {
        return hit.clone().unwrap_or_default();
    }
    let snap = snapshot().await;
    let verdict = diagnosis(&snap);
    *cache.write().await = Some((std::time::Instant::now(), verdict.clone()));
    verdict.unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_pair(family: IpFamily) -> ProbeResult {
        ProbeResult::ok(200, Some(family))
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__net_diagnosis__healthy_is_silent() {
        let snap = NetSnapshot {
            any: ok_pair(IpFamily::V4),
            github: ProbeResult::ok(200, Some(IpFamily::V4)),
            portal: ProbeResult::ok(204, None),
        };
        assert_eq!(diagnosis(&snap), None);
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__net_diagnosis__ipv6_only_names_the_github_gap() {
        // The hotel shape: dual-stack endpoints ride v6, GitHub's
        // v4-only API dies. The teaching must name IPv4/AAA and the
        // proxy lever, and say model pulls still work.
        let snap = NetSnapshot {
            any: ok_pair(IpFamily::V6),
            github: ProbeResult::failed("refused"),
            portal: ProbeResult::failed("blocked"),
        };
        let d = diagnosis(&snap).expect("must diagnose");
        assert!(d.contains("IPv6-only"), "{d}");
        assert!(d.contains("IPv4"), "{d}");
        assert!(d.contains("HTTPS_PROXY"), "{d}");
        assert!(d.contains("huggingface.co"), "{d}");
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__net_diagnosis__captive_portal_named_when_intercepting() {
        let snap = NetSnapshot {
            any: ProbeResult::ok(302, Some(IpFamily::V4)),
            github: ProbeResult::failed("refused"),
            portal: ProbeResult::ok(200, Some(IpFamily::V4)),
        };
        let d = diagnosis(&snap).expect("must diagnose");
        assert!(d.contains("captive portal"), "{d}");
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__net_diagnosis__github_only_block_suggests_token_or_wait() {
        let snap = NetSnapshot {
            any: ok_pair(IpFamily::V4),
            github: ProbeResult::failed("timed out"),
            portal: ProbeResult::ok(204, None),
        };
        let d = diagnosis(&snap).expect("must diagnose");
        assert!(d.contains("GitHub-side"), "{d}");
        assert!(d.contains("GH_TOKEN"), "{d}");
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__net_diagnosis__fully_offline_says_so() {
        let snap = NetSnapshot {
            any: ProbeResult::failed("no route"),
            github: ProbeResult::failed("no route"),
            portal: ProbeResult::failed("no route"),
        };
        let d = diagnosis(&snap).expect("must diagnose");
        assert!(d.contains("no egress"), "{d}");
    }
}
