//! Client for the blazar capability-lane registry: a plain JSON file
//! that maps "architecture mainline can't load yet" to curated fork
//! lanes (repo + immutable commit) known to support it. The registry
//! PUBLISHES compatibility; it never defines it — every entry is still
//! built from source through the same validated fork-lane pipeline as
//! a user-pinned fork, so a hostile or corrupt registry can at worst
//! point at a repo that fails validation, never bypass it.
//!
//! The server side is a static file (a raw GitHub URL works); this
//! module is intentionally client + format contract only.

use std::collections::BTreeSet;
use std::time::Duration;

use anyhow::{Context as _, Result};

/// Registry JSON: an array of lane entries. Fields are validated at
/// parse time; invalid entries are skipped with a warning, never fatal
/// (a broken registry must not take down model serving).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RegistryLane {
    /// Stable identifier used by `blazar engine install --lane <id>`.
    pub id: String,
    /// `owner/name` GitHub slug (validated by the same rules as
    /// user-pinned forks).
    pub repo: String,
    /// Immutable commit SHA the lane is built from.
    pub ref_sha: String,
    /// Upstream llama.cpp PR that would make this lane obsolete, when
    /// known — display-only provenance.
    pub upstream_pr: Option<u64>,
    /// Architectures the lane advertises (from its llama-arch.cpp).
    pub architectures: BTreeSet<String>,
    /// Backends the lane is known to build on (cpu, cuda, ...).
    pub backends: BTreeSet<String>,
    #[serde(default)]
    pub status: LaneStatus,
    #[serde(default)]
    pub note: Option<String>,
    /// Unix epoch seconds the entry was added.
    #[serde(default)]
    pub added_at: i64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LaneStatus {
    /// Known-good, installable.
    #[default]
    Active,
    /// Withdrawn (broken build, yanked): refused at install, hidden
    /// from offers, never auto-suggested.
    Retired,
}

impl LaneStatus {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Retired => "retired",
        }
    }
}

/// Default registry location. Served as a static JSON file from the
/// blazar repo until the ecosystem grows a dedicated home; the URL is
/// overridable so users can self-host or pin a mirror.
pub const DEFAULT_REGISTRY_URL: &str =
    "https://raw.githubusercontent.com/santanu20/blazar/main/registry/capability-lanes.json";

/// Env override, same knob class as `BLAZAR_GH_BASE` (tests point it
/// at a local fixture server).
pub const REGISTRY_URL_ENV: &str = "BLAZAR_CAPABILITY_REGISTRY";

/// Registry documents are a few KB of JSON; anything past this cap is a
/// misconfigured or hostile mirror, not a catalog.
const MAX_BODY_BYTES: usize = 1024 * 1024;

/// Shared pooled client for registry access (the per-request timeout in
/// [`fetch`] stays the bounding clock).
#[must_use]
pub fn http() -> &'static reqwest::Client {
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .build()
            .expect("build capability registry client")
    })
}

/// Bounded fetch of the registry file. Every entry is validated
/// (non-empty id, repo slug shape, 4..=40-hex SHA); invalid entries are
/// dropped with a warning — one bad row must not hide the good ones.
pub async fn fetch(client: &reqwest::Client, url: &str) -> Result<Vec<RegistryLane>> {
    let mut resp = client
        .get(url)
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .with_context(|| format!("capability registry fetch failed: {url}"))?;
    if !resp.status().is_success() {
        anyhow::bail!("capability registry {url} answered HTTP {}", resp.status());
    }
    let body = read_body_bounded(&mut resp, url, MAX_BODY_BYTES).await?;
    parse_registry(&body)
}

/// Read a response body capped at `cap` bytes: the registry URL is
/// user-configurable input, so a hostile mirror must not be able to
/// buffer unbounded memory behind the fetch timeout.
async fn read_body_bounded(resp: &mut reqwest::Response, url: &str, cap: usize) -> Result<String> {
    if let Some(len) = resp.content_length() {
        if len > cap as u64 {
            anyhow::bail!(
                "capability registry {url} declares {len} bytes — over the {cap}-byte cap; \
                 point capability_registry_url at a sane mirror"
            );
        }
    }
    let mut body = Vec::with_capacity(4096);
    while let Some(chunk) = resp
        .chunk()
        .await
        .with_context(|| format!("capability registry {url}: read response body"))?
    {
        if body.len() + chunk.len() > cap {
            anyhow::bail!(
                "capability registry {url} exceeded the {cap}-byte body cap — refusing to \
                 buffer a runaway mirror"
            );
        }
        body.extend_from_slice(&chunk);
    }
    String::from_utf8(body).context("capability registry: response is not valid UTF-8")
}

/// Parse + validate a registry document. Public so tests (and curious
/// users) can check a file offline against the same rules.
pub fn parse_registry(text: &str) -> Result<Vec<RegistryLane>> {
    let raw: Vec<serde_json::Value> =
        serde_json::from_str(text).context("capability registry: not a JSON array")?;
    let mut lanes = Vec::new();
    for (idx, value) in raw.iter().enumerate() {
        match serde_json::from_value::<RegistryLane>(value.clone()) {
            Ok(lane) if lane_entry_valid(&lane) => lanes.push(lane),
            Ok(lane) => {
                tracing::warn!(
                    "capability registry: entry {} ({}) failed validation — skipped",
                    idx,
                    lane.id
                );
            }
            Err(e) => {
                tracing::warn!("capability registry: entry {idx} malformed ({e}) — skipped");
            }
        }
    }
    Ok(lanes)
}

fn lane_entry_valid(lane: &RegistryLane) -> bool {
    !lane.id.trim().is_empty()
        && super::build::validate_repo_slug(&lane.repo).is_ok()
        && super::build::validate_commit_sha(&lane.ref_sha).is_ok()
}

/// Active lanes advertising an architecture, newest-first by
/// `added_at` (ties broken by id for determinism).
#[must_use]
pub fn offers_for_arch<'a>(lanes: &'a [RegistryLane], arch: &str) -> Vec<&'a RegistryLane> {
    let mut hits: Vec<_> = lanes
        .iter()
        .filter(|l| l.status == LaneStatus::Active && l.architectures.contains(arch))
        .collect();
    hits.sort_by(|a, b| b.added_at.cmp(&a.added_at).then_with(|| a.id.cmp(&b.id)));
    hits
}

/// Resolve the effective registry URL: explicit config > env > default.
/// `Some("")` (config set to empty) means the user disabled the
/// registry — no fetch is attempted anywhere.
#[must_use]
pub fn resolve_registry_url(configured: Option<&str>) -> Option<String> {
    if let Some(url) = configured {
        return (!url.trim().is_empty()).then(|| url.trim().to_string());
    }
    let from_env = std::env::var(REGISTRY_URL_ENV).unwrap_or_default();
    if from_env.trim().is_empty() {
        Some(DEFAULT_REGISTRY_URL.to_string())
    } else {
        Some(from_env.trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn active_lane_json() -> String {
        r#"[
            {
                "id": "qwen35-fork",
                "repo": "acme/llama.cpp",
                "ref_sha": "7c81a9f06111c2d3f4e5a6b7c8d9e0f1a2b3c4d5",
                "upstream_pr": 18234,
                "architectures": ["qwen35", "qwen35_moe"],
                "backends": ["cpu", "cuda"],
                "note": "temporary until PR merges",
                "added_at": 1789800000
            },
            {
                "id": "old-qwen35",
                "repo": "other/llama.cpp",
                "ref_sha": "0000000000000000000000000000000000000001",
                "architectures": ["qwen35"],
                "backends": ["cpu"],
                "status": "retired",
                "added_at": 1789700000
            },
            {
                "id": "bad-slug",
                "repo": "not a slug",
                "ref_sha": "0000000000000000000000000000000000000002",
                "architectures": ["x"],
                "backends": ["cpu"],
                "added_at": 1789600000
            },
            {
                "id": "branch-not-sha",
                "repo": "acme/llama.cpp",
                "ref_sha": "main",
                "architectures": ["y"],
                "backends": ["cpu"],
                "added_at": 1789500000
            }
        ]"#
        .to_string()
    }

    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__parse_registry__keeps_valid_drops_invalid() {
        let lanes = parse_registry(&active_lane_json()).unwrap();
        let ids: Vec<_> = lanes.iter().map(|l| l.id.as_str()).collect();
        assert_eq!(ids, vec!["qwen35-fork", "old-qwen35"]);
        assert_eq!(lanes[0].upstream_pr, Some(18234));
        assert_eq!(lanes[0].status, LaneStatus::Active);
        assert_eq!(lanes[1].status, LaneStatus::Retired);
    }

    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__parse_registry__non_array_is_error() {
        assert!(parse_registry("{\"not\": \"array\"}").is_err());
        assert!(parse_registry("not json").is_err());
    }

    #[tokio::test]
    #[allow(non_snake_case)]
    async fn unit__fetch__refuses_oversized_registry_body() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        let server = MockServer::start().await;
        // 2 MiB of body against the 1 MiB cap: a hostile mirror must not
        // be buffered whole behind the fetch timeout.
        Mock::given(method("GET"))
            .and(path("/reg.json"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![b'x'; 2 * MAX_BODY_BYTES]))
            .mount(&server)
            .await;
        let url = format!("{}/reg.json", server.uri());
        let err = fetch(http(), &url).await.unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("cap"), "unexpected error: {msg}");
    }

    #[tokio::test]
    #[allow(non_snake_case)]
    async fn unit__fetch__parses_small_catalog_end_to_end() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/reg.json"))
            .respond_with(ResponseTemplate::new(200).set_body_string(active_lane_json()))
            .mount(&server)
            .await;
        let url = format!("{}/reg.json", server.uri());
        let lanes = fetch(http(), &url).await.unwrap();
        let ids: Vec<_> = lanes.iter().map(|l| l.id.as_str()).collect();
        assert_eq!(ids, vec!["qwen35-fork", "old-qwen35"]);
    }

    /// The shipped default registry document must parse clean through
    /// the same validator the daemon uses — a malformed shipped file
    /// would 404-and-fail-open silently otherwise. Reads the file from
    /// the repo root relative to this crate, so it never depends on a
    /// machine-specific path.
    #[tokio::test]
    #[allow(non_snake_case)]
    async fn unit__fetch__shipped_default_registry_parses() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../registry/capability-lanes.json"
        );
        let body = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        let lanes = parse_registry(&body).expect("shipped registry parses");
        assert!(
            lanes.iter().any(|l| l.id == "instella-moe"),
            "shipped registry carries the validated instella-moe lane"
        );
        for lane in &lanes {
            assert_eq!(lane.status, LaneStatus::Active, "lane {} active", lane.id);
        }
    }

    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__offers_for_arch__active_newest_first_retired_excluded() {
        let lanes = parse_registry(&active_lane_json()).unwrap();
        let offers = offers_for_arch(&lanes, "qwen35");
        assert_eq!(offers.len(), 1);
        assert_eq!(offers[0].id, "qwen35-fork");
        assert!(offers_for_arch(&lanes, "nope").is_empty());
    }

    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__resolve_registry_url__config_env_default_layers() {
        // Config wins, empty config disables.
        assert_eq!(
            resolve_registry_url(Some("https://example.test/reg.json")),
            Some("https://example.test/reg.json".to_string())
        );
        assert_eq!(resolve_registry_url(Some("  ")), None);
        // Env fills in when config is absent.
        std::env::set_var(REGISTRY_URL_ENV, "http://127.0.0.1:9/reg.json");
        assert_eq!(
            resolve_registry_url(None),
            Some("http://127.0.0.1:9/reg.json".to_string())
        );
        std::env::remove_var(REGISTRY_URL_ENV);
        assert_eq!(
            resolve_registry_url(None),
            Some(DEFAULT_REGISTRY_URL.to_string())
        );
    }

    #[test]
    #[allow(non_snake_case)] // suite convention: unit__scenario__expected
    fn unit__lane_status__serde_lowercase_and_default_active() {
        // Status defaults to active when absent.
        let lane: RegistryLane = serde_json::from_str(
            r#"{"id":"a","repo":"acme/llama.cpp","ref_sha":"7c81a9f0",
                "architectures":[],"backends":[]}"#,
        )
        .unwrap();
        assert_eq!(lane.status, LaneStatus::Active);
        let text = serde_json::to_string(&LaneStatus::Retired).unwrap();
        assert_eq!(text, "\"retired\"");
    }
}
