//! Client for the blazar capability-lane registry: a plain JSON file
//! that maps "architecture mainline can't load yet" to curated fork
//! lanes (repo + immutable commit) known to support it. The registry
//! PUBLISHES compatibility; it never defines it - every entry is still
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
    /// known - display-only provenance.
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
        blazar_core::tls::ensure_tls_provider();
        reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .build()
            .expect("build capability registry client")
    })
}

/// Bounded fetch of the registry file. Every entry is validated
/// (non-empty id, repo slug shape, 4..=40-hex SHA); invalid entries are
/// dropped with a warning - one bad row must not hide the good ones.
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
    let body = read_body_bounded(&mut resp, url, MAX_BODY_BYTES, "capability registry").await?;
    parse_registry(&body)
}

/// Read a response body capped at `cap` bytes: the registry URL is
/// user-configurable input, so a hostile mirror must not be able to
/// buffer unbounded memory behind the fetch timeout. Shared with the
/// draft-pairing registry (same fetch contract, different file).
async fn read_body_bounded(
    resp: &mut reqwest::Response,
    url: &str,
    cap: usize,
    label: &str,
) -> Result<String> {
    if let Some(len) = resp.content_length()
        && len > cap as u64
    {
        anyhow::bail!(
            "{label} {url} declares {len} bytes - over the {cap}-byte cap; \
                 point the registry URL at a sane mirror"
        );
    }
    let mut body = Vec::with_capacity(4096);
    while let Some(chunk) = resp
        .chunk()
        .await
        .with_context(|| format!("{label} {url}: read response body"))?
    {
        if body.len() + chunk.len() > cap {
            anyhow::bail!(
                "{label} {url} exceeded the {cap}-byte body cap - refusing to \
                 buffer a runaway mirror"
            );
        }
        body.extend_from_slice(&chunk);
    }
    String::from_utf8(body).context(format!("{label}: response is not valid UTF-8"))
}

/// Bounded read wrapper for sibling registry clients (draft-pairing
/// registry): identical contract, registry-specific label in errors.
pub(crate) async fn read_body_bounded_shared(
    resp: &mut reqwest::Response,
    url: &str,
    cap: usize,
    label: &str,
) -> Result<String> {
    read_body_bounded(resp, url, cap, label).await
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
                    "capability registry: entry {} ({}) failed validation - skipped",
                    idx,
                    lane.id
                );
            }
            Err(e) => {
                tracing::warn!("capability registry: entry {idx} malformed ({e}) - skipped");
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
/// registry - no fetch is attempted anywhere.
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

/// Compiled-in trust anchor: the base64 of the raw 32-byte ed25519
/// public key that signs `registry/capability-lanes.json`. Auto-install
/// only consumes registry content whose ed25519 signature verifies
/// against this key (or an explicit user override). Empty string would
/// mean "no shipped anchor" - the key below is the real one; the
/// matching private key never enters the repository.
pub const EMBEDDED_REGISTRY_PUBKEY_B64: &str = "WnTgolleCcVJEMo0hjvywAS5PsKJZIGh0d8ETBdx+Xw=";

/// Outcome of checking a fetched registry document against its
/// published signature. This is a gate on *automation only* - unsigned
/// or invalid registries still parse and can be installed manually
/// (warn-but-proceed); the daemon just refuses to build from them
/// unattended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignatureState {
    /// Signature at `<registry-url>.sig` verified against the effective
    /// ed25519 public key.
    Verified,
    /// No signature published (`.sig` absent or unreachable). The
    /// document is intact as served, but its origin is unproven.
    Unsigned,
    /// A signature is published but does not verify, or the effective
    /// public key is malformed. Treated as untrusted, not as fatal.
    Invalid,
}

impl SignatureState {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Verified => "verified",
            Self::Unsigned => "unsigned",
            Self::Invalid => "invalid-signature",
        }
    }
}

/// The effective verification key: an explicit config pin wins over
/// the compiled-in anchor (mirrors the URL override - a user hosting
/// their own registry signs it with their own key). `None` means the
/// configured value is malformed base64 or not 32 bytes, which callers
/// surface as [`SignatureState::Invalid`] rather than silently
/// skipping verification.
pub fn effective_pubkey(configured: Option<&str>) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    let b64 = configured.unwrap_or(EMBEDDED_REGISTRY_PUBKEY_B64);
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .map_err(|e| format!("capability_registry_pubkey is not valid base64: {e}"))?;
    if bytes.len() != 32 {
        return Err(format!(
            "capability_registry_pubkey must decode to 32 bytes (raw ed25519 public key), got {}",
            bytes.len()
        ));
    }
    Ok(bytes)
}

/// Verify a registry document against its detached signature with the
/// given raw ed25519 public key.
pub fn verify_registry_signature(body: &[u8], sig_b64: &str, pubkey: &[u8]) -> Result<(), String> {
    use base64::Engine as _;
    use ring::signature::UnparsedPublicKey;
    let sig = base64::engine::general_purpose::STANDARD
        .decode(sig_b64.trim())
        .map_err(|e| format!("registry .sig is not valid base64: {e}"))?;
    let key = UnparsedPublicKey::new(&ring::signature::ED25519, pubkey);
    key.verify(body, &sig)
        .map_err(|e| format!("registry signature does not verify: {e}"))
}

/// Fetch + parse the registry and report its signature state. The
/// registry document itself must fetch (a registry that cannot be
/// fetched has no offer to gate); only the signature leg is
/// soft-failing. Every fetch logs the document's sha256 so an operator
/// can correlate what the daemon saw with what the signer signed.
pub async fn fetch_verified(
    client: &reqwest::Client,
    url: &str,
    pubkey_config: Option<&str>,
) -> Result<(Vec<RegistryLane>, SignatureState)> {
    let mut resp = client
        .get(url)
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .with_context(|| format!("capability registry fetch failed: {url}"))?;
    if !resp.status().is_success() {
        anyhow::bail!("capability registry {url} answered HTTP {}", resp.status());
    }
    let body = read_body_bounded(&mut resp, url, MAX_BODY_BYTES, "capability registry").await?;
    let lanes = parse_registry(&body)?;

    let digest = {
        use sha2::Digest as _;
        let mut h = sha2::Sha256::new();
        h.update(body.as_bytes());
        hex::encode(h.finalize())
    };

    let state = match effective_pubkey(pubkey_config) {
        Ok(pk) => match fetch_signature(client, url).await {
            Ok(sig_b64) => {
                if verify_registry_signature(body.as_bytes(), &sig_b64, &pk).is_ok() {
                    SignatureState::Verified
                } else {
                    tracing::warn!(
                        registry_sha256 = %digest,
                        "capability registry signature INVALID - auto-install refused, manual install still possible"
                    );
                    SignatureState::Invalid
                }
            }
            Err(e) => {
                tracing::debug!(error = %e, "capability registry has no verifiable signature");
                SignatureState::Unsigned
            }
        },
        Err(e) => {
            tracing::warn!("{e} - treating the registry as untrusted");
            SignatureState::Invalid
        }
    };
    if state == SignatureState::Verified {
        tracing::debug!(registry_sha256 = %digest, "capability registry signature verified");
    }
    Ok((lanes, state))
}

/// Best-effort fetch of the detached signature at `<url>.sig`. Any
/// transport failure or non-200 maps to `Err` (unsigned).
async fn fetch_signature(client: &reqwest::Client, url: &str) -> Result<String> {
    let sig_url = format!("{url}.sig");
    let mut resp = client
        .get(&sig_url)
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .with_context(|| format!("signature fetch failed: {sig_url}"))?;
    if !resp.status().is_success() {
        anyhow::bail!("signature {sig_url} answered HTTP {}", resp.status());
    }
    read_body_bounded(&mut resp, &sig_url, 8192, "registry signature").await
}

/// Generate a fresh ed25519 signing pair. Returns
/// `(pkcs8_private_key_der, raw_public_key_32b)` - callers encode.
pub fn registry_keygen() -> Result<(Vec<u8>, Vec<u8>)> {
    use ring::signature::KeyPair as _;
    let rng = ring::rand::SystemRandom::new();
    let pkcs8 = ring::signature::Ed25519KeyPair::generate_pkcs8(&rng)
        .map_err(|e| anyhow::anyhow!("ed25519 keygen failed: {e}"))?;
    let pair = ring::signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref())
        .map_err(|e| anyhow::anyhow!("generated key rejected by its own parser: {e}"))?;
    Ok((pkcs8.as_ref().to_vec(), pair.public_key().as_ref().to_vec()))
}

/// Sign a registry document with a PKCS8 (DER) private key, returning
/// the raw 64-byte ed25519 signature.
pub fn registry_sign(body: &[u8], pkcs8_der: &[u8]) -> Result<Vec<u8>> {
    let pair = ring::signature::Ed25519KeyPair::from_pkcs8(pkcs8_der).map_err(|e| {
        anyhow::anyhow!(
            "private key is not a PKCS8 ed25519 key in the form `blazar registry keygen` prints (openssl genpkey emits a version-0 document this parser does not accept): {e}"
        )
    })?;
    Ok(pair.sign(body).as_ref().to_vec())
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
    /// the same validator the daemon uses - a malformed shipped file
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
        assert_eq!(offers_for_arch(&lanes, "nope").len(), 0);
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
        crate::test_env::set_env(REGISTRY_URL_ENV, "http://127.0.0.1:9/reg.json");
        assert_eq!(
            resolve_registry_url(None),
            Some("http://127.0.0.1:9/reg.json".to_string())
        );
        crate::test_env::remove_env(REGISTRY_URL_ENV);
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

#[cfg(test)]
mod sign_tests {
    use super::*;

    #[test]
    // suite convention: unit__scenario__expected (6b)
    #[allow(non_snake_case)]
    fn unit__registry_keygen_sign_verify__roundtrip_and_openssl_key_parse() {
        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD;

        // 1. ring-generated pair roundtrips through our own API.
        let (pkcs8, raw_pk) = registry_keygen().expect("keygen");
        assert_eq!(raw_pk.len(), 32);
        let body = br#"{"lanes":[]}"#;
        let sig = registry_sign(body, &pkcs8).expect("sign generated key");
        assert_eq!(sig.len(), 64);
        let sig_b64 = b64.encode(&sig);
        assert!(verify_registry_signature(body, &sig_b64, &raw_pk).is_ok());

        // 2. openssl genpkey emits a version-0/no-public-key PKCS8 that
        //    ring's ed25519 template structurally rejects; the teaching
        //    must point the operator at `blazar registry keygen`.
        let openssl_der_hex = "302e020100300506032b657004220420c762152167c796efddcc4aef4c60020dace465e49e35f6a3729c6f6a4d8eb905";
        let der = hex::decode(openssl_der_hex).expect("fixture hex");
        let rejected = registry_sign(body, &der).expect_err("openssl form must be refused");
        assert!(
            rejected.to_string().contains("blazar registry keygen"),
            "rejection must teach the supported key form: {rejected}"
        );

        // 3. a signature from a different key must not verify.
        let (other_pkcs8, other_pk) = registry_keygen().expect("second keygen");
        // Two keygens must yield distinct keys, or pinning would be theater.
        assert_ne!(other_pk, effective_pubkey(None).expect("embedded key"));
        let _ = other_pkcs8;
        let foreign_sig = b64.encode(registry_sign(body, &other_pkcs8).expect("sign second key"));
        assert!(
            verify_registry_signature(body, &foreign_sig, &raw_pk).is_err(),
            "a signature from a different key must not verify"
        );
    }

    /// Minimal parseable registry document local to this module - the
    /// signature legs below exercise `fetch_verified`, not the parser
    /// (the parser has its own battery in `mod tests`).
    fn tiny_registry_json() -> String {
        r#"[
            {
                "id": "tiny-fork",
                "repo": "acme/llama.cpp",
                "ref_sha": "7c81a9f06111c2d3f4e5a6b7c8d9e0f1a2b3c4d5",
                "architectures": ["tiny"],
                "backends": ["cuda"]
            }
        ]"#
        .to_string()
    }

    #[tokio::test]
    // suite convention: unit__scenario__expected (6b)
    #[allow(non_snake_case)]
    async fn unit__fetch_verified__correct_signature_reports_verified() {
        use base64::Engine as _;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let (pkcs8, raw_pk) = registry_keygen().expect("keygen");
        let body = tiny_registry_json();
        let sig_b64 = base64::engine::general_purpose::STANDARD
            .encode(registry_sign(body.as_bytes(), &pkcs8).expect("sign"));

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/reg.json"))
            .respond_with(ResponseTemplate::new(200).set_body_string(body.clone()))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/reg.json.sig"))
            .respond_with(ResponseTemplate::new(200).set_body_string(sig_b64))
            .mount(&server)
            .await;

        // The pair is freshly generated, so the PIN (not the embedded
        // anchor) is the key that must verify it - the same layering a
        // self-hosted registry configures.
        let pinned = base64::engine::general_purpose::STANDARD.encode(&raw_pk);
        let (lanes, state) =
            fetch_verified(http(), &format!("{}/reg.json", server.uri()), Some(&pinned))
                .await
                .expect("fetch_verified succeeds");
        assert_eq!(state, SignatureState::Verified);
        assert_eq!(lanes[0].id, "tiny-fork");
    }

    #[tokio::test]
    // suite convention: unit__scenario__expected (6b)
    #[allow(non_snake_case)]
    async fn unit__fetch_verified__missing_signature_leg_is_unsigned() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // registry present, .sig absent (wiremock 404s unmatched paths).
        Mock::given(method("GET"))
            .and(path("/reg.json"))
            .respond_with(ResponseTemplate::new(200).set_body_string(tiny_registry_json()))
            .mount(&server)
            .await;

        let (lanes, state) = fetch_verified(http(), &format!("{}/reg.json", server.uri()), None)
            .await
            .expect("registry body must still parse");
        assert_eq!(
            state,
            SignatureState::Unsigned,
            "absent .sig is unsigned, not a transport error"
        );
        assert_eq!(lanes.len(), 1);
    }

    #[tokio::test]
    // suite convention: unit__scenario__expected (6b)
    #[allow(non_snake_case)]
    async fn unit__fetch_verified__tampered_body_is_invalid() {
        use base64::Engine as _;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let (pkcs8, raw_pk) = registry_keygen().expect("keygen");
        // sign the ORIGINAL body, serve a TAMPERED one (ref_sha flipped):
        // the detached signature must not validate against edited bytes.
        // The signer's own key is pinned so the ONLY failure cause is
        // the tamper, not a key mismatch.
        let original = tiny_registry_json();
        let sig_b64 = base64::engine::general_purpose::STANDARD
            .encode(registry_sign(original.as_bytes(), &pkcs8).expect("sign"));
        let tampered = original.replace("7c81a9f0", "00000000");
        let pinned = base64::engine::general_purpose::STANDARD.encode(&raw_pk);

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/reg.json"))
            .respond_with(ResponseTemplate::new(200).set_body_string(tampered))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/reg.json.sig"))
            .respond_with(ResponseTemplate::new(200).set_body_string(sig_b64))
            .mount(&server)
            .await;

        let (lanes, state) =
            fetch_verified(http(), &format!("{}/reg.json", server.uri()), Some(&pinned))
                .await
                .expect("fetch itself succeeds; the verdict carries the refusal");
        assert_eq!(state, SignatureState::Invalid);
        assert_eq!(
            lanes.len(),
            1,
            "lanes are still returned - the gate refuses automation, not parsing"
        );
    }

    #[test]
    // suite convention: unit__scenario__expected (6b)
    #[allow(non_snake_case)]
    fn unit__effective_pubkey__layers_and_rejects_malformed_pins() {
        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD;

        // default: the embedded production key, raw 32 bytes.
        let embedded = effective_pubkey(None).expect("embedded key decodes");
        assert_eq!(
            embedded,
            b64.decode(EMBEDDED_REGISTRY_PUBKEY_B64)
                .expect("const decodes")
        );

        // a config pin replaces the embedded key.
        let (other_pkcs8, other_pk) = registry_keygen().expect("keygen");
        let _ = other_pkcs8;
        let pinned = effective_pubkey(Some(&b64.encode(&other_pk))).expect("pinned key accepted");
        assert_eq!(pinned, other_pk);

        // malformed pins fail closed with teaching, never fall back to
        // the embedded key (a typo must not silently re-trust upstream).
        // Note "A"*43 + "=" is shape-VALID (decodes to 32 zero bytes) -
        // a zero key is nonsense, but its shape is the key's shape.
        for bad in [
            "!!not-base64!!",        // not base64 at all
            "AAAA",                  // decodes to 3 bytes
            "A".repeat(44).as_str(), // decodes to 33 bytes (wrong length)
        ] {
            let err = effective_pubkey(Some(bad)).expect_err("malformed pin must be rejected");
            assert!(
                err.contains("capability_registry_pubkey"),
                "error names the knob: {err}"
            );
        }
    }
}
