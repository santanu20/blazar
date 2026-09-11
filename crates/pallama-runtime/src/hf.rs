//! Hugging Face model registry client + pull orchestration.
//!
//! Security invariants (ollama complaint #7 / CVE-2025-51471 class):
//! - outbound HTTPS only to HF hosts via a redirect-host allowlist
//! - the HF token is attached ONLY to first-party `huggingface.co` requests
//!   and is never logged, stored, or forwarded to CDN hosts
//! - LFS sha256 (when the API provides it) is verified; mismatch deletes
//!   the partial file and fails the pull
//!
//! Store layout stays plain files (`models/*.gguf`) so any tool can use
//! them directly — no blob storage, ever (complaint #4).

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use pallama_core::gguf;
use pallama_core::store::{ModelRow, Store};
use pallama_core::PallamaDirs;

use crate::events::PallamaEvent;

use crate::events::EventBus;

pub const HF_API_BASE: &str = "https://huggingface.co";
pub const DEFAULT_QUANT: &str = "Q4_K_M";

/// Redirect-host allowlist. Anything else is refused mid-redirect.
const EXTRA_DOWNLOAD_HOSTS: &[&str] = &["cdn-lfs.huggingface.co", "cas-bridge.xethub.hf.co"];

#[must_use]
pub fn is_allowed_download_host(host: &str) -> bool {
    host == "huggingface.co"
        || host.ends_with(".huggingface.co")
        || host == "hf.co"
        || host.ends_with(".hf.co")
        || EXTRA_DOWNLOAD_HOSTS.contains(&host)
}

// ---------------------------------------------------------------------------
// Naming (complaint #6: registry name = actual repo name, never an alias)
// ---------------------------------------------------------------------------

/// `Qwen/Qwen2.5-0.5B-Instruct-GGUF` -> `qwen2.5-0.5b-instruct`.
#[must_use]
pub fn registry_name(repo: &str) -> String {
    let tail = repo.rsplit('/').next().unwrap_or(repo);
    let lowered = tail.to_lowercase();
    lowered
        .strip_suffix("-gguf")
        .unwrap_or(&lowered)
        .to_string()
}

#[derive(Debug, Clone, PartialEq)]
pub struct PullTarget {
    pub repo: String,
    pub quant: String,
}

/// Accepts `owner/repo:QUANT`, `owner/repo` (default quant), or a catalog
/// short name (resolved via the embedded catalog).
pub fn parse_pull_target(input: &str) -> Result<PullTarget> {
    let input = input.trim();
    if input.contains('/') {
        let (repo, quant) = match input.split_once(':') {
            Some((r, q)) => (r, q.to_uppercase()),
            None => (input, DEFAULT_QUANT.to_string()),
        };
        if repo.is_empty() || quant.is_empty() {
            return Err(anyhow!(
                "invalid pull target {input:?}: empty repo or quant"
            ));
        }
        return Ok(PullTarget {
            repo: repo.to_string(),
            quant,
        });
    }
    let entry = pallama_core::resolve(input).map_err(|e| anyhow!("{e}"))?;
    Ok(PullTarget {
        repo: entry.repo.clone(),
        quant: DEFAULT_QUANT.to_string(),
    })
}

// ---------------------------------------------------------------------------
// HF API shapes (verified against a live `?blobs=true` response; fixture:
// crates/pallama-runtime/tests/fixtures/qwen2.5-0.5b-blobs.json)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct HfLfs {
    pub sha256: String,
    #[serde(default)]
    pub size: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HfSibling {
    pub rfilename: String,
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(default)]
    pub lfs: Option<HfLfs>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HfGgufInfo {
    #[serde(default)]
    pub total: Option<u64>,
    #[serde(default)]
    pub architecture: Option<String>,
    #[serde(default)]
    pub context_length: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HfModelInfo {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub siblings: Vec<HfSibling>,
    #[serde(default)]
    pub gguf: Option<HfGgufInfo>,
}

/// One file to download (a model shard or an mmproj).
#[derive(Debug, Clone)]
pub struct FilePlan {
    pub filename: String,
    pub bytes: u64,
    pub sha256: Option<String>,
}

/// Result of file selection for a pull.
#[derive(Debug, Clone)]
pub struct SelectedFiles {
    /// Shards in order; single-element vec for unsplit models.
    pub shards: Vec<FilePlan>,
    /// Parsed quant actually chosen (may differ from requested on fallback).
    pub quant: String,
    pub quant_fallback: bool,
    pub mmproj: Option<FilePlan>,
}

/// Pick model files from repo siblings: quant match (case-insensitive),
/// shard sets (`-00001-of-0000N`), and an optional mmproj projector.
/// Falls back to the smallest single GGUF (flagged) when the requested
/// quant is absent — never silently: the fallback is recorded in the row.
#[allow(clippy::case_sensitive_file_extension_comparisons)] // operand pre-lowercased
pub fn select_files(siblings: &[HfSibling], wanted_quant: &str) -> Result<SelectedFiles> {
    let wanted = wanted_quant.to_lowercase();
    let ggufs: Vec<&HfSibling> = siblings
        .iter()
        .filter(|s| s.rfilename.to_lowercase().ends_with(".gguf")) // pre-lowercased above
        .collect();
    if ggufs.is_empty() {
        return Err(anyhow!("repo has no .gguf files"));
    }

    // Group: unsplit vs sharded, keyed by quant-bearing base name.
    // Shard marker: `-00001-of-00002.gguf` (gguf-split convention).
    let mut sharded: std::collections::BTreeMap<(String, u32), Vec<(u32, &HfSibling)>> =
        std::collections::BTreeMap::new();
    let mut singles: Vec<&HfSibling> = Vec::new();
    for s in &ggufs {
        match parse_shard_marker(&s.rfilename) {
            Some((idx, count, base)) => {
                sharded.entry((base, count)).or_default().push((idx, s));
            }
            None => singles.push(s),
        }
    }

    let quant_of = |fname: &str| -> Option<String> {
        // Works on full filenames ("m-q4_k_m.gguf") and shard bases
        // ("m-q4_k_m"): the quant token is the trailing path-free segment.
        let lower = fname.to_lowercase();
        let leaf = lower.rsplit('/').next().unwrap_or(&lower);
        let stem = leaf.strip_suffix(".gguf").unwrap_or(leaf);
        let token = stem.rsplit('-').next()?;
        // Recognized quant shapes: q4_k_m, q8_0, iq4_xs, fp16, bf16, f16, q2_k...
        let t = token.trim_start_matches('.');
        if t.starts_with('q')
            || t.starts_with("iq")
            || t.starts_with("bx")
            || t == "fp16"
            || t == "bf16"
            || t == "f16"
            || t == "f32"
        {
            Some(t.to_string())
        } else {
            None
        }
    };

    // Try requested quant among singles first, then shard sets.
    for s in &singles {
        if quant_of(&s.rfilename).as_deref() == Some(wanted.as_str()) {
            return Ok(finish(
                vec![plan(s)],
                wanted_quant.to_string(),
                false,
                siblings,
            ));
        }
    }
    for ((base, _count), group) in &sharded {
        if quant_of(base).as_deref() == Some(wanted.as_str()) {
            let mut ordered: Vec<&HfSibling> = group.iter().map(|(_, s)| *s).collect();
            ordered.sort_by_key(|s| parse_shard_marker(&s.rfilename).map_or(0, |m| m.0));
            let plans: Vec<FilePlan> = ordered.iter().map(|s| plan(s)).collect();
            return Ok(finish(plans, wanted_quant.to_string(), false, siblings));
        }
    }

    // Fallback: smallest single GGUF (deterministic, most likely to run).
    if let Some(smallest) = singles.iter().min_by_key(|s| plan(s).bytes) {
        let q = quant_of(&smallest.rfilename).unwrap_or_else(|| "unknown".into());
        return Ok(finish(
            vec![plan(smallest)],
            q.to_uppercase(),
            true,
            siblings,
        ));
    }
    // No singles at all: pick the smallest shard set by total size.
    let (_, group) = sharded
        .iter()
        .min_by_key(|(_, g)| g.iter().map(|(_, s)| plan(s).bytes).sum::<u64>())
        .ok_or_else(|| anyhow!("repo has GGUFs but no complete file set"))?;
    let mut ordered: Vec<&HfSibling> = group.iter().map(|(_, s)| *s).collect();
    ordered.sort_by_key(|s| parse_shard_marker(&s.rfilename).map_or(0, |m| m.0));
    let plans: Vec<FilePlan> = ordered.iter().map(|s| plan(s)).collect();
    let q = quant_of(&ordered[0].rfilename).unwrap_or_else(|| "unknown".into());
    Ok(finish(plans, q.to_uppercase(), true, siblings))
}

/// `base-q4_k_m-00001-of-00002.gguf` -> `(1, 2, "base-q4_k_m")`.
/// Accepts only the gguf-split 5-digit `-of-` tail with idx in 1..=count.
fn parse_shard_marker(fname: &str) -> Option<(u32, u32, String)> {
    parse_shard_marker_pub(fname)
}

/// Public wrapper (shard-set file enumeration is shared with model removal).
#[must_use]
pub fn parse_shard_marker_pub(fname: &str) -> Option<(u32, u32, String)> {
    let lower = fname.to_lowercase();
    let stem = lower.strip_suffix(".gguf")?;
    let of_pos = stem.rfind("-of-")?;
    let count: u32 = stem[of_pos + 4..].parse().ok()?;
    let before = &stem[..of_pos];
    let idx_start = before.rfind('-')? + 1;
    let idx: u32 = before[idx_start..].parse().ok()?;
    if idx < 1 || idx > count || count < 2 {
        return None;
    }
    let base = before[..idx_start - 1].to_string();
    Some((idx, count, base))
}

fn plan(s: &HfSibling) -> FilePlan {
    FilePlan {
        filename: s.rfilename.clone(),
        bytes: s.lfs.as_ref().and_then(|l| l.size).or(s.size).unwrap_or(0),
        sha256: s.lfs.as_ref().map(|l| l.sha256.clone()),
    }
}

#[allow(clippy::case_sensitive_file_extension_comparisons)] // operand pre-lowercased
fn finish(
    shards: Vec<FilePlan>,
    quant: String,
    quant_fallback: bool,
    siblings: &[HfSibling],
) -> SelectedFiles {
    let mmproj = siblings
        .iter()
        .filter(|s| {
            let lower = s.rfilename.to_lowercase();
            lower.starts_with("mmproj") && lower.ends_with(".gguf")
        })
        .min_by_key(|s| plan(s).bytes)
        .map(plan);
    SelectedFiles {
        shards,
        quant,
        quant_fallback,
        mmproj,
    }
}

/// Rough bits-per-weight for a quant label; display-only param estimation.
fn quant_bpw(quant: &str) -> f64 {
    let q = quant.to_lowercase();
    match q.as_str() {
        "q2_k" | "q2_k_s" => 3.35,
        "q3_k_s" => 3.5,
        "q3_k_m" => 3.91,
        "q3_k_l" => 4.27,
        "q4_0" | "q4_1" => 4.55,
        "iq4_xs" => 4.25,
        "q4_k_s" => 4.5,
        "q4_k_m" => 4.85,
        "q5_0" | "q5_1" => 5.7,
        "q5_k_s" => 5.54,
        "q5_k_m" => 5.69,
        "q6_k" => 6.59,
        "q8_0" => 8.5,
        "fp16" | "f16" | "bf16" => 16.0,
        "f32" => 32.0,
        _ => 5.0,
    }
}

/// Estimated parameter count from file size + quant (labeled estimate,
/// never used for launch decisions).
#[must_use]
#[allow(clippy::cast_precision_loss)] // display-only estimate
pub fn est_params(bytes: u64, quant: &str) -> f64 {
    let bpw = quant_bpw(quant);
    if bpw <= 0.0 {
        return 0.0;
    }
    (bytes as f64 * 8.0 / bpw) / 1e9
}

// ---------------------------------------------------------------------------
// HTTP client
// ---------------------------------------------------------------------------

pub struct HfClient {
    http: reqwest::Client,
    api_base: reqwest::Url,
    dl_base: reqwest::Url,
    token: Option<String>,
    /// Parallel byte-range connections for large downloads (see
    /// `hf_parallel`). 1 = classic single-stream lane.
    pub(crate) download_connections: u32,
    /// Test-only extra redirect-allowed hosts (wiremock).
    extra_hosts: Vec<String>,
}

impl HfClient {
    pub fn new(token: Option<String>) -> Result<Self> {
        Self::with_bases(HF_API_BASE, HF_API_BASE, token, Vec::new())
    }

    /// Builder: set parallel download connections.
    #[must_use]
    pub fn with_download_connections(mut self, connections: u32) -> Self {
        self.download_connections = connections;
        self
    }

    #[allow(clippy::needless_pass_by_value)] // Vec is stored
    pub fn with_bases(
        api_base: &str,
        dl_base: &str,
        token: Option<String>,
        extra_hosts: Vec<String>,
    ) -> Result<Self> {
        // Empty env token degrades to anonymous (an empty Bearer is an
        // invalid credential, not a missing one).
        let token = token.filter(|t| !t.trim().is_empty());
        let extra = extra_hosts.clone();
        let policy = reqwest::redirect::Policy::custom(move |attempt| {
            let host = attempt.url().host_str().unwrap_or_default().to_string();
            if host.is_empty() {
                return attempt.error("redirect target has no host");
            }
            if is_allowed_download_host(&host) || extra.contains(&host) {
                attempt.follow()
            } else {
                attempt.error(format!(
                    "blocked redirect to non-allowlisted host {host:?} (HF hosts only)"
                ))
            }
        });
        let http = reqwest::Client::builder()
            .redirect(policy)
            // Downloads are multi-hundred-MB: no total cap, bounded stalls.
            .connect_timeout(Duration::from_secs(30))
            .read_timeout(Duration::from_mins(2))
            .build()
            .context("build HF http client")?;
        Ok(Self {
            http,
            api_base: reqwest::Url::parse(api_base)?,
            dl_base: reqwest::Url::parse(dl_base)?,
            token,
            download_connections: 8,
            extra_hosts,
        })
    }

    /// Token rides ONLY on first-party huggingface.co requests (test
    /// stand-ins included); CDN hops never see it.
    fn token_for(&self, url: &reqwest::Url) -> Option<String> {
        let host = url.host_str()?;
        let first_party = host == "huggingface.co" || host.ends_with(".huggingface.co");
        (first_party || self.extra_hosts.iter().any(|h| h == host))
            .then(|| self.token.clone())
            .flatten()
    }

    /// GET /api/models/{repo}?blobs=true
    pub async fn model_info(&self, repo: &str) -> Result<HfModelInfo> {
        let url = self
            .api_base
            .join(&format!("api/models/{repo}?blobs=true"))
            .map_err(|e| anyhow!("bad repo {repo:?}: {e}"))?;
        let mut req = self.http.get(url.clone());
        if let Some(token) = self.token_for(&url) {
            req = req.bearer_auth(token);
        }
        let resp = req.send().await.context("HF models API request failed")?;
        match resp.status() {
            reqwest::StatusCode::OK => {}
            reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN => {
                return Err(anyhow!(
                    "repo {repo} is gated or private (status {}). Set HF_TOKEN if you have access",
                    resp.status()
                ));
            }
            reqwest::StatusCode::NOT_FOUND => {
                return Err(anyhow!("repo {repo} not found on Hugging Face"));
            }
            reqwest::StatusCode::TOO_MANY_REQUESTS => {
                let retry = resp
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("a while");
                return Err(anyhow!("HF rate limited; retry after {retry}"));
            }
            other => {
                return Err(anyhow!("HF models API {other} for {repo}"));
            }
        }
        let info: HfModelInfo = resp.json().await.context("decode models API JSON")?;
        Ok(info)
    }

    /// Stream one file to `dest` (via `.part` + atomic rename), resuming
    /// from a previous partial when present. Returns final byte count.
    /// `on_progress` is called with (downloaded, total) after each chunk.
    #[allow(clippy::too_many_lines)] // flat probe->parallel->resume->verify pipeline by design
    pub async fn download_file(
        &self,
        repo: &str,
        plan: &FilePlan,
        dest: &Path,
        mut on_progress: impl FnMut(u64, u64),
    ) -> Result<u64> {
        let url = self
            .dl_base
            .join(&format!(
                "{repo}/resolve/main/{}",
                url_encode_path(&plan.filename)
            ))
            .map_err(|e| anyhow!("bad download URL for {}: {e}", plan.filename))?;
        // Parallel byte-range lane first: engages only when the expected
        // size (from pull metadata) can pay for it and the server proves
        // Range support on a probe; every other shape falls through to
        // the classic lane below.
        if self.download_connections > 1
            && (plan.bytes == 0 || plan.bytes >= crate::hf_parallel::MIN_PARALLEL_BYTES)
        {
            let token = self.token_for(&url);
            if let Some(bytes) = crate::hf_parallel::try_parallel(
                &self.http,
                token.as_deref(),
                &url,
                plan,
                dest,
                self.download_connections,
                &mut on_progress,
            )
            .await?
            {
                return Ok(bytes);
            }
        }
        let part = sibling_part_path(dest);
        let mut have: u64 = 0;
        let mut hasher = Sha256::new();

        if part.exists() {
            let len = std::fs::metadata(&part)
                .map_err(|e| anyhow!("stat {}: {e}", part.display()))?
                .len();
            // Seed the hasher with existing bytes.
            let existing = tokio::fs::File::open(&part).await?;
            let mut reader = tokio::io::BufReader::new(existing);
            let mut buf = vec![0u8; 1 << 20];
            loop {
                let n = reader.read(&mut buf).await?;
                if n == 0 {
                    break;
                }
                hasher.update(&buf[..n]);
            }
            have = len;
        }

        let mut req = self.http.get(url.clone());
        if let Some(token) = self.token_for(&url) {
            req = req.bearer_auth(token);
        }
        if have > 0 {
            req = req.header(reqwest::header::RANGE, format!("bytes={have}-"));
        }
        let mut resp = req.send().await.context("download request failed")?;
        let status = resp.status();
        if !status.is_success() && status.as_u16() != 206 {
            return Err(anyhow!(
                "download {} failed: {} {}",
                plan.filename,
                status.as_str(),
                status.canonical_reason().unwrap_or("")
            ));
        }
        // Server ignored the Range: restart from zero.
        if have > 0 && status.as_u16() != 206 {
            have = 0;
            hasher = Sha256::new();
        }
        let total = if status.as_u16() == 206 {
            have + resp.content_length().unwrap_or(0)
        } else {
            resp.content_length().unwrap_or(plan.bytes)
        };

        let mut file = if status.as_u16() == 206 {
            tokio::fs::OpenOptions::new()
                .append(true)
                .open(&part)
                .await?
        } else {
            tokio::fs::File::create(&part).await?
        };
        let mut downloaded = have;
        on_progress(downloaded, total);
        while let Some(chunk) = resp.chunk().await? {
            file.write_all(&chunk).await?;
            hasher.update(&chunk);
            downloaded += chunk.len() as u64;
            on_progress(downloaded, total);
        }
        file.flush().await?;
        drop(file);

        if let Some(expected) = &plan.sha256 {
            let got = format!("{:x}", hasher.finalize());
            if !got.eq_ignore_ascii_case(expected) {
                let _ = tokio::fs::remove_file(&part).await;
                return Err(anyhow!(
                    "sha256 mismatch for {}: expected {expected}, got {got}; partial deleted",
                    plan.filename
                ));
            }
        }
        tokio::fs::rename(&part, dest)
            .await
            .with_context(|| format!("finalize {}", dest.display()))?;
        Ok(downloaded)
    }
}

fn url_encode_path(path: &str) -> String {
    path.split('/')
        .map(|seg| {
            seg.bytes()
                .map(|b| match b {
                    b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                        char::from(b).to_string()
                    }
                    _ => format!("%{b:02X}"),
                })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("/")
}

pub(crate) fn sibling_part_path(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_os_string();
    s.push(".part");
    PathBuf::from(s)
}

// ---------------------------------------------------------------------------
// Search + fit (no download)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct SearchEntry {
    pub id: String,
    #[serde(default)]
    pub downloads: Option<u64>,
    #[serde(default)]
    pub likes: Option<u64>,
    #[serde(default)]
    pub siblings: Vec<HfSibling>,
    #[serde(default)]
    pub gguf: Option<HfGgufInfo>,
}

/// Quant names advertised by a repo's GGUF filenames, canonical uppercase,
/// deduped, alphabetically sorted. Powers the search table's QUANTS column
/// so `pallama pull <REPO>[:quant]` can be chosen from the listing itself.
///
/// A filename segment qualifies only if it parses as `(IQ|TQ|BF|F|Q)` +
/// digit + `[A-Za-z0-9_]*` — model-size tags (`3B`), dates (`2511`) and
/// finetune words (`heretic`, `fable`) don't. Shards (`-00002-of-00005`,
/// digit-only tail) and `mmproj-*` projectors are excluded by the same
/// rules.
pub fn quant_tokens(names: impl IntoIterator<Item = impl AsRef<str>>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for name in names {
        let Some(quant) = quant_token(name.as_ref()) else {
            continue;
        };
        if !out.contains(&quant) {
            out.push(quant);
        }
    }
    out.sort();
    out
}

fn quant_token(filename: &str) -> Option<String> {
    if !std::path::Path::new(filename)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
    {
        return None;
    }
    let lower = filename.to_ascii_lowercase();
    if lower.starts_with("mmproj") {
        return None;
    }
    let stem = &lower[..lower.len() - ".gguf".len()];
    let last = stem.rsplit(['.', '-']).next()?;
    for prefix in ["iq", "tq", "bf", "q", "f"] {
        if let Some(rest) = last.strip_prefix(prefix) {
            let mut chars = rest.chars();
            if matches!(chars.next(), Some(d) if d.is_ascii_digit())
                && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                return Some(last.to_ascii_uppercase());
            }
        }
    }
    None
}

impl HfClient {
    /// GGUF-filtered model search (complaint #15: discovery beyond a
    /// registry; any community quant is findable).
    pub async fn search(&self, query: &str, limit: u32) -> Result<Vec<SearchEntry>> {
        let url = self
            .api_base
            .join(&format!(
                "api/models?search={}&filter=gguf&limit={limit}&sort=downloads&direction=-1&expand[]=gguf&expand[]=likes&expand[]=siblings",
                url_encode_path(query)
            ))
            .map_err(|e| anyhow!("bad search URL: {e}"))?;
        let resp = self
            .http
            .get(url)
            .send()
            .await
            .context("HF search request")?;
        if !resp.status().is_success() {
            return Err(anyhow!("HF search returned {}", resp.status()));
        }
        resp.json().await.context("decode search results")
    }
}

/// One row of a fit preview.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FitRow {
    pub quant: String,
    pub file: String,
    pub bytes: u64,
    pub fits_vram: bool,
    pub kv_bytes_at_default_ctx: u64,
    pub recommended_ctx: u32,
    /// ctx that fits when the KV cache runs `q8_0` (halved): shows the
    /// headroom the profile ladder's `q8_0` grade buys on this GPU.
    pub recommended_ctx_q8: u32,
}

/// Pre-download compatibility preview (complaint #14): given a repo's
/// sibling list, produce per-quant fit rows against the local hardware.
/// Uses the same KV math as the profile compiler's rule 6.
#[must_use]
pub fn fit_rows(siblings: &[HfSibling], vram_bytes: u64, default_ctx: u32) -> Vec<FitRow> {
    let mut rows = Vec::new();
    for s in siblings {
        let lower = s.rfilename.to_lowercase();
        let Some(stem) = lower.strip_suffix(".gguf") else {
            continue;
        };
        if stem.contains("-of-") {
            continue; // shard parts: fit uses the set total via sibling sums
        }
        let bytes = s.lfs.as_ref().and_then(|l| l.size).or(s.size).unwrap_or(0);
        if bytes == 0 {
            continue;
        }
        let quant = stem.rsplit('-').next().unwrap_or("unknown").to_uppercase();
        // KV estimate for the default ctx, assuming q8_0 KV when tight
        // (same trigger as the compiler's rule 6).
        let kv_f16 = kv_estimate_f16(default_ctx);
        // F105: mirror the compiler's rule-6 KV ladder (f16 -> q8 -> q4)
        // instead of the degenerate `kv_f16.min(kv_f16 / 2)`, which only
        // ever tested the q8 grade.
        let fits = bytes + kv_f16 <= vram_bytes
            || bytes + kv_f16 / 2 <= vram_bytes
            || bytes + kv_f16 / 4 <= vram_bytes;
        let recommended = if fits {
            default_ctx
        } else {
            // shrink ctx until KV fits alongside the weights
            let mut ctx = default_ctx;
            while ctx > 1024 && bytes + kv_estimate_f16(ctx) > vram_bytes {
                ctx /= 2;
            }
            ctx
        };
        // Same shrink loop with halved KV: what q8_0 KV buys (rule-6 grade).
        let mut ctx_q8 = default_ctx;
        while ctx_q8 > 1024 && bytes + kv_estimate_f16(ctx_q8) / 2 > vram_bytes {
            ctx_q8 /= 2;
        }
        rows.push(FitRow {
            quant,
            file: s.rfilename.clone(),
            bytes,
            fits_vram: fits,
            kv_bytes_at_default_ctx: kv_f16,
            recommended_ctx: recommended,
            recommended_ctx_q8: ctx_q8,
        });
    }
    rows.sort_by_key(|r| std::cmp::Reverse(r.bytes));
    rows
}

fn kv_estimate_f16(ctx: u32) -> u64 {
    // Without per-arch metadata pre-download, report a conservative
    // 1.5 GiB-per-16k KV allowance; the real number comes from GGUF at
    // load time (profile rule 6) and is shown in `pallama show`.
    1_572_864_000u64.saturating_mul(u64::from(ctx)) / 16_384
}

// ---------------------------------------------------------------------------
// Pull orchestration
// ---------------------------------------------------------------------------

pub struct Puller {
    pub dirs: PallamaDirs,
    pub client: HfClient,
    pub bus: EventBus,
}

/// RAII lockfile guard: released (removed) on drop, panic-safe.
#[derive(Debug)]
struct PullLock {
    path: PathBuf,
}

impl PullLock {
    fn acquire(dirs: &PallamaDirs, name: &str) -> Result<Self> {
        std::fs::create_dir_all(dirs.run_dir())?;
        let path = dirs.run_dir().join(format!("pull-{name}.lock"));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut f) => {
                use std::io::Write;
                let _ = writeln!(f, "{}", std::process::id());
                Ok(Self { path })
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                // The lockfile carries the owning pid: a dead owner means
                // the pull crashed (kill -9, power loss) and the lock is
                // stale — steal it immediately instead of blocking for the
                // 6h age fallback. Live owner -> refuse, naming the pid.
                let path_display = path.display().to_string();
                let owner = std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|raw| raw.trim().parse::<u32>().ok());
                let owner_alive = owner.is_some_and(pid_alive);
                if !owner_alive {
                    // Unknown/corrupt lockfile with no parseable pid falls
                    // back to the age heuristic so a corrupt file cannot
                    // wedge pulls forever either.
                    let age = std::fs::metadata(&path)
                        .and_then(|m| m.modified())
                        .ok()
                        .and_then(|t| t.elapsed().ok())
                        .unwrap_or_default();
                    if owner.is_none() && age < Duration::from_hours(6) {
                        return Err(anyhow!(
                            "pull already in progress for {name} (lockfile {path_display}, unreadable owner); delete it if you are sure no pull is running"
                        ));
                    }
                    let _ = std::fs::remove_file(&path);
                    return Self::acquire(dirs, name);
                }
                Err(anyhow!(
                    "pull already in progress for {name} by pid {} (lockfile {path_display})",
                    owner.unwrap_or_default()
                ))
            }
            Err(e) => Err(anyhow!("create lock {}: {e}", path.display())),
        }
    }
}

impl Drop for PullLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Is a process with this pid alive? Linux: `/proc` — no shell-out.
/// Other unixes: `kill -0` (signal 0 = existence probe, no delivery).
/// Windows: `tasklist` filter. Pid reuse can false-positive; the age
/// fallback and manual delete remain as escape hatches.
fn pid_alive(pid: u32) -> bool {
    #[cfg(target_os = "linux")]
    {
        std::path::Path::new(&format!("/proc/{pid}")).exists()
    }
    #[cfg(all(unix, not(target_os = "linux")))]
    {
        std::process::Command::new("kill")
            .arg("-0")
            .arg(pid.to_string())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    }
    #[cfg(windows)]
    {
        std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH"])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).contains(&pid.to_string()))
            .unwrap_or(false)
    }
}

impl Puller {
    /// Pull a model into the store. `target` = `owner/repo[:QUANT]` or a
    /// catalog short name.
    pub async fn pull(&self, target: &str) -> Result<ModelRow> {
        let parsed = parse_pull_target(target)?;
        let name = registry_name(&parsed.repo);
        let _lock = PullLock::acquire(&self.dirs, &name)?;
        self.pull_locked(&parsed, &name).await
    }

    async fn pull_locked(&self, target: &PullTarget, name: &str) -> Result<ModelRow> {
        let info = self.client.model_info(&target.repo).await?;
        let selected = select_files(&info.siblings, &target.quant)?;

        let models_dir = self.dirs.models_dir();
        std::fs::create_dir_all(&models_dir)?;

        let total_bytes: u64 = selected.shards.iter().map(|s| s.bytes).sum::<u64>()
            + selected.mmproj.as_ref().map_or(0, |m| m.bytes);
        let bar = indicatif::ProgressBar::new(total_bytes);
        bar.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("{msg} {bar:30} {bytes}/{total_bytes} ({eta})")
                .expect("valid template"),
        );
        bar.set_message(format!("pull {name}:{}", selected.quant));

        let mut last_publish = 0u64;
        let mut progress = |downloaded: u64, total: u64| {
            bar.set_position(downloaded);
            // Throttle bus traffic: ~every 16 MiB.
            if downloaded.saturating_sub(last_publish) >= 16 << 20 || downloaded == total {
                last_publish = downloaded;
                self.bus.publish(PallamaEvent::PullProgress {
                    name: name.to_string(),
                    downloaded,
                    total,
                });
            }
        };

        let mut shard_paths = Vec::new();
        for (i, shard) in selected.shards.iter().enumerate() {
            let mut progress_one = |d: u64, t: u64| {
                // Aggregate across files for the shared bar.
                let before: u64 = selected.shards[..i].iter().map(|s| s.bytes).sum();
                progress(before + d.min(t), total_bytes);
            };
            let dest = unique_dest(&models_dir, &shard.filename, &target.repo);
            self.client
                .download_file(&target.repo, shard, &dest, &mut progress_one)
                .await
                .inspect_err(|e| {
                    self.bus.publish(PallamaEvent::PullFailed {
                        name: name.to_string(),
                        error: e.to_string(),
                    });
                })?;
            shard_paths.push(dest);
        }
        let mmproj_dest = selected
            .mmproj
            .as_ref()
            .map(|mm| unique_dest(&models_dir, &mm.filename, &target.repo));
        if let (Some(mm), Some(dest)) = (&selected.mmproj, &mmproj_dest) {
            self.client
                .download_file(&target.repo, mm, dest, &mut progress)
                .await
                .inspect_err(|e| {
                    // F104: mmproj failure must reach /api/pull watchers
                    // like a shard failure does, not vanish via `?`.
                    self.bus.publish(PallamaEvent::PullFailed {
                        name: name.to_string(),
                        error: format!("mmproj: {e}"),
                    });
                })?;
        }
        bar.finish_and_clear();

        let (row, pull_warning) = build_model_row(
            name,
            target,
            &info,
            &selected,
            &shard_paths,
            mmproj_dest.as_ref(),
        )?;
        if let Some(w) = &pull_warning {
            tracing::warn!(model = %name, "{w}");
        }
        let store = Store::open(&self.dirs)?;
        store.upsert_model(&row)?;
        self.bus.publish(PallamaEvent::ModelPulled {
            name: name.to_string(),
            warning: pull_warning.clone(),
        });
        if selected.quant_fallback {
            tracing::warn!(
                "quant {} not found in {}; pulled {} instead",
                target.quant,
                target.repo,
                selected.quant
            );
        }
        Ok(row)
    }
}

/// Post-download metadata + store row. Health warning: `None` when the
/// header parses; a LOUD warning otherwise (the engine will most likely
/// refuse to load the file — the qwen3.5-9b quantizer-metadata class of
/// failure). GGUF header facts win over HF-provided metadata.
#[allow(clippy::too_many_arguments)] // cohesive pull-facts tuple; splitting hides the fallback chain
fn build_model_row(
    name: &str,
    target: &PullTarget,
    info: &HfModelInfo,
    selected: &SelectedFiles,
    shard_paths: &[PathBuf],
    mmproj_dest: Option<&PathBuf>,
) -> Result<(ModelRow, Option<String>)> {
    let pull_warning = gguf_health_warning(&shard_paths[0], &target.repo, &info.siblings);
    let gguf_meta = gguf::read_metadata_file(&shard_paths[0]).ok();
    let arch = gguf_meta
        .as_ref()
        .map(|m| m.architecture.clone())
        .or_else(|| info.gguf.as_ref().and_then(|g| g.architecture.clone()));
    let ctx_train = gguf_meta
        .as_ref()
        .and_then(|m| m.context_length)
        .or_else(|| info.gguf.as_ref().and_then(|g| g.context_length));

    let bytes: u64 = selected.shards.iter().map(|s| s.bytes).sum();
    Ok((
        ModelRow {
            name: name.to_string(),
            repo: target.repo.clone(),
            quant: selected.quant.clone(),
            path: shard_paths[0].display().to_string(),
            bytes: i64::try_from(bytes).unwrap_or(i64::MAX),
            sha256: selected.shards[0].sha256.clone(),
            mmproj_path: mmproj_dest.as_ref().map(|d| d.display().to_string()),
            shards: i64::try_from(selected.shards.len()).unwrap_or(i64::MAX),
            arch,
            params: Some(est_params(bytes, &selected.quant)),
            ctx_train: ctx_train.and_then(|c| i64::try_from(c).ok()),
            pulled_at: i64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())
                .unwrap_or(i64::MAX),
        },
        pull_warning,
    ))
}

/// Post-download GGUF health check: `None` when the header parses;
/// otherwise a warning naming the failure and the other quants of the
/// same repo (precheck for the load-refusal class of breakage).
#[must_use]
pub fn gguf_health_warning(path: &Path, repo: &str, siblings: &[HfSibling]) -> Option<String> {
    if gguf::read_metadata_file(path).is_ok() {
        return None;
    }
    let mut alts: Vec<String> = siblings
        .iter()
        .filter(|s| {
            let lower = s.rfilename.to_lowercase();
            let is_gguf = std::path::Path::new(&s.rfilename)
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
            is_gguf && !lower.contains("-of-")
        })
        .filter_map(|s| {
            let lower = s.rfilename.to_lowercase();
            let stem = lower.strip_suffix(".gguf")?;
            stem.rsplit('-').next().map(str::to_uppercase)
        })
        .collect();
    alts.sort();
    alts.dedup();
    Some(format!(
        "GGUF metadata unreadable in {} — the engine will likely refuse to load it; `pallama rm` and try another quant of {repo}: {}",
        path.display(),
        alts.join(", ")
    ))
}

fn unique_dest(dir: &Path, filename: &str, repo: &str) -> PathBuf {
    let flat = Path::new(filename);
    let leaf = flat.file_name().unwrap_or_default();
    let dest = dir.join(leaf);
    if !dest.exists() {
        return dest;
    }
    // Collision: disambiguate with repo-derived path parts so two repos
    // shipping the same leaf name never clobber each other. Subdir path
    // first (matches the repo's internal layout), then the repo slug for
    // flat filenames (e.g. two Qwen3.5-9B-Q4_K_M.gguf from different
    // repos — verified live: unsloth base vs unsloth MTP collide).
    if let Some(parent) = flat.parent() {
        if !parent.as_os_str().is_empty() {
            let slug: String = parent.to_string_lossy().replace(['/', '\\'], "--");
            return dir.join(format!("{slug}--{}", leaf.to_string_lossy()));
        }
    }
    let repo_slug = repo.replace(['/', '\\'], "--");
    dir.join(format!("{repo_slug}--{}", leaf.to_string_lossy()))
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn unit__unique_dest__flat_collision_disambiguates_by_repo() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        std::fs::write(dir.join("m.gguf"), b"x").unwrap();
        // No collision: leaf name kept verbatim.
        assert_eq!(
            unique_dest(dir, "fresh.gguf", "unsloth/Base-GGUF"),
            dir.join("fresh.gguf")
        );
        // Flat collision: repo slug prefix — two repos shipping the same
        // leaf never clobber each other (live case: unsloth base vs MTP
        // repos both ship Qwen3.5-9B-Q4_K_M.gguf).
        assert_eq!(
            unique_dest(dir, "m.gguf", "unsloth/Qwen3.5-9B-MTP-GGUF"),
            dir.join("unsloth--Qwen3.5-9B-MTP-GGUF--m.gguf")
        );
        // Subdir filename collision: internal path slug wins (unchanged).
        assert_eq!(
            unique_dest(dir, "org/repo/m.gguf", "unsloth/Other"),
            dir.join("org--repo--m.gguf")
        );
    }

    #[test]
    fn unit__quant_tokens__real_filename_shapes() {
        // Shapes harvested live from HF nanbeige search results.
        let files = [
            "Nanbeige4.2-3B-Q4_K_M.gguf",               // dash separator
            "nanbeige4.1-3b-q8_0.gguf",                 // lowercase repo style
            "nanbeige-16b-base-32k.Q2_K.gguf",          // dot separator
            "Nanbeige4.2-3B-BF16.gguf",
            "Parable-Nanbeige4.2-3B-Claude-Fable-5-heretic.i1-IQ1_M.gguf",
            "model-00001-of-00002.gguf",                // shard tail: digits only
            "mmproj-model-F16.gguf",                    // projector: excluded
            "README.md",                                // not gguf
            "Nanbeige4-3B-Thinking-2511.gguf",          // tag tails: not quants
        ];
        assert_eq!(
            quant_tokens(files),
            ["BF16", "IQ1_M", "Q2_K", "Q4_K_M", "Q8_0"]
        );
    }

    #[test]
    fn unit__quant_tokens__dedups_case_insensitively() {
        let files = ["m.Q8_0.gguf", "m-q8_0.gguf"];
        assert_eq!(quant_tokens(files), ["Q8_0"]);
    }

    #[test]
    fn unit__quant_tokens__no_gguf_files_is_empty() {
        assert!(quant_tokens(["README.md", "config.json"]).is_empty());
    }

    #[test]
    fn unit__gguf_health_warning__bad_header_names_alternatives() {
        let tmp = tempfile::tempdir().unwrap();
        let bad = tmp.path().join("bad.gguf");
        std::fs::write(&bad, b"not a gguf at all").unwrap();
        let sibs = vec![
            sib("m-Q4_K_M.gguf", 1, None),
            sib("m-Q8_0.gguf", 1, None),
            sib("m-00001-of-00002.gguf", 1, None), // shard: excluded
            sib("README.md", 1, None),             // not gguf: excluded
        ];
        let w = gguf_health_warning(&bad, "o/m", &sibs).expect("warning");
        assert!(w.contains("refuse to load"), "{w}");
        assert!(w.contains("o/m"), "{w}");
        assert!(w.contains("Q4_K_M") && w.contains("Q8_0"), "{w}");
        assert!(!w.contains("README"), "{w}");
    }

    #[test]
    fn unit__gguf_health_warning__valid_header_is_none() {
        // Minimal valid GGUF v3 header with one kv (architecture string).
        let tmp = tempfile::tempdir().unwrap();
        let good = tmp.path().join("good.gguf");
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        let k = "general.architecture";
        b.extend_from_slice(&(k.len() as u64).to_le_bytes());
        b.extend_from_slice(k.as_bytes());
        b.extend_from_slice(&8u32.to_le_bytes());
        let v = "qwen3";
        b.extend_from_slice(&(v.len() as u64).to_le_bytes());
        b.extend_from_slice(v.as_bytes());
        std::fs::write(&good, b).unwrap();
        assert_eq!(gguf_health_warning(&good, "o/m", &[]), None);
    }

    fn sib(name: &str, size: u64, sha: Option<&str>) -> HfSibling {
        HfSibling {
            rfilename: name.to_string(),
            size: Some(size),
            lfs: sha.map(|s| HfLfs {
                sha256: s.to_string(),
                size: Some(size),
            }),
        }
    }

    fn sibling_json(name: &str, size: u64, sha: &str) -> serde_json::Value {
        serde_json::json!({
            "rfilename": name,
            "size": size,
            "lfs": {"sha256": sha, "size": size, "pointerSize": 135}
        })
    }

    #[test]
    fn unit__registry_name__strips_gguf_and_lowercases() {
        assert_eq!(
            registry_name("Qwen/Qwen2.5-0.5B-Instruct-GGUF"),
            "qwen2.5-0.5b-instruct"
        );
        assert_eq!(registry_name("ggml-org/Qwen3-0.6B-GGUF"), "qwen3-0.6b");
        assert_eq!(registry_name("a/b"), "b");
    }

    #[test]
    fn unit__parse_target__explicit_repo_with_quant() {
        let t = parse_pull_target("Qwen/Qwen2.5-0.5B-Instruct-GGUF:q8_0").unwrap();
        assert_eq!(t.repo, "Qwen/Qwen2.5-0.5B-Instruct-GGUF");
        assert_eq!(t.quant, "Q8_0");
    }

    #[test]
    fn unit__parse_target__catalog_short_name() {
        let t = parse_pull_target("qwen3-0.6b").unwrap();
        assert_eq!(t.repo, "ggml-org/Qwen3-0.6B-GGUF");
        assert_eq!(t.quant, "Q4_K_M");
    }

    #[test]
    fn unit__parse_target__catalog_prefix_resolves() {
        let t = parse_pull_target("qwen3-coder").unwrap();
        assert!(t.repo.starts_with("unsloth/Qwen3-Coder"));
    }

    #[test]
    fn unit__select_files__quant_match_case_insensitive() {
        let sibs = vec![
            sib("model-fp16.gguf", 1000, None),
            sib("model-Q4_K_M.gguf", 400, None),
        ];
        let sel = select_files(&sibs, "q4_k_m").unwrap();
        assert_eq!(sel.shards.len(), 1);
        assert_eq!(sel.shards[0].filename, "model-Q4_K_M.gguf");
        assert!(!sel.quant_fallback);
    }

    #[test]
    fn unit__select_files__shard_set_enumerated_in_order() {
        let sibs = vec![
            sib("big-q4_k_m-00002-of-00002.gguf", 700, None),
            sib("big-q4_k_m-00001-of-00002.gguf", 700, None),
            sib("big-q8_0.gguf", 1200, None),
        ];
        let sel = select_files(&sibs, "Q4_K_M").unwrap();
        assert_eq!(sel.shards.len(), 2, "both shards queued");
        assert_eq!(sel.shards[0].filename, "big-q4_k_m-00001-of-00002.gguf");
        assert_eq!(sel.shards[1].filename, "big-q4_k_m-00002-of-00002.gguf");
    }

    #[test]
    fn unit__select_files__mmproj_detected() {
        let sibs = vec![
            sib("model-q4_k_m.gguf", 400, None),
            sib("mmproj-model-f16.gguf", 50, None),
            sib("mmproj-model-q8_0.gguf", 30, None),
        ];
        let sel = select_files(&sibs, "Q4_K_M").unwrap();
        assert_eq!(
            sel.mmproj.as_ref().unwrap().filename,
            "mmproj-model-q8_0.gguf"
        );
    }

    #[test]
    fn unit__select_files__default_quant_fallback_smallest() {
        let sibs = vec![
            sib("model-q8_0.gguf", 900, None),
            sib("model-q3_k_m.gguf", 300, None),
        ];
        let sel = select_files(&sibs, "Q4_K_M").unwrap();
        assert!(sel.quant_fallback);
        assert_eq!(sel.quant, "Q3_K_M");
        assert_eq!(sel.shards[0].filename, "model-q3_k_m.gguf");
    }

    #[test]
    fn unit__select_files__no_gguf__named_error() {
        let sibs = vec![sib("README.md", 10, None)];
        let err = select_files(&sibs, "Q4_K_M").unwrap_err();
        assert!(err.to_string().contains("no .gguf"));
    }

    #[test]
    fn unit__est_params__sane_order_of_magnitude() {
        // 0.5B-class Q4_K_M file ~ 400 MiB
        let p = est_params(400 * 1024 * 1024, "Q4_K_M");
        assert!(p > 0.3 && p < 1.2, "got {p}B");
    }

    #[test]
    fn unit__allowlist__known_hosts_only() {
        assert!(is_allowed_download_host("huggingface.co"));
        assert!(is_allowed_download_host("cdn-lfs.huggingface.co"));
        assert!(is_allowed_download_host("cas-bridge.xethub.hf.co"));
        assert!(is_allowed_download_host("transfer.xethub.hf.co"));
        assert!(!is_allowed_download_host("evil.example.com"));
        assert!(!is_allowed_download_host("huggingface.co.evil.com"));
    }

    // ------------------------------------------------------------------
    // wiremock integration
    // ------------------------------------------------------------------

    fn payload(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    #[tokio::test]
    async fn integration__pull_single_file__downloads_verifies_stores() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = PallamaDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        dirs.ensure().unwrap();

        let content = b"gguf-bytes-here".to_vec();
        let sha = payload(&content);
        let api = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/owner/m-repo"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "owner/m-repo",
                "siblings": [ sibling_json("m-repo-q4_k_m.gguf", content.len() as u64, &sha) ]
            })))
            .mount(&api)
            .await;
        let dl = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/owner/m-repo/resolve/main/m-repo-q4_k_m.gguf"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(content.clone()))
            .mount(&dl)
            .await;

        let client = HfClient::with_bases(
            &api.uri(),
            &dl.uri(),
            None,
            vec![
                api.uri().trim_start_matches("http://").to_string(),
                dl.uri().trim_start_matches("http://").to_string(),
            ],
        )
        .unwrap();
        let puller = Puller {
            dirs: dirs.clone(),
            client,
            bus: EventBus::default(),
        };
        let row = puller.pull("owner/m-repo:Q4_K_M").await.unwrap();

        assert_eq!(row.name, "m-repo");
        assert_eq!(row.quant, "Q4_K_M");
        let path = Path::new(&row.path);
        assert!(path.exists(), "file must land at stable path");
        assert_eq!(std::fs::read(path).unwrap(), content);
        assert!(!Path::new(&format!("{}.part", row.path)).exists());

        let store = Store::open(&dirs).unwrap();
        assert_eq!(store.get_model("m-repo").unwrap().unwrap().quant, "Q4_K_M");
    }

    #[tokio::test]
    async fn integration__pull_sha_mismatch__partial_deleted_and_error() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = PallamaDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        dirs.ensure().unwrap();
        let content = b"corrupt-me".to_vec();
        let wrong_sha = payload(b"something-else");
        let api = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/o/r"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "siblings": [ sibling_json("r-q4_k_m.gguf", content.len() as u64, &wrong_sha) ]
            })))
            .mount(&api)
            .await;
        let dl = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/o/r/resolve/main/r-q4_k_m.gguf"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(content))
            .mount(&dl)
            .await;
        let client = HfClient::with_bases(
            &api.uri(),
            &dl.uri(),
            None,
            vec![host_of(&api.uri()), host_of(&dl.uri())],
        )
        .unwrap();
        let puller = Puller {
            dirs: dirs.clone(),
            client,
            bus: EventBus::default(),
        };
        let err = puller.pull("o/r").await.unwrap_err();
        assert!(err.to_string().contains("sha256 mismatch"), "{err}");
        assert!(
            dirs.models_dir().read_dir().unwrap().count() == 0,
            "no partial left behind"
        );
        assert!(Store::open(&dirs)
            .unwrap()
            .list_models()
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn integration__resume_from_part_file__range_and_append() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = PallamaDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        dirs.ensure().unwrap();
        let full = b"0123456789abcdef".to_vec();
        let sha = payload(&full);
        let prefix = &full[..8];

        let api = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/o/r"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "siblings": [ sibling_json("r-q4_k_m.gguf", full.len() as u64, &sha) ]
            })))
            .mount(&api)
            .await;
        let dl = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/o/r/resolve/main/r-q4_k_m.gguf"))
            .and(wiremock::matchers::header("Range", "bytes=8-"))
            .respond_with(
                ResponseTemplate::new(206)
                    .insert_header("Content-Range", "bytes 8-15/16")
                    .set_body_bytes(full[8..].to_vec()),
            )
            .mount(&dl)
            .await;

        // Pre-existing partial.
        std::fs::write(dirs.models_dir().join("r-q4_k_m.gguf.part"), prefix).unwrap();

        let client = HfClient::with_bases(
            &api.uri(),
            &dl.uri(),
            None,
            vec![host_of(&api.uri()), host_of(&dl.uri())],
        )
        .unwrap();
        let puller = Puller {
            dirs,
            client,
            bus: EventBus::default(),
        };
        let row = puller.pull("o/r").await.unwrap();
        let got = std::fs::read(&row.path).unwrap();
        assert_eq!(got, full, "resumed file must equal full content");
    }

    #[tokio::test]
    async fn integration__redirect_to_non_allowlisted_host__blocked() {
        let api = MockServer::start().await;
        let dl = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/o/r"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "siblings": [ sibling_json("r.gguf", 4, &payload(b"abcd")) ]
            })))
            .mount(&api)
            .await;
        Mock::given(method("GET"))
            .and(path("/o/r/resolve/main/r.gguf"))
            .respond_with(
                ResponseTemplate::new(302)
                    .insert_header("Location", "http://evil.example.com/exfil"),
            )
            .mount(&dl)
            .await;

        let client = HfClient::with_bases(
            &api.uri(),
            &dl.uri(),
            None,
            vec![host_of(&api.uri()), host_of(&dl.uri())],
        )
        .unwrap();
        let plan = FilePlan {
            filename: "r.gguf".into(),
            bytes: 4,
            sha256: None,
        };
        let err = client
            .download_file("o/r", &plan, Path::new("/tmp/never-r.gguf"), |_, _| {})
            .await
            .unwrap_err();
        let chain = format!("{err:#}");
        assert!(chain.contains("evil.example.com"), "{chain}");
    }

    #[tokio::test]
    async fn integration__token_only_on_first_party_host() {
        // huggingface-mock redirects to cdn-mock; assert token arrives at
        // neither the CDN nor leaks via any second hop.
        let first = MockServer::start().await;
        let cdn = MockServer::start().await;
        let body = b"tok".to_vec();
        Mock::given(method("GET"))
            .and(path("/api/models/o/r"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "siblings": [ sibling_json("r.gguf", 3, &payload(&body)) ]
            })))
            .mount(&first)
            .await;
        // First-party download hop redirects to CDN.
        Mock::given(method("GET"))
            .and(path("/o/r/resolve/main/r.gguf"))
            .respond_with(
                ResponseTemplate::new(302)
                    .insert_header("Location", format!("{}/lfs/r.gguf", cdn.uri())),
            )
            .expect(1)
            .mount(&first)
            .await;
        Mock::given(method("GET"))
            .and(path("/lfs/r.gguf"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body.clone()))
            .expect(1)
            .mount(&cdn)
            .await;

        let client = HfClient::with_bases(
            &first.uri(),
            &first.uri(),
            Some("SECRET-TOKEN".into()),
            vec![host_of(&first.uri()), host_of(&cdn.uri())],
        )
        .unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("r.gguf");
        let plan = FilePlan {
            filename: "r.gguf".into(),
            bytes: 3,
            sha256: Some(payload(&body.clone())),
        };
        client
            .download_file("o/r", &plan, &dest, |_, _| {})
            .await
            .unwrap();

        // Inspect what the CDN actually received.
        let requests = cdn.received_requests().await.unwrap();
        let cdn_req = requests
            .iter()
            .find(|r| r.url.path() == "/lfs/r.gguf")
            .expect("cdn hit");
        assert!(
            cdn_req.headers.get("authorization").is_none(),
            "token must NEVER reach a CDN host"
        );
    }

    #[tokio::test]
    async fn integration__gated_repo__clear_error() {
        let api = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/o/gated"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&api)
            .await;
        let client =
            HfClient::with_bases(&api.uri(), &api.uri(), None, vec![host_of(&api.uri())]).unwrap();
        let err = client.model_info("o/gated").await.unwrap_err();
        assert!(
            err.to_string().contains("gated") && err.to_string().contains("HF_TOKEN"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn integration__lock_contention__second_pull_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = PallamaDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        dirs.ensure().unwrap();
        let held = PullLock::acquire(&dirs, "m").unwrap();
        let err = PullLock::acquire(&dirs, "m").unwrap_err();
        assert!(err.to_string().contains("already in progress"), "{err}");
        drop(held);
        assert!(
            PullLock::acquire(&dirs, "m").is_ok(),
            "lock released on drop"
        );
    }

    #[tokio::test]
    async fn integration__lock_stale__dead_owner_pid_stolen_immediately() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = PallamaDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        dirs.ensure().unwrap();
        // A real, now-exited process: its pid is genuinely dead, no race.
        let mut child =
            std::process::Command::new(std::env::var("EXE_TRUE").unwrap_or_else(|_| {
                if cfg!(windows) {
                    "cmd".into()
                } else {
                    "true".into()
                }
            }))
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let dead_pid = child.id();
        let _ = child.wait();
        let lock = dirs.run_dir().join("pull-m.lock");
        std::fs::write(&lock, format!("{dead_pid}\n")).unwrap();
        // Fresh file, well under 6h: must STILL be stolen because the
        // owner is provably dead.
        let got = PullLock::acquire(&dirs, "m");
        assert!(got.is_ok(), "dead-owner lock stolen: {}", got.unwrap_err());
    }

    #[tokio::test]
    async fn integration__lock_live__owner_pid_named_in_error() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = PallamaDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        dirs.ensure().unwrap();
        let lock = dirs.run_dir().join("pull-m.lock");
        std::fs::write(&lock, format!("{}\n", std::process::id())).unwrap();
        let err = PullLock::acquire(&dirs, "m").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("already in progress"), "{msg}");
        assert!(msg.contains(&std::process::id().to_string()), "{msg}");
        std::fs::remove_file(&lock).unwrap();
    }

    #[tokio::test]
    async fn integration__pull_shard_set__both_parts_and_count_recorded() {
        let tmp = tempfile::tempdir().unwrap();
        let dirs = PallamaDirs {
            config_dir: tmp.path().join("cfg"),
            data_dir: tmp.path().join("data"),
        };
        dirs.ensure().unwrap();
        let p1 = b"part-one--".to_vec();
        let p2 = b"part-two".to_vec();
        let api = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/o/big"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "siblings": [
                    sibling_json("big-q4_k_m-00001-of-00002.gguf", p1.len() as u64, &payload(&p1)),
                    sibling_json("big-q4_k_m-00002-of-00002.gguf", p2.len() as u64, &payload(&p2)),
                ]
            })))
            .mount(&api)
            .await;
        let dl = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/o/big/resolve/main/big-q4_k_m-00001-of-00002.gguf"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(p1))
            .mount(&dl)
            .await;
        Mock::given(method("GET"))
            .and(path("/o/big/resolve/main/big-q4_k_m-00002-of-00002.gguf"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(p2))
            .mount(&dl)
            .await;

        let client = HfClient::with_bases(
            &api.uri(),
            &dl.uri(),
            None,
            vec![host_of(&api.uri()), host_of(&dl.uri())],
        )
        .unwrap();
        let puller = Puller {
            dirs: dirs.clone(),
            client,
            bus: EventBus::default(),
        };
        let row = puller.pull("o/big:Q4_K_M").await.unwrap();
        assert_eq!(row.shards, 2);
        assert!(
            row.path.ends_with("-00001-of-00002.gguf"),
            "first shard is the launch path"
        );
        assert!(Path::new(&row.path).exists());
        let second = dirs.models_dir().join("big-q4_k_m-00002-of-00002.gguf");
        assert!(second.exists(), "second shard stored alongside");
    }

    fn host_of(uri: &str) -> String {
        uri.rsplit_once("://")
            .map_or(uri, |(_, h)| h)
            .rsplit_once(':')
            .map_or(uri, |(h, _)| h)
            .to_string()
    }
}
