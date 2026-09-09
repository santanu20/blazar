//! GitHub release client for upstream llama.cpp binaries.
//! API truth verified live (2026-09-05): releases carry `bNNNNN` tags
//! (prereleases, 27 assets) plus `vX.Y.Z` stable tags whose single asset
//! `nightly-tag.txt` contains the current b-tag. Assets are named
//! `llama-{tag}-bin-{asset}.tar.gz|.zip` and expose `digest:
//! sha256:...` — verified before extraction.

use anyhow::{anyhow, Context, Result};
use pallama_core::config::UpdateChannel;
use serde::Deserialize;
use sha2::{Digest, Sha256};

pub const LLAMA_CPP_REPO: &str = "ggml-org/llama.cpp";

#[derive(Debug, Clone, Deserialize)]
pub struct GhAsset {
    pub name: String,
    #[serde(default)]
    pub digest: Option<String>,
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(default)]
    pub browser_download_url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GhRelease {
    pub tag_name: String,
    #[serde(default)]
    pub prerelease: bool,
    #[serde(default)]
    pub assets: Vec<GhAsset>,
    /// ISO-8601 publish time, e.g. "2026-09-07T06:49:18Z". GitHub serves
    /// release metadata before assets finish uploading (~2 min stagger);
    /// freshness decides whether a missing asset is worth waiting for.
    #[serde(default)]
    pub published_at: Option<String>,
}

impl GhRelease {
    /// Publish time as unix seconds, if present and well-formed.
    #[must_use]
    pub fn published_epoch(&self) -> Option<i64> {
        iso_to_epoch(self.published_at.as_deref()?)
    }
}

/// Parse GitHub's Zulu ISO-8601 ("YYYY-MM-DDTHH:MM:SS[.fff]Z") to unix
/// seconds. No datetime dependency: days-from-civil (Howard Hinnant).
/// Names mirror the published algorithm.
#[allow(clippy::many_single_char_names, clippy::unreadable_literal)]
fn iso_to_epoch(iso: &str) -> Option<i64> {
    let (date, rest) = iso.split_once('T')?;
    let time = rest.trim_end_matches('Z');
    let time = time.split('.').next().unwrap_or(time);
    let mut d_parts = date.split('-');
    let (y, m, d) = (d_parts.next()?, d_parts.next()?, d_parts.next()?);
    let mut t_parts = time.split(':');
    let (h, mi, s) = (t_parts.next()?, t_parts.next()?, t_parts.next()?);
    let (y, m, d): (i64, i64, i64) = (y.parse().ok()?, m.parse().ok()?, d.parse().ok()?);
    let (h, mi, s): (i64, i64, i64) = (h.parse().ok()?, mi.parse().ok()?, s.parse().ok()?);
    let (y_adj, m_adj) = if m <= 2 { (y - 1, m + 9) } else { (y, m - 3) };
    let era = if y_adj >= 0 { y_adj } else { y_adj - 399 } / 400;
    let yoe = y_adj - era * 400;
    let doy = (153 * m_adj + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    Some(days * 86_400 + h * 3_600 + mi * 60 + s)
}

pub struct GhClient {
    http: reqwest::Client,
    base: reqwest::Url,
    token: Option<String>,
}

#[must_use]
pub fn btag_number(tag: &str) -> Option<u64> {
    // Suffix-tolerant: source-built engines carry a provenance suffix
    // (`b10816-cuda`) but compare by their leading upstream build number.
    tag.strip_prefix('b')?
        .split(|c: char| !c.is_ascii_digit())
        .next()?
        .parse()
        .ok()
}

/// Same-upstream-build comparison: b-tags compare by build number (so
/// `b10816-cuda` == `b10816`); anything else falls back to equality.
#[must_use]
pub fn same_build(a: &str, b: &str) -> bool {
    match (btag_number(a), btag_number(b)) {
        (Some(x), Some(y)) => x == y,
        _ => a == b,
    }
}

impl GhClient {
    pub fn new(token: Option<String>) -> Result<Self> {
        // PALLAMA_GH_BASE: mirrors/tests override the GitHub API base for
        // the llama.cpp engine lane (same knob class as the installer's
        // PALLAMA_INSTALL_BASE_URL). Unset = the real API.
        let base =
            std::env::var("PALLAMA_GH_BASE").unwrap_or_else(|_| "https://api.github.com".into());
        Self::with_base(&base, token)
    }

    pub fn with_base(base: &str, token: Option<String>) -> Result<Self> {
        // An empty env var (GH_TOKEN="") must degrade to anonymous —
        // "Bearer " is an invalid credential and 401s where no-token
        // would succeed.
        let token = token.filter(|t| !t.trim().is_empty());
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("accept", "application/vnd.github+json".parse()?);
        headers.insert("user-agent", "pallama (llama.cpp orchestrator)".parse()?);
        if let Some(t) = &token {
            headers.insert("x-github-api-version", "2022-11-28".parse()?);
            let _ = t; // used per-request below
        }
        let http = reqwest::Client::builder()
            .default_headers(headers)
            // Release assets are 30-400 MB: connect/read timeouts, no total cap.
            .connect_timeout(std::time::Duration::from_secs(30))
            .read_timeout(std::time::Duration::from_mins(2))
            .build()
            .context("build GitHub client")?;
        Ok(Self {
            http,
            base: reqwest::Url::parse(base)?,
            token,
        })
    }

    fn auth(&self, rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.token {
            Some(t) => rb.bearer_auth(t),
            None => rb,
        }
    }

    /// List recent releases (b-tags and v-tags together).
    pub async fn list_releases(&self) -> Result<Vec<GhRelease>> {
        self.list_releases_repo(LLAMA_CPP_REPO).await
    }

    /// `list_releases` for an arbitrary repo (whisper.cpp, pallama self, ...).
    pub async fn list_releases_repo(&self, repo: &str) -> Result<Vec<GhRelease>> {
        let url = self
            .base
            .join(&format!("repos/{repo}/releases?per_page=30"))
            .unwrap();
        let resp = self
            .auth(self.http.get(url.clone()))
            .send()
            .await
            .context("GitHub releases request failed")?;
        match resp.status() {
            reqwest::StatusCode::OK => {}
            reqwest::StatusCode::FORBIDDEN | reqwest::StatusCode::TOO_MANY_REQUESTS => {
                return Err(anyhow!(
                    "GitHub API rate limited ({}). Set GH_TOKEN for 5000 req/hr",
                    resp.status()
                ));
            }
            other => return Err(anyhow!("GitHub releases API {other}")),
        }
        let releases: Vec<GhRelease> = resp.json().await.context("decode releases JSON")?;
        Ok(releases)
    }

    /// Single release for an arbitrary repo: `releases/latest` or
    /// `releases/tags/{version}`. Used by `pallama upgrade` (self-update).
    pub async fn release_by(&self, repo: &str, version: Option<&str>) -> Result<GhRelease> {
        let path = match version {
            Some(v) => format!("repos/{repo}/releases/tags/{v}"),
            None => format!("repos/{repo}/releases/latest"),
        };
        let url = self.base.join(&path).unwrap();
        let resp = self
            .auth(self.http.get(url))
            .send()
            .await
            .context("GitHub release request failed")?;
        match resp.status() {
            reqwest::StatusCode::OK => {}
            reqwest::StatusCode::FORBIDDEN | reqwest::StatusCode::TOO_MANY_REQUESTS => {
                return Err(anyhow!(
                    "GitHub API rate limited ({}). Set GH_TOKEN",
                    resp.status()
                ));
            }
            other => return Err(anyhow!("GitHub release API {other} for {path}")),
        }
        let release: GhRelease = resp.json().await.context("decode release JSON")?;
        Ok(release)
    }

    /// Newest `bNNNNN` release by build number.
    pub async fn latest_b_release(&self) -> Result<GhRelease> {
        let releases = self.list_releases().await?;
        releases
            .into_iter()
            .filter(|r| btag_number(&r.tag_name).is_some())
            .max_by_key(|r| btag_number(&r.tag_name).unwrap_or(0))
            .ok_or_else(|| anyhow!("no b-tagged llama.cpp releases found"))
    }

    /// Resolve the llama.cpp target for an update channel.
    /// `Latest` = newest b-tag (the prerelease firehose); `Stable` = the
    /// newest vX.Y.Z via `releases/latest`, dereferenced through its
    /// `nightly-tag.txt` to the concrete b-tag it ships.
    pub async fn channel_b_release(&self, channel: UpdateChannel) -> Result<GhRelease> {
        match channel {
            UpdateChannel::Latest => self.latest_b_release().await,
            UpdateChannel::Stable => {
                let stable = self.release_by(LLAMA_CPP_REPO, None).await?;
                if stable.tag_name.starts_with('v') {
                    self.resolve_tag(&stable.tag_name).await
                } else {
                    Ok(stable)
                }
            }
        }
    }

    /// Resolve the target release of an arbitrary repo (whisper.cpp,
    /// pallama self) for an update channel. These upstreams publish
    /// through GitHub's `/releases/latest` only — no prerelease
    /// firehose — so both channels coincide; the knob is meaningful for
    /// the llama.cpp lane (`channel_b_release`).
    pub async fn channel_repo_release(
        &self,
        repo: &str,
        channel: UpdateChannel,
    ) -> Result<GhRelease> {
        let _ = channel;
        self.release_by(repo, None).await
    }

    /// Resolve a user-provided tag: `bNNNN` verbatim; `vX.Y.Z` reads its
    /// `nightly-tag.txt` asset to find the current b-tag.
    pub async fn resolve_tag(&self, tag: &str) -> Result<GhRelease> {
        if tag.starts_with('v') {
            let url = self
                .base
                .join(&format!("repos/{LLAMA_CPP_REPO}/releases/tags/{tag}"))
                .unwrap();
            let resp = self
                .auth(self.http.get(url.clone()))
                .send()
                .await
                .context("GitHub release-by-tag request failed")?;
            if !resp.status().is_success() {
                return Err(anyhow!("release {tag} not found ({})", resp.status()));
            }
            let rel: GhRelease = resp.json().await.context("decode release")?;
            let nightly = rel
                .assets
                .iter()
                .find(|a| a.name == "nightly-tag.txt")
                .ok_or_else(|| anyhow!("release {tag} has no nightly-tag.txt asset"))?;
            let txt = self
                .download_asset_bytes(nightly)
                .await
                .context("download nightly-tag.txt")?;
            let inner = String::from_utf8_lossy(&txt).trim().to_string();
            return self.release_by_tag(&inner).await;
        }
        self.release_by_tag(tag).await
    }

    pub async fn release_by_tag(&self, tag: &str) -> Result<GhRelease> {
        self.release_by_tag_repo(LLAMA_CPP_REPO, tag).await
    }

    /// `release_by_tag` for an arbitrary repo (mistral.rs engine lane).
    pub async fn release_by_tag_repo(&self, repo: &str, tag: &str) -> Result<GhRelease> {
        let url = self
            .base
            .join(&format!("repos/{repo}/releases/tags/{tag}"))
            .unwrap();
        let resp = self
            .auth(self.http.get(url.clone()))
            .send()
            .await
            .context("GitHub release-by-tag request failed")?;
        match resp.status() {
            reqwest::StatusCode::OK => Ok(resp.json().await.context("decode release")?),
            reqwest::StatusCode::NOT_FOUND => Err(anyhow!("{repo} release {tag} not found")),
            other => Err(anyhow!("GitHub API {other} for {repo} tag {tag}")),
        }
    }

    /// Newest mistral.rs release by semver (`vtag_semver`, not string
    /// order — v0.10.0 > v0.9.3).
    pub async fn latest_mistralrs_release(&self) -> Result<GhRelease> {
        let releases = self.list_releases_repo(MISTRALRS_REPO).await?;
        releases
            .into_iter()
            .filter(|r| vtag_semver(&r.tag_name).is_some())
            .max_by_key(|r| vtag_semver(&r.tag_name).unwrap_or((0, 0, 0)))
            .ok_or_else(|| anyhow!("no v-tagged mistral.rs releases found"))
    }

    /// Download an asset fully into memory, verifying its sha256 digest
    /// when the release metadata provides one. Assets are ≤ ~400 MB.
    pub async fn download_asset_bytes(&self, asset: &GhAsset) -> Result<Vec<u8>> {
        let url = reqwest::Url::parse(&asset.browser_download_url)
            .with_context(|| format!("asset url {:?}", asset.name))?;
        let mut req = self.http.get(url.clone());
        if url.host_str() == Some("api.github.com") {
            req = self.auth(req);
        }
        let resp = req.send().await.context("asset download failed")?;
        if !resp.status().is_success() {
            return Err(anyhow!(
                "asset download {} returned {}",
                asset.name,
                resp.status()
            ));
        }
        let bytes = resp.bytes().await.context("read asset body")?;
        if let Some(digest) = &asset.digest {
            let expected = digest
                .strip_prefix("sha256:")
                .unwrap_or(digest)
                .to_lowercase();
            let got = format!("{:x}", Sha256::digest(&bytes));
            if got != expected {
                return Err(anyhow!(
                    "sha256 mismatch for {}: expected {expected}, got {got}",
                    asset.name
                ));
            }
        } else {
            tracing::warn!(
                "asset {} has no digest in release metadata; skipping sha verify",
                asset.name
            );
        }
        Ok(bytes.to_vec())
    }

    /// Stream an asset to `dest` with incremental sha256 — for GiB-class
    /// assets (mistral.rs CUDA prebuilts are 0.8-1.1 GiB) that must not
    /// be buffered whole in memory. The client's read-timeout is an
    /// idle-gap cap, not a total deadline, so slow links survive.
    /// Returns bytes written.
    pub async fn download_asset_file(
        &self,
        asset: &GhAsset,
        dest: &std::path::Path,
    ) -> Result<u64> {
        let url = reqwest::Url::parse(&asset.browser_download_url)
            .with_context(|| format!("asset url {:?}", asset.name))?;
        let mut req = self.http.get(url.clone());
        if url.host_str() == Some("api.github.com") {
            req = self.auth(req);
        }
        let resp = req.send().await.context("asset download failed")?;
        if !resp.status().is_success() {
            return Err(anyhow!(
                "asset download {} returned {}",
                asset.name,
                resp.status()
            ));
        }
        if let Some(len) = resp.content_length() {
            tracing::info!(
                "downloading {} ({} MiB) to {}",
                asset.name,
                len / (1024 * 1024),
                dest.display()
            );
        }
        let mut file =
            std::fs::File::create(dest).with_context(|| format!("create {}", dest.display()))?;
        let mut hasher = Sha256::new();
        let mut total: u64 = 0;
        let mut resp = resp;
        while let Some(chunk) = resp.chunk().await.context("read asset stream")? {
            use std::io::Write;
            file.write_all(&chunk).context("write asset chunk")?;
            hasher.update(&chunk);
            total += chunk.len() as u64;
        }
        if let Some(digest) = &asset.digest {
            let expected = digest
                .strip_prefix("sha256:")
                .unwrap_or(digest)
                .to_lowercase();
            let got = format!("{:x}", hasher.finalize());
            if got != expected {
                return Err(anyhow!(
                    "sha256 mismatch for {}: expected {expected}, got {got}",
                    asset.name
                ));
            }
        } else {
            tracing::warn!(
                "asset {} has no digest in release metadata; skipping sha verify",
                asset.name
            );
        }
        Ok(total)
    }
}

/// One asset preference entry. `Versioned` matches any
/// `llama-{tag}-bin-{prefix}{X.Y}{suffix}.{ext}` and picks the highest
/// version, so upstream bumps (cuda-13.3 -> 13.4, rocm-10.0 -> 11.0)
/// keep working without a pallama release.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Candidate {
    Exact(&'static str),
    Versioned {
        prefix: &'static str,
        suffix: &'static str,
    },
    Cpu(&'static str),
}

/// Ordered asset preference for this machine. `Cpu` entries are
/// last-resort: they resolve only when every GPU variant is absent and
/// the caller warns loudly (never a silent CPU fallback).
#[must_use]
pub fn pick_asset(
    os: &str,
    arch: &str,
    vendor: Option<crate::engine::manifest::Vendor>,
) -> Vec<Candidate> {
    use crate::engine::manifest::Vendor;
    use Candidate::{Cpu, Exact, Versioned};
    match (os, arch) {
        ("linux", "x86_64" | "x64" | "amd64") => match vendor {
            Some(Vendor::Nvidia) => vec![Exact("ubuntu-vulkan-x64"), Cpu("ubuntu-x64")],
            Some(Vendor::Amd) => vec![
                Versioned {
                    prefix: "ubuntu-rocm-",
                    suffix: "-x64",
                },
                Exact("ubuntu-vulkan-x64"),
                Cpu("ubuntu-x64"),
            ],
            Some(Vendor::Intel) => vec![
                Exact("ubuntu-sycl-fp16-x64"),
                Exact("ubuntu-vulkan-x64"),
                Cpu("ubuntu-x64"),
            ],
            _ => vec![Cpu("ubuntu-x64")],
        },
        ("linux", "aarch64" | "arm64") => vec![Exact("ubuntu-vulkan-arm64"), Cpu("ubuntu-arm64")],
        ("macos", _) => vec![Exact("macos-arm64"), Exact("macos-x64")],
        ("windows", "x86_64" | "x64" | "amd64") => match vendor {
            Some(Vendor::Nvidia) => vec![
                Versioned {
                    prefix: "win-cuda-",
                    suffix: "-x64",
                },
                Exact("win-vulkan-x64"),
                Cpu("win-cpu-x64"),
            ],
            Some(Vendor::Amd) => vec![
                Versioned {
                    prefix: "win-rocm-",
                    suffix: "-x64",
                },
                Exact("win-vulkan-x64"),
                Cpu("win-cpu-x64"),
            ],
            _ => vec![Exact("win-vulkan-x64"), Cpu("win-cpu-x64")],
        },
        ("windows", "aarch64" | "arm64") => vec![Cpu("win-cpu-arm64")],
        _ => vec![Cpu("ubuntu-x64")],
    }
}

/// A resolved asset: exact release-asset file name, the label stored in
/// the engine row, and whether this is a CPU fallback behind missing
/// GPU variants (caller must warn).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetPick {
    pub name: String,
    pub label: String,
    pub cpu_fallback: bool,
}

/// Walk the candidate list against the release's actual assets and
/// return the first match. `None` = nothing usable (not even CPU).
#[must_use]
pub fn resolve_asset(
    release: &GhRelease,
    os: &str,
    arch: &str,
    vendor: Option<crate::engine::manifest::Vendor>,
) -> Option<AssetPick> {
    let candidates = pick_asset(os, arch, vendor);
    let wants_gpu = candidates.iter().any(|c| !matches!(c, Candidate::Cpu(_)));
    let tag = &release.tag_name;
    for candidate in &candidates {
        match candidate {
            Candidate::Exact(s) | Candidate::Cpu(s) => {
                let name = asset_filename(tag, s);
                if release.assets.iter().any(|a| a.name == name) {
                    let cpu = matches!(candidate, Candidate::Cpu(_)) && wants_gpu;
                    return Some(AssetPick {
                        name,
                        label: (*s).to_string(),
                        cpu_fallback: cpu,
                    });
                }
            }
            Candidate::Versioned { prefix, suffix } => {
                let ext = if prefix.starts_with("win-") {
                    "zip"
                } else {
                    "tar.gz"
                };
                let head = format!("llama-{tag}-bin-{prefix}");
                let tail = format!("{suffix}.{ext}");
                let mut best: Option<(Vec<u32>, String)> = None;
                for a in &release.assets {
                    let Some(middle) = a.name.strip_prefix(&head) else {
                        continue;
                    };
                    let Some(version) = middle.strip_suffix(&tail) else {
                        continue;
                    };
                    let parts: Option<Vec<u32>> =
                        version.split('.').map(|p| p.parse().ok()).collect();
                    let Some(parts) = parts else { continue };
                    if best.as_ref().is_none_or(|(b, _)| parts > *b) {
                        best = Some((parts, version.to_string()));
                    }
                }
                if let Some((_, version)) = best {
                    let label = format!("{prefix}{version}{suffix}");
                    return Some(AssetPick {
                        name: format!("llama-{tag}-bin-{label}.{ext}"),
                        label,
                        cpu_fallback: false,
                    });
                }
            }
        }
    }
    None
}

/// Asset file name for a release tag: `llama-{tag}-bin-{suffix}.tar.gz|zip`.
#[must_use]
pub fn asset_filename(tag: &str, suffix: &str) -> String {
    let ext = if suffix.starts_with("win-") {
        "zip"
    } else {
        "tar.gz"
    };
    format!("llama-{tag}-bin-{suffix}.{ext}")
}

pub const MISTRALRS_REPO: &str = "EricLBuehler/mistral.rs";

/// CUDA toolkit variants mistral.rs publishes prebuilts for, as the
/// digit-run used in asset names (12.8 -> 128). Ordered oldest-first;
/// derivation walks it descending to find the newest the driver allows.
pub const MISTRALRS_CUDAS: [u32; 6] = [128, 129, 130, 131, 132, 133];

/// GPU compute caps (sm) mistral.rs publishes prebuilts for.
pub const MISTRALRS_SMS: [u32; 7] = [80, 86, 89, 90, 100, 120, 121];

/// Parse `vX.Y.Z` numerically: missing patch = 0, `-rc.N` style
/// pre-release suffixes ignored. Currency for v-tagged repos (mistral.rs,
/// pallama self) must compare numerically — lexicographic order calls
/// v0.10.0 a "downgrade" from v0.9.3.
#[must_use]
pub fn vtag_semver(tag: &str) -> Option<(u64, u64, u64)> {
    let core = tag.strip_prefix('v')?;
    let core = core.split('-').next()?;
    let mut it = core.split('.');
    let maj: u64 = it.next()?.parse().ok()?;
    let min: u64 = it.next()?.parse().ok()?;
    let patch: u64 = it.next().map_or(0, |p| p.parse().unwrap_or(0));
    Some((maj, min, patch))
}

/// Ordered mistral.rs asset preferences for this machine (exact names:
/// mistral.rs asset names carry no tag component). The list is derived
/// from live driver/GPU facts, never from a compat matrix:
/// - Linux + Nvidia: `mistralrs-cuda{NNN}-sm{SM}-x86_64-unknown-linux-gnu`
///   for every published NNN the driver supports (driver CUDA 13.0
///   allows 130, not 131), newest first; a trailing CPU build marked
///   `cpu_fallback` covers driver < 12.8. `sm` is the compute cap
///   verbatim (8.9 -> 89); caps outside the published set are a
///   teaching error, not a silent mismatch.
/// - Everything else: the CPU/Metal build that exists for the platform.
pub fn mistralrs_asset_picks(
    os: &str,
    arch: &str,
    vendor: Option<crate::engine::manifest::Vendor>,
    driver_cuda: Option<(u32, u32)>,
    compute_cap: Option<(u32, u32)>,
) -> Result<Vec<AssetPick>> {
    use crate::engine::manifest::Vendor;
    let cpu_linux = |a: &str, fallback: bool| AssetPick {
        name: format!("mistralrs-cpu-{a}-unknown-linux-gnu.tar.gz"),
        label: "cpu".into(),
        cpu_fallback: fallback,
    };
    match (os, arch) {
        ("macos", "aarch64" | "arm64") => Ok(vec![AssetPick {
            name: "mistralrs-metal-aarch64-apple-darwin.tar.gz".into(),
            label: "metal".into(),
            cpu_fallback: false,
        }]),
        ("macos", _) => Err(anyhow!(
            "mistral.rs publishes no prebuilt for macOS x86_64 (Metal/arm64 only)"
        )),
        ("windows", "x86_64" | "x64" | "amd64") => Ok(vec![AssetPick {
            name: "mistralrs-cpu-x86_64-pc-windows-msvc.zip".into(),
            label: "cpu".into(),
            cpu_fallback: false,
        }]),
        ("windows", _) => Err(anyhow!(
            "mistral.rs publishes no prebuilt for Windows ARM64"
        )),
        ("linux", "x86_64" | "x64" | "amd64") => {
            if vendor == Some(Vendor::Nvidia) {
                let Some((dmaj, dmin)) = driver_cuda else {
                    return Ok(vec![cpu_linux("x86_64", true)]);
                };
                let Some((cmaj, cmin)) = compute_cap else {
                    return Ok(vec![cpu_linux("x86_64", true)]);
                };
                let sm = cmaj * 10 + cmin;
                if !MISTRALRS_SMS.contains(&sm) {
                    return Err(anyhow!(
                        "GPU compute cap {cmaj}.{cmin} (sm{sm}) is outside the mistral.rs \
                         prebuilt set (sm{MISTRALRS_SMS:?}); install the CPU build or compile from source"
                    ));
                }
                let floor = dmaj * 10 + dmin;
                let mut picks: Vec<AssetPick> = MISTRALRS_CUDAS
                    .iter()
                    .rev()
                    .filter(|&&nnn| nnn <= floor)
                    .map(|&nnn| AssetPick {
                        name: format!("mistralrs-cuda{nnn}-sm{sm}-x86_64-unknown-linux-gnu.tar.gz"),
                        label: format!("cuda{nnn}-sm{sm}"),
                        cpu_fallback: false,
                    })
                    .collect();
                picks.push(cpu_linux("x86_64", true));
                Ok(picks)
            } else {
                Ok(vec![cpu_linux("x86_64", false)])
            }
        }
        ("linux", "aarch64" | "arm64") => Ok(vec![cpu_linux("aarch64", false)]),
        _ => Err(anyhow!("mistral.rs publishes no prebuilt for {os}/{arch}")),
    }
}

/// First preference whose exact asset name exists on the release.
#[must_use]
pub fn resolve_mistralrs_asset(release: &GhRelease, picks: &[AssetPick]) -> Option<AssetPick> {
    picks
        .iter()
        .find(|p| release.assets.iter().any(|a| a.name == p.name))
        .cloned()
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::engine::manifest::Vendor;
    use Candidate::{Cpu, Exact, Versioned};

    fn rel(tag: &str, assets: &[&str]) -> GhRelease {
        GhRelease {
            tag_name: tag.to_string(),
            prerelease: true,
            assets: assets
                .iter()
                .map(|n| GhAsset {
                    name: (*n).to_string(),
                    digest: None,
                    size: None,
                    browser_download_url: String::new(),
                })
                .collect(),
            published_at: None,
        }
    }

    #[test]
    fn unit__btag_number__plain_and_suffixed() {
        assert_eq!(btag_number("b10816"), Some(10816));
        assert_eq!(btag_number("b10816-cuda"), Some(10816));
        assert_eq!(btag_number("b10816-cpu"), Some(10816));
        assert_eq!(btag_number("local"), None);
        assert_eq!(btag_number("v0.1.0"), None);
        assert_eq!(btag_number("b-cuda"), None);
    }

    #[test]
    fn unit__same_build__number_first() {
        assert!(same_build("b10816", "b10816-cuda"));
        assert!(same_build("b10816-cuda", "b10816"));
        assert!(!same_build("b10816", "b10817"));
        assert!(same_build("local", "local"));
        assert!(!same_build("local", "b10816"));
    }

    #[test]
    fn unit__vtag_semver__numeric_order() {
        assert_eq!(vtag_semver("v0.9.3"), Some((0, 9, 3)));
        assert_eq!(vtag_semver("v0.10.0"), Some((0, 10, 0)));
        assert_eq!(vtag_semver("v1.8"), Some((1, 8, 0)));
        assert_eq!(vtag_semver("v2.0.0-rc.1"), Some((2, 0, 0)));
        assert_eq!(vtag_semver("b10857"), None);
        assert_eq!(vtag_semver("vx.y.z"), None);
        assert!(vtag_semver("v0.10.0") > vtag_semver("v0.9.3"));
    }

    #[test]
    fn unit__mistralrs_picks__nvidia_newest_allowed_cuda_first() {
        let picks = mistralrs_asset_picks(
            "linux",
            "x86_64",
            Some(Vendor::Nvidia),
            Some((13, 0)),
            Some((8, 9)),
        )
        .unwrap();
        assert_eq!(
            picks[0].name,
            "mistralrs-cuda130-sm89-x86_64-unknown-linux-gnu.tar.gz"
        );
        assert_eq!(picks[0].label, "cuda130-sm89");
        assert!(!picks[0].cpu_fallback);
        // newer-than-driver variants excluded, older kept as fallbacks
        assert!(picks.iter().take(3).map(|p| p.name.as_str()).eq([
            "mistralrs-cuda130-sm89-x86_64-unknown-linux-gnu.tar.gz",
            "mistralrs-cuda129-sm89-x86_64-unknown-linux-gnu.tar.gz",
            "mistralrs-cuda128-sm89-x86_64-unknown-linux-gnu.tar.gz",
        ]));
        let cpu = picks.last().unwrap();
        assert_eq!(cpu.label, "cpu");
        assert!(cpu.cpu_fallback);
    }

    #[test]
    fn unit__mistralrs_picks__old_driver_cpu_only_loud_fallback() {
        let picks = mistralrs_asset_picks(
            "linux",
            "x86_64",
            Some(Vendor::Nvidia),
            Some((12, 2)),
            Some((8, 9)),
        )
        .unwrap();
        assert_eq!(picks.len(), 1);
        assert_eq!(picks[0].label, "cpu");
        assert!(picks[0].cpu_fallback);
    }

    #[test]
    fn unit__mistralrs_picks__unknown_driver_or_cap_falls_back() {
        for driver in [None, Some((11, 0))] {
            let picks = mistralrs_asset_picks(
                "linux",
                "x86_64",
                Some(Vendor::Nvidia),
                driver,
                Some((8, 9)),
            )
            .unwrap();
            assert_eq!(picks.len(), 1);
            assert!(picks[0].cpu_fallback);
        }
        let picks =
            mistralrs_asset_picks("linux", "x86_64", Some(Vendor::Nvidia), Some((13, 0)), None)
                .unwrap();
        assert_eq!(picks.len(), 1);
        assert!(picks[0].cpu_fallback);
    }

    #[test]
    fn unit__mistralrs_picks__unsupported_compute_cap_teaches() {
        let err = mistralrs_asset_picks(
            "linux",
            "x86_64",
            Some(Vendor::Nvidia),
            Some((13, 0)),
            Some((11, 0)),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("sm110"), "err: {err}");
        assert!(err.contains("CPU build"));
    }

    #[test]
    fn unit__mistralrs_picks__non_nvidia_and_other_platforms() {
        let picks =
            mistralrs_asset_picks("linux", "x86_64", Some(Vendor::Amd), None, None).unwrap();
        assert_eq!(picks.len(), 1);
        assert_eq!(picks[0].label, "cpu");
        assert!(!picks[0].cpu_fallback);
        let picks = mistralrs_asset_picks("macos", "arm64", None, None, None).unwrap();
        assert_eq!(picks[0].name, "mistralrs-metal-aarch64-apple-darwin.tar.gz");
        let picks =
            mistralrs_asset_picks("windows", "x64", Some(Vendor::Nvidia), None, None).unwrap();
        assert_eq!(picks[0].name, "mistralrs-cpu-x86_64-pc-windows-msvc.zip");
        assert!(mistralrs_asset_picks("macos", "x86_64", None, None, None).is_err());
    }

    #[test]
    fn unit__resolve_mistralrs_asset__first_present_wins() {
        let picks = mistralrs_asset_picks(
            "linux",
            "x86_64",
            Some(Vendor::Nvidia),
            Some((13, 0)),
            Some((8, 9)),
        )
        .unwrap();
        // release carries only the cuda128 fallback variant
        let release = rel(
            "v0.9.3",
            &["mistralrs-cuda128-sm89-x86_64-unknown-linux-gnu.tar.gz"],
        );
        let got = resolve_mistralrs_asset(&release, &picks).unwrap();
        assert_eq!(got.label, "cuda128-sm89");
        // nothing present -> None (caller teaching-errors)
        let empty = rel("v0.9.3", &["some-other-asset.txt"]);
        assert!(resolve_mistralrs_asset(&empty, &picks).is_none());
    }

    #[test]
    fn unit__asset_matrix__linux_nvidia__vulkan_then_cpu_resort() {
        assert_eq!(
            pick_asset("linux", "x86_64", Some(Vendor::Nvidia)),
            vec![Exact("ubuntu-vulkan-x64"), Cpu("ubuntu-x64")]
        );
    }

    #[test]
    fn unit__asset_matrix__linux_amd__rocm_glob_then_vulkan_then_cpu() {
        assert_eq!(
            pick_asset("linux", "x86_64", Some(Vendor::Amd)),
            vec![
                Versioned {
                    prefix: "ubuntu-rocm-",
                    suffix: "-x64"
                },
                Exact("ubuntu-vulkan-x64"),
                Cpu("ubuntu-x64"),
            ]
        );
    }

    #[test]
    fn unit__asset_matrix__linux_intel__sycl_then_vulkan_then_cpu() {
        assert_eq!(
            pick_asset("linux", "x86_64", Some(Vendor::Intel)),
            vec![
                Exact("ubuntu-sycl-fp16-x64"),
                Exact("ubuntu-vulkan-x64"),
                Cpu("ubuntu-x64"),
            ]
        );
    }

    #[test]
    fn unit__asset_matrix__linux_cpu__plain() {
        assert_eq!(pick_asset("linux", "x86_64", None), vec![Cpu("ubuntu-x64")]);
        assert_eq!(
            pick_asset("linux", "x86_64", Some(Vendor::Other)),
            vec![Cpu("ubuntu-x64")]
        );
    }

    #[test]
    fn unit__asset_matrix__linux_arm__vulkan_then_cpu() {
        assert_eq!(
            pick_asset("linux", "aarch64", None),
            vec![Exact("ubuntu-vulkan-arm64"), Cpu("ubuntu-arm64")]
        );
    }

    #[test]
    fn unit__asset_matrix__macos_arm_first() {
        assert_eq!(
            pick_asset("macos", "arm64", None),
            vec![Exact("macos-arm64"), Exact("macos-x64")]
        );
    }

    #[test]
    fn unit__asset_matrix__windows_variants() {
        assert_eq!(
            pick_asset("windows", "x64", Some(Vendor::Nvidia)),
            vec![
                Versioned {
                    prefix: "win-cuda-",
                    suffix: "-x64"
                },
                Exact("win-vulkan-x64"),
                Cpu("win-cpu-x64"),
            ]
        );
        assert_eq!(
            pick_asset("windows", "x64", Some(Vendor::Amd)),
            vec![
                Versioned {
                    prefix: "win-rocm-",
                    suffix: "-x64"
                },
                Exact("win-vulkan-x64"),
                Cpu("win-cpu-x64"),
            ]
        );
        assert_eq!(
            pick_asset("windows", "x64", None),
            vec![Exact("win-vulkan-x64"), Cpu("win-cpu-x64")]
        );
        assert_eq!(
            pick_asset("windows", "arm64", None),
            vec![Cpu("win-cpu-arm64")]
        );
    }

    #[test]
    fn unit__asset_filename__extension_by_platform() {
        assert_eq!(
            asset_filename("b10816", "ubuntu-vulkan-x64"),
            "llama-b10816-bin-ubuntu-vulkan-x64.tar.gz"
        );
        assert_eq!(
            asset_filename("b10816", "win-cuda-13.3-x64"),
            "llama-b10816-bin-win-cuda-13.3-x64.zip"
        );
    }

    #[test]
    fn unit__resolve__version_bump__max_version_wins() {
        let r = rel(
            "b10833",
            &[
                "llama-b10833-bin-ubuntu-rocm-10.0-x64.tar.gz",
                "llama-b10833-bin-ubuntu-rocm-11.2-x64.tar.gz",
            ],
        );
        let p = resolve_asset(&r, "linux", "x86_64", Some(Vendor::Amd)).unwrap();
        assert_eq!(p.name, "llama-b10833-bin-ubuntu-rocm-11.2-x64.tar.gz");
        assert_eq!(p.label, "ubuntu-rocm-11.2-x64");
        assert!(!p.cpu_fallback);
    }

    #[test]
    fn unit__resolve__cuda_major_minor__numeric_order_not_lexical() {
        // 9.1 vs 10.0: lexical compare would call "9.1" bigger.
        let r = rel(
            "b1",
            &[
                "llama-b1-bin-win-cuda-9.1-x64.zip",
                "llama-b1-bin-win-cuda-10.0-x64.zip",
            ],
        );
        let p = resolve_asset(&r, "windows", "x86_64", Some(Vendor::Nvidia)).unwrap();
        assert_eq!(p.label, "win-cuda-10.0-x64");
    }

    #[test]
    fn unit__resolve__upload_race__vulkan_missing_cpu_present__fallback_flagged() {
        // The exact live b10833 scenario: nvidia box, vulkan asset not
        // uploaded yet, CPU asset visible.
        let r = rel("b10833", &["llama-b10833-bin-ubuntu-x64.tar.gz"]);
        let p = resolve_asset(&r, "linux", "x86_64", Some(Vendor::Nvidia)).unwrap();
        assert_eq!(p.label, "ubuntu-x64");
        assert!(p.cpu_fallback, "must be flagged, never silent");
    }

    #[test]
    fn unit__resolve__no_assets__none() {
        let r = rel("b10833", &[]);
        assert!(resolve_asset(&r, "linux", "x86_64", Some(Vendor::Nvidia)).is_none());
    }

    #[test]
    fn unit__resolve__vendor_falls_through_to_vulkan() {
        let r = rel(
            "b1",
            &[
                "llama-b1-bin-ubuntu-sycl-fp32-x64.tar.gz",
                "llama-b1-bin-ubuntu-vulkan-x64.tar.gz",
            ],
        );
        // Intel: fp16 exact miss -> vulkan (sycl-fp32 is not a candidate).
        let p = resolve_asset(&r, "linux", "x86_64", Some(Vendor::Intel)).unwrap();
        assert_eq!(p.label, "ubuntu-vulkan-x64");
        assert!(!p.cpu_fallback);
    }

    #[test]
    fn unit__resolve__non_numeric_version_ignored() {
        let r = rel(
            "b1",
            &[
                "llama-b1-bin-win-cuda-hipx-x64.zip",
                "llama-b1-bin-win-vulkan-x64.zip",
            ],
        );
        let p = resolve_asset(&r, "windows", "x86_64", Some(Vendor::Nvidia)).unwrap();
        assert_eq!(p.label, "win-vulkan-x64");
    }

    #[test]
    fn unit__iso_epoch__github_formats() {
        // Pinned to the live b10833 publish time observed via the API.
        assert_eq!(iso_to_epoch("2026-09-07T06:49:18Z"), Some(1_788_763_758));
        assert_eq!(
            iso_to_epoch("2026-09-07T06:49:18.123Z"),
            Some(1_788_763_758)
        );
        assert_eq!(iso_to_epoch("bogus"), None);
    }

    #[test]
    fn unit__published_epoch__field_routing() {
        let mut r = rel("b1", &[]);
        assert_eq!(r.published_epoch(), None);
        r.published_at = Some("2026-09-07T06:49:18Z".into());
        assert_eq!(r.published_epoch(), Some(1_788_763_758));
    }
}
