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
    tag.strip_prefix('b')?.parse().ok()
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
        let url = self
            .base
            .join(&format!("repos/{LLAMA_CPP_REPO}/releases/tags/{tag}"))
            .unwrap();
        let resp = self
            .auth(self.http.get(url.clone()))
            .send()
            .await
            .context("GitHub release-by-tag request failed")?;
        match resp.status() {
            reqwest::StatusCode::OK => Ok(resp.json().await.context("decode release")?),
            reqwest::StatusCode::NOT_FOUND => Err(anyhow!("llama.cpp release {tag} not found")),
            other => Err(anyhow!("GitHub API {other} for tag {tag}")),
        }
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
