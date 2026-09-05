//! GitHub release client for upstream llama.cpp binaries.
//! API truth verified live (2026-09-05): releases carry `bNNNNN` tags
//! (prereleases, 27 assets) plus `vX.Y.Z` stable tags whose single asset
//! `nightly-tag.txt` contains the current b-tag. Assets are named
//! `llama-{tag}-bin-{asset}.tar.gz|.zip` and expose `digest:
//! sha256:...` — verified before extraction.

use anyhow::{anyhow, Context, Result};
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
}

pub struct GhClient {
    http: reqwest::Client,
    base: reqwest::Url,
    token: Option<String>,
}

fn btag_number(tag: &str) -> Option<u64> {
    tag.strip_prefix('b')?.parse().ok()
}

impl GhClient {
    pub fn new(token: Option<String>) -> Result<Self> {
        Self::with_base("https://api.github.com", token)
    }

    pub fn with_base(base: &str, token: Option<String>) -> Result<Self> {
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
        let url = self
            .base
            .join(&format!("repos/{LLAMA_CPP_REPO}/releases?per_page=30"))
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
                return Err(anyhow!("GitHub API rate limited ({}). Set GH_TOKEN", resp.status()));
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
            return Err(anyhow!("asset download {} returned {}", asset.name, resp.status()));
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
            tracing::warn!("asset {} has no digest in release metadata; skipping sha verify", asset.name);
        }
        Ok(bytes.to_vec())
    }
}

/// Pick the right asset suffix for this machine, given the GPU vendor
/// detected on the engine (or None for CPU-only). Matrix verified against
/// the b10816 asset list.
#[must_use] 
pub fn pick_asset(os: &str, arch: &str, vendor: Option<crate::engine::manifest::Vendor>) -> Vec<&'static str> {
    // Preference order; first existing asset wins.
    match (os, arch) {
        ("linux", "x86_64" | "x64" | "amd64") => match vendor {
            Some(crate::engine::manifest::Vendor::Nvidia) => vec!["ubuntu-vulkan-x64"],
            Some(crate::engine::manifest::Vendor::Amd) => {
                vec!["ubuntu-rocm-10.0-x64", "ubuntu-vulkan-x64"]
            }
            Some(crate::engine::manifest::Vendor::Intel) => {
                vec!["ubuntu-sycl-fp16-x64", "ubuntu-vulkan-x64"]
            }
            _ => vec!["ubuntu-x64"],
        },
        ("linux", "aarch64" | "arm64") => vec!["ubuntu-vulkan-arm64", "ubuntu-arm64"],
        ("macos", _) => vec!["macos-arm64", "macos-x64"],
        ("windows", "x86_64" | "x64" | "amd64") => match vendor {
            Some(crate::engine::manifest::Vendor::Nvidia) => {
                vec!["win-cuda-13.3-x64", "win-vulkan-x64"]
            }
            Some(crate::engine::manifest::Vendor::Amd) => vec!["win-rocm-10.0-x64", "win-vulkan-x64"],
            _ => vec!["win-vulkan-x64"],
        },
        ("windows", "aarch64" | "arm64") => vec!["win-cpu-arm64"],
        _ => vec!["ubuntu-x64"],
    }
}

/// Asset file name for a release tag: `llama-{tag}-bin-{suffix}.tar.gz|zip`.
#[must_use] 
pub fn asset_filename(tag: &str, suffix: &str) -> String {
    let ext = if suffix.starts_with("win-") { "zip" } else { "tar.gz" };
    format!("llama-{tag}-bin-{suffix}.{ext}")
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn unit__asset_matrix__linux_nvidia__vulkan_only() {
        assert_eq!(pick_asset("linux", "x86_64", Some(crate::engine::manifest::Vendor::Nvidia)), vec!["ubuntu-vulkan-x64"]);
    }

    #[test]
    fn unit__asset_matrix__linux_amd__rocm_then_vulkan() {
        assert_eq!(
            pick_asset("linux", "x86_64", Some(crate::engine::manifest::Vendor::Amd)),
            vec!["ubuntu-rocm-10.0-x64", "ubuntu-vulkan-x64"]
        );
    }

    #[test]
    fn unit__asset_matrix__linux_intel__sycl_then_vulkan() {
        assert_eq!(
            pick_asset("linux", "x86_64", Some(crate::engine::manifest::Vendor::Intel)),
            vec!["ubuntu-sycl-fp16-x64", "ubuntu-vulkan-x64"]
        );
    }

    #[test]
    fn unit__asset_matrix__linux_cpu__plain() {
        assert_eq!(pick_asset("linux", "x86_64", None), vec!["ubuntu-x64"]);
        assert_eq!(pick_asset("linux", "x86_64", Some(crate::engine::manifest::Vendor::Other)), vec!["ubuntu-x64"]);
    }

    #[test]
    fn unit__asset_matrix__linux_arm__vulkan_then_cpu() {
        assert_eq!(pick_asset("linux", "aarch64", None), vec!["ubuntu-vulkan-arm64", "ubuntu-arm64"]);
    }

    #[test]
    fn unit__asset_matrix__macos_arm_first() {
        assert_eq!(pick_asset("macos", "arm64", None), vec!["macos-arm64", "macos-x64"]);
    }

    #[test]
    fn unit__asset_matrix__windows_variants() {
        assert_eq!(
            pick_asset("windows", "x64", Some(crate::engine::manifest::Vendor::Nvidia)),
            vec!["win-cuda-13.3-x64", "win-vulkan-x64"]
        );
        assert_eq!(
            pick_asset("windows", "x64", Some(crate::engine::manifest::Vendor::Amd)),
            vec!["win-rocm-10.0-x64", "win-vulkan-x64"]
        );
        assert_eq!(pick_asset("windows", "x64", None), vec!["win-vulkan-x64"]);
        assert_eq!(pick_asset("windows", "arm64", None), vec!["win-cpu-arm64"]);
    }

    #[test]
    fn unit__asset_filename__extension_by_platform() {
        assert_eq!(asset_filename("b10816", "ubuntu-vulkan-x64"), "llama-b10816-bin-ubuntu-vulkan-x64.tar.gz");
        assert_eq!(asset_filename("b10816", "win-cuda-13.3-x64"), "llama-b10816-bin-win-cuda-13.3-x64.zip");
    }
}
