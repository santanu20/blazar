//! Client for the blazar draft-pairing registry: curated
//! speculative-decoding draft recommendations per model family -
//! combined single-file MTP bundles (the draft head ships inside the
//! target GGUF) and separate EAGLE3 draft-head repos. The registry
//! publishes SUGGESTIONS; the live bench (`blazar tune <model>
//! --spec`) stays the adoption gate - a registry entry never bypasses
//! it, exactly like the capability-lane registry publishes
//! compatibility but never defines it.
//!
//! Server side is a static JSON file (a raw GitHub URL works); this
//! module is intentionally client + format contract only, mirroring
//! the capability-lane registry's fetch contract (bounded body,
//! validate-and-skip rows, advisory for callers).

use std::time::Duration;

use anyhow::{Context as _, Result};

/// Default registry location: the repo's own `main` branch, so a
/// merged PR updates every install without a release.
pub const DEFAULT_REGISTRY_URL: &str =
    "https://raw.githubusercontent.com/santanu20/blazar/main/registry/draft-pairings.json";

/// Env override for the registry URL (tests and self-hosted mirrors).
pub const REGISTRY_URL_ENV: &str = "BLAZAR_DRAFT_REGISTRY";

/// Registry bodies are a handful of curated rows; 256KB is already
/// generous and keeps a hostile mirror from buffering memory.
const MAX_BODY_BYTES: usize = 256 * 1024;

/// How a curated draft ships.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DraftMode {
    /// Target GGUF already contains the MTP/NextN head - pull the
    /// target and set `spec = "auto"`; the auto-MTP gate engages the
    /// embedded head when the engine supports it.
    Combined,
    /// Standalone draft model (EAGLE3 head or tiny same-family
    /// instruct model) - pull it and set `spec = "draft"`.
    Separate,
}

impl DraftMode {
    /// One-line adoption guidance, printed under the curated block.
    #[must_use]
    pub fn hint(self) -> &'static str {
        match self {
            Self::Combined => {
                "pull the target, set spec = \"auto\" - the auto-MTP gate arms the embedded head"
            }
            Self::Separate => {
                "pull the draft, set spec = \"draft\" (bench proves net-positive first)"
            }
        }
    }
}

impl std::fmt::Display for DraftMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Combined => "combined",
            Self::Separate => "separate",
        })
    }
}

/// One curated pairing row. `family` uses the same coarse token
/// `blazar drafts` derives from a model name (first segment split on
/// `-`, `.`, `:` - e.g. "qwen3.5-9b" -> "qwen3"); `models` lists the
/// example targets the entry was curated for so the user picks the
/// right sibling, never a guessed pairing.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DraftPairing {
    pub family: String,
    pub mode: DraftMode,
    /// `owner/name` Hugging Face slug.
    pub repo: String,
    /// Example target model names this pairing fits (display only).
    #[serde(default)]
    pub models: Vec<String>,
    pub note: Option<String>,
}

/// Effective registry URL: env override > default. Unlike the
/// capability registry there is no config-file field - this surface is
/// advisory, so the escape hatch stays minimal.
#[must_use]
pub fn registry_url() -> String {
    let from_env = std::env::var(REGISTRY_URL_ENV).unwrap_or_default();
    if from_env.trim().is_empty() {
        DEFAULT_REGISTRY_URL.to_string()
    } else {
        from_env.trim().to_string()
    }
}

/// Bounded fetch of the registry file. Every entry is validated
/// (non-empty family, `owner/name` slug shape, known mode); invalid
/// entries are dropped with a warning - one bad row must not hide the
/// good ones.
pub async fn fetch(client: &reqwest::Client, url: &str) -> Result<Vec<DraftPairing>> {
    let mut resp = client
        .get(url)
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .with_context(|| format!("draft registry fetch failed: {url}"))?;
    if !resp.status().is_success() {
        anyhow::bail!("draft registry {url} answered HTTP {}", resp.status());
    }
    let body = crate::engine::capability_registry::read_body_bounded_shared(
        &mut resp,
        url,
        MAX_BODY_BYTES,
        "draft registry",
    )
    .await?;
    parse_registry(&body)
}

/// Parse + validate a registry document. Public so tests (and curious
/// users) can check a file offline against the same rules.
pub fn parse_registry(text: &str) -> Result<Vec<DraftPairing>> {
    let raw: Vec<serde_json::Value> =
        serde_json::from_str(text).context("draft registry: not a JSON array")?;
    let mut out = Vec::new();
    for (idx, value) in raw.iter().enumerate() {
        match serde_json::from_value::<DraftPairing>(value.clone()) {
            Ok(p) if entry_valid(&p) => out.push(p),
            Ok(p) => {
                tracing::warn!(
                    "draft registry: entry {idx} ({}) failed validation - skipped",
                    p.repo
                );
            }
            Err(e) => {
                tracing::warn!("draft registry: entry {idx} malformed ({e}) - skipped");
            }
        }
    }
    Ok(out)
}

/// Advisory data, so validation is shape-only: non-empty family token
/// and an `owner/name` slug (no whitespace). Repo existence and model
/// fit are the curator's job; the bench gate is the user's.
fn entry_valid(p: &DraftPairing) -> bool {
    let slug_ok = match p.repo.split_once('/') {
        Some((owner, name)) => {
            !owner.trim().is_empty()
                && !name.trim().is_empty()
                && !p.repo.contains(char::is_whitespace)
        }
        None => false,
    };
    !p.family.trim().is_empty() && slug_ok
}

/// Entries matching a family token (case-insensitive; the CLI token is
/// lowercased already), most useful first: combined bundles need no
/// second model, so they sort ahead of separate drafts.
#[must_use]
pub fn pairings_for_family<'a>(entries: &'a [DraftPairing], family: &str) -> Vec<&'a DraftPairing> {
    let mut hits: Vec<_> = entries
        .iter()
        .filter(|p| p.family.eq_ignore_ascii_case(family))
        .collect();
    hits.sort_by_key(|p| std::cmp::Reverse(matches!(p.mode, DraftMode::Combined)));
    hits
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // test names narrate: unit__subject__contract
    use super::*;

    #[test]
    fn unit__parse_registry__valid_rows_and_skips() {
        let text = r#"[
            {"family":"qwen3","mode":"combined","repo":"unsloth/Qwen3.5-9B-MTP-GGUF",
             "models":["qwen3.5-9b"],"note":"combined MTP bundle"},
            {"family":"qwen3","mode":"separate","repo":"williamliao/Qwen3-8B-EAGLE3-Speculator-GGUF",
             "models":["qwen3-8b"]},
            {"family":"","mode":"combined","repo":"x/y"},
            {"family":"z","mode":"combined","repo":"not-a-slug"},
            {"family":"z","mode":"weird","repo":"a/b"}
        ]"#;
        let rows = parse_registry(text).unwrap();
        assert_eq!(rows.len(), 2, "{rows:?}");
        assert_eq!(rows[0].repo, "unsloth/Qwen3.5-9B-MTP-GGUF");
        assert_eq!(rows[0].mode, DraftMode::Combined);
        assert_eq!(rows[1].mode, DraftMode::Separate);
        assert!(rows[1].note.is_none() && rows[1].models == vec!["qwen3-8b".to_string()]);
    }

    #[test]
    fn unit__parse_registry__rejects_non_array() {
        assert!(parse_registry("{}").is_err());
        assert!(parse_registry("not json").is_err());
    }

    #[test]
    fn unit__pairings_for_family__case_insensitive_combined_first() {
        let rows = parse_registry(
            r#"[
                {"family":"qwen3","mode":"separate","repo":"a/b"},
                {"family":"qwen3","mode":"combined","repo":"c/d"},
                {"family":"gemma","mode":"separate","repo":"e/f"}
            ]"#,
        )
        .unwrap();
        let hits = pairings_for_family(&rows, "QWEN3");
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].repo, "c/d", "combined sorts first");
        assert!(pairings_for_family(&rows, "llama").is_empty());
    }

    #[test]
    fn unit__registry_url__env_override_and_default() {
        // Env-var tests race under a parallel runner; assert the pure
        // default path only (the override path is exercised live via
        // BLAZAR_DRAFT_REGISTRY in the CLI validation).
        assert!(registry_url().contains("draft-pairings.json"));
    }

    #[test]
    fn unit__draft_mode__hint_and_display() {
        assert_eq!(DraftMode::Combined.to_string(), "combined");
        assert_eq!(DraftMode::Separate.to_string(), "separate");
        assert!(DraftMode::Combined.hint().contains("auto"));
        assert!(DraftMode::Separate.hint().contains("draft"));
    }
}
