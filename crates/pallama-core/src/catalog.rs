use std::sync::LazyLock;

use serde::Deserialize;

use crate::error::{CoreError, CoreResult};

/// Short-name -> HF repo catalog. Embedded at build time; every entry was
/// verified to exist and be publicly pullable via the HF API
/// (GET /api/models/{repo} -> 200, gated=false) at catalog-freeze time.
/// The registry name shown to users is ALWAYS the real repo name
/// (lowercased, `-gguf` suffix stripped) — never a marketing alias.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CatalogEntry {
    /// Short name users type, e.g. "qwen3-coder-30b".
    pub short_name: String,
    /// HF repo, e.g. "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF".
    pub repo: String,
}

/// Speculative-decoding draft pairs: models matching `model_glob` (prefix
/// match on registry name) use `draft_repo` via `--spec-type draft-simple`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SpecPair {
    /// Prefix matched against the resolved model name, e.g. "qwen3".
    pub model_prefix: String,
    /// llama-server spec type from the manifest's supported set.
    pub spec_type: String,
    /// HF repo of the draft model.
    pub draft_repo: String,
}

#[derive(Debug, Deserialize)]
pub struct Catalog {
    pub entries: Vec<CatalogEntry>,
    pub spec_pairs: Vec<SpecPair>,
}

pub const CATALOG_JSON: &str = include_str!("catalog.json");
#[must_use]
pub fn catalog() -> &'static Catalog {
    static CAT: LazyLock<Catalog> = LazyLock::new(|| {
        serde_json::from_str(CATALOG_JSON)
            .unwrap_or_else(|e| panic!("embedded catalog.json is invalid: {e}"))
    });
    &CAT
}

/// Resolve a user-typed name to a catalog entry:
/// exact match -> unique prefix match -> error with levenshtein<=3 suggestions.
pub fn resolve(name: &str) -> CoreResult<&'static CatalogEntry> {
    let cat = catalog();
    let wanted = name.to_lowercase();
    if let Some(exact) = cat.entries.iter().find(|e| e.short_name == wanted) {
        return Ok(exact);
    }
    let prefixed: Vec<&CatalogEntry> = cat
        .entries
        .iter()
        .filter(|e| e.short_name.starts_with(&wanted))
        .collect();
    match prefixed.len() {
        1 => Ok(prefixed[0]),
        0 => {
            let suggestions: Vec<&str> = cat
                .entries
                .iter()
                .filter(|e| levenshtein(&e.short_name, &wanted) <= 3)
                .map(|e| e.short_name.as_str())
                .collect();
            Err(CoreError::Catalog(if suggestions.is_empty() {
                format!("no catalog model matching {name:?}; try `pallama search <query>` or pull an explicit owner/repo:quant")
            } else {
                format!(
                    "no catalog model matching {name:?}; did you mean: {}?",
                    suggestions.join(", ")
                )
            }))
        }
        _ => Err(CoreError::Catalog(format!(
            "ambiguous name {name:?} matches multiple models: {}",
            prefixed
                .iter()
                .map(|e| e.short_name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ))),
    }
}
/// Spec-draft pair for a resolved model name, if the catalog has one.
#[must_use]
pub fn spec_pair_for(model: &str) -> Option<&'static SpecPair> {
    catalog()
        .spec_pairs
        .iter()
        .find(|p| model.starts_with(&p.model_prefix))
}

/// Classic DP edit distance; catalog sizes are tiny, O(nm) is fine.
#[must_use]
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn unit__catalog_embedded__parses_and_unique() {
        let cat = catalog();
        assert!(
            cat.entries.len() >= 6,
            "catalog should carry a useful default set"
        );
        let mut names: Vec<&str> = cat.entries.iter().map(|e| e.short_name.as_str()).collect();
        names.sort_unstable();
        let n = names.len();
        names.dedup();
        assert_eq!(names.len(), n, "duplicate short_name in catalog");
    }

    #[test]
    fn unit__catalog_repos__well_formed() {
        for e in &catalog().entries {
            assert!(
                e.repo.contains('/') && !e.repo.ends_with('/'),
                "repo must be owner/name: {}",
                e.repo
            );
        }
    }

    #[test]
    fn unit__resolve_exact_and_case() {
        assert_eq!(
            resolve("qwen3-0.6b").unwrap().repo,
            "ggml-org/Qwen3-0.6B-GGUF"
        );
        assert_eq!(resolve("QWEN3-0.6B").unwrap().short_name, "qwen3-0.6b");
    }

    #[test]
    fn unit__resolve_unique_prefix() {
        // Exactly one catalog entry starts with this prefix.
        let e = resolve("qwen3-coder").unwrap();
        assert!(e.repo.starts_with("unsloth/Qwen3-Coder"));
    }

    #[test]
    fn unit__resolve_missing__levenshtein_suggestion() {
        let err = resolve("qwen3-0.5b").unwrap_err().to_string();
        assert!(
            err.contains("qwen3-0.6b"),
            "near-miss should suggest: {err}"
        );
    }

    #[test]
    fn unit__resolve_missing_far__search_hint() {
        let err = resolve("zzzzzzzzzz").unwrap_err().to_string();
        assert!(err.contains("pallama search"), "{err}");
    }

    #[test]
    fn unit__spec_pair__prefix_match() {
        let pair = spec_pair_for("qwen3-coder-30b").expect("qwen3 family has a draft pair");
        assert_eq!(pair.spec_type, "draft-simple");
        assert!(pair.draft_repo.starts_with("ggml-org/Qwen3-0.6B"));
        assert!(spec_pair_for("gemma3-4b").is_none());
    }

    #[test]
    fn unit__levenshtein__known_values() {
        assert_eq!(levenshtein("", ""), 0);
        assert_eq!(levenshtein("abc", "abc"), 0);
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("qwen3-0.5b", "qwen3-0.6b"), 1);
    }
}
