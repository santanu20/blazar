//! Build-time capability mining: read the architecture table out of a
//! llama.cpp source tree and bake it into the engine manifest, so
//! Blazar can answer "which installed lane advertises architecture X"
//! without spawning anything.
//!
//! The source of truth is `src/llama-arch.cpp`'s `LLM_ARCH_NAMES` map
//! (verified against upstream master: one `{ LLM_ARCH_X, "name" }`
//! entry per line, 150+ architectures, terminal
//! `{ LLM_ARCH_UNKNOWN, "(unknown)" }` sentinel). Advertisement is a
//! CANDIDATE filter only — the runtime model load is the verification,
//! and a load failure downgrades the claim (see the supervisor's
//! unknown-architecture classifier).
//!
//! Failure policy: a missing or unparseable file yields an EMPTY set
//! plus a caller-visible warning — never a failed build. An engine
//! with no mined set simply never matches architecture-based routing.

use std::collections::BTreeSet;
use std::path::Path;

/// Path of the architecture table inside a llama.cpp checkout.
const LLAMA_ARCH_TABLE: &str = "src/llama-arch.cpp";
/// Upstream's terminal sentinel name — a placeholder, not a loadable
/// architecture.
const UNKNOWN_SENTINEL: &str = "(unknown)";

/// Mine the architecture names from a llama.cpp source tree. See the
/// module docs for the parse contract.
#[must_use]
pub fn mine_architectures(src_root: &Path) -> BTreeSet<String> {
    let table = src_root.join(LLAMA_ARCH_TABLE);
    let text = match std::fs::read_to_string(&table) {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!(
                "architecture mining skipped: cannot read {}: {e} — the \
                 engine installs without an advertised architecture set",
                table.display()
            );
            return BTreeSet::new();
        }
    };
    parse_arch_table(&text)
}

/// Parse the `LLM_ARCH_NAMES = { ... };` block: every quoted string on
/// an entry line is an architecture name.
#[must_use]
pub fn parse_arch_table(text: &str) -> BTreeSet<String> {
    let mut archs = BTreeSet::new();
    let mut in_table = false;
    for line in text.lines() {
        if !in_table {
            if line.contains("LLM_ARCH_NAMES") {
                in_table = true;
            }
            continue;
        }
        let trimmed = line.trim();
        if trimmed.starts_with('}') || trimmed.starts_with(';') {
            break;
        }
        // Entry shape: `{ LLM_ARCH_LLAMA, "llama" },`
        if let Some(name) = entry_name(trimmed) {
            archs.insert(name);
        }
    }
    archs
}

/// Extract the quoted architecture name from one table entry line;
/// `None` for non-entry lines and the `(unknown)` sentinel.
fn entry_name(line: &str) -> Option<String> {
    if !line.starts_with("{ LLM_ARCH_") {
        return None;
    }
    let open = line.find('"')?;
    let rest = &line[open + 1..];
    let close = rest.find('"')?;
    let name = &rest[..close];
    if name.is_empty() || name == UNKNOWN_SENTINEL {
        return None;
    }
    Some(name.to_string())
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    /// Slice of the real upstream table (verified master 2026-09):
    /// varied name shapes — plain, dotted, dashed, sentinel tail.
    const REAL_TABLE_SLICE: &str = r#"
static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
    { LLM_ARCH_CLIP,             "clip"             }, // dummy, only used by llama-quantize
    { LLM_ARCH_LLAMA,            "llama"            },
    { LLM_ARCH_LLAMA4,           "llama4"           },
    { LLM_ARCH_GPTNEOX,          "gptneox"          },
    { LLM_ARCH_MODERN_BERT,      "modern-bert"      },
    { LLM_ARCH_NOMIC_BERT_MOE,   "nomic-bert-moe"   },
    { LLM_ARCH_QWEN2VL,          "qwen2vl"          },
    { LLM_ARCH_UNKNOWN,          "(unknown)"        },
};

static const std::map<std::string, llm_kv_type> LLM_KV_NAMES = {
    { "general.architecture",         LLM_KV_GENERAL_ARCHITECTURE },
"#;

    #[test]
    fn unit__parse_arch_table__real_shape() {
        let archs = parse_arch_table(REAL_TABLE_SLICE);
        assert_eq!(
            archs,
            BTreeSet::from([
                "clip".to_string(),
                "llama".to_string(),
                "llama4".to_string(),
                "gptneox".to_string(),
                "modern-bert".to_string(),
                "nomic-bert-moe".to_string(),
                "qwen2vl".to_string(),
            ])
        );
    }

    #[test]
    fn unit__parse_arch_table__no_table_is_empty() {
        // A fork that renames the table (or a random file) must mine
        // nothing — never a guess.
        assert!(parse_arch_table("int main() {}\n").is_empty());
        assert!(parse_arch_table("").is_empty());
    }

    #[test]
    fn unit__mine_architectures__tree_layout_and_missing() {
        let tmp = tempfile::tempdir().unwrap();
        // Missing tree: empty set, no panic.
        assert!(mine_architectures(tmp.path()).is_empty());
        // Real layout: src/llama-arch.cpp relative to the checkout root.
        let src = tmp.path().join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("llama-arch.cpp"), REAL_TABLE_SLICE).unwrap();
        let archs = mine_architectures(tmp.path());
        assert!(archs.contains("llama"));
        assert!(archs.contains("nomic-bert-moe"));
        assert!(!archs.contains("(unknown)"));
        assert_eq!(archs.len(), 7);
    }
}
