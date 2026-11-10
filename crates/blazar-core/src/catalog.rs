use std::sync::LazyLock;

use serde::Deserialize;

use crate::error::{CoreError, CoreResult};

/// Short-name -> HF repo catalog. Embedded at build time; every entry was
/// verified to exist and be publicly pullable via the HF API
/// (GET /api/models/{repo} -> 200, gated=false) at catalog-freeze time.
/// The registry name shown to users is ALWAYS the real repo name
/// (lowercased, `-gguf` suffix stripped) - never a marketing alias.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CatalogEntry {
    /// Short name users type, e.g. "qwen3-coder-30b".
    pub short_name: String,
    /// HF repo, e.g. "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF".
    pub repo: String,
    /// Git ref the entry's files live on, when not `main`. Live case:
    /// CLIP's safetensors conversion only exists on `refs/pr/15`.
    /// A user-passed `--revision` always wins over this pin.
    #[serde(default)]
    pub revision: Option<String>,
    /// Modality class tag driving `blazar doctor`'s coverage census and
    /// per-class pull suggestions: embeddings | clip | speech | asr |
    /// diffusion. Absent = plain text LLM (the historical default, so
    /// text entries stay untouched).
    #[serde(default)]
    pub class: Option<String>,
    /// Official chat-template filename pinned for this model's GGUF
    /// family, when the published GGUFs ship without one embedded
    /// (live case: glm-ocr, upstream llama.cpp #31298 - default framing
    /// can loop). Resolved through [`chat_template_source`], vendored
    /// to the user's config dir at pull time, and suggested as an
    /// explicit `[model_overrides.<name>] chat_template_file` pin.
    #[serde(default)]
    pub chat_template: Option<String>,
}

/// Speculative-decoding draft pairs: models matching `model_glob` (prefix
/// match on registry name) use `draft_repo` via `--spec-type draft-simple`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SpecPair {
    /// Prefix matched against the resolved model name, e.g. "qwen3-".
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

/// Embedded catalog-pinned chat templates. One typed arm per pinned
/// file: additions land consciously, with the content diffed against
/// the vendor's canonical source at review time.
#[must_use]
pub fn chat_template_source(file: &str) -> Option<&'static str> {
    match file {
        "glm-ocr.jinja" => Some(include_str!("catalog-templates/glm-ocr.jinja")),
        _ => None,
    }
}

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
                format!(
                    "no catalog model matching {name:?}; try `blazar search <query>` or pull an explicit owner/repo:quant"
                )
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

/// Spec-draft pair constrained to one engine spec type (e.g. the manual
/// `spec = "eagle3"` mode must resolve an eagle3 head, not the generic
/// draft-simple sibling that plain auto would pick).
#[must_use]
pub fn spec_pair_for_typed(model: &str, spec_type: &str) -> Option<&'static SpecPair> {
    catalog()
        .spec_pairs
        .iter()
        .find(|p| model.starts_with(&p.model_prefix) && p.spec_type == spec_type)
}

/// Spec-draft pair for a model under a spec MODE (the
/// `is_valid_spec_mode` vocabulary): typed modes resolve their typed
/// head, file-less modes (off/mtp/ngram*) never consume a draft file,
/// anything else (auto, unknown families) falls back to the generic
/// catalog pair. Single source for spawn-time draft resolution and the
/// `spec_autopull` lane so both agree on what "a draft exists" means.
#[must_use]
pub fn pair_for_spec_mode(model: &str, spec_mode: &str) -> Option<&'static SpecPair> {
    match spec_mode {
        "eagle3" => spec_pair_for_typed(model, "draft-eagle3"),
        "dflash" => spec_pair_for_typed(model, "draft-dflash"),
        "dspark" => spec_pair_for_typed(model, "draft-dspark"),
        // DSpine drafts are HF safetensors dirs (sglang-only lane);
        // GGUF lanes refuse the mode with teaching at compile instead
        // of consuming the pair.
        "dspine" => spec_pair_for_typed(model, "draft-dspine"),
        // off/mtp/ngram never consume an external draft file.
        "off" | "mtp" | "ngram" | "ngram-map-k" | "ngram-map-k4v" | "ngram-mod" | "ngram-cache" => {
            None
        }
        _ => spec_pair_for(model),
    }
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

/// Split an on-demand `LoRA` variant request (`model+adapter`) into its
/// base model name and adapter stem. `+` cannot occur in pulled or
/// imported model names, so the split is unambiguous; `model:lora` was
/// rejected because colon is ollama's `model:tag` separator. Degenerate
/// forms (`+adapter`, `model+`, `model++x`) pass through unchanged and
/// fail later model resolution with the user's own spelling.
#[must_use]
pub fn split_lora_suffix(requested: &str) -> (&str, Option<&str>) {
    match requested.split_once('+') {
        Some((base, stem)) if !base.is_empty() && !stem.is_empty() => (base, Some(stem)),
        _ => (requested, None),
    }
}

/// Registered adapter name for a `model+adapter` request stem. The
/// sglang lane registers adapters as `--lora-paths <name>=<path>` where
/// `<name>` is the adapter path's file stem, and the engine then picks
/// the adapter per request by that exact name (its native
/// `base:adapter` model syntax). This mirrors the supervisor's lane
/// resolution (file-name OR file-stem match, exactly one) so every
/// spelling it accepts maps to the name the child actually registered -
/// including the dotted-directory edge (`.../my.adapter.v2` matches a
/// `+my.adapter.v2` request but registers as `my.adapter`).
/// `None` = no unambiguous match (the caller's own teaching covers it).
#[must_use]
pub fn lora_registered_name(rows: &[crate::store::LoraRow], stem: &str) -> Option<String> {
    let matches: Vec<&crate::store::LoraRow> = rows
        .iter()
        .filter(|l| {
            let p = std::path::Path::new(&l.path);
            p.file_name().is_some_and(|s| s == stem) || p.file_stem().is_some_and(|s| s == stem)
        })
        .collect();
    match matches.as_slice() {
        [only] => std::path::Path::new(&only.path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned()),
        _ => None,
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn unit__lora_registered_name__stem_matching_and_dotted_edge() {
        let row = |id: i64, path: &str| crate::store::LoraRow {
            id,
            model_name: "m".into(),
            path: path.into(),
            scale: 1.0,
        };
        let rows = vec![
            row(1, "/models/adapters/anonymizer"),
            row(2, "/models/adapters/my.adapter.v2"),
        ];
        // Plain stem -> the registered (file-stem) name.
        assert_eq!(
            lora_registered_name(&rows, "anonymizer").as_deref(),
            Some("anonymizer")
        );
        // Dotted directory: a file-name spelling registers under the
        // file STEM - the colon stamp must use that name.
        assert_eq!(
            lora_registered_name(&rows, "my.adapter.v2").as_deref(),
            Some("my.adapter")
        );
        // No match and ambiguous matches resolve to nothing (the
        // caller's own teaching covers those).
        assert_eq!(lora_registered_name(&rows, "absent"), None);
        let dup = vec![row(3, "/a/x"), row(4, "/b/x")];
        assert_eq!(lora_registered_name(&dup, "x"), None);
    }

    #[test]
    fn unit__split_lora_suffix__variants_and_degenerates() {
        assert_eq!(split_lora_suffix("m+foo"), ("m", Some("foo")));
        assert_eq!(
            split_lora_suffix("owner/m:tag+foo"),
            ("owner/m:tag", Some("foo"))
        );
        // Degenerate forms pass through unchanged (model resolution
        // reports the user's own spelling, never a wrong base match).
        assert_eq!(split_lora_suffix("m+"), ("m+", None));
        assert_eq!(split_lora_suffix("+foo"), ("+foo", None));
        assert_eq!(split_lora_suffix("m"), ("m", None));
        assert_eq!(split_lora_suffix("m:tag"), ("m:tag", None));
    }

    #[test]
    fn unit__pair_for_spec_mode__typed_fileless_and_generic() {
        // qwen3 carries both a generic pair and an eagle3 head in the
        // catalog (mirrors the resolve_draft_path pins).
        assert!(pair_for_spec_mode("qwen3-8b", "auto").is_some());
        assert!(pair_for_spec_mode("qwen3-8b", "eagle3").is_some());
        // Typed DSPINE pair resolves the tencent drafter, and only for
        // the 8B prefix it is pinned to.
        let dspine = pair_for_spec_mode("qwen3-8b", "dspine").expect("dspine pair for qwen3-8b");
        assert_eq!(dspine.spec_type, "draft-dspine");
        assert_eq!(dspine.draft_repo, "tencent/Qwen3-8B-DSpine-b16");
        assert!(pair_for_spec_mode("m1", "dspine").is_none());
        // dspine is typed: it must not leak into the generic auto pair
        // (auto keeps resolving the draft-simple head for qwen3).
        let auto = pair_for_spec_mode("qwen3-8b", "auto").expect("auto pair");
        assert_eq!(auto.spec_type, "draft-simple");
        // File-less modes never consume an external draft - at any
        // model, regardless of catalog content.
        for mode in ["off", "mtp", "ngram", "ngram-map-k", "ngram-mod"] {
            assert!(
                pair_for_spec_mode("qwen3-8b", mode).is_none(),
                "{mode} must never resolve a draft file"
            );
        }
        // Unknown model families have no pair under any file mode.
        assert!(pair_for_spec_mode("m1", "auto").is_none());
        // 2026 families: ggml-org ships gpt-oss-20b as the speculative
        // drafter for its 120B sibling, and gemma-4-e2b drafts the 12B.
        // Both are draft-simple (plain same-family smaller model).
        let oss = pair_for_spec_mode("gpt-oss-120b", "auto").expect("auto pair for gpt-oss-120b");
        assert_eq!(oss.spec_type, "draft-simple");
        assert_eq!(oss.draft_repo, "ggml-org/gpt-oss-20b-GGUF");
        let g4 = pair_for_spec_mode("gemma-4-12b", "auto").expect("auto pair for gemma-4-12b");
        assert_eq!(g4.spec_type, "draft-simple");
        assert_eq!(g4.draft_repo, "unsloth/gemma-4-E2B-it-GGUF");
        // The gpt-oss-20b entry itself is a plain pull target - no pair
        // of its own (it IS the drafter).
        assert!(pair_for_spec_mode("gpt-oss-20b", "auto").is_none());
    }

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
        assert!(err.contains("blazar search"), "{err}");
    }

    #[test]
    fn unit__catalog_breadth__non_llm_classes_resolve() {
        // One entry per non-LLM class the catalog now covers; each repo
        // carries a live pull receipt or a freeze-time HF listing check.
        for (name, repo) in [
            ("bge-small-en", "mradermacher/bge-small-en-GGUF"),
            ("bge-reranker-v2-m3", "gpustack/bge-reranker-v2-m3-GGUF"),
            ("laya", "ggml-org/Laya-GGUF"),
            ("clip-vit-base-patch32", "openai/clip-vit-base-patch32"),
            ("qwen3-tts-1.7b", "ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF"),
            ("whisper-base", "openai/whisper-base"),
            ("moss-image", "leejet/MOSS-Image-Base-GGUF"),
            ("flux-3-klein", "leejet/flux-3-image-klein-GGUF"),
            ("qwen-image-edit", "QuantStack/Qwen-Image-Edit-GGUF"),
            ("wan2.1-t2v", "Comfy-Org/Wan_2.1_ComfyUI_repackaged"),
            // The steering target for `glmocr` pulls: the official GGUF
            // exports under the native glm4 arch llama-server loads.
            ("glm-ocr", "ggml-org/GLM-OCR-GGUF"),
            // Clover MoE landed upstream (b12069+); the Diffuspace mirror
            // is the only ungated Q4_K_M hosting verified at freeze.
            ("clover-35b-a3b", "Diffuspace/Clover-35B-A3B-Preview-GGUF"),
            // Multimodal embedder (gemma-embedding2 arch + mmproj): text
            // rides the child embeddings lane, images the G2b content-
            // array path. No Q4_K_M upstream - quant fallback is fine.
            ("embeddinggemma-2", "ggml-org/embeddinggemma-2-GGUF"),
            // The SigLIP candle lane's reference model: safetensors on
            // main (no revision pin needed), sparse config verified.
            ("siglip-base-patch16-224", "google/siglip-base-patch16-224"),
        ] {
            let e = resolve(name).unwrap_or_else(|err| panic!("{name}: {err}"));
            assert_eq!(e.repo, repo, "{name}");
        }
    }

    #[test]
    fn unit__catalog_revision__clip_pins_pr15_and_only_clip() {
        // CLIP's safetensors conversion lives on refs/pr/15 - the entry
        // pins it so `blazar pull clip-vit-base-patch32` works bare.
        assert_eq!(
            resolve("clip-vit-base-patch32")
                .unwrap()
                .revision
                .as_deref(),
            Some("refs/pr/15")
        );
        // Every other entry pulls from main: no accidental pins.
        for e in &catalog().entries {
            if e.short_name != "clip-vit-base-patch32" {
                assert!(
                    e.revision.as_deref().is_none_or(str::is_empty),
                    "unexpected revision pin on {}",
                    e.short_name
                );
            }
        }
    }

    #[test]
    fn unit__spec_pair__prefix_match() {
        let pair = spec_pair_for("qwen3-coder-30b").expect("qwen3 family has a draft pair");
        assert_eq!(pair.spec_type, "draft-simple");
        assert!(pair.draft_repo.starts_with("ggml-org/Qwen3-0.6B"));
        assert!(spec_pair_for("gemma3-4b").is_none());
    }

    #[test]
    fn unit__spec_pair_typed__eagle3_pair_order_preserved() {
        // Typed lookup finds the eagle3 head for the exact family...
        let pair =
            spec_pair_for_typed("qwen3-8b", "draft-eagle3").expect("qwen3-8b has an eagle3 pair");
        assert_eq!(pair.spec_type, "draft-eagle3");
        assert!(pair.draft_repo.starts_with("williamliao/Qwen3-8B-EAGLE3"));
        // ...while untyped auto lookup on the SAME name keeps the generic
        // draft-simple sibling (order in catalog.json is precedence).
        assert_eq!(
            spec_pair_for("qwen3-8b").expect("auto pair").spec_type,
            "draft-simple"
        );
        // Other families have no eagle3 pair yet.
        assert!(spec_pair_for_typed("qwen3-14b", "draft-eagle3").is_none());
    }

    #[test]
    fn unit__levenshtein__known_values() {
        assert_eq!(levenshtein("", ""), 0);
        assert_eq!(levenshtein("abc", "abc"), 0);
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("qwen3-0.5b", "qwen3-0.6b"), 1);
    }

    #[test]
    #[allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    fn unit__chat_template_source__catalog_pins_resolve_and_unknown_refuses() {
        let glm = resolve("glm-ocr").expect("glm-ocr cataloged");
        let file = glm.chat_template.as_deref().expect("template pinned");
        assert_eq!(file, "glm-ocr.jinja");
        let body = chat_template_source(file).expect("pinned template embedded");
        // The official zai-org GLM-OCR template opens with the GLM
        // grounding prefix; anything else means the asset drifted.
        assert!(body.starts_with("[gMASK]<sop>"), "{body:.60}");
        // Unknown pins resolve to None: callers skip guidance rather
        // than vendoring an empty file.
        assert!(chat_template_source("nope.jinja").is_none());
        // Entries without a pin stay untouched by the field's default.
        assert!(
            resolve("qwen3-8b")
                .expect("qwen3-8b cataloged")
                .chat_template
                .is_none()
        );
    }
}
