//! sd.cpp diffusion component sets.
//!
//! A diffusion GGUF pull is not one file: the `DiT` checkpoint that names
//! the repo is unservable without its `VAE` and text encoder(s) (sd-server
//! refuses to boot on the `DiT` alone — verified live against
//! master-890-74988b2). Component files live in DIFFERENT upstream
//! repos picked per model family; the pairing is upstream policy (which
//! `VAE` a model version expects), not something derivable from the `DiT`
//! bytes, so it lives in this curated table. Every entry is verified
//! against the repos it names before landing here.
//!
//! Families differ in flag dialect, which is why specs carry their own
//! flag: Qwen-Image pairs `--vae`/`--llm` (a Qwen3-VL encoder), FLUX
//! pairs `--vae`/`--t5xxl`/`--clip_l` (T5-XXL + CLIP-L, upstream
//! `docs/flux.md`). The store keeps one flag-keyed set per row.

use crate::hf::{FilePlan, HfModelInfo};
use std::path::Path;

/// One component's upstream location. `path` may contain a `{quant}`
/// placeholder resolved against the `DiT`'s pulled quant.
pub struct ComponentSource {
    pub repo: &'static str,
    pub repo_path: &'static str,
}

/// One family component: the sd-server flag it rides, where it comes
/// from, and whether a pull may complete without it.
pub struct ComponentSpec {
    /// sd-server argv flag (`--vae`, `--llm`, `--t5xxl`, `--clip_l`...).
    pub flag: &'static str,
    pub source: ComponentSource,
    /// Required components gate pull success and re-pull repair; optional
    /// ones (vision encoders) skip with a warning when absent upstream.
    pub required: bool,
    /// Quant used when the `DiT`'s quant has no matching component file
    /// (component repos publish fewer quants than `DiT` converters cut).
    pub fallback_quant: Option<&'static str>,
}

/// A diffusion model family: detection token plus the component specs
/// sd-server needs to boot it.
pub struct DiffusionFamily {
    /// Lowercased token matched against the pull repo (`Qwen-Image-2.1`
    /// in any owner's repo name matches). `DiT` conversions keep the
    /// family name in the repo, so the token is stable across mirrors.
    pub token: &'static str,
    pub display: &'static str,
    pub components: &'static [ComponentSpec],
}

/// Verified families. Keep this list honest: an entry asserts the
/// component pairing boots on sd-server, proven live before landing.
const FAMILIES: &[DiffusionFamily] = &[
    DiffusionFamily {
        token: "qwen-image-2.1",
        display: "Qwen-Image-2.1",
        components: &[
            ComponentSpec {
                flag: "--vae",
                source: ComponentSource {
                    repo: "Comfy-Org/Qwen-Image-2.1",
                    repo_path: "vae/qwen_image_2.1_vae_bf16.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--llm",
                source: ComponentSource {
                    repo: "Qwen/Qwen3-VL-8B-Instruct-GGUF",
                    repo_path: "Qwen3VL-8B-Instruct-{quant}.gguf",
                },
                required: true,
                fallback_quant: Some("Q4_K_M"),
            },
            ComponentSpec {
                flag: "--llm_vision",
                source: ComponentSource {
                    repo: "Qwen/Qwen3-VL-8B-Instruct-GGUF",
                    repo_path: "mmproj-Qwen3VL-8B-Instruct-F16.gguf",
                },
                required: false,
                fallback_quant: None,
            },
        ],
    },
    DiffusionFamily {
        // Covers FLUX.1-dev, FLUX.1-schnell and their GGUF mirrors;
        // FLUX.1-Kontext rides the same component set (upstream
        // docs/flux.md + docs/kontext.md). FLUX.2 is a different family
        // (Mistral text encoder) and stays unmatched until curated.
        token: "flux.1",
        display: "FLUX.1",
        components: &[
            ComponentSpec {
                flag: "--vae",
                source: ComponentSource {
                    // The dev repo is gated; schnell ships the identical
                    // `ae.safetensors` (autoencoder shared across FLUX.1).
                    repo: "black-forest-labs/FLUX.1-schnell",
                    repo_path: "ae.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--t5xxl",
                source: ComponentSource {
                    repo: "city96/t5-v1_1-xxl-encoder-gguf",
                    repo_path: "t5-v1_1-xxl-encoder-{quant}.gguf",
                },
                required: true,
                fallback_quant: Some("Q4_K_M"),
            },
            ComponentSpec {
                flag: "--clip_l",
                source: ComponentSource {
                    repo: "comfyanonymous/flux_text_encoders",
                    repo_path: "clip_l.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
        ],
    },
];

/// Family for a pull repo, if the repo carries a known family token.
#[must_use]
pub fn diffusion_family(repo: &str) -> Option<&'static DiffusionFamily> {
    let lower = repo.to_lowercase();
    FAMILIES.iter().find(|f| lower.contains(f.token))
}

/// Every family display name, for teaching messages.
#[must_use]
pub fn supported_families() -> Vec<&'static str> {
    FAMILIES.iter().map(|f| f.display).collect()
}

/// Whether the family of `repo` (when known) carries a vision encoder
/// for image edits. `None` = not a diffusion family repo.
#[must_use]
pub fn family_supports_edits(repo: &str) -> Option<bool> {
    diffusion_family(repo).map(|f| f.components.iter().any(|c| c.flag == "--llm_vision"))
}

/// Resolve one component against its repo listing: the exact
/// `{quant}`-substituted path when it exists, else `None` (caller picks
/// fallback or skip).
#[must_use]
pub fn component_plan(
    info: &HfModelInfo,
    source: &ComponentSource,
    quant: &str,
) -> Option<FilePlan> {
    let wanted = source.repo_path.replace("{quant}", quant);
    info.siblings
        .iter()
        .find(|s| s.rfilename.eq_ignore_ascii_case(&wanted))
        .map(|s| FilePlan {
            filename: s.rfilename.clone(),
            bytes: s.lfs.as_ref().and_then(|l| l.size).or(s.size).unwrap_or(0),
            sha256: s.lfs.as_ref().map(|l| l.sha256.clone()),
        })
}

/// The repo-relative leaf name a component downloads as (flat into the
/// models dir, mirroring the `DiT` and mmproj conventions).
#[must_use]
pub fn component_leaf(source: &ComponentSource, quant: &str) -> String {
    source
        .repo_path
        .replace("{quant}", quant)
        .rsplit('/')
        .next()
        .unwrap_or(source.repo_path)
        .to_string()
}

/// A row's set is repair-worthy when any REQUIRED family component is
/// absent from the row or dead on disk. Optional components (vision)
/// never trigger repair — they only disable edits.
#[must_use]
pub fn required_component_missing(row: &blazar_core::ModelRow, family: &DiffusionFamily) -> bool {
    family.components.iter().filter(|c| c.required).any(|c| {
        row.component(c.flag)
            .is_none_or(|p| !Path::new(p).is_file())
    })
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use blazar_core::store::ComponentFile;

    fn row_with(components: &[(&str, &str)]) -> blazar_core::ModelRow {
        blazar_core::ModelRow {
            name: "m".into(),
            repo: "r".into(),
            quant: "Q4_K_M".into(),
            path: "p.gguf".into(),
            bytes: 1,
            sha256: None,
            mmproj_path: None,
            components: components
                .iter()
                .map(|(f, p)| ComponentFile::new(f, p))
                .collect(),
            shards: 1,
            arch: None,
            params: None,
            ctx_train: None,
            pulled_at: 1,
        }
    }

    #[test]
    fn unit__diffusion_family__token_matches_owner_and_case_variants() {
        let f = diffusion_family("abenzerps/Qwen-Image-2.1-GGUF").unwrap();
        assert_eq!(f.display, "Qwen-Image-2.1");
        assert!(diffusion_family("leejet/qwen-image-2.1-gguf").is_some());
        assert!(diffusion_family("Comfy-Org/Qwen-Image-2.1").is_some());
        // v1 (different model) and unrelated repos stay unmatched —
        // their `VAE` pairing is NOT interchangeable with 2.1.
        assert!(diffusion_family("cppee/Qwen-Image-GGUF").is_none());
        assert!(diffusion_family("qwen/qwen3-8b-gguf").is_none());
        // FLUX.1 in all its housings: official dev/schnell, GGUF mirrors,
        // Kontext (same component set).
        assert!(diffusion_family("black-forest-labs/FLUX.1-dev").is_some());
        assert!(diffusion_family("city96/FLUX.1-schnell-gguf").is_some());
        assert!(diffusion_family("black-forest-labs/FLUX.1-Kontext-dev").is_some());
        assert!(diffusion_family("QuantStack/FLUX.1-dev-GGUF").is_some());
        // FLUX.2 is a different family (Mistral TE) — not curatable by
        // reusing the FLUX.1 set.
        assert!(diffusion_family("black-forest-labs/FLUX.2-dev").is_none());
        assert_eq!(supported_families(), vec!["Qwen-Image-2.1", "FLUX.1"]);
    }

    #[test]
    fn unit__family_supports_edits__vision_flag_decides() {
        assert_eq!(family_supports_edits("x/Qwen-Image-2.1-GGUF"), Some(true));
        assert_eq!(family_supports_edits("city96/FLUX.1-dev-gguf"), Some(false));
        assert_eq!(family_supports_edits("qwen/qwen3-8b-gguf"), None);
    }

    #[test]
    fn unit__component_leaf__flattens_subdirs_and_resolves_quant() {
        let te = ComponentSource {
            repo: "Qwen/Qwen3-VL-8B-Instruct-GGUF",
            repo_path: "Qwen3VL-8B-Instruct-{quant}.gguf",
        };
        assert_eq!(
            component_leaf(&te, "Q4_K_M"),
            "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
        );
        let vae = ComponentSource {
            repo: "Comfy-Org/Qwen-Image-2.1",
            repo_path: "vae/qwen_image_2.1_vae_bf16.safetensors",
        };
        assert_eq!(
            component_leaf(&vae, "Q4_K_M"),
            "qwen_image_2.1_vae_bf16.safetensors"
        );
    }

    #[test]
    fn unit__required_component_missing__required_gates_optional_never_does() {
        let family = diffusion_family("city96/FLUX.1-dev-gguf").unwrap();
        // Never-attached row = repair.
        assert!(required_component_missing(&row_with(&[]), family));
        // All required present and alive (vision absent: irrelevant for
        // FLUX, which has none).
        let tmp = tempfile::tempdir().unwrap();
        let vae = tmp.path().join("ae.safetensors");
        let t5 = tmp.path().join("t5.gguf");
        let clip = tmp.path().join("clip_l.safetensors");
        for f in [&vae, &t5, &clip] {
            std::fs::write(f, b"x").unwrap();
        }
        let mut row = row_with(&[]);
        row.components = vec![
            ComponentFile::new("--vae", vae.to_str().unwrap()),
            ComponentFile::new("--t5xxl", t5.to_str().unwrap()),
            ComponentFile::new("--clip_l", clip.to_str().unwrap()),
        ];
        assert!(!required_component_missing(&row, family));
        // Dead required file = repair (stale store row).
        row.components[1].path = tmp.path().join("gone.gguf").to_str().unwrap().into();
        assert!(required_component_missing(&row, family));
        // Qwen vision encoder is optional: its absence never repairs.
        let qwen = diffusion_family("x/Qwen-Image-2.1-GGUF").unwrap();
        let mut qrow = row_with(&[]);
        qrow.components = vec![
            ComponentFile::new("--vae", vae.to_str().unwrap()),
            ComponentFile::new("--llm", t5.to_str().unwrap()),
        ];
        assert!(!required_component_missing(&qrow, qwen));
    }
}
