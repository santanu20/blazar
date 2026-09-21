//! sd.cpp diffusion component sets.
//!
//! A diffusion GGUF pull is not one file: the `DiT` checkpoint that names
//! the repo is unservable without its `VAE` and text encoder (sd-server
//! refuses to boot on the `DiT` alone — verified live against
//! master-890-74988b2). Component files live in DIFFERENT upstream
//! repos picked per model family; the pairing is upstream policy (which
//! `VAE` a model version expects), not something derivable from the `DiT`
//! bytes, so it lives in this curated table. Every entry is verified
//! against the repos it names before landing here.

use crate::hf::{FilePlan, HfModelInfo};

/// One component's upstream location. `path` may contain a `{quant}`
/// placeholder resolved against the `DiT`'s pulled quant.
pub struct ComponentSource {
    pub repo: &'static str,
    pub repo_path: &'static str,
}

/// A diffusion model family: detection token plus the component sources
/// sd-server needs to boot it.
pub struct DiffusionFamily {
    /// Lowercased token matched against the pull repo (`Qwen-Image-2.1`
    /// in any owner's repo name matches). `DiT` conversions keep the
    /// family name in the repo, so the token is stable across mirrors.
    pub token: &'static str,
    pub display: &'static str,
    /// Image decoder — required (sd-server dies mid-load without it).
    pub vae: ComponentSource,
    /// Prompt text encoder — required (the `DiT` cannot embed prompts).
    pub text_encoder: ComponentSource,
    /// Quant used when the `DiT`'s quant has no matching `TE` file (`TE`
    /// repos publish fewer quants than `DiT` converters cut).
    pub text_encoder_fallback_quant: &'static str,
    /// Vision encoder for image edits (`--llm_vision`) — optional:
    /// skipped with a warning when upstream does not carry it.
    pub vision_encoder: Option<ComponentSource>,
}

/// Verified families. Keep this list honest: an entry asserts the
/// component pairing boots on sd-server, proven live before landing.
const FAMILIES: &[DiffusionFamily] = &[DiffusionFamily {
    token: "qwen-image-2.1",
    display: "Qwen-Image-2.1",
    vae: ComponentSource {
        repo: "Comfy-Org/Qwen-Image-2.1",
        repo_path: "vae/qwen_image_2.1_vae_bf16.safetensors",
    },
    text_encoder: ComponentSource {
        repo: "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        repo_path: "Qwen3VL-8B-Instruct-{quant}.gguf",
    },
    text_encoder_fallback_quant: "Q4_K_M",
    vision_encoder: Some(ComponentSource {
        repo: "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        repo_path: "mmproj-Qwen3VL-8B-Instruct-F16.gguf",
    }),
}];

/// Family for a pull repo, if the repo carries a known family token.
#[must_use]
pub fn diffusion_family(repo: &str) -> Option<&'static DiffusionFamily> {
    let lower = repo.to_lowercase();
    FAMILIES.iter().find(|f| lower.contains(f.token))
}

/// Every family token, for teaching messages.
#[must_use]
pub fn supported_families() -> Vec<&'static str> {
    FAMILIES.iter().map(|f| f.display).collect()
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

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

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
        assert_eq!(supported_families(), vec!["Qwen-Image-2.1"]);
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
}
