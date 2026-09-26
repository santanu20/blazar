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
//!
//! A second family shape exists: STANDALONE checkpoints (SD 1.5, SDXL)
//! carry their `VAE` and text encoders inside the single `.safetensors`
//! file, so sd-server boots them with `-m/--model <file>` alone. Those
//! families carry no component specs — the pull pins the model file
//! itself and the row stores a self-referencing `--model` component
//! that marks it for diffusion routing.

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

/// What a family generates. sd-server derives its serving mode from the
/// LOADED model (capabilities advertises `supported_modes` per boot) —
/// there is no boot-time mode flag in current builds — so the gateway
/// uses this field to teach the right surface (`/v1/videos` vs
/// `/v1/images`) before any child exists.
pub enum FamilyMode {
    Img,
    Vid,
}

/// A diffusion model family: detection token plus the component specs
/// sd-server needs to boot it.
pub struct DiffusionFamily {
    /// Lowercased token matched against the pull repo (`Qwen-Image-2.1`
    /// in any owner's repo name matches). `DiT` conversions keep the
    /// family name in the repo, so the token is stable across mirrors.
    pub token: &'static str,
    pub display: &'static str,
    /// Substrings that veto an otherwise-matching repo. Tokens are
    /// broad by design (`qwen-image` must survive any mirror name), so
    /// siblings sharing the prefix but pairing DIFFERENT components
    /// (2.1, 2512, Edit) are kept out until curated for real.
    pub excludes: &'static [&'static str],
    /// Standalone-checkpoint families name the exact repo files that
    /// ARE the model (`sd_xl_base_1.0.safetensors`). `None` = component
    /// family; the pull picks the `DiT` by quant as usual.
    pub standalone_files: Option<&'static [&'static str]>,
    pub components: &'static [ComponentSpec],
    pub mode: FamilyMode,
}

/// Verified families. Keep this list honest: an entry asserts the
/// component pairing boots on sd-server, proven live before landing.
const FAMILIES: &[DiffusionFamily] = &[
    DiffusionFamily {
        token: "qwen-image-2.1",
        display: "Qwen-Image-2.1",
        excludes: &[],
        standalone_files: None,
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
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // Qwen-Image v1 (upstream docs/qwen_image.md): the DiT pairs
        // the v1 qwen_image_vae — NOT the 2.1 one — and a Qwen2.5-VL-7B
        // text encoder. The broad token needs excludes: 2.1 has its own
        // entry above, 2512's VAE pairing is unverified here, and Edit
        // is a separate instruction-tuned DiT with its own set.
        token: "qwen-image",
        display: "Qwen-Image",
        excludes: &["qwen-image-2.1", "qwen-image-2512", "qwen-image-edit"],
        standalone_files: None,
        components: &[
            ComponentSpec {
                flag: "--vae",
                source: ComponentSource {
                    repo: "Comfy-Org/Qwen-Image_ComfyUI",
                    repo_path: "split_files/vae/qwen_image_vae.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--llm",
                source: ComponentSource {
                    // Dot-separated quants in this repo; fallback Q4_K_M
                    // verified live (4683072512 bytes).
                    repo: "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF",
                    repo_path: "Qwen2.5-VL-7B-Instruct.{quant}.gguf",
                },
                required: true,
                fallback_quant: Some("Q4_K_M"),
            },
        ],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // Covers FLUX.1-dev, FLUX.1-schnell and their GGUF mirrors;
        // FLUX.1-Kontext rides the same component set (upstream
        // docs/flux.md + docs/kontext.md). FLUX.2 has its own curated
        // entry below (Mistral text encoder).
        token: "flux.1",
        display: "FLUX.1",
        excludes: &[],
        standalone_files: None,
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
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // Z-Image-Turbo and Z-Image base share one component set
        // (upstream docs/z_image.md): the FLUX.1 autoencoder plus a
        // Qwen3-4B text encoder. The token cannot collide with
        // Qwen-Image ("qwen-image" never contains "z-image").
        token: "z-image",
        display: "Z-Image",
        excludes: &[],
        standalone_files: None,
        components: &[
            ComponentSpec {
                flag: "--vae",
                source: ComponentSource {
                    // Same file the FLUX.1 family pulls — byte-exact
                    // reuse kicks in when both families live on one
                    // box. BFL repos are gated:auto: the pull surfaces
                    // the HF_TOKEN teaching until the license is
                    // accepted.
                    repo: "black-forest-labs/FLUX.1-schnell",
                    repo_path: "ae.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--llm",
                source: ComponentSource {
                    repo: "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                    repo_path: "Qwen3-4B-Instruct-2507-{quant}.gguf",
                },
                required: true,
                fallback_quant: Some("Q4_K_M"),
            },
        ],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // Curated to the Chroma1-HD line, the current head of
        // silveroxides/Chroma-GGUF (the repo also carries 50+
        // chroma-unlocked-v* variants under the same t5xxl + FLUX.1
        // VAE contract; HD quants verified: Q4_0/Q8_0/BF16 — the
        // fallback lands on Q4_0 for K-quants the repo never cut).
        // Upstream docs/chroma.md.
        token: "chroma",
        display: "Chroma",
        excludes: &[],
        standalone_files: None,
        components: &[
            ComponentSpec {
                flag: "--vae",
                source: ComponentSource {
                    // docs/chroma.md names the FLUX.1-dev autoencoder;
                    // BFL repo, gated:auto (see the Z-Image note).
                    repo: "black-forest-labs/FLUX.1-dev",
                    repo_path: "ae.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--t5xxl",
                source: ComponentSource {
                    // The doc-proven fp16 encoder; no quant variants
                    // exist for this safetensors file.
                    repo: "comfyanonymous/flux_text_encoders",
                    repo_path: "t5xxl_fp16.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
        ],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // Token "flux.2-dev" deliberately excludes the klein variants:
        // klein pairs Qwen3 text encoders while dev pairs
        // Mistral-Small (upstream docs/flux2.md) — a shared token
        // would mis-attach the 24B Mistral encoder to klein pulls.
        // Klein stays unmatched until curated separately.
        token: "flux.2-dev",
        display: "FLUX.2-dev",
        excludes: &[],
        standalone_files: None,
        components: &[
            ComponentSpec {
                flag: "--vae",
                source: ComponentSource {
                    // Repo filename is ae.safetensors (the
                    // flux2_ae.safetensors in doc commands is a local
                    // rename). BFL repo, gated:auto.
                    repo: "black-forest-labs/FLUX.2-dev",
                    repo_path: "ae.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--llm",
                source: ComponentSource {
                    repo: "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
                    repo_path: "Mistral-Small-3.2-24B-Instruct-2506-{quant}.gguf",
                },
                required: true,
                fallback_quant: Some("Q4_K_M"),
            },
        ],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // SDXL: the most-downloaded image model on HF. A STANDALONE
        // checkpoint — sd_xl_base_1.0.safetensors embeds the VAE and
        // both CLIP text encoders, so sd-server boots it with
        // `-m/--model` alone (upstream docs/sd.md txt2img example). The
        // token also catches sdxl-turbo housings; finetune checkpoints
        // (Juggernaut, RealVis, Pony...) live in repos whose names
        // carry no family token and stay uncurated by design.
        token: "stable-diffusion-xl",
        display: "SDXL",
        excludes: &[],
        standalone_files: Some(&["sd_xl_base_1.0.safetensors"]),
        components: &[],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // SD 1.x: second most-downloaded family on HF, same standalone
        // contract as SDXL (v1-5-pruned-emaonly embeds VAE+CLIP). The
        // v1 token cannot collide with SDXL or SD3.5 repos (neither
        // contains "stable-diffusion-v1"). SD3/3.5 pair a different
        // component set (clip_l+clip_g+t5xxl) and stay uncurated.
        token: "stable-diffusion-v1",
        display: "SD 1.5",
        excludes: &[],
        standalone_files: Some(&["v1-5-pruned-emaonly.safetensors", "sd-v1-4.ckpt"]),
        components: &[],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // Wan 2.1 T2V 1.3B (upstream docs/wan.md): a video diffusion
        // model — the DiT is a standalone safetensors that still needs
        // its VAE and the umt5-xxl text encoder attached. NOTE: the
        // doc's `-M vid_gen` is the sd-cli/library interface; sd-server
        // has no mode flag (verified live on master-890: `-m` is short
        // for `--model`) — it derives the serving mode from the loaded
        // model, and video parameters (frames/fps) ride requests.
        // Sibling variants (fun/vace/i2v/flf2v) pair different or
        // extra components and stay out; Wan 2.2 A14B needs a second
        // high-noise DiT and is deferred with it.
        token: "wan",
        display: "Wan 2.1 T2V",
        // Broad token on purpose: the two real housings spell it
        // differently (official `Wan2.1-T2V-1.3B`, Comfy-Org
        // `Wan_2.1_ComfyUI_repackaged`) — "wan_2.1" alone would miss
        // the official repos. "2.2" vetoes both Wan 2.2 spellings;
        // the variant suffixes pair different component contracts.
        excludes: &["2.2", "fun", "vace", "i2v", "flf2v"],
        standalone_files: Some(&[
            "split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors",
            "split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors",
        ]),
        components: &[
            ComponentSpec {
                flag: "--vae",
                source: ComponentSource {
                    repo: "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                    repo_path: "split_files/vae/wan_2.1_vae.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--t5xxl",
                source: ComponentSource {
                    repo: "city96/umt5-xxl-encoder-gguf",
                    repo_path: "umt5-xxl-encoder-{quant}.gguf",
                },
                required: true,
                fallback_quant: Some("Q4_K_M"),
            },
        ],
        mode: FamilyMode::Vid,
    },
];

/// Family for a pull repo, if the repo carries a known family token.
/// Table order matters: narrower tokens (qwen-image-2.1) are listed
/// before broader ones (qwen-image), and `excludes` veto the rest.
#[must_use]
pub fn diffusion_family(repo: &str) -> Option<&'static DiffusionFamily> {
    let lower = repo.to_lowercase();
    FAMILIES
        .iter()
        .find(|f| lower.contains(f.token) && !f.excludes.iter().any(|x| lower.contains(x)))
}

/// The exact repo file a standalone family pulls (first listed name
/// the repo actually hosts wins: SD 1.5 repos ship either the v1-5
/// single file or the v1-4 ckpt, never both).
#[must_use]
pub fn standalone_file_plan(info: &HfModelInfo, family: &DiffusionFamily) -> Option<FilePlan> {
    let wanted = family.standalone_files?;
    info.siblings
        .iter()
        .find(|s| wanted.iter().any(|w| s.rfilename.eq_ignore_ascii_case(w)))
        .map(|s| FilePlan {
            filename: s.rfilename.clone(),
            bytes: s.lfs.as_ref().and_then(|l| l.size).or(s.size).unwrap_or(0),
            sha256: s.lfs.as_ref().map(|l| l.sha256.clone()),
        })
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

/// The serving surface a repo's family targets, for pre-boot teaching.
/// `None` = not a diffusion family repo (or an uncurated one).
#[must_use]
pub fn family_mode(repo: &str) -> Option<&'static FamilyMode> {
    diffusion_family(repo).map(|f| &f.mode)
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
    use crate::hf::{HfLfs, HfSibling};
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
        // v1 and unrelated repos stay unmatched by 2.1 — their `VAE`
        // pairing is NOT interchangeable with 2.1.
        assert!(diffusion_family("qwen/qwen3-8b-gguf").is_none());
        // FLUX.1 in all its housings: official dev/schnell, GGUF mirrors,
        // Kontext (same component set).
        assert!(diffusion_family("black-forest-labs/FLUX.1-dev").is_some());
        assert!(diffusion_family("city96/FLUX.1-schnell-gguf").is_some());
        assert!(diffusion_family("black-forest-labs/FLUX.1-Kontext-dev").is_some());
        assert!(diffusion_family("QuantStack/FLUX.1-dev-GGUF").is_some());
        // Z-Image: turbo + base conversions share the set; the token
        // never collides with Qwen-Image.
        assert!(diffusion_family("leejet/Z-Image-Turbo-GGUF").is_some());
        assert!(diffusion_family("leejet/Z-Image-GGUF").is_some());
        assert!(diffusion_family("unsloth/Z-Image-GGUF").is_some());
        // Chroma: the preconverted GGUF repo and the safetensors origin.
        assert!(diffusion_family("silveroxides/Chroma-GGUF").is_some());
        assert!(diffusion_family("lodestones/Chroma").is_some());
        // FLUX.2-dev has its own Mistral-TE set; the klein variants
        // pair Qwen3 encoders and must NOT match it.
        assert!(diffusion_family("city96/FLUX.2-dev-gguf").is_some());
        assert!(diffusion_family("black-forest-labs/FLUX.2-dev").is_some());
        assert!(diffusion_family("leejet/FLUX.2-klein-9B-GGUF").is_none());
        assert!(diffusion_family("leejet/FLUX.2-klein-base-4B-GGUF").is_none());
        assert_eq!(
            supported_families(),
            vec![
                "Qwen-Image-2.1",
                "Qwen-Image",
                "FLUX.1",
                "Z-Image",
                "Chroma",
                "FLUX.2-dev",
                "SDXL",
                "SD 1.5",
                "Wan 2.1 T2V"
            ]
        );
        // Wan 2.1 T2V: the repackaged tree and GGUF-style housings
        // match; sibling variants with different component contracts
        // are vetoed, and Wan 2.2 needs a second DiT (deferred).
        assert!(diffusion_family("Comfy-Org/Wan_2.1_ComfyUI_repackaged").is_some());
        assert!(diffusion_family("Wan-AI/Wan2.1-T2V-1.3B").is_some());
        assert!(diffusion_family("Comfy-Org/Wan2.1-Fun-1.3B-InP").is_none());
        assert!(diffusion_family("Wan-AI/Wan2.2-T2V-A14B-GGUF").is_none());
        assert!(diffusion_family("QuantStack/Wan2.2-T2V-A14B-GGUF").is_none());
    }

    #[test]
    fn unit__diffusion_family__v1_token_matches_and_excludes_vetoes() {
        // v1 repos ride the v1 set (2.1 entry listed first never
        // catches them: no "2.1" in the repo name).
        let f = diffusion_family("QuantStack/Qwen-Image-GGUF").unwrap();
        assert_eq!(f.display, "Qwen-Image");
        assert!(diffusion_family("cppee/Qwen-Image-GGUF").is_some());
        assert!(diffusion_family("Qwen/Qwen-Image").is_some());
        // 2.1 repos resolve to the 2.1 entry even though the v1 token
        // substring-matches too (table order + excludes both guard).
        assert_eq!(
            diffusion_family("abenzerps/Qwen-Image-2.1-GGUF")
                .unwrap()
                .display,
            "Qwen-Image-2.1"
        );
        // Excluded siblings stay uncurated until verified for real.
        assert!(diffusion_family("unsloth/Qwen-Image-2512-GGUF").is_none());
        assert!(diffusion_family("Qwen/Qwen-Image-Edit-2509").is_none());
        // Standalone families.
        assert_eq!(
            diffusion_family("stabilityai/stable-diffusion-xl-base-1.0")
                .unwrap()
                .display,
            "SDXL"
        );
        assert!(diffusion_family("stabilityai/sdxl-turbo").is_none());
        assert_eq!(
            diffusion_family("stable-diffusion-v1-5/stable-diffusion-v1-5")
                .unwrap()
                .display,
            "SD 1.5"
        );
        assert!(diffusion_family("CompVis/stable-diffusion-v1-4").is_some());
        // SD3.5 pairs clip_l+clip_g+t5xxl — different contract, not
        // curated, and neither standalone token matches it.
        assert!(diffusion_family("stabilityai/stable-diffusion-3.5-medium").is_none());
    }

    #[test]
    fn unit__standalone_file_plan__first_listed_file_the_repo_hosts_wins() {
        let sdxl = diffusion_family("stabilityai/stable-diffusion-xl-base-1.0").unwrap();
        // Repo hosts the base file: exact plan with size + sha.
        let info = HfModelInfo {
            id: "stabilityai/stable-diffusion-xl-base-1.0".into(),
            siblings: vec![
                HfSibling {
                    rfilename: "sd_xl_base_1.0.safetensors".into(),
                    size: Some(6_938_078_334),
                    lfs: Some(HfLfs {
                        size: Some(6_938_078_334),
                        sha256: "aa".into(),
                    }),
                },
                HfSibling {
                    rfilename: "unet/diffusion_pytorch_model.fp16.safetensors".into(),
                    size: Some(5_135_149_760),
                    lfs: None,
                },
            ],
            gguf: None,
        };
        let plan = standalone_file_plan(&info, sdxl).unwrap();
        assert_eq!(plan.filename, "sd_xl_base_1.0.safetensors");
        assert_eq!(plan.bytes, 6_938_078_334);
        assert_eq!(plan.sha256.as_deref(), Some("aa"));
        // SD 1.5: repo without the v1-5 file but WITH v1-4 ckpt picks
        // the second listed name.
        let sd15 = diffusion_family("CompVis/stable-diffusion-v1-4").unwrap();
        let info2 = HfModelInfo {
            id: "CompVis/stable-diffusion-v1-4".into(),
            siblings: vec![HfSibling {
                rfilename: "sd-v1-4.ckpt".into(),
                size: Some(4_265_383_744),
                lfs: None,
            }],
            gguf: None,
        };
        assert_eq!(
            standalone_file_plan(&info2, sd15).unwrap().filename,
            "sd-v1-4.ckpt"
        );
        // Component families never standalone-plan.
        let flux = diffusion_family("city96/FLUX.1-dev-gguf").unwrap();
        assert!(standalone_file_plan(&info, flux).is_none());
        // Repo hosting none of the listed names = no plan (caller
        // teaches instead of guessing a shard).
        let info3 = HfModelInfo {
            siblings: vec![HfSibling {
                rfilename: "something-else.safetensors".into(),
                size: Some(1),
                lfs: None,
            }],
            ..fake_info()
        };
        assert!(standalone_file_plan(&info3, sdxl).is_none());
    }

    fn fake_info() -> HfModelInfo {
        HfModelInfo {
            id: "x/y".into(),
            siblings: vec![],
            gguf: None,
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__family_mode__vid_families_and_img_families_partition() {
        use crate::diffusion::FamilyMode;
        // Wan is the video lane; every curated image family reports Img.
        assert!(matches!(
            family_mode("Comfy-Org/Wan_2.1_ComfyUI_repackaged"),
            Some(&FamilyMode::Vid)
        ));
        for repo in [
            "abenzerps/Qwen-Image-2.1-GGUF",
            "city96/FLUX.1-dev-gguf",
            "stabilityai/stable-diffusion-xl-base-1.0",
        ] {
            assert!(
                matches!(family_mode(repo), Some(&FamilyMode::Img)),
                "{repo} must be an image family"
            );
        }
        // Not a diffusion repo (and not a curated one): no mode claim.
        assert!(family_mode("Qwen/Qwen2.5-0.5B-Instruct-GGUF").is_none());
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__family_supports_edits__vision_flag_decides() {
        assert_eq!(family_supports_edits("x/Qwen-Image-2.1-GGUF"), Some(true));
        assert_eq!(family_supports_edits("city96/FLUX.1-dev-gguf"), Some(false));
        assert_eq!(
            family_supports_edits("leejet/Z-Image-Turbo-GGUF"),
            Some(false)
        );
        assert_eq!(
            family_supports_edits("silveroxides/Chroma-GGUF"),
            Some(false)
        );
        // v1 pairs no vision encoder (Edit is a separate DiT); the
        // standalone families have no components at all.
        assert_eq!(
            family_supports_edits("QuantStack/Qwen-Image-GGUF"),
            Some(false)
        );
        assert_eq!(
            family_supports_edits("stabilityai/stable-diffusion-xl-base-1.0"),
            Some(false)
        );
        assert_eq!(
            family_supports_edits("stable-diffusion-v1-5/stable-diffusion-v1-5"),
            Some(false)
        );
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
