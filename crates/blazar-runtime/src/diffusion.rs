//! sd.cpp diffusion component sets.
//!
//! A diffusion GGUF pull is not one file: the `DiT` checkpoint that names
//! the repo is unservable without its `VAE` and text encoder(s) (sd-server
//! refuses to boot on the `DiT` alone - verified live against
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
//! families carry no component specs - the pull pins the model file
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
/// LOADED model (capabilities advertises `supported_modes` per boot) -
/// there is no boot-time mode flag in current builds - so the gateway
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

/// Verified families. Keep this list honest: an entry's component
/// pairing comes from the upstream sd.cpp docs (docs/<family>.md) and
/// every source file is verified against its HF repo listing before
/// landing; entries carrying a receipt note in their comment are
/// additionally boot-proven live on sd-server.
const FAMILIES: &[DiffusionFamily] = &[
    DiffusionFamily {
        // Qwen-Image-Edit (upstream docs/qwen_image_edit.md): an
        // instruction-tuned DiT for image editing - it rides the SAME
        // vae and text encoder as Qwen-Image v1 (byte-identical
        // sources), plus the Qwen2.5-VL mmproj: the edit surface sends
        // before/after images through the encoder (--llm_vision),
        // family_supports_edits keys on exactly that flag. Listed first
        // because the token is narrower than both v1 and 2.1; the
        // 2509/2511 mirror repos (QuantStack, unsloth) ride the same
        // token. GGUF housings spell the file with underscores
        // (Qwen_Image_Edit-Q4_K_M.gguf - QuantStack, verified).
        token: "qwen-image-edit",
        display: "Qwen-Image-Edit",
        excludes: &[],
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
                    repo: "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF",
                    repo_path: "Qwen2.5-VL-7B-Instruct-{quant}.gguf",
                },
                required: true,
                fallback_quant: Some("Q4_K_M"),
            },
            ComponentSpec {
                flag: "--llm_vision",
                source: ComponentSource {
                    repo: "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF",
                    repo_path: "Qwen2.5-VL-7B-Instruct.mmproj-Q8_0.gguf",
                },
                required: false,
                fallback_quant: None,
            },
        ],
        mode: FamilyMode::Img,
    },
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
        // the v1 qwen_image_vae - NOT the 2.1 one - and a Qwen2.5-VL-7B
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
                    // Same file the FLUX.1 family pulls - byte-exact
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
        // VAE contract; HD quants verified: Q4_0/Q8_0/BF16 - the
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
        // Mistral-Small (upstream docs/flux2.md) - a shared token
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
        // checkpoint - sd_xl_base_1.0.safetensors embeds the VAE and
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
        // contains "stable-diffusion-v1"). SD3/3.5 pair clip_l+clip_g
        // +t5xxl and carry their own entry below.
        token: "stable-diffusion-v1",
        display: "SD 1.5",
        excludes: &[],
        standalone_files: Some(&["v1-5-pruned-emaonly.safetensors", "sd-v1-4.ckpt"]),
        components: &[],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // Wan 2.2 TI2V 5B (upstream docs/wan.md): single-DiT video model
        // - unlike the A14B pair it carries no high-noise expert. The
        // QuantStack mirror hosts one DiT GGUF per quant at the repo
        // root (no low/high ambiguity), so the gguf lane's quant pick
        // serves it directly and this entry only contributes the
        // component set. The Wan 2.2 VAE (1.4GB) is a higher-fidelity
        // decoder than the 2.1 VAE and is required by this checkpoint.
        token: "wan2.2-ti2v",
        display: "Wan 2.2 TI2V 5B",
        excludes: &[],
        standalone_files: None,
        components: &[
            ComponentSpec {
                flag: "--vae",
                source: ComponentSource {
                    repo: "QuantStack/Wan2.2-TI2V-5B-GGUF",
                    repo_path: "VAE/Wan2.2_VAE.safetensors",
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
    DiffusionFamily {
        // Wan 2.2 T2V A14B (upstream docs/wan.md): a MoE pair - the doc
        // command passes the low-noise expert as --diffusion-model and
        // the high-noise expert as --high-noise-diffusion-model. The
        // QuantStack mirror hosts both experts in LowNoise//HighNoise/
        // subdirs at identical per-quant sizes, so the gguf lane's
        // largest-wins pick cannot choose a role between them; the
        // standalone list pins a coherent Q2_K pair (the disk-reality
        // quant for a 2x5.3GB MoE on consumer hardware) with the
        // high-noise sibling attached as an exact-path component. Other
        // quants stay reachable through exact-filename pulls
        // (`repo:LowNoise/<file>.gguf`); a quant-symmetric default pair
        // would need pull-time sibling selection, deliberately not
        // built until a second family needs it.
        token: "wan2.2-t2v",
        display: "Wan 2.2 T2V A14B",
        excludes: &[],
        standalone_files: Some(&["LowNoise/Wan2.2-T2V-A14B-LowNoise-Q2_K.gguf"]),
        components: &[
            ComponentSpec {
                flag: "--high-noise-diffusion-model",
                source: ComponentSource {
                    repo: "QuantStack/Wan2.2-T2V-A14B-GGUF",
                    repo_path: "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q2_K.gguf",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                // The doc's A14B command reuses the Wan 2.1 VAE; the
                // mirror ships a copy next to the experts.
                flag: "--vae",
                source: ComponentSource {
                    repo: "QuantStack/Wan2.2-T2V-A14B-GGUF",
                    repo_path: "VAE/Wan2.1_VAE.safetensors",
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
    DiffusionFamily {
        // Wan 2.1 T2V 1.3B (upstream docs/wan.md): a video diffusion
        // model - the DiT is a standalone safetensors that still needs
        // its VAE and the umt5-xxl text encoder attached. NOTE: the
        // doc's `-M vid_gen` is the sd-cli/library interface; sd-server
        // has no mode flag (verified live on master-890: `-m` is short
        // for `--model`) - it derives the serving mode from the loaded
        // model, and video parameters (frames/fps) ride requests.
        // Sibling variants (fun/vace/i2v/flf2v) pair different or
        // extra components and stay out; Wan 2.2 T2V and TI2V carry
        // their own entries above.
        token: "wan",
        display: "Wan 2.1 T2V",
        // Broad token on purpose: the two real housings spell it
        // differently (official `Wan2.1-T2V-1.3B`, Comfy-Org
        // `Wan_2.1_ComfyUI_repackaged`) - "wan_2.1" alone would miss
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
    DiffusionFamily {
        // SD 3.5 (upstream docs/sd3.md): DiT + clip_l + clip_g + t5xxl.
        // The doc command pairs t5xxl_fp16 from the ungated Comfy fp8
        // repo (avoids the gated stabilityai text encoders); both
        // large (16.5GB) and medium (5.1GB) DiT are standalone
        // safetensors that carry no components. The token cannot hit
        // v1/xl repos (no "stable-diffusion-3" substring there).
        token: "stable-diffusion-3",
        display: "SD 3.5",
        excludes: &[],
        standalone_files: Some(&["sd3.5_large.safetensors", "sd3.5_medium.safetensors"]),
        components: &[
            ComponentSpec {
                flag: "--clip_g",
                source: ComponentSource {
                    repo: "Comfy-Org/stable-diffusion-3.5-fp8",
                    repo_path: "text_encoders/clip_g.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--clip_l",
                source: ComponentSource {
                    repo: "Comfy-Org/stable-diffusion-3.5-fp8",
                    repo_path: "text_encoders/clip_l.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--t5xxl",
                source: ComponentSource {
                    repo: "Comfy-Org/stable-diffusion-3.5-fp8",
                    repo_path: "text_encoders/t5xxl_fp16.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
        ],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // FLUX.3 Klein (upstream docs/flux3.md): the DiT pairs the
        // FLUX.2 VAE and a Qwen3-VL-4B text encoder. VAE source is the
        // ungated Comfy-Org byte-copy of the FLUX.2 ae (the BFL repo
        // is gated); the klein-base GGUF repo rides the same token.
        // The leejet housing has no Q4_K_M - quants run Q2_K..Q8_0
        // including Q4_K (verified), so the fallback names Q4_K.
        token: "flux-3-image-klein",
        display: "FLUX.3 Klein",
        excludes: &[],
        standalone_files: None,
        components: &[
            ComponentSpec {
                flag: "--vae",
                source: ComponentSource {
                    repo: "Comfy-Org/MOSS-Image",
                    repo_path: "vae/flux2-vae.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--llm",
                source: ComponentSource {
                    repo: "Qwen/Qwen3-VL-4B-Instruct-GGUF",
                    repo_path: "Qwen3VL-4B-Instruct-{quant}.gguf",
                },
                required: true,
                fallback_quant: Some("Q4_K_M"),
            },
        ],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // MiniMax-H3 (upstream docs/minimax_h3.md): jointly generates
        // video and stereo audio - the audio VAE is a second required
        // component, a flag no other family uses. The leejet GGUF repo
        // hosts BOTH the DiT and the qwen3vl text encoder; fl2va and
        // ref2va DiT variants (plus _pruned) share this component set,
        // and same-quant picks resolve largest-file-wins, so the
        // pruned DiT needs its exact-filename pull.
        token: "minimax",
        display: "MiniMax-H3",
        excludes: &[],
        standalone_files: None,
        components: &[
            ComponentSpec {
                flag: "--vae",
                source: ComponentSource {
                    repo: "Comfy-Org/MiniMax-H3",
                    repo_path: "vae/minimax_h3_video_vae_fp16.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--audio-vae",
                source: ComponentSource {
                    repo: "Comfy-Org/MiniMax-H3",
                    repo_path: "vae/minimax_h3_audio_vae_fp32.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--llm",
                source: ComponentSource {
                    repo: "leejet/MiniMax-H3-GGUF",
                    repo_path: "qwen3vl_32b_minimax_h3-{quant}.gguf",
                },
                required: true,
                fallback_quant: Some("Q4_K_M"),
            },
        ],
        mode: FamilyMode::Vid,
    },
    DiffusionFamily {
        // MOSS-Image (upstream docs/moss_image.md): 5B DiT + FLUX.2
        // VAE + Qwen3.5-9B text encoder + an explicit tokenizer file -
        // the doc command passes --tokenizer with a locally renamed
        // copy of MOSS's tokenizer.json. The doc DiT filename carries
        // an _attn suffix that does not exist in any HF listing (doc
        // drift); the int8_convrot file is the real one. Base and SFT
        // housings (GGUF leejet, safetensors Comfy) all ride the token.
        // Live-boot receipt: MOSS-Image-Base Q8_0 pulled and rendered
        // through /v1/images/generations on 2026-11-02 (this lane).
        token: "moss-image",
        display: "MOSS-Image",
        excludes: &[],
        standalone_files: Some(&[
            "diffusion_models/moss_image_base_int8_convrot.safetensors",
            "diffusion_models/moss_image_base_bf16.safetensors",
        ]),
        components: &[
            ComponentSpec {
                flag: "--vae",
                source: ComponentSource {
                    repo: "Comfy-Org/MOSS-Image",
                    repo_path: "vae/flux2-vae.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--llm",
                source: ComponentSource {
                    repo: "unsloth/Qwen3.5-9B-GGUF",
                    repo_path: "Qwen3.5-9B-{quant}.gguf",
                },
                required: true,
                fallback_quant: Some("Q4_K_M"),
            },
            ComponentSpec {
                flag: "--tokenizer",
                source: ComponentSource {
                    repo: "OpenMOSS-Team/MOSS-Image-Base",
                    repo_path: "tokenizer/tokenizer.json",
                },
                required: true,
                fallback_quant: None,
            },
        ],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // Krea 2 (upstream docs/krea2.md): DiT + wan 2.1 VAE (same
        // bytes as the wan entry) + Qwen3-VL-4B encoder. The GGUF
        // housing splits quants across BASE/ and TURBO/ subdirs with
        // identical quant names - same-quant picks resolve
        // largest-file-wins deterministically and both variants pair
        // this same component set, so the ambiguity is cosmetic; a
        // deliberate Base-vs-Turbo choice uses the exact-filename
        // pull.
        token: "krea-2",
        display: "Krea 2",
        excludes: &[],
        standalone_files: None,
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
                flag: "--llm",
                source: ComponentSource {
                    repo: "Qwen/Qwen3-VL-4B-Instruct-GGUF",
                    repo_path: "Qwen3VL-4B-Instruct-{quant}.gguf",
                },
                required: true,
                fallback_quant: Some("Q4_K_M"),
            },
        ],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // Iris 3B (upstream docs/iris.md): pixel-space DiT - no VAE
        // component at all - plus the Qwen3-VL-4B encoder. The
        // standalone list names the ROOT model.safetensors only: the
        // depth/upscaler/turbo subdirs ship their own model.safetensors
        // but standalone_file_plan matches the exact root path first.
        token: "iris",
        display: "Iris 3B",
        excludes: &[],
        standalone_files: Some(&["model.safetensors"]),
        components: &[ComponentSpec {
            flag: "--llm",
            source: ComponentSource {
                repo: "Qwen/Qwen3-VL-4B-Instruct-GGUF",
                repo_path: "Qwen3VL-4B-Instruct-{quant}.gguf",
            },
            required: true,
            fallback_quant: Some("Q4_K_M"),
        }],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // Lens (upstream docs/lens.md): DiT + gpt-oss-20b encoder +
        // its tokenizer.json + the FLUX.2 VAE (in-repo ungated copy).
        // The doc command uses UD-Q8_0, absent from the unsloth
        // listing; Q4_K_M is the verified canonical file (exact path,
        // no quant placeholder).
        token: "lens",
        display: "Lens",
        excludes: &[],
        standalone_files: Some(&["diffusion_models/lens_bf16.safetensors"]),
        components: &[
            ComponentSpec {
                flag: "--llm",
                source: ComponentSource {
                    repo: "unsloth/gpt-oss-20b-GGUF",
                    repo_path: "gpt-oss-20b-Q4_K_M.gguf",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--tokenizer",
                source: ComponentSource {
                    repo: "openai/gpt-oss-20b",
                    repo_path: "tokenizer.json",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--vae",
                source: ComponentSource {
                    repo: "Comfy-Org/Lens",
                    repo_path: "vae/flux2-vae.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
        ],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // Fibo 2.0 (upstream docs/fibo2.md): DiT + FLUX.2 VAE +
        // Qwen3-VL-4B encoder (GGUF housing; the Comfy repo also ships
        // the encoder as safetensors). The edit/turbo DiT variants
        // live in the same repo and pair this same set.
        token: "fibo",
        display: "Fibo 2.0",
        excludes: &[],
        standalone_files: Some(&[
            "diffusion_models/fibo2_dev_int8_convrot.safetensors",
            "diffusion_models/fibo2_dev_bf16.safetensors",
        ]),
        components: &[
            ComponentSpec {
                flag: "--vae",
                source: ComponentSource {
                    repo: "Comfy-Org/Fibo-2.0",
                    repo_path: "vae/flux2-vae.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--llm",
                source: ComponentSource {
                    repo: "Qwen/Qwen3-VL-4B-Instruct-GGUF",
                    repo_path: "Qwen3VL-4B-Instruct-{quant}.gguf",
                },
                required: true,
                fallback_quant: Some("Q4_K_M"),
            },
        ],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // Ming-Image 0.1 Design (upstream docs/ming_image.md): DiT +
        // its own VAE + the Ling-mini text encoder (safetensors, not
        // GGUF - the int8_convrot 19.5GB file is the usable size;
        // bf16 is 36.7GB, w4a8 12.8GB) + the mllm tokenizer.
        token: "ming-image",
        display: "Ming-Image 0.1 Design",
        excludes: &[],
        standalone_files: Some(&[
            "diffusion_models/ming_image_0.1_design_int8_convrot.safetensors",
            "diffusion_models/ming_image_0.1_design_bf16.safetensors",
        ]),
        components: &[
            ComponentSpec {
                flag: "--vae",
                source: ComponentSource {
                    repo: "Comfy-Org/Ming-Image",
                    repo_path: "vae/ming_image_vae_bf16.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--llm",
                source: ComponentSource {
                    repo: "Comfy-Org/Ming-Image",
                    repo_path: "text_encoders/ming_image_0.1_ling_mini_2.0_int8_convrot.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--tokenizer",
                source: ComponentSource {
                    repo: "inclusionAI/Ming-Image-0.1-Design",
                    repo_path: "mllm/tokenizer.json",
                },
                required: true,
                fallback_quant: None,
            },
        ],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // HiDream-O1 (upstream docs/hidream_o1_image.md): a fully
        // standalone checkpoint - no components, the DiT embeds
        // everything. dev listed first (the doc command's variant);
        // fp8_scaled is the size-conscious alternative.
        token: "hidream",
        display: "HiDream-O1",
        excludes: &[],
        standalone_files: Some(&[
            "checkpoints/hidream_o1_image_dev_bf16.safetensors",
            "checkpoints/hidream_o1_image_dev_fp8_scaled.safetensors",
            "checkpoints/hidream_o1_image_bf16.safetensors",
        ]),
        components: &[],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // PixArt-Sigma XL-2 1024-MS (upstream docs/pixart.md): 0.6B DiT
        // as a single standalone transformer file plus the SDXL-class
        // VAE from the same repo. The doc's text encoder is the
        // original T5-v1.1-XXL shipped as a 19GB two-shard safetensors
        // pair - pull components are single files, so the entry pairs
        // the byte-equivalent city96 GGUF encoder the FLUX.1 entry
        // already uses (2.5GB at Q4_K_M); the tokenizer is embedded in
        // the T5 vocabulary per the doc, no extra file. Sigma is listed
        // before alpha: both tokens are disjoint, order is defense.
        token: "pixart-sigma",
        display: "PixArt-Sigma",
        excludes: &[],
        standalone_files: Some(&["transformer/diffusion_pytorch_model.safetensors"]),
        components: &[
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
                flag: "--vae",
                source: ComponentSource {
                    repo: "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
                    repo_path: "vae/diffusion_pytorch_model.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
        ],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // PixArt-alpha XL-2 1024-MS (upstream docs/pixart.md): same
        // contract as Sigma with the alpha checkpoint's own VAE (the
        // doc warns the VAE must match the checkpoint's latent space).
        // Token "pixart-xl" cannot hit the Sigma repo ("pixart-sigma-
        // xl..." never contains the literal "pixart-xl").
        token: "pixart-xl",
        display: "PixArt-Alpha",
        excludes: &[],
        standalone_files: Some(&["transformer/diffusion_pytorch_model.safetensors"]),
        components: &[
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
                flag: "--vae",
                source: ComponentSource {
                    repo: "PixArt-alpha/PixArt-XL-2-1024-MS",
                    repo_path: "vae/diffusion_pytorch_model.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
        ],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // Anima (upstream docs/anima.md): DiT + qwen_image_vae + the
        // Qwen3-0.6B-Base encoder. "animatediff" is excluded because
        // AnimateDiff is a different sd.cpp feature that merely shares
        // the token, not this family.
        token: "anima",
        display: "Anima",
        excludes: &["animatediff"],
        standalone_files: None,
        components: &[
            ComponentSpec {
                flag: "--vae",
                source: ComponentSource {
                    repo: "circlestone-labs/Anima",
                    repo_path: "split_files/vae/qwen_image_vae.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--llm",
                source: ComponentSource {
                    repo: "mradermacher/Qwen3-0.6B-Base-GGUF",
                    repo_path: "Qwen3-0.6B-Base.{quant}.gguf",
                },
                required: true,
                fallback_quant: Some("Q4_K_M"),
            },
        ],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // Boogu Image (upstream docs/boogu_image.md): DiT + FLUX.1
        // VAE + Qwen3-VL-8B encoder. The fp8_scaled standalone comes
        // first (half the bf16 bytes). Edit variants need the vision
        // sidecar. The VAE repo is gated; the pull teaches on a 401
        // until access is granted (same posture as the flux.1 entry).
        token: "boogu",
        display: "Boogu Image",
        excludes: &[],
        standalone_files: Some(&[
            "diffusion_models/boogu_image_base_fp8_scaled.safetensors",
            "diffusion_models/boogu_image_base_bf16.safetensors",
        ]),
        components: &[
            ComponentSpec {
                flag: "--vae",
                source: ComponentSource {
                    repo: "black-forest-labs/FLUX.1-schnell",
                    repo_path: "ae.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--llm",
                source: ComponentSource {
                    repo: "unsloth/Qwen3-VL-8B-Instruct-GGUF",
                    repo_path: "Qwen3-VL-8B-Instruct-{quant}.gguf",
                },
                required: true,
                fallback_quant: Some("Q4_K_M"),
            },
            ComponentSpec {
                flag: "--llm_vision",
                source: ComponentSource {
                    repo: "unsloth/Qwen3-VL-8B-Instruct-GGUF",
                    repo_path: "mmproj-F16.gguf",
                },
                required: false,
                fallback_quant: None,
            },
        ],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // ERNIE-Image (upstream docs/ernie_image.md): DiT + FLUX.2
        // VAE (ungated in-repo copy) + Ministral-3 text encoder. The
        // turbo repos ride the same token and component set.
        token: "ernie",
        display: "ERNIE-Image",
        excludes: &[],
        standalone_files: None,
        components: &[
            ComponentSpec {
                flag: "--vae",
                source: ComponentSource {
                    repo: "Comfy-Org/ERNIE-Image",
                    repo_path: "vae/flux2-vae.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--llm",
                source: ComponentSource {
                    repo: "unsloth/Ministral-3-3B-Instruct-2512-GGUF",
                    repo_path: "Ministral-3-3B-Instruct-2512-{quant}.gguf",
                },
                required: true,
                fallback_quant: Some("Q4_K_M"),
            },
        ],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // LLaDA-Image (upstream docs/llada_image.md): diffusion-LM
        // DiT + its own text_encoder GGUF + embeddings-connectors +
        // tokenizer + VAE. The GGUF housing offers only f16 and q8_0,
        // so the fallback lands on Q8_0. Component files are sourced
        // from the BASE repo: the turbo repos carry different file
        // names but byte-identical sizes (verified), and one housing
        // keeps the pairing repairable. Quant tokens in this repo are
        // lowercase (q4_K), so the fallback pins q4_K.
        token: "llada-image",
        display: "LLaDA-Image",
        excludes: &[],
        standalone_files: None,
        components: &[
            ComponentSpec {
                flag: "--llm",
                source: ComponentSource {
                    repo: "fszontagh/LLaDA-Image-GGUF",
                    repo_path: "llada-image-text_encoder-{quant}.gguf",
                },
                required: true,
                fallback_quant: Some("q4_K"),
            },
            ComponentSpec {
                flag: "--embeddings-connectors",
                source: ComponentSource {
                    repo: "fszontagh/LLaDA-Image-GGUF",
                    repo_path: "llada-image-connectors.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            // The edit-surface connectors (connectors-edit.safetensors,
            // same repo) are deliberately not attached: sd.cpp takes one
            // --embeddings-connectors file, so the edit variant is a
            // deliberate file swap, not an automatic sidecar.
            ComponentSpec {
                flag: "--tokenizer",
                source: ComponentSource {
                    repo: "inclusionAI/LLaDA-Image",
                    repo_path: "tokenizer/tokenizer.json",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--vae",
                source: ComponentSource {
                    repo: "inclusionAI/LLaDA-Image",
                    repo_path: "vae/diffusion_pytorch_model.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
        ],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // LongCat-Image (upstream docs/longcat_image.md): DiT +
        // FLUX.1 VAE + the Qwen2.5-VL-7B encoder (same bytes as the
        // qwen-image family). Edit repos ride the token.
        token: "longcat",
        display: "LongCat",
        excludes: &[],
        standalone_files: None,
        components: &[
            ComponentSpec {
                flag: "--vae",
                source: ComponentSource {
                    repo: "black-forest-labs/FLUX.1-schnell",
                    repo_path: "ae.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--llm",
                source: ComponentSource {
                    repo: "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF",
                    repo_path: "Qwen2.5-VL-7B-Instruct.{quant}.gguf",
                },
                required: true,
                fallback_quant: Some("Q4_K_M"),
            },
            ComponentSpec {
                flag: "--llm_vision",
                source: ComponentSource {
                    repo: "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF",
                    repo_path: "Qwen2.5-VL-7B-Instruct.mmproj-Q8_0.gguf",
                },
                required: false,
                fallback_quant: None,
            },
        ],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // LTX-2.3 (upstream docs/ltx2.md): 22B dev DiT + video VAE +
        // audio VAE + embeddings-connectors + gemma-3-12b encoder.
        // The repo also carries a smaller distilled-1.1/ subdir whose
        // quants collide by name - same-quant picks resolve
        // largest-file-wins (the dev model), matching the doc command.
        token: "ltx-2.3",
        display: "LTX-2.3",
        excludes: &[],
        standalone_files: None,
        components: &[
            ComponentSpec {
                flag: "--vae",
                source: ComponentSource {
                    repo: "unsloth/LTX-2.3-GGUF",
                    repo_path: "vae/ltx-2.3-22b-dev_video_vae.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--audio-vae",
                source: ComponentSource {
                    repo: "unsloth/LTX-2.3-GGUF",
                    repo_path: "vae/ltx-2.3-22b-dev_audio_vae.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--embeddings-connectors",
                source: ComponentSource {
                    repo: "unsloth/LTX-2.3-GGUF",
                    repo_path: "text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--llm",
                source: ComponentSource {
                    repo: "unsloth/gemma-3-12b-it-GGUF",
                    repo_path: "gemma-3-12b-it-{quant}.gguf",
                },
                required: true,
                fallback_quant: Some("Q4_K_M"),
            },
        ],
        mode: FamilyMode::Vid,
    },
    DiffusionFamily {
        // LTX-2.5 (upstream docs/ltx2.md): same transformer/VAE
        // architecture family as 2.3 but pairs a Gemma 4 encoder with
        // the projection embedded (single safetensors, no separate
        // connectors). The doc mandates the CONV video-VAE variant.
        // Component repo is gated; the pull teaches on a 401 until
        // access is granted.
        token: "ltx-2.5",
        display: "LTX-2.5",
        excludes: &[],
        standalone_files: None,
        components: &[
            ComponentSpec {
                flag: "--vae",
                source: ComponentSource {
                    repo: "Lightricks/LTX-2.5",
                    repo_path: "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--audio-vae",
                source: ComponentSource {
                    repo: "Lightricks/LTX-2.5",
                    repo_path: "vae/ltx-2.5-audio-vae-bf16.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--llm",
                source: ComponentSource {
                    repo: "Lightricks/LTX-2.5",
                    repo_path: "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
        ],
        mode: FamilyMode::Vid,
    },
    DiffusionFamily {
        // Mage-Flow (upstream docs/mage_flow.md): DiT + own VAE +
        // Qwen3-VL-4B encoder (same bytes as z-image). The doc command
        // renames the VAE locally; the repo listing name is the
        // listing-verified one used here. Edit variants need the
        // vision sidecar.
        token: "mage-flow",
        display: "Mage-Flow",
        excludes: &[],
        standalone_files: Some(&[
            "diffusion_models/mage_flow_int8_convrot.safetensors",
            "diffusion_models/mage_flow_bf16.safetensors",
        ]),
        components: &[
            ComponentSpec {
                flag: "--vae",
                source: ComponentSource {
                    repo: "Comfy-Org/Mage-Flow",
                    repo_path: "vae/mage_flow_vae_bf16.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--llm",
                source: ComponentSource {
                    repo: "Qwen/Qwen3-VL-4B-Instruct-GGUF",
                    repo_path: "Qwen3VL-4B-Instruct-{quant}.gguf",
                },
                required: true,
                fallback_quant: Some("Q4_K_M"),
            },
            ComponentSpec {
                flag: "--llm_vision",
                source: ComponentSource {
                    repo: "Qwen/Qwen3-VL-4B-Instruct-GGUF",
                    repo_path: "mmproj-F16.gguf",
                },
                required: false,
                fallback_quant: None,
            },
        ],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // MiniT2I (upstream docs/minit2i.md): DiT + flan-t5-large
        // riding the --t5xxl flag (doc-exact command). No VAE and no
        // tokenizer file in the pairing. The smallest curated family:
        // the live boot receipt for this wave pulled both files.
        token: "minit2i",
        display: "MiniT2I",
        excludes: &[],
        standalone_files: Some(&["minit2i-b-16/transformer/diffusion_pytorch_model.safetensors"]),
        components: &[ComponentSpec {
            flag: "--t5xxl",
            source: ComponentSource {
                repo: "google/flan-t5-large",
                repo_path: "model.safetensors",
            },
            required: true,
            fallback_quant: None,
        }],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // Ovis-Image (upstream docs/ovis_image.md): 7B DiT + FLUX.1
        // VAE + the ovis_2.5 safetensors text encoder. The GGUF
        // housing offers only Q4_0 and Q8_0, so the fallback pins
        // Q4_0.
        token: "ovis",
        display: "Ovis",
        excludes: &[],
        standalone_files: None,
        components: &[
            ComponentSpec {
                flag: "--vae",
                source: ComponentSource {
                    repo: "black-forest-labs/FLUX.1-schnell",
                    repo_path: "ae.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--llm",
                source: ComponentSource {
                    repo: "Comfy-Org/Ovis-Image",
                    repo_path: "split_files/text_encoders/ovis_2.5.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
        ],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // Z-Image L2P (upstream docs/z_image_l2p.md): pixel-space
        // long-context variant - no VAE component (local decoder),
        // Qwen3-4B encoder (same bytes as z-image).
        token: "l2p",
        display: "Z-Image L2P",
        excludes: &[],
        standalone_files: Some(&["model-1k-merge.safetensors"]),
        components: &[ComponentSpec {
            flag: "--llm",
            source: ComponentSource {
                repo: "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                repo_path: "Qwen3-4B-Instruct-2507-{quant}.gguf",
            },
            required: true,
            fallback_quant: Some("Q4_K_M"),
        }],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // HunyuanVideo 1.5 (upstream docs/hunyuan_video.md): DiT +
        // causal video VAE + Qwen2.5-VL-7B fp8 main conditioning +
        // ByT5 Small GlyphXL glyph conditioning (both ride their doc
        // flags). The repackaged repo id 301-redirects to the
        // canonical Comfy-Org/Hunyuan-Video-1.5 used here. Token
        // carries the hyphenated spelling so Hunyuan text-LLM repos
        // (no "-video") never false-match.
        token: "hunyuan-video",
        display: "HunyuanVideo 1.5",
        excludes: &[],
        standalone_files: Some(&[
            "split_files/diffusion_models/hunyuanvideo1.5_720p_t2v_fp16.safetensors",
        ]),
        components: &[
            ComponentSpec {
                flag: "--vae",
                source: ComponentSource {
                    repo: "Comfy-Org/Hunyuan-Video-1.5",
                    repo_path: "split_files/vae/hunyuanvideo15_vae_fp16.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--llm",
                source: ComponentSource {
                    repo: "Comfy-Org/Hunyuan-Video-1.5",
                    repo_path: "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
            ComponentSpec {
                flag: "--t5xxl",
                source: ComponentSource {
                    repo: "Comfy-Org/Hunyuan-Video-1.5",
                    repo_path: "split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors",
                },
                required: true,
                fallback_quant: None,
            },
        ],
        mode: FamilyMode::Vid,
    },
    DiffusionFamily {
        // Lingbot Video (upstream docs/lingbot_video.md): dense 1.3B
        // DiT + wan 2.1 VAE (byte-reuse) + Qwen3-VL-4B encoder. The
        // moe-30b housing rides the same token and component set. The
        // smallest curated video family: the live boot receipt for
        // this wave rendered through /v1/videos/generations.
        token: "lingbot",
        display: "Lingbot",
        excludes: &[],
        standalone_files: Some(&["transformer/diffusion_pytorch_model.safetensors"]),
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
                flag: "--llm",
                source: ComponentSource {
                    repo: "Qwen/Qwen3-VL-4B-Instruct-GGUF",
                    repo_path: "Qwen3VL-4B-Instruct-{quant}.gguf",
                },
                required: true,
                fallback_quant: Some("Q4_K_M"),
            },
        ],
        mode: FamilyMode::Vid,
    },
    DiffusionFamily {
        // SSD-1B (upstream docs/distilled_sd.md): distilled SDXL
        // subset shipped as a full A1111 checkpoint - no component
        // pairing, the single file is the whole model.
        token: "ssd-1b",
        display: "SSD-1B",
        excludes: &[],
        standalone_files: Some(&["SSD-1B-A1111.safetensors"]),
        components: &[],
        mode: FamilyMode::Img,
    },
    DiffusionFamily {
        // Segmind Vega (upstream docs/distilled_sd.md): same A1111
        // full-checkpoint contract as SSD-1B.
        token: "vega",
        display: "Segmind Vega",
        excludes: &[],
        standalone_files: Some(&["segmind-vega.safetensors"]),
        components: &[],
        mode: FamilyMode::Img,
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

/// How a repo houses a standalone-capable family. Families whose
/// canonical DiT is a single safetensors checkpoint (SD 1.5, SDXL,
/// MOSS-Image, SD 3.5...) also ship DiT GGUF mirrors under the same
/// family token - those must ride the gguf lane, where
/// `attach_diffusion_set` completes the identical component set, not
/// the standalone lane (which would reject them for hosting none of
/// the pinned checkpoint files).
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum StandaloneHousing {
    /// Repo hosts one of the family's pinned standalone files: the
    /// standalone pull lane owns it.
    CheckpointHosted,
    /// Repo carries DiT GGUF files instead: fall through to the gguf
    /// lane (same component set attaches).
    GgufMirror,
    /// Neither: a diffusers-shard or transformer-shard housing the
    /// safetensors lane would mis-register - teach instead.
    DiffusersShards,
}

/// Classify a repo listing against a standalone-capable family.
#[must_use]
pub fn standalone_housing(info: &HfModelInfo, family: &DiffusionFamily) -> StandaloneHousing {
    if standalone_file_plan(info, family).is_some() {
        return StandaloneHousing::CheckpointHosted;
    }
    if info
        .siblings
        .iter()
        .any(|s| s.rfilename.to_lowercase().ends_with(".gguf"))
    {
        return StandaloneHousing::GgufMirror;
    }
    StandaloneHousing::DiffusersShards
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
/// never trigger repair - they only disable edits.
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
            last_used_at: 1,
        }
    }

    #[test]
    fn unit__diffusion_family__token_matches_owner_and_case_variants() {
        let f = diffusion_family("abenzerps/Qwen-Image-2.1-GGUF").unwrap();
        assert_eq!(f.display, "Qwen-Image-2.1");
        assert!(diffusion_family("leejet/qwen-image-2.1-gguf").is_some());
        assert!(diffusion_family("Comfy-Org/Qwen-Image-2.1").is_some());
        // v1 and unrelated repos stay unmatched by 2.1 - their `VAE`
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
                "Qwen-Image-Edit",
                "Qwen-Image-2.1",
                "Qwen-Image",
                "FLUX.1",
                "Z-Image",
                "Chroma",
                "FLUX.2-dev",
                "SDXL",
                "SD 1.5",
                "Wan 2.2 TI2V 5B",
                "Wan 2.2 T2V A14B",
                "Wan 2.1 T2V",
                "SD 3.5",
                "FLUX.3 Klein",
                "MiniMax-H3",
                "MOSS-Image",
                "Krea 2",
                "Iris 3B",
                "Lens",
                "Fibo 2.0",
                "Ming-Image 0.1 Design",
                "HiDream-O1",
                "PixArt-Sigma",
                "PixArt-Alpha",
                "Anima",
                "Boogu Image",
                "ERNIE-Image",
                "LLaDA-Image",
                "LongCat",
                "LTX-2.3",
                "LTX-2.5",
                "Mage-Flow",
                "MiniT2I",
                "Ovis",
                "Z-Image L2P",
                "HunyuanVideo 1.5",
                "Lingbot",
                "SSD-1B",
                "Segmind Vega"
            ]
        );
        // Wan 2.1 T2V: the repackaged tree and GGUF-style housings
        // match; sibling variants with different component contracts
        // are vetoed. Wan 2.2 T2V now carries its own pair entry.
        assert!(diffusion_family("Comfy-Org/Wan_2.1_ComfyUI_repackaged").is_some());
        assert!(diffusion_family("Wan-AI/Wan2.1-T2V-1.3B").is_some());
        assert!(diffusion_family("Comfy-Org/Wan2.1-Fun-1.3B-InP").is_none());
        assert_eq!(
            diffusion_family("QuantStack/Wan2.2-T2V-A14B-GGUF")
                .unwrap()
                .display,
            "Wan 2.2 T2V A14B"
        );
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
        // Edit repos ride the dedicated Edit entry (listed before v1);
        // the 2509 mirror carries the same component contract.
        assert_eq!(
            diffusion_family("Qwen/Qwen-Image-Edit-2509")
                .unwrap()
                .display,
            "Qwen-Image-Edit"
        );
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
        // SD3.5 now carries its own entry (clip_l+clip_g+t5xxl).
        assert_eq!(
            diffusion_family("stabilityai/stable-diffusion-3.5-medium")
                .unwrap()
                .display,
            "SD 3.5"
        );
    }

    #[test]
    fn unit__diffusion_family__g3_curation_token_matrix() {
        // Every G3-curated family resolves from its real repo spelling
        // (case-insensitive) with the curated display name.
        for (repo, display) in [
            ("QuantStack/Qwen-Image-Edit-GGUF", "Qwen-Image-Edit"),
            ("stabilityai/stable-diffusion-3.5-large", "SD 3.5"),
            ("leejet/flux-3-image-klein-GGUF", "FLUX.3 Klein"),
            ("leejet/flux-3-image-klein-base-GGUF", "FLUX.3 Klein"),
            ("leejet/MiniMax-H3-GGUF", "MiniMax-H3"),
            ("leejet/MOSS-Image-Base-GGUF", "MOSS-Image"),
            ("Comfy-Org/MOSS-Image", "MOSS-Image"),
            ("leejet/MOSS-Image-SFT-GGUF", "MOSS-Image"),
            ("realrebelai/KREA-2_GGUFs", "Krea 2"),
            ("speridlabs/iris-3b", "Iris 3B"),
            ("Comfy-Org/Lens", "Lens"),
            ("Comfy-Org/Fibo-2.0", "Fibo 2.0"),
            ("Comfy-Org/Ming-Image", "Ming-Image 0.1 Design"),
            ("Comfy-Org/HiDream-O1-Image", "HiDream-O1"),
            ("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", "PixArt-Sigma"),
            ("PixArt-alpha/PixArt-XL-2-1024-MS", "PixArt-Alpha"),
            // G3c families: every real repo spelling resolves.
            ("Bedovyy/Anima-GGUF", "Anima"),
            ("JusteLeo/Anima2-GGUF", "Anima"),
            ("Comfy-Org/Boogu-Image", "Boogu Image"),
            ("unsloth/ERNIE-Image-GGUF", "ERNIE-Image"),
            ("unsloth/ERNIE-Image-Turbo-GGUF", "ERNIE-Image"),
            ("fszontagh/LLaDA-Image-GGUF", "LLaDA-Image"),
            ("fszontagh/LLaDA-Image-Turbo-GGUF", "LLaDA-Image"),
            ("vantagewithai/LongCat-Image-GGUF", "LongCat"),
            ("unsloth/LTX-2.3-GGUF", "LTX-2.3"),
            ("vantagewithai/LTX-2.5-GGUF", "LTX-2.5"),
            ("Comfy-Org/Mage-Flow", "Mage-Flow"),
            ("MiniT2I/MiniT2I", "MiniT2I"),
            ("leejet/Ovis-Image-7B-GGUF", "Ovis"),
            ("zhen-nan/L2P", "Z-Image L2P"),
            ("Comfy-Org/Hunyuan-Video-1.5", "HunyuanVideo 1.5"),
            ("robbyant/lingbot-video-dense-1.3b", "Lingbot"),
            ("segmind/SSD-1B", "SSD-1B"),
            ("segmind/Segmind-Vega", "Segmind Vega"),
        ] {
            assert_eq!(
                diffusion_family(repo).map(|f| f.display),
                Some(display),
                "{repo} must resolve to {display}"
            );
        }
        // Deferred or uncurated housings stay honestly unmatched:
        // Ideogram4 requires a local fp8->bf16 conversion before
        // serving, and the underscore safetensors housings of Wan 2.2
        // (ComfyUI repackaged style) are not curated GGUF mirrors.
        assert!(diffusion_family("ideogram/Ideogram4").is_none());
        assert!(diffusion_family("Comfy-Org/Wan2.2_ComfyUI_repackaged").is_none());
        assert!(diffusion_family("Wan-AI/Wan_2.2_T2V_A14B_GGUF").is_none());
        // G3c deferrals: PiD ships .pth backbone-dependent sets with a
        // value-taking --vae-format flag, SeFi is fully gated, and the
        // SenseNova shard-dir contract cannot ride the file-based
        // component pull. The anima exclusion keeps AnimateDiff (a
        // different sd.cpp feature sharing the token) out.
        assert!(diffusion_family("nvidia/PiD").is_none());
        assert!(diffusion_family("SeFi-Image/SeFi-Image-base").is_none());
        assert!(diffusion_family("someuser/animatediff-motion-module").is_none());
    }

    #[test]
    fn unit__diffusion_family__wan22_a14b_pair_pins_the_low_noise_expert() {
        // The QuantStack mirror lists LowNoise//HighNoise/ siblings at
        // identical per-quant sizes: a size-based gguf-lane pick could
        // not choose a role between them. The family pins the primary
        // through standalone_files, so the plan must return the exact
        // low-noise file regardless of sibling order or size ties.
        let family = diffusion_family("QuantStack/Wan2.2-T2V-A14B-GGUF").unwrap();
        let gguf = |name: &str| HfSibling {
            rfilename: name.to_string(),
            size: Some(5_300_000_000),
            lfs: Some(HfLfs {
                size: Some(5_300_000_000),
                sha256: "x".to_string(),
            }),
        };
        let info = HfModelInfo {
            id: "QuantStack/Wan2.2-T2V-A14B-GGUF".into(),
            siblings: vec![
                gguf("HighNoise/Wan2.2-T2V-A14B-HighNoise-Q2_K.gguf"),
                gguf("LowNoise/Wan2.2-T2V-A14B-LowNoise-Q2_K.gguf"),
                gguf("HighNoise/Wan2.2-T2V-A14B-HighNoise-Q8_0.gguf"),
                gguf("LowNoise/Wan2.2-T2V-A14B-LowNoise-Q8_0.gguf"),
            ],
            gguf: None,
        };
        let plan = standalone_file_plan(&info, family).unwrap();
        assert_eq!(plan.filename, "LowNoise/Wan2.2-T2V-A14B-LowNoise-Q2_K.gguf");
        // The high-noise expert rides as an exact-path component, so
        // the pair stays quant-coherent without sibling guessing.
        let high = family
            .components
            .iter()
            .find(|c| c.flag == "--high-noise-diffusion-model")
            .expect("a14b entry carries the high-noise expert");
        let high_plan = component_plan(&info, &high.source, "Q4_K_M").unwrap();
        assert_eq!(
            high_plan.filename,
            "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q2_K.gguf"
        );
        // TI2V 5B: single-DiT repo with a unique file per quant - the
        // gguf lane serves the DiT, the family only pins the VAE.
        let ti2v = diffusion_family("QuantStack/Wan2.2-TI2V-5B-GGUF").unwrap();
        assert!(ti2v.standalone_files.is_none());
        let vae = ti2v
            .components
            .iter()
            .find(|c| c.flag == "--vae")
            .expect("ti2v entry carries the Wan 2.2 VAE");
        let vae_info = HfModelInfo {
            id: "QuantStack/Wan2.2-TI2V-5B-GGUF".into(),
            siblings: vec![HfSibling {
                rfilename: "VAE/Wan2.2_VAE.safetensors".into(),
                size: Some(1_410_000_000),
                lfs: Some(HfLfs {
                    size: Some(1_410_000_000),
                    sha256: "y".to_string(),
                }),
            }],
            gguf: None,
        };
        assert_eq!(
            component_plan(&vae_info, &vae.source, "Q4_K_M")
                .unwrap()
                .filename,
            "VAE/Wan2.2_VAE.safetensors"
        );
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
        // Wan (both generations) and MiniMax-H3 are the video lanes
        // (H3 jointly generates video + stereo audio); every curated
        // image family reports Img.
        assert!(matches!(
            family_mode("Comfy-Org/Wan_2.1_ComfyUI_repackaged"),
            Some(&FamilyMode::Vid)
        ));
        assert!(matches!(
            family_mode("QuantStack/Wan2.2-TI2V-5B-GGUF"),
            Some(&FamilyMode::Vid)
        ));
        assert!(matches!(
            family_mode("QuantStack/Wan2.2-T2V-A14B-GGUF"),
            Some(&FamilyMode::Vid)
        ));
        assert!(matches!(
            family_mode("leejet/MiniMax-H3-GGUF"),
            Some(&FamilyMode::Vid)
        ));
        // G3c video families: both LTX generations (audio+video joint)
        // and the two smaller video stacks.
        assert!(matches!(
            family_mode("unsloth/LTX-2.3-GGUF"),
            Some(&FamilyMode::Vid)
        ));
        assert!(matches!(
            family_mode("vantagewithai/LTX-2.5-GGUF"),
            Some(&FamilyMode::Vid)
        ));
        assert!(matches!(
            family_mode("Comfy-Org/Hunyuan-Video-1.5"),
            Some(&FamilyMode::Vid)
        ));
        assert!(matches!(
            family_mode("robbyant/lingbot-video-dense-1.3b"),
            Some(&FamilyMode::Vid)
        ));
        for repo in [
            "abenzerps/Qwen-Image-2.1-GGUF",
            "city96/FLUX.1-dev-gguf",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "leejet/MOSS-Image-Base-GGUF",
            "Comfy-Org/HiDream-O1-Image",
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            "Bedovyy/Anima-GGUF",
            "Comfy-Org/Mage-Flow",
            "MiniT2I/MiniT2I",
            "segmind/SSD-1B",
            "segmind/Segmind-Vega",
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
        // The Edit DiT's own entry pairs --llm_vision: the edit
        // surface (before/after images through the encoder) is live.
        assert_eq!(
            family_supports_edits("QuantStack/Qwen-Image-Edit-GGUF"),
            Some(true)
        );
    }

    #[test]
    fn unit__component_plan__g3_flags_cover_tokenizer_audio_vae_and_exact_files() {
        // Moss --tokenizer: exact path, no quant placeholder.
        let moss = diffusion_family("leejet/MOSS-Image-Base-GGUF").unwrap();
        let tok = moss
            .components
            .iter()
            .find(|c| c.flag == "--tokenizer")
            .expect("moss entry carries --tokenizer");
        let info = HfModelInfo {
            id: "OpenMOSS-Team/MOSS-Image-Base".into(),
            siblings: vec![HfSibling {
                rfilename: "tokenizer/tokenizer.json".into(),
                size: Some(20_000_000),
                lfs: None,
            }],
            gguf: None,
        };
        let plan = component_plan(&info, &tok.source, "Q4_K_M").unwrap();
        assert_eq!(plan.filename, "tokenizer/tokenizer.json");
        // MiniMax --audio-vae: the family-unique second VAE flag.
        let h3 = diffusion_family("leejet/MiniMax-H3-GGUF").unwrap();
        assert!(
            h3.components
                .iter()
                .any(|c| c.flag == "--audio-vae" && c.required),
            "minimax entry must carry a required --audio-vae"
        );
        // Lens --llm: an exact GGUF file with no quant placeholder -
        // any quant string must still resolve the fixed name.
        let lens = diffusion_family("Comfy-Org/Lens").unwrap();
        let llm = lens
            .components
            .iter()
            .find(|c| c.flag == "--llm")
            .expect("lens entry carries --llm");
        let info2 = HfModelInfo {
            id: "unsloth/gpt-oss-20b-GGUF".into(),
            siblings: vec![HfSibling {
                rfilename: "gpt-oss-20b-Q4_K_M.gguf".into(),
                size: Some(11_620_000_000),
                lfs: None,
            }],
            gguf: None,
        };
        let plan = component_plan(&info2, &llm.source, "Q8_0").unwrap();
        assert_eq!(plan.filename, "gpt-oss-20b-Q4_K_M.gguf");
    }

    #[test]
    fn unit__component_plan__g3c_flags_cover_connectors_lowercase_quants_and_exact_t5() {
        // LLaDA --embeddings-connectors: the new doc-exact flag rides
        // as a fixed file, and the text encoder carries this repo's
        // lowercase quant tokens (q4_K) with the fallback pinned to the
        // same spelling.
        let llada = diffusion_family("fszontagh/LLaDA-Image-GGUF").unwrap();
        let connectors = llada
            .components
            .iter()
            .find(|c| c.flag == "--embeddings-connectors")
            .expect("llada entry carries --embeddings-connectors");
        assert!(connectors.required);
        let info = HfModelInfo {
            id: "fszontagh/LLaDA-Image-GGUF".into(),
            siblings: vec![
                HfSibling {
                    rfilename: "llada-image-connectors.safetensors".into(),
                    size: Some(754_200_000),
                    lfs: None,
                },
                HfSibling {
                    rfilename: "llada-image-text_encoder-q4_K.gguf".into(),
                    size: Some(9_692_600_000),
                    lfs: None,
                },
            ],
            gguf: None,
        };
        let plan = component_plan(&info, &connectors.source, "Q4_K_M").unwrap();
        assert_eq!(plan.filename, "llada-image-connectors.safetensors");
        let encoder = llada
            .components
            .iter()
            .find(|c| c.flag == "--llm")
            .expect("llada entry carries --llm");
        assert_eq!(encoder.fallback_quant.as_deref(), Some("q4_K"));
        let enc_plan = component_plan(&info, &encoder.source, "q4_K").unwrap();
        assert_eq!(enc_plan.filename, "llada-image-text_encoder-q4_K.gguf");
        // LTX-2.3 rides the same connectors flag with the gemma
        // encoder, plus its audio VAE.
        let ltx = diffusion_family("unsloth/LTX-2.3-GGUF").unwrap();
        assert!(
            ltx.components
                .iter()
                .any(|c| c.flag == "--embeddings-connectors"
                    && c.source.repo_path.contains("embeddings_connectors"))
        );
        assert!(ltx.components.iter().any(|c| c.flag == "--audio-vae"));
        // MiniT2I --t5xxl: flan-t5-large as a fixed fp32 file - the
        // doc command reuses the t5xxl flag for this encoder.
        let minit2i = diffusion_family("MiniT2I/MiniT2I").unwrap();
        let t5 = minit2i
            .components
            .iter()
            .find(|c| c.flag == "--t5xxl")
            .expect("minit2i entry carries --t5xxl");
        assert_eq!(t5.source.repo, "google/flan-t5-large");
        let t5_info = HfModelInfo {
            id: "google/flan-t5-large".into(),
            siblings: vec![HfSibling {
                rfilename: "model.safetensors".into(),
                size: Some(3_132_700_000),
                lfs: None,
            }],
            gguf: None,
        };
        let t5_plan = component_plan(&t5_info, &t5.source, "Q4_K_M").unwrap();
        assert_eq!(t5_plan.filename, "model.safetensors");
    }

    #[test]
    fn unit__standalone_file_plan__iris_root_model_wins_over_subdir_variants() {
        // The iris repo nests depth//upscaler//turbo/ subdirs that each
        // carry their own model.safetensors; the curated root plan must
        // pick the exact root file, never a subdir sibling.
        let iris = diffusion_family("speridlabs/iris-3b").unwrap();
        let info = HfModelInfo {
            id: "speridlabs/iris-3b".into(),
            siblings: vec![
                HfSibling {
                    rfilename: "model.safetensors".into(),
                    size: Some(11_950_000_000),
                    lfs: None,
                },
                HfSibling {
                    rfilename: "turbo/model.safetensors".into(),
                    size: Some(5_980_000_000),
                    lfs: None,
                },
                HfSibling {
                    rfilename: "upscaler/model.safetensors".into(),
                    size: Some(2_000_000_000),
                    lfs: None,
                },
            ],
            gguf: None,
        };
        let plan = standalone_file_plan(&info, iris).unwrap();
        assert_eq!(plan.filename, "model.safetensors");
        assert_eq!(plan.bytes, 11_950_000_000);
    }

    #[test]
    fn unit__standalone_housing__gguf_mirrors_fall_through_to_the_gguf_lane() {
        // A DiT GGUF mirror of a standalone-capable family (MOSS-Image,
        // SD 3.5) must ride the gguf lane + attach_diffusion_set, not be
        // rejected by the standalone lane for hosting none of the pinned
        // checkpoint files. Live receipt: leejet/MOSS-Image-Base-GGUF:Q4_K
        // pull failed with the standalone teaching before this split.
        let moss = diffusion_family("leejet/MOSS-Image-Base-GGUF").unwrap();
        let gguf_info = HfModelInfo {
            id: "leejet/MOSS-Image-Base-GGUF".into(),
            siblings: vec![HfSibling {
                rfilename: "MOSS-Image-Base-Q4_K.gguf".into(),
                size: Some(3_330_000_000),
                lfs: None,
            }],
            gguf: None,
        };
        assert_eq!(
            standalone_housing(&gguf_info, moss),
            StandaloneHousing::GgufMirror
        );
        // The Comfy housing of the same family hosts the pinned file:
        // standalone lane owns it.
        let comfy_info = HfModelInfo {
            id: "Comfy-Org/MOSS-Image".into(),
            siblings: vec![HfSibling {
                rfilename: "diffusion_models/moss_image_base_int8_convrot.safetensors".into(),
                size: Some(5_410_000_000),
                lfs: None,
            }],
            gguf: None,
        };
        assert_eq!(
            standalone_housing(&comfy_info, moss),
            StandaloneHousing::CheckpointHosted
        );
        // Diffusers shards (unet/, no root checkpoint, no gguf): teach.
        let shards_info = HfModelInfo {
            id: "stabilityai/stable-diffusion-3.5-diffusers".into(),
            siblings: vec![HfSibling {
                rfilename: "transformer/diffusion_pytorch_model-00001-of-00002.safetensors".into(),
                size: Some(5_000_000_000),
                lfs: None,
            }],
            gguf: None,
        };
        let sd3 = diffusion_family("stabilityai/stable-diffusion-3.5-large").unwrap();
        assert_eq!(
            standalone_housing(&shards_info, sd3),
            StandaloneHousing::DiffusersShards
        );
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
