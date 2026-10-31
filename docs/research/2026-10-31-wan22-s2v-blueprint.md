# Wan 2.2 S2V lane — implementation blueprint (evidence-gathered, not yet built)

**Why blueprint, not code:** the only audio-conditioned video model upstream is Wan2.2-S2V-14B
(recipe `wan22-s2v`: features `init_image` + `ref_audios`). Its DiT is a **4-shard
~32.6 GB safetensors set** — no variant fits an 8 GB dev card (TI2V-5B verified: features
`flow_shift`+`init_image` only, NO `ref_audios`, no audio slot). A curated family that cannot be
booted even once on the development box would ship unvalidated surface — this repo's
anti-pattern list forbids it. Everything below is verified evidence so a capable-hardware
session can execute in one wave.

## Upstream contract (verified from master)

`examples/server/recipes/wan22-s2v.json`: slots `diffusion_model` (required), `t5xxl`
(required, UMT5-XXL), `audio_encoder` (required, "Wav2Vec2 Large English"), `vae`|`tae`
(required-any; VAE label = "Wan 2.1 VAE"), `size_multiple: 16`, fixed fps 16, frame_step 4,
defaults euler/flow_shift 3/33f/cfg 4.5/20 steps; `disabled_features`: img2img, mask_image.

## File inventory (HF API, 2026-10-31)

| Slot | Official source | Size | Notes |
|---|---|---|---|
| diffusion_model | `Wan-AI/Wan2.2-S2V-14B` `diffusion_pytorch_model-0000{1..4}-of-00004.safetensors` + `.index.json` | 9.97+9.89+9.96+2.77 = 32.59 GB | **shard SET + index** |
| t5xxl | same repo `models_t5_umt5-xxl-enc-bf16.pth` 11.36 GB — prefer existing curated `city96/umt5-xxl-encoder-gguf` (`umt5-xxl-encoder-{quant}.gguf`, Q4_K_M fallback) | ~3 GB Q4 | reuse wan2.1 component |
| audio_encoder | same repo `wav2vec2-large-xlsr-53-english/` (directory; weights under `google/` — pytorch_model.bin class). Safer source: the safetensors release of `wav2vec2-large-xlsr-53-english` (verify sd.cpp loader's accepted formats before choosing) | ~1.2 GB | config knob `sdcpp_audio_encoder` already spawns `--audio-encoder` |
| vae | prefer `Comfy-Org/Wan_2.1_ComfyUI_repackaged` `split_files/vae/wan_2.1_vae.safetensors` (same as wan2.1 row) | ~500 MB | official repo ships `.pth` |

## Gaps to build (in order)

1. **Shard-set standalone pulls** — `standalone_file_plan` (crates/blazar-runtime/src/diffusion.rs)
   picks ONE file; the S2V DiT needs set-of-shards + index semantics (match
   `diffusion_pytorch_model-*-of-*.safetensors` by base, pull all, store row path = first shard,
   like GGUF shard rows already do in `remove_model`). VRAM-gate + store row need the full-set
   bytes.
2. **Component flag wiring** — `--audio-encoder` rides `ComponentSpec.flag` (generic mechanism,
   same as `--vae`/`--t5xxl`); manual override stays `sdcpp_audio_encoder` config.
3. **Family row** — token `"s2v"` (narrow, before `"wan"`; excludes none — "s2v" cannot collide
   with wan2.1 tokens), `FamilyMode::Vid`, hardware-floor doc line (~36 GB VRAM class or heavy
   offload).
4. **Receipts on capable hardware** — family-resolution + component-plan + shard-set unit pins
   can land anywhere; the BOOT + ref_audios render receipt needs ≥40 GB-class VRAM (or patient
   CPU-offload run). Until then the row must not ship.

## Already-landed pieces this session (2026-10-31)

- `/v1/videos` ref-media relay + structural gate + capability gate + scratch pricing — live
  proven on wan2.1 with ref_videos frames (74 s render, VP8 WebM QC'd).
- `generation_features.ref_audios` capability pre-empt wired (fires on advertised-false;
  recipe-less models advertise true and the child stays content authority).
