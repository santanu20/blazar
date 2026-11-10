# Universal Model-Architecture Coverage - Research and Live Validation

Date: 2026-11-02
Scope: can Blazar serve every model family a local user can reasonably pull
(text LLMs, BERT-class embedders/rerankers/classifiers, CLIP/vision encoders,
VLM chat, diffusion image/video, ASR, TTS, omni/audio-input models)?
Method: code audit of all four crates + strings probes of the installed engine
binaries + upstream doc/API verification + live probes against both the
installed daemon (0.29.0 release, port 11435) and a worktree-built sandbox
daemon (uncommitted M-wave code, port 11599, isolated XDG data dir).
No product code was changed in this session.

Evidence tags used below:
[probe] = live HTTP probe against a real daemon/child (receipt captured)
[strings] = binary strings probe of an installed engine
[code] = verified source line in this repo
[api] = HF / GitHub API or fetched upstream doc

## 1. Validated coverage matrix

| Capability | Status | Evidence |
|---|---|---|
| Text LLM chat (GGUF, llama.cpp lane) | COVERED | store rows qwen3/gemma/granite...; live 32-149 t/s receipts in prior sessions [probe] |
| Text LLM chat (safetensors, sglang + mistral.rs lanes) | COVERED | qwen2.5-vl AWQ on sglang, granite-4.0-hybrid, mistral-7b rows [probe] |
| MLX lane (Apple) | COVERED (engine present, cross-platform gate) | mlx-0.32.2 installed; qwen2.5 mlx rows [code] |
| BERT embeddings (GGUF, child --embeddings lane) | COVERED - PASS | bge-small-en dim=384, cos(synonyms)=0.881 > cos(unrelated)=0.838 [probe] |
| BERT rerank | COVERED - PASS | bge-reranker-v2-m3 scores [4.94, -8.27, -11.04], top-1 correct [probe] |
| Classifier / decisions | COVERED - PASS | laya "turn the lights on" -> home automation 0.877 via child /v1/systemone [probe] |
| CLIP image+text embeddings (in-process candle lane, M2) | COVERED on worktree, NOT in 0.29.0 release | clip-vit-base-patch32 image_url -> 512-dim vector on sandbox daemon [probe]; release daemon 400s (predates M2) [probe] |
| VLM chat (image/video/audio input) | COVERED | qwen3vl-8b 47 t/s, qwen3.5-9b vision 32 t/s receipts; proxy forwards image_url/video_url/input_audio [code proxy.rs:197] |
| Image generation + edits/upscale/controlnet | COVERED | sdcpp master-1030; /v1/images/* routes [code lib.rs]; qwen-image-2.1 row live |
| Video generation | COVERED | wan_2.1 74 s render receipt (prior session); /v1/videos Sora verbs [code] |
| ASR (whisper.cpp engine) | COVERED | whisper b5454 engine + whisper-base row [probe prior] |
| ASR (safetensors whisper on sglang) | COVERED | WhisperForConditionalGeneration row routed sglang 0.5.23 [probe prior] |
| TTS (piper) | COVERED | piper voice rows, /v1/audio/speech + voices + transcode [probe prior] |
| TTS (llama.cpp /tts, qwen3-tts family) | COVERED - CLOSED G1 2026-11-02 | engine speech lane live receipts |
| Multimodal embeddings via VLM mmproj | COVERED gateway-side - CLOSED G2 2026-11-02; engine input-parts = b12020 limit (master-only), forwarding engine-ready | see G2 |
| Audio/video parts on /v1/embeddings | FORWARDED to child (engine 400 on b12020 - master-only feature) | see G2 |
| Diffusion family breadth | COVERED - 39 curated families incl. standalone/GGUF housing split, dual-DiT MoE (Wan 2.2), encoder-only + pixel-space shapes, PixArt w/ GGUF-T5; Ideogram4/PiD/SeFi/SenseNova/tiny-SD deferred w/ evidence [code diffusion.rs; 6 live render receipts] |
| glmocr / ggmlc arches (ollama-exported) | COVERED G4 - both pull lanes steer to the official native-arch repos (glm-ocr catalog entry, glm4 + auto mmproj); gateway OCR receipt live; spec-auto MTP arch-gated | 4 steering arms + OCR [probe] |
| llama.cpp in-process ASR lane | COVERED - engine-row lane forwards multipart to child `/v1/audio/transcriptions`, normalizes the event shape; whisper-enc towers are the quality target | live probe + [code] |
| CLIP breadth (SigLIP, ViT-L, GGUF mmproj reuse) | COVERED - config-driven candle lane, any CLIP size from its own config.json (G6); SigLIP/mmproj ride the G2 child path | live receipt g6-live-receipts.json [probe] |
| Catalog discovery beyond text LLMs | COVERED - 10 non-LLM entries + entry revision pins (G7 CLOSED 2026-11-02) [probe] |
| Kokoro / piper1-gpl voice frontier | UPSTREAM-BOUND G8 - kokoro absent from b12020 arch table; Qwen3-TTS/pockettts/LFM3-Audio reachable via G1 lane today | [api][strings] |

## 2. Engine ground truth (what the shipped engines can already do)

llama.cpp b12020 (the engine Blazar ships and keeps current):

- ~152 architecture tokens in libllama.so [strings], spanning text LLMs
  (llama/qwen/gemma/deepseek/glm/gpt-oss/mistral/kimi/minimax...), BERT-class
  encoders (jina-bert v2/v3, jina embeddings v5, nomic-bert + moe, modern-bert,
  neo-bert, eurobert, mmbert, roberta-bpe, gemma-embedding/2), t5encoder,
  RWKV world/6/arwkv7, mamba/mamba2, hybrid granite-moe, OCR (paddleocr,
  granite-docling, deepseek-ocr), and qwen3tts (audio generation).
- libmtmd.so [strings]: vision towers for ~40 VLM families (qwen2/3-vl,
  internvl, pixtral, gemma4 vision variants, glm4v/5v, deepseek vl/ocr,
  minicpm-v 4.6/4.7, kimi-vl, hunyuan-vl, step3vl, dots.ocr, voxtral,
  granite-speech, mimo-audio, nemotron-omni, lfm3a, moondream-class...),
  AUDIO INPUT (mtmd_audio_stream, whisper encoder), AUDIO GENERATION
  (qwen3tts_gen, pockettts, lfm3a, soprano, muse variants; speaker_ref
  voice-clone), and VIDEO INPUT (frame extraction).
- llama-server routes present in the shipped binary [strings]: /tts,
  /v1/audio/transcriptions, /v1/embeddings (multimodal input parts),
  /v1/rerank, /reranking, /v1/systemone, /infill, input_audio, input_video,
  speaker_ref.

sd.cpp master-1030 [api README + strings]: image families SD1/2/Turbo, SDXL,
SD3/3.5, FLUX.1/2/3, Chroma, Qwen-Image(+2.1/edit/turbo), PixArt, MOSS, Fibo,
Lens, Krea2, Mage-Flow, SeFi, HiDream-O1, Ideogram4, LLaDA, Ming, Iris,
z-image; edit families Kontext/LongCat/Boogu; video families Wan 2.1/2.2,
MiniMax-H3, LTX-2.3+. Blazar's FAMILIES table curates 8 of ~30.

## 3. Gap register (validated, prioritized)

### G1 - llama.cpp native TTS lane not surfaced (P0) - CLOSED 2026-11-02
IMPLEMENTED and live-validated this session (engine speech lane).

What shipped:
- /v1/audio/speech now routes by model field: a store row that resolves
  to the llamacpp lane enters the engine /tts lane; piper voice-ids,
  piper rows, and remote intents fall through to the existing piper
  paths byte-identically.
- Engine lane: gates in order (stream:true -> 400 teaching, piper-native
  knobs -> 400 teaching, modality-cache fast-teach when a cached list
  lacks "audio", mmproj audio-codec pre-check with speech-specific
  teaching), admission + ensure on the @vision lane (mmproj attach),
  POST /tts {input, voice?, lang?, speaker_ref_b64?} to the child,
  RIFF/WAVE validation of the child response, -1 dBFS peak limiting, then
  the full egress family: wav (audio/wav), pcm/pcm16 (raw s16le +
  x-blazar-pcm-format header), mp3/opus/aac/flac (existing ffmpeg
  transcode).
- Root-cause fix uncovered by live validation: llama-server b12020
  SEGFAULTS in process_tts_slots() when a qwen3tts child is spawned with
  --embeddings --pooling last (the daemon's embed-any-model parity arm)
  and then hit on /tts. profile.rs section 10c now skips every embedding
  arm for speech-generation arches (speech_generation_arch(), exact-match
  "qwen3tts" today); VLM/embedding rows keep the flags.
- /v1/audio/capabilities now advertises both speech lanes (piper +
  engine) with formats and knobs.

Live receipts (sandbox daemon 11599, worktree build, qwen3-tts-12hz-1.7b-base
Q4_K_M + auto-attached mmproj, CUDA): wav 403,244 B 24 kHz mono 8.4 s
(13.8 s cold wall), pcm raw + s16le;rate=24000;channels=1;bits=16 header,
mp3 ID3 36,909 B, stream:true 400 piper-teaching, speed knob 400
piper-teaching, piper regression 200 RIFF 145,056 B, non-TTS llamacpp row
honest 400 audio-codec teaching, capabilities lanes=[engine,piper].
Preserved: bench-artifacts/20261102-universal-coverage/ (wav, pcm, mp3,
capabilities, receipts json). Full suites: blazar-core + blazar-gateway
1342/1342 nextest green.

Historical research record (pre-implementation):
The gateway's /v1/audio/speech + /v1/audio/voices were piper-only
(gateway tts.rs: model = piper voice id). Meanwhile the shipped llama.cpp
b12020 serves /tts natively with voice-clone (speaker_ref) support, and
Qwen3-TTS GGUFs are plentiful (official ggml-org 1.7B Base; Serveurperso
community repacks at 924k downloads; cstr 0.6b customvoice).

Engine receipt: standalone llama-server b12020, CPU (-ngl 0),
Qwen3-TTS-12Hz-1.7B-Base Q4_K_M + mmproj Q8_0 ->
POST /tts {"input":"Blazar speech generation capability probe."}
-> 200 audio/wav, 145,964 bytes, 24 kHz mono 16-bit, 3.0 s of audio in
3.6 s wall time. WAV preserved at
bench-artifacts/20261102-universal-coverage/tts-qwen3-1.7b-base-q4km-cpu.wav.

Note: the community cstr 0.6b GGUF failed to load on b12020
("key not found: qwen3tts.context_length" - older conversion layout); the
official ggml-org quant loads cleanly. Pull-time arch teaching remains
future work (small).

### G2 - multimodal embeddings blocked at the gateway (P0) - CLOSED 2026-11-02
Shipped: model-aware dispatch in the /v1/embeddings handler. Requests
whose model resolves to a CLIPModel row keep the candle fast path; every
other row (VLM with mmproj, bert, speech, unresolvable names, remote
intent) now falls through to the proxy path, where vision-aware
admission (VISION_PROJECTOR_MISSING teaching, mmproj-aware @vision-lane
spawn) and byte-faithful child forwarding take over. Pure predicate
`clip_lane_serves(arch)` + unit pin; 786/786 gateway tests green.

Live receipts (sandbox daemon, worktree build - bench-artifacts/
20261102-universal-coverage/g2-live-receipts.json + g2-live-receipts-p1.json):
- qwen3vl-8b-instruct + image_url/input_audio -> forwarded to the
  resident @vision child (spawn + warm-peg on /v1/embeddings proven in
  the daemon log); the child answers the request itself [probe]
- bge-small-en + image part -> 400 VISION_PROJECTOR_MISSING teaching
  (admission gate, honest) [probe]
- clip-vit-base-patch32 image -> 512-d candle lane unchanged [probe]
- bge-small-en text-only -> 384-d unchanged [probe]

Engine-limit finding (probed on the resident child directly): llama.cpp
b12020 rejects image_url / input_audio parts on BOTH /v1/embeddings and
the native /embeddings endpoint ("prompt elements must be a string..."
for every dialect shape). Multimodal embedding inputs are a llama.cpp
master-only feature at the time of writing. The gateway forwarding is
engine-ready: when a future engine rev ships the feature, multimodal
embeddings light up with zero gateway changes. The candle CLIP lane
remains the image-embedding path for CLIPModel rows today.

G2b addendum (2026-11-03, engine b12072): the engine-limit verdict
resolved as WRONG-SHAPE, not missing feature. The engine accepts
multimodal embedding parts only in the OpenRouter content-array shape
(input elements carrying a content array of parts) - the direct
{type:image_url} element stays rejected. One gateway gap did hide behind
that shape: body_needs_vision scanned input elements directly, so a
content-array request spawned the child WITHOUT its mmproj and the child
refused it itself. Fixed across all three scanners (body_needs_vision,
req_has_image_input, candle CLIP item parser - each unwraps
input[i].content). Live receipts on the user's b12072 engine (g2b-live-
receipts.json): VLM content-array image 200 dim 4096 unit-norm 6.3 s
cold, mixed text+image 200, CLIP candle serves BOTH shapes at 512-d,
bert text 384-d unchanged, direct-shape VLM still returns the child's
honest 400 verbatim. 790/790 gateway tests green.

Historical record (pre-fix, worktree M2 behavior): the old shape-only
intercept served image input exclusively from CLIPModel rows and
refused VLM rows with "no CLIP model named ..." - the reachability gap
this closure removed.

### G3 - diffusion family curation gap (P1) - CLOSED 2026-11-02
Shipped: FAMILIES table 8 -> 20 entries (diffusion.rs). Added
Qwen-Image-Edit, SD 3.5, FLUX.3 Klein, MiniMax-H3 (first Vid family with
--audio-vae), MOSS-Image (--tokenizer surface), Krea 2, Iris 3B, Lens,
Fibo 2.0, Ming-Image, HiDream-O1 - each entry paired from its upstream
docs/<family>.md command and verified file-by-file against the HF repo
listing (head comment carries the two-tier honesty contract). Deliberate
non-entries with evidence: Ideogram4 (upstream requires a local fp8->bf16
python conversion of two transformers before any engine sees the weights -
a converter is engine-side capability, out of gateway scope by design).
PixArt and Wan 2.2 landed 2026-11-03 (G3b below) once the GGUF ecosystem
replaced the blockers cited at research time. Unknown families still pull
with the teaching naming the supported set. Architecture fix that landed with it: pull_standalone_lane now
classifies standalone-capable families by housing (StandaloneHousing) -
repos hosting the pinned checkpoint ride the standalone lane, DiT GGUF
mirrors of the same family (MOSS-Image, SD 3.5) fall through to the GGUF
lane where attach_diffusion_set completes the identical component set
(previously such mirrors were rejected as "hosts none of the checkpoint
files"). Live receipt: blazar pull leejet/MOSS-Image-Base-GGUF:Q4_K
fetched DiT + --vae flux2-vae + --llm Qwen3.5-9B (Q4_K_M fallback) +
--tokenizer, row components recorded, and a 512x512 PNG rendered through
/v1/images/generations in 37s (bench-artifacts/20261102-universal-coverage/
g3-moss-image-render.png; structurally QC'd - valid PNG, exact-size IDAT,
varied content). Historical record (pre-fix): the table covered 8 of ~30
upstream families; unknown-family pulls proceeded with a teaching warning
but no VAE/text-encoder auto-attach.

### G3b - deferred-family curation (P1) - CLOSED 2026-11-03
Shipped: FAMILIES 20 -> 24 entries, gateway-layer only (no engine
capability implemented in blazar - the m0617 mandate). Wan 2.2 TI2V-5B =
single-DiT GGUF lane (root one-file-per-quant repo, Wan2.2_VAE + umt5
byte-reuse). Wan 2.2 T2V-A14B = first dual-DiT MoE family: LowNoise
expert pinned as the standalone main (Low/HighNoise dirs ship
byte-identical per-quant sizes - largest-wins could never choose a
role), HighNoise expert attached as an exact-path
--high-noise-diffusion-model component with no quant fallback; bare
`blazar pull QuantStack/Wan2.2-T2V-A14B-GGUF` resolves the whole Q2_K
pair + VAE + byte-reused umt5 in one command. PixArt-Sigma/-alpha: DiT +
GGUF T5-XXL encoder (same bytes the FLUX.1 lane uses) instead of the
19GB sharded safetensors text encoder; each carries its own VAE (docs
warn latent spaces differ). Live receipts on 8 GB VRAM (sandbox
gateway, no mocks): TI2V-5B 320x320 webm 213 s via
/v1/videos/generations (ffprobe vp8-clean); A14B 256x256 webm 146 s
(overcommit + offload); PixArt-Sigma 512x512 PNG 23 s via
/v1/images/generations (PNG CRC-clean, exact-size IDAT) - this render
PROVES the GGUF-T5 pairing rather than assuming it
(bench-artifacts/20261102-universal-coverage/g3b-live-receipts.json +
3 render artifacts). Bonus receipts: interrupted component download
resumed from the v2 chunk ledger on relaunch; umt5 byte-reuse skipped a
re-download across families.

### G3c - remaining-family curation (P1) - CLOSED 2026-11-03
Shipped: FAMILIES 24 -> 39 entries - every remaining sd.cpp image/video
family with an expressible pull contract. Fifteen new entries (Anima,
Boogu-Image, ERNIE-Image, LLaDA-Image, LongCat-Image, LTX-2.3, LTX-2.5,
Mage-Flow, MiniT2I, Ovis-Image, Z-Image L2P, HunyuanVideo 1.5, Lingbot
Video, SSD-1B, Segmind Vega), each paired from its upstream
docs/<family>.md command and verified file-by-file against the HF repo
listing. New component surfaces exercised: --embeddings-connectors
(LLaDA, LTX-2.3), encoder-only families (MiniT2I pixel-space DiT +
exact-path flan-t5-large, no VAE; Z-Image L2P --llm only), A1111
full-checkpoint zero-component entries (SSD-1B, Vega), a canonical
redirect-chasing repo id (Comfy-Org/Hunyuan-Video-1.5), and lowercase
GGUF quant tokens (LLaDA text encoder q4_K). Companion root-cause fix:
the sdcpp spawn gate mandated a --vae/--model flag per component set,
which falsely refused the encoder-only and pixel-space shapes - now
shape-agnostic (any non-empty family set boots; empty set + per-file
existence checks kept verbatim). Live receipts on 8 GB VRAM (sandbox
gateway, no mocks): MiniT2I 512x512 PNG 7 s via
/v1/images/generations; lingbot-video-dense-1.3b 256x256 webm 33 s via
/v1/videos/generations after the VRAM admission gate taught the
overcommit lever (bench-artifacts/20261102-universal-coverage/
g3c-live-receipts.json + 2 render artifacts). Deferred with evidence:
PiD (.pth weights + value-taking --vae-format flag, not expressible as
flag+file component), SeFi (org-gated repos + no ungated hosting for
its qwen3-vl-2b encoder), SenseNova (8-shard + tokenizer-directory
contract the child spawn cannot pass as files), tiny-SD class (need
offline diffusers prep; concedo sdxs GGUFs remain exact-file pullable).

### G4 - ollama-exported glmocr / ggmlc GGUFs unloadable (P2) - CLOSED 2026-11-03 (pull-side steering)

Research correction (2026-11-03, docs/research/2026-11-03-g4-g8-engine-bound-architecture.md):
this was never an engine gap. The official ggml-org/GLM-OCR-GGUF exports
under the native `glm4` arch llama-server already loads, plus a standard
clip mmproj (range-request header probes); `glmocr` is only ollama's
export-tooling rename, and `ggmlc` is the same story for ggml-org/Laya-GGUF
(native `modern-bert`).

Shipped design, live-validated on the sandbox (receipts:
bench-artifacts/20261102-universal-coverage/g4-steering-live-receipts.json):

- `hfmeta::ollama_remapped_arch` - exact-match rename map to the official
  repo + catalog short-name.
- Both pull lanes steer at their choke points, present-row arm AND fresh
  download arm: the rename pull aborts with teaching
  (`pull the official build instead: blazar pull <short>`), downloaded
  files are deleted, no row lands. Verified on all four arms
  (HF laya-typed-decisions, registry glm-ocr; the registry lane unlinks
  its ollama-store hardlink without touching the original blob).
- Catalog carries `glm-ocr -> ggml-org/GLM-OCR-GGUF`; the happy-path pull
  lands arch glm4 (Q8_0 950 MB) with the mmproj auto-attached (484 MB).
- Gateway OCR receipt through /v1/chat/completions: exact transcript
  "HELLO BLAZAR 42" from a rendered-text PNG, 2.5 s on the 8 GB card
  (g4-ocr-gateway-receipt.json).
- Root-cause companion fix: `spec = "auto"` no longer engages draft-mtp on
  unproven arches. glm4/GLM-OCR carries a baked-in MTP head and b12020
  fails its draft context init (`unordered_map::at`, child exits at load);
  bisected live - the identical argv without spec flags serves chat + OCR
  fine. `mtp_arch_proven` gates embedded MTP to receipt-carrying arches
  (qwen family) with a teaching warning elsewhere; explicit `spec = "mtp"`
  still forces it for users who want to gamble.

Historical record (2026-11-02 verdict, superseded): stock llama-server
cannot load the renamed GGUFs (502 ENGINE_CRASHED); filings blazar#2,
ggml-org#19335 (closed via PR #19677).

### G8 - kokoro / piper1-gpl voice frontier (P2) - UPSTREAM-BOUND (documented 2026-11-02)
kokoro-82m GGUFs + voice packs exist on HF, but kokoro is not a llama.cpp
arch (b12020 strings; the mtmd audio-gen family covers qwen3tts/
pockettts/lfm3a/soprano/muse instead) and piper1-gpl upstream docs were
unreachable (404) this session. With G1's engine speech lane shipped,
the modern voice-quality frontier is reachable today: Qwen3-TTS
(custom-voice variants), pockettts, LFM3-Audio - all through
`/v1/audio/speech` with piper remaining the zero-download baseline.
kokoro needs its own engine lane or upstream adoption; revisit at the
next engine upgrade (mtmd gen-audio table grows every release).

### G5 - llama.cpp in-process ASR lane not surfaced (P2) - CLOSED 2026-11-02
Shipped: the engine-row ASR lane. `/v1/audio/transcriptions` (and
`/translations`) route by model field: a name resolving to a pulled
llamacpp row rides the child's native `/v1/audio/transcriptions` (mtmd
audio tower via the `@vision` sidecar instance), with the whitelisted
multipart rebuilt, the child's buffered
`{"type":"transcript.text.done"}` event normalized to OpenAI
`{"text": ...}`, and already-OpenAI bodies passed through unchanged.
Engine capability proven live before implementation (b12020 +
Voxtral-Mini Q4_K_M + mmproj Q8_0 standalone: multipart accepted, audio
tokens consumed, prompt/temperature honored, verbose_json honestly
rejected). Live battery through the sandbox gateway (receipt
`g5-live-receipts.json`): engine-row transcription 200 normalized
(cold spawn 6.3 s wall), whisper.cpp lane regression returned the exact
ground-truth transcript on the same WAV, stream + no-sidecar rows teach
400 with both remedies. Quality caveat, live-proven and documented:
conversational audio towers (voxtral class) answer conversationally on
this endpoint - the internal transcript is correct (bare-audio chat
quoted it verbatim), the endpoint's fixed prompt does not unlock it;
whisper-encoder-class towers are the designed target and light this
lane up fully with zero gateway changes.

### G6 - CLIP lane breadth (P2) - CLOSED 2026-11-02
Shipped: the candle CLIP lane is config-driven. It reads the row's own
`config.json` (verified against live pulls) and translates
`vision_config`/`text_config` into candle's `ClipConfig`, so ViT-B/32,
B/16, L/14 and any other CLIP tower load from the same code path and
preprocess at the model's own `image_size`. Rows without a `config.json`
fall back to the ViT-B/32 preset with the assumption named in any load
error (never mis-attributed to the weights). `hidden_act` outside
`quick_gelu` refuses with teaching (candle 0.11 implements only
QuickGelu for CLIP towers). SigLIP stays out by design: it is a
different HF architecture and already rides the G2 child-forwarding
path. Live receipts: `openai/clip-vit-large-patch14` pulled (1.6 GiB
F32, config image=224 patch=14 hidden=1024 projection=768) and served a
768-d unit-norm image embedding through `/v1/embeddings` in 21.7 s CPU
cold - a tower that previously failed with "expected ViT-B/32 layout";
ViT-B/32 regression unchanged at 512-d
(`bench-artifacts/20261102-universal-coverage/g6-live-receipts.json`).
Historical record (pre-fix): runtime/clip.rs hardcoded
`ClipConfig::vit_base_patch32()` and `IMAGE_SIZE = 224`, so every
non-B/32 CLIP row failed with the compatibility error.

### G7 - catalog is text-only (P1) - CLOSED 2026-11-02
Shipped: ten non-LLM catalog entries, each backed by a live pull receipt or
a freeze-time HF verification (200 + ungated): `bge-small-en` (embeddings),
`bge-reranker-v2-m3` (rerank), `laya` (decisions/classify),
`clip-vit-base-patch32` (CLIP, pinned to `refs/pr/15` where the safetensors
conversion lives), `qwen3-tts-1.7b` (speech), `whisper-base` (ASR), and
`moss-image` / `flux-3-klein` / `qwen-image-edit` / `wan2.1-t2v` (diffusion).
Catalog entries gained an optional `revision` pin (user `--revision` wins)
so non-main-file repos are honestly expressible; `blazar pull
clip-vit-base-patch32` now plans from the pinned ref with no flags.
Live receipts: bare short-name pulls landed rows (28 MiB GGUF bert, 581 MiB
CLIP safetensors from `refs/pr/15`), the pulled bert served `/v1/embeddings`
(384-d), the pulled CLIP row served the candle image lane (512-d)
(`bench-artifacts/20261102-universal-coverage/g7-*.json`). Rejected at
freeze: QuantStack Qwen-Image-2.1 GGUF mirrors (HF 401). Piper voices stay
out by design - `/v1/audio/voices?search=` is their discovery lane.
Historical record (pre-fix): catalog.json carried 35 text-LLM entries only,
so `blazar plan`-style discovery could not propose the non-text lanes the
runtime already supported.

### G8 - kokoro / piper1-gpl voice frontier (P2) - UPSTREAM-BOUND (documented 2026-11-02)
kokoro-82m GGUFs + voice packs exist on HF, but kokoro is not a llama.cpp
arch (b12020 strings; the mtmd audio-gen family covers qwen3tts/
pockettts/lfm3a/soprano/muse instead) and piper1-gpl upstream docs were
unreachable (404) this session. With G1's engine speech lane shipped,
the modern voice-quality frontier is reachable today: Qwen3-TTS
(custom-voice variants), pockettts, LFM3-Audio - all through
`/v1/audio/speech` with piper remaining the zero-download baseline.
kokoro needs its own engine lane or upstream adoption; revisit at the
next engine upgrade (mtmd gen-audio table grows every release).

## 4. Non-gaps (verified working, no action)

BERT embeddings/rerank/classify (child --embeddings/--reranking + systemone
dialect normalization), CLIP candle lane (M2, worktree), VLM chat across
three engine lanes with image/video/audio part forwarding (proxy.rs),
diffusion image gen + edits + upscale + controlnet, wan video gen,
whisper ASR x2, piper TTS + voices + transcode, hybrid-mamba/MoE GGUFs,
t5encoder, responses/realtime/batches/jobs planes, ollama API surface.

## 5. Implementation waves - EXECUTED 2026-11-02 (session r12)

All implementable gaps shipped with live no-mock receipts; the register
is closed except the two upstream-bound verdicts:

- W1 (P0): DONE - G1 engine speech lane (CLOSED) + G2 model-aware
  multimodal-embedding dispatch (CLOSED; b12020 engine-side image-part
  support is master-only, gateway lands it free on engine upgrade).
- W2 (P1): DONE - G3 diffusion FAMILIES 8 -> 20 + housing split
  (CLOSED) + G7 catalog breadth, ten non-LLM entries + revision pin
  (CLOSED).
- W3 (P2): DONE - G5 engine ASR lane (CLOSED) + G6 config-driven CLIP
  lane (CLOSED) + G8 voice frontier verdict (UPSTREAM-BOUND, modern
  voices reachable via G1) + G4 glmocr/ggmlc verdict (UPSTREAM-BOUND,
  filings tracked, fail-fast + teaching at every layer today).

## 6. Reproduction

Sandbox daemon (worktree code, isolated store):

    cd <repo>
    CARGO_TARGET_DIR=$SCRATCH/blazar-sbx-tgt cargo build --workspace
    # XDG skeleton: symlink models/, engines/ into $XDG_DATA_HOME/blazar/,
    # $SCRATCH = any disposable working dir (scratch checkout / probe space)
    # real dirs for run/ sessions/ speccache/ voices/, python sqlite3 .backup
    # of store.db/blazar.db/models.db, port patched to 11599
    XDG_CONFIG_HOME=$SCRATCH/blazar-sbx/config \
    XDG_DATA_HOME=$SCRATCH/blazar-sbx/data \
      $SCRATCH/blazar-sbx-tgt/debug/blazar serve

TTS engine capability (standalone child, CPU):

    cd $SCRATCH/blazar-tts-probe   # ggml-org Qwen3-TTS-1.7B-Base Q4_K_M + mmproj Q8_0
    LD_LIBRARY_PATH=<engines>/b12020-cuda/llama-b12020 llama-server \
      -m qwen3-tts-1.7b-base-Q4_K_M.gguf --mmproj mmproj-qwen3-tts-Q8_0.gguf \
      --port 18234 -ngl 0 -c 4096
    curl -X POST http://127.0.0.1:18234/tts \
      -H 'Content-Type: application/json' \
      -d '{"input":"Blazar speech generation capability probe."}' -o receipt.wav

Raw probe log for this session: bench-artifacts/20261102-universal-coverage/
(tts-qwen3-1.7b-base-q4km-cpu.wav - 24 kHz mono WAV, 3.0 s).
