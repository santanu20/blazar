# Blazar Runnable Architecture Inventory - 2026-11-03

Every model class and architecture blazar can run today, verified against the
installed engines (not marketing claims). Companion to
`2026-11-02-universal-architecture-coverage.md` (gap register + waves).

Method: binary strings probes of the installed engine builds
(llama.cpp b12020-cuda libllama.so + libmtmd.so + libllama-server-impl.so,
re-probed 2026-11-03), upstream release notes (b12058-b12069 fetched live),
and blazar's own live receipts (bench-artifacts/20261102-universal-coverage/).
Engine lanes: llamacpp b12020, sglang 0.5.23, mistral.rs v0.9.4-lane,
sdcpp master-1030, whisper.cpp b5454, piper, mlx 0.32.2 (mac), candle
in-process (CLIP).

## 1. Text LLMs (llamacpp lane, ~150 GGUF arches)

Old-generation through current-generation, all loadable:

- Classic: gpt2, gptj, gptneox, gpt3-finnish, starcoder, starcoder2, falcon,
  falcon3, falcon-h1, mamba, mamba2, rwkv, rwkv6, rwkv6qwen2, rwkv-world,
  arwkv7, Vicuna (vicuna, vicuna-orca), OpenELM, Jais (jais, jais2),
  Exaone (exaone, exaone3, exaone4, exaone-moe), Sea-Lion (orion).
- Llama line: llama (1/2/3), llama4, smollm, smollm3, smolvlm, stablelm,
  stablelm2, minicpm, minicpm3, minicpm5, internlm2, command-r, chameleon,
  Refact (seed-coder), Plamo (plamo, plamo2, plamo3), grok, grok-2.
- Qwen line: qwen3, qwen3moe, qwen3next, qwen35, qwen35moe, qwen4exp,
  qwen2moe (2.5 class rides qwen2 arch family), plus vision variants in
  section 2.
- DeepSeek line: deepseek (v2/v3), deepseek2, deepseek3, deepseek4,
  deepseek41, deepseek-coder, deepseek-llm, deepseek-r1-qwen,
  dots1, dots3note (dots.ocr).
- GLM line: chatglm, chatglm3, chatglm4, glm4moe, glm5, glm5-next, glm-dsa,
  glmedge (GLM-OCR official GGUFs ride glm4 - the G4 steering target).
- Gemma line: gemma, gemma2, gemma3, gemma3n, gemma4, gemma4-assistant.
- Granite line: granite, granite-4.0, granite-4.1, granitehybrid (mamba-MoE),
  granitemoe, graniteswitch, granite-docling.
- Mistral line: mistral (v1/v3/v3-tekken/v7/v7-tekken/large4/3/4).
- MoE giants: dbrx, arctic, bailing family (bailing, bailing2, bailingmoe,
  bailingmoe2, bailingmoe3, bailing-think), hunyuan, hunyuan-dense,
  hunyuan-moe, kimi-k2, kimi-k3, kimi-linear, minimax-01, minimax-m2,
  minimax-m3, gpt-oss, phi (phi-2/3/4 + phimoe), olmo, olmo2, olmoe,
  nemotron.
- Non-llamacpp text lanes: sglang (transformers safetensors - bf16/AWQ/GPTQ;
  verified live: granite hybrid, qwen AWQ), mistral.rs (mistral-family
  safetensors + GGUF), mlx (Apple Silicon MLX models; verified live: qwen2.5).

## 2. Vision + multimodal LLMs

Two mechanisms, both live-proven:

- Native VLM GGUF arches (llamacpp): qwen2vl, qwen3vl, qwen3vlmoe, pixtral,
  smolvlm, chatglm4/glm4v class, hunyuan-vl, llama4, minicpm-v class.
- mtmd projector towers (mmproj sidecar, ~50 clip_graph loaders in
  libmtmd.so b12020): llava, llava-1.6, cogvlm, siglip-tower VLMs, qwen2vl /
  qwen3vl / qwen3a, internvl, pixtral, gemma4v / gemma4uv / gemma4ua /
  gemma4a, glm4v / glm5v, deepseek4v / deepseek-ocr / deepseek-ocr2,
  minicpm-v / minicpmv4_6, kimi-vl / kimi-k25, hunyuan-vl, step3vl, youtuvl,
  dots.ocr, lfm3vl, ling3vl, mimovl, yasa2, exaone4_5, granite4-vision,
  nemotron-v2-vl, moondream class, paddleocr, mobilenetv5, minimax-m3 tower.
- Video input: mtmd video config (frame extraction) - chat lanes accept
  video_url parts (proxy forwards, live-proven).
- sglang vision: qwen2.5-vl AWQ verified live.
- OCR: GLM-OCR (official ggml-org GGUF = glm4 + clip mmproj, gateway receipt
  'HELLO BLAZAR 42'), deepseek-ocr class, dots.ocr, paddleocr, granite
  docling.

## 3. Embeddings (text)

- bert family (GGUF): modern-bert, nomic-bert, nomic-bert-moe, mmbert,
  jina-bert-v2/v3 + jina-{de,es,v1-en,v2-code,v2-de,v2-es,v5-nano},
  roberta-bpe, eurobert, nemotron-3.5-embed, granite-embed-multi-{97m,311m},
  gemma-embedding, gemma-embedding2, t5encoder (umt5-xxl verified live).
- sglang lane: transformers embedding models (bert safetensors class).
- Late-chunking lane: any embedding row (gateway feature).
- Token-level: /embeddings native pooling none surface.

## 4. CLIP (image embeddings) + rerank + classify

- CLIP image embeddings: candle in-process lane, config-driven (any CLIP
  size - ViT-B/32 512-d and ViT-L/14 768-d live receipts); GGUF CLIP arch
  also in the table.
- Rerank: bert cross-encoders (bge-reranker-v2-m3 live receipt, scores
  ordered); --reranking surface on llama-server.
- Classify / decisions: modern-bert classifiers (laya live receipt) via
  /v1/classify + /v1/systemone native dialect + /v1/decisions.

## 5. Audio input (ASR / transcription / translation)

- whisper.cpp engine: full ggml whisper family (tiny..large-v3, turbo,
  distil-whisper) - exact-transcript live receipt.
- llama-server mtmd whisper_enc tower: /v1/audio/transcriptions on
  audio-tower rows (G5 lane; voxtral live receipt).
- sglang whisper safetensors (whisper-base row live).
- Audio parts in chat: input_audio forwarding live-proven.
- Audio-tower VLMs: voxtral_rt, parakeet, conformer, granite-speech,
  mimo-audio, nemotron-omni, d1omni, muse_spark, qwen3a, dots3note-a.

## 6. Speech generation (TTS)

- piper lane: rhasspy VITS voices (en_US-amy-medium live receipt),
  ffmpeg transcode to mp3/opus/aac/flac, pcm streaming.
- Engine speech lane (G1): llama-server native /tts - arch qwen3tts
  (Qwen3-TTS-12Hz verified live, WAV/PCM/MP3 receipts) + mtmd gen towers:
  pockettts (+seanet codec, speaker cloning + predefined-voice packs
  upstream b12069), lfm3a (LFM3-Audio), soprano, muse_glimmer, muse_spark.
- Voice cloning: speaker_ref passthrough on /v1/audio/speech.

## 7. Diffusion - image generation + editing

35 curated families (FAMILIES table, each doc-paired + HF-verified; live
render receipts on 8 GB VRAM):

SD 1.5, SDXL, SD 3.5 (large+medium), FLUX.1 (dev/schnell/Kontext),
FLUX.2-dev, FLUX.3 Klein, Z-Image, Z-Image L2P, Chroma, Qwen-Image,
Qwen-Image-2.1, Qwen-Image-Edit, MOSS-Image, Krea 2, Iris 3B, Lens,
Fibo 2.0, Ming-Image, HiDream-O1, PixArt-Sigma, PixArt-Alpha, Anima,
Boogu-Image, ERNIE-Image, LLaDA-Image, LongCat-Image, Mage-Flow, MiniT2I,
Ovis-Image, SSD-1B, Segmind Vega (+ distilled-SD rides the sd tokens).

Capabilities: text-to-image, image-to-image, inpaint, upscale (ESRGAN +
diffusion), controlnet, edits (Kontext/Qwen-Edit class), variations.
Any sd.cpp-supported family outside the table still pulls (DiT + teaching
warning) and boots with manual component placement.

## 8. Diffusion - video generation

Wan 2.1 (T2V/I2V), Wan 2.2 TI2V-5B (live webm receipt), Wan 2.2 T2V-A14B
dual-DiT MoE (live webm receipt), MiniMax-H3 (video + stereo audio),
LTX-2.3, LTX-2.5 (video + stereo audio), HunyuanVideo 1.5, Lingbot Video
(live webm receipt). OpenAI-shaped /v1/videos/generations + Sora verbs.

## 9. Omni / realtime / agentic surfaces

- Omni models: qwen3-omni class (qwen3vlmoe + audio), nemotron-omni,
  muse-spark (text+vision+audio gen), d1omni.
- /v1/realtime websocket plane (gateway) + llama-server /v1/realtime
  (upstream).
- Responses API, tools/function-calling, structured output, MCP, batches,
  jobs - full gateway planes (unchanged from 0.29.0).

## 10. What blazar CANNOT run today (and why)

| Item | Reason | Path |
|---|---|---|
| Ideogram4 | upstream requires offline fp8->bf16 python conversion of 2 transformers (in-engine capability) | upstream ships convertible weights |
| PiD | .pth weights + value-taking --vae-format flag (pull contract not expressible) | upstream GGUF/fp8 repack |
| SeFi | org-gated repos, no ungated encoder hosting | gating lifts |
| SenseNova | 8-shard + tokenizer-directory spawn contract | upstream single-file repack |
| tiny-SD class | offline diffusers prep needed | concedo sdxs GGUFs = exact-file pullable |
| kokoro TTS | no engine builds the arch (piper1-gpl rejected: VITS-only, pip distribution) | G1 lane picks it up free if upstream adds it |

Resolved after the engine update + G2b gateway fix (2026-11-03, receipts
`bench-artifacts/20261102-universal-coverage/g2b-live-receipts.json`):

- Multimodal embedding inputs: LIVE on the current engine (b12072). The
  engine accepts image parts on /v1/embeddings in the OpenRouter
  content-array shape (`input: [{"content": [parts]}]`); the gateway now
  detects that shape for vision-aware spawn (mmproj attaches) and the
  candle CLIP lane serves it too (dim 4096 VLM receipt, 512-d CLIP
  receipt, all text/shape regressions green).
- Clover: runnable since the b12069+ engine update (clover_moe arch
  token present); any Clover GGUF pulls and boots through the llamacpp
  lane as-is.

## 11. Upstream delta (b12020 -> b12069, 2026-11-03)

Engine update (`blazar engine update`) picks all of this up with zero
gateway changes: Clover AR-MoE arch (PR #31306), PocketTTS predefined voice
packs + voices listed in GET /props (PR #31429 - candidate to surface in
/v1/audio/capabilities), opencl iq4_xs, plus routine perf work.

One gateway-side fix did land with the update (G2b, 2026-11-03): the
multimodal-embedding content-array shape was invisible to the gateway's
body scanners, so children spawned without their projector and rejected
the request. `body_needs_vision`, `req_has_image_input` and the candle
CLIP item parser all unwrap `input[i].content` now - live receipts on
b12072 (VLM 4096-d, CLIP 512-d both shapes, bert/text regressions).

## 12. Endpoint map (class -> API surface)

| Class | Endpoints |
|---|---|
| Text LLM | /v1/chat/completions, /v1/completions, /v1/messages, /v1/responses, /api/* (ollama) |
| Vision LLM | same + image_url/video_url parts |
| Embeddings | /v1/embeddings (+ /api/embeddings) |
| Rerank | /v1/rerank, /rerank, /reranking |
| Classify | /v1/classify, /v1/systemone, /v1/decisions, /v1/score |
| CLIP | /v1/embeddings (image parts) |
| ASR | /v1/audio/transcriptions, /v1/audio/translations |
| TTS | /v1/audio/speech, /v1/audio/voices |
| Image gen/edit | /v1/images/generations, edits, variations, upscale |
| Video gen | /v1/videos/generations + Sora verbs |
| Realtime | /v1/realtime (WS) |

Receipts: bench-artifacts/20261102-universal-coverage/ (g1-g7, g3b, g3c,
g4, g5, g6, doctor). Live-validated rows in the real store cover every
class above except the deferred five in section 10.
