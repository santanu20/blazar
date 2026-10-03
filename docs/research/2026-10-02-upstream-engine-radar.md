---
layout: doc
title: "Upstream Engine Radar"
description: "Research note: upstream releases worth tracking per lane."
doc_kind: "Research note"
---

# Upstream Engine Radar — 2026-10-02

Harvest: merged PRs (last 2 weeks), top open issues by reactions, most-active
open PRs, latest releases for all six engine lanes. Everything below cites the
upstream PR/issue number. Goal: what Blazar should **adopt now**, what it must
**be ready for**, and what to **watch**.

| Engine | Latest release | Velocity signal |
|---|---|---|
| llama.cpp | v0.5.0 (2026-09-23) + b-builds | GLM-5.3-Flash, systemone API, adaptive-MTP PR hot |
| mistral.rs | v0.9.4 (2026-09-24) | delimiter semantics change, Qwen3.8-Flash-Next |
| sglang | v0.5.21 (2026-10-02) | router endpoints, HiCache modes, Apple MPS PR |
| stable-diffusion.cpp | master-929 | upscale endpoint merged, PixArt/Qwen-Image wave |
| whisper.cpp | b5130 (2026-09-11), v1.9.4 tags alongside | **repo moved ggerganov → ggml-org**, VAD-on-GPU |
| mlx-lm | v0.31.3 GH / 0.32.x PyPI | DeepSeek-V4.1, tool-args parsing fix |

## A — Adopt now (shipped upstream, Blazar action available today)

### llama.cpp

- **`/v1/systemone` decision API** (PR 29818, merged): BERT/Qwen-style
  embedding models served as "decision models" (laya, julia-1, lev, openjev,
  kev) behind one endpoint. Blazar's compat census already carries
  `/v1/systemone` — the gateway should route it as a passthrough on llamacpp
  lanes (no translation needed) and teach it in the API spec when a decision
  GGUF is pulled.
- **GLM-5.3-Flash / GLM5-Next** (PR 27773, merged) and **nimble decision
  model** (PR 29844): pure model-support additions — verify GGUF pull + ctx
  defaults flow through Blazar's model matrix unaided; add to the model
  registry's known-good list when spot-checked.
- **Probabilistic drafter + rejection sampling for simple draft and MTP**
  (PR 27694, merged): changes speculative-decode acceptance behavior, not the
  CLI surface. Blazar bench spec-config axes should pin drafter mode so
  speed-vs-quality A/Bs stay comparable across engine bumps.

### stable-diffusion.cpp

- **`POST /sdcpp/v1/upscale`** (PR 2026, merged): standalone ESRGAN upscale
  over HTTP — no diffusion model load, synchronous. Before this, upscale was
  only reachable via `hires` inside a generation (which even aborts
  Qwen-Image-2.1 on some shapes). **Action:** expose upscale on Blazar's
  image API (wrap or passthrough), gate on sdcpp ≥ master-929 via manifest
  feature probe.
- **Async job progress on `GET /sdcpp/v1/jobs/{id}`** (PR 1884, open but
  small): distinguishes slow vs stuck jobs — Blazar's media job polling
  should surface `progress` when present and stay correct when absent.
- **Configurable Qwen cache types + early cache scheduling** (PR 2045,
  merged): `qwen_image_2_1_prefix_cache_type` with quantized storage options
  and FP16-only-under-FlashAttention logic — a VRAM win to pass through as a
  per-model knob once the flag is probed into the sdcpp manifest.
- **PixArt family** (PR 2047), **LLaDA-Image** (1968), **Ming-Image** (2063),
  Qwen-Image-2.1 fix cluster (2054/2057/2058/2048): model-matrix refresh for
  the image lane.

### mistral.rs

- **Delimiters only as special tokens** (PR 2455, merged v0.9.4): raw-text
  `<think>`/tool delimiters are no longer honored — reasoning arrives only
  via `reasoning_content`. Blazar's raw-think suppression stays as
  belt-and-braces for old builds and other lanes; the mistral.rs dialect
  normalization (enable_thinking/effort) remains the correct control path.
  No code change needed — documented understanding.
- **Qwen3.8-Flash-Next** (PR 2462): model-matrix refresh.

### sglang

- **sgl-router gains OpenAI `/v1/embeddings` + native `/generate`** (PRs
  42191/42147, merged): parity signal — Blazar already serves both surfaces;
  for the sglang lane, mapping Ollama-compat `/api/generate` to sglang's
  native `/generate` (instead of translating to chat-completions) is a
  candidate latency win — measure before adopting.
- **`cache_unfinished_req` → `checkpoint_req` rename** (PR 41520): if Blazar
  scripts or docs reference sglang internals, update the name.

### mlx-lm

- **tool_calls `function.arguments` parsing fix** (PR 1904) and **top_p
  keep-most-likely-token fix** (PR 1912): correctness fixes relevant to
  Blazar's tool lane parity — raise the recommended mlx-lm floor (0.32.x) so
  pip installs pick them up.

### whisper.cpp

- **Silero VAD on GPU** (PR 4083, merged): `--vad-gpu`-class flag (probe
  exact name) — pass-through knob for the transcription lane on CUDA hosts.
- **Release-asset layout evolving** (PR 4029: macOS CLI binaries added): the
  asset-selection whitelist in the installer should stay pattern-based, not
  index-based.

## B — Be ready (in flight; prep now, light the path on merge)

### llama.cpp

- **Adaptive MTP** — PR 27210 (56👍): `--spec-type draft-mtp-adaptive` +
  `--spec-draft-n-max` (recommend ≥8; counting state machine climbs/drops
  draft depth). **Prep:** extend profile.rs speculative argv surface with
  the two flags behind a manifest feature gate; add a bench spec axis. When
  merged, Blazar ships support same-day.
- **Disk-based context checkpoint offloading** — issue 20697 (57👍):
  `--cache-disk` to spill ctx checkpoints off UMA RAM. Natural extension of
  Blazar's session save/restore; plan a per-model disk-cache knob + doctor
  disk-budget row. Watch for the implementing PR.
- **DiffusionGemma** (PR 24423, 190👍 — highest-signal PR in the repo) and
  **LLADA 2.0 diffusion** (PR 17454): deep dive 2026-10-03
  (see `2026-10-03-diffusion-llm-readiness.md`). Neither touches
  `tools/server` — PR 24423 ships a CLI plus a stdin/stdout logits-worker
  binary driven by a Python loop; **no HTTP serving exists upstream**.
  Meanwhile `llada`/`llada-moe` are already mainline-loadable, so
  llama-server would decode them autoregressively into garbage without
  crashing. **Shipped:** route-time paradigm guard
  (`DIFFUSION_PARADIGM_ARCHS` in blazar-core, mirrored in the list ENGINE
  column) refuses with teaching; arch set pre-carries `llada2` and
  `diffusion-gemma` so merge day needs no code change. Streaming-tolerance
  audit checklist lives in the review doc for the serving day.
- **Scheduler UMA ring buffer** (PR 27311, 51💬): makes host buffers viable
  on integrated GPUs — perf-only, but changes memory-reporting shapes
  (`alloc_buffer_n` iface, PR 23671); keep blazar VRAM accounting defensive.
- **FP8 GGUF** (PR 10055), **SM120 CUTLASS MoE MXFP4/NVFP4 prefill**
  (PR 26704), **SparseK attention** (PR 16817), **`--numa mirror`**
  (PR 16000): quant/backend matrix growth — GGUF magic/version bumps will
  follow; keep the GGUF parser version-tolerant and ngl/flag probing dynamic.

### mistral.rs

- **RMCP streamable client** (PR 1531): MCP client support server-side —
  relevant to Blazar's tool/orchestration roadmap; track protocol surface.
- **OpenTelemetry support** (PR 1479): Blazar diagnostics could forward
  engine OTel spans when present.
- **KvCacheCodec pluggable hook** (PR 2116) + **no-op KV connector seam**
  (PR 2364): extension points Blazar's cache orchestration can eventually
  drive; no external surface yet.
- **Idle CPU reduction** (PR 775): matches Blazar's own idle-sip work;
  verify with a bench axis when it lands.
- **XDNA NPU** (issue 1254, 28👍) / **ROCm** (issues 431/1345): new asset
  classes for the installer matrix — far out.

### sglang

- **Apple Silicon (MPS) platform** (PR 36780, open as of 2026-10-03): the
  standard Torch runner on macOS — the moment it merges, Blazar's sglang lane
  becomes viable on Mac hardware. **Prep:** soften the sglang-fit doctor gate
  (today CUDA-shaped) to accept an MPS platform marker; keep mlx as the
  default Mac lane.
- **Native `/generate` endpoint** (PR 42147 — merged, but **router-only**):
  all 20 changed files live under `experimental/sgl-router/`; Blazar spawns
  the sglang server directly and never the router, so the endpoint is
  unreachable today. Re-evaluate only if the server itself gains
  `/generate`. Radar item 6 resolved as a no-op (verified 2026-10-03).
- **HiCache L2 `buffer_only` mode** (PR 20535, 10👍/40💬) and **distributed
  KVCache for agentic workloads** (issue 21846 roadmap): shared host-memory
  staging instead of per-worker duplication — config knobs to pass through
  once stable.
- **Kimi K3 / auto-tuner / HiSparse / PD-disagg roadmaps** (issues 32607,
  13363, 28874, 21703): watch; no Blazar surface yet.

### whisper.cpp — repo rename (act soon, low cost)

Upstream moved **ggerganov/whisper.cpp → ggml-org/whisper.cpp** (API 301,
repo id 541269386). Blazar's **engine lane already targets ggml-org**
(`WHISPER_REPO` in crates/blazar-runtime/src/whisper.rs:21 and
engine/gh.rs:1071) — correct. Still stale on the old name:

- `WHISPER_MODEL_REPO = "ggerganov/whisper.cpp"` (whisper.rs:24) — the ggml
  model catalog listing/pull path. Works today via GitHub redirect; update
  before redirects ever retire.
- Comments/help text: engine_kind.rs:29, main.rs doctor/search help,
  manifest.rs:846 ("verified against ggerganov/whisper.cpp b5130").
- Release scheme note: v1.9.x tags (no assets) now ship **alongside** b-tags
  (with assets) — Blazar's b-tag-only asset logic remains correct; keep
  ignoring v-tags for binaries but don't be surprised by dual tagging.

### mlx-lm

- **`--prompt-cache-file` for mlx_lm.server** (issue 1178): pairs with
  Blazar's session/KV-checkpoint story on the Mac lane — map to session
  save/restore when it exists.
- **DFlash block-diffusion speculative** (issue 1135), **Qwen3-Omni**
  (issue 497, 20👍), **ParoQuant** (issue 977): watch.
- **Known upstream wart:** malformed tool-calls around 20k prompt tokens
  (issue 1061) — keep Blazar's tool-call validation strict so the failure is
  caught and taught, not passed through as garbage.

## C — Watchlist (signal, no near-term surface)

llama.cpp: ANE backend (85👍), DirectML (39👍), XDNA (35👍), Mamba GPU
(32👍), Vulkan allreduce/TP (PR 25051), grammar-sampling speedup (22👍),
Orpheus TTS/SNAC codec (25👍 — a local-GGUF TTS lane for Blazar someday),
`/v1/responses` bugs (issue 14702 — stay strict where upstream is loose).
sglang: free-threaded Python (22889), CurveZMQ security fix (21567 — pick up
in the next sglang bump), LLaDA2.2 MoE (31768). sdcpp: LTX-2 (18👍),
MiniMax-H3 video (15👍), Z-Image ControlNets (1033/1074), text2music (870),
flash-attn-by-default (PR 1186). mistral.rs: KBNF grammar (815), Flux.2-klein
(1879), MXFP4_MOE GGUF dtype 39 (2117). mlx-lm: distributed inference (409).

## Priority shortlist for Blazar

1. sdcpp **upscale endpoint** exposure — DONE (wave1).
2. **Adaptive-MTP flags** pre-wired behind a feature gate — DONE (wave1,
   `spec = "mtp-adaptive"`).
3. **whisper.cpp rename cleanup** — DONE (wave1).
4. llama.cpp **systemone** passthrough + docs — shipped by the gateway work;
   API-spec entry pending.
5. **VAD-on-GPU** knob — DONE (wave1).
6. sglang **native `/generate`** — RESOLVED NO-OP (router-only, 2026-10-03).
7. **Diffusion-LLM readiness** — DONE (2026-10-03): paradigm guard shipped,
   design review + streaming-tolerance checklist in
   `2026-10-03-diffusion-llm-readiness.md`.
