# Engine changelog & API-change audit — 2026-10-29

Delta window: 2026-10-21 (prior engine-currency audit + surface re-diff baseline
b11785/installed-b11946) → 2026-10-29. Method: GitHub release/PR/compare APIs per
engine lane, diffed against Blazar's emitted-flag union, route-call surface
(`images.rs` sdcpp calls, gateway census), version/channel parsing
(`parse_version`, `channel_b_release`, `resolve_cuda_asset`), and manifest gating.

## 1. Version deltas

| Engine | Installed | Latest upstream | Delta | Update verdict |
|---|---|---|---|---|
| llama.cpp | b11946-cuda | **b11982** (2026-10-29) | 20 builds, 19 PRs | SAFE — pure kernel/correctness wave, no server API/flag change |
| mistral.rs | v0.9.3 (pinned) | v0.9.5 (2026-10-19) | no new release | PIN CORRECT — #2503 fix merged to master 2026-10-21, still unreleased |
| sglang | 0.5.22 | 0.5.22 | none | CURRENT |
| mlx-lm | 0.32.2 | 0.32.2 | none | CURRENT (qwen3 tool-marker wall upstream-unchanged) |
| sdcpp | master-1017-be48649 | **master-1024-77b72a2** (2026-10-28) | 7 commits | SAFE — additive server change, our routes untouched |
| whisper.cpp | b5454 | v1.9.5 (2026-10-06) | 0 commits | CURRENT — b5454 ≡ v1.9.5 (verified identical trees) |
| piper | 2023.11.14-2 | unchanged | — | CURRENT |

## 2. llama.cpp b11946 → b11982 (the only fast lane)

Full 19-PR changelog classified. **Zero new server flags, zero endpoint changes,
zero deprecations** — the first window since b11588 with no `tools/server` surface
movement.

| PR | Build | Class | Blazar impact |
|---|---|---|---|
| #30557 stop qwen3-coder reasoning/content before template separators | b11951 | chat-template correctness (`common/chat.cpp`, peg-parser) | rides along on update — qwen-coder family template handling fixed |
| #31288 CUDA top-k MoE fusion for 384 experts | b11953 | perf kernel | MoE GGUF decode speedup on CUDA lane (granite-4.0-tiny etc.) — bench-worthy |
| #31188 CUDA fuse MoE gate/up + GLU in MMQ | b11962 | perf kernel | same — combined MoE win, bench before/after |
| #31201 IQ4 quant creation speedup (precomputed halfway points) | b11969 | tooling perf | faster IQ4 quantization; conversion-time only |
| #31336 bound control-vector layer indices + M-RoPE position loops | b11972 | robustness (`common`, mtmd/paddleocr) | rides along — OOB guard for ctrl vectors + vision M-RoPE |
| #31309 version bump 0.7→**0.8.0** | b11960 | milestone | **non-event for Blazar** — CMakeLists `LLAMA_VERSION_MINOR` only; `--version` keeps `version: 0.8.0-dev (build 11982, commit …)` shape; `parse_version` splits on the `build` token (manifest.rs:983) → unaffected. Tags stay `bNNNN` → `channel_b_release` unaffected. Verified against local b11946 output |
| #31254/#31344 test-constant + CUDA-graph-reservation fixes | b11954/78 | test/infra | none |
| #27472 SVE-256 gemm_q6_K, #30976 SYCL dequant vectorization, #31110 SYCL argmax, #31125 SYCL DPAS flash-attn quantized KV, #30995/#31050/#31277/#31331 Hexagon conv2d/IM2COL/CONCAT/UPSCALE, #30412 OpenCL adreno q4_k GEMM, #31334 OpenVINO SCALE views, #31317 s390x CI | various | other-backend kernels | platform-breadth rides along (any-hardware goal); no x86-CUDA effect |

Asset readiness (update path): b11982 ships the full 38-asset matrix including
`llama-b11982-bin-ubuntu-cuda-12.8-x64` and `-cuda-13.4-{x64,arm64}` —
`resolve_cuda_asset` picks by driver as designed. Overlay channel (`BLAZAR_ENGINE_REPO`)
not required; upstream assets consumed directly since b10969.

## 3. mistral.rs — pin analysis re-validated

- v0.9.5 remains latest **release**; our v0.9.3 pin is still the correct call
  (v0.9.5 = upstream #2502 eternal-503 on 8GB cards).
- **#2503 "fix runtime activation overestimation" merged to master 2026-10-21,
  still unreleased.** master = v0.9.5 + 8 commits. Watch trigger stays armed
  (MEMORY action item): on next release → `blazar engine update mistralrs`, rerun
  e2e lane, revisit kv-cache reservation + paged ladder follow-up.
- Next-release incoming content: LFM3 text+VL support (#2515), ReflectionAI Flint
  (#2507), Metal blockwise-FP8 decode/prefill accel (#2512), Windows CUDA build
  fix (#2501), marlin `<string>` fix (#2500). All model-support/perf — no argv
  surface change flagged yet; re-diff `--help` at release time per the §6 method.

## 4. sdcpp master-1017 → master-1024

| Commit | Class | Blazar impact |
|---|---|---|
| #2221 recipe model defaults + idempotent model loading | **server API (additive)** | `/sdcpp/v1/models/load` gains optional recipe-local `models` defaults + `already_loaded: true` no-op semantics. **Blazar never calls models/load** (gateway call surface = capabilities, img_gen, vid_gen, upscale, jobs/{id}, jobs/{id}/cancel — images.rs) → zero impact; idempotent loads are a future optimization for our media lane if we ever adopt server-side model management |
| #2235 INT8 diffusion CUDA fusions | perf | image-lane TTFT win — bench after update |
| #2228 integrated GPUs in automatic memory planning | hw | correctness on iGPU boxes (any-hardware) |
| #2227 honor requested hires sampling step count | fix | hires-path correctness |
| #2237 Chroma distinct version id, #2230 Node-24 CI | infra | none |

Route table for all six Blazar-called routes unchanged in master-1024.

## 5. Preparedness assessment (is Blazar ready for these changes?)

| Seam | Verdict | Evidence |
|---|---|---|
| Version parsing after 0.8.0 bump | READY | `parse_version` keys on `build NNNN` token; bump touched only CMake soversion; local b11946 `version: 0.7.0-dev (build 11946…)` shape → b11982 same shape with 0.8.0-dev |
| Channel tag resolution | READY | tags remain `bNNNN`; `channel_b_release` pattern untouched |
| CUDA asset pick | READY | ubuntu-cuda 12.8/13.4 assets present on b11982; `resolve_cuda_asset` driver-gating unchanged |
| Manifest flag gating | READY | no flags added/removed in window; `offline`, `moe_cache_mib`, `spec_draft_n_ubatch`, `-fa on\|off\|auto`, `--spec-type` family all still manifest-gated (warn-and-skip) |
| sdcpp route surface | READY | #2221 touches only `/sdcpp/v1/models/load` (not called by gateway); extra status fields tolerated |
| mistral.rs next release | WATCH ARMED | unpinned automatically? No — pinned exact-tag v0.9.3 by design; manual release-triggered action in MEMORY |
| whisper | CURRENT | b5454 ≡ v1.9.5 (0-commit compare) |
| sglang / mlx-lm / piper | CURRENT | no releases since install |

## 6. Recommended actions (ordered)

1. **`blazar engine update llamacpp`** → b11982-cuda. Rides in: qwen3-coder
   template fix, MoE CUDA fusions, M-RoPE/ctrl-vector bounds. Then rerun the
   llamacpp e2e lane (validate.py engines phase).
2. **Bench MoE fusion delta** on granite-4.0-tiny-preview Q4_K_M (the store's
   MoE GGUF): same-harness A/B b11946 vs b11982, per benchmarking rules
   (receipt + JSON/CSV + report). Expect decode-tok/s gain on expert-heavy steps.
3. **`blazar engine update sdcpp`** → master-1024. Rides in: INT8 CUDA fusions
   (bench image TTFT), iGPU memory planning, hires fix.
4. **On next mistral.rs release**: unpin (drop exact-tag v0.9.3), update, rerun
   mistralrs lane; re-diff `--help` for LFM3/Flint-era argv changes; revisit
   kv-cache reservation + paged ladder follow-up (the 8GB admission blockage may
   dissolve per #2503).
5. Keep tracked-not-adopted llama.cpp endpoints as-is (/v1/stream,
   /v1/streams/lookup, /v1/messages/count_tokens, GCP-compat) — no upstream
   movement this window; /v1/realtime already shipped natively (Blazar's own
   cascade, E2E-sealed 2026-10-27).

## 7. Utilization inventory (what Blazar already exploits)

Verified current, no regression this window: 41 llama.cpp flags incl. spec-type
family + typed ngram + `--lookup-cache-dynamic` persistence + `--offline` +
`--moe-cache-mib` + `--spec-draft-n-ubatch`; mlx `MlxTuning` full pin surface;
mistralrs extra_args escape hatch + pa-fraction flag-rename tracking;
sglang launch_server grammar + LoRA one-child hot-attach; sdcpp six-route
surface incl. upscale; Responses-API uniform emulation (kept deliberately over
llama.cpp's native /v1/responses for cross-engine contract).

## Posture decisions (2026-10-29 evening wave — live-verified on b11982)

### `POST /tts` (llama.cpp child TTS) — WATCH, not adopt

Route IS registered in b11982 (`ctx_http.post("/tts", routes.post_tts)`,
llama_server.cpp route table; `tts_ctx` initializes per-slot under
`has_cap_tts()`, server-context sources). No `--tts*` CLI flag exists —
the lane is purely capability-gated on loading a TTS-shaped model
(OuteTTS-style checkpoints). None are cataloged, and Blazar's TTS
surface is piper-backed (realtime voice lane + `audio_speech`), which
serves arbitrary speaker/voice needs today. Adopt only if a TTS GGUF
class lands in the catalog as a first-class lane; until then mounting
`/tts` through the gateway would be dead surface (H24 violation
waiting to happen).

### `--tools` / `--mcp-servers-*` (child-side MCP client) — NOT adopt

`--help` on the installed b11982 binary: every knob in this family is
marked *experimental — "do not enable in untrusted environments"*:
`--tools read_file,file_glob_search,grep_search,...` (host filesystem
access from the engine child), `--tools-runtime ssh:<target>` (remote
execution), `--mcp-servers-config/--mcp-servers-json` (child-side MCP
client definitions), `--ui-mcp-proxy` (CORS proxy), `-ag` mega-switch.
Blazar's gateway already owns MCP brokering (`[[mcp]]` config +
`/api/mcp`): routing tools through a second, child-side MCP client
would split the security control point and expose the engine host's
filesystem with no gateway-side scoping. Posture: gateway remains the
sole MCP broker; re-evaluate when upstream de-experiments the surface.

### `--tts` / `--tools` summary

| Surface | b11982 state | Blazar posture |
|---------|--------------|----------------|
| `POST /tts` | registered, `has_cap_tts()`-gated, no CLI flag | watch — adopt on catalog demand |
| `--tools` family | experimental, host-file + ssh runtime | reject for now — gateway owns MCP |
