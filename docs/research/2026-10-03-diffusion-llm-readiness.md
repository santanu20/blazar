# Diffusion-LLM Readiness Review — 2026-10-03

Deep dive following the upstream engine radar's highest-signal item: block-diffusion
LLMs arriving in llama.cpp (DiffusionGemma PR 24423, 190👍). Every fact below was
verified against upstream sources on 2026-10-03; verified vs inferred is marked.

## Upstream state (verified)

| Stream | State | Serving surface |
|---|---|---|
| DiffusionGemma (llama.cpp PR 24423) | open, 29 commits, 87 comments | **None over HTTP.** Ships `llama-diffusion-cli` plus `llama-diffusion-gemma-server` — a stdin/stdout logits worker (one request-file per line, canvas-row logits written back), driven by a Python block-diffusion loop. `tools/server` untouched. |
| LLADA 2.0 (llama.cpp PR 17454) | open, WIP, mergeable=False, 49 commits | **None.** Model loading (`src/models/llada2.cpp`) + arch tables + the diffusion CLI only. |
| LLADA 1 / LLADA-MoE | **merged, mainline today** | **None, and silently dangerous.** The arches live in `libllama`'s loader table (verified: `llada`, `llada-moe` present in the installed build's `libllama.so`), but `tools/server` contains zero llada/diffusion handling (verified via GitHub code search: 11 files match `llada`, none under `tools/server`). |

The paradigm problem, precisely: llama-server's generation loop is autoregressive
token-by-token decode. A block-diffusion model needs a different loop — iterate a
masked canvas over parallel refinement steps. Load one in llama-server and the model
*loads* (arch known) and then *decodes garbage while the child stays healthy*
(inferred from the zero server-side handling + the PR author deliberately shipping a
separate binary; not executed here). No crash means Blazar's capability rescue —
which keys on the `unknown model architecture: 'x'` death — never fires.

## What Blazar does about it (this review's outcome)

**D1 — route-time paradigm guard (implemented).**
`blazar_core::DIFFUSION_PARADIGM_ARCHS = ["llada", "llada-moe", "llada2", "diffusion-gemma"]`
(single source of truth in `gguf.rs`). At the supervisor's route decision — right
after `read_model_meta` succeeds — a GGUF whose arch is in the set and for which NO
installed lane advertises the arch refuses with a teaching error naming the
paradigm, the tracking PRs, and the fork-lane escape hatch. The list ENGINE column
(`routed_engine_lane`) mirrors the same check so CLI and gateway never disagree.
A fork lane advertising the arch (a diffusion-capable build, when one ships)
bypasses the guard — the same lane-capability symmetry the unknown-arch rescue
uses. Merge-day for either PR requires exactly one edit: the arch is already in the
set.

**D2 — when a servable surface lands.** Upstream's current direction is dedicated
binaries, not llama-server integration. Whichever lands first — llama-server
gaining a diffusion decode mode, or a stabilised HTTP-capable binary — Blazar adds
a capability lane for it (the `engine build --fork` machinery already covers
bring-your-own builds). The existing kvless-component teaching
(`diffusion/model-component … sdcpp`) keeps image-DiT files pointed at the right
lane; that path is unrelated to text diffusion.

**D3 — streaming tolerance checklist for the serving day.** Block-diffusion
streams break the append-only delta assumption. Audit targets, in order of
exposure:

1. SSE translation accumulators (`translate.rs` ollama/anthropic re-emission) —
   `push_str` accumulation assumes append-only content.
2. `SseThinkFilter` / `ThinkSplitter` — stateful text streaming; whole-block
   rewrites would need re-feeding semantics, not append.
3. Sentinel judges and sniffers — content-accumulation heuristics.
4. TTFT/TPOT observers — a diffusion server's first frame is a whole block;
   TPOT is meaningless (report blocks/s × block size instead).
5. Semcache / session cache keys — shape-agnostic today; verify.
6. Token accounting — completion_tokens vs emitted characters diverge when
   blocks rewrite.

Bench suites need the same shift: TTFT = time to first block; decode tok/s is not
a scalar; quality checkers that stream-parse partial output must become
block-tolerant.

## Related finding — sglang native `/generate` (radar item 6, resolved as no-op)

PR 42147 merged, but all 20 changed files live under `experimental/sgl-router/`.
Blazar spawns the sglang *server* directly (`launch_server`); the router is not in
Blazar's lane, so the endpoint is unreachable. Re-evaluate only if the server
itself gains a native `/generate`. Radar updated accordingly.

## Status refresh (2026-10-03, GH API)

- adaptive MTP PR 27210: open — pre-wire stance (wave1) unchanged.
- cache-disk issue 20697: open, no implementing PR yet.
- sglang MPS PR 36780: open. OTel PR 1479: open. RMCP PR 1531: open.
  HiCache buffer_only PR 20535: open. Idle-CPU PR 775: open.
