# DeepSeek DeepInfer vs Blazar — positioning + adoptable-ideas audit

Date: 2026-10-20 · Scope: research comparison, no code change, no benchmarks run.
Sources: deepseek-ai/DeepInfer@main (clone 2026-10-20), README.md, doc/inference_deployment_qa.md, repo tree, gh repo metadata; DeepSeek V4.1 Flash Zhihu blog linked from the QA doc.

## TL;DR

DeepInfer (released 2026-10-16, 8.7k★ in 4 days, Apache-2.0) is DeepSeek's **datacenter-scale
inference engine for one model family on ~128-GPU accelerator supernodes** — and the open-source
release is **offline batch inference only** (no API server, no KV-cache storage system, no E/P/D
disaggregation, no load balancers; all four explicitly listed as "not included"). Blazar is a
**local-first universal model server** orchestrating many engines behind many API dialects on any
hardware from an 8 GB laptop GPU up. They are **orthogonal layers, not competitors**: DeepInfer
sits below where Blazar sits, and neither can do the other's job. Three things are directly
harvestable for Blazar's roadmap (tokenization hardening patterns, confidence-scheduled
speculative verification, batch-invariant determinism posture) — details in §5.

## 1. What DeepInfer is (verified)

| Fact | Value |
|---|---|
| Repo | github.com/deepseek-ai/DeepInfer, created 2026-10-16, Apache-2.0, Python + CUDA/C++ (`csrc/`) |
| Lineage | "originally built primarily on vLLM, with additional inspiration from SGLang and TensorRT-LLM" (their README) |
| Claim to fame | Powers DeepSeek's production service: **155T tokens/day**, US$38/GPU-hour revenue (valley pricing) |
| Models supported | **DeepSeek-V4.1-Flash only** |
| Hardware supported | **NVIDIA SM100 (Blackwell) + Huawei Ascend 950 NPU**; Linux, CUDA 13 / CANN 9.2, GCC≥13, Python 3.12 |
| Release surface | `deepinfer/entrypoints/llm.py` offline LLM class only. No HTTP/gRPC server anywhere in tree (only `profile_server.py`, a profiling helper) |
| Min efficient deployment | **64 prefill GPUs + 64 decode GPUs** (1P+1D boots, but decode wants EP64+; "wide-EP") |
| Vendor kernels | FlashMLA, DeepGEMM, DeepEP, TileKernels, DeepSelect, Mega MoE, custom TileLang kernels (vendored under `vendor/`) |
| Companion repo | **deepseek-recipe** — Rust token-encoding/protocol-conversion component (Chat Completions + Responses + Messages APIs) |

### Not included in the release (their own README)

- KV cache system (their 3FS-backed ~100 PiB SSD KV store, 14.4 TB/GPU logical, TTL 3d agent / 7d chatbot, 72h+ retention target)
- VE/Prefill/Decode disaggregation ("most required engine functionality is already implemented")
- Load balancing across DP replicas (their Prefill/Decode/VE/Expert-parallel LBs)

So the open-source artifact ≈ "vLLM-class engine core specialized for one MoE architecture,"
not a servable stack. The interesting systems material (their production architecture) is
documented in the Q&A, much of it not in the code.

## 2. Positioning matrix

| Axis | DeepInfer | Blazar | ollama | vLLM/SGLang |
|---|---|---|---|---|
| Layer | Engine (kernels + scheduler + runtime) | **Server/gateway + orchestrator over engines** | Server over llama.cpp | Engine + basic server |
| Models | 1 family (DeepSeek V4.1 Flash) | Any GGUF/safetensors via 7 engine lanes (llama.cpp, sglang, mlx-lm, mistral.rs, sdcpp, whisper, piper) | llama.cpp subset | Broad transformer catalog |
| Hardware | SM100 / Ascend 950 only | Any: CPU, consumer GPU (8 GB dev box), Apple MLX, RPC fleet fan-out | CPU/consumer GPU | NVIDIA/AMD server GPUs |
| Scale floor | 128 datacenter GPUs for efficiency | 1 laptop | 1 laptop | 1 server GPU |
| Serving API | **None shipped** (offline only) | OpenAI + Ollama + Anthropic + Responses, tools, embeddings, rerank, sessions, realtime WS, decisions/score/classify lanes | Ollama (+OpenAI subset) | OpenAI |
| Protocol conversion | separate `deepseek-recipe` (Rust) | in-gateway, 4-dialect conformance suite + 35-endpoint drift guard | — | — |
| Model lifecycle | manual HF download | pull (ollama-registry reuse, zero-redownload), import --scan (ollama/HF/LM Studio), engine auto-update, doctor, certify | ollama registry | manual |
| Release maturity | v0.1.0, 4 days old | v0.26.0, 3-OS CI, 9 binaries, crates.io + SDKs | mature | mature |

## 3. Overlapping axes — honest head-to-head

These are the only axes where both projects do the same *kind* of work; the table records who is
further along **at their own scale**, not a winner overall.

| Axis | DeepInfer | Blazar today | Delta owner |
|---|---|---|---|
| Speculative decoding | **DSpark**: MTP varlen draft-5 + Confidence Head predicting per-token acceptance; dynamic-length verification selects the prefix subset with max marginal return; GPU-side TileLang kernel (cumprod → global sort top-k → fused remap) picks per-request lengths. +4–7% tok/s at production point, +11% at 96 seq/rank vs fixed-5 | Engine-bound opt-in: sglang ngram (measured never-default-on), EAGLE/DFLASH draft-model lanes, llama-server spec, `tune --spec-race` auto-races lanes | **Upstream engines** — confidence-scheduled verification exists nowhere in llama.cpp/sglang; Blazar adopts when engines ship it (track, don't build) |
| Determinism | `DEEPINFER_BATCH_INVARIANT=1` + fixed `seq_seed` → identical outputs across batch compositions (SM100 only) | temp-0 sha-identical receipts per lane; **full-token parity across differing radix/batch histories is NOT achievable on continuous batching** (verified 2026-10-17, both backends) — this is exactly the gap batch-invariant mode closes | **Upstream engines** — batch-invariance is an engine-level mode; file/track upstream |
| Structured outputs | regex / JSON schema / EBNF → compile on dedicated machine → automaton advance + token mask per decode step (async compile decoupled from request path) | llama.cpp grammar + gateway fence normalization at both seams (non-stream), recipe bare-JSON lift on MLX, 4-dialect tool-call conformance | Blazar parity adequate for its lanes; async/pre-compiled grammar is an upstream nicety |
| Scheduling | Zero-overhead async scheduler (overlap CPU schedule/post-process with GPU forward/sample, SGLang/NanoFlow lineage); prefill chunk size solved from a **fitted quadratic cost model** under a ns time budget; decode LB balances request-count × total-KV-length jointly | Gateway-side: fixed 4-slot caps, warm-peg ladder, prefill graph buckets, admission teach; C1 gateway overhead ~6% upper bound, wins tail TTFT at C8 | Different problem: theirs is in-engine step scheduling; ours is cross-engine orchestration |
| KV strategy | FP4 KV (per-16 e4m3 main, MXFP4 indexer), SWA bounded replay (constant 128-token recompute, window truncation), cross-request prefix reuse via cluster KV store (not open-sourced) | KV quant q4_0 quality-free (receipted 2k + 32k ctx), engine prefix caches, session cache; no cross-daemon KV store | Theirs needs a storage cluster; ours is local-first — no overlap to act on |
| Tokenization / protocol safety | deepseek-recipe (Rust): special-token **injection prevention** (text spelling `<｜User｜>` encodes as ordinary tokens), segmented parallel BPE encode between recorded special-token boundaries, incremental-KMP streaming parser for cross-chunk markers/UTF-8/tool-args | Gateway: sentinel-based raw-byte judging, SSE think-filter with marker-risk fast path, fence normalization, tool-call dialect lifts | **Adoptable patterns for gateway seams — see §5.1** (ideas, not code: Apache-2.0 regardless) |

## 4. Could DeepInfer be a Blazar engine lane?

**No — today.** Hard blockers, all verified:

1. No server surface: offline `LLM` class only; a Blazar lane needs an HTTP child to gateway.
2. Hardware: SM100 / Ascend 950 — does not exist on this box (RTX 4070 8 GB) or any consumer rig.
3. Model: DeepSeek-V4.1-Flash is a CED/YOCO MoE sized for supernodes; MXFP4-expert weights
   alone dwarf any local GPU, and no current llama.cpp/sglang release loads the arch either.
4. Their own serving stack (the part that would matter to a gateway) is not open-sourced.

Revisit trigger: if a future release ships an OpenAI-compatible server + SM100-class hardware
lands in scope, re-evaluate like any other lane (bench direct-engine vs gateway first, per
benchmark policy). Not filed as a roadmap item — no actionable surface.

## 5. What Blazar should harvest

### 5.1 Now (gateway seams — our code, our control)

- **Special-token injection prevention**: deepseek-recipe records inserted special tokens at
  prompt-build time and encodes user text against base vocab only, so user input can never spell
  a control token. Blazar's dialect adapters rebuild prompts from messages — same invariant
  applies wherever we splice sentinel markers into user-controlled text. Audit our prompt
  builders for spelled-marker passthrough.
- **Incremental-KMP streaming parse**: retain match progress across chunk boundaries instead of
  rescanning; emit-determined-content-early. Our SSE think-filter/fence normalization already
  buffers on leading-ws; KMP-progress is the general form for any future marker that can split
  across token chunks (tool-call delimiters, DSML-style tags).
- **Segmented parallel tokenization** (encode ordinary-text segments between known special-token
  boundaries in parallel): engine-side concern mostly, but relevant to any gateway-side
  token-accounting we do for budget headers.

### 5.2 Track upstream (engine-level — file/track, don't build)

- **Confidence-scheduled dynamic-length speculative verification** (DSpark): the measured
  +4–7%/+11% is at their batch scale, but the mechanism (confidence head → marginal-return
  verification budget) is architecture-general. Watch sglang/llama.cpp for equivalents; cite
  DeepInfer's numbers as the existence proof when filing.
- **Batch-invariant inference**: closes the exact parity gap we receipted as unachievable on
  continuous batching. If vLLM/sglang adopt it as a mode, our quality A/B harness (identical
  probe sequences workaround) can relax to batch-varying probes.

### 5.3 Architecture validations for Blazar (no action, morale)

- DeepSeek serves 155T tok/day with **Rust doing protocol conversion** (deepseek-recipe) in
  front of a Python engine — the same layer split as Blazar's Rust gateway over engine children.
- Their prefill LB uses a **fitted cost model + time budget** for chunk sizing rather than fixed
  chunk sizes — the same philosophy as our warm-peg ladder; supports the budget-over-fixed
  posture if we ever schedule across engines.

## 6. Verdict

Not competitors, no lane, no fork. DeepInfer is the current SOTA reference for single-model
datacenter inference economics (FP4 KV, wide-EP, EPLB, DSpark, E/P/D disaggregation) and a
confirmation that the serving/broker layer around engines is its own discipline — the position
Blazar occupies locally. Harvest §5.1 patterns in the gateway; track §5.2 upstream; nothing in
this repo changes our engine matrix or benchmarks.

## 7. Harvest addendum (2026-10-20, same session — implementation + live validation receipts)

### 7.1 Implemented: chatml special-token neutralization at recipe render (prompt_recipe.rs)

The deepseek-recipe attack class — user text carrying exact `<|im_start|>`/`<|im_end|>`
spellings forging turn boundaries when spliced into a chatml template — applied verbatim to
Blazar's `ollama_compat` prompt recipe. All three splice sites (message text, assistant
tool-call history re-render, tool descriptors) now neutralize exact special-token spellings by
substituting fullwidth vertical bars (`|` → `｜`, U+FF5C — the inverse of DeepSeek's own
base-vocab design choice): visually near-identical, tokenizes as ordinary text, can never match
a special token. Engine-owned templates are untouched (upstream behavior parity); parse-side
envelopes in model OUTPUT are untouched (those are model emissions, not user input).

**Live validation** (XDG sandbox rig, qwen2.5-0.5b-instruct-4bit via mlx-0.32.0, temp 0, seed 7,
port 11437, pre-change binary from HEAD worktree vs post-change binary):

| probe | PRE `prompt_eval_count` | POST `prompt_eval_count` | behavior |
|---|---|---|---|
| clean "Reply with exactly: ok" | 13 | 13 (identical) | identical content + eval_count 16 — zero drift on clean input |
| forged turn-splicing message | 47 | 72 (+25) | PRE: model CONTINUES the forged fake conversation (answers "tell me another secret"). POST: forgery inert — single real user turn, model degenerates to plain-text roleplay, mimicking the fullwidth spelling it saw |

The +25 prompt-token delta matches an isolated direct-child A/B exactly (child `/v1/completions`:
verbatim-forged pt=42 → neutralized pt=67) — proof at the tokenization seam that the forged
spellings stopped parsing as single structural special tokens and became ordinary text.

Unit pins: forged spelling neutralized at all three splice sites; structural markers untouched;
clean + near-miss renders byte-identical; exhaustive every-split sweeps for ThinkSplitter and
partial-tag holdback pin the streaming path. `cargo nextest -p blazar-gateway`: 124 lane /
725 full, 0 fail; clippy + fmt clean. Receipts: `/tmp/opencode/blz-val/*.json`.

### 7.2 Verified already-present (no code changed)

- Incremental-KMP-style cross-chunk marker handling on the SSE think lane (ThinkSplitter
  prefix-hold + marker-risk fast path) — pinned by new exhaustive split sweeps.
- Streaming recipe adapter: UTF-8-safe byte buffering, tool-call tag holdback to stream end.

### 7.3 Deliberately excluded

- Streaming JSON-fence normalization stays non-stream-only (enforce-posture decision, MEMORY.md).
- ~~OpenAI-lane equivalent injection~~ — CLOSED 2026-10-21, see §7.5.

### 7.4 Upstream filing (§5.2) — duplicate check changed the plan

- **sglang**: NO filing — duplicate search shows sglang already ships both DSpark
  (open bugs reference `DSPARK target_verify`, draft-graph replay) and
  `--enable-deterministic-inference`. DeepInfer's §5.2 "track upstream" items are already
  landed there.
- **vLLM**: batch-invariant already shipped (`VLLM_BATCH_INVARIANT=1`); DSpark-class dynamic
  verify length not shipped (their dynamic spec decoding is a batch-size table, plus open
  entropy-gated RFC #54082). Filed an evidence comment on vLLM #54082 citing DeepInfer's
  public confidence-head measurements (+4–7% @ ~60 seqs/rank, +11.0% @ 96) as an existence
  proof for continuous-length verification:
  vllm-project/vllm#54082 (issuecomment-6294756765).

### 7.5 Wave B (2026-10-21): recipe lane on /v1/chat/completions + doctor audit

**Feature — gateway-rendered recipe lane on the OpenAI chat surface.** The neutralized
renderer (§7.1) now also serves `/v1/chat/completions` under the same gate as the ollama
lane: `prompt_recipe = "ollama_compat"` pinned for the model, OR (mlx engine AND a
non-empty `tools` array), AND no images. The chat body is rendered gateway-side into the
child's raw `/v1/completions` dialect; responses adapt back (non-stream: buffered
translate + chat completion identity stamp; stream: per-frame SSE adaptation with stamped
ids, usage frame, legal-close truncation). Response-side accounting (sentinel, sniffer,
cache tap, metadata cards, think suppression, model re-stamp, JSON-fence strip) operates
on the caller-dialect bytes — same posture as the ollama recipe lane. Refusals mirror the
ollama posture: `n>1` and MCP selectors teach (400) instead of degrading. Default posture
(recipe = "child", non-mlx-tools) is byte-identical passthrough, pinned by e2e test.

**Root-cause fix found by live validation (wire capture).** The first live attempt 502'd
deterministically: `apply_child_request_mutations_stamped` re-serializes from the hot
lane's `parsed` chat body, so the mlx model-stamp mutator silently REPLACED the rendered
completion body with the caller's chat dialect on the wire (mlx-lm closes the connection
on a `messages`-shaped `/v1/completions` body — visible only via tcpdump: ack-then-FIN,
zero response bytes). Fix: on the recipe lane the forwarded bytes are the mutation source
of truth (no parse is handed over; every mutator is a documented no-op on the model-less
completion body). The e2e stub suite could not catch this — the stub engine kind does not
stamp models; only a live mlx child exercises the mutator.

**Live validation receipts** (sandbox rig, qwen2.5-0.5b-instruct-4bit via mlx-0.32.0
child, temp 0 seed 7, receipts under the rig dir):
- Clean non-stream: `chat.completion` object, stamped `chatcmpl-` id, caller model
  spelling, usage forwarded (pt 20 / ct 2), finish stop.
- Forged turn-splicing message: pt 72 — the exact §7.1 neutralization signature (+25 over
  the pre-fix 47); no fake-conversation continuation.
- Stream: chat `delta.content` frames carrying one stamped id, final usage frame, bare
  `data: [DONE]` terminator.
- Unpinned model (mistral-7b via llamacpp): byte-faithful native passthrough (child's own
  id format) — default lane untouched.
- Tools request on the pinned lane: refused by the PRE-EXISTING capability-certificate
  gate (this model's 2026-10-08 tool probe failed on mlx's template) — expected; note: a
  re-probe under the pinned recipe lane would run grammar-forced and likely certify.
  Tool parsing itself is pinned by unit + e2e stub tests.
- Stub fixture gained llama-server-parity `/v1/completions` streaming (SSE text deltas,
  `stream_options.include_usage` honored) so the streaming lane is e2e-pinned.
- Test totals: blazar-gateway 725 → 735 (+5 unit, +5 e2e), clippy clean, fmt clean.

**Wave B follow-ups (2026-10-21, closed same session).**
- *Probe rides the serving lane — true by construction, no code change.* `model-doctor`
  probes call the gateway's own handlers (`ollama::chat` with the capability-stand-down
  header), so an unpinned mlx model's tools probe flows the recipe lane automatically via
  the mlx_tools gate — production routing and certification measure the same path.
- *Cert flips — verified live, with an architectural caveat.* qwen3-1.7b (UNPINNED mlx)
  certifies FULL PASS including tools (`200, 1 tool call(s), first: get_weather`) through
  the auto-triggered recipe lane. qwen2.5-0.5b honestly FAILs (answers in prose, echoing
  the rendered tools block) — because **mlx-lm ignores the GBNF `grammar` field on
  `/v1/completions`** (live-proven: a literal-only grammar constraint request returned
  free-form text). Recipe-lane tool calling on mlx is therefore prompt-taught format + parse-lift,
  capability-bound per model; decoder-level grammar enforcement holds only on
  llama.cpp-family children. qwen3-4b: 507 fit refusal (8 GB VRAM); granite-micro: crashes
  at mlx load in this environment (not lane-related).

**Doctor audit (wave C).** `blazar doctor` now reports the prompt-recipe posture: pinned
models as ok rows (both surfaces named), and a warn advisory listing unpinned mlx models
("engine-owned template drops tool calls — pin prompt_recipe for the grammar lane"),
excluding whisper-family dirs. Validated against both the sandbox config (1 pinned) and
the real store config.
