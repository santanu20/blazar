# Pallama perf roadmap — researched 2026-09-05

What the fast engines do that pallama *can* adopt, ranked by local-lane ROI.
Sources: vendored upstream llama.cpp b10816 (`references/llama.cpp-master`,
every flag below grep-verified present), vLLM 0.26/V1 design docs + release
notes (2026), SGLang docs + 2026-Q1 roadmap, chatforest SGLang review
(RadixAttention up to 6.4× over vLLM on prefix-heavy workloads).

## Where vLLM/SGLang genuinely win on a LOCAL box (conceded)

| Workload | Why they win | Sourced magnitude |
|---|---|---|
| Long-prompt prefill (RAG, code-agent contexts) | chunked prefill + fused attention kernels; llama.cpp has NO prefill/decode interleaving (verified b10816) | large, workload-dependent |
| Prefix-repeated traffic (agent harness resends system prompt every call) | RadixAttention shares KV across requests | up to 6.4× throughput on prefix-heavy (SGLang review) |
| 3-10 parallel streams (agent tools firing concurrently) | higher continuous-batching ceiling | grows with concurrency |
| Batch decode on Ada+ GPUs | FP8 weights/KV on tensor cores + Marlin kernels; GGUF lane has no FP8 path | ~1.5-2× at batch |

## Status (2026-09-05, as built)

- **A1 DONE**: `spec = "ngram"` config value → `--spec-type ngram-simple`
  (opt-in; `auto` still requires a draft pair — measured adoption only).
- **A2 DONE**: `cpu_range = "lo-hi"` config → `--cpu-range` (validated,
  `lscpu -e` documented for discovery; no topology guessing in code).
- **A3 DONE**: `ubatch` TuningOverride → `--ubatch-size`; NOT a tune-grid
  axis (argmax objective is tg; ubatch moves pp — tripling bench time for a
  knob the objective can't see would be weightless). Set via extra_args or
  adopted programmatically.
- **A4 VERIFIED NO-OP**: `--kv-offload` default ENABLED upstream (b10816
  arg.cpp:2418) — pallama already gets it; nothing to emit.
- **A5 DONE**: `poll = N` config (1..=100) → `--poll`.
- **A6 VERIFIED NO-OP**: `--cache-idle-slots` default enabled (with
  --cache-ram); `--defrag-thold` is DEPRECATED upstream — never emit.
- **A7 VERIFIED NO-OP**: `--context-shift` default enabled upstream.
- **A8 DONE**: `reasoning_format` config → `--reasoning-format`
  (none|deepseek|deepseek-legacy).
- **B1 DEFERRED (verified reasoning)**: the child's unified KV pool with
  `--cache-reuse` already dedupes prefixes WITHIN an instance regardless of
  slot; router-level prefix routing only adds value with multiple instances
  of the SAME model — non-default locally. Build when multi-instance routing
  exists, not before.
- **B2 DEFERRED**: draft auto-pairing needs a curated (model ↔ EAGLE3/MTP
  head) dataset; inventing pairs would violate the no-fabrication rule.
- **B3 DEFERRED**: SLO tiers = queue rework; the existing priority queue
  covers the local lane.
- **B4 DONE**: `pallama upgrade [--version] [--dry-run]` — same asset
  naming + API-digest verification as engine updates; atomic self-replace
  (unix); e2e-tested against a fake release server incl. tamper rejection.
| A3 | **prefill batch tuning** | `--batch-size`, `--ubatch-size` | bigger ubatch = faster long-prompt prefill up to VRAM | tune --search grid already exists; add b/ub axes + adopt |
| A4 | **KV offload** | `--kv-offload`, `--kv-unified` | spill KV to RAM instead of 400ing when ctx > VRAM | profile rule: enable when fit says ctx exceeds VRAM; pairs with cache_ram clamp |
| A5 | **busy-poll decode** | `--poll`, `--poll-batch` | removes poll-sleep latency per step (TTFT) | config knob `poll = false` default, surfaced in `tune` |
| A6 | **KV hygiene** | `--defrag-thold`, `--cache-idle-slots` | defrag + idle-slot cache reuse for long sessions | verify defaults in manifest probe; tune defrag for REPL profiles |
| A7 | **context shift** | `--context-shift` / `--no-context-shift` | rolling window instead of restart-at-overflow | overlay default for `run` REPL profile |
| A8 | **reasoning controls** | `--reasoning-format/-budget/-effort/-preserve` | API-level reasoning params (thinking budgets) | gateway passthrough: OpenAI `reasoning_effort` → child flags |

## Tier B — orchestrator-level architecture (the pallama-native wins)

| # | Feature | Inspired by | Design | Effort |
|---|---|---|---|---|
| B1 | **Prefix-aware slot routing** ("RadixAttention-lite at the router") | SGLang radix cache | gateway hashes the first N tokens (system prompt), sticky-routes to the slot/instance whose `--cache-reuse 256` pool already holds it; falls back to least-loaded. Agent traffic (resend-same-system-prompt) gets warm-prefix TTFT every call | medium — routing table in gateway, no child changes |
| B2 | **Draft auto-pairing** | vLLM/SGLang spec artillery | catalog pairs models with HF EAGLE3/MTP heads (qwen3-0.6B ↔ 8B class); `pull` suggests `--draft`, `spec=auto` adopts after bench proves net-positive | medium |
| B3 | **SLO admission tiers** | production schedulers | priority queue already exists; add TTFT-SLO classes (interactive vs batch) that reorder the queue by deadline not just priority | medium |
| B4 | **`pallama upgrade`** | — | self-update via release-asset digest verifier (installer machinery exists) | small |

## Tier C — not implementable in pallama's lane (honesty)

Tensor parallelism, PagedAttention/paged KV kernels, piecewise CUDA graphs
(vLLM 0.26 dual-mode), FP8 weight paths, FA4, PD disaggregation (single-box
irrelevant). These live in kernel/engine land; pallama's contract is
upstream-first, so they arrive (or don't) with llama.cpp. Fixing them would
mean forking the engine — the one thing pallama exists to avoid.

## Recommended order

A1 (ngram spec) → A2 (core pinning) → B1 (prefix routing) → A3 (tune axes)
→ A4 (kv-offload) → B2 (draft pairing). A1+A2 are days of work with the
largest single-stream local gains; B1 is the architectural differentiator no
other local stack has.
