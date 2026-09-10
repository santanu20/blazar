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
- **A7 DONE (corrected)**: `context_shift = true` overlay knob (opt-in).
  VERIFIED b10833+: `--context-shift` is DEFAULT DISABLED upstream (the
  earlier "default enabled" note was b10819-era and stale); we only emit
  the flag when explicitly enabled.
- **A8 DONE**: `reasoning_format` config → `--reasoning-format`
  (none|deepseek|deepseek-legacy).
- **B1 SHIPPED 2026-09-07** (multi-instance prefix routing): per-model
  `replicas = N` overlay (1..=8) spawns parallel instances keyed `model#N`;
  the gateway hashes each chat's stable prefix (system + first user turn,
  1 KiB head) and sticky-routes per prefix — new prefixes GROW a fresh
  warm-cache replica, known prefixes reuse theirs (SGLang-radix-lite at
  the router, per-instance `--cache-reuse` pools). Live-validated on
  qwen2.5-0.5b replicas=2: distinct prefixes → 2 children, repeat turns
  172/120 ms (warm), clamp held at capacity. Original deferral note: the
  child's unified KV pool with `--cache-reuse` already dedupes prefixes
  WITHIN an instance; router-level routing needed multi-instance first.
- **B2 DEFERRED**: draft auto-pairing needs a curated (model ↔ EAGLE3/MTP
  head) dataset; inventing pairs would violate the no-fabrication rule.
- **B3 SHIPPED 2026-09-06** (SLO classes High=2s/Normal=30s/Low=120s + EDF queue + deadline overrides; `pallama_slo_deadline_exceeded_total` burn counter added 2026-09-07). Original note: SLO tiers = queue rework; the existing priority queue
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
| B1 | **Prefix-aware slot routing** ("RadixAttention-lite at the router") — SHIPPED 2026-09-07 (`replicas` overlay + prefix-hash sticky affinity; see note above) | SGLang radix cache | gateway hashes the first N tokens (system prompt), sticky-routes to the slot/instance whose `--cache-reuse 256` pool already holds it; falls back to least-loaded. Agent traffic (resend-same-system-prompt) gets warm-prefix TTFT every call | shipped |
| B2 | **Draft auto-pairing** — SHIPPED 2026-09-07 (catalog `spec_pairs` prefix-match; missing draft = hard error naming the pull cmd — `--spec-draft-hf` upstream-broken b10840, revisit on fix; accept-rate gauge `pallama_spec_accept_rate`, live-verified 0.577) | vLLM/SGLang spec artillery | catalog pairs models with draft heads; `spec=auto` emits `--spec-type/--spec-draft-model/--spec-draft-n-max` | shipped |
| B3 | **SLO admission tiers** — SHIPPED 2026-09-06 (queue deadline reorder + `pallama_slo_deadline_exceeded_total`) | production schedulers | priority queue already exists; TTFT-SLO classes reorder the queue by deadline | shipped |
| B4 | **`pallama upgrade`** | — | self-update via release-asset digest verifier (installer machinery exists) | small |
| B5 | **Model metadata intelligence** — SHIPPED 2026-09-07: `{arch}.attention.*` key-shape fix (head_count was silently None on ALL real GGUFs — headline bug), MLA `key_length`/`value_length` upper-bound KV math (A21), per-layer SWA `sliding_window` + `full_attention_interval` provable-only sizing (A22), structural GGUF lint (H4) | — | all VRAM math (fit/coresidency/cache-ram/preflight) now rides loader-truth geometry | shipped |
| B6 | **Self-healing engine (J2/J3)** — SHIPPED 2026-09-07: spawn-fail streak (>=2 models) or `--version`-probe failure → auto-rollback to previous engine tag + `EngineRolledBack` event; spawn-time VRAM re-probe teaching warn | — | crash-loops after engine updates self-recover | shipped |
| B7 | **Loading core** — SHIPPED 2026-09-08: predictive preloading (Markov transitions, `predictive_preload`, reaper pre-spawns next model, `model_preloaded` event — live-proven), auto multi-GPU bin-packing (unset `devices` + >1 GPU → max-free-card pick at spawn, ALL VRAM math scoped per-card fixing the summed-pool bug — live-proven), `adaptive_slots` (default-on, 60s sustained concurrency → in-memory `-np+1`, cap 4, `slots_auto_adopted`), lookup-cache knobs (`-lcs`/`-lcd`), per-model `slots` overlay; `--defrag-thold` deprecated upstream = cut honestly | — | cold-start killer + per-card honesty; fixed the only guard-across-await in the codebase (reap snapshot-first — runtime-freeze root cause) | shipped |

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
