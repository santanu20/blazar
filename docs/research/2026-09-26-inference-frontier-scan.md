---
layout: doc
title: "Inference Frontier Scan"
description: "Research note: survey of the local-inference engine landscape."
doc_kind: "Research note"
---

# Inference Frontier Research Scan — Blazar Opportunity Map

**Date:** 2026-09-26 · **Scope:** cutting-edge LLM-serving research + production practices (2024–2026) mapped to Blazar's position as a single-node, cross-platform, multi-engine inference gateway. · **Method:** web metasearch (4 batch waves, 24 queries) + direct arXiv abstract verification (27 IDs resolved via arxiv.org) + live llama.cpp README/releases fetch. No code changed.

---

## 1. Landscape — where the frontier moved

| Cluster | State of the art | Key sources |
|---|---|---|
| Serving architecture | vLLM V1 (~1.7× V0), SGLang radix prefix cache + zero-overhead CPU scheduler; trend = "inference control plane" (llm-d) decoupling routing/cache/flow-control from engines | arXiv:2609.23130 |
| Prefill/decode interference | Chunked prefill (Sarathi) now adaptive/deadline-aware; **disaggregation only pays at ~1000+ GPU scale** — chunking wins locally | arXiv:2308.16369, 2609.07883; aimastery.page analysis |
| Speculative decoding | EAGLE-3 dynamic trees in production; async batched self-spec (ASPIRE); edge-cloud spec (AceSpec); MoE spec limits quantified; **n-gram prompt-lookup (PLD+) free win for agent/RAG** | arXiv:2503.01840, 2609.17943, 2412.01447 |
| KV cache | Prefix-reuse eviction: fancy policies barely beat LRU on agentic traces; KV offload/quantization mature (NVFP4); CPU-GPU physically partitioned KV for MoE | arXiv:2609.28870, 2604.08426, 2609.14507; lmsys.org blog 2026-09-16 |
| Gateways | Joint model-routing + KV-action optimization formalized ("Unified AI Gateway"); llm-d flow control + TTFT/TPOT prediction sidecars + SLO routing; agentic per-step routing (AgentRouter) | arXiv:2609.06940, 2609.22951; llm-d DeepWiki |
| Diffusion LLMs | Mercury-class dLLMs (1000+ tok/s); adaptive/learnable parallel decoding 22–57×; IO-aware KV for dLLMs (Flash-dLLM) | arXiv:2509.25188, 2609.26796 |
| Local/consumer | Hybrid dense+MoE local scheduling studied (SlotBank, 24 GB M4 Air); MoE CPU-GPU co-execution in llama.cpp (`-cmoe`); ktransformers | SSRN 7404578; ktransformers PyPI |
| Cold start | Snapshot/restore 6.5× (460 s→70 s); weight-streaming loaders; container dissect: 64% pull / 33% model load | Parasail + dreaming.press 2026-06; NVIDIA Run:ai Model Streamer |

**Engine surface already exposed by llama-server (verified live 2026-09-26, b11200 era):** continuous batching, `-np` slots, speculative decoding incl. draft KV types (`-ctkd/-ctvd`) and **lookup-decode static/dynamic caches (`-lcs/-lcd`)**, JSON-schema/grammar constraints, rerank + embeddings + Responses routes, KV offload + K/V quant types, `--kv-unified[-per-slot]`, ctx checkpoints (32/slot), `--cache-ram`, SWA, MoE CPU offload (`-cmoe/-ncmoe`), multi-GPU split modes, mmap/lazy load modes, per-phase CPU affinity/prio, NUMA, Snapdragon NPU/Hexagon prebuilts.

---

## 2. Opportunity map — research → Blazar feature

### A. Gateway request intelligence

**A1. SLO-aware admission control + priority queue (kill the Ollama 503 class).**
Research: llm-d flow control (bounded admission, priority dispatch, per-tenant queues) + latency-prediction sidecars (TTFT/TPOT estimators); deadline-aware chunking (arXiv:2609.07883).
Blazar: per-request `priority` + optional TTFT/TPOT SLO fields in all 3 dialects; gateway-side predicted-latency model fed by existing bench/telemetry; deadline violation → reshape (smaller ubatch, chunk limit) or graceful queue-position feedback instead of 503/queue-until-OOM. Ollama's documented failure mode (queue + reject on full, per-model `OLLAMA_NUM_PARALLEL`, memory-driven stalls — docs.ollama.com FAQ) is the reference complaint set to beat publicly.

**A2. Joint routing + cache-action decision ("Unified AI Gateway", arXiv:2609.06940).**
Formalizes exactly Blazar's vantage: at request time jointly choose model, execution site (here: engine lane/replica/slot) and KV-cache action (reuse/evict/migrate) under quality/latency/cost constraints. Blazar already has routing + cache visibility + fit ledger → close the loop into one cost-model decision, exposed in routing-decision trace.

**A3. Agentic per-step routing (AgentRouter, arXiv:2609.22951; RouteLLM arXiv:2406.18665; FrugalGPT cascade arXiv:2305.05176).**
60–80% of agentic trajectory steps are solvable by smaller co-resident models. Blazar: optional cascade mode — small model answers first, escalate on confidence signal (logprob mass / self-check / grammar validity) to the bigger lane; per-session policy; huge VRAM+latency win for tool-call-heavy clients.

**A4. Hybrid response cache: exact + opt-in semantic.**
Exact prefix/prompt cache is engine-side; gateway can add (a) cross-engine exact-response cache keyed on (model, params, normalized messages), (b) opt-in semantic cache via local embedding model with threshold + namespace isolation (GPTCache pattern, github.com/ginkgo-project? → use github.com/zilliztech/GPTCache). Keep semantic OFF by default (accuracy risk), surface hit provenance in response metadata (Blazar already exposes cache-hit info — extend with `cache.kind`).

### B. Speculative decoding orchestration (Blazar differentiator)

**B1. n-gram prompt-lookup auto-engagement (PLD+, arXiv:2412.01447).**
Highest ROI/effort ratio on this whole scan: `-lcs/-lcd` lookup caches already exist in llama-server. Blazar: detect agent/RAG/summarize traffic patterns (high input-output n-gram overlap — Blazar already detects prefix-busting), auto-enable lookup decoding for those sessions, persist dynamic lookup cache per model in the store, report acceptance-rate telemetry. Zero draft-model VRAM cost.

**B2. Draft-pair catalog with closed-loop tuning (EAGLE-3 arXiv:2503.01840; Medusa arXiv:2401.10774; MoE spec limits — aimodels.fyi on Qwen3-Coder-30B+EAGLE-3).**
Blazar already "selects compatible draft candidates" → add: measured acceptance-rate per (target, draft, batch-level) from bench harness; auto-disable spec at high concurrency (the documented batch-size cliff); per-workload draft-length policy (ASPIRE arXiv:2609.17943 shows per-request optimal draft length varies widely — shape hints where engine exposes them).

**B3. Spec-decode parity across engines.** SGLang lane already has EAGLE-3/JIT; expose one Blazar-level `speculation` config translated per-engine (draft path, KV dtype for draft via `-ctkd/-ctvd`, tree size) with honest capability matrix.

### C. KV-cache economics

**C1. Cache-aware steering + honest eviction (arXiv:2609.28870).**
Finding: sophisticated eviction ≈ LRU on real agentic traces; the win is *steering*, not clever eviction. Blazar: route same-prefix traffic to the slot/replica holding the prefix (SGLang radix semantics for its lane; slot-pin for llama.cpp unified-KV mode), keep replacement simple, quantify and expose cached-token economics (OpenAI-style `prompt_tokens_details.cached_tokens` parity in all 3 dialects).

**C2. Cross-restart KV/prompt-cache tiering (Mooncake arXiv:2407.00079 pattern, single-node analog; LMCache; KV-offload eval arXiv:2604.08426).**
Mooncake's insight: KV cache is *the* scarce resource → tier it (VRAM→DRAM→SSD) independently of engine lifetime. Blazar already has session checkpoints + `--cache-ram` plumbing: extend to persistent per-model prompt-cache files (llama.cpp prompt cache save/restore), sized from the fit ledger against free RAM, restored on model reload — kills repeated prefill across daemon restarts/updates (Blazar restarts engines on update today).

**C3. Per-phase quant advisory (Disaggregated Quantization, arXiv:2609.26333).**
Prefill tolerates lower precision than decode. Engine-level today, but Blazar `fit`/profiles can recommend K/V cache types + `cache_ram_mb` per workload class and validate quality via the existing perplexity bench lane.

### D. Scheduling & memory

**D1. Adaptive chunked-prefill shaping (Sarathi arXiv:2308.16369; deadline-aware arXiv:2609.07883).**
Single-node verdict: chunking > disaggregation below datacenter scale. Blazar: per-workload ubatch/chunk presets (chat = small chunks protect ITL; long-context batch = big chunks maximize PP), adaptive on observed ITL violations from slot metrics.

**D2. Hybrid dense+MoE slot banking (SlotBank, SSRN 7404578).**
Studies exactly Blazar's regime (consumer 24 GB, dense + sparse MoE co-residency). Blazar fit ledger → "slot bank" admission: MoE lanes with `-cmoe/-ncmoe` splits sized so a dense model stays resident; per-child scratch gate (today's G2 blocker: replicas not benchable on 8 GiB — this is the design path to unblock).

**D3. Multi-GPU + NPU lanes.** Engine now ships split modes (none/layer/row/tensor) + Snapdragon Hexagon NPU prebuilts. Blazar: expose split-mode in fit/profile for multi-GPU hosts; add engine-kind coverage for Android/Win-ARM targets.

### E. Frontier lanes

**E1. Diffusion-LLM lane (watch/prototype).** Mercury-class speed claims (1100 tok/s), learnable parallel decoding 22–57× (arXiv:2509.25188), Flash-dLLM IO-aware KV (arXiv:2609.26796). No GGUF-class local engine yet — track ggml-org; first-mover gateway support would be a headline feature.

**E2. Structured-output guarantee across dialects (arXiv:2609.23742).**
Constrained decoding eliminates structural failures even in 0.6–4B models. Blazar: map `response_format`/`json_schema` (OpenAI), `format` (Ollama), tool schemas (Anthropic) → engine grammar (llama.cpp `-j`/XGrammar) uniformly; add "guarantee receipt" in diagnostics (constraint-active flag per token stream).

**E3. Test-time-compute primitives (arXiv:2509.09864 latency+token-aware TTC; arXiv:2609.14995).**
Gateway-level `best_of_n`, parallel-sampling fan-out (`n>1` across slots), budget-aware early stop (latency vs marginal quality). No local gateway ships this; natural extension of Blazar's slot/replica control.

### F. Lifecycle

**F1. Cold-start war (Parasail snapshot 6.5×; NVIDIA Run:ai streamer; Modal evidence).**
Blazar: predictive preload from session affinity (usage patterns → pre-warm likely-next model into RAM/page-cache), idle-to-RAM demotion tiers (GPU→CPU weights, instant re-promote), keep-alive priorities per model. mmap/lazy modes are already engine flags to orchestrate.

---

## 3. Priority matrix

| # | Feature | Cluster | Impact | Effort | Risk | Order |
|---|---|---|---|---|---|---|
| 1 | PLD/n-gram lookup auto-spec + persisted lookup cache | B1 | High | Low | Low | **now** |
| 2 | SLO admission + priority queue + latency predictor | A1 | High | Med | Med | **now** |
| 3 | Cached-token economics + cache-aware steering | C1 | Med-High | Med | Low | next |
| 4 | Cross-restart KV/prompt-cache tiering | C2 | High | Med | Med | next |
| 5 | Draft-pair closed-loop tuning + batch-cliff auto-off | B2 | Med-High | Med | Low | next |
| 6 | Structured-output cross-dialect guarantee | E2 | Med | Low | Low | next |
| 7 | Local cascade router (FrugalGPT-style) | A3 | High | Med-High | Med | wave 2 |
| 8 | Hybrid exact+semantic response cache (opt-in) | A4 | Med | Med | Med | wave 2 |
| 9 | Adaptive chunked-prefill shaping | D1 | Med | Med | Low | wave 2 |
| 10 | MoE+dense slot banking (fit-ledger driven) | D2 | High | High | Med | wave 2 |
| 11 | TTC primitives (best-of-N fan-out) | E3 | Med-High | Med | Low | wave 2 |
| 12 | Predictive preload / idle-to-RAM tiers | F1 | Med | Med | Low | wave 3 |
| 13 | Multi-GPU split-mode + NPU lanes | D3 | Med | Med | Low | wave 3 |
| 14 | Joint routing+KV-action cost model | A2 | High | High | Med | wave 3 |
| 15 | dLLM lane | E1 | High (if engine lands) | High | High | watch |

## 4. Anti-map — do NOT chase (single-node local)

- **Prefill/decode disaggregation across nodes** (DistServe arXiv:2401.09670, Mooncake cluster mode): wrong scale; chunked prefill wins below ~1000 GPUs. Take only the *KV-tiering* idea (C2).
- **Kubernetes-class control planes** (llm-d full stack): Blazar is one binary, one host (today); borrow flow-control/predictor *patterns* only.
- **Semantic caching default-on**: ungrounded-answer risk on a local box; ship opt-in with namespaces.
- **Fancy KV eviction policies**: evidence says ≈LRU (arXiv:2609.28870); spend the effort on steering instead.

## 5. Verified citations

arXiv (all resolved via arxiv.org 2026-09-26): 2407.00079 Mooncake · 2609.06940 Unified AI Gateway · 2609.07883 Deadline-Aware Chunking · 2609.17943 ASPIRE · 2609.23130 vLLM/llm-d survey · 2609.26333 Disaggregated Quantization · 2609.28870 Prefix-reuse eviction · 2609.24847 SPECTRA · 2604.08426 KV offloading · 2609.14507 Partitioned KV MoE · 2609.22157 PAGE · 2509.25188 Learnable Parallel dLLM · 2609.22951 AgentRouter · 2609.23742 Constrained decoding small LLMs · 2509.09864 Latency-aware TTC · 2609.14995 TTC rethinking · 2609.26796 Flash-dLLM · 2308.16369 Sarathi · 2401.09670 DistServe · 2503.01840 EAGLE-3 · 2404.14469 SnapKV · 2406.02069 PyramidKV · 2306.14048 H2O · 2406.18665 RouteLLM · 2305.05176 FrugalGPT · 2401.10774 Medusa · 2412.01447 PLD+.

Web (fetched via metasearch 2026-09-26): SlotBank preprint (SSRN 7404578) · lmsys.org NVFP4 KV blog (2026-09-16) · llm-d DeepWiki (flow control, latency prediction) · Parasail snapshotting blog (2026-06-29) · dreaming.press scale-to-zero (2026-06-27) · NVIDIA Run:ai Model Streamer (2025-10-02) · ktransformers PyPI · docs.ollama.com FAQ (queue behavior) · ggml-org/llama.cpp README+releases (live) · kvcache-ai/Mooncake GitHub · blog.lmcache.ai SageMaker post. Excluded: arXiv:2312.02997 (resolved to an unrelated paper — GPTCache cited via GitHub zilliztech/GPTCache instead).


---

## ROI × Feasibility Matrix (codebase-grounded addendum, 2026-09-26)

Re-scored against the live codebase (codegraph + rg audit), not the paper
baseline. Material correction: **six of the fifteen opportunities are already
substantially shipped** — the frontier scan underestimated Blazar's surface.

### Already-built baseline (verified in code)

| ID | Shipped surface (file:line) | Remaining delta |
|----|------------------------------|-----------------|
| A1 | PriorityQueue EDF+SLO tiers+WFQ+burn counter (queue.rs:107-136); predictive 429 early-reject on TTFT p90 warm/cold (proxy.rs:1611-1647); reserved interactive capacity MT4; adaptive slots demand-sized adoption (supervisor.rs:7295+) | class-detection polish only — effectively DONE |
| B1 | n-gram spec family ngram/map-k/k4v/mod/cache (profile.rs:4938); persisted dynamic .lcache (rule 14); eagle3/mtp/auto draft pairs + placement battery | **spec=auto → dense when no draft pair (profile.rs:5062); never falls back to n-gram** |
| A4 | SemanticCache opt-in headers, cosine, TTL+LRU, 8 integration tests | in-memory only — complete as designed |
| C1 | cache_hit_ratio correct denominator (ollama.rs:3068); scrape-summed prompt_tokens_cached_total + warm/cold TTFT; CacheBustTracker sentinel (cache_bust.rs:39) | steering (decisions), visibility done |
| E2 | gateway structured-output lint grammar/format/response_format + D6 verdict LRU (sentinel.rs:756-788) | cross-engine translation guarantee |
| F1p | idle_sleep 300s / idle_timeout 1800s / session pins 900s (config.rs) | predictive preload tier |

Also verified: /v1/responses + /v1/streams/lookup routes exist (no best-of-N);
late_chunking is embeddings-only (R1) — no prefill-chunk shaping exists (D1 = real
gap); kv cache_ram/kv_offload/ctx-checkpoint wiring present in profile.rs (C2
partial); best_of appears only in whisper.rs.

### Ranked by ROI (value ÷ effort) with feasibility gates

| Rank | Item | Value | Effort | Feasibility | Verdict |
|------|------|-------|--------|-------------|---------|
| 1 | **B1-δ auto→ngram fallback** | High: 1.5-3x decode on agent traffic, zero VRAM, no draft pull | XS (<1d): one branch at profile.rs:5062 + warning swap + pins | Certain — engine flags already emitted elsewhere | DO NOW |
| 2 | **B2 draft closed-loop + batch-cliff auto-off** | High: prevents spec pessimization under batch (ASPIRE-verified effect) | M: reuse sum_child_counter scrape infra; llama-lane first | High — spec counters already scraped | DO NEXT |
| 3 | **C1-δ cache-aware steering** | Med-High: turn existing visibility into routing decisions (eviction paper: steer > evict) | M: policy layer atop live signals | High — signals exist | DO NEXT |
| 4 | E2-δ cross-dialect structured output | High correctness win | M-L: per-engine constraint surfaces differ | Med — mistral/sglang grammar support uneven | NEXT |
| 5 | C2-δ cross-restart KV restore (ctxcp + cache_ram tiers) | High: kills cold-start on restart | L: engine-version sensitive, restore-path edge cases | Med — wiring exists, behavior risk | NEXT |
| 6 | E3 best-of-N fan-out on /v1/responses | Med: TTC primitives, route exists | M | High | WAVE 2 |
| 7 | A3 agentic cascade router | High IF quality trusted | L + eval harness | Med — needs bench receipts first (AgentRouter 60-80% routable) | WAVE 2 |
| 8 | D1 adaptive chunked-prefill shaping | Med: TTFT/TPOT balance single-node | M: -ubatch surface exists | Med | WAVE 2 |
| 9 | D2 MoE+dense slot banking | High on ≥24GB hw | L | **Low-local: NOT live-validatable on 8GiB dev box (G2 verdict)** | WAVE 2/3 |
| 10 | C3 per-phase quant advisory | Med | M: advisory = safe | High (kv ladder exists) | WAVE 3 |
| 11 | F1 predictive preload | Med | M | High | WAVE 3 |
| 12 | A2 joint routing+KV-action cost model | Med | L | Low — needs B2+C1 data flywheel first | WAVE 3 |
| 13 | D3 multi-GPU split + NPU lanes | Med, hw-gated audience | L | Low-local: no NPU/multi-GPU to validate on | WAVE 3 |
| 14 | B3 cross-engine spec parity | Med | L | Low — engines lag llama here | WATCH |
| 15 | E1 dLLM lane | — | — | Engine dep absent | WATCH |

Effort scale: XS <1d · S 1-3d · M 1-2wk · L 3wk+. Feasibility "Low-local" =
cannot be live-validated on current dev hardware, which project rules require.

### Sequence recommendation

1. **B1-δ now** (XS, certain, immediate agent-traffic win; bench receipt via
   existing harness captures delta on repeat-bench).
2. **B2 + C1-δ next** (share the metrics-scrape foundation; both feed A2 later).
3. **E2-δ + C2-δ** as the following pair (correctness + cold-start).
4. Wave 2/3 unchanged from scan, with D2/D3 explicitly gated on hardware
   availability for live validation.
