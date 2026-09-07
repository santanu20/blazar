# Pallama Frontier-100 — the SOTA feature matrix

Ranked backlog to keep pallama frontier-level. Statuses: ✅ shipped · 🟡 partial
(extend) · ❌ new. Lane = where it lives (GW gateway / CORE / SUP supervisor /
EM engine-manifest / UP upstream-polled / REF refused-by-design). Evidence keys:
[ACL26]=Awesome-KV-Cache survey (ACL 2026, fetched 2026-09-06) · [HICACHE]=SGLang
HiCache design docs · [LMC]=LMCache · [MOON]=Mooncake/KVCacheD · [DYN]=NVIDIA
Dynamo tier-aware router · [DWIKI]=llama.cpp DeepWiki memory chapter (indexed
b96806, Sep 2 2026) · [ARG]=upstream arg.cpp b10819 (vendored, grep-verified) ·
[V027]=vLLM v0.27 notes · [SGL]=SGLang release notes · [O33]=ollama 0.33.3 ·
[AUD]=docs/frontier-audit.md.

Ground rule (why-not-just-implement-PagedAttention): attention KERNELS live in
llama-server's C++; pallama's zero-fork contract means kernel-class features
arrive via upstream releases (engine manifest auto-exposes them). ollama, as a
fork, is in the same boat with worse lag. Everything above the kernel — routing,
retention policy at request level, accounting, persistence — is ours. Upstream
b10819 has NO paged/radix KV (grep: zero hits); its prefix sharing = unified KV
buffer + K-shift `--cache-reuse` [ARG][DWIKI]. vLLM/SGLang own PagedAttention /
RadixAttention in-engine [V027][SGL].

## A. KV cache & memory management (22)

| # | Feature | Status | Lane | Effort | Evidence |
|---|---|---|---|---|---|
| A1 | Surface `--kv-unified` / `--no-kv-unified` (single shared KV buffer; upstream default when slots auto) | ❌ | CORE compiler | S | [ARG]1726-1730 |
| A2 | Surface `--kv-unified-per-slot N` | ❌ | CORE | S | [ARG]1652 |
| A3 | Surface `--swa-full` + `--ctx-checkpoints N` | ❌ | CORE | S | [ARG]1692-1700 |
| A4 | Surface `--no-kv-offload`, `--load-mode`/direct-io knobs | ❌ | CORE | S | [ARG]876-884, 2417 |
| A5 | Emit `--kv-unified` automatically when slots>1 + engine supports | ❌ | CORE | S | [ARG]+DWIKI unified default |
| A6 | KV quant ladder (auto q8→q4 by capacity math) | ✅ | CORE | — | have (profile.rs) |
| A7 | `tune --search` axis: cache-reuse {0,128,256,512} measured | ❌ | SUP | S | HiCache hit-ratio tuning analog [HICACHE] |
| A8 | **Prefix-hash sticky routing** — RadixAttention-lite at the gateway (B1): hash first N tokens, route to the instance whose cache holds it | ✅ | GW | — | shipped 2026-09-07 (`replicas` overlay + prefix affinity; live-validated sticky warm 172/120 ms) |
| A9 | Cache-hit tokens per request surfaced (usage.pallama_cache_hit_tokens from child /slots) | ❌ | GW | S | [HICACHE] hit metrics |
| A10 | Session KV checkpoints save/restore/erase | ✅ | SUP | — | have (34.6/14.5 ms measured) |
| A11 | Auto-checkpoint on idle (save slot state without user asking) | ❌ | SUP | S | extends A10; [LMC] write-back idea |
| A12 | Checkpoint write policy knob (on-idle / every-turn write-through) | ❌ | SUP | S | [HICACHE] write_back/write_through analog |
| A13 | Hot-prefix pinning: never-evict list for known system prompts | ❌ | GW | M | [DYN] hot-tier concept |
| A14 | Post-load warmup replay: re-issue last-N hot prompts after load | ❌ | GW | M | TTFT meta-cache analog [O33] |
| A15 | KV budget planner: multi-model co-residency math incl unified-KV sizing | ✅ | CORE | S | 2026-09-07: kv_est_bytes in Profile + supervisor co-residency check downgrades candidate KV to q8_0 pre-spawn |
| A16 | Adaptive cache-ram clamp by measured hit rate | ✅ | SUP | S | 2026-09-07: child /metrics poller -> EWMA -> CacheHint -> 20/30/40% clamp; pallama_prefix_cache_hit_rate gauge |
| A17 | L3 disk KV offload | UP | UP | — | upstream lane ([V027] multi-tier; [MOON] SSD offload exist elsewhere) |
| A18 | Prefetch-policy semantics on session restore (best-effort/wait/timeout) | ❌ | GW | S | [HICACHE] prefetch policies |
| A19 | Eviction-policy config (saliency/H2O-class) | UP | UP | — | [ACL26] eviction family — kernel-side |
| A20 | KIVI-class asymmetric 2-bit KV | UP | UP | — | [ACL26] KIVI; engine q4/q8 today |
| A21 | MLA/DSA attention-type awareness in model metadata → profile decisions | ❌ | CORE | S | [DWIKI] MLA in llama-kv-cache |
| A22 | SWA window-size awareness in ctx/VRAM math | ❌ | CORE | S | [DWIKI] ISWA dual-cache |

## B. Scheduling & batching (10)

| # | Feature | Status | Lane | Effort | Evidence |
|---|---|---|---|---|---|
| B1 | Live slots auto-tune (`tune --slots`) | ✅ | SUP | — | shipped today |
| B2 | Extend slots candidates {1..8} + workload profiles (agent vs batch) | 🟡 | SUP | S | extends B1 |
| B3 | SLO admission tiers (interactive vs batch deadlines reorder queue) | ❌ | GW | M | perf-roadmap B3 |
| B4 | Priority queue with header override | ✅ | GW | — | have |
| B5 | Weighted fair queuing between keys (share, not just order) | ❌ | GW | M | LiteLLM-class fairness |
| B6 | Long-prefill pacing (protect short-request TTFT while big prompts run) | ❌ | GW | M | [SGL] PrefillDelayer #24768 |
| B7 | Adaptive spec throttle by load (batch-size-aware steps) | ❌ | CORE | M | [SGL] #24055/#25940 |
| B8 | Identical-prompt single-flight coalescing | ❌ | GW | M | dedup; LiteLLM router has |
| B9 | Queue-depth-driven slots resize (restart-safe) | ❌ | SUP | M | extends B1/B2 |
| B10 | num_ctx header restart | ✅ | GW | — | have (133 ms measured) |

## C. Routing & multi-instance (8)

| # | Feature | Status | Lane | Effort | Evidence |
|---|---|---|---|---|---|
| C1 | `[[remotes]]` prefix routing (vLLM/MLX/another pallama) | ✅ | GW | — | shipped today |
| C2 | Remote health-aware failover (mark down, retry alternate remote) | ❌ | GW | S | [DYN] router concepts |
| C3 | Least-inflight load balancing across duplicate remotes | ❌ | GW | S | standard LB |
| C4 | Cache-aware remote routing (route to the remote holding the prefix) | ❌ | GW | M | [DYN] tier-aware KV routing |
| C5 | Per-model device pinning override | ✅ | CORE | S | 2026-09-07: model_overrides devices (replaces global) |
| C6 | Per-model rpc_servers override | ❌ | CORE | S | global exists |
| C7 | Federated ps/why across remotes | ❌ | GW | S | extends C1 probe |
| C8 | Shadow/canary model compare (local vs remote via sentinel diff) | ❌ | GW | M | novel; uses sentinel |

## D. Protocol & API surface (10)

| # | Feature | Status | Lane | Effort | Evidence |
|---|---|---|---|---|---|
| D1 | Responses registry: previous_response_id chaining + store | ✅ | GW | — | shipped today |
| D2 | GET /v1/responses/{id} retrieval (stored responses addressable) | ❌ | GW | S | OpenAI surface completion |
| D3 | ollama lane embeddings/rerank translate parity | ✅ | GW | S | 2026-09-07: /api/embed (new-style) + /api/rerank lanes, engine 501 passthrough teaching |
| D4 | Batch API (/v1/batch: async jobs, results table) | ❌ | GW | M | OpenAI batch shape |
| D5 | Strict-mode function calling (strict:true schema compile + validate) | ❌ | GW | M | OpenAI Responses strict-by-default (fetched) |
| D6 | GBNF/JSON-schema grammar cache per model+schema | ❌ | GW | M | compile cost amortization |
| D7 | Anthropic translate when engine lacks /v1/messages | ❌ | GW | M | llamactl parity |
| D8 | Capability discovery endpoint (/.well-known/pallama) | ❌ | GW | S | novel; clients introspect |
| D9 | Cache-hit tokens in response usage (pallama_cache_hit_tokens) | ❌ | GW | S | pairs A9 |
| D10 | Byte-faithful proxy for all OpenAI-family routes | ✅ | GW | — | have (zero-tax) |

## E. Security & multi-tenancy (8)

| # | Feature | Status | Lane | Effort | Evidence |
|---|---|---|---|---|---|
| E1 | Virtual keys: scope + rpm/tpm/daily budgets + usage | ✅ | GW | — | shipped today |
| E2 | Key rotation (regenerate secret, keep usage history) | ❌ | GW | S | lifecycle |
| E3 | Wildcard model scopes ("qwen3*", "vllm:*") | ❌ | GW | S | extends scope check |
| E4 | Audit log (key, model, tokens, trace → jsonl) | ❌ | GW | S | enterprise table stakes |
| E5 | Auto self-signed dev cert (tls on zero-config for LAN) | ❌ | GW | S | DX |
| E6 | Request body size limits | ❌ | GW | S | hardening |
| E7 | PII scrub opt-in for why/logs | ✅ | GW | S | 2026-09-07: pii_scrub config; hand-scanned email/secret/IPv4 scrubber on why+watch output |
| E8 | Unix-socket bind option | ❌ | GW | S | tighter-than-loopback |

## F. Observability (8)

| # | Feature | Status | Lane | Effort | Evidence |
|---|---|---|---|---|---|
| F1 | OTLP trace export (bounded, default-off) | ✅ | GW | — | shipped today |
| F2 | Per-key usage gauges in /metrics | ❌ | GW | S | pairs E1 |
| F3 | why/watch with filters (key/model/code) | 🟡 | GW | S | watch exists |
| F4 | p999 TTFT/TPOT + SLO burn alerts | ✅ | GW | S | 2026-09-07: p50/p99/p999 gauges in /metrics + pallama_slo_deadline_exceeded_total burn counter |
| F5 | Slot-level cache-hit scrape (child /slots → hit tokens) | ❌ | GW | S | pairs A9 |
| F6 | Bench history store + regression detection on engine update | ❌ | SUP | M | [V027] CI-class rigor |
| F7 | Engine-update bench gate (auto-rollback on t/s regression) | ❌ | SUP | M | builds on F6 |
| F8 | doctor: remotes + keys + quota health checks | ❌ | CORE | S | extends doctor |

## G. Speculative decoding (6)

| # | Feature | Status | Lane | Effort | Evidence |
|---|---|---|---|---|---|
| G1 | ngram + draft-model spec, persistent spec cache | ✅ | SUP | — | have |
| G2 | Draft auto-pairing from catalog (EAGLE3/MTP heads) | ❌ | CORE | M | perf-roadmap B2 |
| G3 | Spec accept-rate metrics in /metrics + why | ❌ | GW | S | [SGL] spec observability |
| G4 | v0.4.0 spec-type exposure audit (DSpark/DFlash class flags post-update) | ❌ | EM | S | [ARG] spec flags; AUD checklist |
| G5 | Synthetic spec acceptance options surfacing | ❌ | EM | S | llama.cpp #27711 (v0.4.0) |
| G6 | Multi-draft trees (topk>1) | UP | UP | — | [SGL] Spec V2 trees; kernel-side |

## H. Model & engine management (8)

| # | Feature | Status | Lane | Effort | Evidence |
|---|---|---|---|---|---|
| H1 | `pallama quantize` (engine's own llama-quantize) | ✅ | SUP | — | shipped today |
| H2 | Quant advisor: best quant for VRAM from catalog (fit calc) | ❌ | CORE | S | DX; pairs fit cmd |
| H3 | imatrix quantization (llama-imatrix: calibrate + importance-matrix quants) | ❌ | SUP | M | builds on H1; upstream tool in same tarball class |
| H4 | GGUF lint on import: detect known-bad quantizer variants (the qwen3.5 rejection class) | ❌ | CORE | S | real incident in MEMORY |
| H5 | Engine side-by-side + rollback + digest-verified update | ✅ | SUP | — | have |
| H6 | Per-model engine pin (compat matrix: model X pinned to tag Y) | ❌ | SUP | S | multi-engine value |
| H7 | Register local builds (PALLAMA_ENGINE_PATH) | ✅ | SUP | — | have |
| H8 | whisper.cpp managed engine lane (install/bench/model pull) | ✅ | SUP | — | shipped 2026-09-07: `whisper --install/--pull/--list` + lazy local server on `/v1/audio/transcriptions` (live e2e 200 in 0.62 s; bench cut — non-goal) |

## I. Quantization & perf (8)

| # | Feature | Status | Lane | Effort | Evidence |
|---|---|---|---|---|---|
| I1 | KV quant ladder + per-model override | ✅ | CORE | — | have |
| I2 | tune --search grid argmax (bench-proven) | ✅ | SUP | — | have |
| I3 | tune axes: cache-reuse + ubatch (pairs A7) | ❌ | SUP | S | extends I2 |
| I4 | --override-tensor presets (moe-cpu-offload et al.) | ✅ | CORE | S | 2026-09-07: tensor_preset knob, validated vocabulary |
| I5 | VRAM preflight on ctx raise (num_ctx header path) | ❌ | GW | S | capacity math exists |
| I6 | Vulkan per-driver quirk table (flags by driver) | ❌ | EM | S | bench evidence class |
| I7 | llama-bench regression CI on every engine update | ❌ | SUP | M | pairs F6/F7 |
| I8 | FP8/weight-quant lanes | UP | UP | — | GGUF lane has none; kernel-side |

## J. Reliability & ops (6)

| # | Feature | Status | Lane | Effort | Evidence |
|---|---|---|---|---|---|
| J1 | Crash circuit breaker | ✅ | SUP | — | have |
| J2 | Escalating: repeated crashes → auto engine rollback (use prev tag) | ❌ | SUP | M | extends J1 |
| J3 | Daemon-side OOM preflight at load (MemAvailable < model+headroom → named error) | ❌ | GW | S | validate.py logic ported |
| J4 | Config hot-reload (SIGHUP: keys/remotes/cors live) | ❌ | GW | M | registries already live |
| J5 | Wedged-child auto-evict from sentinel stall detection | ❌ | SUP | S | sentinel exists |
| J6 | `pallama snapshot` (store+config backup/restore) | ❌ | CORE | S | ops table stakes |

## K. Client & ecosystem UX (6)

| # | Feature | Status | Lane | Effort | Evidence |
|---|---|---|---|---|---|
| K1 | `pallama launch` (env wiring for agent CLIs) | ✅ | CLI | — | shipped today |
| K2 | Launcher profiles (presets per CLI: claude/dsh/codex shapes) | ❌ | CLI | S | extends K1 |
| K3 | Client-compat pins CI (Claude Code/Codex/Continue/ollama-native) | ✅ | tests | — | shipped today |
| K4 | More client pins (Open WebUI, Jan, LM Studio shapes) | ❌ | tests | S | extends K3 |
| K5 | Shell completions (bash/zsh/fish/pwsh via clap_complete) | ❌ | CLI | S | missing basics |
| K6 | doctor --fix (auto-repair PATH, stale engines, config migrations) | ❌ | CORE | S | DX closer |

## Count & shape

- **Total: 100** — ✅ 31 shipped · 🟡 2 partial · ❌ 61 new · UP 6 (upstream-polled, not ours to build) · REF 0 new refusals — wire-everything wave 2026-09-07: +36 launch knobs (spec-draft placement, ngram typed tuning + tune --ngram, reasoning budget/effort, vision, YaRN set, sched extras, warmup/repack/keep, override-kv, control vectors), A15/A16/C5/D3/E7/F4/I4 closed
- Effort mix of the 62 new: S 44 · M 18 · L 0 — **deliberately no large items**: everything big is either shipped or correctly parked in UP (kernel land).
- Top-10 by ROI: A8 (prefix sticky routing — the one 6.4×-class win we can own), A1-A5 (unified-KV surfacing — free perf, engine already ships it), B3 (SLO tiers), F6+F7 (bench-gated engine updates), G2 (draft auto-pairing), A13 (hot-prefix pinning), E4 (audit log), D5 (strict tools), K5 (completions).

## The honest boundary (restated)

PagedAttention, RadixAttention-in-engine, kernel eviction (H2O/KIVI), FP8/NVFP4,
tensor parallelism, PD-disaggregation transfers (Mooncake/NIXL) = engine/kernel
land [ACL26][V027][SGL][MOON]. They arrive with upstream releases; our engine
manifest auto-exposes new flags and this document's UP rows track them. What we
uniquely own — and what this list maximizes — is everything AROUND the kernel:
routing, retention-at-request-level, accounting, persistence, protocol breadth,
and reliability. ollama cannot follow us here without shedding their fork.
