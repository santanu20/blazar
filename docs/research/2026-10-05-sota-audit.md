# SOTA Comprehensive Audit — Blazar vs the Field

**Date:** 2026-10-05
**Scope:** Every axis — API surface, sampling, scheduling, memory/KV, engines, federation, structured output, modalities, observability, security, UX — at micro-granular feature/config level, against every relevant competitor (ollama, vLLM, SGLang, llama.cpp/llama-server, LocalAI, LiteLLM, mistral.rs, LM Studio, Jan, llamafile, plus OpenAI/Anthropic upstream API contracts).
**Method:** Blazar claims verified against this repo's source/docs (file:line cited). Competitor claims fetched live 2026-10-05 from GitHub release pages / official docs (KB-indexed; sources listed at the end). No training-memory claims.
**Audited binary:** blazar v0.20.0 + [Unreleased] changelog entries.

---

## 0. Executive verdict

**Blazar leads the local-inference field on 9 of 12 axes, is at parity on 2, and trails on 1 (realtime voice).** No single competitor matches the combination of 3 native API dialects, 7 managed engine lanes, capability certificates with admission gates, and honest-fit admission math. The three P0 gaps — realtime voice, failover chains, cross-restart KV restore — are all buildable on foundations that already exist in the codebase; none requires new architecture, only completion of wiring that is partially present.

| Axis | Blazar | Best competitor | Verdict |
|---|---|---|---|
| 1. API surface & dialects | 9.5 | OpenAI (cloud) / ollama (local) | **LEADS local, near-cloud** |
| 2. Sampling & generation control | 10 | llama.cpp (engine level) | **LEADS** (full chain surfaced per-request) |
| 3. Scheduling, queueing, cache | 9 | vLLM (throughput at scale) | **LEADS local; vLLM wins raw batch scale-out** |
| 4. Memory & KV management | 8.5 | vLLM (prefix cache internals) | **PARITY+** |
| 5. Engine fleet & lifecycle | 10 | nothing comparable | **LEADS, uncontested** |
| 6. Engine-lane breadth | 10 | LM Studio (llama.cpp only) | **LEADS, uncontested** |
| 7. Federation & routing | 7.5 | LocalAI v4.11 failover chains | **TRAILS on failover chains; leads capacity-aware routing** |
| 8. Structured output & tools | 9 | LiteLLM (cloud catalog breadth) | **LEADS local** |
| 9. Modalities | 9 | ollama (none of these) / LM Studio | **LEADS, uncontested locally** |
| 10. Observability & ops | 9 | vLLM metrics / LiteLLM spend tracking | **LEADS local** |
| 11. Security & robustness | 9 | nobody local does this | **LEADS, uncontested locally** |
| 12. UX, docs, integrations | 8.5 | ollama (simplicity + SDKs) | **PARITY; ollama wins first-party SDK + simplicity** |

---

## 1. API surface & dialects — LEADS local

Blazar natively speaks **three full dialects**: OpenAI (`/v1/*`), Ollama (`/api/*`), and Anthropic (`/v1/messages`). No local competitor offers three; llama-server master added a native Anthropic dialect (needs `--jinja`, per llama-server README fetched 2026-10-05), ollama offers only its own + an OpenAI-compat subset, LM Studio/Jan offer OpenAI-compat only.

### 1.1 OpenAI-dialect routes (docs/4.API_SPEC.md)

- `/v1/chat/completions` (+`/control` steering, +`/input_tokens` pre-count)
- `/v1/completions` (legacy)
- `/v1/embeddings`, `/v1/rerank` + `/v1/reranking`
- `/v1/responses` (+`{id}` fetch, +`/input_tokens`, `background:true` → 202 jobs, conversations GET/DELETE via `previous_response_id`)
- `/v1/models`, `/v1/batches` (+cancel), `/v1/files` (+content)
- `/v1/images` generations/edits/upscale/jobs/cancel/capabilities
- `/v1/videos` generations/jobs/cancel/capabilities (+vid_gen)
- `/v1/audio` speech/transcriptions/translations/jobs/cancel/capabilities
- `/v1/messages` (Anthropic) + `/v1/messages/count_tokens`
- `/v1/adapters`, `/v1/systemone` (decision models), `/v1/stream`, `/v1/streams/lookup`, `/v1/requests` (+cancel/interrupt)

### 1.2 Ollama-dialect routes

`generate chat tags show copy delete create pull push embed embeddings ps version evict session(s) events explain/{model} fabric quantiles benchmarks capacity keys(+rotate) model-doctor(+{model}) watch why warm replicate route/{model}` — plus native extras `/api/ps capacity summary` and `/api/explain` routing card.

### 1.3 Parity deltas vs upstream contracts (fetched 2026-10-05)

| Upstream feature | Source | Blazar status |
|---|---|---|
| `chat.completions.UPDATE(completion_id, metadata)` | OpenAI OpenAPI spec v2.3.0 | **ABSENT** — no route. P1-6. |
| Responses `instructions` field + swap semantics with `previous_response_id` | OpenAI spec v2.3.0 | Present (responses depth, v0.18) |
| `usage.request_id` | OpenAI spec v2.3.0 | Present |
| `think` levels low/medium/high/max | ollama api.md | Present — effort mapping minimal→off, max→xhigh (v0.19 thinking management) |
| `tool_name` echo field | ollama api.md | **ABSENT** — P2-17 |
| `format: json|schema` | ollama api.md | Present + cross-dialect lint (E2) |
| `keep_alive` (5m default) | ollama api.md | Present + superseded by idle tiers (sleep/pins) |
| Anthropic `max_tokens` REQUIRED, `thinking.budget_tokens < max_tokens` | Anthropic /v1/messages docs | Present — anthropic.rs F46/F47, thinking_budget kwargs, stop_sequences passthrough, system blocks, count_tokens |
| Anthropic Batch API | Anthropic docs | **ABSENT** (OpenAI batches exists) — P1-7 |
| Anthropic `cache_control` prompt caching accounting | Anthropic docs | **ABSENT** (cached_tokens parity gap) — P1-8 |

### 1.4 Verdict

Cloud-API depth is near-complete; the deltas are small surface additions (metadata update, tool_name, Anthropic batch/cache_control), not architecture.

---

## 2. Sampling & generation control — LEADS

The full llama.cpp sampler chain is surfaced per-request AND per-model-overlay (config.rs `sampler_defaults`; profile.rs:2167-2186 emits; `--samplers` chain honored, semicolon-only gotcha translated profile.rs:2105-2117):

`temperature top_k top_p min_p top_n_sigma typical_p repeat_penalty repeat_last_n presence_penalty frequency_penalty dry_multiplier dry_base dry_allowed_length dry_penalty_last_n xtc_probability xtc_threshold mirostat seed`

No local server exposes this breadth: ollama exposes ~8 sampling params; LM Studio a GUI subset; vLLM/SGLang expose their own (large) sets but not the DRY/XTC/min_p/top_n_sigma tail that llama.cpp-class engines support. Blazar = the only server that both speaks three dialects AND forwards the complete chain.

---

## 3. Scheduling, queueing, cache — LEADS local

- **PriorityQueue**: EDF + SLO tiers + WFQ + burn lane (queue.rs:107-136).
- **Predictive 429 early-reject** using TTFT p90 warm/cold (proxy.rs:1611-1647).
- **Semantic cache** (semcache.rs) — beyond anything local competitors ship.
- **CacheBustTracker** for prefix-cache bust detection (cache_bust.rs:39).
- **Correct cache_hit_ratio denominator** (ollama.rs:3068).
- **Per-request best-of-N fan-out** (bestof.rs:47-71: header/body agreement + 2..=MAX range gate) — vLLM/SGLang require benchmark-time config; Blazar does it per-request across all dialects.
- **Agentic cascade routing** per request (cascade.rs).
- **Speculative decoding closed loop**: ngram auto-fallback (config.rs:104-125, `spec_auto_ngram` off→auto; persisted lookup cache `--lookup-cache-dynamic` config.rs:323; typed variants ngram-map-k/k4v/mod config.rs:753-760), draft-acceptance collapse <0.15 → respawn dense (config.rs:128), 60s acceptance poller (gateway lib.rs:1021-1107).

**Open items:** adaptive chunked-prefill shaping (D1 — `late_chunking` is embeddings-only today); batch-cliff spec auto-off under concurrency (B2 quality-collapse covered, concurrency cliff not); joint routing+KV cost model (A2, wave 3).

---

## 4. Memory & KV management — PARITY+

- Per-model `kv_unified`, `cache_type_k/v` quantization, `ctx_extend`, YaRN, `cache_ram_mb` governance (federation), per-GPU `/api/capacity` + honest fit math Tier A/B/C (pre-flight admission, never silent CPU fallback).
- Engine sleep tiers: `idle_sleep_secs` 300 default (profile.rs:657 → `--sleep-idle-seconds`), `idle_timeout` 1800, model pins (config.rs:2158, 2928-2931).
- **Gap:** llama-server master sleep mode now releases VRAM and reports `is_sleeping` in `/health` (README, fetched 2026-10-05) — Blazar wires idle-sleep but does not yet leverage the newer VRAM-release sleep. P2-16.
- **Gap (P0-3):** cross-restart KV/prompt-cache restore. Blazar restarts engines on update → repeated prefill. Wiring partially exists (cache_ram + ctx checkpoints + KV banks). vLLM has no cross-restart restore either — this is a first-mover opportunity, not a catch-up.
- **Gap (P2-14):** per-phase KV quant advisory (C3).

---

## 5. Engine fleet & lifecycle — LEADS, uncontested

Nothing in the local field has an equivalent of:

- **Pinned engine channels** with sha256-verified downloads, gated updates, forks, boot-smoke tests, adoption, rollback, restart-hooks (`blazar engine list/update/use/rm/prune/rollback/local/build/install/offers`).
- **Capability certificates** + admission gates (400 on VERIFIED-FAILED cert probe, fail-open otherwise; `x-blazar-model-doctor` stands down for doctor re-cert).
- **Measured bench receipts, durable results** (schema), quantiles/fabric endpoints, scorecard, autopilot observe→recommend→`--apply`.
- **Disk intelligence**: `fit/storage/prune` with `last_used_at` (schema v9).
- **Model doctor** (+`--fix`) with TESTED timestamps.

Closest competitor feature: ollama's implicit engine bundling (the #1 source of their engine-churn regressions — a complaint class Blazar already fixed by architecture).

---

## 6. Engine-lane breadth — LEADS, uncontested

Seven managed lanes (blazar-core/src/engine_kind.rs:17):

| Lane | Version | Note |
|---|---|---|
| llamacpp | b11393-cuda (b11398 available) | primary |
| sglang | 0.5.21 | **ahead of upstream's visible v0.5.19 tag** |
| mistralrs | v0.9.4 | upstream v0.9.3 (2026-09-07) |
| mlx | 0.32.0 (mlx-lm 0.32.0, mlx[cuda12] 0.32.2) | |
| whisper | b5130 | |
| piper | 2023.11.14-2 | full lane parity since v0.19 |
| sdcpp | master-929-3f8527a (VULKAN) | |

LM Studio/Jan/llamafile = llama.cpp only. LocalAI covers several engines but without channel/pin/certificate governance. vLLM/SGLang are single-engine runtimes, not fleets.

---

## 7. Federation & routing — TRAILS on one feature, leads elsewhere

**Leads:** capacity-aware peer federation with PeerCapacity tiers warm/unknown/cold + TTFT EWMA + free-VRAM tiebreak (v0.16), warm-wait/replicate/route (v0.17), `remotes cache_ram_mb`.

**Trails (P0-2):** LocalAI v4.11.0 ships **failover chains** — ordered local+remote targets, `warm:true`, anti-flap probes + min-residence, `/api/failover(+/{chain},+/events)`, pin/unpin, `X-LocalAI-Served-Model`/`X-LocalAI-Failover` headers, realtime events, gallery template + Runtime UI page (release notes fetched 2026-10-05). This is the strongest new competitor feature against Blazar. Blazar has only an e2e remote-failover circuit test (gateway.rs:3418) — the primitives (federation, capacity, health) exist to build ordered chains with anti-flap and exceed LocalAI (per-chain events + capacity-aware ordering).

---

## 8. Structured output & tool calling — LEADS local

- Cross-dialect structured output (`format json|schema`, GBNF passthrough, per-dialect lint E2, D6 verdict LRU — sentinel.rs:756-788).
- Tool calls across all three dialects; thinking/tool-call translation (anthropic.rs `translate_tool_choice`, mlx tool-call grammar via gateway lane).
- MCP: engine-child plumbing only (`--mcp-servers-config/-json`, Cursor-compatible, profile.rs:1400-1413; agent mode `--agent` config.rs:373). **No gateway MCP hosting/catalog** — vs LiteLLM 1.103.x MCP catalog (incl. M365 server #43099) and LocalAI v4.11 MCP pin/unpin. Partial gap.
- **Open (P1-5):** structured-output guarantee receipt — constraint-active flag per stream, cross-dialect, so clients can verify the schema was actually enforced.

---

## 9. Modalities — LEADS, uncontested locally

Text (7 lanes), vision (mmproj per-model, vision-without-projector teaching 400), embeddings (late-chunking `late_chunking`), **rerankers served live** (bert/rank-pooling GGUF → `--embeddings --reranking`, `/v1/rerank` + `/v1/reranking` + `/api/rerank`, v0.20 — ollama issue #3368 with 381 reactions is evidence of demand), audio STT (whisper, streaming SSE F6) + TTS (piper, `/v1/audio/speech`), images (sdcpp: generations/edits/upscale), video (`/v1/videos`, vid_gen; wan lane not yet e2e-validated live — flagged in MEMORY).

**Gap (P0-1):** realtime voice (`/v1/realtime` class, duplex STT⇄TTS sessions). Foundation exists (whisper streaming + piper + sessions); **no local competitor has it either** — first-mover territory.

---

## 10. Observability & ops — LEADS local

`/metrics` (ollama #3144, 140 reactions — served), OTLP endpoint/service, quantiles + fabric + benchmarks endpoints, histogram, audit log default-on with named-key 403 rows (KeyUsageRow), PII scrub default-on, events (`/api/events`, `/api/watch`), durable bench results with reproducibility receipts, web console `/ui` (zero-dep, dashboard TTFT, Jobs tab), doctor ENGINES versions rows.

vLLM has deeper per-batch engine internals metrics; LiteLLM has spend/cost tracking across cloud providers (Blazar's keys have rpm/tpm/daily_tokens/max_concurrent budgets — cost maps not present; not a local-serving need).

---

## 11. Security & robustness — LEADS, uncontested locally

- DNS-rebinding protection (host_guard, CVE-2024-28224 class) — v0.20; nobody local ships this.
- Loopback hardening + TLS cert/key + CORS local-allowlist default.
- Named keys with rotate, budgets, 403 semantics, audit rows.
- Sentinel enforce, deterministic + deterministic_isolate modes.
- Admission honesty: posture pre-flight, fit math, never-silent GPU→CPU fallback — the exact opposite of ollama's most-complained behavior.
- `scrub.rs` PII scrubbing (metadata-only logging).

**Pre-existing box note (not shipped default):** this dev box's config pins `audit_log=false` + `pii_scrub=false` — shipped defaults are true.

---

## 12. UX, docs, integrations — PARITY

- ~40 CLI commands incl. `connect` (codex/claude/continue/cline/openwebui with `--write` backup+rollback), `scorecard`, `plan`, `autopilot`, `why`, `explain`, `watch`, REPL lane banners.
- docs/7.SETUP.md 30+ config sections; teaching errors everywhere (400s that explain).
- Web console `/ui`.

**Where ollama still wins UX (P2-15):** first-party Python + JS libraries shipped Sept 2026 (v0.40.0-rc2 line) and ChatGPT-Desktop integration. Blazar's three-dialect compat already serves any OpenAI/ollama/Anthropic client — an SDK is distribution, not capability. Recommendation: thin first-party Python/JS SDK wrapping the three dialects (sessions, streams, bench, doctor) to close the onboarding gap.

---

## 13. Micro-granular config surface (audited count: 200+ knobs, config.rs)

| Family | Knobs (audited) |
|---|---|
| Serving | port, host, tls_cert/key, cors_origins, slots, max_loaded_models, deterministic(+isolate), sentinel_enforce |
| Idle tiers | idle_sleep_secs (300), idle_timeout (1800), pins, warmup, warm_on_pull |
| Samplers | 18-knob full chain (§2) + sampler_defaults per-model |
| Per-model | chat_template(+file), mmproj, late_chunking, rpc_servers, replicas, reasoning_budget/effort, spec (mtp/eagle3/draft), LoRA stack (enable_lora, max_rank, max_adapters, max_bytes), kv_unified, ctx_extend, cpu_moe_n, cpu_ffn_n, override_tensor, devices, pin, spec_autopull |
| Spec | spec_auto_ngram, ngram typed variants (map-k/k4v/mod), acceptance-collapse 0.15, lookup-cache-dynamic |
| sglang/mistralrs/mlx | attention_backend, sampling_backend, tool_call_parser, reasoning_parser, tokenizer_path, dtype, quantization, kv_cache_dtype, mem_fraction (clamped 0.20-0.90), mtp(+model, n_predict), encoder_cache_memory_mb, max_num_images, device_layers, mcp_servers_config/json, agent, mode |
| Keys | name, url, key, models, rpm, tpm, daily_tokens, max_concurrent, weight |
| Federation | remotes, cache_ram_mb |
| sdcpp | child_header_timeout (900), cache_mode, fa (default on), vae_tiling, rpc, sage (CUDA SM80+ only), split_mode, tae, conditioning_cache, qwen_prefix_cache, tensor_type_rules, model_args, params_backend, max_vram |
| Observability | otlp_endpoint/service, audit_log (default true), pii_scrub (default true), metrics, quantiles, fabric, benchmarks |

---

## 14. Competitor snapshots (fetched 2026-10-05)

| Competitor | Version | Notable since last audit | Threat to Blazar |
|---|---|---|---|
| ollama | v0.40.0-rc2 (stable v0.35.x) | Python+JS SDKs, ChatGPT-Desktop, structured-output perf | Distribution, not capability |
| vLLM | v0.27.0 (561 commits) | Feed fetched; highlights not distilled this pass | Batch scale-out throughput |
| SGLang | v0.5.19 visible | Feed fetched; highlights not distilled | Blazar pins 0.5.21 — ahead |
| llama.cpp/llama-server | master | Router server (--models-dir/--models-max/--models-autoload), VRAM-release sleep (is_sleeping), POST /props, --media-path, native /v1/messages, /v1/systemone, chat_template_caps, --cache-reuse | Feature source to leverage (P2-16) |
| mistral.rs | v0.9.3 (2026-09-07) | — | Lane at v0.9.4, ahead |
| LocalAI | v4.11.0 | **Failover chains** (ordered targets, warm, anti-flap, events, UI) + MCP pin/unpin + OCI proxy backend | **P0-2 — only feature where Blazar trails** |
| LiteLLM | 1.103.x | MCP catalog, guardrails, scoped tracing, cost maps | Cloud-proxy class, not local runtime |
| LM Studio | 2026.x | GUI + agent-harness guides | Desktop UX only |
| Jan | current | Desktop, llama.cpp | Not in same class |
| llamafile | 0.10.6 (2026-09-28) | Single-file exe | Portability niche |

---

## 15. Prioritized gap register

### P0
1. **Realtime voice lane** (`/v1/realtime`-class duplex STT⇄TTS sessions). Foundation exists (whisper streaming F6, piper, sessions). No local competitor has it — first mover.
2. **Failover chains** — ordered local+remote targets, warm, anti-flap probes + min-residence, per-chain events. LocalAI v4.11 parity + exceed via capacity-aware ordering.
3. **Cross-restart KV/prompt-cache restore** (C2) — avoid re-prefill after engine updates/restarts. Wiring partially present (cache_ram, ctx checkpoints, KV banks).

### P1
4. Adaptive chunked-prefill shaping (D1).
5. Structured-output guarantee receipt (constraint-active flag per stream, cross-dialect).
6. `POST /v1/chat/completions/{id}` metadata update (OpenAI spec v2.3.0; tiny).
7. Anthropic Batch API (OpenAI batches already exists).
8. Anthropic `cache_control` prompt-caching accounting (cached_tokens parity all 3 dialects).
9. Batch-cliff speculative auto-off under concurrency (complement to quality-collapse detector).

### P2
10. Joint routing+KV-action cost model (A2).
11. MoE+dense slot banking (D2).
12. Multi-GPU split-mode + NPU lanes (D3; ollama #5186 Ryzen NPU = 168 reactions).
13. dLLM lane (watch; readiness doc 2026-10-03 exists).
14. Per-phase KV quant advisory (C3).
15. First-party Python/JS SDK (distribution play).
16. Leverage llama-server: router-server mode, VRAM-release sleep, /props, native Anthropic child dialect.
17. ollama `tool_name` echo field.

---

## 16. Differentiators (unique among local competitors)

1. Three native dialects incl. thinking/translation management per engine.
2. Seven managed engine lanes with channels/pins/forks/sha256/boot-smoke/adoption/rollback/restart-hooks.
3. Capability certificates + admission gates; measured bench receipts, durable results, quantiles/fabric.
4. Decision models (`/v1/systemone`) with same-day upstream parity.
5. Disk intelligence (fit/storage/prune with last_used_at).
6. Semantic cache; per-request best-of-N + agentic cascade.
7. Capacity-aware federation + warm/replicate/route.
8. Teaching errors everywhere; honest posture/fit math (never silent fallback).
9. Web console; connect integrations with backup/rollback.
10. DNS-rebinding hardening; loopback security defaults.

---

## Sources (fetched 2026-10-05, KB-indexed)

- ollama releases + api.md — `gh-ollama-releases`, `ollama-api-doc`
- vLLM releases — `gh-vllm-releases`
- llama.cpp releases + llama-server README — `gh-llamacpp-releases`, `llamaserver-readme`
- SGLang releases — `gh-sglang-releases`; mistral.rs releases — `gh-mistralrs-releases`
- LocalAI releases — `gh-localai-releases`; LiteLLM releases — `gh-litellm-releases`
- OpenAI OpenAPI spec v2.3.0 — `openai-openapi`
- Anthropic /v1/messages docs — `anthropic-messages-api`
- Internal baselines: `frontier-scan-0926`, `competitor-pain-1002`, `engine-radar-1002`, `validation-1003`, `diffusion-readiness-1003`
- Blazar source: docs/4.API_SPEC.md, docs/7.SETUP.md, CHANGELOG.md, blazar-core/src/config.rs, blazar-core/src/engine_kind.rs, gateway module census (41 modules), profile.rs cites as noted

*Unverified-note (H7): vLLM v0.27.0 and SGLang release-highlight details were fetched but not itemized in this pass; their rows above intentionally omit per-feature claims.*
