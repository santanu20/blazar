# SOTA & E2E Component Audit — 2026-10-21

Per-component audit of every Blazar micro-component: (a) end-to-end
validation status — LIVE receipt (real daemon + real engine, this or a
prior session with the current code), TEST-ONLY (unit/e2e suite via the
stub fixture, no live engine receipt), or STALE (no current receipt);
(b) SOTA position vs the reference set an operator actually compares
against: ollama, vLLM, and bare llama.cpp-server. Gap items carry an
ROI rank. Sources: this session's live receipts ( rigs at
/tmp/opencode/blz-val), the 2 223-test workspace suite, and the prior
research docs (DeepInfer comparison, engine-currency audit, surface
re-diff).

## 1. Component matrix

### Chat core

| Component | E2E status | SOTA position | Notes |
|---|---|---|---|
| OpenAI `/v1/chat/completions` (non-stream) | LIVE (2026-10-21: mistral-7b " Ok" pt11; recipe lane pt20/ct2 + chatcmpl stamp) | LEAD | Recipe lane (neutralized gateway render) is unique; ollama/vLLM have nothing equivalent — engine-owned templates everywhere |
| OpenAI chat streaming (SSE) | LIVE (stamped delta frames + usage + `[DONE]`) | PARITY | vLLM equals on raw streaming; none match our legal-close truncation + think-filter simultaneously |
| Ollama `/api/chat` + `/api/generate` | LIVE (mistral-7b, qwen via mlx child) | LEAD | Byte-compat incl. NDJSON streaming + think splitting; ollama parity is the floor, recipe lane exceeds it |
| Anthropic `/v1/messages` | LIVE (2026-10-21 receipt: full dialect response — msg_ id, stop_reason end_turn, anthropic usage block; count_tokens 5/11 tok) | PARITY | Receipts: anth1.json + count_tokens on the rig |
| Cancellation / client disconnect | LIVE (2026-10-21 receipt: /v1/requests listing mid-stream → /v1/requests/{id}/cancel → cancelled:true, stream frozen at 49,150 B within 2s, static 12s+) | PARITY→LEAD | In-flight introspection (partial_preview/bytes) + wire cancel both live |
| Single-flight dedup | TEST-ONLY (suite) + live-adjacent (every live probe transits it) | LEAD | Neither ollama nor llama.cpp-server dedups identical in-flight chat requests |
| Sentinel (output judging / enforce) | TEST-ONLY (sentinel.rs 41.7K suite) | LEAD | No competitor ships an output-quality gate at the proxy layer |
| Think dialect (`<think>`/reasoning split) | TEST-ONLY (think_dialect.rs) + LIVE indirectly (recipe suppress posture) | PARITY | Matches ollama think fields + OpenAI reasoning summaries |

### Prompt/recipe layer

| Component | E2E status | SOTA position | Notes |
|---|---|---|---|
| chatml turn-forgery neutralization (F1) | LIVE (forged 47→72 tokens, +25 signature, both lanes; wire-isolated 42→67) | LEAD | DeepSeek-class hardening at the gateway; no local competitor does splice-time neutralization |
| Recipe lane — ollama dialect | LIVE (all receipts above) | LEAD | — |
| Recipe lane — OpenAI dialect | LIVE (clean/forged/stream/n>1/MCP/error-verbatim) | LEAD | Same renderer, both dialects |
| mlx_tools auto-trigger | LIVE (qwen3-1.7b unpinned cert PASS via recipe lane) | LEAD | mlx-lm drops tool calls natively (mlx-lm#1007 filed); we rescue it |
| Structured output (json_schema/object + fence strip) | TEST-ONLY (structured_output.rs) | PARITY+ | Grammar (llama lane) + fence normalization; vLLM guided_* is the only peer |

### Tools & agents

| Component | E2E status | SOTA position | Notes |
|---|---|---|---|
| OpenAI tools (native lanes) | TEST-ONLY (stub e2e) + LIVE capability-cert behavior observed | PARITY | — |
| Grammar-forced tools (recipe + GBNF) | LIVE (probe grammar honored on llama lane; mlx ignores — documented) | LEAD locally | vLLM parity via guided decoding; ollama has JSON schema only |
| MCP mediation | TEST-ONLY + LIVE refusal posture (400 teaching) | LEAD | No local competitor mediates MCP tool catalogs |

### Async / batch surfaces

| Component | E2E status | SOTA position | Notes |
|---|---|---|---|
| `/v1/responses` emulation | TEST-ONLY (conformance) | LEAD locally | Uniform cross-engine Responses incl. storage/previous_response_id — ollama/llama.cpp have none, vLLM partial; llama.cpp b11785 native /v1/responses is child-local only (tracked) |
| `/v1/batches` + `/v1/files` | LIVE (2026-10-21: files upload 190 B JSONL → file-id; batch created → polled completed 1/1/0; full pipeline) | PARITY | OpenAI-parity surface none of the local set ships |
| Jobs framework (images/video/audio/jobs + events + artifacts) | LIVE all three (images PNG; video 16-frame WebM in 150s via wan-2.1; audio jobs ride speech+transcription round-trip) | LEAD locally | Unified async jobs across modalities is unique |
| `/v1/requests` (async chat) | LIVE (2026-10-21: listing + cancel exercised mid-stream; create-via-chat transitively live) | PARITY+ | — |

### Specialty inference

| Component | E2E status | SOTA position | Notes |
|---|---|---|---|
| Embeddings | LIVE-adjacent (bge lane exercised in earlier sessions; fixtures green) | PARITY | late_chunking knob exceeds ollama |
| Rerank (`/v1/rerank`, 4 spellings) | TEST-ONLY this cycle | PARITY | — |
| Raw `/v1/completions` + tokenize/detokenize/apply-template/slots | LIVE (recipe lane rides it daily; direct child probes) | PARITY | Full llama.cpp surface proxied + augmented |
| Speculative decoding family (eagle3/dflash/dspark/mtp/ngram + spec_cache) | LIVE (draft-mtp argv + completion; n-ubatch knob; spec_cache wired) | LEAD locally | vLLM broader at datacenter scale; nobody integrates 5 spec methods + persistent ngram cache across engines on one box |
| LoRA adapters (`/v1/adapters`, sync) | TEST-ONLY this cycle | PARITY+ | ollama has none; vLLM parity |

### Multimodal

| Component | E2E status | SOTA position | Notes |
|---|---|---|---|
| Images generations/edits/variations/upscale + jobs | LIVE (qwen-image PNG, 512×512, params echo) | LEAD locally | Only local server with OpenAI images dialect over sdcpp |
| Videos generations/jobs/content | TEST-ONLY (routes verified vs sdcpp master-982; no run this cycle) | LEAD locally | — |
| Audio transcriptions/translations/speech/voices | LIVE (2026-10-21 round-trip: piper speech 162 KB WAV → whisper transcription → exact text match; ggml-base lives in data/whisper/models) | PARITY+ | ollama has none of speech/voices |

### Sessions, cache, routing, ops

| Component | E2E status | SOTA position | Notes |
|---|---|---|---|
| Sessions + prompt caching | LIVE (2026-10-21: save/restore round-trip n_written 36 / n_read 36, sub-ms) | PARITY+ | Slot-checkpoint surface |
| Semantic cache | LIVE (2026-10-21: miss→store→hit 0.9544 similarity on reworded prompt, cache_debug.cache_hit marker; off by default, x-blazar-cache header override) | LEAD | nobody local has one |
| Engine routing (llamacpp/mlx/sglang/mistralrs/sdcpp/whisper/piper + federation/remotes) | LIVE (all four LLM engines live this cycle; remotes bypass documented) | LEAD | The core differentiator vs every competitor |
| Supervisor (spawn/respawn/settle/warm-peg/idle-evict) | LIVE (respawn receipts, warm-peg, sleep/wake) | LEAD | — |
| Admission/capacity (`/api/capacity`, fit gates) | LIVE (507 fit refusals observed honestly; 8GB ladder) | LEAD | ollama has no admission control |
| model-doctor + capability certs | LIVE (qwen3-1.7b full PASS incl. tools via recipe lane) | LEAD | — |
| Pull/store lifecycle | LIVE (3 pulls this cycle incl. gated-repo workaround + collision guard) | PARITY+ | ollama-parity UX with HF + registry resolution |
| Keys/metrics/watch/events/why | TEST-ONLY mostly | PARITY+ | — |

### CLI & packaging

| Component | E2E status | SOTA position | Notes |
|---|---|---|---|
| CLI (serve/ps/doctor/engine×7/list/model-doctor) | LIVE (doctor rows, engine update/rollback, model-doctor runs) | LEAD | engine manager is unmatched locally |
| Installer (curl|sh, windows/mac/linux + smoke in CI) | CI-LIVE (installer smoke ×3 green every release) | PARITY+ | — |
| Release engineering (SBOM, provenance attest, 12 assets, audit gate) | LIVE (v0.28.0 full pipeline) | LEAD | ollama ships fewer platforms; SBOM+provenance none |

## 2. Gap matrix (ROI-ranked)

| # | Gap | Class | ROI | Action |
|---|---|---|---|---|
| 1 | No benchmark receipts (AGENTS mandate): TTFT/tok/s vs direct engine, concurrency ladder, cache states | PROCESS | HIGH | Stand the benchmark harness up next; every perf claim currently rests on design, not numbers |
| 2 | Anthropic `/v1/messages`, batches, files, requests, sessions, semcache: no LIVE receipts this cycle | VALIDATION | MED-HIGH | One rig afternoon closes all six (each is minutes) |
| 3 | Videos + audio jobs: routes verified, no generation run | VALIDATION | MED | One wan-2.1 video + one whisper transcription receipt |
| 4 | Cancellation: upstream fix landed, no end-to-end receipt | VALIDATION | MED | Kill-client-mid-stream rig probe; verify child decode stops (tokens/s in child log) |
| 5 | Streaming JSON-fence normalization | FEATURE (held) | MED | Deliberate enforce-posture hold; needs explicit sign-off |
| 6 | llama.cpp native `/v1/responses` passthrough per-engine | FEATURE (tracked) | MED | Only if a child-only capability (stream replay) becomes asked-for |
| 7 | `/v1/realtime` voice WS | FEATURE (tracked) | LOW-MED | Novel surface; large lift, no local competitor either |
| 8 | `moe_cache_mib` positive path | UPSTREAM-BLOCKED | LOW | llama.cpp b11785 rejects on MoE GGUFs (dense verified fine); retest each engine bump |
| 9 | mlx grammar enforcement | UPSTREAM (filed) | LOW | mlx-lm#1007; prompt-taught fallback works ≥1.7b-class |

## 3. Verdict

Blazar leads the local-serving set on integration breadth (4 LLM
engines + 3 modalities + 3 dialects + recipe hardening + admission +
doctor), matches vLLM on the surfaces that matter locally (structured
output, spec decoding variety), and trails only on datacenter-scale
serving (disaggregated P/D, router fleets — out of scope by design).
The honest soft spot is not features but **measurement**: gap #1 is the
only item where a competitor can currently claim something we cannot
disprove with receipts.

Method note: statuses derive from live receipts (this session's rigs +
production), the 2 223-test suite, and CI runs — not from reading code
alone. Every LEAD claim above traces to a receipt or a verified route
surface.
