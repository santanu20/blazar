# Pallama feature research — 2026-09-05

What can be built next, grounded in (a) code-verified pallama state, (b) the
vendored upstream llama.cpp reference (b10817/b10819-class,
`references/llama.cpp-master`), (c) first-party docs + web sources cited per
section. Nothing below is asserted from training memory; every upstream flag
or route named here was grep-verified in the vendored tree.

## 1. KV cache quantization — status: HAVE (auto), gaps remain

**Present (code-verified):**

- `core/src/profile.rs` rule 6: when the model is GPU-resident and estimated
  KV (`2*blocks*kv_heads*head_dim*ctx*2` bytes, f16) + weights would exceed
  0.9× VRAM → emits `--cache-type-k q8_0 --cache-type-v q8_0`. GGUF-field
  gaps skip the rule with named warnings; `head_count_kv` falls back to
  `head_count` (matches upstream default for non-GQA models).
- Manual override: `TuningOverrides::kv_quant` (adopted via `tune`).
- `runtime/src/bench.rs` tune grids already bench `-ctk/-ctv` alternatives,
  so measured adoption is wired.

**Gaps (upstream headroom, verified in `tools/server/README.md`):**

| Gap | Evidence | Work |
|---|---|---|
| Fixed at q8_0 — K/V accept `f32,f16,bf16,q8_0,q4_0,q4_1,iq4_nl,q5_0,q5_1` | server README flag table | config knob `cache_type = "auto\|q8_0\|q4_0\|…"`; rule 6 ladder q8_0 → q4_0 at >0.95× VRAM |
| Quantized V requires Flash Attention | `src/llama-context.cpp`: "quantized V cache was requested, but this requires Flash Attention" | non-issue: pallama always emits `--flash-attn` (manifest-gated) |
| Draft-model KV types unexposed | `--spec-draft-type-k/-v` exist | fold into spec-draft profile when `spec = "auto"` |
| `fit` doesn't show KV-quant effect on ctx headroom | `pallama fit` rows compute f16 KV | one extra row: "with q8_0 KV: ctx NNNNN" |

## 2. Unforwarded upstream API surfaces — pure gateway wins

`gateway/src/lib.rs:96-124` routes 5 OpenAI paths + ollama API. The vendored
`tools/server/server.cpp` serves more; the byte-stream proxy needs only new
routes + model-affinity semantics:

| Route (upstream, verified) | What it is | Value |
|---|---|---|
| `POST /v1/responses` | OpenAI **Responses API** (the current-gen OpenAI surface; Codex-CLI-class clients speak it) | biggest client-compat win on the list |
| `POST /v1/audio/transcriptions` | audio-in via the MTMD stack (mp3/wav/flac through `minin`) — omni-class models + mmproj | STT lane without a new engine |
| `POST /infill` | FIM/infill code completion (upstream ships `--fim-qwen-*` preset flags) | code-completion clients |
| `POST /v1/chat/completions/control` | control-vector steering API (`--control-vector*` flags exist) | steering/preset feature |
| `GET/POST/DEL /v1/stream(s)` | named persistent streams + `lookup` (stream sharing/dedup) — GET/POST/DELETE handlers verified in server.cpp | long-running shared generations |
| WebUI static routes | **child serves a built-in Web UI, default ENABLED** (`--ui` default on, `--ui-config` JSON) — currently unreachable through the gateway | `pallama ui` = zero-build chat GUI |
| `--slot-save-path` + slot save/restore | KV-cache checkpoint files per slot | session resume (see §4) |

## 3. Upstream capabilities pallama can orchestrate (config knobs + small logic)

All flags verified in `common/arg.cpp` of the vendored tree:

| Capability | Flags | Pallama shape |
|---|---|---|
| **Router mode** (multi-model in ONE llama-server) | run without `--model`; `--models-dir`, `--models-preset` (INI), `--models-max` (default 4), `--models-autoload`, preset `load-on-startup` | strategic: pallama *generates* the preset INI from its model store; child does internal LRU. Complements (not replaces) the supervisor: pallama keeps fit/tune/engine-mgmt/ollama-API. Sources: aixfunda.substack.com router-mode article; freshlab.es guide; openwebui docs |
| **Persistent n-gram spec cache** | `--lookup-cache-static` / `--lookup-cache-dynamic` (server-supported, verified `LLAMA_EXAMPLE_SERVER`) | per-model cache file under `~/.local/share/pallama/speccache/` → speculation stays warm across restarts; pairs with `spec = "ngram"` |
| **YaRN context extension** | `--rope-scaling {none,linear,yarn}`, `--rope-scale`, `--yarn-orig-ctx` + beta flags | `ctx_extend = 2.0` overlay knob with warning (quality tradeoff, not free ctx) |
| **Fine-grained MoE offload** | `--override-tensor` (pattern→device), `--n-cpu-moe N` (count, finer than bool), `--override-tensor-draft` | per-model overlay presets: "experts on CPU, rest GPU" beats blunt `--cpu-moe` on hybrid boxes |
| **Agent mode / MCP** | `--agent` (built-in tools + CORS proxy), `--tools read_file,grep_search,exec_shell_command,…`, `--tools-runtime docker:/podman:/ssh:` sandboxing, `--mcp-servers-config/json` (Cursor-compatible) | opt-in `agent = true` config with LOUD warnings (tools execute commands); tools-runtime sandboxing documented as the safe path |
| **Slot KV checkpoints** | `--slot-save-path` + `/slots/{id}?action=save\|restore` | `pallama session save/resume` (see §4) |
| **Vision tuning** | `--image-max-tokens`, `--mtmd-batch-max-tokens`, `--media-path` (local media via file://) | overlay knobs; media-path for local-file vision UX |
| **SWA control** | `--swa-full` (full-size SWA cache; PR 13194) | overlay knob for SWA models (Gemma-2/3 class) |
| **Fit floor** | `--fit-ctx N` (min ctx `--fit` may shrink to) | config passthrough `fit_min_ctx` |
| **KV pool mode** | `--kv-unified` / `--kv-unified-per-slot` | profile knob when multi-instance prefix reuse matters |
| **Embedding-dedicated instances** | `--embeddings` (restrict server to embedding use) | `pallama pull nomic-embed` → auto profile: embeddings-only, tiny ctx, never spec/FA | 
| **Child auth hardening** | `--api-key` (multiple), `--api-key-file` | defense-in-depth: gateway injects a random per-spawn child key even on loopback |
| **Slot routing similarity** | `--slot-prompt-similarity` | tune knob for child-side prefix-aware slot picking at `-np > 1` (the upstream cousin of gateway B1) |
| **Draft-side knobs** | `--cpu-moe-draft`, `--cpu-range-draft`, `--gpu-layers-draft`, `--device-draft`, `--spec-draft-n-min/p-min` | full draft-model profile control when `spec = "auto"` |

## 4. Pallama-native gateway features (borrowed, differentiated)

| Feature | Inspired by | Design | Effort |
|---|---|---|---|
| **Session checkpoints** (`pallama session save/resume/list`) | upstream slot-save (§3) + vLLM sleep-mode ergonomics | gateway route → child `/slots/{id}?action=save`; files under `sessions/<model>/`; ollama-API surface: extend `/api/chat` options? | medium — novel, no local stack ships it |
| **Prefix-aware routing** | SGLang radix (6.4× sourced, perf-roadmap) + upstream `--slot-prompt-similarity` | gateway hashes first-N prompt tokens → sticky instance/slot routing; only pays off multi-instance same-model (B1 deferred correctly) | medium |
| **Request dedup/coalescing** | upstream `/v1/streams/lookup` (exists!) | forward the route + document; gateway-level identical-request fanout is YAGNI while upstream provides it | small (forward) |
| **Web dashboard** | llamactl (React UI, verified from its README) | two options: (a) forward the child WebUI (zero build, per-model), (b) pallama's own minimal dashboard on the gateway (ps/queue/metrics — data already in `/metrics` + `/api/ps`) | small (a) → medium (b) |
| **Management API** | llamactl management endpoints + keys | gateway `/mgmt/*` routes (pull/evict/engine) behind management keys → enables remote dashboards; CLI keeps store-direct fast path | medium |
| **Per-key quotas** | existing `api_keys` | token-budget counters per key in gateway | medium |
| **Engine-arch precheck at pull** | the qwen3.5 rejection incident (upstream b10817/b10819 reject its quantizer metadata — MEMORY 2026-09-05) | probe GGUF arch string vs engine build support *before* download completes; `pull` warns + suggests alternative quant | small-medium, prevents a real failure class |
| **`pallama doctor`** | LM Studio diagnostics UX | GPU/driver/VRAM/port/conflict/config sanity in one command; reuses `probe.rs` + `fit` machinery | small |

## 5. New lanes

| Lane | Basis | Verdict |
|---|---|---|
| **STT via omni models** | `/v1/audio/transcriptions` is live upstream (audio in = MTMD, mp3/wav/flac verified in README) | forward the route (§2); `search`/`pull` tag audio-capable repos by mmproj presence |
| **whisper.cpp as sibling engine** | engine-manager already downloads+verifies GGML-org release tarballs; whisper.cpp releases are the same shape | real but a new lane: `pallama whisper file.mp3` — decide by user demand |
| **Router-mode integration** | §3 row 1 | recommended experiment: generate preset INI from store, one child serving N small models — complements supervisor for the many-small-models case |
| **vLLM/SGLang engine** | prior research (frontier-comparison.md) + `Engine` trait ready (`engine_impl.rs:56-75`, `Arc<dyn Engine>` in supervisor.rs:77) | medium-large; opt-in only (Python, CUDA/ROCm, no native Windows) |
| **MLX engine (mac)** | llamactl ships MLX backend (verified README) | same trait question as vLLM; Apple-only lane |
| **Docker-wrapped engines** | llamactl pattern + upstream `--tools-runtime docker:` | keep out of core (breaks zero-Docker brand); document `engine register-local` for containerized binaries |

## 6. Status update 2026-09-05 (post-implementation)

Items 1, 3, 4, 5, 6, 7, 8 from the table below shipped in the beat-ollama CLI wave (this session):

- Forwarded routes (item 1): /v1/responses (+/responses), /v1/audio/transcriptions (+/audio/transcriptions), /infill, /v1/chat/completions/control, /v1/chat/completions/input_tokens, /v1/responses/input_tokens, /v1/messages/count_tokens, /tokenize, /detokenize, /apply-template — all byte-stream proxied with model-affinity routing; multipart model extraction for audio (boundary-aware, header-window-scoped, unit-tested against decoy bytes). Streams routes (/v1/stream(s)) intentionally NOT forwarded: per-instance conv_id sessions carry no model field — needs a gateway-level conv-id registry; revisit if demand appears.
- KV-quant ladder (item 3): rule 6 is now capacity math — none -> q8_0 (KV/2) -> q4_0 (KV/4) against the 0.9x-VRAM budget — plus cache_type config (any upstream type; f16-class forces off) and a CTX@KV_Q8 column in pallama fit.
- Persistent spec caches (item 4): spec_cache = true default -> --lookup-cache-dynamic <data>/speccache/<model>.lcache when spec = ngram (manifest-gated, warn-skip).
- Session checkpoints (item 7): --slot-save-path <data>/sessions/<model>/ (per-model dir, supervisor pre-creates), gateway POST /api/session (save/restore/erase, filename-sanitized) + GET /api/session?model=, CLI pallama session save|restore|rm|list. Round-trip e2e-tested vs the stub incl. traversal rejection + missing-checkpoint 404.
- pallama doctor (item 6): config / port (incl. ollama-11434 conflict class) / engine / hardware / disk (statvfs) / model-health checks in one table.
- YaRN + MoE + agent knobs (item 8): ctx_extend (yarn + warning), cpu_moe_n (--n-cpu-moe), override_tensor (--override-tensor list, overlay-replaces-global), agent = true (opt-in --agent with exec-tools warning) — all validated, overlay-capable, env-overridable.
- Engine-arch precheck (item 5): NOT shipped — upstream exposes no per-arch support probe to orchestrate honestly (a hardcoded arch list would violate the dynamic/no-fabrication rules); doctor surfaces GGUF parse failures instead.

Not shipped (deferred with reasons): item 2 (pallama ui) — user directive: CLI-only, no UI. Items 10-12 (MCP gating beyond the agent switch, management API, vLLM/MLX engines) — separate waves; see section 5.

## 7. Second wave (same day): router mode, prefix affinity, pull precheck

- Router mode SHIPPED (item 9): router = true spawns ONE llama-server with --models-preset — an INI pallama generates from the store (per-model sections carry each model's compiled profile: ctx, cache-type, MoE knobs, per-model sessions dir; [*] carries server knobs). Upstream does autoload-on-request + LRU at --models-max. Verified LIVE on b10819: single front process + per-model children, chat routed by body model, ps translated from the child's /models, pallama stop MODEL forwards the engine unload, unknown models fail fast against the store, per-request num_ctx becomes an explicit 400 naming the model_overrides fix. One design correction from the live run: NO front-level --slot-save-path — a CLI value cascades to model children and overrides their per-model INI paths.
- Prefix affinity SHIPPED (honest scope): slot_prompt_similarity config -> --slot-prompt-similarity (upstream default 0.1). The originally-specced gateway-level slot steering is IMPOSSIBLE today: upstream has no id_slot request field (schema grep) — documented instead of built on a guess.
- Pull precheck SHIPPED: post-download GGUF header validation — ModelPulled event + CLI + /api/pull NDJSON carry a warning naming the failure, the pallama rm hint, and the sibling quants of the same repo (the qwen3.5-9b load-refusal class). import already failed fast.
- Router preset generation skips unreadable/uncompilable models with warnings (one bad model cannot take the router down — live-verified: qwen3.5-9b skipped, qwen2.5 served).

## 6. Ranked by ROI (pre-implementation ranking, kept for history)

| # | Feature | Why first | Size |
|---|---|---|---|
| 1 | Forward `/v1/responses`, `/infill`, `/v1/audio/transcriptions`, `/v1/streams`, control, webui routes | pure proxy-route additions; biggest client-compat surface per LOC; byte-stream path already proven zero-tax | small |
| 2 | `pallama ui` (child WebUI through the gateway) | default-enabled upstream, currently dead weight; instant GUI | small |
| 3 | KV-quant ladder + `cache_type` knob + fit row | rule 6 exists; q4_0 headroom verified; fits the "solved complaint" brand (VRAM pressure) | small |
| 4 | Persistent lookup caches for `spec = "ngram"` | two flags + path management; warm speculation after restart | small |
| 5 | Engine-arch precheck at pull | prevents the exact qwen3.5-class failure users hit | small-medium |
| 6 | `pallama doctor` | cheap trust-builder; machinery exists | small |
| 7 | Session save/resume on slot-save-path | novel; no competitor ships it | medium |
| 8 | YaRN `ctx_extend` + override-tensor MoE presets + vision knobs | overlay surface already validates unknown keys as errors | small-medium |
| 9 | Router-mode experiment (generated preset INI) | strategic architecture option; upstream LRU for free | medium |
| 10 | Agent/MCP gating (`agent = true` + tools-runtime docs) | powerful but security-sensitive; ship after docs | medium |
| 11 | Management API + dashboard | unlocks remote ops | medium-large |
| 12 | vLLM / MLX engine adapters | trait-ready; big surface (store format, profile, bench) | large |

## 8. Research addendum 2026-09-09 — research-grade features (post-frontier-100)

Deep-research pass over current llama.cpp tip (b10853), SGLang, vLLM 0.26,
and the 2026 semantic-cache/embedding/spec-decode literature. Every upstream
capability claim below was grep-verified in the vendored tree (noted per row)
or fetched fresh this session (artifacts under /tmp/opencode/*.json). These
rows are NEW — none appear in frontier-100.md (grep-verified: no
late-chunking / semantic-cache / logprob rows exist there).

| # | Feature | Evidence | Lane | Size | ROI |
|---|---|---|---|---|---|
| R1 | ✅ SHIPPED 2026-09-09 — `latechunk.rs` (piece-walk → chunk spans → mean-pool → L2); opt-in `[model_overrides."<model>"] late_chunking = true` + `late_chunking_max_tokens` (default 8192); profile forces `--embeddings --pooling none` + ubatch floor (`LATE_CHUNK_UBATCH_DEFAULT` 2048, explicit `ubatch_size` wins; upstream `!slot.can_split()` caps one doc at one micro-batch — full-ctx ubatch OOMs tight GPUs, live-pinned); matrix fetched via legacy `/embedding` route (`content: [ids]` — OAI `/v1/embeddings` rejects pooling=none on stable b10809); 3 input shapes (string / string-array / int-array pretokenized) across /api/embed, /api/embeddings, /v1/embeddings; 413 guards (cap / ctx-band / ubatch); TPM charged | jina.ai + arXiv 2409.04701v3 (Jul 2025); vendored `server-context.cpp` supports LLAMA_POOLING_TYPE_NONE (grep-verified); live-proven on sandbox 11499 (b10809, qwen2.5-0.5b: 2-chunk cos-sim 0.915, 896-dim, L2=1.0; 3202-tok doc → clean 413) | GW | M | **highest** — no local stack ships it; pure gateway math on upstream capability |
| R2 | **Logprob confidence scoring in sentinel** ✅ SHIPPED 2026-09-09 — `Accum` aggregates `choices[].logprobs.content[].logprob` (mean/min/tokens) → `SentinelRecord.logprob_*` → `pallama why` prints `conf=mean/min(Ntok)`; ollama-lane clients request via `logprobs`+`top_logprobs` (translated 1:1) | extends the semantic-reliability moat; zero upstream changes; display-only by design (no new Codes/thresholds) | GW | S | high — unique observability, cheap |
| R3 | ✅ **SHIPPED 2026-09-09** — session-pinned eviction: `x-pallama-session: <name>` header pins the addressed model (canonical name) for `session_keep_secs` (default 900, 0=off); ONE axum middleware (innermost layer) covers every lane; reaper `reap_once` skips idle-eviction for pinned models (sleep-mark still applies), `victim_key` demotes pinned to last-resort, `evict_model` releases pins (force wins), `keep_alive=0` loses to a live pin; `POST /api/session {"action":"close","session"}` releases (idempotent), `GET /api/sessions` lists live pins, `pallama_sessions_live` gauge. Granularity = model (orchestrator layer; slot-save bank already covers KV). Live-proven 11499: pin held model 62s past a 30s idle timeout, close→evict within 15s, keep_alive=0 deferred to pin, gauge 1→0 | SGLang #29173 unified session radix cache (opt-in there); pairs with our PrefixKey routing + slot-save sessions | GW | M | high — protects agent workloads from cache churn |
| R4 | ✅ SHIPPED 2026-09-09 — **Opt-in semantic cache**: `[semantic_cache]` config (enabled/model/ttl_secs=600/threshold=0.90/max_entries=256) + per-request headers `x-pallama-cache: on\|off`, `-ttl`, `-threshold`; ollama lane `/api/chat` non-stream only; exact-filter (lane+model+API-key) + cosine best-match; responses carry `cache_debug{cache_hit,hit_type,similarity,cache_id}` + `x-pallama-cache` header; embed via /tokenize → legacy /embedding (mean-pool/L2); embed failure = bypass (never fails request); malformed overrides = 400; metrics `pallama_semantic_cache_{hits,misses,stores,embed_failures}_total` + entries gauge. LIVE 11499: identical → hit sim 1.000 (44ms), paraphrase → hit 0.956, different → miss+store, off → bypass, stream → bypass; structured-output requests (grammar / any non-null `format`) bypass in BOTH directions — lookup and store (live-pinned post-R9: a malformed-grammar request was served a cached 200 instead of the child's 400; fixed same session, pinned by integration test) | Bifrost gateway-native plugin shape; 2026 consensus: opt-in, threshold-gated, never default (correctness risk) | GW | M | ✅ done (v1 lane cut: OpenAI byte-proxy lane documented as follow-up) |
| R5 | ✅ **SHIPPED 2026-09-09** — perplexity-gated quantize: `pallama quantize --verify [--max-degradation P]` (default 10%) runs engine-bundled llama-perplexity on base + output over a fixed diverse probe corpus (`quantize.rs` `VERIFY_PROBE_CORPUS` ~8KB one-pass natural text — a cycled corpus drives PPL→1.0 and hides damage, live-pinned); failing output is DELETED and never registered (`quantized output removed (...): perplexity gate FAILED: PPL a -> b (+x% degradation, gate y%)`); `--allow-requantize` passthrough (upstream flag, opt-in — quantizing an already-quantized source is upstream-disabled by default) makes --verify exercisable on local quants. Live-proven sandbox XDG: Q8_0 pass (+0.3% vs 10% gate), Q2_K fail (+28.7% vs 2% gate → file gone, no store row, RC=1) | vLLM-class accuracy discipline (fp32 lm_head lesson, 0.26.0); tool already in our tarballs | CORE | S-M | high — trust in quantize output, differentiator vs ollama |
| R6 | **Cache-hit tokens in usage** ✅ SHIPPED 2026-09-09 — `prompt_eval_cached_count` on ollama-lane non-stream + stream final chunk; /v1 chat STREAMS get `stream_options.include_usage=true` injected (additive, never overrides) so usage reaches clients; gateway `CacheObs` classifies every completed response warm/cold | frontier A9/D9; vLLM 0.26 `num_cache_creation_tokens` analog | GW | S | high — agents use it for cost/prefill decisions |
| R7 | **KV offload metrics gauges → reshaped as prompt-cache observability** ✅ SHIPPED 2026-09-09 (gateway-feasible 3/4) — `/metrics` now emits: `pallama_cache_hit_ratio` (authoritative, scrape-summed child counters), `pallama_prompt_cached_tokens_observed_total` + `pallama_prompt_tokens_observed_total` + `pallama_ttft_unclassified_total` (gateway-observed), `pallama_ttft_warm_seconds`/`pallama_ttft_cold_seconds` histograms (the lookup-delay analog). Offload byte gauges remain upstream-lane | vLLM 0.26 offloading metrics maturity | GW | S | medium — completes observability story |
| R8 | **Hybrid-linear KV awareness** ✅ SHIPPED 2026-09-09 — GGUF `recurrent_layers` bool-array parsing + arch classification (`AttentionClass` Full/HybridLinear/Recurrent mirroring upstream `llm_arch_is_recurrent`/`llm_arch_is_hybrid` lists); `kv_f16_bytes` counts ONLY full-attention layers when the recurrent/full split is provable (explicit array trunk-prefix, or interval fallback `(i+1)%interval!=0` for qwen35/qwen35moe/qwen3next/qwen4exp/minimax-01 — vendored models/*.cpp semantics); pure-recurrent archs (mamba/rwkv*) estimate Some(0) (constant SSM state rides the 512MiB headroom); unprovable hybrid (e.g. kimi-k3 without layer map) stays conservative full-layer upper bound + profile WARNING naming the missing metadata; per-layer sliding windows honored inside the trunk (recurrent-aware token sum). Live-proven 11499: real Qwen3.5-9B GGUF (carries `full_attention_interval=4`) → `pallama coreside` KV@CTX **128M @4096** vs 512M pre-R8 (8/32 full layers), all full-attention rows byte-identical, spawn argv unchanged (no KV-ladder surprise), no spurious warnings | llama.cpp b10853 (Sep 8 2026) recurrent-state rollback arrives via engine releases (b10819-class tree lacks kimi-k3 in `llm_arch_supports_rs_rollback` — catalog-ready now); SGLang day-0 KDA-aware prefix caching | EM | S | ✅ done — KV math + readiness; per-arch ctx=1M preflight scaling arrives when such GGUFs land locally |
| R9 | **Structured-output pre-validation + raw GBNF passthrough** ✅ SHIPPED 2026-09-09 (reshaped) — sentinel `structured_output_error` top-level lint wired pre-admission in `/api/chat`: garbage ollama `format` (unknown string/number/bool/array) 400s with a teaching error (was silently ignored — intentional fail-fast change), schema-object shape sanity (`properties`/`required`/`type` top-level only, no recursion — exotic-valid never rejected), `format`(schema)+`grammar` mutual exclusion mirrors child `json_schema+grammar` rejection, client-sent `response_format` type-set mirrors child (json_object/json_schema/empty; server-common.cpp:1168-1180), `grammar` type+cap checks (256KiB `MAX_GRAMMAR_BYTES`, empty=absent); translate passes raw GBNF `grammar` 1:1 to the child (skips per-request schema→grammar conversion, grammar string wins upstream). Malformed GBNF itself still fails gracefully at the child (400 task error, engine alive — vLLM-0.26 parity confirmed live). Live-proven 11499: all 4 lint paths 400 pre-admission, GBNF `root ::= "yes" \| "no"` constrains real model output to exactly `yes`/`no`, malformed GBNF → child 400 passthrough + next request fine, /v1 byte-proxy unchanged. Gateway-side schema→GBNF converter cache = documented L-effort follow-up (1268-line converter, language-equivalence fidelity risk) | vLLM 0.26: grammar compile failure no longer crashes engine; regex-compile timeout | GW | S | ✅ done — pre-validation + passthrough; converter-cache follow-up documented |
| R10 | ❌ **REJECTED — OUT OF SCOPE (user mandate 2026-09-09)**: gateway context compaction (history folding/summarization of oldest turns) is an **agent-harness feature, not a model-server feature**. Pallama mutates conversation content NEVER — the server's contract is transport/scheduling/caching/quantization/observability/admission; conversation-content policy (compaction, memory, summarization) belongs to the agent harness layer. The server's correct answer to an over-ctx prompt is the existing fail-fast preflight 400 with a teaching error (already shipped, K1) — the agent decides how to shorten its own history. Cancelled mid-plan, zero code written | context-editing research line stays relevant — for HARNESS projects, not Pallama | N/A | N/A | closed — scope boundary recorded in MEMORY.md 2026-09-09 |

**Sequencing recommendation:** R6 + R2 (small, immediate) → R1 (headline) →
R3 → R5 → R4 (opt-in, after R1 gives the embedding lane) → R7/R8/R9 as
fillers → R10 as its own wave.

### 8.1 Feasibility audit 2026-09-09 (code-verified against Pallama + vendored b10819-class tree)

| # | Verdict | Load-bearing evidence (file:line, all verified this session) | Adjustment |
|---|---|---|---|
| R1 | ✅ FEASIBLE (M) | `--pooling {none,mean,cls,last,rank}` arg.cpp:2315; per-token matrix response server-context.cpp:2168-2191 (pooling=none → `res->embedding` = unnormalized per-TOKEN vectors; pooling=mean → normalized single + break); `/tokenize` route already forwarded; profile compiles argv per model → `pooling = "none"` override slots in | normalize AFTER span-pooling; fit interplay (ctx must cover full doc) |
| R2 | ✅ FEASIBLE (S-M) | `logprobs`/`n_probs` alias server-schema.cpp:180; emission both shapes server-task.cpp:376-434; sentinel already parses `Bytes`/`Value` streams (sentinel.rs:300-304) | decide injection policy (only-if-requested vs transparent inject) |
| R3 | ✅ SHIPPED 2026-09-09 (GW, M) | eviction ladder reap_once→evict supervisor.rs:1700-1713; `evict()` = single choke point w/ bank_save supervisor.rs:1552-1561; sessions today = stateless file ops (ollama.rs:1356-1438); PrefixKey heat tracking exists | shipped as header-pinned model registry (SessionRegistry in pallama-runtime, middleware touch, reaper consult, demote-not-exclude victim ordering) |
| R4 | ✅ SHIPPED (M) | AppState.http + ensure_with_admission machinery reusable for internal embed calls; body-rewrite precedent = UsageSniffer wrap (keys.rs:578-608); cosine sim hand-rolled, zero new deps | opt-in only; embedding model = local engine instance |
| R5 | ✅ SHIPPED 2026-09-09 (S) — engine bundle ALREADY ships `llama-perplexity` + `llama-imatrix` (verified: ~/.local/share/pallama/engines/b10809/llama-b10809/); find_quantize_bin precedent quantize.rs:84-101 | pure CLI orchestration: run ppl, report delta, no download |
| R6 | ✅ TRIVIAL (S) | upstream ALREADY emits `prompt_tokens_details.cached_tokens`/`input_tokens_details.cached_tokens` (server-task.cpp:370,592,703,725) + `prompt_tokens_cached_total` metric; byte-proxied /v1 routes pass it through untouched | work = ollama-API translate + ps/why surfacing only |
| R7 | ⚠️ SPLIT — 2/3 gateway-feasible | full upstream metric table verified (server-task.cpp:1520-1615): prompt_tokens_cached_total, prompt_tokens_total, prompt_seconds_total, requests_deferred/processing, spec-decode 3x, busy slots — NO kv-offload byte gauges, NO lookup-delay histograms | ❌ blocked (upstream lane): offload read/write byte counters + sync/async op histograms. ✅ gateway-now: (a) cache-hit ratio as cached/(cached+processed) — upstream `prompt_tokens_total` EXCLUDES cached tokens, so cached/processed can exceed 1 (live-pinned 2026-09-09: 2.27 pre-fix), (b) lookup-delay ANALOG = cold-vs-warm prefill TTFT split (gateway TTFT Histogram × response `cached_tokens`), (c) expose Pallama-owned EWMA hit-rate + cache-ram clamp pct as gauges, (d) queue-pressure from requests_deferred |
| R8 | ✅ SHIPPED 2026-09-09 (S) — `arch` column already in store (store.rs:39,102); profile rules branch on GGUF fields today (head_count_kv fallback, profile.rs:2300-class) | shipped as GGUF-truth KV math (recurrent_layers array + interval fallback) rather than arch-keyed special-casing — metadata-driven, zero hardcoding; KV ladder now fed the fractional truth automatically |
| R9 | ✅ SHIPPED 2026-09-09 (S) — json_schema→grammar is converted PER REQUEST server-side (server-schema.cpp:251-269) | shipped as pre-validation (top-level format/grammar/response_format lint, pre-admission) + raw GBNF passthrough (grammar string skips child conversion entirely); converter-cache itself = documented L follow-up (1268-line upstream converter, fidelity risk) |
| R10 | ✅ FEASIBLE (M-L) | enforce_prompt_fits hook exists (preflight.rs:31-45: 90% est threshold → exact count via running child); translate layer owns message history; session bank + checkpoints live | history-rewrite policy + KV-checkpoint interplay needs design care (quality risk) |

**Audit result: 9/10 green (R5/R6 cheaper than the research pass assumed), 1
downgraded (R7 → gateway-metrics reshaping, offload split refused to upstream
lane).** Sequencing unchanged; R7 slot swaps to the reshaped metrics row.

**Wave 2026-09-09: R2 + R6 + R7-lite SHIPPED** (tests: 171 gateway suite green;
live-validated same day). Sequencing for the rest: R1 (headline) → R3 → R5 →
R4 (after R1) → R8/R9 as fillers → R10 as its own wave.

**Sources (read 2026-09-09):** llama.cpp b10850-b10853 release notes;
SGLang 582-PR release notes (session radix cache #29173, Rust frontend,
weight-cache daemon); vLLM 0.26.0 changelog (411 commits); Bifrost/getmaxim/
NeuralTrust semantic-cache docs; arXiv 2409.04701v3 (late chunking);
arXiv 2607.19223 AdaFlash (adaptive speculation); techtimes DFlash coverage.
Search artifacts: /tmp/opencode/{llamacpp_b10853,sglang,semcache,vllm026}.json.

## Sources

- Vendored upstream `references/llama.cpp-master`: `common/arg.cpp` (flag
  definitions + help text), `tools/server/server.cpp` (route table),
  `tools/server/README.md` (flag tables incl. K/V cache types), 
  `src/llama-context.cpp` (quantized-V-requires-FA throw).
- pallama code: `core/src/profile.rs` (rules 1-13), `gateway/src/lib.rs`
  (route table), `runtime/src/engine_impl.rs` (Engine trait).
- Web (read 2026-09-05): aixfunda.substack.com "The new Router mode in llama
  cpp server"; freshlab.es "Local LLM Router Mode"; docs.openwebui.com
  llama.cpp alternatives page; github.com/lordmathis/llamactl (README:
  multi-backend, dashboard, LRU, remote instances).
- Prior internal docs: `docs/perf-roadmap.md` (tier A/B status),
  `docs/frontier-comparison.md` (vLLM/SGLang sourced numbers),
  `docs/ollama-gap-analysis.md` (complaint matrix).
