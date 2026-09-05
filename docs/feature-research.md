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
