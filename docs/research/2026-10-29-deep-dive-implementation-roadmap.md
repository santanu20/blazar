# Deep-dive implementation roadmap — 2026-10-29

Companion to `2026-10-29-engine-changelog-audit.md`. Method: commit-level upstream
deltas (compare APIs, master not just releases), llama.cpp master `server.cpp` route
table (52 routes) diffed against Blazar's gateway census, hot open-PR radar per
engine, Blazar code/git cross-reference (profile.rs, supervisor.rs, realtime.rs,
store.rs). Every item below names its evidence.

## 1. What the release-based audit missed (commit-level findings)

Releases carry 1 PR each; master is 51 commits ahead of b11946 — **32 commits
beyond releases**, incl. two server PRs:

| Finding | Evidence | Class |
|---|---|---|
| **#31287 tool calls in /v1/realtime** | server.cpp master `ctx_http.ws("/v1/realtime")` + PR | capability Blazar's own realtime lacks |
| **#31350 restore spec checkpoint for drafts > n_rs_seq** | master commit | spec-decode correctness (our governor/ngram lanes) |
| #31321 reuse tensor staging buffer, skip direct-io bounce | master commit | model LOAD speed (cold start) |
| #31300 remove DSV4 simple batch splitting | master commit | DeepSeek-V4 decode change |
| #31283 mtmd video input for LFM3 | master commit | multimodal video |
| UI wave (search, OGG attachments, MCP checkboxes…) | ~14 commits | N/A (child webui unused) |
| kernel wave (Metal MXFP4 mat-vec, OpenCL iq2, CDNA sparse attn, SVE gemv) | master | rides along |

## 2. Route-table diff: llama.cpp master (52) vs Blazar gateway

Blazar already mounts the previously "tracked-not-adopted" family:
`/v1/streams/lookup`, `/v1/stream`, `/v1/messages/count_tokens`, `/props`,
`/v1/chat/completions/input_tokens`, `/v1/responses/input_tokens`, `/slots{,/{id}}`,
`/v1/systemone`, rerank aliases (lib.rs:356-516). Remaining gaps:

| Gap | Upstream shape | Recommendation |
|---|---|---|
| `/lora-adapters` GET/POST | "LoRA adapters hotswap" — attach/detach adapters on a live child | **IMPLEMENT** — mirror the sglang one-child hot-attach pattern (colon routing `base:adapter`, hot rm/add same-pid) onto the llamacpp lane; today llamacpp LoRA is spawn-time only (profile.rs:744 `--lora`/`--lora-scaled`, `--lora-init-without-apply`, reserved `--lora-max-rank/-max-adapters`) → adapter switch costs a respawn |
| `/tts` POST | `routes.post_tts` — built-in TTS from TTS-capable GGUFs | **EVALUATE→IMPLEMENT** — route `/v1/audio/speech` to child `/tts` when the model probes TTS-capable (manifest feature gate); piper stays default voice lane. Adds a TTS engine lane with zero new processes |
| `/tools` GET/POST | `tools.setup(params.server_tools, mcp_mgr, …)` — **server-side MCP client + tool runtime** in llama-server | **WATCH/DECIDE** — gateway owns MCP (`[[mcp]]` config, /api/mcp); child-side MCP is an alternative tool-execution plane. Possible harvest: surface child tool capability in /api/mcp status; don't wire execution until product sign-off |
| `/models/load`, `/models/unload`, `/models/sse`, DEL `/models`, POST/DEL `/v1/stream`, `/cors-proxy`, `/embedding` (legacy), `/v1/health` | server-side model hot-swap + stream GC + CORS relay | **NOT ADOPTING (deliberate)** — Blazar owns model lifecycle, streams, and posture; these duplicate the supervisor's job |

## 3. Realtime tool calls — spec for Blazar (highest-value new capability)

Upstream #31287 proves the shape; Blazar's `realtime.rs` is audio+chat only
(no tool parse/format events). Implementation sketch:
1. `session.update` accepts `tools[]` (OpenAI realtime shape: type=function,
   name, description, parameters) — validate + cap count.
2. Model output containing tool-call syntax in the text channel → emit
   `response.function_call` + `conversation.item.create` (function item),
   wait for `conversation.item.create` (function_call_output) from the client
   before continuing the turn — the existing session loop already multiplexes
   client↔child frames; add the function-item state.
3. Reuse the gateway's existing tool-call adapter (ThinkSplitter/holdback lane,
   prompt.rs parse-lift) for the child dialects; realtime then inherits every
   engine's tool certificate.
4. E2E: extend `tests/realtime.rs` with a scripted WS client that answers a
   function call (pattern: piper→PCM cascade test).

## 4. In-flight upstream radar (be-ready list)

- **llama.cpp**: #31372/#31376 DSpark/Spark2.5 draft models → when merged, add
  `catalog.json` spec_pairs + auto-pair rules (low effort, release-time);
  #31374 reasoning-budget end-tag (watch our reasoning-effort args still parse);
  #31368 rollback-snapshot guard (cache correctness, rides); #31366 Intel hybrid
  thread fix (CPU lane).
- **mistral.rs**: #2518/#2519 tool-call parser fixes (hunyuan closing tag in
  string arg, literal "null" arg) — pairs perfectly with our tools-cert lanes on
  the next release; #2514/#2510/#2509 CPU quant perf wave; #2511 Kolibri-1 model.
- **mlx-lm**: #2016 LFM3 (+3, hot) — same LFM3 wave as mistral.rs #2515 and
  llama.cpp video-input #31283; #2008 quantized-KV packed width fix; #2011
  partial_rotary_factor fix. All ride along on 0.32.3+; then rerun the mlx lane
  (qwen3 tool-marker wall may also move — check issue before pinning hopes).
- **sglang**: steady dflash/AMD-sa stream; nothing Blazar-visible this week.

## 5. Blazar-side deferred items (code-verified, unchanged priority)

| Item | Seam | ROI |
|---|---|---|
| Load-aware slot re-provisioning (static `-np 2` pin; C4 0.56×/C8 0.61× queueing; adaptive_slots 60s-sustained window misses bursts) | supervisor.rs `adaptive_slots_tick` (6878) + profile -np pin | **HIGH** (biggest measured e2e gap) |
| Benchmarks wave | bench_matrix.py + receipt machinery | HIGH (blocked on approval) |
| `mlx_signal` classifier: mlx-community quant dirs misroute to sglang | store.rs:366+ (feeds ShardFormat::detect) | MED bug |
| Orphan adoption sweep on daemon restart (§9 teardown) | daemon boot sweep exists for pulls; untracked children can survive stop/start | MED |
| Streaming fence normalization (non-stream only today) | proxy.rs fence posture | MED (needs sign-off) |
| Pull-era bugs: piper quant fuzzy-mismatch, `blazar rm` name/file, whisper/piper ENGINE column dagger, mistralrs `--tok-model-id` auto-derive | flagged entries | LOW-MED each |
| Size-aware spec policy (0.5b win / 9B loss, n=2) | spec governor follow-up | LOW until more data points |

## 6. Consolidated action list (ordered)

1. `blazar engine update llamacpp` → b11982+ (qwen3-coder template fix, MoE CUDA
   fusions, M-RoPE bounds; later #31321 = faster cold load) + rerun llamacpp lane.
2. `blazar engine update sdcpp` → master-1024 (INT8 CUDA fusions, iGPU planning).
3. **Implement realtime tool calls** (§3) — flagship user-facing capability.
4. **Implement llamacpp LoRA hot-attach** (§2) — sglang-pattern parity, kills respawn.
5. Bench MoE-fusion delta (granite-4.0-tiny, b11946 vs b11982, receipt campaign).
6. Load-aware slot re-provisioning design spike (§5 row 1) — measure-first per §8.
7. Evaluate `/tts` child lane; decide `/tools` posture (product call).
8. On next mistral.rs/mlx-lm releases: unpin/update, rerun lanes, re-diff `--help`,
   add LFM3 + DSpark entries to model matrix.
9. Weekly cadence for this audit (it caught 32 un-release'd commits incl. 2 server PRs).
