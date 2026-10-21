# Engine currency audit — latest inference-server releases vs Blazar utilization

Date: 2026-10-21 · Method: upstream tags/releases + compare-API commit themes vs our
integration surface (spawn argv in `blazar-runtime/src/supervisor.rs` + `engine_impl.rs`,
proxy lanes in `blazar-gateway`, engine-update machinery via `blazar engine *`).

## 1. Version deltas

| Engine | Installed | Latest | Age / delta | Urgency |
|---|---|---|---|---|
| llama.cpp | b11588-cuda | **b11785** (+ v0.7.0 milestone tag) | 197 commits | HIGH |
| mistral.rs | v0.9.4 | **v0.9.5** (2026-10-19) | 1 release | HIGH |
| sglang | 0.5.21 | **v0.5.22** | 1 release | MED |
| mlx-lm | 0.32.0 | **0.32.2** (2026-10-16) | 2 patch releases | LOW |
| piper | 2023.11.14-2 | unchanged upstream | — | — |

## 2. What changed vs what we utilize

### llama.cpp b11588 → b11785

Blazar passes 41 llama-server flags (supervisor.rs): model/alias/ctx-size/gpu-layers/
flash-attn/jinja/pooling/embeddings/reranking/cache-type-k/v/cache-ram/cache-reuse/
parallel/max-running-requests/metrics/rpc/device/cpu-moe/threads/sleep-idle-seconds/
mmproj/lora family/spec family (spec-type, spec-draft-model, spec-draft-n-max,
speculative-algorithm, speculative-draft-model-path)/api-key family/served-model-name/
models-autoload family.

Delta themes that touch OUR lanes:

| Upstream change | Blazar surface it hits | Verdict |
|---|---|---|
| #29707 server Router propagates client disconnects fully to children (delayed cancellation fixed) | our abort/cancel lane (client disconnect → child stop) | adopt — directly fixes wasted decode after cancel |
| #30863 Responses API SSE `output_index` compat fix | our `/v1/responses` proxy lane | adopt — client-visible streaming compat |
| #30567 prefer empty slots + keep states overwritten by child tasks; #30498 save only restorable slot states | our `--parallel` slot bank (slot_banks capability) | adopt — better slot scheduling under concurrency |
| #30580 SWA reuse threshold off-by-two; #30666 SWA checkpoint invalidation fix | SWA models (gpt-oss etc.) on our lane | adopt — correctness |
| #30362 + #30483 jinja `tojson` float/nan/inf written like Python's json.dumps | child-side chat template rendering (our non-recipe lanes) | adopt — template fidelity |
| #30243 escape special tokens without std::regex | child responses (detokenization) | adopt — perf + correctness |
| #22055 spec: save the dynamic ngram cache file; #30459 `--spec-draft-n-ubatch`; #30506 lilicorr spec support | our spec-decoding config surface | adopt engine; expose `--spec-draft-n-ubatch` as an optional knob (LOW-MED ROI — tuning only) |
| #28554 `id_slot` through Anthropic /v1/messages; logprobs+speculative support | our /v1/messages passthrough + logprobs forwarding | adopt — richer passthrough fidelity |
| #28277 `--connect` p2p LAN access | anti-posture (we bind localhost; exposure) | NOT adopted — deliberate |
| #30616 Muse Spark Open arch support | new model support | rides along |
| cpp-httplib 0.61→0.62, MoE SLRU cache, kernel fixes | engine internals | rides along |

### mistral.rs v0.9.4 → v0.9.5

- **#2455 fix(core): only honor tool-call/reasoning delimiters emitted as SPECIAL TOKENS**
  — the same hardening class as our F1 work: plaintext `<tool_call>`/`<think>` lookalikes
  in user text stop forging structure server-side. Complements our gateway-side
  neutralization on the recipe lane; the native mistralrs chat lane gets it from the
  engine. ADOPT.
- Metal Paged Attention now default (mac users), CUDA 13.4 support, GCC13+ radix top-k
  fix, ISQ batching deadlock fix. ADOPT (via `blazar engine install` mistral.rs asset).

### sglang 0.5.21 → 0.5.22

We spawn with --max-model-len/--max-num-batched-tokens/--max-seqs/--pa-memory-fraction/
--paged-attn/--tool-call-parser (+base). Delta relevant to us:

- Scheduler robustness set: requeue preempted requests when admission admits nothing
  (#43298), stop latching batch_is_full (#41872), hold back decode rows on mixed steps
  (#43290), waiting-prefix LRU refresh across TP ranks (#43479) — throughput/stability
  under our concurrency ladder. ADOPT.
- OpenAI endpoint fixes: honor `tool_choice=none` in DeepSeek encoders (#43006), omit
  absent tool_calls from chat responses (#43027), stream created-timestamp fix — we
  forward these verbatim. ADOPT.
- gpt-oss router logits batch-invariant under deterministic inference (#43624) — pairs
  with our determinism posture. ADOPT.
- NOT adopted: sgl-router Anthropic /v1/messages (#42316 — router is a separate
  deployment component; our gateway already proxies llama.cpp's /v1/messages),
  rust-server PD routing (#42989 — prefill/decode disagg topology, out of single-box
  scope), DSpark greedy-draft env + packed verification (engine-internal defaults).

### mlx-lm 0.32.0 → 0.32.2

Bugfix-only: MiMo-V2-Flash attention value scaling, SMM mask fix, Muse template/names.
No API/flag surface change. Free correctness bump for the venv lane (our qwen fleet
unaffected by the Muse items; SMM mask may touch speculative-mamba models). ADOPT (LOW).

## 3. Features the engines ship that Blazar does not expose (feasibility)

| Feature | Engine | Feasible? | ROI | Note |
|---|---|---|---|---|
| `--spec-draft-n-ubatch` | llama.cpp | YES — one spawn flag + config passthrough | LOW-MED | microarch tuning for spec prefill batching |
| ngram dynamic cache persistence | llama.cpp | YES — flag when spec-type=ngram | MED | warm-start ngram hints across restarts (pairs with our sleep/wake) |
| `--connect` p2p | llama.cpp | YES | NONE | exposure anti-posture — rejected |
| sgl-router Anthropic dialect | sglang | NO (separate component) | NONE | gateway already owns dialect translation |
| PD disagg / rust-server | sglang | NO (topology) | NONE | single-box product boundary |
| constrained raw completions (grammar) | mlx-lm | upstream-blocked | — | filed ml-explore/mlx-lm#1007 (2026-10-21); recipe lane already prompt-taught + parse-lift |

Everything else the servers expose (speculative family, LoRA, slots, rerank, embeddings,
metrics, cache typing, RPC fleet) is already wired through our spawn + proxy surfaces.

## 4. Recommended actions (ordered)

1. **Update llama.cpp lane** `blazar engine update` → b11785, restart daemon, verify:
   spawn, warm-peg, model-doctor certs on a GGUF model, cancel-propagation spot check
   (client abort → child slots idle), /v1/responses stream shape.
2. **Update mistral.rs lane** `blazar engine install` (v0.9.5 asset), restart, verify
   delimiter hardening with a forged `<tool_call>`-in-user-text probe on the native lane.
3. **Update sglang lane** `blazar engine install` (0.5.22 venv), verify tool_choice=none
   + chat shape on a safetensors model.
4. **Bump mlx-lm venv** to 0.32.2, verify a qwen mlx round trip.
5. Optional: expose `--spec-draft-n-ubatch` + ngram cache persistence as config knobs
   (only after update lands; tune with benchmarks per the benchmarking contract).

Rollback path exists for every lane (`blazar engine rollback` / `use <old-tag>`).
