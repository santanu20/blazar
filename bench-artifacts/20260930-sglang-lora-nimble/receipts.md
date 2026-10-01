# 20260930 — sglang runtime LoRA: validated doors, spawn preflight, argv proof, Nimble attempt

Campaign: sglang-lora-nimble · Box: RTX 4070 Laptop 8188 MiB, 13674 MiB RAM, sglang 0.5.20 (venv engine), gateway :11435 · Binary: target/debug/blazar (deployed to /usr/local/bin/blazar, daemon pid 576592)

## 1. CLI attach door (`validate_lora_attach`, crates/blazar-cli/src/main.rs)

| # | Input | Result |
|---|-------|--------|
| T1 | `lora add qwen3-1.7b /nonexistent/adapter-dir` | rc=1, teaching: adapter path not found |
| T2 | `lora add qwen3-1.7b <real .gguf>` | rc=1, teaching: .gguf/.bin adapters serve on llamacpp engine |
| T3 | `lora add no-such-model <real PEFT dir>` | rc=1, teaching: no such model — `blazar pull` |
| T4 | `lora add qwen3.5-9b <PEFT dir>` (GGUF file-lane row) | rc=1, teaching: PEFT dirs serve on sglang engine |
| T5 | `lora add qwen3-1.7b ~/.local/share/blazar/loras/eternis-anonymizer-qwen3-1.7b` | rc=0, row #3 attached |
| T6 | `lora add qwen3.5-9b-bf16 ~/.local/share/blazar/loras/bespoke-nimble-9b` | rc=0, row #5 attached |

## 2. Spawn preflight (`resolve_lora_lane`, crates/blazar-runtime/src/supervisor.rs)

- Missing-on-disk row (sqlite-inserted, path `/tmp/gone-lora-dir`) → request `qwen3-1.7b+gone-lora-dir` → HTTP 404:
  `lora /tmp/gone-lora-dir (row #4) attached to "qwen3-1.7b" is missing on disk — re-download the adapter or blazar lora rm 4 the row`
- Pre-fix behavior (row #1 era): nonexistent path was accepted at `lora add` with rc=0 — garbage row reached spawn. Root cause fixed at the door; spawn check is the second gate.

## 3. Live child argv proof (/proc/<pid>/cmdline, sglang 0.5.20)

- Dense mode (co-actor's plain `qwen3-1.7b` request, eternis row attached):
  `python -m sglang.launch_server --model-path .../qwen3-1.7b.d --host 127.0.0.1 --port 37679 --served-model-name qwen3-1.7b --context-length 16384 --enable-lora --lora-paths eternis-anonymizer-qwen3-1=.../eternis-anonymizer-qwen3-1.7b ...`
- Variant mode (request `qwen3.5-9b-bf16+bespoke-nimble-9b`):
  `python -m sglang.launch_server --model-path .../qwen3.5-9b-bf16.d --host 127.0.0.1 --port 40167 --served-model-name qwen3.5-9b-bf16 --context-length 16384 --enable-lora --lora-paths bespoke-nimble-9b=.../bespoke-nimble-9b ...`
- Raw captures: /tmp/nimble_argv_576739.txt (dense), nimble_argv_594144.txt + nimble_argv_594425.txt (variant).

## 4. Variant serve proof (workload-class: PEFT adapter on exact-base)

- Request `qwen3-1.7b+eternis-anonymizer-qwen3-1` (eternis/eternis_anonymizer_lora_Qwen3-1.7B_7jul_r16, r16 alpha16, classic Qwen3 targets k/v/down/up/q/gate/o_proj): 62.3 s cold spawn → completion reply 200.
- Wrong stem (full dotted dir name) → 404 with teaching (file_stem semantics: dots split names — request the stem, not the dir).

## 5. Bespoke-Nimble-9B attempt on this box — honest hardware wall

- Base: `~/.local/share/blazar/models/qwen3.5-9b-bf16.d` (Qwen/Qwen3.5-9B snapshot, 18 GiB, 16/16 files, HF_HUB_DISABLE_XET=1) adopted by boot preflight as sglang row `qwen3.5-9b-bf16`.
- Adapter: `bespokelabs/Bespoke-Nimble-9b` PEFT dir (r16 alpha32, targets include GDN in_proj_qkv/in_proj_z/in_proj_b/in_proj_a/out_proj + classic qwen modules), 173 MiB safetensors.
- Result: sglang crashed at layer weight-init (`MergedColumnParallelLinear.create_weights` under offloader wrap_modules) on both crash-loop attempts; gateway surfaced 502 `engine crashed` with full traceback, supervisor gave up cleanly, no hang, no slot leak (`blazar ps`: no models loaded).
- Verdict: LoRA wiring correct end-to-end (argv above proves emission for this exact spawn); failure is 18 GiB BF16 vs 8 GiB VRAM + ~5 GiB free RAM. Nimble class on 8 GB cards requires a quantized Qwen3.5-9B base (no official FP8/AWQ ships from Qwen today) or a bigger GPU.
- Typed-decision serving pattern (schema enum + logit_bias + logprobs) independently proven on qwen3-1.7b via gateway: receipts /tmp/nimble_pattern_receipts.json (json_schema strict enum answer + top_logprobs passthrough; logit_bias ±100 → forced token stream, warm 0.1 s).

## 6. Guards re-run after all edits

- `cargo nextest -p blazar-runtime -p blazar-cli`: 742/742 PASS.
- clippy: clean on touched code (2 pre-existing warnings in untouched doctor region).

## 7. Root-cause fixes: nested-config planning, row heal, dotted adapter names (2026-09-30 evening)

Three defects surfaced by the Nimble attempt, each fixed at its owning layer:

- **Nested config descent** (`crates/blazar-core/src/hfmeta.rs`): geometry/ctx/dtype keys absent at the top level now fall back into `text_config` (Qwen3.5/Qwen3-VL/Llama-4 wrapper shape; top level keeps priority, flat configs identical). `full_attention_interval >= 2` reduces the KV layer count to the full-attention layers (Qwen3.5-9B: 32 layers / interval 4 → 8 KV layers) — hybrid GDN layers hold constant state, counting all layers overestimates KV 4x.
- **Dir-row heal** (`crates/blazar-runtime/src/models.rs`): boot reconcile re-measures existing safetensors dir rows (bytes/shards/ctx/arch/params) when disk drifted — a row adopted mid-download froze 3.1 GiB for an 18 GiB dir because second boots skip owned dirs. Heals only on a COMPLETE readable dir (weights present + config parses); never touches sha256/quant/mmproj/pulled_at.
- **Spawn measurement belt** (`crates/blazar-runtime/src/supervisor.rs`): planning re-measures dir rows from disk at spawn (fail-open to row bytes), so frozen sizes can't feed the fit ladder even mid-session.
- **Dotted adapter names** (`resolve_lora_lane`): `model+adapter` suffix now matches the full file/dir name (stem shorthand unchanged) — `qwen3-1.7b+eternis-anonymizer-qwen3-1.7b` no longer requires the dot-truncated stem.

Live proofs (gateway :11435, daemon 663606, sglang 0.5.20):

| Check | Result |
|---|---|
| Boot heal of `qwen3.5-9b-bf16` | row 3.1 GiB → **18.0 GiB**, ctx `-` → **262144**, shards 4 |
| 9B+nimble request (guard on, default) | **0s** teaching refusal: `insufficient memory ... MemAvailable 8168 < floor 9717 (model 18411)` — no spawn, no crash-loop |
| 9B+nimble request (guard off, opt-in) | admission reservation refuses (18 GiB > 8 GiB card, nothing evictable) → bounded 2-min queue → 503; still zero blind spawns |
| Dense `qwen3-1.7b` after all edits | 43s cold spawn, reply ok (keep-pin) |
| Variant by FULL dotted name | 35s cold spawn, reply ok; child argv `--lora-paths eternis-anonymizer-qwen3-1=...` (live /proc proof) |
| Full suite | `cargo nextest -p blazar-core -p blazar-runtime -p blazar-cli`: **1126/1126 PASS**, clippy 0 warnings |

Pre-fix behavior for the same 9B request (receipt §5): stale 3.1 GiB row + KvGeom None → fit ladder skipped → blind spawn → sglang `create_weights` crash-loop → 502s. The whole chain is closed: honest sizes, real geometry, teaching refusals.

Not committed — shared worktree, co-actor holds index.

## 7b. AllSlotsBusy conflation fix (same day, follow-up)

Root cause: `admission_blocked` returned one bool for two worlds — "residents in the way" (evictable, queue correct) and "incoming floor > every card even empty" (physically unschedulable). The latter surfaced as a 120 s `all_slots_busy` queue timeout on an idle-capable verdict known at entry.

Fix (root cause, at the classifier): `SupervisionError::ModelTooLarge(String)` variant + `largest_card_budget(name, incoming)` classifier (Some(best) only when floor exceeds EVERY known candidate card empty; None on any unjudgeable card = fail open) wired into the admission `Refuse` arm; `admission_candidates` extraction shared with `admission_blocked` so the two can never disagree. Gateway maps it 503 immediately (no queue). The post-reservation `Refuse` arm deliberately keeps `AllSlotsBusy`: a card fit at admission, only a concurrent reservation holds it (genuinely temporary).

Live receipt: qwen3-1.7b held in-flight (long stream), then `qwen3.5-9b-bf16+bespoke-nimble-9b` → **0.073 s** 503: `admission floor 19.2 GiB exceeds every GPU budget (largest card 8.0 GiB) even with the box empty — waiting for a slot can never change this. Use a smaller quant (blazar fit), lower the ctx/slots, or serve on a larger card`. Hold stream completed untouched (finish_reason length). Tests: unit__largest_card_budget__impossible_vs_busy_vs_fail_open (impossible/busy/exact-edge/fail-open/arm-reachability), unit__supervision_error__model_too_large_maps_to_503_teaching; full blazar-runtime+blazar-gateway 1056/1056 PASS.

## 8. Parallelism-aware planner (auto TP for SGLang, 2026-10-01)

Planner root cause: fit ladder and admission judged unsharded weights, so multi-GPU boxes either crashed at NCCL spawn (unpinned oversize) or refused loads that TP could serve.

| Proof | Result |
|---|---|
| `plan_auto_tp` pure fn | [8188,8188]MiB: 12000MiB→Some(2), 8000→None, 17500→None; [8188]→None; [16000,4000] 17000→None (bottleneck); 3×8188 20000→Some(3) |
| Emission, unpinned | 12GiB model, 2×8GiB → Tier A full, `--tp-size 2` + NCCL teaching warning, mem-fraction 0.756, no offload |
| Emission, manual wins | tp_size=2 pin + auto=3 → only `--tp-size 2`, no auto warning |
| Ladder per-rank honesty | 18GiB model, 2×8GiB → per-rank 9GiB > 7.2GiB budget → Tier C `--cpu-offload-gb 3` + per-rank host share, not a blind Tier A |
| Admission spanning bypass | dual-GPU fixture, 6000MiB resident: spanning=false + 9000MiB → blocked; spanning=true + 9000 → admitted (15000≤16376 summed pool); spanning=true + 12000 → blocked (18000>16376) |
| Refusal teaching | impossible + spanning → combined-pool numbers `even sharded across N cards (X GiB per rank)` |
| Old-engine contract | base 0.5.19 flag fixture unchanged: tp_size pin on old engine still warn-skips (fixture split: base = old surface, extended = new surface with --tp-size) |

Honest boundary: this box is single-GPU — the live daemon takes the planner-None path everywhere (regression below proves byte-same behavior); TP correctness is proven by the unit fixtures and emission tests above, not by a live multi-GPU spawn. Suite: 1594/1594 across 4 crates, clippy clean.

Live single-GPU regression after deploy: dense `qwen3-1.7b` serve ok (44.7 s cold, real reply); `qwen3.5-9b-bf16+bespoke-nimble-9b` → byte-same pre/post-deploy guard refusal (MemAvailable teaching; spawn_mem_guard fires ahead of admission when on). The ModelTooLarge 503 mapping and text are pinned byte-exact by unit__supervision_error__model_too_large_maps_to_503_teaching; single card means no sharding note appears (correct: planner None, spanning false). healthz ok.

## 9. GPU rank pinning + mistral.rs layer-spanning (2026-10-01)

- Engine trait `spawn(argv, endpoint, spawn_env)`: per-spawn env applied AFTER `engine_env`, same-key override logged (never silent); all four engine impls + router/per-instance call sites updated; router passes empty (single-process by construction).
- Env merge proof on a REAL child (`/bin/sh printenv`, unit__spawn_child_env__spawn_env_overrides_engine_env_last_wins): spawn-time value wins the shared key, engine-only key survives, override visible in logs.
- Pin contract (unit__spanning_env_from_probe__atomic_pci_pair_or_fail_open): ranks 2 + 2 probed NVIDIA cards -> exactly [CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=0,1]; prefix for manual tp < card count; probe short / non-spanning / single rank -> EMPTY env (fail-open). Indices derive from a FRESH nvidia-smi probe (PCI order) at spawn time, never the mixed llamacpp census (enumeration-order mismatch hazard).
- mistral.rs spanning (unit__plan_mistralrs_sum_span__summed_pool_not_bottleneck): 17 GiB on 16+4 GiB pair -> Some(2) (uneven layer split legal; sglang TP refuses the same load per-rank 8.5>4); fits-one-card / over-sum / single-card -> None. Ground truth: `mistralrs serve --help` — `-n --device-layers` "Omit for automatic device mapping"; no --tp-size exists on mistral.rs 0.9.4.
- Admission/wiring: mistralrs arm rides the existing spanning bypass (summed pool) + spanning-aware ModelTooLarge; `wants_pick` now gated on !spanning (single-card scoping would mis-scope every sharded estimate, incl. manual tp and mistral.rs).
- Single-GPU live boundary: this box probes 1 NVIDIA card, so spanning_spawn_env and both arms stay inert by construction; regression below proves byte-identical behavior. Multi-GPU correctness is pinned by the unit fixtures above (no multi-GPU box available).
- Suite after the change: 1597/1597 PASS (4-crate nextest; 1 slow + 9 leaky pre-existing); clippy 0 warnings.

## 10. 20261001 — Federation v1: peer presence + bare-name fallback (LIVE, two real daemons)

Design: local store-only resolution was the seam. Fallback lives at the openai_proxy lane only (after split_remote, before body rewrites): local miss (pure `not found`, ambiguous stays local) -> peers_serving() (60s TTL presence cache of peer /v1/models, live-filter, least-in-flight, config-order stable) -> forward_with_health with `peer:model` (inherits STRIP_REQUEST: caller `authorization` never forwarded; remote.key becomes bearer). /v1/models merges peer ids (owned_by blazar-remote, snapshot semantics). Kill switch: `remote_fallback = false` (default true, meaningful only with remotes).

Box: primary :11435 (systemd) + sandbox peer fed2 :11436 (XDG_CONFIG_HOME/XDG_DATA_HOME isolated store+config; engine rows sqlite-copied; engines dir symlinked to primary binaries; GGUF reflink-copied, adopted as fedtest-0.6b).

| Proof | Request to PRIMARY | Result |
|---|---|---|
| P1 explicit prefix | `fed2:fedtest-0.6b` | 3s, real completion ("**OK**") from peer child |
| P2 bare fallback | `fedtest-0.6b` (no local row) | 0s warm, real completion "peer-ok" routed to fed2 |
| P3 models merge | GET /v1/models | lists `fedtest-0.6b:q4_0` (owned_by blazar-remote) |
| P4 peer down | bare request after kill -TERM fed2 | 0s `remote "fed2" unreachable` — fast-fail, no hang; presence stale-within-TTL routes to forward which fails honestly (explicit-route contract); after TTL expiry/no claim -> existing local 404 teaching |
| P5 local regression | `qwen3-1.7b` | 40s cold spawn, reply ok — local lane untouched |

Sandbox gotchas receipted: `ln -sfn` nests when target dir exists (rm first); nohup daemons die on tool timeout (use setsid); env-isolated daemons unfindable by pkill -f (match /proc/N/environ or ss port); registry pull stalled twice (reflink copy instead — pull-dialect filename with quant tail required); config.toml `remotes = []` inline means sed-replace not append (TOML duplicate key = boot fail).

State after test: fed2 killed, primary config restored (remotes empty, backup held outside the repo), sandbox removed, primary healthz ok. Tests: gateway+core+runtime suites green, clippy 0. Not committed (shared index).

## 11. 20261001 — HW-agnostic pinning (Vulkan) + federation breadth

**Vulkan pinning contracts (ground-truth experiments, real sd-server vulkan build):** E1 plain run lists `0 = Intel(R) Graphics (RPL-S) / 1 = NVIDIA RTX 4070` (llvmpipe auto-excluded by ggml). E2 `GGML_VK_VISIBLE_DEVICES=0` → only Intel, still labeled `Vulkan0`. E3 `GGML_VK_VISIBLE_DEVICES=1` → only NVIDIA, RELABELED `Vulkan0` — the env FILTERS AND RENUMBERS child ids. E4 `CUDA_VISIBLE_DEVICES=0` on a vulkan child changes nothing — CUDA env does not touch Vulkan enumeration.

**New pure core:** `spanning_env_from_census(ranks, probed_nvidia, census_names, lane_allows_vk, argv_device_free)` — Vulkan priority for llamacpp spans whose census names parse as `Vulkan<N>` (strict subset → `GGML_VK_VISIBLE_DEVICES=<ids>`; identity set → no emission), else the existing CUDA/ROCm PCI pair, else empty fail-open. `parse_vk_id` strict token parse. Census provenance = the llamacpp child's own `--list-devices` ids, so no vendor lists. Emission requires device-id-free argv (`--tensor-split` ratios/`--tp-size` only) — the renumber hazard is void; comment at the emission site audits any future device-id argv. Unit: unit__spanning_env_from_census__vulkan_priority_cuda_fallback_fail_open (subset / identity no-op / CUDA names skip / malformed fail-open / count-short fail-open / non-llamacpp skip / argv-ids → CUDA pair keeps its shipped probe-count contract).

**Federation breadth:** shared `try_fallback_forward` + `FallbackLane::{OpenAi, OllamaChat}` — one miss-detection path (pure not-found, ambiguity stays local), `peers_serving_with_refresh` adds one throttled forced re-probe per remote (≤1/10s, `PeerPresence.last_forced`, pure gate `force_refresh_due` unit-tested) so a peer that gains a model inside the 60s TTL is found. Hooks: OpenAI lane (chat/completions/embeddings/images via the proxy hook), ollama `/api/chat` (translated remote round-trip, token budgets ride like the explicit lane), `/v1/images/generations` (byte forward, peer gate owns caps). `/api/generate` stays local-only — it refuses remote prefixes by design (F14); fallback honors the same contract.

**Live proofs (primary :11435 + sandbox peer fed2 :11436, real daemons, real children):**
- P6 `/api/chat` bare `fedtest-0.6b` → peer reply `peer-ok` through the ollama translator ✓
- P7 `/v1/embeddings` bare → real 1024-dim vector from peer ✓
- P8 peer renamed its model + restarted; primary presence cache TTL-fresh with the OLD listing; bare `fedrenamed-0.6b` → discovered via throttled re-probe → `refresh-ok` ✓ (without the fix: local 404 for up to 60s)
- Regression: local qwen3-1.7b serve ok (after freeing the shared GPU), 9B+nimble guard teaching byte-same, `/api/generate` bare peer name → local 404 by contract ✓
- Contention finding (honest, not a regression): a peer on the SAME box spawns children the primary cannot evict (separate ledgers) — a marginal-fit spawn crashed at sglang prefill cuda-graph capture with 4.4GB held by the peer's 0.6b child. Production peers belong on separate boxes; documented here as the one-GPU-box caveat.

**Suite:** 1453/1453 (core+runtime+gateway) + supervisor VK tests, clippy 0. Config restored (remotes = []), sandbox removed, single daemon on :11435.
