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

- Missing-on-disk row (sqlite-inserted, path `/tmp/opencode/gone-lora-dir`) → request `qwen3-1.7b+gone-lora-dir` → HTTP 404:
  `lora /tmp/opencode/gone-lora-dir (row #4) attached to "qwen3-1.7b" is missing on disk — re-download the adapter or blazar lora rm 4 the row`
- Pre-fix behavior (row #1 era): nonexistent path was accepted at `lora add` with rc=0 — garbage row reached spawn. Root cause fixed at the door; spawn check is the second gate.

## 3. Live child argv proof (/proc/<pid>/cmdline, sglang 0.5.20)

- Dense mode (co-actor's plain `qwen3-1.7b` request, eternis row attached):
  `python -m sglang.launch_server --model-path .../qwen3-1.7b.d --host 127.0.0.1 --port 37679 --served-model-name qwen3-1.7b --context-length 16384 --enable-lora --lora-paths eternis-anonymizer-qwen3-1=.../eternis-anonymizer-qwen3-1.7b ...`
- Variant mode (request `qwen3.5-9b-bf16+bespoke-nimble-9b`):
  `python -m sglang.launch_server --model-path .../qwen3.5-9b-bf16.d --host 127.0.0.1 --port 40167 --served-model-name qwen3.5-9b-bf16 --context-length 16384 --enable-lora --lora-paths bespoke-nimble-9b=.../bespoke-nimble-9b ...`
- Raw captures: /tmp/opencode/nimble_argv_576739.txt (dense), nimble_argv_594144.txt + nimble_argv_594425.txt (variant).

## 4. Variant serve proof (workload-class: PEFT adapter on exact-base)

- Request `qwen3-1.7b+eternis-anonymizer-qwen3-1` (eternis/eternis_anonymizer_lora_Qwen3-1.7B_7jul_r16, r16 alpha16, classic Qwen3 targets k/v/down/up/q/gate/o_proj): 62.3 s cold spawn → completion reply 200.
- Wrong stem (full dotted dir name) → 404 with teaching (file_stem semantics: dots split names — request the stem, not the dir).

## 5. Bespoke-Nimble-9B attempt on this box — honest hardware wall

- Base: `~/.local/share/blazar/models/qwen3.5-9b-bf16.d` (Qwen/Qwen3.5-9B snapshot, 18 GiB, 16/16 files, HF_HUB_DISABLE_XET=1) adopted by boot preflight as sglang row `qwen3.5-9b-bf16`.
- Adapter: `bespokelabs/Bespoke-Nimble-9b` PEFT dir (r16 alpha32, targets include GDN in_proj_qkv/in_proj_z/in_proj_b/in_proj_a/out_proj + classic qwen modules), 173 MiB safetensors.
- Result: sglang crashed at layer weight-init (`MergedColumnParallelLinear.create_weights` under offloader wrap_modules) on both crash-loop attempts; gateway surfaced 502 `engine crashed` with full traceback, supervisor gave up cleanly, no hang, no slot leak (`blazar ps`: no models loaded).
- Verdict: LoRA wiring correct end-to-end (argv above proves emission for this exact spawn); failure is 18 GiB BF16 vs 8 GiB VRAM + ~5 GiB free RAM. Nimble class on 8 GB cards requires a quantized Qwen3.5-9B base (no official FP8/AWQ ships from Qwen today) or a bigger GPU.
- Typed-decision serving pattern (schema enum + logit_bias + logprobs) independently proven on qwen3-1.7b via gateway: receipts /tmp/opencode/nimble_pattern_receipts.json (json_schema strict enum answer + top_logprobs passthrough; logit_bias ±100 → forced token stream, warm 0.1 s).

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
