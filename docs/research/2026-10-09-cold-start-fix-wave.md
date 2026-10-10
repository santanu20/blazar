# sglang cold-start fix wave — receipts (2026-10-09)

Scope approved by user: cut `blazar run qwen3-1.7b` cold start with zero quality
loss, then a wider e2e perf sweep to find and fix similar issues. Raw artifacts:
`bench-artifacts/20261009-sglang-cold-start/` (JSONL probe logs, per-run child
stderr with timestamps, driver + runner scripts for reproduction).

## Environment

| Axis | Value |
|---|---|
| Machine | Lenovo Legion 5 16IRX9, RTX 4070 Laptop 8188 MiB |
| OS / driver | Linux; driver-managed CUDA via engine venv |
| Blazar | dev build deployed 2026-10-09 14:51 IST (systemd `blazar`) |
| Engine | sglang 0.5.21 (pip venv, persistent triton/inductor caches under `engines/sglang-0.5.21/cache/`) |
| Model | qwen3-1.7b BF16 safetensors (3.8 GiB), routed llamacpp→sglang |
| Probe harness | `sglang_instr.py` — replicates the daemon's exact argv on a scratch port; temp-0 quality sha, 6416-token prefill, 256-token decode tps |

## Cold-start anatomy (before)

Daemon timeline 2026-10-09 06:56 spawn: 2.2 s admission+profile → child exec →
ready ≈ 30.1 s; first request total 33.8 s. Engine self-reported (from
`/get_server_info`): `load_weight` 1.19 s, `kv_cache_allocation` 0.29 s, decode
CUDA-graph capture 0.81 s (bs [1,2,4,8]), **prefill CUDA-graph capture 7.19 s
(35+ aggregate-token buckets up to 2048+)**, `scheduler_e2e` 12.51 s,
`tokenizer_e2e` 18.85 s (whole launch window: 2× torch import + scheduler
subprocess + init — not tokenizer work).

Controlled A/B on the scratch harness (identical probe sequence each run, temp-0):

| Run | Prefill graph config | ready_t | prefill capture | long-prefill (6416 tok) | decode tps | quality sha |
|---|---|---|---|---|---|---|
| baseline v1 (cold JIT) | sglang default (42 buckets) | 41.2 s | 17.1 s | — | — | — |
| qbase (warm JIT, round 2) | sglang default | 32.2 s | 9.9 s | 1.267 s | 66.3 | fd6bdb78f76c38e1 |
| cap512 | `--cuda-graph-max-bs-prefill 512` | 31.2 s | 8.0 s | 1.276 s | 64.0 | fd6bdb78f76c38e1 |
| list3 | explicit 3-size list | 27.2 s | 5.2 s | 1.273 s | 65.0 | fd6bdb78f76c38e1 |
| qlist3 [256,1024,2048] | explicit 3-size list | 28.3 s | 5.5 s | 1.239 s | 66.2 | fd6bdb78f76c38e1 |
| two512 [512,2048] | explicit 2-size list | 27.2 s | 5.1 s | 1.271 s | 66.1 | fd6bdb78f76c38e1 |
| cfg2 (config JSON) | prefill.bs [256,2048] | 27.2 s | 5.1 s | 1.252 s | 66.1 | fd6bdb78f76c38e1 |

Quality sha identical across all six runs → **the graph-flag axis is
temp-0-output-neutral**; explicit small bucket lists cut prefill capture
~5 s and total ready ~5 s with no prefill-latency or tps cost.

## Honest floor

Child exec → ready cannot go below ~27–28 s on this machine/engine today:
`launch_server` imports ~8 s + scheduler-subprocess torch re-import ~5.3 s +
LoRA JIT/KV-kernel ~3–4 s (upstream `--enable-lora` cost) + prefill graphs
~5.5 s + decode ~1 s + warmup ~1.5 s. The planning target of ≤21 s was not
achievable; deviation reported here rather than hidden. Remaining local lever
is REPL pre-spawn (below), which moves the wait off the user's first turn.

## Changes shipped

1. **Derived prefill graph buckets** (`crates/blazar-core/src/profile.rs`): on
   engines exposing the 0.5.21 split graph surface, all VRAM tiers now emit
   `--cuda-graph-bs-prefill <chunk/8> <chunk/2> <chunk>` sized from the
   effective chunked-prefill size (default 2048 → [256, 1024, 2048]), pinning
   `--chunked-prefill-size` when unset. User pins (`extra_args` or tuning
   knobs) suppress derived emission; the legacy 0.5.19 flag surface keeps
   byte-identical behavior. Knob mapping on the split surface:
   `cuda_graph_bs`→`--cuda-graph-bs-decode`, `cuda_graph_max_bs`→
   `--cuda-graph-max-bs-decode` (combined flags no longer exist upstream).
   6 new unit tests + `sglang_flags_0521` fixture; suite 2073/2073.
2. **REPL pre-spawn** (`crates/blazar-cli/src/main.rs run_repl`): at banner
   time the CLI fire-and-forgets `POST /api/warm {"model", "wait": true}` —
   standard low-priority admission, coalesces with the first real turn, silent
   on error. Live proof: banner 09:43:59.962 → spawn 09:43:59.971 (9 ms) →
   ready 09:44:26.38 — **banner→ready 26.4 s with zero user input**; a
   human's first message lands warm.
3. **Warm-peg surface ladder** (`crates/blazar-runtime/src/supervisor.rs`):
   the post-spawn peg no longer assumes chat. `--reranking` children are
   probed on `/rerank`; other `--embeddings` children on `/v1/embeddings`
   then `/v1/systemone`. Served-model stamping (`--served-model-name` /
   `--model` echo) replaces flat names so mlx children stop treating the
   request model as a hub repo id. Result: the every-few-minutes 500/404 WARN
   loop (laya, bge-small-en, bge-reranker-v2-m3, mlx qwen2.5-*-4bit) is gone;
   each lane now logs `warm-peg ok` on its real surface.
4. **mlx offline hardening** (`crates/blazar-runtime/src/engine_impl.rs`):
   `HF_HUB_OFFLINE=1` pinned into mlx child env (user-set value survives).
   Bare-name requests that previously did a live hub fetch → 401 → 404 now
   fail fast in ~50 ms with the local cache-miss message. Isolation test
   (5 requests + 40 s idle): child survives — no crash-risk regression.

## Post-fix numbers (daemon path, live)

- Cold `/api/warm {"wait":true}` for qwen3-1.7b: **31.5 s** total (was 33.8 s
  first-request; child exec→ready now 26–28 s depending on JIT cache state).
- Live child argv verified: `--cuda-graph-bs-prefill 256 1024 2048
  --chunked-prefill-size 2048 --kv-cache-dtype fp8_e5m2
  --cuda-graph-max-bs-decode 256 --mem-fraction-static 0.681`.
- Tier-B replica (exact daemon argv incl. fp8 KV): prefill capture 5.85 s over
  exactly the 3 derived buckets; decode 3.78 s (36 sizes ≤256);
  `tokenizer_e2e` 21.05 s. Its tps was not persisted by the harness (driver
  omission); perf axis proven at Tier A above.

## Quality verdict (no loss from this wave)

- Graph-config axis: temp-0 sha identical across 6 controlled runs (above).
- fp8-KV difference seen on daemon spawns is the **pre-existing, designed
  Tier-B VRAM-pressure posture** (f16 KV would not fit alongside the 667 MiB
  residue on the card at spawn time); daemon teaches it at spawn and exposes
  `models.<name>.sglang.kv_cache_dtype` to override. Replica of the exact
  Tier-B argv reproduces the daemon text char-for-char for the first 100
  chars; late-token divergence tracks differing radix-cache/batch histories —
  full-token temp-0 equality across differing cache histories is not
  achievable on continuous-batching engines and was never the baseline.

## Reproduce

```bash
# scratch harness (replicates daemon argv; per-run tagged logs + JSONL probes)
cd bench-artifacts/20261009-sglang-cold-start
python3 sglang_instr.py            # env: INSTR_TAG, INSTR_PORT, INSTR_EXTRA_JSON
bash run_ab3.sh                    # round-3 A/B trio
# daemon path
curl -s -X POST 127.0.0.1:11435/api/warm -H 'content-type: application/json' \
  -d '{"model":"qwen3-1.7b","wait":true}'
```

## Known WARNs intentionally untouched this wave

- `child_auth` fallback `--api-key` visible in `/proc/<pid>/cmdline` (needs
  keyfile support for the sglang lane).
- Spec-draft GGUF refusal teaching (Qwen3-0.6B draft is GGUF; sglang EAGLE
  needs safetensors) — informational, correct behavior.
- `engines re-rooted` notice on daemon start.
