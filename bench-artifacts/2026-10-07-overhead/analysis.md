# Relay-overhead analysis — 2026-10-07-overhead campaign

Question this campaign answers: **what does routing through the blazar gateway cost,
compared to driving the same engine directly?** Prior cold-start audit
(2026-10-05) showed a 107-vs-124 t/s gap (~13%) in campaign-context conditions;
an indicative warm C=1 spot-check on the 0.6B model showed ~0%. This campaign
re-measures both paths under one harness (bench_matrix v4), same engine build,
same model, same machine.

## Verdict

| Path (llama.cpp b11429-cuda, Qwen3.5-9B Q4_K_M) | Decode t/s | TTFT p50 | Notes |
|---|---:|---:|---|
| Direct, 1 slot x ctx 4096 | 41.1 | 121 ms | baseline |
| Direct, 1 slot x ctx 16384 | 41.5 | 122 ms | ctx-insensitive |
| Gateway (sandbox, default 4x65536 profile) | blocked | blocked | host RAM gate, see below |
| Gateway, nearest-topology C=1 (4-slot child) | 37.2 | ~193 ms p99 tail | 6-9% aggregate gap incl. slot skew |

1. **Single-stream matched-cell overhead: NOT MEASURED — environment-blocked.**
   The sandbox admission gate requires ~7,731 MiB host MemAvailable
   (model 5,366 + safety headroom); this 13.3 GiB-RAM box floor sat at
   7,092-7,724 MiB across four attempts (desktop + real daemon resident).
   The gate is correct to refuse; retrying later when RAM is free costs ~5 min
   via harness resume (it retries only failed cells). The blocked cells are
   recorded as failures in cells.jsonl — nothing is silently missing.
2. **Nearest-topology C=1 comparison (direct np=1 vs gateway 4-slot child):
   ~6% aggregate decode gap.** This is an UPPER bound on relay overhead — it
   also contains the 4-slot child's per-stream batching cost and the gateway's
   admission bookkeeping. The indicative 0.6B warm spot-check (~0%) bounds the
   pure relay copy path from below.
3. **Concurrency "gap" is slot topology, not relay cost.** The direct lane
   spawns `-np C` (slots = C); the gateway's tuned profile fixed the child at
   4 slots x ctx 65536 for every level. At C=8: direct 130 vs gateway 86 t/s
   measures 8-slot vs 4-slot batching, not the relay. Tellingly, the gateway
   WINS tail latency at high concurrency: TTFT p99 469 ms vs direct 941 ms
   (C=8) and 267 vs 412 ms (C=4) — admission + queueing smooths burst tails.
4. **Greedy transparency: 7/20 byte-exact at temp 0, text-similarity ratio
   ~1.0.** Divergence is multi-slot batching numerics in the engine (4-slot
   child vs 1-slot direct reference), not gateway translation drift — the
   direct determinism lane itself is 20/20 exact. On a slot-matched child the
   prior campaigns show byte-exact forwarding.
5. **Actionable tuning finding:** the default profile's fixed 4-slot choice
   caps C=8 aggregate throughput on this box. Workload-aware slot reshaping
   (the adaptive-reshape machinery) or a bench-tuned C=8 profile should
   recover the 8-slot aggregate when saturated load is expected. This is a
   profile/tune follow-up, not a gateway defect.

## Environment-blocked and failed cells (preserved, not hidden)

- `blazar b11429-cuda config=default` and `config=single-stream`: 4 attempts,
  each refused by the sandbox MemAvailable gate (best attempt 7 MiB short).
- `direct v0.9.4 (mistral.rs) ctx 4096/16384 x np1, 16384 x np4`: model +
  BF16 vision projector exceeds this 8 GiB card by 133 MiB — deterministic
  environment bound, correctly recorded.
- `greedy_gw v0.9.4`: same RAM gate.

## Reproducibility receipt

| Field | Value |
|---|---|
| Date | 2026-10-07 |
| Blazar binary | repo release build of main @ 424dce3, `blazar 0.22.0`, sha256 6b102a67...f9c66a53b |
| Engine | llama.cpp b11429-cuda (llama-server sha256 prefix 5a2b208943ca0491) |
| Model | Qwen3.5-9B-Q4_K_M.gguf (5,366 MiB) |
| GPU | RTX 4070 Laptop 8 GiB (nvidia-smi driver 595.91.07; inventory cell reads 580.173.02 — both stamped) |
| Host | i7-14650HX, 13.3 GiB usable RAM, kernel 7.0.0-38-generic, AC power |
| Harness | scripts/bench_matrix.py, HARNESS_VERSION 4 |
| Command | `python3 scripts/bench_matrix.py --model qwen3.5-9b --providers direct blazar --conc-sweep 1,2,4,8 --skip-ppl --skip-tools --skip-quality --skip-media --skip-features --skip-ctxcurve --skip-reshape --skip-idle --skip-variants --blazar-bin ./target/release/blazar --artifacts-dir bench-artifacts/2026-10-07-overhead --md bench-artifacts/2026-10-07-overhead/report.md` |
| Raw cells | `cells.jsonl` in this directory (append-only, last-wins per cell key) |
| Follow-up | re-run the same command when MemAvailable > 7.8 GiB: resume retries only the blocked blazar/greedy_gw cells (~5 min) |
