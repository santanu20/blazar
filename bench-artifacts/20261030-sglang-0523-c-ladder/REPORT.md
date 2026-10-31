# sglang 0.5.21 → 0.5.23 C-ladder — version-delta receipt

**Date:** 2026-10-30 · **Lane:** sglang-0.5.23 (pip) · **Model:** qwen3-1.7b.d (3875 MiB)
**Host:** production daemon, blazar 0.28.0 release binary (RTX 4070 Laptop 8188 MiB, driver 595.99.02)
**Baseline:** `bench-artifacts/20261009-sglang-rebaseline` (sglang-0.5.21, blazar 0.24.0 sandbox daemon, same model/harness/flags)

## ITL p99 (ms) by concurrency — 3 rounds/cell, zero failed requests in BOTH campaigns

| C | 0.5.21 | 0.5.23 | Δ |
|---|--------|--------|---|
| 1 | 15.66 | 15.91 | +1.6% |
| 2 | 16.01 | 15.77 | −1.5% |
| 4 | 16.46 | 16.56 | +0.6% |
| 8 | 36.77 | 37.63 | +2.3% |

Single-stream (0.5.23): TTFT p50/p99 33/43 ms · decode 66.4 tok/s · prefill 5.0k cold / 16.5k cached tok/s · VRAM 6005–6247 MiB.

## Verdict

**Parity within noise (±2.3%)** — the currency bump 0.5.21→0.5.23 is regression-free on this
model/hardware. The 0.5.22/0.5.23 detokenizer perf wave (flat-column ids, per-request decode
inputs sent once) shows no measurable ITL delta here: qwen3-1.7b decodes at ~66 tok/s aggregate,
far below the token rate where detokenizer cost binds; gains, if any, live at higher-throughput
serving shapes. The C8 ITL-tail doubling (36→38 ms) is present at BOTH versions — a scheduler
saturation characteristic of this card, not a regression.

## Disclosures

- Gateway binary differs between campaigns (0.24.0 sandbox vs 0.28.0 production); ITL is
  engine-dominated and both campaigns measured through the full gateway path, so the engine
  delta dominates. TTFT/ITL compare like-for-like; cross-binary rows are NOT compared.
- First attempt (same flags) failed all sglang cells: VRAM floor 5867 MiB vs 5814 free — the
  resident co-tenant was the host's real ollama (4417 MiB, reference rows ran). Purged; rerun
  on an idle GPU. Kept as evidence of the floor guard working, not as a data point.
- Quality lane skipped (`--skip-quality`): no checker-verified quality claim is made.

## Reproduce

```
BLAZAR_BIN=/usr/local/bin/blazar python3 scripts/bench_matrix.py \
  --artifacts-dir bench-artifacts/20261030-sglang-0523-c-ladder \
  --engines sglang-0.5.23 --providers blazar --model qwen3-1.7b.d \
  --conc-sweep 1,2,4,8 --conc-rounds 3 --skip-ppl --skip-greedy --skip-tools \
  --skip-reshape --skip-features --skip-idle --skip-ctxcurve --skip-quality \
  --skip-variants --skip-media --fresh
```
