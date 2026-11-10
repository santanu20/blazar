# TurboQuant fork KV-quality receipts — 2026-11-07

Campaign: llama-perplexity A/B on the turboquant fork engine, identical to the
20261005-kv-quality campaign (same corpus, chunks 4, ngl 99, only cache types varied).

- model: `Qwen3.5-9B-Q4_K_M.gguf` (store name `qwen3.5-9b`)
- engine: `fork-AtomicBot-ai_atomic-llama-cpp-turboquant-8d935078-cuda`
  (built via `blazar engine build cuda --fork AtomicBot-ai/atomic-llama-cpp-turboquant@8d935078`)
- gpu: NVIDIA GeForce RTX 4070 Laptop GPU 8GiB / driver 595.99.02
- corpus: `../20261005-kv-quality/corpus.txt` (identical bytes)
- raw logs: one per rung, named `<engine>-<ctx>-<rung>.log`

## Results

| ctx | rung | ppl | +/- err |
|---|---|---|---|
| 2048 | f16 | 5.3651 | 0.22360 |
| 2048 | turbo3/turbo3 | 5.3912 | 0.22587 |
| 2048 | turbo4/turbo4 | 5.3754 | 0.22413 |
| 16384 | turbo3/turbo3 | 3.8744 | 0.05918 |
| 16384 | turbo4/turbo4 | 3.8798 | 0.05943 |

- turbo3 (+0.026) and turbo4 (+0.010) vs f16 at ctx 2048: inside the +/-0.22 error bars.
- turbo3 vs turbo4 at ctx 16384: delta 0.005, inside the +/-0.059 error bars.
- Cross-ctx ppl values are NOT comparable (window scale differs); compare within-ctx only.
- f16 baseline not run at 16384: it does not fit this 8GiB card next to the 5.2GiB weights.

## 32k finding (negative result)

`turbo3` at ctx 32768 on this card dies silently ~3s into the first chunk (process
exit, no CUDA error, no kernel OOM/Xid trace) — reproduced 2/2. No 32k turbo
receipt is claimed; the log stub is kept as evidence. Upstream q8_0/q4_0 32k
receipts (b11417, 20261005 campaign) are unaffected.
