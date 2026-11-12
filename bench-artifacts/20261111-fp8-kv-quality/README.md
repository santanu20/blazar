# FP8 KV quality receipts — 2026-11-11

Campaign: llama-perplexity A/B for the OCP-FP8 E4M3 KV cache grade
(upstream PR #28392, first shipped in b12284), same protocol as the
20261005 / 20261107 campaigns: fixed corpus, chunks 4, ngl 99, only
cache types varied.

- model: `qwen2.5-0.5b-instruct-q4_k_m.gguf` (store name `qwen2.5-0.5b-instruct`,
  qwen2 arch - deliberately non-MLA: the compile gate drops fp8 pins on
  MLA/DeepSeek-class models where upstream excludes them)
- engine: `b12284-cuda` (ubuntu-cuda-12.8-x64), `NATIVE_FP8 = 1`,
  `FA_QUANTS = ... f8_e4m3-f8_e4m3` in the boot line (engagement proof in
  each f8 log)
- gpu: NVIDIA GeForce RTX 4070 Laptop GPU 8GiB
- corpus: `../20261005-kv-quality/corpus.txt` (identical bytes)
- raw logs: one per rung, named `<ctx>-<rung>.log`

## Results

| ctx | rung | ppl | +/- err |
|---|---|---|---|
| 2048 | f16 | 7.0511 | 0.32082 |
| 2048 | f8_e4m3/f8_e4m3 | 7.1867 | 0.32603 |
| 16384 | f16 | 4.6458 | 0.07021 |
| 16384 | f8_e4m3/f8_e4m3 | 4.7867 | 0.07270 |

- ctx 2048: +0.136 - inside the +/-0.32 error bars (parity at short ctx).
- ctx 16384: **+0.141 - outside the +/-0.07 error bars (~2x)**. A real,
  measurable long-context quality cost, consistent with FP8's 2-byte ->
  1-byte dynamic-range squeeze stressing long-window retrieval.
- Cross-ctx ppl values are NOT comparable (window scale differs).

## Decision recorded

`f8_e4m3` stays **pin-only**: typed in the `cache_type` vocabulary, planned
as 1 byte/element, MLA-gated and engine-probe-gated at compile - but the
automatic fit ladder never selects it (registry rows above are the honest
basis: short-ctx parity, long-ctx cost). Capacity-constrained operators
trade ~3% ppl at 16k ctx for 2x KV capacity deliberately.
