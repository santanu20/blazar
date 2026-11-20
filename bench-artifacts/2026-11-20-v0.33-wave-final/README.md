# v0.33.0 wave-final baseline (2026-11-20)

# v0.33.0 wave-final baseline — analysis notes (2026-11-20)

Binary: release build at 0c3929d (`blazar 0.33.0`), model qwen2.5-0.5b-instruct q4_k_m,
RTX 4070 Laptop, AC powered, quiet box at launch (other actor's test suite drained first).
Raw cells: cells.jsonl (89 last-wins records); charts in plots/; headline report BENCHMARK.md.

## Headline (same-engine, same-model comparisons)

| lane | decode p50 | notes |
|---|---|---|
| direct llama.cpp fork (cuda) | 346.5 t/s | 17-cell median basis |
| blazar gateway, same engine | 329.8 t/s | ~4.8% aggregate overhead at 0.5b scale |
| ollama host, same engine class | 344.4 t/s | blazar within noise of ollama while adding pools/keys/observability |
| mistralrs v0.9.5 | 241.5 t/s (1 lane) | qwen2 multimodal GGUF unsupported by native loader (stderr kept) |

- TTFT p99 across 28 gateway cells: 3.5-24 ms; TTFT p50 single-digit ms.
- Prefill cold ~18.4k t/s; warm/cached lane ~2x (see cells, cache_reuse axis).
- VRAM efficiency: ~563 t/s/GiB direct (630 MiB peak).
- Cold load: 1.0-1.5 s (0.5b q4 on GPU); ollama cold parity 1.995 s.
- Concurrency ladder (C=1..8): zero failed requests across 4 conc cells (errors panel omitted
  by the renderer for exactly this reason); reshape sustained C=8 300s completed.
- KV ppl rungs: kv=f16 ppl 8.6685 +/- 0.4202 (2.3 s wall) — quantized rung comparisons in cells.
- Quality: 7 suites ran (code/instruct/multilingual/niah/reason/safety/schema), 3 scored rows,
  qc drift gate clean (no config knob drifted greedy output outside allowlist).

## Investigate-list outcomes

- per-GB-VRAM: measured above; first-class metric for future A/Bs.
- eviction cost vs saved capacity: reshape + idle-wake/evict-ladder lanes captured
  (lifecycle-cold-idle.svg); a dedicated eviction-cost A/B would need paired forced-evict
  runs — not in this campaign, flagged for the next one.
- streaming cancellation latency: not emitted by this campaign's lanes — needs a dedicated
  cancel-probe cell (abort mid-stream, measure engine slot-release). Flagged.
- SQLite mutex contention: not observable from campaign cells; gateway now ships
  blazar_queue_wait_* + admission rejection counters (wave #14) which bound the
  scheduling-side share of any future contention finding. A store-specific probe remains open.

## Honest failures recorded

- mistralrs cells: `multimodal GGUF architecture qwen2 is not supported by the native
  multimodal loader` — upstream capability gap, stderr preserved per cell.
- Ancient llama builds (2023.11.14-2, b5454, master-1043): several cells failed child-health
  on this model/flag set — per-cell stderr kept; these builds are bench-eligibility residue,
  not regressions (excluded set stamped in features.txt: 8 benchable, 4 excluded).

## Renderer fix riding this campaign

bench_charts.py hbar_svg crashed on a degenerate all-equal scale (0/0) — parity deltas were
all zero (perfect parity). Fixed with a unit-range guard; --render-only re-rendered the full
report from preserved cells without re-measurement.
txt Release binary 0c3929d, qwen2.5-0.5b-instruct q4_k_m, RTX 4070 Laptop, AC, quiet box.
Raw cells/plots/stderr kept locally (repo convention: summaries only). Reproduce via campaign_cmd.txt.
