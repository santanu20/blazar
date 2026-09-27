# spec_auto_ngram default decision — dense vs ngram-lookup fallback

- **Date:** 2026-09-27 · **Blazar:** feature/frontier-roi (B1-δ) · **Engine:** b11202-cuda (llama.cpp, ubuntu-cuda-12.8-x64)
- **Model:** qwen2.5-0.5b-instruct Q4_K_M (no catalog draft pair → auto falls back to ngram)
- **Setup:** isolated XDG daemon (port 11499), greedy (seed 42, temp 0), `num_predict` 96, 6 runs/workload interleaved, one warm-up gen excluded per mode. Mode = daemon default (`spec=auto`, `spec_auto_ngram=true`) vs `BLAZAR_SPEC_AUTO_NGRAM=false`; engaged mode verified via `/proc/<pid>/cmdline` (`--spec-type ngram-simple` + `--lookup-cache-dynamic …​.lcache` present/absent).

## Result (mean / median eval tok/s)

| Workload | ngram fallback | dense | delta |
|---|---|---|---|
| Repetitive JSON-lines continuation | 642.87 / 679.96 | 334.79 / 350.29 | **+92.0%** |
| Non-repetitive (6 distinct questions) | 333.54 / 339.60 | 343.98 / 354.47 | **−3.0%** |

Repetitive runs also show the dynamic lookup cache warming across requests (428 → 740 tok/s over the six runs).

## Decision

`spec_auto_ngram` **stays `true`** (default). The fallback's upside on patterned traffic (agent logs, structured continuation — the exact traffic the fallback targets) is ~1.9×, while the worst observed cost on fresh prose is −3.0%, inside the ±5% acceptance band set before the run. Raw per-request rows: `cells.jsonl`; machine summary: `summary.json`.
