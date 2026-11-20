# Probe campaign: cancel latency, eviction cost, store contention

Run: 2026-11-20 · blazar 0.33.0 @ 271e267 (release build) · model qwen2.5-0.5b-instruct-q4_k_m (468 MiB) · engine fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda · RTX 4070 Laptop (8 GB) · linux 6.x, AC powered, box quiet (no parallel cargo/tests).

Lanes (`--probes cancel,evict,store --probe-reps 5`, reference decode/cold/tools/reshape cells in `cells.jsonl` were measured in the same campaign earlier in the day and resumed-skipped on the final pass):

## Cancel latency (`probe-cancel`)

Streaming chat aborted mid-stream (>= 8 content chunks seen), wall time from `resp.close()` to the engine slot reporting `in_flight == 0`.

| path | p50 | p90 | max |
|---|---|---|---|
| gateway (OpenAI SSE) | 1.3 ms | 1.8 ms | 2.0 ms |
| direct llama-server | 1.0 ms | - | - |

`tokens_seen_median` = 8. Gateway cancel tax over a bare engine is ~0.3 ms: client disconnect reaches the child slot immediately; no queued-drain wait.

## Eviction cost (`probe-evict`)

Victim auto-picked: `gemma3-270m` (278 MiB gguf). `max_loaded_models = 1`.

- explicit `POST /api/evict` teardown wall: **310.2 ms**
- cold first-token (fresh spawn, cold page cache): 8083.3 ms
- pressure first-token (A resident, B's admission evicts A first): median 6654.7 ms
- `evict_cost_ms` = pressure − cold = **0 (clipped)** — the pressure path was *faster* than the cold path because the second pass rides the OS page cache.

Caveat (by design, recorded honestly): the cold-vs-pressure delta is contaminated by disk-cache warmth at this model size; treat `evict_wall_ms` (310 ms) and the post-conditions (A gone, B resident, `gpu_after` 5534 MiB with B + CUDA context) as the reliable eviction-cost signals, and see the pressure/cold raw walls for load-shape comparison.

## SQLite store contention (`probe-store`)

Sandbox store copy (WAL, synchronous=FULL, 1.16 MiB), k = 1/4/16 independent connections, ops cycling upsert-with-commit / SUM aggregate / small scan:

| k | upsert p50 | upsert p99 | SUM p50 | scan p50 | busy retries |
|---|---|---|---|---|---|
| 1 | 0.578 ms | 1.438 ms | 0.025 ms | 0.013 ms | 0 |
| 4 | 0.733 ms | 55.363 ms | 0.033 ms | 0.018 ms | 0 |
| 16 | 0.690 ms | 504.156 ms | 0.018 ms | 0.015 ms | 0 |

Verdict for the "measure SQLite mutex contention before changing persistence" question: median write cost is flat (~0.6-0.7 ms) at every concurrency; contention shows only as a p99 write tail (55 ms @ 4 writers, 504 ms @ 16) absorbed by busy-timeout, reads stay tens-of-µs. No persistence redesign is justified by this data; if the write tail ever matters, serialize/batch writers (single flusher) rather than swapping the store.

## Reproduce

```
python3 scripts/bench_matrix.py --model qwen2.5-0.5b-instruct \
  --providers --skip-ppl --skip-greedy --skip-features --skip-conc --skip-idle \
  --skip-quality --skip-variants --skip-media --skip-ctxcurve \
  --probes cancel,evict,store --probe-reps 5 \
  --artifacts-dir bench-artifacts/<dir> --blazar-bin <release blazar>
```

Raw receipts: `cells.jsonl` (rows `provider = probe-cancel|probe-evict|probe-store`), `campaign_cmd.txt`, `summary.txt`, `benchmark.md`.
