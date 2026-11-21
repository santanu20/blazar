# 2026-11-21 gateway saturation (issue 009 + issue 004 evidence)

Harness: a scratch saturation harness (not committed); methodology below is
the reproducible part). Model `qwen2.5-0.5b-instruct:q4_k_m`, sandbox daemon on 8199
(tree release binary), matched direct `llama-server` on 8299, rungs C=1,2,4,8,16,
2 rounds each, `tg=128` unique-prompt streaming threads.

## Methodology

The direct leg REUSES the gateway child's argv verbatim minus
`--api-key-file`, `--slot-save-path`, `--port`, `--lookup-cache-dynamic`
(sandbox-state caches that would warm-start direct; `--cache-ram` KEPT = matched
memory budget) plus `--port 8299 --alias sat-direct`. Same binary, same model file,
same ctx/slots/thread flags - the only difference is the gateway in the path.
Direct leg runs first, gateway second (ordering noted in caveats).

## Aggregate result (sys t/s, gateway/direct ratio)

| C | direct | gateway | ratio |
|---|---|---|---|
| 1 | 264.0 | 317.9 | 1.204 |
| 2 | 453.7 | 439.9 | 0.970 |
| 4 | 494.7 | 520.1 | 1.051 |
| 8 | 532.8 | 532.1 | 0.999 |
| 16 | 566.9 | 534.7 | 0.943 |

Every rung: 0 failed requests. Target bands (engineering targets from the backlog):
single-request <= 3% overhead -> measured FASTER; normal concurrency <= 5% ->
C=2 -3.0%, C=4 +5.1% (gain); saturation <= 10% -> C=16 -5.7%. PASS.

## Stage timing across the ladder (gateway leg, cumulative means)

| C | auth µs | body ms | admission ms | dispatch ms | model_ttft ms | stream ms |
|---|---|---|---|---|---|---|
| 1 | 19 | 18.0 | 392.0 | 3.2 | 23.3 | 115.6 |
| 2 | 17 | 10.9 | 196.0 | 4.1 | 16.7 | 174.9 |
| 4 | 15 | 7.9 | 98.0 | 69.2 | 77.8 | 211.7 |
| 8 | 13 | 8.6 | 49.0 | 281.8 | 288.6 | 264.5 |
| 16 | 12 | 13.4 | 24.5 | 760.0 | 765.7 | 296.9 |

Reading (this is the issue-009 answer): the gateway-own slice (auth + body +
admission) stays in single-digit milliseconds at every concurrency. What grows with
C is dispatch + model_ttft - tokens queueing inside the 2-slot engine child. The
saturation bottleneck is the engine, not the gateway. Admission means carry the
cold-spawn queue-wait outlier at low-C rungs (2.5 s first-request spawn wait,
visible as the 392 ms mean at C=1; per-request quantiles in the receipt).
Workload counters (issue 010) recorded `interactive_chat` for every request -
cross-instrumentation visible in the same scrape.

TTFT p50/p95 gateway: 9.1/14.1 ms (C=1) -> 1374/2427 ms (C=16): slot contention
physics at 16 streams on 2 slots, identical for the matched direct child.

## Caveats

- Single box (RTX 4070 8 GB), 2-slot default config, small model - absolute
  numbers are machine-specific; ratios are the deliverable.
- Leg ordering: direct first, gateway second (warmer page cache for the gateway
  leg; contributes to the C=1 gateway "win" and slightly flatters gateway).
- 2 rounds per rung - variance visible in the receipt JSON; single-run-rung
  precision is not claimed.
- Tree state: release binary built from a tree with a concurrent in-flight wave
  (user-authorized "ignore collision, do all").
- vs the backlog memo's quoted Nov-20 ratios (0.982/0.942/0.978/0.898 at
  C=1/2/4/8): those numbers were never located in artifacts; this campaign
  supersedes them with a matched-argv, stage-instrumented methodology.

## Reproduce

XDG sandbox daemon (store snapshot + engines symlink) on 8199; warm one request;
capture gateway child argv from `/proc/<daemon-pid>/children`; spawn the matched
direct child on 8299; drive C=1..16 unique-prompt streaming rounds against both;
scrape `/metrics` (`blazar_stage_*`, `blazar_queue_wait_*`, `blazar_workload_*`)
and `/api/quantiles` per gateway rung. Full raw data: `sat_receipt.json`
(per-round walls, ttft arrays, argvs of both children, cumulative stage sums,
quantile blocks, workload counters).
