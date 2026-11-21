Title: Profile and reduce gateway saturation overhead

Priority: P1 | Wave: 3 | Backlog: B05 | Effort: M | Depends: 004

## Problem

The 2026-11-20 campaign measured gateway-vs-direct at matched settings
(conc-blazar vs conc-direct legs in
bench-artifacts/2026-11-20-v0.33-wave-final/cells.jsonl). Overhead at low
concurrency is modest; saturation shows the largest gap. Exact ratios must be
reproduced from the cells before any fix is judged: numbers quoted outside
bench-artifacts receipts are not evidence.

## Approach

1. Reproduce the conc ladder from the 2026-11-20 campaign; extract
   gateway/direct ratios per concurrency from cells.jsonl.
2. Use the stage timing from issue 004 to name the largest measured
   contributor before touching code. Candidates by suspicion only: middleware
   layers, streaming body wrappers, body parsing/buffering, queue-lock
   contention, store mutex wait, per-request allocation, metrics
   serialization.
3. Fix the measured bottleneck; add a conc-lane regression gate so gateway
   overhead has a budget.
4. Never alter inference parameters to flatter gateway numbers; any such
   change must be identified in the receipt.

## Acceptance criteria

- [ ] Ratios reproduced from receipts, not quoted
- [ ] Stage-level evidence names the dominant contributor
- [ ] C=8 overhead measurably improved against the reproduced baseline
- [ ] Regression gate enforces an overhead budget in the bench suite
- [ ] Receipts identify any inference-parameter differences

## Files

crates/blazar-gateway/src/ (stage evidence from 004), scripts/bench_matrix.py
