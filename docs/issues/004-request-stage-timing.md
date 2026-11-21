Title: Request-stage timing in the gateway

Priority: P0 | Wave: 1 | Backlog: B05.1 | Effort: S-M | Depends: -

## Problem

The gateway records end-to-end ttft/itl quantiles
(crates/blazar-gateway/src/histogram.rs) and admission wait in the queue
(crates/blazar-gateway/src/queue.rs), but there is no per-stage breakdown of
where non-model time goes. Without it, gateway-vs-direct overhead (issue 009)
cannot be attributed to auth, body handling, admission, routing, dispatch, or
streaming wrappers.

## Approach

1. Define stage boundaries: request received, auth, body handling, admission
   queue, routing, engine dispatch, first token, last token.
2. Record stage durations independently per request alongside the existing
   histograms.
3. Expose p50/p95/p99 per stage through the existing metrics surface.
4. Prove zero measurable hot-path cost: A/B bench with and without
   instrumentation before merging.

## Acceptance criteria

- [ ] Per-stage durations recorded for every request
- [ ] p50/p95/p99 per stage exposed through the metrics surface
- [ ] Stage table documented in docs
- [ ] Bench A/B shows no measurable latency regression from instrumentation

## Files

crates/blazar-gateway/src/histogram.rs, crates/blazar-gateway/src/queue.rs,
crates/blazar-gateway/src/lib.rs
