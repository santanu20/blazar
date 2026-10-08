# Gateway relay performance analysis — 2026-10-08

Question: the 2026-10-07 overhead campaign measured a ~5-6% single-stream
gateway residual (non-stream +17.9 ms / +5.5% per request; streaming
TTFT +1.5 ms fixed, ITL +0.14 ms/token). This note attributes the cost
at the code level and records the engineering verdict.

## Measurement context

- Release build of the integrated tree (all feature branches merged),
  Qwen2.5-0.5B-Instruct child, 2 slots, full GPU offload, warm cache,
  3 warm-up + 20 measured non-stream requests (max_tokens 256, temp 0,
  seed 42) and 12 streaming runs per path; `perf record -g` at 499 Hz
  over a 32-request streaming burst.
- Direct path = same child llama-server, same parameters, bypassing the
  gateway.

## Attribution chain (three passes over the source)

1. **The relay is already frame-passthrough.** The streaming path
   (`proxy.rs` relay, ~1403-1434) forwards child bytes untouched via
   `stream::iter(vec![Ok(bytes)])`; think-suppression and model-restamp
   transform only on their own lanes. The hypothesized
   "parse-transform-re-emit per chunk" does not exist on the common path.
2. **The cache tap is bounded telemetry, not a copy hazard.** `CacheTap`
   is an 8 KiB tail ring feeding cache-hit receipt accounting
   (`/api/explain` evidence), not the semantic cache, and not an
   unbounded per-request buffer.
3. **Sentinel does not parse on the hot path.** `SentinelFeed::bytes` is
   a `try_send` into the analyzer task; serde analysis happens off the
   relay, and the whole observer is config-gated (`sentinel = false` →
   inert feed, covered by an integration test).

What remains per chunk: a refcount clone + channel try_send (~100 ns),
the bounded tap append (~150 ns), three mutex lock/unlock on
optional-filter cells (~150 ns when inactive), and — the dominant term —
the scheduler hop structure: child hyper → gateway task → combinator
chain → axum body → client, versus the direct child → client. The perf
profile agrees: `tokio runtime Context::run`, allocator `_int_free`,
`drop_glue<serde_json::Value>`, and `memmove` are all present but
spread thin; no single gateway function is hot.

## Quantified verdict

The five lock acquisitions per chunk total ~250 ns: 0.009% of the
inter-token budget at 340 t/s (0.5B) and 0.0006% at 42 t/s (9B). A
lock-trim or ownership-restructure of the filter cells would touch
behavior-bearing lanes (think suppression, model restamping, usage
accounting) for a reward below measurement noise. Rejected:
correctness risk without measurable latency recovery.

The residual is structural (one extra task hop per chunk, one extra
full-body JSON round-trip for non-stream responses that must be
inspected before forwarding) plus same-box contention (engine, daemon,
and desktop share this laptop). No safe code change at the current
architecture level recovers a measurable fraction without losing
product behavior (sentinel quality observation, cache receipts,
dialect translation) or re-architecting the body pipeline.

## Revisit conditions

- Batch-heavy serving profile (many concurrent streams) where per-chunk
  fixed costs amortize differently — re-measure before touching code.
- A frame-level passthrough prototype that skips the combinator chain
  for untransformed lanes could be evaluated against a strict
  byte-parity + conformance suite; expected reward is the per-token
  scheduler hop, risk is subtle framing regressions on split chunks.
- If a future feature moves inspection off the forwarded body entirely
  (e.g. sentinel sampling), re-profile from scratch.

## Reproducibility

Numbers and method: `bench-artifacts/2026-10-07-overhead/` campaign
(receipt in `analysis.md`) plus the 2026-10-08 profiling pass described
above (release build, perf 499 Hz). Box: RTX 4070 Laptop 8 GiB,
i7-14650HX, AC power, driver 595.91.07.
