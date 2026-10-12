# E2E perf audit — 2026-10-12

Question: after the cold-start wave and the 2026-10-08 relay analysis,
where does the remaining single-stream gateway overhead actually go,
and can any of it be recovered without touching behavior? This audit
profiles the whole gateway path end-to-end under a streaming decode
load, attributes every microsecond it can, ships the two fixes that
were provably free, and re-runs the campaign to confirm no regression.

## Measurement context

- Two campaigns, identical matrix: qwen2.5-0.5b-instruct (Q4_K_M),
  llama.cpp b11539-cuda, direct engine vs gateway, concurrency sweep
  1/2/4/8, greedy 20-run determinism probes (receipts in
  `bench-artifacts/20261012-e2e-perf-audit/` pre-fix and
  `bench-artifacts/20261012-e2e-perf-audit-postfix/` post-fix).
- Pre-fix binary sha256 `e24e1876a6168a5a…`, post-fix
  `45a486452ff12794…` (both recorded beside their artifacts).
- CPU attribution: a sandboxed daemon under a repeating greedy
  streaming chat load (1024-token essay completions), profiled with
  `perf record -F 499 --call-graph dwarf` for ~50 s, with
  `/proc/<pid>/task/*/{stat,sched}` deltas taken around each run to
  split CPU time from context switches and off-CPU wait.

## What the 210 µs/token residual is made of

The 2026-10-07 campaign shape reproduced: C1 gateway decode ~6% under
direct at matched slots, gateway dominant at C4+ (re-confirming the
+45%/+85% wins), greedy gateway-transparency 14/20 exact with ratio
0.941 (the misses are known multi-slot numerics, not gateway logic).

Attribution of the ~210 µs/token single-stream gap:

| Slice | Share | Where |
|---|---|---|
| Daemon CPU | ~96 µs/token | stime-dominated: kernel loopback double-hop (nft/tcp/skb) ~35-40%, tokio+hyper+allocator ~35%, per-frame observation ~10-15% (off-relay), relay closure ~5-8 µs |
| Off-CPU wakeup | ~115 µs/token | structural: one extra task hop per chunk (matches 2026-10-08) |

The surprise was in *where inside* the daemon CPU went: the SSE
think-suppression filter — not the relay — held the largest single
code-level share (`SseThinkFilter::feed` 1.48% plus a ~6% serde /
`BTreeMap` cluster behind it).

## Root cause: suppression parsed every frame it did not touch

`suppress_think` arms for every plain chat stream on every engine
(asking-for-thinking is the only opt-out), and the filter's `feed`
parsed *every* SSE frame — UTF-8 lossy conversion, line drain, full
`serde_json` round trip, think-marker scan — even when the chunk was
an ordinary token delta that could never contain a marker.

## Fixes (both byte-identical by construction)

1. **Identity fast path in `SseThinkFilter::feed`** (`translate.rs`).
   When the filter carries no partial state and the chunk is
   line-complete, it now scans the raw bytes for marker risk
   (`<t`, `</`, `<"`, `\u` byte pairs — covering every think-marker
   opening, every proper-prefix tail, and `\u003c`-style escapes) and
   returns the chunk untouched when none is present. A real 384-token
   gateway dump (97 KB) contained zero such bytes: the fast path is
   the common case on real traffic. Marker-bearing or partial-line
   chunks take the original path unchanged. Pinned by five new unit
   tests (byte-exact passthrough, marker-after-skip, bare-`<` tail,
   `\u003c` escapes, cross-chunk reassembly).
2. **Clone elision in request mutation** (`proxy.rs`). The request
   mutation stage used to deep-clone every JSON body before learning
   whether any mutator would fire; on the llama.cpp lane the model
   stamp is `None` and conforming bodies mutated nothing. Four
   read-only predicates (usage, system-first, stamp, think) now
   pre-check the borrowed value and return the original bytes when no
   mutator would fire; firing bodies take the same mutators in the
   same order. Each predicate is conservative: it may say "needed"
   for a body a mutator would leave alone, never the reverse. Pinned
   by two new unit tests, including a 24-body corpus asserting
   predicate == mutator outcome for all four pairs.

Suite: 2146/2146 workspace tests green (2139 baseline + 7 new,
PASS→FAIL = 0), `cargo fmt` clean, clippy clean (one pre-existing
transitive future-incompat note).

## Post-fix evidence

- **Profile:** `SseThinkFilter::feed` 1.48% → 0.37% (marker-scan
  only, ~0.4 µs/token); the relay-side parse cluster is gone. Total
  daemon CPU/token unchanged within noise (93.2 → 96.0 µs, ±5% run
  variance) — the recovered work is real but small next to the
  kernel/structural slices, as the 2026-10-08 note predicted.
- **A/B campaign:** every gateway cell moved within the band the
  *direct control lane* itself moved between runs (decode ±3%,
  concurrency sums ±10-20% on this shared desktop): default 325.4 →
  324.0 t/s, matched 329.0 → 319.5, C8 sums 6323.6 → 6511.8 with ITL
  p99 6.2 → 5.8 ms. Greedy gateway-transparency identical at 14/20
  exact, ratio 0.941 — behavior byte-identical in vivo, not just by
  proof.
- **Quality:** zero change — greedy transcripts, determinism 20/20
  engine self-consistency, and the transparency ratio are all
  unchanged.

## What remains, and when to revisit

- ~96 µs/token daemon CPU is mostly kernel loopback + runtime
  plumbing; the observation task's per-frame JSON parse is off-relay
  and below the structural floor at C1, while C8 tails already win —
  left as-is, now attributed.
- ~115 µs/token off-CPU wakeup latency is the extra task hop per
  chunk; recovering it means re-architecting the body pipeline, which
  the 2026-08 note already priced as behavior-bearing.
- Revisit conditions: a lane where engines emit think markers in
  ordinary content (the fast path's scan is the only added cost), or
  a profiling run that moves the kernel slice.
