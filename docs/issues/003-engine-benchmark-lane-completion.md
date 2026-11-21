Title: Benchmark lane completion for all supported engines

Priority: P0 | Wave: 1 | Backlog: B04 | Effort: M (code) + machine time | Depends: 002, 004

## Problem

The bench infrastructure is already a full coordinator: scripts/bench_matrix.py
(12K+ lines) runs direct vs Blazar-gateway vs ollama legs, concurrency
ladders, context sweeps, media/quality/ppl/tools lanes, power sampling, and
writes receipted cells (bench-artifacts/*/cells.jsonl with argv, version,
resource, and power fields) tracked by bench-artifacts/INDEX.md.

The 2026-11-20 final campaign render covers llamacpp (fork lane), mistralrs,
ollama-host, piper (media-tts), sdcpp, and whisper. Missing entirely: sglang,
vllm, kokoro. mistralrs also lacks a conc-direct leg, so its gateway-vs-direct
concurrency comparison is incomplete. sglang lanes are proven runnable on
this hardware by earlier receipts (bench-artifacts/20260928-sglang-safetensors,
bench-artifacts/20260930-sglang-lora-nimble, and later campaigns).

## Approach

1. Add per-engine lanes reusing the existing coordinator; small models that
   fit the dev GPU (8 GB VRAM constraint - document it, do not skip it).
2. Per lane: baseline, direct-vs-gateway concurrency ladder (C=1,2,4,8),
   quality receipt (greedy parity / ppl), known-limitations note in the
   campaign README, INDEX.md row.
3. Close the mistralrs conc-direct gap.
4. Use the stage timing from 004 to attribute residual gateway-vs-direct
   deltas.
5. Where a lane cannot run on available hardware, record an
   engine_exclusion_reasons entry rather than silently omitting the cell.

## Acceptance criteria

- [ ] Reproducible baseline receipt per supported engine, or an explicit exclusion reason
- [ ] Direct-vs-gateway comparison at C=1,2,4,8 for every runnable engine
- [ ] Quality receipt per engine lane
- [ ] mistralrs conc-direct leg present
- [ ] Known-limitations section per campaign; INDEX.md updated

## Files

scripts/bench_matrix.py, scripts/bench_charts.py, bench-artifacts/INDEX.md

## Notes

bench_matrix.py evolves quickly; rebase lane work on current HEAD rather
than carrying a long-lived fork of the coordinator.
