Title: Add vLLM and Kokoro capability certification

Priority: P0 | Wave: 1 | Backlog: B01 | Effort: M | Depends: 001

## Problem

- vLLM (the PagedAttention throughput flagship for shard directories) has no
  section in docs/engine-coverage.md, no entry in
  scripts/first_class_flags.json, and sits in LANES_WITHOUT_REGISTRY in
  scripts/engine_coverage.py. No flag-probe dump exists on disk yet; the only
  vLLM-adjacent artifact is
  docs/research/2026-11-19-llamacpp-vllm-kv-import-data-plane.md.
- Kokoro has a coverage section that reads "0 flags probed" with all
  disposition counts at zero: it is a zero-CLI-flag Python shim, so flag
  probing is the wrong instrument for it. Its capability truth must come from
  disposition (what Blazar knows it serves) plus live serving-path
  certification.

## Approach

vLLM:

1. Capture the flag surface (the `vllm serve` full-help dump) into the probe
   format scripts/engine_coverage.py consumes.
2. Build the scripts/first_class_flags.json lane from live emission truth:
   spawn the engine through Blazar, record which flags it actually accepted.
3. Certify the serving path end to end: pull a small shard model that fits
   the dev GPU, serve, verify generation, streaming, and cancellation.
4. Render the docs/engine-coverage.md section from the registry.

Kokoro:

1. Record a disposition-based capability profile (modality: TTS; model
   format; concurrency behavior; zero-flag argv composition).
2. Live-certify the serving path (spawn, generate, verify audio output).

## Acceptance criteria

- [ ] vllm lane in scripts/first_class_flags.json backed by spawn emission truth
- [ ] vllm section rendered in docs/engine-coverage.md with disposition counts
- [ ] kokoro profile documents the zero-flag shim reality with a serving-path receipt
- [ ] Both lanes removed from LANES_WITHOUT_REGISTRY
- [ ] At least one live certification receipt per lane in scripts/certification_receipts.json

## Files

scripts/first_class_flags.json, scripts/engine_coverage.py,
scripts/certification_receipts.json, docs/engine-coverage.md
