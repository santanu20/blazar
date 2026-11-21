Title: Cross-engine KV interoperability certification matrix

Priority: P1 | Wave: 5 | Backlog: B07 | Effort: L | Depends: -

## Problem

The KV interchange substrate is real and landed: KvCarrier with the
llama.cpp slot-bank and MLX codecs (crates/blazar-runtime/src/kv/interchange.rs),
the vllm-to-mlx staged preload and the sglang HiCache L3 codec (repo history:
2c0fc86, 73c13d3), and the reliability sweep (2004ccd). What does not exist
is a pair-wise certification matrix proving actual cache reuse, per
source/target pair, with live engines.

## Approach

1. Compatibility matrix artifact: every supported source x target pair with
   an explicit verdict (Exact / Verified / N/A with reason). Verdicts come
   only from live tests.
2. Per-pair live test: generate a prefix on the source, export the cache,
   validate the manifest, import on the target, rerun the exact prefix,
   verify tokens were actually reused (not merely that the file converted),
   compare output correctness, measure saved prefill time.
3. Numerical validation: no silent dtype or geometry conversion; any
   conversion is explicit and recorded.
4. Crash consistency: an interrupted import never exposes a partial cache.
5. No automatic KV migration until both correctness and measured prefill
   benefit are proven per pair.

## Acceptance criteria

- [ ] Matrix artifact with a live-test verdict per pair
- [ ] Token-reuse proof (not file-conversion success) per verified pair
- [ ] Correctness comparison and prefill savings recorded in receipts
- [ ] Crash-consistency test for interrupted imports
- [ ] Automatic migration stays off until per-pair proof exists

## Files

crates/blazar-runtime/src/kv/ (interchange + codec modules),
scripts/bench_matrix.py (reuse-measurement lane)
