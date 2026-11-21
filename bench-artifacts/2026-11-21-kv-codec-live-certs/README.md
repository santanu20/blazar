# Cross-engine KV codec live certification - vllm -> llama.cpp (2026-11-21)

Pair: `qwen2.5-coder-0.5b-instruct` (HF safetensors dir, vllm 0.34.0 lane,
[kv.translation] + [vllm.kv_offload] enabled) -> `qwen2.5-coder-0.5b-f16`
(F16 GGUF converted from the same HF dir via ggml-org/llama.cpp
convert_hf_to_gguf.py, stamped general.base_model_id, llama.cpp b12578 lane).

## What ran (XDG-isolated sandbox daemon, tree release binary)
1. Warm: two `/v1/completions` runs of the 1575-token raw prefix on the vllm
   lane; warm usage reported `cached_tokens: 1536` (engine-side tier proof).
2. `blazar kv import --from qwen2.5-coder-0.5b-instruct --to
   qwen2.5-coder-0.5b-f16 --tokens tokens.json` -> "imported 24 block(s) x
   64 token(s)" into the slot-save bank `_auto-16384`.
3. Spawn on the llama.cpp lane: daemon logged `restored banked session
   _auto-16384 into slot 0 (1536 tokens into slot KV)` BEFORE first use;
   same-prefix request HTTP 200 in 4.06 s (cold incl. spawn + mlock),
   greedy continuation coherent.

## Root-cause fix this certification forced (product bug)
Blazar's `VllmFsSpec::fields_json` folded a 5-field digest and a 2-level
block path; vllm 0.34.0's FileMapper digests a 10-field config
(pcp/dcp/dtype/kv_cache_groups/inference_engine/parallel_agnostic=false)
and lays blocks out as `<h3>/<h2>_g<N>/<hash>.bin`. The reverse import
could never find an engine-written tier. Fixed in
crates/blazar-runtime/src/kv/repack.rs with a live-captured golden test
(digest `22d6ff76ec48`, see engine-tier-config.json).

## Honest limits (evidence, not assumptions)
- sglang <-> vllm pairs stay `codec_shipped`: Blazar's hicache host-tier
  guard refuses the 13.7 GiB RAM class (needs ~15 GiB headroom; the
  refusal itself is the receipt) - uncertifiable on this box.
- mlx <-> vllm pairs stay `codec_shipped`: mlx lane needs a Metal host.
- Prefill-savings A/B on the 0.5B pair is spawn-noise dominated; the
  restore log (1536 tokens into slot KV) is the reuse proof. A measured
  timing delta belongs to a follow-up on a bigger model.
- Teardown: port freed, no orphan children, VRAM back to 16 MiB.
