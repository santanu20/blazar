# All-Engine Reusable KV Cache: Feasibility Research + ROI

Date: 2026-11-19
Wave: all-engine KV reuse research (user mandate). Parents: `2026-11-19-kv-translation-framework.md` (PR4 policy), `2026-11-19-llamacpp-vllm-kv-import-data-plane.md` (shipped data plane, live-proven `cached_tokens 320/343`).

## Question

Can KV prefix reuse extend to EVERY relevant Blazar lane (llama.cpp, sglang,
vllm, mistral.rs, MLX) — and if yes, what architecture maximizes reuse per
unit of effort and risk?

## Landscape conclusion (evidence-first)

**No engine-agnostic KV interchange standard exists anywhere upstream.** Every
engine owns a private cache substrate; the only cross-engine byte movers today
are (a) LMCache MP — same-layout only; cross-layout retrieval returns garbage
(proven live on this box, lmcache 0.5.8rc1; upstream 2026-Q3 roadmap
LMCache/LMCache#4025 has NO cross-layout converter in any of its seven
sections) — and (b) Blazar's own `repack.rs` (llama.cpp→vllm, live-proven).
Therefore all-engine reuse, if wanted, must be Blazar-built.

## Per-lane findings (primary-source pinned)

| Lane | External KV surface | Format determinism | Injection path | Verdict |
|---|---|---|---|---|
| llama.cpp | slot-save banks (POST /slots) | fully pinned (Branch B) | server API, already spoken | **done** (source side of shipped converter) |
| vllm 0.34.0 | FS offload tier (hash-addressed .bin + config.json) | fully pinned + self-describing | connector reads rank dir at boot | **done** (sink side shipped; reverse = LOW effort) |
| MLX (mlx-lm) | **official file prompt cache** (safetensors + metadata) | pinned from mlx-lm source + tests | CLI reads/writes; server = in-memory LRU only → import needs a venv-side wrapper that pre-loads the cache | **GOLD participant** |
| sglang 0.5.23 | HiCache file backend (default keys) + opt-in unified L3 scheme | default keys NOT self-describing (in-file TODO admits dtype/layout absent); unified L3 = msgspec namespace + page rectangles, cross-deployment by design | engine-only reader; external writer must pin page layout + hash chain | feasible, **highest effort** |
| mistral.rs | none (in-memory prefix_cacher; zero serde/persist code) | n/a | none today | **blocked upstream**; honest Unsupported + optional upstream PR |

### MLX detail (the new gold)

`mlx_lm.cache_prompt --prompt-cache-file f.safetensors` writes
`mx.save_safetensors(tree_flatten([c.state for c in cache]))` with metadata
`{model, tokenizer_config, cache_classes, scalars}`; layout
`[batch=1, n_kv_heads, seq, head_dim]` — token-major, separate K/V, the
closest substrate of all engines to llama.cpp rows. Token ids are NOT stored
(an interchange carrier must carry them). Implemented + closed upstream
(ml-explore/mlx-examples#917, 2024-08-30). The blazar lane runs
`python -m mlx_lm.server`, whose `LRUPromptCache` is in-memory only — import
requires a wrapper module in the venv that pre-loads `load_prompt_cache()`
into the LRU before serving (no upstream dependency, medium effort).

### sglang detail

Installed 0.5.23 `hicache_storage.py`: default file keys carry topology
suffixes but the payload is undocumented bytes — the source's own TODO:
"Include KV dtype and stored layout in keys …; raw file payloads do not
describe their format". The opt-in unified L3 key scheme
(`hicache_key_scheme.py`, `_SCHEMA_VERSION=1`) is a deliberate
cross-deployment design (layer-range × head-range rectangles, logical dtype,
K+V both) — external writing is feasible but means pinning the msgspec
namespace, `page_unified.py` layout, and the page-hash chain: the largest
single-format effort in the fleet. Upstream storage backends are churning
(hf3fs/mooncake/nixl/…) — pin any codec to the lane's installed version.

### mistral.rs detail

`prefix_cacher.rs` (1,975 lines): `CacheKey{tokens, adapter_generation}` →
candle GPU tensors, `IndexMap`, no serialization, no injection API. Blocked
upstream. Options: contribute save/load to prefix_cacher (plausible upstream
PR from us — candle tensors serialize fine), or in-process embed later. Until
then: honest `Unsupported` capability with a teaching.

### Reverse direction (vllm → llama.cpp)

Both formats are fully pinned by Branch B: read vllm tier files (addressing
already replicated in `repack.rs`), repack into a slot-save container
(including the `server_tokens` envelope), inject via the slot-restore API
blazar already speaks. Effort LOW, symmetric to the shipped direction.

## ROI matrix (effort anchored on the shipped Branch B converter = same class of work)

| Direction | Value | Effort | Risk | ROI | Wave |
|---|---|---|---|---|---|
| llama.cpp→vllm | — | — | — | **shipped** (`a8cdbdb`, live-proven) | done |
| vllm→llama.cpp | warm-start GGUF lane from HF-lane prefixes | LOW | low (both formats pinned + oracle fixtures exist) | **highest** | 1 |
| BKI carrier + codec registry | makes every future lane N-codec not N² | MED | low (pure refactor atop shipped parts) | **high** (multiplier) | 1 |
| mlx export→BKI | unlocks mlx prefixes into every other lane | LOW-MED | low (official format, source-pinned) | high | 1 |
| mlx import (venv wrapper) | warm-start mlx lane from any other | MED | med (wrapper in venv, no upstream dep) | medium-high | 2 |
| sglang unified-L3 codec | full mesh incl. sglang | HIGH | med-high (format churn; version-pinned) | medium | later |
| mistral.rs codec | mesh completeness | HIGH | high (no upstream surface at all) | low now | upstream-PR track |
| cross-MODEL KV (different weights) | — | — | — | **unsound** (KV is weights-derived; only text-layer semantic cache is honest) | refused |

## Proposed architecture: Blazar KV Interchange (BKI)

A canonical in-repo interchange carrier + per-lane codecs, reusing everything
already shipped:

- **BKI carrier**: token ids + identity manifest (PR4 constructors: model /
  tokenizer / config digests, rope fingerprint, weight provenance) + layout
  descriptor (layers, kv_heads, head_dim, dtype, block_tokens,
  attention_layout) + per-layer K/V byte rows.
- **Codec registry** (gated by `[kv.translation]`): each lane contributes
  `import X → BKI` and `export BKI → X`; PR4 `translation_verdict` stays the
  eligibility gate; existing `repack.rs` parsers/writers become the first
  codecs verbatim.
- **CLI generalization**: `blazar kv import --from A --to B` resolves the
  codec pair from the registry; `blazar doctor` enumerates every admitted
  pair with the same honesty rows as today.
- **Wave 1** (on approval): carrier + registry + vllm→llama.cpp + mlx export,
  live-proven E2E each. **Wave 2**: mlx import wrapper, sglang unified-L3.
  mistral.rs rides the upstream-PR track.

Why BKI over more pairwise converters: at 5 lanes pairwise is 20 directions;
codecs are 10 — and each new lane (sglang later, any future engine) costs one
codec instead of N.

## Sources

- mlx-lm main: `mlx_lm/models/cache.py`, `cache_prompt.py`,
  `tests/test_prompt_cache.py`; issue ml-explore/mlx-examples#917 (closed,
  implemented 2024-08-30).
- sglang 0.5.23 installed venv: `mem_cache/hicache_storage.py` (HiCacheFile
  :728, in-file TODO), `hicache_key_scheme.py`, `pool_host/page_unified.py`;
  blog sglang.io/blog/hicache.
- mistral.rs master: `mistralrs-core/src/prefix_cacher.rs` (keyword sweep:
  zero serde/persist surface).
- LMCache/LMCache#4025 (2026-Q3 roadmap, api.github.com body) — no
  cross-layout converter; corruption evidence in
  `2026-11-18-cross-engine-kv-lmcache.md`.
- Blazar receipts: Branch B live proof (`cached_tokens 320/343` gateway-level),
  `docs/research/2026-11-19-llamacpp-vllm-kv-import-data-plane.md`.
- Papers: LMCache arXiv:2510.09665, CacheBlend arXiv:2405.16444.

## mlx-lm prompt-cache format (pinned empirically, wave C)

`mlx_lm.cache_prompt --model <dir> --prompt <text> --prompt-cache-file f.safetensors`
writes a safetensors file whose header metadata carries `0.model` (the exact
model path string - free provenance), `0.tokenizer_config` (json), `1.{i}`
(the cache class per layer - plain `KVCache` for dense models), and
`2.{i}.0`/`2.{i}.1` (scalar typemap entries). Tensors: `{i}.0` = K, `{i}.1` =
V, both `[1, n_kv_heads, capacity, head_dim]` with capacity grown in 256-token
chunks (342 tokens -> capacity 512), plus `{i}.2` = I32 scalar = used offset,
which must agree across all layers. Cache dtype follows the model weights
(bf16 for the qwen2.5-1.5b BF16 dir on this box). Token ids are NOT stored -
the interchange carrier carries them, supplied by the caller.

### Dtype oracle (decisive for the vllm sink)

Engine-written vLLM tier block files on this box decode as **f16**, not the
model's bfloat16: f16 interpretation gives plausible post-RoPE magnitudes
(rms 1.79, max 3.81), bf16 gives garbage (rms 9.46, max 318). This
retroactively explains why the Branch B F16-GGUF-sourced import restored
cleanly, and forces the mlx codec to convert bf16 rows numerically
(round-to-nearest-even, overflow-refusing) rather than bit-copying.

### Live findings (wave C, 2026-11-19)

Proof chain (all through the real gateway, qwen2.5-1.5b-instruct, debug
daemon, prod config snapshot/restore):

1. `mlx_lm cache_prompt` on the HF dir cached 342 tokens (bf16). Token
   extraction subtlety: `cache_prompt` APPLIES THE CHAT TEMPLATE when the
   tokenizer has one (`messages=[user]`, `continue_final_message=True`,
   `add_generation_prompt=False`) - raw `encode` gives 318 ids and mismatches
   the 342 cells. The vLLM chat route's own template produces the identical
   342-token prefix, so the blocks hash-match.
2. Import: `blazar kv import --from M --to M --cache-file f --tokens ids` ->
   28 layers x 342 tokens, bf16 source converted to f16, 5 blocks x 64,
   22-token tail dropped.
3. Same-prefix chat: request 1 = cached 0 (async external lookup lands one
   pass late - consistent with every wave); request 2 = **cached_tokens
   320/353, sane and correct reply**.
4. Source disambiguation (the engine also re-stores blocks it computes): all
   5 block files carry the import's write timestamp and were never rewritten;
   the 5 chain hashes recomputed through vLLM's own `hash_block_tokens` over
   the supplied ids match those files exactly. Older wave-B-era files in the
   same rank dir have different hashes (different template id chains) and were
   not involved. The loaded bytes were mlx-derived - the bf16->f16 RNE
   conversion and head-gather repack are byte-correct for a live engine.
