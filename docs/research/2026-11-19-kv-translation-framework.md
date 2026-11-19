# KV Translation Framework (PR4) — gate-complete cross-lane eligibility

Date: 2026-11-19
Status: shipped as policy framework; data-plane converter banked as the next wave
Plan: `dp-20261119-kv-translation-framework-pr4` (Branch A scored 8.05/10; Branch B feasibility 3/10 this wave)

## Problem

The KV campaign shipped persistent reuse inside each lane (llama.cpp slot banks, vllm
OffloadingConnector, sglang+LMCache daemon) and refused every cross-layout transfer with
honest errors. What was missing is the *decision layer*: given a KV cache stored under
lane A's layout, may lane B reuse it, and if not, why exactly? Until PR4 the
compatibility function was a placeholder — any registered converter id blessed any
layout pair (`converters.first()`), and nothing constructed real layouts or identities,
so no surface could answer the question truthfully.

The live evidence motivating a *typed* answer: on 2026-11-19 (LMCache wave, qwen2.5-1.5b)
sglang stored 256 tokens into the shared daemon; vllm retrieved them (74.6% external
prefix-cache hit) and generated garbage — LMCache 0.5.8rc1 copies stored objects into
the registered tensors with no cross-layout conversion (`v1/multiprocess/modules/
lmcache_driven_transfer.py:984`; `rg 'def convert|reformat'` over `v1/` = zero hits).
A naive "a converter exists" verdict would repeat that corruption. The verdict layer
must know *which* differences a converter can bridge and which are hard blocks.

## Lane layout facts (live-registered, 2026-11-19)

Constants in `crates/blazar-runtime/src/kv/translation.rs` cite these registrations:

| lane | attention_layout | block_tokens | source |
|---|---|---|---|
| llama.cpp | `sep-kv-token-major` | 1 | slot save/restore path, engine-native |
| sglang | `sep-kv-block-major` (`NL_X_NB_BS_NH_HS`) | page (config `page_size`, default 1) | daemon registration log, sglang 0.5.23 run |
| vllm | `fused-kv-block-major` (`NL_X_NB_BS_NH_CS`, cs = 2×head_dim) | 64 | daemon registration log, vllm 0.34.0 run; engine `lmcache_mp_connector.py` |

The corruption pair (sglang vs vllm on equal geometry and dtype) differs in exactly two
axes: K|V split/fuse and token granularity — the composition `defuse+repage`.

## Design

- **Typed built-ins, not a plugin API.** `BuiltInConverter::{Defuse, Repage}` with
  chain-step semantics: `eligible` requires the converter's own field to be the *only*
  remaining difference; `apply` pulls that field toward the target. A chain composes
  depth-first (depth ≤ 2, registry order) into a canonical id `defuse+repage`.
  First-class extensibility would be speculation — no third converter exists to model.
- **Hard blocks before any converter:** tensor geometry (layers / kv_heads / head_dim),
  dtype casts (value-changing, never silent), parallel reshuffle (tp/pb — no blazar lane
  reports tp>1, semantics unverified). Each block carries its reason string.
- **Knob:** `[kv.translation] enabled` (default off) + `allow` (empty = all built-ins;
  a list narrows; unknown ids fail config load naming the known set). Knob-off = empty
  registry → every layout gap is `Incompatible` with a teaching that names the knob.
- **Identity, honestly:** HF dirs pin `model_sha256` (store row) + `sha256(tokenizer.json)`
  + `sha256(config.json)` + rope fingerprint (`rope_theta`, plus `rope_scaling`
  type/factor when declared). GGUF rows use `gguf-embedded:<file-sha>` markers for
  tokenizer/config — embedded blobs get no fabricated sub-digests. Any missing leg →
  `None` → the surface reports identity-incomplete and stays Exact-only.
- **Layout constructors** from already-parsed metadata (`GgufMeta`, `HfMeta::kv:
  KvGeom{layers,kv_heads,head_dim}`), None-honest when geometry is absent.
- **Doctor verdict table:** per local HF-dir model, sglang↔vllm verdict — `Exact`,
  `Transformable via <chain> - no bridge is wired for this converter yet, transfers stay
  engine-internal`, or the hard-block reason. Two-layer honesty: policy says what
  *would* be safe; nothing pretends a data plane exists.

## Deliberately not in this wave

- **Data-plane converter (Branch B)** — actually repacking bytes between llama.cpp slot
  dumps and vllm/LMCache object format: two engine-internal binary formats to
  reverse-engineer plus digest formulas. Banked as the next wave; its eligibility
  contract is exactly this framework's output.
- **TP/PP bridging** — no lane reports tp>1; modeling a reshuffle without a way to
  verify it would be a silent-corruption generator, not a feature.
- **Bank-path loosening** — session banks stay PR1-strict (engine_sha-bound);
  compatibility is consulted for reporting only.

## Verification

- Unit truth tables: eligibility per converter and per composed chain; hard blocks fire
  ahead of converters; knob-off teaching; allowlist narrowing and unknown-id refusal at
  config load; HF identity pins all four digests and degrades to None per missing leg;
  GGUF embedded markers; the corruption pair composes `defuse+repage` in both
  directions under a full registry and stays Incompatible under a narrowed one.
- CLI: `kv_translation_rows` over a fake HF model dir (knob off teaches, knob on
  reports the composed chain with the no-bridge caveat).
- Live: `blazar doctor` cross-lane table for `qwen2.5-1.5b-instruct` knob-off and
  knob-on (see CHANGELOG entry).

## Sources

- Live daemon registration logs + engine runs, 2026-11-19 (LMCache wave receipts:
  `~/.local/share/blazar/kv/lmcache/daemon.log`).
- LMCache 0.5.8rc1 sdist: `v1/multiprocess/modules/lmcache_driven_transfer.py`,
  `v1/common/kv_format/` (format detection/specs), `v1/multiprocess/config.py`.
- vLLM 0.34.0: `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py`.
- SGLang 0.5.23: `python/sglang/srt/mem_cache/storage/lmcache/lmcache_unified_radix_cache.py`.
