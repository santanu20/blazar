# Cross-engine KV interoperability certification matrix - receipt

- Date: 2026-11-21
- Binary: tree debug build (`blazar 0.33.0` dev tree)
- Scope: issue 011 (docs/issues/011-kv-interop-certification-matrix.md)

## What landed

- `crates/blazar-runtime/src/kv/certification.rs`: the canonical 4x4
  pair-wise matrix (llama.cpp / vllm / sglang / mlx, both directions)
  with an explicit verdict per pair:

  | Verdict | Meaning |
  |---|---|
  | `exact` | same-engine round trip through the native restore path |
  | `live_certified` | real engines, proven token reuse, receipt cited |
  | `codec_shipped` | codec + byte-identity unit proofs landed; live pair certificate pending |
  | `unavailable` | no codec lane exists for the pair |

- The committed registry gained a `kv_matrix` section (schema v3,
  `scripts/first_class_flags.json`), pinned both directions by
  `integration__knob_registry__committed_kv_matrix_matches_live_certification`.
- `scripts/engine_coverage.py --check` gates the matrix (4x4 coverage,
  exactly-once pairs, live_certified implies an existing receipt
  artifact, receipt only on live_certified) and
  `docs/engine-coverage.md` renders it as a "Cross-engine KV
  interoperability" section.

## Live evidence status (honest, per pair)

- `llama.cpp -> vllm`: **live certified** - receipt
  `docs/research/2026-11-19-llamacpp-vllm-kv-import-data-plane.md`
  (gateway chat through the imported prefix reported `cached_tokens > 0`).
- The five remaining codec pairs (`vllm -> llama.cpp`, `mlx <-> vllm`,
  `sglang <-> vllm`): codec shipped with byte-identity unit proofs;
  their live pair certificates (both engines up, prefix export, import,
  reuse + prefill savings measured) are PENDING and named as such in
  the matrix - the gap is explicit, never presumed compatible.
- mlx pair certificates require a Metal host for the mlx lane; this
  CUDA-only box cannot run them.

## Crash consistency (verified live against the real writer/reader)

`unit__kv_certification__interrupted_page_publish_never_exposes_partial_cache`
drives `carrier_to_sglang_pages` + `sglang_pages_to_carrier` on a real
tempdir:

1. A successful publish leaves digest-named objects only - zero `.part`
   files survive.
2. An interrupted publish (garbage under a `.part` name, digest object
   absent) leaves the reader walking the digest chain to the last
   complete object - the partial bytes are never surfaced. A crash can
   cost a cold page; it can never serve wrong bytes.

The vllm block writer follows the same write-aside + rename discipline
(repack.rs, interchange.rs fs-tier export).

## Migration gate

`[kv.translation] enabled` defaults to `false` (config.rs) and
`unit__kv_certification__migration_stays_off_by_default` pins that the
default registry admits zero pairs. No pair promotes to automatic
migration without a live certificate + measured prefill savings.

## Verification receipts

- `cargo nextest run -p blazar-runtime -E 'test(kv_certification) +
  test(knob_registry)'`: 11 passed.
- `cargo clippy -p blazar-runtime --all-targets`: no findings.
- `python3 scripts/engine_coverage.py --check`: consistent (gate green).
- Committed JSON regenerated from live emission (version 3).

## Pending follow-ups (tracked in the matrix itself)

1. Live pair certificates for the five codec-shipped lanes (vllm +
   llama.cpp same-model prefix exchange; sglang + vllm; mlx pairs on a
   Metal host).
2. Prefill-savings measurement per certified pair (bench lane reuse
   from scripts/bench_matrix.py) before any automatic-migration
   promotion.
