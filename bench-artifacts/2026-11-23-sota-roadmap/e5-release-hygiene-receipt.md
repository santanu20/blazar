# E5 release hygiene - receipt

Roadmap item E5 (licence notices, installer checksums, multi-file
GGUF import), implemented on the sota-roadmap worktree and proven
live on the release binary. Every claim below has a log behind it.

## Scope delivered

1. **Licence notices.** The models store gained a `license` column
   (schema v25 migration, guarded ALTER). Ollama-registry pulls capture
   the licence layer (256 KiB size guard, best-effort, never fails the
   pull; the local store reuse lane reads the notice blob zero-network).
   Hugging Face pulls capture the model card `cardData.license`.
   `/api/show` and `blazar show` print the recorded licence; rows with
   none say the honest upstream pointer instead of a fabricated text.
2. **Installer checksums.** Pip lanes (SGLang, vLLM, MLX, LiteRT,
   kokoro) verify their venv after install by re-hashing every file the
   `*.dist-info/RECORD` manifests list (RFC 4180 quoted paths,
   hashless entries, venv-escape refusal) and stamp the engine row with
   `pip:<sha256>`. `blazar engine verify <tag>` re-checks any recorded
   digest: pip lanes re-walk the RECORDs, GitHub lanes re-hash the
   cached download archive, and a digest whose cache is gone is an
   honest exit-0 skip. Pre-feature rows report `unverified` and say so.
3. **Multi-file GGUF import.** `blazar import` on any leaf of a
   `-00001-of-00002.gguf` set imports the whole set (every leaf linked,
   row shards=N, first-leaf digest, summed bytes - the HF-lane
   convention), refuses an incomplete set naming the missing leaf, and
   serves identically to the unsplit file.

## Verification chain

- cargo fmt --all --check clean
- cargo clippy --workspace --all-targets --all-features -- -D warnings: 0 findings
- cargo nextest run --workspace: 3327 tests run, 3327 passed, 0 skipped (+40 E5 pins over the 3287 pre-E5 baseline)
- check_config_doc_drift, engine_coverage --check, bench_render_selftest, bench_archive --selftest + --check, check_hygiene, check_glyphs: all rc=0 (re-run after the doc edits)
- cargo build --release -p blazar: 63,927,088 bytes (2026-11-26 23:36)

## New pins (RED captured first for every task)

- T1 store: unit__model_roundtrip__license_round_trips_and_replaces, unit__migrate_v25__adds_models_license_to_legacy_stores
- T2 registry: 6 wiremock pins (captures_license_layer, absent_leaves_none, oversized_skipped_without_fetch, non_utf8_skipped, fetch_failure_never_fails_pull, local_store_license_blob_is_read)
- T3 HF: unit__hf_model_info__card_data_license_decoded_and_unknown_fields_tolerated, integration__pull_single_file__captures_card_license, + 2 existing pins extended
- T4 surface: e2e__show_licence_surfaces_row_text_and_honest_fallback
- T5 pip verify: 10 pip_verify pins + 2 install_with_rollback pins (record stamps pip: digest; mismatch rolls back and preserves the old engine)
- T6 engine verify: 7 pins (pip reverify, pip drift, unverified row, GH cache rehash, sha256: prefix strip + cache miss, notfound + dirmissing)
- T7 shard import: 5 shard_set_leaves pins + 4 import_model pins
- T5 hardening (found live): unit__verify_venv_record__nested_vendor_record_is_ignored

## Live evidence (release binary, no mocks)

### engine verify on the real host lanes

```text
=== engine verify b12728-cuda ===
verified engine b12728-cuda: fa933e8c59cecfca0f4eda3358afca026fb1557f857a7e86896820fabef7b0af (cached archive re-hashed (220573131 bytes at ~/.cache/blazar/downloads/84b7966945105833-llama-b12728-bin-ubuntu-cuda-12.8-x64.tar.gz))
RC=0
=== engine verify kokoro-1.0 ===
verified engine kokoro-1.0: beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a (cached archive re-hashed (325505369 bytes at ~/.cache/blazar/downloads/d8b9971f715c8e70-kokoro-v1.0.onnx))
RC=0
=== engine verify sglang-0.5.23 ===
blazar: engine sglang-0.5.23 records no usable digest (row predates digest recording) - reinstall the lane to record one
RC=1
```

### pip-lane verify against the real venvs (sandboxed store copy)

```text
[1] sglang-0.5.23 (seeded wrong digest) rc=1
blazar: engine sglang-0.5.23 failed verification: venv file build_backend.py drifted from its RECORD: expected sha256 54a69428da79de604d4b7d21d96cfa7574ca37c48191bcd391669af5cb28b20d, got 59480145c7bb8be2bb86d4dda092763953d22d550056ff03bfe704b5af204f89 - reinstall the lane to restore it - reinstall the lane to restore a known-good engine

PASS: real post-install rewrite (flashinfer build_backend.py) detected

[2] mlx-0.32.2 (seeded wrong digest) rc=1
blazar: engine mlx-0.32.2 failed verification: row records pip:0000000000000000000000000000000000000000000000000000000000000000 but the live venv now fingerprints pip:ad5f8ac9f3c6c1edaa52b93e6705d37e2e6312e445465c3ae0b65d7ba0134c4c - the venv changed after install; reinstall the lane - reinstall the lane to restore a known-good engine

[3] mlx-0.32.2 restamped rc=0
verified engine mlx-0.32.2: pip:ad5f8ac9f3c6c1edaa52b93e6705d37e2e6312e445465c3ae0b65d7ba0134c4c (venv re-verified against its RECORD manifests)

PASS: pip-lane engine verify live arm (drift on sglang; verified cycle on mlx-0.32.2)
```

### licence E2E + shard-set import E2E (real daemon, real splitter)

```text
validate: mem floor overridden via BLAZAR_VALIDATE_MEM_FLOOR_MIB -> 2048 MiB
[a1] /api/show licence surfaced verbatim (78 chars)
[a2] blazar model show carries the licence line
[c1] llama-gguf-split produced 2 leaves (491400256 bytes total)
[c2] import rc=0
imported qwen2-5b-split-e5 (469 MiB q4_k_m, arch qwen2, ctx_train Some(32768)) -> <sandbox>/share/blazar/models/qwen2-5b-split-e5-q4_k_m-00001-of-00002.gguf
zero extra disk used (link, not copy)
2 shards installed as one model: the row points at leaf 1, the engine discovers the rest by naming convention

[c3] row: shards=2 path=leaf-1 sha256=first-leaf bytes=sum (491400256)
validate: mem floor overridden via BLAZAR_VALIDATE_MEM_FLOOR_MIB -> 2048 MiB
[c4] split-model greedy output: 'banana'
[c5] unsplit-model greedy output: 'banana'
[c6] greedy outputs byte-identical between split and unsplit
[c7] incomplete-set import rc=1
blazar: shard set is incomplete: <sandbox> is missing (found 1 of 2 leaves beside <sandbox>); split models import only as a full set

PASS: licence E2E + shard-set import/serve/compare + incomplete-set refusal
```

## Reproduction

The live arms run through the repository validation harness
(scripts/validate.py Sandbox/Daemon) against the release binary;
the unit and integration pins run under `cargo nextest run --workspace`.
The release checklist (docs/RELEASE-CHECKLIST.md) carries the hygiene
items as standing release gates.

