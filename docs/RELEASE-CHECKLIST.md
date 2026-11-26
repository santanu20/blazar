# Release checklist

Run before every tag. Every line is a gate: a check that cannot be ticked
blocks the release, and a skipped check must say so in the release notes.

## 1. Build and static gates

| Gate | Command |
|------|---------|
| Formatting | `cargo fmt --all --check` |
| Lint | `cargo clippy --workspace --all-targets --all-features -- -D warnings` |
| Tests | `cargo nextest run --workspace` |
| Config doc drift | `python3 scripts/check_config_doc_drift.py` |
| Engine coverage | `python3 scripts/engine_coverage.py --check` |
| Bench renderer | `python3 scripts/bench_render_selftest.py` |
| Bench archive | `python3 scripts/bench_archive.py --selftest` and `--check` |
| Repo hygiene | `bash scripts/check_hygiene.sh` |
| Glyph policy | `python3 scripts/check_glyphs.py` |
| Release binary | `cargo build --release -p blazar` builds clean |

## 2. Release hygiene

These items close the release-hygiene gaps tracked in the SOTA roadmap
(licence notices, installer checksums, multi-file GGUF import).

### Licence notices

- A model pulled from the ollama registry records the licence layer's text;
  a model pulled from Hugging Face records the model card `cardData.license`.
  Confirm end-to-end on a fresh pull: `blazar show <model>` prints the
  licence line and `POST /api/show` returns the same text.
- Rows with no licence on record (imported, adopted, pre-schema-v25 pulls)
  show the honest upstream pointer - never a fabricated licence.

### Installer checksums

- Every lane's engine row carries a digest: GitHub lanes the verified
  release-manifest sha256, pip lanes a `pip:<sha256>` RECORD fingerprint.
  Confirm on a fresh install: the row's SHA256 column in `blazar engine list`
  is populated, not `unverified`.
- `blazar engine verify <tag>` passes on one GitHub lane of the release
- `blazar engine verify` (no tag) sweeps every installed lane: each report rendered, honest skips tolerated, zero FAIL lines
  matrix and one pip lane. The honest outcomes are part of the contract:
  a missing cache blob on a GitHub lane is an exit-0 skip; drift on any
  lane is a loud failure naming the file and teaching the reinstall.

### Multi-file GGUF import

- A split GGUF (`-00001-of-00002.gguf` naming) imports as one model:
  `blazar import <leaf-1>` links every shard, the row records the shard
  count, and serving that model produces output identical to the unsplit
  file (greedy, same prompt).
- An incomplete set (one leaf missing) is refused with the missing leaf
  named, and no row is written.

## 3. Receipts

Each hygiene check above lands a receipt pair (raw log + markdown) under
`bench-artifacts/` for the release, so the release notes can link evidence
instead of asserting it. The 2026-11 E5 campaign receipts live at
`bench-artifacts/2026-11-23-sota-roadmap/e5-release-hygiene-receipt.md`.
The matched ollama 0.40.17 head-to-head (warm ladder, cold start, the
per-request gateway overhead lane in microseconds, and the temp-0 quality
battery) is receipted at
`bench-artifacts/2026-11-23-sota-roadmap/m1-ollama-head-to-head-receipt.md`.
`python3 scripts/bench_archive.py --check` keeps the tracked receipt set
in step with what the docs cite.
