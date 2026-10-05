# MTP spec-decode currency receipt — 2026-10-05

Question (from the perf-vs-quality audit): is blazar's native-MTP lane
(`--spec-type draft-mtp`, llama.cpp PR #29761) current against upstream, and
what is it waiting on?

## Verdict

**Fully current. The pinned engine `b11393-cuda` already carries the entire
draft-mtp emission surface** — the lane can fire today without an engine
update. It is dormant for exactly one reason: no MTP-bearing GGUF exists to
serve (none on this box, none published upstream). The gating chain from
`profile.rs` is satisfied end-to-end on the current pin:

| Gate (profile.rs) | Requirement | b11393-cuda manifest | State |
|---|---|---|---|
| spec-mode enum | `mtp` modeled | supervisor/profile spec enum has `mtp` | OK |
| `--spec-type` advertised | flag in manifest | `--spec-type` present (329 flags) | OK |
| `spec_types` contains `draft-mtp` | choice list | full list incl. `draft-mtp`, `draft-dflash`, `draft-dspark`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache` | OK |
| `--spec-draft-n-max` | companion flag | present | OK |
| `--spec-draft-backend-sampling` | companion flag | present (with `--no-` variant) | OK |
| GGUF carries MTP layers | `mtp_layers` non-null (`{arch}.n_predict_layers` / `nextn_predict_layers`) | **absent on 11/11 local GGUFs** | **BLOCKER — model side** |

The earlier working assumption that b11393 (built Sep 26) predates the MTP
merge (Oct 1, c061df198 "Qwen4Exp: add MTP") was wrong: the unified
`--spec-type` vocabulary with `draft-mtp` as a choice was already upstream by
build 11393. The Oct 1 merge wired MTP for the Qwen4Exp architecture; the
flag surface predates it.

## Evidence

1. **Pinned engine manifest** (authoritative source — `blazar.db` engines
   table, `b11393-cuda` row, read-only query 2026-10-05): `spec_types` =
   `[none, draft-simple, draft-eagle3, draft-mtp, draft-dflash, draft-dspark,
   ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache]`;
   `--spec-type`, `--spec-draft-n-max`, `--spec-draft-backend-sampling` all in
   `flags`.
2. **Newest upstream release b11408** (probe ran OUTSIDE the store — tarball
   `llama-b11408-bin-ubuntu-cuda-12.8-x64.tar.gz` downloaded to a temp dir,
   `--help` captured, zero store/daemon mutation, temp dir removed after this
   receipt landed): identical spec-type vocabulary; `--spec-draft-n-max`
   default 3; `--spec-draft-backend-sampling` at help line 364. Full help:
   `b11408-full-help.txt`, spec-family excerpt: `b11408-spec-family-help.txt`.
3. **Local GGUF sweep** (2026-10-05, python GGUF header parse): 11/11 models
   on this box carry no `*predict_layers*` metadata key — Laya, Qwen3-0.6B,
   Qwen3.5-9B-Q4_K_M, Qwen3VL-8B, reranker, bge, mmprojs, qwen-image,
   qwen2.5-0.5b, umt5.
4. **Upstream model availability** (HF API, 2026-10-05): `ggml-org` author
   search "qwen4" → zero results; "Qwen4Exp gguf" → zero; "qwen4exp" → only
   tiny fixtures/experiments, no real checkpoints. **No MTP-bearing GGUF is
   published anywhere yet.**

## Consequences

- No engine update required for draft-mtp capability. The
  "engine {tag} lacks draft-mtp — run: blazar engine update" warning path
  never triggers on b11393+ for capability reasons.
- The lane auto-fires the moment an MTP-bearing GGUF lands in the store:
  `spec = "auto"` (default, opportunistic) → embedded MTP head wins over
  catalog draft-pair → n-gram fallback. No config change, no code change.
- Memory math note: Wave M (2026-09-30) reverted draft-PAIR MTP because a
  separate draft model did not fit 8 GiB. Embedded MTP heads ride the model's
  own weights (~+1 layer) — that objection does not apply; the memory math
  changes in blazar's favor when the artifacts appear.
- Future additive surface (no action now — the lane is generic over
  `spec_types`): upstream vocabulary blazar could opportunistically adopt once
  measured: `draft-dflash`, `draft-dspark`, `ngram-map-k`, `ngram-map-k4v`,
  `ngram-mod`, plus the ngram tuning knobs (`--spec-ngram-*-size-*`,
  `--spec-ngram-mod-n-min/n-max`). Also tracked separately:
  draft-mtp-adaptive (PR #27210, still open upstream), already gated in
  profile.rs.

## Reproduce

- Manifest: `sqlite3 ~/.local/share/blazar/blazar.db "SELECT manifest FROM
  engines WHERE tag='b11393-cuda'"` (read-only) → `spec_types`, `flags`.
- Upstream probe: download `llama-b11408-bin-ubuntu-cuda-12.8-x64.tar.gz`
  from the ggml-org/llama.cpp release `b11408`, extract, run
  `./llama-b11408/bin/llama-server --help`.
- GGUF sweep: parse each model's GGUF header for keys matching
  `*predict_layers*`.
