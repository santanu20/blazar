# DFlash lane + score/decisions routes + engine currency — 2026-10-13

**Subject:** the research-driven SOTA gap wave: DFLASH speculative decoding on the sglang lane, gateway proxy routes for sglang-native `/v1/score` + `/v1/decisions`, and engine currency (llama.cpp b11580 → b11588).

**Verdict: implemented and live-validated within what this box can express.** The DFlash lane is code-complete and unit-pinned; the real engine accepts its flags and runs them until a toolchain defect (this box's flashinfer JIT cannot compile the DFLASH verify kernels under either the system CUDA 12.0 nvcc or the venv's bundled CUDA 13.4 nvcc) kills the child at graph capture — loud, root-caused, receipted. `/v1/score` and `/v1/decisions` round-trip 200 through the gateway on a live sglang child. b11588 is active with a zero-flag delta (331 → 331).

## 1. Research → gap selection

SOTA scan (sglang v0.5.21 + vllm v0.31.0 release notes, llama.cpp b11588): the 2026-10-09 DFlash exclusion ("needs `--speculative-draft-model-path`") is liftable — the installed 0.5.21 confirms the full flag surface (`--speculative-algorithm {EAGLE,EAGLE3,NEXTN,STANDALONE,NGRAM,DFLASH,DSPARK,UNO}`, draft-path/revision/load-format, `--speculative-dflash-block-size`, plus ngram external-corpus and dspark table knobs). DFLASH rides the EAGLE-family draft machinery (venv source: `dflash_info_v2.py` carries Eagle-shaped draft state through `EagleDraftWorker`), and the public checkpoint ecosystem is real — the canonical pair per upstream docs is `meta-llama/Llama-3.1-8B-Instruct` + `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat`; z-lab publishes drafts down to 4B-class targets (`Qwen3-4B-DFlash-b16`, a 5-layer ~0.8B head, BF16 ≈ 1.6 GiB).

Planning: yggdrasil `dp-20261013-sglang-dflash-score-latest-engines` — approach "extend the draft-pair lane" scored 8.35/10 over generic pass-through (6.35, YAGNI) and defer-DFlash (6.35, fails the directive). Deviation from the finalized steps, with reason: pair-lane routing by catalog `spec_type` was dropped — catalog `draft-dflash` pairs are GGUF (llama.cpp-lane-only per the 2026-09-30 loader receipt), so the sglang-native path is the knob lane (`spec_algorithm = "dflash"` + `spec_draft_path`), which is what shipped.

## 2. Engine currency (T1)

`blazar engine update`: **b11588-cuda active** (was b11580). Full `--help` diff: **zero flags added or removed** (331 → 331; the MTP image-embedding work landed engine-side, not CLI-side). Nothing to map or teach. sglang already at 0.5.21 (latest release); mistralrs v0.9.4, MLX 0.32.0 unchanged.

## 3. The DFlash knob lane (T2)

- `SglangTuning` gains `spec_draft_path` (safetensors draft dir; required with `spec_algorithm = "dflash"` — config validation rejects a bare dflash at load with the 2026-10-09 receipt cited) and `speculative_dflash_block_size` (`--speculative-dflash-block-size`, emitted only when the dflash arm engages).
- `push_sglang_speculative` (profile.rs): the `"dflash"` arm now resolves the draft via `sglang_spec_draft_dir` — present/non-empty, not a GGUF file (loader receipt), engine advertises `--speculative-draft-model-path` — then emits `--speculative-algorithm DFLASH --speculative-draft-model-path <dir>` + block-size + the shared depth knobs. Every refusal path teaches and stays dense; dense spawns remain byte-identical.
- Depth knobs (`spec_num_steps`, `spec_num_draft_tokens`) moved into `push_sglang_spec_depth_knobs`, whose doc always claimed "shared by every engaging speculation arm" — the EAGLE3 pair lane now actually emits them (an inconsistency fix, not a behavior break; flags are order-independent).
- Registry: `--speculative-dflash-block-size` added to `SGLANG_SUPPORTED`; the fixture matrix gained a dflash posture; `scripts/first_class_flags.json` regenerated (committed-JSON match test green).
- Hint block: family lines for both knobs (completeness pin enforces it).
- Docs: `docs/config-code-paths.md` SglangTuning 62 → 64 fields, drift gate green.

**Pins** (all in `cargo nextest run --workspace`, 2185/2185): dflash+draft argv emission, dflash-without-draft dense+teaching, dflash-GGUF refusal, config vocabulary (dflash validates only with a draft), plus the pre-existing ngram/EAGLE3-pair/flag-missing suites.

## 4. Score + decisions routes (T3)

Schemas read from the installed engine before coding: `ScoringRequest` (query/items, `label_token_ids` for CausalLM lanes), `DecisionRequest` (input + discriminated questions: choice `{name, description}` / score levels / yes_no; `extra=forbid`). Gateway: `/v1/score` + `/v1/decisions` mounted on `openai_proxy` (byte-stream passthrough with model-affinity routing), added to the upstream path list and to `SGLANG_SERVES_PATHS` (source receipt: installed 0.5.21 `http_server.py` lines 1955/1963). The llama-inventory drift guard is untouched by design (it pins llama.cpp's README; sglang-native routes follow the systemone precedent). The per-lane gate truth-table test iterates the const, so the new paths are covered on all lanes automatically.

## 5. Live validation (T5) — no mocks

Pulled for real: `Qwen/Qwen3-4B` (7.6 GiB) + `z-lab/Qwen3-4B-DFlash-b16` (1.1 GiB draft) into the store.

**DFLASH engagement — blocked on this box, loud and receipted:**
- Direct engine runs (wrapper + manual): DFLASH + draft reach the verify-JIT stage and die at graph capture. System nvcc (CUDA 12.0): NVCC Internal Compiler Error in `kvcache.cuh(174)`. Venv-bundled nvcc (CUDA 13.4, via the engine wrapper's `CUDA_HOME`): `#error "CUDA compiler and CUDA toolkit headers are incompatible"` (flashinfer 0.6.18 JIT vs mixed include paths). Clean cache, `--sampling-backend pytorch`, and `--attention-backend triton` do not bypass it — `_dflash_top_p_renorm_prob` calls the flashinfer sampling JIT directly. Same defect family as the receipted ngram flashinfer issue (upstream #43412); on a box with a clean matching CUDA toolkit these kernels compile (upstream CI proves the path). Receipts: `/tmp/opencode/dflash-live/{manual-child2,manual-child3,wrapper-child}.log` slices copied under `bench-artifacts/20261013-dflash-score/`.
- Through blazar: the first request carries the spawn, but the 7.5 GiB target's multi-minute load exceeds the gateway's 2-minute slot wait, and the spawn admission preflight (J3-family, raw-weights comparison) does not model `cpu_offload_gb`, so a weights>free-VRAM model never reaches child exec through the queue. **Follow-up filed:** make the sglang admission preflight offload-aware (the llama lane's ngl ladder already is).

**Score/decisions — PASS end-to-end** (`bench-artifacts/20261013-dflash-score/dflash_score_probe_dense.json`): dense `qwen2.5-1.5b-instruct` (sglang 0.5.21 child, real spawn + 38.4 s load), then through the gateway: chat 200; `/v1/score` 200 (Paris 0.990 on "The capital of France is", Berlin 0.809 on the Germany context, croissants mass tiny — correct ordering); `/v1/decisions` 200 (choice "front desk" 0.9999983, label_mass 0.9+). First attempt 400s were payload-shape education from the engine's own pydantic (`label_token_ids` required for CausalLM; option field is `description`) — the routes relayed them faithfully.

## 6. Gates (T6)

- `cargo fmt --check`: clean.
- `cargo clippy --workspace --all-targets -- -D warnings`: 0 errors (after extracting `sglang_spec_draft_dir`, collapsing one if, backticking a doc path).
- `cargo nextest run --workspace`: **2185/2185** (+2 over the release: the dflash argv + dflash-GGUF pins).
- Full `scripts/validate.py` battery: launched via a wait-guard (a concurrent session's OCR batch holds ~5 GiB RAM; the watcher fires the battery the moment MemAvailable ≥ 6 GiB and no OCR processes remain; done-marker EXIT=N at `/tmp/opencode/dflash-live/validate.done`).

## 7. Reproduce

```bash
cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo nextest run --workspace
# dense lane score/decisions round-trip (real store, isolated config)
BLAZAR_PROBE_MODE=dense BLAZAR_PROBE_MODEL=qwen2.5-1.5b-instruct \
BLAZAR_PROBE_DRAFT_DIR="$HOME/.local/share/blazar/models/qwen3-4b-dflash-b16.d" \
python3 <probe rig>   # shape documented in §5; method exact
# DFLASH engagement (needs a CUDA-toolchain-clean box)
blazar pull Qwen/Qwen3-4B && blazar pull z-lab/Qwen3-4B-DFlash-b16
# then model_overrides.<model>.sglang: spec_algorithm="dflash", spec_draft_path=<draft dir>
python3 scripts/validate.py
```

## 8. Follow-ups

1. Offload-aware sglang admission preflight (J3 compare should subtract `cpu_offload_gb`; the engine-side fit ladder already does the equivalent at spawn).
2. DFLASH engagement re-validation on a box with a matching CUDA toolkit (or upstream flashinfer JIT fix); the knob lane is ready and will engage the moment the kernels compile.
3. Optional knobs when drafts exist for them: DSPARK table paths, UNO draft LoRA, ngram external-corpus seeding (all verified present in the installed engine's CLI surface; deliberately not wired — no live artifact today).
