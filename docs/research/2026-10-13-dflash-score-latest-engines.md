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
- Direct engine runs (wrapper + manual): DFLASH + draft reach the verify-JIT stage and die at graph capture. System nvcc (CUDA 12.0): NVCC Internal Compiler Error in `kvcache.cuh(174)`. Venv-bundled nvcc (CUDA 13.4, via the engine wrapper's `CUDA_HOME`): `#error "CUDA compiler and CUDA toolkit headers are incompatible"` (flashinfer 0.6.18 JIT vs mixed include paths). Clean cache, `--sampling-backend pytorch`, and `--attention-backend triton` do not bypass it — `_dflash_top_p_renorm_prob` calls the flashinfer sampling JIT directly. Same defect family as the receipted ngram flashinfer issue (upstream #43412); on a box with a clean matching CUDA toolkit these kernels compile (upstream CI proves the path). Receipt slices preserved under `bench-artifacts/20261013-dflash-score/` (manual-child2, manual-child3, wrapper-child).
- Through blazar: the first request carries the spawn, but the 7.5 GiB target's multi-minute load exceeds the gateway's 2-minute slot wait, and the spawn admission preflight (J3-family, raw-weights comparison) does not model `cpu_offload_gb`, so a weights>free-VRAM model never reaches child exec through the queue. **Follow-up filed:** make the sglang admission preflight offload-aware (the llama lane's ngl ladder already is).

**Score/decisions — PASS end-to-end** (`bench-artifacts/20261013-dflash-score/dflash_score_probe_dense.json`): dense `qwen2.5-1.5b-instruct` (sglang 0.5.21 child, real spawn + 38.4 s load), then through the gateway: chat 200; `/v1/score` 200 (Paris 0.990 on "The capital of France is", Berlin 0.809 on the Germany context, croissants mass tiny — correct ordering); `/v1/decisions` 200 (choice "front desk" 0.9999983, label_mass 0.9+). First attempt 400s were payload-shape education from the engine's own pydantic (`label_token_ids` required for CausalLM; option field is `description`) — the routes relayed them faithfully.

## 6. Gates (T6)

- `cargo fmt --check`: clean.
- `cargo clippy --workspace --all-targets -- -D warnings`: 0 errors (after extracting `sglang_spec_draft_dir`, collapsing one if, backticking a doc path).
- `cargo nextest run --workspace`: **2185/2185** (+2 over the release: the dflash argv + dflash-GGUF pins).
- Full `scripts/validate.py` battery: launched via a wait-guard (a concurrent session's OCR batch holds ~5 GiB RAM; the watcher fires the battery the moment MemAvailable ≥ 6 GiB and no OCR processes remain; completion recorded in a done-marker file carrying the exit code).

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

## 9. Resolution — 2026-10-16: the JIT blocker root-caused and fixed, DFLASH engaged

**Follow-up 2 is closed.** The "toolchain defect" was blazar's own bootstrap assumption, not the box: sglang floats the CUDA compiler dists and pip resolves them to the newest CUDA line, while the torch wheel pins its headers to one exact line. Live matrix: `nvidia-cuda-nvcc`/`nvidia-nvvm`/`nvidia-cuda-crt` all 13.4.92 vs torch 2.13.0+cu130 and `nvidia-cuda-runtime` 13.0.96 headers. Two failure layers, both receipted in `bench-artifacts/20261016-dflash-engaged/`:

1. flashinfer's bundled cccl guard (libcudacxx `cuda_toolkit.h:41`) compares the 13.4 compiler against the 13.0 toolkit headers → `#error "CUDA compiler and CUDA toolkit headers are incompatible"`.
2. Aligning nvcc alone is not enough: `nvidia-nvvm` (separate dist) ships the cicc frontend — 13.4 cicc emits PTX `.version 9.4` that the 13.0 ptxas rejects ("Unsupported .version 9.4; current version is '9.0'"). The pin must cover the trio.

**Fixes:** `sglang_install.rs` now reads `torch.version.cuda` from the fresh venv and re-resolves `nvidia-cuda-nvcc`/`nvidia-nvvm`/`nvidia-cuda-crt` onto that exact line at install time (`torch_cuda_pin_specs`); the live venv was repaired in place the same way (`uv pip install 'nvidia-cuda-nvcc==13.0.*' 'nvidia-cuda-crt==13.0.*' 'nvidia-nvvm==13.0.*'`). A minimal repro (`flashinfer.sampling.top_p_renorm_probs` — the exact op `_dflash_top_p_renorm_prob` needs) went from the cccl `#error` to compile+run clean. `nvidia-cuda-nvdisasm` stays floating: no 13.0-line wheels exist and it is off the compile path.

**Engagement (live, no mocks):** Qwen3-4B + z-lab DFlash-b16 draft, direct engine run — draft runner up ("Initialized DFLASH draft runner... DFLASH draft runner ready", fused KV materialization n_layers=5), **target verify CUDA graph captured** (the exact stage that died before), DFLASH sampling pre-warmed, `_dflash_accept_bonus_contig_kernel` device-loaded, greedy completion correct ("The capital of France is Paris. The capital of Germany is Berlin..."), clean SIGTERM drain. Working posture on the 8 GiB box: `--cpu-offload-gb 4 --mem-fraction-static 0.80 --cuda-graph-max-bs-decode 2 --cuda-graph-max-bs-prefill 512 --context-length 2048` (prefill capture at 2048 tokens with 0.90 was the only remaining OOM — capacity, not defect; full-capture attempt receipted in the artifact log).

**Also fixed on the way (both latent spawn-breakers on 0.5.21):** `--cuda-graph-max-bs` split into `-decode`/`-prefill` (blazar's emitter already picks the new spelling — profile.rs `push_sglang_*`; only this manual probe used the stale flag), and `--cpu-offload-gb` is integer-only — fractional `cpu_offload_gb` pins now quantize UP at emission with a teaching warning (`unit__sglang__fractional_offload_pin_rounds_up_to_whole_gb`).

**New follow-up:** flashinfer's own JIT cache lands in `~/.cache/flashinfer/<ver>/<arch>/cached_ops/` despite the wrapper's `SGLANG_CACHE_DIR` export (observed live: `~/.cache/flashinfer/0.6.18/89/cached_ops/sampling`) — so `engine rm` will not reclaim those compiled kernels and the sglang_install.rs cache-scoping comment overstates coverage. Find flashinfer's cache-root override in the installed 0.6.18 source before wiring it (env var name not yet verified).

## 10. Resolution — 2026-10-16 (later): DFLASH through blazar, admission + memory model honest

The direct-engine engagement above still could not ride the gateway: two blazar-side sizing layers were wrong, each receipted live through the failing spawn chain (`answer4/5/6.json` in the artifact dir).

**Layer 1 — admission (ModelTooLarge false refusal).** The admission floor charged weights + standing overheads and ignored both the `cpu_offload_gb` pin (subtraction) and the engaged draft pair (addition): qwen3-4b + 4 GiB pin was refused as an "impossible floor" though its real VRAM demand is ~6 GiB. Fixed via a shared pure derive (`sglang_effective_offload_bytes`) so the gate and the emitted argv cannot disagree; draft bytes measured from the draft dir (`root_safetensors_bytes`); J3 host-RAM guard counts the offload (the 2026-10-16 dev-box OOM-reboot class); J3 VRAM warn judges on-GPU weights.

**Layer 2 — spec memory model (three interlocked defects).** sglang 0.5.21 semantics, verified in venv source: `mem_fraction_static` bounds only weights+draft+KV (draft loads pre-pool and IS counted); everything else — verify/prefill CUDA graphs, the DFLASH prewarm chain (`(rows, vocab)` fp32 tensors scaled by the decode-graph ceiling), activations — allocates from the `(1-frac)` remainder (`kv_cache_configurator._profile_available_bytes`: pool = free − pre·(1−frac)).

1. Graph ceiling used the dense per-size slope: a spec lane derived ceiling 201 whose x8-9 verify graphs + prewarm needed ~2.9 GiB dynamic vs 2.38 available → OOM at `prewarm_sampling` (dflash_worker_v2.py:760). Ceiling now pays `SGLANG_SPEC_GRAPH_MULT = 8`; Tier A+spec spawns get a bounded list where upstream default 256 used to ride unbounded.
2. The reserve was charged into the fraction NUMERATOR (grows the static pool, starves the dynamic side it protects — the exact inversion that OOM'd). It now LOWERS THE CEILING: `frac <= (vram − reserves)/vram`.
3. Draft weights never entered the static derive → emitted 0.605 under the engine's own 0.6076 minimum-viable floor → death at `alloc_memory_pool`. Draft bytes now charged into `mem-fraction-static` exactly as the engine counts them.

Reserve sized 256 MiB by receipt convergence: the engine's live floor budget admitted `2000 MiB activations + ~300` (matching `SGLANG_ACTIVATION_RESERVE_MIB` + 256), and the now-bounded ceilings shrink the prewarm tail to a few hundred MiB. An interim 1024 calibration over-corrected — the ceiling fell below the engine floor on tight cards (infeasible-by-construction emit); both failure directions are receipted (`answer5/6.json`).

**Result (live, through the gateway, no mocks):** the previously-refusing request now spawns — argv `--cpu-offload-gb 4` (once; double-emission also fixed) + `--mem-fraction-static 0.705` + ceiling-18 capture list, boot-to-warm ~70 s (vs ~260 s at ceiling 201 — the bounded list cuts capture time too), greedy answer correct ("The capital of France is **Paris**."), and the ENGINE'S OWN `/get_server_info` reports `speculative_algorithm: DFLASH` — engagement proven from inside the engine, not from gateway metadata. Full receipt: `bench-artifacts/20261016-dflash-engaged/through-blazar-engagement-receipt.md`.

## §11 — Open follow-ups closed (2026-10-17)

**flashinfer cache escape → fixed at the wrapper.** Root cause (H7, read from
installed source): sglang only `setdefault`s `FLASHINFER_WORKSPACE_BASE`
inside `redirect_third_party_caches()` (srt/environ.py), whose docstring
admits it "silently does nothing if FlashInfer was imported — it resolves
its workspace at import time"; flashinfer's `jit/env.py` freezes
`FLASHINFER_CACHE_DIR = $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer` as a
module-level constant. Any import order touching flashinfer first froze the
cache at `~/.cache/flashinfer` where `engine rm` cannot reclaim it — live
evidence: 8 files escaped there during the Oct-15/16 DFLASH runs while 23
landed engine-scoped in the same window. Fix: the sglang-server wrapper now
exports `FLASHINFER_WORKSPACE_BASE` (engine cache dir, `:-` override
semantics) before exec; the install template gained a pure `wrapper_script()`
(pinned by unit test) and the smoke env mirrors it. Live proof: child
`/proc/<pid>/environ` carries the var; across 4 non-greedy requests + 2
spawns on the patched wrapper, `~/.cache/flashinfer` gained ZERO new files.

**ngram greedy-only constraint → retired.** The constraint existed because
non-greedy verify sampling drove the flashinfer JIT through the same
compiler-trio mismatch fixed in §9. Live revalidation on the fixed toolchain
(qwen3-1.7b, NGRAM engaged per `/get_server_info`, RTX 4070, through-blazar):
temperature 0.7 + top_p 0.95 (pytorch sampling default), temperature 1.0 +
top_k 40, and temperature 0.8 + top_p 0.9 under an explicit
`sampling_backend = "flashinfer"` pin all completed 200 with coherent
sampled output; no Ninja failures, engine stayed up. The
`attention_backend = "triton"` pairing note retires with it (flashinfer
backends work when pinned; triton remains the global portability default).
Artifacts: `bench-artifacts/20261017-ngram-revalidation/`.
