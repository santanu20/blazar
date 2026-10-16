# ngram revalidation + flashinfer cache scoping — 2026-10-17

## env
- blazar daemon (systemd), port 11435, engine sglang-0.5.21 (venv, compiler trio pinned 13.0.88 per §9)
- model qwen3-1.7b (4,063,515,592 B), overlay: spec_algorithm="ngram" (+ sampling_backend="flashinfer" for run 3)
- wrapper patched with FLASHINFER_WORKSPACE_BASE export (code fix: wrapper_script() in crates/blazar-runtime/src/engine/sglang_install.rs)

## evidence
- /proc/<child>/environ: FLASHINFER_WORKSPACE_BASE=<engine>/cache present on live spawn
- /get_server_info: speculative_algorithm NGRAM; attention triton / sampling pytorch (run 1-2), sampling flashinfer (run 3)
- run 1: temp 0.7 top_p 0.95 -> 200, 120 tok, coherent capitals (answer_pytorch_t07.json)
- run 2: temp 1.0 top_k 40 -> 200 "Elephant"
- run 3: temp 0.8 top_p 0.9 flashinfer sampling -> cold 200 in 31.6s (gateway trace, boot incl), warm 200 in ~2s; engine alive
- cache deltas across all runs: engine-scope 35 -> 35 files, ~/.cache/flashinfer 16 -> 16 (zero new escapes)
- baselines: 8 of the 16 home files dated Oct 15+ (pre-patch wrapper), engine had 23 Oct-15+ files — the mixed-write signature of the import-order race
