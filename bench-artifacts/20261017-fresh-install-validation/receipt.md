# Fresh-install validation — installer wrapper emission + CUDA compiler-trio pin

**Date:** 2026-10-17 · **Blazar:** working tree @ c3e5187+uncommitted (binary v6, rebuilt from the gated tree)
**Host:** RTX 4070 Laptop (sm_89), Linux, daemon via systemd, port 11435

## Why

Two live-validation gaps from the 2026-10-16/17 waves: (1) the cache-scoped wrapper
(FLASHINFER_WORKSPACE_BASE) had only been live-patched into an existing engine, never
emitted by a fresh install; (2) the installer's compiler-trio pin
(`pin_jit_compiler_to_torch`) was unit-tested but never observed on a fresh install.

## What ran

1. Rebuilt binary from the gated tree; `sudo install` + daemon restart (healthz ok).
2. `blazar stop qwen3-1.7b`; `blazar engine rm sglang-0.5.21` → **8826 MiB reclaimed**,
   engine dir gone, `~/.cache/flashinfer` untouched (16 files).
3. `blazar engine install --kind sglang` (bare) → built **0.5.19** = `SGLANG_DEFAULT_VERSION`
   (documented flag-contract pin; version moves ride `engine update --kind sglang <v>` —
   not a bug). Install log shows the pin firing on the fresh venv:
   `nvidia-cuda-{nvcc,crt}==13.4.92→13.0.88`, `nvidia-nvvm==13.4.92→13.0.88`,
   `torch.version.cuda = 13.0`.
4. Fresh wrapper emission verified: both `SGLANG_CACHE_DIR` and `FLASHINFER_WORKSPACE_BASE`
   exported to the engine cache dir before exec (`sglang-server` lines 4-5).
5. `blazar engine update --kind sglang 0.5.21` → installed + activated 0.5.21 (trio 13.0.88
   again), **auto-pruned 0.5.19 (8.97 GB)**, daemon restarted. Single sglang lane restored.
6. Cold-JIT proof through the daemon: POST /v1/chat/completions `qwen3-1.7b`
   temp 0.8 top_p 0.9 (overlay: spec_algorithm=ngram, sampling_backend=flashinfer).
   Empty engine cache → first request JIT-compiled from zero.

## Results

- HTTP 200, coherent answer (colors + correct hex codes) — `cold-answer.json`
- Engine truth (`/get_server_info` port 45379): `speculative_algorithm: NGRAM`,
  `sampling_backend: flashinfer`, `attention_backend: triton`
- Engine-scoped cache: 0 → **242 files** (all JIT artifacts landed inside the engine dir)
- Home escape check: `~/.cache/flashinfer` **16 → 16** (zero new escapes)
- `engine rm` reclamation: 8826 MiB (incl. all engine-scoped JIT cache) — the fix's purpose

## Verdict

Both gaps closed live: fresh installs emit the cache-scoped wrapper and pin the compiler
trio to torch's CUDA line with no manual steps. No code changes were needed — the
installer paths worked as designed.

## Reproduce

```
blazar engine rm sglang-<v> && blazar engine install --kind sglang   # default pin
blazar engine update --kind sglang 0.5.21                            # explicit version
curl -s :11435/v1/chat/completions -d '{"model":"qwen3-1.7b",...,"temperature":0.8}'
curl -s :<engine_port>/get_server_info   # port from /api/ps blazar_endpoint
```

Artifacts: `install.log`, `update.log`, `cold-answer.json` (this dir).
