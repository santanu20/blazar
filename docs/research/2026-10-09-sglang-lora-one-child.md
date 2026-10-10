# sglang LoRA: one-child variant routing + hot-attach (2026-10-09)

## Problem (root cause)

A `model+adapter` request on the sglang lane spawned a DEDICATED child per
variant — a second full copy of the base weights in VRAM — and even then the
adapter was never applied: sglang selects LoRA per request only through its
native `base:adapter` colon model syntax (serving_base.py `_parse_model_parameter`
splits on `:` alone; serving_chat.py resolves the adapter from the model field).
The gateway stamped the body model `base+stem`, which selects nothing — the
child silently answered with base weights. Silent wrong-weights + 2x VRAM.

## Fix

1. **Alias variant lanes onto the base child** (`supervisor.rs
   sglang_variant_rides_base`, called from `ensure_routed_opts` before replica-key
   derivation): validates the stem through the same lane resolution teachings
   (`resolve_lora_lane`), evicts stale adapter-less residents, then routes
   `base+stem` traffic to the base instance — one child, one warm KV cache, one
   loading-coalescing slot shared across base and variants.
2. **Colon stamp at the gateway** (`proxy.rs sglang_variant_stamp` +
   `child_model_stamp_for`): rewrites the child-visible model to
   `base:<registered-adapter-name>` (name = adapter path file stem, matched by
   file name OR file stem — identical to lane resolution). Applied on chat,
   embeddings, embed, rerank, generate lanes; response model is restamped back
   to the caller's `+` spelling. Prediction path mirrors it for pre-spawn
   decisions.
3. **Hot-attach/detach without respawn** (`lora_hot_attach` +
   `POST /api/lora/sync` + CLI notify): `blazar lora add/rm` still write the
   store, then notify the daemon; a resident, adapter-capable (`--enable-lora`)
   sglang child receives the engine's own `/load_lora_adapter` /
   `/unload_lora_adapter` admin call. Everything else (nothing resident,
   non-sglang lanes, adapter-less spawns) defers to the next spawn, which the
   store row already drives — the CLI prints exactly which happened.
4. **llamacpp/mistralrs unchanged**: adapters are baked into spawn argv
   (`--lora-file`) there, so the dedicated variant child stays (aliasing
   returns false for non-sglang kinds).

## Verification (live, RTX 4070 Laptop 8GB, qwen3-1.7b + eternis-anonymizer)

- Cold variant request → ONE child (6.5GB GPU), 200 OK; child /v1/models lists
  base + adapter.
- Adapter selection proven on the SAME child, temp 0, thinking off, identical
  prompt: colon model → "Repeat exactly: ..." (adapter behavior), plain model →
  clean echo (base behavior).
- Hot rm → adapter vanishes from child /v1/models, same pid, sync 17ms.
- Hot add → adapter returns, same pid, sync 36ms.
- Warm variant request after re-attach: 122ms.
- Gates: nextest 2080/2080, clippy clean, fmt clean.
- Raw evidence: `bench-artifacts/20261009-sglang-lora-one-child/`.

## Upstream context

sglang 0.5.21 `/load_lora_adapter` + `/unload_lora_adapter` are
ADMIN_OPTIONAL endpoints (io_struct.py LoadLoraAdapterReq); runtime load
requires `--enable-lora` at spawn (model_runner maybe_init_lora_manager) —
blazar emits it exactly when the model row has adapters, so the hot-attach
path covers attach/remove churn on an already-capable child, and the
supervisor evicts pre-lora residents when variant traffic arrives.

## Incidental finding (flagged, not this wave)

Daemon restart leaves engine children orphaned (an untracked mlx child survived
a stop/start cycle holding 516MiB). Needs a §9 lifecycle follow-up: teardown on
SIGTERM must sweep children, or startup must adopt/kill strays.
