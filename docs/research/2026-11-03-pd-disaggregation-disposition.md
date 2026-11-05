# P/D Disaggregation Disposition - Blazar

Date: 2026-11-03
Wave: capability-validation (F2 cell)
Status: DISPOSITION - defer sglang PD lane; watch upstream llama.cpp; keep prefix-affinity routing as the in-box mitigation

## What P/D disaggregation is

Prefill and decode run on separate engine instances (possibly separate boxes):
the prefill worker computes the prompt KV and ships it to the decode worker,
which then generates. Wins: each phase gets hardware sized for it (prefill is
compute-bound, decode is bandwidth-bound), long prompts stop blocking decode
token streams, and both pools scale independently.

## Evidence gathered (live + source-cited, 2026-11-03)

### llamacpp lane (b12072, active engine)

- Slot save/restore EXISTS and works: POST /slots/{id}?action=save|restore with
  filename in the JSON body; live-verified n_saved 41 / n_written 1,619,508 and
  a warm-slot restore hit (cached 27, wall 201 ms) on this exact engine.
- There is NO cross-process KV-transfer API. State leaves the process only via
  slot state files (llama_state_seq_save_file), and the restore path in
  server-context.cpp (~4267-4286) explicitly falls back to full prompt
  re-processing on a fresh child when the checkpoint cannot be guaranteed to
  load ("restored from a slot file, not guaranteed to load - fall back to full
  prompt re-processing"). Verified against the fetched b12072 source
  (server-context.cpp, indexed receipt).
- Consequence: a llama.cpp P/D split today would re-prefill on the decode side
  for every fresh child - the transfer would save nothing the local KV cache
  does not already save. Not implementable without upstream support.

### sglang lane (0.5.23, installed engine)

- The engine manifest carries the FULL disaggregation flag surface:
  --disaggregation-mode, --disaggregation-transfer-backend,
  --disaggregation-bootstrap-port, --disaggregation-decode-*,
  --disaggregation-ib-*, --enable-pd-role-switch, --mooncake-ib-device,
  --prefill-only-disable-kv-cache, --dist-init-addr, --nnodes, and friends.
- The reference topology needs mooncake (transfer engine) + redis-like
  metadata store + at least two engine instances (prefill-only and
  decode-only) coordinated by a router - a multi-instance service, not a
  spawn-shape change.
- Blazar's supervisor currently owns one serving instance per model key on a
  single host; a PD lane means managing instance PAIRS with different argv,
  lifecycles, and health surfaces.

### What blazar already has that captures part of the win

- Prefix-affinity routing (affinity_hash, B1, gateway proxy.rs): repeated
  conversation prefixes stick to the instance that already holds their KV -
  the cheapest possible "KV locality" and it is live.
- Session banking (F3, hardened this wave): KV checkpoints survive idle
  evict and rehydrate the respawn, closing most of the restart-prefill cost
  on a single box.
- ggml-RPC fleet (F1, hardened this wave): multi-box weight/KV placement
  inside one engine process - distributed memory without a P/D process split.

## Disposition

1. DEFER the sglang PD lane until a multi-box deployment target exists. The
   engine supports it; blazar gains nothing wiring it for a single laptop.
   Trigger to revisit: a real second-GPU-box target or user demand for
   prefill-heavy multi-tenant serving.
2. WATCH upstream llama.cpp for a KV-transfer API between processes (the
   pieces - slot state serialization - exist; transfer does not). Revisit
   when a stable cross-process transfer lands.
3. KEEP prefix-affinity + session banking as the shipped mitigation; they
   cover the single-box restart/affinity costs that P/D would otherwise
   target first.

## Upstream watch entries (mirrored to MEMORY.md)

- llama.cpp: cross-process KV transfer (none as of b12072; slot files only).
- sglang: disaggregation surface stable since 0.4.x era; mooncake backend is
  the transfer engine of record.
