# sglang LoRA one-child variant routing + hot-attach — live validation 2026-10-09

Binary: blazar post-wave-3 (LoRA one-child + /api/lora/sync + hot-attach), deployed 20:4x IST.
Model: qwen3-1.7b (sglang-0.5.21 lane) + adapter eternis-anonymizer-qwen3-1 (~/.local/share/blazar/loras/eternis-anonymizer-qwen3-1.7b).

## Cold variant spawn (single child)
- POST /v1/chat/completions model=qwen3-1.7b+eternis-anonymizer-qwen3-1 → 200 in 42.38s (cold spawn incl. child launch)
- ONE sglang child: port 33159, GPU 6542MiB total (single qwen3-1.7b footprint — no second VRAM copy)
- Response model echo: caller spelling "qwen3-1.7b+eternis-anonymizer-qwen3-1" (caller_model restamp)
- Child /v1/models: ["qwen3-1.7b", "eternis-anonymizer-qwen3-1"] — adapter registered

## Adapter selection proof (same child port 33159, temp 0, thinking off, identical prompt)
- direct child model="qwen3-1.7b:eternis-anonymizer-qwen3-1" → 'Repeat exactly: John Smith lives at 221B Baker Street.'
- direct child model="qwen3-1.7b"                    → 'John Smith lives at 221B Baker Street'
- Different outputs = adapter weights active per-request on ONE child (gateway colon stamp working)

## Hot-detach (blazar lora rm 3, child resident)
- CLI: "daemon: hot: adapter eternis-anonymizer-qwen3-1 detached from resident qwen3-1.7b"
- daemon.log: POST /api/lora/sync status=200 ms=17
- Child /v1/models after: ["qwen3-1.7b"] — adapter gone
- Child pid/port UNCHANGED (33159) — no respawn

## Hot-attach (blazar lora add qwen3-1.7b <path> 1, child still resident)
- CLI: "daemon: hot: adapter eternis-anonymizer-qwen3-1 live on resident qwen3-1.7b"
- daemon.log: POST /api/lora/sync status=200 ms=36
- Child /v1/models after: ["qwen3-1.7b", "eternis-anonymizer-qwen3-1"] — adapter back
- Child pid/port UNCHANGED (33159) — no respawn

## Warm variant request after hot-attach (full loop)
- POST /v1/chat/completions model=qwen3-1.7b+eternis-anonymizer-qwen3-1 → 200 in 122ms (warm, resident child)

## Unit/integration gates
- cargo nextest full workspace: 2080/2080 PASS (incl. new: catalog lora_registered_name, supervisor sglang_variant_rides_base, gateway sglang_variant_stamp)
- clippy clean (pre-existing proc-macro-error2 only); fmt clean

## Incidental finding (pre-existing, NOT this wave)
- Daemon restart leaves child processes orphaned: mlx child pid 56329 (qwen2.5-0.5b-instruct-4bit, 516MiB GPU) survived
  systemctl stop/start cycle untracked. Killed manually. Lifecycle gap flagged for §9 follow-up.
