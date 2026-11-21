# Effective-runtime verification receipt (issue 008, Wave C)

Date: 2026-11-20/21. Tree: debug build of working tree (0.33.0 crate version, daemon identifies as 0.34.4 line). Engine: llama.cpp b12578-cuda. Model: qwen2.5-0.5b-instruct:q4_k_m (GGUF). Hardware: RTX 4070 8 GB.

## What was built

- `crates/blazar-runtime/src/effective.rs`: post-spawn verification of the *effective* child configuration against the argv Blazar asked for, via the child's `/props` surface. Warn-only (a drifted-but-healthy child is named, never rejected); one bounded 5 s loopback GET; non-llama.cpp lanes skip. Only argv-pinned values are verified — engine defaults are the engine's business.
- `supervisor.rs` (dedicated-spawn Ok arm, before bank restore): llamacpp-gated probe + per-finding `WARN` on target `blazar::effective`, clean verify at `DEBUG`.

## Root-cause story (why the first compare was wrong)

Round 1 flagged: `argv pinned --ctx-size 32768 total but the child serves 65536 (32768 x 2 slots)`. Investigation:

1. Full child argv dumped from `/proc/<pid>/cmdline` (42 tokens; no duplicate `--ctx-size`; auth via `--api-key-file`).
2. Child's live `/props` (with bearer auth): `total_slots 2, n_ctx 32768`.
3. Single-variable A/B on scratch children, identical `-c 32768 -np 2`: plain → 16384/slot (upstream total semantics); `+--kv-unified` → 32768/slot reported.
4. VRAM discriminator (`nvidia-smi --query-compute-apps`): classic 956 MiB vs unified 972 MiB (+16 MiB) — the KV pool does **not** double under `--kv-unified`.

Conclusion: `--kv-unified` = one shared pool of `ctx_total` tokens; `/props` reports the pool size as each slot's `n_ctx`. Blazar's capacity model (`crates/blazar-core/src/profile.rs`: `--ctx-size` is total, KV scales with total, unified = no per-slot duplication) is correct — the verify layer's `n_ctx × total_slots` reading double-counted under unified geometry.

Fix: `KvGeometry { Classic, Unified, UnifiedPerSlot }` detected from argv tokens; Classic keeps slot-multiplied semantics, Unified compares the pool reading flat (256-token alignment window), UnifiedPerSlot skips the ctx verdict honestly (fork geometry unverified). 5 new unit tests pin the daemon-child shape.

## Round-2 live receipt (post-fix binary)

- XDG-isolated sandbox daemon (port 8199, `RUST_LOG=debug`, snapshot store + engines symlink).
- Cold `POST /v1/chat/completions` → HTTP 200 in 3.7 s, content `EFFECTIVE VERIFY ROUND 2` (`resp-round2.json`).
- Daemon log: `DEBUG effective config verified against /props` for the fresh spawn (`verify-line-round2.txt`) — the round-1 false positive is gone.
- Teardown: SIGTERM → reaped, port freed, zero orphan `llama-server`/`vllm`/EngineCore processes, VRAM back to 16 MiB.

## Repro

```bash
# sandbox daemon (snapshot store + engines symlink per 20261120-kokoro-vllm-cert pattern)
XDG_DATA_HOME=<scratch>/data XDG_CONFIG_HOME=<scratch>/config \
  RUST_LOG=debug setsid ./target/debug/blazar serve &
curl -s -X POST http://127.0.0.1:8199/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-0.5b-instruct:q4_k_m","messages":[{"role":"user","content":"ping"}],"max_tokens":8}'
rg 'effective config' <daemon-log>
```

Artifacts: `resp-round2.json` (round-2 reply), `verify-line-round2.txt` (clean-verify log line). Round-1 evidence lives in the sandbox daemon log tail quoted in the session record.
