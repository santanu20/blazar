# Through-blazar DFLASH engagement — live receipt (2026-10-16, third wave: memory-model fix)

Reproducibility receipt for the spec-lane memory sizing fix (unreleased at time of run).

## Environment

- Blazar: dev build on HEAD c3e5187 + working tree (admission fix + spec memory-model fix; `/usr/local/bin/blazar` swapped from `target/release/blazar`, `cargo build --release -p blazar`)
- Engine: sglang-0.5.21 lane at `~/.local/share/blazar/engines/sglang-0.5.21/` (venv CUDA compiler trio pinned 13.0.88 — see CHANGELOG trio entry)
- Model: qwen3-4b (store row `~/.local/share/blazar/models/qwen3-4b.d`, 8,044,982,000 B)
- Draft: z-lab Qwen3-4B-DFlash-b16 (`~/.local/share/blazar/models/qwen3-4b-dflash-b16.d`, 1,074,860,568 B)
- Hardware: RTX 4070 Laptop GPU (8,188 MiB total; ~7.7 GiB free at spawn), 32 GiB RAM (~10 GiB available; Chrome closed)
- OS: Linux 6.x, systemd blazar daemon, port 11435
- Config overlay (machine-local, `~/.config/blazar/config.toml`):

```toml
[model_overrides."qwen3-4b".sglang]
spec_algorithm = "dflash"
spec_draft_path = "~/.local/share/blazar/models/qwen3-4b-dflash-b16.d"
cpu_offload_gb = 4
```

## Emitted argv (captured live via /proc/<pid>/cmdline of the launch_server child)

```
--speculative-algorithm DFLASH
--speculative-draft-model-path ~/.local/share/blazar/models/qwen3-4b-dflash-b16.d
--cpu-offload-gb 4                       # emitted exactly once (double-push fixed)
--mem-fraction-static 0.705              # ceiling clamp (vram-2000-256)/vram; above engine floor 0.6076
--cuda-graph-max-bs-decode <= 18         # journal: "capture list <= 18 ... x8 verify multiplier and 256 MiB spec-chain reserve already deducted"
```

## Failure chain this wave fixed (each stage live-observed)

1. v2 posture (pre-fix): ceiling 201, frac 0.688 → OOM at `prewarm_sampling` (446 MiB alloc failed, 172 MiB free; dflash_worker_v2.py:760 scaled_logits).
2. v3 (ceiling form, reserve 1024): ceiling 4, frac 0.605 → engine floor refusal: "Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=0.605. Raise above 0.608 (minimum viable = 1 - available/pre = 0.6076). If using speculative decoding, draft weights are now counted."
3. v5 (final): graph-ceiling x8 multiplier + reserve 256 (receipt-converged with the engine's own `2000 + ~300` live budget) + draft bytes charged into the static derive → frac 0.705, spawn healthy.

## Result

- POST /v1/chat/completions qwen3-4b "What is the capital of France?" (max_tokens 24, temperature 0)
  → 200, `"The capital of France is **Paris**."` (greedy-correct, finish stop)
- Boot to warm ~70 s (b2's manual run at ceiling 201 took ~260 s — the bounded capture list also cuts boot time)
- `/api/ps`: state ready, engine sglang-0.5.21, teaching warnings present
- Engine's own `127.0.0.1:40687/get_server_info`: `speculative_algorithm: DFLASH` (live engine truth, not gateway metadata)
- spawn settle: card used 93%

## Artifacts

- answer7.json — the 200 completion (this wave)
- answer4/5/6.json — the crash chain receipts (502 pool-floor / prewarm OOM)
- engagement-child.log, engagement-result.txt — b2's manual-engine receipts (same flags, direct engine)

## Reproduce

```sh
cargo build --release -p blazar && sudo install -m 755 target/release/blazar /usr/local/bin/blazar
sudo systemctl restart blazar
curl -s http://127.0.0.1:11435/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-4b","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":24,"temperature":0}'
# argv evidence: pgrep -f launch_server; tr '\0' '\n' < /proc/<pid>/cmdline
# engine truth:    curl -s http://127.0.0.1:<engine-port>/get_server_info | jq .speculative_algorithm
```
