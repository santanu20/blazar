# Backend default A/B — triton/pytorch (default) vs flashinfer/flashinfer

**Date:** 2026-10-17 · **Blazar:** working tree @ c3e5187+uncommitted (v6 binary)
**Model:** qwen3-1.7b (dense, spec off — spec pins removed to isolate the backend variable)
**Engine:** sglang 0.5.21, fresh install, JIT cache warm (engine-scoped)
**Method:** greedy (temp 0) 256-token streams, 1 warmup spawn + 3 measured trials per arm,
TTFT + decode tok/s from `stream_options.include_usage`; each arm = separate child via
config overlay + daemon restart; backend truth verified via `/get_server_info` per arm.

## Results

| Arm | attention / sampling | decode tok/s (3 trials) | median | TTFT warm | cold spawn |
|---|---|---|---|---|---|
| A (default) | triton / pytorch | 66.3 / 64.6 / 66.3 | **66.3** | 41 ms | 46.1 s |
| B | flashinfer / flashinfer | 66.0 / 65.8 / 66.0 | 66.0 | 42-54 ms | 72.6 s |

Completion lengths differ across arms (192 vs 178) — different attention kernels give
different floating-point paths even at temp 0; expected, does not affect the throughput
conclusion.

## Verdict

**Keep triton + pytorch as global portability defaults.** Decode throughput is a wash
(66.3 vs 66.0 tok/s, ~0.5% — within noise), while the flashinfer arm pays **+26 s cold
boot** (flashinfer attention init on 0.6.18). No perf case for flipping the default; the
pin knobs (`attention_backend`, `sampling_backend` per model overlay) remain the right
surface for lanes that need flashinfer (e.g. spec verify sampling).

Portability note: any future revisit must also account for existing installs whose venvs
predate the compiler-trio pin — a default flip would JIT-crash those until they reinstall
the engine.

## Reproduce

```
# arm A: no overlay (defaults)
# arm B: [model_overrides."qwen3-1.7b".sglang] attention_backend="flashinfer" sampling_backend="flashinfer"
sudo systemctl restart blazar
python3 harness.py   # streaming request with stream_options.include_usage for per-token timing
curl -s :<engine_port>/get_server_info   # verify arm truth
```
