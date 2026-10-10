# mlx kv-bits A/B — qwen2.5-0.5b-instruct-4bit, RTX 4070 (CUDA backend), 2026-10-10

Fixed workload: /api/chat greedy temp 0, num_predict 128, identical prompt, wall-clock
e2e (mlx lane adapter fills no eval_duration — wall is the same metric both arms).
3 warm calls per arm; child argv verified per arm (`--kv-bits 8` present / absent).

| Arm | warm wall s (x3) | mean | child VRAM |
|---|---|---|---|
| kv_bits 8 (+group 64 default) | 0.476 / 0.410 / 0.406 | 0.430 | 631 MiB |
| dense (unset) | 0.247 / 0.198 (cold first call 3.38s excluded) | 0.222 | 739 MiB |

## Verdict
kv-bits 8 is ~1.9x SLOWER decode wall on a 0.5B short-context model and saves only
108 MiB — the KV pool is tiny here, so per-step dequant overhead dominates. kv-bits
is a long-context/large-KV memory lever, not a speed knob; stays opt-in with the
upstream batching-disable caveat. Dense remains the right default for this class.
Binary c37217ca, daemon lane, config restored (unset) after the run.
