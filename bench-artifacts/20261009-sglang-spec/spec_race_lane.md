# sglang --spec-race lane receipt — 2026-10-09

Run: `blazar tune qwen3-1.7b --spec-race` (blazar 24aebf74, daemon stopped, VRAM 15 MiB idle)
Engine: sglang-0.5.21 (active engine was llamacpp; lane resolved sglang-0.5.21, printed note)
Model: qwen3-1.7b (BF16 safetensors dir), single-stream greedy wall tok/s, live spawns.

| mode  | tok/s |
|-------|-------|
| dense |  64.5 |
| ngram |  68.9 |

Verdict: ngram +7% over dense on the search-lane prompts -> lane printed adopt
commands (config set sglang.spec_algorithm ngram + sglang.attention_backend
triton) + greedy-only caveat. NOT auto-adopted (per-model [sglang] overlay is a
whole-struct replace; a partial write would drop sibling pins).

Cross-reference: live_battery.json same model measured ngram -16% @C1 on diverse
fixed probes (and -44% @C8 aggregate, +12s cold). ngram gain is
workload-dependent (prompt n-gram hit rate): +7% on repetitive search prompts,
negative on diverse traffic and under concurrency. Opt-in posture is correct.

Negative-path receipts:
- `blazar tune qwen3-1.7b` (no --spec-race): teaching bail "safetensors dirs
  tune via the sglang spec race" (replaces the previous cryptic GGUF read error
  on directories).
- MLX dirs: clear bail "MLX dirs serve on the mlx lane, which has no tune
  surface yet".
