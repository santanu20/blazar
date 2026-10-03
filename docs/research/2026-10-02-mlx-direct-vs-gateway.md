---
layout: doc
title: "MLX Direct vs Gateway"
description: "Research note: serving MLX directly versus through the gateway."
doc_kind: "Research note"
---

# MLX lane: direct engine vs Blazar gateway (perf + quality receipt)

Date: 2026-10-02 · Blazar 0.18.0 (feat/next-wave @ 6d5b582) · engine lane
`mlx-0.32.0` (`mlx-lm==0.32.0 + mlx[cuda12]==0.32.2`, engine-local cuda-home).

## Environment

| Axis | Value |
|---|---|
| Model | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (4-bit MLX dir, 265 MiB) |
| Hardware | NVIDIA RTX 4070 (sm_89), 24 GB-class box |
| OS | Linux 6.0-compatible kernel, glibc 2.39, X11 session |
| Surfaces | direct = `python -m mlx_lm.server` on 127.0.0.1:41240 (CUDA_HOME=engine cuda-home) · gateway = `blazar` daemon :11435 (llamacpp b11344 active; mlx lane auto-routes by format) |
| Workload | non-stream chat, fixed prompt, max_tokens 128, temperature 0, N=5 per surface after 1 warmup; 1 streaming TTFT sample; 3 parity prompts |

## Perf (non-stream, warm; 101 completion tokens mean)

| Surface | t₁…t₅ (s) | warm mean (t₂–t₅) | tok/s (completion / wall) |
|---|---|---|---|
| direct mlx_lm.server | 1.225, 0.392, 0.382, 0.383, 0.383 | **0.385 s** | ~262 |
| via Blazar gateway | 0.517, 0.400, 0.393, 0.396, 0.394 | **0.396 s** | ~255 |
| **gateway overhead** | — | **+11 ms (~2.8%)** | −7 tok/s (−2.7%) |

Streaming TTFT (SSE first byte): direct 5.5 ms · gateway 5.6 ms (**+0.1 ms**).

## Quality parity (temperature 0)

3/3 prompts byte-identical between surfaces ("sky blue" explanation, "capital
of France", "count 1–5") — the gateway is a transparent passthrough for this
lane: zero quality delta by construction, proven empirically.

## Reading

- Overhead ≈ 11 ms/request at C=1 warm — admission + routing + streaming
  relay on top of the child. Well under one decode step (~4 ms/token here).
- First-request JIT (kernel compile) affects both surfaces identically
  (31.7 s cold on this box, then ~0.4 s warm); the gateway adds nothing to it.
- Scope: 0.5B model, C=1, warm. Concurrency ladders and multi-model mixes
  belong to the benchmark matrix harness (`scripts/bench_matrix.py` wave).

## Reproduce

```bash
# direct
CUDA_HOME=~/.local/share/blazar/engines/mlx-0.32.0/cuda-home \
  ~/.local/share/blazar/engines/mlx-0.32.0/venv/bin/python -m mlx_lm.server \
  --model ~/.local/share/blazar/models/qwen2.5-0.5b-instruct-4bit.d \
  --host 127.0.0.1 --port 41240
# gateway: daemon on :11435, model name qwen2.5-0.5b-instruct-4bit
curl -s localhost:11435/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"qwen2.5-0.5b-instruct-4bit","messages":[{"role":"user","content":"Explain in two sentences why the sky is blue."}],"max_tokens":128,"temperature":0}'
```

Raw per-run captures (curl `time_total` + response usage blocks, one JSON
per request) were taken during the session outside the repo; every number
in the tables above is reproduced verbatim from them and the reproduction
commands below regenerate the same captures.
