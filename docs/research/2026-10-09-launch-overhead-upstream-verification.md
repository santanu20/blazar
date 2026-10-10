# Launch-overhead axis: upstream CUDA-graph status verification — 2026-10-09

Outcome: **no upstream filing made.** The mechanism blazar wanted to propose
(Metal-style command-buffer pre-encoding/replay for fixed-shape decode, the
uzu `CommandBufferEncoding::end_encoding` → `Executable` replay technique)
already exists on the CUDA lane in mainline llama.cpp, and the remaining
multi-slot gap was explicitly declined by maintainers. Filing would have been
a duplicate.

## Verified upstream state (gh, 2026-10-09)

| Piece | Status | Reference |
|---|---|---|
| CUDA graph capture/replay for decode (Volta+Turing sm_70/75+) | Shipped — runtime graphs with `GGML_CUDA_DISABLE_GRAPHS` opt-out | PR #25749 ("Enable CUDA graphs on Volta+Turing"), referenced by issue #25835 (VRAM leak follow-up, closed) |
| CUDA graphs on Pascal (sm_61) | Open PR, +40% MoE / +7% dense claimed | PR #27721 |
| CUDA graphs for multi-slot decode via shape-stable padded ubatch | **Closed `not_planned` 2026-10-06** | Issue #27009 (state_reason verified via API) |
| SYCL graph record/replay | Open PR | PR #28725 |
| Metal command-buffer pre-encoding/replay | No issue/PR found (searched "command buffer replay", "pre-encode") | — |

## Why this closes the track

- Blazar's CUDA lanes (llamacpp b11429-cuda, this box RTX 4070 sm_89) already
  ride the shipped decode-path graphs; nothing to request.
- The only mechanism gap (multi-slot shape-stable graphs) was declined
  upstream as not planned — a re-file from us would be noise.
- A Metal-lane filing would carry zero Metal-side measurements from us
  (Linux/CUDA shop); that fails our own evidence bar for a filing
  (repro receipts attached). Revisit if blazar gains a Metal measurement lane.

## Context

Track C came from the uzu competitive analysis (2026-10-09): uzu's six speed
mechanisms include Metal command-buffer replay (their Metal-only CUDA-graph
equivalent). On CUDA, ggml beat them to it. Speculative-decode adoption
(tracks A/B) landed separately: see `bench-artifacts/20261009-sglang-spec/`
and the sglang upstream issue
https://github.com/sgl-project/sglang/issues/43412
(flashinfer verify-sampling JIT crash on non-greedy requests, 0.5.21 bundle).
