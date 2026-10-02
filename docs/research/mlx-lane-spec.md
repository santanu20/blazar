# MLX engine lane — engineering spec (F8, staged)

Status: spec only. Implementation is intentionally staged until Mac hardware
is available for live validation — shipping a macOS-only lane blind from a
Linux box would violate the project's live-validation rule and risk a dead
feature. Tracked as the round-4 audit's F8 (docs/research/
2026-10-02-competitor-pain-points-audit.md section 10).

## Why

Ollama ships an MLX engine as its Apple-Silicon headline (their release notes
claim state-of-the-art Mac performance for recent models). Blazar's Mac story
today is llama.cpp Metal + mistral.rs Metal — solid, but not the MLX ceiling
for some model classes. Mac users are a large slice of the local-LLM
audience; an MLX lane is the last major platform lane we do not cover.

## Lane shape (mirrors the sglang venv lane)

- Runtime: `mlx-lm` (Python) exposes an OpenAI-compatible server
  (`mlx_lm.server`) on a localhost port. The lane spawns it as a supervised
  child exactly like the sglang lane: venv install, pinned package version,
  loopback bind, gateway-minted child bearer auth, health + boot-smoke gate.
- Install/update: PyPI-driven (`pip install mlx-lm==<pin>`), not GitHub
  release assets. The engine manager grows a pip-package lane kind whose
  "tag" is the mlx-lm version; doctor currency compares against PyPI
  (reuse `live_sglang_currency`'s PyPI probe shape).
- Platform gate: `cfg(target_os = "macos")` in the runtime; on Linux the
  install path refuses with a teaching error naming the constraint and the
  lanes that DO run on Linux. No silent absence.
- Posture: Apple Silicon unified memory means RAM is the VRAM pool. The fit
  math reuses the CPU/RAM posture path with the Mac memory-pressure signal
  (and the existing `MemAvailable` hard floor) instead of a discrete-GPU
  census. `blazar explain` and `/api/capacity` must name the unified-memory
  arithmetic.

## Surfaces (integration contract at implementation time)

Runtime lane module + `EngineKind::Mlx` + engine manager install/update/use/
prune + doctor currency row + posture fit + `/api/ps` device fields
(`blazar_device`: "mlx") + SETUP honesty row + USAGE engine row + CHANGELOG +
API surface unchanged (OpenAI-compat passthrough) + tests (unit-pinnable
parts: platform gate, version compare, posture math) + live validation.

## Validation plan (requires Mac hardware)

1. Install lane on Apple Silicon; boot-smoke gate passes; child on loopback.
2. Chat completion 200 with correct `system_fingerprint`; TTFT/TPOT recorded.
3. Fit governance: a model larger than unified memory produces the teaching
   refusal with the arithmetic named, never an OOM crash.
4. Doctor currency row live against PyPI; `engine update --kind mlx` walk.
5. Restart durability + teardown leak checks per the lifecycle contract.

## Non-goals

- No MLX on Linux (framework is Apple-only; the gate says so loudly).
- No custom kernels or framework forks — we wrap upstream `mlx-lm` releases,
  same posture as every other lane.
