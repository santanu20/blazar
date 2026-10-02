# MLX lane (F8) — implemented design

Status: **implemented on `feat/next-wave`** (2026-10-02). The original spec
staged this lane behind Mac hardware; MLX now ships an official CUDA backend,
so the lane ships for Linux+NVIDIA too and was live-validated on the dev box.

## What the lane is

`EngineKind::Mlx` — a pip-venv lane wrapping `mlx_lm.server` (Apple's official
OpenAI-compatible server), identical in shape to the sglang lane:

- **Linux + NVIDIA:** `mlx-lm==0.32.0` + `mlx[cuda12]==0.32.2` (dual pin).
- **Apple Silicon:** `mlx-lm` only (plain `mlx` resolves on Darwin).
- **CPU-only Linux / other platforms:** install refuses with teaching (the
  CUDA wheels are the point; CPU-only Linux gains nothing over llamacpp).

### Why `mlx[cuda12]` and not the `mlx-cuda` wheel

The standalone `mlx-cuda` PyPI wheel (0.30.0) was the pre-0.32 distribution.
Since mlx-lm 0.32, CUDA rides the **standard `mlx` package as extras**
(`mlx[cuda12]`, `mlx[cuda13]`, `mlx[cpu]`) — that is what mlx-lm's own
dependency metadata resolves. `cuda12` (not `cuda13`) is pinned for the
broadest driver base (>= 525) and one tested surface. The dual version pins
are visible in the store row's asset label:
`pip:mlx-lm==0.32.0+mlx[cuda12]==0.32.2`.

## Install / update / doctor

- `blazar engine install --kind mlx` — venv (uv when present) + shim
  `mlx-server` (`python -m mlx_lm.server`) + boot-smoke
  (`import mlx_lm.server`) inside the same rollback discipline as sglang
  (install_with_rollback): an unbootable venv never becomes an activatable
  row. Disk preflight requires 4 GiB free; installs stream pip output.
- `blazar engine update --kind mlx [version]` + `--all` walk (venv lanes are
  adjacent in the fixed walk order: llamacpp, mistralrs, sglang, mlx, sdcpp,
  whisper).
- `blazar doctor` shows a PyPI currency row for the lane (`mlx-lm pip lane`).
- `lane_max_n(Mlx) = 8` (plane default — unprobed, never under-promised).

## Serving

- Spawn: `mlx-server --model <dir> --host 127.0.0.1 --port <p>` + generic
  `model_overrides.argv` passthrough. Loopback bind is load-bearing
  isolation (not every mlx-lm build ships `--api-key`); the supervisor's
  child-auth mint appends the secret when the installed build supports it
  (flag surface is probed into the manifest at install).
- Health: poll `GET /v1/models` until 200 — mlx_lm.server has no dedicated
  `/health` route; the OpenAI surface answers once weights are loaded.
- Warm peg: same JIT class as sglang (python import chain is torch-class
  slow; `[warm_peg] mlx` overrides, default on).
- TCP transport only (like sglang); Linux venv gets the same
  `LD_LIBRARY_PATH` nvidia-lib surgery so the `libmlxcuda` extension finds
  the pip-provided CUDA libs.
- Posture: rides the sglang-proven non-GGUF path (VRAM ledger via probe);
  no context-length flags exist on mlx_lm.server, so no KV preflight
  applies.

## Model routing (two-sided, never silent)

- Detection: `ModelRow::is_mlx()` — a strict `mlx` token in the row's
  name/repo/path **and** the path being a directory (MLX dirs are
  quantized-safetensors-shaped; the token + dir gate is the discriminator,
  consulted before `is_quantized_safetensors()`).
- `route_format(mlx = true)` routes to the Mlx lane **before** the
  quantized-safetensors arm — an installed sglang must never shadow an MLX
  dir, and only mlx-lm can decode those quants.
- Absent lane: `LaneError::MlxUnserved` teaching at serve time —
  `blazar engine install --kind mlx`; pulls still succeed (files are
  lane-agnostic) and the search/pull hints name the install command.
- Pull: mlx-community dirs ride the existing safetensors-dir machinery
  (sharded safetensors + index.json + config.json, which carries the
  quantization block that makes the dir MLX).

## Config

No `[mlx]` table exists yet — extra flags ride the generic
`[model_overrides.<name>] argv` passthrough. The knob-hint block says so
explicitly rather than advertising an unsettable table. A real table lands
if/when mlx-lm knobs need first-class pins.

## Honest positioning

Native CUDA lanes (llamacpp) remain faster for models available as
GGUF/safetensors. The mlx lane's value: the **mlx-only quant ecosystem**
(mlx-community repos) becomes servable, on both CUDA Linux and Apple
Silicon, under one gateway (keys, admission, metrics, n-choices plane
rules).

## Validation plan (executed)

Live no-mock on the dev box (Linux + NVIDIA): real venv install with CUDA
wheels, shim `--help` probe receipt, pull
`mlx-community/Qwen2.5-0.5B-Instruct-4bit`, chat 200 through the gateway,
`blazar bench` vs the same-size GGUF on llamacpp (receipt names both quant
formats), full workspace suite, hygiene gate, deploy ritual
(stop → copy → md5 verify → start).
