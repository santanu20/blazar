# Hugging Face file-type coverage

Which model file types Pallama can **pull** and **serve**, per engine lane,
as of 2026-09-17 (v0.6.x era). Evidence tags:

- `PROVEN` — exercised live on this project's test bench (RTX 4070 8 GiB,
  Linux, driver 580) with the cited result.
- `EXPECTED` — the engine documents native support and the flag is wired
  through Pallama's tuning surface, but no live bench run exists yet.
- `CONVERT-FIRST` — no engine serves it natively; convert with the
  upstream tool, then pull the converted artifact.
- `N/A` — no Pallama engine has a runtime for it (documented honestly
  rather than half-wired).

## Summary table

| File type | Pull lane | llamacpp | sglang 0.5.19 | mistral.rs v0.9.3 | Verdict |
|---|---|---|---|---|---|
| GGUF single-file (any quant: Q2_K…Q8_0, IQ, M2, F16, BF16, TQ) | catalog + HF-name | **native** | refused (teaching) | native | `PROVEN` (Q4_K_M + F16 + Q8_0-class via llamacpp; mistral.rs GGUF lane serving) |
| GGUF sharded (`-00001-of-…`) | catalog + HF-name | native (auto-stitch) | refused | native (shard staging) | `PROVEN` (staging lane pinned by unit test) |
| GGUF + embedded MTP/nextn head | HF-name | **native** (`--spec-type draft-mtp` auto) | refused | mtp via `--mtp-model` lane | `PROVEN` (qwen3.5-9b-mtp, 1.35x dense, control 1.00x) |
| safetensors dir, BF16/F16 | HF-name | refused | **native** | native | `PROVEN` (sglang 0.612 quality; mistral.rs 0.575) |
| safetensors dir, AWQ | HF-name | refused | **native** (awq_marlin auto) | AWQ lane exists | `PROVEN` on sglang: marlin ~1.7x BF16 same-window decode, 369 tok/s through gateway = zero overhead |
| safetensors dir, GPTQ | HF-name | refused | **native** (auto-detect) | GPTQ lane | `PROVEN` on sglang (0.5B Int4 pull + serve bench, see below) |
| safetensors dir, FP8 (e4m3/e5m2) | HF-name | refused | native on Ada/Hopper | no | `PROVEN` — RedHatAI/Qwen2.5-0.5B-Instruct-FP8-dynamic (876 MiB): cold 31.7 s (needs `model_load_timeout_secs = 600` + `cuda_graph_backend_prefill = "disabled"` on this laptop), warm ~260-275 tok/s triton w8a8-dynamic |
| PyTorch `.bin` pickle shards (legacy HF) | HF-name | refused | loads via transformers loader | loads via loader | `EXPECTED` — works upstream, prefer safetensors remasters; unbenched here |
| MLX (`.mlx` dirs, mlx-lm format) | HF-name fetches files | no | no | no | `N/A` on Linux — Apple-silicon ecosystem format; convert with `mlx-lm convert` (on a Mac) to safetensors, then serve on any lane |
| EXL2 / ExLlamaV2 | fetches files | no | no | no | `N/A` — ExLlamaV2 runtime is not among Pallama's engines |
| bitsandbytes int8/nf4 | fetches files | no | no | no | `N/A` — training-time quant; no serving runtime in the three engines |
| AQLM / HQQ / EETQ / SqueezeLLM | fetches files | no | `--quantization` knob paths exist upstream | no | `EXPECTED` at best via sglang `quantization` tuning knob + `extra_args`; niche, unbenched, treat as unsupported until proven |
| GGUF+LoRA adapter (`.gguf`/`.safetensors` lora) | fetches files | native (`--lora`) | PEFT/`--lora-paths` | native `--lora alias=path` | `PROVEN` wiring (llamacpp `--lora` + per-pair scale; mistral.rs ALIAS=SOURCE; sglang `--enable-lora`) |

Pull lanes: the **catalog** lane searches GGUF repos (`pallama search`);
the **HF-name** lane (`pallama pull Qwen/Qwen2.5-0.5B-Instruct-AWQ`) fetches
any HF repo layout (safetensors dirs, quantized variants, even MLX/EXL2
files) — pulling always works; *serving* is where the table above applies.

## Live probes behind this table

| Probe | Result |
|---|---|
| qwen2.5-0.5b Q4_K_M GGUF → llamacpp | matrix bench, 508-618 tok/s single-stream |
| qwen2.5-0.5b-instruct-fp16 GGUF → llamacpp | quant-matched matrix row (0.561 quality) |
| qwen2.5-0.5b-instruct BF16 → sglang | 0.612 quality, 752 tok/s conc4 (matrix) |
| qwen2.5-0.5b-instruct BF16 → mistral.rs | 0.575 quality, 26 ms TTFT, 8 s cold (matrix + PA fix) |
| Qwen2.5-0.5B-Instruct-AWQ → sglang | marlin ~1.7x BF16 same-window; 708 MiB weights; cold 21.8 s with prefill-graph knob |
| AWQ vanilla path (`quantization = "awq"`) | **env-blocked**: sglang JIT kernel fails under nvcc ≤ 12.0 (`assert` undefined in `awq_dequantize.cuh`) — upgrade CUDA or sglang |
| qwen3.5-9b-mtp GGUF (nextn head) → llamacpp | draft-mtp 1.35x dense, control 1.00x |
| mistral.rs `--isq` (in-situ quant) | **upstream-broken** in v0.9.3: request-time `model_error` on every quant level, both sources, free or busy GPU (Pallama passthrough itself proven) |
| flashinfer attention (sglang) | **env-blocked**: JIT nvcc 12.0 rejects `--compress-mode=size` |

## Notes

- The sglang `quantization` knob (`model_overrides.<m>.sglang.quantization`)
  forces a specific dequant path; the default is auto-detect from the
  repo's `quantization_config`, which is what the proven rows used.
- Engine-format mismatches fail loudly with a teaching error naming the
  remedy (`pallama engine use`, `engine install --kind`, or conversion).
- With `[engine_routing] mode = "auto"`, the format column of this table
  is exactly what routes: GGUF → llamacpp, safetensors → sglang (policy
  may prefer mistral.rs for latency), nothing-installed → strict teaching.
