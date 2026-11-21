# Blazar benchmark matrix

- **date**: 2026-11-21 11:48:11
- **model**: `qwen2.5-coder-0.5b-instruct.d` (953 MiB)
- **gpu**: NVIDIA GeForce RTX 4070 Laptop GPU / driver 595.99.02
- **engines**: 2023.11.14-2, b12578-cuda, b5907, inventory, kokoro-1.0, litert-0.19.2, master-1043-3be2378, mlx-0.32.2, sglang-0.5.23, v0.9.5, vllm-0.34.0
- **harness**: bench_matrix v4 - `--model qwen2.5-coder-0.5b-instruct --media-only --media-runs 3 --fresh`
- **blazar**: `blazar 0.34.0` (sandbox daemon binary)
- **quality lane**: not-measured - no quality rows in this campaign (lane not reached for the selected providers/engines)

## Notable findings

- 3 cell(s) aborted on ENVIRONMENT guards (GPU/RAM co-residency), not product behavior - see Failed cells.

## Engine coverage

_Engine selection policy: latest installed version per engine kind._
| Engine | Kind | Lane | ok cells | err cells | Status |
|---|---|---|---:|---:|---|
| 2023.11.14-2 | piper | media (tts) | 2 | 0 | benchmarked |
| mlx-0.32.2 | mlx | text (direct venv + gateway) | 0 | 0 | model-format sweep: GGUF/safetensors campaign; this lane serves MLX-layout checkpoints (rerun with --model <mlx-row>) |
| v0.9.5 | mistralrs | text (direct + gateway) | 0 | 0 | no cells in this campaign |
| master-1043-3be2378 | sdcpp | media | 2 | 3 | benchmarked |
| sglang-0.5.23 | sglang | text (direct venv + gateway) | 0 | 0 | no cells in this campaign |
| kokoro-1.0 | kokoro | media (tts) | 1 | 0 | benchmarked |
| litert-0.19.2 | openaihost | text (direct venv + gateway) | 0 | 0 | model-format sweep: GGUF/safetensors campaign; this lane serves .litertlm bundles (rerun with --model <litertlm-row>) |
| vllm-0.34.0 | vllm | text (gateway venv) | 0 | 0 | no cells in this campaign |
| b12578-cuda | llamacpp | text (direct + gateway) | 0 | 0 | no cells in this campaign |
| b5907 | whisper | media | 1 | 0 | benchmarked |
| fork-AtomicBot-ai_atomic-llama-cpp-turboquant-8d935078-cuda | - | - | 0 | 0 | excluded: latest-per-kind policy: superseded by b12578-cuda |
| fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | - | - | 0 | 0 | excluded: latest-per-kind policy: superseded by b12578-cuda |

## Feature matrix

| feature | ollama(documented) |
|---|---|
| `anthropic-api` | - |
| `ctx-override` | Y |
| `embeddings` | Y |
| `grammar-gbnf` | - |
| `json-schema` | Y |
| `kv-quant` | - |
| `lora-adapter` | Y |
| `metrics-endpoint` | - |
| `paged-attn` | - |
| `parallel-np` | Y |
| `quant-on-load` | - |
| `rerank` | - |
| `slots-sessions` | - |
| `spec-decode` | - |
| `tokenize-endpoint` | Y |
| `vision-mmproj` | Y |

## Failed cells


**environment** (box/co-residency guards - NOT blazar defects):

- `master-1043-3be2378` / media-image / {'size': '512x512', 'steps': [4], 'runs': 3}: media image cell crashed: MemAvailable 6680 MiB < needed ~7899 MiB (model 5500 MiB + headroom). A co-resident blazar/ollama engine is likely holding memory: ...
- `master-1043-3be2378` / media-image / {'size': '512x512', 'steps': [4], 'runs': 3, 'config': 'fa_off'}: media image cell crashed: MemAvailable 6233 MiB < needed ~7899 MiB (model 5500 MiB + headroom). A co-resident blazar/ollama engine is likely holding memory: ...
- `master-1043-3be2378` / media-image / {'size': '512x512', 'steps': [4], 'runs': 3, 'config': 'vae_tiling_on'}: media image cell crashed: MemAvailable 7478 MiB < needed ~7899 MiB (model 5500 MiB + headroom). A co-resident blazar/ollama engine is likely holding memory: ...

## Reading this report

- `direct` = raw child spawn on a probed free port (no gateway); `blazar` = full gateway path inside a sandboxed daemon (resolved engine argv in `child_argv`); `ollama` = HTTP-only reference against the host service.
- decode counts ALL emitted tokens (content + reasoning); engine `usage` counters are authoritative when present. GPU/power peaks sampled at ~1.2 s cadence.
- per-run spread, cold-start/resource detail, and every raw cell live in `cells.jsonl`; tables here are per-config medians.
