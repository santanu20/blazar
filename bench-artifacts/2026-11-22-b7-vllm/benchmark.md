# Blazar benchmark matrix

- **date**: 2026-11-22 15:29:05
- **model**: `qwen3-8b-awq.d` (5831 MiB)
- **gpu**: NVIDIA GeForce RTX 4070 Laptop GPU / driver 595.99.02
- **engines**: inventory, vllm-0.34.0
- **harness**: bench_matrix v4 - `--blazar-bin target/release/blazar --model qwen3-8b-awq --engines vllm-0.34.0 --quality-only --artifacts-dir bench-artifacts/2026-11-22-b7-vllm --md B7-vllm.md`
- **blazar**: `blazar 0.34.3` (sandbox daemon binary)
- **quality lane**: error - quality lane ran but produced no scored rows: quality cell crashed: MemAvailable 5781 MiB < needed ~8294 MiB (model 5816 MiB + headroom). A co-resident blazar/ollama 

## Notable findings

- 1 cell(s) aborted on ENVIRONMENT guards (GPU/RAM co-residency), not product behavior - see Failed cells.

## Engine coverage

_Engine selection policy: latest installed version per engine kind._
| Engine | Kind | Lane | ok cells | err cells | Status |
|---|---|---|---:|---:|---|
| vllm-0.34.0 | vllm | text (gateway venv) | 0 | 1 | attempted, all cells errored (see cells.jsonl) |
| 2023.11.14-2 | - | - | 0 | 0 | excluded: kind 'piper' has no bench lane in this harness |
| b12606-cuda | - | - | 0 | 0 | excluded: kind 'llamacpp' has no bench lane in this harness |
| b5907 | - | - | 0 | 0 | excluded: kind 'whisper' has no bench lane in this harness |
| fork-AtomicBot-ai_atomic-llama-cpp-turboquant-8d935078-cuda | - | - | 0 | 0 | excluded: latest-per-kind policy: superseded by b12606-cuda |
| fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | - | - | 0 | 0 | excluded: latest-per-kind policy: superseded by b12606-cuda |
| kokoro-1.0 | - | - | 0 | 0 | excluded: kind 'kokoro' has no bench lane in this harness |
| litert-0.19.2 | - | - | 0 | 0 | excluded: model-format sweep: this campaign sweeps a non-litertlm model; the openaihost lane serves .litertlm bundles (rerun with --model <litertlm-row>) |
| master-1043-3be2378 | - | - | 0 | 0 | excluded: kind 'sdcpp' has no bench lane in this harness |
| mlx-0.32.2 | - | - | 0 | 0 | excluded: kind 'mlx' has no bench lane in this harness |
| sglang-0.5.22 | - | - | 0 | 0 | excluded: model-format sweep: this campaign sweeps a GGUF file; sglang serves HF safetensors checkpoints (rerun with --model <safetensors-row>) |
| v0.9.5 | - | - | 0 | 0 | excluded: kind 'mistralrs' has no bench lane in this harness |

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

- `vllm-0.34.0` / quality / {'config': 'default'}: quality cell crashed: MemAvailable 5781 MiB < needed ~8294 MiB (model 5816 MiB + headroom). A co-resident blazar/ollama engine is likely holding memory: stop...

## Reading this report

- `direct` = raw child spawn on a probed free port (no gateway); `blazar` = full gateway path inside a sandboxed daemon (resolved engine argv in `child_argv`); `ollama` = HTTP-only reference against the host service.
- decode counts ALL emitted tokens (content + reasoning); engine `usage` counters are authoritative when present. GPU/power peaks sampled at ~1.2 s cadence.
- per-run spread, cold-start/resource detail, and every raw cell live in `cells.jsonl`; tables here are per-config medians.
