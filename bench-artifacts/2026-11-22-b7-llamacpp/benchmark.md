# Blazar benchmark matrix

- **date**: 2026-11-22 15:13:00
- **model**: `qwen2.5-0.5b-instruct-q4_k_m.gguf` (468 MiB)
- **gpu**: NVIDIA GeForce RTX 4070 Laptop GPU / driver 595.99.02
- **engines**: b12606-cuda, inventory
- **harness**: bench_matrix v4 - `--blazar-bin target/release/blazar --model qwen2.5-0.5b-instruct --engines b12606-cuda --quality-only --artifacts-dir bench-artifacts/2026-11-22-b7-llamacpp --md B7-llamacpp.md`
- **blazar**: `blazar 0.34.3` (sandbox daemon binary)
- **quality lane**: ran - 2 scored row(s); suites: code, instruct, multilingual, niah, reason, safety, schema

## Engine coverage

_Engine selection policy: latest installed version per engine kind._
| Engine | Kind | Lane | ok cells | err cells | Status |
|---|---|---|---:|---:|---|
| b12606-cuda | llamacpp | text (direct + gateway) | 2 | 0 | benchmarked |
| 2023.11.14-2 | - | - | 0 | 0 | excluded: kind 'piper' has no bench lane in this harness |
| b5907 | - | - | 0 | 0 | excluded: kind 'whisper' has no bench lane in this harness |
| fork-AtomicBot-ai_atomic-llama-cpp-turboquant-8d935078-cuda | - | - | 0 | 0 | excluded: latest-per-kind policy: superseded by b12606-cuda |
| fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | - | - | 0 | 0 | excluded: latest-per-kind policy: superseded by b12606-cuda |
| kokoro-1.0 | - | - | 0 | 0 | excluded: kind 'kokoro' has no bench lane in this harness |
| litert-0.19.2 | - | - | 0 | 0 | excluded: model-format sweep: this campaign sweeps a non-litertlm model; the openaihost lane serves .litertlm bundles (rerun with --model <litertlm-row>) |
| master-1043-3be2378 | - | - | 0 | 0 | excluded: kind 'sdcpp' has no bench lane in this harness |
| mlx-0.32.2 | - | - | 0 | 0 | excluded: kind 'mlx' has no bench lane in this harness |
| sglang-0.5.22 | - | - | 0 | 0 | excluded: model-format sweep: this campaign sweeps a GGUF file; sglang serves HF safetensors checkpoints (rerun with --model <safetensors-row>) |
| v0.9.5 | - | - | 0 | 0 | excluded: kind 'mistralrs' has no bench lane in this harness |
| vllm-0.34.0 | - | - | 0 | 0 | excluded: model-format sweep: this campaign sweeps a GGUF file; the vllm lane serves HF safetensors checkpoints (rerun with --model <safetensors-row>) |

## Charts

- deterministic SVG renders of the cells below; source of truth is `cells.jsonl`, charts live in `plots/`.

### Quality suites (chart)

<p align="center"><img src="plots/quality-suites.svg" alt="Quality suites (chart)"></p>

_Deterministic-checker pass rates per suite (reasoning, instruction following, code, schema, needle-in-haystack, multilingual, safety) across runtimes on the identical seeded task set. Source: 2026-11-22-b7-llamacpp/cells.jsonl._

### Gateway quality parity (chart)

<p align="center"><img src="plots/quality-parity.svg" alt="Gateway quality parity (chart)"></p>

_Per-suite pass-rate delta of routing through blazar vs the same engine build driven directly; bars hugging the zero axis are the gateway being output-transparent. Source: 2026-11-22-b7-llamacpp/cells.jsonl._

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

## Reading this report

- `direct` = raw child spawn on a probed free port (no gateway); `blazar` = full gateway path inside a sandboxed daemon (resolved engine argv in `child_argv`); `ollama` = HTTP-only reference against the host service.
- decode counts ALL emitted tokens (content + reasoning); engine `usage` counters are authoritative when present. GPU/power peaks sampled at ~1.2 s cadence.
- per-run spread, cold-start/resource detail, and every raw cell live in `cells.jsonl`; tables here are per-config medians.
