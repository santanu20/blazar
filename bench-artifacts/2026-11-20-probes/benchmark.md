# Blazar benchmark matrix

- **date**: 2026-11-20 21:03:35
- **model**: `qwen2.5-0.5b-instruct-q4_k_m.gguf` (468 MiB)
- **gpu**: NVIDIA GeForce RTX 4070 Laptop GPU / driver 595.99.02
- **engines**: 2023.11.14-2, b5454, fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda, inventory, litert-0.19.2, master-1043-3be2378, mlx-0.32.2, ollama-host, probe, sglang-0.5.23, v0.9.5
- **harness**: bench_matrix v4 - `--model qwen2.5-0.5b-instruct --providers --skip-ppl --skip-greedy --skip-features --skip-conc --skip-idle --skip-quality --skip-variants --skip-media --skip-ctxcurve --probes cancel,evict,store --probe-reps 5 --artifacts-dir bench-artifacts/2026-11-20-probes --blazar-bin target/release/blazar`
- **blazar**: `blazar 0.33.0` (sandbox daemon binary)
- **quality lane**: skipped - --skip-quality: no checker-verified receipts back any speed number in this campaign
- all blazar-owned rows measured by `blazar 0.33.0`

## Notable findings

- 4 cell(s) aborted on ENVIRONMENT guards (GPU/RAM co-residency), not product behavior - see Failed cells.

## Engine coverage

_Engine selection policy: latest installed version per engine kind._
| Engine | Kind | Lane | ok cells | err cells | Status |
|---|---|---|---:|---:|---|
| 2023.11.14-2 | piper | media (tts) | 0 | 0 | no cells in this campaign |
| b5454 | whisper | media | 0 | 0 | no cells in this campaign |
| mlx-0.32.2 | mlx | text (direct venv + gateway) | 0 | 0 | model-format sweep: GGUF/safetensors campaign; this lane serves MLX-layout checkpoints (rerun with --model <mlx-row>) |
| v0.9.5 | mistralrs | text (direct + gateway) | 2 | 4 | benchmarked |
| master-1043-3be2378 | sdcpp | media | 0 | 0 | no cells in this campaign |
| sglang-0.5.23 | sglang | text (direct venv + gateway) | 0 | 0 | no cells in this campaign |
| litert-0.19.2 | openaihost | text (direct venv + gateway) | 0 | 0 | model-format sweep: GGUF/safetensors campaign; this lane serves .litertlm bundles (rerun with --model <litertlm-row>) |
| fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | llamacpp | text (direct + gateway) | 4 | 2 | benchmarked |
| b12521-cuda | - | - | 0 | 0 | excluded: latest-per-kind policy: superseded by fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda |
| fork-AtomicBot-ai_atomic-llama-cpp-turboquant-8d935078-cuda | - | - | 0 | 0 | excluded: latest-per-kind policy: superseded by fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda |
| kokoro-1.0 | - | - | 0 | 0 | excluded: kind 'kokoro' has no bench lane in this harness |
| vllm-0.34.0 | - | - | 0 | 0 | excluded: kind 'vllm' has no bench lane in this harness |

## Charts

- deterministic SVG renders of the cells below; source of truth is `cells.jsonl`, charts live in `plots/`.

### Single-stream decode throughput (chart)

<p align="center"><img src="plots/speed-single-stream.svg" alt="Single-stream decode throughput (chart)"></p>

_Median decode t/s per runtime and engine; whiskers span the interquartile range of the 5 runs; the dashed line marks the fastest direct engine. Higher is better. Source: 2026-11-20-probes/cells.jsonl._

### Lifecycle: cold start and idle wake (chart)

<p align="center"><img src="plots/lifecycle-cold-idle.svg" alt="Lifecycle: cold start and idle wake (chart)"></p>

_Seconds to first token after a cold start (page cache dropped) and after idle-policy expiry; blazar keeps weights resident while ollama reloads from disk. Warm-daemon ollama caveat applies. Lower is better. Source: 2026-11-20-probes/cells.jsonl._

## Speed (single-stream, medians)

| engine | provider | params | n | ttft p50 (ms) | ttft p99 (ms) | itl p99 (ms) | decode t/s | prefill t/s (cached) | VRAM peak (MiB) |
|---||---||---||---||---||---||---||---||---||
| fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | blazar | config=default child: np=2 ctx=32768 | 1 | 9 | 11 | 6 | 308.3 | 34165.9 | 1279 |
| fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | blazar | config=single-stream child: np=1 ctx=16384 | 1 | 9 | 12 | 4 | 337.2 | 62753.3 | 1193 |
| ollama-host | ollama | reference=True NOTE: serves 'qwen2.5:0.5b' - t/s NOT comparable | 1 | 13 | 19 | 4 | 346.7 | 128605.9 | 726 |
| v0.9.5 | blazar | config=default child: ctx=16384 | 1 | 6 | 8 | - | 0.0 | - | 6248 |
| v0.9.5 | blazar | config=single-stream child: ctx=16384 | 1 | 5 | 5 | - | 0.0 | - | 6154 |

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

**product** (engine/gateway behavior):

- `v0.9.5` / blazar (2 cells): zero tokens served (model failed to load/serve) [{'config': 'default'}; {'config': 'single-stream'}]

**environment** (box/co-residency guards - NOT blazar defects):

- `fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda` / blazar / {'config': 'default'}: blazar cell crashed: MemAvailable 7821 MiB < needed ~8019 MiB (model 5596 MiB + headroom). A co-resident blazar/ollama engine is likely holding memory: stop ...
- `fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda` / blazar / {'config': 'single-stream'}: blazar cell crashed: MemAvailable 7797 MiB < needed ~8019 MiB (model 5596 MiB + headroom). A co-resident blazar/ollama engine is likely holding memory: stop ...
- `v0.9.5` / blazar / {'config': 'default'}: blazar cell crashed: MemAvailable 7627 MiB < needed ~8019 MiB (model 5596 MiB + headroom). A co-resident blazar/ollama engine is likely holding memory: stop ...
- `v0.9.5` / blazar / {'config': 'single-stream'}: blazar cell crashed: MemAvailable 7743 MiB < needed ~8019 MiB (model 5596 MiB + headroom). A co-resident blazar/ollama engine is likely holding memory: stop ...

## Reading this report

- `direct` = raw child spawn on a probed free port (no gateway); `blazar` = full gateway path inside a sandboxed daemon (resolved engine argv in `child_argv`); `ollama` = HTTP-only reference against the host service.
- decode counts ALL emitted tokens (content + reasoning); engine `usage` counters are authoritative when present. GPU/power peaks sampled at ~1.2 s cadence.
- per-run spread, cold-start/resource detail, and every raw cell live in `cells.jsonl`; tables here are per-config medians.
