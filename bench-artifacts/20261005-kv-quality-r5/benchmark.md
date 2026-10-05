# Blazar benchmark matrix

- **date**: 2026-10-05 17:08:28
- **model**: `Qwen3.5-9B-Q4_K_M.gguf` (5366 MiB)
- **gpu**: NVIDIA GeForce RTX 4070 Laptop GPU / driver 595.91.07
- **engines**: b11393-cuda, inventory, ollama-host
- **harness**: bench_matrix v4 — `--artifacts-dir bench-artifacts/20261005-kv-quality-r5 --model qwen3.5-9b --engines b11393-cuda --providers blazar --skip-greedy --skip-tools --skip-features --skip-idle --skip-reshape --skip-ctxcurve --skip-variants --skip-media --skip-ppl --skip-conc --runs 1`
- **blazar**: `blazar 0.20.0` (sandbox daemon binary)
- **quality lane**: ran — 1 scored row(s); suites: code, instruct, multilingual, niah, reason, safety, schema
- all blazar-owned rows measured by `blazar 0.20.0`

## Speed (serving, streaming)

| engine | kind | provider | params | ttft p50 (ms) | ttft p99 (ms) | itl p50 (ms) | itl p99 (ms) | decode t/s | prefill t/s (cold) | prefill t/s (cached) | tokens src |
|---||---||---||---||---||---||---||---||---||---||---||
| b11393-cuda | llamacpp | blazar | config=default child: np=4 ctx=65536 | 191 | 191 | 25 | 26 | 40.7 | 1338.6 | 1338.6 | chunks |
| b11393-cuda | llamacpp | blazar | config=single-stream child: np=1 ctx=16384 | 121 | 121 | 24 | 28 | 40.9 | 1297.2 | 1297.2 | chunks |
| ollama-host | ollama | ollama | reference=True NOTE: serves 'qwen3.5:9b' - t/s NOT comparable | 121 | 121 | 25 | 75 | 40.7 | 1404.7 | 1404.7 | engine_counters |

## Resources & cold start

| engine | provider | params | load s | daemon boot s | cold 1st req s | GPU peak (MiB) | GPU power (W) | RSS peak (MiB) | teardown |
|---||---||---||---||---||---||---||---||---||
| b11393-cuda | blazar | config=default | - | 0.53 | 5.49 | 6332 | 56.2 | 1836 | ok |
| b11393-cuda | blazar | config=single-stream | - | 0.52 | 4.93 | 6078 | 55.1 | 1682 | ok |
| ollama-host | ollama | reference=True | - | - | - | 6566 | 55.0 | - | ok |

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

- `direct` = raw child spawn on a probed free port (no gateway).
- `blazar` = full gateway path inside a sandboxed daemon (profile compiler, routing, auth); `child_argv` in cells.jsonl holds the resolved engine argv.
- `ollama` = HTTP-only reference against the host service, one cell.
- decode counts ALL emitted tokens (content + reasoning/thinking); `usage`/engine counters are authoritative when present (`tokens src`).
- prefill t/s (cold) = prompt tokens / first-token time on an uncached token-targeted prompt; (cached) = same prompt re-sent (child prompt-cache path). ollama prefill uses engine-side prompt_eval counters, which EXCLUDE template tokens — ollama prefill reads high relative to the 512-token lanes.
- blazar speed rows show the resolved slot/context shape (`child: np=… ctx=…`) parsed from the recorded child argv — auto-slots may differ from the direct rows' explicit np.
- decode-lane TTFT rides the child's prompt cache after run 1 (warm path); the prefill-lane cold/cached pair is the honest cache story at real prompt sizes.
- GPU/power peaks sampled at ~1.2 s cadence (max across NVIDIA GPUs); very short bursts may undersample.
- Cells append to `cells.jsonl` and resume across reruns (keyed on engine/provider/params/model/harness-version).
