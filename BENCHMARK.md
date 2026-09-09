# Pallama benchmark matrix

- **date**: 2026-09-09 21:54:28
- **model**: `Qwen3.5-9B-Q4_K_M.gguf` (5417 MiB)
- **gpu**: NVIDIA GeForce RTX 4070 Laptop GPU / driver 580.173.02
- **engines**: b10809-cuda (llamacpp)
- **harness**: bench_matrix v1 — `--model Qwen3.5 --md BENCHMARK.md --artifacts-dir ~/.cache/pallama-bench-matrix/20260909-195826 --skip-ppl --skip-greedy --providers ollama --engines b10809-cuda`

## Speed (serving, streaming)

| engine | kind | provider | params | ttft p50 (ms) | decode t/s | prefill t/s | GPU peak (MiB) |
|---|---|---|---|---|---|---|---|
| b10809 | llamacpp | direct | ctx=4096 np=1 | 151 | 39.0 | 26.5 | 5208 |
| b10809 | llamacpp | direct | ctx=4096 np=4 | 151 | 39.1 | 25.1 | 5337 |
| b10809 | llamacpp | direct | ctx=16384 np=1 | 156 | 39.0 | 25.5 | 5583 |
| b10809 | llamacpp | direct | ctx=16384 np=4 | 135 | 39.0 | 25.9 | 5722 |
| b10809-cuda | llamacpp | direct | ctx=4096 np=1 | 119 | 41.4 | 28.4 | 5314 |
| b10809-cuda | llamacpp | direct | ctx=4096 np=4 | 110 | 41.4 | 29.0 | 5460 |
| b10809-cuda | llamacpp | direct | ctx=16384 np=1 | 110 | 41.7 | 29.3 | 5710 |
| b10809-cuda | llamacpp | direct | ctx=16384 np=4 | 110 | 41.4 | 28.7 | 5848 |
| b10809 | llamacpp | pallama | config=default | 144 | 38.9 | 26.1 | 0 |
| b10809-cuda | llamacpp | pallama | config=default | 132 | 33.8 | 24.8 | 0 |
| ollama-host | ollama | ollama | reference=True | 76072 | 37.0 | 12.9 | 0 |

## Quality — perplexity (identical pinned args)

| engine | perplexity | wall (s) | note |
|---|---|---|---|
| b10809 | 27.6528 | 12.2 | lower = better text fit |
| b10809-cuda | 27.1942 | 9.8 | lower = better text fit |
| v0.9.3 | - | - | llama-perplexity is llama-server-family only |

## Quality — greedy parity vs llama.cpp-direct reference

| engine | exact matches | ratio mean | ratio min | first divergence (median chars) |
|---|---|---|---|---|
| b10809 | 20/20 | 1.0 | 1.0 | 267.0 |
| b10809-cuda | 15/20 | 0.8624 | 0.1421 | 236.5 |
| v0.9.3 | -/- | - | - | - |

## Feature matrix

| feature | b10809-cuda | ollama(documented) |
|---|---|---|
| `anthropic-api` | - | - |
| `ctx-override` | Y | Y |
| `embeddings` | Y | Y |
| `grammar-gbnf` | Y | - |
| `json-schema` | Y | Y |
| `kv-quant` | Y | - |
| `lora-adapter` | Y | Y |
| `metrics-endpoint` | Y | - |
| `paged-attn` | Y | - |
| `parallel-np` | Y | Y |
| `quant-on-load` | Y | - |
| `rerank` | Y | - |
| `slots-sessions` | Y | - |
| `spec-decode` | Y | - |
| `tokenize-endpoint` | Y | Y |
| `vision-mmproj` | Y | Y |

## Failed cells

- `v0.9.3` / direct / {'ctx': 4096, 'np': 1}: child exited rc=1 during load; last output: Error: Num GPU blocks is 0. This means there is not enough memory. Either reduce the memory amount/utilization/context size or disable PagedAttention. | 
- `v0.9.3` / direct / {'ctx': 4096, 'np': 4}: child exited rc=1 during load; last output: Error: Num GPU blocks is 0. This means there is not enough memory. Either reduce the memory amount/utilization/context size or disable PagedAttention. | 
- `v0.9.3` / direct / {'ctx': 16384, 'np': 1}: child exited rc=1 during load; last output: Error: Num GPU blocks is 0. This means there is not enough memory. Either reduce the memory amount/utilization/context size or disable PagedAttention. | 
- `v0.9.3` / direct / {'ctx': 16384, 'np': 4}: child exited rc=1 during load; last output: Error: Num GPU blocks is 0. This means there is not enough memory. Either reduce the memory amount/utilization/context size or disable PagedAttention. | 
- `v0.9.3` / pallama / {'config': 'default'}: pallama cell crashed: HTTP Error 502: Bad Gateway
- `v0.9.3` / greedy / {'greedy': True}: child failed to become healthy
- `v0.9.3` / ppl / {'ppl': 2048}: llama-perplexity is llama-server-family only

## Reading this report

- `direct` = raw child spawn on a probed free port (no gateway).
- `pallama` = full gateway path inside a sandboxed daemon (profile compiler, routing, auth).
- `ollama` = HTTP-only reference against the host service, one cell.
- GPU peaks are sampled at ~1.2 s cadence; very short bursts may undersample.
- Cells append to `cells.jsonl` and resume across reruns (keyed on engine/provider/params/model/harness-version).
