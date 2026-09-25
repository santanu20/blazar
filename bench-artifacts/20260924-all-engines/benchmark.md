# Blazar benchmark matrix

- **date**: 2026-09-25 01:25:35
- **model**: `Qwen3.5-9B-Q4_K_M.gguf` (5417 MiB)
- **gpu**: NVIDIA GeForce RTX 4070 Laptop GPU / driver 580.178.04
- **engines**: b11147-cuda, b5130, inventory, master-890-74988b2, ollama-host, piper, v0.9.3
- **harness**: bench_matrix v2 — `--blazar-bin target/release/blazar --model Qwen3.5-9B --providers blazar --skip-media --skip-ppl --skip-ctxcurve --skip-tools --skip-greedy --artifacts-dir bench-artifacts/20260924-all-engines --md BENCHMARK.md`
- **blazar**: `blazar 0.11.0` (sandbox daemon binary)
- all blazar-owned rows measured by `blazar 0.11.0`

## Notable findings

- gateway vs direct decode (`b11147-cuda`): 41.0 vs 41.7 t/s = -1.7% (parity).
- gateway vs direct decode (`b11147-cuda`): 41.1 vs 41.7 t/s = -1.5% (parity).
- gateway vs direct decode (`v0.9.3`): 40.9 vs 19.6 t/s = +109.1% (gain).
- gateway vs direct decode (`v0.9.3`): 40.9 vs 19.6 t/s = +109.3% (gain).
- gateway vs direct decode (`v0.9.3`): 40.9 vs 19.6 t/s = +109.1% (gain).
- concurrency system throughput (`b11147-cuda`, 1 streams): 39.2 vs direct 38.8 t/s = 1.01x.
- concurrency system throughput (`b11147-cuda`, 2 streams): 39.6 vs direct 38.8 t/s = 1.02x.
- concurrency system throughput (`b11147-cuda`, 4 streams): 39.8 vs direct 38.8 t/s = 1.02x.
- concurrency system throughput (`b11147-cuda`, 8 streams): 39.5 vs direct 38.8 t/s = 1.02x.
- kv=q8_0 (`b11147-cuda`): decode -1.3% vs baseline
- spec=ngram-simple (`b11147-cuda`): decode -2.4% vs baseline
- mmproj=True (`b11147-cuda`): decode -1.7% vs baseline
- gateway greedy transparency (`b11147-cuda`): 20/20 exact vs same-engine direct — transparent.
- 1 cell(s) aborted on ENVIRONMENT guards (GPU/RAM co-residency), not product behavior — see Failed cells.

## Speed (serving, streaming)

| engine | kind | provider | params | ttft p50 (ms) | ttft p99 (ms) | itl p50 (ms) | itl p99 (ms) | decode t/s | prefill t/s (cold) | prefill t/s (cached) | tokens src |
|---||---||---||---||---||---||---||---||---||---||---||
| b11147-cuda | llamacpp | direct | ctx=4096 np=1 | 110 | 114 | 24 | 25 | 41.7 | 1342.6 | 8201.4 | chunks |
| b11147-cuda | llamacpp | direct | ctx=4096 np=4 | 109 | 112 | 24 | 25 | 41.5 | 1364.6 | 8307.5 | chunks |
| b11147-cuda | llamacpp | direct | ctx=16384 np=1 | 109 | 112 | 24 | 25 | 41.5 | 1367.9 | 8292.5 | chunks |
| b11147-cuda | llamacpp | direct | ctx=16384 np=4 | 108 | 112 | 24 | 25 | 41.5 | 1407.7 | 8209.3 | chunks |
| b11147-cuda | llamacpp | direct | ctx=4096 kv=q8_0 np=1 | 108 | 119 | 24 | 25 | 41.1 | 1374.1 | 8312.1 | chunks |
| b11147-cuda | llamacpp | direct | ctx=4096 np=1 spec=ngram-simple | 118 | 122 | 25 | 25 | 40.7 | 1343.9 | 7642.3 | chunks |
| b11147-cuda | llamacpp | direct | ctx=4096 mmproj=True np=1 | 116 | 121 | 24 | 26 | 41.0 | 1339.0 | 7727.7 | chunks |
| b11147-cuda | llamacpp | blazar | config=default child: np=1 ctx=16384 | 120 | 123 | 24 | 25 | 41.0 | 1365.6 | 7505.3 | chunks |
| b11147-cuda | llamacpp | blazar | config=single-stream child: np=1 ctx=16384 | 118 | 124 | 24 | 25 | 41.1 | 1317.7 | 7309.2 | chunks |
| v0.9.3 | mistralrs | blazar | config=default child: np=2 ctx=8192 | 125 | 129 | 24 | 26 | 40.9 | 1338.7 | 7538.8 | chunks |
| v0.9.3 | mistralrs | blazar | config=paged_attn_off child: np=2 ctx=8192 | 122 | 125 | 24 | 25 | 40.9 | 1326.5 | 7556.7 | chunks |
| v0.9.3 | mistralrs | blazar | config=single-stream child: np=1 ctx=16384 | 124 | 127 | 24 | 26 | 40.9 | 1333.3 | 7244.6 | chunks |
| ollama-host | ollama | ollama | reference=True ⚠ serves 'qwen3.5:9b' — t/s NOT comparable | 127 | 133 | 25 | 75 | 40.6 | 1364.7 | 8357.6 | engine_counters |
| v0.9.3 | mistralrs | direct | ctx=4096 np=1 pa=off | 124 | 137 | 50 | 64 | 19.6 | 221.7 | 244.1 | chunks |
| ollama-host | ollama | ctxcurve-ollama | ctx=2048 | 168 | - | - | - | 40.2 | - | - | - |
| ollama-host | ollama | ctxcurve-ollama | ctx=8192 | 149 | - | - | - | 40.3 | - | - | - |
| ollama-host | ollama | ctxcurve-ollama | ctx=16384 | 135 | - | - | - | 40.7 | - | - | - |
| b11147-cuda | llamacpp | ctxcurve-blazar | ctx=2048 child: np=3 ctx=6144 | 132 | 203 | 24 | 32 | 40.6 | 1287.9 | 6536.9 | chunks |
| b11147-cuda | llamacpp | ctxcurve-blazar | ctx=8192 child: np=1 ctx=8192 | 125 | 331 | 24 | 26 | 41.0 | 1321.8 | 7010.5 | chunks |
| b11147-cuda | llamacpp | ctxcurve-blazar | ctx=16384 child: np=1 ctx=16384 | 126 | 328 | 24 | 26 | 40.9 | 1301.0 | 7545.5 | chunks |
| v0.9.3 | mistralrs | ctxcurve-blazar | ctx=2048 child: np=4 ctx=8192 | 125 | 180 | 24 | 26 | 40.9 | 1329.7 | 7486.6 | chunks |
| v0.9.3 | mistralrs | ctxcurve-blazar | ctx=8192 child: np=1 ctx=8192 | 128 | 328 | 24 | 26 | 41.0 | 1311.9 | 7148.8 | chunks |
| v0.9.3 | mistralrs | ctxcurve-blazar | ctx=16384 child: np=1 ctx=16384 | 124 | 325 | 24 | 26 | 40.9 | 1314.5 | 7500.3 | chunks |

## Resources & cold start

| engine | provider | params | load s | daemon boot s | cold 1st req s | GPU peak (MiB) | GPU power (W) | RSS peak (MiB) | teardown |
|---||---||---||---||---||---||---||---||---||
| b11147-cuda | direct | ctx=4096 np=1 | 4.04 | - | - | 5319 | 55.9 | 5718 | ok |
| b11147-cuda | direct | ctx=4096 np=4 | 2.01 | - | - | 5459 | 55.2 | 5770 | ok |
| b11147-cuda | direct | ctx=16384 np=1 | 2.01 | - | - | 5709 | 55.9 | 5770 | ok |
| b11147-cuda | direct | ctx=16384 np=4 | 2.01 | - | - | 5847 | 55.5 | 5770 | ok |
| b11147-cuda | direct | ctx=4096 kv=q8_0 np=1 | 2.02 | - | - | 5261 | 56.0 | 5770 | ok |
| b11147-cuda | direct | ctx=4096 np=1 spec=ngram-simple | 2.54 | - | - | 5319 | 57.3 | 5771 | ok |
| b11147-cuda | direct | ctx=4096 mmproj=True np=1 | 3.01 | - | - | 6439 | 55.2 | 5776 | ok |
| b11147-cuda | blazar | config=default | - | 0.53 | 5.0 | 5715 | 56.3 | 1992 | ok |
| b11147-cuda | blazar | config=single-stream | - | 0.53 | 5.29 | 5715 | 56.3 | 1993 | ok |
| v0.9.3 | blazar | config=default | - | 0.52 | 5.0 | 5501 | 55.6 | 1990 | ok |
| v0.9.3 | blazar | config=paged_attn_off | - | 0.52 | 5.0 | 5501 | 55.4 | 1990 | ok |
| v0.9.3 | blazar | config=single-stream | - | 0.52 | 4.97 | 5715 | 55.5 | 1993 | ok |
| ollama-host | ollama | reference=True | - | - | - | 6626 | 55.5 | - | ok |
| v0.9.3 | direct | ctx=4096 np=1 pa=off | 7.52 | - | - | 6914 | 41.9 | 7443 | ok |
| ollama-host | ctxcurve-ollama | ctx=2048 | 7.91 | - | - | 6338 | - | - | ok |
| ollama-host | ctxcurve-ollama | ctx=8192 | 6.44 | - | - | 6446 | - | - | ok |
| ollama-host | ctxcurve-ollama | ctx=16384 | 5.71 | - | - | 6622 | - | - | ok |
| b11147-cuda | ctxcurve-blazar | ctx=2048 | - | - | - | 5486 | - | - | ok |
| b11147-cuda | ctxcurve-blazar | ctx=8192 | - | - | - | 5452 | - | - | ok |
| b11147-cuda | ctxcurve-blazar | ctx=16384 | - | - | - | 5710 | - | - | ok |
| v0.9.3 | ctxcurve-blazar | ctx=2048 | - | - | - | 5596 | - | - | ok |
| v0.9.3 | ctxcurve-blazar | ctx=8192 | - | - | - | 5452 | - | - | ok |
| v0.9.3 | ctxcurve-blazar | ctx=16384 | - | - | - | 5841 | - | - | ok |

## Concurrency (parallel streams)

| engine | provider | streams | sys t/s | sum stream t/s | ttft max (ms) | ttft spread (ms) | itl p99 (ms) | ok/errors | wall (s) |
|---||---||---||---||---||---||---||---||---||
| b11147-cuda | conc-direct | conc=1 | 38.8 | 40.8 | 179 | 0.0 | 28.25 | 1/0 | 3.3 |
| b11147-cuda | conc-blazar | conc=1 rounds=3 | 39.2 | 123.2 | 309 | 201.9 | 26.74 | 3/0 | 9.8 |
| v0.9.3 | conc-blazar | conc=1 rounds=3 | 39.4 | 123.3 | 209 | 101.0 | 26.76 | 3/0 | 9.75 |
| ollama-host | conc-ollama | conc=1 rounds=3 | 22.6 | 120.9 | 7206 | 7094.7 | 75.47 | 3/0 | 16.98 |
| b11147-cuda | conc-direct | conc=2 | 71.0 | 73.9 | 164 | 0.6 | 28.56 | 2/0 | 3.6 |
| b11147-cuda | conc-blazar | conc=2 rounds=3 | 39.6 | 246.5 | 3497 | 3384.8 | 25.45 | 6/0 | 19.41 |
| v0.9.3 | conc-blazar | conc=2 rounds=3 | 68.1 | 219.1 | 381 | 182.9 | 29.24 | 6/0 | 11.28 |
| ollama-host | conc-ollama | conc=2 rounds=3 | 30.8 | 243.9 | 8723 | 8617.8 | 74.59 | 6/0 | 24.92 |
| b11147-cuda | conc-direct | conc=4 | 110.7 | 115.9 | 241 | 14.1 | 37.39 | 4/0 | 4.63 |
| b11147-cuda | conc-blazar | conc=4 rounds=3 | 39.8 | 494.0 | 9901 | 9789.9 | 25.5 | 12/0 | 38.61 |
| v0.9.3 | conc-blazar | conc=4 rounds=3 | 69.2 | 439.8 | 4145 | 3946.1 | 29.05 | 12/0 | 22.21 |
| ollama-host | conc-ollama | conc=4 rounds=3 | 34.0 | 485.8 | 15846 | 15732.0 | 74.98 | 12/0 | 45.23 |
| b11147-cuda | conc-direct | conc=8 | 133.5 | 141.2 | 477 | 11.3 | 60.49 | 8/0 | 7.67 |
| b11147-cuda | conc-blazar | conc=8 rounds=3 | 39.5 | 982.3 | 23007 | 22887.6 | 26.01 | 24/0 | 77.86 |
| v0.9.3 | conc-blazar | conc=8 rounds=3 | 68.7 | 874.4 | 11871 | 11659.1 | 29.36 | 24/0 | 44.71 |
| ollama-host | conc-ollama | conc=8 rounds=3 | 36.3 | 972.4 | 29036 | 28922.8 | 74.81 | 24/0 | 84.64 |
- sys t/s = total tokens / wall (true system throughput); sum stream t/s = sum of per-stream rates. sum >> sys means streams were serialized (queued on a single slot) rather than served concurrently.


## Quality — perplexity (identical pinned args)

| engine | perplexity | ± err | wall (s) | note |
|---|---|---|---|---|
| v0.9.3 | - | - | - | llama-perplexity is llama-server-family only |
| b11147-cuda | 16.7503 | 0.87774 | 8.6 | lower = better text fit |
- corpus: deterministic offline repo text (code-heavy) — PARITY-ONLY; absolute PPL is not comparable to published wiki-text perplexities.

## Quality — greedy parity vs `b11147-cuda`-direct (backend numerics)

| engine | exact matches | ratio mean | ratio min | first divergence (median chars) |
|---|---|---|---|---|
| b11147-cuda *(self — trivially 1.0) | 20/20 | 1.0 | 1.0 | 798.5 |
| v0.9.3 | 0/20 | 0.5261 | 0.0 | 0.0 |

## Quality — gateway transparency (blazar path vs direct, same engine)

| engine | exact matches | ratio mean | ratio min | first divergence (median chars) |
|---|---|---|---|---|
| b11147-cuda | 20/20 | 1.0 | 1.0 | 798.5 |
- expectation: 20/20 exact, ratio 1.0. A miss has TWO possible causes: gateway translation defect (sampler remap / template drift), or multi-slot batching numerics (child -np > 1 changes float reduction order; near-tie logits flip). Pin `slots = 1` and re-run: still <20/20 = translation defect, 20/20 = slot-count numerics (upstream physics).
## Feature matrix

| feature | b11147-cuda | v0.9.3 | ollama(documented) |
|---|---|---|---|
| `anthropic-api` | - | Y | - |
| `ctx-override` | Y | Y | Y |
| `embeddings` | Y | - | Y |
| `grammar-gbnf` | Y | - | - |
| `json-schema` | Y | Y | Y |
| `kv-quant` | Y | Y | - |
| `lora-adapter` | Y | Y | Y |
| `metrics-endpoint` | Y | Y | - |
| `paged-attn` | Y | Y | - |
| `parallel-np` | Y | Y | Y |
| `quant-on-load` | Y | Y | - |
| `rerank` | Y | - | - |
| `slots-sessions` | Y | - | - |
| `spec-decode` | Y | Y | - |
| `tokenize-endpoint` | Y | - | Y |
| `vision-mmproj` | Y | Y | Y |

## Failed cells

**product** (engine/gateway behavior):

- `v0.9.3` / direct / {'ctx': 4096, 'np': 1}: child exited rc=1 during load; last output: Error: Num GPU blocks is 0. This means there is not enough memory. Either reduce the memory amount/utilization/context size or disable PagedAttention. | 
- `v0.9.3` / direct / {'ctx': 4096, 'np': 4}: child exited rc=1 during load; last output: Error: Num GPU blocks is 0. This means there is not enough memory. Either reduce the memory amount/utilization/context size or disable PagedAttention. | 
- `v0.9.3` / direct / {'ctx': 16384, 'np': 1}: child exited rc=1 during load; last output: Error: Num GPU blocks is 0. This means there is not enough memory. Either reduce the memory amount/utilization/context size or disable PagedAttention. | 
- `v0.9.3` / direct / {'ctx': 16384, 'np': 4}: child exited rc=1 during load; last output: Error: Num GPU blocks is 0. This means there is not enough memory. Either reduce the memory amount/utilization/context size or disable PagedAttention. | 
- `v0.9.3` / ppl / {'ppl': 2048}: llama-perplexity is llama-server-family only

**environment** (box/co-residency guards — NOT blazar defects):

- `v0.9.3` / reshape / {'reshape': True}: reshape cell crashed: MemAvailable 6986 MiB < needed ~7795 MiB (model 5417 MiB + headroom). A co-resident blazar/ollama engine is likely holding memory: stop it for the validation window.

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
