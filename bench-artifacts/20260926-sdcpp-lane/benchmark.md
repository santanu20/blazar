# Blazar benchmark matrix

- **date**: 2026-09-26 19:46:51
- **model**: `Qwen3.5-9B-Q4_K_M.gguf` (5366 MiB)
- **gpu**: NVIDIA GeForce RTX 4070 Laptop GPU / driver 595.91.07
- **engines**: b11193-cuda, b5130, inventory, master-919-19bbbca, ollama-host, piper, v0.9.4
- **harness**: bench_matrix v2 — `--artifacts-dir bench-artifacts/20260926-sdcpp-lane --md bench-artifacts/20260926-sdcpp-lane/REPORT.md --allow-battery`
- **blazar**: `blazar 0.11.0` (sandbox daemon binary)
- all blazar-owned rows measured by `blazar 0.11.0`

## Notable findings

- gateway vs direct decode (`b11193-cuda`): 41.4 vs 41.5 t/s = -0.2% (parity).
- gateway vs direct decode (`b11193-cuda`): 41.5 vs 41.5 t/s = -0.1% (parity).
- concurrency system throughput (`b11193-cuda`, 4 streams): 70.3 vs direct 112.8 t/s = 0.62x.
- kv=q8_0 (`b11193-cuda`): decode -1.2% vs baseline
- spec=ngram-simple (`b11193-cuda`): decode -1.2% vs baseline
- mmproj=True (`b11193-cuda`): decode +0.4% vs baseline
- gateway greedy transparency (`b11193-cuda`): 19/20 exact vs same-engine direct — NOT TRANSPARENT.

## Speed (serving, streaming)

| engine | kind | provider | params | ttft p50 (ms) | ttft p99 (ms) | itl p50 (ms) | itl p99 (ms) | decode t/s | prefill t/s (cold) | prefill t/s (cached) | tokens src |
|---||---||---||---||---||---||---||---||---||---||---||
| b11193-cuda | llamacpp | direct | ctx=4096 np=1 | 127 | 129 | 24 | 25 | 41.5 | 1306.9 | 7275.9 | chunks |
| b11193-cuda | llamacpp | direct | ctx=4096 np=4 | 124 | 127 | 24 | 25 | 41.4 | 1305.4 | 7442.9 | chunks |
| b11193-cuda | llamacpp | direct | ctx=16384 np=1 | 122 | 126 | 24 | 25 | 41.4 | 1366.8 | 7344.0 | chunks |
| b11193-cuda | llamacpp | direct | ctx=16384 np=4 | 128 | 131 | 24 | 25 | 41.2 | 1308.6 | 7331.7 | chunks |
| b11193-cuda | llamacpp | direct | ctx=4096 kv=q8_0 np=1 | 120 | 125 | 24 | 26 | 41.0 | 1363.0 | 7546.7 | chunks |
| b11193-cuda | llamacpp | direct | ctx=4096 np=1 spec=ngram-simple | 123 | 125 | 24 | 26 | 41.0 | 1327.2 | 7591.7 | chunks |
| b11193-cuda | llamacpp | direct | ctx=4096 mmproj=True np=1 | 123 | 125 | 24 | 25 | 41.7 | 1345.7 | 7517.1 | chunks |
| b11193-cuda | llamacpp | blazar | config=default child: np=2 ctx=8192 | 122 | 126 | 24 | 25 | 41.4 | 1319.0 | 7328.5 | chunks |
| b11193-cuda | llamacpp | blazar | config=single-stream child: np=1 ctx=16384 | 123 | 125 | 24 | 25 | 41.5 | 1302.0 | 7304.5 | chunks |
| v0.9.4 | mistralrs | blazar | config=default child: np=2 ctx=8192 | 125 | 126 | 24 | 25 | 41.4 | 1327.5 | 7166.5 | chunks |
| v0.9.4 | mistralrs | blazar | config=paged_attn_off child: np=2 ctx=8192 | 124 | 127 | 24 | 25 | 41.5 | 1322.9 | 7136.9 | chunks |
| v0.9.4 | mistralrs | blazar | config=single-stream child: np=1 ctx=16384 | 123 | 126 | 24 | 25 | 41.5 | 1329.3 | 7240.1 | chunks |
| ollama-host | ollama | ollama | reference=True ⚠ serves 'qwen3.5:9b' — t/s NOT comparable | 125 | 127 | 25 | 75 | 40.5 | 1442.4 | 7701.3 | engine_counters |
| b11193-cuda | llamacpp | ctxcurve-blazar | ctx=2048 child: np=4 ctx=8192 | 110 | 163 | 24 | 25 | 41.7 | 1394.4 | 8074.2 | chunks |
| b11193-cuda | llamacpp | ctxcurve-blazar | ctx=8192 child: np=1 ctx=8192 | 110 | 295 | 24 | 25 | 41.7 | 1376.4 | 8172.7 | chunks |
| b11193-cuda | llamacpp | ctxcurve-blazar | ctx=16384 child: np=1 ctx=16384 | 108 | 300 | 24 | 25 | 41.8 | 1363.5 | 8146.3 | chunks |
| v0.9.4 | mistralrs | ctxcurve-blazar | ctx=2048 child: np=4 ctx=8192 | 110 | 160 | 24 | 25 | 41.7 | 1363.0 | 8145.9 | chunks |
| v0.9.4 | mistralrs | ctxcurve-blazar | ctx=8192 child: np=1 ctx=8192 | 108 | 301 | 24 | 25 | 41.7 | 1358.7 | 8107.7 | chunks |
| v0.9.4 | mistralrs | ctxcurve-blazar | ctx=16384 child: np=1 ctx=16384 | 113 | 294 | 24 | 25 | 41.8 | 1374.0 | 8133.3 | chunks |
| ollama-host | ollama | ctxcurve-ollama | ctx=2048 | 111 | - | - | - | 40.9 | - | - | - |
| ollama-host | ollama | ctxcurve-ollama | ctx=8192 | 118 | - | - | - | 40.8 | - | - | - |
| ollama-host | ollama | ctxcurve-ollama | ctx=16384 | 110 | - | - | - | 40.9 | - | - | - |

## Resources & cold start

| engine | provider | params | load s | daemon boot s | cold 1st req s | GPU peak (MiB) | GPU power (W) | RSS peak (MiB) | teardown |
|---||---||---||---||---||---||---||---||---||
| b11193-cuda | direct | ctx=4096 np=1 | 6.05 | - | - | 5270 | 55.1 | 5413 | ok |
| b11193-cuda | direct | ctx=4096 np=4 | 2.55 | - | - | 5416 | 56.3 | 5719 | ok |
| b11193-cuda | direct | ctx=16384 np=1 | 2.51 | - | - | 5666 | 56.7 | 5714 | ok |
| b11193-cuda | direct | ctx=16384 np=4 | 2.51 | - | - | 5804 | 56.4 | 5716 | ok |
| b11193-cuda | direct | ctx=4096 kv=q8_0 np=1 | 2.51 | - | - | 5210 | 56.7 | 5717 | ok |
| b11193-cuda | direct | ctx=4096 np=1 spec=ngram-simple | 2.11 | - | - | 5270 | 55.4 | 5709 | ok |
| b11193-cuda | direct | ctx=4096 mmproj=True np=1 | 3.01 | - | - | 6400 | 56.4 | 5721 | ok |
| b11193-cuda | blazar | config=default | - | 0.54 | 4.94 | 5452 | 55.4 | 1983 | ok |
| b11193-cuda | blazar | config=single-stream | - | 0.52 | 4.9 | 5666 | 55.8 | 1987 | ok |
| v0.9.4 | blazar | config=default | - | 0.52 | 4.57 | 5458 | 55.8 | 1984 | ok |
| v0.9.4 | blazar | config=paged_attn_off | - | 0.52 | 4.56 | 5458 | 56.3 | 1984 | ok |
| v0.9.4 | blazar | config=single-stream | - | 0.52 | 4.54 | 5672 | 55.3 | 1987 | ok |
| ollama-host | ollama | reference=True | - | - | - | 6566 | 55.2 | - | ok |
| b11193-cuda | ctxcurve-blazar | ctx=2048 | - | - | - | 5546 | - | - | ok |
| b11193-cuda | ctxcurve-blazar | ctx=8192 | - | - | - | 5396 | - | - | ok |
| b11193-cuda | ctxcurve-blazar | ctx=16384 | - | - | - | 5660 | - | - | ok |
| v0.9.4 | ctxcurve-blazar | ctx=2048 | - | - | - | 5546 | - | - | ok |
| v0.9.4 | ctxcurve-blazar | ctx=8192 | - | - | - | 5402 | - | - | ok |
| v0.9.4 | ctxcurve-blazar | ctx=16384 | - | - | - | 5666 | - | - | ok |
| ollama-host | ctxcurve-ollama | ctx=2048 | 4.76 | - | - | 6368 | - | - | ok |
| ollama-host | ctxcurve-ollama | ctx=8192 | 4.5 | - | - | 6566 | - | - | ok |
| ollama-host | ctxcurve-ollama | ctx=16384 | 4.49 | - | - | 6830 | - | - | ok |

## Concurrency (parallel streams)

| engine | provider | streams | sys t/s | sum stream t/s | ttft max (ms) | ttft spread (ms) | itl p99 (ms) | ok/errors | wall (s) |
|---||---||---||---||---||---||---||---||---||
| b11193-cuda | conc-direct | conc=4 | 112.8 | 118.1 | 244 | 10.6 | 35.96 | 4/0 | 4.54 |
| b11193-cuda | conc-blazar | conc=4 rounds=3 | 70.3 | 447.5 | 4075 | 3877.7 | 28.49 | 12/0 | 21.83 |
| v0.9.4 | conc-blazar | conc=4 rounds=3 | 70.4 | 447.8 | 4075 | 3878.0 | 28.4 | 12/0 | 21.82 |
| ollama-host | conc-ollama | conc=4 rounds=3 | 34.3 | 489.7 | 15676 | 15555.8 | 74.35 | 12/0 | 44.82 |
- sys t/s = total tokens / wall (true system throughput); sum stream t/s = sum of per-stream rates. sum >> sys means streams were serialized (queued on a single slot) rather than served concurrently.


## Quality — perplexity (identical pinned args)

| engine | perplexity | ± err | wall (s) | note |
|---|---|---|---|---|
| b11193-cuda | 17.3512 | 0.91919 | 8.0 | lower = better text fit |
| v0.9.4 | - | - | - | llama-perplexity is llama-server-family only |
- corpus: deterministic offline repo text (code-heavy) — PARITY-ONLY; absolute PPL is not comparable to published wiki-text perplexities.

## Quality — greedy parity vs `b11193-cuda`-direct (backend numerics)

| engine | exact matches | ratio mean | ratio min | first divergence (median chars) |
|---|---|---|---|---|
| b11193-cuda *(self — trivially 1.0) | 20/20 | 1.0 | 1.0 | 928.5 |
| v0.9.4 | -/- | - | - | - |

## Quality — gateway transparency (blazar path vs direct, same engine)

| engine | exact matches | ratio mean | ratio min | first divergence (median chars) |
|---|---|---|---|---|
| b11193-cuda | 19/20 | 0.9508 | 0.0169 | 835.5 |
- expectation: 20/20 exact, ratio 1.0. A miss has TWO possible causes: gateway translation defect (sampler remap / template drift), or multi-slot batching numerics (child -np > 1 changes float reduction order; near-tie logits flip). Pin `slots = 1` and re-run: still <20/20 = translation defect, 20/20 = slot-count numerics (upstream physics).
## Feature matrix

| feature | b11193-cuda | v0.9.4 | ollama(documented) |
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

- `v0.9.4` / direct / {'ctx': 4096, 'np': 1}: child exited rc=1 during load; last output: Error: multimodal GGUF requires its original `config.json`, but the GGUF files do not identify one unambiguous Hugging Face base model; pass `--tok-model-id <original-model-id>` | 
- `v0.9.4` / direct / {'ctx': 4096, 'np': 4}: child exited rc=1 during load; last output: Error: multimodal GGUF requires its original `config.json`, but the GGUF files do not identify one unambiguous Hugging Face base model; pass `--tok-model-id <original-model-id>` | 
- `v0.9.4` / direct / {'ctx': 16384, 'np': 1}: child exited rc=1 during load; last output: Error: multimodal GGUF requires its original `config.json`, but the GGUF files do not identify one unambiguous Hugging Face base model; pass `--tok-model-id <original-model-id>` | 
- `v0.9.4` / direct / {'ctx': 16384, 'np': 4}: child exited rc=1 during load; last output: Error: multimodal GGUF requires its original `config.json`, but the GGUF files do not identify one unambiguous Hugging Face base model; pass `--tok-model-id <original-model-id>` | 
- `v0.9.4` / direct / {'ctx': 4096, 'np': 1, 'pa': 'off'}: child exited rc=1 during load; last output: Error: multimodal GGUF requires its original `config.json`, but the GGUF files do not identify one unambiguous Hugging Face base model; pass `--tok-model-id <original-model-id>` | 
- `v0.9.4` / ppl / {'ppl': 2048}: llama-perplexity is llama-server-family only
- `v0.9.4` / greedy / {'greedy': True}: child failed to become healthy; stderr tail: 'Error: multimodal GGUF requires its original `config.json`, but the GGUF files do not identify one unambiguous Hugging Face base model; pass `--tok-model-id <original-model-id>`\n'

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
