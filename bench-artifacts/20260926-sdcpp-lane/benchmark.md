# Blazar benchmark matrix

- **date**: 2026-10-06 16:00:26
- **model**: `Qwen3.5-9B-Q4_K_M.gguf` (5366 MiB)
- **gpu**: NVIDIA GeForce RTX 4070 Laptop GPU / driver 595.91.07
- **engines**: b11193-cuda, b5130, inventory, master-919-19bbbca, ollama-host, piper, v0.9.4
- **harness**: bench_matrix v4 — `regenerated (slim format)`
- **blazar**: `blazar 0.11.0` (sandbox daemon binary)
- **quality lane**: not-measured — no quality rows in this campaign (lane not reached for the selected providers/engines)
- all blazar-owned rows measured by `blazar 0.11.0`

## Notable findings

- gateway vs direct decode (`b11193-cuda`, 2 configs): median -0.1%, 2/2 within ±5% (parity).
- concurrency system throughput (`b11193-cuda`, 4 streams): 70.3 vs direct 112.8 t/s = 0.62x.
- kv=q8_0 (`b11193-cuda`): decode -1.2% vs baseline
- spec=ngram-simple (`b11193-cuda`): decode -1.2% vs baseline
- mmproj=True (`b11193-cuda`): decode +0.4% vs baseline
- gateway greedy transparency (`b11193-cuda`): 19/20 exact vs same-engine direct — NOT TRANSPARENT.

## Charts

- deterministic SVG renders of the cells below; source of truth is `cells.jsonl`, charts live in `plots/`.

### Single-stream decode throughput (chart)

<p align="center"><img src="plots/speed-single-stream.svg" alt="Single-stream decode throughput (chart)"></p>

_Median decode t/s per runtime and engine; whiskers span the interquartile range of the 5 runs; the dashed line marks the fastest direct engine. Higher is better. Source: 20260926-sdcpp-lane/cells.jsonl._

### Gateway overhead (chart)

<p align="center"><img src="plots/gateway-overhead.svg" alt="Gateway overhead (chart)"></p>

_Decode t/s delta of routing through blazar relative to driving the same engine build directly; left of zero means the gateway path won. Source: 20260926-sdcpp-lane/cells.jsonl._

### Concurrency scaling (chart)

<p align="center"><img src="plots/concurrency-throughput.svg" alt="Concurrency scaling (chart)"></p>

_Aggregate system tokens/s as parallel streams are added; flat-to-rising means the scheduler keeps the device saturated. Higher is better. Source: 20260926-sdcpp-lane/cells.jsonl._

### Concurrency tail latency (chart)

<p align="center"><img src="plots/concurrency-ttft.svg" alt="Concurrency tail latency (chart)"></p>

_Worst-case first-token wait per stream as concurrency rises (log scale) - the tail the scheduler must bound. Lower is better. Source: 20260926-sdcpp-lane/cells.jsonl._

### Resource cost vs concurrency (chart)

<p align="center"><img src="plots/concurrency-vram.svg" alt="Resource cost vs concurrency (chart)"></p>

_Peak VRAM footprint as parallel streams (and their KV caches) stack up. Source: 20260926-sdcpp-lane/cells.jsonl._

### Long-context degradation (chart)

<p align="center"><img src="plots/ctx-curve.svg" alt="Long-context degradation (chart)"></p>

_Single-stream decode t/s as prompt context grows (log x-axis) - KV-cache pressure made visible. Source: 20260926-sdcpp-lane/cells.jsonl._

### Memory vs context (chart)

<p align="center"><img src="plots/vram-vs-context.svg" alt="Memory vs context (chart)"></p>

_Peak VRAM as prompt context grows (log x-axis) - the KV-cache slope that sets the usable context ceiling. Source: 20260926-sdcpp-lane/cells.jsonl._

### Lifecycle: cold start and idle wake (chart)

<p align="center"><img src="plots/lifecycle-cold-idle.svg" alt="Lifecycle: cold start and idle wake (chart)"></p>

_Seconds to first token after a cold start (page cache dropped) and after idle-policy expiry; blazar keeps weights resident while ollama reloads from disk. Warm-daemon ollama caveat applies. Lower is better. Source: 20260926-sdcpp-lane/cells.jsonl._

### KV-quant perplexity (chart)

<p align="center"><img src="plots/ppl-kv.svg" alt="KV-quant perplexity (chart)"></p>

_Perplexity per KV-cache quantization configuration (whiskers: standard error); the dashed line marks the unquantized f16 run. Lower is better; compare only within one context rung. Source: 20260926-sdcpp-lane/cells.jsonl._

## Speed (single-stream, medians)

| engine | provider | params | n | ttft p50 (ms) | ttft p99 (ms) | itl p99 (ms) | decode t/s | prefill t/s (cached) | VRAM peak (MiB) |
|---||---||---||---||---||---||---||---||---||
| b11193-cuda | blazar | config=default child: np=2 ctx=8192 | 1 | 122 | 126 | 25 | 41.4 | 7328.5 | 5452 |
| b11193-cuda | blazar | config=single-stream child: np=1 ctx=16384 | 1 | 123 | 125 | 25 | 41.5 | 7304.5 | 5666 |
| b11193-cuda | ctxcurve-blazar | ctx=16384 child: np=1 ctx=16384 | 1 | 108 | 300 | 25 | 41.8 | 8146.3 | 5660 |
| b11193-cuda | ctxcurve-blazar | ctx=2048 child: np=4 ctx=8192 | 1 | 110 | 163 | 25 | 41.7 | 8074.2 | 5546 |
| b11193-cuda | ctxcurve-blazar | ctx=8192 child: np=1 ctx=8192 | 1 | 110 | 295 | 25 | 41.7 | 8172.7 | 5396 |
| b11193-cuda | direct | ctx=16384 np=1 | 1 | 122 | 126 | 25 | 41.4 | 7344.0 | 5666 |
| b11193-cuda | direct | ctx=16384 np=4 | 1 | 128 | 131 | 25 | 41.2 | 7331.7 | 5804 |
| b11193-cuda | direct | ctx=4096 kv=q8_0 np=1 | 1 | 120 | 125 | 26 | 41.0 | 7546.7 | 5210 |
| b11193-cuda | direct | ctx=4096 mmproj=True np=1 | 1 | 123 | 125 | 25 | 41.7 | 7517.1 | 6400 |
| b11193-cuda | direct | ctx=4096 np=1 | 1 | 127 | 129 | 25 | 41.5 | 7275.9 | 5270 |
| b11193-cuda | direct | ctx=4096 np=1 spec=ngram-simple | 1 | 123 | 125 | 26 | 41.0 | 7591.7 | 5270 |
| b11193-cuda | direct | ctx=4096 np=4 | 1 | 124 | 127 | 25 | 41.4 | 7442.9 | 5416 |
| ollama-host | ctxcurve-ollama | ctx=16384 | 1 | 110 | - | - | 40.9 | - | 6830 |
| ollama-host | ctxcurve-ollama | ctx=2048 | 1 | 111 | - | - | 40.9 | - | 6368 |
| ollama-host | ctxcurve-ollama | ctx=8192 | 1 | 118 | - | - | 40.8 | - | 6566 |
| ollama-host | ollama | reference=True NOTE: serves 'qwen3.5:9b' - t/s NOT comparable | 1 | 125 | 127 | 75 | 40.5 | 7701.3 | 6566 |
| v0.9.4 | blazar | config=default child: np=2 ctx=8192 | 1 | 125 | 126 | 25 | 41.4 | 7166.5 | 5458 |
| v0.9.4 | blazar | config=paged_attn_off child: np=2 ctx=8192 | 1 | 124 | 127 | 25 | 41.5 | 7136.9 | 5458 |
| v0.9.4 | blazar | config=single-stream child: np=1 ctx=16384 | 1 | 123 | 126 | 25 | 41.5 | 7240.1 | 5672 |
| v0.9.4 | ctxcurve-blazar | ctx=16384 child: np=1 ctx=16384 | 1 | 113 | 294 | 25 | 41.8 | 8133.3 | 5666 |
| v0.9.4 | ctxcurve-blazar | ctx=2048 child: np=4 ctx=8192 | 1 | 110 | 160 | 25 | 41.7 | 8145.9 | 5546 |
| v0.9.4 | ctxcurve-blazar | ctx=8192 child: np=1 ctx=8192 | 1 | 108 | 301 | 25 | 41.7 | 8107.7 | 5402 |

## Concurrency (parallel streams, medians)

| provider | engine | params | n | sys t/s | ttft max (ms) | itl p99 (ms) | ok/errors |
|---||---||---||---||---||---||---||---||
| conc-blazar | b11193-cuda | conc=4 rounds=3 | 1 | 70.3 | 4075 | 28 | 12/0 |
| conc-blazar | v0.9.4 | conc=4 rounds=3 | 1 | 70.4 | 4075 | 28 | 12/0 |
| conc-direct | b11193-cuda | conc=4 | 1 | 112.8 | 244 | 36 | 4/0 |
| conc-ollama | ollama-host | conc=4 rounds=3 | 1 | 34.3 | 15676 | 74 | 12/0 |
- sys t/s = total tokens / wall (true system throughput); flat sys t/s with growing ttft max means streams queued on a fixed slot count instead of being served concurrently.


## Quality — perplexity (identical pinned args)

| engine | perplexity | +/- err | wall (s) | note |
|---|---|---|---|---|
| b11193-cuda | 17.3512 | 0.91919 | 8.0 | lower = better text fit |
| v0.9.4 | - | - | - | llama-perplexity is llama-server-family only |
| v0.9.4 | - | - | - | llama-perplexity is llama-server-family only |
- corpus: deterministic offline repo text (code-heavy) — PARITY-ONLY; absolute PPL is not comparable to published wiki-text perplexities.

## Quality — greedy parity vs `reference`-direct (backend numerics)

| engine | exact matches | ratio mean | ratio min | first divergence (median chars) |
|---|---|---|---|---|
| b11193-cuda | 20/20 | 1.0 | 1.0 | 928.5 |
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

- `v0.9.4` / direct (10 cells): child exited rc=1 during load; last output: Error: multimodal GGUF requires its original `config.json`, but the GGUF files do not identify one unambiguous Hu... [{'ctx': 4096, 'np': 1}; {'ctx': 4096, 'np': 4}; {'ctx': 16384, 'np': 1}; {'ctx': 16384, 'np': 4}; +6 more]
- `v0.9.4` / ppl (2 cells): llama-perplexity is llama-server-family only [{'ppl': 2048}; {'ppl': 2048}]
- `v0.9.4` / greedy / {'greedy': True}: child failed to become healthy; stderr tail: 'Error: multimodal GGUF requires its original `config.json`, but the GGUF files do not identify one unambiguous ...

## Reading this report

- `direct` = raw child spawn on a probed free port (no gateway); `blazar` = full gateway path inside a sandboxed daemon (resolved engine argv in `child_argv`); `ollama` = HTTP-only reference against the host service.
- decode counts ALL emitted tokens (content + reasoning); engine `usage` counters are authoritative when present. GPU/power peaks sampled at ~1.2 s cadence.
- per-run spread, cold-start/resource detail, and every raw cell live in `cells.jsonl`; tables here are per-config medians.
