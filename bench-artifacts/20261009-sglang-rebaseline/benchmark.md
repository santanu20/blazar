# Blazar benchmark matrix

- **date**: 2026-10-09 21:09:01
- **model**: `qwen3-1.7b.d` (3890 MiB)
- **gpu**: NVIDIA GeForce RTX 4070 Laptop GPU / driver 595.99.02
- **engines**: inventory, ollama-host, sglang-0.5.21
- **harness**: bench_matrix v4 — `--artifacts-dir bench-artifacts/20261009-sglang-rebaseline --engines sglang-0.5.21 --providers blazar --model qwen3-1.7b --conc-sweep 1,2,4,8 --conc-rounds 3 --skip-ppl --skip-greedy --skip-tools --skip-reshape --skip-features --skip-idle --skip-ctxcurve --skip-quality --skip-variants --skip-media --fresh`
- **blazar**: `blazar 0.24.0` (sandbox daemon binary)
- **quality lane**: skipped — --skip-quality: no checker-verified receipts back any speed number in this campaign
- all blazar-owned rows measured by `blazar 0.24.0`

## Charts

- deterministic SVG renders of the cells below; source of truth is `cells.jsonl`, charts live in `plots/`.

### Single-stream decode throughput (chart)

<p align="center"><img src="plots/speed-single-stream.svg" alt="Single-stream decode throughput (chart)"></p>

_Median decode t/s per runtime and engine; whiskers span the interquartile range of the 5 runs; the dashed line marks the fastest direct engine. Higher is better. Source: 20261009-sglang-rebaseline/cells.jsonl._

### Concurrency scaling (chart)

<p align="center"><img src="plots/concurrency-throughput.svg" alt="Concurrency scaling (chart)"></p>

_Aggregate system tokens/s as parallel streams are added; flat-to-rising means the scheduler keeps the device saturated. Higher is better. Source: 20261009-sglang-rebaseline/cells.jsonl._

### Concurrency tail latency (chart)

<p align="center"><img src="plots/concurrency-ttft.svg" alt="Concurrency tail latency (chart)"></p>

_Worst-case first-token wait per stream as concurrency rises (log scale) - the tail the scheduler must bound. Lower is better. Dashed curves carry the median time-to-first-byte, which upper-bounds queue wait under burst arrival. Source: 20261009-sglang-rebaseline/cells.jsonl._

### Resource cost vs concurrency (chart)

<p align="center"><img src="plots/concurrency-vram.svg" alt="Resource cost vs concurrency (chart)"></p>

_Peak VRAM footprint as parallel streams (and their KV caches) stack up. Source: 20261009-sglang-rebaseline/cells.jsonl._

### Resource cost vs concurrency (chart)

<p align="center"><img src="plots/concurrency-power.svg" alt="Resource cost vs concurrency (chart)"></p>

_Peak GPU board power per concurrency level - the energy price of keeping the device saturated. Source: 20261009-sglang-rebaseline/cells.jsonl._

### Lifecycle: cold start and idle wake (chart)

<p align="center"><img src="plots/lifecycle-cold-idle.svg" alt="Lifecycle: cold start and idle wake (chart)"></p>

_Seconds to first token after a cold start (page cache dropped) and after idle-policy expiry; blazar keeps weights resident while ollama reloads from disk. Warm-daemon ollama caveat applies. Lower is better. Source: 20261009-sglang-rebaseline/cells.jsonl._

## Speed (single-stream, medians)

| engine | provider | params | n | ttft p50 (ms) | ttft p99 (ms) | itl p99 (ms) | decode t/s | prefill t/s (cached) | VRAM peak (MiB) |
|---||---||---||---||---||---||---||---||---||
| ollama-host | ollama | reference=True NOTE: serves 'qwen3:1.7b' - t/s NOT comparable | 1 | 39 | 40 | 7 | 158.6 | 61201.8 | 2321 |
| sglang-0.5.21 | blazar | config=default | 1 | 40 | 49 | 17 | 66.0 | 9518.9 | 6307 |
| sglang-0.5.21 | blazar | config=single-stream | 1 | 39 | 47 | 16 | 66.2 | 9577.6 | 6093 |

## Concurrency (parallel streams, medians)

| provider | engine | params | n | sys t/s | ttft max (ms) | itl p99 (ms) | ok/errors |
|---||---||---||---||---||---||---||---||
| conc-blazar | sglang-0.5.21 | conc=1 rounds=3 | 1 | 63.9 | 42 | 16 | 3/0 |
| conc-blazar | sglang-0.5.21 | conc=2 rounds=3 | 1 | 118.2 | 50 | 16 | 6/0 |
| conc-blazar | sglang-0.5.21 | conc=4 rounds=3 | 1 | 205.8 | 51 | 16 | 12/0 |
| conc-blazar | sglang-0.5.21 | conc=8 rounds=3 | 1 | 218.4 | 1722 | 37 | 24/0 |
| conc-ollama | ollama-host | conc=1 rounds=3 | 1 | 81.5 | 2259 | 7 | 3/0 |
| conc-ollama | ollama-host | conc=2 rounds=3 | 1 | 105.5 | 3172 | 8 | 6/0 |
| conc-ollama | ollama-host | conc=4 rounds=3 | 1 | 125.6 | 4787 | 8 | 12/0 |
| conc-ollama | ollama-host | conc=8 rounds=3 | 1 | 137.0 | 8286 | 8 | 24/0 |
- sys t/s = total tokens / wall (true system throughput); flat sys t/s with growing ttft max means streams queued on a fixed slot count instead of being served concurrently.


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
