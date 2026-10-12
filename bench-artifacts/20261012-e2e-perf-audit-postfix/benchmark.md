# Blazar benchmark matrix

- **date**: 2026-10-12 17:47:20
- **model**: `qwen2.5-0.5b-instruct-q4_k_m.gguf` (468 MiB)
- **gpu**: NVIDIA GeForce RTX 4070 Laptop GPU / driver 595.99.02
- **engines**: b11539-cuda, inventory, ollama-host
- **harness**: bench_matrix v4 — `--model qwen2.5-0.5b-instruct --engines b11539-cuda --providers direct blazar --conc-sweep 1,2,4,8 --skip-ppl --skip-tools --skip-quality --skip-media --skip-features --skip-ctxcurve --skip-reshape --skip-idle --skip-variants --blazar-bin ./target/release/blazar --artifacts-dir bench-artifacts/20261012-e2e-perf-audit-postfix --md bench-artifacts/20261012-e2e-perf-audit-postfix/report.md`
- **blazar**: `blazar 0.26.0` (sandbox daemon binary)
- **quality lane**: skipped — --skip-quality: no checker-verified receipts back any speed number in this campaign
- all blazar-owned rows measured by `blazar 0.26.0`

## Notable findings

- gateway vs direct decode (`b11539-cuda`, 2 configs): median -6.7%, 0/2 within ±5% (parity); outliers: single-stream -8.0%, default -6.7%.
- concurrency system throughput (`b11539-cuda`, 1 streams): 285.5 vs direct 295.2 t/s = 0.97x.
- concurrency system throughput (`b11539-cuda`, 2 streams): 374.0 vs direct 481.4 t/s = 0.78x.
- concurrency system throughput (`b11539-cuda`, 4 streams): 456.4 vs direct 662.7 t/s = 0.69x.
- concurrency system throughput (`b11539-cuda`, 8 streams): 472.3 vs direct 929.8 t/s = 0.51x — serialized/queued or wall-inflated (see note).
- gateway greedy transparency (`b11539-cuda`): 14/20 exact vs same-engine direct — NOT TRANSPARENT.

## Charts

- deterministic SVG renders of the cells below; source of truth is `cells.jsonl`, charts live in `plots/`.

### Single-stream decode throughput (chart)

<p align="center"><img src="plots/speed-single-stream.svg" alt="Single-stream decode throughput (chart)"></p>

_Median decode t/s per runtime and engine; whiskers span the interquartile range of the 5 runs; the dashed line marks the fastest direct engine. Higher is better. Source: 20261012-e2e-perf-audit-postfix/cells.jsonl._

### Gateway overhead (chart)

<p align="center"><img src="plots/gateway-overhead.svg" alt="Gateway overhead (chart)"></p>

_Decode t/s delta of routing through blazar relative to driving the same engine build directly; left of zero means the gateway path won. Source: 20261012-e2e-perf-audit-postfix/cells.jsonl._

### Concurrency scaling (chart)

<p align="center"><img src="plots/concurrency-throughput.svg" alt="Concurrency scaling (chart)"></p>

_Aggregate system tokens/s as parallel streams are added; flat-to-rising means the scheduler keeps the device saturated. Higher is better. Source: 20261012-e2e-perf-audit-postfix/cells.jsonl._

### Concurrency tail latency (chart)

<p align="center"><img src="plots/concurrency-ttft.svg" alt="Concurrency tail latency (chart)"></p>

_Worst-case first-token wait per stream as concurrency rises (log scale) - the tail the scheduler must bound. Lower is better. Dashed curves carry the median time-to-first-byte, which upper-bounds queue wait under burst arrival. Source: 20261012-e2e-perf-audit-postfix/cells.jsonl._

### Resource cost vs concurrency (chart)

<p align="center"><img src="plots/concurrency-vram.svg" alt="Resource cost vs concurrency (chart)"></p>

_Peak VRAM footprint as parallel streams (and their KV caches) stack up. Source: 20261012-e2e-perf-audit-postfix/cells.jsonl._

### Resource cost vs concurrency (chart)

<p align="center"><img src="plots/concurrency-power.svg" alt="Resource cost vs concurrency (chart)"></p>

_Peak GPU board power per concurrency level - the energy price of keeping the device saturated. Source: 20261012-e2e-perf-audit-postfix/cells.jsonl._

### Lifecycle: cold start and idle wake (chart)

<p align="center"><img src="plots/lifecycle-cold-idle.svg" alt="Lifecycle: cold start and idle wake (chart)"></p>

_Seconds to first token after a cold start (page cache dropped) and after idle-policy expiry; blazar keeps weights resident while ollama reloads from disk. Lower is better. Source: 20261012-e2e-perf-audit-postfix/cells.jsonl._

## Speed (single-stream, medians)

| engine | provider | params | n | ttft p50 (ms) | ttft p99 (ms) | itl p99 (ms) | decode t/s | prefill t/s (cached) | VRAM peak (MiB) |
|---||---||---||---||---||---||---||---||---||
| b11539-cuda | blazar | config=default child: np=2 ctx=32768 | 1 | 8 | 11 | 5 | 324.0 | 72329.9 | 1272 |
| b11539-cuda | blazar | config=single-stream child: np=1 ctx=16384 | 1 | 9 | 10 | 4 | 319.5 | 71711.0 | 1064 |
| b11539-cuda | direct | ctx=16384 np=1 | 1 | 6 | 9 | 4 | 343.9 | 81076.4 | 790 |
| b11539-cuda | direct | ctx=16384 np=4 | 1 | 7 | 8 | 4 | 341.7 | 84421.0 | 778 |
| b11539-cuda | direct | ctx=4096 np=1 | 1 | 6 | 10 | 4 | 347.4 | 80673.0 | 634 |
| b11539-cuda | direct | ctx=4096 np=4 | 1 | 6 | 9 | 4 | 341.7 | 78083.7 | 632 |

## Concurrency (parallel streams, medians)

| provider | engine | params | n | sys t/s | ttft max (ms) | itl p99 (ms) | ok/errors |
|---||---||---||---||---||---||---||---||
| conc-blazar | b11539-cuda | conc=1 rounds=3 | 1 | 285.5 | 11 | 5 | 3/0 |
| conc-blazar | b11539-cuda | conc=2 rounds=3 | 1 | 374.0 | 67 | 6 | 6/0 |
| conc-blazar | b11539-cuda | conc=4 rounds=3 | 1 | 456.4 | 524 | 5 | 12/0 |
| conc-blazar | b11539-cuda | conc=8 rounds=3 | 1 | 472.3 | 1278 | 6 | 24/0 |
| conc-direct | b11539-cuda | conc=1 rounds=3 | 1 | 295.2 | 8 | 4 | 3/0 |
| conc-direct | b11539-cuda | conc=2 rounds=3 | 1 | 481.4 | 20 | 7 | 6/0 |
| conc-direct | b11539-cuda | conc=4 rounds=3 | 1 | 662.7 | 33 | 9 | 12/0 |
| conc-direct | b11539-cuda | conc=8 rounds=3 | 1 | 929.8 | 40 | 10 | 24/0 |
- sys t/s = total tokens / wall (true system throughput); flat sys t/s with growing ttft max means streams queued on a fixed slot count instead of being served concurrently.


## Quality — greedy parity vs `b11539-cuda`-direct (backend numerics)

| engine | exact matches | ratio mean | ratio min | first divergence (median chars) |
|---|---|---|---|---|
| b11539-cuda *(self — trivially 1.0) | 20/20 | 1.0 | 1.0 | 976.5 |

## Quality — gateway transparency (blazar path vs direct, same engine)

| engine | exact matches | ratio mean | ratio min | first divergence (median chars) |
|---|---|---|---|---|
| b11539-cuda | 14/20 | 0.9407 | 0.4378 | 894.0 |
- expectation: 20/20 exact, ratio 1.0. A miss has TWO possible causes: gateway translation defect (sampler remap / template drift), or multi-slot batching numerics (child -np > 1 changes float reduction order; near-tie logits flip). Pin `slots = 1` and re-run: still <20/20 = translation defect, 20/20 = slot-count numerics (upstream physics).
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

- `ollama-host` / ollama / {'reference': True}: no ollama model comparable to 'qwen2.5-0.5b-instruct' (have: glm-ocr:latest, qwen3:1.7b, qwen3.5:9b) — pull a matching tag
- `ollama-host` / cold-ollama / {'cold': True}: no ollama model comparable to 'qwen2.5-0.5b-instruct' (have: glm-ocr:latest, qwen3:1.7b, qwen3.5:9b) — pull a matching tag
- `ollama-host` / conc-ollama (4 cells): no ollama model comparable to 'qwen2.5-0.5b-instruct' [{'conc': 1, 'rounds': 3}; {'conc': 2, 'rounds': 3}; {'conc': 4, 'rounds': 3}; {'conc': 8, 'rounds': 3}]
- `ollama-host` / greedy_ollama / {'greedy_ollama': True}: no same-family ollama model for greedy reference

## Reading this report

- `direct` = raw child spawn on a probed free port (no gateway); `blazar` = full gateway path inside a sandboxed daemon (resolved engine argv in `child_argv`); `ollama` = HTTP-only reference against the host service.
- decode counts ALL emitted tokens (content + reasoning); engine `usage` counters are authoritative when present. GPU/power peaks sampled at ~1.2 s cadence.
- per-run spread, cold-start/resource detail, and every raw cell live in `cells.jsonl`; tables here are per-config medians.
