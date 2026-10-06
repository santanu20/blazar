# Blazar benchmark matrix

- **date**: 2026-10-06 16:00:26
- **model**: `qwen3-1.7b.d` (3890 MiB)
- **gpu**: NVIDIA GeForce RTX 4070 Laptop GPU / driver 595.91.07
- **engines**: inventory, ollama-host, piper, sglang-0.5.19
- **harness**: bench_matrix v4 — `--model qwen3-1.7b --engines sglang-0.5.19 --artifacts-dir bench-artifacts/20260928-sglang-safetensors --md bench-artifa`
- **blazar**: `blazar 0.13.0` (sandbox daemon binary)
- **quality lane**: not-measured — no quality rows in this campaign (lane not reached for the selected providers/engines)
- all blazar-owned rows measured by `blazar 0.13.0`

## Notable findings

- 2 cell(s) aborted on ENVIRONMENT guards (GPU/RAM co-residency), not product behavior — see Failed cells.

## Charts

- deterministic SVG renders of the cells below; source of truth is `cells.jsonl`, charts live in `plots/`.

### Single-stream decode throughput (chart)

<p align="center"><img src="plots/speed-single-stream.svg" alt="Single-stream decode throughput (chart)"></p>

_Median decode t/s per runtime and engine; whiskers span the interquartile range of the 5 runs; the dashed line marks the fastest direct engine. Higher is better. Source: 20260928-sglang-safetensors/cells.jsonl._

### Concurrency scaling (chart)

<p align="center"><img src="plots/concurrency-throughput.svg" alt="Concurrency scaling (chart)"></p>

_Aggregate system tokens/s as parallel streams are added; flat-to-rising means the scheduler keeps the device saturated. Higher is better. Source: 20260928-sglang-safetensors/cells.jsonl._

### Concurrency tail latency (chart)

<p align="center"><img src="plots/concurrency-ttft.svg" alt="Concurrency tail latency (chart)"></p>

_Worst-case first-token wait per stream as concurrency rises (log scale) - the tail the scheduler must bound. Lower is better. Source: 20260928-sglang-safetensors/cells.jsonl._

### Resource cost vs concurrency (chart)

<p align="center"><img src="plots/concurrency-vram.svg" alt="Resource cost vs concurrency (chart)"></p>

_Peak VRAM footprint as parallel streams (and their KV caches) stack up. Source: 20260928-sglang-safetensors/cells.jsonl._

### Long-context degradation (chart)

<p align="center"><img src="plots/ctx-curve.svg" alt="Long-context degradation (chart)"></p>

_Single-stream decode t/s as prompt context grows (log x-axis) - KV-cache pressure made visible. Source: 20260928-sglang-safetensors/cells.jsonl._

### Memory vs context (chart)

<p align="center"><img src="plots/vram-vs-context.svg" alt="Memory vs context (chart)"></p>

_Peak VRAM as prompt context grows (log x-axis) - the KV-cache slope that sets the usable context ceiling. Source: 20260928-sglang-safetensors/cells.jsonl._

### Lifecycle: cold start and idle wake (chart)

<p align="center"><img src="plots/lifecycle-cold-idle.svg" alt="Lifecycle: cold start and idle wake (chart)"></p>

_Seconds to first token after a cold start (page cache dropped) and after idle-policy expiry; blazar keeps weights resident while ollama reloads from disk. Warm-daemon ollama caveat applies. Lower is better. Source: 20260928-sglang-safetensors/cells.jsonl._

## Speed (single-stream, medians)

| engine | provider | params | n | ttft p50 (ms) | ttft p99 (ms) | itl p99 (ms) | decode t/s | prefill t/s (cached) | VRAM peak (MiB) |
|---||---||---||---||---||---||---||---||---||
| ollama-host | ctxcurve-ollama | ctx=16384 | 1 | 42 | - | - | 157.4 | - | 3234 |
| ollama-host | ctxcurve-ollama | ctx=2048 | 1 | 43 | - | - | 156.5 | - | 1588 |
| ollama-host | ctxcurve-ollama | ctx=8192 | 1 | 40 | - | - | 156.3 | - | 2322 |
| ollama-host | ollama | reference=True NOTE: serves 'qwen3:1.7b' - t/s NOT comparable | 1 | 33 | 39 | 7 | 157.5 | 60940.3 | 2320 |
| sglang-0.5.19 | blazar | config=cg_bs_16 | 1 | 33 | 34 | 15 | 67.6 | 10972.5 | 6470 |
| sglang-0.5.19 | blazar | config=cg_bs_256 | 1 | 33 | 33 | 15 | 67.6 | 10967.7 | 6804 |
| sglang-0.5.19 | blazar | config=chunked_prefill_4096 | 1 | 33 | 33 | 15 | 67.6 | 10988.9 | 6738 |
| sglang-0.5.19 | blazar | config=default | 1 | 33 | 33 | 15 | 67.6 | 10985.4 | 6470 |
| sglang-0.5.19 | blazar | config=deterministic_on | 1 | 59 | 62 | 27 | 39.5 | 6273.0 | 6742 |
| sglang-0.5.19 | blazar | config=kv_dtype_bf16 | 1 | 33 | 34 | 15 | 67.7 | 10941.7 | 6416 |
| sglang-0.5.19 | blazar | config=kv_dtype_e4m3 | 1 | 33 | 34 | 16 | 67.4 | 10978.3 | 6474 |
| sglang-0.5.19 | blazar | config=kv_unified_off | 1 | 33 | 33 | 15 | 67.6 | 10927.9 | 6490 |
| sglang-0.5.19 | blazar | config=kv_unified_on | 1 | 33 | 33 | 16 | 67.6 | 11080.0 | 6450 |
| sglang-0.5.19 | blazar | config=page_64 | 1 | 33 | 34 | 16 | 67.6 | 10669.5 | 6490 |
| sglang-0.5.19 | blazar | config=radix_session_on | 1 | 33 | 34 | 15 | 67.7 | 11005.0 | 6484 |
| sglang-0.5.19 | blazar | config=single-stream | 1 | 33 | 33 | 15 | 67.6 | 11048.7 | 6340 |
| sglang-0.5.19 | blazar | config=spec_off | 1 | 33 | 33 | 15 | 67.6 | 10897.4 | 6482 |
| sglang-0.5.19 | blazar | config=torch_compile_on | 1 | 32 | 33 | 15 | 68.7 | 11274.0 | 7070 |
| sglang-0.5.19 | ctxcurve-blazar | ctx=16384 | 1 | 33 | 33 | 15 | 67.6 | 11140.6 | 6470 |
| sglang-0.5.19 | ctxcurve-blazar | ctx=2048 | 1 | 33 | 34 | 15 | 67.7 | 10907.4 | 4792 |
| sglang-0.5.19 | ctxcurve-blazar | ctx=8192 | 1 | 33 | 33 | 15 | 67.6 | 11111.5 | 5490 |

## Concurrency (parallel streams, medians)

| provider | engine | params | n | sys t/s | ttft max (ms) | itl p99 (ms) | ok/errors |
|---||---||---||---||---||---||---||---||
| conc-blazar | sglang-0.5.19 | conc=4 rounds=3 | 1 | 254.0 | 41 | 16 | 12/0 |
| conc-ollama | ollama-host | conc=4 rounds=3 | 1 | 122.2 | 5066 | 8 | 12/0 |
- sys t/s = total tokens / wall (true system throughput); flat sys t/s with growing ttft max means streams queued on a fixed slot count instead of being served concurrently.


## Quality — perplexity (identical pinned args)

| engine | perplexity | +/- err | wall (s) | note |
|---|---|---|---|---|
| sglang-0.5.19 | - | - | - | no llama-perplexity binary on this box |
| sglang-0.5.19 | - | - | - | no PPL line in llama-perplexity output (exit 1); tail: " model from ~/.local/share/blazar/models/qwen3-1.7b.d\n0.00.566.159 E llama_model_load_from_file_impl: failed to load model\n0.00.566.164 E cmn  common_init_: failed to load model '~/.local/share/blazar/models/qwen3-1.7b.d'\n0.00.566.173 E llama_perplexity: unable to load model\n" |
| sglang-0.5.19 | - | - | - | lower = better text fit |
- corpus: deterministic offline repo text (code-heavy) — PARITY-ONLY; absolute PPL is not comparable to published wiki-text perplexities.

## Feature matrix

| feature | sglang-0.5.19 | ollama(documented) |
|---|---|---|
| `anthropic-api` | - | - |
| `ctx-override` | - | Y |
| `embeddings` | - | Y |
| `grammar-gbnf` | - | - |
| `json-schema` | - | Y |
| `kv-quant` | - | - |
| `lora-adapter` | - | Y |
| `metrics-endpoint` | - | - |
| `paged-attn` | - | - |
| `parallel-np` | - | Y |
| `quant-on-load` | - | - |
| `rerank` | - | - |
| `slots-sessions` | - | - |
| `spec-decode` | - | - |
| `tokenize-endpoint` | - | Y |
| `vision-mmproj` | - | Y |

## Failed cells

**product** (engine/gateway behavior):

- `sglang-0.5.19` / blazar (7 cells): cold probe failed: HTTP Error 502: Bad Gateway [{'config': 'mem_frac_0.90'}; {'config': 'hicache_on'}; {'config': 'memory_saver_on'}; {'config': 'mem_frac_0.90'}; +3 more]
- `sglang-0.5.19` / ppl / {'ppl': 2048}: no llama-perplexity binary on this box
- `sglang-0.5.19` / ppl / {'ppl': 2048}: no PPL line in llama-perplexity output (exit 1); tail: " model from ~/.local/share/blazar/models/qwen3-1.7b.d\n0.00.566.159 E llama_model_load_from_file_impl...

**environment** (box/co-residency guards — NOT blazar defects):

- `sglang-0.5.19` / blazar (2 cells): GPU memory floor exceeded before cell [{'config': 'mem_frac_0.90'}; {'config': 'hicache_on'}]

## Reading this report

- `direct` = raw child spawn on a probed free port (no gateway); `blazar` = full gateway path inside a sandboxed daemon (resolved engine argv in `child_argv`); `ollama` = HTTP-only reference against the host service.
- decode counts ALL emitted tokens (content + reasoning); engine `usage` counters are authoritative when present. GPU/power peaks sampled at ~1.2 s cadence.
- per-run spread, cold-start/resource detail, and every raw cell live in `cells.jsonl`; tables here are per-config medians.
