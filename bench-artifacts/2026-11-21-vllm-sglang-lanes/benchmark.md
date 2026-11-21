# Blazar benchmark matrix

- **date**: 2026-11-21 11:37:14
- **model**: `qwen2.5-coder-0.5b-instruct.d` (953 MiB)
- **gpu**: NVIDIA GeForce RTX 4070 Laptop GPU / driver 595.99.02
- **engines**: inventory, ollama-host, sglang-0.5.23, vllm-0.34.0
- **harness**: bench_matrix v4 - `--model qwen2.5-coder-0.5b-instruct --engines vllm-0.34.0 sglang-0.5.23 --providers blazar --runs 3 --skip-media --artifacts-dir ~/.cache/blazar-bench-matrix/20261121-111035`
- **blazar**: `blazar 0.34.0` (sandbox daemon binary)
- **quality lane**: ran - 1 scored row(s); suites: code, instruct, multilingual, niah, reason, safety, schema
- all blazar-owned rows measured by `blazar 0.34.0`

## Notable findings

- 2 cell(s) aborted on ENVIRONMENT guards (GPU/RAM co-residency), not product behavior - see Failed cells.

## Engine coverage

_Engine selection policy: latest installed version per engine kind._
| Engine | Kind | Lane | ok cells | err cells | Status |
|---|---|---|---:|---:|---|
| sglang-0.5.23 | sglang | text (direct venv + gateway) | 24 | 1 | benchmarked |
| vllm-0.34.0 | vllm | text (gateway venv) | 10 | 2 | benchmarked |
| 2023.11.14-2 | - | - | 0 | 0 | excluded: kind 'piper' has no bench lane in this harness |
| b12578-cuda | - | - | 0 | 0 | excluded: kind 'llamacpp' has no bench lane in this harness |
| b5907 | - | - | 0 | 0 | excluded: kind 'whisper' has no bench lane in this harness |
| fork-AtomicBot-ai_atomic-llama-cpp-turboquant-8d935078-cuda | - | - | 0 | 0 | excluded: latest-per-kind policy: superseded by b12578-cuda |
| fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | - | - | 0 | 0 | excluded: latest-per-kind policy: superseded by b12578-cuda |
| kokoro-1.0 | - | - | 0 | 0 | excluded: kind 'kokoro' has no bench lane in this harness |
| litert-0.19.2 | - | - | 0 | 0 | excluded: model-format sweep: this campaign sweeps a non-litertlm model; the openaihost lane serves .litertlm bundles (rerun with --model <litertlm-row>) |
| master-1043-3be2378 | - | - | 0 | 0 | excluded: kind 'sdcpp' has no bench lane in this harness |
| mlx-0.32.2 | - | - | 0 | 0 | excluded: kind 'mlx' has no bench lane in this harness |
| v0.9.5 | - | - | 0 | 0 | excluded: kind 'mistralrs' has no bench lane in this harness |

## Charts

- deterministic SVG renders of the cells below; source of truth is `cells.jsonl`, charts live in `plots/`.

### Single-stream decode throughput (chart)

<p align="center"><img src="plots/speed-single-stream.svg" alt="Single-stream decode throughput (chart)"></p>

_Median decode t/s per runtime and engine; whiskers span the interquartile range of the 5 runs; the dashed line marks the fastest direct engine. Higher is better. Source: 20261121-111035/cells.jsonl._

### Concurrency scaling (chart)

<p align="center"><img src="plots/concurrency-throughput.svg" alt="Concurrency scaling (chart)"></p>

_Aggregate system tokens/s as parallel streams are added; flat-to-rising means the scheduler keeps the device saturated. Higher is better. Source: 20261121-111035/cells.jsonl._

### Concurrency tail latency (chart)

<p align="center"><img src="plots/concurrency-ttft.svg" alt="Concurrency tail latency (chart)"></p>

_Worst-case first-token wait per stream as concurrency rises (log scale) - the tail the scheduler must bound. Lower is better. Dashed curves carry the median time-to-first-byte, which upper-bounds queue wait under burst arrival. Source: 20261121-111035/cells.jsonl._

### Resource cost vs concurrency (chart)

<p align="center"><img src="plots/concurrency-vram.svg" alt="Resource cost vs concurrency (chart)"></p>

_Peak VRAM footprint as parallel streams (and their KV caches) stack up. Source: 20261121-111035/cells.jsonl._

### Resource cost vs concurrency (chart)

<p align="center"><img src="plots/concurrency-power.svg" alt="Resource cost vs concurrency (chart)"></p>

_Peak GPU board power per concurrency level - the energy price of keeping the device saturated. Source: 20261121-111035/cells.jsonl._

### Long-context degradation (chart)

<p align="center"><img src="plots/ctx-curve.svg" alt="Long-context degradation (chart)"></p>

_Single-stream decode t/s as prompt context grows (log x-axis) - KV-cache pressure made visible. Source: 20261121-111035/cells.jsonl._

### Memory vs context (chart)

<p align="center"><img src="plots/vram-vs-context.svg" alt="Memory vs context (chart)"></p>

_Peak VRAM as prompt context grows (log x-axis) - the KV-cache slope that sets the usable context ceiling. Source: 20261121-111035/cells.jsonl._

### Lifecycle: cold start and idle wake (chart)

<p align="center"><img src="plots/lifecycle-cold-idle.svg" alt="Lifecycle: cold start and idle wake (chart)"></p>

_Seconds to first token after a cold start (page cache dropped) and after idle-policy expiry; blazar keeps weights resident while ollama reloads from disk. Warm-daemon ollama caveat applies. Lower is better. Source: 20261121-111035/cells.jsonl._

### Quality suites (chart)

<p align="center"><img src="plots/quality-suites.svg" alt="Quality suites (chart)"></p>

_Deterministic-checker pass rates per suite (reasoning, instruction following, code, schema, needle-in-haystack, multilingual, safety) across runtimes on the identical seeded task set. Source: 20261121-111035/cells.jsonl._

## Speed (single-stream, medians)

| engine | provider | params | n | ttft p50 (ms) | ttft p99 (ms) | itl p99 (ms) | decode t/s | prefill t/s (cached) | VRAM peak (MiB) |
|---||---||---||---||---||---||---||---||---||
| ollama-host | ctxcurve-ollama | ctx=16384 | 1 | 8 | - | - | 335.4 | - | 838 |
| ollama-host | ctxcurve-ollama | ctx=2048 | 1 | 8 | - | - | 326.4 | - | 606 |
| ollama-host | ctxcurve-ollama | ctx=8192 | 1 | 8 | - | - | 328.7 | - | 726 |
| ollama-host | ollama | reference=True NOTE: serves 'qwen2.5:0.5b' - t/s NOT comparable | 1 | 16 | 18 | 4 | 338.5 | 144112.6 | 726 |
| sglang-0.5.23 | blazar | config=chunked_prefill_4096 | 1 | 20 | 22 | 5 | 201.9 | 28349.9 | 2300 |
| sglang-0.5.23 | blazar | config=default | 1 | 18 | 23 | 5 | 201.9 | 25759.9 | 2104 |
| sglang-0.5.23 | blazar | config=deterministic_on | 1 | 38 | 47 | 12 | 85.0 | 16462.8 | 1936 |
| sglang-0.5.23 | blazar | config=kv_dtype_bf16 | 1 | 19 | 30 | 5 | 201.0 | 27514.1 | 2106 |
| sglang-0.5.23 | blazar | config=kv_dtype_e4m3 | 1 | 20 | 22 | 5 | 199.2 | 26635.7 | 2154 |
| sglang-0.5.23 | blazar | config=kv_unified_off | 1 | 19 | 22 | 5 | 201.7 | 26134.3 | 2104 |
| sglang-0.5.23 | blazar | config=kv_unified_on | 1 | 18 | 121 | 5 | 201.5 | 30083.0 | 2106 |
| sglang-0.5.23 | blazar | config=mem_frac_0.90 | 1 | 19 | 21 | 5 | 201.1 | 27755.5 | 6664 |
| sglang-0.5.23 | blazar | config=memory_saver_on | 1 | 19 | 23 | 5 | 201.7 | 26791.3 | 2104 |
| sglang-0.5.23 | blazar | config=page_64 | 1 | 20 | 21 | 5 | 201.5 | 22089.8 | 2102 |
| sglang-0.5.23 | blazar | config=radix_session_on | 1 | 19 | 20 | 5 | 201.8 | 28669.6 | 2106 |
| sglang-0.5.23 | blazar | config=single-stream | 1 | 20 | 31 | 6 | 201.6 | 27689.5 | 1908 |
| sglang-0.5.23 | blazar | config=spec_off | 1 | 19 | 21 | 5 | 202.0 | 28917.6 | 2104 |
| sglang-0.5.23 | blazar | config=torch_compile_on | 1 | 24 | 27 | 10 | 99.6 | 27844.9 | 2358 |
| sglang-0.5.23 | ctxcurve-blazar | ctx=16384 | 1 | 20 | 22 | 6 | 202.2 | 26857.5 | 2104 |
| sglang-0.5.23 | ctxcurve-blazar | ctx=2048 | 1 | 19 | 23 | 5 | 202.1 | 26950.7 | 2006 |
| sglang-0.5.23 | ctxcurve-blazar | ctx=8192 | 1 | 19 | 22 | 5 | 202.2 | 26175.1 | 2038 |
| vllm-0.34.0 | blazar | config=default child: ctx=16384 | 1 | 20 | 20 | 6 | 189.8 | 17339.5 | 6222 |
| vllm-0.34.0 | blazar | config=single-stream child: ctx=16384 | 1 | 19 | 19 | 6 | 189.5 | 19631.0 | 6222 |
| vllm-0.34.0 | ctxcurve-blazar | ctx=16384 child: ctx=16384 | 1 | 18 | 19 | 7 | 188.8 | 20126.5 | 6222 |
| vllm-0.34.0 | ctxcurve-blazar | ctx=2048 child: ctx=2048 | 1 | 18 | 27 | 6 | 192.4 | 19365.4 | 6222 |
| vllm-0.34.0 | ctxcurve-blazar | ctx=8192 child: ctx=8192 | 1 | 21 | 21 | 6 | 189.5 | 20252.7 | 6222 |

## Concurrency (parallel streams, medians)

| provider | engine | params | n | sys t/s | ttft max (ms) | itl p99 (ms) | ok/errors |
|---||---||---||---||---||---||---||---||
| conc-blazar | sglang-0.5.23 | conc=4 rounds=3 | 1 | 328.4 | 49 | 6 | 12/0 |
| conc-blazar | vllm-0.34.0 | conc=4 rounds=3 | 1 | 469.6 | 34 | 7 | 12/0 |
| conc-ollama | ollama-host | conc=4 rounds=3 | 1 | 192.6 | 2628 | 4 | 12/0 |
- sys t/s = total tokens / wall (true system throughput); flat sys t/s with growing ttft max means streams queued on a fixed slot count instead of being served concurrently.


## Quality - perplexity (identical pinned args)

| engine | perplexity | +/- err | wall (s) | note |
|---|---|---|---|---|
| sglang-0.5.23 | - | - | - | lower = better text fit |
| vllm-0.34.0 | - | - | - | lower = better text fit |
- corpus: deterministic offline repo text (code-heavy) - PARITY-ONLY; absolute PPL is not comparable to published wiki-text perplexities.

## Feature matrix

| feature | sglang-0.5.23 | vllm-0.34.0 | ollama(documented) |
|---|---|---|---|
| `anthropic-api` | - | - | - |
| `ctx-override` | - | Y | Y |
| `embeddings` | - | - | Y |
| `grammar-gbnf` | - | - | - |
| `json-schema` | - | - | Y |
| `kv-quant` | - | - | - |
| `lora-adapter` | - | - | Y |
| `metrics-endpoint` | - | - | - |
| `paged-attn` | - | - | - |
| `parallel-np` | - | - | Y |
| `quant-on-load` | - | - | - |
| `rerank` | - | - | - |
| `slots-sessions` | - | - | - |
| `spec-decode` | - | - | - |
| `tokenize-endpoint` | - | - | Y |
| `vision-mmproj` | - | - | Y |

## Failed cells

**product** (engine/gateway behavior):

- `vllm-0.34.0` / tools / {'tools': True}: tools lane: every scenario failed transport - {"error":{"message":"\"auto\" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set",...

**environment** (box/co-residency guards - NOT blazar defects):

- `sglang-0.5.23` / quality / {'config': 'default'}: mem_guard: GPU too busy for quality lane
- `vllm-0.34.0` / quality / {'config': 'default'}: mem_guard: GPU too busy for quality lane

## Reading this report

- `direct` = raw child spawn on a probed free port (no gateway); `blazar` = full gateway path inside a sandboxed daemon (resolved engine argv in `child_argv`); `ollama` = HTTP-only reference against the host service.
- decode counts ALL emitted tokens (content + reasoning); engine `usage` counters are authoritative when present. GPU/power peaks sampled at ~1.2 s cadence.
- per-run spread, cold-start/resource detail, and every raw cell live in `cells.jsonl`; tables here are per-config medians.
