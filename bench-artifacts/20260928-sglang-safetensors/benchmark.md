# Blazar benchmark matrix

- **date**: 2026-09-29 08:57:19
- **model**: `qwen3-1.7b.d` (3890 MiB)
- **gpu**: NVIDIA GeForce RTX 4070 Laptop GPU / driver 595.91.07
- **engines**: inventory, ollama-host, piper, sglang-0.5.19
- **harness**: bench_matrix v2 — `--model qwen3-1.7b --engines sglang-0.5.19 --artifacts-dir bench-artifacts/20260928-sglang-safetensors --md bench-artifacts/20260928-sglang-safetensors/BENCHMARK.md`
- **blazar**: `blazar 0.13.0` (sandbox daemon binary)
- all blazar-owned rows measured by `blazar 0.13.0`

## Speed (serving, streaming)

| engine | kind | provider | params | ttft p50 (ms) | ttft p99 (ms) | itl p50 (ms) | itl p99 (ms) | decode t/s | prefill t/s (cold) | prefill t/s (cached) | tokens src |
|---||---||---||---||---||---||---||---||---||---||---||
| sglang-0.5.19 | sglang | blazar | config=default | 33 | 33 | 15 | 15 | 67.6 | 7471.1 | 10985.4 | chunks |
| sglang-0.5.19 | sglang | blazar | config=single-stream | 33 | 33 | 15 | 15 | 67.6 | 6300.3 | 11048.7 | chunks |
| sglang-0.5.19 | sglang | blazar | config=kv_unified_on | 33 | 33 | 15 | 16 | 67.6 | 6315.2 | 11080.0 | chunks |
| sglang-0.5.19 | sglang | blazar | config=kv_unified_off | 33 | 33 | 15 | 15 | 67.6 | 7322.9 | 10927.9 | chunks |
| sglang-0.5.19 | sglang | blazar | config=spec_off | 33 | 33 | 15 | 15 | 67.6 | 6507.5 | 10897.4 | chunks |
| sglang-0.5.19 | sglang | blazar | config=deterministic_on | 59 | 62 | 25 | 27 | 39.5 | 4788.8 | 6273.0 | chunks |
| sglang-0.5.19 | sglang | blazar | config=radix_session_on | 33 | 34 | 15 | 15 | 67.7 | 7395.2 | 11005.0 | chunks |
| sglang-0.5.19 | sglang | blazar | config=chunked_prefill_4096 | 33 | 33 | 15 | 15 | 67.6 | 6342.4 | 10988.9 | chunks |
| sglang-0.5.19 | sglang | blazar | config=page_64 | 33 | 34 | 15 | 16 | 67.6 | 7125.1 | 10669.5 | chunks |
| sglang-0.5.19 | sglang | blazar | config=torch_compile_on | 32 | 33 | 15 | 15 | 68.7 | 6973.2 | 11274.0 | chunks |
| ollama-host | ollama | ollama | reference=True ⚠ serves 'qwen3:1.7b' — t/s NOT comparable | 33 | 39 | 6 | 7 | 157.5 | 3705.1 | 60940.3 | engine_counters |
| sglang-0.5.19 | sglang | ctxcurve-blazar | ctx=2048 | 33 | 34 | 15 | 15 | 67.7 | 7182.3 | 10907.4 | chunks |
| sglang-0.5.19 | sglang | ctxcurve-blazar | ctx=8192 | 33 | 33 | 15 | 15 | 67.6 | 6341.9 | 11111.5 | chunks |
| sglang-0.5.19 | sglang | ctxcurve-blazar | ctx=16384 | 33 | 33 | 15 | 15 | 67.6 | 7557.6 | 11140.6 | chunks |
| ollama-host | ollama | ctxcurve-ollama | ctx=2048 | 43 | - | - | - | 156.5 | - | - | - |
| ollama-host | ollama | ctxcurve-ollama | ctx=8192 | 40 | - | - | - | 156.3 | - | - | - |
| ollama-host | ollama | ctxcurve-ollama | ctx=16384 | 42 | - | - | - | 157.4 | - | - | - |

## Resources & cold start

| engine | provider | params | load s | daemon boot s | cold 1st req s | GPU peak (MiB) | GPU power (W) | RSS peak (MiB) | teardown |
|---||---||---||---||---||---||---||---||---||
| sglang-0.5.19 | blazar | config=default | - | 0.55 | 32.31 | 6470 | 55.1 | - | ok |
| sglang-0.5.19 | blazar | config=single-stream | - | 0.53 | 31.24 | 6340 | 55.1 | - | ok |
| sglang-0.5.19 | blazar | config=kv_unified_on | - | 0.53 | 30.24 | 6450 | 55.6 | - | ok |
| sglang-0.5.19 | blazar | config=kv_unified_off | - | 0.53 | 31.25 | 6490 | 55.6 | - | ok |
| sglang-0.5.19 | blazar | config=spec_off | - | 0.52 | 31.29 | 6482 | 56.0 | - | ok |
| sglang-0.5.19 | blazar | config=deterministic_on | - | 0.52 | 36.14 | 6742 | 55.4 | - | ok |
| sglang-0.5.19 | blazar | config=radix_session_on | - | 0.53 | 30.24 | 6484 | 55.1 | - | ok |
| sglang-0.5.19 | blazar | config=chunked_prefill_4096 | - | 0.53 | 37.3 | 6738 | 55.9 | - | ok |
| sglang-0.5.19 | blazar | config=page_64 | - | 0.53 | 32.17 | 6490 | 55.5 | - | ok |
| sglang-0.5.19 | blazar | config=torch_compile_on | - | 0.52 | 99.36 | 7070 | 55.0 | - | ok |
| ollama-host | ollama | reference=True | - | - | - | 2320 | 54.9 | - | ok |
| sglang-0.5.19 | ctxcurve-blazar | ctx=2048 | - | - | - | 4792 | - | - | ok |
| sglang-0.5.19 | ctxcurve-blazar | ctx=8192 | - | - | - | 5490 | - | - | ok |
| sglang-0.5.19 | ctxcurve-blazar | ctx=16384 | - | - | - | 6470 | - | - | ok |
| ollama-host | ctxcurve-ollama | ctx=2048 | 2.55 | - | - | 1588 | - | - | ok |
| ollama-host | ctxcurve-ollama | ctx=8192 | 2.95 | - | - | 2322 | - | - | ok |
| ollama-host | ctxcurve-ollama | ctx=16384 | 2.03 | - | - | 3234 | - | - | ok |

## Concurrency (parallel streams)

| engine | provider | streams | sys t/s | sum stream t/s | ttft max (ms) | ttft spread (ms) | itl p99 (ms) | ok/errors | wall (s) |
|---||---||---||---||---||---||---||---||---||
| sglang-0.5.19 | conc-blazar | conc=4 rounds=3 | 254.0 | 771.1 | 41 | 8.6 | 16.38 | 12/0 | 6.05 |
| ollama-host | conc-ollama | conc=4 rounds=3 | 122.2 | 1869.2 | 5066 | 5021.9 | 7.86 | 12/0 | 12.57 |
- sys t/s = total tokens / wall (true system throughput); sum stream t/s = sum of per-stream rates. sum >> sys means streams were serialized (queued on a single slot) rather than served concurrently.


## Quality — perplexity (identical pinned args)

| engine | perplexity | ± err | wall (s) | note |
|---|---|---|---|---|
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

- `sglang-0.5.19` / blazar / {'config': 'mem_frac_0.90'}: cold probe failed: HTTP Error 502: Bad Gateway
- `sglang-0.5.19` / blazar / {'config': 'hicache_on'}: cold probe failed: HTTP Error 502: Bad Gateway
- `sglang-0.5.19` / blazar / {'config': 'memory_saver_on'}: cold probe failed: HTTP Error 502: Bad Gateway

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
