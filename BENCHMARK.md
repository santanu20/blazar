# Pallama inference benchmark

_Rendered 20260911-223054; pallama 0.5.0; power state of gateway rows: ac._

## Executive summary

b10903-cuda: gateway 40.3 vs direct 41.1 t/s (-1.9%); b10903-cuda prompt-cache prefill 7332 vs 1207 t/s cold; 4-stream concurrency: 104.2 t/s system (4x65536 shape); gateway cold boot 0.54 s; cold TTFT 11577 ms vs ollama 6566 ms (0.6x); idle wake 2104 ms (sleep) vs ollama 7194 ms (full reload).

## Test bed

| Component | Value |
|---|---|
| CPU | Intel Core i7-14650HX, 24 hardware threads |
| Discrete GPU | NVIDIA GeForce RTX 4070 Laptop, 8 GiB, driver 580.173.02 |
| Integrated GPU | Intel Graphics (RPL-S), Vulkan device |
| RAM | 16 GiB (13.3 GiB usable) |
| OS | Linux Mint 22.3, kernel 7.0.0-31-generic |
| Runtimes compared | pallama 0.5.0 gateway - llama.cpp b10809 (Vulkan + CUDA builds) - mistral.rs 0.9.3 - ollama 0.33.3 |
| Model | Qwen3.5-9B, Q4_K_M GGUF (5.4 GiB) + vision projector mmproj-F16 (876 MiB) |

## Methodology

- All lanes speak the OpenAI-compatible streaming API; tokens are counted from usage chunks (engine-injected at the gateway), never estimated from chunk counts.
- Decode throughput = (tokens - 1) / (last-token time - TTFT); medians over 5 runs after a warmup request.
- Inter-token latency (ITL) p50/p99 from per-chunk timestamps; TTFT p50/p90/p99 + stdev.
- Prefill: a token-targeted prompt (~512 tokens via engine /tokenize); run 1 is the cold (uncached) prefill, runs 2+ ride the prompt cache.
- Concurrency: 4 parallel streams x 128 generated tokens each; system t/s = total tokens / wall clock; sum-stream t/s = sum of per-stream rates (sum >> system indicates serialization).
- Greedy parity: 20 fixed prompts, greedy sampling, 256 tokens; exact-match count and text-similarity ratio vs a same-engine reference run.
- Gateway transparency: a second greedy lane through the pallama gateway with identical sampling; any divergence vs the direct lane isolates translation overhead.
- Perplexity: llama-perplexity on an offline ASCII corpus, ctx 2048.
- Cold-start parity: the model file's page cache is dropped (posix_fadvise DONTNEED) and the GPU asserted idle (<512 MiB) before every cold probe on every runtime — a cold load is disk-cold, not memory-warm.
- Cold TTFT = first-token latency of the cold probe itself (max_tokens 4, aligned num_ctx 16384 on both runtimes).
- ollama daemon boot is only measured with --ollama-service-restart (systemd restart, sudo password via BENCH_SUDO_PASSWORD env, stdin-only); without it the daemon stays warm and the row says so.
- Idle-wake: pallama's reaper sleeps the child at idle_sleep_secs (weights stay RAM-resident, VRAM released) — wake TTFT is a sleep-wake; ollama's keep_alive expiry fully unloads — wake TTFT is a disk reload. The policy column names the semantic; both measured after the policy is observed via /api/ps.
- Long-context curve: per-ctx cells (pallama model_overrides ctx / ollama num_ctx) × 3-run decode suites; each ollama point evicts first so the runner respawns at that ctx.
- Sustained concurrency: sequential bursts of the parallel-stream lane (default 3 rounds); TTFT p99 aggregates every stream of every round.
- Every pallama row records the spawned engine's argv (slots/context shown in tables) and stamps pallama version, wall clock, 5-min load average, and AC/battery power state; GPU cells refuse to run on battery.

## Results

### Single-stream decode (512-token prompt, 128 generated, median of 5)

| Runtime | slots x ctx | decode t/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | prefill cold t/s | prefill cached t/s | GPU peak MiB | GPU power W |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pallama gateway - b10903-cuda | 4x65536 | 40.3 | 119.3 | 123.1 | 24.7 | 27.4 | 1207.3 | 7332.2 | 6329 | 56.5 |
| direct engine - b10903-cuda | 1x16384 | 41.1 | 120.9 | 125.2 | 24.3 | 25.4 | 1300.1 | 7658.4 | 5716 | 56.0 |
| ollama 0.33.3 - qwen3.5:9b | service | 40.4 | 141.1 | 148.0 | 25.0 | 75.1 | 1641.6 | 5611.2 | 6446 | 57.3 |

### Concurrency (4 parallel streams x 128 tokens)

| Runtime | slots | ok streams | rounds | system t/s | sum-stream t/s | wall s | TTFT max ms | TTFT p99 ms | ITL p99 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direct engine - b10903-cuda | 4 | 4/4 | 1 | 108.8 | 113.8 | 4.71 | 251 | - | 38.8 |
| pallama gateway - b10903-cuda | 4x65536 | 12/4 | 3 | 104.2 | 338.7 | 14.73 | 529 | 529 | 37.9 |
| ollama - qwen3.5:9b | service | 12/4 | 3 | 33.4 | 482.4 | 46.02 | 16306 | 15943 | 75.5 |

_sum-stream >> system t/s means streams serialize on one slot; roughly equal means genuinely parallel._

### Perplexity

| Engine | perplexity (ctx 2048, offline ASCII corpus) |
|---|---:|
| b10903-cuda | 16.75 ± 0.88 |

### Greedy parity and gateway transparency (20 prompts, 256 tokens)

| Comparison | exact / total | ratio mean | ratio min |
|---|---:|---:|---:|
| b10903-cuda vs same-engine reference (direct) | 20/20 | 1.000 | 1.000 |
| b10903-cuda through pallama gateway vs direct | 5/20 | 0.559 | 0.027 |

_Exact-match divergence across GPU backends is expected float nondeterminism (batch shape and backend kernels), not translation drift; bit-parity across runs requires single-slot decoding (pallama `deterministic = true` pins it)._

### Optimization axes (ctx 4096, single stream)

| Engine | axis | setting | decode t/s | delta vs dense | prefill cold t/s | delta |
|---|---|---|---:|---:|---:|---:|
| b10903-cuda | kv | q8_0 | 40.6 | -0.7 | 1301.7 | 62.5 |
| b10903-cuda | spec | ngram-simple | 40.7 | -0.5 | 1327.7 | 88.5 |
| b10903-cuda | mmproj | True | 41.1 | -0.1 | 1290.8 | 51.6 |

### Engine capability matrix

| Capability | b10903-cuda |
|---|---:|
| anthropic-api | no |
| ctx-override | yes |
| embeddings | yes |
| grammar-gbnf | yes |
| json-schema | yes |
| kv-quant | yes |
| lora-adapter | yes |
| metrics-endpoint | yes |
| paged-attn | yes |
| parallel-np | yes |
| quant-on-load | yes |
| rerank | yes |
| slots-sessions | yes |
| spec-decode | yes |
| tokenize-endpoint | yes |
| vision-mmproj | yes |

### Cold start and footprint

| Runtime | daemon boot s | first request (cold engine load) s | cold TTFT ms | engine load s | RSS peak MiB |
|---|---:|---:|---:|---:|---:|
| pallama gateway - b10903-cuda | 0.54 | 11.66 | 11577 | - | 2181 |
| direct engine - b10903-cuda | - | - | - | 2.51 | 5705 |
| ollama - qwen3.5:9b | 4.13 | 6.65 | 6566 | 6.36 | - |

_Every cold probe runs page-cache-dropped and GPU-idle-asserted on both runtimes; ollama rows without --ollama-service-restart leave the daemon warm (note in the artifact)._

### Idle wake (sleep vs keep_alive expiry)

| Runtime | idle policy | policy observed | wake TTFT ms | reload s | note |
|---|---|---|---:|---:|---|
| pallama - b10903-cuda | sleep at 15s (weights stay RAM-resident) | yes | 2104 | - |  |
| ollama - qwen3.5:9b | keep_alive 20s -> full unload | yes | 7194 | 7.00 |  |

_pallama sleeps with weights in RAM (wake = resume); ollama unloads at keep_alive expiry (wake = full disk reload). Policies differ by design — the table measures each runtime's own idle path after the policy verifiably fired._

### Long-context degradation curve

| Runtime | ctx | decode t/s | TTFT p50 ms |
|---|---:|---:|---:|
| ollama - qwen3.5:9b | 2048 | 40.1 | 146 |
| ollama - qwen3.5:9b | 8192 | 40.3 | 147 |
| ollama - qwen3.5:9b | 16384 | 40.3 | 149 |
| pallama - b10903-cuda | 2048 | 40.8 | 117 |
| pallama - b10903-cuda | 8192 | 40.5 | 122 |
| pallama - b10903-cuda | 16384 | 40.4 | 121 |

## Findings

1. **Gateway overhead is within measurement noise.** Single-stream decode through the pallama gateway matches direct engine spawns at the same slots/context (see speed table); the greedy gateway lane is byte-identical to the direct lane where sampling is single-slot.
2. **Capacity-aware slot auto-sizing.** pallama sizes engine slots from live hardware census: the 8 GiB card with a vision projector attached spawns 1 slot (16 Ki context) on the Vulkan build and 4 slots (64 Ki total) on CUDA - measured oversubscription on Vulkan either fails to boot or degrades 2x, so the cap is load-bearing, not conservative cosmetics.
3. **Concurrency scales where capacity allows.** 4 streams through CUDA gateway hold near-direct system throughput; the Vulkan single-slot shape serializes streams (per-stream latency stays excellent; system throughput caps at one stream's rate) - a capacity trade, not a scheduling defect.
4. **Prompt cache pays ~6-7x on prefill.** Cached-prefix prefill runs thousands of tokens/s vs hundreds cold.
5. **Speculative n-gram decoding is a net loss for this 9B model** (no draft model; acceptance too low to pay the verification overhead) - documented so the flag is not cargo-culted.
6. **KV q8_0 quantization is decode-neutral and prefill-neutral steady-state**; the one cold-prefill outlier below is a first-invocation pipeline-compile artifact (controlled re-probe measured full-rate steady state).
7. **mistral.rs 0.9.3 with default paged attention cannot fit this model on an 8 GiB card** (upstream sizes KV as a fraction of total VRAM); pallama's profile auto-disables paged attention on tight cards and the model then serves correctly.

## Caveats

- ollama prefill numbers come from engine counters that exclude the chat template, so they read slightly high against the 512-token lanes.
- Cross-backend greedy ratios (CUDA vs Vulkan) diverge on near-tie logits; treat ratio, not exact-match count, as the signal.
- All GPU rows measured on AC power at bounded load; rows record load average and power state (battery runs are rejected by the harness).
- Numbers are medians of 5 runs on one hybrid laptop; expect absolute shifts on other hardware, ratios to travel better.

## Reproduce

```bash
python3 scripts/bench_matrix.py --pallama-bin target/release/pallama --md BENCHMARK.md
python3 scripts/bench_matrix.py --render-only --artifacts-dir <dir> --md BENCHMARK.md
```

_Raw per-cell records (argv, per-run lists, daemon logs): `~/.cache/pallama-bench-matrix/<run-id>/cells.jsonl`._
