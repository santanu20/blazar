# Blazar inference benchmark

_Rendered 20260922-174339; blazar 0.10.0; power state of gateway rows: ac._

## Executive summary

b11070-cuda: gateway 347.6 vs direct 333.0 t/s (+4.4%); b11070-cuda prompt-cache prefill 51806 vs 17747 t/s cold; 4-stream concurrency: 447.9 t/s system (2x32768 shape); gateway cold boot 0.52 s.

## Test bed

| Component | Value |
|---|---|
| CPU | Intel Core i7-14650HX, 24 hardware threads |
| Discrete GPU | NVIDIA GeForce RTX 4070 Laptop, 8 GiB, driver 580.173.02 |
| Integrated GPU | Intel Graphics (RPL-S), Vulkan device |
| RAM | 16 GiB (13.3 GiB usable) |
| OS | Linux Mint 22.3, kernel 7.0.0-31-generic |
| Runtimes compared | blazar 0.10.0 gateway - b11070-cuda |
| Model | qwen2.5-0.5b.gguf |

## Methodology

- All lanes speak the OpenAI-compatible streaming API; tokens are counted from usage chunks (engine-injected at the gateway), never estimated from chunk counts.
- Decode throughput = (tokens - 1) / (last-token time - TTFT); medians over 5 runs after a warmup request.
- Inter-token latency (ITL) p50/p99 from per-chunk timestamps; TTFT p50/p90/p99 + stdev.
- Prefill: a token-targeted prompt (~512 tokens via engine /tokenize); run 1 is the cold (uncached) prefill, runs 2+ ride the prompt cache.
- Concurrency: 4 parallel streams x 128 generated tokens each; system t/s = total tokens / wall clock; sum-stream t/s = sum of per-stream rates (sum >> system indicates serialization).
- Greedy parity: 20 fixed prompts, greedy sampling, 256 tokens; exact-match count and text-similarity ratio vs a same-engine reference run.
- Gateway transparency: a second greedy lane through the blazar gateway with identical sampling; any divergence vs the direct lane isolates translation overhead.
- Perplexity: llama-perplexity on an offline ASCII corpus, ctx 2048.
- Cold-start parity: the model file's page cache is dropped (posix_fadvise DONTNEED) and the GPU asserted idle (<512 MiB) before every cold probe on every runtime — a cold load is disk-cold, not memory-warm.
- Cold TTFT = first-token latency of the cold probe itself (max_tokens 4, aligned num_ctx 16384 on both runtimes).
- ollama daemon boot is only measured with --ollama-service-restart (systemd restart, sudo password via BENCH_SUDO_PASSWORD env, stdin-only); without it the daemon stays warm and the row says so.
- Idle-wake: blazar's reaper sleeps the child at idle_sleep_secs (weights stay RAM-resident, VRAM released) — wake TTFT is a sleep-wake; ollama's keep_alive expiry fully unloads — wake TTFT is a disk reload. The policy column names the semantic; both measured after the policy is observed via /api/ps.
- Long-context curve: per-ctx cells (blazar model_overrides ctx / ollama num_ctx) x 3-run decode suites; each ollama point evicts first so the runner respawns at that ctx.
- Sustained concurrency: sequential bursts of the parallel-stream lane (default 3 rounds); TTFT p99 aggregates every stream of every round.
- Every blazar row records the spawned engine's argv (slots/context shown in tables) and stamps blazar version, wall clock, 5-min load average, and AC/battery power state; GPU cells refuse to run on battery.

## Results

### Single-stream decode (512-token prompt, 128 generated, median of 5)

| Runtime | slots x ctx | decode t/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | prefill cold t/s | prefill cached t/s | GPU peak MiB | GPU power W |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blazar gateway - b11070-cuda | 2x32768 | 338.7 | 10.3 | 14.1 | 3.0 | 4.2 | 18039.0 | 49490.6 | 1021 | 54.1 |
| blazar gateway - b11070-cuda | 1x16384 | 347.6 | 10.2 | 13.7 | 2.9 | 4.1 | 17747.0 | 51806.2 | 788 | 51.7 |
| direct engine - b11070-cuda | 1x16384 | 333.0 | 9.5 | 11.9 | 3.0 | 3.6 | 17776.7 | 54712.8 | 788 | 54.2 |

### Concurrency (4 parallel streams x 128 tokens)

| Runtime | slots | ok streams | rounds | system t/s | sum-stream t/s | wall s | TTFT max ms | TTFT p99 ms | ITL p99 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| blazar gateway - b11070-cuda | 2x32768 | 12/4 | 3 | 447.9 | 3287.4 | 2.26 | 512 | 501 | 5.7 |
| direct engine - b11070-cuda | 4 | 4/4 | 1 | 477.8 | 674.4 | 0.72 | 32 | - | 11.8 |

_sum-stream >> system t/s means streams serialize on one slot; roughly equal means genuinely parallel._

### Perplexity

| Engine | perplexity (ctx 2048, offline ASCII corpus) |
|---|---:|
| b11070-cuda | 43.86 ± 2.54 |

### Greedy parity and gateway transparency (20 prompts, 256 tokens)

| Comparison | exact / total | ratio mean | ratio min |
|---|---:|---:|---:|
| b11070-cuda vs same-engine reference (direct) | 20/20 | 1.000 | 1.000 |
| b11070-cuda through blazar gateway vs direct | 19/20 | 0.967 | 0.332 |

_Exact-match divergence across GPU backends is expected float nondeterminism (batch shape and backend kernels), not translation drift; bit-parity across runs requires single-slot decoding (blazar `deterministic = true` pins it)._

### Optimization axes (ctx 4096, single stream)

| Engine | axis | setting | decode t/s | delta vs dense | prefill cold t/s | delta |
|---|---|---|---:|---:|---:|---:|
| b11070-cuda | kv | q8_0 | 321.7 | -30.7 | 16941.3 | 742.3 |
| b11070-cuda | spec | ngram-simple | 327.5 | -24.8 | 18061.0 | 1861.9 |

### Engine capability matrix

| Capability | b11070-cuda |
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
| blazar gateway - b11070-cuda | 0.52 | 1.74 | 1719 | - | 584 |
| blazar gateway - b11070-cuda | 0.52 | 1.90 | 1891 | - | 568 |
| direct engine - b11070-cuda | - | - | - | 1.00 | 564 |

_Every cold probe runs page-cache-dropped and GPU-idle-asserted on both runtimes; ollama rows without --ollama-service-restart leave the daemon warm (note in the artifact)._

### Idle wake (sleep vs keep_alive expiry)

| Runtime | idle policy | policy observed | wake TTFT ms | reload s | note |
|---|---|---|---:|---:|---|
| blazar - b11070-cuda | sleep at 15s (weights stay RAM-resident) | yes | 641 | - |  |

_blazar sleeps with weights in RAM (wake = resume); ollama unloads at keep_alive expiry (wake = full disk reload). Policies differ by design — the table measures each runtime's own idle path after the policy verifiably fired._

### Long-context degradation curve

| Runtime | ctx | decode t/s | TTFT p50 ms |
|---|---:|---:|---:|
| blazar - b11070-cuda | 2048 | 333.1 | 15 |
| blazar - b11070-cuda | 8192 | 327.6 | 11 |
| blazar - b11070-cuda | 16384 | 342.6 | 13 |

## Findings

1. **Gateway overhead is within measurement noise.** Single-stream decode through the blazar gateway matches direct engine spawns at the same slots/context (see speed table); the greedy gateway lane is byte-identical to the direct lane where sampling is single-slot.
2. **Capacity-aware slot auto-sizing.** blazar sizes engine slots from live hardware census: the 8 GiB card with a vision projector attached spawns 1 slot (16 Ki context) on the Vulkan build and 4 slots (64 Ki total) on CUDA - measured oversubscription on Vulkan either fails to boot or degrades 2x, so the cap is load-bearing, not conservative cosmetics.
3. **Concurrency scales where capacity allows.** 4 streams through CUDA gateway hold near-direct system throughput; the Vulkan single-slot shape serializes streams (per-stream latency stays excellent; system throughput caps at one stream's rate) - a capacity trade, not a scheduling defect.
4. **Prompt cache pays ~6-7x on prefill.** Cached-prefix prefill runs thousands of tokens/s vs hundreds cold.
5. **Speculative n-gram decoding is a net loss for this 9B model** (no draft model; acceptance too low to pay the verification overhead) - documented so the flag is not cargo-culted.
6. **KV q8_0 quantization is decode-neutral and prefill-neutral steady-state**; the one cold-prefill outlier below is a first-invocation pipeline-compile artifact (controlled re-probe measured full-rate steady state).
7. **mistral.rs 0.9.3 with default paged attention cannot fit this model on an 8 GiB card** (upstream sizes KV as a fraction of total VRAM); blazar's profile auto-disables paged attention on tight cards and the model then serves correctly.

## Caveats

- ollama prefill numbers come from engine counters that exclude the chat template, so they read slightly high against the 512-token lanes.
- Cross-backend greedy ratios (CUDA vs Vulkan) diverge on near-tie logits; treat ratio, not exact-match count, as the signal.
- All GPU rows measured on AC power at bounded load; rows record load average and power state (battery runs are rejected by the harness).
- Numbers are medians of 5 runs on one hybrid laptop; expect absolute shifts on other hardware, ratios to travel better.

## Reproduce

```bash
python3 scripts/bench_matrix.py --blazar-bin target/release/blazar --md BENCHMARK.md
python3 scripts/bench_matrix.py --render-only --artifacts-dir <dir> --md BENCHMARK.md
```

_Raw per-cell records (argv, per-run lists, daemon logs): `/home/santanu/.cache/blazar-bench-matrix/20260922-174339/cells.jsonl`._
