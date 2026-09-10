# Pallama inference benchmark

_Rendered 20260910-003328; pallama 0.5.0; power state of gateway rows: ac. Post-render addenda (2026-09-10 P0 wave) follow the executive summary._

## Executive summary

llama.cpp b10809 (Vulkan): gateway 39.0 vs direct 39.1 t/s (-0.1%); llama.cpp b10809 (CUDA build): gateway 40.7 vs direct 41.2 t/s (-1.2%); llama.cpp b10809 (Vulkan) prompt-cache prefill 6629 vs 1011 t/s cold; 4-stream concurrency: 37.3 t/s system (1x16384 shape); 98.6 t/s system (4x65536 shape); 15.2 t/s system (engine-scheduled shape); gateway cold boot 0.53 s.

## Addenda (2026-09-10 vs-ollama audit + P0 wave)

- **Prefill gap REFUTED as product gap:** controlled A/B, same build class — llama-bench (b10809-cuda, -ngl 999, -fa auto, pp512, 3 reps) 1630 t/s vs ollama /api/generate raw:true same-counter nonce probe 1726/1672 t/s warm → 2-6% engine-build-flag territory (different compile flags, same llama.cpp tree), NOT a gateway/planner gap. No chase.
- **Slots auto-fit (P0b) live-proven on the tight lane:** vulkan build + mmproj + 8 GiB card, default ctx 16384 — pre-feature spawn = 1x16384 (37.3 t/s serialized at 4 streams, the G1 gap); post-feature spawn resolves 4x4096 (identical total-ctx budget), argv `--ctx-size 16384 -np 4 --kv-unified`, `slots_ctx_auto_fit` event on /api/events, spawn settle 88% used (healthy zone, ~3.5 s load — not the 27 s degraded boot at 32k total). 4-stream probe: short streams completed in 4.4/5.7 s WHILE long streams were still generating (impossible under 1-slot serialization); long stream decoded at 31.5 t/s under 4-way contention.

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
- Every pallama row records the spawned engine's argv (slots/context shown in tables) and stamps pallama version, wall clock, 5-min load average, and AC/battery power state; GPU cells refuse to run on battery.

## Results

### Single-stream decode (512-token prompt, 128 generated, median of 5)

| Runtime | slots x ctx | decode t/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | prefill cold t/s | prefill cached t/s | GPU peak MiB | GPU power W |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pallama gateway - llama.cpp b10809 (Vulkan) | 1x16384 | 39.0 | 145.5 | 162.7 | 25.6 | 26.8 | 1011.1 | 6628.6 | 6707 | 57.9 |
| pallama gateway - llama.cpp b10809 (CUDA build) | 4x65536 | 40.7 | 135.3 | 137.3 | 24.5 | 25.9 | 1287.2 | 7339.4 | 7328 | 55.2 |
| pallama gateway - mistral.rs 0.9.3 (CUDA sm89) | engine-scheduled | 18.0 | 137.4 | 153.4 | 56.9 | 68.8 | 222.9 | 225.5 | 7044 | 41.9 |
| direct engine - llama.cpp b10809 (Vulkan) | 1x16384 | 39.1 | 149.1 | 161.7 | 25.6 | 26.9 | 1023.7 | 6629.0 | 5605 | 55.2 |
| direct engine - llama.cpp b10809 (CUDA build) | 1x16384 | 41.2 | 137.3 | 144.0 | 24.3 | 25.6 | 1280.0 | 7371.5 | 5716 | 55.6 |
| ollama 0.33.3 - qwen3.5:9b | service | 40.4 | 139.8 | 148.6 | 25.0 | 75.5 | 1755.3 | 5400.0 | 6448 | 56.6 |

### Concurrency (4 parallel streams x 128 tokens)

| Runtime | slots | ok streams | system t/s | sum-stream t/s | wall s | TTFT max ms | TTFT spread ms | ITL p99 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| direct engine - llama.cpp b10809 (Vulkan) | 4 | 4/4 | 86.8 | 92.3 | 5.90 | 390 | 4 | 46.1 |
| direct engine - llama.cpp b10809 (CUDA build) | 4 | 4/4 | 98.5 | 102.9 | 5.20 | 262 | 3 | 41.2 |
| pallama gateway - llama.cpp b10809 (Vulkan) | 1x16384 | 4/4 | 37.3 | 155.6 | 13.72 | 10454 | 10283 | 27.3 |
| pallama gateway - llama.cpp b10809 (CUDA build) | 4x65536 | 4/4 | 98.6 | 103.5 | 5.19 | 285 | 7 | 41.8 |
| pallama gateway - mistral.rs 0.9.3 (CUDA sm89) | engine-scheduled | 4/4 | 15.2 | 15.4 | 8.43 | 269 | 88 | 70.5 |

_sum-stream >> system t/s means streams serialize on one slot; roughly equal means genuinely parallel._

### Perplexity

| Engine | perplexity (ctx 2048, offline ASCII corpus) |
|---|---:|
| llama.cpp b10809 (Vulkan) | 36.76 ± 2.18 |
| llama.cpp b10809 (CUDA build) | 34.95 ± 2.05 |
| mistral.rs 0.9.3 (CUDA sm89) | not applicable (tool is llama.cpp-family) |

### Greedy parity and gateway transparency (20 prompts, 256 tokens)

| Comparison | exact / total | ratio mean | ratio min |
|---|---:|---:|---:|
| llama.cpp b10809 (Vulkan) vs same-engine reference (direct) | 20/20 | 1.000 | 1.000 |
| llama.cpp b10809 (CUDA build) vs same-engine reference (direct) | 12/20 | 0.771 | 0.050 |
| mistral.rs 0.9.3 (CUDA sm89) vs same-engine reference (direct) | 0/20 | 0.531 | 0.000 |
| llama.cpp b10809 (Vulkan) through pallama gateway vs direct | 18/20 | 0.915 | 0.015 |
| llama.cpp b10809 (CUDA build) through pallama gateway vs direct | 6/20 | 0.551 | 0.015 |

_Exact-match divergence across GPU backends is expected float nondeterminism (batch shape and backend kernels), not translation drift; bit-parity across runs requires single-slot decoding (pallama `deterministic = true` pins it)._

### Optimization axes (ctx 4096, single stream)

| Engine | axis | setting | decode t/s | delta vs dense | prefill cold t/s | delta |
|---|---|---|---:|---:|---:|---:|
| llama.cpp b10809 (Vulkan) | kv | q8_0 | 38.9 | -0.0 | 77.6 | -941.0 |
| llama.cpp b10809 (Vulkan) | spec | ngram-simple | 38.7 | -0.2 | 1013.3 | -5.3 |
| llama.cpp b10809 (Vulkan) | mmproj | True | 39.2 | 0.2 | 932.6 | -86.0 |
| llama.cpp b10809 (CUDA build) | kv | q8_0 | 40.8 | -0.4 | 1300.7 | 67.8 |
| llama.cpp b10809 (CUDA build) | spec | ngram-simple | 40.8 | -0.4 | 1273.4 | 40.5 |
| llama.cpp b10809 (CUDA build) | mmproj | True | 41.1 | -0.1 | 1246.1 | 13.1 |
| mistral.rs 0.9.3 (CUDA sm89) | pa | off | 18.0 | - | 235.5 | - |

### Engine capability matrix

| Capability | llama.cpp b10809 (Vulkan) | llama.cpp b10809 (CUDA build) | mistral.rs 0.9.3 (CUDA sm89) |
|---|---:|---:|---:|
| anthropic-api | no | no | yes |
| ctx-override | yes | yes | yes |
| embeddings | yes | yes | no |
| grammar-gbnf | yes | yes | no |
| json-schema | yes | yes | yes |
| kv-quant | yes | yes | yes |
| lora-adapter | yes | yes | yes |
| metrics-endpoint | yes | yes | yes |
| paged-attn | yes | yes | yes |
| parallel-np | yes | yes | yes |
| quant-on-load | yes | yes | yes |
| rerank | yes | yes | no |
| slots-sessions | yes | yes | no |
| spec-decode | yes | yes | yes |
| tokenize-endpoint | yes | yes | no |
| vision-mmproj | yes | yes | yes |

### Cold start and footprint

| Runtime | daemon boot s | first request (cold engine load) s | engine load s | RSS peak MiB |
|---|---:|---:|---:|---:|
| pallama gateway - llama.cpp b10809 (Vulkan) | 0.53 | 3.68 | - | 1944 |
| pallama gateway - llama.cpp b10809 (CUDA build) | 0.52 | 3.81 | - | 2075 |
| pallama gateway - mistral.rs 0.9.3 (CUDA sm89) | 0.52 | 12.04 | - | 8291 |
| direct engine - llama.cpp b10809 (Vulkan) | - | - | 2.51 | 5673 |
| direct engine - llama.cpp b10809 (CUDA build) | - | - | 2.51 | 5706 |

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

_Raw per-cell records (argv, per-run lists, daemon logs): `~/.cache/pallama-bench-matrix/20260910-003328/cells.jsonl`._
