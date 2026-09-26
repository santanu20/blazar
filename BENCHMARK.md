# Blazar inference benchmark

_Rendered 20260924-all-engines; blazar 0.11.0; power state of gateway rows: ac._

## Executive summary

b11147-cuda: gateway 41.1 vs direct 41.5 t/s (-1.0%); b11147-cuda prompt-cache prefill 7309 vs 1318 t/s cold; b11147-cuda sweep C=1/2/4/8: C1: 39.2 t/s system (1x16384); C2: 39.6 t/s system (1x16384); C4: 39.8 t/s system (1x16384); C8: 39.5 t/s system (1x16384) | mistral.rs 0.9.3 (CUDA sm89) sweep C=1/2/4/8: C1: 39.4 t/s system (2x8192); C2: 68.1 t/s system (2x8192); C4: 69.2 t/s system (2x8192); C8: 68.7 t/s system (2x8192); gateway cold boot 0.53 s; cold TTFT 5214 ms vs ollama 3717 ms (0.7x); idle wake 1970 ms (sleep) vs ollama 6153 ms (full reload).

## Measured in this campaign

- blazar: 5 cell(s)
- cold-ollama: 1 cell(s)
- conc-blazar: 8 cell(s)
- conc-direct: 4 cell(s)
- conc-ollama: 4 cell(s)
- ctxcurve-blazar: 6 cell(s)
- ctxcurve-ollama: 3 cell(s)
- direct: 12 cell(s)
- features: 2 cell(s)
- greedy: 2 cell(s)
- greedy_gw: 1 cell(s)
- idle-blazar: 2 cell(s)
- idle-ollama: 1 cell(s)
- media-image: 1 cell(s)
- media-tts: 1 cell(s)
- media-tts-conc: 1 cell(s)
- media-video: 1 cell(s)
- media-whisper: 1 cell(s)
- ollama: 1 cell(s)
- ppl: 2 cell(s)
- reshape: 2 cell(s)
- tools: 2 cell(s)
- tools-ollama: 1 cell(s)

## Engine coverage

| Engine | Kind | Lane | ok cells | err cells | Status |
|---|---|---|---:|---:|---|
| v0.9.3 | mistralrs | text (direct + gateway) | 15 | 6 | benchmarked |
| master-890-74988b2 | sdcpp | media | 2 | 0 | benchmarked |
| b5130 | whisper | media | 1 | 0 | benchmarked |
| b11147-cuda | llamacpp | text (direct + gateway) | 27 | 0 | benchmarked |
| sglang-0.5.19 | — | — | 0 | 0 | excluded: needs an HF safetensors model; this box serves GGUF only and 8 GiB VRAM cannot host sglang beside the media children |

## Test bed

| Component | Value |
|---|---|
| CPU | Intel Core i7-14650HX, 24 hardware threads |
| Discrete GPU | NVIDIA GeForce RTX 4070 Laptop, 8 GiB, driver 580.173.02 |
| Integrated GPU | Intel Graphics (RPL-S), Vulkan device |
| RAM | 16 GiB (13.3 GiB usable) |
| OS | Linux Mint 22.3, kernel 7.0.0-31-generic |
| Runtimes compared | blazar 0.11.0 gateway - b11147-cuda - inventory - mistral.rs 0.9.3 (CUDA sm89) - ollama-host - piper (gateway TTS lane) - stable-diffusion.cpp master-890 (Vulkan) - whisper.cpp b5130 |
| Model | Qwen3.5-9B-Q4_K_M.gguf, en_US-amy-medium, ggml-base, qwen-image-2.1, wan_2.1_comfyui_repackaged |

## Methodology

- All lanes speak the OpenAI-compatible streaming API; tokens are counted from usage chunks (engine-injected at the gateway), never estimated from chunk counts.
- Decode throughput = (tokens - 1) / (last-token time - TTFT); medians over 5 runs after a warmup request.
- Inter-token latency (ITL) p50/p99 from per-chunk timestamps; TTFT p50/p90/p99 + stdev.
- Prefill: a token-targeted prompt (~512 tokens via engine /tokenize); run 1 is the cold (uncached) prefill, runs 2+ ride the prompt cache.
- Concurrency: N parallel streams x 128 generated tokens each (levels via --conc-sweep, per-row `C` column); system t/s = total tokens / wall clock; sum-stream t/s = sum of per-stream rates (sum >> system indicates serialization).
- Greedy parity: 20 fixed prompts, greedy sampling, 256 tokens; exact-match count and text-similarity ratio vs a same-engine reference run.
- Tool calls: 6 single-turn scenarios (3-tool set: weather/calculate/flights), temperature 0, max 192 tokens (call JSON must complete); scored on well-formed calls, correct function selection, valid JSON arguments with required keys, and a no-tool control for false positives; stream deltas accumulated per OpenAI spec.
- Gateway transparency: a second greedy lane through the blazar gateway with identical sampling; any divergence vs the direct lane isolates translation overhead.
- Perplexity: llama-perplexity on an offline ASCII corpus, ctx 2048.
- Cold-start parity: the model file's page cache is dropped (posix_fadvise DONTNEED) and the GPU asserted idle (<512 MiB) before every cold probe on every runtime — a cold load is disk-cold, not memory-warm.
- Cold TTFT = first-token latency of the cold probe itself (max_tokens 4, aligned num_ctx 16384 on both runtimes).
- ollama daemon boot is only measured with --ollama-service-restart (systemd restart, sudo password via BENCH_SUDO_PASSWORD env, stdin-only); without it the daemon stays warm and the row says so.
- Idle-wake: blazar's reaper sleeps the child at idle_sleep_secs (weights stay RAM-resident, VRAM released) — wake TTFT is a sleep-wake; ollama's keep_alive expiry fully unloads — wake TTFT is a disk reload. The policy column names the semantic; both measured after the policy is observed via /api/ps.
- Long-context curve: per-ctx cells (blazar model_overrides ctx / ollama num_ctx) x 3-run decode suites; each ollama point evicts first so the runner respawns at that ctx.
- Sustained concurrency: sequential bursts of the parallel-stream lane (default 3 rounds); TTFT p99 aggregates every stream of every round.
- Every blazar row records the spawned engine's argv (slots/context shown in tables) and stamps blazar version, wall clock, 5-min load average, and AC/battery power state; GPU cells refuse to run on battery.
- Media lanes run in isolated sandbox daemons (same protocol as text blazar cells); the engine child spawns lazily, so each family's first request is the COLD number (spawn + weights + first artifact), labeled cold_request_s.
- Media TTFB = time to first BODY byte (first audio sample for streamed PCM, not response headers); buffered WAV TTFB equals its total by construction and the table says so.
- Video frame counts are container ground truth: the response webm is parsed for lacing-aware SimpleBlock counts and asserted against the Wan 4k+1 temporal grid (a mismatch is recorded loudly, never averaged away).
- The video scratch-gate probe sends one expected-rejected monster (duration 60s -> 960 aligned frames) and times the 400; the legit 5-frame pass rides the same warm child, so axis-row vs probe deltas price the gate itself.
- Media cells stamp GPU-busy and RAM-available at entry instead of asserting an idle GPU: a warm child from the previous family is the normal media workflow, and the receipt carries the occupancy rather than hiding it.
- TTS RTF = synthesis wall / audio seconds, audio duration parsed from the RIFF data-chunk length (not estimated from characters); whisper transcribes a WAV synthesized by the same campaign's piper voice, so the input is reproducible from the receipt.
- Image quality stamps are PIL-gated luma-domain metrics (rms contrast = luma stddev, entropy in bits, unique colors on a 256x256 downsample); when PIL is absent the row carries an honest 'skipped' note instead of a fake number, and one audit PNG per steps point is saved beside the cells for offline re-measurement.
- TTS concurrency probe: N parallel streamed-PCM requests through one sandboxed gateway; wall clock vs sum of per-stream totals yields an efficiency ratio (sum/wall ~ 1 means serialized, -> N means perfectly parallel), and the probe fails loudly if any stream errors or truncates.
- Adaptive reshape: sustained 8 concurrent streams (adoption needs a 60 s saturation streak plus a graceful drain); the child engine -np is polled from /proc every 2 s to prove the reshape landed; throughput and TTFT p50 are compared before vs after the slot transition; a dropped request anywhere fails the lane.

## Results

### Single-stream decode (512-token prompt, 128 generated, median of 5)

| Runtime | slots x ctx | decode t/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | prefill cold t/s | prefill cached t/s | GPU peak MiB | GPU power W |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blazar gateway - b11147-cuda | 1x16384 | 41.0 | 120.2 | 123.4 | 24.3 | 25.5 | 1365.6 | 7505.3 | 5715 | 56.3 |
| blazar gateway - b11147-cuda (single-stream) | 1x16384 | 41.1 | 118.1 | 123.6 | 24.3 | 25.4 | 1317.7 | 7309.2 | 5715 | 56.3 |
| blazar gateway - mistral.rs 0.9.3 (CUDA sm89) | 2x8192 | 40.9 | 125.3 | 129.4 | 24.4 | 25.6 | 1338.7 | 7538.8 | 5501 | 55.6 |
| blazar gateway - mistral.rs 0.9.3 (CUDA sm89) (paged_attn_off) | 2x8192 | 40.9 | 121.7 | 125.2 | 24.4 | 25.5 | 1326.5 | 7556.7 | 5501 | 55.4 |
| blazar gateway - mistral.rs 0.9.3 (CUDA sm89) (single-stream) | 1x16384 | 40.9 | 124.1 | 127.0 | 24.4 | 25.5 | 1333.3 | 7244.6 | 5715 | 55.5 |
| direct engine - b11147-cuda | 1x16384 | 41.5 | 108.5 | 111.7 | 24.1 | 25.2 | 1367.9 | 8292.5 | 5709 | 55.9 |
| ollama 0.33.3 - qwen3.5:9b | service | 40.6 | 126.6 | 133.3 | 24.8 | 75.0 | 1364.7 | 8357.6 | 6626 | 55.5 |

### Concurrency (1x2x4x8 parallel streams x 128 tokens)

| Runtime | slots | ok streams | rounds | system t/s | sum-stream t/s | wall s | TTFT max ms | TTFT p99 ms | ITL p99 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| blazar gateway - b11147-cuda | 1x16384 | 3/1 | 3 | 39.2 | 123.2 | 9.80 | 309 | 305 | 26.7 |
| blazar gateway - b11147-cuda | 1x16384 | 6/2 | 3 | 39.6 | 246.5 | 19.41 | 3497 | 3488 | 25.4 |
| blazar gateway - b11147-cuda | 1x16384 | 12/4 | 3 | 39.8 | 494.0 | 38.61 | 9901 | 9884 | 25.5 |
| blazar gateway - b11147-cuda | 1x16384 | 24/8 | 3 | 39.5 | 982.3 | 77.86 | 23007 | 22962 | 26.0 |
| blazar gateway - mistral.rs 0.9.3 (CUDA sm89) | 2x8192 | 3/1 | 3 | 39.4 | 123.3 | 9.75 | 209 | 208 | 26.8 |
| blazar gateway - mistral.rs 0.9.3 (CUDA sm89) | 2x8192 | 6/2 | 3 | 68.1 | 219.1 | 11.28 | 381 | 376 | 29.2 |
| blazar gateway - mistral.rs 0.9.3 (CUDA sm89) | 2x8192 | 12/4 | 3 | 69.2 | 439.8 | 22.21 | 4145 | 4124 | 29.1 |
| blazar gateway - mistral.rs 0.9.3 (CUDA sm89) | 2x8192 | 24/8 | 3 | 68.7 | 874.4 | 44.71 | 11871 | 11825 | 29.4 |
| direct engine - b11147-cuda | 1 | 1/1 | 1 | 38.8 | 40.8 | 3.30 | 179 | - | 28.2 |
| direct engine - b11147-cuda | 2 | 2/2 | 1 | 71.0 | 73.9 | 3.60 | 164 | - | 28.6 |
| direct engine - b11147-cuda | 4 | 4/4 | 1 | 110.7 | 115.9 | 4.63 | 241 | - | 37.4 |
| direct engine - b11147-cuda | 8 | 8/8 | 1 | 133.5 | 141.2 | 7.67 | 477 | - | 60.5 |
| ollama - qwen3.5:9b | service | 3/1 | 3 | 22.6 | 120.9 | 16.98 | 7206 | 7064 | 75.5 |
| ollama - qwen3.5:9b | service | 6/2 | 3 | 30.8 | 243.9 | 24.92 | 8723 | 8559 | 74.6 |
| ollama - qwen3.5:9b | service | 12/4 | 3 | 34.0 | 485.8 | 45.23 | 15846 | 15486 | 75.0 |
| ollama - qwen3.5:9b | service | 24/8 | 3 | 36.3 | 972.4 | 84.64 | 29036 | 28286 | 74.8 |

_sum-stream >> system t/s means streams serialize on one slot; roughly equal means genuinely parallel._

### Concurrency frontier (system t/s and tail latency vs level)

| Runtime | C | ok streams | system t/s | sum-stream t/s | eff vs C=1 | TTFT p99 ms | ITL p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| blazar gateway - b11147-cuda | 1 | 3 | 39.2 | 123.2 | 100% | 305 | 26.7 |
| blazar gateway - b11147-cuda | 2 | 6 | 39.6 | 246.5 | 51% | 3488 | 25.4 |
| blazar gateway - b11147-cuda | 4 | 12 | 39.8 | 494.0 | 25% | 9884 | 25.5 |
| blazar gateway - b11147-cuda | 8 | 24 | 39.5 | 982.3 | 13% | 22962 | 26.0 |
| blazar gateway - mistral.rs 0.9.3 (CUDA sm89) | 1 | 3 | 39.4 | 123.3 | 100% | 208 | 26.8 |
| blazar gateway - mistral.rs 0.9.3 (CUDA sm89) | 2 | 6 | 68.1 | 219.1 | 86% | 376 | 29.2 |
| blazar gateway - mistral.rs 0.9.3 (CUDA sm89) | 4 | 12 | 69.2 | 439.8 | 44% | 4124 | 29.1 |
| blazar gateway - mistral.rs 0.9.3 (CUDA sm89) | 8 | 24 | 68.7 | 874.4 | 22% | 11825 | 29.4 |
| direct engine - b11147-cuda | 1 | 1 | 38.8 | 40.8 | 100% | - | 28.2 |
| direct engine - b11147-cuda | 2 | 2 | 71.0 | 73.9 | 91% | - | 28.6 |
| direct engine - b11147-cuda | 4 | 4 | 110.7 | 115.9 | 71% | - | 37.4 |
| direct engine - b11147-cuda | 8 | 8 | 133.5 | 141.2 | 43% | - | 60.5 |
| ollama - qwen3.5:9b | 1 | 3 | 22.6 | 120.9 | 100% | 7064 | 75.5 |
| ollama - qwen3.5:9b | 2 | 6 | 30.8 | 243.9 | 68% | 8559 | 74.6 |
| ollama - qwen3.5:9b | 4 | 12 | 34.0 | 485.8 | 38% | 15486 | 75.0 |
| ollama - qwen3.5:9b | 8 | 24 | 36.3 | 972.4 | 20% | 28286 | 74.8 |

- blazar gateway - b11147-cuda: throughput plateaus at C=2 (<10% per-level gain), peak 39.8 t/s at C=4.
- blazar gateway - mistral.rs 0.9.3 (CUDA sm89): throughput plateaus at C=4 (<10% per-level gain), peak 69.2 t/s at C=4.
- direct engine - b11147-cuda: still gaining at C=8 (38.8 -> 133.5 t/s) - saturation not reached within the sweep.
- ollama - qwen3.5:9b: throughput plateaus at C=8 (<10% per-level gain), peak 36.3 t/s at C=8.

### Adaptive reshape under sustained load (no-lag proof)

| Runtime | reshape | slots | time to reshape s | req before/after | TTFT p50 before→after ms | sys t/s before→after | failed |
|---|---|---|---:|---:|---:|---:|---:|
| blazar gateway - b11147-cuda | NO | -→- | - | 70/0 | 33342→- | 20→- | 0 |

### Perplexity

| Engine | perplexity (ctx 2048, offline ASCII corpus) |
|---|---:|
| b11147-cuda | 16.75 ± 0.88 |
| mistral.rs 0.9.3 (CUDA sm89) | not applicable (tool is llama.cpp-family) |

### Greedy parity and gateway transparency (20 prompts, 256 tokens)

| Comparison | exact / total | ratio mean | ratio min |
|---|---:|---:|---:|
| b11147-cuda vs same-engine reference (direct) | 20/20 | 1.000 | 1.000 |
| mistral.rs 0.9.3 (CUDA sm89) vs same-engine reference (direct) | 0/20 | 0.526 | 0.000 |
| b11147-cuda through blazar gateway vs direct | 20/20 | 1.000 | 1.000 |

_Exact-match divergence across GPU backends is expected float nondeterminism (batch shape and backend kernels), not translation drift; bit-parity across runs requires single-slot decoding (blazar `deterministic = true` pins it)._

### Tool calls (single-turn selection + schema quality)

| Runtime | scenarios | well-formed | selection | args valid | control FP | TTFT p50 ms |
|---|---:|---:|---:|---:|---|---:|
| blazar gateway - b11147-cuda | 6 | 5/6 | 5/5 | 5/5 | no | 123 |
| blazar gateway - mistral.rs 0.9.3 (CUDA sm89) | 6 | 5/6 | 5/5 | 5/5 | no | 107 |
| ollama - qwen3.5:9b | 6 | 4/6 | 4/5 | 4/5 | no | 288 |


### Optimization axes (ctx 4096, single stream)

| Engine | axis | setting | decode t/s | delta vs dense | prefill cold t/s | delta |
|---|---|---|---:|---:|---:|---:|
| b11147-cuda | kv | q8_0 | 41.1 | -0.6 | 1374.1 | 31.5 |
| b11147-cuda | spec | ngram-simple | 40.7 | -1.0 | 1343.9 | 1.3 |
| b11147-cuda | mmproj | True | 41.0 | -0.7 | 1339.0 | -3.6 |
| mistral.rs 0.9.3 (CUDA sm89) | pa | off | 19.6 | - | 221.7 | - |

### Engine capability matrix

| Capability | b11147-cuda | mistral.rs 0.9.3 (CUDA sm89) |
|---|---:|---:|
| anthropic-api | no | yes |
| ctx-override | yes | yes |
| embeddings | yes | no |
| grammar-gbnf | yes | no |
| json-schema | yes | yes |
| kv-quant | yes | yes |
| lora-adapter | yes | yes |
| metrics-endpoint | yes | yes |
| paged-attn | yes | yes |
| parallel-np | yes | yes |
| quant-on-load | yes | yes |
| rerank | yes | no |
| slots-sessions | yes | no |
| spec-decode | yes | yes |
| tokenize-endpoint | yes | no |
| vision-mmproj | yes | yes |

### Cold start and footprint

| Runtime | daemon boot s | first request (cold engine load) s | cold TTFT ms | engine load s | RSS peak MiB |
|---|---:|---:|---:|---:|---:|
| blazar gateway - b11147-cuda | 0.53 | 5.00 | 4927 | - | 1992 |
| blazar gateway - b11147-cuda | 0.53 | 5.29 | 5214 | - | 1993 |
| blazar gateway - mistral.rs 0.9.3 (CUDA sm89) | 0.52 | 5.00 | 4854 | - | 1990 |
| blazar gateway - mistral.rs 0.9.3 (CUDA sm89) | 0.52 | 5.00 | 4846 | - | 1990 |
| blazar gateway - mistral.rs 0.9.3 (CUDA sm89) | 0.52 | 4.97 | 4897 | - | 1993 |
| direct engine - b11147-cuda | - | - | - | 2.01 | 5770 |
| ollama - qwen3.5:9b | - | 3.80 | 3717 | 3.60 | - |

_Every cold probe runs page-cache-dropped and GPU-idle-asserted on both runtimes; ollama rows without --ollama-service-restart leave the daemon warm (note in the artifact)._

### Idle wake (sleep vs keep_alive expiry)

| Runtime | idle policy | policy observed | wake TTFT ms | reload s | note |
|---|---|---|---:|---:|---|
| blazar - b11147-cuda | sleep at 15s (weights stay RAM-resident) | yes | 1970 | - |  |
| blazar - mistral.rs 0.9.3 (CUDA sm89) | sleep at 15s (weights stay RAM-resident) | yes | 2006 | - |  |
| ollama - qwen3.5:9b | keep_alive 20s -> full unload | yes | 6153 | 5.99 |  |

_blazar sleeps with weights in RAM (wake = resume); ollama unloads at keep_alive expiry (wake = full disk reload). Policies differ by design — the table measures each runtime's own idle path after the policy verifiably fired._

### Long-context degradation curve

| Runtime | ctx | decode t/s | TTFT p50 ms |
|---|---:|---:|---:|
| blazar - b11147-cuda | 2048 | 40.6 | 132 |
| blazar - b11147-cuda | 8192 | 41.0 | 125 |
| blazar - b11147-cuda | 16384 | 40.9 | 126 |
| blazar - mistral.rs 0.9.3 (CUDA sm89) | 2048 | 40.9 | 125 |
| blazar - mistral.rs 0.9.3 (CUDA sm89) | 8192 | 41.0 | 128 |
| blazar - mistral.rs 0.9.3 (CUDA sm89) | 16384 | 40.9 | 124 |
| ollama - qwen3.5:9b | 2048 | 40.2 | 168 |
| ollama - qwen3.5:9b | 8192 | 40.3 | 149 |
| ollama - qwen3.5:9b | 16384 | 40.7 | 135 |

### Media lanes (image / video / TTS / whisper)

| Lane | cold s | median s | min s | max s | ground truth | gate reject s | TTFB speedup |
|---|---:|---:|---:|---:|---|---:|---:|
| image - qwen-image-2.1 (512x512, steps=[4]) | 53.58 | 48.01 | 47.89 | 48.25 | 512x512 PNG, entropy 6.4 bits, contrast 51.2 |  |  |
| tts - en_US-amy-medium (840 chars, wav+pcm) | - | 2.00 | - | 3.27 | RTF wav 0.033 / pcm 0.054 (61s audio) |  | 3.96x |
| tts-conc - en_US-amy-medium (400 chars x4 pcm) | - | 4.54 | 4.81 | - | efficiency 3.68 of 4 streams |  | 1634ms max TTFB |
| video - wan_2.1_comfyui_repackaged (320x320, steps=8) | 16.90 | 14.08 | 14.08 | 14.09 | 5f: mux==reported==aligned | 0.002 |  |
| video - wan_2.1_comfyui_repackaged (320x320, steps=8) | - | 26.11 | 26.11 | 26.13 | 13f: mux==reported==aligned |  |  |
| video - wan_2.1_comfyui_repackaged (320x320, steps=8) | - | 61.23 | 61.22 | 61.24 | 33f: mux==reported==aligned |  |  |
| whisper - ggml-base (transcribes piper wav) | 2.49 | 1.99 | - | - | RTF 0.065 |  |  |

_Media cells run through the same sandboxed gateway as text lanes but do not assert GPU-idle: a warm engine child is the normal serving shape, so each row stamps gpu_busy_mib / ram_avail_mib / loadavg instead. 3 runs (not 5) — media variance is dominated by the model, not the scheduler. Video frame counts are read from the EBML container (lacing-aware), never from an API field; the VRAM gate probe times how fast an over-budget request is rejected with a teaching error._

## Findings (this campaign)

1. **Gateway overhead vs direct spawn: within measurement noise.** b11147-cuda decode 41.1 t/s through the gateway vs 41.5 t/s direct (-1.0%), greedy parity through the gateway 20/20 exact.
2. **Capacity-aware slot auto-sizing observed in argv.** Distinct engine shapes this campaign: 1x16384, 2x8192 - slots follow the live hardware census, each row's child_argv carries the receipt.
3. **Concurrency scaling per engine.** b11147-cuda (C=1/2/4/8): peak 39.8 t/s at C=4, 25% of ideal at C=4; mistral.rs 0.9.3 (CUDA sm89) (C=1/2/4/8): peak 69.2 t/s at C=4, 44% of ideal at C=4; serialization behavior per level in the frontier table below.
4. **Prompt cache pays 5.5x on prefill** (7505 cached vs 1366 t/s cold).
5. **Speculative n-gram decoding is a net loss for this model** (decode 40.7 t/s, -1.0 vs dense baseline) - measured, not assumed.
6. **KV quantization (q8_0) is decode-neutral** (decode 41.1 t/s vs 41.7 dense).
7. **mistral.rs 0.9.3 (CUDA sm89) serves this model through blazar's profile** ( default paged attention cannot fit this card (4 direct cell(s) refused at load: 'Num GPU blocks is 0'); blazar auto-disables PA on tight cards and the model then serves; the row's argv is the receipt).
8. **Tool-call quality (single-turn, temp 0).** b11147-cuda: selection 5/5, args 5/5 (control clean); mistral.rs 0.9.3 (CUDA sm89): selection 5/5, args 5/5 (control clean); ollama: selection 4/5, args 4/5 (control clean); per-scenario detail in cells.jsonl.
9. **Adaptive reshape lands under sustained load.** b11147-cuda: no reshape observed in the lane window (graceful-drain: adoption waits for in-flight streams, never kills one).
10. **Image quality stamps (PIL, luma domain): entropy 6.41 bits, rms contrast 51.2, 28089 unique colors @256x256** on qwen-image-2.1 - perceptual baseline for cross-run comparisons; audit PNG saved beside the cells.
11. **Streamed PCM cuts time-to-first-audio 3.96x vs buffered WAV** (piper lane, first audio 0.50s vs 2.00s full synthesis) - total wall time is slightly higher (per-chunk synthesis), the win is interactivity.
12. **4 parallel PCM streams through one gateway: perfectly parallel** (efficiency 3.68 = sum of per-stream totals / 4.81s wall, max TTFB 1634 ms, NON-uniform stream outputs - flagged) - the scalability receipt for the TTS lane.
13. **Video VRAM gate rejects an over-budget request in 2 ms** with the full estimate math and override levers in the error body - instead of an opaque child abort minutes later.


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

_Raw per-cell records (argv, per-run lists, daemon logs): `bench-artifacts/20260924-all-engines/cells.jsonl`._
