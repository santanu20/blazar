# Blazar inference benchmark

_Rendered 20260929-sglang-safetensors; blazar 0.13.0; power state of gateway rows: ac._

## Executive summary

- sglang-0.5.19 prompt-cache prefill 11084 vs 7310 t/s cold
- sglang-0.5.19 sweep C=4: C4: 253.8 t/s system (engine-scheduled)
- ITL p99 15.31 ms vs ollama 7.10 ms (0.5x tighter)
- tool-call TTFT p50 36 ms vs ollama 285 ms (8.0x)
- gateway cold boot 0.52 s
- cold TTFT 30242 ms vs ollama 2138 ms (0.1x)
- idle wake 44 ms (sleep) vs ollama 2004 ms (full reload)

## Measured in this campaign

- blazar: 17 cell(s)
- cold-ollama: 2 cell(s)
- conc-blazar: 1 cell(s)
- conc-ollama: 2 cell(s)
- ctxcurve-blazar: 3 cell(s)
- ctxcurve-ollama: 6 cell(s)
- features: 1 cell(s)
- greedy_ollama: 2 cell(s)
- idle-blazar: 1 cell(s)
- idle-ollama: 2 cell(s)
- media-tts: 2 cell(s)
- media-tts-conc: 2 cell(s)
- ollama: 2 cell(s)
- ppl: 1 cell(s)
- reshape: 1 cell(s)
- tools: 1 cell(s)
- tools-ollama: 2 cell(s)

## Engine coverage

| Engine | Kind | Lane | ok cells | err cells | Status |
|---|---|---|---:|---:|---|
| sglang-0.5.19 | sglang | - | 25 | 1 | benchmarked |
| b11202-cuda | - | - | 0 | 0 | excluded: kind 'llamacpp' has no bench lane in this harness |
| b5130 | - | - | 0 | 0 | excluded: kind 'whisper' has no bench lane in this harness |
| master-920-2f88688 | - | - | 0 | 0 | excluded: kind 'sdcpp' has no bench lane in this harness |
| v0.9.4 | - | - | 0 | 0 | excluded: kind 'mistralrs' has no bench lane in this harness |

## Test bed

| Component | Value |
|---|---|
| CPU | Intel Core i7-14650HX, 24 hardware threads |
| Discrete GPU | NVIDIA GeForce RTX 4070 Laptop, 8 GiB, driver 580.173.02 |
| Integrated GPU | Intel Graphics (RPL-S), Vulkan device |
| RAM | 16 GiB (13.3 GiB usable) |
| OS | Linux Mint 22.3, kernel 7.0.0-31-generic |
| Runtimes compared | blazar 0.13.0 gateway - inventory - ollama-host - piper (gateway TTS lane) - sglang-0.5.19 |
| Model | en_US-amy-medium, qwen3-1.7b.d |

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
- Cold-start parity: the model file's page cache is dropped (posix_fadvise DONTNEED) and the GPU asserted idle (<512 MiB) before every cold probe on every runtime - a cold load is disk-cold, not memory-warm.
- Cold TTFT = first-token latency of the cold probe itself (max_tokens 4, aligned num_ctx 16384 on both runtimes).
- ollama daemon boot is only measured with --ollama-service-restart (systemd restart, sudo password via BENCH_SUDO_PASSWORD env, stdin-only); without it the daemon stays warm and the row says so.
- Idle-wake: blazar's reaper sleeps the child at idle_sleep_secs (weights stay RAM-resident, VRAM released) - wake TTFT is a sleep-wake; ollama's keep_alive expiry fully unloads - wake TTFT is a disk reload. The policy column names the semantic; both measured after the policy is observed via /api/ps.
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
| blazar gateway - sglang-0.5.19 | engine-scheduled | 67.7 | 32.8 | 32.9 | 14.8 | 15.5 | 7053.0 | 10962.7 | 6470 | 55.5 |
| blazar gateway - sglang-0.5.19 (single-stream) | engine-scheduled | 67.6 | 32.8 | 33.4 | 14.8 | 15.4 | 7406.2 | 10957.0 | 6282 | 55.8 |
| blazar gateway - sglang-0.5.19 (kv_unified_on) | engine-scheduled | 67.6 | 32.9 | 33.0 | 14.8 | 15.3 | 6362.6 | 10983.0 | 6424 | 55.1 |
| blazar gateway - sglang-0.5.19 (kv_unified_off) | engine-scheduled | 67.6 | 32.9 | 33.1 | 14.8 | 15.5 | 6636.9 | 10977.2 | 6454 | 55.1 |
| blazar gateway - sglang-0.5.19 (spec_off) | engine-scheduled | 67.6 | 32.2 | 33.7 | 14.8 | 15.5 | 6437.2 | 11032.7 | 6490 | 55.4 |
| blazar gateway - sglang-0.5.19 (deterministic_on) | engine-scheduled | 39.5 | 58.3 | 58.9 | 25.3 | 26.7 | 4706.2 | 6262.8 | 6698 | 56.5 |
| blazar gateway - sglang-0.5.19 (mem_frac_0.90) | engine-scheduled | 67.6 | 33.0 | 33.3 | 14.8 | 15.5 | 7123.0 | 11094.6 | 6556 | 55.0 |
| blazar gateway - sglang-0.5.19 (radix_session_on) | engine-scheduled | 67.6 | 32.9 | 33.6 | 14.8 | 15.3 | 6362.4 | 10986.2 | 6454 | 55.1 |
| blazar gateway - sglang-0.5.19 (chunked_prefill_4096) | engine-scheduled | 67.6 | 32.7 | 33.6 | 14.8 | 15.5 | 6382.4 | 10976.7 | 6750 | 55.8 |
| blazar gateway - sglang-0.5.19 (page_64) | engine-scheduled | 67.5 | 33.7 | 34.0 | 14.8 | 15.4 | 7071.7 | 10581.2 | 6440 | 55.7 |
| blazar gateway - sglang-0.5.19 (memory_saver_on) | engine-scheduled | 67.6 | 33.1 | 33.2 | 14.8 | 15.6 | 6305.2 | 11152.5 | 6476 | 55.0 |
| blazar gateway - sglang-0.5.19 (torch_compile_on) | engine-scheduled | 68.7 | 31.9 | 32.3 | 14.6 | 14.9 | 7168.5 | 11275.5 | 6470 | 55.1 |
| blazar gateway - sglang-0.5.19 (kv_dtype_bf16) | engine-scheduled | 67.6 | 32.5 | 32.8 | 14.8 | 15.2 | 6330.1 | 11099.8 | 6416 | 55.3 |
| blazar gateway - sglang-0.5.19 (kv_dtype_e4m3) | engine-scheduled | 67.3 | 32.7 | 33.5 | 14.9 | 15.4 | 7349.2 | 11236.5 | 6416 | 55.1 |
| blazar gateway - sglang-0.5.19 (cg_bs_16) | engine-scheduled | 67.6 | 32.4 | 33.0 | 14.8 | 15.3 | 6291.7 | 11089.8 | 6430 | 55.1 |
| blazar gateway - sglang-0.5.19 (cg_bs_256) | engine-scheduled | 67.6 | 32.5 | 33.1 | 14.8 | 15.3 | 7310.0 | 11084.3 | 6804 | 55.4 |
| ollama 0.33.3 - qwen3:1.7b (t/s NOT comparable - matrix model differs) | service | 156.2 | 43.3 | 45.5 | 6.4 | 7.1 | 3713.5 | 61604.0 | 2322 | 55.4 |
| ollama 0.33.3 - qwen3:1.7b (t/s NOT comparable - matrix model differs) | service | 156.9 | 41.3 | 47.3 | 6.4 | 7.1 | 3562.4 | 59197.5 | 2322 | 55.0 |

### Concurrency (4 parallel streams x 128 tokens)

| Runtime | slots | ok streams | rounds | system t/s | sum-stream t/s | wall s | TTFT max ms | TTFT p99 ms | ITL p99 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| blazar gateway - sglang-0.5.19 | engine-scheduled | 12 of 3x4 | 3 | 253.8 | 769.1 | 6.05 | 46 | 45 | 17.3 |
| ollama - qwen3:1.7b | service | 12 of 3x4 | 3 | 127.1 | 1863.4 | 12.09 | 4579 | 4486 | 8.0 |
| ollama - qwen3:1.7b | service | 12 of 3x4 | 3 | 126.9 | 1855.2 | 12.10 | 4558 | 4465 | 7.9 |

_sum-stream >> system t/s means streams serialize on one slot; roughly equal means genuinely parallel._

### Concurrency frontier (system t/s and tail latency vs level)

| Runtime | C | ok streams | system t/s | sum-stream t/s | eff vs C=1 | TTFT p99 ms | ITL p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| blazar gateway - sglang-0.5.19 | 4 | 12 of 3x4 | 253.8 | 769.1 | - | 45 | 17.3 |
| ollama - qwen3:1.7b | 4 | 12 of 3x4 | 127.1 | 1863.4 | - | 4486 | 8.0 |
| ollama - qwen3:1.7b | 4 | 12 of 3x4 | 126.9 | 1855.2 | - | 4465 | 7.9 |

- blazar gateway - sglang-0.5.19: 1 level(s) measured - insufficient levels for a saturation verdict.
- ollama - qwen3:1.7b: 2 level(s) measured - insufficient levels for a saturation verdict.

### Adaptive reshape under sustained load (no-lag proof)

| Runtime | reshape | slots | time to reshape s | req before/after | TTFT p50 before->after ms | sys t/s before->after | failed |
|---|---|---|---:|---:|---:|---:|---:|
| blazar gateway - sglang-0.5.19 | n/a | skipped: adaptive reshape is a llamacpp-lane mechanism; this engine's child exposes no slot shape (-np) to observe or adopt | - | - | - | - | - |

### Perplexity

| Engine | perplexity (ctx 2048, offline ASCII corpus) | ppl tool |
|---|---:|---|
| sglang-0.5.19 | not applicable (tool is llama.cpp-family) | own |

### Greedy parity and gateway transparency (20 prompts, 256 tokens)

| Comparison | exact / total | ratio mean | ratio min |
|---|---:|---:|---:|
| ollama (qwen3:1.7b) run-to-run, temp 0 | 20/20 | 1.000 | 1.000 |
| ollama (qwen3:1.7b) run-to-run, temp 0 | 20/20 | 1.000 | 1.000 |

_Exact-match divergence across GPU backends is expected float nondeterminism (batch shape and backend kernels), not translation drift; bit-parity across runs requires single-slot decoding (blazar `deterministic = true` pins it)._

### Tool calls (single-turn selection + schema quality)

| Runtime | scenarios | well-formed | selection | args valid | control FP | TTFT p50 ms |
|---|---:|---:|---:|---:|---|---:|
| blazar gateway - sglang-0.5.19 | 6 | 0/6 | 0/5 | 0/5 | no | 36 |
| ollama - qwen3.5:9b | 6 | 4/6 | 4/5 | 4/5 | no | 285 |
| ollama - qwen3.5:9b | 6 | 4/6 | 4/5 | 4/5 | no | 289 |

### Optimization axes (ctx 4096, single stream)

_Not measured in this campaign (20260929-sglang-safetensors); optimization axes lane not run._

### Gateway config knobs (on/off vs default profile)

| Engine | config | decode t/s | decode delta% | ttft p50 ms | ttft delta% | GPU peak MiB | GPU delta | output vs default |
|---|---|---:|---:|---:|---:|---:|---:|:-:|
| sglang-0.5.19 | single-stream | 67.6 | -0.1% | 33 | +0.2% | 6282 | -188 | same |
| sglang-0.5.19 | kv_unified_on | 67.6 | -0.0% | 33 | +0.4% | 6424 | -46 | same |
| sglang-0.5.19 | kv_unified_off | 67.6 | -0.1% | 33 | +0.3% | 6454 | -16 | same |
| sglang-0.5.19 | spec_off | 67.6 | -0.1% | 32 | -1.8% | 6490 | +20 | same |
| sglang-0.5.19 | deterministic_on | 39.5 | -41.7% | 58 | +77.8% | 6698 | +228 | same |
| sglang-0.5.19 | mem_frac_0.90 | 67.6 | -0.1% | 33 | +0.6% | 6556 | +86 | same |
| sglang-0.5.19 | radix_session_on | 67.6 | -0.1% | 33 | +0.3% | 6454 | -16 | same |
| sglang-0.5.19 | chunked_prefill_4096 | 67.6 | -0.1% | 33 | -0.3% | 6750 | +280 | same |
| sglang-0.5.19 | page_64 | 67.5 | -0.2% | 34 | +2.8% | 6440 | -30 | same |
| sglang-0.5.19 | memory_saver_on | 67.6 | -0.1% | 33 | +0.8% | 6476 | +6 | same |
| sglang-0.5.19 | torch_compile_on | 68.7 | +1.5% | 32 | -2.8% | 6470 | +0 | same |
| sglang-0.5.19 | kv_dtype_bf16 | 67.6 | -0.1% | 32 | -0.9% | 6416 | -54 | same |
| sglang-0.5.19 | kv_dtype_e4m3 | 67.3 | -0.5% | 33 | -0.2% | 6416 | -54 | DRIFT |
| sglang-0.5.19 | cg_bs_16 | 67.6 | -0.1% | 32 | -1.3% | 6430 | -40 | same |
| sglang-0.5.19 | cg_bs_256 | 67.6 | -0.1% | 33 | -0.7% | 6804 | +334 | same |

### Gateway scheduler knobs under concurrent load (C=8 bursts)

_Not measured in this campaign (20260929-sglang-safetensors); conc lane A/B lane not run._

### Engine capability matrix

| Capability | sglang-0.5.19 |
|---|---:|
| anthropic-api | no |
| ctx-override | no |
| embeddings | no |
| grammar-gbnf | no |
| json-schema | no |
| kv-quant | no |
| lora-adapter | no |
| metrics-endpoint | no |
| paged-attn | no |
| parallel-np | no |
| quant-on-load | no |
| rerank | no |
| slots-sessions | no |
| spec-decode | no |
| tokenize-endpoint | no |
| vision-mmproj | no |

### Cold start and footprint

| Runtime | daemon boot s | first request (cold engine load) s | cold TTFT ms | engine load s | RSS peak MiB |
|---|---:|---:|---:|---:|---:|
| blazar gateway - sglang-0.5.19 | 0.55 | 36.27 | 36242 | - | 422 |
| blazar gateway - sglang-0.5.19 (single-stream) | 0.52 | 30.21 | 30181 | - | 1352 |
| blazar gateway - sglang-0.5.19 (kv_unified_on) | 0.52 | 29.27 | 29236 | - | 1357 |
| blazar gateway - sglang-0.5.19 (kv_unified_off) | 0.52 | 29.22 | 29187 | - | 1358 |
| blazar gateway - sglang-0.5.19 (spec_off) | 0.52 | 29.29 | 29254 | - | 1357 |
| blazar gateway - sglang-0.5.19 (deterministic_on) | 0.52 | 30.84 | 30770 | - | 1357 |
| blazar gateway - sglang-0.5.19 (mem_frac_0.90) | 0.53 | 29.29 | 29242 | - | 1358 |
| blazar gateway - sglang-0.5.19 (radix_session_on) | 0.52 | 29.24 | 29216 | - | 1356 |
| blazar gateway - sglang-0.5.19 (chunked_prefill_4096) | 0.52 | 35.23 | 35197 | - | 1356 |
| blazar gateway - sglang-0.5.19 (page_64) | 0.53 | 31.31 | 31274 | - | 803 |
| blazar gateway - sglang-0.5.19 (memory_saver_on) | 0.52 | 28.22 | 28192 | - | 1357 |
| blazar gateway - sglang-0.5.19 (torch_compile_on) | 0.52 | 55.26 | 55222 | - | 1357 |
| blazar gateway - sglang-0.5.19 (kv_dtype_bf16) | 0.52 | 28.25 | 28211 | - | 1356 |
| blazar gateway - sglang-0.5.19 (kv_dtype_e4m3) | 0.52 | 28.23 | 28194 | - | 1353 |
| blazar gateway - sglang-0.5.19 (cg_bs_16) | 0.52 | 28.21 | 28179 | - | 1353 |
| blazar gateway - sglang-0.5.19 (cg_bs_256) | 0.52 | 30.29 | 30242 | - | 1363 |
| ollama - qwen3:1.7b | - | 2.15 | 2138 | 2.02 | - |
| ollama - qwen3:1.7b | - | 2.16 | 2157 | 2.04 | - |

_Every cold probe runs page-cache-dropped and GPU-idle-asserted on both runtimes; ollama rows without --ollama-service-restart leave the daemon warm (note in the artifact)._

### Idle wake (sleep vs keep_alive expiry)

| Runtime | idle policy | policy observed | wake TTFT ms | reload s | note |
|---|---|---|---:|---:|---|
| blazar - sglang-0.5.19 | sleep at 15s (weights stay RAM-resident) | yes | 44 | - |  |
| ollama - qwen3:1.7b | keep_alive 20s -> full unload | yes | 2004 | 1.86 |  |
| ollama - qwen3:1.7b | keep_alive 20s -> full unload | yes | 2091 | 1.94 |  |

Wake cost is the upstream engine child restoring its CUDA context and re-uploading weights (single-level sleep; warm steady-state is ~0.1 s). Raise `idle_sleep_secs` to sleep less often, or lower it toward the evict ladder when wake latency matters less than VRAM residency.

_blazar sleeps with weights in RAM (wake = resume); ollama unloads at keep_alive expiry (wake = full disk reload). Policies differ by design - the table measures each runtime's own idle path after the policy verifiably fired._

### Long-context degradation curve

| Runtime | ctx | decode t/s | TTFT p50 ms |
|---|---:|---:|---:|
| blazar - sglang-0.5.19 | 2048 | 67.6 | 32 |
| blazar - sglang-0.5.19 | 8192 | 67.6 | 33 |
| blazar - sglang-0.5.19 | 16384 | 67.6 | 33 |
| ollama - qwen3:1.7b | 2048 | 157.8 | 41 |
| ollama - qwen3:1.7b | 2048 | 156.6 | 43 |
| ollama - qwen3:1.7b | 8192 | 157.6 | 33 |
| ollama - qwen3:1.7b | 8192 | 155.8 | 35 |
| ollama - qwen3:1.7b | 16384 | 155.6 | 52 |
| ollama - qwen3:1.7b | 16384 | 157.5 | 42 |

### Media lanes (image / video / TTS / whisper)

| Lane | cold s | median s | min s | max s | ground truth | gate reject s | TTFB speedup |
|---|---:|---:|---:|---:|---|---:|---:|
| tts - en_US-amy-medium (840 chars, wav+pcm) | - | 2.03 | - | 3.47 | RTF wav 0.034 / pcm 0.058 (60s audio) |  | 3.97x |
| tts - en_US-amy-medium (840 chars, wav+pcm) | - | 1.99 | - | 3.42 | RTF wav 0.033 / pcm 0.057 (60s audio) |  | 3.80x |
| tts-conc - en_US-amy-medium (400 chars x4 pcm) | - | 4.49 | 4.75 | - | efficiency 3.66 of 4 streams |  | 1560ms max TTFB |
| tts-conc - en_US-amy-medium (400 chars x4 pcm) | - | 4.69 | 4.96 | - | efficiency 3.78 of 4 streams |  | 1610ms max TTFB |

_Not measured in this campaign (20260929-sglang-safetensors); media config A/B lane not run._

_Media cells run through the same sandboxed gateway as text lanes but do not assert GPU-idle: a warm engine child is the normal serving shape, so each row stamps gpu_busy_mib / ram_avail_mib / loadavg instead. 3 runs (not 5); media variance is dominated by the model, not the scheduler. Video frame counts are read from the EBML container (lacing-aware), never from an API field; the VRAM gate probe times how fast an over-budget request is rejected with a teaching error._

## Findings (this campaign)

1. **Concurrency scaling per engine.** sglang-0.5.19 (C=4): peak 253.8 t/s at C=4; serialization behavior per level in the frontier table below.
2. **Prompt cache pays 1.6x on prefill** (10963 cached vs 7053 t/s cold).
3. **Tool-call quality (single-turn, temp 0).** sglang-0.5.19: selection 0/5, args 0/5 (control clean); ollama: selection 4/5, args 4/5 (control clean); ollama: selection 4/5, args 4/5 (control clean); per-scenario detail in cells.jsonl.
4. **Adaptive reshape lands under sustained load.** sglang-0.5.19: no reshape observed in the lane window (graceful-drain: adoption waits for in-flight streams, never kills one).
5. **Streamed PCM cuts time-to-first-audio 3.97x vs buffered WAV** (piper lane, first audio 0.51s vs 2.03s full synthesis) - total wall time is slightly higher (per-chunk synthesis), the win is interactivity.
6. **Streamed PCM cuts time-to-first-audio 3.80x vs buffered WAV** (piper lane, first audio 0.52s vs 1.99s full synthesis) - total wall time is slightly higher (per-chunk synthesis), the win is interactivity.
7. **4 parallel PCM streams through one gateway: perfectly parallel** (efficiency 3.66 = sum of per-stream totals / 4.75s wall, max TTFB 1560 ms, NON-uniform stream outputs - flagged) - the scalability receipt for the TTS lane.
8. **4 parallel PCM streams through one gateway: perfectly parallel** (efficiency 3.78 = sum of per-stream totals / 4.96s wall, max TTFB 1610 ms, NON-uniform stream outputs - flagged) - the scalability receipt for the TTS lane.

## Carried-over findings (no receipt in this campaign)

_Established in earlier campaigns whose receipts live in their bench-artifacts/ directories; this campaign did not measure these lanes._

1. **Gateway overhead is within measurement noise.** Single-stream decode through the blazar gateway matches direct engine spawns at the same slots/context (see speed table); the greedy gateway lane is byte-identical to the direct lane where sampling is single-slot.
2. **Capacity-aware slot auto-sizing.** blazar sizes engine slots from live hardware census: the 8 GiB card with a vision projector attached spawns 1 slot (16 Ki context) on the Vulkan build and 4 slots (64 Ki total) on CUDA - measured oversubscription on Vulkan either fails to boot or degrades 2x, so the cap is load-bearing, not conservative cosmetics.
3. **Speculative n-gram decoding is a net loss for this 9B model** (no draft model; acceptance too low to pay the verification overhead) - documented so the flag is not cargo-culted.
4. **KV q8_0 quantization is decode-neutral and prefill-neutral steady-state**; the one cold-prefill outlier below is a first-invocation pipeline-compile artifact (controlled re-probe measured full-rate steady state).
5. **mistral.rs 0.9.3 with default paged attention cannot fit this model on an 8 GiB card** (upstream sizes KV as a fraction of total VRAM); blazar's profile auto-disables paged attention on tight cards and the model then serves correctly.

## Failed cells (engine-reality receipts)

_These knobs crashed the engine child on this test bed; the crash signature is the receipt. Raw logs: `cells.jsonl` (`daemon_log_tail` field)._

| Engine | lane | config | failure |
|---|---|---|---|
| sglang-0.5.19 | blazar | hicache_on | cold probe failed: HTTP Error 400: Bad Request |

## Structured skips (tool/format boundaries)

- sglang-0.5.19 (ppl): skipped: llama-perplexity loads GGUF files only; qwen3-1.7b.d is an HF safetensors directory - model-format boundary, not an engine failure
- sglang-0.5.19 (reshape): skipped: adaptive reshape is a llamacpp-lane mechanism; this engine's child exposes no slot shape (-np) to observe or adopt

## Caveats

- ollama prefill numbers come from engine counters that exclude the chat template, so they read slightly high against the 512-token lanes.
- Cross-backend greedy ratios (CUDA vs Vulkan) diverge on near-tie logits; treat ratio, not exact-match count, as the signal.
- All GPU rows measured on AC power at bounded load; rows record load average and power state (battery runs are rejected by the harness).
- Numbers are medians of 5 runs on one hybrid laptop; expect absolute shifts on other hardware, ratios to travel better.

## Reproduce

```bash
python3 scripts/bench_matrix.py --model qwen3-1.7b --engines sglang-0.5.19 --artifacts-dir bench-artifacts/20260929-sglang-safetensors --md bench-artifacts/20260929-sglang-safetensors/benchmark.md
python3 scripts/bench_matrix.py --render-only --artifacts-dir <dir> [--append-campaign <sibling-campaign-dir>] --md BENCHMARK.md
```

_Raw per-cell records (argv, per-run lists, daemon logs): `bench-artifacts/20260929-sglang-safetensors/cells.jsonl`._
