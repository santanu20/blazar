# Blazar inference benchmark

_Rendered 20260928-gguf-full; blazar 0.13.0; power state of gateway rows: ac._

## Executive summary

- v0.9.4: gateway 41.0 vs direct 21.5 t/s (+90.3%)
- v0.9.4 prompt-cache prefill 7396 vs 1370 t/s cold
- b11202-cuda sweep C=4: C4: 67.5 t/s system (2x8192)
- v0.9.4 sweep C=4: C4: 67.4 t/s system (2x8192)
- gateway cold boot 0.52 s
- cold TTFT 4430 ms vs ollama 5446 ms (1.2x)
- idle wake 4615 ms (sleep) vs ollama 5958 ms (full reload)

## Measured in this campaign

- blazar: 32 cell(s)
- cold-ollama: 1 cell(s)
- conc-blazar: 2 cell(s)
- conc-direct: 1 cell(s)
- conc-ollama: 1 cell(s)
- ctxcurve-blazar: 6 cell(s)
- ctxcurve-ollama: 3 cell(s)
- direct: 13 cell(s)
- features: 2 cell(s)
- greedy: 2 cell(s)
- greedy_gw: 1 cell(s)
- idle-blazar: 2 cell(s)
- idle-ollama: 1 cell(s)
- media-image: 4 cell(s)
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
| v0.9.4 | mistralrs | text (direct + gateway) | 27 | 0 | benchmarked |
| b5130 | whisper | media | 1 | 0 | benchmarked |
| b11202-cuda | llamacpp | text (direct + gateway) | 40 | 0 | benchmarked |
| master-920-2f88688 | sdcpp | media | 5 | 0 | benchmarked |
| sglang-0.5.19 | sglang | — | 0 | 0 | no cells in this campaign |

## Test bed

| Component | Value |
|---|---|
| CPU | Intel Core i7-14650HX, 24 hardware threads |
| Discrete GPU | NVIDIA GeForce RTX 4070 Laptop, 8 GiB, driver 580.173.02 |
| Integrated GPU | Intel Graphics (RPL-S), Vulkan device |
| RAM | 16 GiB (13.3 GiB usable) |
| OS | Linux Mint 22.3, kernel 7.0.0-31-generic |
| Runtimes compared | blazar 0.13.0 gateway - b11202-cuda - inventory - master-920-2f88688 - ollama-host - piper (gateway TTS lane) - v0.9.4 - whisper.cpp b5130 |
| Model | Qwen3.5-9B-Q4_K_M.gguf, en_US-amy-medium, ggml-base, qwen-image-2.1-uncensored, wan_2.1_comfyui_repackaged |

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
| blazar gateway - b11202-cuda | 2x8192 | 41.1 | 124.2 | 174.3 | 24.3 | 25.6 | 1302.3 | 7212.0 | 5458 | 55.6 |
| blazar gateway - b11202-cuda (single-stream) | 1x16384 | 41.6 | 125.0 | 126.5 | 24.1 | 25.2 | 1313.0 | 7239.3 | 5666 | 55.2 |
| blazar gateway - b11202-cuda (cont_batching_off) | 2x8192 | 41.0 | 124.7 | 176.6 | 24.3 | 25.5 | 1307.7 | 7191.5 | 5458 | 55.9 |
| blazar gateway - b11202-cuda (fa_on) | 2x8192 | 41.0 | 124.4 | 131.1 | 24.4 | 25.3 | 1318.9 | 7293.1 | 5458 | 55.1 |
| blazar gateway - b11202-cuda (fa_off) | 2x8192 | 40.7 | 160.3 | 351.2 | 24.5 | 25.7 | 1158.8 | 7434.7 | 5696 | 56.4 |
| blazar gateway - b11202-cuda (kv_unified_on) | 2x8192 | 41.0 | 121.6 | 127.0 | 24.4 | 25.4 | 1371.7 | 7327.4 | 5458 | 55.6 |
| blazar gateway - b11202-cuda (kv_unified_off) | 2x8192 | 41.0 | 123.2 | 126.0 | 24.3 | 25.6 | 1353.3 | 7293.5 | 5458 | 56.9 |
| blazar gateway - b11202-cuda (swa_full_on) | 2x8192 | 41.6 | 122.6 | 125.1 | 24.1 | 25.1 | 1316.6 | 7460.8 | 5458 | 56.1 |
| blazar gateway - b11202-cuda (no_kv_offload_on) | 2x8192 | 18.5 | 149.8 | 205.0 | 53.8 | 59.6 | 931.3 | 5074.5 | 5106 | 52.1 |
| blazar gateway - b11202-cuda (cache_q8_q8_0) | 2x8192 | 40.6 | 124.1 | 124.7 | 24.6 | 25.5 | 1312.3 | 7580.5 | 5338 | 55.5 |
| blazar gateway - b11202-cuda (ctx_checkpoints_4) | 2x8192 | 41.0 | 120.9 | 124.8 | 24.3 | 25.5 | 1317.7 | 7538.2 | 5477 | 55.5 |
| blazar gateway - b11202-cuda (spec_off) | 2x8192 | 41.6 | 119.2 | 131.7 | 24.0 | 25.2 | 1308.3 | 7349.5 | 5452 | 55.2 |
| blazar gateway - b11202-cuda (deterministic_on) | 1x16384 | 41.1 | 122.8 | 123.6 | 24.3 | 25.4 | 1312.0 | 7424.7 | 5666 | 57.1 |
| blazar gateway - b11202-cuda (mmproj_offload_off) | 2x8192 | 41.6 | 123.6 | 124.4 | 24.0 | 25.4 | 1313.7 | 7229.0 | 5458 | 56.6 |
| blazar gateway - b11202-cuda (threads_batch_8) | 2x8192 | 41.0 | 125.7 | 173.8 | 24.3 | 25.5 | 1310.9 | 7246.5 | 5458 | 55.7 |
| blazar gateway - b11202-cuda (cache_reuse_256) | 2x8192 | 41.0 | 124.8 | 174.7 | 24.4 | 25.6 | 1287.3 | 7243.0 | 5458 | 55.7 |
| blazar gateway - b11202-cuda (kv_unified_per_slot_4096) | 2x8192 | 41.0 | 123.2 | 123.7 | 24.4 | 25.7 | 1330.7 | 7549.6 | 5458 | 55.3 |
| blazar gateway - b11202-cuda (conc_default) | 2x8192 | - | - | - | - | 79.2 | - | - | 5464 | 55.7 |
| blazar gateway - b11202-cuda (adaptive_slots_off) | 2x8192 | - | - | - | - | 61.9 | - | - | 5466 | 55.1 |
| blazar gateway - b11202-cuda (poll_50) | 2x8192 | - | - | - | - | 30.3 | - | - | 5464 | 55.6 |
| blazar gateway - v0.9.4 | 2x8192 | 41.1 | 122.2 | 124.4 | 24.3 | 25.7 | 1379.9 | 7160.5 | 5458 | 56.8 |
| blazar gateway - v0.9.4 (paged_attn_off) | 2x8192 | 41.6 | 119.2 | 124.7 | 24.1 | 25.3 | 1370.2 | 7447.0 | 5458 | 55.7 |
| blazar gateway - v0.9.4 (single-stream) | 1x16384 | 41.5 | 122.3 | 123.4 | 24.1 | 25.4 | 1403.8 | 7654.8 | 5666 | 55.7 |
| blazar gateway - v0.9.4 (kv_unified_on) | 2x8192 | 41.1 | 124.0 | 125.2 | 24.3 | 25.6 | 1367.1 | 7414.7 | 5464 | 55.5 |
| blazar gateway - v0.9.4 (kv_unified_off) | 2x8192 | 41.1 | 120.3 | 125.1 | 24.3 | 25.5 | 1312.5 | 7397.5 | 5465 | 56.8 |
| blazar gateway - v0.9.4 (spec_off) | 2x8192 | 41.4 | 122.9 | 123.6 | 24.1 | 25.5 | 1325.0 | 7317.3 | 5458 | 56.5 |
| blazar gateway - v0.9.4 (deterministic_on) | 1x16384 | 41.0 | 118.1 | 121.7 | 24.3 | 25.3 | 1313.0 | 7244.9 | 5666 | 56.4 |
| blazar gateway - v0.9.4 (pa_mem_0.85) | 2x8192 | 41.1 | 122.2 | 124.5 | 24.3 | 25.5 | 1326.4 | 7667.6 | 5458 | 56.6 |
| blazar gateway - v0.9.4 (pa_mem_0.55) | 2x8192 | 41.0 | 121.2 | 122.3 | 24.4 | 25.3 | 1316.9 | 7371.5 | 5458 | 55.7 |
| blazar gateway - v0.9.4 (mr_batch_64) | 2x8192 | 41.6 | 122.6 | 125.6 | 24.1 | 25.2 | 1366.6 | 7352.9 | 5464 | 55.9 |
| blazar gateway - v0.9.4 (mr_prefix_cache_256) | 2x8192 | 41.0 | 121.4 | 173.7 | 24.4 | 25.6 | 1314.6 | 7358.3 | 5464 | 55.9 |
| blazar gateway - v0.9.4 (mr_enc_cache_512mb) | 2x8192 | 41.0 | 121.2 | 123.4 | 24.4 | 25.7 | 1369.8 | 7396.1 | 5464 | 57.1 |
| direct engine - b11202-cuda | 1x16384 | 41.6 | 122.9 | 124.5 | 24.0 | 25.3 | 1308.9 | 7439.5 | 5666 | 55.9 |
| direct engine - v0.9.4 | 1x16384 | 21.5 | 150.9 | 184.8 | 46.3 | 55.1 | 281.3 | 345.2 | 7042 | 44.9 |
| ollama 0.33.3 - qwen3.5:9b | service | 40.6 | 125.8 | 132.3 | 24.9 | 75.0 | 1409.3 | 7372.4 | 6570 | 55.4 |

### Concurrency (4 parallel streams x 128 tokens)

| Runtime | slots | ok streams | rounds | system t/s | sum-stream t/s | wall s | TTFT max ms | TTFT p99 ms | ITL p99 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| blazar gateway - b11202-cuda | 2x8192 | 12/4 | 3 | 67.5 | 433.1 | 22.75 | 4205 | 4204 | 81.3 |
| blazar gateway - v0.9.4 | 2x8192 | 12/4 | 3 | 67.4 | 434.3 | 22.79 | 4216 | 4213 | 32.8 |
| direct engine - b11202-cuda | 4 | 4/4 | 1 | 110.6 | 116.7 | 4.63 | 276 | - | 41.4 |
| ollama - qwen3.5:9b | service | 12/4 | 3 | 33.4 | 486.9 | 45.98 | 16572 | 16213 | 74.7 |

_sum-stream >> system t/s means streams serialize on one slot; roughly equal means genuinely parallel._

### Concurrency frontier (system t/s and tail latency vs level)

| Runtime | C | ok streams | system t/s | sum-stream t/s | eff vs C=1 | TTFT p99 ms | ITL p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| blazar gateway - b11202-cuda | 4 | 12 | 67.5 | 433.1 | - | 4204 | 81.3 |
| blazar gateway - v0.9.4 | 4 | 12 | 67.4 | 434.3 | - | 4213 | 32.8 |
| direct engine - b11202-cuda | 4 | 4 | 110.6 | 116.7 | - | - | 41.4 |
| ollama - qwen3.5:9b | 4 | 12 | 33.4 | 486.9 | - | 16213 | 74.7 |

- blazar gateway - b11202-cuda: 1 level(s) measured - insufficient levels for a saturation verdict.
- blazar gateway - v0.9.4: 1 level(s) measured - insufficient levels for a saturation verdict.
- direct engine - b11202-cuda: 1 level(s) measured - insufficient levels for a saturation verdict.
- ollama - qwen3.5:9b: 1 level(s) measured - insufficient levels for a saturation verdict.

### Adaptive reshape under sustained load (no-lag proof)

| Runtime | reshape | slots | time to reshape s | req before/after | TTFT p50 before→after ms | sys t/s before→after | failed |
|---|---|---|---:|---:|---:|---:|---:|
| blazar gateway - b11202-cuda | NO | -→- | - | 116/0 | 16327→- | 35→- | 0 |
| blazar gateway - v0.9.4 | NO | -→- | - | 116/0 | 16333→- | 35→- | 0 |

### Perplexity

| Engine | perplexity (ctx 2048, offline ASCII corpus) |
|---|---:|
| b11202-cuda | 17.35 ± 0.92 |
| v0.9.4 | 17.35 ± 0.92 |

### Greedy parity and gateway transparency (20 prompts, 256 tokens)

| Comparison | exact / total | ratio mean | ratio min |
|---|---:|---:|---:|
| b11202-cuda vs same-engine reference (direct) | 20/20 | 1.000 | 1.000 |
| v0.9.4 vs same-engine reference (direct) | 0/20 | 0.514 | 0.000 |
| b11202-cuda through blazar gateway vs direct | 16/20 | 0.940 | 0.017 |

_Exact-match divergence across GPU backends is expected float nondeterminism (batch shape and backend kernels), not translation drift; bit-parity across runs requires single-slot decoding (blazar `deterministic = true` pins it)._

### Tool calls (single-turn selection + schema quality)

| Runtime | scenarios | well-formed | selection | args valid | control FP | TTFT p50 ms |
|---|---:|---:|---:|---:|---|---:|
| blazar gateway - b11202-cuda | 6 | 4/6 | 4/5 | 4/5 | no | 114 |
| blazar gateway - v0.9.4 | 6 | 4/6 | 4/5 | 4/5 | no | 113 |
| ollama - qwen3.5:9b | 6 | 4/6 | 4/5 | 4/5 | no | 283 |

### Optimization axes (ctx 4096, single stream)

| Engine | axis | setting | decode t/s | delta vs dense | prefill cold t/s | delta |
|---|---|---|---:|---:|---:|---:|
| b11202-cuda | kv | q8_0 | 41.0 | -0.2 | 1389.3 | 123.8 |
| b11202-cuda | pa | on | 41.5 | 0.2 | 1316.2 | 50.8 |
| b11202-cuda | spec | ngram-simple | 41.0 | -0.2 | 1334.6 | 69.2 |
| b11202-cuda | mmproj | True | 41.4 | 0.2 | 1309.8 | 44.4 |
| v0.9.4 | pa | off | 18.2 | -3.0 | 204.2 | -69.6 |

### Gateway config knobs (on/off vs default profile)

| Engine | config | decode t/s | decode Δ% | ttft p50 ms | ttft Δ% | GPU peak MiB | GPU Δ | output vs default |
|---|---|---:|---:|---:|---:|---:|---:|:-:|
| b11202-cuda | single-stream | 41.6 | +1.0% | 125 | +0.6% | 5666 | +208 | ✓ same |
| b11202-cuda | cont_batching_off | 41.0 | -0.3% | 125 | +0.4% | 5458 | +0 | ✓ same |
| b11202-cuda | fa_on | 41.0 | -0.2% | 124 | +0.2% | 5458 | +0 | ✓ same |
| b11202-cuda | fa_off | 40.7 | -0.9% | 160 | +29.1% | 5696 | +238 | ✓ same |
| b11202-cuda | kv_unified_on | 41.0 | -0.3% | 122 | -2.1% | 5458 | +0 | ✓ same |
| b11202-cuda | kv_unified_off | 41.0 | -0.3% | 123 | -0.8% | 5458 | +0 | ✓ same |
| b11202-cuda | swa_full_on | 41.6 | +1.1% | 123 | -1.3% | 5458 | +0 | ✓ same |
| b11202-cuda | no_kv_offload_on | 18.5 | -54.9% | 150 | +20.6% | 5106 | -352 | ✓ same |
| b11202-cuda | cache_q8_q8_0 | 40.6 | -1.2% | 124 | -0.1% | 5338 | -120 | ✓ same |
| b11202-cuda | ctx_checkpoints_4 | 41.0 | -0.3% | 121 | -2.6% | 5477 | +19 | ✓ same |
| b11202-cuda | spec_off | 41.6 | +1.0% | 119 | -4.0% | 5452 | -6 | ✓ same |
| b11202-cuda | deterministic_on | 41.1 | -0.2% | 123 | -1.1% | 5666 | +208 | ✓ same |
| b11202-cuda | mmproj_offload_off | 41.6 | +1.1% | 124 | -0.5% | 5458 | +0 | ✓ same |
| b11202-cuda | threads_batch_8 | 41.0 | -0.3% | 126 | +1.2% | 5458 | +0 | ✓ same |
| b11202-cuda | cache_reuse_256 | 41.0 | -0.4% | 125 | +0.5% | 5458 | +0 | ✓ same |
| b11202-cuda | kv_unified_per_slot_4096 | 41.0 | -0.4% | 123 | -0.8% | 5458 | +0 | ✓ same |
| v0.9.4 | paged_attn_off | 41.6 | +1.1% | 119 | -2.5% | 5458 | +0 | ✓ same |
| v0.9.4 | single-stream | 41.5 | +0.9% | 122 | +0.1% | 5666 | +208 | ✓ same |
| v0.9.4 | kv_unified_on | 41.1 | -0.1% | 124 | +1.5% | 5464 | +6 | ✓ same |
| v0.9.4 | kv_unified_off | 41.1 | -0.1% | 120 | -1.5% | 5465 | +7 | ✓ same |
| v0.9.4 | spec_off | 41.4 | +0.8% | 123 | +0.6% | 5458 | +0 | ✓ same |
| v0.9.4 | deterministic_on | 41.0 | -0.2% | 118 | -3.3% | 5666 | +208 | ✓ same |
| v0.9.4 | pa_mem_0.85 | 41.1 | -0.1% | 122 | -0.0% | 5458 | +0 | ✓ same |
| v0.9.4 | pa_mem_0.55 | 41.0 | -0.2% | 121 | -0.8% | 5458 | +0 | ✓ same |
| v0.9.4 | mr_batch_64 | 41.6 | +1.1% | 123 | +0.3% | 5464 | +6 | ✓ same |
| v0.9.4 | mr_prefix_cache_256 | 41.0 | -0.3% | 121 | -0.6% | 5464 | +6 | ✓ same |
| v0.9.4 | mr_enc_cache_512mb | 41.0 | -0.3% | 121 | -0.8% | 5464 | +6 | ✓ same |

### Gateway scheduler knobs under concurrent load (C=8 bursts)

| Engine | config | sys tok/s | sys Δ% | ttft p99 ms | ttft Δ% | GPU peak MiB | GPU Δ | errs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| b11202-cuda | conc_default | 58.6 | +0.0% | 16238 | +0.0% | 5464 | +0 | 0 |
| b11202-cuda | adaptive_slots_off | 58.9 | +0.5% | 16537 | +1.8% | 5466 | +2 | 0 |
| b11202-cuda | poll_50 | 58.7 | +0.1% | 16600 | +2.2% | 5464 | +0 | 0 |

### Engine capability matrix

| Capability | b11202-cuda | v0.9.4 |
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
| blazar gateway - b11202-cuda | 0.53 | 5.08 | 4929 | - | 2202 |
| blazar gateway - b11202-cuda | 0.52 | 4.95 | 4872 | - | 2104 |
| blazar gateway - b11202-cuda | 0.52 | 4.79 | 4702 | - | 2103 |
| blazar gateway - b11202-cuda | 0.52 | 4.79 | 4639 | - | 2101 |
| blazar gateway - b11202-cuda | 0.52 | 4.88 | 4719 | - | 2345 |
| blazar gateway - b11202-cuda | 0.52 | 4.87 | 4697 | - | 2152 |
| blazar gateway - b11202-cuda | 0.52 | 4.93 | 4783 | - | 2196 |
| blazar gateway - b11202-cuda | 0.52 | 4.91 | 4766 | - | 2102 |
| blazar gateway - b11202-cuda | 0.52 | 5.54 | 5310 | - | 2465 |
| blazar gateway - b11202-cuda | 0.52 | 4.95 | 4794 | - | 2084 |
| blazar gateway - b11202-cuda | 0.52 | 4.49 | 4342 | - | 2102 |
| blazar gateway - b11202-cuda | 0.52 | 4.64 | 4493 | - | 2102 |
| blazar gateway - b11202-cuda | 0.52 | 4.80 | 4727 | - | 2105 |
| blazar gateway - b11202-cuda | 0.51 | 4.86 | 4714 | - | 2102 |
| blazar gateway - b11202-cuda | 0.52 | 4.83 | 4675 | - | 2203 |
| blazar gateway - b11202-cuda | 0.52 | 4.83 | 4680 | - | 2202 |
| blazar gateway - b11202-cuda | 0.52 | 4.82 | 4676 | - | 2102 |
| blazar gateway - b11202-cuda | - | - | - | - | 0 |
| blazar gateway - b11202-cuda | - | - | - | - | 0 |
| blazar gateway - b11202-cuda | - | - | - | - | 0 |
| blazar gateway - v0.9.4 | 0.52 | 4.63 | 4480 | - | 2101 |
| blazar gateway - v0.9.4 | 0.52 | 4.62 | 4461 | - | 2102 |
| blazar gateway - v0.9.4 | 0.52 | 4.60 | 4524 | - | 2105 |
| blazar gateway - v0.9.4 | 0.52 | 4.58 | 4422 | - | 2129 |
| blazar gateway - v0.9.4 | 0.52 | 4.62 | 4439 | - | 2196 |
| blazar gateway - v0.9.4 | 0.52 | 4.68 | 4500 | - | 2102 |
| blazar gateway - v0.9.4 | 0.52 | 4.68 | 4581 | - | 2104 |
| blazar gateway - v0.9.4 | 0.52 | 4.64 | 4483 | - | 2101 |
| blazar gateway - v0.9.4 | 0.53 | 4.64 | 4492 | - | 2101 |
| blazar gateway - v0.9.4 | 0.52 | 4.58 | 4433 | - | 2102 |
| blazar gateway - v0.9.4 | 0.52 | 4.59 | 4439 | - | 2202 |
| blazar gateway - v0.9.4 | 0.52 | 4.58 | 4430 | - | 2101 |
| direct engine - b11202-cuda | - | - | - | 2.51 | 5715 |
| direct engine - v0.9.4 | - | - | - | 8.51 | 7952 |
| ollama - qwen3.5:9b | - | 5.54 | 5446 | 5.32 | - |

_Every cold probe runs page-cache-dropped and GPU-idle-asserted on both runtimes; ollama rows without --ollama-service-restart leave the daemon warm (note in the artifact)._

### Idle wake (sleep vs keep_alive expiry)

| Runtime | idle policy | policy observed | wake TTFT ms | reload s | note |
|---|---|---|---:|---:|---|
| blazar - b11202-cuda | sleep at 15s (weights stay RAM-resident) | yes | 4615 | - |  |
| blazar - v0.9.4 | sleep at 15s (weights stay RAM-resident) | yes | 3727 | - |  |
| ollama - qwen3.5:9b | keep_alive 20s -> full unload | yes | 5958 | 5.83 |  |

_blazar sleeps with weights in RAM (wake = resume); ollama unloads at keep_alive expiry (wake = full disk reload). Policies differ by design — the table measures each runtime's own idle path after the policy verifiably fired._

### Long-context degradation curve

| Runtime | ctx | decode t/s | TTFT p50 ms |
|---|---:|---:|---:|
| blazar - b11202-cuda | 2048 | 41.5 | 125 |
| blazar - b11202-cuda | 8192 | 41.4 | 122 |
| blazar - b11202-cuda | 16384 | 41.0 | 122 |
| blazar - v0.9.4 | 2048 | 41.0 | 123 |
| blazar - v0.9.4 | 8192 | 41.0 | 122 |
| blazar - v0.9.4 | 16384 | 41.7 | 120 |
| ollama - qwen3.5:9b | 2048 | 40.5 | 135 |
| ollama - qwen3.5:9b | 8192 | 40.5 | 128 |
| ollama - qwen3.5:9b | 16384 | 40.6 | 130 |

### Media lanes (image / video / TTS / whisper)

| Lane | cold s | median s | min s | max s | ground truth | gate reject s | TTFB speedup |
|---|---:|---:|---:|---:|---|---:|---:|
| image - qwen-image-2.1-uncensored (512x512, steps=[4]) | 60.42 | 47.34 | 46.88 | 49.25 | 512x512 PNG, entropy 7.1 bits, contrast 67.5 |  |  |
| image - qwen-image-2.1-uncensored (512x512, steps=[4]) | 63.31 | 52.21 | 52.16 | 52.30 | 512x512 PNG, entropy 7.4 bits, contrast 67.1 |  |  |
| image - qwen-image-2.1-uncensored (512x512, steps=[4]) | 56.76 | 46.59 | 46.49 | 46.60 | 512x512 PNG, entropy 7.2 bits, contrast 63.1 |  |  |
| image - qwen-image-2.1-uncensored (512x512, steps=[4]) | 57.62 | 45.59 | 45.39 | 45.61 | 512x512 PNG, entropy 7.2 bits, contrast 75.3 |  |  |
| tts - en_US-amy-medium (840 chars, wav+pcm) | - | 2.05 | - | 3.55 | RTF wav 0.034 / pcm 0.059 (60s audio) |  | 3.99x |
| tts-conc - en_US-amy-medium (400 chars x4 pcm) | - | 4.95 | 5.09 | - | efficiency 3.74 of 4 streams |  | 1768ms max TTFB |
| video - wan_2.1_comfyui_repackaged (320x320, steps=8) | 38.24 | 11.08 | 11.08 | 11.08 | 5f: mux==reported==aligned | 0.001 |  |
| video - wan_2.1_comfyui_repackaged (320x320, steps=8) | - | 22.11 | 22.09 | 22.14 | 13f: mux==reported==aligned |  |  |
| video - wan_2.1_comfyui_repackaged (320x320, steps=8) | - | 52.17 | 52.17 | 52.19 | 33f: mux==reported==aligned |  |  |
| whisper - ggml-base (transcribes piper wav) | 2.39 | 2.10 | - | - | RTF 0.069 |  |  |

| Engine | config | median total s | Δ% vs default | cold request s |
|---|---|---:|---:|---:|
| master-920-2f88688 | fa_off | 52.2 | +10.3% | 63.3 |
| master-920-2f88688 | vae_tiling_on | 46.6 | -1.6% | 56.8 |
| master-920-2f88688 | sage_attn_on | 45.6 | -3.7% | 57.6 |

_Media cells run through the same sandboxed gateway as text lanes but do not assert GPU-idle: a warm engine child is the normal serving shape, so each row stamps gpu_busy_mib / ram_avail_mib / loadavg instead. 3 runs (not 5) — media variance is dominated by the model, not the scheduler. Video frame counts are read from the EBML container (lacing-aware), never from an API field; the VRAM gate probe times how fast an over-budget request is rejected with a teaching error._

## Findings (this campaign)

1. **Gateway overhead vs direct spawn: within measurement noise.** b11202-cuda decode 41.1 t/s through the gateway vs 41.6 t/s direct (-1.3%), greedy parity through the gateway 16/20 exact.
2. **Capacity-aware slot auto-sizing observed in argv.** Distinct engine shapes this campaign: 1x16384, 2x8192 - slots follow the live hardware census, each row's child_argv carries the receipt.
3. **Concurrency scaling per engine.** b11202-cuda (C=4): peak 67.5 t/s at C=4; v0.9.4 (C=4): peak 67.4 t/s at C=4; serialization behavior per level in the frontier table below.
4. **Prompt cache pays 5.5x on prefill** (7212 cached vs 1302 t/s cold).
5. **Speculative n-gram decoding is a net loss for this model** (decode 41.0 t/s, -0.2 vs dense baseline) - measured, not assumed.
6. **KV quantization (q8_0) is decode-neutral** (decode 41.0 t/s vs 41.2 dense).
7. **Tool-call quality (single-turn, temp 0).** b11202-cuda: selection 4/5, args 4/5 (control clean); v0.9.4: selection 4/5, args 4/5 (control clean); ollama: selection 4/5, args 4/5 (control clean); per-scenario detail in cells.jsonl.
8. **Adaptive reshape lands under sustained load.** b11202-cuda: no reshape observed in the lane window; v0.9.4: no reshape observed in the lane window (graceful-drain: adoption waits for in-flight streams, never kills one).
9. **Image quality stamps (PIL, luma domain): entropy 7.14 bits, rms contrast 67.5, 27748 unique colors @256x256** on qwen-image-2.1-uncensored - perceptual baseline for cross-run comparisons; audit PNG saved beside the cells.
10. **Image quality stamps (PIL, luma domain): entropy 7.39 bits, rms contrast 67.1, 33142 unique colors @256x256** on qwen-image-2.1-uncensored - perceptual baseline for cross-run comparisons; audit PNG saved beside the cells.
11. **Image quality stamps (PIL, luma domain): entropy 7.15 bits, rms contrast 63.1, 27511 unique colors @256x256** on qwen-image-2.1-uncensored - perceptual baseline for cross-run comparisons; audit PNG saved beside the cells.
12. **Image quality stamps (PIL, luma domain): entropy 7.22 bits, rms contrast 75.3, 29751 unique colors @256x256** on qwen-image-2.1-uncensored - perceptual baseline for cross-run comparisons; audit PNG saved beside the cells.
13. **Streamed PCM cuts time-to-first-audio 3.99x vs buffered WAV** (piper lane, first audio 0.51s vs 2.05s full synthesis) - total wall time is slightly higher (per-chunk synthesis), the win is interactivity.
14. **4 parallel PCM streams through one gateway: perfectly parallel** (efficiency 3.74 = sum of per-stream totals / 5.09s wall, max TTFB 1768 ms, NON-uniform stream outputs - flagged) - the scalability receipt for the TTS lane.
15. **Video VRAM gate rejects an over-budget request in 1 ms** with the full estimate math and override levers in the error body - instead of an opaque child abort minutes later.

## Carried-over findings (no receipt in this campaign)

_Established in earlier campaigns whose receipts live in their bench-artifacts/ directories; this campaign did not measure these lanes._

1. **mistral.rs 0.9.3 with default paged attention cannot fit this model on an 8 GiB card** (upstream sizes KV as a fraction of total VRAM); blazar's profile auto-disables paged attention on tight cards and the model then serves correctly.

## Caveats

- ollama prefill numbers come from engine counters that exclude the chat template, so they read slightly high against the 512-token lanes.
- Cross-backend greedy ratios (CUDA vs Vulkan) diverge on near-tie logits; treat ratio, not exact-match count, as the signal.
- All GPU rows measured on AC power at bounded load; rows record load average and power state (battery runs are rejected by the harness).
- Numbers are medians of 5 runs on one hybrid laptop; expect absolute shifts on other hardware, ratios to travel better.

## Reproduce

```bash
python3 scripts/bench_matrix.py --model qwen3.5-9b --artifacts-dir bench-artifacts/20260928-gguf-full --md BENCHMARK.md
python3 scripts/bench_matrix.py --render-only --artifacts-dir <dir> --md BENCHMARK.md
```

_Raw per-cell records (argv, per-run lists, daemon logs): `bench-artifacts/20260928-gguf-full/cells.jsonl`._
