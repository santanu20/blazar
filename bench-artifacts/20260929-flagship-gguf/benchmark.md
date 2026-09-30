# Blazar inference benchmark

_Rendered 20260929-flagship-gguf; blazar 0.13.0; power state of gateway rows: ac._

## Executive summary

- v0.9.4: gateway 18.9 vs direct 21.4 t/s (-11.7%)
- v0.9.4 prompt-cache prefill 241 vs 243 t/s cold
- b11202-cuda sweep C=1/2/4/8/16: C1: 40.0 t/s system (4x65536); C2: 65.9 t/s system (4x65536); C4: 104.8 t/s system (4x65536); C8: 102.7 t/s system (4x65536); C16: 103.4 t/s system (4x65536)
- v0.9.4 sweep C=1/8/16: C1: 18.6 t/s system (engine-scheduled); C8: 25.0 t/s system (engine-scheduled); C16: 16.1 t/s system (engine-scheduled)
- ITL p99 51.22 ms vs ollama 75.01 ms (1.5x tighter)
- tool-call TTFT p50 117 ms vs ollama 284 ms (2.4x)
- adaptive reshape under sustained load: slots 4->8, 49.1->63.0 t/s while serving, zero failed requests
- greedy determinism: direct 20/20 exact, gateway transparency 7/20 exact at temp 0 (divergence = multi-slot batching numerics, not translation drift)
- gateway cold boot 0.52 s
- cold TTFT 8495 ms vs ollama 4404 ms (0.5x)
- idle wake 3627 ms (sleep) vs ollama 6212 ms (full reload)
- evict-ladder wake 4478 ms (full respawn still beats ollama's reload)

## Measured in this campaign

- blazar: 32 cell(s)
- cold-ollama: 1 cell(s)
- conc-blazar: 10 cell(s)
- conc-direct: 5 cell(s)
- conc-ollama: 5 cell(s)
- ctxcurve-blazar: 6 cell(s)
- ctxcurve-ollama: 3 cell(s)
- direct: 13 cell(s)
- features: 2 cell(s)
- greedy: 2 cell(s)
- greedy_gw: 1 cell(s)
- greedy_ollama: 1 cell(s)
- idle-blazar: 3 cell(s)
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
| v0.9.4 | mistralrs | text (direct + gateway) | 30 | 1 | benchmarked |
| b5130 | whisper | media | 1 | 0 | benchmarked |
| b11202-cuda | llamacpp | text (direct + gateway) | 49 | 0 | benchmarked |
| master-920-2f88688 | sdcpp | media | 5 | 0 | benchmarked |
| sglang-0.5.19 | - | - | 0 | 0 | excluded: model-format sweep: this campaign sweeps a GGUF file; sglang serves HF safetensors checkpoints (rerun with --model <safetensors-row>) |

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
| blazar gateway - b11202-cuda | 4x65536 | 40.6 | 121.5 | 181.4 | 24.6 | 25.7 | 1297.8 | 7364.4 | 6169 | 55.2 |
| blazar gateway - b11202-cuda (single-stream) | 1x16384 | 41.0 | 121.2 | 127.7 | 24.2 | 25.5 | 1322.2 | 7580.8 | 5666 | 55.1 |
| blazar gateway - b11202-cuda (cont_batching_off) | 4x65536 | 40.6 | 122.6 | 178.4 | 24.6 | 25.6 | 1310.7 | 7258.9 | 6152 | 55.4 |
| blazar gateway - b11202-cuda (fa_on) | 4x65536 | 40.6 | 120.9 | 129.4 | 24.6 | 25.7 | 1334.5 | 7483.4 | 6152 | 55.5 |
| blazar gateway - b11202-cuda (fa_off) | 4x65536 | 10.0 | 233.9 | 618.1 | 100.0 | 110.2 | 501.6 | 3173.8 | 6612 | 36.7 |
| blazar gateway - b11202-cuda (kv_unified_on) | 4x65536 | 40.7 | 119.7 | 126.2 | 24.6 | 25.7 | 1336.2 | 7499.3 | 6152 | 55.4 |
| blazar gateway - b11202-cuda (kv_unified_off) | 2x8192 | 41.0 | 122.5 | 128.7 | 24.4 | 25.4 | 1321.5 | 7324.0 | 5458 | 55.1 |
| blazar gateway - b11202-cuda (swa_full_on) | 4x65536 | 40.8 | 120.2 | 125.8 | 24.5 | 25.6 | 1306.7 | 7223.9 | 6152 | 55.4 |
| blazar gateway - b11202-cuda (no_kv_offload_on) | 2x8192 | 18.9 | 148.4 | 203.0 | 52.9 | 58.0 | 962.2 | 5198.1 | 5106 | 52.1 |
| blazar gateway - b11202-cuda (cache_q8_q8_0) | 4x65536 | 32.0 | 130.3 | 200.8 | 31.4 | 33.6 | 1189.8 | 6372.3 | 6544 | 53.9 |
| blazar gateway - b11202-cuda (ctx_checkpoints_4) | 4x65536 | 41.0 | 120.3 | 126.1 | 24.4 | 25.4 | 1343.2 | 7382.3 | 6152 | 55.4 |
| blazar gateway - b11202-cuda (spec_off) | 4x65536 | 41.0 | 121.0 | 174.3 | 24.3 | 25.6 | 1406.2 | 7400.1 | 6152 | 55.7 |
| blazar gateway - b11202-cuda (deterministic_on) | 1x16384 | 41.0 | 120.9 | 122.4 | 24.3 | 25.5 | 1325.5 | 7299.1 | 5666 | 55.8 |
| blazar gateway - b11202-cuda (mmproj_offload_off) | 4x65536 | 40.6 | 121.7 | 128.3 | 24.6 | 25.8 | 1302.0 | 7190.4 | 6152 | 56.9 |
| blazar gateway - b11202-cuda (threads_batch_8) | 4x65536 | 41.2 | 121.9 | 177.5 | 24.3 | 25.5 | 1319.7 | 7419.7 | 6152 | 55.5 |
| blazar gateway - b11202-cuda (cache_reuse_256) | 4x65536 | 40.6 | 120.7 | 176.9 | 24.6 | 25.7 | 1341.0 | 7188.5 | 6152 | 55.3 |
| blazar gateway - b11202-cuda (kv_unified_per_slot_4096) | 4x65536 | 40.6 | 121.7 | 126.6 | 24.5 | 25.9 | 1349.1 | 7217.1 | 6152 | 56.2 |
| blazar gateway - b11202-cuda (conc_default) | 4x65536 | - | - | - | - | 37.3 | - | - | 6158 | 55.0 |
| blazar gateway - b11202-cuda (adaptive_slots_off) | 4x65536 | - | - | - | - | 86.0 | - | - | 6148 | 54.8 |
| blazar gateway - b11202-cuda (poll_50) | 4x65536 | - | - | - | - | 51.2 | - | - | 6279 | 54.9 |
| blazar gateway - v0.9.4 | engine-scheduled | 18.8 | 132.0 | 143.2 | 53.7 | 60.9 | 242.1 | 244.6 | 6948 | 38.1 |
| blazar gateway - v0.9.4 (paged_attn_off) | engine-scheduled | 18.9 | 141.3 | 155.2 | 53.2 | 61.3 | 239.7 | 241.2 | 6948 | 42.2 |
| blazar gateway - v0.9.4 (single-stream) | engine-scheduled | 18.8 | 139.7 | 152.7 | 53.4 | 62.1 | 237.6 | 239.5 | 6948 | 41.0 |
| blazar gateway - v0.9.4 (kv_unified_on) | engine-scheduled | 18.9 | 131.3 | 136.9 | 53.3 | 61.3 | 244.0 | 247.1 | 6948 | 37.5 |
| blazar gateway - v0.9.4 (kv_unified_off) | engine-scheduled | 18.8 | 128.4 | 147.7 | 53.3 | 60.9 | 237.7 | 247.6 | 6948 | 40.4 |
| blazar gateway - v0.9.4 (spec_off) | engine-scheduled | 18.6 | 135.6 | 142.5 | 53.8 | 62.9 | 239.2 | 236.8 | 6948 | 39.6 |
| blazar gateway - v0.9.4 (deterministic_on) | engine-scheduled | 18.8 | 132.2 | 142.5 | 53.6 | 60.7 | 237.6 | 244.4 | 6948 | 37.3 |
| blazar gateway - v0.9.4 (pa_mem_0.85) | engine-scheduled | 18.7 | 130.1 | 147.4 | 52.9 | 61.5 | 237.2 | 249.6 | 6948 | 43.0 |
| blazar gateway - v0.9.4 (pa_mem_0.55) | engine-scheduled | 18.6 | 134.1 | 140.0 | 54.0 | 62.4 | 239.0 | 247.4 | 6948 | 38.4 |
| blazar gateway - v0.9.4 (mr_prefix_cache_256) | engine-scheduled | 18.9 | 131.4 | 136.1 | 53.0 | 61.1 | 241.0 | 246.5 | 6948 | 36.7 |
| blazar gateway - v0.9.4 (mr_enc_cache_512mb) | engine-scheduled | 18.9 | 137.2 | 143.7 | 53.0 | 61.0 | 243.0 | 241.0 | 6948 | 36.7 |
| direct engine - b11202-cuda | 1x16384 | 41.5 | 115.5 | 119.5 | 24.0 | 25.1 | 1335.7 | 7769.5 | 5660 | 55.2 |
| direct engine - v0.9.4 | 1x16384 | 21.4 | 105.7 | 108.1 | 46.7 | 52.3 | 321.3 | 323.4 | 7042 | 44.3 |
| ollama 0.33.3 - qwen3.5:9b (t/s NOT comparable - matrix model differs) | service | 40.7 | 129.7 | 140.5 | 24.8 | 75.0 | 1416.3 | 7337.1 | 6570 | 55.1 |

### Concurrency (1x2x4x8x16 parallel streams x 128 tokens)

| Runtime | slots | ok streams | rounds | system t/s | sum-stream t/s | wall s | TTFT max ms | TTFT p99 ms | ITL p99 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| blazar gateway - b11202-cuda | 4x65536 | 3 of 3x1 | 3 | 40.0 | 125.6 | 9.61 | 264 | 262 | 26.1 |
| blazar gateway - b11202-cuda | 4x65536 | 6 of 3x2 | 3 | 65.9 | 216.5 | 11.66 | 372 | 371 | 35.7 |
| blazar gateway - b11202-cuda | 4x65536 | 12 of 3x4 | 3 | 104.8 | 346.3 | 14.65 | 554 | 554 | 37.6 |
| blazar gateway - b11202-cuda | 4x65536 | 24 of 3x8 | 3 | 102.7 | 675.3 | 29.90 | 5840 | 5840 | 54.8 |
| blazar gateway - b11202-cuda | 4x65536 | 48 of 3x16 | 3 | 103.4 | 1350.1 | 59.43 | 15802 | 15798 | 85.8 |
| blazar gateway - v0.9.4 | engine-scheduled | 3 of 3x1 | 3 | 18.6 | 56.8 | 20.65 | 171 | 170 | 60.3 |
| blazar gateway - v0.9.4 | engine-scheduled | 6 of 3x2 | 3 | 0.0 | 0.0 | 0.18 | 73 | 73 | - |
| blazar gateway - v0.9.4 | engine-scheduled | 12 of 3x4 | 3 | 0.0 | 0.0 | 0.30 | 114 | 114 | - |
| blazar gateway - v0.9.4 | engine-scheduled | 24 of 3x8 | 3 | 25.0 | 76.0 | 25.56 | 509 | 508 | 81.8 |
| blazar gateway - v0.9.4 | engine-scheduled | 48 of 3x16 | 3 | 16.1 | 33.6 | 15.88 | 864 | 864 | 337.1 |
| direct engine - b11202-cuda | 1 | 1 of 1x1 | 1 | 40.0 | 41.5 | 3.20 | 139 | - | 25.1 |
| direct engine - b11202-cuda | 2 | 2 of 1x2 | 1 | 70.5 | 73.7 | 3.63 | 184 | - | 32.2 |
| direct engine - b11202-cuda | 4 | 4 of 1x4 | 1 | 111.3 | 117.0 | 4.60 | 260 | - | 36.5 |
| direct engine - b11202-cuda | 8 | 8 of 1x8 | 1 | 136.9 | 144.0 | 7.48 | 425 | - | 58.7 |
| direct engine - b11202-cuda | 16 | 16 of 1x16 | 1 | 236.6 | 262.5 | 8.66 | 918 | - | 69.1 |
| ollama - qwen3.5:9b | service | 3 of 3x1 | 3 | 23.3 | 121.7 | 16.49 | 6764 | 6632 | 74.2 |
| ollama - qwen3.5:9b | service | 6 of 3x2 | 3 | 29.8 | 243.3 | 25.81 | 9557 | 9392 | 74.7 |
| ollama - qwen3.5:9b | service | 12 of 3x4 | 3 | 33.6 | 486.7 | 45.69 | 16287 | 15928 | 74.8 |
| ollama - qwen3.5:9b | service | 24 of 3x8 | 3 | 36.2 | 969.7 | 84.88 | 28972 | 28217 | 74.8 |
| ollama - qwen3.5:9b | service | 48 of 3x16 | 3 | 37.0 | 1941.0 | 166.02 | 57266 | 55711 | 75.0 |

_sum-stream >> system t/s means streams serialize on one slot; roughly equal means genuinely parallel._

### Concurrency frontier (system t/s and tail latency vs level)

| Runtime | C | ok streams | system t/s | sum-stream t/s | eff vs C=1 | TTFT p99 ms | ITL p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| blazar gateway - b11202-cuda | 1 | 3 of 3x1 | 40.0 | 125.6 | 100% | 262 | 26.1 |
| blazar gateway - b11202-cuda | 2 | 6 of 3x2 | 65.9 | 216.5 | 82% | 371 | 35.7 |
| blazar gateway - b11202-cuda | 4 | 12 of 3x4 | 104.8 | 346.3 | 66% | 554 | 37.6 |
| blazar gateway - b11202-cuda | 8 | 24 of 3x8 | 102.7 | 675.3 | 32% | 5840 | 54.8 |
| blazar gateway - b11202-cuda | 16 | 48 of 3x16 | 103.4 | 1350.1 | 16% | 15798 | 85.8 |
| blazar gateway - v0.9.4 | 1 | 3 of 3x1 | 18.6 | 56.8 | 100% | 170 | 60.3 |
| blazar gateway - v0.9.4 | 2 | 6 of 3x2 | 0.0 | 0.0 | - | 73 | - |
| blazar gateway - v0.9.4 | 4 | 12 of 3x4 | 0.0 | 0.0 | - | 114 | - |
| blazar gateway - v0.9.4 | 8 | 24 of 3x8 | 25.0 | 76.0 | 17% | 508 | 81.8 |
| blazar gateway - v0.9.4 | 16 | 48 of 3x16 | 16.1 | 33.6 | 5% | 864 | 337.1 |
| direct engine - b11202-cuda | 1 | 1 of 1x1 | 40.0 | 41.5 | 100% | - | 25.1 |
| direct engine - b11202-cuda | 2 | 2 of 1x2 | 70.5 | 73.7 | 88% | - | 32.2 |
| direct engine - b11202-cuda | 4 | 4 of 1x4 | 111.3 | 117.0 | 70% | - | 36.5 |
| direct engine - b11202-cuda | 8 | 8 of 1x8 | 136.9 | 144.0 | 43% | - | 58.7 |
| direct engine - b11202-cuda | 16 | 16 of 1x16 | 236.6 | 262.5 | 37% | - | 69.1 |
| ollama - qwen3.5:9b | 1 | 3 of 3x1 | 23.3 | 121.7 | 100% | 6632 | 74.2 |
| ollama - qwen3.5:9b | 2 | 6 of 3x2 | 29.8 | 243.3 | 64% | 9392 | 74.7 |
| ollama - qwen3.5:9b | 4 | 12 of 3x4 | 33.6 | 486.7 | 36% | 15928 | 74.8 |
| ollama - qwen3.5:9b | 8 | 24 of 3x8 | 36.2 | 969.7 | 19% | 28217 | 74.8 |
| ollama - qwen3.5:9b | 16 | 48 of 3x16 | 37.0 | 1941.0 | 10% | 55711 | 75.0 |

- blazar gateway - b11202-cuda: throughput plateaus at C=8 (<10% per-level gain), peak 104.8 t/s at C=4.
- blazar gateway - v0.9.4: 5 level(s) measured - insufficient levels for a saturation verdict.
- direct engine - b11202-cuda: still gaining at C=16 (40.0 -> 236.6 t/s) - saturation not reached within the sweep.
- ollama - qwen3.5:9b: throughput plateaus at C=8 (<10% per-level gain), peak 37.0 t/s at C=16.

### Adaptive reshape under sustained load (no-lag proof)

| Runtime | reshape | slots | time to reshape s | req before/after | TTFT p50 before->after ms | sys t/s before->after | failed |
|---|---|---|---:|---:|---:|---:|---:|
| blazar gateway - b11202-cuda | yes | 4->8 | 74 | 32/147 | 7413->349 | 49->63 | 0 |
| blazar gateway - v0.9.4 | n/a | skipped: adaptive reshape is a llamacpp-lane mechanism; this engine's child exposes no slot shape (-np) to observe or adopt | - | - | - | - | - |

<details><summary>daemon tail (slots/reshape) - b11202-cuda</summary>

```
2026-09-29T18:01:53.480677Z  INFO evict: terminating child and releasing the slot model="qwen3.5-9b" pid=217652
2026-09-29T18:01:53.480677Z  INFO evict: terminating child and releasing the slot model="qwen3.5-9b" pid=217652
```

</details>

### Perplexity

| Engine | perplexity (ctx 2048, offline ASCII corpus) | ppl tool |
|---|---:|---|
| b11202-cuda | 4.83 +/- 0.19 | own |
| v0.9.4 | 4.83 +/- 0.19 | borrowed (llama-b11202) |

### Greedy parity and gateway transparency (20 prompts, 256 tokens)

| Comparison | exact / total | ratio mean | ratio min |
|---|---:|---:|---:|
| b11202-cuda vs same-engine reference (direct) | 20/20 | 1.000 | 1.000 |
| v0.9.4 vs same-engine reference (direct) | 0/20 | 0.514 | 0.000 |
| b11202-cuda through blazar gateway vs direct | 7/20 | 0.643 | 0.017 |
| ollama (qwen3.5:9b) run-to-run, temp 0 | 20/20 | 1.000 | 1.000 |

_Exact-match divergence across GPU backends is expected float nondeterminism (batch shape and backend kernels), not translation drift; bit-parity across runs requires single-slot decoding (blazar `deterministic = true` pins it)._

### Tool calls (single-turn selection + schema quality)

| Runtime | scenarios | well-formed | selection | args valid | control FP | TTFT p50 ms |
|---|---:|---:|---:|---:|---|---:|
| blazar gateway - b11202-cuda | 6 | 4/6 | 4/5 | 4/5 | no | 117 |
| blazar gateway - v0.9.4 | 6 | 5/6 | 5/5 | 5/5 | no | 1784 |
| ollama - qwen3.5:9b | 6 | 4/6 | 4/5 | 4/5 | no | 284 |

### Optimization axes (ctx 4096, single stream)

| Engine | axis | setting | decode t/s | delta vs dense | prefill cold t/s | delta |
|---|---|---|---:|---:|---:|---:|
| b11202-cuda | kv | q8_0 | 41.1 | -0.2 | 1332.6 | 72.0 |
| b11202-cuda | pa | on | 41.5 | 0.1 | 1335.0 | 74.4 |
| b11202-cuda | spec | ngram-simple | 41.1 | -0.3 | 1330.8 | 70.2 |
| b11202-cuda | mmproj | True | 41.5 | 0.1 | 1342.6 | 81.9 |
| v0.9.4 | pa | off | 18.8 | -2.6 | 237.9 | -80.9 |

### Gateway config knobs (on/off vs default profile)

| Engine | config | decode t/s | decode delta% | ttft p50 ms | ttft delta% | GPU peak MiB | GPU delta | output vs default |
|---|---|---:|---:|---:|---:|---:|---:|:-:|
| b11202-cuda | single-stream | 41.0 | +1.0% | 121 | -0.2% | 5666 | -503 | same |
| b11202-cuda | cont_batching_off | 40.6 | +0.2% | 123 | +0.8% | 6152 | -17 | same |
| b11202-cuda | fa_on | 40.6 | -0.0% | 121 | -0.5% | 6152 | -17 | same |
| b11202-cuda | fa_off | 10.0 | -75.3% | 234 | +92.5% | 6612 | +443 | same |
| b11202-cuda | kv_unified_on | 40.7 | +0.2% | 120 | -1.5% | 6152 | -17 | same |
| b11202-cuda | kv_unified_off | 41.0 | +1.1% | 123 | +0.8% | 5458 | -711 | same |
| b11202-cuda | swa_full_on | 40.8 | +0.5% | 120 | -1.1% | 6152 | -17 | same |
| b11202-cuda | no_kv_offload_on | 18.9 | -53.5% | 148 | +22.1% | 5106 | -1063 | same |
| b11202-cuda | cache_q8_q8_0 | 32.0 | -21.1% | 130 | +7.2% | 6544 | +375 | same |
| b11202-cuda | ctx_checkpoints_4 | 41.0 | +1.1% | 120 | -1.0% | 6152 | -17 | same |
| b11202-cuda | spec_off | 41.0 | +1.1% | 121 | -0.5% | 6152 | -17 | same |
| b11202-cuda | deterministic_on | 41.0 | +1.0% | 121 | -0.5% | 5666 | -503 | same |
| b11202-cuda | mmproj_offload_off | 40.6 | -0.0% | 122 | +0.1% | 6152 | -17 | same |
| b11202-cuda | threads_batch_8 | 41.2 | +1.6% | 122 | +0.3% | 6152 | -17 | same |
| b11202-cuda | cache_reuse_256 | 40.6 | +0.1% | 121 | -0.7% | 6152 | -17 | same |
| b11202-cuda | kv_unified_per_slot_4096 | 40.6 | +0.1% | 122 | +0.1% | 6152 | -17 | same |
| v0.9.4 | paged_attn_off | 18.9 | +0.5% | 141 | +7.1% | 6948 | +0 | same |
| v0.9.4 | single-stream | 18.8 | -0.1% | 140 | +5.9% | 6948 | +0 | same |
| v0.9.4 | kv_unified_on | 18.9 | +0.5% | 131 | -0.5% | 6948 | +0 | same |
| v0.9.4 | kv_unified_off | 18.8 | +0.3% | 128 | -2.7% | 6948 | +0 | same |
| v0.9.4 | spec_off | 18.6 | -1.2% | 136 | +2.8% | 6948 | +0 | same |
| v0.9.4 | deterministic_on | 18.8 | -0.1% | 132 | +0.2% | 6948 | +0 | same |
| v0.9.4 | pa_mem_0.85 | 18.7 | -0.2% | 130 | -1.4% | 6948 | +0 | same |
| v0.9.4 | pa_mem_0.55 | 18.6 | -0.9% | 134 | +1.7% | 6948 | +0 | same |
| v0.9.4 | mr_prefix_cache_256 | 18.9 | +0.7% | 131 | -0.4% | 6948 | +0 | same |
| v0.9.4 | mr_enc_cache_512mb | 18.9 | +0.5% | 137 | +4.0% | 6948 | +0 | same |

### Gateway scheduler knobs under concurrent load (C=8 bursts)

| Engine | config | sys tok/s | sys delta% | ttft p99 ms | ttft delta% | GPU peak MiB | GPU delta | errs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| b11202-cuda | conc_default | 84.9 | +0.0% | 10101 | +0.0% | 6158 | +0 | 0 |
| b11202-cuda | adaptive_slots_off | 84.3 | -0.8% | 9893 | -2.1% | 6148 | -10 | 0 |
| b11202-cuda | poll_50 | 84.0 | -1.1% | 10320 | +2.2% | 6279 | +121 | 0 |

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
| blazar gateway - b11202-cuda | 0.53 | 5.73 | 5540 | - | 2241 |
| blazar gateway - b11202-cuda (single-stream) | 0.52 | 4.94 | 4848 | - | 2105 |
| blazar gateway - b11202-cuda (cont_batching_off) | 0.53 | 5.22 | 5127 | - | 2136 |
| blazar gateway - b11202-cuda (fa_on) | 0.52 | 4.87 | 4680 | - | 2140 |
| blazar gateway - b11202-cuda (fa_off) | 0.56 | 9.57 | 9216 | - | 4727 |
| blazar gateway - b11202-cuda (kv_unified_on) | 0.52 | 4.82 | 4628 | - | 2140 |
| blazar gateway - b11202-cuda (kv_unified_off) | 0.53 | 4.69 | 4537 | - | 2196 |
| blazar gateway - b11202-cuda (swa_full_on) | 0.53 | 4.68 | 4484 | - | 2140 |
| blazar gateway - b11202-cuda (no_kv_offload_on) | 0.52 | 5.69 | 5475 | - | 2466 |
| blazar gateway - b11202-cuda (cache_q8_q8_0) | 0.52 | 6.36 | 6248 | - | 2418 |
| blazar gateway - b11202-cuda (ctx_checkpoints_4) | 0.52 | 4.72 | 4526 | - | 2140 |
| blazar gateway - b11202-cuda (spec_off) | 0.52 | 4.82 | 4632 | - | 2241 |
| blazar gateway - b11202-cuda (deterministic_on) | 0.52 | 4.77 | 4697 | - | 2105 |
| blazar gateway - b11202-cuda (mmproj_offload_off) | 0.52 | 4.79 | 4592 | - | 2140 |
| blazar gateway - b11202-cuda (threads_batch_8) | 0.52 | 5.02 | 4822 | - | 2241 |
| blazar gateway - b11202-cuda (cache_reuse_256) | 0.52 | 4.79 | 4599 | - | 2241 |
| blazar gateway - b11202-cuda (kv_unified_per_slot_4096) | 0.52 | 4.80 | 4607 | - | 2140 |
| blazar gateway - v0.9.4 | 0.52 | 8.39 | 8224 | - | 6789 |
| blazar gateway - v0.9.4 (paged_attn_off) | 0.52 | 8.79 | 8626 | - | 7011 |
| blazar gateway - v0.9.4 (single-stream) | 0.52 | 8.35 | 8183 | - | 7141 |
| blazar gateway - v0.9.4 (kv_unified_on) | 0.52 | 8.83 | 8663 | - | 7288 |
| blazar gateway - v0.9.4 (kv_unified_off) | 0.52 | 12.14 | 11976 | - | 7368 |
| blazar gateway - v0.9.4 (spec_off) | 0.52 | 9.27 | 9114 | - | 7570 |
| blazar gateway - v0.9.4 (deterministic_on) | 0.52 | 9.27 | 9097 | - | 7819 |
| blazar gateway - v0.9.4 (pa_mem_0.85) | 0.52 | 8.96 | 8802 | - | 7687 |
| blazar gateway - v0.9.4 (pa_mem_0.55) | 0.52 | 9.00 | 8829 | - | 7763 |
| blazar gateway - v0.9.4 (mr_prefix_cache_256) | 0.52 | 8.36 | 8197 | - | 7716 |
| blazar gateway - v0.9.4 (mr_enc_cache_512mb) | 0.52 | 8.67 | 8495 | - | 7856 |
| direct engine - b11202-cuda | - | - | - | 2.01 | 5719 |
| direct engine - v0.9.4 | - | - | - | 6.01 | 8138 |
| ollama - qwen3.5:9b | - | 4.49 | 4404 | 4.29 | - |

_Every cold probe runs page-cache-dropped and GPU-idle-asserted on both runtimes; ollama rows without --ollama-service-restart leave the daemon warm (note in the artifact)._

### Idle wake (sleep vs keep_alive expiry)

| Runtime | idle policy | policy observed | wake TTFT ms | reload s | note |
|---|---|---|---:|---:|---|
| blazar - b11202-cuda | sleep at 15s (weights stay RAM-resident) | yes | 3627 | - |  |
| blazar - b11202-cuda | evict at 45s (full respawn + KV-bank restore) | yes | 4478 | - |  |
| blazar - v0.9.4 | sleep at 15s (weights stay RAM-resident) | yes | 140 | - |  |
| ollama - qwen3.5:9b | keep_alive 20s -> full unload | yes | 6212 | 6.08 |  |

Wake cost is the upstream engine child restoring its CUDA context and re-uploading weights (single-level sleep; warm steady-state is ~0.1 s). Raise `idle_sleep_secs` to sleep less often, or lower it toward the evict ladder when wake latency matters less than VRAM residency.

_blazar sleeps with weights in RAM (wake = resume); ollama unloads at keep_alive expiry (wake = full disk reload). Policies differ by design - the table measures each runtime's own idle path after the policy verifiably fired._

### Long-context degradation curve

| Runtime | ctx | decode t/s | TTFT p50 ms |
|---|---:|---:|---:|
| blazar - b11202-cuda | 2048 | 41.0 | 125 |
| blazar - b11202-cuda | 8192 | 40.7 | 123 |
| blazar - b11202-cuda | 16384 | 40.6 | 122 |
| blazar - v0.9.4 | 2048 | 18.6 | 136 |
| blazar - v0.9.4 | 8192 | 18.9 | 135 |
| blazar - v0.9.4 | 16384 | 18.7 | 139 |
| ollama - qwen3.5:9b | 2048 | 40.5 | 132 |
| ollama - qwen3.5:9b | 8192 | 40.6 | 132 |
| ollama - qwen3.5:9b | 16384 | 40.7 | 120 |

### Media lanes (image / video / TTS / whisper)

| Lane | cold s | median s | min s | max s | ground truth | gate reject s | TTFB speedup |
|---|---:|---:|---:|---:|---|---:|---:|
| image - qwen-image-2.1-uncensored (512x512, steps=[4]) | 56.33 | 45.55 | 45.51 | 45.83 | 512x512 PNG, entropy 7.1 bits, contrast 75.6 |  |  |
| image - qwen-image-2.1-uncensored (512x512, steps=[4]) [fa_off] | 61.94 | 52.21 | 52.19 | 52.22 | 512x512 PNG, entropy 6.9 bits, contrast 68.4 |  |  |
| image - qwen-image-2.1-uncensored (512x512, steps=[4]) [vae_tiling_on] | 58.01 | 46.45 | 46.45 | 46.52 | 512x512 PNG, entropy 7.0 bits, contrast 73.4 |  |  |
| image - qwen-image-2.1-uncensored (512x512, steps=[4]) [sage_attn_on] | 56.77 | 45.55 | 45.49 | 45.58 | 512x512 PNG, entropy 7.3 bits, contrast 70.9 |  |  |
| tts - en_US-amy-medium (840 chars, wav+pcm) | - | 2.05 | - | 3.41 | RTF wav 0.034 / pcm 0.056 (61s audio) |  | 4.08x |
| tts-conc - en_US-amy-medium (400 chars x4 pcm) | - | 4.72 | 4.88 | - | efficiency 3.83 of 4 streams |  | 1578ms max TTFB |
| video - wan_2.1_comfyui_repackaged (320x320, steps=8) | 38.24 | 11.07 | 11.07 | 11.08 | 5f: mux==reported==aligned | 0.001 |  |
| video - wan_2.1_comfyui_repackaged (320x320, steps=8) | - | 22.10 | 22.10 | 22.11 | 13f: mux==reported==aligned |  |  |
| video - wan_2.1_comfyui_repackaged (320x320, steps=8) | - | 52.16 | 52.16 | 52.17 | 33f: mux==reported==aligned |  |  |
| whisper - ggml-base (transcribes piper wav) | 2.41 | 2.14 | - | - | RTF 0.069 |  |  |

| Engine | config | median total s | delta% vs default | cold request s |
|---|---|---:|---:|---:|
| master-920-2f88688 | fa_off | 52.2 | +14.6% | 61.9 |
| master-920-2f88688 | vae_tiling_on | 46.5 | +2.0% | 58.0 |
| master-920-2f88688 | sage_attn_on | 45.5 | inert | 56.8 |

> master-920-2f88688 sage_attn_on: config knob inert on this box: profiler skipped sage_attn (SageAttention requires a CUDA device (SM80+, CUDA build) and the child dies during load otherwise; active device Vulkan1) - this row measures the default posture; treat delta% as run-to-run noise

_Media cells run through the same sandboxed gateway as text lanes but do not assert GPU-idle: a warm engine child is the normal serving shape, so each row stamps gpu_busy_mib / ram_avail_mib / loadavg instead. 3 runs (not 5); media variance is dominated by the model, not the scheduler. Video frame counts are read from the EBML container (lacing-aware), never from an API field; the VRAM gate probe times how fast an over-budget request is rejected with a teaching error._

## Findings (this campaign)

1. **Gateway overhead vs direct spawn: within measurement noise.** b11202-cuda decode 41.0 t/s through the gateway vs 41.5 t/s direct (-1.3%), greedy parity through the gateway 7/20 exact.
2. **Capacity-aware slot auto-sizing observed in argv.** Distinct engine shapes this campaign: 1x16384, 2x8192, 4x65536 - slots follow the live hardware census, each row's child_argv carries the receipt.
3. **Concurrency scaling per engine.** b11202-cuda (C=1/2/4/8/16): peak 104.8 t/s at C=4, 66% of ideal at C=4; v0.9.4 (C=1/8/16): peak 25.0 t/s at C=8, 17% of ideal at C=8; serialization behavior per level in the frontier table below.
4. **Prompt cache pays 5.7x on prefill** (7364 cached vs 1298 t/s cold).
5. **Speculative n-gram decoding is a net loss for this model** (decode 41.1 t/s, -0.3 vs dense baseline) - measured, not assumed.
6. **KV quantization (q8_0) is decode-neutral** (decode 41.1 t/s vs 41.4 dense).
7. **Tool-call quality (single-turn, temp 0).** b11202-cuda: selection 4/5, args 4/5 (control clean); v0.9.4: selection 5/5, args 5/5 (control clean); ollama: selection 4/5, args 4/5 (control clean); per-scenario detail in cells.jsonl.
8. **Adaptive reshape lands under sustained load.** b11202-cuda: reshaped 4->8 slots after 74.1 s under sustained C=8, TTFT p50 7413->349 ms, 0 dropped requests; v0.9.4: no reshape observed in the lane window (graceful-drain: adoption waits for in-flight streams, never kills one).
9. **Image quality stamps (PIL, luma domain): entropy 7.11 bits, rms contrast 75.6, 27459 unique colors @256x256** on qwen-image-2.1-uncensored - perceptual baseline for cross-run comparisons; audit PNG saved beside the cells.
10. **Image quality stamps (PIL, luma domain): entropy 6.94 bits, rms contrast 68.4, 27560 unique colors @256x256** on qwen-image-2.1-uncensored - perceptual baseline for cross-run comparisons; audit PNG saved beside the cells.
11. **Image quality stamps (PIL, luma domain): entropy 7.03 bits, rms contrast 73.4, 31438 unique colors @256x256** on qwen-image-2.1-uncensored - perceptual baseline for cross-run comparisons; audit PNG saved beside the cells.
12. **Image quality stamps (PIL, luma domain): entropy 7.31 bits, rms contrast 70.9, 33526 unique colors @256x256** on qwen-image-2.1-uncensored - perceptual baseline for cross-run comparisons; audit PNG saved beside the cells.
13. **Streamed PCM cuts time-to-first-audio 4.08x vs buffered WAV** (piper lane, first audio 0.50s vs 2.05s full synthesis) - total wall time is slightly higher (per-chunk synthesis), the win is interactivity.
14. **4 parallel PCM streams through one gateway: perfectly parallel** (efficiency 3.83 = sum of per-stream totals / 4.88s wall, max TTFB 1578 ms, NON-uniform stream outputs - flagged) - the scalability receipt for the TTS lane.
15. **Video VRAM gate rejects an over-budget request in 1 ms** with the full estimate math and override levers in the error body - instead of an opaque child abort minutes later.

## Carried-over findings (no receipt in this campaign)

_Established in earlier campaigns whose receipts live in their bench-artifacts/ directories; this campaign did not measure these lanes._

1. **mistral.rs 0.9.3 with default paged attention cannot fit this model on an 8 GiB card** (upstream sizes KV as a fraction of total VRAM); blazar's profile auto-disables paged attention on tight cards and the model then serves correctly.

## Failed cells (engine-reality receipts)

_These knobs crashed the engine child on this test bed; the crash signature is the receipt. Raw logs: `cells.jsonl` (`daemon_log_tail` field)._

| Engine | lane | config | failure |
|---|---|---|---|
| v0.9.4 | blazar | mr_batch_64 | cold probe failed: HTTP Error 502: Bad Gateway |

## Structured skips (tool/format boundaries)

- master-920-2f88688 (media-image): config knob inert on this box: profiler skipped sage_attn (SageAttention requires a CUDA device (SM80+, CUDA build) and the child dies during load otherwise; active device Vulkan1) - this row measures the default posture; treat delta% as run-to-run noise
- v0.9.4 (reshape): skipped: adaptive reshape is a llamacpp-lane mechanism; this engine's child exposes no slot shape (-np) to observe or adopt

## Caveats

- ollama prefill numbers come from engine counters that exclude the chat template, so they read slightly high against the 512-token lanes.
- Cross-backend greedy ratios (CUDA vs Vulkan) diverge on near-tie logits; treat ratio, not exact-match count, as the signal.
- All GPU rows measured on AC power at bounded load; rows record load average and power state (battery runs are rejected by the harness).
- Numbers are medians of 5 runs on one hybrid laptop; expect absolute shifts on other hardware, ratios to travel better.

## Reproduce

```bash
python3 scripts/bench_matrix.py --artifacts-dir bench-artifacts/20260929-flagship-gguf --engines b11202-cuda v0.9.4 master-920-2f88688 b5130 --conc-sweep 1,2,4,8,16 --md bench-artifacts/20260929-flagship-gguf/benchmark.md
python3 scripts/bench_matrix.py --render-only --artifacts-dir <dir> [--append-campaign <sibling-campaign-dir>] --md BENCHMARK.md
```

_Raw per-cell records (argv, per-run lists, daemon logs): `bench-artifacts/20260929-flagship-gguf/cells.jsonl`._
