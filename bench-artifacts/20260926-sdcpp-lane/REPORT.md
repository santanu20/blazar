# Blazar inference benchmark

_Rendered 20260926-sdcpp-lane; blazar 0.11.0; power state of gateway rows: ac._

## Executive summary

b11193-cuda: gateway 41.5 vs direct 41.4 t/s (+0.2%); b11193-cuda prompt-cache prefill 7305 vs 1302 t/s cold; b11193-cuda sweep C=4: C4: 70.3 t/s system (2x8192) | v0.9.4 sweep C=4: C4: 70.4 t/s system (2x8192); gateway cold boot 0.52 s; cold TTFT 4827 ms vs ollama 4242 ms (0.9x); idle wake 3056 ms (sleep) vs ollama 4934 ms (full reload).

## Measured in this campaign

- blazar: 5 cell(s)
- cold-ollama: 1 cell(s)
- conc-blazar: 2 cell(s)
- conc-direct: 1 cell(s)
- conc-ollama: 1 cell(s)
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
| v0.9.4 | mistralrs | text (direct + gateway) | 11 | 7 | benchmarked |
| master-919-19bbbca | sdcpp | media | 2 | 0 | benchmarked |
| b5130 | whisper | media | 1 | 0 | benchmarked |
| b11193-cuda | llamacpp | text (direct + gateway) | 21 | 0 | benchmarked |
| sglang-0.5.19 | — | — | 0 | 0 | excluded: needs an HF safetensors model; this box serves GGUF only and 8 GiB VRAM cannot host sglang beside the media children |

## Test bed

| Component | Value |
|---|---|
| CPU | Intel Core i7-14650HX, 24 hardware threads |
| Discrete GPU | NVIDIA GeForce RTX 4070 Laptop, 8 GiB, driver 580.173.02 |
| Integrated GPU | Intel Graphics (RPL-S), Vulkan device |
| RAM | 16 GiB (13.3 GiB usable) |
| OS | Linux Mint 22.3, kernel 7.0.0-31-generic |
| Runtimes compared | blazar 0.11.0 gateway - b11193-cuda - inventory - master-919-19bbbca - ollama-host - piper (gateway TTS lane) - v0.9.4 - whisper.cpp b5130 |
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
| blazar gateway - b11193-cuda | 2x8192 | 41.4 | 121.6 | 125.8 | 24.1 | 25.3 | 1319.0 | 7328.5 | 5452 | 55.4 |
| blazar gateway - b11193-cuda (single-stream) | 1x16384 | 41.5 | 122.9 | 125.4 | 24.1 | 25.2 | 1302.0 | 7304.5 | 5666 | 55.8 |
| blazar gateway - v0.9.4 | 2x8192 | 41.4 | 124.9 | 125.8 | 24.1 | 25.3 | 1327.5 | 7166.5 | 5458 | 55.8 |
| blazar gateway - v0.9.4 (paged_attn_off) | 2x8192 | 41.5 | 123.6 | 126.5 | 24.1 | 25.0 | 1322.9 | 7136.9 | 5458 | 56.3 |
| blazar gateway - v0.9.4 (single-stream) | 1x16384 | 41.5 | 123.2 | 126.3 | 24.1 | 25.4 | 1329.3 | 7240.1 | 5672 | 55.3 |
| direct engine - b11193-cuda | 1x16384 | 41.4 | 122.3 | 125.6 | 24.2 | 25.1 | 1366.8 | 7344.0 | 5666 | 56.7 |
| ollama 0.33.3 - qwen3.5:9b | service | 40.5 | 125.4 | 126.7 | 24.9 | 75.0 | 1442.4 | 7701.3 | 6566 | 55.2 |

### Concurrency (4 parallel streams x 128 tokens)

| Runtime | slots | ok streams | rounds | system t/s | sum-stream t/s | wall s | TTFT max ms | TTFT p99 ms | ITL p99 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direct engine - b11193-cuda | 4 | 4/4 | 1 | 112.8 | 118.1 | 4.54 | 244 | - | 36.0 |
| blazar gateway - b11193-cuda | 2x8192 | 12/4 | 3 | 70.3 | 447.5 | 21.83 | 4075 | 4055 | 28.5 |
| blazar gateway - v0.9.4 | 2x8192 | 12/4 | 3 | 70.4 | 447.8 | 21.82 | 4075 | 4054 | 28.4 |
| ollama - qwen3.5:9b | service | 12/4 | 3 | 34.3 | 489.7 | 44.82 | 15676 | 15317 | 74.3 |

_sum-stream >> system t/s means streams serialize on one slot; roughly equal means genuinely parallel._

### Concurrency frontier (system t/s and tail latency vs level)

| Runtime | C | ok streams | system t/s | sum-stream t/s | eff vs C=1 | TTFT p99 ms | ITL p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| blazar gateway - b11193-cuda | 4 | 12 | 70.3 | 447.5 | - | 4055 | 28.5 |
| blazar gateway - v0.9.4 | 4 | 12 | 70.4 | 447.8 | - | 4054 | 28.4 |
| direct engine - b11193-cuda | 4 | 4 | 112.8 | 118.1 | - | - | 36.0 |
| ollama - qwen3.5:9b | 4 | 12 | 34.3 | 489.7 | - | 15317 | 74.3 |

- blazar gateway - b11193-cuda: 1 level(s) measured - insufficient levels for a saturation verdict.
- blazar gateway - v0.9.4: 1 level(s) measured - insufficient levels for a saturation verdict.
- direct engine - b11193-cuda: 1 level(s) measured - insufficient levels for a saturation verdict.
- ollama - qwen3.5:9b: 1 level(s) measured - insufficient levels for a saturation verdict.

### Adaptive reshape under sustained load (no-lag proof)

| Runtime | reshape | slots | time to reshape s | req before/after | TTFT p50 before→after ms | sys t/s before→after | failed |
|---|---|---|---:|---:|---:|---:|---:|
| blazar gateway - b11193-cuda | NO | -→- | - | 118/0 | 16177→- | 36→- | 0 |
| blazar gateway - v0.9.4 | NO | -→- | - | 118/0 | 15993→- | 36→- | 0 |

### Perplexity

| Engine | perplexity (ctx 2048, offline ASCII corpus) |
|---|---:|
| b11193-cuda | 17.35 ± 0.92 |
| v0.9.4 | not applicable (tool is llama.cpp-family) |

### Greedy parity and gateway transparency (20 prompts, 256 tokens)

| Comparison | exact / total | ratio mean | ratio min |
|---|---:|---:|---:|
| b11193-cuda vs same-engine reference (direct) | 20/20 | 1.000 | 1.000 |
| v0.9.4 vs same-engine reference (direct) | None/None | - | - |
| b11193-cuda through blazar gateway vs direct | 19/20 | 0.951 | 0.017 |

_Exact-match divergence across GPU backends is expected float nondeterminism (batch shape and backend kernels), not translation drift; bit-parity across runs requires single-slot decoding (blazar `deterministic = true` pins it)._

### Tool calls (single-turn selection + schema quality)

| Runtime | scenarios | well-formed | selection | args valid | control FP | TTFT p50 ms |
|---|---:|---:|---:|---:|---|---:|
| blazar gateway - b11193-cuda | 6 | 4/6 | 4/5 | 4/5 | no | 115 |
| blazar gateway - v0.9.4 | 6 | 4/6 | 4/5 | 4/5 | no | 114 |
| ollama - qwen3.5:9b | 6 | 4/6 | 4/5 | 4/5 | no | 288 |


### Optimization axes (ctx 4096, single stream)

| Engine | axis | setting | decode t/s | delta vs dense | prefill cold t/s | delta |
|---|---|---|---:|---:|---:|---:|
| b11193-cuda | kv | q8_0 | 41.0 | -0.5 | 1363.0 | 56.1 |
| b11193-cuda | spec | ngram-simple | 41.0 | -0.5 | 1327.2 | 20.3 |
| b11193-cuda | mmproj | True | 41.7 | 0.2 | 1345.7 | 38.8 |

### Engine capability matrix

| Capability | b11193-cuda | v0.9.4 |
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
| blazar gateway - b11193-cuda | 0.54 | 4.94 | 4781 | - | 1983 |
| blazar gateway - b11193-cuda | 0.52 | 4.90 | 4827 | - | 1987 |
| blazar gateway - v0.9.4 | 0.52 | 4.57 | 4425 | - | 1984 |
| blazar gateway - v0.9.4 | 0.52 | 4.56 | 4409 | - | 1984 |
| blazar gateway - v0.9.4 | 0.52 | 4.54 | 4467 | - | 1987 |
| direct engine - b11193-cuda | - | - | - | 2.51 | 5714 |
| ollama - qwen3.5:9b | - | 4.32 | 4242 | 4.14 | - |

_Every cold probe runs page-cache-dropped and GPU-idle-asserted on both runtimes; ollama rows without --ollama-service-restart leave the daemon warm (note in the artifact)._

### Idle wake (sleep vs keep_alive expiry)

| Runtime | idle policy | policy observed | wake TTFT ms | reload s | note |
|---|---|---|---:|---:|---|
| blazar - b11193-cuda | sleep at 15s (weights stay RAM-resident) | yes | 3056 | - |  |
| blazar - v0.9.4 | sleep at 15s (weights stay RAM-resident) | yes | 2972 | - |  |
| ollama - qwen3.5:9b | keep_alive 20s -> full unload | yes | 4934 | 4.81 |  |

_blazar sleeps with weights in RAM (wake = resume); ollama unloads at keep_alive expiry (wake = full disk reload). Policies differ by design — the table measures each runtime's own idle path after the policy verifiably fired._

### Long-context degradation curve

| Runtime | ctx | decode t/s | TTFT p50 ms |
|---|---:|---:|---:|
| blazar - b11193-cuda | 2048 | 41.7 | 110 |
| blazar - b11193-cuda | 8192 | 41.7 | 110 |
| blazar - b11193-cuda | 16384 | 41.8 | 108 |
| blazar - v0.9.4 | 2048 | 41.7 | 110 |
| blazar - v0.9.4 | 8192 | 41.7 | 108 |
| blazar - v0.9.4 | 16384 | 41.8 | 113 |
| ollama - qwen3.5:9b | 2048 | 40.9 | 111 |
| ollama - qwen3.5:9b | 8192 | 40.8 | 118 |
| ollama - qwen3.5:9b | 16384 | 40.9 | 110 |

### Media lanes (image / video / TTS / whisper)

| Lane | cold s | median s | min s | max s | ground truth | gate reject s | TTFB speedup |
|---|---:|---:|---:|---:|---|---:|---:|
| image - qwen-image-2.1-uncensored (512x512, steps=[4]) | 58.79 | 50.13 | 49.58 | 52.94 | 512x512 PNG, entropy 7.3 bits, contrast 68.5 |  |  |
| video - wan_2.1_comfyui_repackaged (320x320, steps=8) | 41.23 | 11.07 | 11.07 | 11.08 | 5f: mux==reported==aligned | 0.001 |  |
| video - wan_2.1_comfyui_repackaged (320x320, steps=8) | - | 22.10 | 22.10 | 22.10 | 13f: mux==reported==aligned |  |  |
| video - wan_2.1_comfyui_repackaged (320x320, steps=8) | - | 52.16 | 52.15 | 52.17 | 33f: mux==reported==aligned |  |  |
| tts - en_US-amy-medium (840 chars, wav+pcm) | - | 1.93 | - | 3.17 | RTF wav 0.032 / pcm 0.052 (60s audio) |  | 4.03x |
| tts-conc - en_US-amy-medium (400 chars x4 pcm) | - | 4.50 | 4.58 | - | efficiency 3.91 of 4 streams |  | 1481ms max TTFB |
| whisper - ggml-base (transcribes piper wav) | 2.16 | 1.90 | - | - | RTF 0.062 |  |  |

_Media cells run through the same sandboxed gateway as text lanes but do not assert GPU-idle: a warm engine child is the normal serving shape, so each row stamps gpu_busy_mib / ram_avail_mib / loadavg instead. 3 runs (not 5) — media variance is dominated by the model, not the scheduler. Video frame counts are read from the EBML container (lacing-aware), never from an API field; the VRAM gate probe times how fast an over-budget request is rejected with a teaching error._

## Findings (this campaign)

1. **Gateway overhead vs direct spawn: within measurement noise.** b11193-cuda decode 41.5 t/s through the gateway vs 41.4 t/s direct (+0.2%), greedy parity through the gateway 19/20 exact.
2. **Capacity-aware slot auto-sizing observed in argv.** Distinct engine shapes this campaign: 1x16384, 2x8192 - slots follow the live hardware census, each row's child_argv carries the receipt.
3. **Concurrency scaling per engine.** b11193-cuda (C=4): peak 70.3 t/s at C=4; v0.9.4 (C=4): peak 70.4 t/s at C=4; serialization behavior per level in the frontier table below.
4. **Prompt cache pays 5.6x on prefill** (7276 cached vs 1307 t/s cold).
5. **Speculative n-gram decoding is a net loss for this model** (decode 41.0 t/s, -0.5 vs dense baseline) - measured, not assumed.
6. **KV quantization (q8_0) is decode-neutral** (decode 41.0 t/s vs 41.5 dense).
7. **Tool-call quality (single-turn, temp 0).** b11193-cuda: selection 4/5, args 4/5 (control clean); v0.9.4: selection 4/5, args 4/5 (control clean); ollama: selection 4/5, args 4/5 (control clean); per-scenario detail in cells.jsonl.
8. **Adaptive reshape lands under sustained load.** b11193-cuda: no reshape observed in the lane window; v0.9.4: no reshape observed in the lane window (graceful-drain: adoption waits for in-flight streams, never kills one).
9. **Image quality stamps (PIL, luma domain): entropy 7.30 bits, rms contrast 68.5, 29348 unique colors @256x256** on qwen-image-2.1-uncensored - perceptual baseline for cross-run comparisons; audit PNG saved beside the cells.
10. **Video VRAM gate rejects an over-budget request in 1 ms** with the full estimate math and override levers in the error body - instead of an opaque child abort minutes later.
11. **Streamed PCM cuts time-to-first-audio 4.03x vs buffered WAV** (piper lane, first audio 0.48s vs 1.93s full synthesis) - total wall time is slightly higher (per-chunk synthesis), the win is interactivity.
12. **4 parallel PCM streams through one gateway: perfectly parallel** (efficiency 3.91 = sum of per-stream totals / 4.58s wall, max TTFB 1481 ms, NON-uniform stream outputs - flagged) - the scalability receipt for the TTS lane.

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
python3 scripts/bench_matrix.py --blazar-bin target/release/blazar --md BENCHMARK.md
python3 scripts/bench_matrix.py --render-only --artifacts-dir <dir> --md BENCHMARK.md
```

_Raw per-cell records (argv, per-run lists, daemon logs): `bench-artifacts/20260926-sdcpp-lane/cells.jsonl`._
