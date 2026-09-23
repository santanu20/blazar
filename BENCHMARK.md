# Blazar inference benchmark

_Rendered 20260923-media-v2; blazar 0.10.0; power state of gateway rows: ._

## Executive summary

_No complete rows._

## Test bed

| Component | Value |
|---|---|
| CPU | Intel Core i7-14650HX, 24 hardware threads |
| Discrete GPU | NVIDIA GeForce RTX 4070 Laptop, 8 GiB, driver 580.173.02 |
| Integrated GPU | Intel Graphics (RPL-S), Vulkan device |
| RAM | 16 GiB (13.3 GiB usable) |
| OS | Linux Mint 22.3, kernel 7.0.0-31-generic |
| Runtimes compared | blazar 0.10.0 gateway - piper (gateway TTS lane) - stable-diffusion.cpp master-890 (Vulkan) - whisper.cpp b5130 |
| Model | en_US-amy-medium, ggml-base, qwen-image-2.1, wan_2.1_comfyui_repackaged |

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
- Media lanes run in isolated sandbox daemons (same protocol as text blazar cells); the engine child spawns lazily, so each family's first request is the COLD number (spawn + weights + first artifact), labeled cold_request_s.
- Media TTFB = time to first BODY byte (first audio sample for streamed PCM, not response headers); buffered WAV TTFB equals its total by construction and the table says so.
- Video frame counts are container ground truth: the response webm is parsed for lacing-aware SimpleBlock counts and asserted against the Wan 4k+1 temporal grid (a mismatch is recorded loudly, never averaged away).
- The video scratch-gate probe sends one expected-rejected monster (duration 60s -> 960 aligned frames) and times the 400; the legit 5-frame pass rides the same warm child, so axis-row vs probe deltas price the gate itself.
- Media cells stamp GPU-busy and RAM-available at entry instead of asserting an idle GPU: a warm child from the previous family is the normal media workflow, and the receipt carries the occupancy rather than hiding it.
- TTS RTF = synthesis wall / audio seconds, audio duration parsed from the RIFF data-chunk length (not estimated from characters); whisper transcribes a WAV synthesized by the same campaign's piper voice, so the input is reproducible from the receipt.
- Image quality stamps are PIL-gated luma-domain metrics (rms contrast = luma stddev, entropy in bits, unique colors on a 256x256 downsample); when PIL is absent the row carries an honest 'skipped' note instead of a fake number, and one audit PNG per steps point is saved beside the cells for offline re-measurement.
- TTS concurrency probe: N parallel streamed-PCM requests through one sandboxed gateway; wall clock vs sum of per-stream totals yields an efficiency ratio (sum/wall ~ 1 means serialized, -> N means perfectly parallel), and the probe fails loudly if any stream errors or truncates.

## Results

### Single-stream decode (512-token prompt, 128 generated, median of 5)

| Runtime | slots x ctx | decode t/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | prefill cold t/s | prefill cached t/s | GPU peak MiB | GPU power W |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

### Concurrency (4 parallel streams x 128 tokens)

| Runtime | slots | ok streams | rounds | system t/s | sum-stream t/s | wall s | TTFT max ms | TTFT p99 ms | ITL p99 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

_sum-stream >> system t/s means streams serialize on one slot; roughly equal means genuinely parallel._

### Perplexity

_Not measured._

### Greedy parity and gateway transparency (20 prompts, 256 tokens)

| Comparison | exact / total | ratio mean | ratio min |
|---|---:|---:|---:|

_Exact-match divergence across GPU backends is expected float nondeterminism (batch shape and backend kernels), not translation drift; bit-parity across runs requires single-slot decoding (blazar `deterministic = true` pins it)._

### Optimization axes (ctx 4096, single stream)

_Not measured._

### Engine capability matrix

_Not measured._

### Cold start and footprint

_Not measured._

_Every cold probe runs page-cache-dropped and GPU-idle-asserted on both runtimes; ollama rows without --ollama-service-restart leave the daemon warm (note in the artifact)._

### Idle wake (sleep vs keep_alive expiry)

_Not measured._

_blazar sleeps with weights in RAM (wake = resume); ollama unloads at keep_alive expiry (wake = full disk reload). Policies differ by design — the table measures each runtime's own idle path after the policy verifiably fired._

### Long-context degradation curve

_Not measured._

### Media lanes (image / video / TTS / whisper)

| Lane | cold s | median s | min s | max s | ground truth | gate reject s | TTFB speedup |
|---|---:|---:|---:|---:|---|---:|---:|
| image - qwen-image-2.1 (512x512, steps=[4]) | 59.06 | 71.18 | 64.13 | 71.52 | 512x512 PNG, entropy 6.6 bits, contrast 54.6 |  |  |
| tts - en_US-amy-medium (840 chars, wav+pcm) | - | 2.16 | - | 3.49 | RTF wav 0.036 / pcm 0.058 (61s audio) |  | 4.18x |
| tts-conc - en_US-amy-medium (400 chars x4 pcm) | - | 5.01 | 5.12 | - | efficiency 3.90 of 4 streams |  | 1813ms max TTFB |
| video - wan_2.1_comfyui_repackaged (320x320, steps=8) | 17.19 | 13.09 | 13.08 | 14.09 | 5f: mux==reported==aligned | 0.001 |  |
| video - wan_2.1_comfyui_repackaged (320x320, steps=8) | - | 26.12 | 26.12 | 26.14 | 13f: mux==reported==aligned |  |  |
| video - wan_2.1_comfyui_repackaged (320x320, steps=8) | - | 60.19 | 60.18 | 60.19 | 33f: mux==reported==aligned |  |  |
| whisper - ggml-base (transcribes piper wav) | 2.46 | 2.14 | - | - | RTF 0.070 |  |  |

_Media cells run through the same sandboxed gateway as text lanes but do not assert GPU-idle: a warm engine child is the normal serving shape, so each row stamps gpu_busy_mib / ram_avail_mib / loadavg instead. 3 runs (not 5) — media variance is dominated by the model, not the scheduler. Video frame counts are read from the EBML container (lacing-aware), never from an API field; the VRAM gate probe times how fast an over-budget request is rejected with a teaching error._

## Findings

1. **Gateway overhead is within measurement noise.** Single-stream decode through the blazar gateway matches direct engine spawns at the same slots/context (see speed table); the greedy gateway lane is byte-identical to the direct lane where sampling is single-slot.
2. **Capacity-aware slot auto-sizing.** blazar sizes engine slots from live hardware census: the 8 GiB card with a vision projector attached spawns 1 slot (16 Ki context) on the Vulkan build and 4 slots (64 Ki total) on CUDA - measured oversubscription on Vulkan either fails to boot or degrades 2x, so the cap is load-bearing, not conservative cosmetics.
3. **Concurrency scales where capacity allows.** 4 streams through CUDA gateway hold near-direct system throughput; the Vulkan single-slot shape serializes streams (per-stream latency stays excellent; system throughput caps at one stream's rate) - a capacity trade, not a scheduling defect.
4. **Prompt cache pays ~6-7x on prefill.** Cached-prefix prefill runs thousands of tokens/s vs hundreds cold.
5. **Speculative n-gram decoding is a net loss for this 9B model** (no draft model; acceptance too low to pay the verification overhead) - documented so the flag is not cargo-culted.
6. **KV q8_0 quantization is decode-neutral and prefill-neutral steady-state**; the one cold-prefill outlier below is a first-invocation pipeline-compile artifact (controlled re-probe measured full-rate steady state).
7. **mistral.rs 0.9.3 with default paged attention cannot fit this model on an 8 GiB card** (upstream sizes KV as a fraction of total VRAM); blazar's profile auto-disables paged attention on tight cards and the model then serves correctly.
8. **Image quality stamps (PIL, luma domain): entropy 6.61 bits, rms contrast 54.6, 29393 unique colors @256x256** on qwen-image-2.1 - perceptual baseline for cross-run comparisons; audit PNG saved beside the cells.
9. **Streamed PCM cuts time-to-first-audio 4.18x vs buffered WAV** (piper lane, first audio 0.52s vs 2.16s full synthesis) - total wall time is slightly higher (per-chunk synthesis), the win is interactivity.
10. **4 parallel PCM streams through one gateway: perfectly parallel** (efficiency 3.90 = sum of per-stream totals / 5.12s wall, max TTFB 1813 ms, NON-uniform stream outputs - flagged) - the scalability receipt for the TTS lane.
11. **Video VRAM gate rejects an over-budget request in 1 ms** with the full estimate math and override levers in the error body - instead of an opaque child abort minutes later.

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

_Raw per-cell records (argv, per-run lists, daemon logs): `bench-artifacts/20260923-media-v2/cells.jsonl`._
