# Blazar inference benchmark

_Rendered 20260924-text-frontier; blazar 0.11.0; power state of gateway rows: ac._

## Executive summary

b11147-cuda: gateway 40.7 vs direct 41.0 t/s (-0.9%); b11147-cuda prompt-cache prefill 7997 vs 1343 t/s cold; b11147-cuda sweep C=1/2/4/8: C1: 38.7 t/s system (1x16384); C2: 39.0 t/s system (1x16384); C4: 39.3 t/s system (1x16384); C8: 39.3 t/s system (1x16384); gateway cold boot 0.52 s; cold TTFT 6156 ms vs ollama 5997 ms (1.0x); idle wake 4365 ms (sleep) vs ollama 6484 ms (full reload).

## Measured in this campaign

- blazar: 2 cell(s)
- cold-ollama: 1 cell(s)
- conc-blazar: 4 cell(s)
- conc-direct: 4 cell(s)
- conc-ollama: 4 cell(s)
- direct: 7 cell(s)
- features: 1 cell(s)
- greedy: 1 cell(s)
- greedy_gw: 1 cell(s)
- idle-blazar: 1 cell(s)
- idle-ollama: 1 cell(s)
- ollama: 1 cell(s)

## Engine coverage

_Engine inventory not stamped (campaign predates coverage stamping); coverage cannot be audited for this artifact._

## Test bed

| Component | Value |
|---|---|
| CPU | Intel Core i7-14650HX, 24 hardware threads |
| Discrete GPU | NVIDIA GeForce RTX 4070 Laptop, 8 GiB, driver 580.173.02 |
| Integrated GPU | Intel Graphics (RPL-S), Vulkan device |
| RAM | 16 GiB (13.3 GiB usable) |
| OS | Linux Mint 22.3, kernel 7.0.0-31-generic |
| Runtimes compared | blazar 0.11.0 gateway - b11147-cuda - ollama-host |
| Model | Qwen3.5-9B-Q4_K_M.gguf |

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

## Results

### Single-stream decode (512-token prompt, 128 generated, median of 5)

| Runtime | slots x ctx | decode t/s | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | prefill cold t/s | prefill cached t/s | GPU peak MiB | GPU power W |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blazar gateway - b11147-cuda | 1x16384 | 40.8 | 127.6 | 133.9 | 24.5 | 25.6 | 1295.0 | 6783.8 | 5840 | 55.3 |
| blazar gateway - b11147-cuda | 1x16384 | 40.7 | 129.9 | 136.1 | 24.5 | 25.6 | 1343.2 | 7997.0 | 5715 | 56.4 |
| direct engine - b11147-cuda | 1x16384 | 41.0 | 120.6 | 129.2 | 24.3 | 25.8 | 1329.4 | 7645.1 | 5715 | 56.4 |
| ollama 0.33.3 - qwen3.5:9b | service | 40.0 | 172.9 | 196.7 | 25.3 | 76.0 | 1212.9 | 7508.6 | 6447 | 55.4 |

### Concurrency (1x2x4x8 parallel streams x 128 tokens)

| Runtime | slots | ok streams | rounds | system t/s | sum-stream t/s | wall s | TTFT max ms | TTFT p99 ms | ITL p99 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| blazar gateway - b11147-cuda | 1x16384 | 3/1 | 3 | 38.7 | 122.3 | 9.93 | 324 | 321 | 26.0 |
| blazar gateway - b11147-cuda | 1x16384 | 6/2 | 3 | 39.0 | 244.0 | 19.71 | 3580 | 3571 | 26.2 |
| blazar gateway - b11147-cuda | 1x16384 | 12/4 | 3 | 39.3 | 488.8 | 39.12 | 9990 | 9981 | 26.2 |
| blazar gateway - b11147-cuda | 1x16384 | 24/8 | 3 | 39.3 | 975.1 | 78.21 | 23054 | 23038 | 26.0 |
| direct engine - b11147-cuda | 1 | 1/1 | 1 | 38.8 | 40.8 | 3.30 | 182 | - | 25.8 |
| direct engine - b11147-cuda | 2 | 2/2 | 1 | 69.9 | 72.9 | 3.66 | 178 | - | 29.9 |
| direct engine - b11147-cuda | 4 | 4/4 | 1 | 108.3 | 114.3 | 4.73 | 282 | - | 38.6 |
| direct engine - b11147-cuda | 8 | 8/8 | 1 | 131.7 | 139.2 | 7.78 | 477 | - | 63.6 |
| ollama - qwen3.5:9b | service | 3/1 | 3 | 22.3 | 120.1 | 17.21 | 7332 | 7188 | 75.8 |
| ollama - qwen3.5:9b | service | 6/2 | 3 | 28.5 | 239.4 | 26.95 | 10284 | 10118 | 76.1 |
| ollama - qwen3.5:9b | service | 12/4 | 3 | 32.6 | 481.9 | 47.17 | 17537 | 17172 | 75.8 |
| ollama - qwen3.5:9b | service | 24/8 | 3 | 35.0 | 960.5 | 87.65 | 31172 | 30411 | 75.8 |

_sum-stream >> system t/s means streams serialize on one slot; roughly equal means genuinely parallel._

### Concurrency frontier (system t/s and tail latency vs level)

| Runtime | C | ok streams | system t/s | sum-stream t/s | eff vs C=1 | TTFT p99 ms | ITL p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| blazar gateway - b11147-cuda | 1 | 3 | 38.7 | 122.3 | 100% | 321 | 26.0 |
| blazar gateway - b11147-cuda | 2 | 6 | 39.0 | 244.0 | 50% | 3571 | 26.2 |
| blazar gateway - b11147-cuda | 4 | 12 | 39.3 | 488.8 | 25% | 9981 | 26.2 |
| blazar gateway - b11147-cuda | 8 | 24 | 39.3 | 975.1 | 13% | 23038 | 26.0 |
| direct engine - b11147-cuda | 1 | 1 | 38.8 | 40.8 | 100% | - | 25.8 |
| direct engine - b11147-cuda | 2 | 2 | 69.9 | 72.9 | 90% | - | 29.9 |
| direct engine - b11147-cuda | 4 | 4 | 108.3 | 114.3 | 70% | - | 38.6 |
| direct engine - b11147-cuda | 8 | 8 | 131.7 | 139.2 | 42% | - | 63.6 |
| ollama - qwen3.5:9b | 1 | 3 | 22.3 | 120.1 | 100% | 7188 | 75.8 |
| ollama - qwen3.5:9b | 2 | 6 | 28.5 | 239.4 | 64% | 10118 | 76.1 |
| ollama - qwen3.5:9b | 4 | 12 | 32.6 | 481.9 | 36% | 17172 | 75.8 |
| ollama - qwen3.5:9b | 8 | 24 | 35.0 | 960.5 | 20% | 30411 | 75.8 |

- blazar gateway - b11147-cuda: throughput plateaus at C=2 (<10% per-level gain), peak 39.3 t/s at C=8.
- direct engine - b11147-cuda: still gaining at C=8 (38.8 -> 131.7 t/s) - saturation not reached within the sweep.
- ollama - qwen3.5:9b: throughput plateaus at C=8 (<10% per-level gain), peak 35.0 t/s at C=8.

### Perplexity

_Not measured in this campaign (20260924-text-frontier); perplexity lane not run._

### Greedy parity and gateway transparency (20 prompts, 256 tokens)

| Comparison | exact / total | ratio mean | ratio min |
|---|---:|---:|---:|
| b11147-cuda vs same-engine reference (direct) | 20/20 | 1.000 | 1.000 |
| b11147-cuda through blazar gateway vs direct | 20/20 | 1.000 | 1.000 |

_Exact-match divergence across GPU backends is expected float nondeterminism (batch shape and backend kernels), not translation drift; bit-parity across runs requires single-slot decoding (blazar `deterministic = true` pins it)._

### Tool calls (single-turn selection + schema quality)

_Not measured in this campaign (20260924-text-frontier); tool calls lane not run._


### Optimization axes (ctx 4096, single stream)

| Engine | axis | setting | decode t/s | delta vs dense | prefill cold t/s | delta |
|---|---|---|---:|---:|---:|---:|
| b11147-cuda | kv | q8_0 | 40.6 | -0.3 | 1360.7 | 147.6 |
| b11147-cuda | spec | ngram-simple | 40.5 | -0.3 | 1326.6 | 113.5 |
| b11147-cuda | mmproj | True | 41.0 | 0.1 | 1291.5 | 78.4 |

### Engine capability matrix

| Capability | b11147-cuda |
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
| blazar gateway - b11147-cuda | 0.52 | 5.56 | 5485 | - | 1993 |
| blazar gateway - b11147-cuda | 0.52 | 6.23 | 6156 | - | 1993 |
| direct engine - b11147-cuda | - | - | - | 2.52 | 5771 |
| ollama - qwen3.5:9b | - | 6.08 | 5997 | 5.84 | - |

_Every cold probe runs page-cache-dropped and GPU-idle-asserted on both runtimes; ollama rows without --ollama-service-restart leave the daemon warm (note in the artifact)._

### Idle wake (sleep vs keep_alive expiry)

| Runtime | idle policy | policy observed | wake TTFT ms | reload s | note |
|---|---|---|---:|---:|---|
| blazar - b11147-cuda | sleep at 15s (weights stay RAM-resident) | yes | 4365 | - |  |
| ollama - qwen3.5:9b | keep_alive 20s -> full unload | yes | 6484 | 6.30 |  |

_blazar sleeps with weights in RAM (wake = resume); ollama unloads at keep_alive expiry (wake = full disk reload). Policies differ by design — the table measures each runtime's own idle path after the policy verifiably fired._

### Long-context degradation curve

_Not measured in this campaign (20260924-text-frontier); long-context curve lane not run._

### Media lanes (image / video / TTS / whisper)

_Not measured in this campaign (20260924-text-frontier); media lane not run._

_Media cells run through the same sandboxed gateway as text lanes but do not assert GPU-idle: a warm engine child is the normal serving shape, so each row stamps gpu_busy_mib / ram_avail_mib / loadavg instead. 3 runs (not 5) — media variance is dominated by the model, not the scheduler. Video frame counts are read from the EBML container (lacing-aware), never from an API field; the VRAM gate probe times how fast an over-budget request is rejected with a teaching error._

## Findings (this campaign)

1. **Gateway overhead vs direct spawn: within measurement noise.** b11147-cuda decode 40.7 t/s through the gateway vs 41.0 t/s direct (-0.9%), greedy parity through the gateway 20/20 exact.
2. **Concurrency scaling per engine.** b11147-cuda (C=1/2/4/8): peak 39.3 t/s at C=8, 13% of ideal at C=8; serialization behavior per level in the frontier table below.
3. **Prompt cache pays 5.2x on prefill** (6784 cached vs 1295 t/s cold).
4. **Speculative n-gram decoding is a net loss for this model** (decode 40.5 t/s, -0.3 vs dense baseline) - measured, not assumed.
5. **KV quantization (q8_0) is decode-neutral** (decode 40.6 t/s vs 40.8 dense).

## Carried-over findings (no receipt in this campaign)

_Established in earlier campaigns whose receipts live in their bench-artifacts/ directories; this campaign did not measure these lanes._

1. **Capacity-aware slot auto-sizing.** blazar sizes engine slots from live hardware census: the 8 GiB card with a vision projector attached spawns 1 slot (16 Ki context) on the Vulkan build and 4 slots (64 Ki total) on CUDA - measured oversubscription on Vulkan either fails to boot or degrades 2x, so the cap is load-bearing, not conservative cosmetics.
2. **mistral.rs 0.9.3 with default paged attention cannot fit this model on an 8 GiB card** (upstream sizes KV as a fraction of total VRAM); blazar's profile auto-disables paged attention on tight cards and the model then serves correctly.
3. **Gateway passes OpenAI tools verbatim** (tools-aware validation, no schema rewriting); tool-call quality is the engine's own - selection and argument schema are scored per scenario with a no-tool control for false positives.


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

_Raw per-cell records (argv, per-run lists, daemon logs): `bench-artifacts/20260924-text-frontier/cells.jsonl`._
