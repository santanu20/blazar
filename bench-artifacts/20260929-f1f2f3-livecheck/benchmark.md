# Blazar inference benchmark

_Rendered 20260929-f1f2f3-livecheck; blazar 0.13.0; power state of gateway rows: ac._

## Executive summary

- b11202-cuda prompt-cache prefill 8330 vs 1362 t/s cold
- b11202-cuda sweep C=4/4: C4: 105.0 t/s system (4x65536); C4: 105.3 t/s system (4x65536)
- v0.9.4 sweep C=4/4: C4: 20.8 t/s system (engine-scheduled); C4: 17.8 t/s system (engine-scheduled)
- ITL p99 25.02 ms vs ollama 73.97 ms (3.0x tighter)
- adaptive reshape under sustained load: slots 4->8, 50.6->64.8 t/s while serving, zero failed requests
- gateway cold boot 0.52 s
- cold TTFT 3912 ms vs ollama 4607 ms (1.2x)

## Measured in this campaign

- blazar: 8 cell(s)
- cold-ollama: 2 cell(s)
- conc-blazar: 4 cell(s)
- ollama: 2 cell(s)
- reshape: 3 cell(s)

## Engine coverage

| Engine | Kind | Lane | ok cells | err cells | Status |
|---|---|---|---:|---:|---|
| v0.9.4 | mistralrs | text (direct + gateway) | 6 | 2 | benchmarked |
| b11202-cuda | llamacpp | text (direct + gateway) | 5 | 2 | benchmarked |
| b5130 | - | - | 0 | 0 | excluded: kind 'whisper' has no bench lane in this harness |
| master-920-2f88688 | - | - | 0 | 0 | excluded: kind 'sdcpp' has no bench lane in this harness |
| sglang-0.5.19 | - | - | 0 | 0 | excluded: model-format sweep: this campaign sweeps a GGUF file; sglang serves HF safetensors checkpoints (rerun with --model <safetensors-row>) |

## Test bed

| Component | Value |
|---|---|
| CPU | Intel Core i7-14650HX, 24 hardware threads |
| Discrete GPU | NVIDIA GeForce RTX 4070 Laptop, 8 GiB, driver 580.173.02 |
| Integrated GPU | Intel Graphics (RPL-S), Vulkan device |
| RAM | 16 GiB (13.3 GiB usable) |
| OS | Linux Mint 22.3, kernel 7.0.0-31-generic |
| Runtimes compared | blazar 0.13.0 gateway - b11202-cuda - inventory - ollama-host - v0.9.4 |
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
| blazar gateway - b11202-cuda | 4x65536 | 41.2 | 107.8 | 117.5 | 24.3 | 25.1 | 1392.4 | 8119.5 | 6151 | 56.1 |
| blazar gateway - b11202-cuda (single-stream) | 1x16384 | 41.6 | 107.5 | 111.5 | 24.0 | 25.0 | 1362.1 | 8330.2 | 5665 | 55.5 |
| blazar gateway - v0.9.4 | engine-scheduled | 21.7 | 119.5 | 152.6 | 45.7 | 56.7 | 275.4 | 266.2 | 6947 | 44.3 |
| blazar gateway - v0.9.4 (single-stream) | engine-scheduled | 21.6 | 121.9 | 133.8 | 45.9 | 55.6 | 261.0 | 269.0 | 6947 | 44.3 |
| ollama 0.33.3 - qwen3.5:9b (t/s NOT comparable - matrix model differs) | service | 41.1 | 107.7 | 115.9 | 24.6 | 74.0 | 1351.7 | 8186.5 | 6569 | 55.7 |
| ollama 0.33.3 - qwen3.5:9b (t/s NOT comparable - matrix model differs) | service | 41.2 | 110.0 | 111.1 | 24.5 | 73.7 | 1430.3 | 8498.6 | 6569 | 55.4 |

### Concurrency (4 parallel streams x 128 tokens)

| Runtime | slots | ok streams | rounds | system t/s | sum-stream t/s | wall s | TTFT max ms | TTFT p99 ms | ITL p99 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| blazar gateway - b11202-cuda | 4x65536 | 12 of 3x4 | 3 | 105.0 | 349.1 | 14.62 | 535 | 528 | 79.2 |
| blazar gateway - b11202-cuda | 4x65536 | 12 of 3x4 | 3 | 105.3 | 347.9 | 14.59 | 489 | 488 | 76.5 |
| blazar gateway - v0.9.4 | engine-scheduled | 12 of 3x4 | 3 | 20.8 | 63.8 | 18.46 | 245 | 245 | 54.0 |
| blazar gateway - v0.9.4 | engine-scheduled | 12 of 3x4 | 3 | 17.8 | 36.5 | 14.37 | 282 | 282 | 79.5 |

_sum-stream >> system t/s means streams serialize on one slot; roughly equal means genuinely parallel._

### Concurrency frontier (system t/s and tail latency vs level)

| Runtime | C | ok streams | system t/s | sum-stream t/s | eff vs C=1 | TTFT p99 ms | ITL p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| blazar gateway - b11202-cuda | 4 | 12 of 3x4 | 105.0 | 349.1 | - | 528 | 79.2 |
| blazar gateway - b11202-cuda | 4 | 12 of 3x4 | 105.3 | 347.9 | - | 488 | 76.5 |
| blazar gateway - v0.9.4 | 4 | 12 of 3x4 | 20.8 | 63.8 | - | 245 | 54.0 |
| blazar gateway - v0.9.4 | 4 | 12 of 3x4 | 17.8 | 36.5 | - | 282 | 79.5 |

- blazar gateway - b11202-cuda: 2 level(s) measured - insufficient levels for a saturation verdict.
- blazar gateway - v0.9.4: 2 level(s) measured - insufficient levels for a saturation verdict.

### Adaptive reshape under sustained load (no-lag proof)

| Runtime | reshape | slots | time to reshape s | req before/after | TTFT p50 before->after ms | sys t/s before->after | failed |
|---|---|---|---:|---:|---:|---:|---:|
| blazar gateway - b11202-cuda | yes | 4->8 | 74 | 32/154 | 7138->316 | 51->65 | 0 |
| blazar gateway - v0.9.4 | NO (detection void) | -->- | - | 1143/0 | 2064->- | 363->- | 0 |
| blazar gateway - v0.9.4 | NO (detection void) | -->- | - | 1128/0 | 2082->- | 359->- | 0 |

<details><summary>daemon tail (slots/reshape) - b11202-cuda</summary>

```
2026-09-29T16:02:51.664931Z  INFO evict: terminating child and releasing the slot model="qwen3.5-9b" pid=166067
2026-09-29T16:02:51.664931Z  INFO evict: terminating child and releasing the slot model="qwen3.5-9b" pid=166067
```

</details>

<details><summary>daemon tail (slots/reshape) - v0.9.4</summary>

```
2026-09-29T14:04:27.545022Z  INFO evict: terminating child and releasing the slot model="qwen3.5-9b" pid=69589
2026-09-29T14:04:27.545022Z  INFO evict: terminating child and releasing the slot model="qwen3.5-9b" pid=69589
```

</details>

<details><summary>daemon tail (slots/reshape) - v0.9.4</summary>

```
2026-09-29T15:07:52.380934Z  INFO evict: terminating child and releasing the slot model="qwen3.5-9b" pid=125045
2026-09-29T15:07:52.380934Z  INFO evict: terminating child and releasing the slot model="qwen3.5-9b" pid=125045
```

</details>

### Perplexity

_Not measured in this campaign (20260929-f1f2f3-livecheck); perplexity lane not run._

### Greedy parity and gateway transparency (20 prompts, 256 tokens)

_Not measured in this campaign (20260929-f1f2f3-livecheck); greedy parity lane not run._

_Exact-match divergence across GPU backends is expected float nondeterminism (batch shape and backend kernels), not translation drift; bit-parity across runs requires single-slot decoding (blazar `deterministic = true` pins it)._

### Tool calls (single-turn selection + schema quality)

_Not measured in this campaign (20260929-f1f2f3-livecheck); tool calls lane not run._

### Optimization axes (ctx 4096, single stream)

_Not measured in this campaign (20260929-f1f2f3-livecheck); optimization axes lane not run._

### Gateway config knobs (on/off vs default profile)

| Engine | config | decode t/s | decode delta% | ttft p50 ms | ttft delta% | GPU peak MiB | GPU delta | output vs default |
|---|---|---:|---:|---:|---:|---:|---:|:-:|
| b11202-cuda | single-stream | 41.6 | +1.1% | 108 | -0.3% | 5665 | -486 | same |
| v0.9.4 | single-stream | 21.6 | -0.5% | 122 | +1.9% | 6947 | +0 | same |

### Gateway scheduler knobs under concurrent load (C=8 bursts)

_Not measured in this campaign (20260929-f1f2f3-livecheck); conc lane A/B lane not run._

### Engine capability matrix

_Not measured in this campaign (20260929-f1f2f3-livecheck); capability matrix lane not run._

### Cold start and footprint

| Runtime | daemon boot s | first request (cold engine load) s | cold TTFT ms | engine load s | RSS peak MiB |
|---|---:|---:|---:|---:|---:|
| blazar gateway - b11202-cuda | 0.55 | 4.12 | 3934 | - | 2140 |
| blazar gateway - b11202-cuda (single-stream) | 0.52 | 4.01 | 3912 | - | 2105 |
| blazar gateway - v0.9.4 | 0.52 | 8.01 | 7873 | - | 6584 |
| blazar gateway - v0.9.4 (single-stream) | 0.53 | 7.55 | 7399 | - | 6650 |
| ollama - qwen3.5:9b | - | 4.70 | 4607 | 4.48 | - |
| ollama - qwen3.5:9b | - | 4.44 | 4361 | 4.25 | - |

_Every cold probe runs page-cache-dropped and GPU-idle-asserted on both runtimes; ollama rows without --ollama-service-restart leave the daemon warm (note in the artifact)._

### Idle wake (sleep vs keep_alive expiry)

_Not measured in this campaign (20260929-f1f2f3-livecheck); idle wake lane not run._

_blazar sleeps with weights in RAM (wake = resume); ollama unloads at keep_alive expiry (wake = full disk reload). Policies differ by design - the table measures each runtime's own idle path after the policy verifiably fired._

### Long-context degradation curve

_Not measured in this campaign (20260929-f1f2f3-livecheck); long-context curve lane not run._

### Media lanes (image / video / TTS / whisper)

_Not measured in this campaign (20260929-f1f2f3-livecheck); media lane not run._

_Not measured in this campaign (20260929-f1f2f3-livecheck); media config A/B lane not run._

_Media cells run through the same sandboxed gateway as text lanes but do not assert GPU-idle: a warm engine child is the normal serving shape, so each row stamps gpu_busy_mib / ram_avail_mib / loadavg instead. 3 runs (not 5); media variance is dominated by the model, not the scheduler. Video frame counts are read from the EBML container (lacing-aware), never from an API field; the VRAM gate probe times how fast an over-budget request is rejected with a teaching error._

## Findings (this campaign)

1. **Capacity-aware slot auto-sizing observed in argv.** Distinct engine shapes this campaign: 1x16384, 4x65536 - slots follow the live hardware census, each row's child_argv carries the receipt.
2. **Concurrency scaling per engine.** b11202-cuda (C=4/4): peak 105.3 t/s at C=4; v0.9.4 (C=4/4): peak 20.8 t/s at C=4; serialization behavior per level in the frontier table below.
3. **Prompt cache pays 5.8x on prefill** (8119 cached vs 1392 t/s cold).
4. **Adaptive reshape lands under sustained load.** b11202-cuda: reshaped 4->8 slots after 74.1 s under sustained C=8, TTFT p50 7138->316 ms, 0 dropped requests; v0.9.4: reshape verdict VOID - api/ps never reported a slots reading for the model — reshape verdict is a measurement void, not a product result; v0.9.4: reshape verdict VOID - api/ps never reported a slots reading for the model — reshape verdict is a measurement void, not a product result (graceful-drain: adoption waits for in-flight streams, never kills one).

## Carried-over findings (no receipt in this campaign)

_Established in earlier campaigns whose receipts live in their bench-artifacts/ directories; this campaign did not measure these lanes._

1. **Gateway overhead is within measurement noise.** Single-stream decode through the blazar gateway matches direct engine spawns at the same slots/context (see speed table); the greedy gateway lane is byte-identical to the direct lane where sampling is single-slot.
2. **Speculative n-gram decoding is a net loss for this 9B model** (no draft model; acceptance too low to pay the verification overhead) - documented so the flag is not cargo-culted.
3. **KV q8_0 quantization is decode-neutral and prefill-neutral steady-state**; the one cold-prefill outlier below is a first-invocation pipeline-compile artifact (controlled re-probe measured full-rate steady state).
4. **mistral.rs 0.9.3 with default paged attention cannot fit this model on an 8 GiB card** (upstream sizes KV as a fraction of total VRAM); blazar's profile auto-disables paged attention on tight cards and the model then serves correctly.
5. **Gateway passes OpenAI tools verbatim** (tools-aware validation, no schema rewriting); tool-call quality is the engine's own - selection and argument schema are scored per scenario with a no-tool control for false positives.

## Failed cells (engine-reality receipts)

_These knobs crashed the engine child on this test bed; the crash signature is the receipt. Raw logs: `cells.jsonl` (`daemon_log_tail` field)._

| Engine | lane | config | failure |
|---|---|---|---|
| b11202-cuda | blazar | default | blazar cell crashed: MemAvailable 7345 MiB < needed ~7731 MiB (model 5366 MiB + headroom). A co-resi |
| b11202-cuda | blazar | single-stream | blazar cell crashed: MemAvailable 7381 MiB < needed ~7731 MiB (model 5366 MiB + headroom). A co-resi |
| v0.9.4 | blazar | default | blazar cell crashed: MemAvailable 7456 MiB < needed ~7731 MiB (model 5366 MiB + headroom). A co-resi |
| v0.9.4 | blazar | single-stream | blazar cell crashed: MemAvailable 7371 MiB < needed ~7731 MiB (model 5366 MiB + headroom). A co-resi |

## Caveats

- ollama prefill numbers come from engine counters that exclude the chat template, so they read slightly high against the 512-token lanes.
- Cross-backend greedy ratios (CUDA vs Vulkan) diverge on near-tie logits; treat ratio, not exact-match count, as the signal.
- All GPU rows measured on AC power at bounded load; rows record load average and power state (battery runs are rejected by the harness).
- Numbers are medians of 5 runs on one hybrid laptop; expect absolute shifts on other hardware, ratios to travel better.

## Reproduce

```bash
python3 scripts/bench_matrix.py --artifacts-dir bench-artifacts/20260929-f1f2f3-livecheck --engines b11202-cuda v0.9.4 --providers blazar --skip-ppl --skip-greedy --skip-tools --skip-features --skip-idle --skip-ctxcurve --skip-media --skip-variants --md bench-artifacts/20260929-f1f2f3-livecheck/benchmark.md
python3 scripts/bench_matrix.py --render-only --artifacts-dir <dir> [--append-campaign <sibling-campaign-dir>] --md BENCHMARK.md
```

_Raw per-cell records (argv, per-run lists, daemon logs): `bench-artifacts/20260929-f1f2f3-livecheck/cells.jsonl`._
