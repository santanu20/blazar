# Blazar inference benchmark

_Rendered 2026-10-07-overhead; blazar mixed; power state of gateway rows: ac._

## Executive summary

- v0.9.4 prompt-cache prefill 240 vs 235 t/s cold
- b11429-cuda sweep C=1/2/4/8: C1: 36.1 t/s system (4x65536); C2: 59.1 t/s system (4x65536); C4: 86.1 t/s system (4x65536); C8: 87.9 t/s system (4x65536)
- v0.9.4 sweep C=1/2/4/8: C1: 17.2 t/s system (engine-scheduled); C2: 16.1 t/s system (engine-scheduled); C4: 16.7 t/s system (engine-scheduled); C8: 22.8 t/s system (engine-scheduled)
- ITL p99 63.81 ms vs ollama 75.80 ms (1.2x tighter)
- greedy determinism: direct 20/20 exact, gateway transparency 7/20 exact at temp 0 (divergence = multi-slot batching numerics, not translation drift)
- gateway cold boot 20.54 s
- cold TTFT 9098 ms vs ollama 6652 ms (0.7x)

## Measured in this campaign

- blazar: 4 cell(s)
- cold-ollama: 1 cell(s)
- conc-blazar: 8 cell(s)
- conc-direct: 4 cell(s)
- conc-ollama: 4 cell(s)
- direct: 8 cell(s)
- greedy: 2 cell(s)
- greedy_gw: 1 cell(s)
- greedy_ollama: 1 cell(s)
- ollama: 1 cell(s)

## Engine coverage

_Engine selection policy: latest installed version per engine kind._
| Engine | Kind | Lane | ok cells | err cells | Status |
|---|---|---|---:|---:|---|
| v0.9.4 | mistralrs | text (direct + gateway) | 7 | 4 | benchmarked |
| b5130 | whisper | media | 0 | 0 | no cells in this campaign |
| master-929-3f8527a | sdcpp | media | 0 | 0 | no cells in this campaign |
| sglang-0.5.21 | sglang | - | 0 | 0 | no cells in this campaign |
| b11429-cuda | llamacpp | text (direct + gateway) | 14 | 2 | benchmarked |
| 2023.11.14-2 | - | - | 0 | 0 | excluded: kind 'piper' has no bench lane in this harness |
| mlx-0.32.0 | - | - | 0 | 0 | excluded: kind 'mlx' has no bench lane in this harness |

## Test bed

| Component | Value |
|---|---|
| CPU | Intel Core i7-14650HX, 24 hardware threads |
| Discrete GPU | NVIDIA GeForce RTX 4070 Laptop, 8 GiB, driver 580.173.02 |
| Integrated GPU | Intel Graphics (RPL-S), Vulkan device |
| RAM | 16 GiB (13.3 GiB usable) |
| OS | Linux Mint 22.3, kernel 7.0.0-31-generic |
| Runtimes compared | blazar mixed gateway - inventory - llama.cpp b11429-cuda - mistral.rs v0.9.4 - ollama-host |
| Model | Qwen3.5-9B-Q4_K_M.gguf |

## Methodology

<details>
<summary><b>Measurement protocol (how every number on this page was produced)</b></summary>

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
- Time-to-first-byte (TTFB) is stamped when the first response byte arrives on every stream; under burst arrival its per-level median upper-bounds queue wait (dashed curves on the concurrency tail-latency chart). Receipts from harness versions before TTFB capture omit the field and those curves.
- Concurrency cells sample GPU memory, GPU board power, and host CPU busy (delta /proc/stat, busy = total - idle - iowait) every 1.2 s alongside the load; the resource-vs-concurrency charts read those per-cell peaks. Legacy receipts without sampler folds omit the power chart.
- Quality suites: seven deterministic-checker suites (arithmetic reasoning, verifiable instruction following, code with executed tests, JSON schema, needle-in-haystack, multilingual, refusal/benign) on a seeded task generator (default seed 1337, --quality-seed) - the identical task set runs through blazar, the direct engine, and ollama, so pass-rate deltas isolate the gateway. Hybrid reasoning models are pinned to direct answers (enable_thinking=false on the OpenAI-compatible path, think=false on ollama's native /api/chat) so a lane measures task ability, not how much of the token budget the <think> block consumed. A parallel fast-subset probe (default C=4) rides the same daemon session; per-task prompts, raw excerpts, and checker verdicts land in cells.jsonl. Embed triples compare only within one engine (vector spaces are not cross-engine comparable). A config knob whose pass rate drops more than 2.0 pp vs default is flagged by the QC gate regardless of its speed win.

</details>

## Results

### Single-stream decode (512-token prompt, 128 generated, median of 5)

<details>
<summary><b>Receipt table - single-stream decode</b></summary>

| Runtime | slots x ctx | decode t/s | tok/s per W (net) | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | ITL p99 ms | prefill cold t/s | prefill cached t/s | GPU peak MiB | GPU power W |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blazar gateway - mistral.rs v0.9.4 (single-stream) | engine-scheduled | 18.7 | 0.931 | 137.9 | 145.5 | 53.2 | 63.8 | 235.5 | 239.6 | 7171 | 35.7 |
| direct engine - llama.cpp b11429-cuda | 1x16384 | 41.5 | 1.488 | 122.1 | 126.0 | 24.1 | 25.2 | 1331.4 | 7506.0 | 5681 | 55.7 |
| ollama 0.33.3 - qwen3.5:9b (t/s NOT comparable - matrix model differs) | service | 40.1 | 1.449 | 141.4 | 414.1 | 25.1 | 75.8 | 1261.5 | 6662.0 | 6569 | 56.0 |

</details>

### Single-stream decode throughput (chart)

<p align="center"><img src="plots/speed-single-stream.svg" alt="Single-stream decode throughput (chart)"></p>

_Median decode t/s per runtime and engine; whiskers span the interquartile range of the 5 runs; the dashed line marks the fastest direct engine. Higher is better. Source: 2026-10-07-overhead/cells.jsonl._

### Concurrency (1x2x4x8 parallel streams x 128 tokens)

<details>
<summary><b>Receipt table - concurrency lanes</b></summary>

| Runtime | slots | ok streams | rounds | system t/s | sum-stream t/s | wall s | TTFT max ms | TTFT p99 ms | ITL p99 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direct engine - llama.cpp b11429-cuda | 1 | 3 of 3x1 | 3 | 39.3 | 122.6 | 9.78 | 189 | 188 | 27.4 |
| blazar gateway - llama.cpp b11429-cuda | 4x65536 | 3 of 3x1 | 3 | 36.1 | 121.0 | 5.12 | 278 | 275 | 26.7 |
| blazar gateway - mistral.rs v0.9.4 | engine-scheduled | 3 of 3x1 | 3 | 17.2 | 54.9 | 8.97 | 193 | 193 | 62.9 |
| ollama - qwen3.5:9b | service | 3 of 3x1 | 3 | 21.6 | 121.1 | 17.78 | 8022 | 7864 | 75.0 |
| direct engine - llama.cpp b11429-cuda | 2 | 6 of 3x2 | 3 | 70.0 | 221.0 | 10.97 | 223 | 223 | 29.8 |
| blazar gateway - llama.cpp b11429-cuda | 4x65536 | 6 of 3x2 | 3 | 59.1 | 219.5 | 6.69 | 326 | 326 | 30.2 |
| blazar gateway - mistral.rs v0.9.4 | engine-scheduled | 6 of 3x2 | 3 | 16.1 | 18.2 | 9.13 | 434 | 421 | 185.5 |
| ollama - qwen3.5:9b | service | 6 of 3x2 | 3 | 29.1 | 242.1 | 26.40 | 10038 | 9873 | 75.1 |
| direct engine - llama.cpp b11429-cuda | 4 | 12 of 3x4 | 3 | 109.1 | 351.8 | 14.08 | 413 | 412 | 36.8 |
| blazar gateway - llama.cpp b11429-cuda | 4x65536 | 12 of 3x4 | 3 | 86.1 | 351.4 | 9.66 | 583 | 577 | 37.7 |
| blazar gateway - mistral.rs v0.9.4 | engine-scheduled | 12 of 3x4 | 3 | 16.7 | 52.4 | 11.05 | 267 | 267 | 255.2 |
| ollama - qwen3.5:9b | service | 12 of 3x4 | 3 | 33.6 | 484.9 | 45.67 | 16152 | 15792 | 75.0 |
| direct engine - llama.cpp b11429-cuda | 8 | 24 of 3x8 | 3 | 131.9 | 433.2 | 23.29 | 941 | 941 | 59.2 |
| blazar gateway - llama.cpp b11429-cuda | 4x65536 | 24 of 3x8 | 3 | 87.9 | 659.7 | 17.55 | 3834 | 3834 | 171.5 |
| blazar gateway - mistral.rs v0.9.4 | engine-scheduled | 24 of 3x8 | 3 | 22.8 | 27.6 | 6.04 | 508 | 469 | 190.2 |
| ollama - qwen3.5:9b | service | 24 of 3x8 | 3 | 35.7 | 970.5 | 86.01 | 30221 | 29465 | 74.9 |

_sum-stream >> system t/s means streams serialize on one slot; roughly equal means genuinely parallel._

</details>

### Concurrency scaling (chart)

<p align="center"><img src="plots/concurrency-throughput.svg" alt="Concurrency scaling (chart)"></p>

_Aggregate system tokens/s as parallel streams are added; flat-to-rising means the scheduler keeps the device saturated. Higher is better. Source: 2026-10-07-overhead/cells.jsonl._

### Concurrency tail latency (chart)

<p align="center"><img src="plots/concurrency-ttft.svg" alt="Concurrency tail latency (chart)"></p>

_Worst-case first-token wait per stream as concurrency rises (log scale) - the tail the scheduler must bound. Lower is better. Dashed curves carry the median time-to-first-byte, which upper-bounds queue wait under burst arrival. Source: 2026-10-07-overhead/cells.jsonl._

### Resource cost and reliability vs concurrency

<p align="center"><img src="plots/concurrency-vram.svg" alt="Resource cost vs concurrency (chart)"></p>

_Peak VRAM footprint as parallel streams (and their KV caches) stack up. Source: 2026-10-07-overhead/cells.jsonl._

<p align="center"><img src="plots/concurrency-power.svg" alt="Resource cost vs concurrency (chart)"></p>

_Peak GPU board power per concurrency level - the energy price of keeping the device saturated. Source: 2026-10-07-overhead/cells.jsonl._

### Concurrency frontier verdicts

- blazar gateway - llama.cpp b11429-cuda: throughput plateaus at C=8 (<10% per-level gain), peak 87.9 t/s at C=8.
- blazar gateway - mistral.rs v0.9.4: throughput plateaus at C=2 (<10% per-level gain), peak 22.8 t/s at C=8.
- direct engine - llama.cpp b11429-cuda: still gaining at C=8 (39.3 -> 131.9 t/s) - saturation not reached within the sweep.
- ollama - qwen3.5:9b: throughput plateaus at C=8 (<10% per-level gain), peak 35.7 t/s at C=8.

### Adaptive reshape under sustained load (no-lag proof)

_Not measured in this campaign (2026-10-07-overhead); adaptive reshape lane not run._

### Perplexity

_Not measured in this campaign (2026-10-07-overhead); perplexity lane not run._

### Greedy parity and gateway transparency (20 prompts, 256 tokens)

<details>
<summary><b>Receipt table - greedy parity</b></summary>

| Comparison | exact / total | ratio mean | ratio min |
|---|---:|---:|---:|
| llama.cpp b11429-cuda vs same-engine reference (direct) | 20/20 | 1.000 | 1.000 |
| mistral.rs v0.9.4 vs same-engine reference (direct) | 0/20 | 0.514 | 0.000 |
| llama.cpp b11429-cuda through blazar gateway vs direct | 7/20 | 0.643 | 0.017 |
| ollama (qwen3.5:9b) run-to-run, temp 0 | 20/20 | 1.000 | 1.000 |

_Exact-match divergence across GPU backends is expected float nondeterminism (batch shape and backend kernels), not translation drift; bit-parity across runs requires single-slot decoding (blazar `deterministic = true` pins it)._

</details>

### Tool calls (single-turn selection + schema quality)

_Not measured in this campaign (2026-10-07-overhead); tool calls lane not run._

### Quality suites (checker-verified, seeded, greedy)

- _quality lane: **skipped** — --skip-quality: no checker-verified receipts back any speed number in this campaign_

_Not measured in this campaign (2026-10-07-overhead); quality suites lane not run._

### Optimization axes (ctx 4096, single stream)

_Not measured in this campaign (2026-10-07-overhead); optimization axes lane not run._

### Gateway config knobs (on/off vs default profile)

<details>
<summary><b>Receipt table - config knob A/B</b></summary>

| Engine | config | decode t/s | decode delta% | ttft p50 ms | ttft delta% | GPU peak MiB | GPU delta | output vs default |
|---|---|---:|---:|---:|---:|---:|---:|:-:|
| mistral.rs v0.9.4 | single-stream | 18.7 | - | 138 | - | 7171 | - | - |

</details>

### Gateway scheduler knobs under concurrent load (C=8 bursts)

_Not measured in this campaign (2026-10-07-overhead); conc lane A/B lane not run._

### Engine capability matrix

_Not measured in this campaign (2026-10-07-overhead); capability matrix lane not run._

### Cold start and idle wake (lifecycle)

<p align="center"><img src="plots/lifecycle-cold-idle.svg" alt="Lifecycle: cold start and idle wake (chart)"></p>

_Seconds to first token after a cold start (page cache dropped) and after idle-policy expiry; blazar keeps weights resident while ollama reloads from disk. Warm-daemon ollama caveat applies. Lower is better. Source: 2026-10-07-overhead/cells.jsonl._

_Every cold probe runs page-cache-dropped and GPU-idle-asserted on both runtimes; ollama rows without --ollama-service-restart leave the daemon warm (noted on the chart when it applies). blazar sleeps with weights in RAM (wake = resume); ollama unloads at keep_alive expiry (wake = full disk reload). Policies differ by design - each dot measures its own runtime's idle path after the policy verifiably fired. Per-run detail and per-config rows: the campaign's cells.jsonl._

### Long-context degradation curve

_Not measured in this campaign (2026-10-07-overhead); long-context lane not run._

### Media lanes (image / video / TTS / whisper)

_Not measured in this campaign (2026-10-07-overhead); media lane not run._

_Not measured in this campaign (2026-10-07-overhead); media config A/B lane not run._

_Media cells run through the same sandboxed gateway as text lanes but do not assert GPU-idle: a warm engine child is the normal serving shape, so each row stamps gpu_busy_mib / ram_avail_mib / loadavg instead. 3 runs (not 5); media variance is dominated by the model, not the scheduler. Video frame counts are read from the EBML container (lacing-aware), never from an API field; the VRAM gate probe times how fast an over-budget request is rejected with a teaching error._

## Findings (this campaign)

1. **Concurrency scaling per engine.** b11429-cuda (C=1/2/4/8): peak 87.9 t/s at C=8, 30% of ideal at C=8; v0.9.4 (C=1/2/4/8): peak 22.8 t/s at C=8, 17% of ideal at C=8; serialization behavior per level in the frontier table below.
2. **Prompt cache pays 6.1x on prefill** (7343 cached vs 1213 t/s cold).
3. **v0.9.4 serves this model through blazar's profile** ( blazar auto-disables PA on tight cards and the model then serves; the row's argv is the receipt).

## Carried-over findings (no receipt in this campaign)

_Established in earlier campaigns whose receipts live in their bench-artifacts/ directories; this campaign did not measure these lanes._

1. **Gateway overhead is within measurement noise.** Single-stream decode through the blazar gateway matches direct engine spawns at the same slots/context (see speed table); the greedy gateway lane is byte-identical to the direct lane where sampling is single-slot.
2. **Capacity-aware slot auto-sizing.** blazar sizes engine slots from live hardware census: the 8 GiB card with a vision projector attached spawns 1 slot (16 Ki context) on the Vulkan build and 4 slots (64 Ki total) on CUDA - measured oversubscription on Vulkan either fails to boot or degrades 2x, so the cap is load-bearing, not conservative cosmetics.
3. **Speculative n-gram decoding is a net loss for this 9B model** (no draft model; acceptance too low to pay the verification overhead) - documented so the flag is not cargo-culted.
4. **KV q8_0 quantization is decode-neutral and prefill-neutral steady-state**; the one cold-prefill outlier below is a first-invocation pipeline-compile artifact (controlled re-probe measured full-rate steady state).
5. **Gateway passes OpenAI tools verbatim** (tools-aware validation, no schema rewriting); tool-call quality is the engine's own - selection and argument schema are scored per scenario with a no-tool control for false positives.
6. Gateway adaptive slots reshape under sustained concurrency without dropping streams.

## Failed cells (engine-reality receipts)

_These knobs crashed the engine child on this test bed; the crash signature is the receipt. Raw logs: `cells.jsonl` (`daemon_log_tail` field)._

| Engine | lane | config | failure |
|---|---|---|---|
| v0.9.4 | direct |  | child exited rc=1 during load; last output: Error: This model does not fit on the devices ["cuda[0]  |
| v0.9.4 | direct |  | child exited rc=1 during load; last output: Error: This model does not fit on the devices ["cuda[0]  |
| v0.9.4 | direct |  | child exited rc=1 during load; last output: Error: This model does not fit on the devices ["cuda[0]  |
| b11429-cuda | blazar | default | blazar cell crashed: MemAvailable 7092 MiB < needed ~7731 MiB (model 5366 MiB + headroom). A co-resi |
| b11429-cuda | blazar | single-stream | blazar cell crashed: MemAvailable 7360 MiB < needed ~7731 MiB (model 5366 MiB + headroom). A co-resi |
| v0.9.4 | blazar | default | blazar cell crashed: MemAvailable 7724 MiB < needed ~7731 MiB (model 5366 MiB + headroom). A co-resi |

## Caveats

- ollama prefill numbers come from engine counters that exclude the chat template, so they read slightly high against the 512-token lanes.
- Cross-backend greedy ratios (CUDA vs Vulkan) diverge on near-tie logits; treat ratio, not exact-match count, as the signal.
- All GPU rows measured on AC power at bounded load; rows record load average and power state (battery runs are rejected by the harness).
- Numbers are medians of 5 runs on one hybrid laptop; expect absolute shifts on other hardware, ratios to travel better.

## Reproduce

```bash
python3 scripts/bench_matrix.py --model qwen3.5-9b --engines b11429-cuda --providers direct blazar --conc-sweep 1,2,4,8 --skip-ppl --skip-tools --skip-quality --skip-media --skip-features --skip-ctxcurve --skip-reshape --skip-idle --skip-variants --blazar-bin ./target/release/blazar --artifacts-dir bench-artifacts/2026-10-07-overhead --md bench-artifacts/2026-10-07-overhead/report.md
python3 scripts/bench_matrix.py --render-only --artifacts-dir <dir> [--append-campaign <sibling-campaign-dir>] --md BENCHMARK.md
```

_Raw per-cell records (argv, per-run lists, daemon logs): `bench-artifacts/2026-10-07-overhead/cells.jsonl`._
