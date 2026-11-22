# Blazar inference benchmark

_Rendered 2026-11-22-b7-llamacpp; blazar 0.34.3._

## Executive summary

- _No complete rows._

## Measured in this campaign

- quality: 1 cell(s)
- quality-direct: 1 cell(s)

## Engine coverage

_Engine selection policy: latest installed version per engine kind._
| Engine | Kind | Lane | ok cells | err cells | Status |
|---|---|---|---:|---:|---|
| b12606-cuda | llamacpp | text (direct + gateway) | 2 | 0 | benchmarked |
| 2023.11.14-2 | - | - | 0 | 0 | excluded: kind 'piper' has no bench lane in this harness |
| b5907 | - | - | 0 | 0 | excluded: kind 'whisper' has no bench lane in this harness |
| fork-AtomicBot-ai_atomic-llama-cpp-turboquant-8d935078-cuda | - | - | 0 | 0 | excluded: latest-per-kind policy: superseded by b12606-cuda |
| fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | - | - | 0 | 0 | excluded: latest-per-kind policy: superseded by b12606-cuda |
| kokoro-1.0 | - | - | 0 | 0 | excluded: kind 'kokoro' has no bench lane in this harness |
| litert-0.19.2 | - | - | 0 | 0 | excluded: model-format sweep: this campaign sweeps a non-litertlm model; the openaihost lane serves .litertlm bundles (rerun with --model <litertlm-row>) |
| master-1043-3be2378 | - | - | 0 | 0 | excluded: kind 'sdcpp' has no bench lane in this harness |
| mlx-0.32.2 | - | - | 0 | 0 | excluded: kind 'mlx' has no bench lane in this harness |
| sglang-0.5.22 | - | - | 0 | 0 | excluded: model-format sweep: this campaign sweeps a GGUF file; sglang serves HF safetensors checkpoints (rerun with --model <safetensors-row>) |
| v0.9.5 | - | - | 0 | 0 | excluded: kind 'mistralrs' has no bench lane in this harness |
| vllm-0.34.0 | - | - | 0 | 0 | excluded: model-format sweep: this campaign sweeps a GGUF file; the vllm lane serves HF safetensors checkpoints (rerun with --model <safetensors-row>) |

## Test bed

| Component | Value |
|---|---|
| CPU | Intel Core i7-14650HX, 24 hardware threads |
| Discrete GPU | NVIDIA GeForce RTX 4070 Laptop, 8 GiB, driver 580.173.02 |
| Integrated GPU | Intel Graphics (RPL-S), Vulkan device |
| RAM | 16 GiB (13.3 GiB usable) |
| OS | Linux Mint 22.3, kernel 7.0.0-31-generic |
| Runtimes compared | blazar 0.34.3 gateway - inventory - llama.cpp b12606-cuda |
| Model | qwen2.5-0.5b-instruct-q4_k_m.gguf |

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

_Not measured in this campaign (2026-11-22-b7-llamacpp); single-stream speed lane not run._

### Concurrency (N parallel streams x 128 tokens)

_Not measured in this campaign (2026-11-22-b7-llamacpp); concurrency lane not run._

_sum-stream >> system t/s means streams serialize on one slot; roughly equal means genuinely parallel._

### Concurrency frontier verdicts

_Not measured in this campaign (2026-11-22-b7-llamacpp); concurrency lane not run._

### Adaptive reshape under sustained load (no-lag proof)

_Not measured in this campaign (2026-11-22-b7-llamacpp); adaptive reshape lane not run._

### Perplexity

_Not measured in this campaign (2026-11-22-b7-llamacpp); perplexity lane not run._

### Greedy parity and gateway transparency (20 prompts, 256 tokens)

_Not measured in this campaign (2026-11-22-b7-llamacpp); greedy parity lane not run._

_Exact-match divergence across GPU backends is expected float nondeterminism (batch shape and backend kernels), not translation drift; bit-parity across runs requires single-slot decoding (blazar `deterministic = true` pins it)._

### Tool calls (single-turn selection + schema quality)

_Not measured in this campaign (2026-11-22-b7-llamacpp); tool calls lane not run._

### Quality suites (checker-verified, seeded, greedy)

- _quality lane: **ran** - 2 scored row(s); suites: code, instruct, multilingual, niah, reason, safety, schema_

<details>
<summary><b>Receipt table - quality suites</b></summary>

| Runtime | Engine | config | reason | instruct | code | schema | niah | multilingual | safety | overall | conc quality |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | llama.cpp b12606-cuda | direct | 0/12 | 4/10 | 2/5 | 5/6 | 6/6 | 3/8 | 12/12 | 32/59 | - |
| blazar | llama.cpp b12606-cuda | default | 0/12 | 4/10 | 2/5 | 6/6 | 6/6 | 3/8 | 12/12 | 33/59 | 7/18 @C=4 |

</details>

- Gateway parity on llama.cpp b12606-cuda: overall pass 54.2% direct vs 55.9% through blazar (+1.7 pp) on the identical seeded task set.
- Quality under load on llama.cpp b12606-cuda: 7/18 at C=4 (38.9%) vs 33.3% serial on the same fast subset - held.

### Quality suites (chart)

<p align="center"><img src="bench-artifacts/2026-11-22-b7-llamacpp/plots/quality-suites.svg" alt="Quality suites (chart)"></p>

_Deterministic-checker pass rates per suite (reasoning, instruction following, code, schema, needle-in-haystack, multilingual, safety) across runtimes on the identical seeded task set. Source: 2026-11-22-b7-llamacpp/cells.jsonl._

### Gateway quality parity (chart)

<p align="center"><img src="bench-artifacts/2026-11-22-b7-llamacpp/plots/quality-parity.svg" alt="Gateway quality parity (chart)"></p>

_Per-suite pass-rate delta of routing through blazar vs the same engine build driven directly; bars hugging the zero axis are the gateway being output-transparent. Source: 2026-11-22-b7-llamacpp/cells.jsonl._

### Optimization axes (ctx 4096, single stream)

_Not measured in this campaign (2026-11-22-b7-llamacpp); optimization axes lane not run._

### Gateway config knobs (on/off vs default profile)

_Not measured in this campaign (2026-11-22-b7-llamacpp); config knob A/B lane not run._

### Gateway scheduler knobs under concurrent load (C=8 bursts)

_Not measured in this campaign (2026-11-22-b7-llamacpp); conc lane A/B lane not run._

### Engine capability matrix

_Not measured in this campaign (2026-11-22-b7-llamacpp); capability matrix lane not run._

### Cold start and idle wake (lifecycle)

_Not measured in this campaign (2026-11-22-b7-llamacpp); cold/idle lanes not run._

_Every cold probe runs page-cache-dropped and GPU-idle-asserted on both runtimes; ollama rows without --ollama-service-restart leave the daemon warm (noted on the chart when it applies). blazar sleeps with weights in RAM (wake = resume); ollama unloads at keep_alive expiry (wake = full disk reload). Policies differ by design - each dot measures its own runtime's idle path after the policy verifiably fired. Per-run detail and per-config rows: the campaign's cells.jsonl._

### Long-context degradation curve

_Not measured in this campaign (2026-11-22-b7-llamacpp); long-context lane not run._

### Media lanes (image / video / TTS / whisper)

_Not measured in this campaign (2026-11-22-b7-llamacpp); media lane not run._

_Not measured in this campaign (2026-11-22-b7-llamacpp); media config A/B lane not run._

_Media cells run through the same sandboxed gateway as text lanes but do not assert GPU-idle: a warm engine child is the normal serving shape, so each row stamps gpu_busy_mib / ram_avail_mib / loadavg instead. 3 runs (not 5); media variance is dominated by the model, not the scheduler. Video frame counts are read from the EBML container (lacing-aware), never from an API field; the VRAM gate probe times how fast an over-budget request is rejected with a teaching error._

## Findings (this campaign)

_No complete findings from this campaign's cells; see carried-over findings below._

## Carried-over findings (no receipt in this campaign)

_Established in earlier campaigns whose receipts live in their bench-artifacts/ directories; this campaign did not measure these lanes._

1. **Gateway overhead is within measurement noise.** Single-stream decode through the blazar gateway matches direct engine spawns at the same slots/context (see speed table); the greedy gateway lane is byte-identical to the direct lane where sampling is single-slot.
2. **Capacity-aware slot auto-sizing.** blazar sizes engine slots from live hardware census: the 8 GiB card with a vision projector attached spawns 1 slot (16 Ki context) on the Vulkan build and 4 slots (64 Ki total) on CUDA - measured oversubscription on Vulkan either fails to boot or degrades 2x, so the cap is load-bearing, not conservative cosmetics.
3. **Concurrency scales where capacity allows.** 4 streams through CUDA gateway hold near-direct system throughput; the Vulkan single-slot shape serializes streams (per-stream latency stays excellent; system throughput caps at one stream's rate) - a capacity trade, not a scheduling defect.
4. **Prompt cache pays ~6-7x on prefill.** Cached-prefix prefill runs thousands of tokens/s vs hundreds cold.
5. **Speculative n-gram decoding is a net loss for this 9B model** (no draft model; acceptance too low to pay the verification overhead) - documented so the flag is not cargo-culted.
6. **KV q8_0 quantization is decode-neutral and prefill-neutral steady-state**; the one cold-prefill outlier below is a first-invocation pipeline-compile artifact (controlled re-probe measured full-rate steady state).
7. **mistral.rs 0.9.3 with default paged attention cannot fit this model on an 8 GiB card** (upstream sizes KV as a fraction of total VRAM); blazar's profile auto-disables paged attention on tight cards and the model then serves correctly.
8. **Gateway passes OpenAI tools verbatim** (tools-aware validation, no schema rewriting); tool-call quality is the engine's own - selection and argument schema are scored per scenario with a no-tool control for false positives.
9. Gateway adaptive slots reshape under sustained concurrency without dropping streams.

## Caveats

- ollama prefill numbers come from engine counters that exclude the chat template, so they read slightly high against the 512-token lanes.
- Cross-backend greedy ratios (CUDA vs Vulkan) diverge on near-tie logits; treat ratio, not exact-match count, as the signal.
- All GPU rows measured on AC power at bounded load; rows record load average and power state (battery runs are rejected by the harness).
- Numbers are medians of 5 runs on one hybrid laptop; expect absolute shifts on other hardware, ratios to travel better.

## Reproduce

```bash
python3 scripts/bench_matrix.py --blazar-bin target/release/blazar --model qwen2.5-0.5b-instruct --engines b12606-cuda --quality-only --artifacts-dir bench-artifacts/2026-11-22-b7-llamacpp --md B7-llamacpp.md
python3 scripts/bench_matrix.py --render-only --artifacts-dir <dir> [--append-campaign <sibling-campaign-dir>] --md BENCHMARK.md
```

_Raw per-cell records (argv, per-run lists, daemon logs): `bench-artifacts/2026-11-22-b7-llamacpp/cells.jsonl`._
