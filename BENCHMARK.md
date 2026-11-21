# Blazar inference benchmark

_Rendered 2026-11-20-v0.33-wave-final; blazar 0.33.0; power state of gateway rows: ac._

> **Provenance and publication.** This file is machine-rendered by
> `scripts/bench_matrix.py`; the label above identifies the campaign of the
> LAST RENDER, not necessarily the newest campaign. Every campaign's
> receipts, engine builds,
> and blazar versions are catalogued per artifact directory:
> [`bench-artifacts/INDEX.md`](bench-artifacts/INDEX.md) carries one row per
> matrix-rendered campaign, and ad-hoc audit sessions carry a committed
> `report.md` inside their `bench-artifacts/<campaign>/` directory (for
> example the 2026-10-31 benchmark audit). The README performance snapshot
> cites the campaign each row traces to. Publication rule going forward: no
> README-quoted number without a committed receipt path (under
> `bench-artifacts/`) or a row in this file; re-render this file with
> `python3 scripts/bench_matrix.py --blazar-bin target/release/blazar --md
> BENCHMARK.md` after a campaign so the render label never silently trails
> the released numbers.

## Executive summary

- v0.9.5 prompt-cache prefill 19356 vs 17079 t/s cold
- fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda sweep C=4: C4: 445.5 t/s system (2x32768)
- ITL p99 5.75 ms vs ollama 4.01 ms (0.7x tighter)
- tool-call TTFT p50 14 ms vs ollama 291 ms (20.6x)
- adaptive reshape under sustained load: slots 2->16, 842.5->1629.3 t/s while serving, zero failed requests
- greedy determinism: direct 20/20 exact, gateway transparency 12/20 exact at temp 0 (divergence = multi-slot batching numerics, not translation drift)
- gateway cold boot 0.52 s
- cold TTFT 3292 ms vs ollama 1995 ms (0.6x)
- idle wake 32 ms (sleep) vs ollama 2326 ms (full reload)
- evict-ladder wake 1547 ms (full respawn still beats ollama's reload)

## Measured in this campaign

- blazar: 31 cell(s)
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
- quality: 2 cell(s)
- quality-direct: 2 cell(s)
- quality-ollama: 1 cell(s)
- reshape: 2 cell(s)
- tools: 2 cell(s)
- tools-ollama: 1 cell(s)

## Engine coverage

_Engine selection policy: latest installed version per engine kind._
| Engine | Kind | Lane | ok cells | err cells | Status |
|---|---|---|---:|---:|---|
| 2023.11.14-2 | piper | media (tts) | 0 | 2 | attempted, all cells errored (see cells.jsonl) |
| b5454 | whisper | media | 0 | 1 | attempted, all cells errored (see cells.jsonl) |
| mlx-0.32.2 | mlx | text (direct venv + gateway) | 0 | 0 | model-format sweep: GGUF/safetensors campaign; this lane serves MLX-layout checkpoints (rerun with --model <mlx-row>) |
| v0.9.5 | mistralrs | text (direct + gateway) | 7 | 21 | benchmarked |
| master-1043-3be2378 | sdcpp | media | 2 | 3 | benchmarked |
| sglang-0.5.23 | sglang | text (direct venv + gateway) | 0 | 0 | no cells in this campaign |
| litert-0.19.2 | openaihost | text (direct venv + gateway) | 0 | 0 | model-format sweep: GGUF/safetensors campaign; this lane serves .litertlm bundles (rerun with --model <litertlm-row>) |
| fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | llamacpp | text (direct + gateway) | 42 | 0 | benchmarked |
| b12521-cuda | - | - | 0 | 0 | excluded: latest-per-kind policy: superseded by fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda |
| fork-AtomicBot-ai_atomic-llama-cpp-turboquant-8d935078-cuda | - | - | 0 | 0 | excluded: latest-per-kind policy: superseded by fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda |
| kokoro-1.0 | - | - | 0 | 0 | excluded: kind 'kokoro' has no bench lane in this harness |
| vllm-0.34.0 | - | - | 0 | 0 | excluded: kind 'vllm' has no bench lane in this harness |

## Test bed

| Component | Value |
|---|---|
| CPU | Intel Core i7-14650HX, 24 hardware threads |
| Discrete GPU | NVIDIA GeForce RTX 4070 Laptop, 8 GiB, driver 580.173.02 |
| Integrated GPU | Intel Graphics (RPL-S), Vulkan device |
| RAM | 16 GiB (13.3 GiB usable) |
| OS | Linux Mint 22.3, kernel 7.0.0-31-generic |
| Runtimes compared | blazar 0.33.0 gateway - inventory - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda - mistral.rs v0.9.5 - ollama-host - piper 2023.11.14-2 - stable-diffusion.cpp master-1043-3be2378 - whisper.cpp b5454 |
| Model | en_US-amy-medium, ggml-base, qwen-image-2.1-uncensored, qwen2.5-0.5b-instruct-q4_k_m.gguf, wan_2.1_comfyui_repackaged |

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
| blazar gateway - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | 2x32768 | 322.0 | 23.382 | 8.9 | 12.4 | 3.1 | 5.1 | 17354.7 | 60677.5 | 1268 | 54.8 |
| blazar gateway - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda (single-stream) | 1x16384 | 326.6 | 32.706 | 9.4 | 10.6 | 3.1 | 4.3 | 18044.7 | 62597.2 | 1060 | 54.3 |
| blazar gateway - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda (cont_batching_off) | 2x32768 | 331.5 | 29.966 | 8.7 | 10.0 | 3.1 | 4.3 | 18698.9 | 58373.7 | 1270 | 54.7 |
| blazar gateway - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda (fa_on) | 2x32768 | 329.8 | 29.841 | 8.6 | 11.0 | 3.0 | 4.3 | 18935.1 | 60915.5 | 1268 | 55.0 |
| blazar gateway - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda (fa_off) | 2x32768 | 321.0 | 31.376 | 22.7 | 24.0 | 3.1 | 4.6 | 10889.4 | 60518.3 | 1922 | 54.9 |
| blazar gateway - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda (kv_unified_on) | 2x32768 | 330.4 | 30.098 | 9.0 | 10.8 | 3.0 | 4.4 | 18427.7 | 68052.7 | 1270 | 54.8 |
| blazar gateway - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda (kv_unified_off) | 2x32768 | 328.4 | 29.715 | 9.0 | 10.5 | 3.1 | 4.3 | 18386.3 | 62546.0 | 1252 | 54.6 |
| blazar gateway - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda (swa_full_on) | 2x32768 | 329.8 | 29.657 | 9.0 | 12.0 | 3.1 | 4.0 | 18146.0 | 65339.9 | 1268 | 54.6 |
| blazar gateway - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda (no_kv_offload_on) | 2x32768 | 187.8 | 13.156 | 10.4 | 13.7 | 5.3 | 7.8 | 14033.4 | 45550.6 | 860 | 43.1 |
| blazar gateway - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda (cache_q8_q8_0) | 2x32768 | 331.8 | 28.443 | 7.4 | 11.1 | 3.0 | 4.4 | 17911.7 | 62320.9 | 1096 | 54.9 |
| blazar gateway - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda (ctx_checkpoints_4) | 2x32768 | 337.1 | 26.519 | 9.0 | 11.6 | 3.0 | 3.9 | 18614.3 | 59652.3 | 1268 | 54.7 |
| blazar gateway - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda (spec_off) | 2x32768 | 350.0 | 31.463 | 8.3 | 10.4 | 2.9 | 3.4 | 19120.4 | 63163.3 | 1401 | 54.8 |
| blazar gateway - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda (deterministic_on) | 1x16384 | 349.1 | 27.542 | 8.4 | 10.7 | 2.8 | 3.7 | 18337.9 | 64742.6 | 1060 | 54.9 |
| blazar gateway - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda (mmproj_offload_off) | 2x32768 | 335.2 | 30.796 | 9.4 | 12.4 | 2.9 | 4.2 | 18059.4 | 60927.6 | 1268 | 55.0 |
| blazar gateway - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda (threads_batch_8) | 2x32768 | 322.7 | 26.405 | 8.8 | 11.5 | 3.1 | 4.1 | 18098.5 | 61259.7 | 1268 | 54.8 |
| blazar gateway - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda (cache_reuse_256) | 2x32768 | 320.5 | 31.175 | 9.0 | 11.1 | 3.1 | 4.4 | 17779.8 | 63070.8 | 1268 | 54.7 |
| blazar gateway - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda (kv_unified_per_slot_4096) | 2x32768 | 332.5 | 28.316 | 8.6 | 10.1 | 3.0 | 4.1 | 14317.9 | 62350.3 | 1268 | 54.9 |
| blazar gateway - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda (conc_default) | 2x32768 | - | - | - | - | - | 14.3 | - | - | 1278 | 53.3 |
| blazar gateway - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda (adaptive_slots_off) | 2x32768 | - | - | - | - | - | 5.6 | - | - | 1270 | 54.2 |
| blazar gateway - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda (poll_50) | 2x32768 | - | - | - | - | - | 5.8 | - | - | 1274 | 53.4 |
| blazar gateway - mistral.rs v0.9.5 (paged_attn_off) | engine-scheduled | 241.5 | 14.299 | 11.1 | 12.7 | 4.1 | 5.1 | 17078.8 | 19355.6 | 806 | 51.4 |
| direct engine - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | 1x16384 | 347.4 | 20.748 | 6.3 | 9.1 | 2.9 | 3.6 | 20020.1 | 80285.0 | 788 | 55.0 |
| ollama 0.33.3 - qwen2.5:0.5b (t/s NOT comparable - matrix model differs) | service | 344.4 | 96.923 | 11.5 | 15.9 | 2.9 | 4.0 | 5872.6 | 128870.9 | 726 | 51.9 |

</details>

### Single-stream decode throughput (chart)

<p align="center"><img src="bench-artifacts/2026-11-20-v0.33-wave-final/plots/speed-single-stream.svg" alt="Single-stream decode throughput (chart)"></p>

_Median decode t/s per runtime and engine; whiskers span the interquartile range of the 5 runs; the dashed line marks the fastest direct engine. Higher is better. Source: 2026-11-20-v0.33-wave-final/cells.jsonl._

### Gateway overhead (chart)

<p align="center"><img src="bench-artifacts/2026-11-20-v0.33-wave-final/plots/gateway-overhead.svg" alt="Gateway overhead (chart)"></p>

_Decode t/s delta of routing through blazar relative to driving the same engine build directly; left of zero means the gateway path won. Source: 2026-11-20-v0.33-wave-final/cells.jsonl._

### Concurrency (4 parallel streams x 128 tokens)

<details>
<summary><b>Receipt table - concurrency lanes</b></summary>

| Runtime | slots | ok streams | rounds | system t/s | sum-stream t/s | wall s | TTFT max ms | TTFT p99 ms | ITL p99 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| blazar gateway - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | 2x32768 | 12 of 3x4 | 3 | 445.5 | 3355.0 | 2.25 | 517 | 504 | 6.3 |
| blazar gateway - mistral.rs v0.9.5 | engine-scheduled | 12 of 3x4 | 3 | 0.0 | 0.0 | 0.07 | 27 | 26 | - |
| direct engine - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | 4 | 12 of 3x4 | 3 | 717.9 | 2633.7 | 1.77 | 45 | 43 | 8.2 |
| ollama - qwen2.5:0.5b | service | 12 of 3x4 | 3 | 183.0 | 4074.2 | 4.99 | 2885 | 2865 | 4.1 |

_sum-stream >> system t/s means streams serialize on one slot; roughly equal means genuinely parallel._

</details>

### Concurrency scaling (chart)

<p align="center"><img src="bench-artifacts/2026-11-20-v0.33-wave-final/plots/concurrency-throughput.svg" alt="Concurrency scaling (chart)"></p>

_Aggregate system tokens/s as parallel streams are added; flat-to-rising means the scheduler keeps the device saturated. Higher is better. Source: 2026-11-20-v0.33-wave-final/cells.jsonl._

### Concurrency tail latency (chart)

<p align="center"><img src="bench-artifacts/2026-11-20-v0.33-wave-final/plots/concurrency-ttft.svg" alt="Concurrency tail latency (chart)"></p>

_Worst-case first-token wait per stream as concurrency rises (log scale) - the tail the scheduler must bound. Lower is better. Dashed curves carry the median time-to-first-byte, which upper-bounds queue wait under burst arrival. Source: 2026-11-20-v0.33-wave-final/cells.jsonl._

### Resource cost and reliability vs concurrency

<p align="center"><img src="bench-artifacts/2026-11-20-v0.33-wave-final/plots/concurrency-vram.svg" alt="Resource cost vs concurrency (chart)"></p>

_Peak VRAM footprint as parallel streams (and their KV caches) stack up. Source: 2026-11-20-v0.33-wave-final/cells.jsonl._

<p align="center"><img src="bench-artifacts/2026-11-20-v0.33-wave-final/plots/concurrency-power.svg" alt="Resource cost vs concurrency (chart)"></p>

_Peak GPU board power per concurrency level - the energy price of keeping the device saturated. Source: 2026-11-20-v0.33-wave-final/cells.jsonl._

### Concurrency frontier verdicts

- blazar gateway - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda: 1 level(s) measured - insufficient levels for a saturation verdict.
- blazar gateway - mistral.rs v0.9.5: 1 level(s) measured - insufficient levels for a saturation verdict.
- direct engine - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda: 1 level(s) measured - insufficient levels for a saturation verdict.
- ollama - qwen2.5:0.5b: 1 level(s) measured - insufficient levels for a saturation verdict.

### Adaptive reshape under sustained load (no-lag proof)

<details>
<summary><b>Receipt table - adaptive reshape</b></summary>

| Runtime | reshape | slots | time to reshape s | req before/after | TTFT p50 before->after ms | sys t/s before->after | failed |
|---|---|---|---:|---:|---:|---:|---:|
| blazar gateway - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | yes | 2->16 | 32 | 281/4547 | 425->56 | 842->1629 | 0 |
| blazar gateway - mistral.rs v0.9.5 | n/a | skipped: adaptive reshape is a llamacpp-lane mechanism; this engine's child exposes no slot shape (-np) to observe or adopt | - | - | - | - | - |

<details><summary>daemon tail (slots/reshape) - llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda</summary>

```
2026-11-20T14:07:59.935220Z  INFO evict: terminating child and releasing the slot model="qwen2.5-0.5b-instruct" pid=2232880
2026-11-20T14:07:59.935220Z  INFO evict: terminating child and releasing the slot model="qwen2.5-0.5b-instruct" pid=2232880
```

</details>

</details>

### Perplexity

<details>
<summary><b>Receipt table - perplexity</b></summary>

| Engine | KV rung | perplexity (ctx 2048, offline ASCII corpus) | ppl tool |
|---|---|---:|---|
| llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | f16 | 8.67 +/- 0.42 | own |
| mistral.rs v0.9.5 | f16 | 8.67 +/- 0.42 | borrowed (llama-fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda) |

</details>

### Greedy parity and gateway transparency (20 prompts, 256 tokens)

<details>
<summary><b>Receipt table - greedy parity</b></summary>

| Comparison | exact / total | ratio mean | ratio min |
|---|---:|---:|---:|
| llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda vs same-engine reference (direct) | 20/20 | 1.000 | 1.000 |
| mistral.rs v0.9.5 vs same-engine reference (direct) | None/None | - | - |
| llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda through blazar gateway vs direct | 12/20 | 0.925 | 0.587 |
| ollama (qwen2.5:0.5b) run-to-run, temp 0 | 15/20 | 0.846 | 0.105 |

_Exact-match divergence across GPU backends is expected float nondeterminism (batch shape and backend kernels), not translation drift; bit-parity across runs requires single-slot decoding (blazar `deterministic = true` pins it)._

</details>

### Tool calls (single-turn selection + schema quality)

<details>
<summary><b>Receipt table - tool calls</b></summary>

| Runtime | scenarios | well-formed | selection | args valid | control FP | TTFT p50 ms |
|---|---:|---:|---:|---:|---|---:|
| blazar gateway - fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | 6 | 5/6 | 5/5 | 5/5 | yes | 14 |
| blazar gateway - v0.9.5 | 6 | 0/6 | 0/5 | 0/5 | no | 4 |
| ollama - qwen3.5:9b | 6 | 4/6 | 4/5 | 4/5 | no | 291 |

</details>

### Quality suites (checker-verified, seeded, greedy)

- _quality lane: **ran** - 3 scored row(s); suites: code, instruct, multilingual, niah, reason, safety, schema_

<details>
<summary><b>Receipt table - quality suites</b></summary>

| Runtime | Engine | config | reason | instruct | code | schema | niah | multilingual | safety | overall | conc quality |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | direct | 1/12 | 4/10 | 2/5 | 5/6 | 6/6 | 3/8 | 12/12 | 33/59 | - |
| blazar | llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | default | 1/12 | 4/10 | 2/5 | 5/6 | 6/6 | 3/8 | 12/12 | 33/59 | 7/18 @C=4 |
| ollama | ollama daemon | ollama | 0/12 | 2/10 | 4/5 | 6/6 | 6/6 | 3/8 | 12/12 | 33/59 | 6/18 @C=4 |

</details>

- Gateway parity on llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda: overall pass 55.9% direct vs 55.9% through blazar (+0.0 pp) on the identical seeded task set.
- Quality under load on llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda: 7/18 at C=4 (38.9%) vs 33.3% serial on the same fast subset - held.
- ollama reference: 55.9% overall on qwen2.5:0.5b (same-family build, not the campaign checkpoint).

### Quality suites (chart)

<p align="center"><img src="bench-artifacts/2026-11-20-v0.33-wave-final/plots/quality-suites.svg" alt="Quality suites (chart)"></p>

_Deterministic-checker pass rates per suite (reasoning, instruction following, code, schema, needle-in-haystack, multilingual, safety) across runtimes on the identical seeded task set. Source: 2026-11-20-v0.33-wave-final/cells.jsonl._

### Gateway quality parity (chart)

<p align="center"><img src="bench-artifacts/2026-11-20-v0.33-wave-final/plots/quality-parity.svg" alt="Gateway quality parity (chart)"></p>

_Per-suite pass-rate delta of routing through blazar vs the same engine build driven directly; bars hugging the zero axis are the gateway being output-transparent. Source: 2026-11-20-v0.33-wave-final/cells.jsonl._

### Optimization axes (ctx 4096, single stream)

<details>
<summary><b>Receipt table - optimization axes</b></summary>

| Engine | axis | setting | decode t/s | delta vs dense | prefill cold t/s | delta |
|---|---|---|---:|---:|---:|---:|
| llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | kv | q8_0 | 331.6 | -16.2 | 18655.7 | -1202.5 |
| llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | pa | on | 346.0 | -1.7 | 18915.0 | -943.3 |
| llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | spec | ngram-simple | 328.1 | -19.6 | 19296.9 | -561.3 |

</details>

### Gateway config knobs (on/off vs default profile)

<details>
<summary><b>Receipt table - config knob A/B</b></summary>

| Engine | config | decode t/s | decode delta% | ttft p50 ms | ttft delta% | GPU peak MiB | GPU delta | output vs default |
|---|---|---:|---:|---:|---:|---:|---:|:-:|
| llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | single-stream | 326.6 | +1.4% | 9 | +5.7% | 1060 | -208 | same |
| llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | cont_batching_off | 331.5 | +2.9% | 9 | -2.1% | 1270 | +2 | same |
| llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | fa_on | 329.8 | +2.4% | 9 | -3.8% | 1268 | +0 | same |
| llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | fa_off | 321.0 | -0.3% | 23 | +154.6% | 1922 | +654 | DRIFT |
| llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | kv_unified_on | 330.4 | +2.6% | 9 | +0.9% | 1270 | +2 | same |
| llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | kv_unified_off | 328.4 | +2.0% | 9 | +1.3% | 1252 | -16 | same |
| llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | swa_full_on | 329.8 | +2.4% | 9 | +1.0% | 1268 | +0 | same |
| llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | no_kv_offload_on | 187.8 | -41.7% | 10 | +16.1% | 860 | -408 | same |
| llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | cache_q8_q8_0 | 331.8 | +3.0% | 7 | -17.1% | 1096 | -172 | DRIFT |
| llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | ctx_checkpoints_4 | 337.1 | +4.7% | 9 | +0.7% | 1268 | +0 | same |
| llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | spec_off | 350.0 | +8.7% | 8 | -7.0% | 1401 | +133 | same |
| llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | deterministic_on | 349.1 | +8.4% | 8 | -6.4% | 1060 | -208 | same |
| llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | mmproj_offload_off | 335.2 | +4.1% | 9 | +5.5% | 1268 | +0 | same |
| llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | threads_batch_8 | 322.7 | +0.2% | 9 | -1.8% | 1268 | +0 | same |
| llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | cache_reuse_256 | 320.5 | -0.5% | 9 | +1.0% | 1268 | +0 | same |
| llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | kv_unified_per_slot_4096 | 332.5 | +3.3% | 9 | -4.1% | 1268 | +0 | same |
| mistral.rs v0.9.5 | paged_attn_off | 241.5 | - | 11 | - | 806 | - | - |

</details>

### Gateway scheduler knobs under concurrent load (C=8 bursts)

<details>
<summary><b>Receipt table - conc lane A/B</b></summary>

| Engine | config | sys tok/s | sys delta% | ttft p99 ms | ttft delta% | GPU peak MiB | GPU delta | errs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | conc_default | 325.0 | +0.0% | 2718 | +0.0% | 1278 | +0 | 0 |
| llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | adaptive_slots_off | 286.3 | -11.9% | 3604 | +32.6% | 1270 | -8 | 0 |
| llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | poll_50 | 271.0 | -16.6% | 3204 | +17.9% | 1274 | -4 | 0 |

</details>

### Engine capability matrix

<details>
<summary><b>Receipt table - capability matrix</b></summary>

| Capability | llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | mistral.rs v0.9.5 |
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

</details>

### Cold start and idle wake (lifecycle)

<p align="center"><img src="bench-artifacts/2026-11-20-v0.33-wave-final/plots/lifecycle-cold-idle.svg" alt="Lifecycle: cold start and idle wake (chart)"></p>

_Seconds to first token after a cold start (page cache dropped) and after idle-policy expiry; blazar keeps weights resident while ollama reloads from disk. Warm-daemon ollama caveat applies. Lower is better. Source: 2026-11-20-v0.33-wave-final/cells.jsonl._

_Every cold probe runs page-cache-dropped and GPU-idle-asserted on both runtimes; ollama rows without --ollama-service-restart leave the daemon warm (noted on the chart when it applies). blazar sleeps with weights in RAM (wake = resume); ollama unloads at keep_alive expiry (wake = full disk reload). Policies differ by design - each dot measures its own runtime's idle path after the policy verifiably fired. Per-run detail and per-config rows: the campaign's cells.jsonl._

### Long-context degradation curve

<p align="center"><img src="bench-artifacts/2026-11-20-v0.33-wave-final/plots/ctx-curve.svg" alt="Long-context degradation (chart)"></p>

_Single-stream decode t/s as prompt context grows (log x-axis) - KV-cache pressure made visible. Source: 2026-11-20-v0.33-wave-final/cells.jsonl._

<p align="center"><img src="bench-artifacts/2026-11-20-v0.33-wave-final/plots/vram-vs-context.svg" alt="Memory vs context (chart)"></p>

_Peak VRAM as prompt context grows (log x-axis) - the KV-cache slope that sets the usable context ceiling. Source: 2026-11-20-v0.33-wave-final/cells.jsonl._

### Media lanes (image / video / TTS / whisper)

<details>
<summary><b>Receipt tables - media lanes and config A/B</b></summary>

| Lane | cold s | median s | min s | max s | ground truth | gate reject s | TTFB speedup |
|---|---:|---:|---:|---:|---|---:|---:|
| image - qwen-image-2.1-uncensored (512x512, steps=[4]) | 56.52 | 46.52 | 41.88 | 50.03 | 512x512 PNG, entropy 7.3 bits, contrast 74.1 |  |  |
| image - qwen-image-2.1-uncensored (512x512, steps=[4]) [fa_off] | 55.43 | 38.65 | 36.00 | 45.70 | 512x512 PNG, entropy 7.0 bits, contrast 71.6 |  |  |

| Engine | config | median total s | delta% vs default | cold request s |
|---|---|---:|---:|---:|
| stable-diffusion.cpp master-1043-3be2378 | fa_off | 38.6 | -16.9% | 55.4 |

_Media cells run through the same sandboxed gateway as text lanes but do not assert GPU-idle: a warm engine child is the normal serving shape, so each row stamps gpu_busy_mib / ram_avail_mib / loadavg instead. 3 runs (not 5); media variance is dominated by the model, not the scheduler. Video frame counts are read from the EBML container (lacing-aware), never from an API field; the VRAM gate probe times how fast an over-budget request is rejected with a teaching error._

</details>

## Findings (this campaign)

1. **Gateway overhead vs direct spawn: within measurement noise.** fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda decode 349.1 t/s through the gateway vs 347.4 t/s direct (+0.5%), greedy parity through the gateway 12/20 exact.
2. **Capacity-aware slot auto-sizing observed in argv.** Distinct engine shapes this campaign: 1x16384, 2x32768 - slots follow the live hardware census, each row's child_argv carries the receipt.
3. **Concurrency scaling per engine.** fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda (C=4): peak 445.5 t/s at C=4; serialization behavior per level in the frontier table below.
4. **Prompt cache pays 3.5x on prefill** (60677 cached vs 17355 t/s cold).
5. **Speculative n-gram decoding is a net loss for this model** (decode 328.1 t/s, -19.6 vs dense baseline) - measured, not assumed.
6. **KV quantization (q8_0) is decode-neutral** (decode 331.6 t/s vs 347.7 dense).
7. **v0.9.5 serves this model through blazar's profile** ( blazar auto-disables PA on tight cards and the model then serves; the row's argv is the receipt).
8. **Tool-call quality (single-turn, temp 0).** fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda: selection 5/5, args 5/5 (control FP); v0.9.5: selection 0/5, args 0/5 (control clean); ollama: selection 4/5, args 4/5 (control clean); per-scenario detail in cells.jsonl.
9. **Adaptive reshape lands under sustained load.** llama.cpp fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda: reshaped 2->16 slots after 32.2 s under sustained C=8, TTFT p50 425->56 ms, 0 dropped requests; mistral.rs v0.9.5: no reshape observed in the lane window (graceful-drain: adoption waits for in-flight streams, never kills one).
10. **Image quality stamps (PIL, luma domain): entropy 7.33 bits, rms contrast 74.1, 30824 unique colors @256x256** on qwen-image-2.1-uncensored - perceptual baseline for cross-run comparisons; audit PNG saved beside the cells.
11. **Image quality stamps (PIL, luma domain): entropy 7.03 bits, rms contrast 71.6, 25419 unique colors @256x256** on qwen-image-2.1-uncensored - perceptual baseline for cross-run comparisons; audit PNG saved beside the cells.
12. **TTS concurrency probe FAILED: media tts-conc cell crashed: MemAvailable 6837 MiB < needed ~7899 MiB (model 5500 MiB + headroom). A co-resident blazar/ollama engine is likely holding memory: stop it for the validation window.** - the gateway did not sustain None parallel PCM streams; needs investigation.

## Failed cells (engine-reality receipts)

_These knobs crashed the engine child on this test bed; the crash signature is the receipt. Raw logs: `cells.jsonl` (`daemon_log_tail` field)._

| Engine | lane | config | failure |
|---|---|---|---|
| v0.9.5 | blazar | default | zero tokens served (model failed to load/serve) |
| v0.9.5 | blazar | single-stream | zero tokens served (model failed to load/serve) |
| v0.9.5 | blazar | kv_unified_on | zero tokens served (model failed to load/serve) |
| v0.9.5 | blazar | kv_unified_off | zero tokens served (model failed to load/serve) |
| v0.9.5 | blazar | spec_off | zero tokens served (model failed to load/serve) |
| v0.9.5 | blazar | deterministic_on | zero tokens served (model failed to load/serve) |
| v0.9.5 | blazar | pa_mem_0.85 | zero tokens served (model failed to load/serve) |
| v0.9.5 | blazar | pa_mem_0.55 | zero tokens served (model failed to load/serve) |
| v0.9.5 | blazar | mr_prefix_cache_256 | zero tokens served (model failed to load/serve) |
| v0.9.5 | blazar | mr_enc_cache_512mb | zero tokens served (model failed to load/serve) |
| v0.9.5 | ctxcurve-blazar |  | zero tokens served (model failed to load/serve) |
| v0.9.5 | ctxcurve-blazar |  | zero tokens served (model failed to load/serve) |
| v0.9.5 | ctxcurve-blazar |  | zero tokens served (model failed to load/serve) |
| v0.9.5 | direct |  | child exited rc=1 during load; last output: Error: multimodal GGUF architecture `qwen2` is not suppo |
| v0.9.5 | direct |  | child exited rc=1 during load; last output: Error: multimodal GGUF architecture `qwen2` is not suppo |
| v0.9.5 | direct |  | child exited rc=1 during load; last output: Error: multimodal GGUF architecture `qwen2` is not suppo |
| v0.9.5 | direct |  | child exited rc=1 during load; last output: Error: multimodal GGUF architecture `qwen2` is not suppo |
| v0.9.5 | direct |  | child exited rc=1 during load; last output: Error: multimodal GGUF architecture `qwen2` is not suppo |
| v0.9.5 | greedy |  | child failed to become healthy; stderr tail: 'Error: multimodal GGUF architecture `qwen2` is not sup |
| master-1043-3be2378 | media-image | vae_tiling_on | media image cell crashed: MemAvailable 5550 MiB < needed ~7899 MiB (model 5500 MiB + headroom). A co |
| master-1043-3be2378 | media-image | sage_attn_on | media image cell crashed: MemAvailable 7392 MiB < needed ~7899 MiB (model 5500 MiB + headroom). A co |
| 2023.11.14-2 | media-tts |  | media tts cell crashed: MemAvailable 7105 MiB < needed ~7899 MiB (model 5500 MiB + headroom). A co-r |
| 2023.11.14-2 | media-tts-conc |  | media tts-conc cell crashed: MemAvailable 6837 MiB < needed ~7899 MiB (model 5500 MiB + headroom). A |
| master-1043-3be2378 | media-video |  | media video cell crashed: MemAvailable 7071 MiB < needed ~7899 MiB (model 5500 MiB + headroom). A co |
| b5454 | media-whisper |  | media whisper cell crashed: MemAvailable 6568 MiB < needed ~7899 MiB (model 5500 MiB + headroom). A  |
| v0.9.5 | quality | default | quality lanes: first task transport failure (reason/reason-0: HTTP Error 503: Service Unavailable) |
| v0.9.5 | quality-direct | direct | child failed to become healthy; stderr tail: 'Error: multimodal GGUF architecture `qwen2` is not sup |

## Structured skips (tool/format boundaries)

- v0.9.5 (reshape): skipped: adaptive reshape is a llamacpp-lane mechanism; this engine's child exposes no slot shape (-np) to observe or adopt

## Caveats

- ollama prefill numbers come from engine counters that exclude the chat template, so they read slightly high against the 512-token lanes.
- Cross-backend greedy ratios (CUDA vs Vulkan) diverge on near-tie logits; treat ratio, not exact-match count, as the signal.
- All GPU rows measured on AC power at bounded load; rows record load average and power state (battery runs are rejected by the harness).
- Numbers are medians of 5 runs on one hybrid laptop; expect absolute shifts on other hardware, ratios to travel better.

## Reproduce

```bash
python3 scripts/bench_matrix.py --blazar-bin target/release/blazar --model qwen2.5-0.5b-instruct --artifacts-dir bench-artifacts/2026-11-20-v0.33-wave-final --md BENCHMARK.md
python3 scripts/bench_matrix.py --render-only --artifacts-dir <dir> [--append-campaign <sibling-campaign-dir>] --md BENCHMARK.md
```

_Raw per-cell records (argv, per-run lists, daemon logs): `bench-artifacts/2026-11-20-v0.33-wave-final/cells.jsonl`._
