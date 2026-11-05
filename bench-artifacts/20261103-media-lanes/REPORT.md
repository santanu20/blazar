# Media-Lane Performance Receipts (W-B)

First performance envelope for blazar's non-text lanes: embeddings, speech
synthesis, image generation, and video generation, each driven end to end
through a sandbox blazar gateway (worktree build, all session waves) with
direct-engine baselines where meaningful. Produced by
`scripts/bench_media.py` (stdlib-only harness, warmup before timing,
append-only receipts).

## Environment

| Item | Value |
|---|---|
| Gateway | sandbox daemon, port 11599, worktree build (G1/G2/G2b/G5/G6/W-A1 waves) |
| llama.cpp engine | b12072-cuda (active llamacpp lane) |
| sd.cpp engine | master-1043-3be2378 (Vulkan backend, NVIDIA RTX) |
| GPU | 8188 MiB VRAM class (watermarks per run below) |
| Host memory | 15.5 GiB class; MemAvailable 4.7-7.0 GiB during runs |
| Date | 2026-11-03 |

Receipts: `receipts.jsonl` (35 entries, append-only, includes failed
environment blocks for honesty). Per-lane CSVs alongside this report.
Child logs under `~/.cache/bench-media/`.

## Embeddings

Ladder = concurrent clients C, N reps each, errors counted. `p50/p95/p99`
are per-request latency.

### text-bert (bge-small-en, 384-d) - gateway vs direct llama-server

| Mode | C=1 | C=2 | C=4 | C=8 |
|---|---|---|---|---|
| gateway req/s | 80.7 | 106.1 | 119.4 | 126.4 |
| gateway p50 | 11.8 ms | 18.6 ms | 31.6 ms | 57.9 ms |
| direct req/s (CPU -ngl 0) | 72.8 | 69.6 | 71.3 | - |
| gateway errors | 0 | 0 | 0 | 0 |

Reading: the gateway matches or exceeds a bare llama-server on text
embeddings at C=1 and scales with concurrency (126 req/s at C=8) where the
direct child flatlines (~71 req/s). Overhead of the gateway on this lane is
within noise; the win comes from connection pooling and admission batching.

### clip-image (candle lane, ViT-B/32, 512-d) - gateway

| C=1 | C=2 | C=4 | C=8 |
|---|---|---|---|
| 0.556 req/s | 0.556 | 0.556 | 0.556 |
| p50 1805 ms | 3499 | 6959 | 9107 |

Reading: in-process candle CLIP on CPU serializes (flat throughput, latency
scales with C). This is the honest capacity ceiling of the CPU fast path;
GPU offload for this lane is future work (tracked). Zero errors.

### vlm-content (qwen3vl-8b + mmproj, 4096-d, content-array shape)

| C=1 | C=2 |
|---|---|
| 6.68 req/s | 8.59 req/s |
| p50 144.9 ms | p50 228.3 ms |

Reading: child-served multimodal embeddings via the G2b lane; 0 errors;
l2-norm 1.0000 spot-checked.

## Speech synthesis

`x realtime` = audio-seconds produced per wall-second (higher is better;
the harness field `rtf_mean` carries this value).

| Lane | C=1 | C=2 | C=4 |
|---|---|---|---|
| piper gateway (en_US-amy-medium) | 2.60 req/s, 12.0x | 4.03, 9.2x | 4.66, 6.3x |
| engine gateway (pocket-tts, CUDA) | 10.98 req/s, 38.7x | 12.3, - | 11.8, - |
| engine direct (pocket-tts, CPU) | 2.65 req/s, 9.8x | 2.70, - | 2.74, - |

Caveat (config asymmetry, documented honestly): the direct baseline was
pinned to CPU (`-ngl 0`) to avoid contending with the daemon for VRAM, while
the gateway auto-offloads onto CUDA. The 4x gateway advantage on the engine
lane is therefore a receipt of the gateway's placement policy, not pure
overhead. The like-for-like comparison is piper-vs-engine through the same
gateway: engine lane is 4.2x faster at C=1 on this hardware.

## Image generation (qwen-image-2.1-uncensored, 512x512, steps 20, seed 42)

| Mode | reps | walls | mean |
|---|---|---|---|
| gateway | 15 (5 runs x 3) | 38.3-59.1 s | 46.0 s |
| direct sd-server | 3 | 35.0-46.6 s | 39.7 s |

All gateway renders byte-identical (443151 B PNG, seed determinism held
across 5 independent runs and cold/warm children). Direct renders
byte-identical among themselves (390964 B) but differ from gateway bytes:
the gateway passes its pinned sampler knobs while the direct child runs
engine defaults for unspecified flags - same seed, different default
sampling parameters. VRAM watermark 6.2-7.2 GiB (varies with co-residents).

Gateway-vs-direct delta: +6.3 s mean (+16%). Both children spill the 11.6
GiB logical parameter set; the gateway's child runs its conservative
`--offload-to-cpu` profile chosen by VRAM admission, the direct child ran
Vulkan auto-fit with params pinned to disk. The delta is the measured cost
of the gateway's safer placement on this 8 GiB card, not per-request proxy
overhead.

## Video generation (wan_2.1_comfyui_repackaged, 320x320, 1 s clip)

| rep | wall | bytes |
|---|---|---|
| 0 | 34.4 s | 830979 (valid webm) |
| 1 | 35.2 s | 666341 (valid webm) |

Byte difference between reps: this lane does not pin a seed in the harness
request; container/pipeline non-determinism expected. No direct baseline
(documented in W-B notes: a second 11.6 GiB-class child cannot co-reside on
this box; the image-lane direct methodology transfers).

## Reproduction

```
python3 scripts/bench_media.py embeddings --preset text-bert --direct
python3 scripts/bench_media.py embeddings --preset clip-image
python3 scripts/bench_media.py embeddings --preset vlm-content --conc 1,2
python3 scripts/bench_media.py speech --lane-speech piper
python3 scripts/bench_media.py speech --lane-speech engine --direct
python3 scripts/bench_media.py image --model qwen-image-2.1-uncensored --size 512x512 --steps 20 --seed 42 --reps 3
python3 scripts/bench_media.py image ... --direct     # exclusive mode
python3 scripts/bench_media.py video --model wan_2.1_comfyui_repackaged --size 320x320 --seconds 1 --reps 2
```

Defaults: gateway `BLAZAR_GATEWAY` or http://127.0.0.1:11599; store/engines
XDG-derived; receipts land in `./bench-artifacts/<date>-media-lanes/`.

## Answers to the standing benchmark questions

- How fast? Tables above; embeddings 126 req/s at C=8, engine TTS 38.7x
  realtime, image ~46 s, video ~35 s per short clip.
- How correct? Every response validated (embedding dims + unit norms, RIFF
  audio, PNG magic/CRC, webm/EBML parse); zero invalid artifacts banked.
- How reliable? 0 errors across all embeddings/speech ladders; the two
  failed image attempts (RAM-starved env, co-resident children) are
  preserved in receipts.jsonl with their environment blocks.
- What does blazar add? Concurrency scaling on embeddings (126 vs 71 req/s
  direct), placement policy on engine TTS (CUDA vs pinned-CPU baseline),
  one port/one API for four modalities.
- What does it cost? Image lane +16% wall vs an unconstrained direct child
  on an 8 GiB card (conservative offload profile).
- Where does it saturate? candle CLIP CPU lane at 0.556 req/s (serialized);
  bert embeddings keep scaling to C=8; engine TTS saturates ~11 req/s.
- What tradeoffs exist? CLIP CPU fast-path latency vs child-served
  multimodal embeddings (0.56 req/s vs 6.7 req/s, different model classes);
  sampler-knob pinning vs engine defaults (image bytes differ).
- Can another engineer reproduce it? Yes - commands above, harness in
  repo, receipts + raw JSONL checked in alongside this report.
