# Engine-utilization wave — live spawn receipts (2026-10-10)

Binary: d2419f5f (release, from this wave). Box: RTX 4070 Laptop 8188 MiB, driver 595.99.02.
Daemon: systemd `blazar` on :11435. Every row = real child spawned through the daemon,
argv read from `/proc/<pid>/cmdline`, completion via `/v1/chat/completions` (temp 0).

## sglang lane (qwen3-1.7b, bf16 safetensors dir)

| config knobs | child argv proof | result |
|---|---|---|
| grammar_backend=xgrammar, retraction_policy=priority, enable_priority_scheduling=true | `--grammar-backend xgrammar --retraction-policy priority --enable-priority-scheduling` | 200, "OK." |
| retraction_policy=priority WITHOUT enable_priority_scheduling (pre-fix binary) | engine died during load | sglang: `ValueError: --retraction-policy priority requires --enable-priority-scheduling` → now caught at config-validate time in blazar |

Finding: daemon reads config at spawn; config-set after daemon start requires a daemon
restart to affect child argv (warm children keep their spawn-time argv).

## mlx lane (qwen2.5-0.5b-instruct-4bit, mlx-community dir, Linux CUDA backend)

| config knobs | child argv proof | result |
|---|---|---|
| mlx.kv_bits=8, mlx.kv_group_size=64 | `--kv-bits 8 --kv-group-size 64` | 200, "OK" |

First-class MLX knobs reach a live mlx_lm.server child on the CUDA backend.

## mistralrs lane (qwen3-1.7b pinned via model_overrides.<m>.engine = mistralrs)

| config knobs | child argv proof | result |
|---|---|---|
| mistralrs.pa_memory_mb=2048 | `--pa-memory-mb 2048` (and NO `--pa-memory-fraction`) | 200, "OK." |
| pa_memory_mb + derived fraction together (pre-fix binary) | engine crashed at argv parse | mistralrs: `--pa-memory-fraction cannot be used with --pa-memory-mb` → both fraction emission sites now yield to the absolute pin |

## Cleanup

All knobs unset after proof; model children unloaded; daemon healthy on d2419f5f.

## Task 7 live proofs (binary c37217ca, 2026-10-10)

| Lane | Knob | Child argv proof | Result |
|---|---|---|---|
| mistralrs v0.9.4 | `mistralrs.isq = "q8_0"` | `--isq q8_0` (+ derived `--pa-memory-fraction 0.79` coexists) | qwen3-1.7b 200, "OK." — ISQ quantizes at load, admission stays checkpoint-based |
| whisper b5454 | `whisper_beam_size = 5` | `--beam-size 5` (daemon.log child argv) | piper-TTS wav → " The quick brown fox jumps over the lazy dog.\n" exact |
| sdcpp | 14 component knobs | unit-level only (`unit__compile_sdcpp__tuning_knobs_emitted_and_gated`, 14 pairs incl. `--backend clip=cpu,vae=cuda0`) | no component assets pulled on this box — live spawn would die on missing weights; argv emission + validation (ip_adapter requires clip_vision) proven at compile layer |
| mlx | kv-bits perf A/B | functional proof in Task 6 rows (`--kv-bits 8 --kv-group-size 64` + answer) | C=1 t/s + VRAM A/B deferred: kv-bits disables batching upstream, single-stream compare is the right metric and needs an idle-GPU window |

All knobs unset after proofs; daemon restarted clean.

## Task 7 pending items — closed (2026-10-10, binary c37217ca)

**mlx kv-bits C=1 A/B** (qwen2.5-0.5b-instruct-4bit, CUDA backend, greedy 128-tok fixed
prompt, warm wall e2e x3, child argv verified): kv_bits 8 = 0.430s mean / 631 MiB;
dense = 0.222s mean / 739 MiB. kv-bits ~1.9x slower, saves 108 MiB → memory lever for
long-context only, stays opt-in. Full table: mlx_kv_bits_ab.md.

**sdcpp component lane END-TO-END** (RealESRGAN_x4plus.pth 63.9MB, GitHub release):
`sdcpp_upscale_model` + `sdcpp_hires_upscalers_dir` knobs → child argv carries BOTH
flags → /v1/images/generations 512x512 200 (66.2s) → POST /v1/images/upscale?model=...
with the generated png + scale 4 → 200 in 14.1s, output 2048x2048 (6.05MB png).
Asset-never-auto-pulled contract held: unset knobs + child stopped after proof.
