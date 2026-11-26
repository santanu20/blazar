# LiteRT-LM lane live receipt: x86 Linux host, CPU runtime

Receipt for roadmap item H4. Hardware class: engine lane `openaihost`
(LiteRT-LM) on the CPU runtime, on a host without an NPU. The GPU and NPU
backends of this runtime were not run: this box has no NPU device and its
GPU is not one the runtime lists for this bundle's use.

## Environment

| Item | Value |
|---|---|
| CPU | Intel Core i7-14650HX (14th generation, 16 cores, no NPU) |
| GPUs | NVIDIA GeForce RTX 4070 Laptop GPU 8188 MiB (not used by this lane), Intel RPL-S iGPU (not used) |
| Memory | 13674 MiB |
| NPU | none (no `/dev/accel`) |
| System | Linux 7.0.0-43-generic, glibc 2.39 |
| Runtime | LiteRT-LM 0.20.0 (PyPI, 2026-11-23) in a per-engine venv |
| Model | litert-community/SmolLM2-360M-Instruct, alias `smollm2-360m-instruct`, 373 MB |
| Blazar | 0.34.3, branch `sota-roadmap-20261123`, HEAD `517578e` plus uncommitted work-tree changes |
| Load | concurrency 1, temperature 0 |

The bundle reports its own capabilities through `litert-lm describe`:

```text

```

## Setup

The daemon ran from the release binary against a sandbox store (a copy of
the main store, active engine `b12686-cuda`). The model reaches the lane by
its format: the daemon logs `engine routing: spawn uses a routed adapter
model="smollm2-360m-instruct" from="llamacpp" routed_to="litert-0.20.0"`.
The serving child is one shared `litert-lm serve --host 127.0.0.1 --port
<port>` process per installed lane. The LiteRT registry lives in the home
directory, outside the sandbox; this run only read it.

## Results through Blazar (run 1)

| Measurement | Value |
|---|---|
| First request (cold: child start, model load, 21 tokens) | 4.44 s, HTTP 200, finish stop |
| Warm chat, 21 tokens, non-stream | 2.654 s |
| Warm chat, 21 tokens, stream | TTFT 114.1 ms, total 2294.3 ms |
| Decode 200 tokens, non-stream, 3 runs | 14.066 s, 10.064 s, 13.055 s = 14.5, 20.3, 15.6 tok/s wall |
| Decode 200 tokens, stream | 206 frames, 14.1 frames/s after the first |
| Serving child | RSS 1262 MiB loaded, 226 MiB after the model was released |
| Memory available | 4122 MiB start, 3588 MiB loaded, 6181 MiB end |
| `/api/ps` | engine `litert-0.20.0`, state `ready`, size 373719040 bytes |
| `/api/fabric` | assignment engine `litert-0.20.0`, model key `_openaihost` |
| Errors | none (every request HTTP 200) |

Run 1's decode numbers were taken while the host was busy (another
session's test run started, and a runaway research process of this session
was still spinning on one core). The comparison below is the reference.

## Direct engine against the same lane through Blazar

Same `litert-lm serve` binary, same model alias, same 200-token request,
same host, arms alternated every round (4 rounds, 2 generations per arm and
round).

| Arm | Samples | Median | Min | Max | Mean |
|---|---|---|---|---|---|
| direct `litert-lm serve` | 8 | 19.4 tok/s | 14.1 | 21.7 | 18.9 |
| through Blazar | 8 | 17.0 tok/s | 12.1 | 21.7 | 17.2 |

| Round | Order | direct mean | through Blazar mean |
|---|---|---|---|
| 1 | direct then blazar | 15.75 | 14.3 |
| 2 | blazar then direct | 21.7 | 21.55 |
| 3 | direct then blazar | 21.55 | 16.55 |
| 4 | blazar then direct | 16.4 | 16.25 |

Streamed decode: direct not reported tok/s
(0 sample), through Blazar
19.8 tok/s (4 samples). TTFT on a
27-token prompt: 136.1 ms through Blazar,
45.8 ms direct.

Reading: the ranges overlap and the per-round gap swings between 0 and
5 tok/s, so this is not proof of zero overhead. The CPU runtime uses every
core, and the gateway's own per-frame work competes with it. On the GPU
lane measured in H2 the same comparison showed no difference (326.0 against
325.0 tok/s).

## The runtime's own benchmark (direct, no gateway)

```text
Backend                    : cpu
Cache                      : no
Prefill speed:        180.33 tokens/s
Decode speed:         24.21 tokens/s
Init time:            0.9393 s
Time to first token:  1.4843 s
```

## Tool call

Request with one `get_weather` tool and `tool_choice: auto`: HTTP 200, no
`tool_calls`, finish `stop`, and a plain-text answer that invents the
weather. The bundle reports `Supports Function Call: NO`, so this is the
model's capability, not a gateway fault. The direct engine answers the same
request with `tool_calls: null`. The answer text contains a mojibake
sequence before `C` (raw bytes 32 32 c3 82 c2 b0 43 20 28 37); the
same bytes come from the direct engine, so the model prints them itself.

## Notes and limits

- The lane rewrites its compiled-model cache next to the model at load
  (`model.litertlm_<timestamp>_<size>.xnnpack_cache`, about 366 MB, same
  size, new mtime). That is the lane's disk cache mode; the model file is
  not modified.
- The shim records the absolute venv path at install time, so a copied
  engines tree still runs the original venv.
- CPU runtime on one host, one small model, single requests. No GPU or NPU
  backend of this runtime was run.
- `blazar doctor` prints no hardware-receipt row for this lane: the registry
  tracks GPU classes and this lane states no GPU backend.

## Artifacts

- `litert-lm-receipt.json` (this receipt, machine readable)
- `h4_litert_live.json`, `h4_litert_ab.json`, `h4_litert_direct_check.json` (raw observations)
