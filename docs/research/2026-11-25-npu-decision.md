# ADR: NPU runtimes and Blazar lanes (H4 decision)

**Date:** 2026-11-25 - **Status:** ACCEPTED - **Decision session:** dp-20261125-blazar-h4-npu-decision-litert-receipt (branch B, score 7.50)

## Context

Roadmap item H4 asked for three things: a live receipt for the LiteRT-LM lane, research on
OpenVINO and ONNX Runtime GenAI, and a written decision on NPUs. Two defects were found on
the way and are closed in the same change:

- The hardware receipt registry (`crates/blazar-core/src/hardware_receipts.rs`) knew only
  CUDA, ROCm, SYCL, Vulkan and Metal builds. An installed llama.cpp OpenVINO build read as
  "no backend", so `blazar doctor` printed no row for the most experimental accelerator
  build Blazar can install. The registry now tracks the class `llamacpp openvino intel` as
  experimental, with no live receipt.
- Blazar's automatic GPU pick passed `--device OPENVINO0` to that build, because the build
  lists the CPU with the host RAM as its memory and that row always wins a most-free pick.
  The build aborts a spawn whose `--device` names a row other than the one its environment
  variable selected. The pick is now skipped for that build (D3).

## Landscape

Checked 2026-11-25 against primary sources (S1-S13). Nothing below rests on a search
summary.

| Runtime | Reaches an NPU with | Model format | Serving shape | State today | Src |
|---|---|---|---|---|---|
| llama.cpp OpenVINO backend | Intel CPU, Intel GPU, Intel NPU | GGUF | `llama-server` binary, prebuilt for Ubuntu and Windows x64, plus Snapdragon Linux/Android arm64 | Work in progress upstream; a missing device is a hard error | S1, S2 |
| OpenVINO GenAI | Intel CPU, GPU, NPU | OpenVINO IR (export step) | Python and C++ API, no server binary | Stable release train | S3-S5 |
| LiteRT-LM | CPU, GPU and NPU backends | `.litertlm` bundle | `litert-lm serve`, OpenAI endpoints | `run` takes `--backend`; `serve` has no backend switch | S6, S7 |
| ONNX Runtime GenAI | Execution providers: Intel OpenVINO, Qualcomm QNN | ONNX | Library API | generate() is a preview API | S8-S10 |
| Lemonade (AMD-backed) | AMD XDNA2 NPU, Radeon | GGUF, FLM, ONNX | Server speaking the OpenAI, Anthropic and Ollama APIs | Community project | S11 |
| ollama | none | n/a | n/a | Intel NPU #3004, AMD NPU #5186 and Snapdragon NPU #5360 requested; all open | S13 |

Not verified on 2026-11-25: which NPUs and operating systems the LiteRT-LM NPU backend
supports; the Qualcomm side of Lemonade; any AMD or Snapdragon NPU run; the ONNX Runtime
GenAI source (its repositories answered HTTP 404 from the research machine while other
repositories resolved); Microsoft Foundry Local hardware support; any published Intel
NPU-versus-GPU figure.

## Evidence from one host

One machine, so these are receipts and not generalizations: Intel Core i7-14650HX
(16 cores, no NPU), NVIDIA GeForce RTX 4070 Laptop GPU 8188 MiB, 13674 MiB RAM, Linux
7.0.0, glibc 2.39. Only the NVIDIA OpenCL vendor driver is installed, so the OpenVINO GPU
device cannot run here.

### LiteRT-LM lane (litert-lm 0.20.0, `smollm2-360m-instruct`)

The bundle's own report: text only, default backend CPU, max context 8192, "Supports
Function Call: NO".

| Measurement | Result |
|---|---|
| Runtime's own benchmark (CPU backend, cache off) | prefill 180.33 tok/s, decode 24.21 tok/s, init 0.94 s, first token 1.48 s |
| First request through Blazar (cold) | 4.44 s for 21 tokens |
| Warm chat, then a warm stream | 2.65 s; 114 ms to first token, 14.1 frames/s |
| A/B, 4 rounds alternated, 8 samples per arm, 200-token generations | direct 19.5 tok/s median (14.1 to 21.7); through Blazar 17.0 tok/s median (12.1 to 21.7) |
| Request carrying a weather tool | HTTP 200, no tool call; the direct server answers the same prompt with `tool_calls: null` |
| Serving child | `litert-lm serve --host 127.0.0.1 --port <port>`; 1262 MiB RSS loaded, 226 MiB after release |

The A/B ranges overlap and the gap per round is 0 to 5 tok/s. That is not proof of zero
overhead: the CPU backend uses every core, and the gateway's per-frame work competes with
it. The lane also rewrote its compiled-model cache (about 366 MB) next to the model at
load; the model file itself was untouched.

### llama.cpp OpenVINO prebuilt b12686 (Ubuntu x64)

Installed through `blazar engine update` with the asset override: 79.8 s, 2 devices and
341 flags registered, the installed CUDA engine stayed active.

| Measurement | Result |
|---|---|
| `GGML_OPENVINO_DEVICE=NPU` | abort (signal 6): the device "is not available" |
| `GGML_OPENVINO_DEVICE=GPU` | exit code 1 with the usage text (cause not determined) |
| OpenVINO build on its CPU device (`qwen2.5-0.5b-instruct` Q4_K_M, 200-token generations) | 41.0 tok/s median, 45.2 tok/s streaming; first token 23.5 ms; first request 8.5 s |
| CPU backend of the CUDA build, same model | 52.2 tok/s median, 55.8 tok/s streaming; first token 20.6 ms; first request 2.1 s |
| Memory available before and after load | 4820 to 2891 MiB (OpenVINO), 4943 to 3733 MiB (CPU backend); NVIDIA VRAM unchanged |

Answers differ between the two backends at temperature 0 (both coherent). On this host the
OpenVINO build is slower than the plain CPU backend and holds about 0.6 GB more host
memory. That agrees with the independent Core Ultra measurement (S12): an NPU is not a
speed feature while the CPU is idle.

### What the device listing really is

`--list-devices` printed two rows: `OPENVINO0` for the CPU, with the host RAM as its
memory (13674 MiB), and `OPENVINO1` for the NVIDIA card (7807 MiB), although OpenVINO
supports Intel GPUs only. The build takes its device from `GGML_OPENVINO_DEVICE` alone and
aborts a spawn whose `--device` names another row. With that build active the daemon banner
shows `2 GPU(s), 21481 MiB VRAM`: the RAM row plus the NVIDIA card.

## Decision

- **D1 - no new engine kind for NPUs now.** Every runtime that can serve a model on an NPU
  is a vendor server, a preview API, or work in progress upstream. None of them justifies a
  supervised lane, an installer path and a lifecycle test suite.
- **D2 - Intel NPUs, and Intel GPUs without a Vulkan driver, are reached through the
  prebuilt llama.cpp OpenVINO build.** One build covers CPU, GPU and NPU from GGUF, so no
  second model format enters the store. It is opt-in: the user sets the asset override and
  `GGML_OPENVINO_DEVICE`. The installer default does not change. The receipt registry
  flags the class `llamacpp openvino intel` as experimental until a machine with an Intel
  NPU or an Intel GPU leaves a receipt.
- **D3 - Blazar never passes `--device` for that build.** Its listing rows are the menu
  for `GGML_OPENVINO_DEVICE`, not `--device` targets. Pinned by
  `integration__auto_gpu_pick__openvino_build_takes_its_device_from_the_env` and
  `unit__takes_device_from_env__only_an_all_openvino_census` in `crates/blazar-runtime`.
- **D4 - no OpenVINO GenAI lane.** It would add OpenVINO IR as a second model format for
  devices the llama.cpp build already serves, and it ships no server binary to supervise.
- **D5 - AMD XDNA2 and Snapdragon NPU runtimes are vendor servers.** They are reached
  through `[[remotes]]` like any OpenAI-compatible endpoint, the same lane as the openai
  hosts (`2026-11-07-openai-host-lane-phase-a.md`), not as supervised lanes: there is no
  install path and no lifecycle to own.
- **D6 - ONNX Runtime GenAI is not a candidate now.** Its generate() API is a preview, its
  page makes no NPU statement of its own, and its source could not be read.
- **D7 - the LiteRT-LM lane stays the lane for `.litertlm` bundles**, receipted on the CPU
  runtime only. Its `serve` command has no backend switch, so Blazar cannot select the NPU
  backend, and its NPU support is not verified.
- **Installer defaults are unchanged.** The OpenVINO build is never picked automatically:
  it is work in progress upstream, slower than the plain CPU backend on the measured host,
  and a missing device is a hard error rather than a fallback.

Rejected: an engine kind for OpenVINO GenAI (D4); reading the accelerator rows of that
listing into the fit ladder as VRAM (the CPU row reports host RAM, and the right
representation cannot be designed without the hardware, see below).

## Using the OpenVINO build today

```toml
# Top level of blazar.toml: the llama.cpp asset that `blazar engine update` installs.
engine_asset = "ubuntu-openvino-2026.4.1-x64"

# Extra environment for engine children and probes.
[engine_env]
# CPU, GPU or NPU. A device this machine lacks is a hard error, not a fallback.
GGML_OPENVINO_DEVICE = "NPU"

[model_overrides."my-model"]
engine = "b12686" # the tag `blazar engine list` prints for the OpenVINO build
slots = 1         # the NPU backend serves one slot
ctx = 4096        # bounded on purpose: the default is the model's training context
```

1. Install the Intel NPU driver (OpenVINO does not ship it). The GPU device needs the
   Intel graphics driver.
2. Set `engine_asset` for the llama.cpp release in use and run `blazar engine update
   <tag>`. The asset name changes with every llama.cpp release.
3. On a machine with an NVIDIA card and an active CUDA engine the new build is registered
   but the CUDA engine stays active; `blazar engine use <tag>` switches. Pin models with
   `engine` to run both builds side by side.
4. `blazar doctor` lists the class as `experimental` until a receipt exists for it.
5. Do not set `devices` on a model served by that build: the engine aborts unless the
   list names the environment-selected device. The automatic pick already leaves it alone.

No NPU number exists in this record. The first run on a Core Ultra machine writes it.

## What reopens the decision

- A machine with an Intel NPU or an Intel GPU is available: leave receipts for the NPU and
  GPU devices, then revisit how the device census is represented.
- Upstream llama.cpp leaves the OpenVINO backend's work-in-progress state.
- LiteRT-LM `serve` gains a backend switch: then receipt its NPU backend.
- An AMD or Snapdragon machine, or a scriptable vendor server, becomes available.
- ONNX Runtime GenAI leaves preview.
- OpenVINO GenAI ships an OpenAI-compatible server binary.

## Consequences

- Intel NPU and Intel GPU users have a working path today with one setting and one
  variable, and `blazar doctor` says plainly that the path is experimental.
- The store gains no new model format, and the installer never surprises a host without
  the accelerator.
- The LiteRT-LM lane is receipted on speed, streaming and tool-call behaviour for the CPU
  runtime; its GPU and NPU backends stay unreceipted.
- Known limit: with the OpenVINO build active the hardware banner counts the RAM row as a
  GPU (`2 GPU(s), 21481 MiB VRAM` on the measured host). Placement estimates that read
  that banner are optimistic on such a host until the census is revisited.

## Sources

- S1 https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENVINO.md
- S2 https://github.com/ggml-org/llama.cpp/releases/tag/b12686
- S3 https://github.com/openvinotoolkit/openvino.genai
- S4 https://pypi.org/project/openvino-genai/
- S5 https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/system-requirements.html
- S6 https://github.com/google-ai-edge/LiteRT-LM
- S7 https://pypi.org/project/litert-lm/
- S8 https://onnxruntime.ai/docs/genai/
- S9 https://pypi.org/project/onnxruntime-genai/
- S10 https://onnxruntime.ai/docs/execution-providers/
- S11 https://github.com/lemonade-sdk/lemonade
- S12 https://github.com/TonyA2024/NPU-vs-CPU-LLM-Inference-Benchmark
- S13 https://github.com/ollama/ollama/issues/3004 (and issues 5186, 5360, 2169 and 1590
  in the same repository)
