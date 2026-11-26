# llama.cpp OpenVINO build: live receipt on a host without an NPU or an Intel GPU

Live run of the llama.cpp OpenVINO prebuilt (b12686, asset ubuntu-openvino-2026.4.1-x64) through Blazar on one laptop that has an NVIDIA dGPU and no NPU. It records: the install, what the device listing means on this host, how the CPU device of the OpenVINO build serves against the CPU backend of the CUDA build, which devices refuse to run, the doctor row, and the automatic GPU pick that this build must not receive.

## Environment

| Item | Value |
|---|---|
| Binary | ~/Projects/blazar-sota/target/release/blazar |
| Engine tag | b12686 |
| Asset | ubuntu-openvino-2026.4.1-x64 |
| Host CPU | Intel(R) Core(TM) i7-14650HX |
| Host RAM MiB | 13674 |
| Kernel | 7.0.0-43-generic |
| Model | __sb__qwen2.5-0.5b-instruct-q4_k_m.gguf |
| Prompt | Count from 1 to 120 in words. |
| Decode tokens per generation | 200 |
| System service pid (stable) | 1509489 |
| Main store engine names unchanged | True |

## Setup

Every run used a throwaway copy of the engine store (a sandbox data directory); the main store was compared before and after and its engine names did not change. The system daemon's pid was recorded before and after every leg and never changed. The OpenVINO build was installed with the `engine_asset` override; on a host that already serves with a CUDA engine the update registers the build and keeps the CUDA engine active. Serving arms alternate: the OpenVINO build with `GGML_OPENVINO_DEVICE=CPU`, and the CPU backend of the installed CUDA build of the same release (`--gpu-layers 0` appended).

## Install

Exit code 0 in 79.8 s.

```text
gate: baseline engine b12129-cuda pruned - skipping the comparison (nothing to gate against)
engine b12686 registered (build 12686, 2 devices, 341 flags); the active CUDA engine was kept - run `blazar engine use b12686` to switch
```

| Engine row after the update | Asset | Kind | Active |
|---|---|---|---|
| fork-AtomicBot-ai_atomic-llama-cpp-turboquant-8d935078-cuda | built-fork | llamacpp | 0 |
| fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda | built-fork | llamacpp | 0 |
| b12686-cuda | ubuntu-cuda-12.8-x64 | llamacpp | 1 |
| b12686 | ubuntu-openvino-2026.4.1-x64 | llamacpp | 0 |

## Device listing

```text
OPENVINO0: GGML_OPENVINO_DEVICE=CPU (selected) - Intel(R) Core(TM) i7-14650HX (13674 MiB, 13674 MiB free)
OPENVINO1: GGML_OPENVINO_DEVICE=GPU - NVIDIA GeForce RTX 4070 Laptop GPU (dGPU) (7807 MiB, 7807 MiB free)
```

The listing reports the CPU with the host RAM as device memory and the NVIDIA card through the OpenVINO GPU plugin; free equals total in a fresh process. Setting the env to a device the build cannot use is a hard failure, not a fallback:

- `GGML_OPENVINO_DEVICE=GPU`: exit code 1 --
  `error while handling argument "--list-devices": Exception from src/inference/src/cpp/core.cpp:312:`
  `Exception from src/inference/src/dev/plugin.cpp:94:`
- `GGML_OPENVINO_DEVICE=NPU`: exit by signal 6 --
  `/home/runner/work/llama.cpp/llama.cpp/ggml/src/ggml-openvino/ggml-openvino-extra.cpp:98: GGML OpenVINO Backend: GGML_OPENVINO_DEVICE=NPU is not available. Set it to one of the available OpenVINO devices:`
  `  CPU: Intel(R) Core(TM) i7-14650HX`
  `  GPU: NVIDIA GeForce RTX 4070 Laptop GPU (dGPU)`

## Serving, steady state

One model, four legs (two per arm, alternating). Per leg: cold chat, one long generation, then one non-stream and one streamed 200-token generation at temperature 0. The OpenVINO leg passes `--gpu-layers 999` and (pre-fix) `--device OPENVINO0`; the CPU-backend leg appends `--gpu-layers 0`.

| Leg | Placement | `--device` | `--gpu-layers` | Cold chat s | First long gen s | Non-stream tok/s | Stream decode tok/s | Stream TTFT ms | Mem avail idle->loaded MiB | VRAM idle->loaded MiB | Child RSS MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| openvino-cpu | OpenVINO build, CPU device | OPENVINO0 | 999 | 8.5 | 4.9 | 41.2 | 40.7 | 22 | 4820->2891 | 1076->1076 | 1845 |
| cpu-backend | CPU backend of the CUDA build | None | 0 | 2.2 | 4.0 | 50.4 | 50.1 | 24 | 4943->3733 | 1076->1665 | 1343 |
| openvino-cpu | OpenVINO build, CPU device | OPENVINO0 | 999 | 8.2 | 4.9 | 40.7 | 49.7 | 25 | 4955->3149 | 1076->1076 | 1846 |
| cpu-backend | CPU backend of the CUDA build | None | 0 | 2.0 | 3.8 | 54.0 | 61.4 | 18 | 4610->3331 | 1076->1665 | 1344 |

| Arm (medians) | Non-stream tok/s | Stream decode tok/s | Stream TTFT ms | Cold chat s |
|---|---|---|---|---|
| OpenVINO build, CPU device | 41.0 | 45.2 | 24 | 8.4 |
| CPU backend of the CUDA build | 52.2 | 55.8 | 21 | 2.1 |

## Quality

Every request returned HTTP 200. At temperature 0 each backend repeats its own answer across its legs and the two backends answer differently (both coherent): Hello! It's nice to meet you! vs Hello! It's a pleasure to meet you!.

## Direct probe (no gateway)

The engine binary was started directly with `GGML_OPENVINO_DEVICE=CPU`: ready, one chat returned HTTP 200 in 0.22 s. Key log lines:

```text
0.00.059.250 I cmn  common_fit_a: fitting params to device memory ...
0.00.188.298 I common_fit_params: successfully fit params to free device memory
0.00.211.387 I llama_prepare_model_devices: using device OPENVINO0 (GGML_OPENVINO_DEVICE=CPU (selected) - Intel(R) Core(TM) i7-14650HX) (unknown id) - 13674 MiB free
0.00.421.798 I load_tensors: offloaded 25/25 layers to GPU
0.00.421.801 I load_tensors:   CPU_Mapped model buffer size =    89.26 MiB
0.00.421.802 I load_tensors:    OPENVINO0 model buffer size =   383.47 MiB
0.03.627.306 I llama_kv_cache:  OPENVINO0 KV buffer size =    12.00 MiB
0.03.630.433 I sched_reserve:  OPENVINO0 compute buffer size =    33.75 MiB
0.03.630.435 I sched_reserve: OPENVINO0_HOST compute buffer size =     4.52 MiB
```

## Doctor row

- **hardware receipts** (warn): experimental, no live receipt on this hardware class: b12686 (openvino, intel), v0.9.5 (cuda, nvidia); receipted: b12686-cuda (cuda, nvidia), master-1043-3be2378 (vulkan, nvidia), mlx-0.32.2 (cuda, nvidia) (docs/engines.md, Hardware receipts)

## The automatic GPU pick

```text
hardware: 16 cores, 13674 MiB RAM, 2 GPU(s), 21481 MiB VRAM
2026-11-25T10:16:00.998478Z  INFO engine routing: spawn uses a routed adapter model="qwen2.5-0.5b-instruct" from="llamacpp" routed_to="b12686"
2026-11-25T10:16:00.998478Z  INFO engine routing: spawn uses a routed adapter model="qwen2.5-0.5b-instruct" from="llamacpp" routed_to="b12686"
2026-11-25T10:16:01.367515Z  INFO auto GPU pick: most free VRAM (manual `devices` unset) model="qwen2.5-0.5b-instruct" device=OPENVINO0 free_mib=13674
2026-11-25T10:16:01.367515Z  INFO auto GPU pick: most free VRAM (manual `devices` unset) model="qwen2.5-0.5b-instruct" device=OPENVINO0 free_mib=13674
```

The daemon picked the OPENVINO0 row (host RAM reported as device memory, 13674 MiB free beats the NVIDIA card's 7807) and passed --device OPENVINO0; the run predates the fix.

crates/blazar-runtime/src/supervisor.rs now skips the automatic GPU pick when every device row an engine reports is an OPENVINO row: that build takes its device from GGML_OPENVINO_DEVICE and aborts when --device names a non-selected row. Pinned by:
- `unit__takes_device_from_env__only_an_all_openvino_census`
- `integration__auto_gpu_pick__openvino_build_takes_its_device_from_the_env`

## Reading

The OpenVINO build serves through its CPU device at 41.0 tok/s median non-stream against 52.2 tok/s on the CPU backend of the CUDA build of the same release: on this host the OpenVINO CPU device is the slower way to run the same model on the same cores. Its cold start is also slower and its memory footprint larger. The GPU and NPU devices never ran, so this is a CPU-device receipt, not an accelerator receipt: the registry keeps the OpenVINO class experimental and unreceipted.

## Limits

- One host, one driver stack, one llama.cpp build (b12686).
- The host has no NPU and no Intel GPU: the GPU and NPU devices never served a request (the GPU env probe dies at --list-devices with an OpenCL runtime mismatch, the NPU env aborts the process).
- Two legs per arm, single request at a time; no concurrency ladder.
- The auto-pick observation was recorded with the pre-fix binary; the fix landed after the run and is covered by its own tests.
- The census still counts the OpenVINO CPU row as a GPU in the daemon banner (2 GPU(s), 21481 MiB VRAM); known distortion, not changed here.

## Artifacts

- Raw run data: `h4_openvino_live.json` (harness `h4_openvino_live.py`)
- This receipt: `bench-artifacts/2026-11-23-sota-roadmap/llamacpp-openvino-receipt.{md,json}`
- Decision: `docs/research/2026-11-25-npu-decision.md`
