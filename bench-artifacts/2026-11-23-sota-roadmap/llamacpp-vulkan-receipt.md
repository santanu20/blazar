# llama.cpp Vulkan build: live receipt on an NVIDIA dGPU and an Intel iGPU

Live run of the llama.cpp Vulkan prebuilt through Blazar on one laptop that has an NVIDIA discrete GPU and an Intel integrated GPU. It records three things: how fast the Vulkan build serves on each GPU against the CUDA build and the CPU backend, which build the installer picks on a host that has a GPU but no vendor GPU runtime, and what the ROCm and SYCL prebuilts do on this host.

## Environment

| Item | Value |
|---|---|
| GPU, discrete | NVIDIA GeForce RTX 4070 Laptop GPU, 8188 MiB (PCI 0000:01:00.0) |
| GPU, integrated | Intel RPL-S graphics (PCI 0000:00:02.0), shares system memory |
| CPU and RAM | 16 cores, 13674 MiB |
| NVIDIA driver | 595.99.02 |
| OS | Linux 7.0.0-43-generic, glibc 2.39 |
| llama.cpp | b12686: ubuntu-vulkan-x64 (Vulkan) and ubuntu-cuda-12.8-x64 (CUDA) |
| Blazar | 0.34.3, branch sota-roadmap-20261123, HEAD 517578e plus uncommitted work-tree changes |
| Concurrency | 1 request at a time |
| Sampling | temperature 0, 200-token generations |
| Models | qwen2.5-0.5b-instruct Q4_K_M (469 MiB), qwen3-4b-instruct-2507 (2382 MiB) |
| System service | the system Blazar service ran under the same process before and after every run (not touched) |

## Setup

Every run used a throwaway copy of the engine store (a sandbox data directory); the main store was compared before and after each run and did not change. The serving runs installed the Vulkan build with the `engine_asset` override set to `ubuntu-vulkan-x64`: on a host that already serves with a CUDA engine, `engine update` keeps that engine and installs no second build of the same release. The installer pick below ran without the override.

## Installer pick on a host without a vendor GPU runtime

`blazar engine update b12686` ran with the automatic asset choice in a mount namespace that hides `/proc/driver/nvidia` and on a PATH without `nvidia-smi`, with no ROCm and no oneAPI installed. Exit code 0.

```text
lane llamacpp now holds only b12686
engine b12686 active (build 12686, 2 devices, 341 flags) (newer than stable v1.0.0)
INFO no vendor GPU runtime is installed but Vulkan enumerates a discrete GPU: picking the Vulkan build
INFO engine asset cache hit bytes=37073391 url=https://github.com/ggml-org/llama.cpp/releases/download/b12686/llama-b12686-bin-ubuntu-vulkan-x64.tar.gz
INFO pruned superseded engine b12686-cuda (1151012638 bytes) after update to b12686
```

| Engine row after the update | Asset | Kind | Active |
|---|---|---|---|
| b12686 | ubuntu-vulkan-x64 | llamacpp | 1 |

## Serving, steady state

Two models, four placements each. Per placement and model: the daemon starts the model, answers one cold chat, runs one long generation (the Vulkan driver compiles its pipelines there; it is not measured), then repeats one non-stream and one streamed 200-token generation at temperature 0. The table shows the medians over the repeats; the iGPU placements repeat fewer times because they are slow and hold shared system memory. The CPU arm is the same Vulkan binary with `--gpu-layers 0` appended.

| Model | Placement | `--device` | Repeats | Non-stream tok/s | Stream decode tok/s | Stream TTFT ms | Cold chat s | VRAM used MiB | Child RSS MiB | Memory available, min MiB |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b-instruct | Vulkan build, auto pick | Vulkan1 | 3 | 293.5 | 298.7 | 5.4 | 1.75 | 2194 | 372 | 4125 |
| qwen2.5-0.5b-instruct | CUDA build | none | 3 | 332.2 | 347.6 | 5.3 | 2.10 | 2338 | 624 | 3874 |
| qwen2.5-0.5b-instruct | CPU backend of the Vulkan build | Vulkan1 | 3 | 44.4 | 44.1 | 28.8 | 2.34 | 1543 | 1170 | 3268 |
| qwen2.5-0.5b-instruct | Vulkan build on the Intel iGPU | Vulkan0 | 2 | 10.2 | 13.7 | 163.1 | 5.95 | 1076 | 338 | 3345 |
| qwen3-4b-instruct-2507 | CUDA build | none | 3 | 65.4 | 66.3 | 44.9 | 5.56 | 6931 | 2450 | 2645 |
| qwen3-4b-instruct-2507 | Vulkan build, auto pick | Vulkan1 | 3 | 63.9 | 65.6 | 42.3 | 5.06 | 6911 | 2105 | 2994 |
| qwen3-4b-instruct-2507 | Vulkan build on the Intel iGPU | Vulkan0 | 1 | 9.6 | 9.1 | 199.4 | 18.45 | 1076 | 3436 | 650 |
| qwen3-4b-instruct-2507 | CPU backend of the Vulkan build | Vulkan1 | 3 | 24.2 | 24.7 | 109.3 | 8.05 | 3875 | 5239 | 1339 |

VRAM is the whole-card figure while the model is loaded (idle card before the arms: 1076 MiB). Memory available is the host minimum during the arm (idle before the arms: 4319 MiB).

- qwen2.5-0.5b-instruct: 4 placements, every request HTTP 200; the short chat answer was "Hello! It's a pleasure to meet you!" on every placement; the 200-token counting text was identical on every placement.
- qwen3-4b-instruct-2507: 4 placements, every request HTTP 200; the short chat answer was "Hello! \ud83d\ude0a" on every placement; the 200-token counting text was identical on every placement.

## Serving, single-trial first pass

The first pass measured one trial per placement and took the first long Vulkan generation as its number, so it mixes cold pipeline compilation into the result. It is kept as a raw observation; the steady-state table above is the reference.

| Model | Placement | First 200-token generation s | Stream decode tok/s |
|---|---|---|---|
| qwen2.5-0.5b-instruct | Vulkan build, auto pick | 3.51 | 294.5 |
| qwen2.5-0.5b-instruct | Vulkan build on the Intel iGPU | 28.59 | 11.3 |
| qwen2.5-0.5b-instruct | CUDA build | 0.62 | 332.0 |
| qwen3-4b-instruct-2507 | Vulkan build, auto pick | 4.57 | 52.9 |
| qwen3-4b-instruct-2507 | Vulkan build on the Intel iGPU | 23.31 | 10.0 |
| qwen3-4b-instruct-2507 | CUDA build | 2.73 | 78.1 |

## Engine placement, binary run directly

`llama-server` of the Vulkan build started by hand on the small model with verbosity 4, no gateway. Device list it prints:

```text
Available devices:
  Vulkan0: Intel(R) Graphics (RPL-S) (10256 MiB, 4633 MiB free)
  Vulkan1: NVIDIA GeForce RTX 4070 Laptop GPU (8188 MiB, 6732 MiB free)
```

- `-m <sandbox>/share/blazar/models/__sb__qwen2.5-0.5b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 47551 --ctx-size 2048 --flash-attn auto -lv 4 --device Vulkan1 --gpu-layers 999`
  - `0.00.392.667 I llama_prepare_model_devices: using device Vulkan1 (NVIDIA GeForce RTX 4070 Laptop GPU) (0000:01:00.0) - 6711 MiB free`
  - `0.00.499.426 I load_tensors: offloading output layer to GPU`
  - `0.00.499.428 I load_tensors: offloading 23 repeating layers to GPU`
  - `0.00.499.428 I load_tensors: offloaded 25/25 layers to GPU`
  - `0.00.499.431 I load_tensors:   CPU_Mapped model buffer size =    89.26 MiB`
  - `0.00.499.432 I load_tensors:      Vulkan1 model buffer size =   373.71 MiB`
  - `0.00.589.155 I resolve_fused_ops: Flash Attention enabled`
  - `0.00.590.787 W resolve_fused_ops: layer 0 is assigned to device Vulkan1 but Flash Attention over K/V rows is assigned to device CPU (usually due to missing support)`
  - `0.00.590.789 W resolve_fused_ops: Flash Attention over K/V rows not supported, set to disabled`
  - `0.00.602.433 I sched_reserve:    Vulkan1 compute buffer size =    37.51 MiB`
  - `0.00.602.436 I sched_reserve: Vulkan_Host compute buffer size =     9.03 MiB`
  - `0.00.630.611 I srv  server_on_li: listening on http://127.0.0.1:47551`
- `-m <sandbox>/share/blazar/models/__sb__qwen2.5-0.5b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 42737 --ctx-size 2048 --flash-attn auto -lv 4 --device Vulkan0 --gpu-layers 999`
  - `0.00.266.555 I llama_prepare_model_devices: using device Vulkan0 (Intel(R) Graphics (RPL-S)) (0000:00:02.0) - 4554 MiB free`
  - `0.00.359.566 I load_tensors: offloading output layer to GPU`
  - `0.00.359.566 I load_tensors: offloading 23 repeating layers to GPU`
  - `0.00.359.567 I load_tensors: offloaded 25/25 layers to GPU`
  - `0.00.359.568 I load_tensors:      Vulkan0 model buffer size =   373.71 MiB`
  - `0.00.359.569 I load_tensors:  Vulkan_Host model buffer size =    89.26 MiB`
  - `0.00.828.874 I resolve_fused_ops: Flash Attention enabled`
  - `0.00.829.860 W resolve_fused_ops: layer 0 is assigned to device Vulkan0 but Flash Attention over K/V rows is assigned to device CPU (usually due to missing support)`
  - `0.00.829.860 W resolve_fused_ops: Flash Attention over K/V rows not supported, set to disabled`
  - `0.00.837.406 I sched_reserve:    Vulkan0 compute buffer size =    37.51 MiB`
  - `0.00.837.408 I sched_reserve: Vulkan_Host compute buffer size =     9.03 MiB`
  - `0.01.302.479 I srv  server_on_li: listening on http://127.0.0.1:42737`
- `-m <sandbox>/share/blazar/models/__sb__qwen2.5-0.5b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 42053 --ctx-size 2048 --flash-attn auto -lv 4 --device Vulkan1 --gpu-layers 0`
  - `0.00.383.908 I llama_prepare_model_devices: using device Vulkan1 (NVIDIA GeForce RTX 4070 Laptop GPU) (0000:01:00.0) - 6711 MiB free`
  - `0.00.485.823 I load_tensors: offloading 0 repeating layers to GPU`
  - `0.00.485.825 I load_tensors: offloaded 0/25 layers to GPU`
  - `0.00.485.829 I load_tensors:   CPU_Mapped model buffer size =   462.96 MiB`
  - `0.00.498.875 I resolve_fused_ops: Flash Attention enabled`
  - `0.00.499.144 I resolve_fused_ops: Flash Attention over K/V rows enabled`
  - `0.00.502.237 I sched_reserve:    Vulkan1 compute buffer size =    37.25 MiB`
  - `0.00.502.240 I sched_reserve: Vulkan_Host compute buffer size =     9.06 MiB`
  - `0.00.558.935 I srv  server_on_li: listening on http://127.0.0.1:42053`
- `-m <sandbox>/share/blazar/models/__sb__qwen2.5-0.5b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 45621 --ctx-size 2048 --flash-attn auto -lv 4 --gpu-layers 0`
  - `0.00.362.521 I llama_prepare_model_devices: using device Vulkan1 (NVIDIA GeForce RTX 4070 Laptop GPU) (0000:01:00.0) - 6711 MiB free`
  - `0.00.466.881 I load_tensors: offloading 0 repeating layers to GPU`
  - `0.00.466.882 I load_tensors: offloaded 0/25 layers to GPU`
  - `0.00.466.885 I load_tensors:   CPU_Mapped model buffer size =   462.96 MiB`
  - `0.00.480.003 I resolve_fused_ops: Flash Attention enabled`
  - `0.00.480.292 I resolve_fused_ops: Flash Attention over K/V rows enabled`
  - `0.00.483.734 I sched_reserve:    Vulkan1 compute buffer size =    37.25 MiB`
  - `0.00.483.736 I sched_reserve: Vulkan_Host compute buffer size =     9.06 MiB`
  - `0.00.542.052 I srv  server_on_li: listening on http://127.0.0.1:45621`

## ROCm and SYCL prebuilts on this host

| Asset | Starts here? | Census | What it loads |
|---|---|---|---|
| ubuntu-cuda-12.8-x64 | yes | 1 device (CUDA0) | libcuda.so.1 from the NVIDIA driver; ships cudart and cublas |
| ubuntu-vulkan-x64 | yes | 2 devices (Vulkan0 Intel, Vulkan1 NVIDIA) | the Vulkan loader and the installed GPU drivers |
| ubuntu-rocm-10.0-x64 | yes | 0 devices (no AMD GPU) | ships libamdhip64.so.7, libhipblas.so.3, librocblas.so.5; nothing missing |
| ubuntu-sycl-fp16-x64 | no | none, binary exits 127 at --version | 12 oneAPI libraries it does not ship (libsycl, libsvml, libimf, libirng, libiomp5, libintlc, libmkl_*, libdnnl, libze_loader) |

`blazar engine update` with the SYCL asset exits 1 after the probe: the binary cannot start without the Intel runtime, so nothing is registered. The ROCm asset installs and registers with an empty device list. `ldd` and `readelf -d` on the cached tarballs gave the library lists.

## Reading

- On the NVIDIA card the Vulkan build serves at 86 % of the CUDA build for the 0.5B model (298.7 against 347.6 stream tok/s) and 99 % for the 4B model (65.6 against 66.3).
- The first pass put the 4B Vulkan arm at 68 % of CUDA (52.9 against 78.1 tok/s). The gap between this card's two backends moves from pass to pass by more than the difference between them on the 4B model.
- On the Intel iGPU the Vulkan build is 3.2 times slower than the CPU backend of the same binary at 0.5B (13.7 against 44.1 tok/s) and 2.7 times at 4B (9.1 against 24.7). This is why the installer arm asks for a discrete GPU: an integrated-only host keeps the CPU build.
- The 4B model on the iGPU left the host with 650 MiB of memory available while loaded.
- The first long generation on a Vulkan device is slower than the ones after it (the driver compiles its pipelines). For 200 tokens: 3.51 s in the first pass against 0.72 s for the warm-up generation of the steady-state run (qwen2.5-0.5b-instruct); 4.57 s in the first pass against 3.23 s for the warm-up generation of the steady-state run (qwen3-4b-instruct-2507).
- Time to first token above is measured on a prompt of about 40 tokens. Prefill on a long prompt was not measured, so the install-time warning that Vulkan is slower on the first token is neither confirmed nor refuted by this receipt.
- With `--gpu-layers 0` the Vulkan build still creates the device context: VRAM in use rose by about 467 MiB on the 0.5B CPU arm.
- llama.cpp keeps plain flash attention on Vulkan and disables its newer 'Flash Attention over K/V rows' path on this build (log line in the placement probe).

## Limits

- One host, one driver version, one llama.cpp build.
- iGPU placements repeat 1 to 2 times; their medians are rough.
- Single-request latency only; no concurrency ladder in this receipt.
- Prefill on long prompts and draft-model effects were not measured (the 4B arms ran with a draft model attached).

## Artifacts

The machine-readable receipt is `llamacpp-vulkan-receipt.json` next to this file. It carries the installer pick, every arm with its medians and answers, the placement probe and the vendor runtime captures.

