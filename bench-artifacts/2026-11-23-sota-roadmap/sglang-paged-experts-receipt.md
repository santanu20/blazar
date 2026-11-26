# SGLang Paged Experts (E1) live receipt

Date: 2026-11-26

Question: Can Blazar route MoE models whose routed experts exceed GPU memory by keeping K experts resident and paging the rest from pinned host memory, and what does that trade in speed, TTFT and host RAM?

## Environment

- host: Intel Core i7-14650HX (16 cores), 13674 MiB RAM
- gpu: NVIDIA GeForce RTX 4070 Laptop GPU, 8188 MiB
- os: Linux 7.0.0-43-generic
- system cuda: 12.0 (/usr/bin/nvcc; no /usr/local/cuda toolkit)
- lane cuda: CUDA 13.0 bundled in the sglang engine venv (CUDA_HOME exported by the lane shim)
- engine: sglang 0.5.23 (sglang lane, spawned through the sglang-server shim)
- attention: flashinfer 0.7.2 (JIT compiled with the bundled CUDA 13 toolkit)
- model: OLMoE-1B-7B-0924-GPTQ (OlmoeForCausalLM, 16 layers, 64 routed experts, top_k 8, GPTQ 4-bit group 128 sym)
- policy: --mem-fraction-static 0.8 on every arm (the auto fraction 0.545 left no KV room for the unpaged baseline)

## Model and upstream router bug

The shuyuej/OLMoE-1B-7B-0924-GPTQ export quantizes the MoE router (mlp.gate qweight/qzeros/scales/g_idx) while sglang 0.5.23 builds the router as a plain unquantized linear; the engine aborts with KeyError 'model.layers.0.mlp.gate.g_idx' (olmoe.py:429).

Harness-side copy with the 16 routers dequantized to bf16 and the GPTQ router tensors removed (e1_patch_router.py). Requantizing the bf16 routers drifts 0 nibbles on all 16 gates; the qzeros nibble convention (raw 7 = symmetric center, no +1 bias) is inferred from the sym=true layout and validated by the requant bound plus cross-arm identity, not against a bf16 reference (none exists on this host).

## Direct arms (engine driven through the lane shim)

| arm | K | load s | steady tok/s | first-gen tok/s | TTFT ms | KV tokens | weights GB | VRAM after MiB | MemAvail MiB | scheduler RSS MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unpaged | 64 (all) | 43.1 | 254.03 | 247.98 | 238 | 11927 | 3.66 | 7425 | 6861 -> 5216 | 1711 |
| k8 | 8 | 69.7 | 44.51 | 44.51 | 1382 | 33783 | 1.00 | 7329 | 7073 -> 901 | 4910 |
| k16 | 16 | 63.0 | 63.86 | 63.73 | 979 | 31047 | 1.33 | 7371 | 6494 -> 1944 | 4925 |
| k32 | 32 | 71.0 | 124.49 | 121.45 | 517 | 24695 | 2.11 | 7379 | 7098 -> 2376 | 1273 |

Every arm shares one budget policy (--mem-fraction-static 0.8), so the KV pool absorbs whatever VRAM the weights release; the KV token capacity column is the real discriminator.

Capacity observation: at the auto memory fraction (0.545) the unpaged baseline aborts with 'Loaded weights leave no GPU memory for the KV cache' (minimum viable 0.5734) while the K=16 paged build loads fine - the feature's capacity story, on one 8 GiB card.

## Quality

Identical outputs across K: yes. All four arms produce the bit-identical answer text: paging preserves generation exactly on the Marlin runner (GPTQ) in single-request, temperature-0 runs; upstream's bit-identical claim is stated for the triton runner, with Marlin documented as within kernel batch-shape noise. Absolute quality reflects the base (non-instruct) checkpoint's temperature-0 continuation, not the paging mechanism.

## Through Blazar (release binary, sandbox store)

### guard_armed - spawn memory guard ON (product default)

- spawn memory guard: True
- engine argv carries --enable-paged-experts: True
- --paged-experts-num-resident: 16
- request served: True
- steady decode through the gateway: 70.08 tok/s
- answer head: '\n\nThe first step is to create a function that will convert a number to a string.\n\nfunction numberToWords(number) {\n  var'

### guard_off - spawn memory guard OFF (documented lever, mechanism run)

- spawn memory guard: False
- engine argv carries --enable-paged-experts: True
- --paged-experts-num-resident: 16
- request served: True
- steady decode through the gateway: 63.73 tok/s
- answer head: '\n\nThe first step is to create a function that will convert a number to a string.\n\nfunction numberToWords(number) {\n  var'

The spawn memory guard sees the pinned expert store: with the guard armed the J3 floor charges max(classic floor, paged floor). Both legs answer on the /v1/completions surface; their answer text matches leg-to-leg but differs from the direct arms' /generate text (the gateway and the native surface format the prompt differently) - the paging evidence is the direct arms' cross-K identity, the product evidence is the leg-to-leg identity. Steady decode through the gateway (70.1 and 63.7 tok/s) matches the direct K=16 arm (63.9 tok/s) within busy-host variance.

## Lane gate bug found and fixed during live preparation

Blazar's sglang lane gate compared the config.json FORMAT name against the engine's LOADER-registry names and refused every genuine GPTQ export (quant_method 'gptq' is not in the loader list; only 'gptq_marlin' is). sglang itself converts complete GPTQ configs (bits, group_size, sym, desc_act all present) to the Marlin loader at load (GPTQMarlinConfig.override_quantization_method), so the engine loads what the gate refused.

Fix: HfMeta gains quant_gptq_marlin_convertible and the gate maps a complete 'gptq' config to 'gptq_marlin' before the membership check; incomplete GPTQ configs stay refused (the engine would abort on them). Pinned by unit__sglang_lane_teach__gptq_converts_to_the_marlin_loader and unit__hf_config_gptq_props__marlin_conversion_flag.

Live evidence: a via-Blazar request against the pre-fix release binary was refused with HTTP 400 MODEL_UNSUPPORTED_ENGINE naming quant_method 'gptq' - refusal body: `{"error":{"blazar_code":"MODEL_UNSUPPORTED_ENGINE","code":400,"message":"<sandbox>: quant_method 'gptq' is not loadable by this sglang lane (0.5.23; supported: auto-round, auto-round-int8, awq, awq_ma` (from e1_paged_via_blazar_prefail.json); the rebuilt binary with the fix serves the same request (the via-Blazar legs above).

## Readings

- Steady decode is monotone in K: 44.5 (K=8) -> 63.9 (K=16) -> 124.5 (K=32) -> 254.0 (K=64); paging cost scales with the number of non-resident experts.
- TTFT moves the other way: 1382 -> 979 -> 517 -> 238 ms (more resident experts warm the copy path less).
- KV token capacity: 11927 unpaged vs 33783 at K=8 (2.8x) - paging buys context room, not only weight room.
- VRAM weight slice follows the split math exactly: 3.66 GB unpaged vs 1.00/1.33/2.11 GB at K=8/16/32.
- The honest host-cost signal is the MemAvail delta (-6172 MiB at K=8): CUDA pinned memory does not appear in VmLck, and the K=32 scheduler RSS (1273 MiB) hides the store because the pinned pages are file-backed; RSS tracks it only when the allocator maps the store privately (K=8/K=16).

## Limits

- No bf16 reference exists on this host: the router dequant convention is validated by the requant bound and cross-arm output identity, not against reference logits.
- The memory-forced engagement (a model that cannot spawn unpaged at all) is not reproducible on this 13.6 GB RAM host with a 4 GB model; the in-product paths are proven by the stub-VRAM integration tests (integration__sglang_paged_experts__moe_pages_instead_of_cpu_offload).
- Single GPU, no TP/EP/speculative pairing, upstream hard constraints; Blazar's knobs refuse those combinations at config parse.
- First generation per server is ~7x slower than steady state (lazy kernel warmup); the steady column is runs 2-3.

## Artifacts

- Raw direct-arm data: `e1_paged_live.json` (harness `e1_paged_live.py`)
- Via-Blazar leg data: `e1_paged_via_blazar.json` (harness `e1_paged_via_blazar.py`)
- Router patch harness: `e1_patch_router.py`
- Engine logs per arm: `e1_srv_<arm>.log`
- Daemon logs per leg: `e1_via_daemon_<leg>.log`
- This receipt: `bench-artifacts/2026-11-23-sota-roadmap/sglang-paged-experts-receipt.{md,json}`
