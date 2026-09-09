# Pallama vs vLLM / SGLang — honest frontier comparison (2026-09)

Question asked: *"is pallama at equal frontier level with vLLM and SGLang in
all aspects?"*

**Straight answer: no — and nobody should claim it is.** Pallama is not a
kernel-level serving engine; it is an orchestrator over llama.cpp's
`llama-server`. At **concurrency 1 it is at parity (or ahead)**; at
**datacenter concurrency it is behind**, because llama.cpp's slot-based
batching is not PagedAttention. Where pallama wins outright is the dimension
vLLM/SGLang deliberately ignore: heterogeneous single-box UX — any GPU vendor,
any quant, CPU fallback, zero Python, zero config, plain files.

Everything below is sourced or code-verified. No parity is asserted that
wasn't measured.

## Capability matrix

Sources: SGLang server-args docs (docs.sglang.io, read 2026-09-05), vLLM
feature set as documented in 2026 comparisons + project docs, llama.cpp
upstream b10816 (vendored reference in this repo), pallama code + step-H
measurements.

| Dimension | vLLM | SGLang | pallama (llama.cpp backend) |
|---|---|---|---|
| Continuous batching | PagedAttention, token-level | same class + overhead-free scheduler | slot-based continuous batching (`-np` slots); real but smaller ceiling |
| Throughput @ concurrency 1 | ~58 tok/s (7B, 4090, sourced bench) | similar | **parity** — 40-42 t/s measured on 9B Q4 (step H head-to-head vs ollama); ollama-vs-vLLM sourced bench shows engine parity at c=1 (62 vs 58) |
| Throughput @ concurrency 16-64 | **950 tok/s @ c=64 (23× ollama, sourced)** | same frontier class | behind by design; slots share ctx, no paged KV; slots=1 default (full single-client speed) |
| KV memory mgmt | PagedAttention (paged, fragmentation-free) | RadixAttention (prefix-tree reuse) | per-slot contiguous + `--cache-reuse 256` chunk reuse + KV q8_0 auto-quant; no paging |
| Tensor/pipeline parallelism | TP + PP, multi-node | TP + DP + multi-node (`--nnodes`) + DCP for MLA | **NO** — llama.cpp has no TP; `rpc_servers` splits *layers* across boxes (`--rpc`), a different (weaker) mechanism |
| Weight quantization | FP8, AWQ, GPTQ, INT8, Marlin kernels | FP8 (`--quantization`), FP8 KV, bitsandbytes, GGUF load-format | **any GGUF quant** (Q2-Q8, IQ, K-quants) — the broadest quant library; no FP8 weights |
| KV quant | FP8 | `--kv-cache-dtype fp8_e4m3/e5m2` | q8_0/q4 auto under VRAM pressure (rule 6) |
| Speculative decoding | yes | yes | yes — 7 upstream modes (EAGLE3, MTP, n-gram, DFlash/Spark), `spec = "auto"` adopts when draft pair pulled |
| Prefix caching | automatic (paged) | Radix cache (best-in-class) | `--cache-reuse` chunk cache (weaker than radix) |
| Structured output | guided decoding | xgrammar | verified live (step H: strict json_schema round-trip) |
| Multi-LoRA hot paths | batched LoRA | multi-LoRA | `lora add/rm/list` → `--lora`/`--lora-scaled` (no batched mixing) |
| Hardware breadth | CUDA, ROCm, CPU, XPU, HPU, TPU backends | CUDA/ROCm-centric, CPU | **any llama.cpp backend: CUDA, Vulkan, Metal, SYCL, OpenCL, CPU** — wins on odd/dual-vendor boxes (verified: this Intel iGPU + NVIDIA box) |
| Quantized-model variety | curated checkpoint formats | same | every HF GGUF repo, day one (zero fork: upstream runs new archs immediately) |
| Deployment | Python venv/Docker, GPU host assumed | same | one static binary, any OS incl. Windows/macOS, no Python |
| APIs | OpenAI (+extras) | OpenAI + gRPC mode | OpenAI + **ollama drop-in** (port 11434) + Anthropic `/v1/messages` |
| Ops surface | server, metrics, health | server, router for DP, metrics, health | daemon, pidfile, priority queue, eviction ladder, trace ids, merged metrics |
| Observability | mature (OTel tracing) | mature | request log + trace id + merged `/metrics`; no distributed tracing |
| Chunked prefill | yes | yes (`--chunked-prefill-size`) | prompt processing splits into `--ubatch-size` micro-batches, but no prefill/decode interleaving (verified: no chunked-prefill flag in upstream b10816 source) |

## Where each wins

**vLLM/SGLang win:** multi-user throughput (10-25× at high concurrency,
sourced), TP/PP multi-GPU sharding, FP8/AWQ kernels, radix/paged KV,
disaggregated serving, gRPC, multi-node fleets. If you run a 24/7 API for
dozens of concurrent users on A100/H100 class hardware, pallama is the wrong
tool and this project will say so.

**pallama wins:** single-user/small-team local inference on whatever hardware
you own — dual-vendor GPUs (Vulkan+CUDA in one box, verified), old GPUs, CPU
boxes, Macs, Windows; any GGUF ever published (zero engine fork = new
architectures day one); plain-file store both APIs on the ollama port with
zero reconfiguration for existing clients; 30%-of-RAM cache clamp so a 13 GiB
laptop doesn't swap itself to death (measured this session). No Python, no
Docker, no YAML, no cluster.

**Ollama head-to-head, same box (2026-09-08, RTX 4070 laptop, Qwen3.5-9B-class, 128 tok, temp 0):**

- Cold load+gen: pallama 8.8s vs ollama 13.1s (pallama -33%).
- Warm single-stream: parity, ~40 t/s both (3.4s vs 3.3s).
- 4-parallel aggregate (2026-09-08 re-run, exclusive GPU per side, identical prompt/temp/128 tok):
  pallama `slots = 4` overlay = **59.5 tok/s** vs ollama default = **28.8 tok/s** — pallama
  **2.07×** (batched decode across 4 slots; ollama's runner loads `-np 1` on this 8 GiB card
  and serializes). At pallama slots=1 the same bench gives ~36 tok/s — the parallel gap
  versus ollama inverts into a 2× win with one overlay knob.
- Bench discipline note: head-to-heads MUST run exclusively (one daemon's model at a time).
  Concurrent daemons race for VRAM — the loser silently CPU-falls-back (observed: ollama at
  4.2 t/s while pallama held the 4070).
- Backend note: ollama runs CUDA on this box; pallama Linux runs Vulkan because upstream
  llama.cpp publishes **no Linux CUDA release assets** (verified across 12 consecutive
  releases: only vulkan/rocm/sycl/openvino/cpu for ubuntu; CUDA is Windows-only). The
  measured Vulkan-vs-CUDA delta on this card is ~4%. Asset preference in `engine/gh.rs`
  already picks the best available per vendor (rocm for AMD, sycl-fp16 for Intel,
  versioned win-cuda for Windows NVIDIA) — the residual Linux NVIDIA gap is
  upstream-asset-bound, not a pallama pick bug.

**Sourced benchmark anchors**

- Ollama 62 t/s vs vLLM 58 t/s at c=1; 41 vs 950 t/s at c=64 (Qwen2.5-7B,
  RTX 4090) — talkingtech.io, 2026-06-04. Engine-class parity at c=1, vLLM
  23× at c=64. llama.cpp-class engines (pallama's backend) sit in the ollama
  row of that table.
- pallama 40.4-41.9 t/s vs ollama 40.8 t/s (qwen3.5:9b Q4_K_M, RTX 4070
  CUDA) with lower TTFT variance — this repo, step H, 2026-09-05.
- SGLang capabilities (TP/DP/DCP, FP8 weights+KV, chunked prefill, radix
  cache, deterministic mode, gRPC) — docs.sglang.io server_arguments, read
  2026-09-05.

## Roadmap items that close real gaps (ranked by ROI)

1. **Slots > 1 auto-profile** — `tune --search` already benches; extend to
   pick `-np` by measured concurrency workload (cheap, real batching gains).
2. **Prefill/decode interleaving** — upstream lacks it entirely (verified b10816); would need upstream work, not a pallama knob — listed for honesty, not actionable in pallama today
3. **`pallama upgrade`** — DONE 2026-09-05 (self-update, digest-verified, atomic; see `pallama upgrade --help`).
4. Windows service registration (install.ps1 + `--with-service`).
5. gRPC gateway — only if a datacenter use case actually appears; otherwise
   it is scope creep.
