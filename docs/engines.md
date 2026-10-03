---
layout: doc
title: Engines
description: The seven serving engines Blazar orchestrates — roles, model formats, and how routing picks between them.
doc_kind: Documentation
---
## One gateway, seven engines

Blazar does not replace inference engines — it orchestrates them. Each lane
is the real upstream binary, managed end-to-end: download or build, pin,
configure, supervise, and route requests to it.

| Engine | Role | Typical workloads |
|---|---|---|
| **llama.cpp** | Mainstream local text serving | GGUF and quantized local models |
| **mistral.rs** | Alternative text runtime | Supported GGUF and safetensors models |
| **SGLang** | Safetensors-oriented serving | AWQ, GPTQ, FP8 safetensors models |
| **MLX** | Apple-Silicon and CUDA-backed serving | MLX quantized model directories (mlx-community) |
| **stable-diffusion.cpp** | Media generation | Diffusion images and video |
| **whisper.cpp** | Speech recognition | Transcription and translation |
| **piper** | Offline speech synthesis | Local TTS voices |

## Format-driven routing

Routing follows model format and engine capability:

```text
GGUF                      → llama.cpp / mistral.rs
Quantized safetensors     → SGLang
Plain safetensors         → SGLang / mistral.rs
MLX quant directories     → MLX
Diffusion component sets  → stable-diffusion.cpp
Audio transcription       → whisper.cpp
Offline TTS               → piper
```

The default preserves single-active-engine behavior; per-model engine pins
give deterministic placement. Curated capability lanes cover GGUF
architectures the mainstream llama.cpp lane does not yet support.

## Engine lifecycle

Every lane supports the same management surface:

```sh
blazar engine update --all        # refresh every lane to its latest
blazar engine install --kind ...  # add a lane by engine kind
blazar engine pin <tag>           # hold a lane at a known version
blazar doctor versions            # lane + binary health overview
```

Ask Blazar why a model landed on a lane:

```sh
blazar explain <model>            # the routing decision, with reasons
```

## Per-engine notes

### llama.cpp

The default text lane for GGUF checkpoints. Slot-parallel serving, KV-cache
quantization, and n-gram speculative decoding. Curated capability lanes
extend coverage to GGUF architectures the mainstream build does not ship.

### mistral.rs

The second text lane: GGUF and supported safetensors models, with its own
serving and speculation paths. A per-model pin moves any supported checkpoint
here deterministically.

### SGLang

The throughput lane for quantized safetensors directories — AWQ, GPTQ, FP8.
Blazar derives its parallel sizes, memory fraction, and KV-cache budget from
your hardware before the server ever starts.

### MLX

Apple-Silicon-native (and CUDA-backed) serving for mlx-community quantized
directories. Installed as a pip environment (`mlx-lm`) and managed like any
other lane.

### stable-diffusion.cpp

Diffusion images (and video component sets) on CPU or GPU, including Vulkan
devices. Served over the same OpenAI-style image API surface.

### whisper.cpp

Speech transcription and translation, with streaming capture and Silero VAD
gating for live audio.

### piper

Offline neural TTS with local voices — no network, no cloud service, one
API surface shared with the rest of the media lanes.
