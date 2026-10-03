---
layout: page
title: "Blazar — local AI gateway and runtime orchestrator"
order: 1
---

<img src="assets/BLAZAR_Banner.webp" alt="Blazar banner" width="896" />

## The control plane for local AI inference.

Unified APIs. Intelligent routing. Resource-aware scheduling. Multi-engine serving.

**OpenAI · Ollama · Anthropic**

llama.cpp · mistral.rs · SGLang · MLX-LM · sd.cpp · whisper.cpp · piper

Blazar is a local AI gateway and runtime orchestrator in one Rust binary. Applications talk to one endpoint; Blazar manages the model store, picks or pins the right inference engine, fits workloads to your hardware, and controls concurrency, lifecycle, and diagnostics.

Bring your client. Bring your model. Blazar handles the serving stack.

```text
                           Applications
      ┌───────────────────────┬────────────────────────┐
      │                       │                        │
  OpenAI SDK            Ollama clients         Anthropic SDK
      │                       │                        │
      └───────────────────────┴────────────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │      BLAZAR       │
                    │───────────────────│
                    │ API gateway       │
                    │ Routing           │
                    │ Admission/queue   │
                    │ Model store       │
                    │ VRAM/KV fit       │
                    │ Sessions/cache    │
                    │ Diagnostics       │
                    │ Engine lifecycle  │
                    └─────────┬─────────┘
                              │
      ┌───────────┬───────────┴───────────┬───────────┐
      │           │           │           │           │
        llama.cpp  mistral.rs    SGLang        MLX
      │           │           │           │           │
      └────────────── Text / Embeddings ──────────────┘
                              │
                 ┌────────────┼─────────────┐
                 │            │             │
              sd.cpp       whisper.cpp     piper
               Images       Speech         TTS
               Video        to text
```

## Get started

```sh
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/santanu20/blazar/v0.19.0/scripts/install.sh | sh
```

Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/santanu20/blazar/v0.19.0/scripts/install.ps1 | iex
```

Then:

```sh
blazar search qwen3 instruct     # find models on Hugging Face
blazar pull qwen3-0.6b           # registry shortname
blazar serve                     # gateway on http://127.0.0.1:11435
blazar run qwen3-0.6b            # interactive REPL
```

Works with the OpenAI, Ollama, and Anthropic client SDKs — point them at the local gateway and keep your existing code.

## Documentation

| Guide | What it covers |
|-------|----------------|
| [System overview](1.SYSTEM_OVERVIEW.html) | What Blazar is, the problems it solves, core concepts |
| [Architecture](2.ARCHITECTURE.html) | Crate layout, daemon design, request lifecycle |
| [Data model](3.DATA_MODEL.html) | Stores, rows, and on-disk formats |
| [API specification](4.API_SPEC.html) | Every HTTP surface with request/response examples |
| [Business rules](6.BUSINESS_RULES.html) | Routing decisions, scheduling and admission policy |
| [Setup](7.SETUP.html) | Installation, engines, models, configuration reference |
| [Usage](9.USAGE.html) | Day-2 workflows: sessions, cache, diagnostics, federation |
| [Compatibility contract](8.DO_NOT_BREAK.html) | The behaviors Blazar guarantees not to break |
| [Scientific method](10.SCIENTIFIC.html) | How Blazar measures: benchmarks, statistics, controls |
| [Configuration code paths](config-code-paths.html) | Every config key traced to the code that reads it |

## Research notes

Design investigations that shaped the engine lanes:

- [Inference frontier scan](research/2026-09-26-inference-frontier-scan.html)
- [Competitor pain-points audit](research/2026-10-02-competitor-pain-points-audit.html)
- [MLX direct vs gateway](research/2026-10-02-mlx-direct-vs-gateway.html)
- [Upstream engine radar](research/2026-10-02-upstream-engine-radar.html)
- [Diffusion LLM readiness](research/2026-10-03-diffusion-llm-readiness.html)
- [MLX lane spec](research/mlx-lane-spec.html)

## Reference

- [Benchmark methodology and results](https://github.com/santanu20/blazar/blob/main/BENCHMARK.md) — how Blazar is measured
- [Changelog](https://github.com/santanu20/blazar/blob/main/CHANGELOG.md) — release history
- [GitHub repository](https://github.com/santanu20/blazar) — source, issues, releases
