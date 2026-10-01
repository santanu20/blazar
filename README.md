<div align="center">

# Blazar

**The control plane for local AI inference.**

OpenAI · Ollama · Anthropic compatible &nbsp;·&nbsp; Multi-engine &nbsp;·&nbsp; VRAM-aware &nbsp;·&nbsp; Multimodal

[![CI](https://github.com/santanu20/blazar/actions/workflows/ci.yml/badge.svg)](https://github.com/santanu20/blazar/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/santanu20/blazar)](https://github.com/santanu20/blazar/releases)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)
[![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](#installation)

**Current release: `0.14.0`**

</div>

> **Blazar is a local AI gateway and runtime orchestrator for heterogeneous inference workloads.** It gives applications one stable endpoint while managing models, engines, VRAM/KV fit, concurrency, lifecycle, diagnostics, and multimodal workloads in one Rust binary.

<div align="center">

<img src="docs/assets/blazar-logo.png" width="280" alt="Blazar logo">

**[Get started in 60 seconds](#60-second-quickstart)** &nbsp;·&nbsp; **[See benchmark evidence](#benchmark-snapshot)** &nbsp;·&nbsp; **[Read the API reference](docs/4.API_SPEC.md)**

</div>

---

## Table of contents

- [What is Blazar?](#what-is-blazar)
- [Why use Blazar?](#why-use-blazar)
- [Who should use Blazar?](#who-should-use-blazar)
- [Blazar vs. common approaches](#blazar-vs-common-approaches)
- [Benchmark snapshot](#benchmark-snapshot)
- [Where does it fit?](#where-does-it-fit)
- [How it works](#how-it-works)
- [60-second quickstart](#60-second-quickstart)
- [Use your existing clients](#use-your-existing-clients)
- [Model management](#model-management)
- [Multi-engine serving](#multi-engine-serving)
- [Resource-aware scheduling](#resource-aware-scheduling)
- [Advanced request orchestration](#advanced-request-orchestration)
- [Advanced feature index](#advanced-feature-index)
- [Multimodal workloads](#multimodal-workloads)
- [Sessions, cache, and warm starts](#sessions-cache-and-warm-starts)
- [Diagnostics and observability](#diagnostics-and-observability)
- [Safe engine lifecycle](#safe-engine-lifecycle)
- [Security and network behavior](#security-and-network-behavior)
- [Configuration](#configuration)
- [Common commands](#common-commands)
- [Installation](#installation)
- [Reproducing benchmarks](#reproducing-benchmarks)
- [Architecture and engineering](#architecture-and-engineering)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Credits](#credits)
- [License](#license)

---

## What is Blazar?

Blazar is a **local AI gateway and runtime orchestrator**.

Applications talk to one gateway. Blazar manages the model store, selects or pins an appropriate inference engine, fits workloads to available resources, controls concurrency and lifecycle, and exposes operational diagnostics.

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
        ┌─────────────────────┼─────────────────────────┐
        │                     │                         │
   llama.cpp            mistral.rs                  SGLang
        │                     │                         │
        └────────────── Text / Embeddings ─────────────┘
                              │
                 ┌────────────┼─────────────┐
                 │            │             │
              sd.cpp       whisper.cpp     piper
               Images       Speech         TTS
               Video        to text
```

The underlying engines still perform inference. **Blazar is the control plane around those engines.**

### The core idea

**Bring your client. Bring your model. Blazar handles the serving stack.**

### At a glance

| Capability | Blazar provides |
|---|---|
| **One gateway** | OpenAI-, Ollama-, and Anthropic-compatible APIs |
| **Multiple runtimes** | llama.cpp, mistral.rs, SGLang, stable-diffusion.cpp, whisper.cpp, and piper |
| **Measured auto-tuning** | Speculative decoding, KV-cache quantization, and flash attention selected per hardware by benchmarked rules, not guesswork |
| **Resource control** | VRAM/KV fit, slots, admission, co-residency, and lifecycle management |
| **Model operations** | Pull, import, inspect, pin, tune, snapshot, and restore |
| **Production controls** | API keys, TLS, CORS, audit logging, metrics, traces, and diagnostics |
| **Multimodal** | Text, embeddings, image, video, speech-to-text, and TTS |

---

## Why use Blazar?

Blazar is useful when local inference has outgrown the "run one server for one model" workflow.

| Need | Blazar's approach |
|---|---|
| Multiple applications use different APIs | One gateway supports **OpenAI, Ollama, and Anthropic** compatible surfaces. |
| Different models need different runtimes | Capability-aware lanes across **llama.cpp, mistral.rs, SGLang, stable-diffusion.cpp, whisper.cpp, and piper**. |
| GPU memory and context are difficult to manage | Fit planning, KV-aware sizing, slots, admission control, and co-residency planning. |
| Local serving leaves speed and efficiency behind | Auto-tuned posture — speculative decoding, KV quantization ladder, flash attention — chosen per hardware from measured rules, with built-in tok/s-per-watt benchmarking to prove the result. |
| Restarts and reloads cost minutes | Session checkpoints restore conversational state, and warm-on-pull avoids the first-request cold path. |
| Models and engines become operationally messy | One CLI, model store, engine store, configuration surface, and lifecycle manager. |
| Engine updates can introduce regressions | Verified side-by-side installs, probing, regression gates, explicit activation, and rollback. |
| Failures are hard to diagnose | `doctor`, `why`, `watch`, metrics, trace IDs, and actionable errors. |
| AI CLIs need repeated environment setup | `blazar launch` prepares the local API environment and ensures the daemon is available. |
| Local deployments need controls | API keys, TLS, CORS, audit logging, PII scrubbing, OTLP, and explicit remote routing. |

### What Blazar is not

Blazar does **not** replace the inference engines, train models, or require a hosted service. It gives applications a stable local control surface while the selected runtime performs the actual model execution.

## Who should use Blazar?

Blazar is a strong fit when you:

- run AI models locally on CPU and/or one or more GPUs;
- need OpenAI, Ollama, or Anthropic compatibility without maintaining separate gateways;
- run multiple models, formats, or inference runtimes on the same machine;
- care about VRAM, KV-cache capacity, concurrency, model residency, and cold starts;
- want controlled engine upgrades, verification, rollback, and capability-aware routing;
- are building agents, developer tools, internal AI services, or multimodal pipelines.

Blazar is intentionally **not** necessary for every deployment. If you only need one model, one runtime, and one local process with no orchestration requirements, a direct engine server may be simpler.

## Blazar vs. common approaches

Blazar occupies the **orchestration layer between applications and inference runtimes**. The comparison below is about architectural role, not a universal ranking.

| Approach | Best when | What Blazar adds |
|---|---|---|
| **Direct engine server** | One runtime, one model, minimal control plane | Gateway compatibility, model operations, admission, routing, lifecycle, diagnostics |
| **Ollama** | Simple local model management and Ollama-compatible clients | Multiple engine families, explicit routing, deeper resource planning, lifecycle controls, operational diagnostics |
| **SGLang / vLLM-style runtime** | High-throughput serving centered on one runtime family | A higher-level local control plane across heterogeneous runtimes and workload types |
| **Custom scripts / multiple daemons** | Highly bespoke infrastructure | One API surface, one configuration model, one model/engine store, and one operator workflow |

---

## Benchmark snapshot

Blazar is designed to add a **control and orchestration plane around inference engines without becoming the performance bottleneck** — and the published campaign below measures that claim directly.

The repository includes reproducible benchmark campaigns covering raw engine overhead, concurrency, latency, cache behavior, cold starts, idle wake, tool calls, scheduling, model lifecycle, and media workloads.

### Published benchmark snapshot

The latest committed flagship campaign was recorded on **September 29, 2026**. It used an NVIDIA RTX 4070 Laptop GPU (8 GiB), Intel Core i7-14650HX, 16 GiB RAM, and Linux Mint 22.3. The campaign receipt is published in [`BENCHMARK.md`](BENCHMARK.md) and [`bench-artifacts/`](bench-artifacts/).

> **Important:** these are measured results from a specific model, engine build, hardware configuration, workload, and software version. They are evidence of observed behavior, not universal performance guarantees. This campaign ran against Blazar `0.13.0`; the current release is `0.14.0`.

### Headline evidence

| Proof point | Published result |
|---|---:|
| **Gateway overhead** | **~2%** decode-throughput difference vs the same direct engine configuration |
| **Concurrent throughput** | **104.8 t/s** at C=4 in the published Blazar gateway campaign |
| **Adaptive serving** | **49 → 63 t/s** with **0 failed requests** during sustained-load reshape |

### What the benchmark shows

| Workload | Blazar | Reference | Interpretation |
|---|---:|---:|---|
| Single-stream decode behind gateway | **40.6 t/s** | **41.5 t/s** direct engine | ~**2%** measured gateway overhead in this configuration |
| Concurrent system throughput, C=4 | **104.8 t/s** | **33.6 t/s** Ollama reference lane | **~3.1×** measured system throughput |
| Concurrent TTFT p99, C=4 | **554 ms** | **15.9 s** Ollama reference lane | Lower measured tail latency |
| Tool-call TTFT p50 | **117 ms** | **284 ms** Ollama reference lane | **~2.4×** lower measured TTFT |
| Idle wake | **3.63 s** | **6.21 s** Ollama reference lane | Faster measured wake |
| Adaptive serving under sustained load | **49 → 63 t/s** | — | **~28%** gain; **0 failed requests** |

### Why these numbers matter

The benchmark illustrates two distinct benefits of Blazar.

**Low overhead on the fast path.** A direct llama.cpp run measured 41.5 tokens/s while the same engine behind the Blazar gateway measured 40.6 tokens/s. In that configuration, the gateway overhead was approximately 2% for decode throughput.

**More control as workloads become difficult.** The larger value of an orchestration layer appears under concurrency, model lifecycle changes, and sustained load. Blazar can schedule requests, manage slots, control model residency, adapt capacity, and expose the resulting behavior to the operator instead of leaving every decision to an individual engine process.

### Benchmark philosophy

Blazar benchmarks more than raw tokens/s because a local inference platform is an operational system, not just a decoder. Campaigns measure:

- single-stream decode and prefill;
- concurrency and saturation behavior;
- gateway overhead versus the direct engine;
- TTFT and inter-token latency;
- cold starts and idle wake;
- prompt-cache reuse;
- tool-call and structured-output behavior;
- adaptive scheduling;
- model and engine lifecycle;
- image, video, TTS, and Whisper workloads.

Every campaign records the model, engine build, hardware, configuration, workload, and measurement methodology.

For the complete methodology, full result tables, and raw campaign receipts, see [`BENCHMARK.md`](BENCHMARK.md).

---

## Where does it fit?

- **Local development** — keep application code pointed at one endpoint while models and engine lanes swap underneath.
- **Personal AI workstation** — several models on one machine, with VRAM, slots, residency, and idle eviction under one policy.
- **Agents and CLI tools** — `blazar launch <command>` gives every local tool the same gateway and configuration.
- **Homelab and internal services** — start loopback-only; add authentication, TLS, CORS, metrics, and audit as you grow.
- **Multimodal local AI** — text, embeddings, image, video, transcription, and TTS through the same operational model.
- **Heterogeneous hardware** — capability-aware engine selection, multi-GPU placement, and supported RPC offload.

---

## How it works

A typical request path is:

```text
Client request
    │
    ▼
API dialect / route
    │
    ▼
Model + capability resolution
    │
    ▼
Admission + queueing
    │
    ▼
VRAM / KV / slot fit
    │
    ▼
Engine selection or explicit override
    │
    ▼
Managed engine child
    │
    ▼
Response + telemetry + trace
```

The practical workflow is:

```text
fit → pull/import → serve → run → observe → tune
```

---

## 60-second quickstart

### 1. Install

#### Linux / macOS

```sh
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/santanu20/blazar/v0.14.0/scripts/install.sh \
  | BLAZAR_REPO=santanu20/blazar sh
```

#### Windows PowerShell

```powershell
irm https://raw.githubusercontent.com/santanu20/blazar/v0.14.0/scripts/install.ps1 | iex
```

The Blazar binary and inference engines are separate artifacts. The installer can bootstrap the default engine; you can also manage the engine explicitly:

```sh
blazar engine update
```

To skip engine bootstrap during installation, set `BLAZAR_INSTALL_ENGINE=0` in the installer environment.

### 2. Pull a model

```sh
# Registry shortname
blazar pull qwen3-0.6b

# Hugging Face GGUF
blazar pull ggml-org/Qwen3-8B-GGUF:Q4_K_M
```

Or register an existing local GGUF:

```sh
blazar import /path/to/model.gguf --name mymodel
```

### 3. Start the gateway

```sh
blazar serve
```

Default listener:

```text
http://127.0.0.1:11435
```

### 4. Run a model

```sh
blazar run qwen3-0.6b
```

Or call the OpenAI-compatible API:

```sh
curl http://127.0.0.1:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [{"role": "user", "content": "Explain local inference in one paragraph."}]
  }'
```

---

## Use your existing clients

Blazar is designed to be introduced **without rewriting application code**.

### OpenAI-compatible clients

Base URL:

```text
http://127.0.0.1:11435/v1
```

Common surfaces include:

```text
/v1/chat/completions
/v1/completions
/v1/responses
/v1/embeddings
/v1/rerank
/v1/batches
/v1/files
/v1/audio/*
/v1/images/*
/v1/videos/*
```

### Ollama-compatible clients

```sh
export OLLAMA_HOST=http://127.0.0.1:11435
```

Common compatibility routes include:

```text
/api/chat
/api/generate
/api/tags
/api/ps
/api/pull
/api/embeddings
/api/embed
/api/rerank
```

You can use Blazar alongside an existing Ollama installation, then later configure Blazar for `11434` when you want a port-level replacement.

### Anthropic-compatible clients

Base URL:

```text
http://127.0.0.1:11435
```

Primary route:

```text
/v1/messages
```

### AI CLIs and agents

```sh
blazar launch <command>
```

Blazar prepares the relevant OpenAI / Anthropic / Ollama base-URL environment and executes the command.

---

## Model management

Blazar keeps model assets as ordinary local files instead of requiring an opaque runtime-specific blob store.

Default model area:

```text
~/.local/share/blazar/models/
```

Common workflows:

```sh
blazar pull <target>
blazar import <file> --name <name>
blazar list
blazar show <model>
blazar ps
blazar fit <target>
```

For existing GGUF files, import uses hardlinks by default, avoiding an unnecessary second copy of the model data.

---

## Multi-engine serving

Blazar currently manages these runtime lanes:

| Engine | Role | Typical workloads |
|---|---|---|
| **llama.cpp** | Mainstream local text serving | GGUF and quantized local models |
| **mistral.rs** | Alternative text runtime | Supported GGUF and safetensors models |
| **SGLang** | Safetensors-oriented serving | AWQ, GPTQ, FP8 and supported safetensors models |
| **stable-diffusion.cpp** | Media generation | Diffusion images and video |
| **whisper.cpp** | Speech recognition | Transcription and translation |
| **piper** | Offline speech synthesis | Local TTS voices |

### Capability-aware routing

Blazar can route according to model format and engine capability rather than forcing every model through one runtime.

Typical lanes are:

```text
GGUF                      → llama.cpp / mistral.rs
Quantized safetensors     → SGLang
Plain safetensors         → SGLang / mistral.rs
Diffusion component sets  → stable-diffusion.cpp
Audio transcription       → whisper.cpp
Offline TTS               → piper
```

Automatic routing is available through the routing policy; the default configuration preserves the single-active-engine behavior, while explicit per-model engine pins remain available when you need deterministic placement.

For GGUF architectures that are not yet covered by the mainstream llama.cpp lane, Blazar can manage curated **capability lanes** built from immutable fork commits and route only the affected models there.

Peering multiple Blazar gateways into one routing fabric is covered in [`docs/7.SETUP.md`](docs/7.SETUP.md) under **Federation: peers behind one gateway**.

---

## Resource-aware scheduling

Blazar treats **memory, concurrency, and model residency as scheduling problems** rather than leaving all decisions to the underlying engine.

### Fit before download

```sh
blazar fit <model-or-repo>
```

Preview model fit, context limits, and quantization options before committing to a large download.

### VRAM and KV awareness

The serving planner can account for:

- model weights;
- context length;
- KV-cache posture and quantization;
- slot count and parallelism;
- host-memory spill where supported;
- multi-model co-residency;
- device placement and multi-GPU configuration.

### Co-residency planning

```sh
blazar coreside
```

Use this to reason about which models can remain resident together within the machine's available capacity.

### Admission and queueing

When a model is saturated, Blazar can park requests behind bounded admission rather than allowing uncontrolled concurrency to inflate latency and destabilize the workload.

See [`docs/7.SETUP.md`](docs/7.SETUP.md) for the admission levers and the multi-GPU sharding rules (automatic tensor parallelism, rank pinning, manual pins).

### Adaptive capacity

Supported lanes can reshape slot capacity from sustained workload telemetry while still respecting explicit configuration pins.

---

## Advanced request orchestration

Blazar can make decisions **per request**, not only per model.

### Best-of-N

For supported non-streaming request paths, `best_of` can generate multiple candidate answers and select among them using deterministic criteria such as schema validity and clean completion.

### Cascade routing

For supported Ollama-compatible requests, `cascade` can try a smaller or cheaper candidate first and escalate only when the result does not satisfy the decision ladder.

These features are useful for agentic systems that want to trade extra candidate computation for better answer quality or a lower average serving cost.

See [`docs/4.API_SPEC.md`](docs/4.API_SPEC.md) for exact endpoint support and constraints.

---

## Advanced feature index

Every advanced capability is on by default unless noted, and each has full documentation one link away.

| Feature | What it gives you | Details |
|---|---|---|
| **Speculative decoding** | Automatic n-gram self-speculation with a persistent lookup cache; optional draft models for faster lanes | [`docs/7.SETUP.md`](docs/7.SETUP.md#speculative-decoding) |
| **KV-cache quantization** | Automatic K/V grade ladder with context autofit — VRAM headroom and longer contexts without manual tuning | [`docs/7.SETUP.md`](docs/7.SETUP.md#memory-kv--context) |
| **Flash attention** | Enabled automatically where the engine build supports it | [`docs/7.SETUP.md`](docs/7.SETUP.md) |
| **Multi-GPU sharding** | Automatic tensor parallelism with rank pinning; explicit pins always win | [`docs/7.SETUP.md`](docs/7.SETUP.md#multi-gpu-sharding--admission) |
| **Admission and queueing** | Bounded concurrency under saturation — parks or refuses requests rather than destabilizing latency | [`docs/7.SETUP.md`](docs/7.SETUP.md#multi-gpu-sharding--admission) |
| **Federation** | Peer Blazar gateways behind one endpoint: explicit remote routing, least-busy selection, and fallback | [`docs/7.SETUP.md`](docs/7.SETUP.md#federation-peers-behind-one-gateway) |
| **Warm starts and session bank** | Session checkpoints survive unload and restart; conversation-prefix routing keeps hot KV where the next request needs it | [Sessions, cache, and warm starts](#sessions-cache-and-warm-starts) |
| **Request deduplication** | Identical concurrent requests collapse into one computation (`singleflight`) | [`docs/6.BUSINESS_RULES.md`](docs/6.BUSINESS_RULES.md) |
| **Structured output** | Schema- and grammar-constrained generation across API dialects | [`docs/7.SETUP.md`](docs/7.SETUP.md#structured-output-across-dialects) |
| **Tool calling** | Native tool-call translation across the OpenAI-, Ollama-, and Anthropic-compatible surfaces | [`docs/4.API_SPEC.md`](docs/4.API_SPEC.md#tool-calling) |
| **Reasoning budget** | Per-model and per-request thinking-budget control for reasoning models | [`docs/4.API_SPEC.md`](docs/4.API_SPEC.md#reasoning-budget) |
| **Best-of-N and cascade** | Per-request quality and cost ladders (above) | [`docs/4.API_SPEC.md`](docs/4.API_SPEC.md) |
| **Multimodal projector placement** | Vision projectors are detected and offloaded automatically so image understanding does not crowd the text hot path | [`docs/7.SETUP.md`](docs/7.SETUP.md) |
| **Load modes** | `mlock`/`mmap` posture per model — resident serving or fast swap, chosen by the planner | [`docs/7.SETUP.md`](docs/7.SETUP.md) |
| **Idle lifecycle** | Idle sleep, eviction, and watchdog monitoring keep unused capacity cheap | [Safe engine lifecycle](#safe-engine-lifecycle) |

---

## Multimodal workloads

Blazar extends the same gateway and lifecycle model beyond text.

### Text and embeddings

```text
/v1/chat/completions
/v1/responses
/v1/embeddings
/v1/rerank
```

### Image generation

```text
POST /v1/images/generations
POST /v1/images/edits
```

### Video generation

```text
POST /v1/videos/generations
```

Image and video model families are explicitly separated so a video workload is not sent to an image-only route.

### Speech-to-text

```text
POST /v1/audio/transcriptions
POST /v1/audio/translations
```

CLI:

```sh
blazar whisper --install
blazar whisper --pull base
blazar whisper file.wav
```

### Text-to-speech

```text
POST /v1/audio/speech
```

CLI:

```sh
blazar tts --install
blazar tts --pull en_US-amy-medium
blazar tts "hello" --out hello.wav
```

Long-running media operations can use gateway-owned asynchronous job handles where supported by the route.

---

## Sessions, cache, and warm starts

Local AI workloads often repeat the same context. Blazar makes that behavior visible and manageable.

### Sessions

```sh
blazar session save <model>
blazar session restore <model>
```

Session checkpoints can preserve conversational state across model unloads and daemon restarts where the selected engine supports it.

### Prefix-cache visibility

Blazar exposes cache-hit information and diagnostics for common cache-busting patterns. This helps distinguish a genuinely expensive generation from a workload that keeps invalidating its own prompt prefix.

### Semantic cache

An optional semantic cache can reuse compatible responses for sufficiently similar requests. It is opt-in rather than silently changing request behavior.

### Preload and warm-on-pull

Blazar can optionally preload configured models and can warm a freshly pulled model so the next real request can avoid the full cold-start path.

---

## Diagnostics and observability

A central design goal is to answer **"what happened?"** rather than merely returning an error code.

### Health and environment

```sh
blazar doctor
```

Checks cover areas such as system/GPU state, engines, models, runtime, channels, and service-manager integration, with corrective hints where available.

### Explain request behavior

```sh
blazar why
```

Inspect routing, queueing, model state, cache behavior, and sentinel findings using request trace information.

### Live diagnostics

```sh
blazar watch
```

Useful for observing a live workload.

### Metrics

```text
GET /metrics
```

Metrics include request behavior, model decode telemetry, cache activity, routing, and advanced scheduling/orchestration signals.

### Audit and telemetry

Blazar can provide:

- audit JSONL records;
- PII scrubbing for diagnostic output;
- OTLP traces and metrics;
- trace IDs linking gateway activity to individual requests.

External telemetry is configurable and not required for normal local operation.

---

## Safe engine lifecycle

Inference engines evolve quickly. Blazar treats them as **managed, versioned dependencies** rather than replacing one binary in place.

```sh
blazar engine update
blazar engine list
blazar engine use <tag>
blazar engine rollback
```

The engine lifecycle supports:

- SHA-256 verification;
- capability probing;
- side-by-side installations;
- explicit activation;
- regression gates;
- rollback;
- capability-lane management.

This lets you experiment with newer engines without making an existing working installation disposable.

---

## Security and network behavior

Blazar is designed for **local-first, operator-controlled deployment**.

Default gateway endpoint:

```text
127.0.0.1:11435
```

Security and deployment controls include:

- optional API keys with model, rate, token, and concurrency scopes;
- TLS using configured PEM certificate/key pairs;
- explicit CORS policy;
- PII scrubbing in diagnostic surfaces;
- audit logging;
- opt-in OTLP export;
- explicit remote routing rather than mandatory cloud connectivity.

Example key management:

```sh
blazar keys add <name>
blazar keys list
blazar keys rotate <name>
```

Downloaded engines, third-party capability forks, installers, and model weights should be handled according to your own host and supply-chain security requirements.

---

## Configuration

Configuration lives at:

```text
~/.config/blazar/config.toml
```

Prefer the CLI for routine changes:

```sh
blazar config defaults
blazar config list
blazar config get <key>
blazar config set <key> <value>
blazar config unset <key>
```

Major configuration areas include:

| Area | Examples |
|---|---|
| Serving | `host`, `port`, `default_ctx`, `slots`, `max_loaded_models` |
| Memory / quality | `cache_type`, `cache_type_k`, `cache_type_v`, `kv_unified`, `ctx_extend` |
| Scheduling | `singleflight`, `adaptive_slots`, admission and priority controls |
| Speculation | `spec`, draft selection, n-gram lookup settings |
| Routing | `engine_routing`, per-model engine overrides, replicas |
| Access | `keys`, TLS, CORS, remotes |
| Observability | `audit_log`, `pii_scrub`, `otlp_endpoint`, `otlp_service` |
| Model behavior | `chat_template`, sampler defaults, LoRA, projector, warmup |
| Media | diffusion tuning, RPC, audio-lane timeouts |

The authoritative live defaults are available from the binary itself:

```sh
blazar config defaults
```

Authentication is optional. A default local deployment does not require a gateway API key unless you configure keys.

---

## Common commands

| Task | Command |
|---|---|
| Start gateway | `blazar serve` |
| Run a model | `blazar run <model>` |
| Launch an AI CLI | `blazar launch <command>` |
| Pull a model | `blazar pull <target>` |
| Import local GGUF | `blazar import <file> --name <name>` |
| Inspect models | `blazar list` / `blazar show <model>` |
| Inspect running models | `blazar ps` |
| Stop a model | `blazar stop <model>` |
| Check hardware/configuration | `blazar doctor` |
| Explain request behavior | `blazar why` |
| Live diagnostics | `blazar watch` |
| Preview fit | `blazar fit <target>` |
| Plan co-residency | `blazar coreside` |
| Manage engines | `blazar engine update`, `blazar engine list`, `blazar engine use`, `blazar engine rollback`, `blazar engine install`, `blazar engine build`, `blazar engine prune` |
| Manage sessions | `blazar session save`, `blazar session restore` |
| Manage LoRA | `blazar lora add`, `blazar lora rm`, `blazar lora list` |
| Manage projectors | `blazar mmproj ...` |
| Tune a model | `blazar tune <model>` |
| Inspect draft candidates | `blazar drafts <model>` |
| Benchmark | `blazar bench <model>` |
| Manage API keys | `blazar keys list`, `blazar keys add`, `blazar keys rm`, `blazar keys rotate` |
| Transcribe audio | `blazar whisper <file>` |
| Generate speech | `blazar tts "<text>"` |
| Search models | `blazar search ...` |
| Back up state | `blazar snapshot` |
| Self-update | `blazar upgrade` |

Complete CLI surface:

```sh
blazar --help
```

---

## Installation

### Release binaries

Current release targets include:

- Linux: x86_64, aarch64, armv7;
- macOS: Intel and Apple Silicon;
- Windows: x64 and ARM64.

Engine capability and accelerator support depend on the selected runtime and platform.

### Build from source

```sh
cargo install --path crates/blazar-cli
```

Or from a checkout:

```sh
sh scripts/install.sh --build
```

On Windows:

```powershell
irm https://raw.githubusercontent.com/santanu20/blazar/main/scripts/install.ps1 -OutFile install.ps1
.\install.ps1 -Build
```

### Package-manager integrations

Repository integrations are maintained for [Homebrew](packaging/homebrew), [Scoop](packaging/scoop), and [Winget](packaging/winget).

---

## Reproducing benchmarks

Blazar includes a reproducible benchmark harness for performance, quality, and orchestration experiments.

Run:

```sh
blazar bench <model>
```

Or use the repository harness:

```sh
python3 scripts/bench_matrix.py --blazar-bin target/release/blazar --md BENCHMARK.md
```

The benchmark system covers areas such as:

- single-stream decode;
- concurrency and saturation;
- greedy parity;
- perplexity;
- long-context TTFT;
- tool-call and schema quality;
- slot adaptation;
- cold start / idle wake;
- image, video, TTS, and Whisper lanes.

Published numbers are **version-, model-, campaign-, and hardware-specific**. Use them for engineering comparison, not as universal performance guarantees.

See [`BENCHMARK.md`](BENCHMARK.md) and [`bench-artifacts/`](bench-artifacts).

---

## Architecture and engineering

The repository is a Rust workspace with explicit boundaries:

```text
crates/
├── blazar-core       config, model catalog, GGUF metadata, profiles,
│                     hardware, storage, domain logic
├── blazar-runtime    downloads, engines, supervisor, lifecycle,
│                     benchmarking, quantization, upgrades
├── blazar-gateway    HTTP APIs, translation, scheduling, sessions,
│                     cache, media, observability
└── blazar-cli        the `blazar` executable and interactive CLI
```

The workspace denies unsafe Rust by default (`unsafe_code = "deny"`), and CI runs Clippy with warnings treated as failures along with formatting, tests, installer checks, and related validation.

---

## Documentation

The README is the **product entry point**. Detailed operational and implementation material is split into dedicated documents.

| Document | Purpose |
|---|---|
| [`docs/1.SYSTEM_OVERVIEW.md`](docs/1.SYSTEM_OVERVIEW.md) | Product shape and system overview |
| [`docs/2.ARCHITECTURE.md`](docs/2.ARCHITECTURE.md) | Internal architecture and lifecycle |
| [`docs/3.DATA_MODEL.md`](docs/3.DATA_MODEL.md) | SQLite schema and invariants |
| [`docs/4.API_SPEC.md`](docs/4.API_SPEC.md) | Complete HTTP and CLI reference |
| [`docs/6.BUSINESS_RULES.md`](docs/6.BUSINESS_RULES.md) | Routing, admission, validation, eviction, and operational rules |
| [`docs/7.SETUP.md`](docs/7.SETUP.md) | Installation, deployment, environment, and configuration |
| [`docs/8.DO_NOT_BREAK.md`](docs/8.DO_NOT_BREAK.md) | Maintainer invariants and compatibility rules |
| [`docs/9.USAGE.md`](docs/9.USAGE.md) | End-user workflows and troubleshooting |
| [`docs/assets/blazar-logo.png`](docs/assets/blazar-logo.png) | Project logo |
| [`docs/assets/blazar-demo.svg`](docs/assets/blazar-demo.svg) | Captured CLI demo |
| [`docs/10.SCIENTIFIC.md`](docs/10.SCIENTIFIC.md) | VRAM/KV math, GGUF parsing, and numerical methods |
| [`BENCHMARK.md`](BENCHMARK.md) | Benchmark methodology and results |
| [`CHANGELOG.md`](CHANGELOG.md) | Release history |

---

## Project layout

```text
.
├── crates/
│   ├── blazar-core/
│   ├── blazar-runtime/
│   ├── blazar-gateway/
│   └── blazar-cli/
├── docs/
├── registry/
├── packaging/
├── scripts/
├── tests/
├── BENCHMARK.md
├── CHANGELOG.md
├── Cargo.toml
└── README.md
```

---

## Contributing

Before submitting changes:

1. Read [`docs/8.DO_NOT_BREAK.md`](docs/8.DO_NOT_BREAK.md).
2. For architectural changes, read [`docs/2.ARCHITECTURE.md`](docs/2.ARCHITECTURE.md) and [`docs/6.BUSINESS_RULES.md`](docs/6.BUSINESS_RULES.md).
3. Run the repository's standard formatting, Clippy, test, and installer checks.

The CI workflow is the practical source of truth for required automated checks.

---

## Credits

Blazar is an orchestration layer built on upstream open-source projects including:

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs)
- [SGLang](https://github.com/sgl-project/sglang)
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [piper](https://github.com/rhasspy/piper)

Those projects provide the underlying inference runtimes. Model weights remain subject to the licenses and distribution terms of their publishers.

---

## License

Blazar is dual-licensed under either:

- [MIT](LICENSE-MIT)
- [Apache-2.0](LICENSE-APACHE)
