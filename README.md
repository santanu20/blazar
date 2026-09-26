# blazar

> **One local inference gateway for OpenAI, Ollama, and Anthropic clients — with multiple engines, VRAM-aware scheduling, model management, and operator-grade diagnostics in one Rust binary.**

[![CI](https://github.com/santanu20/blazar/actions/workflows/ci.yml/badge.svg)](https://github.com/santanu20/blazar/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/santanu20/blazar)](https://github.com/santanu20/blazar/releases)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)
[![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](#installation)

**Current release: `0.12.0`**

---

## What is Blazar?

Blazar is a **local inference gateway and runtime orchestrator**.

It sits in front of local inference engines and gives applications one stable endpoint while Blazar handles model discovery, engine selection, process lifecycle, memory fit, concurrency, sessions, diagnostics, and verified engine updates.

```text
OpenAI SDK ───────┐
Ollama clients ───┼──► blazar gateway ──► llama.cpp
Anthropic SDK ────┘          │            mistral.rs
                              │            SGLang
                              │            sd.cpp (images / video)
                              │
                              ├── model store
                              ├── VRAM / KV fit
                              ├── routing + scheduling
                              ├── sessions + cache visibility
                              └── diagnostics + engine lifecycle
```

### The core idea

**Bring your client. Bring your model. Blazar handles the serving stack.**

You get one local port, three API dialects, multiple engine backends, and one operational surface.

---

## Why use it?

| Problem | Blazar's approach |
|---|---|
| Multiple clients speak different APIs | One gateway supports **OpenAI + Ollama + Anthropic** APIs |
| Different model families need different runtimes | Capability-driven **engine routing** across llama.cpp, mistral.rs, SGLang, and sd.cpp (diffusion/video) |
| GPU memory is easy to oversubscribe | `blazar fit`, capacity-aware profiles, KV-cache controls, and model co-residency planning |
| Engine upgrades can break working installs | Verified, side-by-side engine installs with **rollback and regression gates** |
| Model stores become opaque and tool-specific | Local model files remain ordinary **GGUF / safetensors** files |
| Failures are hard to diagnose | `blazar doctor`, `blazar why`, `blazar watch`, trace IDs, metrics, and explicit teaching errors |
| Local deployments become operationally messy | One binary, one documented config, system service support, snapshots, and self-update |

Blazar is intentionally an **orchestrator** rather than a reimplementation of model inference. The engines still do the heavy inference work.

---

## 60-second quickstart

### 1. Install

#### Linux / macOS

For the current `0.12.0` release:

```sh
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/santanu20/blazar/v0.12.0/scripts/install.sh \
  | BLAZAR_REPO=santanu20/blazar sh
```

#### Windows PowerShell

```powershell
irm https://raw.githubusercontent.com/santanu20/blazar/v0.12.0/scripts/install.ps1 | iex
```

The release binary does **not** bundle an inference engine. Install/update the engine separately:

```sh
blazar engine update
```

The installer can also bootstrap the engine on supported Unix installs; `blazar engine update` is the explicit, cross-platform control point.

### 2. Pull a model

```sh
# Ollama registry shortname
blazar pull qwen3-0.6b

# Hugging Face GGUF
blazar pull ggml-org/Qwen3-8B-GGUF:Q4_K_M
```

GGUF can also be imported without copying the file:

```sh
blazar import /path/to/model.gguf --name mymodel
```

### 3. Start the gateway

```sh
blazar serve
```

Blazar listens on `127.0.0.1:11435` by default.

### 4. Run a model

In another shell:

```sh
blazar run qwen3-0.6b
```

Or call it directly through the OpenAI-compatible API:

```sh
curl http://127.0.0.1:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [{"role": "user", "content": "Explain local inference in one paragraph."}]
  }'
```

---

## Existing Ollama clients can move over incrementally

Blazar is designed to coexist with Ollama while you test it.

Point an existing Ollama client at Blazar:

```sh
export OLLAMA_HOST=http://127.0.0.1:11435
```

OpenAI clients can use:

```text
http://127.0.0.1:11435/v1
```

Anthropic-compatible clients use:

```text
http://127.0.0.1:11435
```

When you are ready for a full port-level replacement, configure Blazar for `11434`, stop Ollama, and keep the clients unchanged.

### Models stay portable

Blazar keeps model weights as ordinary local files rather than requiring an opaque runtime-specific blob store.

```text
~/.local/share/blazar/models/
```

Existing GGUF files can be registered with `blazar import`; imports use hardlinks by default, so registration does not duplicate the model data.

### Launch agent CLIs preconfigured

`blazar launch` wraps any AI CLI: it exports the OpenAI/Anthropic/Ollama base-URL environment for the local daemon, ensures the daemon is up, then execs the command unchanged.

```sh
blazar launch claude    # or any tool that reads ANTHROPIC_BASE_URL / OPENAI_BASE_URL / OLLAMA_HOST
blazar launch dsh
```

---

## Engines and model formats

Blazar currently orchestrates four engine families:

| Engine | Typical formats / role |
|---|---|
| **llama.cpp** | GGUF; default mainstream lane for quantized GGUF serving |
| **mistral.rs** | GGUF and safetensors paths supported by the runtime |
| **SGLang** | Safetensors; especially AWQ / GPTQ / FP8 on supported accelerators |
| **sd.cpp** | Diffusion + video checkpoints via a prebuilt `sd-server` (CUDA on NVIDIA when upstream ships it, Vulkan on every GPU, CPU/Metal otherwise). Nine curated families — Qwen-Image-2.1, Qwen-Image (v1), FLUX.1, Z-Image, Chroma, FLUX.2-dev (component sets with flag-keyed VAE/text-encoder pulls), SDXL, SD 1.5 (single self-contained checkpoints) and Wan 2.1 T2V (video: DiT + `--vae` + `--t5xxl`) — plus async jobs, SSE progress, native-dialect translation and huggingface hub-cache reuse |

Routing is capability-driven rather than a blind global switch.

Typical policy:

- **GGUF** → llama.cpp, with mistral.rs available as an alternate lane.
- **Quantized safetensors (AWQ/GPTQ/FP8)** → SGLang.
- **Plain safetensors** → SGLang or mistral.rs according to routing policy.
- **Diffusion component sets** → sd.cpp (`blazar engine install --kind sdcpp`); the domain gate is bidirectional — text engines never receive component rows and sd.cpp never receives text models. Pulling a known diffusion family (`Qwen-Image-2.1-GGUF`, FLUX.1 GGUF repos) fetches the full flag-keyed component set (e.g. `--vae` + `--t5xxl` + `--clip_l` for FLUX.1) as one model, re-using byte-exact files already on disk; generation rides `POST /v1/images/generations` (and `/v1/images/edits` when the family ships a vision encoder), with opt-in `"async": true`/`"stream": true` job modes (`GET /v1/images/jobs/{id}`, `POST /v1/images/jobs/{id}/cancel`, `GET /v1/images/capabilities`) and rich native fields (cache engines, LoRA, guidance) shallow-translated automatically. Component pulls reuse byte-exact files from the local huggingface hub cache (hardlinked, `rm`-safe). Video families (Wan 2.1 T2V) serve `POST /v1/videos/generations` with the same job modes at `/v1/videos/jobs/*` — the gateway refuses a video family on the images route and vice versa.
- `engine_routing.mode = "manual"` pins a single active engine when you explicitly want that behavior.
- A per-model engine override wins over automatic routing.
- The ENGINE column in `blazar list` is the routing lane, not a capability guarantee; a `†` cell (plus a footer line, or the `engine_arch_gap` field in `--json`) marks a GGUF architecture the routed llama.cpp build provably cannot load — spawn fails with teaching unless a covering fork lane is installed.
- When an installed lane (e.g. a fork build) advertises an architecture the picked lane provably lacks, all three listings (`list`, `/api/tags`, `/v1/models`) show that lane — the same one the spawn-time capability rescue lands on — so previews never advertise a lane that would crash first.
- Multi-node offload: per-model `rpc_servers` overrides spawn text engines with `--rpc gpu:node...` against your own `ggml-rpc-server` workers; a user-run RPC server is treated as a foreign co-tenant (reported by `doctor`, never swept by the orphan cleaner).

For model fit and engine choice:

```sh
blazar fit <model-or-repo>
blazar list
blazar show <model>
blazar ps
```

---

## One gateway, three API dialects

### OpenAI-compatible

Common surfaces include:

```text
/v1/chat/completions
/v1/completions
/v1/embeddings
/v1/rerank
/v1/responses
/v1/batches
/v1/files
/v1/audio/transcriptions
/v1/audio/translations
/v1/audio/speech
/v1/images/generations
/v1/images/edits
/v1/videos/generations
```

### Ollama-compatible

```text
/api/chat
/api/generate
/api/tags
/api/ps
/api/pull
```

### Anthropic-compatible

```text
/v1/messages
```

### Blazar-native control plane

```text
/api/evict
/api/session
/api/keys
/api/why
/api/watch
/.well-known/blazar
```

See the complete route, payload, error, and configuration reference in [`docs/4.API_SPEC.md`](docs/4.API_SPEC.md).

---

## The features that matter in production

### VRAM-aware serving

Blazar profiles a model against the available hardware before spawning it. `blazar fit` can preview fit and quant alternatives before you download a large model.

KV-cache policy can trade memory against capacity using supported cache types such as `f16`, `q8_0`, and `q4_0`.

```sh
blazar fit qwen3-0.6b
blazar coreside
```

### Engine lifecycle without blind replacement

Engines are installed side-by-side, verified, probed, and switched explicitly. Updates are regression-gated and can roll back when a measured decode regression crosses the configured threshold.

```sh
blazar engine update
blazar engine list
blazar engine use <tag>
blazar engine rollback
```

### Sessions and cache-aware workflows

Session checkpoints can survive model unloads and daemon restarts. The gateway also exposes cache-hit information and detects common prefix-cache busting patterns.

```sh
blazar session save <model>
blazar session restore <model>
blazar why
```

### Diagnostics instead of guesswork

```sh
blazar doctor
blazar why
blazar watch
blazar ps
```

The diagnostic surfaces expose trace IDs, routing decisions, model state, context/slot information, engine failures, and sentinel detections with actionable fix hints.

### Speculative decoding

Blazar can select and expose compatible draft candidates for speculative decoding, while retaining explicit per-request controls.

```sh
blazar drafts <model>
blazar run <model> --no-draft
```

### LoRA and vision support

Blazar supports managed LoRA adapters, per-request adapter variants, and vision projectors for compatible models.

```sh
blazar lora add <model> /path/to/adapter.gguf
blazar lora list
blazar mmproj ...
```

### Images, video, and speech

`blazar run` is multimodal by model kind: text models open a streaming chat loop, diffusion sets generate images or video, and pulled piper voices write WAV clips — with an inline `PROMPT` every lane runs single-shot and exits. The same media lanes are served over HTTP on the routes listed above.

```sh
# image generation (sd.cpp lane)
blazar pull Qwen-Image-2.1-GGUF
blazar run qwen-image-2.1

# text-to-video (Wan 2.1 via sd.cpp)
blazar pull Comfy-Org/Wan_2.1_ComfyUI_repackaged
blazar run wan_2.1_comfyui_repackaged

# speech: install the lanes, pull assets, generate
blazar tts --install && blazar tts --pull en_US-amy-medium
blazar tts "hello" --out hello.wav
blazar whisper --install && blazar whisper --pull base
blazar whisper file.wav
```

Both voice lanes are fully managed: `--list` inventories what is installed, `--pin <tag>`/`--pin none` freezes (or frees) the exact binary version served — the pin is honored across both the engines lane and the legacy tree. Whisper can also transparently use a remote `whisper: [[remotes]]` entry when one is configured.

---

## Capability lanes for architectures not in mainstream llama.cpp yet

When a GGUF architecture is not available in the mainstream engine, Blazar can build a **capability lane** from an immutable llama.cpp fork commit, record the provenance, and route only the models that need it.

```sh
blazar engine offers
blazar engine install --lane <id> --backend cuda
```

Capability lanes are a bridge, not the normal path: mainstream engines take precedence for architectures they already support, and curated lanes can be retired automatically once upstream support arrives.

For untrusted third-party forks, treat the lane as executable code running with your privileges and review the provenance before installing it.

---

## Installation details

### Supported release targets

Current release assets cover:

- **Linux:** x86_64, aarch64, armv7; GNU builds plus static musl fallback where needed.
- **macOS:** Intel and Apple Silicon.
- **Windows:** x64 and ARM64, with an emulated x64 fallback when a native ARM64 asset is unavailable.

Package-manager integrations are also maintained in the repository for **Homebrew, Scoop, and Winget**.

### Build from source

```sh
cargo install --path crates/blazar-cli
```

Or run the repository installer from a checkout:

```sh
sh scripts/install.sh --build
```

On Windows:

```powershell
irm https://raw.githubusercontent.com/santanu20/blazar/main/scripts/install.ps1 -OutFile install.ps1
.\install.ps1 -Build
```

### Update Blazar itself

```sh
blazar upgrade
```

Use `--dry-run` to preview an upgrade or `--version` to pin a release.

---

## Configuration

Blazar uses a documented TOML configuration file:

```text
~/.config/blazar/config.toml
```

Common configuration areas include:

| Area | Examples |
|---|---|
| Serving | `host`, `port`, `default_ctx`, `slots`, `max_loaded_models` |
| Memory / quality | `cache_type`, `kv_unified`, `cache_ram_mb`, `ctx_extend` |
| Scheduling | `singleflight`, `prompt_preflight`, priority / queue controls |
| Speculation | `spec`, draft behavior |
| Routing | `engine_routing`, per-model engine overrides, replicas |
| Access | `[[keys]]`, TLS, CORS |
| Observability | `audit_log`, `pii_scrub`, `otlp_endpoint`, `otlp_service` |
| Model behavior | `chat_template`, samplers, LoRA, mmproj, warmup |
| Media lanes | `sdcpp_flash_attention`, `sdcpp_vae_tiling`, `sdcpp_rpc_servers`, `media_job_wait_secs`, `whisper_idle_secs` |

Inspect or change settings through the CLI instead of hand-editing whenever practical:

```sh
blazar config defaults
blazar config list
blazar config get <key>
blazar config set <key> <value>
```

Auth is optional: adding at least one `[[keys]]` entry activates gateway authentication. Without keys, the local gateway is open.

---

## Security and network behavior

Blazar is designed for local-first operation:

- The gateway binds to `127.0.0.1` by default.
- There is no account requirement and registry push/login commands are intentionally refused.
- Release and engine downloads are SHA-256 verified against release metadata.
- Engine installs are isolated side-by-side rather than replacing a working engine in place.
- Model pulls are resumable and support verification against recorded digests where a content digest is available.
- OTLP observability export is **opt-in** via configuration; it is not a mandatory telemetry service.
- Explicit remote routing can be configured when an operator chooses to use it.

Treat the installer, downloaded engines, third-party forks, and model files according to your own supply-chain and host-security requirements.

---

## Performance

Blazar ships a reproducible benchmark harness; [`BENCHMARK.md`](BENCHMARK.md) holds the methodology and the latest campaign results. Measured lanes: single-stream speed, concurrency sweep (per-level system throughput, efficiency vs C=1, saturation verdicts), greedy parity against the raw engine, perplexity, long-context TTFT curve, tool-call selection/schema quality, adaptive slot reshape under sustained load, cold start, idle wake, and media (image/video/TTS/whisper).

Every campaign writes its receipts to `bench-artifacts/` (see `bench-artifacts/INDEX.md` for the campaign ledger), and the report's engine-coverage table accounts for every installed engine as measured or excluded-with-reason.

Run a benchmark with:

```sh
blazar bench <model>
```

or use the repository harness:

```sh
python3 scripts/bench_matrix.py --blazar-bin target/release/blazar --md BENCHMARK.md
```

**Important:** published benchmark tables are version- and hardware-specific. Do not read historical campaign numbers as guarantees for the current release or for different GPUs.

---

## Quality and engineering discipline

The repository is a Rust workspace with four primary crates:

```text
blazar-core       domain logic, config, store, catalog, GGUF parsing, profiles, hardware
blazar-runtime    downloads, engines, supervisor, lifecycle, benchmarks, quantization
blazar-gateway    HTTP APIs, scheduling, translation, sessions, cache, sentinel, audio lanes
blazar-cli        the `blazar` executable and interactive CLI
```

The project enforces `unsafe_code = deny` at the workspace level and uses Clippy with warnings treated as failures in the CI configuration.

The test surface includes unit/integration coverage for the gateway, lifecycle/supervisor behavior, engine installation, compatibility paths, installer flows, and related network stubs.

---

## Common commands

| Task | Command |
|---|---|
| Start server | `blazar serve` |
| Chat | `blazar run <model>` |
| Generate an image / video / voice clip | `blazar run <diffusion-or-voice-model>` |
| Launch an agent CLI against the daemon | `blazar launch <command>` |
| Unload one model now | `blazar stop <model>` |
| Pull model | `blazar pull <target>` |
| Import local GGUF | `blazar import <file> --name <name>` |
| Derive another quantization | `blazar quantize <model> ...` |
| Attach a vision projector | `blazar mmproj <model> <file>` |
| Alias / copy a model | `blazar cp` / `blazar create` |
| Inspect models | `blazar list` / `blazar show <model>` |
| Inspect running models | `blazar ps` |
| Check hardware / configuration | `blazar doctor` |
| Explain a request | `blazar why` |
| Live diagnostics | `blazar watch` |
| Preview VRAM fit | `blazar fit <target>` |
| Co-residency plan | `blazar coreside` |
| Benchmark | `blazar bench <model>` |
| Tune | `blazar tune <model>` |
| Draft-model candidates | `blazar drafts <model>` |
| Manage engines | `blazar engine update|list|use|rollback|build|local|offers|prune` |
| Transcribe audio | `blazar whisper <file>` |
| Text-to-speech | `blazar tts "<text>"` |
| Manage API keys | `blazar keys list|add|rm|rotate` |
| Save / restore sessions | `blazar session save|restore ...` |
| Manage LoRA | `blazar lora add|rm|list ...` |
| Search Hugging Face | `blazar search ...` |
| Backup state | `blazar snapshot` |
| Shell completions | `blazar completions bash|zsh|fish|powershell` |
| Self-update | `blazar upgrade` |

Run `blazar --help` for the command tree and `docs/4.API_SPEC.md` for the full reference.

---

## Documentation

The README is the product entry point. The detailed documentation is split by job:

| Document | Purpose |
|---|---|
| [`docs/1.SYSTEM_OVERVIEW.md`](docs/1.SYSTEM_OVERVIEW.md) | Product shape, users, workflows, architecture overview |
| [`docs/2.ARCHITECTURE.md`](docs/2.ARCHITECTURE.md) | Internal architecture, lifecycle, queue, supervisor, failure modes |
| [`docs/3.DATA_MODEL.md`](docs/3.DATA_MODEL.md) | SQLite schema, indexes, invariants |
| [`docs/4.API_SPEC.md`](docs/4.API_SPEC.md) | HTTP routes, payloads, errors, CLI surface |
| [`docs/6.BUSINESS_RULES.md`](docs/6.BUSINESS_RULES.md) | Limits, validation, routing, eviction, operational rules |
| [`docs/7.SETUP.md`](docs/7.SETUP.md) | Build, install, deploy, environment variables, configuration |
| [`docs/8.DO_NOT_BREAK.md`](docs/8.DO_NOT_BREAK.md) | Maintainer invariants and compatibility rules |
| [`docs/9.USAGE.md`](docs/9.USAGE.md) | End-user workflows, troubleshooting, FAQ |
| [`docs/10.SCIENTIFIC.md`](docs/10.SCIENTIFIC.md) | VRAM/KV math, GGUF parsing, routing evidence |
| [`BENCHMARK.md`](BENCHMARK.md) | Benchmark methodology and historical results |
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

Blazar is structured as a Rust workspace with explicit boundaries between domain logic, runtime/orchestration, gateway behavior, and CLI concerns.

Before submitting changes, run the repository's normal checks and inspect the contributor invariants in [`docs/8.DO_NOT_BREAK.md`](docs/8.DO_NOT_BREAK.md).

For architectural changes, start with [`docs/2.ARCHITECTURE.md`](docs/2.ARCHITECTURE.md) and [`docs/6.BUSINESS_RULES.md`](docs/6.BUSINESS_RULES.md).

---

## Credit

Blazar is an orchestration layer built on upstream inference projects including:

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs)
- [SGLang](https://github.com/sgl-project/sglang)
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) — the sd.cpp image/video lane
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) — the transcription lane
- [piper](https://github.com/rhasspy/piper) — the offline TTS lane

Those projects provide the underlying inference engines. Model weights remain the property and responsibility of their publishers.

---

## License

Blazar is dual-licensed under either:

- [MIT](LICENSE-MIT)
- [Apache-2.0](LICENSE-APACHE)
