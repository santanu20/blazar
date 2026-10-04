# Blazar

**The control plane for local AI inference.**

Unified OpenAI · Ollama · Anthropic gateway &nbsp;·&nbsp; Multi-engine serving &nbsp;·&nbsp; Resource-aware orchestration &nbsp;·&nbsp; Multimodal

[![CI](https://github.com/santanu20/blazar/actions/workflows/ci.yml/badge.svg)](https://github.com/santanu20/blazar/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/blazar.svg)](https://crates.io/crates/blazar)
[![Release](https://img.shields.io/github/v/release/santanu20/blazar)](https://github.com/santanu20/blazar/releases)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)
[![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](#installation)

<div align="center">

<img src="docs/assets/BLAZAR_Banner.webp" alt="Blazar — the control plane for local AI inference" width="896"/>

![Blazar in action](docs/assets/blazar-demo.svg)

*Live CLI session: single-shot generation, the engine panel with speculation state, the installed-lane inventory, and the routing decision for a model — all from one daemon.*

**[Get started in 60 seconds](#60-second-quickstart)** &nbsp;·&nbsp; **[Why use Blazar](#why-use-blazar)** &nbsp;·&nbsp; **[Benchmark evidence](#benchmark-snapshot)** &nbsp;·&nbsp; **[API reference](docs/4.API_SPEC.md)**

</div>

---

## What is Blazar?

Blazar is a **local AI gateway and runtime orchestrator** in one Rust binary. Applications talk to one endpoint; Blazar manages the model store, picks or pins the right inference engine, fits workloads to your hardware, and controls concurrency, lifecycle, and diagnostics.

**Bring your client. Bring your model. Blazar handles the serving stack.**

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

The engines still perform inference. **Blazar is the control plane around them** — it does not replace engines, train models, or require a hosted service.

### At a glance

| Capability | Blazar provides |
|---|---|
| **One gateway** | OpenAI-, Ollama-, and Anthropic-compatible APIs |
| **Multiple runtimes** | llama.cpp, mistral.rs, SGLang, MLX, stable-diffusion.cpp, whisper.cpp, piper |
| **Resource control** | VRAM/KV fit, slots, admission, co-residency, lifecycle |
| **Reliability** | Durable jobs, request cancel/interrupt, restart-safe response chains |
| **Model operations** | Pull, import, inspect, pin, tune, snapshot, restore |
| **Production controls** | API keys, TLS, CORS, audit logging, metrics, traces |
| **Federation** | Capacity-aware peers with warm, replicate, and route verbs |
| **Multimodal** | Text, embeddings, image, video, speech-to-text, TTS |

---

## 60-second quickstart

### 1. Install

#### Linux / macOS

```sh
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/santanu20/blazar/v0.19.0/scripts/install.sh \
  | BLAZAR_REPO=santanu20/blazar sh
```

#### Windows PowerShell

```powershell
irm https://raw.githubusercontent.com/santanu20/blazar/v0.19.0/scripts/install.ps1 | iex
```

Also available: `cargo install blazar`, or [build from source](#installation).

The Blazar binary and inference engines are separate artifacts. The installer can bootstrap the default engine; manage it explicitly with `blazar engine update`. Set `BLAZAR_INSTALL_ENGINE=0` to skip engine bootstrap.

### 2. Pull a model

```sh
blazar search qwen3 instruct                        # find models on Hugging Face
blazar pull qwen3-0.6b                              # registry shortname
blazar pull ggml-org/Qwen3-8B-GGUF:Q4_K_M           # Hugging Face GGUF
blazar import /path/to/model.gguf --name mymodel    # existing local file
```

### 3. Serve and run

```sh
blazar serve                # gateway on http://127.0.0.1:11435
blazar run qwen3-0.6b       # interactive REPL
```

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

No application rewrite needed — point clients at Blazar's compatible surfaces.

| Client | Base URL | Surfaces |
|---|---|---|
| **OpenAI SDK** | `http://127.0.0.1:11435/v1` | `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/rerank`, `/v1/batches`, `/v1/files`, `/v1/audio/*`, `/v1/images/*`, `/v1/videos/*` |
| **Ollama clients** | `OLLAMA_HOST=http://127.0.0.1:11435` | `/api/chat`, `/api/generate`, `/api/tags`, `/api/ps`, `/api/pull`, `/api/embeddings`, `/api/embed`, `/api/rerank` |
| **Anthropic SDK** | `http://127.0.0.1:11435` | `/v1/messages` |

Blazar runs alongside an existing Ollama install, and can later serve on `11434` as a port-level replacement.

### AI CLIs and agents

```sh
blazar launch <command>     # prepare env vars, then exec the command
blazar connect              # print the client-integration plan
blazar connect --write      # apply with backup; rolls back if the test request fails
blazar connect opencode --write --all-models   # agent-harness menu in one write
blazar connect pi --write                     # pi / oh-my-pi, same treatment
```

`connect` configures Codex, Claude Code, Continue, Cline, Open WebUI, opencode, and pi (oh my pi) to talk to Blazar — capability-driven model menus included (`--all-models` lists every certified chat model; the default pick follows the chat+tools > chat > uncertified ladder). Harnesses without a `connect` lane still work: point any OpenAI-compatible client at `http://127.0.0.1:11435/v1`, any Ollama client at the gateway root, or any Anthropic SDK client at `http://127.0.0.1:11435`.

---

## Why use Blazar?

When local inference has outgrown "run one server for one model", Blazar is the next step.

| Need | Blazar's approach |
|---|---|
| Apps speak different APIs | One gateway with **OpenAI + Ollama + Anthropic** surfaces |
| Different models need different runtimes | Capability-aware lanes across 7 engines |
| GPU memory and context are hard to manage | Fit planning, KV-aware sizing, slots, admission, co-residency |
| Engines update and break things | Verified side-by-side installs, regression gates, rollback |
| Failures are hard to diagnose | `doctor`, `why`, `watch`, metrics, trace IDs, actionable errors |
| Long generations die with the process | Durable jobs and restart-safe response chains in SQLite |
| Several machines should act as one pool | Capacity-aware federation with warm, replicate, route |
| Local deployments need controls | API keys, TLS, CORS, audit logging, PII scrubbing, OTLP |

### Blazar vs. common approaches

Architectural role, not a universal ranking:

| Approach | Best when | What Blazar adds |
|---|---|---|
| **Direct engine server** | One runtime, one model | Gateway compatibility, model ops, admission, routing, lifecycle, diagnostics |
| **Ollama** | Simple local model management | Multiple engine families, explicit routing, deeper resource planning, lifecycle controls |
| **SGLang / vLLM-style runtime** | High-throughput serving on one runtime | A higher-level control plane across heterogeneous runtimes |
| **Custom scripts / many daemons** | Bespoke infrastructure | One API, one config model, one operator workflow |

If you only need one model, one runtime, and one process, a direct engine server is simpler — Blazar is intentionally not necessary for that case.

---

## Engines and routing

| Engine | Role | Typical workloads |
|---|---|---|
| **llama.cpp** | Mainstream local text serving | GGUF and quantized local models |
| **mistral.rs** | Alternative text runtime | Supported GGUF and safetensors models |
| **SGLang** | Safetensors-oriented serving | AWQ, GPTQ, FP8 safetensors models |
| **MLX** | Apple-Silicon and CUDA-backed serving | MLX quantized model directories (mlx-community) |
| **stable-diffusion.cpp** | Media generation | Diffusion images and video |
| **whisper.cpp** | Speech recognition | Transcription and translation |
| **piper** | Offline speech synthesis | Local TTS voices |

```text
GGUF                      → llama.cpp / mistral.rs
Quantized safetensors     → SGLang
Plain safetensors         → SGLang / mistral.rs
MLX quant directories     → MLX
Diffusion component sets  → stable-diffusion.cpp
Audio transcription       → whisper.cpp
Offline TTS               → piper
```

Routing follows model format and engine capability. The default preserves single-active-engine behavior; per-model engine pins give deterministic placement. Curated capability lanes cover GGUF architectures the mainstream llama.cpp lane does not yet support.

---

## Features

### One-word UX on top of the control plane

Beginners get verbs, not knobs. `blazar plan <model>` previews what `run` would decide — identity, the live routing card, measured speed, verified capability, next actions — read-only, never pulling. `blazar run` prints a one-line ready-card of the daemon's real decision (engine, context, KV, speculation, warm state) before the first token, and `--intent agent|batch|coding|reasoning|vision` maps a workload class to queue priority, reasoning defaults, and a multimodal preflight (explicit flags still win). `blazar config preset balanced|fast|quality|agent|max-throughput` moves a handful of root knobs in one validated, printed, reversible step. `blazar scorecard <model>` assembles the stored benchmark and the capability certificate into one dated card — it never re-runs anything. `blazar autopilot` reads every model's stored records and recommends only real work (missing bench, bench newer than the tuned profile, missing or stale certificate — each rec cites its datum); `--apply` runs the list sequentially. The capability certificate also gates admission: a request whose strict needs (tools, images, JSON mode) hit a verified FAIL on the same engine kind is refused with the probe, date, and refresh command before it burns a turn — absent or stale certificates always pass.

### Resource-aware scheduling

Memory, concurrency, and residency are scheduling problems, not side effects. `blazar fit <target>` previews fit, context limits, and quantization before a download. The planner accounts for weights, context, KV-cache posture, slots, host-memory spill, co-residency, and multi-GPU placement. Saturated models get bounded admission instead of uncontrolled concurrency; supported lanes reshape slot capacity from live telemetry.

### Reliability: jobs, requests, durability

Jobs, request cards, and response chains live in a SQLite ledger that survives restarts.

- **Durable jobs** — `/v1/jobs` plane with events, cancel, and artifact endpoints. A gateway crash mid-job marks the row `abandoned` with a teaching error; terminal jobs prune after 7 days.
- **Request cards** — `GET /v1/requests` lists live generations; cancel or interrupt them from another connection, keeping partial output where asked.
- **Durable response chains** — `previous_response_id` chains promote from the ledger across restarts, never served stale.
- **Disk intelligence** — `blazar fit` checks required-vs-available disk before download; `blazar storage` reports where bytes went (per-model VRAM/disk tiers included); `blazar prune --orphans/--unused` reclaims (dry-run by default).

### Federation

Several Blazar instances act as one serving pool. Peers publish `/api/capacity` snapshots; picks prefer warm models, then lowest queue wait, then most free VRAM. `blazar warm`, `blazar replicate`, and `blazar route <model>` manage placement with per-peer outcomes. Non-Blazar remotes degrade to pass-through peers.

### Advanced orchestration

Per-request decisions, not just per-model: `best_of` candidate selection, validated parallel choices (`n` 1–8 with lane ceilings checked up front), cascade routing (try cheap first, escalate on decision-ladder failure), and typed **decision models** (`/v1/systemone`) for state-and-questions workloads.

### Multimodal

Text and embeddings through the routes above, plus:

```text
POST /v1/images/generations | edits | upscale     (ESRGAN upscale serves from a live diffusion child)
POST /v1/videos/generations
POST /v1/audio/transcriptions | translations      (streaming + Silero VAD for long audio)
POST /v1/audio/speech                              (piper TTS)
```

```sh
blazar whisper --install && blazar whisper --pull base && blazar whisper file.wav
blazar tts --install && blazar tts "hello" --out hello.wav
```

### Sessions, cache, warm starts

`blazar session save/restore` checkpoints conversational state across unloads and restarts. Prefix-cache visibility exposes cache-busting patterns; an opt-in semantic cache reuses responses for similar requests; warm-on-pull avoids the cold-start path.

### Safe engine lifecycle

Engines are managed, versioned dependencies — never one binary replaced in place.

```sh
blazar engine update                  # active lane
blazar engine update --all            # every installed lane
blazar engine install --kind <kind>   # llamacpp | mistralrs | sglang | sdcpp | whisper | piper | mlx
blazar engine use <tag> && blazar engine rollback
```

SHA-256 verification, capability probing, side-by-side installs, explicit activation, regression gates, rollback, and per-kind version checks against live upstreams in `blazar doctor`.

### Diagnostics and observability

`blazar doctor` checks system, GPU, engines, models, and channels with corrective hints. `blazar why` explains request behavior from trace data. `blazar explain <model>` prints the effective-config card where every value names its source. `blazar model-doctor <model>` runs real probes (chat, streaming, strict JSON schema, tool-call elicitation, embeddings) and stores a capability certificate. `/api/capacity` gives a live per-GPU census; `/metrics` and OTLP export cover telemetry; audit JSONL and PII scrubbing are built in.

### Web console and the ComputeFabric read-model

`http://127.0.0.1:11435/ui` is a zero-dependency, offline console served by the daemon itself — dashboard (TTFT/TPOT quantiles, cache hit rates, live engines), models, engine inventory, compute (GPU census with per-device tenants and remote peers), stored benchmarks, jobs, and sessions. It only composes existing read-only endpoints, so the page can never mutate state. `/api/fabric` exposes the same ComputeFabric inventory as JSON — CPU, GPUs (live `nvidia-smi` census with manifest fallback), residents joined to their device, external GPU processes, installed engines, and federated peers — a read model by design: it answers "what is there and what is it doing", it never moves work. `/api/quantiles` gives p50/p95/p99 latency blocks (warm/cold split) and cache counters; `/api/benchmarks` lists every stored benchmark and tuning record without re-running anything.

### Security and network behavior

Local-first: loopback by default, no cloud dependency.

- Optional API keys with model, rate, token, and concurrency scopes (`blazar keys add/list/rotate`)
- TLS from PEM pairs, explicit CORS policy
- DNS-rebinding and cross-origin hardening on loopback binds: local browser origins work by default, everything else is refused (`cors_origins` adds origins explicitly, `"*"` opts out)
- Audit logging, PII scrubbing, opt-in OTLP, explicit remote routing

---

## Benchmark snapshot

The latest committed campaign (September 29, 2026; RTX 4070 Laptop 8 GiB, i7-14650HX, Linux Mint 22.3) is published with full receipts in [`BENCHMARK.md`](BENCHMARK.md). It ran against Blazar `0.13.0`; current release is `0.19.0`. These are measured results from one configuration — evidence, not universal guarantees.

| Workload | Blazar | Reference | Interpretation |
|---|---:|---:|---|
| Single-stream decode | **40.6 t/s** | 41.5 t/s direct engine | **~2%** gateway overhead |
| Concurrent throughput, C=4 | **104.8 t/s** | 33.6 t/s Ollama lane | **~3.1×** system throughput |
| Concurrent TTFT p99, C=4 | **554 ms** | 15.9 s Ollama lane | Lower tail latency |
| Tool-call TTFT p50 | **117 ms** | 284 ms Ollama lane | **~2.4×** lower |
| Idle wake | **3.63 s** | 6.21 s Ollama lane | Faster wake |
| Adaptive serving | **49 → 63 t/s** | — | **~28%** gain, 0 failed requests |

Campaigns measure more than tokens/s: prefill, saturation, cache reuse, tool-call and structured-output quality, cold starts, scheduling, lifecycle, and media lanes. Reproduce with `blazar bench <model>` or `python3 scripts/bench_matrix.py --blazar-bin target/release/blazar --md BENCHMARK.md`.

---

## Configuration

Config lives at `~/.config/blazar/config.toml`; prefer the CLI for routine changes.

```sh
blazar config defaults | list | get <key> | set <key> <value> | unset <key>
```

| Area | Examples |
|---|---|
| Serving | `host`, `port`, `default_ctx`, `slots`, `max_loaded_models` |
| Memory / quality | `cache_type`, `kv_unified`, `ctx_extend` |
| Scheduling | `singleflight`, `adaptive_slots`, admission, priorities |
| Speculation | `spec`, draft selection, n-gram lookup |
| Routing | `engine_routing`, per-model overrides, replicas |
| Access | `keys`, TLS, CORS, remotes |
| Observability | `audit_log`, `pii_scrub`, `otlp_endpoint` |
| Media | diffusion tuning, RPC, audio-lane timeouts |

`blazar config defaults` is the authoritative live reference. Authentication is optional for a default local deployment.

---

## Common commands

| Task | Command |
|---|---|
| Start gateway | `blazar serve` |
| Run a model | `blazar run <model>` |
| Run with a workload posture | `blazar run <model> --intent agent` |
| Preview a model before running | `blazar plan <model>` |
| One-page perf + quality card | `blazar scorecard <model>` |
| One-word config posture | `blazar config preset balanced` |
| Observe → recommend per model | `blazar autopilot`, `blazar autopilot --apply` |
| Pull / import models | `blazar pull <target>`, `blazar import <file> --name <name>` |
| Inspect models | `blazar list`, `blazar show <model>`, `blazar ps` |
| Stop / remove | `blazar stop` (daemon), `blazar rm <model>` |
| Quantize / attach projector | `blazar quantize <model> <out-quant>`, `blazar mmproj <model> <mmproj.gguf path>` |
| Tune generation speed | `blazar tune <model>`, `blazar bench <model>` (results stored) |
| Launch an AI CLI | `blazar launch <command>` |
| Connect AI clients | `blazar connect --write` (Claude Code, Codex, Continue, Cline, Open WebUI, opencode, pi) |
| Explain serving config | `blazar explain <model>` |
| Certify capabilities | `blazar model-doctor <model>` |
| Check health / heal | `blazar doctor`, `blazar doctor --fix`, `blazar why`, `blazar watch` |
| Preview fit / co-residency | `blazar fit <target>`, `blazar coreside` |
| Manage engines | `blazar engine update / list / use / rollback / install / build / prune` |
| Federate | `blazar warm`, `blazar replicate`, `blazar route` |
| Sessions / LoRA | `blazar session save/restore`, `blazar lora add/rm/list` |
| Audio | `blazar whisper <file>`, `blazar tts "<text>"` |
| Disk | `blazar storage`, `blazar prune --orphans/--unused` |
| Keys / benchmark / snapshot | `blazar keys ...`, `blazar bench <model>`, `blazar snapshot` |
| Search / self-update | `blazar search ...`, `blazar upgrade` |

Full CLI surface: `blazar --help`. Complete route contract: [`docs/4.API_SPEC.md`](docs/4.API_SPEC.md).

---

## Installation

### Release binaries

- Linux: x86_64, aarch64, armv7
- macOS: Intel and Apple Silicon
- Windows: x64 and ARM64

### crates.io

```sh
cargo install blazar
```

### Build from source

```sh
cargo install --path crates/blazar-cli      # or, from a checkout:
sh scripts/install.sh --build
```

Windows: `irm https://raw.githubusercontent.com/santanu20/blazar/main/scripts/install.ps1 -OutFile install.ps1; .\install.ps1 -Build`

Package-manager integrations: [Homebrew](packaging/homebrew), [Scoop](packaging/scoop), [Winget](packaging/winget).

---

## Architecture

Rust workspace with explicit boundaries; unsafe code denied by default; CI runs Clippy with warnings as failures plus formatting, tests, and installer checks.

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

---

## Documentation

This README is the product entry point; depth lives in dedicated documents. The full series is also hosted at [santanu20.github.io/blazar](https://santanu20.github.io/blazar/).

| Document | Purpose |
|---|---|
| [`docs/1.SYSTEM_OVERVIEW.md`](docs/1.SYSTEM_OVERVIEW.md) | Product shape and system overview |
| [`docs/2.ARCHITECTURE.md`](docs/2.ARCHITECTURE.md) | Internal architecture and lifecycle |
| [`docs/3.DATA_MODEL.md`](docs/3.DATA_MODEL.md) | SQLite schema and invariants |
| [`docs/4.API_SPEC.md`](docs/4.API_SPEC.md) | Complete HTTP and CLI reference |
| [`docs/6.BUSINESS_RULES.md`](docs/6.BUSINESS_RULES.md) | Routing, admission, validation, eviction rules |
| [`docs/7.SETUP.md`](docs/7.SETUP.md) | Installation, deployment, configuration |
| [`docs/8.DO_NOT_BREAK.md`](docs/8.DO_NOT_BREAK.md) | Maintainer invariants and compatibility rules |
| [`docs/9.USAGE.md`](docs/9.USAGE.md) | End-user workflows and troubleshooting |
| [`docs/10.SCIENTIFIC.md`](docs/10.SCIENTIFIC.md) | VRAM/KV math, GGUF parsing, numerical methods |
| [`BENCHMARK.md`](BENCHMARK.md) | Benchmark methodology and results |
| [`CHANGELOG.md`](CHANGELOG.md) | Release history |

---

## Contributing

1. Read [`docs/8.DO_NOT_BREAK.md`](docs/8.DO_NOT_BREAK.md).
2. For architectural changes, read [`docs/2.ARCHITECTURE.md`](docs/2.ARCHITECTURE.md) and [`docs/6.BUSINESS_RULES.md`](docs/6.BUSINESS_RULES.md).
3. Run the standard formatting, Clippy, test, and installer checks — the CI workflow is the source of truth.

---

## Credits

Blazar is an orchestration layer built on upstream open-source projects:

- Inference runtimes: [llama.cpp](https://github.com/ggml-org/llama.cpp) · [mistral.rs](https://github.com/EricLBuehler/mistral.rs) · [SGLang](https://github.com/sgl-project/sglang) · [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) · [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) · [whisper.cpp](https://github.com/ggml-org/whisper.cpp) · [piper](https://github.com/rhasspy/piper)
- Models and architectures: [Silero VAD](https://github.com/snakers4/silero-vad) (speech detection for streaming audio) · [ModernBERT](https://github.com/AnswerDotAI/modernbert) (the System One decision-model family rides ModernBERT-architecture GGUFs)

Those projects provide the inference runtimes. Model weights remain subject to their publishers' licenses and distribution terms.

---

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).
