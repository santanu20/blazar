# pallama

[![CI](https://github.com/santanu20/pallama/actions/workflows/ci.yml/badge.svg)](https://github.com/santanu20/pallama/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/santanu20/pallama?include_prereleases)](https://github.com/santanu20/pallama/releases)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT_OR_Apache--2.0-blue)](#license)
[![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](#install)
[![Engines](https://img.shields.io/badge/engines-llama.cpp%20%7C%20mistral.rs%20%7C%20SGLang-orange)](#credit)

**Every model. Every API. Your hardware at full speed. One binary.**

`pallama` is one Rust binary that turns any machine into a complete local inference server.

| The pitch | The reality |
|---|---|
| **Every model** | Three engines, unmodified, side-by-side: [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` (GGUF) · [mistral.rs](https://github.com/EricLBuehler/mistral.rs) (HF safetensors) · [SGLang](https://github.com/sgl-project/sglang) (safetensors + AWQ/GPTQ on CUDA/ROCm) |
| **Every API** | One gateway speaking the **OpenAI**, **Ollama** and **Anthropic** APIs simultaneously |
| **Full speed** | Engines are probed, profiled and orchestrated — nothing reimplemented, nothing slowed down |
| **One binary** | Port **11435** — pallama's own port, so it never collides with a running ollama; point existing clients at it with zero code changes |

| Guarantee | How it is kept |
|---|---|
| **No fork** | Official upstream engine binaries, sha256-verified, installed side-by-side with atomic switching and rollback |
| **No lock-in** | Models stay plain `.gguf` files you can touch with any tool |
| **No telemetry, no cloud** | The only outbound traffic pallama ever generates is the engine/model downloads you ask for — stated in `--help` and at startup |

**Contents:** [The numbers](#the-numbers) · [Switch from ollama](#switch-from-ollama) · [Who it's for](#who-its-for) · [Install](#install) · [Quickstart](#60-second-quickstart) · [Power-tool CLI](#the-power-tool-cli) · [Ollama complaint table](#every-ollama-complaint-fixed-at-the-root) · [How it works](#how-it-works) · [Quality](#quality) · [Docs](#documentation)

## The numbers

Measured, not marketed. Same laptop, same model (Qwen3.5-9B Q4_K_M), reproducible with one command. Full methodology, raw artifacts and caveats: [BENCHMARK.md](BENCHMARK.md).

| What you feel | pallama | ollama 0.33.3 | Delta |
|---|---:|---:|---:|
| 4-stream concurrent throughput (system t/s) | **104.2** | 33.4 | **3.1x** |
| Tail latency, inter-token p99 (ms) | **27.4** | 75.1 | **2.7x tighter** |
| Wake from idle (ms) — sleep vs full reload | **2104** | 7194 | **3.4x** |
| Prefill on cached prompt (t/s) | **7332** | 5611 | **1.3x** (and 6x over pallama's own cold prefill) |
| Gateway overhead vs direct engine (decode t/s) | 40.3 | 41.1 | **-1.9% (noise)** |
| Daemon boot (s) | **0.54** | 4.13 | **7.7x** |

In practice: **3x more concurrent streams from the GPU you already own**, agent loops that don't stall on tail latency, sessions that wake in a blink, and cached prefill that stops re-paying the long-context tax every turn.

| Test bed | i7-14650HX · RTX 4070 Laptop 8 GiB · Linux · pallama 0.5.0 gateway over llama.cpp b10903-cuda |
|---|---|
| Honesty clause | Cold-load first token is slower (11.6 s vs 6.6 s): pallama defaults to a 16384 context where ollama silently truncates at 2048. Set `ctx` lower and get ollama's load time back — you can't buy ollama's missing 14 Ki of context at any price. Ratios travel across hardware; absolute numbers shift. |

Reproduce it yourself:

```sh
python3 scripts/bench_matrix.py --pallama-bin target/release/pallama --md BENCHMARK.md
```

## Switch from ollama

Low-risk and incremental: pallama runs **alongside** ollama (different port), keeps using the models you already have, and only becomes a drop-in replacement when *you* decide.

| Step | Action |
|---|---|
| 1 · Install | `curl --proto '=https' --tlsv1.2 -fsSL https://raw.githubusercontent.com/santanu20/pallama/main/scripts/install.sh \| sh` — the installer bootstraps the llama.cpp engine automatically (sha256-verified; Linux preflights your GPU driver), so a fresh install serves inference immediately. Windows + source paths: [Install](#install) |
| 2 · Keep your models | `pallama import /path/to/model.gguf --name mymodel` registers GGUF already on disk, hardlinked in place (zero copy; `--copy` for a duplicate). Or re-pull the shortnames you know — they route to registry.ollama.ai, sha256-verified and resumable — and any `owner/repo:QUANT` from Hugging Face works: `pallama pull qwen3-0.6b` · `pallama pull ggml-org/Qwen3-8B-GGUF:Q4_K_M` |
| 3 · Repoint your tools | Zero code changes — same shortnames, same `model:tag` colons, same wire formats, all on `http://127.0.0.1:11435`: `OLLAMA_HOST=http://127.0.0.1:11435 ollama list` · OpenAI SDK `base_url = "http://127.0.0.1:11435/v1"` · Anthropic SDK `base_url = "http://127.0.0.1:11435"` |
| 4 · Go full drop-in | When ready: set `port = 11434` in `config.toml`, remove ollama — every `OLLAMA_HOST`-less client keeps working unchanged |

What changes on disk — and what doesn't:

| | ollama | pallama |
|---|---|---|
| **Models live in** | `~/.ollama/models/` | `~/.local/share/pallama/models/` |
| **A quant on disk** | `blobs/sha256-8f4a8e…` — opaque blob, only ollama reads it | `qwen3-8b-q4_k_m.gguf` — plain file, any tool |
| **Same model, other quant** | `blobs/sha256-1c9d02…` — opaque blob | `qwen3-8b-q8_0.gguf` — plain file, any tool |
| **Model metadata** | `manifests/library/qwen3` — internal hash tree | SQLite store, fully inspectable (`pallama show`) |
| **Configuration** | `OLLAMA_*` env sprawl, undocumented defaults | one documented `~/.config/pallama/config.toml` |

Everything on the pallama side is a plain file or an inspectable row — `pallama import` hardlinks GGUF in place, so nothing is re-downloaded and nothing is duplicated. The payoff is the [measured table above](#the-numbers) plus every long-standing ollama complaint resolved at the root in the [table below](#every-ollama-complaint-fixed-at-the-root).

## Who it's for

| You are | What pallama hands you |
|---|---|
| **The agent builder** | Point your OpenAI, ollama or Anthropic SDK at one port and go — three API dialects, zero code changes. Tool calls, `json_schema` structured output, embeddings, rerank, transcription, speech, batch endpoints, capability discovery at `/.well-known/pallama`. When an agent loop mysteriously stalls, `pallama why [trace]` says exactly what that request did and what went wrong — after the fact, from real telemetry |
| **The ollama refugee** | Every classic complaint fixed at the root — [the table below](#every-ollama-complaint-fixed-at-the-root). Plain `.gguf` files instead of a hashed blob store, any HF quant, multi-shard GGUF pulled natively, `keep_alive` and `num_ctx` honored in both APIs, concurrent streams that actually run in parallel |
| **The power user** | Per-request context sizes with budget validation *before* spawn · KV-cache quantization ladder (f16 → q8_0 → q4_0) with visible VRAM math · slot-level session checkpoints that survive unload and restarts · speculative decoding with per-request control (`options.spec`, `--no-draft`) · LoRA adapters and `model+adapter` variants · vision projectors · GBNF grammars — every capability the installed engine has, probed and exposed, none hand-configured |
| **The cautious operator** | `pallama fit` previews VRAM fit and quant alternatives *before* you download 30 GB · `pallama doctor` diagnoses config, ports, engines, hardware, disk and model health in one table · engine updates are regression-gated (a >10% decode drop auto-rolls-back) · a crash circuit restarts children; an eviction ladder sleeps idle models without killing in-flight requests · systemd/launchd units with `Restart=always` ship in the box |
| **The privacy hardliner** | Local-only by design, stated in `--help` and at startup — no telemetry, no cloud endpoints, no phone-home. HTTPS to Hugging Face only; registry tokens only ever sent to first-party hosts, verified by tests (the CVE-2025-51471 token-exfiltration class cannot happen here) |

## Install

Prebuilt binaries, sha256-verified against release metadata by the installers (the same mechanism as `pallama engine update`):

| OS | Architectures | Notes |
|---|---|---|
| **Linux** | x86_64 · aarch64 · armv7 | glibc ≥ 2.35, or static musl (Alpine and old-glibc hosts auto-fallback; 32-bit ARM boards get static armv7-musl) |
| **macOS** | x86_64 · Apple Silicon | launchd service (`KeepAlive`, `RunAtLoad`) |
| **Windows** | x64 · ARM64 | installs to `%LOCALAPPDATA%\Programs\pallama`, adds user PATH; ARM64 hosts pick the native asset, falling back to emulated x64 with a warning when a release has none |

**Linux / macOS (WSL included):**

```sh
# one-liner from the published repo (override with PALLAMA_REPO=<owner>/<name>;
# inside a clone the repo is derived from the git origin automatically):
curl --proto '=https' --tlsv1.2 -fsSL \
  https://raw.githubusercontent.com/santanu20/pallama/main/scripts/install.sh | sh

# from a source checkout (zero arguments: builds the checkout fresh with
# cargo, then installs system-wide; a missing cc/rust toolchain is
# provisioned first via scripts/bootstrap.sh --minimal — announced, never silent):
sudo sh scripts/install.sh

# or force the source path explicitly (no release channel contact):
sudo sh scripts/install.sh --build
```

**Windows (PowerShell):**

```powershell
irm https://raw.githubusercontent.com/santanu20/pallama/main/scripts/install.ps1 | iex

# or compile from source (rustup is installed via winget when missing):
irm https://raw.githubusercontent.com/santanu20/pallama/main/scripts/install.ps1 -OutFile install.ps1
.\install.ps1 -Build
```

**From source:** `sudo cargo install --path crates/pallama-cli` (recent stable Rust) — or just run the installer from the checkout.

**Staying current:** `pallama upgrade` self-updates the binary from GitHub Releases (sha256-verified, same mechanism as engine installs; `--dry-run` to preview, `--version` to pin).

### What the installer does for you

| Capability | Detail |
|---|---|
| **One-click readiness** | After the binary lands, bootstraps the llama.cpp engine (idempotent — skips when an engine is already active) so a fresh install serves inference immediately; failures are loud warnings, never silent. Opt out `PALLAMA_INSTALL_ENGINE=0` · pre-pull a model `PALLAMA_INSTALL_MODEL=<name>` · engine mirror `PALLAMA_GH_BASE` · daemon cgroup cap `PALLAMA_UNIT_MEMORY_HIGH` (systemd `MemoryHigh`, default `85%` of RAM — soft reclaim/throttle only, never an OOM kill; empty string omits the line) |
| **GPU preflight (Linux)** | Before the engine lands: censuses PCI GPU hardware; when driver userspace is absent, installs it from **first-party distro repos only** (announce-then-act), then walks you through REBOOT → `pallama engine update`, which auto-picks the newest CUDA build your driver supports (runtimes bundled; no CUDA toolkit ever installed). Windows is advise-only (`pallama doctor` warns on driverless GPUs). Opt out `PALLAMA_AUTO_DRIVER=0`. Per-distro detail: [docs/7.SETUP — GPU driver preflight](docs/7.SETUP.md#gpu-driver-preflight-installer) |
| **Old glibc or Alpine** | Automatic fallback to the static musl build. Pin a version `PALLAMA_VERSION=v0.4.0` · mirror `PALLAMA_INSTALL_BASE_URL` · build repo `PALLAMA_CHECKOUT`. No toolchain? Bootstrapped for you (`scripts/bootstrap.sh --minimal`, opt out `PALLAMA_AUTO_BOOTSTRAP=0`) |
| **System-wide only, like ollama's installer** | Root-owned binary in `/usr/local/bin` plus a systemd unit (`Restart=always`, GPU groups, auto-start, restart-on-upgrade; runs as the invoking user — under `sudo` the unit targets `SUDO_USER`, not root) on Linux, or a launchd service on macOS. Deliberately no user-path (`~/.local/bin`) mode: a second copy there is how stale-binary daemon races happen (`pallama doctor` flags any; the installer removes one it finds). Root or sudo required |

### Uninstall

| Mode | Command | Scope |
|---|---|---|
| Full | `sh scripts/uninstall.sh` | Services, binary, unit + drop-ins, config (incl. `gh-token.env`), store, engines, runtime/caches — models **asked first** (sizes shown, default keep; `--remove-models`/`--yes` non-interactive, `--keep-models` skip, `--dry-run` transcript) |
| Quick | `sh scripts/install.sh --uninstall` | Binary + units only; models and config under `~/.local/share/pallama` / `~/.config/pallama` stay until you delete them |

## 60-second quickstart

![pallama run + pallama why](docs/assets/pallama-demo.svg)

_Live capture: `pallama run` streams the answer plus `[profile]` transparency lines — every tuning decision the compiler made, and how to override it — and `pallama why` explains the request after the fact: model, status, finish reason, context, token counts, latency._

```sh
pallama engine update                        # install + activate upstream llama-server
                                             #   (sha256-verified; regression-gated)
pallama pull qwen3-0.6b                      # ollama shortname → registry.ollama.ai
pallama pull ggml-org/Qwen3-8B-GGUF:Q4_K_M   # or any owner/repo:QUANT from HF
pallama run qwen3-0.6b                       # streaming REPL (auto-pulls if missing)
pallama show qwen3.5:9b                      # model:tag colons resolve onto flat rows
OLLAMA_HOST=http://127.0.0.1:11435 ollama list   # existing ollama clients just work
```

Want a CUDA build from source? `pallama engine build cuda` compiles the in-tree ggml-cuda backend from an upstream tag — auto GPU-arch detection, auto CUDA host-compiler matching — installed through the same probe → regression-gate → activate flow as release assets. A `cpu` backend is also available.

Three API dialects on the one port:

| Dialect | Endpoints |
|---|---|
| **OpenAI** | `/v1/chat/completions` · `/v1/completions` · `/v1/embeddings` · `/v1/rerank` · `/v1/messages` · `/v1/responses` · `/v1/batches` · `/v1/files` · `/v1/audio/transcriptions` · `/v1/audio/speech` · `/infill` · `/tokenize` |
| **Ollama** | `/api/chat` · `/api/generate` · `/api/tags` · `/api/ps` · `/api/pull` |
| **pallama-native** | `/api/evict` · `/api/session` + session bank · `/api/keys` · `/api/why` · `/api/watch` · `/.well-known/pallama` capability discovery |

Full route table with payloads and error codes: [docs/4.API_SPEC](docs/4.API_SPEC.md).

## The power-tool CLI

The advanced surface, grouped by job. `--json`/JSONL output on the inspection commands (`list`, `ps`, `show`, `fit`, `doctor`) keeps them script-friendly.

**Models & weights**

| Command | What it does |
|---|---|
| `pallama pull owner/repo:QUANT` \| `shortname` | Hugging Face or registry.ollama.ai, sha256-verified, resumable, multi-shard native |
| `pallama import model.gguf --name x` | Register a GGUF already on disk — hardlinked in place, zero copy |
| `pallama quantize -t Q4_K_M [--imatrix calib.txt]` | Derive new quants locally with the engine's own llama-quantize; imatrix calibration for better Q4 accuracy |
| `pallama create` | Parameter aliases from a Modelfile (`FROM` + `PARAMETER`) — zero-copy, no 30–60 GB blob duplication |
| `pallama lora` / `pallama mmproj` | LoRA adapter management; request `model+adapter` to spawn a variant beside the dense model; attach a vision projector to any model |
| `pallama search --format <tag>` | Search Hugging Face across every weight format |
| `pallama fit [--json]` | VRAM fit + quant alternatives *before* downloading |
| `pallama coreside` | Co-residency plan: which local models fit in VRAM together (weights + f16 KV at each model's ctx) |

**Performance**

| Command | What it does |
|---|---|
| `pallama tune --search` | Measured launch profile: grid argmax over real runs; live probes for slots/n-gram/load tuning |
| `pallama bench` | llama-bench runner for measured, comparable numbers |
| `pallama drafts <model>` | Speculative-decoding candidates — EAGLE3/MTP heads plus small same-family models; steer per request with `options.spec` or `--no-draft` |

**Operations & safety**

| Command | What it does |
|---|---|
| `pallama engine update/use/rollback/build` | sha256-verified engines, side-by-side, regression-gated; CUDA source builds through the same flow |
| `pallama engine build --fork owner/llama.cpp@<sha>` | Temporary capability lane for GGUF archs mainline can't load yet: immutable SHA pin, architecture set mined from the fork's source, provenance in `engine list`; a model that dies on `unknown model architecture` re-routes to an installed advertising lane exactly once and remembers the pin |
| `pallama engine offers` + `engine install --lane <id>` | Curated capability-lane registry: catalog of community fork lanes per missing architecture (`--arch`, `--json`); one-command curated build with auto-retire — when every architecture a lane serves ships upstream, the lane is marked superseded (pins auto-clear, `engine list` shows it) and curated lanes are removed after `fork_retire_days` (user-built forks and pinned lanes are never auto-deleted) |
| `pallama upgrade [--dry-run]` | Self-update the binary from GitHub Releases, sha256-verified |
| `pallama keys` | API key lifecycle — list / add / rm / rotate against the daemon |
| `pallama launch --warm <model> -- <cli>` | Pre-warm a model, exec a CLI, hand it a gateway key |
| `pallama session save/restore` | Slot KV checkpoints that survive unload and daemon restarts |
| `pallama snapshot` | Timestamped backup of config + store + sessions manifest (older ones auto-pruned) |
| `pallama doctor` / `pallama why` / `pallama watch` | One-table diagnosis; post-hoc trace answers; live tail of sentinel detections |
| `pallama whisper` | Audio transcription (wav/mp3/flac/…; `--install`/`--pull`/`--list` manage the model) |
| `pallama tts` | Offline speech synthesis (piper): `--install` the engine, `--pull <voice>` from the voice catalog, `--list` voices; WAV to file or stdout |

Plus `cp`/`rm`/`stop` for model housekeeping, `migrate` (config migration with timestamped backup), and `completions <shell>`.

## Every ollama complaint, fixed at the root

Every row is a real, long-standing ollama complaint with pallama's root-cause resolution. No workarounds — different architecture.

**Engines & speed**

| ollama failure | pallama resolution |
|---|---|
| Vendored engine fork lags upstream; breaks new models | Zero fork. Official upstream binaries, sha256-verified, side-by-side installs, `engine update/use/rollback`. New architectures land when upstream ships them |
| Best backend not published as a prebuilt (Linux CUDA) | `pallama engine build cuda` compiles the in-tree ggml-cuda backend from an upstream tag: auto GPU-arch detection, auto CUDA host-compiler matching, full binary set (`llama-server/-quantize/-imatrix/-perplexity/-bench`), installed as `bNNNN-cuda` through the same probe → regression-gate → activate flow as release assets |
| Slower than llama.cpp (1.8x reports) | No inference reimplementation; byte-stream proxy + measured profiles (`tune --search`) |

**Models & registry**

| ollama failure | pallama resolution |
|---|---|
| Modelfile copies 30–60 GB to change one parameter; silent `{{ .Prompt }}` fallback | No blob copies. `pallama create` aliases `FROM` + `PARAMETER` zero-copy; child always runs `--jinja` (embedded template); params per-request or per-model config overlay |
| Hashed blob lock-in; limited quants | Plain `.gguf` files at `~/.local/share/pallama/models/` — any tool can use them; any HF quant |
| No multi-shard GGUF | Shard sets pulled natively (`-0000N-of-0000M`), first shard launched |
| Registry bottleneck; misleading names | Direct HF pull **and** registry.ollama.ai pull (shortnames route to the ollama registry; Docker-v2 manifest wire, sha256-verified resumable blobs via 307→presigned-CDN with allowlisted redirects, token only ever on first-party registry host — `PALLAMA_REGISTRY_TOKEN` for private namespaces). HF names = actual repo name, never a marketing alias |

**API correctness**

| ollama failure | pallama resolution |
|---|---|
| Opaque 2048 default ctx; silent truncation | Default ctx 16384, shown in `ps`; overflow errors surfaced verbatim with fix hints |
| `keep_alive` confusion | Honored in both APIs: `N` pins N s, `0` evicts after the request, `-1` pins forever; `/api/ps` shows the countdown |
| Per-request `num_ctx` silently ignored | Honored: instance restarts once at the requested size, or a 400 with the max-fitting ctx — never silent truncation. Under unified KV the pin is budget-validated *before* any spawn; a truly unhostable pin is a single teaching 400 naming all four levers (lower `num_ctx` / smaller quant via `pallama fit` / `cache_type = "q8_0"` / `kv_unified = false`) — never a spawn-die-retry storm of 502s |
| "API returns 200 but nothing useful happens"; silent truncation; plain-text instead of tool calls; agents loop on malformed tool args | sentinel: warn-only semantic observation on every chat request — `finish_reason: length` with fix hints, tool-arg JSON + hallucinated-name + parameters-schema checks, `json_schema`/`json_object` validation, empty-response and reasoning-with-no-answer detection, stalled-stream detection, and a pre-inference template-capability check (`x-pallama-warnings` header). `pallama why [trace]` answers any request after the fact |

**Operations & diagnostics**

| ollama failure | pallama resolution |
|---|---|
| Hidden concurrency | `ps` shows slots/ctx/ports/in-flight/spec/cache-hit; loaders stream `X-Pallama-Status: loading`; explicit priority queue (`X-Pallama-Priority`) |
| No session/context persistence | `pallama session save/restore`: slot KV checkpoints that survive unload and daemon restarts |
| No diagnostics when things break | `pallama doctor`: config (incl. stale pins), port conflicts (incl. the ollama-11434 class), engine, hardware, disk, model health (model-dir orphans vs the store, hardlink twins flagged as zero-space), update-currency in one table |
| No discovery; env sprawl | `pallama search` (HF, every weight format via `--format`); every knob in one documented `config.toml`, inspectable via `pallama config` |
| Unbounded cache RAM; blind pulls | `cache_ram_mb` config (auto-capped at 30% of physical RAM); `pallama fit [--json]` previews VRAM fit + quant alternatives *before* downloading — for safetensors repos one aggregate row (full shard set) with the sglang/mistralrs lane hint |
| No KV-quant control; VRAM cliffs | capacity-math ladder: f16 -> q8_0 (KV/2) -> q4_0 (KV/4) at 90% VRAM, or pin any type via `cache_type`; visible in `pallama show` warnings |

**Privacy, security & trust**

| ollama failure | pallama resolution |
|---|---|
| CVE-2025-51471 token exfiltration class | HTTPS to HF only; redirect-host allowlist; token sent only to `huggingface.co` (verified in tests) |
| Cloud pivot; silent off-machine routing | Hard local-only; stated in `--help` and at startup |
| Attribution dodging | Version, banner, and this README credit llama.cpp/ggml/ggerganov |

## How it works

### Engine routing

One active engine serves at a time (`pallama engine use`), but models come in formats engines digest differently. Routing is on by default and picks the lane per model at spawn time:

| Model format | Lane |
|---|---|
| Quantized safetensors (AWQ/GPTQ/FP8) | sglang only |
| GGUF | llama.cpp (mistral.rs as alternate) |
| Plain safetensors | sglang or mistral.rs per `policy` (quality / latency / throughput) |

`mode = "manual"` keeps one engine for everything (byte-identical to pre-routing behavior); a per-model `[model_overrides] engine = ...` pin wins over both modes. Both API dialects route identically; `pallama list`, `/v1/models`, and `/api/tags` all show the resolved engine per model; nothing-can-serve is a teaching error, never a guess. Details + decision table: [docs/2.ARCHITECTURE](docs/2.ARCHITECTURE.md) and [docs/10.SCIENTIFIC](docs/10.SCIENTIFIC.md).

### Architecture

```
pallama-core      pure domain: gguf parser, config, store (SQLite), catalog,
                  capability-driven profile compiler (manifest-gated)
pallama-runtime   tokio: HF client (token-isolated), engine installer+prober,
                  process supervisor (ladder: active→sleep→evict; crash circuit),
                  event bus, llama-bench runner/tuner
pallama-gateway   axum: byte-stream OpenAI proxy, ollama-compat translation,
                  priority admission queue, metrics merge, SSE events
pallama-cli       the `pallama` binary: clap commands, REPL, auto-start
```

### Design principles

| Principle | What it buys you |
|---|---|
| **Capability manifest** | Engines are probed after install; the profile compiler emits only flags the installed build supports — engine drift becomes a data problem, not an outage |
| **Prebuilt CUDA, straight from upstream** | Official llama.cpp ubuntu-cuda assets install automatically on Linux-NVIDIA boxes (same-release first, scan-back for the newest that ships one); `PALLAMA_ENGINE_REPO` opts into a self-hosted overlay for sm-slim builds |
| **Bytes-based VRAM admission** | Heterogeneous models co-reside by actual bytes, not a count heuristic |
| **Eviction ladder** | Child-native sleep at `idle_sleep_secs` → SIGTERM at `idle_timeout_secs`; nothing burns VRAM forever, nothing dies mid-request |
| **Zero-tax proxy** | OpenAI traffic forwarded byte-for-byte; client disconnect aborts the upstream request and frees the slot |
| **Single-pid signals only** | pallama never signals a process group — an errant kill can never take down your shell session or unrelated children |

Full walkthrough: [docs/2.ARCHITECTURE](docs/2.ARCHITECTURE.md).

## Quality

| Gate | Standing |
|---|---|
| Tests | **1099** — compiler tables, wiremock network suites, engine install cycles with a real stub engine, supervisor lifecycle integration, full gateway round-trips over both APIs incl. sentinel suites |
| Lint | `cargo clippy --workspace --all-targets -- -D warnings` clean |
| Live E2E | Real-engine harnesses: `scripts/validate.py`, `scripts/bench_matrix.py` |
| CI | The suite plus shellcheck, ruff, installer e2e on x86_64 **and** arm64 Linux, dependency CVE audit, repo hygiene gates |

Details: [docs/7.SETUP](docs/7.SETUP.md).

## Documentation

The complete manual lives in [`docs/`](docs/):

| Doc | Contents |
|---|---|
| [1.SYSTEM_OVERVIEW](docs/1.SYSTEM_OVERVIEW.md) | problem, users, workflows, tech stack |
| [2.ARCHITECTURE](docs/2.ARCHITECTURE.md) | crates, request lifecycle, supervisor, queue, failure modes |
| [3.DATA_MODEL](docs/3.DATA_MODEL.md) | SQLite store schema, indexes, invariants |
| [4.API_SPEC](docs/4.API_SPEC.md) | every HTTP route + every CLI command (payloads, errors, exit codes) |
| [6.BUSINESS_RULES](docs/6.BUSINESS_RULES.md) | limits, validation, routing + eviction policy |
| [7.SETUP](docs/7.SETUP.md) | env vars, dev setup, build, deploy, full config reference |
| [8.DO_NOT_BREAK](docs/8.DO_NOT_BREAK.md) | invariants + intentional quirks maintainers must preserve |
| [9.USAGE](docs/9.USAGE.md) | end-user tasks, quickstart, troubleshooting, FAQ |
| [10.SCIENTIFIC](docs/10.SCIENTIFIC.md) | KV/VRAM math, GGUF metadata parsing, routing evidence |

## Credit

Pallama is an orchestrator: inference is upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) (ggml, ggerganov and hundreds of contributors), [mistral.rs](https://github.com/EricLBuehler/mistral.rs) and [SGLang](https://github.com/sgl-project/sglang) — the engine authors did the hard parts. Models come from their publishers on Hugging Face.

## License

MIT OR Apache-2.0
