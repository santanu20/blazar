# pallama

**llama.cpp orchestration: ollama-grade UX, zero engine fork.**

One Rust binary — `pallama` — that wraps upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server`: official binaries, side-by-side versions, atomic switching, and a byte-stream gateway exposing both the **OpenAI** and **Ollama** APIs on port **11434** (drop-in `OLLAMA_HOST` replacement).

Local-only by design: **no telemetry, no cloud endpoints**; the only outbound traffic is user-initiated engine/model downloads. Powered by llama.cpp / ggml / ggerganov.

## Quickstart

```sh
pallama engine update                     # install + activate upstream llama-server (sha256-verified)
pallama pull qwen3-0.6b                   # or any owner/repo:QUANT from Hugging Face
pallama run qwen3-0.6b                    # streaming REPL
OLLAMA_HOST=http://127.0.0.1:11434 ollama list   # existing ollama clients just work
```

OpenAI-compatible: `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/rerank`, `/v1/messages` (Anthropic), `/v1/adapters` (LoRA). Ollama-compatible: `/api/chat`, `/api/generate`, `/api/tags`, `/api/pull`, `/api/ps`, `/api/show`, `/api/embeddings`, `/api/events`.

## Why (ollama complaints → pallama resolutions)

| ollama failure | pallama resolution |
|---|---|
| Vendored engine fork lags upstream; breaks new models | Zero fork. Official upstream binaries, sha256-verified, side-by-side installs, `engine update/use/rollback`. New architectures land when upstream ships them |
| Slower than llama.cpp (1.8× reports) | No inference reimplementation; byte-stream proxy + measured profiles (`tune --search`) |
| Modelfile copies 30–60 GB to change one parameter; silent `{{ .Prompt }}` fallback | No Modelfile. GGUF is self-contained truth; child always runs `--jinja` (embedded template); params per-request or per-model config overlay |
| Hashed blob lock-in; limited quants | Plain `.gguf` files at `~/.local/share/pallama/models/` — any tool can use them; any HF quant |
| No multi-shard GGUF | Shard sets pulled natively (`-0000N-of-0000M`), first shard launched |
| Registry bottleneck; misleading names | Direct HF pull; name = actual repo name, never a marketing alias |
| CVE-2025-51471 token exfiltration class | HTTPS to HF only; redirect-host allowlist; token sent only to `huggingface.co` (verified in tests) |
| Cloud pivot; silent off-machine routing | Hard local-only; stated in `--help` and at startup |
| Opaque 2048 default ctx; silent truncation | Default ctx 16384, shown in `ps`; overflow errors surfaced verbatim with fix hints |
| Hidden concurrency | `ps` shows slots/ctx/ports/in-flight; loaders stream `X-Pallama-Status: loading`; explicit priority queue (`X-Pallama-Priority`) |
| Attribution dodging | Version, banner, and this README credit llama.cpp/ggml/ggerganov |
| `keep_alive` confusion | Honored in both APIs: `N` pins N s, `0` evicts after the request, `-1` pins forever; `/api/ps` shows the countdown |
| Per-request `num_ctx` silently ignored | Honored: instance restarts once at the requested size, or a 400 with the max-fitting ctx — never silent truncation |
| Unbounded cache RAM; blind pulls | `cache_ram_mb` config; `pallama fit` previews VRAM fit + quant alternatives *before* downloading |
| No discovery; env sprawl | `pallama search` (HF GGUF); every knob in one documented `config.toml`, inspectable via `pallama config` |

## Commands

| Command | Purpose |
|---|---|
| `serve` / `stop` | Daemon lifecycle (pidfile-guarded, graceful shutdown) |
| `pull` / `rm` / `list` / `show` | Model store (plain GGUF files; rm refuses while running) |
| `run <model>` | Streaming REPL (`/exit /clear /model /sysinfo /profile`) |
| `ps [--reset]` | Live instances: state, ctx, in-flight, endpoint |
| `bench` / `tune --search` | `llama-bench` tables; measured argmax profile adoption |
| `engine list/update/use/rollback` | Upstream engine management with capability manifests |
| `lora add/rm/list` | LoRA adapters served via `--lora`/`--lora-scaled` |
| `search <query>` | HF GGUF search (downloads, likes, sizes) |
| `fit <repo[:quant]>` | Pre-download VRAM fit preview + quant alternatives |
| `config list/get/set` | One knob surface (validates on set) |

Daemon-dependent commands auto-start `pallama serve` (detached, logs at `~/.local/share/pallama/run/daemon.log`).

## Configuration

`~/.config/pallama/config.toml` — created with defaults on first run; `PALLAMA_*` env vars override; secrets (`HF_TOKEN`, `GH_TOKEN`) are env-only, never stored.

```toml
host = "127.0.0.1"
port = 11434              # ollama port: drop-in replacement
default_ctx = 16384
idle_sleep_secs = 300     # child-native GPU sleep (frees VRAM, warm wake)
idle_timeout_secs = 1800  # process eviction after idle
max_loaded_models = 0     # 0 = auto from VRAM / model size
child_transport = "tcp"   # "tcp" (curl-debuggable) | "unix"
engine_asset = "auto"     # or explicit: "ubuntu-vulkan-x64", "win-cuda-13.3-x64", ...
engine_pin = ""           # "" = newest b-tag; else e.g. "b10816"
spec = "off"              # "auto" adopts speculative decoding when a draft pair is pulled
cache_reuse = 256         # prefix-cache chunk reuse (0 disables)
api_keys = []             # non-empty = Bearer auth at the gateway (children stay loopback)
rpc_servers = ""          # e.g. "box1:50052,box2:50052" -> --rpc
cache_ram_mb = 8192       # child prompt-cache budget (0 = unlimited)

[model_overrides."qwen3-coder-30b"]   # per-model overlay; unknown keys are errors
ctx = 32768
```

## Architecture

```
pallama-core      pure domain: gguf parser, config, store (SQLite), catalog,
                  capability-driven profile compiler (12 rules, manifest-gated)
pallama-runtime   tokio: HF client (token-isolated), engine installer+prober,
                  process supervisor (ladder: active→sleep→evict; crash circuit),
                  event bus, llama-bench runner/tuner
pallama-gateway   axum: byte-stream OpenAI proxy, ollama-compat translation,
                  priority admission queue, metrics merge, SSE events
pallama-cli       the `pallama` binary: clap commands, REPL, auto-start
```

Load-bearing ideas:

- **Capability manifest** — after install, the engine is probed (`--version`, `--list-devices`, `--help`). The profile compiler emits *only* flags the installed build actually supports; a missing flag is a hard error naming it and suggesting `engine use <tag>`. Engine drift becomes a data problem, not a code problem.
- **Eviction ladder** — active → child-native sleep at `idle_sleep_secs` (VRAM freed, instant wake) → SIGTERM at `idle_timeout_secs`. Nothing burns VRAM forever; nothing dies mid-request.
- **Zero-tax proxy** — OpenAI traffic is forwarded byte-for-byte (no body parsing, no SSE buffering). Tool calls, structured output, logprobs ride through untouched. Client disconnect aborts the upstream request; the slot frees.
- **Signals are single-pid only** — Pallama never signals process groups; every teardown path is audited and idempotent.

## Verification

145 tests: pure compiler tables, wiremock network suites (resume, sha, allowlist, token isolation), engine install cycles with a real stub engine binary, supervisor lifecycle integration (ladder, capacity, crash-circuit, shutdown), and full gateway round-trips over both APIs. `cargo clippy --workspace --all-targets -- -D warnings` clean.

## Credit

Pallama is an orchestrator: all inference is upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) — ggml, ggerganov and hundreds of contributors did the hard parts. Models come from their publishers on Hugging Face.
