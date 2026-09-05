# pallama

**llama.cpp orchestration: ollama-grade UX, zero engine fork.**

One Rust binary — `pallama` — that wraps upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server`: official binaries, side-by-side versions, atomic switching, and a byte-stream gateway exposing both the **OpenAI** and **Ollama** APIs on port **11434** (drop-in `OLLAMA_HOST` replacement).

Local-only by design: **no telemetry, no cloud endpoints**; the only outbound traffic is user-initiated engine/model downloads. Powered by llama.cpp / ggml / ggerganov.

## Install

Prebuilt binaries for Linux (x86_64/aarch64, glibc ≥ 2.35 or static musl), macOS (x86_64/Apple Silicon), and Windows (x64) are published on GitHub Releases, sha256-verified against the release metadata by the installers (the same mechanism as `pallama engine update`).

**Linux / macOS (WSL included):**

```sh
# from a source checkout (zero arguments: the script auto-detects the local
# build and installs it system-wide with a systemd service):
sh scripts/install.sh

# from a published repo (set PALLAMA_REPO to the owner/name that hosts releases):
export PALLAMA_REPO=owner/pallama
curl --proto '=https' --tlsv1.2 -fsSL \
  "https://raw.githubusercontent.com/${PALLAMA_REPO%%/*}/pallama/main/scripts/install.sh" | sh
```

`--system` installs system-wide like ollama: binary in `/usr/local/bin` plus a `systemctl` unit (`Restart=always`, GPU groups, auto-start). Omit it for the sudo-free `~/.local/bin` install.

Installs to `~/.local/bin/pallama` (add it to PATH if needed). Older glibc than 2.35, or Alpine? The script automatically falls back to the static musl build. Pin a version with `PALLAMA_VERSION=v0.1.0`, or point at a fork/mirror with `PALLAMA_REPO=owner/pallama`. Add `--with-systemd-unit` (download the script and run `sh install.sh --with-systemd-unit`) to install a `systemctl --user` service instead of the default on-demand auto-start.

**Windows (PowerShell):**

```powershell
$env:PALLAMA_REPO = 'owner/pallama'; irm https://raw.githubusercontent.com/owner/pallama/main/scripts/install.ps1 | iex
```

Installs to `%LOCALAPPDATA%\Programs\pallama` and adds it to the user PATH.

**From source:** `cargo install --path crates/pallama-cli` (recent stable Rust).

**Uninstall:** `pallama stop; rm -f ~/.local/bin/pallama; rm -rf ~/.local/share/pallama ~/.config/pallama ~/.cache/pallama` (Windows: stop pallama in Task Manager, delete `%LOCALAPPDATA%\Programs\pallama`, remove the data dirs under `%LOCALAPPDATA%`/`%USERPROFILE%`).

## Quickstart

```sh
pallama engine update                     # install + activate upstream llama-server (sha256-verified)
pallama pull qwen3-0.6b                   # or any owner/repo:QUANT from Hugging Face
pallama run qwen3-0.6b                    # streaming REPL
OLLAMA_HOST=http://127.0.0.1:11434 ollama list   # existing ollama clients just work
```

OpenAI-compatible: `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/rerank`, `/v1/messages` (Anthropic), `/v1/responses` (Responses API), `/v1/audio/transcriptions` (multipart, audio-capable models), `/infill` (FIM), `/v1/chat/completions/control` (steering vectors), `/v1/chat/completions/input_tokens`, `/v1/responses/input_tokens`, `/v1/messages/count_tokens`, `/tokenize`, `/detokenize`, `/apply-template`, `/v1/adapters` (LoRA). Ollama-compatible: `/api/chat`, `/api/generate`, `/api/tags`, `/api/pull`, `/api/ps`, `/api/show`, `/api/embeddings`, `/api/events`. Pallama-native: `/api/evict`, `/api/session` (slot KV checkpoints), `/api/why` (sentinel record ring), `/api/watch` (live SSE sentinel tail), `X-Pallama-Num-Ctx` request header (per-request ctx on the OpenAI path — the protocol has no such field; same restart-once semantics as `options.num_ctx`), `X-Pallama-Enforce` header (agent loops: 422 on malformed tool calls / schema violations for non-stream requests). Sentinel observes chat-completions, legacy completions, and `/v1/responses` (both stream and non-stream) on both APIs.

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
| Unbounded cache RAM; blind pulls | `cache_ram_mb` config (auto-capped at 30% of physical RAM — measured live: unclamped 8 GiB budget on a 13 GiB box drove the child to an 8.3 GiB RSS plateau and system-wide swap death); `pallama fit` previews VRAM fit + quant alternatives *before* downloading, incl. the ctx headroom a q8_0 KV cache buys |
| No KV-quant control; VRAM cliffs | capacity-math ladder: f16 -> q8_0 (KV/2) -> q4_0 (KV/4) at 90% VRAM, or pin any type via `cache_type`; visible in `pallama show` warnings |
| No session/context persistence | `pallama session save/restore`: slot KV checkpoints that survive unload and daemon restarts |
| No diagnostics when things break | `pallama doctor`: config, port conflicts (incl. the ollama-11434 class), engine, hardware, disk, model health in one table |
| No discovery; env sprawl | `pallama search` (HF GGUF); every knob in one documented `config.toml`, inspectable via `pallama config` |
| "API returns 200 but nothing useful happens"; silent truncation; plain-text instead of tool calls; agents loop on malformed tool args | sentinel: warn-only semantic observation on every chat request — `finish_reason: length` with fix hints, tool-arg JSON + hallucinated-name + parameters-schema checks, `json_schema`/`json_object` validation, empty-response and reasoning-with-no-answer detection, stalled-stream detection, and a pre-inference template-capability check (`x-pallama-warnings` header) when the model's chat template cannot render tools. `pallama why [trace]` answers any request after the fact |

## Commands

| Command | Purpose |
|---|---|
| `serve` / `stop` | Daemon lifecycle (pidfile-guarded, graceful shutdown) |
| `pull` / `rm` / `list` / `show` | Model store (plain GGUF files; rm refuses while running) |
| `run <model>` | Streaming REPL (`/exit /clear /model /sysinfo /profile`) |
| `ps [--reset]` | Live instances: state, ctx, in-flight, endpoint |
| `bench` / `tune --search` | `llama-bench` tables; measured argmax profile adoption |
| `engine list/update/use/rollback` | Upstream engine management with capability manifests |
| `config list/get/set` | One knob surface (validates on set) |
| `cp <src> <dst>` / `create <name> -f Modelfile` | ollama-parity aliases: zero-byte hardlink + config overlay (no blob copies; unsupported Modelfile keys are named rejections) |
| `run <model> [prompt…]` / `stop <model>` | inline single-shot generation (`--verbose` counts) / unload a model now |
| `push` / `login` family | refused by design: local-only, no registry or cloud accounts |
| `search <query>` | HF GGUF search (downloads, likes, sizes) |
| `session save/restore/rm/list` | slot KV-cache checkpoints: pause a model's context, resume later (survives unload + restart) |
| `doctor` | one-command diagnostics: config, port conflicts, engine, hardware, disk, model health |
| `why [trace]` | sentinel: what the model returned, what was wrong with it (truncation, invalid tool args, schema violations, empty replies, stalls), which knob fixes it |
| `watch` | live tail of sentinel detections as they happen (SSE; Ctrl-C to stop) |
| `router = true` (config) | one child serves ALL models: preset INI auto-generated per model, engine-native autoload + LRU; `pallama stop <model>` becomes an engine unload |
| `upgrade [--version] [--dry-run]` | Self-update from GitHub Releases (sha256-verified, atomic) |

Daemon-dependent commands auto-start `pallama serve` (detached, logs at `~/.local/share/pallama/run/daemon.log`).

## Configuration

`~/.config/pallama/config.toml` — created with defaults on first run; `PALLAMA_*` env vars override; secrets (`HF_TOKEN`, `GH_TOKEN`) are env-only, never stored.

```toml
host = "127.0.0.1"
port = 11435              # default 11434 (drop-in ollama replacement); change it when ollama runs side-by-side on the same box
idle_sleep_secs = 300     # child-native GPU sleep (frees VRAM, warm wake)
idle_timeout_secs = 1800  # process eviction after idle
max_loaded_models = 0     # 0 = auto from VRAM / model size
child_transport = "tcp"   # "tcp" (curl-debuggable) | "unix"
engine_asset = "auto"     # or explicit: "ubuntu-vulkan-x64", "win-cuda-13.3-x64", ...
engine_pin = ""           # "" = newest b-tag; else e.g. "b10816"
spec = "off"              # "auto" = draft-pair speculation when pulled; "ngram" = self-drafting (no draft model)
cache_reuse = 256         # prefix-cache chunk reuse (0 disables)
api_keys = []             # non-empty = Bearer auth at the gateway (children stay loopback)
rpc_servers = ""          # e.g. "box1:50052,box2:50052" -> --rpc
cache_ram_mb = 8192       # child prompt-cache budget; auto-capped at 30% of RAM (0 = unlimited)
cpu_range = ""            # pin child threads to CPUs "lo-hi" (P/E hybrids: pin P-cores, discover via `lscpu -e`)
poll = 0                  # 1..=100 busy-poll waiting for work (trades idle CPU for TTFT)
reasoning_format = ""     # "" auto | "none" | "deepseek" | "deepseek-legacy" thought-tag extraction
router = false            # ONE llama-server serves every model (upstream router: preset INI generated from the store, engine-side autoload + LRU); false = one child per model
router_max_models = 0    # router mode: max concurrently loaded models (0 = upstream default 4)
slot_prompt_similarity = 0.0  # >0 tunes prefix-affinity slot reuse at slots > 1 (upstream default 0.1; 0 = emit nothing)
cache_type = ""           # "" auto ladder (q8_0 -> q4_0 by capacity math) | explicit: f16, q8_0, q4_0, ...
spec_cache = true         # persist n-gram speculation cache across restarts (spec = "ngram")
sessions = true           # slot KV checkpoints: `pallama session save/restore` (engine-gated)
ctx_extend = 0.0          # YaRN context extension factor; 0 = off. >1.0..=32.0; quality tradeoff
cpu_moe_n = 0             # keep N MoE experts on CPU (finer than the auto heuristic)
override_tensor = []      # upstream --override-tensor entries, e.g. [".ffn_.*_exps.=CPU"]
agent = false             # child --agent: built-in tools + MCP proxy. WARNING: exec_shell_command
sentinel = true           # warn-only response-semantics observation: truncation, tool-call validity,
                         #   schema violations, empty replies, stalls -> `pallama why` (never alters bytes)
sentinel_stall_secs = 30 # stalled-stream threshold (0 = off; 5..=600)
sentinel_enforce = false  # opt-in hard mode: invalid tool args / unknown tools / schema violations
                         #   -> 422 on NON-STREAM chat requests (per-request X-Pallama-Enforce: 1|0 overrides;
                         #   streaming stays warn-only — bytes are already on the wire). Records persist
                         #   across restarts (run/sentinel.jsonl, bounded)

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

222 tests: pure compiler tables, wiremock network suites (resume, sha, allowlist, token isolation), engine install cycles with a real stub engine binary, supervisor lifecycle integration (ladder, capacity, crash-circuit, shutdown), and full gateway round-trips over both APIs — including the sentinel suites (9 detection codes, responses grammar, persistence reload, enforce 422s, live watch SSE, parity-under-observation). `cargo clippy --workspace --all-targets -- -D warnings` clean.

## Credit

Pallama is an orchestrator: all inference is upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) — ggml, ggerganov and hundreds of contributors did the hard parts. Models come from their publishers on Hugging Face.
