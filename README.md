# pallama

**llama.cpp orchestration: ollama-grade UX, zero engine fork.**

One Rust binary — `pallama` — that wraps upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server`: official binaries, side-by-side versions, atomic switching, and a byte-stream gateway exposing both the **OpenAI** and **Ollama** APIs on port **11434** (drop-in `OLLAMA_HOST` replacement).

Local-only by design: **no telemetry, no cloud endpoints**; the only outbound traffic is user-initiated engine/model downloads. Powered by llama.cpp / ggml / ggerganov.

## Install

Prebuilt binaries for Linux (x86_64/aarch64, glibc ≥ 2.35 or static musl), macOS (x86_64/Apple Silicon), and Windows (x64) are published on GitHub Releases, sha256-verified against the release metadata by the installers (the same mechanism as `pallama engine update`).

**Linux / macOS (WSL included):**

```sh
# from a published repo (set PALLAMA_REPO to the owner/name that hosts releases):
export PALLAMA_REPO=owner/pallama
curl --proto '=https' --tlsv1.2 -fsSL \
  "https://raw.githubusercontent.com/${PALLAMA_REPO%%/*}/pallama/main/scripts/install.sh" | sh

# from a source checkout (zero arguments: builds the checkout fresh with
sudo sh scripts/install.sh

# or force the source path explicitly (no release channel contact):
sudo sh scripts/install.sh --build
```

Like ollama's installer: **system-wide only** — root-owned binary in `/usr/local/bin` plus a systemd unit (`Restart=always`, GPU groups, auto-start, restart-on-upgrade). There is deliberately no user-path (`~/.local/bin`) install mode: a second copy there is how stale-binary daemon races happen (`pallama doctor` flags any that already exist, and the installer removes one it finds). Root or sudo is required.

From a checkout the installer compiles fresh with `cargo` first (never a stale `target/release`); a checkout-less `curl | sh` uses the sha256-verified release channel. Older glibc than 2.35, or Alpine? Automatic fallback to the static musl build. Pin a version with `PALLAMA_VERSION=v0.3.0`, a mirror with `PALLAMA_INSTALL_BASE_URL`, or a build repo with `PALLAMA_CHECKOUT`.

**Windows (PowerShell):**

```powershell
$env:PALLAMA_REPO = 'owner/pallama'; irm https://raw.githubusercontent.com/owner/pallama/main/scripts/install.ps1 | iex
```

Installs to `%LOCALAPPDATA%\Programs\pallama` and adds it to the user PATH.

**From source:** `sudo cargo install --path crates/pallama-cli` (recent stable Rust) — or just run the installer from the checkout.

**Uninstall:** `sudo sh scripts/install.sh --uninstall` (removes binary + units; your models and config under `~/.local/share/pallama` / `~/.config/pallama` are user data and stay until you delete them).

## Quickstart

```sh
pallama engine update                     # install + activate upstream llama-server (sha256-verified;
                                           #   regression-gated: >10% decode drop auto-rolls-back; --no-gate skips)
pallama pull qwen3-0.6b                   # or any owner/repo:QUANT from Hugging Face
pallama run qwen3-0.6b                    # streaming REPL
OLLAMA_HOST=http://127.0.0.1:11434 ollama list   # existing ollama clients just work
```

Capability discovery: `GET /.well-known/pallama` (routes, headers, features, engine identity). OpenAI-compatible: `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/rerank`, `/v1/messages` (Anthropic), `/v1/responses` (Responses API), `/v1/audio/transcriptions` (multipart, audio-capable models), `/infill` (FIM), `/v1/chat/completions/control` (steering vectors), `/v1/chat/completions/input_tokens`, `/v1/responses/input_tokens`, `/v1/messages/count_tokens`, `/tokenize`, `/detokenize`, `/apply-template`, `/v1/adapters` (LoRA). Ollama-compatible: `/api/chat`, `/api/generate`, `/api/tags`, `/api/pull`, `/api/ps`, `/api/show`, `/api/embeddings`, `/api/events`. Pallama-native: `/api/evict`, `/api/session` (slot KV checkpoints + `_auto` session bank: KV survives eviction, restored on respawn), `/api/keys` (virtual-key CRUD), `GET /v1/responses/{id}` (stored-response retrieval), `/api/why` (sentinel record ring), `/api/watch` (live SSE sentinel tail), `X-Pallama-Num-Ctx` request header (per-request ctx on the OpenAI path — the protocol has no such field; same restart-once semantics as `options.num_ctx`), `X-Pallama-Enforce` header (agent loops: 422 on malformed tool calls / schema violations for non-stream requests), `X-Pallama-Deadline-Ms` header (SLO: EDF queue ordering; prefill-heavy requests auto-demote one class). Identical non-stream requests single-flight (duplicate waits, then rides the leader's warm prefix). Sentinel observes chat-completions, legacy completions, and `/v1/responses` (both stream and non-stream) on both APIs.

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
| `ps [--reset]` | Live instances: state, ctx, **GPU offload** (`full`/`partial`/`cpu`/`auto` — silent CPU fallback is never silent), in-flight, endpoint |
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
| `keys [list]/add/rm/rotate NAME [--models a,b] [--rpm N] [--tpm N] [--daily-tokens N] [--max-concurrent N]` | Virtual API keys: model scoping (wildcards: `qwen3*`, `vllm:*`), rate limits, daily token budgets, per-key concurrency cap (leased at auth, held while the response streams), live usage, secret rotation (`[[keys]]` in config; `x-api-key` works for Anthropic clients; SIGHUP hot-reloads the key set) |
| `quantize <model> -t Q4_K_M [--imatrix calib.txt] [--name N]` | Derive a new quant with the engine's own llama-quantize; `--imatrix` calibrates an importance matrix first (F16 sources; better Q4 accuracy) |
| `launch [--warm MODEL] [--key K] <cli> [args...]` | Point any OpenAI/Anthropic/ollama-speaking CLI at pallama (env wired, optional pre-warm) |
| `whisper [--install] [--pull SIZE] [--list] [<file>]` | Managed STT lane: installs whisper.cpp server from its releases, pulls ggml models (`tiny`…`large-v3-turbo`), transcribes via the lazy local server on `/v1/audio/transcriptions`; `<remote>:model` still forwards |
| `mmproj <model> <path>` | Attach a vision projector GGUF to an existing model (refuses while running; next spawn gets vision); `import --mmproj <path>` attaches at import time |
| `tune <model> --slots C` / `--ngram` / `--load` / `--replicas` | Live-concurrency `-np` search (adopts on a >5% win, records the engine-update regression baseline); n-gram grid; warmup A/B (adopts `warmup = false` only if cold load + first token is >5% faster); replica scaling probe (8 clients x 3 gens against 1 vs 2 identical children, adopts `replicas = 2` only if >1.3x aggregate) |
| `replicas = N` (overlay) | Parallel instances of one model (1..=8): distinct conversation prefixes each get their own warm-cache child (prefix-hash sticky routing); `ps` shows `model#N` |
| `coreside` | VRAM co-residency plan: which local models fit together (weights + f16 KV @ ctx, hot-first greedy) |
| `drafts <model>` | Speculative-decoding draft candidates (EAGLE3/MTP heads + small same-family) from live HF search |
| `completions <bash\|zsh\|fish\|powershell>` | Shell completions to stdout |
| `snapshot` | Backup config + store to `<data>/snapshots/<ts>/` (models/engines stay: bulk) |
| `[[remotes]]` (config) | Remote OpenAI-compatible engines (vLLM / MLX / another pallama): `model = "name:model"` routes there; `ps` probes health |

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
engine_asset = "auto"     # auto picks by OS/GPU (versioned cuda/rocm = newest); or explicit: "ubuntu-vulkan-x64", ...
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
# kv_unified = true       # unset = auto (emits --kv-unified when slots > 1):
                          #   one shared KV buffer = prefix reuse across sequences
                          #   (the cheap radix). false = --no-kv-unified. Per-model
                          #   [model_overrides] kv_unified exists too.
kv_unified_per_slot = 0   # per-slot token budget inside the unified buffer (0 = off)
swa_full = false          # keep FULL KV for sliding-window layers (quality at memory cost)
ctx_checkpoints = 0       # rolling context checkpoints for SWA models
no_kv_offload = false     # keep all KV on GPU; fail instead of spilling to CPU
load_mode = ""            # "" | mmap | mlock | direct-io
spawn_mem_guard = true    # refuse loads when MemAvailable < model/2 + 512 MiB
                          #   (swap-death prevention; set false to load anyway)
session_bank = true       # KV checkpoints (_auto) survive eviction; restored on respawn
prompt_preflight = true   # refuse prompts that cannot fit ctx (exact /tokenize over
                          #   the 90% estimate threshold) — the engine would silently
                          #   truncate; set false to allow truncation (sentinel still warns)
singleflight = true       # identical non-stream chat requests coalesce (5s bound);
                          #   duplicates ride the leader's warm prefix
spec_cache = true         # persist n-gram speculation cache across restarts (spec = "ngram")
sessions = true           # slot KV checkpoints: `pallama session save/restore` (engine-gated)
ctx_extend = 0.0          # YaRN context extension factor; 0 = off. >1.0..=32.0; quality tradeoff
cpu_moe_n = 0             # keep N MoE experts on CPU (finer than the auto heuristic)
override_tensor = []      # upstream --override-tensor entries, e.g. [".ffn_.*_exps.=CPU"]
devices = []              # explicit --device selection for multi-GPU boxes (names from `pallama doctor`); empty = engine auto
engine_check_secs = 86400 # background llama.cpp release check (0 = off); result in `doctor`/`ps` — never auto-installs
agent = false             # child --agent: built-in tools + MCP proxy. WARNING: exec_shell_command
sentinel = true           # warn-only response-semantics observation: truncation, tool-call validity,
                         #   schema violations, empty replies, stalls -> `pallama why` (never alters bytes)
sentinel_stall_secs = 30 # stalled-stream threshold (0 = off; 5..=600)
sentinel_enforce = false  # opt-in hard mode: invalid tool args / unknown tools / schema violations
                          #   -> 422 on NON-STREAM chat requests (per-request X-Pallama-Enforce: 1|0 overrides;
                          #   streaming stays warn-only — bytes are already on the wire). Records persist
                          #   across restarts (run/sentinel.jsonl, bounded)
tls_cert = ""             # PEM chain + key pair enables HTTPS at the gateway (both-or-neither;
tls_key = ""              #   validated). Empty = plain HTTP (loopback default)
cors_origins = []         # e.g. ["https://chat.example"] or ["*"]; empty = no CORS headers
otlp_endpoint = ""        # OTLP collector URL (e.g. "http://127.0.0.1:4318"): one span per
                          #   request, batched, bounded. Empty = off
otlp_service = "pallama"  # service.name on exported spans

[[keys]]                  # virtual API keys (empty set = authless loopback)
name = "ci"
key = "plm_..."           # `pallama keys add` generates + shows once
models = ["qwen3.5-9b"]   # empty = all models (admin: manages keys via /api/keys)
rpm = 0                   # requests/minute (0 = unlimited)
tpm = 0                   # tokens/minute, counted from response usage
daily_tokens = 0          # total tokens per UTC day
max_concurrent = 0        # parallel in-flight requests (0 = unlimited); leased at
                          #   auth, held until the response body drains — a held
                          #   SSE stream occupies a slot, 429 + Retry-After beyond

[[remotes]]               # external OpenAI-compatible engines (vLLM, MLX, another pallama)
name = "vllm"             # requests with model = "vllm:<id>" route there
url = "http://10.0.0.4:8000"
key = ""                  # optional bearer for the remote

[model_overrides."qwen3-coder-30b"]   # per-model overlay; unknown keys are errors
ctx = 32768
# Overlay keys: ctx, spec, loras, extra_args, cache_type, kv_unified,
# ctx_extend, cpu_moe_n, override_tensor, devices, warmup,
# reasoning_budget, reasoning_effort, replicas

# ---- wire-everything knobs (all default = engine defaults; unset emits nothing)

# spec-draft placement (spec = "auto" with a pulled draft)
# spec_draft_cpu_range = ""       # "lo-hi" CPU set for the draft model
# spec_draft_cpu_strict = false   # strict placement
# spec_draft_device = ""          # dedicated GPU for the draft
# spec_draft_ngl = ""             # draft VRAM layers: N | "auto" | "all"
# spec_draft_threads = 0          # draft thread count
# spec_draft_p_min = 0.0          # min draft probability (greedy accept)
# spec_draft_p_split = 0.10       # split probability
# spec_draft_poll = 0             # draft poll level 0..=100
# spec_draft_prio = 0             # draft priority -1..=3
# spec_draft_prio_batch = 0       # draft batch priority -1..=3
# spec_draft_poll_batch = true    # poll for draft batch work
# spec_draft_cpu_strict_batch = false
# spec_draft_threads_batch = 0
# spec_draft_type_k = ""          # draft KV cache type
# spec_draft_type_v = ""          # draft vocab cache type
# spec_draft_override_tensor = [] # e.g. "exps=CPU"
# spec_draft_n_cpu_moe = 0        # draft MoE experts on CPU
# spec_draft_cpu_moe = false      # all draft experts on CPU
# spec_draft_backend_sampling = true # false = --no-spec-draft-backend-sampling

# n-gram tuning (spec = "ngram"); grid-search with `pallama tune <model> --ngram`
# ngram_size_m = 0                # lookup table size (0 = engine default)
# ngram_size_n = 0                # n-gram length
# ngram_min_hits = 0              # min hits before trusting a draft
# ngram_adaptive_decay = 0        # adaptive spec decay step (spec = "adaptive")
# ngram_adaptive_target = 0.0     # target acceptance rate 0..=1

# reasoning control (cuts wasted thinking tokens on agent traffic)
# reasoning_budget = -1           # -1 unrestricted | 0 end now | N token budget
# reasoning_budget_message = ""   # injected when the budget runs out
# reasoning_effort = ""           # "" | minimal|low|medium|high|xhigh|max
# reasoning_preserve = true       # keep reasoning content in responses

# vision / multimodal (models with an mmproj)
# image_max_tokens = 0            # per-image token ceiling (0 = model)
# image_min_tokens = 0
# mtmd_batch_max_tokens = 0       # image tokens per encode batch
# mmproj_offload = true           # false = keep projector on CPU
# mmproj_auto = true              # false = no projector auto-discovery
# mmproj_device = ""              # dedicated projector device
# embd_normalize = 0              # 1 = L2-normalize embeddings

# YaRN fine-tuning (beyond ctx_extend)
# yarn_orig_ctx = 0
# yarn_ext_factor = -1.0          # >= 0 to set
# yarn_attn_factor = 0.0
# yarn_beta_fast = 0.0
# yarn_beta_slow = 0.0

# scheduling extras
# cpu_strict = false              # strict CPU placement (--cpu-strict 1)
# prio = 0                        # process priority -1..=3
# prio_batch = 0                  # batch-thread priority
# poll_batch = true               # poll for batch work (follows `poll` by default)
# threads_http = 0                # HTTP server threads
# numa = ""                       # "" | distribute | isolate
# check_tensors = false           # verify tensor data on load

# engine behavior toggles (defaults mirror upstream; knobs exist to disable)
# warmup = true                   # false = --no-warmup (faster cold load)
# repack = true                   # weight repacking
# no_host = false                 # bypass host buffer (tight VRAM)
# op_offload = true               # host tensor ops to device
# keep_tokens = 0                 # tokens kept on ctx shift (-1 = all)
# context_shift = false           # opt-in: upstream default is DISABLED
# samplers = ""                   # "temp;top_k;..." ordering + selection

# video input (vision models with ffmpeg support)
# video_ffmpeg_dir = ""           # "" = system ffmpeg
# video_fps = 0.0                 # sampling rate
# video_timestamp_interval = 0.0  # timestamp cadence

# power-user escapes
# override_kv = []                # "KEY=TYPE:VALUE" GGUF metadata repairs
# control_vectors = []            # control vector files
# control_vectors_scaled = []     # "FNAME:SCALE"
# control_vector_layer_range = ""
# tensor_preset = ""              # "" | "moe-cpu-offload" (experts -> CPU)
# pii_scrub = false               # redact emails/secrets/IPs from why/watch
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

Live harnesses (real engine, real model, no mocks):

- `scripts/validate.py` — exhaustive E2E validation: every config knob traced from `config.toml` through the profile compiler to the engine child's actual `/proc` argv, both API surfaces, sentinel, lifecycle behaviors (crash-respawn, eviction, cancellation), CLI, auth — in an isolated XDG sandbox (`--fast` smoke mode, `--phase` filters).
- `scripts/bench_compare.py` — head-to-head benchmark vs **direct llama-server** (launched with the argv cloned from pallama's own child, so the only delta is orchestration) and **ollama** (live service, API-only), plus `llama-bench` as the engine ceiling. Measures cold load, TTFT, decode/prefill tok/s, RSS/VRAM, and times **every feature** (tool calls, structured output, vision, embeddings, sessions, queue behavior, watch/why, config variants incl. router mode) with a route-capability matrix across all three servers. Same isolation contract: never touches port 11434.

## Credit

Pallama is an orchestrator: all inference is upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) — ggml, ggerganov and hundreds of contributors did the hard parts. Models come from their publishers on Hugging Face.
