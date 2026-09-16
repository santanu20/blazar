# pallama

**Multi-engine local inference: llama.cpp, mistral.rs and SGLang orchestrated behind one OpenAI + Ollama + Anthropic gateway.**

One Rust binary — `pallama` — that orchestrates upstream engines unmodified: [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` (GGUF models), [mistral.rs](https://github.com/EricLBuehler/mistral.rs) (HF-style safetensors) and [SGLang](https://github.com/sgl-project/sglang) (safetensors + AWQ/GPTQ on CUDA/ROCm) — official binaries, side-by-side versions, atomic switching, and a byte-stream gateway exposing the **OpenAI**, **Ollama** and **Anthropic** APIs on port **11434** (drop-in `OLLAMA_HOST` replacement).

Local-only by design: **no telemetry, no cloud endpoints**; the only outbound traffic is user-initiated engine/model downloads. Powered by the upstream engines, unmodified.

## Install

Prebuilt binaries for Linux (x86_64/aarch64/armv7, glibc ≥ 2.35 or static musl), macOS (x86_64/Apple Silicon), and Windows (x64/ARM64) are published on GitHub Releases, sha256-verified against the release metadata by the installers (the same mechanism as `pallama engine update`).

**Linux / macOS (WSL included):**

```sh
# from a published repo (set PALLAMA_REPO to the owner/name that hosts releases;
# inside a clone it is derived from the git origin automatically):
export PALLAMA_REPO=owner/pallama
curl --proto '=https' --tlsv1.2 -fsSL \
  "https://raw.githubusercontent.com/${PALLAMA_REPO%%/*}/pallama/master/scripts/install.sh" | sh

# from a source checkout (zero arguments: builds the checkout fresh with
# cargo, then installs system-wide; a missing cc/rust toolchain is
# provisioned first via scripts/bootstrap.sh --minimal — announced, never silent):
sudo sh scripts/install.sh

# or force the source path explicitly (no release channel contact):
sudo sh scripts/install.sh --build
```

**One-click readiness:** after the binary lands, the installer bootstraps the llama.cpp engine (idempotent — skips when an engine is already active) so a fresh install can serve inference immediately; failures are loud warnings, never silent. Opt out with `PALLAMA_INSTALL_ENGINE=0`, pre-pull a model with `PALLAMA_INSTALL_MODEL=<name>`, point engine downloads at a mirror with `PALLAMA_GH_BASE`, or cap the daemon cgroup with `PALLAMA_UNIT_MEMORY_HIGH` (systemd `MemoryHigh`, default `85%` of RAM — soft reclaim/throttle only, never an OOM kill; empty string omits the line) (e.g. a GitHub API proxy). `sudo sh scripts/install.sh` and the PowerShell installer behave the same.

**GPU preflight (Linux):** before the engine lands, the installer censuses PCI GPU hardware (`lspci`, provisioned when missing) and — when the driver userspace is absent — installs it from **first-party distro repos only** (announce-then-act): NVIDIA via apt's `nvidia-driver` / pacman's `nvidia nvidia-utils`; dnf auto-installs `akmod-nvidia` only when RPMFusion is already enabled; zypper and all third-party-only lanes print the exact command instead of running it. AMD/Intel without a Vulkan ICD get the mesa Vulkan drivers. Every install ends with the loud chain: **REBOOT, then `pallama engine update`** — which auto-picks the newest CUDA build the installed driver supports (runtimes are bundled; no CUDA toolkit is ever installed). Opt out with `PALLAMA_AUTO_DRIVER=0` (acknowledged even when the engine bootstrap is off). Windows is advise-only: the PowerShell installer detects NVIDIA hardware without `nvidia-smi` and links the driver download. `pallama doctor` warns (`nvidia driver` / `vulkan driver` rows) whenever PCI GPU hardware is present but its driver is not.

Like ollama's installer: **system-wide only** — root-owned binary in `/usr/local/bin` plus a systemd unit (`Restart=always`, GPU groups, auto-start, restart-on-upgrade; runs as the invoking user — under `sudo` the unit targets `SUDO_USER`, not root) on Linux, or a launchd service (`KeepAlive`, `RunAtLoad`) on macOS. There is deliberately no user-path (`~/.local/bin`) install mode: a second copy there is how stale-binary daemon races happen (`pallama doctor` flags any that already exist, and the installer removes one it finds). Root or sudo is required.

**Clean uninstall:** `sh scripts/uninstall.sh` removes everything the installer put here plus per-user state — services, binary, unit + drop-ins, config (incl. `gh-token.env`), store, engines, runtime/caches — in one pass. Models are the one exception: GGUF + whisper model files are expensive re-downloads, so the script **asks first** (sizes shown, default keep; `--remove-models`/`--yes` for non-interactive full nuke, `--keep-models` to skip the prompt, `--dry-run` for an action transcript). For the quick binary+units-only removal, `sh scripts/install.sh --uninstall` leaves all user data in place.

From a checkout the installer compiles fresh with `cargo` first (never a stale `target/release`); a checkout-less `curl | sh` uses the sha256-verified release channel. No toolchain? It is bootstrapped for you (`scripts/bootstrap.sh --minimal`, opt out with `PALLAMA_AUTO_BOOTSTRAP=0`). Older glibc than 2.35, or Alpine? Automatic fallback to the static musl build (32-bit ARM boards get the static `armv7-unknown-linux-musleabihf` build). Pin a version with `PALLAMA_VERSION=v0.4.0`, a mirror with `PALLAMA_INSTALL_BASE_URL`, or a build repo with `PALLAMA_CHECKOUT`.

**Windows (PowerShell):**

```powershell
$env:PALLAMA_REPO = 'owner/pallama'; irm https://raw.githubusercontent.com/owner/pallama/master/scripts/install.ps1 | iex

# or compile from source (rustup is installed via winget when missing):
irm https://raw.githubusercontent.com/owner/pallama/master/scripts/install.ps1 -OutFile install.ps1
.\install.ps1 -Build -Repo owner/pallama
```

Installs to `%LOCALAPPDATA%\Programs\pallama` and adds it to the user PATH. ARM64 hosts pick the native asset automatically (falling back to the emulated x64 one with a warning when a release has none).

**From source:** `sudo cargo install --path crates/pallama-cli` (recent stable Rust) — or just run the installer from the checkout.

**Uninstall:** `sudo sh scripts/install.sh --uninstall` (removes binary + units; your models and config under `~/.local/share/pallama` / `~/.config/pallama` are user data and stay until you delete them).

## Quickstart

```sh
pallama engine update                     # install + activate upstream llama-server (sha256-verified;
                                           #   regression-gated: >10% decode drop auto-rolls-back; --no-gate skips)
                                           # Linux-NVIDIA? prefers the prebuilt CUDA overlay
                                           #   CUDA overlay (bundled cudart/cublas, no toolkit needed) and
                                           #   falls back to Vulkan when the driver cannot run it
pallama engine build cuda                 # rather compile it? no Linux-CUDA prebuilts upstream? build from source:
                                           #   auto GPU-arch (nvidia-smi), auto host-compiler match
                                           #   (nvcc 12 + gcc 13 -> g++-12), installs as bNNNN-cuda,
                                           #   same probe/gate/activate flow. cpu backend also available
 pallama pull qwen3-0.6b                   # ollama registry shortname (registry.ollama.ai, sha256-verified, resumable)
 pallama pull ggml-org/Qwen3-8B-GGUF:Q4_K_M  # or any owner/repo:QUANT from Hugging Face
 pallama run qwen3-0.6b                    # streaming REPL — not in the store? auto-pulls first, then runs
 pallama show qwen3.5:9b                   # muscle memory? model:tag colons resolve onto flat rows
                                          #   (CLI commands AND gateway /api/chat + OpenAI routes)
 OLLAMA_HOST=http://127.0.0.1:11434 ollama list   # existing ollama clients just work
```

Capability discovery: `GET /.well-known/pallama` (routes, headers, features, engine identity). OpenAI-compatible: `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/rerank`, `/v1/messages` (Anthropic), `/v1/responses` (Responses API), `/v1/audio/transcriptions` (multipart, audio-capable models), `/infill` (FIM), `/v1/chat/completions/control` (steering vectors), `/v1/chat/completions/input_tokens`, `/v1/responses/input_tokens`, `/v1/messages/count_tokens`, `/tokenize`, `/detokenize`, `/apply-template`, `/v1/adapters` (LoRA). Ollama-compatible: `/api/chat`, `/api/generate` (full parity: `system`, `images[]`, `think`, streaming NDJSON — all translated onto the hardened chat bus), `/api/tags`, `/api/pull`, `/api/ps`, `/api/show` (reports `capabilities`: `completion` always, `vision` when an mmproj is attached), `/api/embeddings`, `/api/events`; `/api/chat` maps `logprobs`/`top_logprobs` back onto the ollama-native response shape. Pallama-native: `/api/evict`, `/api/session` (slot KV checkpoints + `_auto` session bank: KV survives eviction, restored on respawn), `/api/keys` (virtual-key CRUD), `GET /v1/responses/{id}` (stored-response retrieval), `/v1/batches` + `/v1/files` (OpenAI Batch API: upload JSONL, async batch jobs, poll/cancel, download results — each line replays through the full auth/quota lane), `/api/why` (sentinel record ring), `/api/watch` (live SSE sentinel tail), `X-Pallama-Num-Ctx` request header (per-request ctx on the OpenAI path — the protocol has no such field; same restart-once semantics as `options.num_ctx`), `X-Pallama-Enforce` header (agent loops: 422 on malformed tool calls / schema violations for non-stream requests), `X-Pallama-Deadline-Ms` header (SLO: EDF queue ordering; prefill-heavy requests auto-demote one class), `X-Pallama-Session` header (session pins: requests carrying it hold the model's eviction for `session_keep_secs`; release via `POST /api/session {"action":"close"}`, list via `GET /api/sessions`), `X-Pallama-Cache`/`-Ttl`/`-Threshold` headers (opt-in semantic cache on ollama `/api/chat` non-stream: similarity-gated hits with `cache_debug` in the response body — see `[semantic_cache]` config). Identical non-stream requests single-flight (duplicate waits, then rides the leader's warm prefix). Sentinel observes chat-completions, legacy completions, and `/v1/responses` (both stream and non-stream) on both APIs.

## Why (ollama complaints → pallama resolutions)

| ollama failure | pallama resolution |
|---|---|
| Vendored engine fork lags upstream; breaks new models | Zero fork. Official upstream binaries, sha256-verified, side-by-side installs, `engine update/use/rollback`. New architectures land when upstream ships them |
| Best backend not published as a prebuilt (Linux CUDA) | `pallama engine build cuda` compiles the in-tree ggml-cuda backend from an upstream tag: auto GPU-arch detection (`nvidia-smi` compute cap), auto CUDA host-compiler matching (e.g. nvcc 12 + gcc 13 → `g++-12`), full binary set (`llama-server/-quantize/-imatrix/-perplexity/-bench`), installed as `bNNNN-cuda` through the same probe → regression-gate → activate flow as release assets |
| Slower than llama.cpp (1.8× reports) | No inference reimplementation; byte-stream proxy + measured profiles (`tune --search`) |
| Modelfile copies 30–60 GB to change one parameter; silent `{{ .Prompt }}` fallback | No Modelfile. GGUF is self-contained truth; child always runs `--jinja` (embedded template); params per-request or per-model config overlay |
| Hashed blob lock-in; limited quants | Plain `.gguf` files at `~/.local/share/pallama/models/` — any tool can use them; any HF quant |
| No multi-shard GGUF | Shard sets pulled natively (`-0000N-of-0000M`), first shard launched |
| Registry bottleneck; misleading names | Direct HF pull **and** registry.ollama.ai pull (`qwen3:0.6b` shortnames route to the ollama registry; Docker-v2 manifest wire, sha256-verified resumable blobs via 307→presigned-CDN with allowlisted redirects, token only ever on first-party registry host — `PALLAMA_REGISTRY_TOKEN` for private namespaces). HF names = actual repo name, never a marketing alias |
| CVE-2025-51471 token exfiltration class | HTTPS to HF only; redirect-host allowlist; token sent only to `huggingface.co` (verified in tests) |
| Cloud pivot; silent off-machine routing | Hard local-only; stated in `--help` and at startup |
| Opaque 2048 default ctx; silent truncation | Default ctx 16384, shown in `ps`; overflow errors surfaced verbatim with fix hints |
| Hidden concurrency | `ps` shows slots/ctx/ports/in-flight; loaders stream `X-Pallama-Status: loading`; explicit priority queue (`X-Pallama-Priority`) |
| Attribution dodging | Version, banner, and this README credit llama.cpp/ggml/ggerganov |
| `keep_alive` confusion | Honored in both APIs: `N` pins N s, `0` evicts after the request, `-1` pins forever; `/api/ps` shows the countdown |
| Per-request `num_ctx` silently ignored | Honored: instance restarts once at the requested size, or a 400 with the max-fitting ctx — never silent truncation. Under unified KV the pin is budget-validated *before* any spawn: the `--cache-ram` budget is raised within the 60% RAM guard when the pin is hostable (auto slots walk down to a pool that fits), and a truly unhostable pin is a single teaching 400 naming all four levers (lower `num_ctx` / smaller quant via `pallama fit` / `cache_type = "q8_0"` / `kv_unified = false`) — never a spawn-die-retry storm of 502s |
| Unbounded cache RAM; blind pulls | `cache_ram_mb` config (auto-capped at 30% of physical RAM — measured live: unclamped 8 GiB budget on a 13 GiB box drove the child to an 8.3 GiB RSS plateau and system-wide swap death); `pallama fit` previews VRAM fit + quant alternatives *before* downloading, incl. the ctx headroom a q8_0 KV cache buys |
| No KV-quant control; VRAM cliffs | capacity-math ladder: f16 -> q8_0 (KV/2) -> q4_0 (KV/4) at 90% VRAM, or pin any type via `cache_type`; visible in `pallama show` warnings |
| No session/context persistence | `pallama session save/restore`: slot KV checkpoints that survive unload and daemon restarts |
| No diagnostics when things break | `pallama doctor`: config (incl. stale pins that mirror retired defaults), port conflicts (incl. the ollama-11434 class), engine, hardware, disk, model health (incl. model-dir orphans vs the store — unmanaged GGUFs get a `pallama import` hint, hardlink twins flagged as zero-space), component update-currency in one table |
| No discovery; env sprawl | `pallama search` (HF GGUF); every knob in one documented `config.toml`, inspectable via `pallama config` |
| "API returns 200 but nothing useful happens"; silent truncation; plain-text instead of tool calls; agents loop on malformed tool args | sentinel: warn-only semantic observation on every chat request — `finish_reason: length` with fix hints, tool-arg JSON + hallucinated-name + parameters-schema checks, `json_schema`/`json_object` validation, empty-response and reasoning-with-no-answer detection, stalled-stream detection, and a pre-inference template-capability check (`x-pallama-warnings` header) when the model's chat template cannot render tools. `pallama why [trace]` answers any request after the fact |

## Commands

| Command | Purpose |
|---|---|
| `serve` / `stop` | Daemon lifecycle (pidfile-guarded, graceful shutdown) |
| `pull` / `rm` / `list` / `show` | Model store (plain GGUF files; rm refuses while running) |
| `run <model>` | Streaming REPL (`/exit /clear /model /sysinfo /profile`); profile decisions print as `[profile] …` once after the first turn of each model |
| `ps [--reset]` | Live instances: state, ctx, **GPU offload + card** (`full@<card>`/`partial@<card>`/`cpu`/`auto` — silent CPU fallback is never silent, and you always see which card holds the model), in-flight, endpoint; `warn[<model>]` lines under the table surface profile decisions (ctx fit, slot auto, offload rationale) |
| `bench` / `tune --search` | `llama-bench` tables; measured argmax profile adoption |
| `scripts/bench_matrix.py` | The single benchmarking entry point. Full matrix: every engine × provider (direct child spawn / pallama gateway via sandbox / ollama reference) × ctx/parallel sweep, plus **quality lanes** — llama-perplexity parity, 20-prompt greedy parity + gateway-transparency lane, feature-capability matrix, concurrency, optimization axes (KV quant / spec decode / projector / paged-attn), soak — into resume-safe `cells.jsonl` artifacts, an internal forensics `benchmark.md`, and a **publication-format report** (`--md BENCHMARK.md`, or `--render-only` to re-render an existing campaign without re-measuring). **Cold-start parity lanes**: page-cache-dropped (`posix_fadvise`) + GPU-idle-asserted cold TTFT on every runtime, optional ollama daemon-boot metric (`--ollama-service-restart`, sudo via `BENCH_SUDO_PASSWORD` env). **Idle-wake lane** (pallama sleep ladder vs ollama keep_alive expiry, policy verified via `/api/ps`), **long-ctx curve** (`--ctxcurve-sweep`), **sustained concurrency** (`--conc-rounds`, per-round + cross-round p99 tails). Model pick is store-row-first (scratch files can't 404 the gateway lanes). Supersedes the earlier `bench_engines`/`bench_compare`/`soak.sh` harnesses. Never touches the real daemon, config, or ollama |
| `engine list/update/use/rollback` | Upstream engine management with capability manifests |
| `engine build cuda\|cpu [--tag bNNNN] [--arch A] [--jobs N]` | Compile llama.cpp from an upstream tag when no prebuilt fits: auto GPU-arch (`nvidia-smi` compute cap), auto CUDA host-compiler match (nvcc 12 + gcc 13 → `g++-12`), installs as `bNNNN-cuda` through the same probe → activate flow. Closes the "no Linux-CUDA prebuilts upstream" gap with upstream's own in-tree ggml-cuda kernels |
| `engine install --kind sglang [version]` | SGLang as a third engine kind for **safetensors models** (non-GGUF lane): uv/pip venv install pinned per release (`sglang==0.5.19` default, `ninja` bundled for flashinfer JIT, venv-PATH shim), Linux x86_64/aarch64 CUDA/ROCm. `pallama pull` grows a safetensors lane — whole-repo downloads (shards + config + tokenizer, LFS sha256-verified, `.part` resume) into `models/<name>.d/` dir rows. Same one-port gateway (`/api/chat`, `/api/generate`, OpenAI `/v1/*`, Anthropic `/v1/messages`) with zero client change. Low-VRAM ladder compiled at spawn: full → KV fp8 (`--kv-cache-dtype`) → CPU offload (`--cpu-offload-gb`, host-RAM-capped) → loud refusal with the numbers; ~45 first-class knobs under `model_overrides.<name>.sglang.*` (tokenizer workers/batching, grammar backend + radix eviction, idle/VRAM hygiene, scheduler tuning, explicit `cuda_graph_bs` list, `tp/dp/pp/ep_size` pins, LoRA rank/backend) + LoRA adapter lane (HF/PEFT adapters via `--enable-lora`; GGUF adapters refused with teaching) + reserved-flag guard on `extra_args`; warm-peg after spawn (`[warm_peg]` table, default true, per-engine `warm_peg.sglang/llamacpp` overrides) pre-opens the concurrent batch shapes so the first real request never pays the one-time JIT/autotune stall; portability defaults (`--attention-backend triton --sampling-backend pytorch`, overridable) survive boxes where system nvcc can't compile flashinfer JIT. GGUF models keep using llamacpp/mistralrs — engine/model mismatches fail with teaching errors |
| `engine install [--tag vX.Y.Z]` | mistral.rs as a second engine kind: prebuilt GPU-aware pick (newest `cudaNNN` the driver supports × exact compute cap, Metal on Apple Silicon, CPU otherwise — loud warn on fallback), activated as a `mistralrs`-kind row. Serving paths: OpenAI chat/completions/embeddings + Anthropic `/v1/messages` via the same gateway; llama-server-only surfaces (`/tokenize`, slot sessions, GBNF grammar, steering, rerank) answer a teaching 400 instead of breaking; decode-regression gate is llama-server-only and skipped with a printed note. ~21 first-class knobs under `model_overrides.<name>.mistralrs.*` (or global `[mistralrs]`): scheduler (batch/prefill-chunk/decode-steps/prefix-cache-n), paged-attention detail (block size, cache type, context len), LoRA dynamic-serving switch + capacity + native `--lora ALIAS=SOURCE` adapter lane (GGUF and safetensors), MTP speculative family, vision caps, metrics/access-log toggles, `device_layers` pins — all flag-gated per engine manifest |
| `config list/get/set/unset/defaults` | One knob surface (validates on set; unset restores built-in defaults) |
| `cp <src> <dst>` / `create <name> -f Modelfile` | ollama-parity aliases: zero-byte hardlink + config overlay (no blob copies; unsupported Modelfile keys are named rejections) |
| `run <model> [prompt…]` / `stop <model>` | inline single-shot generation (`--verbose` counts) / unload a model now |
| `push` / `login` family | refused by design: local-only, no registry or cloud accounts |
| `search <query>` | HF GGUF search (downloads, likes, sizes) |
| `session save/restore/rm/list` | slot KV-cache checkpoints: pause a model's context, resume later (survives unload + restart) |
| `doctor` | one-command diagnostics: config, port conflicts, engine (+ binary `--version` smoke, + GPU enumeration-drift vs the serving child), hardware (incl. PCI GPU present but driverless — `nvidia driver`/`vulkan driver` warnings), disk, model health, sqlite store integrity, service-manager state (systemd/launchd), bind+auth exposure, update currency (llama.cpp `engine update`, whisper.cpp `whisper --install`, pallama `upgrade` — warn-only, never auto-installs) |
| `why [trace]` | sentinel: what the model returned, what was wrong with it (truncation, invalid tool args, schema violations, empty replies, stalls), which knob fixes it |
| `watch` | live tail of sentinel detections as they happen (SSE; Ctrl-C to stop) |
| `router = true` (config) | one child serves ALL models: preset INI auto-generated per model, engine-native autoload + LRU; `pallama stop <model>` becomes an engine unload |
| `upgrade [--version] [--dry-run]` | Self-update from GitHub Releases (sha256-verified, atomic) |
| `keys [list]/add/rm/rotate NAME [--models a,b] [--rpm N] [--tpm N] [--daily-tokens N] [--max-concurrent N]` | Virtual API keys: model scoping (wildcards: `qwen3*`, `vllm:*`), rate limits, daily token budgets, per-key concurrency cap (leased at auth, held while the response streams), live usage, secret rotation (`[[keys]]` in config; `x-api-key` works for Anthropic clients; SIGHUP hot-reloads the key set) |
| `quantize <model> -t Q4_K_M [--imatrix calib.txt] [--verify [--max-degradation P]] [--allow-requantize] [--name N]` | Derive a new quant with the engine's own llama-quantize; `--imatrix` calibrates an importance matrix first (F16 sources; better Q4 accuracy); `--verify` gates the output through llama-perplexity over a fixed probe corpus and deletes+rejects it when perplexity degrades beyond P% (default 10); `--allow-requantize` permits quantizing an already-quantized source (quality risk — pair with `--verify`) |
| `launch [--warm MODEL] [--key K] <cli> [args...]` | Point any OpenAI/Anthropic/ollama-speaking CLI at pallama (env wired, optional pre-warm) |
| `whisper [--install] [--tag TAG] [--pin TAG\|none] [--pull SIZE] [--list] [<file>]` | Managed STT lane: installs whisper.cpp server from its releases (`--tag` installs and pins that release; `--pin` switches the active installed tag without any download, `none` unpins to newest-installed; 3 newest kept, pinned always kept), pulls ggml models (`tiny`…`large-v3-turbo`), transcribes via the lazy local server on `/v1/audio/transcriptions`; `<remote>:model` still forwards |
| `mmproj <model> <path>` | Attach a vision projector GGUF to an existing model (refuses while running; next spawn gets vision); `import --mmproj <path>` attaches at import time |
| `tune <model> --slots C` / `--ngram` / `--load` / `--replicas` / `--cache-reuse` | Live-concurrency `-np` search (adopts on a >5% win, records the engine-update regression baseline); n-gram grid; warmup A/B (adopts `warmup = false` only if cold load + first token is >5% faster); replica scaling probe (8 clients x 3 gens against 1 vs 2 identical children, adopts `replicas = 2` only if >1.3x aggregate); `--cache-reuse` grid {0,256,512} on a long repeated prefix (adopts only if >5% faster and >50 ms — noise-floored) |
| `replicas = N` (overlay) | Parallel instances of one model (1..=8): distinct conversation prefixes each get their own warm-cache child (prefix-hash sticky routing); conversations sharing a system prompt coalesce onto the replica that already holds that system KV; `ps` shows `model#N` |
| `pin = true` (overlay) | Never pick this instance as a capacity-eviction victim (hot-prefix pinning; pressure falls on unpinned instances) |
| Spec draft pairing | `spec = "auto"` pairs via catalog prefix match; the draft must be **pre-pulled** (`pallama pull ggml-org/Qwen3-0.6B-GGUF:Q4_0`) — a missing draft is a hard error naming the pull command. Accept-rate gauge: `pallama_spec_accept_rate` |
| MTP speculation | `spec = "mtp"` emits `--spec-type draft-mtp` for GGUFs that ship the multi-token-prediction head **inside the weights** (no draft model; e.g. unsloth `Qwen3.5-9B-MTP-GGUF` — the common base Q4_K_M carries none, verified). Under the **default** `spec = "auto"` an MTP-bearing GGUF auto-enables draft-mtp when the engine advertises it (emits the ollama trio: `--spec-type draft-mtp --spec-draft-n-max N --spec-draft-backend-sampling`, N = min(head layers, 2)) and beats the catalog draft-pair lane; engines without `draft-mtp` warn-skip to the pair path; an unpulled catalog draft degrades to dense with a pull hint (auto never refuses a spawn — explicit modes fail fast). Detection reads both `n_predict_layers` (llama.cpp) and `nextn_predict_layers` (ollama converter) metadata keys. `spec = "off"` never speculates even on MTP GGUFs |
| Engine auto-rollback | Spawn failures across >=2 models (or a failing `--version` probe) auto-rollback to the previous engine tag + `engine_rolled_back` event; restore with `pallama engine use` |
| CUDA dethrone guard + build advisory | On NVIDIA boxes a newly installed non-CUDA llama.cpp engine never auto-DEMOTEs an installed CUDA engine (Vulkan first-token is measurably slower; override with `pallama engine use`); NVIDIA + Vulkan-only active + no CUDA installed → one-time teaching log: `pallama engine build cuda` (~18% faster first token on this class of card, one-time ~7 min build) |
| EAGLE3 speculation | `spec = "eagle3"` pairs a trained speculator draft (separate GGUF, RedHatAI-style): emits `--spec-type draft-eagle3 --spec-draft-model <path>` + the full draft-placement knob set (device pin, cpu range, threads, ngl). Catalog pair shipped for `qwen3-8b` (`williamliao/Qwen3-8B-EAGLE3-Speculator-GGUF:F16`, verified ungated) — typed lookup, so plain `auto` keeps its generic draft-simple pair and precedence. Engines without `draft-eagle3` fail fast naming `pallama engine update`; unpulled draft hard-errors naming the pull command (never silently dense) |
| dflash/dspark speculation | `spec = "dflash"` / `"dspark"` = block-diffusion drafter lanes (newest upstream spec types; two-file pairing like eagle3, ctx_dft asserted upstream). Engine-gated (b10896 advertises both). Catalog pairs shipped for `qwen3-8b` (ggml-org `dflash-`/`dspark-Qwen3-8B-Q8_0.gguf`, exact-filename pull slot) — **measured -17% vs dense on prose** (acceptance 0.17-0.25, BENCHMARK.md); try on code/templated workloads, never defaulted |
| Lazy tensor residency | `lazy_mode = "auto"\|"on"\|"off"` (global + per-model override) → `--lazy-mode` on deviation from the engine default `auto`: big tensors (>4 GiB) stream from disk on demand instead of sitting resident in RAM — MoE relief on RAM-tight boxes. Emitted only when you deviate (default config stays argv-clean); engine without the flag degrades to a teaching warning |
| Full sampler surface on ollama options | Request `options` accept every llama-server-native sampler: `xtc_probability`/`xtc_threshold`, `top_n_sigma`, `logit_bias`, `dry_multiplier`/`dry_base`/`dry_allowed_length`/`dry_penalty_last_n`/`dry_sequence_breakers` (shape-checked: non-empty string array), `mirostat`/`mirostat_tau`/`mirostat_eta`, `dynatemp_range`/`dynatemp_exponent`, `adaptive_target`/`adaptive_decay`, `samplers` chain, and `adaptive_p: true` (joins the sampler chain post-pass — order-independent with an explicit `samplers` list). Unknown keys still 400 listing them, never silently dropped |
| Server-side tools (experimental) | Global-only quartet `server_tools` / `server_tools_runtime` / `mcp_servers_config` / `mcp_servers_json` → `--tools` / `--tools-runtime` / `--mcp-servers-config` / `--mcp-servers-json` (upstream experimental agent tooling; engine limits CORS to localhost when set). Opt-in, never defaulted, **trusted environments only** (tools include `exec_shell_command`); runtime accepts `docker:`/`podman:` images as passthrough values — pallama itself never requires docker. `mcp_servers_config` path is existence-checked at profile-compile (typo fails before boot, not after); global-only on purpose: security posture must not vary silently per model. Old engines degrade to teaching warnings |
| GGUF metadata lint | `pull`/`import`/`show` warn when context length, attention geometry, or SWA layout is incomplete (KV/VRAM estimates run blind) — warn-only, never blocks |
| Hybrid-linear KV math | Hybrid-linear archs (qwen3-next/qwen3.5/Kimi-K3-class: linear attention + full-attention mix) get a TRUE fractional KV estimate: only full-attention layers count when the GGUF carries `recurrent_layers` or `full_attention_interval`; pure-recurrent archs (mamba, rwkv) estimate ~0 KV (constant state). Missing metadata on a hybrid arch → conservative full-layer upper bound + warning naming what a newer conversion would emit. Feeds `coreside`, fit checks, and the KV-quant ladder automatically |
| `predictive_preload = true` | Reaper learns model transitions (A→B counts >= 3) and pre-spawns the next likely model while the current one idles — kills the cold-start on alternating workloads; emits `model_preloaded` event; 5-min failure backoff |
| `adaptive_slots` (default on) | Sustained admission pressure (requests parked waiting for a slot across 6 reaper ticks = 60 s) auto-adopts `-np +1` (cap 8), then respawns the instance at the first idle drain — the KV bank carries conversations across the reshape; spawn failure rolls the adoption back; per-model `slots` overlay, replicas, and `deterministic` models are excluded; `tune --slots` remains the persistent path. After 5 min of quiet the adoption decays back to the natural shape (per-stream latency recovers; re-adopts in 60 s if demand returns) |
| Slots auto-fit | When capacity caps `slots = auto` at ONE slot and ctx came from the default (not pinned by overlay/tuning/`num_ctx`), the same total-ctx budget is re-spent as parallel shallow slots (e.g. 1x16384 -> 4x4096, floor 4096 — ollama's throughput-first default; identical VRAM/KV budget, concurrent streams batch instead of queueing). Emits `slots_ctx_auto_fit` event + a teaching warning; prompts longer than the per-slot ctx take the existing `num_ctx` restart-once path |
| `lookup_cache_static` / `lookup_cache_dynamic` | Path to a llama.cpp lookup cache file (validated to exist); `static` is read-only, `dynamic` is refreshed by generation — verbatim passthrough of `-lcs`/`-lcd` |
| Auto GPU pick (multi-GPU) | With `devices` unset and >1 GPU, each spawn probes `--list-devices` and picks the **discrete** card with the most free VRAM (integrated cards report shared system RAM as "free" — huge but bandwidth-starved; they are skipped and logged, and used only on integrated-only boxes). All VRAM math (ngl, KV, cache-ram, the 95% co-residency KV-downgrade planner) is scoped to that card, not the summed pool. With a spare discrete card left over, profile warnings suggest `spec_draft_device` / `mmproj_device` on it (teaching only — bench before adopting) |
| Measured VRAM feedback | After every spawn settles, the supervisor re-probes the card and logs the **measured** free-VRAM delta next to the predicted weights+KV estimate — the number to trust when tuning. A card found >95% committed after load (or later, checked at least once a minute while models run) gets a teaching warning: kv-quantize, smaller quant (`pallama fit`), or free co-resident engines (`pallama ps`). Pressure is judged card-level against the card total, so squatters (ollama, a desktop session) count exactly like our own children |
| Auto tensor-split (last resort) | When `devices` / `tensor_split` / `main_gpu` are all unset and weights+KV exceed the best discrete card's free VRAM but fit the discrete cards combined, the supervisor plans a proportional `--tensor-split` automatically — with a loud warning that a layer split buys capacity, not speed (inter-card bandwidth taxes every token) and a smaller quant may serve faster. Any manual pin silences it |
| `coreside` | VRAM co-residency plan: which local models fit together (weights + KV @ ctx — the 512 MiB working-set floor when `--kv-unified` applies, f16 otherwise — hot-first greedy) |
| `drafts <model>` | Speculative-decoding draft candidates (EAGLE3/MTP heads + small same-family) from live HF search |
| `completions <bash\|zsh\|fish\|powershell>` | Shell completions to stdout |
| `snapshot` | Backup config + store to `<data>/snapshots/<ts>/` (models/engines stay: bulk) |
| `[[remotes]]` (config) | Remote OpenAI-compatible engines (vLLM / MLX / another pallama): `model = "name:model"` routes there; `ps` probes health. Duplicate entries under one name form a pool: least-in-flight load balancing + failover circuit (3 consecutive failures marks a member down for 30 s, then a half-open probe) |

Daemon-dependent commands auto-start `pallama serve` (detached, logs at `~/.local/share/pallama/run/daemon.log`).

## Configuration

`~/.config/pallama/config.toml` — created with defaults on first run; `PALLAMA_*` env vars override; secrets (`HF_TOKEN`, `GH_TOKEN`) are env-only, never stored.

```toml
host = "127.0.0.1"
port = 11435              # default 11434 (drop-in ollama replacement); change it when ollama runs side-by-side on the same box
idle_sleep_secs = 300     # child-native GPU sleep (frees VRAM, warm wake)
idle_timeout_secs = 1800  # process eviction after idle
max_loaded_models = 0     # 0 = auto from VRAM / model size
child_transport = "tcp"   # "tcp" (curl-debuggable) — "unix" rejected at load (not implemented)
auto_restart_engine_switch = false  # true: engine use/update/build/install restarts a
                                    #   live daemon itself (user-scope service or
                                    #   passwordless sudo; else falls back to the hint)
# child_auth: unset = auto (TCP children authed, UDS not) | true | false.
#   Per-child secret minted at spawn (0600 keyfile run/<model>.apikey, never in
#   argv on engines with --api-key-file), gateway stamps every child call;
#   direct curl then needs Authorization: Bearer <key>
 # log_level: tracing directive for the daemon, e.g. "debug" or
 #   "pallama=trace,pallama::engine=debug" (shows engine child logs).
 #   Unset = INFO. RUST_LOG env always wins (systemd escape hatch).
 # update_channel: "latest" (default, the daily llama.cpp b-tag firehose)
 #   or "stable" (newest vX.Y.Z, dereferenced through its nightly-tag.txt
 #   to the concrete b-build). Channels are pins, not floors: switching
 #   latest -> stable re-targets downward (regression gate skipped, the
 #   downgrade is the point); explicit `engine update --tag bNNNN`
 #   bypasses the channel. Applies to whisper/self lanes too (they only
 #   publish through /releases/latest, so both channels coincide there).
 #   One-liner: `pallama config set update_channel stable` (invalid values
 #   are rejected, file unchanged).
 update_channel = "latest"
engine_asset = "auto"     # auto picks by OS/GPU (versioned cuda/rocm = newest); or explicit: "ubuntu-vulkan-x64", ...
spec = "auto"             # DEFAULT auto (2026-09-11): opportunistic — draft-mtp when the GGUF
                            #   ships an MTP head (n_predict_layers; measured +50% decode), catalog
                            #   draft pair when pulled, dense + teaching warning otherwise (never
                            #   refuses a spawn); "off" = dense always; explicit modes fail fast:
                            #   "mtp" (head in GGUF), "eagle3" (speculator), "dflash"|"dspark"
                            #   (block-diffusion, engine-gated; qwen3-8b pairs in catalog, -17% prose)
                            #   "ngram" = self-drafting (no draft model);
                            #   typed n-gram engines: "ngram-map-k" | "ngram-map-k4v" | "ngram-mod" | "ngram-cache"
 lazy_mode = "auto"         # tensor residency: auto = engine default (>4GiB on-demand), on = all big
                            #   tensors from disk (mmap), off = resident; per-model override exists
# --- experimental upstream agent tooling (global-only, trusted environments only) ---
# server_tools = "grep_search,read_file"  # or "all"; adds server-side tool calls, CORS -> localhost
# server_tools_runtime = "ssh:gpu-box"    # docker:<img>|podman:<img>|docker-container:<id>|podman-container:<id>|ssh:<host>
# mcp_servers_config = "/etc/pallama/mcp.json"  # Cursor-compatible MCP defs; path checked at compile
# mcp_servers_json = '{"mcpServers":{"fs":{}}}' # inline alternative (mutually exclusive with the path)
cache_reuse = 0          # prefix-cache chunk reuse (0 = off by default: the engine's native slot prompt-cache already covers identical prefixes at 16x, while the flag measured ~0.6s SLOWER cold loads; enable for cross-slot prefix sharing, grid-search via `tune --cache-reuse`)
cache_idle_slots = true   # false emits --no-cache-idle-slots (skip saving idle slots to prompt cache)
predictive_preload = false # reaper pre-spawns the next likely model (>=3 A->B transitions) while the current idles
 adaptive_slots = true     # sustained admission pressure (queued requests) auto-adopts -np +1 (cap 8) and respawns at idle drain; default on
 deterministic = false     # true pins slots = 1 (both engine lanes): multi-slot batches
                           #   perturb logits in near-tie positions, so greedy output under
                           #   slots > 1 is not token-for-token reproducible; costs nothing
                           #   single-client, queues concurrent streams. Per-model
                           #   [model_overrides] deterministic exists too.
# lookup_cache_static = "/path/to/cache.bin"   # -lcs: read-only lookup cache (must exist)
# lookup_cache_dynamic = "/path/to/cache.bin"  # -lcd: lookup cache refreshed by generation
api_keys = []             # non-empty = Bearer auth at the gateway (children stay loopback)
rpc_servers = ""          # e.g. "box1:50052,box2:50052" -> --rpc (per-model
                          #   override: model_overrides.rpc_servers replaces it
                          #   for that model; empty/absent overlay inherits).
                          #   Endpoints are TCP-probed before every spawn: a
                          #   dead one fails fast with a teaching 500 instead
                          #   of crash-looping the child (upstream SIGABRTs)
cache_ram_mb = 8192       # child prompt-cache budget; auto-capped at 30% of RAM (0 = unlimited)
cpu_range = ""            # pin child threads to CPUs "lo-hi" (P/E hybrids: pin P-cores, discover via `lscpu -e`)
poll = 0                  # 1..=100 busy-poll waiting for work (trades idle CPU for TTFT)
reasoning_format = ""     # "" auto | "none" | "deepseek" | "deepseek-legacy" thought-tag extraction
reasoning = ""            # "" engine auto | "on" | "off" | "auto" server-side reasoning switch (--reasoning; per-model override supported)
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
load_mode = ""            # "" = auto (mlock when weights <= 40% RAM — eager page-in measured ~0.7s faster to first token) | mmap | mlock | direct-io
spawn_mem_guard = true    # refuse loads when MemAvailable < model/2 + 512 MiB
                          #   (swap-death prevention; set false to load anyway)
session_bank = true       # KV checkpoints (_auto) survive eviction; restored on respawn
prompt_preflight = true   # refuse prompts that cannot fit ctx (exact /tokenize over
                          #   the 90% estimate threshold) — the engine would silently
                          #   truncate; set false to allow truncation (sentinel still warns)
singleflight = true       # identical non-stream chat requests coalesce (5s bound);
                          #   duplicates ride the leader's warm prefix
spec_cache = true         # persist n-gram speculation cache across restarts (spec = "ngram")
 session_keep_secs = 900   # session pins: `x-pallama-session: <name>` header holds a model's
                           #   eviction for this window (refreshed per pinned request); 0 = off.
                           #   POST /api/session {"action":"close","session":...} releases early;
                           #   GET /api/sessions lists live pins; force stop always wins

[semantic_cache]           # R4 opt-in semantic cache (ollama /api/chat non-stream only)
enabled = false            # off by default — cached answers trade freshness for speed
model = ""                 # REQUIRED when enabled: local embed model (e.g. "bge-m3"); runs
                           #   via its own engine child (/tokenize + /embedding)
ttl_secs = 600             # entry lifetime
threshold = 0.90           # cosine similarity gate for a hit
max_entries = 256          # LRU cap
# Per-request: `x-pallama-cache: on|off` (overrides config), `x-pallama-cache-ttl: <secs>`,
# `x-pallama-cache-threshold: <0.01..=1.0>` (malformed -> 400). Hits carry
# `cache_debug{cache_hit,hit_type,similarity,cache_id}` + `x-pallama-cache: hit; similarity=X`;
# misses store the response and return `x-pallama-cache: miss`. Entries never cross
# API keys; streams bypass entirely; structured-output requests (grammar / non-null
# format) bypass in BOTH directions — lookup and store (constraints change the valid
# output space, a cached unconstrained answer would violate them); embed-model
# failures bypass (request still served).
# Metrics: pallama_semantic_cache_{hits,misses,stores,embed_failures}_total + entries gauge.
sessions = true           # slot KV checkpoints: `pallama session save/restore` (engine-gated)
ctx_extend = 0.0          # YaRN context extension factor; 0 = off. >1.0..=32.0; quality tradeoff
cpu_moe_n = 0             # keep N MoE experts on CPU (finer than the auto heuristic)
cpu_ffn_n = 0             # dense models: keep first N layers' FFN weights on CPU (--n-cpu-ffn)
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
audit_log = false         # E4 audit trail: one JSON line per generation request
                          #   (/api/chat, /api/generate, /v1/chat/completions,
                          #   /v1/completions, /v1/messages, /v1/responses) to
                          #   <data>/log/audit.jsonl — trace id, key name, model,
                          #   status, latency, priority, queue depth. NEVER request
                          #   or response content. Off-path writer (drops + counts
                          #   under pressure: pallama_audit_dropped_total metric);
                          #   rotates at 16 MiB keeping the newest 2000 lines

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
weight = 1                # WFQ share under queue contention (>= 1): among keys
                          #   waiting at the same priority + SLO tier, a weight-3
                          #   key earns ~3x the slots of a weight-1 key (virtual-
                          #   runtime credits; explicit x-pallama-deadline-ms
                          #   waiters keep strict deadline order)

[[remotes]]               # external OpenAI-compatible engines (vLLM, MLX, another pallama)
name = "vllm"             # requests with model = "vllm:<id>" route there
url = "http://10.0.0.4:8000"
key = ""                  # optional bearer for the remote
# Duplicate [[remotes]] entries with the SAME name form a POOL: circuit
# health per member (3 fails -> 30s mark-down), least-inflight LB, and
# conversation-prefix stickiness — repeat prompts return to the member
# whose KV already holds them (bounded affinity table; unbinds on
# failure). The serving member is named in the x-pallama-remote
# response header on both the /v1 and /api lanes.

[model_overrides."qwen3.5-9b"]     # per-model overlay; unknown keys are errors
ctx = 32768
# Overlay keys: ctx, slots, spec, loras, extra_args, cache_type, kv_unified,
# ctx_extend, cpu_moe_n, override_tensor, devices, warmup,
# reasoning_budget, reasoning_effort, replicas, pin,
# chat_template, chat_template_file, sampler_defaults, spm_infill,
# late_chunking, rpc_servers, mmproj
# mmproj = "lazy"  # tri-state projector policy (also accepts true/false):
#                  # lazy (default) — text-only cold spawn (~3.0 s faster,
#                  # 1126 MiB VRAM freed); first vision request respawns
#                  # with the projector, conversation carried via KV bank
#                  # true/attach — always attach the row's projector
#                  # false/skip   — text-only; vision requests fail loudly
# Global default knob: mmproj_policy = "lazy" | "attach" | "skip"

# ---- scaling recipes (measured 2026-09-12, qwen3.5-9b / RTX 4070, greedy-parity
# verified on every shape — all scheduling-only knobs):
# slots = 8 + ctx = 4096        # concurrency-first: 8 streams batch instead of
#                               # queue (system +19%, TTFT p99 5.4x at 8 streams,
#                               # LESS VRAM); per-stream ITL rises ~60% and
#                               # per-slot ctx shrinks — prefer for many short
#                               # concurrent clients
# extra_args = ["--poll", "100", "--threads-http", "4"]
#                               # high-concurrency tail latency: TTFT p99 ~-11%
#                               # at the cost of CPU busy-spin while waiting on
#                               # the GPU; leave default (50) for desktop use
# NOTE: raising -ub above the 512 default is a measured regression on long
# prefill (2111 -> 1728 t/s at 8k tokens, +360 MiB) — leave ubatch alone.

[model_overrides."bge-m3"]          # late chunking (jina arXiv 2409.04701):
late_chunking = true                # doc embedded ONCE per-token (pooling=none),
                                    # gateway mean-pools per chunk span; /api/embed,
                                    # /api/embeddings + /v1/embeddings gain a
                                    # late_chunking:true lane (string / string[] /
                                    # pretokenized int[][] input)
# late_chunking_max_tokens = 8192  # global late-chunk doc budget (413 above)
# ubatch_size = 2048               # explicit ubatch for late models (default 2048;
                                    # one doc must fit one micro-batch)

[model_overrides."qwen3-coder-7b"] # FIM code-infill models (e.g. CodeGemma)
spm_infill = true                  # Suffix/Prefix/Middle token order
chat_template = "chatml"           # or chat_template_file = "/path" (XOR; both set = refuse)

[model_overrides."creative-model"] # per-model sampler DEFAULTS (18 argv flags;
sampler_defaults.temperature = 0.9 # request-body values still override per call;
sampler_defaults.top_k = 50        # extra_args keeps last-word precedence)
sampler_defaults.dry_multiplier = 0.8

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
# typed-variant tuning: map-k/map-k4v/simple share ngram_size_* knobs;
# spec = "ngram-mod" uses these (0 = engine default 24/64/48):
# ngram_mod_n_match = 0           # tokens matched before drafting (1..=1024)
# ngram_mod_n_max = 0             # max draft tokens (0..=1024)
# ngram_mod_n_min = 0             # min draft tokens (0..=1024)

# compute plumbing / multi-GPU split (0/"" = engine defaults; bench `tune` still wins)
# batch_size = 0                  # logical batch (--batch-size)
# ubatch_size = 0                 # physical micro-batch (--ubatch-size)
# threads_batch = 0               # threads for prompt processing (--threads-batch)
# main_gpu = -1                   # primary GPU index (--main-gpu; -1 = engine auto)
# split_mode = ""                 # "" | none | layer | row | tensor (--split-mode)
# tensor_split = ""               # comma ratios per GPU, e.g. "3,1" (--tensor-split)
# models_autoload: router-mode LRU autoload override (unset = upstream default)
#   true = --models-autoload, false = --no-models-autoload

# reasoning control (cuts wasted thinking tokens on agent traffic)
# reasoning = ""                 # "" engine auto-detect | "on" | "off" | "auto" (server-side switch; authoritative for templates that ignore the thinking/enable_thinking request vars)
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

## Engine routing

One active engine serves at a time (`pallama engine use`), but models come in
formats engines digest differently. Routing picks the right lane per model —
on by default (set `mode = "manual"` to keep one engine for everything):

```toml
[engine_routing]
mode = "auto"        # default; "manual" = the active engine serves everything
policy = "quality"   # quality (default) | latency | throughput
```

- **When it fires:** at spawn time — the first request for a model spawns it
  through the routed engine (lazy, same as every spawn). Pulling a model never
  routes anything; there is no separate loading step.
- **Both API dialects route identically** — `/api/chat` (ollama clients) and
  `/v1/chat/completions` go through the same gateway resolver.
- **Per-model pin wins over both modes:**
  `[model_overrides."my-model"] engine = "sglang"` (a tag or a kind).
- **See what would serve what:** `pallama list` (ENGINE column), `/v1/models`,
  and `/api/tags` all carry the resolved engine per model.
- **Evidence-backed route table** (measured, quant-matched, on this class of
  hardware): safetensors → sglang on quality/throughput (0.612 quality,
  747 tok/s conc4) or mistral.rs on latency (26 ms TTFT, 8 s cold); GGUF →
  llamacpp on every policy (mistral.rs GGUF scored 0.462 vs llamacpp's 0.500
  on the same Q4_K_M model); nothing-can-serve → teaching error, never a guess.
- **manual mode is byte-identical** to pre-routing behavior: the active engine
  serves or refuses with its usual teaching.

## Architecture

```
pallama-core      pure domain: gguf parser, config, store (SQLite), catalog,
                  capability-driven profile compiler (19 rules, manifest-gated)
pallama-runtime   tokio: HF client (token-isolated), engine installer+prober,
                  process supervisor (ladder: active→sleep→evict; crash circuit),
                  event bus, llama-bench runner/tuner
pallama-gateway   axum: byte-stream OpenAI proxy, ollama-compat translation,
                  priority admission queue, metrics merge, SSE events
pallama-cli       the `pallama` binary: clap commands, REPL, auto-start
```

Load-bearing ideas:

- **Capability manifest** — after install, the engine is probed (`--version`, `--list-devices`, `--help`). The profile compiler emits *only* flags the installed build actually supports; a missing flag is a hard error naming it and suggesting `engine use <tag>`. Engine drift becomes a data problem, not a code problem. Source-built engines (`engine build cuda|cpu`) flow through the identical probe → store → gate lane as release assets; their tags carry a `-cuda`/`-cpu` suffix that currency checks compare by build number, so `engine update` hints stay correct across prebuilt ↔ built switches.
- **Prebuilt CUDA overlay** — upstream publishes no Linux-CUDA binaries, so this repo's `engine-cuda` workflow compiles pristine upstream b-tags with `-DGGML_CUDA=ON` on CUDA 12/13 images and publishes them as `bNNNN-cuda` releases (bundling cudart/cublas/cublasLt + EULAs — no toolkit install on the target). On Linux-NVIDIA this is zero-touch: `pallama engine update` automatically prefers the highest `ubuntu-cuda-{X.Y}` asset the driver can run (strict ceiling: a driver older than the toolkit stays on Vulkan rather than shipping a child that cannot boot); `PALLAMA_ENGINE_REPO=owner/repo` points the channel at a fork, `pallama engine install bNNNN-cuda` pins an overlay build explicitly. No fork: every asset is the unmodified upstream source at a tagged commit.
- **Bytes-based VRAM admission** — auto capacity sums resident weights against the VRAM budget instead of dividing by the largest model, so heterogeneous pairs co-reside (0.5B + 9B on an 8 GiB card) where a floor heuristic would evict. Explicit `max_loaded_models` still wins as a count; CPU-only boxes stay pinned to one instance; the fresh-probe spawn guard still owns refuse-vs-warn at load time.
- **Eviction ladder** — active → child-native sleep at `idle_sleep_secs` (VRAM freed, instant wake) → SIGTERM at `idle_timeout_secs`. Nothing burns VRAM forever; nothing dies mid-request.
- **Zero-tax proxy** — OpenAI traffic is forwarded byte-for-byte (no body parsing, no SSE buffering). Tool calls, structured output, logprobs ride through untouched. Client disconnect aborts the upstream request; the slot frees.
- **Signals are single-pid only** — Pallama never signals process groups; every teardown path is audited and idempotent.

## Verification

### Cold-start TTFT (measured, 2026-09-12, RTX 4070 laptop, 0.5B)

Method: `pallama stop`, 1 s settle, then `time pallama run
qwen2.5-0.5b-instruct 'Say ok' --max-tokens 5` — three runs, plus a
warm `/api/chat` for the floor.

| Stage | Wall |
|---|---|
| Cold `run` (3 runs) | 1.16–1.94 s |
| Warm `/api/chat` total_duration | 21.7 ms (prompt 5.5 + eval 16.2) |
| Daemon-side spawn (profile → healthy → session-restore) | ~0.86 s |

The cold wall at 0.5B is engine load (mmap + CUDA init) — pallama-side stages
(profile compile, adaptive health poll ~12 ms overshoot, admission)
are milliseconds. A 9B parity re-run at uncontended conditions
(same file, ctx aligned, disk-cold + GPU-idle asserted on both) told a
different story: pallama cold TTFT 11.4 s vs ollama 5.7 s, warm decode
12.4 vs 27.2 t/s — the gap is the capacity-first default profile
(`-np 4` + wide `--ctx-size` + `--kv-unified` hosting a 256k-token KV
pool in the system-RAM `--cache-ram` budget; PCIe-bound decode), not
daemon overhead. For ollama-shaped single-stream speed, pin
`slots = 1` and `kv_unified = false` (or `model_overrides`); a
single-stream-first defaults pass is on the roadmap.

764 tests: pure compiler tables, wiremock network suites (resume, sha, allowlist, token isolation), engine install cycles with a real stub engine binary, supervisor lifecycle integration (ladder, capacity, crash-circuit, shutdown), and full gateway round-trips over both APIs — including the sentinel suites (9 detection codes, responses grammar, persistence reload, enforce 422s, live watch SSE, parity-under-observation). `cargo clippy --workspace --all-targets -- -D warnings` clean.

Dev hygiene: cargo never garbage-collects `target/` — after heavy test sessions `sh scripts/clean-target.sh` wipes the stale incremental-compilation cache (18G observed once; safe when no build is running, refuses otherwise).

Live harnesses (real engine, real model, no mocks):

- `scripts/validate.py` — the single validation entry point: exhaustive E2E battery (every config knob traced from `config.toml` through the profile compiler to the engine child's actual `/proc` argv, both API surfaces, sentinel, lifecycle behaviors, CLI, auth) plus the merged **manifest registry** (command + knob coverage manifests, `--phase=manifests` for a registry sanity echo) — all in an isolated XDG sandbox (`--fast` smoke mode, `--phase` filters).
- `scripts/bench_matrix.py` — benchmarking (see table above): measurement campaigns, quality lanes, and publication-format reporting in one tool (`--render-only` re-renders without re-measuring).

## Credit

Pallama is an orchestrator: inference is upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) (ggml, ggerganov and hundreds of contributors), [mistral.rs](https://github.com/EricLBuehler/mistral.rs) and [SGLang](https://github.com/sgl-project/sglang) — the engine authors did the hard parts. Models come from their publishers on Hugging Face.
