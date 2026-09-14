# Changelog

All notable changes to Pallama are documented here. Format follows
Keep a Changelog; versions follow SemVer. Earlier releases were not
tracked here.

## [Unreleased]

### Changed
- **Engine retention tightened from 3 to 2 installed builds (2026-09-13).** Auto-prune (which runs after every engine install) now keeps the newest engine plus one rollback anchor (~215 MiB each); `local` and the active tag stay protected on top. Users who want deeper history still have `pallama engine use <tag>` re-download on demand.
- **Prebuilt CUDA engine channel slimmed to consumer archs + PTX forward-JIT (2026-09-13).** The overlay CI (`engine-cuda`) built fat binaries across every SASS arch (sm 61–120 per toolkit, both real+virtual per arch). Builds now emit SASS for the consumer set only (`61-real;75-real;86-real;89-real;120` on CUDA 12.8.1, `75-real;86-real;89-real;120` on 13.0) plus full SASS+PTX on the newest arch, so future GPUs JIT forward from its PTX — roughly a third of the nvcc work and a much smaller download. Datacenter archs (sm 70/80/90/100) stay reachable through the source lane (`pallama engine build cuda`). The CI link step also resolves CUDA driver-API symbols via the toolkit stubs (GPU-less runners have no `libcuda.so.1`; the stub's SONAME keeps the runtime NEEDED entry correct on real machines).

### Fixed
- Engine retention is now scoped per engine kind: installing a mistral.rs or sglang engine no longer prunes the newest llama.cpp builds (and vice versa) — each lane keeps its own newest `KEEP_TAGS`; active and `local` stay protected on top. Kind-blind retention deleted cross-lane engines on back-to-back installs (mistralrs install pruned the active CUDA engine; the CUDA reinstall then pruned sglang).
- **Interrupted large downloads no longer 416-loop (found live during the SGLang validation pull).** The parallel downloader preallocates `.part` files to full length (sparse holes); if its resume sidecar was lost mid-download, the next attempt declined the parallel lane and the classic lane sent `bytes=<full-length>-` against a sparse file — 416 with no self-heal. Now: a full-length `.part` without a sidecar is re-fetched in place by the parallel lane (chunk writes are idempotent), a short sidecar-less `.part` is handed to the classic lane from zero (hasher reset, no Range header), and only a mismatched sidecar is discarded. Pinned by wiremock 416-mount tests plus a live interrupted-pull rerun.
- **Test-only port TOCTOU (1-in-10 full-suite flakes).** Three engine tests asserted connect-fails on ports they had just bound-and-dropped; under parallel runs another test's port-0 bind could rebind the released port, flipping "dead endpoint" fixtures alive. Replaced by a `dead_port()` helper that verifies refusal (retrying on a different port if something rebinds) — 10/10 consecutive full-suite runs green.
- Engine probes (`--version`/`--help`/device census) can no longer deadlock or fail blind: `probe_output` drained piped stdout/stderr only after the child exited, so any probe target writing more than the ~64 KiB pipe buffer blocked on write, never exited, and surfaced as a misleading "timed out or failed to spawn" after a full 30 s deadline. Pipes are now drained on reader threads while the child runs (pinned by a 4 MiB flood test), and transient spawn failures (e.g. fork pressure) are logged with their `io::Error` instead of being silently folded into the timeout case.
- **`uninstall.sh --yes` no longer implies `--remove-models` (data-loss fix, found in the 2026-09-13 uninstall/install loop).** Flag parsing aliased `--remove-models | --yes` (last-flag-wins), so `--keep-models --yes` silently deleted every downloaded model. `--yes` now means "skip prompts using SAFE defaults (models kept)" and conflicts hard with `--remove-models`. Also: privilege preflight refuses non-tty unprivileged runs instead of half-deleting (systemd crash-loop split-brain); root-run uninstalls resolve the real user's home via `SUDO_USER` instead of `/root`; SQLite WAL sidecars (`pallama.db-wal/-shm`) are removed with the db; and a trailing `[ … ] && status` made flawless `--remove-models` runs exit 1 (a false test as the script's last command becomes its exit status — POSIX footgun, now an explicit `if` + `exit 0`).
- **`install.sh` now runs every user-state step as the invoking user.** Under `sudo`, toolchain probing, `migrate`, engine bootstrap, model pulls, and the source build previously ran as root — populating `/root` (rustup into `/root/.cargo`, root-owned `target/`, engine rows the service user could never see → permanent engine-less crash-loop). All steps route through an `as_user` wrapper; the health poll falls back to port 11435 (was 11434 — ollama's port, answered green while pallama was dead); fresh installs `enable` without `--now` and start the unit only after the engine bootstrap (previously measured 86 `no engine installed` restarts during the install window).
- Test-suite leaks: `unit__run_list_devices__parses_live_census_output` hand-rolled a `/tmp/pallama-census-<pid>` fixture dir and removed only the script file, and the engine-manager suite's `stub_engine_dir` staged pid-keyed dirs it never removed — together one leaked dir per `cargo test` run (200+ had accumulated). Both fixtures now own `tempfile` guards that clean up on drop.
- `pallama engine update` on an NVIDIA box now warns with the exact recovery command whenever the prebuilt CUDA lane is dropped — driver older than the overlay's newest asset (both versions named), overlay release not yet published for a fresh upstream tag, driver CUDA capability unprobed (reboot path), or pre-CUDA-12 drivers. These lane drops were `info`-level and easy to miss.

### Added
- **Per-arch CUDA engine assets (~60% smaller downloads).** The prebuilt CUDA channel now publishes one slim asset per GPU architecture (`llama-bNNNN-bin-ubuntu-cuda-13.0-sm89-x64.tar.gz`, 9 assets per tag across CUDA 12.8/13.0) instead of a fat multi-arch tarball; the sm120 asset carries PTX for forward JIT on future GPUs. `pallama engine update` picks the exact asset for the local GPU's compute capability, falls back to legacy fat assets during the transition, then to the PTX asset for GPUs newer than sm120. CI wall time drops in parallel (per-arch jobs ~15-20min with ccache vs ~50min serial fat build).
- mistral.rs profiles now reach the FULL `mistralrs serve` surface: every flag the probed engine binary offers forwards verbatim (bools like `--flash-attn` and valued pairs like `--dtype bf16` — attention method, dtype, KV-cache quant, prefix cache, token source, LoRA, device mapping, log control), gated by the same manifest capability check as the sglang lane. Explicit profile values beat derived ones (`--max-model-len` from the profile wins over ctx synthesis, no duplicates); llama.cpp-only dialect words are still dropped with the compiler's existing warning.
- **SGLang engine kind + safetensors model lane (2026-09-14).** `pallama engine install --kind sglang [version]` installs a pinned venv (`sglang==0.5.19` default; uv lane with `--prerelease=allow` for transitive pre-release pins, pip fallback; `ninja` installed and venv-PATH exported in the shim so flashinfer JIT can compile; ≥10 GiB disk preflight; orphan-dir cleanup on every failure path; non-Linux fails fast with a teaching error). `pallama pull <hf-repo>` gains a safetensors lane for repos without GGUFs: root-level shards + config/tokenizer allowlist land in `models/<name>.d/` with LFS sha256 verification, `.part` resume, shard-index coverage checks, and idempotent repulls; GGUF repos keep the existing lane and a repo offering both teaches which lane won. SGLang models serve through the same single-port gateway — `/api/chat`, `/api/generate`, OpenAI `/v1/*`, Anthropic `/v1/messages` — with zero client change and per-child auth. The profile compiler adds a low-VRAM ladder (full → KV fp8 → CPU offload bounded by host RAM → refusal with weights/KV/VRAM numbers — it can never emit a spawn that OOM-crash-loops), `--max-running-requests` from slots, EAGLE3 speculative pair, ~20 first-class tuning knobs under `models.<name>.sglang.*`, a reserved-flag guard on `extra_args` for lifecycle/security flags the ladder owns, and portability-first attention/sampling backend defaults (`triton`/`pytorch`, flag-gated and overridable) for boxes whose system nvcc can't satisfy flashinfer JIT. Unix child transport is refused with a teaching error (SGLang is TCP-only).
- **Overlay-lag fallback for CUDA engine updates (2026-09-13).** When the update channel resolves a target build that the prebuilt overlay has not published yet (upstream moved first, watcher up to an hour behind), `engine update` now installs the newest *published* overlay build the driver can run — never newer than the channel target, never the already-active tag — instead of demoting the box to Vulkan or the hours-long source lane. Exact-tag pins never fall back. With the hourly freshness watcher this bounds the CUDA-download lane to minutes of user time in the common case.
- `pallama engine prune` — manual trigger for the engine retention policy (same keep-newest + protect-active/local logic that runs automatically after each install); useful after lowering retention or cleaning up accumulated builds.
- Installer: `PALLAMA_UNIT_MEMORY_HIGH` sets a systemd `MemoryHigh` soft ceiling on the daemon cgroup (default `85%` of RAM, recomputed by systemd at every unit start — soft reclaim/throttle only, never an OOM kill; empty string omits the line).
- **Sentinel detection codes unified to snake_case everywhere
  (2026-09-13).** The `Code` enum serialized via serde derive as
  PascalCase (`ReasoningNoAnswer`) while every display/filter surface
  used its `as_str()` snake_case form (`reasoning_no_answer`) — the
  persisted `run/sentinel.jsonl` and doctor's offline scan counted the
  former, so doctor's own hint `pallama why --code ReasoningNoAnswer`
  matched nothing. `#[serde(rename_all = "snake_case")]` puts the
  derive path (persistence, doctor) on the same spelling as `/api/why`,
  the CLI, and the filter; existing JSONL histories migrate with a
  one-time rewrite (backup kept alongside). Pinned by a round-trip test
  over all nine codes.
- **`pallama engine update` no-op on source-built engines — now routes
  to the local build lane (2026-09-13).** With a source-built engine
  active (`built-cuda`/`built-cpu` asset) and no prebuilt overlay
  release published, `engine update` resolved the channel target,
  found nothing installable, and exited with "engine b…-cuda already
  active — nothing new installed" behind a dead-end warning (live:
  b10931-cuda active, b10936 target, overlay repo 404). Update now
  detects that state (source-built active + strictly newer upstream
  target + `engine_asset` not pinned to a prebuilt lane) and, when the
  toolchain is present, delegates to the `engine build` flow — same F7
  regression gate, progress lines, and restart hint; explicit `--tag`
  pins and channel downgrades keep the old behavior. Missing
  toolchain now prints the exact install hint instead of the vague
  lane warning.
- **`pallama why` observability dead-end + actionable
  ReasoningNoAnswer detail + validate-daemon orphan leak
  (2026-09-13).** `/api/why` scanned only the newest 100 ring records
  and hard-capped output at 10 — flagged records older than the latest
  clean batch were unreachable, making doctor's `pallama why` hint a
  dead end (live: 45 flagged requests invisible behind 10 clean ones).
  New `/api/why` params `flagged=1` and `limit=<n>` (default 10, capped
  at the 256-record ring) plus `pallama why --flagged/--code/--model/
  --limit`; filters apply before the newest-first cut. Doctor's
  sentinel hint is now directly runnable (`pallama why --flagged`,
  plus `pallama why --code <dominant>`). ReasoningNoAnswer detail now
  names the budget when usage proves it (`finish=length at 5
  completion tokens (reasoning consumed the budget)`). Validate
  sandbox daemons install a Linux PDEATHSIG parent-death guard
  (`PALLAMA_VALIDATE=1` only) and validate.py routes SIGTERM/SIGHUP/
  SIGINT through its atexit cleanup — a `timeout(1)` kill previously
  leaked daemons for hours (live: pid 1134882, 8h).

### Changed
- **Source builds now use ccache/sccache when on PATH.** Every build
  ran cold in a fresh tempdir (by design: probe-after-teardown keeps
  the install check honest), so each engine update recompiled all of
  llama.cpp (10-30 min for CUDA). `detect_toolchain` now also probes
  `ccache`/`sccache` and, when found, configure gains
  `CMAKE_{C,CXX,CUDA}_COMPILER_LAUNCHER` — repeat builds skip
  recompiling unchanged translation units while the per-build tempdir
  discipline stays intact. The cache is purely optional;
  `require_toolchain` never demands it.

### Added
- **`reasoning` config knob — llama.cpp's server-side reasoning
  switch.** `reasoning = "on" | "off" | "auto"` (default `""` = engine
  auto-detect from the chat template) maps to llama-server's
  `--reasoning` flag, global or per-model via
  `[model_overrides.<model>].reasoning`. This is the authoritative
  thinking kill-switch for templates that ignore the
  `thinking`/`enable_thinking` request variables (the ollama `think`
  toggle sets template vars, which qwen-family templates honor but
  e.g. harmony-class ones do not). Invalid values fail config load
  naming the `on|off|auto` set (overlay values validated too);
  default emits no flag, argv byte-identical. Completes Pallama's
  wiring of llama.cpp's full reasoning flag family
  (`--reasoning`, `--reasoning-format`, `--reasoning-effort`,
  `--reasoning-budget`, `--reasoning-budget-message`,
  `--reasoning-preserve`).
- **Ollama drop-in parity: `/api/generate` rides the full chat-bus
  pipeline.** The generate lane previously mapped to a bare
  `/v1/completions` post — no streaming (an SSE request hit a
  `.json()` parse and died), no images, no `system`, no `think`, no
  sentinel enforcement, no TTFT clock. It now translates through the
  same `/v1/chat/completions` core as `/api/chat` (admission,
  preflight, num_ctx restart, keep_alive, priority, sentinel
  enforce+observe, TTFT/TPOT, accounting, evict-after) with a
  generate-shaped wire: `system` becomes `messages[0]`, the raw
  `prompt` the user turn, `images[]` multimodal `image_url` parts
  (mime sniffed from magic bytes — PNG/JPEG/GIF/WEBP; unknown magic
  is a 400, never a silent mislabel), `think` maps to
  `chat_template_kwargs` exactly like the chat lane, and
  `stream:true` returns real NDJSON `response` deltas with a usage
  final line. `template`/`suffix` stay rejected with a teaching 400
  (the engine owns templates). One semantic change: raw generate
  prompts now get the engine's chat template applied — matching real
  ollama, whose generate is templated (raw completions remain one
  `POST /v1/completions` away).
- **Ollama chat `images[]` translate to multimodal parts.**
  Message-level `images[]` were copied verbatim to the child, which ignores
  unknown fields — vision routing (`@vision` respawn, mmproj attach)
  existed but the pixels never reached the model. `chat_to_openai`
  now converts them to `image_url` data-URL parts; text-only
  requests are byte-identical to before.
- **`/api/show` reports `capabilities`.** `["completion"]` always,
  `"vision"` when the store row has an `mmproj` path — the
  evidence-based source, no guessing. Ollama clients that probe
  capabilities (geokit's vision-model discovery) now see vision
  models.
- **`logprobs` mapped back in `/api/chat`.** Requests with
  `logprobs:true` passed through but the response never carried
  them back. Both the non-stream body and every stream chunk now
  clone `choices[].logprobs.content` to the top-level `logprobs`
  array (ollama-native field names — verifier clients parse it
  directly).
- **`pallama doctor` flags config pins that mirror retired defaults**:
  template pins outlive default changes — a config written when
  `cache_reuse = 256` or `spec = "off"` were the defaults silently keeps
  the old behavior after the defaults moved to `0` / `"auto"` (both bit
  live users). Doctor now emits one WARN row naming each stale pin, the
  retired value and the current default, teaching that deleting the
  line adopts the new default while keeping it is legitimate for
  deliberate pins (e.g. `spec = "off"` for bit-exact greedy). Global
  and `model_overrides` spellings both checked; a fresh/default config
  emits no row. The check registry is a maintenance contract: future
  default changes add their entry there.
- **Profile warnings surfaced to users**: compile-time decisions
  (unified-KV ctx fit, slot auto `-np`, gpu-offload rationale,
  cache-ram clamp, dense-fallback) were daemon-log-only. They now ride
  `/api/ps` rows as `pallama_warnings`, print as `[profile] …` lines
  after a `pallama run` stream, and as `warn[<model>]` lines under the
  `ps` table. The REPL prints them once after the first turn of each
  model (re-armed on `/model` switches).
- **`run` auto-pull**: `pallama run <model>` with a model missing from
  the store now pulls it first (the exact `pallama pull` flow —
  progress, resumable `.part`, pull locks, mmproj attach, warnings)
  and starts the chat once the download lands. Triggers only when the
  input parses as an `owner/repo[:QUANT]` ref or a catalog short name;
  store hits never touch the network, and unknown names keep the
  not-found teaching error. `pallama pull` itself is unchanged.
- **Unified-KV pool must fit the `--cache-ram` budget**: with
  `--kv-unified`, the whole KV pool + weights live inside the
  `--cache-ram` system-RAM budget — a default 32k ctx on a 1.7B model
  needs a ~7.3 GiB pool against a ~4.1 GiB budget and the engine child
  died at context creation regardless of free VRAM. The profile
  compiler now shrinks ctx (256-token steps, never below the 4096
  floor) until the f16 KV + weights fit the budget, marks the profile
  `ctx_autofit`, and warns loudly; a pinned ctx is never touched —
  it warns instead (raise the budget via model_overrides
  extra_args `--cache-ram`). Engine-crash errors now carry the
  child's last log lines (e.g. "failed to create context … cudaMalloc
  failed: out of memory") instead of a bare key.
- **Colon-name resolution (ollama muscle memory)**: every model-taking
  command (`show`, `run`, `rm`, `cp` source, `stop`, `bench`, `tune`,
  `mmproj`, `quantize`, `drafts`, `session`, `lora`) resolves
  `model:tag` input onto the flat store row (`qwen3.5:9b` →
  `qwen3.5-9b`), and the gateway's `ensure` choke point applies the
  same rule to API traffic (`/api/chat`, `/api/generate`, the OpenAI
  routes) — colon-gated, so canonical-name requests pay nothing. Exact
  rows always win; a miss on both forms errors with the input verbatim
  plus a flat-form teaching hint (`pallama names are flat; …-form
  would be its flat form`). `pull` is exempt — its colon is the
  `owner/repo:QUANT` separator. The rule lives in
  `Store::resolve_model_name` (one implementation, CLI + gateway).
- **`ps` names the card**: `/api/ps` rows carry `pallama_device` and
  the CLI GPU column renders `full@<card>` (e.g. `full@NVIDIA
  GeForce RTX 4070`) so you always see which card holds the model;
  router mode / unknown placement keeps the bare offload label.

- **Grouped top-level help**: `pallama --help` renders commands by
  category (Serve & Chat / Model Management / Tuning & Benchmarks /
  Engine & Config / Observability / Refused by design) instead of a
  38-row wall. clap has no native subcommand grouping (verified against
  clap 4.6), so a custom renderer reads descriptions and aliases live
  from the clap command tree — the enum stays the single source of
  truth, and `unit__grouped_help__covers_every_subcommand` fails
  `cargo test` if the grouping table and the enum ever disagree.
  Per-command help (`pallama help <cmd>`, `<cmd> --help`), error usage
  and shell completions remain clap-native.
- **`pallama why --watch`**: tails sentinel detections live (same lane
  as `pallama watch`); conflicts with a positional TRACE (loud clap
  error, not a silent precedence).
- **Refusal hint**: the local-only refusal stubs (`push`/`signin`/
  `login`/`signout`/`logout`) now also point at `pallama search` for
  GGUF discovery, not just `pallama pull`.

- **Prebuilt CUDA engine channel** (`engine-cuda` CI workflow + runtime
  overlay): upstream llama.cpp publishes no Linux-CUDA binaries, so the
  workflow compiles pristine upstream b-tags with `-DGGML_CUDA=ON` on
  CUDA 12.8/13.0 devel images (arch matrix per toolkit major, Blackwell
  included) and publishes them as `bNNNN-cuda` releases bundling
  libcudart/libcublas/libcublasLt + NVIDIA EULAs — target machines need
  only the driver, never a toolkit. On Linux-x86_64-NVIDIA,
  `pallama engine update` prefers them automatically on Linux-NVIDIA
  (zero-touch; `PALLAMA_ENGINE_REPO=owner/repo` points at a fork)
  prefers the highest `ubuntu-cuda-{X.Y}-x64` asset the probed driver
  can run (strict ceiling — an older driver stays on the Vulkan lane
  instead of risking an unbootable child); every miss (env unset,
  release absent, asset not runnable) falls back to today's behavior
  unchanged. `pallama engine install bNNNN-cuda` pins an overlay build
  explicitly; `engine_asset` overrides still win over the overlay.

### Fixed

- **Dead `--rpc` endpoint no longer crash-loops the child**: upstream
  llama-server connects RPC backends eagerly at argv-parse and SIGABRTs
  when one is down (ggml-rpc.cpp), so a `rpc_servers` entry pointing at
  a stopped worker turned every spawn into a crash-loop of 502s. Every
  `--rpc` endpoint in the final child argv (global knob, model overlay
  and `extra_args` occurrences alike) is now TCP-probed in parallel
  (2s budget) before the process exists: a dead one refuses the spawn
  with a teaching 500 naming the endpoint, with no child process, no
  engine rollback and no circuit-breaker trip — the engine binary is
  healthy, the worker is not.
- **Bank restore defers while the triggering request runs**: the
  detached slot-restore still queued the request that caused the
  spawn ~1s behind cache priming (the pending request was invisible
  to busy-polls). `ensure_routed` now brackets itself with a pending
  counter and the restore posts only at idle (100ms poll, 30s bound,
  bank preserved for the next spawn). Racing request: 5.44s vs the
  5.3s no-bank floor.
- **Session-bank restore leaves the spawn critical path**: the ~1s
  slot-0 restore POST runs detached behind an fs-only identity
  preflight, so Ready publishes at engine-health time (cold 9B
  5.37s vs ~6.4s) and non-racing first requests pay nothing;
  continuation chats still get the banked KV.
- **Evict teardown no longer pins a std lock across awaits**: the
  evicting mark is a `tokio::sync::Mutex` with an explicit async
  release (every exit path clears it; a leaked mark starved every
  future spawn of that name — the `/api/evict`-timeout wedge class
  observed once live). Concurrent `stop`+`run` churn on a 9B is
  clean 6/6 (was 3/6 with spawn failures).
- **mlock auto-policy charges resident siblings**: the 40%-of-RAM
  gate now counts other live instances' weights, so a churn
  replacement racing its dying predecessor's pinned pages degrades
  to plain mmap instead of double-pinning (2 × 5.4 GiB mlock on
  13.6 GiB RAM live-repro'd as spawn failures).
- **Bench warm cells assert GPU idle**: `gpu_busy_mib` recorded in
  cells.jsonl; contaminated runs self-label instead of silently
  halving both sides' decode numbers.
- **Sub-weights `--cache-ram` budget no longer CPU-splits the model**:
  the unified-KV budget floor now derives from rule 2b's own
  constraint at the ctx floor (weights + f16 KV + 64 MiB, over the
  0.85 compute share) and engages rule 2b at the raw budget — floored
  spawns stop hard-warning by construction and autofit ctx outcomes
  survive.
- **Bench cross-runtime prefill parity**: the ollama lane (no tokenize
  route) sized prompts by char estimate — 371 real tokens vs pallama's
  527 at the same target. Lanes now converge on the engine's own
  `prompt_eval_count`; re-measured same-token counts put pallama at or
  above ollama on every metric (prefill 1346 vs 1337 t/s, decode 41.1
  vs 40.5, ttft 123 vs 131 ms, qwen3.5-9b warm) — the reported 29%
  prefill gap was a measurement artifact.

- **Draft-decline keeps the gpu-layers pin**: the wave-5 unpin
  (draft pairs leave `--gpu-layers` to the engine fitter) now only
  applies when the draft will actually attach; a capacity-declined
  draft (running dense) keeps the pinned fast path instead of
  inheriting the fitter's conservative split.
- **Sub-weights cache-ram budget under unified KV floored at the
  weights** (3.3x decode fix): on a 13.6 GiB RAM / 8 GiB VRAM box the
  30% clamp (4102 MiB) sat below qwen3.5-9b's weights (5417 MiB) while
  `--kv-unified` was on, so the engine's live fitter honored the
  unsatisfiable sysmem budget and CPU-split the layers — 15.6 vs 39.9
  t/s, silently. The budget is now floored at weights + KV floor +
  64 MiB with a loud warning (measured numbers included); a floor over
  60% of RAM keeps the clamp and teaches `kv_unified = false` / smaller
  quant instead. Default-config 9B decode is 40.9 t/s (was 12.4;
  ollama 40.5 same-settings, direct ceiling 41.2).
- **Spawns plan against live census VRAM, not the boot snapshot**:
  single-GPU boxes fell back to the daemon-boot `Hardware` when no
  card pick applied, so a daemon started while a foreign context
  (e.g. the user's ollama model) held VRAM would forever CPU-split
  spawns and decline speculative drafts against stale free-VRAM
  readings (live-repro'd: 7.8 GiB free, planner saw 3.2 GiB). The
  profile now uses scoped-pick > fresh census > boot snapshot, and
  evict invalidates the census cache so our own teardowns are
  re-measured. Measured effect on qwen3-8b + draft pair: 28 t/s
  dense CPU-split → 45.8 t/s spec-accelerated full-GPU.
- **`run` parses its flags instead of swallowing them into the
  prompt**: `trailing_var_arg` made `pallama run m 'Say ok'
  --max-tokens 5 --verbose` send the flag TEXT to the model and
  never parse either flag (live-repro'd: every completion echoed
  "--max-tokens 5"; `--verbose` stats silently absent from the
  wave-6 TTFT notes). Flags now parse normally before the prompt; a
  leading `-` word needs quoting or `--`.
- **Circuit breaker counts crash restarts, not churn**:
  `record_restart` fired on every successful spawn, so a user
  stop→run churn ×4 inside the 60 s window opened the breaker on the
  5th and 503'd until `pallama ps --reset` (live-repro'd). Restarts
  now count only when the spawn consumes an unclean-death mark set by
  the child reaper (non-zero exit); clean teardowns (stop / idle /
  capacity) reset the mark — churn is a cold start, not a crash loop.
- **Draft KV charged in the co-residency planner**: the pressure and
  admission math counted only the dense model's KV
  (`kv_est_bytes`), so a speculative pair's draft KV (device-side,
  often LARGER than its weights — e.g. 2.2 GiB vs 0.35 GiB on the
  eagle3 8B pair) was invisible to card-capacity decisions. The
  profile now adds the draft's f16 KV at the resolved ctx (unified:
  on top of the floor; classic: on top of the quantized dense KV);
  an unreadable draft header degrades to dense-only with a daemon
  warning rather than blocking the spawn.
- **Speculative drafts no longer pin `--gpu-layers 999`**: the draft's
  weights + KV allocate DEVICE-side beyond any planner charge, and a
  hard pin disables the engine's live fitter (`n_gpu_layers already
  set by user to 999, abort`) — shared-GPU spec pairs could
  cudaMalloc-OOM. Spec pairs now leave `--gpu-layers` to the fitter
  (`auto`), in both unified-KV and classic accounting, with a profile
  warning saying so. Dense models keep the pinned fast path.

### Changed

- **Highest-perf defaults from the elimination audit** (2026-09-12,
  raw-engine elim sweep, 9B q4, fadvise-cold + GPU-idle asserted per
  config, 3 reps): `--load-mode mlock` auto-policy — empty `load_mode`
  now pins weights via mlock when they fit a 40% RAM share (eager
  page-in measured ~0.7s faster to first token than lazy mmap faults;
  explicit `load_mode` always wins; low-`RLIMIT_MEMLOCK` boxes degrade
  to a benign upstream warning + plain mmap), and `cache_reuse`
  defaults to 0 — the engine's native slot prompt-cache already covers
  identical prefixes (16x on re-ask, measured) while `--cache-reuse`
  cost ~0.6s on every cold load; opt back in for cross-slot prefix
  sharing. Measured free within noise and left unchanged: `--metrics`,
  `--jinja` (quality path), `-b/-ub` at llama defaults, threads 8/16/24,
  KV q8_0, ctx 4096 vs 16384 warm.

- **Cold-start TTFT measured**: methodology + numbers landed in README
  (Performance). At 0.5B the cold `run` wall is 1.16–1.94 s, all of it
  engine load (mmap + CUDA init); pallama-side stages (profile
  compile, health poll, admission) are milliseconds. A later 9B
  parity re-run at uncontended conditions measured pallama cold
  11.4 s vs ollama 5.7 s and warm decode 12.4 vs 27.2 t/s — traced to
  the capacity-first default profile (`-np 4` + wide `--ctx-size` +
  `--kv-unified` hosting a 256k-token KV pool in the system-RAM
  `--cache-ram` budget, PCIe-bound decode) against ollama's
  single-slot in-VRAM layout. That is a defaults-policy tradeoff, not
  a daemon overhead; a single-stream-first tuning pass is the open
  follow-up. Pin `slots = 1` / `kv_unified = false` (or
  `model_overrides`) for ollama-shaped single-stream speed today.
- **Help branding de-ollama'd**: top-level about, help footer, and the
  four refused-command descriptions (`signin`/`login`/`signout`/
  `logout`) now say "pallama" instead of "ollama.com"/"ollama-grade".
  Factual wire-compat references (the ollama API dialect the gateway
  speaks, `OLLAMA_HOST`, port-11434 hint) intentionally keep the name.

- **Golden harness**: all three `--help` consumers (commands registry
  check, gates manifest cross-check, goldens capture) now share one
  parser, `_help_command_names()`, which understands the grouped help
  format (same sorted name set as before — the byte-identical
  `help.commands` golden proves the renderer dropped nothing);
  regenerated goldens also absorb two latent pre-existing drifts —
  `doctor.check-names` (the `service` row was in the subtraction list
  but the golden predated it) and `config.fresh-keys` (`cpu_ffn_n`
  knob from the working tree).
- **Heterogeneous VRAM admission**: auto capacity is now bytes-based —
  a spawn is admitted while the SUM of resident instance weights plus
  the incoming model stays within total VRAM, replacing the old
  `floor(VRAM / largest_model)` heuristic that locked heterogeneous
  pairs out (0.5B + 9B co-reside on an 8 GiB card; a second 9B still
  evicts). Explicit `max_loaded_models > 0` keeps pure-count semantics,
  CPU-only boxes stay pinned to one instance, and the fresh free-VRAM
  probe (spawn guard) still owns refuse-vs-warn at load time.
  `Supervisor::capacity_for` replaced by `instance_cap` /
  `bytes_admission_active` / `resident_bytes` / `vram_budget_bytes`.

## [0.5.0] — 2026-09-10

Behavior-changing release: spawn-time concurrency, honest capacity math.

### Added (2026-09-12 demand-driven slot reshape — LC4 generalized, live-proven)

- **Demand decay**: after 30 quiet reaper ticks (5 min at zero in-flight),
  an adopted slot shape decays back to the natural capacity shape via the
  same idle-drain respawn (per-stream latency recovers: np8 ITL ~60 ms vs
  np4 ~37 ms). Asymmetric hysteresis by design — scale-up needs 60 s of
  proven pressure, scale-down needs 5 min of quiet; saturation re-raises
  in 60 s if demand returns.
- **Adaptive slots now react to real demand**: the trigger is a gauge over
  requests parked in the admission gate (`AllSlotsBusy` queue waits — the
  same-model concurrency park), not the unobservable `in_flight > slots`
  comparison; saturation across 6 reaper ticks (60 s) adopts `-np +1`
  (cap raised 4 → 8 on the measured np8 scaling win: conc8 sys +19%,
  queue-TTFT 5.4× better), and the reshape **respawns the instance at the
  first idle drain** (KV bank carries conversations) instead of waiting
  for the next natural spawn. Explicit `slots` overlays, replicas, and
  `deterministic` models are excluded; a spawn failure rolls the adoption
  back and clears the reshape queue.
- **`slot_cap()` live resolution** replaces the gateway's hardcoded
  admission cap of 4 (which silently defeated any adopted np5-8 shape):
  adopted value → live child argv `-np` → explicit config → default,
  with replica/vision key canonicalization.
- Live proof: 12-stream sustained load on a 4-slot instance → gauge
  fills (37-74 s parks) → adopt 5 → idle drain → respawn at `-np 5`,
  greedy outputs bit-identical before/after (zero quality loss).
- Four falsified designs on the way (in-flight comparison unreachable
  through admission control; event counter drained per tick; gauge
  bracketed on the multi-model-only `AllSlotsBusy` raise) — probes and
  artifacts in `/tmp/opencode/flagprobe/`.

### Added (2026-09-12 scaling flag-space study — measured, no default changes)
- **Flag-space study (b10903, qwen3.5-9b, RTX 4070)**: probed `-ub 2048/4096`, `--kv-unified-per-slot`, `-tb`/`-Crb`, `--poll 100`/`--threads-http`, and slot-shape scaling (`np8×32768` vs `np4×65536`) across short-prompt, long-prefill (2k/8k), and 8-stream sustained regimes. Greedy parity 5/5 and 2/2 on every config — all levers are scheduling-only, zero quality impact. Verdicts: `-ub` above the 512 default is a measured REGRESSION on long prefill (2111 → 1728 t/s at 8k, +360 MiB) — default stays; slot scaling is the concurrency lever (np8: system +19%, TTFT p99 5.4x better at 8 streams, −246 MiB; np4 keeps per-stream ITL 37 vs 60 ms) and the existing capacity-probed auto-slot rule already picks the balanced shape; `--poll 100` trims conc TTFT p99 ~11% at the cost of CPU busy-spin — documented as a high-concurrency recipe (`model_overrides.<model>.extra_args`), not a default.

### Added (2026-09-11 benchmark-parity wave)
- **MmprojPolicy tri-state (`mmproj` model override + global
  `mmproj_policy` knob), default `lazy`**: text-only cold spawns skip the
  projector read entirely (measured: 875 MiB file, ~3.0 s off cold TTFT
  on the 9B VL row, 1126 MiB VRAM freed); the first vision request
  triggers `ensure_vision` — a projector respawn under an `@vision`
  instance key with the KV bank carrying the conversation. `attach`
  restores the old always-on behavior, `skip` is text-only with a loud
  teaching error on vision requests, `extra_args -mm` still wins, and
  router mode always attaches. Accepts both `mmproj = false` and
  `mmproj = "lazy"` spellings.
- **Spec-decode capacity gate (rule 11)**: catalog draft pairs are now
  gated by measured VRAM — draft file + model + KV floor + spawn
  overhead vs the picked card's free MiB; insufficient → dense with a
  full teaching warning (all numbers in the log), speculation engages
  automatically on cards that fit both. Plus a self-draft guard (a draft
  equal to the main model degrades dense) and a shipped
  `qwen3.5-9b → draft-mtp` catalog pair
  (`unsloth/Qwen3.5-9B-MTP-GGUF`, resolves to a local registry row when
  pulled).
- **Vision-aware admission**: `body_needs_vision` gateway scan across
  all four request shapes (`OpenAI` chat, `OpenAI` responses, Anthropic
  messages, ollama chat/generate `images`), threading a `needs_vision`
  flag through `ensure_with_admission` into the lazy-attach path.
- **Cold-start off the spawn critical path**: the settle census
  (`--list-devices` subprocess, ~100-800 ms) now runs in a detached task
  after the instance is published — first-request latency no longer pays
  it; admission charges the weights floor until the measured settle lands
  (same guarantee as before, later timing).
- **Adaptive health polling** in both engine health checks (llama.cpp,
  mistral.rs): 25 ms start, ×1.6 backoff capped at 150 ms — post-ready
  overshoot drops ~75 ms → ~12 ms.
- **Benchmark cold/idle/ctx lanes** (`scripts/bench_matrix.py`): fadvise
  page-cache drop + GPU-idle assert before every cold probe (all
  runtimes), pallama `cold_ttft_ms` capture, ollama cold lane
  (`load_duration` counters, optional systemd-restart daemon-boot metric
  via `BENCH_SUDO_PASSWORD`), idle-wake lane (pallama sleep ladder vs
  ollama keep_alive expiry, policy verified via `/api/ps`), long-ctx
  curve lane, sustained concurrency rounds with p99 tails, ollama
  concurrency parity cell, store-row-first model pick (scratch files can
  no longer 404 gateway lanes).
- **`model_overrides.<model>.mmproj = false`** — suppress the row's
  projector for text-only spawns (projector cold read is physics: 875 MiB
  ≈ +0.3-0.5 s cold TTFT on the 9B VL row). Loud teaching warning on
  skip; explicit `extra_args -mm` still wins; vision requests on a
  suppressed spawn fail loudly.
- **Census TTL cache** — `live_hardware()` caches the `--list-devices`
  census for 10 s (repeated spawns in a live daemon skip the
  ~0.2-0.26 s subprocess); invalidated on any engine spawn failure so a
  stale census can never gate admission after a failed boot.
- **Harness mmproj fadvise parity** — cold probes drop the page cache on
  the model file AND its projector (a cached projector handed pallama a
  warmer "cold" start than text-only ollama; both files now disk-cold).

### Added
- **Slots auto-fit** (throughput-first parity with ollama's VRAM-tier ctx
  default): when `slots = auto` resolves to ONE slot and the ctx came from
  `default_ctx` (not pinned by model overlay, bench tuning, or a per-request
  `num_ctx`), the same total-ctx capacity budget is re-spent as parallel
  shallow slots — e.g. 1x16384 becomes 4x4096 (floor 4096, ollama's own
  default ctx). Identical VRAM/KV budget under both the unified per-token
  guard and classic KV math (N x ctx/N == 1 x ctx), so boot risk is
  unchanged while concurrent streams batch instead of queueing (measured
  37 -> 98 t/s system at 4 streams on the tight lane). Prompts longer than
  the per-slot ctx take the existing teaching 400 + `num_ctx` restart-once
  path. Surfaces a `slots_ctx_auto_fit` event, `Profile.ctx_autofit`, and a
  loud spawn warning; any ctx pin disables it.
- `deterministic` config knob (global + per-model override, default false):
  pins slots = 1 in both engine lanes so greedy decoding reproduces
  token-for-token — multi-slot batches perturb logits in near-tie
  positions (measured: auto-slots np=4 flipped 14/20 greedy probes vs
  the same child at slots = 1). An explicit `slots > 1` under the pin
  loses loudly (warning), matching how vLLM/OpenAI document
  temperature-0 non-determinism under batching.
- Auto-slots conservative cap for vulkan-class builds: when the GPU
  census mixes integrated + discrete devices (the vulkan enumeration
  signature; CUDA builds list CUDA devices only) AND a projector is
  attached, slot count charges a measured per-token compute-buffer term
  (24 KiB/token of total ctx) against VRAM headroom. Measured on an
  8 GiB card (Qwen3.5-9B Q4_K_M + mmproj, `--kv-unified`): total ctx
  16384 boots clean (88 t/s system at 4 streams), 32768 boots degraded
  (47 t/s, 27 s load), 65536 OOMs or crawls at 96% commitment — while
  projector-free vulkan spawns and CUDA builds with the projector stay
  healthy at the same totals, so only that combination pays the charge.
  A teaching warning names the cap and the overrides.
- **`spec = "mtp"`** (opt-in): emits `--spec-type draft-mtp` for GGUFs whose
  weights ship the multi-token-prediction head (no draft model to pull —
  distinct from `spec = "auto"` pairing and `ngram` self-drafting). Engines
  that don't list `draft-mtp` among their `--spec-type` values fail fast at
  profile-compile naming `pallama engine update`. Verified absent in the
  common Unsloth Qwen3.5-9B Q4_K_M GGUF, so it stays strictly opt-in; bench
  before adopting (ngram precedent: unmeasured speculation can net-lose).
- **`spec = "eagle3"`** (opt-in): pairs a trained EAGLE3 speculator draft
  (separate GGUF) via `--spec-type draft-eagle3 --spec-draft-model`,
  inheriting the full draft-placement knob set. Catalog gained a typed
  eagle3 pair for `qwen3-8b` (williamliao/Qwen3-8B-EAGLE3-Speculator-GGUF,
  verified ungated); typed lookup keeps plain `auto` on its existing
  draft-simple pair — precedence unchanged. Engine without `draft-eagle3`
  or unpulled draft = hard error naming the fix, never silent dense.
- **`spec = "dflash"` / `"dspark"`** (opt-in): block-diffusion drafter
  pass-through on the same two-file pairing as eagle3 (typed catalog pair +
  `--spec-draft-model`), gated on the engine advertising the spec type.
  b10896 advertises both; no catalog pairs until
  verified artifacts exist — the lanes error honestly today and light up
  with a verified draft.
- **`lazy_mode`** (`"auto"` default, global + per-model override): tensor
  residency control — `--lazy-mode` emitted only on deviation from the
  engine default (auto = >4 GiB tensors on-demand from disk via mmap; on =
  all such tensors lazy; off = fully resident). Big-MoE RAM relief;
  engine without the flag degrades to a teaching warning, never an abort.
- **Server-side tools passthrough** (experimental, global-only quartet
  `server_tools` / `server_tools_runtime` / `mcp_servers_config` /
  `mcp_servers_json`): opt-in wiring of upstream's experimental agent
  tooling (`--tools`, `--tools-runtime`, `--mcp-servers-config`,
  `--mcp-servers-json`). Runtime values (`docker:<image>`, `podman:<image>`,
  `docker-container:<id>`, `podman-container:<id>`, `ssh:<host>`) are
  validated prefixes passed through — pallama itself never requires docker.
  The MCP config path is existence-checked at profile-compile (fails before
  boot); the inline JSON is syntax-checked at config validation and the two
  forms are mutually exclusive. Global-only deliberately: tool/security
  posture must not vary silently per model. Engines without the flags
  degrade to teaching warnings. Trusted environments only — the tool set
  includes `exec_shell_command`.
- **MTP auto-detection** (`spec = "auto"` extension): GGUF metadata
  `n_predict_layers` (llama.cpp) / `nextn_predict_layers` (ollama
  converter) — both parsed, first non-zero wins — marks weights with a
  baked-in multi-token-prediction head. Auto lane emits the ollama-parity
  trio (`--spec-type draft-mtp --spec-draft-n-max min(layers,2)
  --spec-draft-backend-sampling`) when the engine advertises draft-mtp,
  superseding the catalog draft-pair; engines lacking it warn-skip to the
  pair path. Manual `spec = "mtp"` now warns on headless GGUFs (engine
  boot would fail). `spec = "off"` never speculates.
- **Pull collision guard**: two repos shipping the same leaf filename
  (verified live: unsloth base vs MTP GGUFs both ship
  `Qwen3.5-9B-Q4_K_M.gguf`) silently clobbered each other on pull; flat
  collisions now disambiguate with a repo-slug prefix
  (`unsloth--Qwen3.5-9B-MTP-GGUF--Qwen3.5-9B-Q4_K_M.gguf`).
  - **Client-disconnect mid-spawn no longer wedges a model** (live-repro'd:
    one timed-out generate during cold load left every later generate on
    that model hanging forever). Loads now run on a detached task —
    `JoinHandle` drop detaches, so the spawn always completes — and the
    supervisor's loader protocol gained a cancellation guard that wakes
    parked waiters with a loud error instead of stranding the `Notify`.
    Product semantic: a load once started runs to completion (like
    ollama), only the disconnected request stops waiting.
- **dflash/dspark catalog pairs + exact-filename pulls**: the quant slot
  of a pull target may now carry an exact `.gguf` filename
  (`pallama pull ggml-org/Qwen3-8B-GGUF:dflash-Qwen3-8B-Q8_0.gguf`) — how
  drafter artifacts sharing a quant token with their target get pulled
  unambiguously; such rows are named after the file stem so they never
  clobber the model row. When several files DO share a requested quant
  token, the largest (the model, not its drafter or an mmproj) wins with
  a warning naming the skipped files. Catalog: `qwen3-8b` shortname now
  points at unsloth (full quant family); dflash/dspark spec pairs
  reference the ggml-org block-diffusion drafters. Measured honestly:
  dflash -17% vs dense on prose (BENCHMARK.md) — strictly opt-in.
  Registry pulls also log when the GGUF carries an MTP head (ollama-side
  conversions merge it; `spec = "auto"` then enables draft-mtp).
- **registry.ollama.ai pull lane**: `pallama pull qwen3:0.6b` (or explicit
  `registry.ollama.ai/ns/model:tag[@sha256:…]`) pulls straight from the
  ollama registry — Docker-v2 manifest parsing (tolerating the lying
  `text/plain` content-type), sha256-verified resumable blob downloads via
  307→presigned-Cloudflare-R2 redirects, wildcard redirect allowlist
  (`*.r2.cloudflarestorage.com`, apex excluded), optional
  `PALLAMA_REGISTRY_TOKEN` attached only to the first-party registry host.
  Model + projector layers land as plain GGUF files under the store's
  collision-safe naming; template/params/license/adapter layers are
  skipped with a log line (the GGUF's own template/sampler defaults are
  authoritative in pallama). Digest pins verify the manifest body hash.
  Gateway `/api/pull` and CLI share the same router (`route_pull`).
- **`docs/registry-ollama-pull.md`** — implementation spec for pulling
  directly from registry.ollama.ai (wire protocol live-verified 2026-09-10:
  Docker-v2 manifest envelope, `…ollama.image.*` layers, 307→Cloudflare-R2
  signed blob redirects, no tags/list). Closes the spec half of the
  ecosystem-gravity gap; implementation follows the same security
  discipline as the HF lane (allowlisted redirects, sha256-verified,
  resumable).
- **`cpu_ffn_n` knob** (`--n-cpu-ffn`, global + per-model overlay + `PALLAMA_CPU_FFN_N`):
  dense-model twin of `cpu_moe_n` — keep the first N layers' FFN weights on
  CPU for fine-grained VRAM trading on dense models (0 = off; `Some(0)`
  overlay is an explicit off per the F114 discipline). Emitted only when
  nonzero; registered in the argv spawn groups and fresh-keys surface.
- **Full sampler surface on ollama request options**: `options` now accepts
  every llama-server-native sampler field 1:1 — XTC (`xtc_probability`,
  `xtc_threshold`), `top_n_sigma`, `logit_bias`, the DRY family
  (`dry_multiplier`/`dry_base`/`dry_allowed_length`/`dry_penalty_last_n`/
  `dry_sequence_breakers`, the latter shape-checked as a non-empty string
  array), mirostat (`mirostat`/`mirostat_tau`/`mirostat_eta`),
  dynatemp (`dynatemp_range`/`dynatemp_exponent`), adaptive
  (`adaptive_target`/`adaptive_decay`), the `samplers` chain, and
  `adaptive_p: true` which joins the sampler chain post-pass (so an
  explicit `samplers` list and `adaptive_p` compose regardless of key
  order). Field names verified against the b10896 request schema +
  live /props; unknown keys still 400 listing them.

### Changed
- **CUDA dethrone guard + build advisory**: installing/updating a non-CUDA
  llama.cpp engine on an NVIDIA box no longer auto-DEMOTEs an installed
  CUDA engine (the previous hardware-blind activation silently cost
  ~18% first-token latency on this class of card when a Vulkan engine
  update landed); `pallama engine use <tag>` still switches explicitly.
  NVIDIA + Vulkan-only active + no CUDA engine installed now logs a
  one-time teaching advisory: `pallama engine build cuda` (one-time ~7 min
  in-tree build; upstream ships no Linux CUDA prebuilts).
- **`spec` default is now `"auto"`** (was `"off"`): an opportunistic lane —
  `draft-mtp` when the GGUF ships an MTP head (measured +50% decode),
  the catalog draft pair when pulled, dense-with-teaching-warning when
  the pair is not (auto never refuses a spawn; hard errors belong to the
  explicit typed modes). Target-side verification keeps output
  lossless. Matches ollama's own MTP auto-enable posture; `spec = "off"`
  still forces dense.
- **Spawn capacity now sizes against live free VRAM** (`capacity_bytes`):
  profile math (slot/KV/offload sizing, paged-attn auto-off) previously
  compared against *total* VRAM while the engine's `--fit` refuses to
  shrink an explicit `--gpu-layers` pin — a busy card (fresh probe
  free ≪ total) produced a 999 pin that fit could not correct and the
  child OOM'd. Capacity now means free-when-probed (idle behavior
  unchanged; a zero probe fails open to total).
- **Validation consolidated into `scripts/validate.py`** — the manifest
  registry (command/knob coverage tables, formerly
  `validate_manifests.py`) is merged in as the MANIFEST REGISTRY section;
  `--phase=manifests` echoes registry sanity (81 command paths, 136 knobs,
  21 model-override fields). One validation script, same gates.
- **Benchmarking consolidated into `scripts/bench_matrix.py`** — the single
  entry point. It now also renders the publication-format report (`--md`,
  or `--render-only` to re-render an existing campaign's `cells.jsonl`
  without re-measuring); the internal forensics report stays at
  `<artifacts>/benchmark.md`. Supersedes and removes `bench_compare.py`,
  `bench_engines.py`, and `soak.sh` (the `--soak` flag covers sustained-load
  probing inside the isolated harness).
- **`slots = 0` (new default) is now pallama capacity-aware auto** instead
  of the upstream `-np -1` passthrough: the total `--ctx-size` scales so
  every slot keeps the resolved per-slot context, and the slot count
  clamps to the KV budget (cache-ram under `--kv-unified`), the model's
  trained context, and — on the classic non-unified path — the 85% VRAM
  envelope, capped at 4. Concurrent streams batch on the GPU instead of
  queueing on one slot (measured ~2.5x system throughput at 4 streams;
  single-stream speed unchanged). Pin `slots = 1` for the previous
  single-client behavior. `Profile.ctx` now reports the per-slot context
  (the argv `--ctx-size` is the scaled total).
- `Hardware::total_vram_mib()` counts discrete GPUs only when any exist:
  integrated GPUs report shared system RAM as VRAM, and summing them
  invented capacity (a vulkan census of [iGPU 10 GiB, 8 GiB discrete]
  read as 18 GiB and over-pinned offload decisions). Integrated-only
  boxes keep the integrated sum.
- KV charges for GPU-capacity decisions (co-residency planner, auto
  tensor-split, `Profile.kv_est_bytes`) are now layout-aware: the
  measured 512 MiB working-set floor under `--kv-unified` (the buffer
  lives in the `--cache-ram` system-RAM budget) instead of the full f16
  estimate that read as 2-4x the real VRAM demand.
- The per-request `num_ctx` VRAM preflight and `pallama coreside` use the
  same unified-aware charge (`kv_unified_for`, single decision source
  with profile compilation): legal `num_ctx` bumps on unified spawns no
  longer read as OOM and refuse; `coreside` plans unified models at the
  512 MiB floor instead of deferring them on phantom pressure.
- `adaptive_slots` now defaults to true (sustained concurrency on a
  1-slot model still auto-adopts `-np +1` in memory, cap 4; set
  `adaptive_slots = false` to pin single-slot until asked).
- mistral.rs engines without an explicit `mistralrs_paged_attn` now
  auto-disable paged attention when weights+projector occupy >75% of
  VRAM (upstream's paged-KV pool computes to zero there and the load
  fails with "Num GPU blocks is 0"); a teaching warning explains the
  fallback and the override.
- Gateway hot path parses each JSON request body once (model
  extraction, usage-flag injection, model-id rewrite, single-flight
  stream detection previously re-parsed the same bytes up to 3x), and
  per-request SQLite connections (open + pragmas at ~16 gateway sites)
  are replaced by one lazily-reused connection; `[[keys]]` secret
  comparison is constant-time. No wire behavior changes.

### Removed
- `engine_pin` config key (+ `PALLAMA_ENGINE_PIN` env override): dead
  surface — written and documented but never read since engine pinning
  moved to the engine marker (`pallama engine use <tag>`). Configs
  carrying the key fail fast with a TOML parse error naming it; delete
  the line to migrate.

### Fixed
- Gateway offload under-count (D1): the offload resolver double-charged
  full f16 KV while emitting `--kv-unified` (which hosts KV in the
  cache-ram budget), dropping tight-fit models into the engine's auto
  band and parking layers on the CPU — measured -22% decode. The unified
  branch now pins full offload on an absolute test (resident + 512 MiB
  KV floor + 700 MiB measured spawn overhead <= VRAM).
- `--cache-reuse` is no longer emitted on multimodal spawns (upstream
  silently disables the combination; the dead flag misled profile
  readers).
