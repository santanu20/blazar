# Changelog

All notable changes to Pallama are documented here. Format follows
Keep a Changelog; versions follow SemVer. Earlier releases were not
tracked here.

## [Unreleased]

### Added
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
- **Cold-start TTFT measured, load-bound (0.5B)**: methodology +
  numbers landed in README (Performance). tl;dr — cold `run` wall
  1.16–1.94 s on the RTX 4070 laptop, all of it engine load (mmap +
  CUDA init); pallama-side stages (profile compile, health poll,
  admission) are milliseconds. The earlier 8.5 s-vs-6.3 s "gap"
  against ollama was an 8B-class model under a contended box — not a
  pallama-side regression. No code change: measurement first, nothing
  ours to fix at this size.
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
