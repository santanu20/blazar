# Changelog

All notable changes to Pallama are documented here. Format follows
Keep a Changelog; versions follow SemVer. Earlier releases were not
tracked here.

## [0.5.0] — 2026-09-10

Behavior-changing release: spawn-time concurrency, honest capacity math.

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
- **`docs/registry-ollama-pull.md`** — implementation spec for pulling
  directly from registry.ollama.ai (wire protocol live-verified 2026-09-10:
  Docker-v2 manifest envelope, `…ollama.image.*` layers, 307→Cloudflare-R2
  signed blob redirects, no tags/list). Closes the spec half of the
  ecosystem-gravity gap; implementation follows the same security
  discipline as the HF lane (allowlisted redirects, sha256-verified,
  resumable).

### Changed
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
