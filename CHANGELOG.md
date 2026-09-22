# Changelog

All notable changes to Pallama are documented here. Format follows
Keep a Changelog; versions follow SemVer. Earlier releases were not
tracked here.

## [Unreleased]

### Added

- **First-class `sdcpp` engine lane: diffusion GGUF serving via stable-diffusion.cpp.** A fourth engine family wraps leejet/stable-diffusion.cpp's `sd-server` (prebuilt releases, Vulkan-first so one asset covers NVIDIA/AMD/Intel; `blazar engine install --kind sdcpp`). Pulling a known diffusion family fetches the whole component set as ONE model with per-file resume and sha256 verification, re-using any byte-exact files already on disk. Components are flag-keyed per family — Qwen-Image-2.1 (`--vae` + `--llm` text encoder + optional `--llm_vision` for edits) and FLUX.1 (`--vae` + `--t5xxl` + `--clip_l`; matches dev/schnell/mirrors, excludes FLUX.2) — stored as a JSON list in one `components` column (schema v6; v5 databases fold their legacy per-flag columns in place, v4 databases upgrade through both steps). Routing gains a bidirectional domain gate: component rows can only route to `sdcpp`, text models can never route to it (`DiffusionUnserved` teaching names the install command when no sdcpp lane exists). The gateway exposes `POST /v1/images/generations` and `POST /v1/images/edits` (OpenAI images shape, multipart relayed untouched) through the same admission/ensure/supervision path as chat; edits on a family without a vision encoder are refused with a family-aware teaching instead of a re-pull loop; `blazar run` refuses component sets with teaching instead of spawning a child that would 404 every send. The teaching is bidirectional at the gateway too: posting a component set to any text or embedding surface (`/v1/chat/completions`, `/api/chat`, `/api/generate`, `/v1/embeddings`, `/v1/messages`, rerank...) answers 400 with the images endpoints instead of booting sd-server into a guaranteed 404. Health checks use sd-server's real ready signal (`/v1/models`), and VRAM-based `--offload-to-cpu` placement follows the same 75% ladder as mistral.rs (resident budget = DiT + every component file) so 8 GB GPUs can serve 9 GB component sets.
- **Diffusion-component GGUFs now fail with teaching instead of a raw parse error.** Image-repo GGUF splits (DiT / text-encoder / VAE components, e.g. `Qwen-Image-2.1-GGUF`) carry no `general.architecture`, so no llama.cpp lane can ever load them — previously a chat attempt surfaced `gguf metadata: config: gguf: missing general.architecture` with zero guidance. The daemon now names the file as a diffusion/model component and points at the `sdcpp` lane (install + re-pull to fetch the set); `blazar list` marks such rows `†` with a dedicated footer line, and `list --json` reports `"engine_arch_gap": "(no architecture metadata)"`. Pull still succeeds — the component files remain on disk for the lane.
- **Engine listings now predict the capability-rescue lane.** The ENGINE field in `blazar list`, `/api/tags`, and `/v1/models` mirrors the router's lane choice — but the supervisor re-routes a spawn to an installed lane that advertises the model's GGUF architecture after the picked lane rejects it, so with a fork lane installed the previews showed a mainstream tag that would crash first. All three previews now show the lane that will ULTIMATELY serve, through the same preference rule the spawn rescue uses (one core `advertising_lanes` fn: mainstream beats fork, newest-first within class). A user pin is never overridden (the spawn rescue has the same gate), and rows whose arch set was never mined stay unpredicted (honest unknown). Live-proven end-to-end: with a lane advertising `instella-moe` installed, all three listings flip to it and the spawn goes straight to that lane with no wasted mainstream crash.
- **`list` now marks engine-architecture gaps instead of implying support.** The ENGINE column shows the routing lane (format/policy/pin), which says nothing about whether that engine's build can load the model's GGUF architecture — an unknown architecture crash-classifies at spawn. A model whose architecture is provably absent from the routed llama.cpp lane's mined arch table now renders as `tag†` with a teaching footer (fork-lane rescue via `blazar engine offers`), and `list --json` rows carry an additive `engine_arch_gap` field for the same case. Rows whose answer is unknowable (pre-v2 manifests with no arch table, non-llama.cpp kinds, models without a parsed arch) and all healthy rows are byte-identical to before.

### Fixed

- **The engine help census no longer drops underscore flags.** The shared `--help` parser filtered flag lines through a dash-era charset, so sd.cpp's five underscore flags (`--llm_vision`, `--clip_vision`, `--qwen2vl_vision`, `--clip_g`, `--clip_l`) never reached the engine manifest — and the strict probed-flags argv gate then refused `--llm_vision` at spawn, silently degrading `/v1/images/edits` to a confusing upstream "generate_image returned no results". The charset now admits `_`, the five flags probe cleanly (152 → 157), and existing installs recover by re-running `blazar engine install --kind sdcpp` (re-probe in place).
- **Memoryless device census rows never reach the GPU pool.** sd.cpp's `--list-devices` prints accelerators without memory numbers (all `total_mib = 0`); copied verbatim into the hardware census they made the GPU picker tie-break on zero and hand a Vulkan device name (or worse, the host `CPU` row) to a CUDA llama-server, which crashed at spawn (`invalid device: CPU`). Census rows without memory totals are now treated as "not a capacity census" — the picker falls back to nvidia-smi when the whole pool is memoryless, and the sd.cpp device parser drops the host CPU row outright (it is not an accelerator). The sdcpp offload ladder also gets real VRAM numbers back for its placement decisions.
- **Component re-pull re-uses byte-exact files even when the store name differs.** The reuse check ran on the store's collision-safe (repo-prefixed) destination name, so a hand-placed byte-exact file under its bare leaf name missed the check and triggered a full multi-GiB re-download next to its own copy — on a nearly-full disk that aborted the whole pull. Reuse is now checked against both the bare leaf and the resolved destination (size gate, then sha256 when known), pinning the truth table.
- **A diffusion-family row without its component set is a repair, not "already present".** The re-pull gate treated a never-attached set (`NULL` components) as fine and the repair path only fired when at least one component path existed, so `blazar pull` on a family-known row reported present without attaching the set. Any row whose repo matches a known diffusion family now requires every required component file to exist; a fixed-binary re-pull attaches the missing set and leaves the DiT untouched.
- **A 0-KV component GGUF on disk is intact, not corrupt.** The re-pull integrity gate required the GGUF metadata parse to succeed, but diffusion DiT files never carry `general.architecture` — sound on-disk sets were pruned as failed-integrity and re-downloaded in full. The gate now walks the container structurally: the exact "missing general.architecture" parse error means the header, KV section, and tensor table all parsed cleanly, so the file is byte-structurally fine for the sdcpp lane.
- **`doctor` port check now verifies daemon identity, not just liveness.** A bound port answering `/healthz` used to be reported as "blazar daemon already answering" — during the rename window a pre-rename pallama daemon (same default port 11435) earned a false ok. The daemon's `/api/version` now advertises `name: blazar`; doctor reports ok only for a named blazar answer, warns with the observed version for a pre-rename/foreign daemon, and keeps the occupied-port hint otherwise.

## [0.10.0] - 2026-09-21

### Changed

- **Project renamed: `pallama` → `blazar`.** Crates (`blazar-core`, `blazar-runtime`, `blazar-gateway`, `blazar-cli`), the binary, the `BLAZAR_*` env prefix, XDG dirs (`~/.config/blazar`, `~/.local/share/blazar`) and every install URL now carry the new name. Pre-1.0 break, no migration: existing installs should re-run the install script and re-pull models from their plain `.gguf` files. The repo rename keeps old GitHub links redirecting; this changelog keeps the historical name in older entries.
- **The product repo no longer hosts llama.cpp builds.** The legacy `engine-cuda` + `overlay-freshness` workflows (hourly cron) published upstream llama.cpp CUDA builds as `bNNNN-cuda` releases here — third-party binaries on the product release page, consumed by no default code path. Deleted outright. The prebuilt CUDA channel consumes upstream `ggml-org/llama.cpp` official ubuntu-cuda assets directly (same-release first, then scan-back); `BLAZAR_ENGINE_REPO` remains the supported opt-in for operators who publish their own overlay builds.

### Fixed

- **Parallel-lane pulls now resume at byte granularity instead of restarting from zero.** The parallel download lane (default ≥8 connections for files ≥32 MiB) split files into 8–128 MiB chunks but persisted its `.part.progress` resume ledger only when a whole chunk finished — with chunks spread across connections, none finished until the file was nearly complete, so any interrupt (Ctrl-C, network drop, daemon kill) left a full-length sparse `.part` with no ledger and the next pull hit the orphan heuristic and re-fetched every byte. The sidecar is now v2 with an aria2-style per-chunk ledger: it is seeded the moment the lane engages, workers count only bytes that survived a positional write (refunding counters when a retried attempt rewinds), and the coordinator snapshots every pending chunk's verified prefix on a 1s cadence plus immediately on each chunk completion. A re-pull continues each chunk exactly at its persisted prefix — verified live against registry.ollama.ai (three interrupted sessions of qwen3:0.6b, 523 MiB: 22 MiB → 246 MiB → 260 MiB, every chunk prefix strictly growing, zero orphan re-fetches). v1 sidecars (done-chunks only) load unchanged and keep their guarantees; the classic single-stream lane and the orphan path (full-length `.part` with no ledger at all — sparse holes are unresolvable) are untouched.

### Added
- **`pallama pull <target> --verify`: re-hash a pulled model against its recorded sha256.** Verify-only mode — no network, no pull lock: streams the file through sha256 on the blocking pool and exits non-zero on a mismatch, printing the expected and actual digests plus the file path. Registry pulls (content digest), HF GGUF pulls (LFS digest) and `model:tag` aliases all resolve; safetensors directory rows are refused loudly (their stored hash is a revision identity, not a content digest) and rows without any digest are reported as unverifiable instead of failing. The `--verify` flow ran live against the interrupted-pull test bed: the completed qwen3:0.6b (498 MiB) verified exit 0 with the digest matching the registry manifest byte-for-byte, a single flipped byte flipped the verdict to exit 1, and restoring it restored exit 0.
- **Ledger persistence is now crash-ordered.** `store_sidecar` fsyncs the `.part` data before writing the ledger, fsyncs the ledger tmp file, renames atomically, and fsyncs the parent directory — a power loss can no longer leave a sidecar claiming bytes the `.part` does not actually hold. The fsync steps degrade with a warning (never abort the download) on filesystems that refuse sync; a missing `.part` still gets its ledger written, which a regression test pins.
- **`pallama search --quant <q>`: filter search results by quantization on each repo's real file set.** The Hub exposes no quant facet, so the filter runs client-side against the same quant set the QUANTS column renders (per-file GGUF tokens, repo-id bit-width markers for safetensors/MLX lanes — filter and table can never disagree). `--quant q4` matches the whole Q4 family (`q4_0`, `q4_k_m`, `q4_k_xl`, `iq4_xs`, …), `--quant Q4_K_M` that exact quant, comma-lists OR (`q4,q5`), and matching is case-insensitive; an `iq4` filter stays i-quant-only. A bare query token that is itself a quant (`pallama search qwen image 2.1 gguf q4`) is lifted out of the Hub full-text query into the filter automatically — such tokens are dead weight server-side — with a stderr notice (stdout stays clean for `--json` pipes); `--format` composes independently. Quant token extraction also widened, verified against a 4,346-file live Hub scan: double-extension layouts (`Q4_K_M.GGUF.gguf`, ~5% of scanned files) and quant-in-folder layouts (`Q4_K_M/model.gguf`) now yield tokens in both the search table and the `pull <repo>:quant` path, which shares the grammar.

## [0.9.1] - 2026-09-21

### Fixed

- **The keep-CUDA update guard now covers windows-x86_64 (CI windows lane had been red since the OS matrix was restored).** `keep_cuda_skip_pred` was hardcoded to linux-x86_64, so a Windows box with an active locally built CUDA engine (`engine build cuda`) downloaded the standard Vulkan asset on every channel update and registered it dormant — the exact wasted-download the guard exists to prevent, and the platform where the dormant copy is largest. The skip now applies on both x64 desktop OSes (macOS stays excluded: no CUDA, Metal lane); surfaced by the windows CI job failing `integration__update_resolved__keep_cuda_skip_downloads_nothing` since commit 426a9d7 restored the full OS matrix after an ubuntu-only era.

- **`engine rm` refuses while a live child serves from the engine and can no longer leave ghost rows.** `/api/ps` (and `pallama ps`) now report the serving engine per row (`pallama_engine`); `engine rm` bails with a stop-first teach when any live child runs the target engine, falls back to refusing on leftover child pidfiles when the daemon is down (possible orphans of a killed daemon), and removes dir+row through the same restorable aside-first unit the retirement sweep uses — a store failure mid-removal restores the directory instead of leaving a live row over a deleted path. Live-observed before the fix: rm of a hot fork lane deleted its tree while the daemon kept serving requests from the unlinked binary.
- **Fork lanes no longer shadow mainstream engines for overlapping models (E2E lifecycle audit).** Lane selection (`serving_lane`, shared by the supervisor, gateway, and CLI) previously took the newest lane of the required kind — so on a box whose global engine is sglang/mistral.rs, installing a fork lane for one exotic architecture silently moved EVERY GGUF model onto the fork. Selection now carries lane provenance (`LaneClass`: manifest `source` field) and a mainstream build always beats a fork shim for same-kind overlaps, newest-first preserved inside each class; the unknown-architecture rescue ordering (`advertising_lanes`) applies the same rule, and an explicit `model_overrides.<model>.engine` pin still forces a fork when wanted. A fork remains eligible whenever it is the only lane of its kind. The default capability registry now ships in-repo (`registry/capability-lanes.json`, served from the default URL) with the live-validated instella-moe lane — previously the URL 404'd and every real user saw an empty catalog. Caller-provided source trees whose git origin is not upstream llama.cpp are stamped as the fork lanes they are (honest provenance: correct tag shape, trust tier, and retention treatment) instead of Upstream. The supervisor's localhost control-plane calls (warm-peg probes, session-bank save/restore) share one pooled HTTP client instead of minting a client per call.
- **Capability-lane audit hardening (7 findings, all root-cause).** Unknown-architecture classification no longer panics on stderr tails containing multi-byte case folds (U+0130/U+212A shifted the byte offsets between original and lowercased strings; extraction now runs entirely on the lowercased copy, which is also the matching key advertised lane manifests carry — mixed-case architecture names now classify). The retirement sweep retires a lane dir aside-first and only deletes the row afterward, restoring the aside on a store failure (no ghost row over a deleted dir), and one un-reclaimable lane warns and frees the rest of the pass instead of aborting the whole sweep. `llama-arch.cpp` mining fetches now carry a 10s per-request timeout (a blackholed route previously stalled daemon startup behind the 2-minute client read timeout). Registry fetches read a bounded body (1 MiB cap, refused loudly — the registry URL is user-configurable and a hostile mirror must not be bufferable behind the fetch timeout) over a shared pooled HTTP client (previously three one-shot clients). Fork repo slugs reject `.`/`..` traversal segments. `engine install --lane` refuses a backend the lane does not advertise (a CUDA build of a cpu-only fork previously died silently in cmake) and `engine offers` gains a BACKENDS column.

### Added
- **Capability-lane registry + supersede lifecycle (`engine offers`, `engine install --lane`, auto-retire).** A curated registry (JSON array of `{id, repo, ref_sha, upstream_pr, architectures, backends, status, note, added_at}`; URL via `capability_registry_url`, env `PALLAMA_CAPABILITY_REGISTRY`, `""` disables) catalogs community fork lanes per missing architecture: `pallama engine offers` lists the catalog (`--arch` filter, `--json`), and `pallama engine install --lane <id> --backend cpu|cuda` builds one — registry data never bypasses the fork validators (defense in depth) and installs carry the curated trust banner. When every architecture a fork lane serves ships upstream, the lane is stamped `superseded by <tag>` (binary upstream lanes get their architecture set mined once from their tag's `llama-arch.cpp` on first refresh — release-installed lanes have no mined set of their own), learned rescue pins clear lazily on next resolve, `engine list` shows the stamp, and curated-trust lanes are deleted after `fork_retire_days` (default 7, 0 disables) once superseded, inactive, and unpinned — user-built forks and pinned lanes are never auto-deleted. The unknown-architecture failure teaching now also names a matching curated lane from the registry when one advertises the missing architecture. Live-validated end-to-end: registry catalog + filtered offers, curated build (152 architectures mined), binary-lane mining + supersede stamping, and a patched-GGUF unknown-arch crash → classification → fork-build teaching + registry offer chain.
- **Capability lanes: run GGUF architectures that only exist in unmerged llama.cpp forks.** `pallama engine build --backend cpu --fork owner/llama.cpp@<commit-sha>` (long form: `--repo owner/llama.cpp --ref <sha>`) clones the fork at that exact commit — never a branch — verifies the checked-out SHA matches the pin, resolves abbreviated pins (>= 4 hex) through the commits API first, mines the architecture table out of the fork's `src/llama-arch.cpp`, and registers the build as an additive engine lane (`fork-owner_repo-<sha8>-<backend>`) whose manifest records full provenance (source, repo, ref pin, base tag, advertised architecture set). Fork lanes never consume the KEEP_TAGS retention budget and are never auto-pruned or sibling-pruned; `engine list` shows the provenance suffix (`fork owner/llama.cpp@7c81a9f0 (base bNNNN)`) and `--json` rows gain additive `source`/`provenance` keys. At spawn time, a model that dies with `unknown model architecture: 'x'` is classified, and if an installed lane advertises that architecture the supervisor re-routes to it exactly once per spawn and remembers the rescue pin for the daemon's lifetime — a user `model_overrides.<model>.engine` pin always wins and is never overwritten; when no lane advertises the architecture the failure teaches the exact `--fork` build command. Upstream source builds now record their own resolved commit in the same provenance fields. Fork builds print an explicit trust banner (third-party code compiled and run with your privileges), skip the b-tag regression gate (they live outside the b-tag currency), and leave sibling engines untouched.

## [0.9.0] - 2026-09-19

### Changed
- **Prebuilt CUDA channel now consumes upstream llama.cpp releases directly; the project builds no llama.cpp artifacts at all.** The engine channel installs upstream's official ubuntu-cuda assets automatically on Linux-NVIDIA boxes (same-release asset first, then a scan-back for the newest release that ships one, then the Vulkan universal asset). The overlay concept survives only as a self-hosted escape hatch: `PALLAMA_ENGINE_REPO` points at a repo publishing `bNNNN-cuda` builds of upstream sources (unset or empty skips the overlay lanes with zero network probes; `pallama engine install bNNNN-cuda` requires it). The hourly freshness watcher, the CUDA build workflow, and every project-published `bNNNN-cuda` tag are gone.

### Added
- **On-demand LoRA variants (`model+adapter`).** Any request may name `model+adapter` (stem = file stem of an adapter attached via `pallama lora add`): the variant spawns as its own instance carrying ONLY that adapter, so the dense model stays warm beside it — per-request adapter swaps with no restarts, no config edits. Base spelling resolves through the existing exact/colon-swap/prefix ladder; unknown or duplicated stems 404 with the exact `pallama lora add`/`lora rm` remedy. `pallama ps` lists the variant as `model+adapter`; session banks, prefix heat, and keep_alive keep keying the plain model. Router mode (single child) rejects plus-syntax with the `/v1/adapters` alternative instead of a confusing child-side 404.
- **Per-request speculation control, spec autopull, and spec visibility.** `options.spec` (ollama dialect) and `X-Pallama-Spec` (OpenAI dialect) queue a spec mode for the model's next spawn — same restart-once semantics as `options.num_ctx`, and same-shape requests never churn a matching instance. The REPL rides it with `pallama run <model> --no-draft` (dense spawn for that whole session, config untouched, re-queued every turn so mid-session evictions keep the shape). Invalid modes 400 with the full vocabulary instead of silently falling to the catalog pair. New opt-in `spec_autopull = true` pulls a missing catalog draft at first spawn instead of degrading to dense (auto) or failing (explicit modes); any autopull failure falls back to exactly the without-autopull behavior — per-model ensure lane only. `pallama ps` and `/api/ps` now show the effective spec mode + draft file per instance (`SPEC` column, `pallama_spec`/`pallama_draft`), and `pallama doctor` gains a chunking row surfacing the effective prefill knobs (llamacpp `ubatch_size`, sglang `chunked_prefill_size`/`mixed_chunk`) with the engine defaults they fall back to.
- **Cache-busting client detection (`cache_bust_system`).** Chatty agent clients that mutate the system prompt or tool list every turn silently destroy prefix-cache reuse — every turn re-prefills the whole conversation. The gateway now fingerprints system+tools per conversation (first-user-message identity, 1 KiB affinity discipline) on `/api/chat` and `/v1/chat/completions`, and after two consecutive mutations on a stable conversation emits one advisory detection into the sentinel ring (`pallama why`), with the fix spelled out: keep the system prompt byte-stable, move per-turn values into the last user message. Purely advisory telemetry — requests are never modified, rejected, or retried.
- **Per-model cache-hit visibility.** The gateway's warm/cold classification now keeps a per-model token split (same lock-free atomics as the global): `pallama ps` gains a `HIT` column and `/api/ps` a `pallama_cache_hit` ratio (null until the gateway has classified a completed response — never a lying 0%), and `/metrics` gains `pallama_gateway_cache_hit_ratio{model=...}` per model. sglang children now carry `--enable-cache-report` by default (it powers the warm/cold split and these surfaces; metrics reporting only, `cache_report = false` opts out) — the scrape-derived `pallama_cache_hit_ratio` remains the engine-side truth.
- **Offline text-to-speech (`pallama tts` + `/v1/audio/speech`).** A piper lane (rhasspy/piper binary from its GitHub release, voices from `rhasspy/piper-voices` on HF) speaks the OpenAI TTS shape locally: `model` IS the piper voice id (`en_US-amy-medium`), output is a full WAV (`response_format` teaches instead of transcoding), `speed` maps to piper's inverse length scale. `pallama tts --install / --pull <voice> / --list / --pin` manage the lane with the same retention semantics as whisper (tag pinning, KEEP_TAGS prune, replace-don't-merge installs); synthesis spawns the one-shot binary per request (`kill_on_drop` bounded, 10k-char input cap) — no server child, nothing pooling. Remote intent (`name:model`) forwards to fleet remotes like every other lane.

## [0.8.0] - 2026-09-18

### Added
- **The REPL asks for thinking on every turn.** The gateway translates the request to the engine's chat-template thinking knobs; reasoning streams dim-italic with a blank line before the bright answer — no tags, no toggles, nothing chat-app-shaped. Models whose template has no thinking markers get a teaching 400 from the gateway; the CLI remembers that per model (reset on `/model`), prints a one-line notice, and re-sends without the knob — capability negotiation, never a silent retry. Reasoning stays out of conversation memory; only the answer is remembered.

### Changed
- **The interactive REPL chat stream is sacred: no `[profile]` lines after replies anymore.** Profile telemetry for a run (KV/ctx fit, slot autofit, load-mode choice, spec fallback) now reaches the user only through `pallama ps` (table `warn[...]` rows and `--json`) — the surfaces you consult on purpose — while pure auto-tuning decision notes (`slots auto`, `slots auto-fit`, `spec=auto` dense fallback, `load-mode mlock auto`, and the `cache_ram_mb` default meeting the adaptive 30%-of-RAM cap) moved to the daemon journal (`journalctl -u pallama`), emitted once per spawn. What stays a `pallama ps` warning is input divergence worth acting on: a user-pinned `cache_ram_mb` that got clamped, a spec draft pair that exists in the catalog but is not pulled, engines lacking flags you requested. Net effect on a default-config small-RAM box: zero warning noise, same decisions, same journal audit trail.
- **README restructured for scannability** with the ollama conversion path up front, and CI hardened: audit-job annotations, ruff pinned at 0.16.8 with a frozen rule set, hermetic e2e XDG pins, tampered-digest guarantee.

### Fixed

- **A failed engine replacement can no longer destroy the working engine it was replacing.** All install lanes (llama.cpp release assets, mistral.rs releases, sglang pip venvs, source builds) now run through one rollback-safe swap: the installed `engines/<tag>` dir is renamed aside before the new install starts at the final path (venvs and extracted archives embed absolute paths, so a scratch-name build cannot be renamed in), restored untouched when anything fails — download, extract, probe rejection, store failure; live incident 2026-09-18: a full disk ate a healthy sglang venv mid-rebuild under the old remove-first flow — and the superseded copy is deleted only after the replacement registers. A replacement briefly needs headroom for both copies; that disk cost is the price of never leaving a tag empty-handed. Leftover rollback copies from crashed runs of the same tag are swept automatically.

### Security
- **rustls bumped to 0.23.45** (RUSTSEC-2026-0285) across the workspace HTTP stack.

## [0.7.0] - 2026-09-18

### Fixed
- **Engine probes no longer misread fork/fd/writeback pressure as a broken engine.** `Command::spawn` transients — EAGAIN/EINTR/ENOMEM (fork pressure), EMFILE/ENFILE (fd pressure), ETXTBSY (write→close→exec writeback race on a just-installed binary, captured live on the install lane, not only tests) — are now classified and retried through one bounded policy (4 attempts, 250 ms backoff) at the shared spawn layer: `probe_output`, the post-crash `verify_engine_binary` probe, and the mistral.rs help census all inherit it. Permanent failures (missing binary, permissions, ENOEXEC/garbage asset) still fail on attempt one, so a genuinely bad engine is rejected just as fast as before; what disappears is the flake class where a healthy engine failed its probe under load and `register_or_clean` deleted a perfectly good engine dir. The retry lives in `pallama-runtime` internals — no config surface, no behavior change for healthy runs.

### Changed
- **Default gateway port: 11434 → 11435.** Pallama now owns its port identity and never collides with a running ollama (which lives on 11434). Config files without an explicit `port` inherit the new default on upgrade — re-point clients (`OLLAMA_HOST=http://127.0.0.1:11435`, OpenAI base URL) accordingly. Drop-in `OLLAMA_HOST` replacement remains available as an opt-in: set `port = 11434` in `config.toml` once ollama is removed from the box. Explicit `port = 11434` pins keep parsing and binding unchanged.

## [0.6.1] - 2026-09-18

First tagged source release of the Pallama server (consolidates the 2026-09-17 engine-routing era entries below under the same version).
### Added
- `pallama fit --json`: fit rows as JSONL with the machine context (`repo`, `vram_bytes`) riding every row — `fits_vram` is meaningless without the hardware that produced it; header, table and lane hints suppressed.
- `pallama show --json`: the model card as one machine-typed object — nested GGUF metadata (null on safetensors rows) and the stored profile with argv/benchmark embedded as real JSON values instead of double-encoded strings (a corrupted row degrades to its raw string rather than failing the listing).
- `--json` on `list`, `ps` and `engine list`: the `search --json` JSONL contract across every table command — machine-typed rows (raw bytes, null-able optionals, full untruncated engine sha256), banners/hints/catalog lines suppressed so stdout is pure data; empty listings stream zero rows.
- `pallama fit` on safetensors repos: one aggregate row (full shard set summed from Hub blob sizes) through the same rule-6 KV ladder as the GGUF lane, with the sglang/mistralrs serving-lane hint; sizeless repos now teach "nothing to preview" instead of printing a bare empty table.
- `pallama search --json`: one JSON object per row (JSONL, matching `doctor --json`) — machine-typed `ctx`/`size_bytes`/`arch` (null without GGUF metadata) and the FULL quants list (no `+N` collapse) for scripting; empty results stream zero rows (`jq -s` reads `[]`), and SIGPIPE is reset to default so `| head`/`| jq` closing early ends quietly instead of panicking.
- `pallama search --format <F>`: weight-format filter beyond the GGUF default — any Hub tag (`safetensors`, `awq`, `gptq`, `fp8`, `mlx`, `onnx`, …) or `any`/`all` for an unfiltered browse; `--format` works in any argument position; new FORMAT column (Hub tags, most-specific-first: an AWQ repo shows `awq`, not its `safetensors` container tag; `gguf` outranks the `mlx` tag GGUF mirrors self-apply), non-GGUF QUANTS mined from repo-id bit-widths (`-8bit`, `-AWQ`, `-GPTQ-Int4`, `IQ4_XS`), and format-aware pull footers (safetensors → sglang/mistralrs lane; MLX → convert guidance). Empty results now name the format and teach valid tags.

Engine routing, the HF format audit, and a hardened engine store — the "pick the right engine for the model" release. Era highlights below; entries were drafted under Unreleased as they landed.

### Added
- **`pallama doctor` routing rows (2026-09-17).** Doctor now resolves every pulled model through the same lane resolver as the listings and warns per-model when nothing can serve it, quoting the exact `pallama engine install --kind <kind>` remedy; a healthy store gets one `routing` ok row naming the active mode.
- **`pallama list` shows the routed engine per model (2026-09-17).** New ENGINE column resolves through the same lane logic as `/v1/models` and `/api/tags` — the three listings can never disagree. Manual mode shows the global active engine; auto mode routes per format+policy; `-` means nothing installed can serve the model.
- **Engine routing v1 — `[engine_routing]` mode + policy + per-model pin (2026-09-16/17).** Default `manual` is byte-identical to 0.6.0. `mode = "auto"` routes each spawn by model format through the evidence-backed table: GGUF → llamacpp (mistral.rs GGUF 0.462 vs llama 0.500/0.526 quality, same Q4_K_M — measured, not assumed); safetensors → sglang on `quality`/`throughput` (0.612 quality / 752 tok/s conc4) or mistral.rs on `latency` (26 ms TTFT, 8 s cold). `policy` routes for real; a per-model `engine = "<tag|kind>"` pin beats both modes. Routed spawns use a per-spawn local adapter — the global active engine and the supervisor arc are untouched — and the gateway now keys mistral.rs quirks off the per-child kind (Instance/EngineRef carry `kind`). `/v1/models` + `/api/tags` advertise the routed engine per model row.
- **`pallama engine use --kind <kind>`** — switch lanes by kind, newest-first.
- **`model_load_timeout_secs`** — raise the hardcoded 180 s health budget for first-ever JIT lanes (marlin repack + graph capture can exceed it).
- **HF file-type coverage audit** (`docs/hf-format-coverage.md`) — pull/serve lane + evidence tag for every HF format: GGUF (any quant, sharded, MTP heads — llamacpp, PROVEN incl 1.35x MTP speedup), safetensors BF16/AWQ/GPTQ/FP8 on sglang (AWQ marlin ~1.7x faster than BF16 same-window; GPTQ 329-358 tok/s; FP8 ~260-275 tok/s w8a8), MLX/EXL2 documented convert-first (no fake support), mistral.rs fallbacks + its ISQ lane verdict (upstream-broken at request time in v0.9.3).
- **Auto-slots ceiling 8 on ≥8 GiB cards** (np-sweep evidence), **llama-server child knobs** (sse ping, timeout, template kwargs, cont-batching, reuse-port, lora-init), **mistral.rs tuning surface** (21 knobs + LoRA ALIAS=SOURCE lane), **dotted config edits** (`config set sglang.grammar_backend xgrammar`), **`config edit [model]`** with schema-pinned knob hints, **installer engine menu**, **`engine update --check`** dry-run.

### Fixed

- **Wrong-lane teaching names the engine you already have (2026-09-17).** Pulling a safetensors model while llamacpp is active in manual mode used to advise `pallama engine install --kind sglang` even when sglang was installed all along — the error could not see the store. The mismatch teaching now appends the installed lane and the zero-flip remedy: `... but sglang-0.5.19 is already installed: pallama engine use sglang-0.5.19 (then restart the daemon), or let routing pick per model: pallama config set engine_routing.mode auto`.
- **Per-kind engine probe** — false rollbacks dethroned healthy sglang/mistral.rs engines on ANY spawn failure (the probe only ever looked for `llama-server`).
- **mistral.rs paged-attention co-tenancy** — the 0.90-of-total PA fraction silently truncated KV into ~60 ms empty responses on shared GPUs (the matrix MULTITURN 0.0 mystery); now derived from free VRAM with a classic-KV fallback. Matrix recovery: quality 0.509 → 0.575, MULTITURN 0.0 → 0.733, conc4 186 → 508.
- **mistral.rs `extra_args` strict manifest-gated passthrough** — the blanket refusal silently disabled engine-only flags (ISQ A/B measured 1.00x doing nothing).
- **Manual-mode mistral.rs model rewrite regression** (introduced + fixed within this release): the pre-spawn predictor now resolves `Ok(None)` to the global kind.
- **Installer hangs forever running `pallama serve` from a comment** — dash executes backticks inside `$( )` heredocs at parse time; also wrote a corrupted unit file.
- **Relocatable engine rows + verified downloads** — `server_path` re-roots to the live data dir; engine downloads verify length AND digest and never leave partials behind (the truncated 14 MiB/768 MiB mistral.rs tarball class).
- **Warm-peg after sglang/llamacpp spawn** (`[warm_peg]`) — health flips 200 before sglang's residual warmup; the first concurrent batch paid a one-time shape JIT (the 39.6 tok/s mystery → 752 at steady state).

### Added
- **FP8 safetensors proven live (2026-09-17).** RedHatAI/Qwen2.5-0.5B-Instruct-FP8-dynamic pulls by HF name and serves on the sglang lane: cold 31.7 s (raise `model_load_timeout_secs` and disable the prefill graph on JIT-laptop boxes), warm ~260-275 tok/s — the w8a8-dynamic class sits between BF16 and 4-bit marlin as expected. Coverage table updated.
- **`/v1/models` and `/api/tags` advertise the routed engine per model (2026-09-17).** Every model row carries an additive `engine` field naming the tag that would serve it right now: manual mode shows the global active engine; `[engine_routing] mode = "auto"` resolves per model through the same format+policy lane as the supervisor (GGUF → llamacpp, safetensors → sglang/mistral.rs per policy); `null` means nothing installed can serve it. One shared resolver (`LaneResolution`) backs the listing, the chat-time model rewrite, and routing — the three can no longer disagree.
- **Engine routing v1 — `[engine_routing]` (2026-09-17).** `mode = "manual" | "auto"` (default `manual` = byte-identical to today: an engine/format mismatch keeps its teaching error). In `auto`, each spawn routes by model format — GGUF → llamacpp, safetensors → sglang, mistral.rs as fallback for both, and a strict teaching error when nothing installed can serve the format — via a per-spawn local adapter: the global active engine row and the supervisor's engine are untouched, co-residency stays guarded by the existing VRAM ladders. A per-model `engine = "<tag-or-kind>"` pin in `[model_overrides."<model>"]` wins over both modes; a pin naming nothing installed fails loudly with the roster. `policy = latency|quality|throughput` now routes (default `quality`): on safetensors, `latency` prefers mistral.rs (26 ms TTFT, 8 s cold) and `quality`/`throughput` prefer sglang (0.612 quality, 752 tok/s conc4); GGUF stays llamacpp under every policy until a quant-matched mistral.rs GGUF matrix row exists. Live-proven both directions: sglang active + GGUF model + auto → routed to llamacpp and served (log: `engine routing: spawn uses a routed adapter`); manual → the existing safetensors/GGUF teaching error, unchanged.

### Added
- **`model_load_timeout_secs` (2026-09-17).** The health-window budget for a spawning engine child was hardcoded at 180s — plenty for precompiled lanes, but a first-ever sglang quantized spawn (marlin repack + JIT + CUDA-graph capture) can exceed it and the daemon kills a child that was making progress. `pallama config set model_load_timeout_secs 600` raises the budget; the knob validates > 0 and feeds the supervisor verbatim.
- **`sglang.cuda_graph_backend_prefill` knob (2026-09-16): unsticks quantized spawns on hybrid-GPU laptops.** sglang 0.5.19's default `breakable` prefill CUDA-graph capture (42 shapes) deadlocks at 0% on hybrid iGPU+dGPU laptops — live-proven with an AWQ marlin load that never finished capturing while decode graphs captured fine. `cuda_graph_backend_prefill = "disabled"` (choices: full/breakable/tc_piecewise/disabled) skips the prefill capture and lets the spawn reach healthy: AWQ cold 21.8 s, warm ~369 tok/s through the gateway (parity with direct-shim 371 = zero gateway overhead). AWQ/marlin lane itself verified end-to-end (quant auto-detected, 0.46 GB weights); same-window A/B (correction of the initial cross-window read): **AWQ marlin decodes ~1.7x FASTER than BF16 on this 8 GiB laptop** (steady ~351 vs ~196 tok/s, back-to-back phases, BF16 got the cooler GPU and still lost — decode is bandwidth-bound) plus 708 MiB vs 953 MiB weights; forcing the vanilla path (`quantization = "awq"`) is blocked on nvcc ≤ 12.0 by an sglang JIT source bug (`assert` undefined in `awq_dequantize.cuh`) — the default marlin path never compiles that kernel.

### Changed
- **Auto engine routing is the default (2026-09-17).** `[engine_routing] mode` flips `manual` → `auto`: a freshly pulled safetensors model now serves through sglang (or mistral.rs per policy) with zero configuration, instead of hitting the wrong-lane 400 out of the box. Set `mode = "manual"` to keep the active engine serving everything — the wrong-lane teaching (now store-aware, see Fixed) covers that path.
- **Auto-slots ceiling 8 on 8 GiB+ cards (2026-09-17).** `slots = 0` (auto) used to cap at 4 slots on every GPU; live `-np` sweep on an 8 GiB RTX 4070 Laptop shows slots=8 sustains 440.7 combined tok/s at conc4 vs 394 at 4 (identical per-request walls = true parallelism, not queueing). The cap is now 8 when the card has ≥ 8 GiB VRAM, 4 below; VRAM/RAM/context mins still bind first. Test expectations that pinned the old cap moved with it.
- **mistral.rs `extra_args` is now strict manifest-gated passthrough (2026-09-16).** The lane used to blanket-refuse every override with a warn-drop ("its 'serve' grammar does not accept llama-server flags"), which silently disabled legitimate mistral.rs-only flags (`--isq` measured a perfect 1.00x A/B doing nothing). Now mirrors the sglang contract: a flag must exist in the engine's probed manifest (unknown → hard error with manifest teaching) and must not collide with the twelve supervisor/ladder-owned flags (reserved → hard error); everything else rides the child argv verbatim, and the argv translator forwards it manifest-gated. Live caveat from the proving experiment: mistral.rs v0.9.3 `--isq` itself is broken upstream on the cuda130-sm89 build (request-time `model_error` for GGUF and safetensors sources, `q4k` and `q8_0`, free or busy GPU) — passthrough verified working end-to-end, documented in `docs/engine-coverage.md`.

### Fixed

- **Routed-spawn mistral.rs requests keep their model rewrite in manual mode (2026-09-17).** The per-child-kind gateway refactor (routing wave) taught the pre-spawn predictor only the routed-lane shape, so in manual mode — where the global active engine serves — the ollama-lane translation stopped rewriting the request model to mistral.rs's literal `default` and every chat came back `model ... was not found` (matrix rerun showed it as an 85-error all-red row and zeroed bench lanes). The predictor now resolves manual mode to the global engine's kind, matching the old daemon-lifetime behavior; live re-proof serves GGUF through the mistral.rs lane 200/OK.
- **mistral.rs silently served empty responses on a GPU shared with another tenant (2026-09-16).** Upstream's default `--pa-memory-fraction 0.90` is a fraction of TOTAL VRAM; with a co-tenant (e.g. ollama holding 6.3 of 8 GiB) the paged-KV pool lands on absent memory and long prompts — 17k-char system + tool schemas — came back empty in ~60 ms while short prompts looked healthy. The profile compiler now derives a conservative `--pa-memory-fraction` from FREE VRAM at spawn (model weights + projector + headroom subtracted, clamped 0.05–0.90, explicit `mistralrs_pa_memory_fraction` pin always wins) and falls back to `--paged-attn off` + classic ctx-sized KV with a loud teaching warning when even the minimum KV doesn't fit (forced-on never emits conflicting flags). Single-tenant and probe-blind boxes are byte-identical to before. Live-proven both ways: the exact failing workload (co-tenant + full tool prompt) went from instant-empty to a real tool call.
- **Engine rollback no longer dethrones healthy sglang/mistral.rs engines (2026-09-16).** The crash-path probe walked for a binary named `llama-server` only — structurally always false on the sglang (`sglang-server` shim) and mistral.rs (`mistralrs`) lanes, so ANY spawn failure on those engines triggered a false `--version probe failed` rollback to llamacpp (first live hit: a config-induced flashinfer JIT crash took the whole sglang lane out of `active`). New `verify_engine_binary` probes per kind — manifest `server_path` for llama/mistral.rs, the venv's `importlib.metadata` for sglang (the shim has no `--version` and would boot torch) — behind a real time budget (5s/15s, std-only thread + channel; the old probe documented a budget it never enforced and could block the spawn path forever). A config-induced crash now yields the request error alone; the genuine-ABI-crash heuristic (2+ models failing with a healthy probe) still rolls back. Live-proven by replaying the exact incident: 502 for the request, no rollback line, sglang stays active.

### Added
- **`pallama engine use --kind <kind>` (2026-09-17).** Shorthand for switching the serving lane by engine kind instead of tag — resolves the newest installed row of that kind (newest-first, the same contract rollback relies on); the plain tag form is unchanged.
- **HF file-type coverage audit (2026-09-17).** `docs/hf-format-coverage.md` — which Hugging Face file types each engine pulls and serves, with live-probe evidence: GGUF (any quant) + GGUF-MTP on llamacpp, BF16/AWQ/GPTQ safetensors on sglang (GPTQ Int4 0.5B proven this release: 329-358 tok/s warm, marlin class), mistral.rs GGUF+BF16 lanes; MLX/EXL2/bitsandbytes honestly N/A on Linux with conversion guidance; upstream/env-blocked lanes (mistral.rs `--isq`, sglang vanilla-AWQ and flashinfer under nvcc <= 12.0) documented with root causes.
- **`pallama engine update --check` (2026-09-16).** Dry-run both lanes: resolves the newest compatible build and prints the full upgrade story — channel, up-to-date-or-available, and for the llamacpp CUDA lane the overlay state (driver vs newest CUDA, overlay-published-but-too-new with the `pallama engine build cuda` hint, or the hourly overlay-lag note) — without downloading, installing, or writing anything. sglang lane: reports PyPI currency the same way; a pinned `--version` is ignored with a note under `--check`.
- **The installer teaches the engine menu (2026-09-16).** Bootstrap stays zero-touch (llamacpp auto-picks the newest driver-compatible CUDA build; skip with `PALLAMA_INSTALL_ENGINE=0`), but every install now prints the one-liner for the other engines — `pallama engine install --kind sglang` / `--kind mistralrs` with honest cost notes (~6 GiB venv, Linux+NVIDIA; ~0.8 GiB) plus `engine list` / `engine use <tag>` for switching — so the choice is discoverable from the install output itself instead of the docs.

## [0.6.0] — 2026-09-16

Full native tuning surface for all three engines, warm-peg, and the
relocatable/verified engine store.

### Added
- **Engine flag coverage audit (2026-09-16): `scripts/engine_coverage.py` + `docs/engine-coverage.md`.** A reproducible answer to "how much of each backend is actually wired": the script reads each engine's probed manifest from the store, greps the profile compiler + argv translators for wired flags, and buckets the long tail (multi-node, observability, per-request, product-specific families). The doc states the two-layer model — first-class knobs for the single-box product, universal `extra_args` passthrough for everything else (manifest-gated on sglang, reserved-seven hard error) — with live numbers: llama-server 174/328 wired, sglang 74/541, mistral.rs 41/93, plus the promotion policy and the proof-artifact map (argv pins, lockstep tests, live e2e, matrix benchmarks).
- **`config edit <model>` hint coverage is pinned in BOTH directions (2026-09-16).** The uncomment-all schema probe already caught stale keys; `unit__knob_hint_block__covers_every_tuning_field` now parses the tuning structs from pallama-core source and fails naming any field that lacks a hint line (and any hint key that is not a field) — a new knob can no longer ship without editor hints.
- **`pallama config edit` opens the config in `$VISUAL`/`$EDITOR` (2026-09-16).** Previously every table-scoped tweak meant `nano $(pallama config path)`-style archaeology. The command bootstraps the config file if missing, runs the editor on it, and validates the result on exit — a hand-broken file surfaces immediately with the offending error instead of at the next daemon boot. No editor set is a teaching error (export EDITOR), not a guess; a non-zero editor exit propagates. With a model name (`config edit <model>`) the editor opens with a comment hint block at the top: the model's engine lane, every tuning knob family with example values, and its `[model_overrides."<model>".<engine>]` skeleton — the block strips itself on save (mangled markers stay as harmless comments), and its keys are unit-pinned to the config schema so hints never drift stale.
- **A successful engine update prunes the lane it updated (2026-09-16).** `KEEP_TAGS = 2` retention kept the previous build as a rollback anchor — visible as "two llama.cpp engines" after every update, each hundreds of MB. A verified, activated update (llama-server prebuilt, `engine build`, and the sglang venv lane alike) now deletes its superseded same-kind siblings outright, printing each freed tag and size; the `local` pseudo-tag and cross-kind engines are never touched, and the non-active paths (keep-CUDA guard, gate failure) keep the incumbent — nothing is deleted unless the new engine actually took over. Rollback after an update therefore fails loudly ("no older engine") instead of silently stepping back to a stale build; reinstalling an older tag is one command.
- **`config set`/`get`/`unset` now reach table knobs via dotted keys (2026-09-16).** `pallama config set sglang.grammar_backend xgrammar`, `model_overrides."qwen 7b".sglang.stream_interval 4` and `mistralrs.prefix_cache_n 0` previously failed with a duplicate-table parse and a "edit the file directly" shrug — the setter only understood root-scope pins. The command now writes/reads/removes a single leaf inside its table surgically: existing lines are replaced in place, new leaves land grouped inside their section, and absent headers (`[model_overrides."<model>".sglang]`) are created after the longest existing parent section with non-bare segments quoted. Candidates validate against the full schema before the file is touched (same reject-and-leave-unchanged contract), `get` prints the pinned line or `<not set>`, and `unset` removes just that leaf (the header and its other keys survive). Unknown dotted paths fail fast via schema probes — `sglang.bogus` is a typo, not a new key. Root-scope bare keys behave exactly as before.
- **Full sglang tuning surface (2026-09-15): 25 new `model_overrides.<name>.sglang.*` knobs (or global `[sglang]` table) + LoRA adapter lane.** The 0.5.19 flag census (541 flags) minus the multi-node service families (PD-disagg/MoE-backend/deepep — wrong product for a single-box server) is now first-class config, flag-gated per engine manifest with the usual warn-skip on older builds: tokenizer family (`tokenizer_mode`/`tokenizer_backend`/`tokenizer_worker_num`/`detokenizer_worker_num`/`dynamic_batch_tokenizer{,_batch_size,_batch_timeout}`), structured output (`grammar_backend` with xgrammar/outlines/llguidance choices, `radix_eviction_policy`, `session_radix_cache`, `mixed_chunk`), idle/VRAM hygiene (`sleep_on_idle`, `memory_saver`, `watchdog_timeout`), scheduler (`cache_report`, `batch_notify_size`, `scheduler_recv_interval`), `cuda_graph_bs` (explicit batch-size list — suppresses the tight-fit ladder's own `--cuda-graph-max-bs`), `max_total_tokens`, parallelism pins `tp/dp/pp/ep_size` (emitted only above 1, with a multi-GPU teaching warning on single-GPU boxes), and LoRA (`max_lora_rank`, `lora_backend`). LoRA adapters ride the lane natively: GGUF/`.bin` adapters are refused with the PEFT-format teaching (sglang loads HF/PEFT only), HF adapters emit `--enable-lora` + `--lora-paths name=path`, and the llamacpp `scale` field warns-and-ignores (upstream has no per-adapter scale). `deterministic = true` now also emits the native `--enable-deterministic-inference` when the engine has it (the slots=1 pin stays — additive, no behavior change). Choice fields validate against the engine's real `--help` grammar at config-parse time.
- **Warm-peg after sglang/llamacpp spawn (the `[warm_peg]` table, `default = true`) — the 39.6 tok/s concurrency mystery solved.** sglang's `/health` flips 200 before residual warmup drains, and its internal warmup covers batch-shape 1 only, so the first concurrent batch paid a one-time ~4.4 s JIT/autotune stall that a freshly-spawned engine (idle-evict cycles) handed straight to the benchmark's only burst — while llamacpp ships precompiled CUDA and pays none of it (steady-state conc4 measured 587 tok/s once warm). The supervisor now pegs the batch shapes at spawn: one single probe (90 s budget, drains the residual queue) then N truly-concurrent probes before the instance publishes Ready; idle-evict respawns re-peg automatically. llamacpp is pegged too (`-np` width, detached — its peg is near-free and a blocking one was measured to tax slow-child spawns) for uniform first-request latency; mistral.rs serves eagerly and skips. Unix-transport and other engines skip (FakeEngine-safe); a failed probe warns and never blocks the spawn. Per-engine opt-out: `pallama config set warm_peg.sglang false` (or `warm_peg.llamacpp`) overrides `default` for that lane only.
- **`pallama config unset <key>` and `pallama config defaults [key]`.** A pinned knob previously had no path back to its built-in default short of hand-editing config.toml — and stale pins outlived default changes (the retired `spec = "off"` / `cache_reuse = 256` pins did exactly that). `unset` removes the top-level pin (validated candidate, atomic write with the usual `.bak-*` trail; never touches `[model_overrides]` tables even when the same knob name appears there), reports the old line plus the default it returns to, and is idempotent when the key is not pinned (running daemons keep their loaded config until restarted; `PALLAMA_*` env overrides still win). `defaults` prints the exact TOML a fresh install writes — or one knob's default line — so "what is the default?" no longer requires a clean box. `pallama config --help` now carries a set → get → unset → defaults examples footer; unknown keys fail fast with the same teaching as `get`/`set`.
- **Full mistral.rs tuning surface (2026-09-16): 21 new `model_overrides.<name>.mistralrs.*` knobs (or global `[mistralrs]` table) + LoRA adapter lane.** The v0.9.3 `mistralrs serve --help` census (82 flags) minus the agent-family/product-CLI surface (search/shell/code-exec/sandbox/mcp — their harness, our gateway owns agentic), ISQ/quantize lanes (Pallama pulls and quantizes itself), and hf-token/cache (Pallama pulls) is now first-class config, flag-gated per engine manifest with the usual warn-skip on older builds: scheduler family (`max_batch_size`, `max_prefill_chunk_tokens`, `max_decode_steps_before_prefill`, `prefix_cache_n` — 0 disables, exempt from the ≥1 floor), paged-attention detail (`pa_block_size`, `pa_cache_type`, `pa_context_len`), LoRA capacity (`enable_lora` dynamic-serving switch + `lora_max_rank`/`lora_max_adapters`/`lora_max_bytes` — bare limits are refused by upstream at boot, so they are gated on `enable_lora` or a preloaded adapter with a teaching warning otherwise), MTP speculative family (`mtp` switch first, then `mtp_model`/`mtp_n_predict`/`mtp_draft_sampling` {auto,greedy,probabilistic}; `mtp_model` without `mtp` is a config-time footgun error), vision caps (`encoder_cache_memory_mb`, `max_num_images`, `max_image_length`), `disable_metrics`/`disable_access_log`, and `device_layers` (`ORD:NUM;...` grammar validated, single-GPU teaching mirrors the sglang parallelism pins). LoRA adapters ride the lane natively: mistral.rs `--lora ALIAS=SOURCE` accepts GGUF and safetensors alike (unlike sglang) — both formats now emit with the file stem as alias, and the llamacpp `scale` field warns-and-ignores (the ALIAS=SOURCE form has no scale). The argv translator forwards the tuning tokens manifest-gated (old engines drop them; the compile-time warning already taught).
- **Six llama-server child knobs (2026-09-16).** The fork-census gap after samplers/aliases/gateway-owned flags: `sse_ping_interval` (-1 disables the SSE keepalive ping), `server_timeout_secs` (`--timeout`), `chat_template_kwargs` (must be a JSON object — validated at parse time), `cont_batching` (tri-state: unset = engine default, false emits the `--no-cont-batching` negation), `reuse_port`, `lora_init_without_apply`. All top-level config pins, flag-gated per engine manifest.

### Changed
- **Engine retention tightened from 3 to 2 installed builds (2026-09-13).** Auto-prune (which runs after every engine install) now keeps the newest engine plus one rollback anchor (~215 MiB each); `local` and the active tag stay protected on top. Users who want deeper history still have `pallama engine use <tag>` re-download on demand.
- **Prebuilt CUDA engine channel slimmed to consumer archs + PTX forward-JIT (2026-09-13).** The overlay CI (`engine-cuda`) built fat binaries across every SASS arch (sm 61–120 per toolkit, both real+virtual per arch). Builds now emit SASS for the consumer set only (`61-real;75-real;86-real;89-real;120` on CUDA 12.8.1, `75-real;86-real;89-real;120` on 13.0) plus full SASS+PTX on the newest arch, so future GPUs JIT forward from its PTX — roughly a third of the nvcc work and a much smaller download. Datacenter archs (sm 70/80/90/100) stay reachable through the source lane (`pallama engine build cuda`). The CI link step also resolves CUDA driver-API symbols via the toolkit stubs (GPU-less runners have no `libcuda.so.1`; the stub's SONAME keeps the runtime NEEDED entry correct on real machines).

### Fixed

- **`install.sh` no longer hangs forever running `pallama serve` from a comment (2026-09-16).** The systemd-unit heredoc is collected inside a command substitution, and dash executes backticked text — and any literal dollar-parenthesis — at PARSE time, before the heredoc is even read. A unit comment reading "`pallama serve`" therefore launched the real server with its stdout wired to the capture pipe: the installer blocked reading a daemon that never exits (40+ minutes, twice, terminal blinking), and once unblocked the `sudo tee` wrote the corrupted capture (the server's startup banner spliced mid-comment) into `/etc/systemd/system/pallama.service`. The backticks are gone, the heredoc carries a NOTE forbidding both syntaxes in its body, and the repaired installer ran end-to-end clean (unit rewritten, service active, healthz green). Reproduced with a 6-line dash repro and `sh -x` on a fakesystemctl sandbox before the fix; verified clean after.
- **The intermittent ~1.9 s test flake in the stale-release pin is root-caused, not just budgeted (2026-09-16).** With NVIDIA driver persistence mode off (laptop default) the GPU drops to P3 after ~45 s idle and the first `nvidia-smi` of a burst re-wakes it — measured 1.9 s 3/3 after idle vs 33-46 ms back-to-back, with CPU load ruled out as the trigger. The GPU-facts probe and the 15 s test budget now document the mechanism (and the `nvidia-smi -pm 1` server-side remedy) where the swing is paid.
- **Engine rows are relocatable: `server_path` re-roots to the live data dir (2026-09-15).** Manifests baked the absolute server path probed at install time, so a row installed under one data dir failed ENOENT at spawn under another (move `XDG_DATA_HOME`, restore a backup, or copy the DB into a sandbox) while the binaries sat intact under the new engines root — the engine-matrix harness hit exactly this: a sandbox DB seeded from the user store pointed b10970-cuda at a pruned user-store path and every request 500'd in <500 ms. On load (`pallama serve` startup and `engine` manager reads), a recorded path that no longer exists is re-anchored when the same `engines/<tag>/...` tail exists under the live engines dir (warn names both roots); anything ambiguous is left untouched so a genuinely missing binary still fails loudly at spawn.
- **Engine downloads verify length AND digest, and never leave partials behind (2026-09-15).** The install lane accepted a cleanly-EOF'd stream as a complete asset: a truncated mistral.rs v0.9.3 archive (14 MiB of a multi-hundred-MiB tarball — HTTP/2 END_STREAM mid-transfer) landed in the store as a valid engine row and only failed at spawn with ENOENT. `download_asset_file` now captures `Content-Length` before the stream loop, removes the partial on EVERY error path (mid-stream error, digest mismatch, length mismatch), and refuses the install naming the mismatch source (response header vs release metadata); `download_asset_bytes` cross-checks the release-API size. (hyper itself rejects most header/body mismatches as decode errors — the in-code guards are defense-in-depth for clean-EOF transports; the guaranteed contract is fail-loud + no partial left on disk.)
- **Image data-URLs tolerate MIME-wrapped base64** (python `base64.encodebytes` and friends emit newlines every 76 chars): the gateway now strips all whitespace from base64 image payloads before sniffing and forwarding. llama-server b10970's strict decoder rejected wrapped payloads with 'Failed to load image or audio file' (b10948 tolerated them; ollama strips server-side), so a whole geokit vision suite errored. Post-fix that suite scores 6/6 1.00 with zero errors; clean payloads pass through byte-identical.
- **mistral.rs now actually serves HF/safetensors rows** (first live attempt caught two argv dialect bugs): directory rows ride `-m/--model-id <dir>` instead of `-f/--quantized-file` (GGUF-only — the HF loader refused to infer a format), `--max-model-len` is emitted only for GGUF loads (the HF loader serves native ctx from config.json and rejects the knob), and the GGUF shard-view staging farm passes directory rows through untouched (an HF tree produced an empty view and a nonexistent staged path). Live-proven: v0.9.3 serving qwen2.5-0.5b-instruct BF16 end-to-end through the gateway on the engine-matrix harness.
- Bare `pallama engine install` no longer silently defaults to `--kind mistralrs` — it prints the lane chooser (all four lanes, their model formats, and the exact install command each). `pallama engine list` now catalogs every installable-but-absent lane plus the separate whisper voice lane (tag or install command), and the engine group help cross-links voice.
- Two daemons, one lock: a session-spawned `pallama serve` holding the pidfile while the systemd unit respawns used to exit 1 in a `Restart=always` crash loop. Lock refusals now carry a `LockHeld` marker and exit the same singleton-conflict code as a port bind conflict (exit 3, already covered by the unit's `RestartPreventExitStatus`), with the remove-pidfile teaching — verified live.
- `pallama quantize` is now crash-atomic: output writes to a hidden `.part-<pid>` sibling promoted by rename only after the verify gates, so an interrupted run never leaves an orphan that blocks the next one. `pallama pull` missteps name the model verbatim (no `library/` prefix) and teach the registry's no-tag-listing reality with a `search` pointer; the upgrade banner prints only on success. `pallama tune --load` announces the model load behind its first probe row; `lora`/`session`/`search` args carry real descriptions; `pallama coreside` states the device-backed KV truth and prints `-` for unmeasurable KV instead of a fake `0M`.
- **Streaming tool calls now arrive as one complete ollama-shaped line** (the dialect real ollama emits): fragment deltas are merged by index, arguments re-assembled and re-parsed (raw-string fallback when malformed), name/id/index nested as ollama clients expect, and a stream that ends mid-call still flushes. The non-stream path reshapes to the same dialect and the request normalizer stringifies object arguments the engine rejects. Clients that replay tool_calls (multi-turn agents — geokit harness) previously sent nameless fragments back and took 500s; MULTITURN 0.00 → 0.60 with zero errors.
- **Template-thinking evidence is cached per model file** (len+mtime stat guard): the absent-`think` default previously re-read and fully parsed the model GGUF on every request — ~60ms of KV-walk on a 5.7 GiB file before dispatch (first-token 133-141ms absent vs 70-80ms explicit). Post-cache all payload classes 92-97ms flat; the HTTP listener also sets `TCP_NODELAY` (correct-by-construction for NDJSON streams). Harness verdict vs ollama 0.34.0: 10/11 axes green — warm TTFT median 0.078 vs 0.080, decode +5.2%, cold load −48.7%, long-gen +2.0%, concurrency +3.6%/+3.5%/p95 −2.8%.
- `think` now defaults OFF on the ollama API for thinking-capable templates, matching ollama semantics. qwen-dialect chat templates render thinking ON when the kwarg is missing, so an unbounded reasoning trace ate the whole `num_predict` budget and the answer never arrived (70k-char `reasoning_no_answer` spirals observed live on the geokit harness). The gate reuses the `think: true` capability evidence (template markers; fail-open without a readable template), explicit `think` values pass through untouched, and user `chat_template_kwargs` still win. Both `/api/chat` and `/api/generate` run the think gates BEFORE translation — a post-translate mutation would never reach the engine (live-caught: the first landing injected `think: false` after the translator had already consumed the field).
- **Wrong-lane model metadata now teaches 400 at the door instead of crashing 500 deep in the engine.** `read_model_meta` is format-aware: a safetensors directory under the llamacpp engine (or GGUF under sglang) is refused pre-spawn with the exact engine switch remedy (`pallama engine install --kind …`, `engine use <tag>`, restart) — previously the daemon fed a directory to the GGUF reader and surfaced `gguf metadata: io: Is a directory (os error 21)` as a 500. Profile-compile teachings ride the same `UnsupportedModel → 400` mapping, so unfit pins and reserved-flag errors reach clients as remedies, not 500s. (Latent twin fixed: mistralrs + safetensors-dir previously took the GGUF reader too; it now reads the HF config.)
- **All engine-kind teachings now name remedies that exist (backend-audit gaps).** (1) The llamacpp wrong-lane message taught `pallama run <model> --engine sglang` — a flag `run` never had; it now teaches the install/use/restart sequence. (2) `extra_args` on a mistralrs model was silently dropped; the compiler now warns naming every ignored flag (its `serve` grammar shares none of the llama-server dialect — a passthrough would hard-fail the boot). (3) The llama-server-only endpoint gate (`/props`, `/slots`, `/tokenize`, `/apply-template`, `/infill`, rerank, stream-lookup) and the `/api/session` slot-checkpoint gate fired only for mistralrs; an sglang child now gets the same teaching 400 (naming its kind) instead of a bare upstream 404, and `action: close` still bypasses the gate. (4) Router-mode and child-auth warnings fork their remedy on engine kind — `engine update` for llamacpp/sglang, `engine install --kind mistralrs` for mistral.rs (which has no update lane), and a non-llamacpp router request is told router mode is llama-server-only rather than sent down an update that can never add `--models-preset`.
- **Unified-KV capacity model rewritten to device truth — loop-arch KV counted, placement corrected, storm class dead.** Two stacked estimator defects made every unified-lane capacity decision judge fiction. (1) KV layer count used `block_count` alone, but loop architectures (nanbeige4.2: `nanbeige.num_loops = 2`) build `n_layer = blocks x loops` — the estimate was exactly HALF the child's allocation (22 vs 44 layers → 2816 vs 5632 MiB f16 KV @ 32768, live-proven by manual llama-server exec). `GgufMeta` now reads `<arch>.num_loops` (clamped 1..=8) and the KV math multiplies it in. (2) The unified lane assumed `--kv-unified` moves the KV pool into `--cache-ram` system RAM (RAM-budget verdicts, raise-to machinery, flat slot caps, floor-based `-ngl 999` pinning); b10948 proves the pool is DEVICE-backed (`CUDA0 KV buffer` with the flag on — it shares one buffer across sequences, `--cache-ram` is the weights-mmap cap per PR #16391). One placement model now governs both lanes: the 2b fit judges GPU VRAM, `unified_ctx_verdict{Fit, Refuse}` refuses only what the device physically cannot host (weights + quant-scaled KV + 700 MiB spawn overhead > VRAM), the 85-100% middle zone serves degraded with `--gpu-layers` left to the engine fitter (the old floor-pin disabled the fitter — "n_gpu_layers already set to 999, abort" — and OOM'd exactly there), and the gateway `num_ctx` preflight uses the same verdict with the model's effective `cache_type` quant applied so its own `cache_type = "q8_0"` teaching is never a dead end. The pinned-slot walk-down and budget-raise machinery from the earlier num_ctx fix are removed with it (the capacity axes are now strictly tighter than the verdict — the split brain they patched is gone). Live: nanbeige pin 32768 = single 400 naming true demand (weights 2455 + KV 5632) and the q8_0 lever; with `cache_type = "q8_0"` set the same pin serves (2455 + 2816 fits); autofit shrinks honestly on true KV; qwen3.5-9b pin 32768 still serves.
- **`think: true` on a non-thinking model now teaches 400 (ollama parity).** The ollama lane silently no-op'd `think` on models whose chat template has no thinking markers, while real ollama answers 400. A marker sniff (`enable_thinking` / `<think>` / `reasoning` in the GGUF chat template) gates both `/api/chat` and `/api/generate` after resolve: hard template evidence produces a 400 naming the cause and remedy; a missing or unreadable template (safetensors rows, legacy files) fails open — never a new failure class on shapes we cannot judge.
- **OpenAI `reasoning_effort` on `/v1/chat/completions` reaches the engine.** llama-server ignores the top-level field, so /v1 clients tuning reasoning got silent no-ops (200, no effect). The chat-completions lane now bridges a non-empty top-level `reasoning_effort` into `chat_template_kwargs.reasoning_effort` — explicit user kwargs win, the top-level field stays for children that read it. `completions`/`responses` paths are untouched (unverified child surfaces), and the file's zero-rewriting contract now names this and `stream_options.include_usage` as the two deliberate injections.
- **Anthropic `thinking.budget_tokens` is honored in `/v1/messages`.** The lane preserved thinking blocks but ignored the request budget. `thinking{type,budget_tokens}` now maps to `chat_template_kwargs {thinking, enable_thinking[, thinking_budget]}`, and `budget_tokens >= max_tokens` is rejected with Anthropic's own `invalid_request_error` shape (spec rule) instead of loading a model that can only emit thinking. Enabled-without-budget is a documented leniency (flags only).
- **`pallama run` renders thinking deltas dimmed on a real terminal.** The streaming REPL printed only `message.content` — reasoning vanished. On a TTY, thinking deltas render inline dimmed (ANSI 2m) with a single separator before the first content token; piped output omits thinking entirely so `jq`/scripts stay clean.
- **`pallama bench`/`tune` find llama-bench in any engine layout and teach the real remedy when it is absent.** The locator hardcoded the classic `engines/<tag>/llama-<tag>/` path, so per-arch slim assets (which unpack into vendor-suffixed dirs like `llama-b10948-bin-ubuntu-cuda-13.0-sm89-x64`) missed the bench binary even if present — and its error sent users down a dead end (`pallama engine update` fetches another server-only slim asset). The search now walks each llamacpp engine dir for the real layout (reusing the engine module's symlink-safe walk; active engine first, shallowest match = deterministic), only accepts a `llama-bench` with `llama-server` in the same dir (the Tuner spawns load/replica probes from the bench dir — a tool-only dir would fail confusingly later), skips mistralrs/sglang rows entirely, and the error names remedies that actually produce a bench binary: `pallama engine build cuda` (the source lane builds the full tool set) or `pallama engine local <dir>` for a full bundle. The validator's engine lanes now run a REAL switch dance when two or more server-bearing engines exist (slim boxes previously fell into a fail-all branch despite having two), and its bench/tune lanes boundary honestly — carrying the remedy text — when no llama-bench is installed, instead of failing red.
- **Colon `model:tag` names resolve by tag, not prefix luck.** The gateway's model resolver stripped everything after `:` and then prefix-matched — the tag was silently discarded. With one `qwen2.5-0.5b` row that accident resolved; the safetensors lane's `qwen2.5-0.5b-instruct` / `qwen2.5-1.5b-instruct` rows exposed it as `model "qwen2.5:0.5b" is ambiguous: [...]` (404) on every gateway surface (`/v1/chat/completions`, `/api/chat`, show, sessions — all ride one choke point). The resolver now applies the store's documented shared rule FIRST — exact case-insensitive match, then the `:`→`-` swap onto an existing row — before the prefix/Levenshtein ladder; the tag is never discarded. A wrong tag (`qwen2.5:7b`) still fails with the ambiguity teaching, and typos keep their `did you mean` hint. The resolver had zero covering tests (why this regressed silently); it now carries five pins including the exact 3-row store shape that failed live.
- **The `num_ctx` respawn OOM storm (21× HTTP 502 from one doomed pin).** A request pinning `options.num_ctx` larger than the warmup shape evicted the instance and re-spawned with the pin — but the profile compiler only WARNED "pinned ctx exceeds budget — will likely fail" and proceeded, so the child died at context creation on every spawn attempt while each retried request re-pinned it (breaker-blind: spawn-phase deaths never counted toward the circuit). Three root-cause layers, one shared verdict: (1) the unified-KV 2b fit and the gateway's `num_ctx` preflight now judge through one `unified_ctx_verdict` — a hostable pin HONORS itself by raising the `--cache-ram` budget within the 60% RAM guard (2a-bis floor algebra; a floor that already covers the pin raises nothing), an unhostable one is refused ONCE with teaching (lower `num_ctx` / smaller quant `pallama fit` / `cache_type = "q8_0"` / `kv_unified = false`) — pre-spawn 400 on the request path, compile error on CLI spawns; (2) auto slots at a pinned ctx are now pool-validated: capacity axes derived np4 at a 32768 pin on a 262k-trained model (pool 131072 no legal budget could host) while np1 hosted the same pin — auto walks down to the largest hostable count (explicit `slots` pins never walk; they conflict loudly); (3) spawn-phase child deaths (load-timeout retries exhausted with `child_died`) now count toward the circuit breaker, so a true death-loop opens `CircuitOpen` (503, `ps --reset`) instead of 502-ing forever — clean churn and load-timeout stay uncounted (wave-7 contract). Explicit `extra_args --cache-ram` owns the budget absolutely (no clamp, no floor, no raise; an unfit pin under it teaches the minimum value to set), and the budget is resolved ONCE before slot derivation so the walk judges against the exact budget the fit sees. Live-proven on qwen3.5-9b/13.6 GiB: pin 32768 walked np4→np1 and answered through the raised-floor budget; pin 131072 returned a single 400 and the running instance survived untouched.
- Request-level `chat_template_kwargs` now forwards 1:1 to the engine on both `/api/chat` and `/api/generate` (audit H1: the field was silently dropped — power users coming from llama-server/vllm/sglang lost template switches with no error). Explicit user keys beat the `think`-derived pair (`think` fills only missing keys); e.g. `"think": true` + `{"chat_template_kwargs": {"enable_thinking": false}` now disables thinking, as does `"enable_thinking": false` alone.
- Engine retention is now scoped per engine kind: installing a mistral.rs or sglang engine no longer prunes the newest llama.cpp builds (and vice versa) — each lane keeps its own newest `KEEP_TAGS`; active and `local` stay protected on top. Kind-blind retention deleted cross-lane engines on back-to-back installs (mistralrs install pruned the active CUDA engine; the CUDA reinstall then pruned sglang).
- **Interrupted large downloads no longer 416-loop (found live during the SGLang validation pull).** The parallel downloader preallocates `.part` files to full length (sparse holes); if its resume sidecar was lost mid-download, the next attempt declined the parallel lane and the classic lane sent `bytes=<full-length>-` against a sparse file — 416 with no self-heal. Now: a full-length `.part` without a sidecar is re-fetched in place by the parallel lane (chunk writes are idempotent), a short sidecar-less `.part` is handed to the classic lane from zero (hasher reset, no Range header), and only a mismatched sidecar is discarded. Pinned by wiremock 416-mount tests plus a live interrupted-pull rerun.
- **Test-only port TOCTOU (1-in-10 full-suite flakes).** Three engine tests asserted connect-fails on ports they had just bound-and-dropped; under parallel runs another test's port-0 bind could rebind the released port, flipping "dead endpoint" fixtures alive. Replaced by a `dead_port()` helper that verifies refusal (retrying on a different port if something rebinds) — 10/10 consecutive full-suite runs green.
- Engine probes (`--version`/`--help`/device census) can no longer deadlock or fail blind: `probe_output` drained piped stdout/stderr only after the child exited, so any probe target writing more than the ~64 KiB pipe buffer blocked on write, never exited, and surfaced as a misleading "timed out or failed to spawn" after a full 30 s deadline. Pipes are now drained on reader threads while the child runs (pinned by a 4 MiB flood test), and transient spawn failures (e.g. fork pressure) are logged with their `io::Error` instead of being silently folded into the timeout case.
- **`uninstall.sh --yes` no longer implies `--remove-models` (data-loss fix, found in the 2026-09-13 uninstall/install loop).** Flag parsing aliased `--remove-models | --yes` (last-flag-wins), so `--keep-models --yes` silently deleted every downloaded model. `--yes` now means "skip prompts using SAFE defaults (models kept)" and conflicts hard with `--remove-models`. Also: privilege preflight refuses non-tty unprivileged runs instead of half-deleting (systemd crash-loop split-brain); root-run uninstalls resolve the real user's home via `SUDO_USER` instead of `/root`; SQLite WAL sidecars (`pallama.db-wal/-shm`) are removed with the db; and a trailing `[ … ] && status` made flawless `--remove-models` runs exit 1 (a false test as the script's last command becomes its exit status — POSIX footgun, now an explicit `if` + `exit 0`).
- **`install.sh` now runs every user-state step as the invoking user.** Under `sudo`, toolchain probing, `migrate`, engine bootstrap, model pulls, and the source build previously ran as root — populating `/root` (rustup into `/root/.cargo`, root-owned `target/`, engine rows the service user could never see → permanent engine-less crash-loop). All steps route through an `as_user` wrapper; the health poll falls back to port 11435 (was 11434 — ollama's port, answered green while pallama was dead); fresh installs `enable` without `--now` and start the unit only after the engine bootstrap (previously measured 86 `no engine installed` restarts during the install window).
- Test-suite leaks: `unit__run_list_devices__parses_live_census_output` hand-rolled a `/tmp/pallama-census-<pid>` fixture dir and removed only the script file, and the engine-manager suite's `stub_engine_dir` staged pid-keyed dirs it never removed — together one leaked dir per `cargo test` run (200+ had accumulated). Both fixtures now own `tempfile` guards that clean up on drop.
- `pallama engine update` on an NVIDIA box now warns with the exact recovery command whenever the prebuilt CUDA lane is dropped — driver older than the overlay's newest asset (both versions named), overlay release not yet published for a fresh upstream tag, driver CUDA capability unprobed (reboot path), or pre-CUDA-12 drivers. These lane drops were `info`-level and easy to miss.

### Added
- `pallama doctor` now renders six grouped sections (SYSTEM, GPU, ENGINES, MODELS, CHANNELS, RUNTIME) with per-section rollups; `--flat` keeps the legacy single table and `--json` emits one object per check for scripting. New checks: GPU driver/CUDA + VRAM headroom + largest-model fit + engine-asset SM-arch match; per-kind engine inventory (llamacpp/mistralrs/sglang, tag + provenance + size) and retention vs the keep-policy; compiler-cache channel (ccache/sccache); daemon uptime + restart count, tempdir hygiene, bench baseline presence, and model type mix (gguf/safetensors/stale).
- `pallama list` gained a TYPE column: `gguf` (single-file GGUF), `safetensors` (HF-style directory for the mistralrs/sglang lanes), `dir?` (unexpected dir layout — inspect with `pallama show`), `missing!` (stale row whose file is gone, matching the VISION column convention).
  The table now auto-sizes every column to its widest cell (long ARCH values like `Qwen2ForCausalLM` no longer overlap CTX), SIZE/CTX right-align, NAME/ARCH truncate with `…` — char-safe, fixing a latent byte-slice panic on multibyte names.
- **Per-arch CUDA engine assets (~60% smaller downloads).** The prebuilt CUDA channel now publishes one slim asset per GPU architecture (`llama-bNNNN-bin-ubuntu-cuda-13.0-sm89-x64.tar.gz`, 9 assets per tag across CUDA 12.8/13.0) instead of a fat multi-arch tarball; the sm120 asset carries PTX for forward JIT on future GPUs. `pallama engine update` picks the exact asset for the local GPU's compute capability, falls back to legacy fat assets during the transition, then to the PTX asset for GPUs newer than sm120. CI wall time drops in parallel (per-arch jobs ~15-20min with ccache vs ~50min serial fat build).
- mistral.rs profiles now reach the FULL `mistralrs serve` surface: every flag the probed engine binary offers forwards verbatim (bools like `--flash-attn` and valued pairs like `--dtype bf16` — attention method, dtype, KV-cache quant, prefix cache, token source, LoRA, device mapping, log control), gated by the same manifest capability check as the sglang lane. Explicit profile values beat derived ones (`--max-model-len` from the profile wins over ctx synthesis, no duplicates); llama.cpp-only dialect words are still dropped with the compiler's existing warning.
- **SGLang engine kind + safetensors model lane (2026-09-14).** `pallama engine install --kind sglang [version]` installs a pinned venv (`sglang==0.5.19` default; uv lane with `--prerelease=allow` for transitive pre-release pins, pip fallback; `ninja` installed and venv-PATH exported in the shim so flashinfer JIT can compile; ≥10 GiB disk preflight; orphan-dir cleanup on every failure path; non-Linux fails fast with a teaching error). `pallama pull <hf-repo>` gains a safetensors lane for repos without GGUFs: root-level shards + config/tokenizer allowlist land in `models/<name>.d/` with LFS sha256 verification, `.part` resume, shard-index coverage checks, and idempotent repulls; GGUF repos keep the existing lane and a repo offering both teaches which lane won. SGLang models serve through the same single-port gateway — `/api/chat`, `/api/generate`, OpenAI `/v1/*`, Anthropic `/v1/messages` — with zero client change and per-child auth. The profile compiler adds a low-VRAM ladder (full → KV fp8 → CPU offload bounded by host RAM → refusal with weights/KV/VRAM numbers — it can never emit a spawn that OOM-crash-loops), `--max-running-requests` from slots, EAGLE3 speculative pair, ~20 first-class tuning knobs under `model_overrides.<name>.sglang.*`, a reserved-flag guard on `extra_args` for lifecycle/security flags the ladder owns, and portability-first attention/sampling backend defaults (`triton`/`pytorch`, flag-gated and overridable) for boxes whose system nvcc can't satisfy flashinfer JIT. Unix child transport is refused with a teaching error (SGLang is TCP-only).
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
- **`pallama doctor` flags model-dir orphans**: GGUF files in the
  models folder that no store row owns (the "folder shows many, `list`
  shows few" confusion) are reported by the models check with a
  `pallama import <file> --name <n>` hint — import to register or
  delete to reclaim. Hardlink twins of registered files (import's
  dedup shape, e.g. case-duplicate leaves) are counted separately with
  the teaching that deleting them reclaims no space (same inode);
  attached projectors are never orphans (they ride their model row).
  Unix uses (dev,ino) twin detection; Windows counts all unreferenced
  GGUFs as orphans. Clean boxes emit byte-identical output (suffix
  only appears when there is something to say).
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
- **GPU driver preflight — the zero-touch last mile**: the Linux
  installer now detects PCI GPU hardware before the engine bootstrap
  (`lspci`; pciutils provisioned when missing) and, when the driver
  userspace is absent, installs it from **first-party distro repos
  only** (announce-then-act, never fatal): NVIDIA via apt's
  `nvidia-driver` metapackage / pacman's `nvidia nvidia-utils`; dnf
  auto-installs `akmod-nvidia` only when RPMFusion is already enabled;
  zypper (and every third-party-only lane) prints the exact command
  instead of running it. AMD/Intel cards without a Vulkan ICD get
  `mesa-vulkan-drivers`/`vulkan-radeon`/`vulkan-intel`. A driver
  install ends with the loud chain: REBOOT, then `pallama engine
  update` — which auto-picks the newest CUDA build the driver supports
  (runtimes bundled; no CUDA toolkit). Opt out with
  `PALLAMA_AUTO_DRIVER=0` (acknowledged even when the engine bootstrap
  is off; skipped entirely under `PALLAMA_INSTALL_ENGINE=0`). The
  Windows installer is advise-only (detects NVIDIA hardware without
  `nvidia-smi`, links the driver download). `pallama doctor` gains
  `nvidia driver` / `vulkan driver` WARN rows when PCI GPU hardware is
  present but its driver is not (new pure helpers
  `probe::parse_pci_vendors`/`pci_gpu_vendors`).
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
