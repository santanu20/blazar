# Changelog

All notable changes to Pallama are documented here. Format follows
Keep a Changelog; versions follow SemVer. Earlier releases were not
tracked here.

## [Unreleased]

### Fixed
- **Engine rollback no longer dethrones healthy sglang/mistral.rs engines (2026-09-16).** The crash-path probe walked for a binary named `llama-server` only — structurally always false on the sglang (`sglang-server` shim) and mistral.rs (`mistralrs`) lanes, so ANY spawn failure on those engines triggered a false `--version probe failed` rollback to llamacpp (first live hit: a config-induced flashinfer JIT crash took the whole sglang lane out of `active`). New `verify_engine_binary` probes per kind — manifest `server_path` for llama/mistral.rs, the venv's `importlib.metadata` for sglang (the shim has no `--version` and would boot torch) — behind a real time budget (5s/15s, std-only thread + channel; the old probe documented a budget it never enforced and could block the spawn path forever). A config-induced crash now yields the request error alone; the genuine-ABI-crash heuristic (2+ models failing with a healthy probe) still rolls back. Live-proven by replaying the exact incident: 502 for the request, no rollback line, sglang stays active.

### Added
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
