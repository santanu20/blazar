# Ollama complaint audit — verified against pallama's code

Method: complaint classes compiled from ollama's issue tracker, Reddit r/ollama,
community troubleshooting guides, and post-mortems (sources cited per row);
each pallama resolution verified against this repository's code (file refs) or
its test suite. Statuses are honest: PARTIAL/NOT-FIXED rows are listed with
what is missing.

Legend: **FIXED** = resolved and test-verified. **PARTIAL** = mitigated with
gaps. **NO** = not solved (tracked as roadmap).

## Verified fixes

| # | Complaint (source) | ollama behavior | pallama resolution | Status / code |
|---|---|---|---|---|
| 1 | Engine fork lags llama.cpp; new architectures break on old vendored runtimes (tracker, recurring) | bundled fork, weeks behind | zero fork: official upstream release binaries, sha-verified, side-by-side, `engine update/use/rollback` | FIXED — `runtime/src/engine/{gh,mod}.rs` |
| 2 | Slower than raw llama.cpp (community benchmarks) | wrapper overhead | byte-stream proxy, zero body parsing on OpenAI path; measured p50 32.1 ms vs 33.2 ms direct (soak, step H) | FIXED — `gateway/src/proxy.rs` |
| 3 | Modelfile copies the whole blob to change one parameter; silent `{{ .Prompt }}` fallback | layer-copied derived blobs | no Modelfile: GGUF is truth, child always runs `--jinja` (embedded template), params per-request or overlay | FIXED — `core/src/profile.rs` rule 1 |
| 4 | Hashed blob store lock-in (`~/.ollama/models/blobs/sha256-…`) | opaque content-addressed store | plain `.gguf` files under `~/.local/share/pallama/models/`, any tool can use them | FIXED — `runtime/src/models.rs` |
| 5 | Limited quant selection; registry bottleneck | curated registry, one quant per tag usually | direct HF pull, any repo/quant, shard sets native, quant fallback (smallest) | FIXED — `runtime/src/hf.rs` |
| 6 | Token exfiltration class (CVE-2025-51471) | redirect handling sent tokens cross-host | HTTPS-only + redirect-host allowlist + token only on first-party host (wiremock-verified: CDN never sees `Authorization`) | FIXED — `runtime/src/hf.rs`, integration test |
| 7 | Cloud pivot / silent off-machine routing fears | cloud features announced | hard local-only, stated in `--help` and startup banner; only egress = user-initiated downloads | FIXED |
| 8 | Opaque 2048 default ctx; silent truncation | default 2048, grows silently | default ctx 16384, shown in `ps`; ctx overflow errors surfaced verbatim with fix hints | FIXED — `core/src/profile.rs`, gateway 400s |
| 9 | KV cache pre-allocated for the model's full declared context (65 GB VRAM for a 7B model; dev.to post-mortem below) | loads at GGUF-declared 128K regardless of need | loads at `default_ctx` (16384); model `context_length` only ever clamps DOWN; per-request `num_ctx` restarts instance once at that size | FIXED — `profile.rs` ctx rules; `gateway/src/ollama.rs:341` |
| 10 | `OLLAMA_NUM_CTX` ignored; per-request `num_ctx` won't shrink a loaded model (dev.to) | env var ignored per-model; no shrink | config + overlay + per-request restart-once; `0`-evict semantics explicit | FIXED (ollama API) — see row 15 for OpenAI-path caveat |
| 11 | `keep_alive` confusion (units, `0` vs `-1`) | inconsistent honoring | honored in both APIs: `N` pins N s, `0` evicts after request, `-1` forever; `/api/ps` shows countdown | FIXED — `gateway/src/ollama.rs:355-367` |
| 12 | Hidden concurrency; requests queue opaquely one-at-a-time (tracker, guides) | OLLAMA_NUM_PARALLEL env, invisible | `ps` shows slots/ctx/in-flight; slots-aware admission gate queues at the gateway with priority (`X-Pallama-Priority`); `-np` configurable | FIXED (visibility) — note: default `slots = 1` = full-speed single client, raise for concurrency |
| 13 | Long-context OOM / unbounded cache RAM (ROCm issue #5741 class) | no cache budget knob | `cache_ram_mb` config → `--cache-ram`; **auto-capped at 30% of physical RAM** (measured: unclamped 8 GiB budget on a 13 GiB box → 8.3 GiB child RSS plateau, swap death); KV quant (`cache-type-k q8_0`) when KV+weights > 90% VRAM | FIXED — `profile.rs` rule 6 + rule 12 clamp (this session) |
| 14 | Blind pulls: no VRAM fit preview before multi-GB download | pull-then-fail | `pallama fit` previews fit + quant alternatives from HF metadata pre-download; `search` shows sizes | FIXED — `core/src/profile.rs` fit rows, `runtime/src/hf.rs` |
| 15 | OpenAI-compatible endpoint has no way to set context (dev.to) | no `num_ctx` on `/v1/*` | per-request ctx honored on the ollama API (`options.num_ctx`); OpenAI path accepts the `X-Pallama-Num-Ctx` extension header (the protocol has no such field) with the same restart-once semantics — non-numeric values get a named 400, never silence | FIXED (2026-09-05, sentinel wave) — `openai.rs`, integration test |
| 16 | Env-var sprawl (`OLLAMA_*` zoo) | 20+ env vars | one validated `config.toml`, `pallama config` surface, env overrides documented | FIXED — `core/src/config.rs` |
| 17 | Marketing model names hide the actual artifact | aliases like `llama3` | model name = actual HF repo name, always | FIXED |
| 18 | No rollback / update anxiety | in-place engine replace | engine versions side-by-side, keep-3 prune, `engine rollback`, atomic active-switch | FIXED — `runtime/src/engine/mod.rs` |
| 19 | Runs as root system service; user/group management friction | systemd `ollama` user + groups | user-local daemon, no sudo anywhere; auto-start on demand | FIXED — `cli/src/main.rs` ensure_daemon |
| 20 | Attribution dodging (llama.cpp/ggml credit) | minimal | version banner, README credit section, engine provenance in store | FIXED |
| 21 | Log opacity when debugging slow/failed requests | terse server logs | per-request access log with trace id (`x-pallama-trace-id`), merged child `/metrics`, daemon log with rotation | FIXED — note: a *manually* redirected daemon logs wherever you point it (live finding this session) |

## Honest gaps (NOT fixed or partial)

| # | Complaint class | Status | What pallama does / what's missing |
|---|---|---|---|
| 22 | Multi-GPU layer split / "uses GPU or CPU, never both" (r/ollama) | PARTIAL | `--gpu-layers auto` handles mixed split (better than nothing); `rpc_servers` config splits layers across boxes (`--rpc`). No tensor parallelism — llama.cpp does not provide it (see frontier doc) |
| 23 | Windows service management (start-on-boot, tray) | PARTIAL | Windows binary + installer exist; no service/scheduled-task management yet (`--with-systemd-unit` is Linux/macOS only). Roadmap |
| 24 | Self-update of the orchestrator binary | **FIXED** | `pallama upgrade [--version] [--dry-run]`: resolves the release, verifies the API sha256 digest, atomic self-replace (unix; Windows stages the binary with instructions); e2e-tested incl. tamper rejection |
| 25 | gRPC serving API | NO | HTTP only (OpenAI + ollama). gRPC is a datacenter-serving feature; see frontier doc |

## New fixed rows (beat-ollama CLI wave, 2026-09-05)

| # | Complaint class | ollama behavior | pallama resolution | Status / code |
|---|---|---|---|---|
| 26 | Modern OpenAI surface (Responses API) missing | chat-completions era only | `/v1/responses` (+ 10 more upstream routes: audio transcriptions, FIM infill, control vectors, token counting, tokenize/apply-template) byte-stream proxied with model routing | FIXED — `gateway/src/lib.rs`, `openai.rs` |
| 27 | No KV-cache control; VRAM cliffs at long ctx | fixed f16 KV | capacity ladder none → q8_0 → q4_0 at 90% VRAM + `cache_type` pin + `fit` CTX@KV_Q8 column | FIXED — `core/src/profile.rs` rule 6 |
| 28 | Context lost on unload/restart | re-prompt from scratch | `pallama session save/restore`: slot KV checkpoints surviving unload + daemon restarts (engine-gated, per-model dirs) | FIXED — `gateway/src/ollama.rs` session handlers, CLI `session` |
| 29 | Speculative cache cold after restart | none (no ngram mode) | persistent `--lookup-cache-dynamic` per model — speculation warm from first request after respawn | FIXED — `profile.rs` rule 14 |
| 30 | No idea why it's broken | opaque logs, GitHub-issue archaeology | `pallama doctor`: config, port conflicts (names the ollama-11434 class), engine manifest, hardware, disk headroom, model health — one table, offline | FIXED — `cli/src/main.rs` doctor |
| 31 | No context extension; hard ctx ceilings | trained-window only | `ctx_extend` YaRN knob (validated range, quality warning surfaced) | FIXED — `profile.rs` rule 16 |
| 32 | Coarse MoE offload | none | `cpu_moe_n` count-based expert offload + `override_tensor` per-tensor patterns, per-model overlays | FIXED — `profile.rs` rules 17-18 |
| 33 | One-process-per-model only; no shared serving | one runner per model, port sprawl | `router = true`: ONE child serves every model (upstream router mode; preset INI auto-generated from the store; engine-native autoload + LRU; `ps`/`stop` adapted) | FIXED — supervisor `spawn_router_instance`, gateway `ps_router` |
| 34 | Bad pulls discovered only at load time | pull succeeds silently, fails on run | post-download GGUF header validation: warning names the failure, the `rm` hint, and sibling quant alternatives (CLI, event bus, `/api/pull` NDJSON) | FIXED — `runtime/src/hf.rs` `gguf_health_warning` |
| 35 | No prefix affinity between concurrent requests | none | `slot_prompt_similarity` knob → upstream `--slot-prompt-similarity` (gateway-level slot steering impossible upstream: no `id_slot` request field — verified, documented) | PARTIAL (child-side only, by upstream constraint) |

## Sentinel wave (2026-09-05): the "200 but nothing happened" class

Backlog source: 2026-09-05 community-sentiment corpus (attachment; classes match the
sources below). The verified gap it exposed: the gateway proved the pipe worked but
never checked whether the semantic operation succeeded.

| # | Complaint class | ollama behavior | pallama resolution | Status / code |
|---|---|---|---|---|
| 36 | Silent semantic failures: truncation, empty 200s, reasoning-with-no-answer, stalled streams — diagnosed by users building external proxies | 200 + whatever the model emitted; debugging = guesswork | sentinel: warn-only observation of every chat-family request (both APIs); detections land in a bounded ring queryable via `pallama why [trace]` + `GET /api/why` + `tracing` target `pallama::sentinel`. Zero body mutation; analyzer on a bounded drop-oldest side-channel (zero-tax preserved, parity-tested) | FIXED — `gateway/src/sentinel.rs`, integration tests |
| 37 | Tool calling broken locally: plain text instead of calls, malformed arguments, hallucinated tools, schema violations | no serving-layer validation; blame lands on "the model" | accumulated tool-call validation: arguments must parse as JSON, names checked against the request's tool list, args validated against the tool's own `parameters` schema (jsonschema, local-only build); PLUS a pre-inference template-capability check — tools requested but the GGUF template cannot render them = `x-pallama-warnings: template_no_tools` header before the first token (the actual root cause of the plain-text failure mode) | FIXED — `sentinel.rs` (`tool_detections`, `template_support`), `GgufMeta.chat_template` |

Boundaries, honestly: response-side detections cannot ride response headers (bytes
are immutable by design — the warn-only contract); they reach the client via `why`/logs.
Ring/channel/accumulation all bounded (cap 256 records / 64-slot channel / 4 MiB
accumulation; overflow marks the record degraded, never blocks). Kill-switch:
`sentinel = false` disables every hook (tested).

Closed since (2026-09-05, follow-up wave): `/v1/responses` event grammar parsed
(streamed function_call fragments merged, `incomplete`+`max_output_tokens` ->
truncation, `input_tokens`/`output_tokens` usage mapping) — both stream and
non-stream, warn path and enforce path. The ring now persists to a bounded JSONL at
`run/sentinel.jsonl` (cap 1 MiB, rotate keeping newest 128; corrupt lines skipped;
`why` survives daemon restarts — tested). `sentinel_enforce` (config) +
`X-Pallama-Enforce: 1|0` (per-request override, either direction): hard violations
(invalid tool args, unknown tool names, schema violations) -> 422 on NON-STREAM
chat requests on both APIs; streaming stays warn-only by physics (bytes already
sent); oversized bodies (> 4 MiB) pass through with a loud warn, never silently.

## Sources

- dev.to — "How Ollama Silently Ate 65GB of My VRAM" (ljkunal, 2026-03-30): KV pre-allocation at declared ctx, ignored `OLLAMA_NUM_CTX`, OpenAI endpoint ctx gap — https://dev.to/ljkunal/how-ollama-silently-ate-65gb-of-my-vram-and-how-i-fixed-it-22pf
- ollama issue #5741 — ROCm memory issues with long contexts — https://github.com/ollama/ollama/issues/5741
- r/ollama — multi-GPU/CPU utilization threads (e.g. "Ollama either used GPUs or CPUs, never both") — https://www.reddit.com/r/ollama/comments/1c5rj62/
- Community troubleshooting corpus: insiderllm.com ollama troubleshooting guide; masteraikit.com "Why is Ollama so slow"; glukhov.org "How Ollama handles parallel requests"
- CVE-2025-51471 — ollama API token exfiltration class
- Live verification on this machine (2026-09-05): cache-ram sizing pathology measured directly (sampler: 8.3 GiB child RSS, mem_avail floor 0.73 GiB) before the 30% clamp landed
