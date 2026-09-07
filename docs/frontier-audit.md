# Pallama Frontier Audit — 2026-09-06

> STATUS 2026-09-06 (same day, implementation wave): TOP moves 1,2,3,4,5,6,
> 8, 9, 10 + whisper lane are IMPLEMENTED (per-key tier incl. quotas/usage;
> `pallama quantize`; client-compat pins + `pallama launch`; OTLP export;
> TLS+CORS; Responses `previous_response_id` registry; win scheduled-task
> service via install.ps1 -WithService; `tune --slots` live search;
> `[[remotes]]` routing covering vLLM/MLX; `pallama whisper`; brew/scoop/
> winget manifests under packaging/). Skipped by user directive: Docker.
> Deferred honestly: full MLX/vLLM engine-manager integrations (the
> `[[remotes]]` lane covers serving them today; native pull/tune for those
> backends is a future wave). 258 tests green, clippy clean.

Deep-search audit against the live frontier (all external data fetched 2026-09-06;
code refs verified against this tree). Goal: world-class in ALL aspects — or an
honest refusal line explaining why not.

## Frontier snapshot (first-party, fetched today)

| Player | Version (date) | Frontier bar they set |
|---|---|---|
| llama.cpp | **b10819 (Sep 5) — pallama's engine IS current**; v0.4.0 stable (Sep 4) | server: per-slot ctx limit (#24124), `data:` URL media (#27735), `preserve_reasoning` default (#28174), reject prefilled assistant tool calls (#27626), synthetic spec acceptance (#27711); new archs: Qwen3.8-Flash-Next, Nemotron-3-Puzzle, DSpark-for-Nemotron-3.5; release tarballs pack **whole `build/bin`** (release.yml:215) incl. llama-bench (+quantize-class tools); llama.app hosted UI exists |
| ollama | **v0.33.3 (Sep 3)** | desktop app + onboarding; **`ollama launch dsh|muse`** (backs agent CLIs: DeepSeek Harness, Muse Code); Responses API + web search; resolved-metadata caching halved TTFT (~995→524 ms); **MLX on Linux amd64** (asset `ollama-linux-amd64-mlx.tar.zst`); Qwen3.8 system-message normalization |
| vLLM | **v0.27.0** | multi-tier KV-cache offloading (fs secondary tier), sleep-mode weight reload, MRv2, **experimental Rust frontend**, batch-invariant FP8 (+28.9% e2e), NVFP4, non-root Docker, DP supervisor — kernel/datacenter class (Tier C for us) |
| SGLang | v0.5.x line | radix prefix cache (6.4× sourced in perf-roadmap), spec artillery — Tier C for us |
| llamactl | current README | multi-backend (llama.cpp/MLX/vLLM), React web dashboard, **remote instances + central routing**, instance groups w/ per-group limits, Anthropic `/v1/messages` endpoint, Docker-wrapped backends |
| LiteLLM (gateway SOTA) | docs today | per-key rate limits, budgets/spend, **virtual keys w/ model scoping**, OTel export, MCP/agent gateway, guardrails |
| OpenAI (protocol SOTA) | docs today | Responses API: `previous_response_id` chaining + Conversations API, `store:false`, **strict-by-default function calling**, `text.format` structured outputs; clients (Codex-CLI class) are Responses-native |

Pallama-verified strengths (do NOT rebuild): engine currency b10819 = latest;
34 routes (20 OpenAI-family incl. `/v1/messages`→child `server.cpp:263`, 14 ollama);
sentinel semantic layer + enforce + persistence; session KV checkpoints; router mode;
KV-quant ladder; persistent spec cache; pull precheck; zero-tax proxy; priority queue;
doctor/why/watch; install story (7 targets) + digest-verified self-upgrade.

## TOP moves to world-class (ranked)

| # | Move | Why (frontier evidence) | Pallama today | Effort |
|---|---|---|---|---|
| 1 | **Per-key security tier**: virtual keys `{key, name, models[], rpm, tpm/day-token budget}`, token-bucket at admission, per-key usage counters surfaced via `/api/keys` + `pallama keys` | LiteLLM sets this bar; api_keys is a flat `Vec<String>` (config.rs:38, bearer check lib.rs:43) — 2026 gateway SOTA requires scoping+quotas+accounting | MISSING | medium |
| 2 | **`pallama quantize <model> --type Q4_K_M`**: orchestrate llama-quantize from the engine dir → new sibling GGUF + `pallama list` provenance | engine tarballs pack the whole `build/bin` (release.yml:215 `tar -C build/bin .`); llama-quantize is a standard target — **verify presence per engine at impl**; no local stack ships one-command quantize (LM Studio hides it in conversion; ollama: none) | MISSING — high-ROI, brand-aligned (plain-GGUF truth) | small-medium |
| 3 | **Client-compat CI suite** (pinned replay fixtures): Claude Code `/v1/messages` (+`anthropic-version`), Codex-CLI `/v1/responses`, Continue/Cline OpenAI, dsh/muse-class launchers; a `pallama launch` wrapper story mirroring `ollama launch` | ollama now SHIPS `ollama launch dsh/muse` (v0.32.x notes); child serves `/v1/messages` natively (server.cpp:263) but pallama has zero anthropic handling/tests (grep clean) | routes exist; no pinned e2e client suite | small |
| 4 | **OTLP/OTel export** (traces + per-request spans; metrics already exist) + per-key token accounting feed from #1 | LiteLLM ships OTel; pallama has `/metrics` + `x-pallama-trace-id` logs only (lib.rs /metrics route) | MISSING | small-medium |
| 5 | **TLS/HTTPS termination option + CORS policy knob** at the gateway | remote/homelab lane (llamactl remote instances; pallama `host` configurable — 0.0.0.0-bind complaint class solved by default 127.0.0.1 but remote use needs encryption); child agent-mode CORS exists upstream | MISSING | medium |
| 6 | **Responses-API depth: gateway conversation registry** for `previous_response_id` chaining + `store` semantics; strict-mode tool calling (`strict:false` passthrough) | OpenAI migration guide (fetched): chaining is THE Responses differentiator; upstream child has NO `previous_response_id`/`store` support (grep empty in server.cpp) → pure gateway value-add no competitor local stack has | MISSING (impossible upstream today — our lane) | medium |
| 7 | **Docker image for pallama itself** (multi-arch, non-root, engine-bake variant) + **brew/scoop/winget/nix** taps | vLLM non-root image; llamactl Docker-ready; pallama has install.sh/ps1 only | MISSING | small-medium |
| 8 | **Windows service registration** (`--with-service`, install.ps1) | own gap-analysis row 23; ollama ships install.ps1 service | PARTIAL (binary+installer, no service) | small |
| 9 | **Slots auto-tune** (tune --search picks `-np` by measured concurrency) | own frontier-comparison roadmap #1; batching = the remaining cheap local win | OPEN | small |
| 10 | **Remote engine instances** (register remote llama-server endpoints, route + ps across hosts) | llamactl ships this exact feature (README fetched) | MISSING (rpc_servers = layer-split only, config.rs:40) | medium-large |
| 11 | **Engine-trait plugins: MLX (now Linux-capable per ollama v0.33.3 asset) and vLLM adapters, opt-in** | ollama MLX-on-Linux changes MLX calculus; `Arc<dyn Engine>` ready (engine_impl.rs); keep upstream-first brand — plugins opt-in, never default | MISSING by design; trait-ready | large |
| 12 | **whisper.cpp sibling engine** (`pallama whisper file.mp3`) | same GGML-org release shape (prior research); STT lane | NOT BUILT (documented decision) | medium |

## Full aspect matrix

| Aspect | Verdict | Bar-setter | Note |
|---|---|---|---|
| API surface breadth | **AT BAR** | OpenAI/ollama | 34 routes incl. Responses/audio/FIM/anthropic-forward; depth items = TOP #6 |
| Semantic reliability | **BEYOND** (unique) | — | sentinel + enforce + `why`/`watch`; nobody ships this |
| Session continuity | **BEYOND** (unique) | — | slot-save checkpoints survive unload+restart |
| Perf orchestration | **AT BAR local** | vLLM (conceded @c=64, sourced 23×) | slots auto-tune (#9) + prefix routing (deferred B1) remain |
| Spec decoding | **AT BAR** | upstream | ngram+draft+persistent cache; DSpark/DFlash arrive via engine manifest — check `--spec-type` exposure after each engine update |
| Engine mgmt | **BEYOND** | ollama | side-by-side, rollback, digest-verified, capability manifests, self-upgrade |
| Model store | **BEYOND** | ollama | plain GGUF, shard sets, pull precheck, hardlink cp/create |
| Multimodal | **AT BAR** | LM Studio | vision e2e (projectors serve); audio-in forwarded; TTS = upstream lane |
| Security | **BEHIND** | LiteLLM | flat api_keys, no TLS/CORS/rate/quota — TOP #1/#5 |
| Observability | **PARTIAL** | LiteLLM/vLLM | /metrics+trace-id good; no OTel — TOP #4 |
| Distributed | **BEHIND** | llamactl | no remote instances — TOP #10 |
| Packaging/deploy | **PARTIAL** | ollama | 7-target binaries+installers; no Docker/packages/service — TOP #7/#8 |
| Agent ecosystem | **PARTIAL** | ollama launch | API-compat yes; no pinned client suite / launch story — TOP #3 |
| Quantization tooling | **BEHIND** | LM Studio | `pallama quantize` — TOP #2 |
| UX/CLI | **AT BAR+** | ollama | full cmd parity + doctor/why/watch; REPL |
| Docs/website | **AT BAR** | — | README + 5 docs; website when public |

## Refused by design (do not "fix")

- Desktop app / own web dashboard — user directive: CLI-only. (Child WebUI route
  forwarding remains a small opt-in if ever wanted.)
- gRPC serving — datacenter feature; scope creep locally.
- Tensor parallelism / paged kernels / FP8 / PD-disagg — kernel land (Tier C);
  arrives (or doesn't) with upstream. Forking = the thing pallama exists to avoid.
- Content-policy guardrails — local-first, single-user trust model; sentinel's
  semantic checks are the local-relevant subset.
- Cloud registry / accounts (`push/login`) — refused surfaces, tested.

## Upstream v0.4.0 items already inside our b10819 binary — exposure checklist

Engine ships them; confirm pallama config surface at next manifest probe:
`--ctx-per-slot`-class flag (#24124), `data:` URL media (client-side usage),
`preserve_reasoning` default (#28174 — our `reasoning_format` knob interplay),
prefilled-tool-call rejection (#27626 — sentinel should classify this child error),
synthetic spec acceptance (#27711 — tune grid candidate).

## Sources (fetched 2026-09-06)

- github.com/ggml-org/llama.cpp/releases (b10819 latest, v0.4.0 notes)
- github.com/ollama/ollama/releases (v0.33.3 + v0.32.x notes)
- github.com/vllm-project/vllm/releases (v0.27.0)
- github.com/sgl-project/sglang/releases
- github.com/lordmathis/llamactl (README)
- docs.litellm.ai/docs/simple_proxy
- developers.openai.com/api/docs/guides/migrate-to-responses
- references/llama.cpp-master (server.cpp:263, release.yml:215, arg.cpp)
- this tree: config.rs, lib.rs, gateway/src/*, engine/gh.rs
