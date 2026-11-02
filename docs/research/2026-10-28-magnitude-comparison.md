# Blazar vs. Magnitude - comprehensive architecture & capability comparison

**Date:** 2026-10-28
**Compared against:** `magnitudedev/magnitude` @ `e596431` (cloned 2026-10-28, HEAD dated 2026-10-27)
**Method:** full-tree inspection of both repositories - workspace/crate structure, LoC (tokei), serving API surface extraction, engine/kernel docs read in full (overview, tuning, compatibility, scheduling, drafting-and-verification), model catalog parse, GitHub/release metadata via API. No benchmarks were executed for this comparison; performance figures cited are each project's own committed claims with their stated methodology.

---

## 1. Executive summary

Magnitude and Blazar occupy **different layers of the same market** ("run open models locally, connect them to agents") and solve **different problems**:

- **Magnitude is a vertically integrated inference engine.** They wrote their own numerical compiler and runtime ("Seismic": language, IR, compiler, CPU/Metal/CUDA/Vulkan backends, ~250k LoC of toolchain alone) and hand-author per-family model kernels (9 families). Their edge is **raw per-device speed**: at first model load, a 60-second evidence-based autotuner searches kernel configurations on your exact silicon and every model ships with a tuned speculative drafter. Claim: **2.17-2.19x decode / 1.32-1.34x prefill vs Ollama** (avg of 11 popular models, Mac + NVIDIA).
- **Blazar is a horizontally integrated control plane.** It wraps the best existing engines (llama.cpp, mistral.rs, SGLang, MLX, sdcpp, whisper.cpp, piper) behind one gateway with **3 native protocol dialects** (OpenAI + Ollama + Anthropic), model ops, admission/scheduling, MCP tool-calling, multimodal, federation, and diagnostics. Its edge is **breadth, compatibility, and orchestration**: ~1.4% gateway overhead at C1, 3x better ITL tail than Ollama, 28x better TTFT p99 under C=4 load, plus entire surface categories Magnitude has no equivalent of.

**Honest bottom line:** on pure single-stream decode speed of Magnitude's curated 30-model catalog, Magnitude's tuned kernels will beat llama.cpp-lane Blazar - that is their entire thesis, and it is credible engineering. Blazar cannot match that layer without owning kernels. Conversely, Magnitude has no embeddings, no rerank, no audio/video/image generation, no Ollama dialect, no MCP gateway, no federation, no multi-engine routing, no model import from existing stores, and supports only 9 model families. Blazar's leverage is everything above the kernel.

---

## 2. Identity and project facts

| Dimension | Blazar | Magnitude |
|---|---|---|
| Repo | santanu20/blazar | magnitudedev/magnitude |
| Created | 2026-09-18 | 2026-06-12 |
| Positioning | Model server / inference runtime + gateway ("competitor of ollama/vllm") | "Run open models as fast as your hardware allows" - inference engine + desktop app |
| Stars / forks | 0 / 0 (personal, pre-community) | **7,395 / 523** (~4.5 months old) |
| Core contributors | 1 (solo) | 2 primary (thrgreenwald 803, anerli 734 commits) + corporate entity (magnitude.dev) |
| License | Dual MIT / Apache-2.0 | Apache-2.0 |
| Primary language | Rust (gateway) | Rust (inference core) + TypeScript/Bun (CLI, desktop) |
| Scale (code LoC) | ~180k Rust across 4 crates + test crate | ~565k Rust in inference workspace (60+ crates) + ~9k TS CLI + ~3k TS desktop + large TS packages/ agent stack |
| Test functions | 2,229 `#[test]` (nextest suite 2,232 green) + validate.py 396-check live battery | 3,371 `#[test]` in inference workspace + evaluation/harness + fixtures |
| Release cadence | v0.28.0 (9-platform CI, SBOM, crates.io, attestations) | CLI 0.2.7->0.2.13 in 17 days; desktop 50.x auto-update acceptance channel |
| Distribution | GitHub release binaries, crates.io, installer script | Signed installers (dmg / exe / deb / rpm / pacman; x64+arm64), auto-updating service, magnitude.dev installer API |
| Community surface | README + docs/ in repo, docs site | Discord, X, docs.magnitude.dev, magnitude.dev/benchmarks |

**Read:** Magnitude has ~40x the community traction and 2 devs + a company behind it; Blazar has deeper verification discipline per feature (receipt-gated shipping) but zero community exposure yet.

---

## 3. Architecture - the fundamental difference

### 3.1 Blazar: control plane around external engines

```
Clients (OpenAI SDK / Ollama clients / Anthropic SDK)
        │  3 native dialects, one port
   Blazar gateway (Rust daemon, ~180k LoC)
   routing | fit-planning | admission | spec governor | MCP loop |
   sessions/cache | jobs ledger | federation | diagnostics
        │  spawn/version/pin/rollback
   Engine children: llama.cpp | mistral.rs | SGLang | MLX | sdcpp | whisper.cpp | piper
        │
   GPU (engine-owned kernels)
```

- Engines are separate upstream binaries, versioned side-by-side, SHA-verified, regression-gated with auto-rollback (`engine update`, `engine use`).
- Blazar never touches tensor math. Its wins are scheduling, residency, cache, protocol, lifecycle (receipts in README/BENCHMARK.md).
- Blast radius of engine improvements is free (llama.cpp b11588->b11785 wave); kernel performance ceiling is the engines'.

### 3.2 Magnitude: own engine, own numerical compiler

```
Desktop (Electron) / CLI (Bun TS) / `magnitude serve`
        │
   service/api + service/server  (~4k LoC API, OpenAI-style REST + SSE)
        │
   engine: admission | batching | scheduling | generation | grammar | state
        │  model programs (typed)
   families/: llama | qwen | qwen35 | qwen4exp | gemma4 | lfm2 |
              nemotron_h | muse_glimmer | dflash   (hand-written kernels)
        │
   Seismic: lang (80k LoC) | compiler (46k) | runtime (50k) |
            backends metal/cuda/vulkan/cpu (34k) | ir (16k) | topology | measure
        │
   CPU / Metal / CUDA (NVRTC on-device) / Vulkan 1.3
```

- **Seismic** is a numerical DSL + compiler + runtime: kernels are authored once as programs; the compiler checks, fuses, resource-analyzes and lowers per backend. CUDA kernels compile on the user's machine via bundled NVRTC 12.9 -> exact-arch CUBIN with **no CUDA toolkit required**.
- **First-load autotuning (tuning.md):** 60-second allowance per model load; measures the *actual serving workload* (4k/64k prefill + decode with drafting), adversarial-cache-sized inputs, weighted `service_cost` objective, numerics as hard admission constraint, deadline-protected confirmation reserves, full retained-evidence traces. This is the most rigorous autotuning spec I have read in a local-inference project - genuinely SOTA-grade methodology.
- **Spec decode doctrine (drafting-and-verification.md):** three guarantees - (1) prefill pays nothing for drafting, (2) verification costs one decode step, (3) drafting costs only the drafter's own reads - with per-guarantee validation protocol. Every catalog model ships a matched drafter (MTP head, DFlash, DSpark, LiLiCorr).
- **Scheduling (scheduling.md):** single physical batch in flight, decode-first with prefill-debt repayment, eviction priced by releasable bytes x replay cost, 6-step capacity-negotiation ladder, typed failure with required/available byte facts. Comparable in rigor to sglang/vLLM schedulers, documented at spec level.

### 3.3 Scheduling head-to-head

| Property | Blazar | Magnitude |
|---|---|---|
| Batching owner | Engine children (llama.cpp continuous batching, sglang scheduler) + gateway admission/slots | Own scheduler (one physical batch, phase-fair, eviction-priced) |
| Admission | Bounded admission under saturation; VRAM/KV fit preview (`fit`); capability certificates | Negotiated preparation with typed capacity facts; 6-step reclaim->fail ladder |
| Slot policy | Adaptive slot reshape from live telemetry (2->8->16 under 60s sustained load; 1.72x throughput receipt) | No fixed request/batch count; aggregate token allowance per phase |
| Spec decode | Engine features (MTP/EAGLE/ngram/draft pairs) + closed-loop governor that parks speculation when acceptance collapses (default-on) | First-class per model; the three-guarantee doctrine; acceptance parity with reference is a defect bar, not a hope |
| Fairness | priorities, singleflight, queue-wait telemetry | Waiting-age + locality credit + accumulated service; preemption debt |

Both serious. Magnitude's is a single-engine spec with total control; Blazar's coordinates engines it does not own - harder problem, coarser levers.

---

## 4. Hardware & platform support

| Surface | Blazar (via engines) | Magnitude (native) |
|---|---|---|
| macOS | x + Apple Silicon (MLX lane is the differentiator; Metal via llama.cpp) | Apple Silicon only, **macOS 15 floor** (Metal); Intel Macs = CPU only |
| Windows | Yes (9-platform CI) | Yes (x64; Vulkan/CUDA/CPU) |
| Linux x64/arm64 | Yes | Yes (deb/rpm/pacman; glibc 2.35 floor) |
| NVIDIA | CUDA via engine builds (llama.cpp upstream coverage incl. older archs) | CUDA **sm_80+ only** (bf16 MMA + cp.async requirement); Turing/Volta -> Vulkan lane; NVRTC 12.9 on-device compile, driver R525+ floor |
| AMD | Via llama.cpp (Vulkan/HIP upstream builds) | Vulkan 1.3: GCN4/Polaris+ on Windows/RADV; logical 32-lane subgroup model with fixed-order reductions for 64-wide GCN |
| Intel | Via llama.cpp | Vulkan: Gen9/Skylake+ (ANV; Windows Arc drivers 101.4499+) |
| CPU | Engine CPU builds | x86-64-v2 floor with runtime tiers up to AVX-512 VNNI; AArch64 NEON (dotprod/i8mm planned) |
| Legacy/odd hardware | Whatever upstream engines ship | Explicit exclusions (phones, Kepler, Adreno X) - documented floors |

**Read:** Magnitude's compatibility doc is exemplary engineering (typed floors, probed tiers, per-vendor coverage tables, unqualified-hardware honesty). Blazar's effective coverage is broader on NVIDIA legacy (engine builds) and equal elsewhere, but inherits whatever upstream ships rather than owning floors.

---

## 5. Model support

| Surface | Blazar | Magnitude |
|---|---|---|
| Catalog size | Unbounded (any engine-supported model) | 30 catalog models, curated |
| Architectures | llama.cpp mainstream lane + curated capability lanes; mistral.rs GGUF+safetensors; SGLang AWQ/GPTQ/FP8; MLX dirs | 9 hand-kernelled families: llama, qwen, qwen35, qwen4exp, gemma4, lfm2, nemotron_h, muse_glimmer, dflash |
| Catalog examples (Magnitude) | - | qwen3.5-4b/9b, qwen3.6-35b-a3b, qwen3.8-27b/flash, glm-5.2, deepseek-v4-flash, gemma-4 (e2b->31b QAT), nemotron-3.x family (up to 550b-a55b), lfm2.5/lfm3 (+VL), minicpm5/minicpm-v-4.7, muse-glimmer-30b, bonsai-8b-q1, laguna-s-2.1 |
| Formats | GGUF, safetensors (AWQ/GPTQ/FP8), MLX directories | Own artifact pipeline over GGUF-derived weights (ggml crate) + HF source resolution |
| Import/reuse | `blazar import --scan`: reuse ollama store, HF cache, LM Studio (byte-verified hardlinks - 10 GiB model appears instantly) | None observed (catalog + HF resolve only) |
| Custom/local GGUF | Yes, any engine-supported arch | No - outside the 9 families |
| Quantized KV | Yes (`cache_type`, per-model) | Yes ("quantized KV halves memory per agent") |
| Vision | Via engine lanes | Catalog VL models (minicpm-v, lfm3-vl, nemotron super-vl) |
| Drafters | Spec pairs (auto qwen3-0.6b pairing), MTP (qwen3.5-9b-mtp receipt), ngram fallback | Per-model matched drafter (MTP, DFlash, DSpark, LiLiCorr) - shipped for *every* supported model |

**Read:** the model-support trade is the strategic fork. Magnitude: "we write optimized kernels for the most popular open-weight families, which is how we beat generalist engines." Blazar: every model the engines run, plus cross-engine routing for the same weights. A user with an out-of-family GGUF (or an existing ollama store) can only use Blazar.

---

## 6. Serving API & protocol surface

Extracted from `inference/engine/serving` + `inference/service/api` (Magnitude) vs. `docs/4.API_SPEC.md` + README (Blazar):

| Endpoint class | Blazar | Magnitude |
|---|---|---|
| OpenAI chat | `/v1/chat/completions` (+streaming, tools, structured output with receipts, metadata cards) | `/v1/chat/completions` |
| OpenAI completions | `/v1/completions` | - (not observed) |
| OpenAI Responses | `/v1/responses` (emulated uniformly across engines + native passthrough; `previous_response_id` chains, background) | `/v1/responses` |
| Ollama dialect | Full: `/api/chat`, `/api/generate`, `/api/tags`, `/api/ps`, `/api/pull`, `/api/embed*`, `/api/rerank`, `/api/create|copy|delete|push` | - (none) |
| Anthropic dialect | `/v1/messages`, `/v1/messages/batches`, count_tokens, cache accounting | `/anthropic/v1/messages`, `/anthropic/v1/messages/count_tokens` |
| Embeddings / rerank | Yes (late-chunking opt-in; rerankers from GGUF) | - |
| Audio STT/TTS | `/v1/audio/*` (whisper + Silero VAD streaming; piper TTS) + `blazar whisper` / `blazar tts` | - |
| Images / video | `/v1/images/*` (gens/edits/ESRGAN), `/v1/videos/*` (wan-2.1 receipt) | - |
| Realtime voice | `/v1/realtime` WebSocket cascade (STT->chat->TTS, E2E-sealed) | - |
| Files / batches / jobs | `/v1/files`, `/v1/batches`, `/v1/requests` (list/cancel), durable jobs ledger | - |
| Control plane | `/api/ps`, `/api/capacity`, `/api/explain`, `/api/failover`, `/api/mcp`, `/api/errors`, `/metrics`, OTLP, audit JSONL | `/api/v1/*`: catalog, installations, instances, hardware, downloads, model-assessments, drafters, HF sources, events, local-models, load-plans (own REST dialect, ~40 routes) |
| Grammar/structured output | Engine grammars (GBNF on llama-family; recipe-lane tool teaching on mlx) + structured-output receipts | grammar crate in engine (spec-level; endpoint exposure not surveyed) |
| Observability | `/metrics`, OTLP, `why`/`explain`, doctor, request-log, PII scrub | `/v1/logs`, `/v1/traces`, tuning traces, usage dashboards |

**Read:** Magnitude exposes the two agent-relevant dialects (OpenAI + Anthropic) well. Blazar's surface is an entire platform (roughly 3-4x the endpoint classes), including every multimodal lane.

---

## 7. Agents & tooling integration

| Surface | Blazar | Magnitude |
|---|---|---|
| One-click agent connect | `blazar connect`: Codex, Claude Code, Continue, Cline, Open WebUI, opencode, pi (oh my pi) - capability-driven model menus | `magnitude connections connect`: pi, opencode, hermes, openclaw, codex, claude-code, oh-my-pi, cline (+ `--install-skill`) |
| MCP | **Gateway-side MCP for any client**: register stdio/HTTP MCP servers once; tools injected as `mcp__<server>__<tool>`; gateway runs the whole tool loop (`[[mcp]]`, `/api/mcp`, per-request opt-in) | - |
| Tool calling | Engine tool-call adapters + recipe lane (prompt-taught tools + parse-lift for engines without native support), capability certificates gate admission | In-engine tool parsing (families own templates); no gateway MCP |
| SDKs | Python + TypeScript, zero-dependency, published lanes (PyPI pending trusted-publisher) | None observed (HTTP API only) |
| Desktop/console | `/ui` zero-dependency web console (TTFT/TPOT dashboard, census, jobs) - read-only by construction | Full Electron desktop app (catalog browsing, recommendations fastest->smartest, usage analytics, settings, tray) |
| Recommendations | `blazar plan` / `autopilot` / `scorecard` (receipt-backed) | Hardware assessment + ranked recommendations with intelligence index |

**Read:** Magnitude's desktop + assessment UX is the more polished consumer on-ramp; Blazar's MCP-loop and tool-certification story is the stronger infrastructure play (any client gains tools with zero client changes).

---

## 8. Performance - claims vs. claims (different references, stated methodology)

**Magnitude (committed claim, repo README/benchmarks page):** avg of 11 popular local models - Mac: 1.32x prefill / **2.19x decode** vs Ollama; NVIDIA: 1.34x / **2.17x decode**. Methodology lives on magnitude.dev/benchmarks (not in repo beyond the SVG).

**Blazar (committed campaigns, BENCHMARK.md with receipts):**

| Metric | Blazar | Reference | Margin |
|---|---:|---:|---|
| C1 decode overhead vs direct engine | 41.07 t/s | 41.64 direct | **-1.4%** |
| ITL p99 | 25.8 ms | 75.0 ms Ollama | **~3x** |
| TTFT p99 @ C=4 | 554 ms | 15.9 s Ollama | **~28x** |
| Throughput @ C=4 | 104.8 t/s | 33.6 t/s Ollama | **~3.1x** |
| Adaptive serving | 49->63 t/s | - | **+28%** |
| Sustained C8 + governor | 779.8 t/s | 452 stuck-at-np2 | **1.72x** |
| Idle wake | 2.4-3.5 s | 6.2 s Ollama | **~2x** |

**Cross-reading (inference, flagged as such):** the two claim sets are not directly comparable - Magnitude attacks the kernel layer where Ollama is weakest (single-stream decode of its curated families); Blazar attacks orchestration (tail latency, scheduling, tools) where it is strongest and measures against *the direct engine*, a much harder baseline than Ollama. If Magnitude's 2x+ decode claims replicate, a Magnitude-lane C1 decode would beat Blazar's llama.cpp lane at single-stream on their 9 families. Blazar's documented advantage at queueing/tail/multimodal surfaces has no Magnitude counterpart to race against.

---

## 9. Security, ops, lifecycle

| Surface | Blazar | Magnitude |
|---|---|---|
| Network posture | Loopback default, scoped API keys (add/rotate), TLS, explicit CORS, DNS-rebinding hardening, audit JSONL, PII scrubbing, opt-in OTLP | Local-only by default; server mode for LAN serving (`magnitude server setup`, systemd, boot-start) |
| Auth on child engines | argv-lane refusal (no secrets in /proc cmdline), keyfile/env lanes, manifest-probed auth | n/a (single process) |
| Update safety | Side-by-side engine versions, SHA-256, regression gates + auto-rollback | Auto-updating app+server+CLI as one unit; update refuses to interrupt a running serve |
| Crash durability | SQLite ledger: durable jobs (abandoned-marking on crash), restart-safe response chains, orphan reclaim (TERM->KILL sweep) | Request lifecycle spec: shutdown rejects/cancels/drains; owner resource-domain release |
| Secrets | env/keyfile; GH_TOKEN API-layer only (documented) | HF credential endpoint (`/api/v1/sources/hugging-face/credential`) |

---

## 10. Feature matrix (detailed)

Legend: PASS shipped + receipt/test-verified | ✚ shipped, verification lighter | ➖ not present | (blank) n/a

| Capability | Blazar | Magnitude |
|---|---|---|
| Custom inference kernels (own engine) | ➖ (by design) | PASS 9 families + Seismic |
| Per-device kernel autotuning | ➖ | PASS 60s first-load, evidence-based |
| Own numerical compiler (IR/fusion/lowering) | ➖ | PASS |
| Spec decode per model | ✚ engine-driven + governor (default-on, closed-loop park/restore) | PASS every model, 3-guarantee doctrine |
| Multi-engine routing | PASS 5 text lanes + 3 media lanes, per-model pins | ➖ |
| Ollama dialect | PASS full | ➖ |
| OpenAI dialect | PASS full incl. Responses emulation | PASS chat + Responses |
| Anthropic dialect | PASS + batches | ✚ messages + count_tokens |
| Embeddings / rerank | PASS | ➖ |
| Speech STT/TTS + realtime | PASS | ➖ |
| Image/video generation | PASS | ➖ |
| Vision chat | ✚ engine lanes | ✚ catalog VL models |
| MCP gateway tool loop | PASS | ➖ |
| Structured output receipts | PASS | ✚ grammar crate |
| Sessions save/restore + KV banking | PASS | ✚ prefix-cache sharing across sessions |
| Semantic cache | PASS opt-in | ➖ |
| Durable jobs/requests ledger | PASS | ➖ |
| Federation / remotes / failover | PASS | ➖ |
| best_of / cascade routing | PASS | ➖ |
| LoRA serving | PASS sglang hot-attach | ➖ (not observed) |
| Model import (ollama/HF/LM Studio reuse) | PASS hardlink reuse | ➖ |
| Hardware fit preview | PASS `fit` | PASS assessment + load-plans |
| Doctor/diagnostics | PASS doctor --fix, why, model-doctor certs | ✚ status, troubleshooting, typed device reasons |
| Web/desktop UI | ✚ `/ui` read-only console | PASS Electron app |
| SDKs | PASS Python + TS | ➖ |
| Cross-platform installers | ✚ 9-platform CI + crates.io | PASS signed dmg/exe/deb/rpm/pacman + auto-update |
| Community traction | ➖ | PASS 7.4k stars, Discord, company |

---

## 11. What Blazar should consider borrowing (ROI-ordered)

1. **Receipt-driven auto-tune pins (rung 2 - codebase reuse).** Magnitude's moat is per-device tuning. Blazar already banks bench receipts per (model, harness, precondition) - extend `blazar tune`/autopilot to *auto-pin* the measured-best knob set (`cache_type`, `spec_*`, `ubatch`, slot shape) from stored receipts on first-run of a new (model, device) pair. Not kernel tuning, but the same "pay once at install, win forever" shape at the orchestration layer, and it composes with existing `model_overrides` + `scorecard` machinery.
2. **Spec-pair coverage per catalog model.** Magnitude ships a matched drafter for *every* model. Blazar's `spec_pairs` covers qwen3-family auto-pairing + MTP receipts; a systematic per-family drafter table (with acceptance-rate receipts per pair) would close the most visible decode-speed gap without owning kernels.
3. **First-load "assessment" UX.** Magnitude's hardware assessment -> ranked model recommendations (fastest->smartest) is the best consumer on-ramp observed here. Blazar has all the ingredients (device census, bench receipts, capability certs) to render an equivalent `blazar plan --install` flow.

**Not recommended:** writing an own-kernel engine lane to chase Magnitude at their layer - that is a multi-100k-LoC bet on a different thesis; Blazar's leverage (protocol breadth, multi-engine, model ops) is where the 3-28x receipts already live.

## 12. Threat assessment (competitive view)

- Magnitude's GTM (desktop app + one-click agent connect + 2x-vs-Ollama headline) targets the same "local AI for agents" users Blazar's gateway does. Their catalog constraint (9 families, no import, no Ollama dialect) is the wedge Blazar currently owns uncontested.
- If Magnitude open-sources deeper agent tooling from their `packages/` stack (harness, skills, acn/icn protocols present in-monorepo), they could grow past inference into Blazar's orchestration layer - worth watching their release notes.
- Blazar's uncontested differentiators to amplify: MCP-for-any-client, multimodal lanes, federation, engine pluralism, import/reuse, verification receipts.

---

## 13. Evidence index

- Magnitude repo @ e596431: README.md, inference/docs/{overview,tuning,compatibility,scheduling}.md, inference/docs/kernels/drafting-and-verification.md, inference/Cargo.toml workspace, inference/catalog/models.json (30 entries), inference/engine/serving/src route extraction, docs/{reference,models}.mdx, packages/ listing, tokei LoC, GitHub API (stars/forks/contributors/releases).
- Blazar repo @ working tree (v0.28.0-era, uncommitted waves per MEMORY.md): README.md (features §, engines table, benchmark snapshot), crates/ LoC, MEMORY.md receipt index, BENCHMARK.md reference.
- All performance numbers are each project's own committed claims; no cross-project benchmark was run for this comparison (would require magnitude.dev model artifacts on the RTX 4070 rig; candidate follow-up: `bench_matrix.py` parity campaign against `magnitude serve` on the shared model intersection).
