# Atomic-Chat vs Blazar — Competitive Research Report

**Date:** 2026-11-06 (rev 2 — Blazar side re-verified against the unreleased working tree, not the GH release)
**Blazar basis:** tree `e48142e` = **v0.29.0 + 46 unreleased commits** (+30,607 lines, 298 files; CHANGELOG `[Unreleased]`), clean tree. The 0.29.0 release tag alone understates the surface below.
**Method:** GitHub API metadata + README/AGENTS.md/PLAN of `AtomicBot-ai/Atomic-Chat`, `AtomicBot-ai/atomic-chat-core`, `AtomicBot-ai/atomic-llama-cpp-turboquant` (fetched live 2026-11-05/06); Blazar side verified against the local tree (`crates/*`, CHANGELOG Unreleased, knob_registry/profile/optimize source, docs/4.API_SPEC.md CLI tables, MEMORY receipts). No claims from training memory.

---

## 1. Bottom line

Atomic-Chat is **not a Blazar competitor in category** — it is an **end-user chat product** (Tauri desktop + iOS/Android apps) with an extracted inference core (`atomic-chat-core`, TypeScript). Its real edge over Blazar is concentrated in two places:

1. **Engine-level kernel work** — a maintained llama.cpp fork (`atomic-llama-cpp-turboquant`, 432 stars) shipping TurboQuant WHT-rotated KV-cache compression (~4.3x), `TQ4_1S`/`TQ3_1S` weight quants, and MTP/NextN speculative decoding that loads the draft head **into the target context** (no second context/tokenizer/KV, async depth-2 scheduler) — validated with mmproj multimodal on the same slot. The fork is Apache-2.0, and Blazar's unreleased `engine build --fork` + `engine offers` lane machinery is the direct absorption path for it (see §7.1).
2. **Accelerator lane breadth** — 12 engines incl. Apple Foundation Models, Foundry Local, LiteRT-LM, GenieX (Qualcomm NPU), PrismML (1/2-bit Bonsai), plus TensorRT-LLM/vLLM/SGLang in self-managed containers (Docker/WSL, consent-gated privileged setup).

Blazar leads on **everything server-side**: 3 API dialects + realtime/batch/rerank/jobs/videos surfaces, multi-machine federation + auto-membership RPC fleet + LAN discovery, cross-store model adoption (ollama/HF-cache/LM-Studio hardlinks), admission/queue/capacity discipline, spec-decode orchestration (`drafts`/`tune --spec`/auto-MTP gating), evidence-gated KV-quant tuning (`kv_quality` receipts + `--cache-type-k/v` global and per-model), evidence-gated optimize recommendations, `engine build --fork` pinned source builds + curated capability-lane registry, typed error catalog, committed gateway-overhead benchmark receipts (~1.4% single-stream), Rust single binary, and a true OSS license (MIT OR Apache-2.0). Atomic-Chat's main repo is **license:NOASSERTION** ("Other") and `atomic-chat-core` has **no license at all** — source-available, not open source; only their llama.cpp fork inherits Apache-2.0.

---

## 2. What Atomic-Chat is (verified facts)

| Dimension | Atomic-Chat (app) | atomic-chat-core | atomic-llama-cpp-turboquant |
|---|---|---|---|
| Role | Tauri 2 desktop + iOS/Android chat app | TS inference core (engine manager + router + OpenAI server `:1337`) | llama.cpp fork (KV/weight compression + spec-decode) |
| Created | 2026-03-31 (~7 mo old) | 2026-09-17 (extraction from app, phased per PLAN.md) | fork, default branch `dev` |
| Stars / forks | **1,943 / 229**, 140+ contributors | 3 / 3 | **432 / 58** |
| Language | TypeScript 15.1MB, Rust 4.9MB (shell), Swift 87KB | TypeScript (Node **and** Bun runtime-agnostic gate) | C/C++ (llama.cpp) |
| License | **"Other" / NOASSERTION** | **none (all-rights-reserved)** | Apache-2.0 (inherited) |
| Cadence | Multiple releases/week (v2.3.5 on 2026-11-04, 18 assets: dmg/exe/AppImage + apt/dnf/AUR repos + App Store + Google Play) | pinned by app releases (v0.21.x) | pushed 2026-10-28 |
| Distribution | Signed desktop installers, Linux distro repos, mobile stores, HF org (`AtomicChat`) publishing curated GGUF collections | bun-compiled single binary, also importable TS library | release binaries + HF collections |

Product surface (app side): chats, assistants w/ per-assistant system prompts, projects w/ conversation tree, artifacts live-preview panel, MCP client connections, one-click external agent launch (Claude Code, Codex, OpenCode, Cline, Goose, …), Integrations tab, auto context-window expansion, reasoning-context tracking.

Cloud providers in-app: OpenAI (incl. ChatGPT OAuth PKCE, callback :1455), Anthropic, Mistral, Groq, MiniMax, Qwen, Moonshot — mixed freely with local models.

---

## 3. Their engine stack (12 lanes)

| Engine | What it runs | Where |
|---|---|---|
| upstream llama.cpp | GGUF, default on Win/Linux | macOS/Win/Linux — CPU/Metal/CUDA/Vulkan |
| **atomic-llama-cpp-turboquant** | GGUF + TurboQuant KV (`turbo3`/`turbo4`) + MTP | same matrix |
| PrismML llama.cpp | Bonsai 1-/2-bit tensors | + ROCm |
| MLX-VLM | MLX incl. VLM, DFlash/EAGLE-3/MTP | macOS Apple Silicon |
| Apple Foundation Models | on-device FM, nothing to download | macOS |
| Foundry Local (Microsoft) | managed host-mode | Win/macOS |
| LiteRT-LM (Google) | PyPI-wheel host-mode w/ own OpenAI server | mobile-class/NPU |
| GenieX (Qualcomm) | AI Hub bundles, NPU | Snapdragon |
| TensorRT-LLM, vLLM, SGLang | container-managed (Docker + NVIDIA toolkit; WSL distro on Windows), memory sized by core | Linux+NVIDIA / Win x64 |
| stable-diffusion.cpp (own build) | images (FLUX, Qwen-Image, Z-Image, Krea) + video (Wan 2.2, LTX-2) | CPU/Metal/CUDA/Vulkan/ROCm |

Speculative decoding: MTP (30–70% claim, "up to 3x on Gemma 4"), DFlash block-diffusion (up to 6x claim, Qwen 3.6 / Gemma 4 / Kimi K2.5, Apple-only, mutually exclusive with MTP), EAGLE-3 (Gemma 4, MLX), MTP on MLX (Qwen 3.5/3.6, DeepSeek V4). Flash-Attention on/off/auto.

### TurboQuant fork deep-dive (the strategically important piece)
- **KV cache:** WHT-rotated low-bit quantization, `-ctk turbo3 -ctv turbo3` → **~4.3x smaller KV**, backend-native kernels: Metal "TurboFlash", CUDA, Vulkan, HIP.
- **Weights:** `TQ4_1S` / `TQ3_1S` quant types (~4x lower claimed).
- **MTP in-context:** assistant head loaded **into the target context** — no second `llama_context`, no second tokenizer, no separate KV; runs on a dedicated scheduler so draft compute overlaps target verification; async depth-2 pipeline; `kv_only_nextn` knob.
- **mmproj + spec on one slot:** multimodal turns fall back to plain target decoding while spec stays armed for text turns (validated Qwen 3.6-35B UDT + NextN + turbo3; Gemma 4-26B + MTP + turbo3).
- **UDT (UD-Turbo) collections:** target+NextN combined `_MTP.gguf` single-file drops; quant masks pin `blk.*.nextn.*`/`mtp.*` to Q8_0 (draft acceptance preserved, ~10MiB cost) and lift `attn_q/attn_k` to Q6_K (pairs with turbo3 KV); Unsloth imatrix + published runbooks; Apache-2.0.
- **Their published benches** (fork README): gemma-E4B f16-mtp 94.3 t/s (turbo3-mtp 67.8 w/ 82.6% quality-retention column), gemma-26B f16-mtp 110.8 vs base 83.6, gemma-31B f16-mtp 21.2 vs 19.4 base.

### atomic-chat-core architecture notes
- Persistent single **owner daemon**: control listener permanent loopback; public inference listener starts/stops independently; CLI/app exit **detaches without unloading**; snapshot + SSE with instance identity rebuilds state on reconnect.
- Serious engineering culture: contract/e2e/app-e2e/runtime-compat/live test tiers, coverage-floor + test-quality + runtime-agnostic lint gates, `engine-surface-diff` script, ~50 dated ADRs in `docs/decisions/`, engine-candidate CI workflow, curated-inventory digest verification.
- Managed runtimes: consent-gated privileged host steps via `pkexec`/`sudo` helper (`host-step exec`), descriptor-qualified distros, never self-installs; opt-in VM-changing live tests behind `ATOMIC_LIVE_MANAGED=1`.
- 2,989 files: `runtime/` 373, `server/` 136, `backend/` 78 (backend packs: catalog/install/optimal/advisor/turboquant release index, hardware-tier selection), `diffusion/` 69, `integrations/` 66, `hardware/` 33 (nvidia-smi, DRM sysfs, PowerShell CIM, Vulkan ICDs), `speculative/` 15, `remote-access/` 23, `telemetry/` 32 (Sentry w/ consent + scrub + dedup caps).
- Phase-1 note: "the core does not download models" — the app side owns model acquisition; core owns engine builds (sdcpp + MLX builds owned by core).

---

## 4. Head-to-head matrix

Blazar column reflects the **unreleased tree** (`e48142e`). Legend: ✅ shipped+evidenced · 🟡 partial/different shape · ❌ absent

| Capability | Atomic-Chat (core where relevant) | Blazar (unreleased) |
|---|---|---|
| **API dialects** | OpenAI-compatible only (`:1337/v1`) + `/atomic/v1` control + SSE | **OpenAI + Ollama + Anthropic** (`/v1/messages`+batches), port-level Ollama replacement possible |
| **Surface breadth** | chat, embeddings, audio, images/video via app tabs | chat/completions/responses/embeddings/**rerank**/**batches/files**/audio/*,images/*,**videos/*, `/api/*` verbs, **realtime voice WS**, Python SDK |
| **MCP** | app **connects to** MCP servers (client) | gateway **runs the MCP tool loop** for engines without MCP (server-side), per-request opt-in, `/api/mcp` |
| **Engines (text)** | 9 lanes incl. TRT-LLM/vLLM/SGLang(containers), Foundry, LiteRT, GenieX, Apple FM, PrismML | llama.cpp, mistral.rs, SGLang (native lane), MLX (Apple+CUDA) + curated capability lanes + **`engine build --fork owner/repo@<sha>`** (pinned source build of any fork, archs mined from source, provenance in `engine list`) + **`engine offers` / `install --lane`** curated lane registry (BACKENDS column, validators, auto-retire when upstream covers the archs) |
| **Engines (media)** | sdcpp (own build) | sdcpp (35 curated diffusion families) + **in-process candle CLIP/SigLIP** + whisper.cpp + piper + engine-native TTS/ASR lanes |
| **Speculative decoding** | **MTP/DFlash/EAGLE-3/NextN + own kernels + UDT collections** (fork-only in-context MTP, spec+mmproj co-residency proven) | spec orchestration shipped: auto engagement gated by `engine_has_mtp && mtp_arch_proven(arch)` (embedded MTP) w/ ngram self-draft fallback, `blazar drafts <model>` (EAGLE3/MTP + same-family draft candidates), `tune --spec` live probes, GGUF MTP-tensor detection + real-MTP tests; gaps: spec+mmproj co-residency receipts, curated draft-GGUF ecosystem pairing |
| **KV-cache quant** | TurboQuant turbo3/turbo4 (~4.3x, own kernels), no published quality gate | **machinery ahead of theirs**: `--cache-type-k/v` engine knobs + global `cache_type{,_k,_v}` + per-model overrides + **evidence gate** (`kv_quality.rs`: llama-ppl A/B receipts per model+GGUF, ctx-floor eligibility, `/api/explain` kv_advisory reports `measured: null` rather than borrowing numbers); turbo-class types light up the moment an engine build offers them (absorbable via fork lane) |
| **Weight quants** | TQ4_1S/TQ3_1S (own) + Q4_K_S/Q5_K_M/Q8_0/F16 | full GGUF quant matrix + quant fallback + curated catalog (45 entries) + `blazar quantize` |
| **Engine-build advisor** | backend-pack advisor: hardware tiers, catalog/recommendation/update endpoints, TurboQuant release index | `optimize.rs` evidence-gated recommendation engine (Severity/`OptimizeRec` cmd+why+datum, benchmark budget), doctor Advisor checks, `engine delta` post-update absorption checklist; delta: no "best build for THIS GPU" tier-ranking surface across installed builds |
| **Model acquisition** | app-side downloads; GGUF metadata + KV estimate; UDT/assistant collections on HF | `blazar pull/search/import --scan`, **cross-store adoption** (ollama/HF cache/LM Studio, sha256 byte-verified hardlinks), ollama-registry reuse, rename-steering, revision pins, `--max-download-mbps`; delta: no auto-pairing of published draft/assistant GGUF collections |
| **Multi-machine** | ❌ single machine (loopback default, LAN opt-in) | **federation (capacity-aware placement) + RPC fleet** (auto join/heartbeat/leave, TTL eviction, crash-restart rejoin) + **UDP LAN discovery + guided onboarding** (`seek_gateways`) |
| **Cloud models** | 7 providers + ChatGPT OAuth in-app | remotes/failover chains/fabric as infrastructure (any OpenAI-compatible + Anthropic endpoints), keys+scrub |
| **Orchestration** | model→provider router | **best_of fan-out, n-way, cascade, systemone decision models, failover anti-flap, jobs durability, admission/queue/capacity, preflight, host_guard, sentinel** |
| **Sessions/cache** | owner-detach keeps model resident; SSE state rebuild | session save/restore + **KV session banking on graceful stop** (warm restart), prefix-cache visibility, opt-in semantic cache, late-chunking, warm-on-pull |
| **Engine lifecycle** | engine builds owned, managed-container provisioning w/ consent UX | supervisor: spawn mem-guard, crash-restart backoff, **engine rollback on bad binary (LC4)**, port-race recovery, `engine update` sha256+regression-gated, idle sleep/wake (~2x faster wake than Ollama) |
| **Observability** | Sentry telemetry (consent+scrub), events, SSE | OTLP opt-in, audit, usage accounting, histograms/quantiles, explain (incl. kv_advisory), model_doctor, metadata cards, web console, 26-code typed error catalog |
| **Perf evidence** | engine-level tables (MTP t/s, KV ratios) | **committed gateway campaigns**: 1.4% single-stream overhead, TTFT p50 120ms parity / 3x lower tail than Ollama, C1 423.7 t/s (1.14x direct), wake 2x faster, + unreleased media-lane envelopes (embeddings 80.7→126.4 req/s C1→C8, piper 12x realtime, seed-determinism receipts), full reproducibility receipts |
| **Hardware fit** | backend-pack hardware tiers + advisor; NPU lanes | VRAM/KV fit modeling, resource-aware routing, degrade ladders, mem guards, `coreside` co-residency planner |
| **Privacy stance** | local-first claims + Sentry reporting (consented) | strictly local; telemetry opt-in; nothing leaves machine without `[[remotes]]` |
| **License** | app NOASSERTION; core none; fork Apache-2.0 | **MIT OR Apache-2.0** everywhere |
| **Infra deps** | Docker+NVIDIA toolkit for 3 lanes; WSL on Win | **none** — single Rust binary, engines are external binaries from upstream releases or pinned fork source builds |
| **Test discipline** | 5-tier tests, gates, ADRs, ~50 decisions | 2,441+ tests (nextest, workspace 2x), live-validation receipt culture, pin-first doctrine |
| **End-user UX** | **full chat app, artifacts, assistants, projects, mobile apps, app stores** | headless server + CLI + web console (by design) |

---

## 5. Where Atomic is ahead — the honest gap list (against the unreleased tree)

1. **Spec-decode kernel depth + co-residency proof (biggest remaining gap).** Their fork loads MTP heads into the target context with an overlapping scheduler, and has **validated spec + mmproj multimodal on one slot** — no Blazar receipt exists for spec+vision co-residency on any build (upstream builds have historically refused that combination). They also curate an ecosystem of tuned draft artifacts (UDT combined `_MTP.gguf`, assistant-head GGUFs, quant masks); Blazar's `drafts` recommends candidates but does not pair published collections.
2. **Turbo-class compression kernels.** turbo3/turbo4 (~4.3x KV) and TQ4_1S/TQ3_1S are real fork-only kernels. Blazar's absorption machinery already exists (`engine build --fork` pinned source builds, `engine offers` curated lanes, `engine delta` flag/spec-type diffing, `--cache-type-k/v` + evidence gate) — what's missing is a **curated lane entry + measured receipts** pointing at a turbo-class fork. This is a curation/validation task, not an architecture gap.
3. **NPU/mobile accelerator lanes.** LiteRT-LM, GenieX, Apple Foundation Models, Foundry Local, PrismML 1/2-bit — five lanes Blazar has no equivalent for. (Counterpoint: Blazar's no-container rule makes the TRT/vLLM/SGLang container lanes structurally uninteresting; Blazar's native SGLang lane is the cleaner answer there.)
4. **End-user distribution & community.** 1.9k stars in 7 months, app stores, distro packages, Discord, HF quant collections, 140+ contributors. Blazar's distribution is binaries + crates.io; the discovery/branding gap is real even for a server product.
5. **Hardware-tier build ranking UX.** Their backend advisor ranks engine builds per hardware tier with a release index. Blazar has the ingredients (probe facts, engine manifests, `engine delta`, optimize recommendations) but no single "best build for this GPU" surface.
6. **Detach-without-unload owner model.** Their daemon survives CLI/app exit with the model resident; Blazar's daemon is itself the owner (KV session banking on graceful stop instead) — Blazar's approach is arguably stronger for a server, but their pattern is worth noting for CLI ergonomics.

## 6. Where Blazar is decisively ahead

1. **Serving surface**: 3 dialect families, realtime WS voice, batch API + files, rerank, videos, jobs, Ollama verb parity — Atomic is OpenAI-compat only.
2. **Multi-machine**: federation + auto-membership RPC fleet + LAN discovery/onboarding; Atomic is single-box.
3. **Model-store interop**: zero-network adoption of existing ollama/HF/LM-Studio stores; rename-steering; catalog; quant fallback; speed caps. Atomic explicitly doesn't download in the core.
4. **Serving discipline**: admission/queue/capacity/preflight/sentinel/failover anti-flap + typed 26-code error catalog — nothing comparable on their side.
5. **Evidence culture at the gateway layer**: committed benchmark campaigns with reproducibility receipts incl. the unreleased media-lane envelopes; their published numbers are engine-level only.
6. **KV-quant safety machinery**: `--cache-type-k/v` + per-model overrides + perplexity-receipt eligibility with per-model ctx floors + honest `measured: null` — they ship faster kernels but no quality-gate machinery at all.
7. **Engine absorption machinery**: `engine build --fork` (immutable-commit source builds, arch mining, provenance) + `engine offers` curated registry + regression-gated `engine update` + rollback + `engine delta` — Blazar can adopt third-party forks as first-class lanes; Atomic hardcodes theirs.
8. **License/privacy posture**: true OSS, no telemetry by default, no container requirement, single Rust binary.
9. **MCP server-side**: gateway runs tool loops for engines — theirs is client-side only.

---

## 7. Recommended actions for Blazar (high-ROI, non-overlapping)

1. **TurboQuant absorption spike (P0 — flips their biggest moat into a Blazar lane):** `blazar engine build --fork AtomicBot-ai/atomic-llama-cpp-turboquant@<sha> --backend cuda` (Apache-2.0 permits) → `blazar engine delta` to bank the new flags/spec types → kv-quality campaign on `--cache-type-k/v turbo3` for 2-3 resident models (receipts into `registry/kv-quant-quality.json`, ppl + ctx-floor) → throughput A/B vs upstream lane → publish as a curated `engine offers` lane. Their fork becomes a validation data point for Blazar's lane machinery instead of a differentiator.
2. **Spec+mmproj co-residency receipts (P1):** on the fork lane from (1), live probes of a vision+MTP row (e.g. qwen3vl-class with embedded nextn): spec armed on text turns, mmproj vision turns served by the same child; pin the profile interplay (which argv combinations each engine build accepts — gate on `engine delta`-mined spec types). Zero code if the profile machinery already composes; receipts + pins are the deliverable.
3. **Draft-ecosystem pairing (P2):** teach `blazar drafts`/pull routing to recognize UDT-style combined `_MTP.gguf` files (target+NextN in one GGUF — `-m`/`-md` same-path pattern) and published assistant-head collections (AtomicChat HF org, unsloth MTP GGUFs) so `drafts` can recommend *installed-store-aware* pairings, not just same-family candidates.
4. **Hardware-tier build ranking (P2):** unify probe facts + engine manifests + `engine delta` into one advisor surface ("best installed build for this GPU; what the newest build adds") — the last UX delta vs their backend advisor.
5. **Watch items:** upstream llama.cpp MTP/NextN trajectory (Blazar's `mtp_arch_proven` allowlist grows from receipts, not vendor claims); their UDT quant-mask conventions (Q8_0 nextn pins, Q6_K attn) as a possible community standard worth surfacing in `blazar quantize` masks.

## 8. Sources

- `api.github.com/repos/AtomicBot-ai/{Atomic-Chat,atomic-chat-core,atomic-llama-cpp-turboquant}` (metadata, languages, releases, commits) — fetched 2026-11-05/06
- Atomic-Chat README (features, engines tables, download matrix, acknowledgements, contributors)
- atomic-chat-core README + AGENTS.md (repo map, commands, managed-runtimes contract, owner/daemon model)
- atomic-llama-cpp-turboquant README (TurboQuant, MTP/NextN, UDT collections, benchmark tables, mmproj+spec)
- Blazar local tree **@ `e48142e` (v0.29.0+46, unreleased)**: `README.md`, `CHANGELOG.md` `[Unreleased]`, `Cargo.toml`, `crates/*` inventory, `blazar-core/src/{kv_quality.rs,knob_registry.rs,optimize.rs,profile.rs,config.rs,gguf.rs}`, `registry/kv-quant-quality.json`, `docs/4.API_SPEC.md` CLI tables, `crates/blazar-runtime/src/discovery.rs`, MEMORY.md session receipts
