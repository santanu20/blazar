# Competitor Pain-Point Audit — ollama / SGLang / llama.cpp (2026-10-02)

Method: live web research (ddgs metasearch across Reddit, HN, Stack Overflow,
GitHub, vendor blogs — raw captures in session) + GitHub Search API
top-reacted OPEN issues for each upstream repo + codebase verification of
every mitigation claimed below (file:line evidence). No claim in this file is
from training memory alone; each row was grepped in this repo on 2026-10-02.

## 1. Where the complaints come from (research receipts)

- ollama top open issues by reactions: #3368 reranking (381), #15051
  go-engine quant (294), #3185 license notices in artifacts (292), #8618
  Janus-Pro vision (273), #7865 MCP (220), #5245 multi-file GGUF import (217),
  #228 home-dir clutter (190), #5186 Ryzen NPU (168), #3144 /metrics (140),
  #1590 Intel Arc (139), #12532 cloud stats (126), #786 image gen (125),
  #162 disable autostart (123), #2006 rate-limit download speed (104),
  #1005 context-window management (87).
- Community threads: "ollama silently truncates at num_ctx 4096/2048" (the
  single most-written-about ollama failure mode across Reddit/blogs/LinkedIn);
  keep_alive 5-minute surprise unloads and cold-start tax; OLLAMA_NUM_PARALLEL
  not effective (esp. Mac MLX); 503 queue-overflow class; RAM/VRAM growth over
  long sessions (0.6.x-era leaks, Mac KV-cache leak fixed ~0.30); tool-call +
  JSON-schema combos returning empty tool arrays; /api/embeddings deprecation
  confusion; pull failures mid-CDN-throttle with restart loops; blob-storage
  opacity; alias-vs-repo naming; CVE-2025-51471 path traversal class.
- SGLang: startup-vs-serving OOM confusion around mem-fraction-static; CUDA
  wheel/torch version matrix pain on install (CUDA-12 docs pulling CUDA-13
  deps, torch clobbering); multi-second-to-minute cold boots (python/torch/
  CUDA-graph composition); hicache host RAM floor; DeepEP/JIT link failures;
  engine crash-recovery (their Aug-2026 sub-second restart postmortem).
- llama-server: "not usable by casual users" (HN); compute-buffer allocation
  failures on tight cards; quantized-V cache hard-requiring flash attention
  (silent config death); broken/missing Jinja chat templates (Gemma/Qwen
  corrections circulating as gists); --cache-ram default holding ~8 GiB host
  RAM; OpenAI /v1/responses gaps (#14702); grammar sampling speed (#4218);
  disk-based context offload still an open FR (#20697).

## 2. Verified mitigation map (ollama complaints)

| User-facing complaint | Blazar mitigation (evidence) | Status |
|---|---|---|
| Silent 4096/2048 context truncation | Pre-admission prompt-token estimate vs effective ctx; loud 400 naming both numbers + how to raise (`preflight.rs:57,166`); `options.num_ctx` restarts instance at requested size (`ollama.rs:1097`, complaint #13); sentinel `CtxTruncated` diagnostics; OpenAI lane refuses what the engine would truncate (`openai.rs:462` FIX8); `explain` surfaces admission ctx discipline | FIXED |
| keep_alive surprise unloads / VRAM squatting | keep_alive honored exactly: >0 pins N s, 0 evicts after request, -1 forever (`ollama.rs:6,1234`, complaint #12); idle ladder sleep/evict knobs; `/api/ps` + `/api/capacity` observability | FIXED |
| Concurrency: serial requests, 503 overflow, NUM_PARALLEL ineffective | Adaptive slots + measured-pressure reshape (4→8 slots, 49.1→63.0 t/s receipt); bounded admission with teaching 503 (2min→503 receipt); queue priorities + SLO deadline counters in /metrics; deterministic-isolate lane | FIXED |
| RAM/VRAM leaks over long sessions | Crash/teardown pins (SIGKILL, circuit, respawn) in `tests/supervisor.rs`; leak-accounting tests (aside/ghost/marks); systemd MemoryHigh ceiling (`install.sh` BLAZAR_UNIT_MEMORY_HIGH) | FIXED |
| Slower than raw llama.cpp | Flagship campaign: 67.5 vs ollama 33.4 t/s concurrent (2x), receipts in `BENCHMARK.md` + `bench-artifacts/`; per-request trace ids + argv forensics | FIXED |
| No reranking (top issue, 381 upvotes) | `/v1/rerank`, `/v1/reranking` (Jina alias), `/api/rerank` with doc-shape normalization (`lib.rs:270,347,382`, `ollama.rs:2640`) | FIXED |
| No image generation | sdcpp lane (qwen-image 2.1, FLUX class), video lane (wan); images gate | FIXED |
| No /metrics | `/metrics` merged children Prometheus + `blazar_*` gauges + histogram buckets (`lib.rs:406`, `ollama.rs:3369`, `histogram.rs`) | FIXED |
| Multi-file GGUF import | gguf-split `-00001-of-0000N` shard sets planned, pulled, stored, deleted together (`hf.rs:321-341`, `store.rs` shards column) | FIXED |
| Home-directory clutter | XDG layout `~/.local/share/blazar/...`, explicit env overrides, loud fallback (`dirs.rs:16-38`) | FIXED |
| MCP support | `mcp_servers_config`/`mcp_servers_json` → `--mcp-servers-config/-json` engine argv with existence checks (`profile.rs:1298-1308`, `config.rs:152`) | FIXED (engine-side) |
| Forced autostart | systemd unit is explicit install output; uninstall removes; no stealth background start | FIXED |
| Pull failures / restart loops / throttle aborts | 6-attempt chunk retry, Retry-After honored, exponential backoff capped 30s (`hf_parallel.rs`, commit b9e3797); resumable `-partial`; progress live; net-probe bounded retry for engine checks | FIXED |
| Tool calls empty with JSON schema / null outputs | `ollama_compat` prompt recipe with GBNF `<tool_call>` grammar (measurably improves error-path tool calling); sentinel structured-output lint fails fast on malformed schema/oversized grammar before admission; cascade best-of-n JSON repair lane | FIXED |
| Embeddings deprecated/confusing | Both `/api/embed` and `/api/embeddings`; generative checkpoints gain an embeddings parity arm (`--embeddings --pooling last`, Wave K O3 receipt: chat + 896-dim embeds co-served); latechunk lane; teaching 501s name the knob when posture lacks it | FIXED |
| Blob storage opacity | Plain files `models/*.gguf` + `.d/` dirs, any tool can use them (`hf.rs:10`, complaint #4) | FIXED |
| Alias naming hides real repo | Registry name = actual HF repo name (`hf.rs:60`, complaint #6) | FIXED |
| Path-traversal class CVE | hf.rs security invariants (complaint #7, CVE-2025-51471 class) | FIXED |
| Requests silently wait on model load | Loading transparency in responses (`proxy.rs:886`, complaint #10) | FIXED |
| No pre-download compatibility check | Compatibility preview before pull (`hf.rs:1423`, complaint #14) | FIXED |
| Poor model discovery | Hub search across every weight format (`hf.rs:1208`, complaint #15) | FIXED |
| Dropped logprobs | OpenAI lane passthrough (`openai.rs`, complaint #1) | FIXED |
| Thinking models emit raw `<think>` / empty content | Template-aware think gates both engines: GGUF metadata probe + HF dir `tokenizer_config.json` probe; default think-off bridge (`inject_default_think_off`, Wave K O2 live receipt) | FIXED |
| No cancel/interrupt | `/v1/requests/{id}/cancel` + `/interrupt`, unified `/v1/jobs` plane, batch + audio cancels | FIXED |
| OpenAI Responses API gaps | Own `/v1/responses` implementation + durable SQLite write-through for `previous_response_id` chaining | FIXED |
| Windows install / GPU pain | install.ps1, scoop, winget channels; Windows driver detection advise-only; doctor warns on missing GPU drivers | FIXED |
| Slow first token after idle | warm_on_pull pre-spawn (cold 1740ms→102ms receipt); KV-bank prefix restore after eviction; mistral.rs resident alternative (139.7ms) | FIXED |

## 3. Verified mitigation map (SGLang complaints)

| Complaint | Blazar mitigation | Status |
|---|---|---|
| OOM: mem-fraction-static tuning cliff | Activation-reserve governance: ceiling = (vram−2000MiB)/vram clamped [0.20,0.90], live clamp receipt 0.900→0.743, PyTorch capture assert class eliminated (Wave I1 + pins) | FIXED |
| hicache host RAM floor crash (upstream reserves 10 GiB) | Host viability gate: budget = ram−10240−4096; teaching 400 with the math instead of SIGKILL 502 (Wave I2) | FIXED |
| Install matrix hell (CUDA gens, torch clobber, JIT link) | Generation-agnostic bundled-toolkit discovery + dev-layout bridging (symlinks) + CUDA_HOME pinning; install-time boot-smoke gate imports `sglang.srt.entrypoints.engine`; venv `nvidia/*/lib` LD_LIBRARY_PATH fix (Culture-13 + 6015adf receipts) | FIXED |
| Cold boot 30s+ | Honest composition receipt (28.3s decomposed: 8.0s torch+CUDA ctx, 7.4s graph capture, blazar adds ~0.5s); mitigations: warm_on_pull, radix/argv receipts, mistral.rs lane for latency-critical residency | MITIGATED (upstream-python floor documented) |
| Crash loops surface as opaque 502s | GGUF-draft refusal guard with teaching warning; supervisor respawn + circuit breaker; engine recovery covered by crash/teardown pins | FIXED |
| Dedicated embedding lane | `sglang.is_embedding` knob → `--is-embedding` dedicated child (commit 9bf2161) | FIXED |

## 4. Verified mitigation map (llama.cpp / llama-server complaints)

| Complaint | Blazar mitigation | Status |
|---|---|---|
| Not usable by casual users | Full UX: pull/serve/doctor/explain/warm/quantize; one config file with validation; first-party driver install flow (announce-then-act); `engine update` auto-picks newest compatible build | FIXED |
| Compute-buffer alloc failures on tight cards | Posture-aware Tier A/B/C fit math (weights+KV+activation reserve), honest refusal receipts instead of child death; unified-KV relocation ladder + `no_kv_offload` knob | FIXED |
| Quantized-V cache dies without flash attention | fa_off posture drops V-side quant to f16, keeps K (Wave H1 pin) | FIXED |
| Broken/missing chat templates | GGUF-metadata template probe + HF-dir template probe; think-gates; `ollama_compat` raw lane as template escape hatch; vision/mmproj posture handling | FIXED |
| --cache-ram silently holds ~8 GiB host RAM | cache_ram_mb plumbing governed by fit ledger against free RAM | FIXED |
| OpenAI API gaps (responses, batches) | Full `/v1` surface incl. responses + batches + files-shaped lanes; Anthropic dialect translation | FIXED |
| Grammar handling | GBNF passthrough (256 KiB cap) + JSON-schema translation + grammar-size lint pre-admission | FIXED |
| Multi-GPU split tuning | Demand-shard planner (smallest-card NCCL bottleneck), tensor-split probe, co-residency planner receipts | FIXED |

## 5. Open gaps found (candidate work, none critical)

1. **User-configurable pull download speed cap** (ollama #2006, 104
   upvotes). `download_connections` (1..=32) shapes parallelism but there is
   no bytes/second ceiling knob for bandwidth-constrained users. Small
   feature: config knob + token-bucket in `hf_parallel.rs` chunk reads.
2. **NPU lanes** (ollama #5186/#3004/#1590 class). No Intel/AMD NPU or
   OpenVINO lane; Vulkan covers Arc/iGPU class compute today. Upstream
   engine coverage question — document as roadmap rather than silently
   absent.
3. **Disk-based context offload** (llama.cpp #20697, open upstream FR).
   Blazar analog = KV banks + session checkpoints + unified-KV host
   relocation; the literal `--cache-disk` does not exist upstream yet. Track
   upstream, wire when it lands.
4. **License notices in release artifacts** (ollama #3185). Repo carries
   LICENSE-MIT/APACHE and scoop/winget manifests declare the license; the
   GitHub-release artifact lint (github-release-engineer gates H25/H26
   family) should assert NOTICE/LICENSE files ship inside every archive.
5. **Grammar sampling speed** (llama.cpp #4218) — upstream kernel work, out
   of blazar's control; cascade lane already avoids pathological re-rolls.

## 6. Complaint-ledger convention

In-repo inline ledger references (keep using when touching these surfaces):
#1 logprobs passthrough (openai.rs), #4 no blob storage, #6 real repo names,
#7 CVE-class path safety (hf.rs), #10 load-wait transparency (proxy.rs),
#12 keep_alive, #13 num_ctx honored (ollama.rs), #14 compat preview,
#15 hub search (hf.rs), #16 delete accepts the name /api/tags itself renders
(gateway ollama.rs delete → shared `proxy::resolve_model` ladder; found by the
F1 live-validation rig, pin `e2e__ollama_delete_accepts_the_tags_rendered_name`).

## 7. Round-2 sweep (2026-10-02, deeper complaint tiers)

Research: hidden perf knobs, env-var confusion, registry outages, auth,
slot/ctx division, Modelfile import, prompt-cache invalidation, sglang
Windows/CVEs. Code-verified outcomes:

| Next-tier complaint | Verdict (evidence) |
|---|---|
| OLLAMA_FLASH_ATTENTION / OLLAMA_KV_CACHE_TYPE hidden env-only knobs users miss | FIXED-BETTER: flash_attention defaults ON (measured DEFAULTS wave), KV quant governed with fa-guard (Wave H1); `blazar explain` surfaces effective config with provenance (1d57c52) |
| MAX_LOADED_MODELS ignored / evict despite ample RAM (v0.13 regression class) | FIXED: card-scoped co-residency planner A15 + per-device VRAM ledger + teaching RAM errors (`supervisor.rs:982,2160,3880`) |
| ollama.com registry outage blocks all pulls (no status page) | FIXED-BETTER: pulls go straight to HF + engine binaries from upstream GitHub releases — no central single-vendor registry in the path |
| No API auth / open port 11434 (recurring security complaint) | FIXED-BETTER: scoped `keys` (bearer identity + model scope + rate/token budgets, `blazar keys add`), TLS cert/key pair, CORS origins, per-child bearer auth (`config.rs:177-180,608-619`) |
| llama-server ctx-size divided across -np slots surprise | FIXED: posture-aware slot math computes real per-slot ctx; `admission_ctx` reads the LIVE child ctx (`preflight.rs:47`), preflight 400s name the effective number |
| Prompt-cache lost / full reprocess after system-prompt change | FIXED: cache_bust fingerprint (sys+tools change detection), unified-KV `--cache-reuse`, KV banks restore prefix after eviction |
| Modelfile import pain / blob copy / silent key drops | FIXED-BETTER: `blazar create -f Modelfile` = no blob copy, refuses-to-fake keys with teaching list (`cli/main.rs:7094`); GGUF import hardlinks; split GGUF wildcards handled natively |
| sglang Windows (uvloop dep, #389/#2249) | UPSTREAM-LIMIT: lane requires Linux/WSL; document WSL guidance (see fix list F3) |
| sglang RCE CVEs (CVE-2026-3059/3060 network-reach class) | MITIGATED-BY-ARCHITECTURE: children always bind 127.0.0.1 (`engine_impl.rs:424` "gateway stays the only public face") + scoped keys + TLS on the public face |
| ollama cloud/registry auth confusion (OLLAMA_API_KEY) | N/A: blazar is local-first; remote lane carries its own optional bearer (`config.rs:1861`) |

## 8. Consolidated implement/fix list (round 1 + round 2 survivors)

Actionable items, all others verified fixed:

- **F1. Pull download speed cap** (ollama #2006, 104 upvotes). **DONE
  2026-10-02**: `download_speed_limit_mb` config + `BLAZAR_DOWNLOAD_SPEED_LIMIT_MB`
  env; shared token bucket (1 s burst, `runtime/throttle.rs`) pacing model
  pulls, registry pulls, TTS/whisper voices, and engine-binary assets;
  progress bars name the cap; validation rejects negative/NaN. Receipts:
  `integration__download_speed_limit__paces_the_classic_lane` (2.5 s measured
  deficit at 1000 B/s cap), 6 throttle unit tests, config env/validation test,
  workspace 1657/1657. **Enforcement hardening after live e2e** (81b156d): the
  first cut let parallel chunk workers sleep concurrently and under-enforced
  the cap ~4x; the bucket lock is now held across the deficit sleep
  (tokio Mutex) with deficit carried as negative tokens. Pin
  `unit__acquire__concurrent_reads_enforce_the_aggregate_cap` (8 workers x
  250 KB @ 1 MB/s = 1.011 s). Live receipt: 45.9 MB registry pull at
  `BLAZAR_DOWNLOAD_SPEED_LIMIT_MB=1.0` measured 0.945 MB/s (46 s pacing +
  ~2.5 s manifest/TLS/store overhead). The same rig surfaced the
  `/api/delete` name-resolution gap, fixed in 31f5f8b — see ledger #16.
- **F2. Release-artifact license gate** (ollama #3185 parity). Extend the
  github-release pre-release checks: assert LICENSE-MIT/LICENSE-APACHE (+
  NOTICE if added) exist inside every published archive; fail the release if
  absent.
- **F3. Hardware/platform honesty rows in SETUP**. sglang lane: Linux/WSL
  requirement named at install + doctor (uvloop upstream limit). NPU class:
  one roadmap line (Vulkan covers Arc/iGPU compute today; NPU/OpenVINO =
  upstream engine question, tracked). Prevents the "why doesn't X work"
  issue-filing class.
- **F4. Homebrew license field** — verify `packaging/homebrew/blazar.rb`
  declares `license "MIT OR Apache-2.0"` like scoop/winget already do;
  one-line patch if missing.
- **F5. Upstream watchlist entry** — llama.cpp `--cache-disk` (#20697):
  wire as a posture knob when upstream ships it; until then KV banks +
  session checkpoints are the in-repo analog. One row in the audit doc is
  the tracker.

Rejected as non-actionable: grammar sampling speed (upstream kernel work),
ollama cloud stats (their telemetry product, not a local-server concern),
DeepSeek/roadmap model-support items (engine-level, arrive via
`engine update`), `--cache-disk` itself (not yet upstream).

## 9. Round-3 sweep — sdcpp + mistral.rs + whisper lanes (2026-10-02)

The three lanes never individually audited. Sources: leejet/stable-diffusion.cpp
+ EricLBuehler/mistral.rs + ggml-org/whisper.cpp top open issues (GitHub API,
2026-10-02) and community search (reddit/dev.to/arXiv on whisper streaming).

### stable-diffusion.cpp

| Complaint | Verdict (evidence) |
|---|---|
| #2081 sd-server accepts requests during startup window, spins CPU, never responds | FIXED-BY-ARCH: children are ready-gated (TCP health probe + boot-smoke) before the gateway routes to them |
| #1988 sd-server needs API-key support | FIXED-BETTER: gateway scoped keys (model scope + budgets) + supervisor-minted child auth (`engine_impl.rs:1043`) |
| #2022/#2073 Vulkan iGPU census reports "available 0.00 MB", model load fails | GOVERNED: `probe.rs:235-241` treats a zero-memory census as empty and falls back to nvidia-smi; `images.rs` submit gate is fail-open where VRAM is unmeasurable, teaching-400 + `vram_overcommit` lever where measurable |
| #2015 `--offload-to-cpu` pins past GTT budget, kills device | GOVERNED: offload is a posture choice with fit math (`profile.rs:3218-3237`), not a blind default; `--vae-tiling` and `--diffusion-fa` are gated knobs (`profile.rs:3005,3111`, `config.rs:532`) |
| #1971 please ship non-AVX-512 CPU builds | RESOLVED upstream: current releases ship one baseline `bin-win-cpu-x64.zip`; our CPU pattern excludes cuda/rocm/vulkan and picks it — no host-CPU detection needed |
| "./"-walk symlink loops under a service daemon cwd | FIXED: `child_cwd` anchor seats every child in its engine install dir (`engine_impl.rs:332-343`) — blazar hit and fixed this class independently |
| #2078/#2003/#1990 artifact/black-frame/Metal-tensor bugs, LTX/MiniMax/Wan video regressions | UPSTREAM kernels — arrive via engine updates |

### mistral.rs

| Complaint | Verdict |
|---|---|
| #2421 projector auto-discovery binds the directory's only mmproj to ANY model, crashes text-only GGUFs | FIXED-BETTER: blazar projector policy default `lazy` = text-only spawn, mmproj attached on demand (`config.rs:780-870`, incl. `mmproj_offload`/`mmproj_auto`/`mmproj_device`) |
| #2419 macOS CPU available memory reported as 0 MB | COVERED-LOUD: our spawn-time guard reads sysinfo `MemAvailable` with a hard floor and fails with a named error (never silent starvation); the 0-MB bug itself was mistral.rs's own reading |
| #2343 concurrent throughput does not scale above serial (H100, v0.9.0) | UPSTREAM PagedAttention bug; blazar admission bounds the queue, reshape observability tracked |
| #2460/#2435/#2411 ISQ deadlocks, GCC-13 CUDA link errors, malformed wheels | MITIGATED-BY-ARCH: pinned engine versions + boot-smoke gate reject bad builds at install time |
| #2441/#2427 tojson abort, tool-call tags leaking into reasoning content | UPSTREAM parser bugs |

### whisper.cpp

| Complaint | Verdict |
|---|---|
| Streaming ASR quality: "simplistic streaming mode, disjoint 30s-padded segments, unsuitable for deployment" (arXiv); "feels laggy for live apps" (dev.to) | **DONE (F6, 2026-10-02):** `stream=true` SSE now ships on the transcription endpoints — progressive chunk decode + rebased rolling segments (see fix list F6) |
| #4075 Metal 10x slowdown with non-default `audio_ctx` | IMMUNE: blazar never sets `audio_ctx` (defaults) |
| #4018 non-ASCII `-m` model path aborts (0xC0000409) on MSVC Windows | WATCHLIST (W1): non-ASCII Windows homes would crash the whisper child; mitigation when implemented = Windows short-path (8.3) normalization of child argv |
| #4041-4043 Go-binding CString leaks | N/A: blazar spawns binaries, no bindings |
| #4090/#4059 unvalidated GGUF header fields → OOB read/write | COVERED (class): pull-layer checksum/provenance invariants |
| #4026 no macOS binaries in releases | TAUGHT: `whisper_asset_patterns` returns an explicit build-from-source instruction for macOS (`gh.rs:1204-1213`) |

### Round-3 additions to the fix list

- **F6 — DONE 2026-10-02: streaming transcriptions shipped.** `stream=true` on
  `/v1/audio/transcriptions` + `/translations` returns SSE: WAV split at PCM
  frame boundaries into `whisper_stream_chunk_ms` windows (default 30 s,
  1-120 s), each transcribed in order on the same lazy child;
  `chunk.completed` events carry rebased segments while later audio still
  decodes, `transcript.completed` closes with the merged transcript. Gate:
  `stream` + `async` = 400; window-shaping fields dropped; hangup stops at the
  next chunk boundary. Live receipt: 66 s jfk-concat WAV, 6 s windows, 11
  chunks + final on the rig daemon.
- **W1 — DONE 2026-10-02.** 8.3 short-path argv guard on Windows
  (best-effort `GetShortPathNameW`, identity off-Windows) before the whisper
  child sees a non-ASCII model path (whisper.cpp #4018 abort class).

## 10. Round-4 intensive ollama sweep (2026-10-02, community-axis research)

Method: GitHub top-reacted refresh (list unchanged vs round 1; #10792 Gemma 3n
195+, #9387 phi4 multimodal 165+ are engine-level model-support, arrive via
`engine update`) + NEW axes: Reddit r/LocalLLaMA, troubleshooting-article
corpus (insiderllm/mrsaynothing/llmcheck 2026), GitHub Discussions, enterprise
logging guides, OpenAI-parity audits, ollama 0.30-0.35 engine-churn reports.

| Round-4 complaint class | Verdict (evidence) |
|---|---|
| Silent GPU→CPU fallback ("tok/s tanks", "ollama ps says CPU", Docker GPU loss over days) — the most-written 2026 troubleshooting article class | FIXED-BY-ARCH: placement is posture-decided pre-flight (teaching 400s carry the numbers), `/api/ps` returns `blazar_device` + `blazar_device_id` per instance (ollama.rs:432-433), per-GPU `/api/capacity`, `blazar explain`, boot-smoke gate at spawn — no runtime silent-fallback path exists |
| Stateful Responses API (ollama compat = non-stateful only, no `previous_response_id`) | FIXED-BETTER: full `previous_response_id` chaining, durable across restarts (store v8 responses registry, responses.rs, openai.rs:836) |
| `n` variants from one prompt (OpenAI chat `n` param) — open ollama compat complaint | GAP → **F7** (proposed): fan-out `n>1` in openai.rs (engines sample per-choice; needs gateway fan + response reshape) |
| MLX engine on Apple Silicon (ollama ships MLX lane for SOTA Mac perf) | ROADMAP → **F8** (doc note): our Mac lanes = llamacpp Metal + mistral.rs Metal; MLX lane = future wave, not silent |
| Engine-churn regressions (#18373 load slower, 0.33.x 5x tok/s drop) | IMMUNE-BY-ARCH: engines pinned per install, updates explicit + F7 boot-smoke gated + benchmarks receipt; no auto-churn |
| Docker GPU/OOM/networking complaint class (whole troubleshooting genre) | N/A-BY-DESIGN: native binaries + engine manager, no containers |
| Logs record full request/response content by default (PII warnings in ollama guides) | FIXED-BY-DESIGN: gateway logs metadata only — no body/prompt/message content logging anywhere in src/ (grep-verified) |
| Enterprise audit ("who said what", multi-user) | FIXED-BETTER: scoped keys named in 403s (keys.rs:53), KeyUsageRow accounting, durable responses/jobs rows correlate identity+request |
| `think: false` timeouts on non-interactive paths (suggestion engines) | FIXED: think-gates (Wave K O2) |
| Grammar/GBNF exposure (ollama declined the PRs) | FIXED: GBNF passthrough (256 KiB cap) |
| Windows GUI stuck-loading / macOS Ventura-Sequoia-Tahoe runner crashes | UPSTREAM-LANE: no GUI surface to break; engine crashes arrive via `engine update`, caught by boot-smoke + doctor currency rows |
| Perf gap vs raw llama-server (5x prompt-eval reports) | FIXED: posture defaults + FA-on default + admission receipts (2x t/s vs ollama measured) |

Round-4 net: two new actionable items — **F7** (`n` choices) and **F8** (MLX
roadmap row). Everything else verified fixed, better, or immune by
architecture.

## 11. Sources (selection)

- GitHub Search API top-reacted open issues: ollama/ollama,
  sgl-project/sglang, ggml-org/llama.cpp (2026-10-02 snapshots).
- Reddit r/ollama + r/LocalLLaMA threads on num_ctx truncation, keep_alive,
  NUM_PARALLEL, memory growth; HN "The local LLM ecosystem doesn't need
  Ollama"; ssdnodes/cohorte/glukhov deep-dives on ollama concurrency and
  queue 503s; knowledgeforagents + LinkedIn on sglang OOM modes;
  docs.sglang.io install pages + CUDA-mismatch bug reports; ramgpt +
  deepwiki on llama-server buffer allocation and chat-template machinery.
- In-repo receipts: BENCHMARK.md, bench-artifacts/ campaigns, MEMORY.md wave
  logs ( Waves D–M), git log 2026-09-26..10-02.
