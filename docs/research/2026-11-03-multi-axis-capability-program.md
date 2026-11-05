# Multi-axis capability program - research and proposed architecture

Date: 2026-11-03. Session r12 follow-up to the universal-architecture program
(G1-G8 all resolved; see 2026-11-02-universal-architecture-coverage.md).
This document is the evidence base and architecture proposal for the five
axes that remain open. No implementation yet - awaiting wave approval.

Evidence classes used throughout: [strings] = binary strings probe of the
installed engine; [api] = live HF API query; [gh] = live GitHub API query;
[code] = blazar source read; [doc] = fetched upstream documentation;
[receipt] = live blazar run artifact in bench-artifacts/.

## Axis A - engine-delta absorption (b12020 -> b12072)

The engine lane moves weekly; blazar must absorb new surfaces deliberately,
not accidentally. Findings on the current b12072 build:

- PocketTTS predefined voices are LIVE in the engine: libllama-server-impl
  exposes `voices` via GET /props ("see voices in GET /props"), libmtmd
  ships `clip_gen_get_voices` + `%zu predefined voices`, and upstream PR
  #31429 documents the design: kyutai/pocket-tts-without-voice-cloning
  weights bake each voice into the mmproj as recovered prompt embeddings;
  the /tts `voice` field (and --tts-voice) selects one; speaker_ref is
  rejected when the model has no voice encoder [strings + doc + gh release
  notes]. HF hosting: kyutai/pocket-tts-without-voice-cloning (12.6k dl),
  lmz/pocket-tts-without-voice-cloning-q8 [api].
- The G1 engine speech lane ALREADY passes `voice` through to the child
  (tts.rs engine_tts_child_body); the gap is discovery only: /v1/audio/voices
  is piper-only and /v1/audio/capabilities does not list engine voices [code].
- Clover arch family (clover / clover_moe) is loadable in b12072+ [strings];
  the GGUF ecosystem is large and hot: Diffuspace/Clover-35B-A3B-Preview-GGUF
  43k downloads, maternion 16.5k, mudler APEX variants [api]. No blazar
  catalog entry; the llamacpp lane spawns it like any text LLM (no special
  handling identified as needed).
- EmbeddingGemma: ggml-org/embeddinggemma-2-GGUF 124k downloads + unsloth
  mirror 155k [api]; arch gemma-embedding(2) present in the engine arch
  table [strings]; blazar already serves it through the child embeddings
  lane once pulled (bert-family census also counts it via the 'embedding'
  substring). Gap = catalog entry only.
- GET /props exposes `modalities` (vision flag) plus build info [doc] -
  a cheap, spawn-time capability signal the runtime already partially uses.

Proposed architecture (small, two waves):

- A1 "engine voices surface": /v1/audio/voices gains an engine section per
  resident speech-capable child (voices read from the child's GET /props,
  TTL-cached exactly like the sdcpp child_capabilities pattern - no boot,
  honest empty when nothing resident); /v1/audio/capabilities engine lane
  documents the `voice` knob + points at the voices endpoint. Catalog:
  `pocket-tts` entry (class speech; mmproj-with-voices verified at impl).
  STATUS: CLOSED 2026-11-03. Shipped: `engines[]` section in
  /v1/audio/voices (resident LlamaCpp children, speech gate =
  speech_generation_arch OR cached 'audio' modality, GET /props probe,
  600 s TTL cache, defensive normalize accepting string arrays and
  {id,name} objects, probe failures logged + retried, no boot - honest
  empty list when nothing resident); capabilities engine-lane voices
  note; `pockettts` joined speech_generation_arch (alongside qwen3tts) so
  the @vision spawn never emits the --embeddings parity arm that
  segfaulted speech models. Live 7/7 on b12072 through the sandbox
  gateway with a self-converted kyutai/pocket-tts-without-voice-cloning
  pair (model 159 MB + mmproj 53 MB baking all 27 English voices):
  default + named-voice WAV synthesis, engines[] listing the 27 voices,
  cache-hit path, capabilities note, piper regression, honest-empty
  before boot (receipts: bench-artifacts/20261102-universal-coverage/
  wa1-live-receipts.json + wa1-default.wav + wa1-giovanni.wav). No
  hosted llama.cpp-format GGUF exists yet (lmz q8 = model-only,
  Serveurperso targets pockettts.cpp) - the catalog `pocket-tts` entry
  is deferred to W-A2 with a conversion recipe note (two convert_hf_to_
  gguf.py runs, the second with --mmproj and a distinct --outfile).
  Implementation note: EngineRef.name carries the lane suffix for lane
  instances - store lookups from live endpoints must strip it via
  model_of_key(&eng.key), or the speech gate silently skips every child.
- A2 "new-arch catalog pass": clover-35b-a3b + embeddinggemma-2 entries
  (text + embeddings classes); engine-update absorption checklist added to
  the engine update flow output: arch-table diff, route diff, release-note
  scan for new model classes - three greps the updater already has the
  binaries for.

## Axis B - performance receipts for the new lanes

Findings on the current harness [code]:

- `blazar bench <model>` is llama-bench-shaped (pp/tg tok/s, TTFT/ITL p95)
  with a spec-race A/B mode; `optimize run` already has live-concurrency
  machinery (C concurrent clients, reps, slots axis).
- bench_arch_refusal deliberately refuses encoder-only arches (bert,
  modern-bert, nomic-bert, t5encoder, xlm-roberta) because llama-bench
  hangs on them [code comment: observed live hours-class hangs].
- No bench profile exists for the lanes this program shipped: speech,
  embeddings, multimodal embeddings, image/video generation.

Proposed architecture (medium, one wave):

- New bench profiles, one per lane family, each reusing the optimize
  C-clients pattern and each emitting the AGENTS.md reproducibility
  receipt (engine + gateway config, cold/warm state, raw JSONL + summary):
  - embeddings: req/s + p50/p95/p99 latency at C=1..8, dim + l2-norm
    validity check per response (bert child lane, candle CLIP lane,
    VLM content-array lane).
  - speech: RTF (audio seconds / wall seconds) + chars/s at C=1..4 for
    the piper lane, the engine /tts lane, and standalone-engine baseline
    (the direct-vs-blazar overhead number the mandate requires).
  - image/video: time-to-first-pixel unavailable by API (buffered), so
    wall-time per fixed seed/size/step-count, plus VRAM watermark via
    nvidia-smi sampling; direct sd-server baseline for overhead.
- Store receipts under bench-artifacts/<date>-media-lanes/ with the JSON +
  markdown report + reproduction commands, per the benchmarking mandate.

## Axis C - ollama complaint audit (top open issues by reactions, live GH API)

Mapped to blazar state:

| ollama issue | reactions | blazar state |
|---|---|---|
| #3368 reranking models | 385 | SOLVED - /v1/rerank live-validated (bge-reranker receipts) |
| #3185 license distribution in artifacts | 297 | OPEN (release-engineering; add to release checklist) |
| #8618 Janus-Pro vision | 272 | covered via llamacpp lane if GGUF exists (verify at impl) |
| #7865 MCP support | 222 | SOLVED - MCP waves shipped earlier |
| #5245 multi-file GGUF import | 221 | SOLVED - shard sets + components + exact-filename pull |
| #10792 Gemma 3n | 195 | SOLVED - gemma3n in engine arch table |
| #228 home-dir clutter | 192 | SOLVED - XDG layout |
| #5186/#3004/#1590/#2169 NPU/Arc/OpenVINO accelerators | 168+139+111+95 | OPEN - the one big class blazar does not own yet |
| #9387 phi4 multimodal | 165 | covered via mtmd towers (verify per-model) |
| #3144 /metrics | 143 | SOLVED - --metrics + obs plane |
| #786 image generation | 124 | SOLVED - 35 diffusion families + video |
| #162 autostart control | 123 | SOLVED - service management |
| #2006 rate-limit download speed | 104 | OPEN - pull has bandwidth PREDICTION only, no user cap [code: rg rate/throttle, no knob] |
| #10989 qwen3-embedding/reranker | 83 | SOLVED - arch table + lanes |

Proposed architecture: (1) publish the solved-table above as part of the
positioning docs (honest, receipt-linked); (2) small wave: `--max-download-
mbps` pull/download cap (config + flag, enforced at the chunk-loop level
where progress already flows); (3) accelerators = research-first sub-axis:
inventory llama.cpp SYCL/Vulkan/OpenCL asset lanes vs blazar's installer
(cuda/vulkan today), propose an asset-lane extension design before any
code (NPU support is upstream-engine-dependent; do not promise).

## Axis D - embedding breadth

- EmbeddingGemma-2: massive GGUF ecosystem [api]; needs only the catalog
  entry (rides the existing child embeddings lane).
- SigLIP: candle-transformers ships models/siglip.rs with a
  Config{text_config, vision_config} shape [code: registry read] - the
  config-driven G6 recipe translates directly. GGUF ecosystem exists
  (cstr/siglip-base/large/so400m, SigLIP2 NaFlex experiments) [api] but
  the canonical serving path is the safetensors row (google/siglip-*),
  same as CLIP today. NaFlex (variable resolution) is SigLIP2-only and
  stays out of wave 1.
- Rerank breadth: bge-reranker covers the flagship; no action identified
  beyond catalog.

Proposed architecture (small-medium, one wave): extend the candle lane to
SiglipModel rows (config-driven translator + honest quick_gelu-class
activation handling exactly as G6; dispatch: `clip_lane_serves` widens to
CLIPModel + SiglipModel), plus the embeddinggemma catalog entry. Estimated
touch: clip.rs (renamed semantics), openai.rs one-line predicate, catalog.

## Axis E - candle PR merge

PR #4087 open upstream. On merge + release: bump the dependency, delete
blazar's hand-written HF-config translator in favor of the crate's own
deserialization. No standalone wave; rides the next dependency bump.

## Proposed wave order

1. W-A1 engine voices surface (small; immediate UX value; b12072 already
   ships the engine side).
2. W-A2 new-arch catalog pass + absorption checklist - STATUS: CLOSED
   2026-11-03. Catalog +2 entries: clover-35b-a3b (Diffuspace mirror,
   the only ungated Q4_K_M hosting verified at freeze; Clover MoE
   landed upstream in b12069+) and embeddinggemma-2 (ggml-org official,
   class embeddings; multimodal embedder - gemma-embedding2 arch +
   mmproj, text rides the child embeddings lane, images the G2b
   content-array path; no Q4_K_M upstream so quant fallback applies).
   pocket-tts stays uncataloged: no hosted llama.cpp-format GGUF exists
   (the W-A1 receipts carry the two-command converter recipe). New CLI
   subcommand `blazar engine delta [TAG]` = the absorption checklist:
   set-diffs the mined manifests (architectures/flags/spec types) of
   two installed llama.cpp rows - pure store lookup, no binary probing;
   honest teachings for single-row installs, unknown tags, and tags on
   other lanes (kind named). Live receipts (wa2-engine-delta-receipts
   .json): real-store teaching paths + sandbox diff on the real b12072
   manifest (168 arches) vs a documented synthetic older row (+clover_
   moe, JSON mode, explicit-tag reverse). Tests 2403/2403 workspace.
 3. W-B media-lane perf receipts (medium; satisfies the benchmarking
    mandate for everything this program shipped).
    STATUS: CLOSED 2026-11-03. Shipped scripts/bench_media.py (stdlib-only,
    warmup-before-timing, concurrency ladders, VRAM watermarks, append-only
    receipts.jsonl + per-lane CSV, direct-engine baselines driven from the
    store row's own components JSON). Nine batteries banked in
    bench-artifacts/20261103-media-lanes/ (+ REPORT.md): bert text
    embeddings gateway 80.7 req/s @ C=1 scaling to 126.4 @ C=8 vs direct
    llama-server flatline ~71 (gateway overhead within noise, pooling win);
    candle CLIP CPU 0.556 req/s flat (honest serialized ceiling);
    qwen3vl content-array multimodal 6.68 req/s @ C=1; piper gateway 12.0x
    realtime; engine TTS gateway 38.7x realtime on CUDA vs 9.8x direct
    pinned CPU (placement-policy receipt, asymmetry documented); image
    qwen-image-2.1 512x512 seed-deterministic (15 gateway reps byte
    identical) mean 46.0s vs direct 39.7s (+16% = conservative offload
    profile on 8 GiB card, not proxy overhead); video wan2.1 320x320 ~35s
    per 1s clip. Companion product fix shipped mid-wave: candle CLIP lane
    served JSON as text/plain (String IntoResponse) - now axum::Json
    (openai.rs clip_embeddings tail). Image --direct mode made exclusive
    after diagnosing silent child death under two 11.6 GiB diffusion
    children co-resident (oom-killer disabled on the bench box).
 4. W-D embedding breadth: SigLIP candle lane + embeddinggemma
    (small-med). STATUS: CLOSED 2026-11-03. New
    crates/blazar-runtime/src/siglip.rs (G6 recipe): sparse-config reader
    leaning on candle serde defaults (google ships only patch_size +
    dims), hidden_act translated to candle's lowercase serde names
    (unknown refused), preprocess reuses the CLIP [-1,1] affine (SigLIP
    processor mean .5/std .5 is identical), resident mmap cache, sole-
    row resolve w/ pull teachings. Gateway dispatch: candle_lane_serves
    widens to CLIPModel+SiglipModel; clip_embeddings resolves per named
    row w/ siglip sole-row fallback for model-less requests; dynamic
    blazar_note. Spawn-gate companion fix: supervisor safetensors gate
    now refuses SiglipModel w/ teaching (was: sglang crash 502 - live
    bisected; CLIP teaching verbatim preserved). Catalog: siglip-base-
    patch16-224 (class clip, google official). Live 4/4 on b12072
    (wd-siglip-live-receipts.json): 768-d l2=1.0 image embed; CLIP 512-d
    regression; sole-CLIP priority for model-less; text-part -> 400
    spawn-gate teaching. embeddinggemma-2 entry already landed in W-A2
    (rides child embeddings + content-array lanes). 2418/2418 green.
    LESSON: candle 0.11 Activation serde names are LOWERCASE (unit pin
    caught the PascalCase misread); /tmp wipe #3 mid-wave - zombie
    daemon from the trash-expunged old build held port 11599 and
    answered probes with stale code (readlink /proc/<pid>/exe revealed
    it; always verify the daemon binary path when a fix 'doesn't work').
 5. W-C download cap + solved-table publication + accelerator research
    note (small; accelerator design deferred to its own research pass).
    STATUS: CLOSED 2026-11-03. CORRECTION: the research claim "no user
    cap" was stale - the throttle infrastructure (config key
    download_speed_limit_mb, BLAZAR_DOWNLOAD_SPEED_LIMIT_MB env,
    chunk-loop pacing shared across parallel connections and engine
    downloads) already lived at HEAD from earlier commits; only the
    per-invocation flag was missing. Shipped: `blazar pull
    --max-download-mbps` (flag > config > unlimited; 0 = unlimited;
    finite >= 0 validated with teaching BEFORE any network traffic;
    log line names the cap when it engages). Live receipts
    (wc-download-cap-receipts.json): -3/nan/inf teach via exit 1 (bare
    dash-values additionally caught by clap's own parser, exit 2, no
    network); 28 MiB pull 31.6 s capped at 1 MB/s vs 4.6 s uncapped
    (6.9x, proving the shared throttle paces the real chunk loop); help
    surface present; sandbox restored (row + files removed). The
    solved-complaints table above (section with the ollama issue
    comparison) IS this doc's publication; accelerators stay a
    research-only note pending their own pass.

All waves follow the session contract: gateway-layer only, live no-mock
validation, receipts to bench-artifacts, research-report + CHANGELOG +
MEMORY flips per wave.
