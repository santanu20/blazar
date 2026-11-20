---
layout: doc
title: Engines
description: The nine serving engines Blazar orchestrates - roles, model formats, and how routing picks between them.
doc_kind: Documentation
---
## One gateway, nine engines

Blazar does not replace inference engines - it orchestrates them. Each lane
is the real upstream binary, managed end-to-end: download or build, keep
current, configure, supervise, and route requests to it. Pip-installed lanes
(SGLang, vLLM, MLX, LiteRT-LM, Kokoro) resolve the package index's latest release
at install time - falling back to the last verified release with a warning
when the index is unreachable - while GitHub-release lanes track their
release channel; `blazar engine pin` freezes a lane when reproducibility
matters.

| Engine | Role | Typical workloads |
|---|---|---|
| **llama.cpp** | Mainstream local text serving | GGUF and quantized local models |
| **mistral.rs** | Alternative text runtime | Supported GGUF and safetensors models |
| **SGLang** | Safetensors-oriented serving | AWQ, GPTQ, FP8 safetensors models |
| **vLLM** | Throughput-first OpenAI-native serving | HF safetensors directories (PagedAttention) |
| **MLX** | Apple-Silicon and CUDA-backed serving | MLX quantized model directories (mlx-community) |
| **stable-diffusion.cpp** | Media generation | Diffusion images and video |
| **whisper.cpp** | Speech recognition | Transcription and translation |
| **piper** | Offline speech synthesis | Local TTS voices |
| **kokoro** | Neural speech synthesis | Kokoro ONNX voices (24 kHz, CPU) |
| **LiteRT-LM** (`openaihost`) | Edge-optimized text serving | `.litertlm` bundles (LiteRT-LM mobile/edge/NPU builds) |

## Format-driven routing

Routing follows model format and engine capability:

```text
GGUF                      -> llama.cpp / mistral.rs
Quantized safetensors     -> SGLang
Plain safetensors         -> policy order (Throughput: vLLM / SGLang / mistral.rs,
                             Quality: SGLang / mistral.rs / vLLM,
                             Latency: mistral.rs / SGLang / vLLM)
MLX quant directories     -> MLX
Diffusion component sets  -> stable-diffusion.cpp
Audio transcription       -> whisper.cpp
Offline TTS               -> piper
Neural TTS (kokoro)       -> kokoro
.litertlm bundles         -> LiteRT-LM (openaihost)
```

The default preserves single-active-engine behavior; per-model engine pins
give deterministic placement. Curated capability lanes cover GGUF
architectures the mainstream llama.cpp lane does not yet support.

## Engine lifecycle

Every lane supports the same management surface:

```sh
blazar engine update --all        # refresh every lane to its latest
blazar engine install --kind ...  # add a lane by engine kind
blazar engine pin <tag>           # hold a lane at a known version
blazar doctor versions            # lane + binary health overview
```

Ask Blazar why a model landed on a lane:

```sh
blazar explain <model>            # the routing decision, with reasons
```

## Engine integration contract

Every lane - llama.cpp, mistral.rs, sglang, vLLM, MLX, sdcpp, OpenAI-host -
implements the same five semantics. An engine is production-ready when all
five hold; each is pinned by tests, not convention:

| Semantic | Contract | Where |
|---|---|---|
| Capability declaration | Probed `Manifest` vocabulary (flags, devices, spec/cache types, trust tier); argv may only emit advertised flags | `engine/manifest.rs` (`probe_kind`, `MANIFEST_PROBE_VERSION` generations), coverage doc below |
| Health | Readiness = the engine's own signal (llama: `{"status":"ok"}` body; sglang: probe-200; sdcpp: `/v1/models`), polled within `model_load_timeout` | `Engine::health_check` + per-lane deadline/signal unit tests |
| Cancellation | Own process group, kernel parent-death tie (Linux PDEATHSIG), `kill_on_drop`; termination is single-pid by design | `spawn_child` in `engine_impl.rs`, `probe::parent_death_tie` |
| Memory accounting | Weights + admission floor gate the spawn; the measured settle delta is observational, exposed at `/api/admission-accuracy` | `admission_floor_bytes`, spawn-settle task, accuracy poller |
| Lifecycle | Instance states Loading/Ready/Sleeping/Evicted/Crashed with one legal transition matrix; evicted/crashed are terminal | `events.rs` (`can_transition`), architecture doc |

Adding a lane means: a `probe_<kind>` manifest parser, an `Engine` impl
honoring the contract above, and a coverage section in this doc. Preflight
refusals (dead `--rpc` endpoints, unsupported transports) happen BEFORE any
child exists - crash-looping after spawn is a contract violation.

## Per-engine notes

### llama.cpp

The default text lane for GGUF checkpoints. Slot-parallel serving, KV-cache
quantization, and n-gram speculative decoding. Curated capability lanes
extend coverage to GGUF architectures the mainstream build does not ship.

### mistral.rs

The second text lane: GGUF and supported safetensors models, with its own
serving and speculation paths. A per-model pin moves any supported checkpoint
here deterministically.

### SGLang

The throughput lane for quantized safetensors directories - AWQ, GPTQ, FP8.
Blazar derives its parallel sizes, memory fraction, and KV-cache budget from
your hardware before the server ever starts. Advanced serving surfaces are
typed knobs with capability-probed emission (older lanes warn-and-degrade
rather than pass dead flags): prefill/decode disaggregation (`disagg_pair`
spawns decode/prefill role children plus a supervised `sgl-router` child -
real mooncake KV transfer, both OpenAI chat and the native `/generate` route
forwarded through it), the HiCache L3 storage tier (`hicache_storage_backend`
family), speculative decoding (NGRAM, EAGLE3 draft pairs, DFlash, DSpark,
DSpine), context parallelism (`cp_size`/`attn_cp_size`/`dcp_size` plus
`cp_strategy`/`dcp_comm_backend` companions), attention data parallelism
(`attn_dp_size`, must divide `tp_size`), CUDA-graph backend selection per
phase, PDMux collocation, and the experimental fused-kernel plugin lanes
(NVIDIA megakernel / AMD monokernel).

### vLLM

The throughput-first lane for plain HF safetensors directories - it wins the
Throughput policy order (Quality and Latency still prefer SGLang and
mistral.rs). Installed as a pip venv (Linux + NVIDIA, ~6-8 GiB incl. torch
and the CUDA wheels), launched through a `vllm-server` shim that execs the
venv's own `vllm serve`. The profile compiler owns the VRAM guardrails:
`--max-model-len` from the pinned context, `--gpu-memory-utilization` always
emitted (default 0.85 - set `[vllm] gpu_memory_utilization` lower when the
GPU is shared with a co-tenant daemon or another lane), and
`--max-num-seqs` when a slot pin exists. Tool calling
(`--enable-auto-tool-choice` + parser selection) is the
incremental-certification wave; v0 ships plain chat, streaming, and strict
`json_schema` output.

Persistent KV reuse ships behind `[vllm.kv_offload]` (opt-in): the child
boots with the built-in OffloadingConnector (`kv_both`) tiering completed
blocks into a pinned CPU buffer (`cpu_bytes_gib`, default 2 GiB) plus a
content-addressed filesystem store (`root_dir`, default
`<data_dir>/kv/vllm`). The store keys blocks by a digest of the run
configuration - a different model or context pin writes a different tree
and reads nothing back, so a mismatch fails cold, never with wrong reuse.
Restarted children reload matching prompt prefixes instead of recomputing
them; the effect is observable as `cached_tokens` in usage (the
`--enable-prompt-tokens-details` flag rides every spawn) and surfaces in
the gateway's warm/cold TTFT split and `prompt_eval_cached_count`.

### MLX

Apple-Silicon-native (and CUDA-backed) serving for mlx-community quantized
directories. Installed as a pip environment (`mlx-lm`) and managed like any
other lane.

Memory and context on this lane are **runtime-managed**: `mlx_lm` sizes its
own caches and decides residency inside its runtime. Blazar's contract with
observers is explicit about what that means:

- `ps` shows `blazar_capacity: "runtime-managed"` - the lane is neither
  unknown-capacity nor unlimited; the engine owns memory admission.
- `blazar_ctx` reports the model's trained window (`max_position_embeddings`
  from config.json, pulled at model download) as a **fit ceiling**: prompt
  preflight bounds requests by it, but it is not an allocation Blazar makes
  and no ctx flag is passed to the server. Rows without config.json metadata
  honestly report `0` and admission falls back to the configured default.
- KV-cache estimate stays unset (`kv_est_bytes: null`) because the runtime,
  not Blazar, sizes the caches - co-residency planning uses the measured
  settle footprint (weights + actual card delta) instead, the same bytes
  admission that gates every other lane.
- Session banking (KV checkpoint save/restore) is a llama-server `/slots`
  API and is not invoked on this lane; conversation warmth comes from the
  engine's own prompt cache (`--prompt-cache-*` knobs tune it).

Known upstream issue: mlx-lm 0.32.0 can silently drop a tool call - the
response reports `finish_reason: "tool_calls"` while `message.tool_calls`
is missing and the generated text is lost (the model emitted call markers
but the parser rejected the text between them; most visible with small
quantized models). Upstream fix pending in
[ml-explore/mlx-lm#1957](https://github.com/ml-explore/mlx-lm/pull/1957);
until it lands and the lane's engine is refreshed, tool-call traffic on
this lane may hit it.

### stable-diffusion.cpp

Diffusion images (and video component sets) on CPU or GPU, including Vulkan
devices. Served over the same OpenAI-style image API surface.

### whisper.cpp

Speech transcription and translation, with streaming capture and Silero VAD
gating for live audio.

### piper

Offline neural TTS with local voices - no network, no cloud service, one
API surface shared with the rest of the media lanes.

### kokoro

Kokoro-82M neural speech synthesis (kokoro-onnx venv + ONNX voice model,
CPU, ~0.4 GiB lane). 24 kHz WAV, 50+ voices (`af_*`/`am_*` English first;
`"model": "kokoro"` picks the `af_heart` default). Voices ride
`/v1/audio/speech` alongside piper's; a resident child enumerates them
under `/v1/audio/voices` (kokoro section). Buffered WAV only - streaming
and mp3/opus/aac/flac continue to ride the piper voices. Install with
`blazar engine install 1.0 --kind kokoro`; the gateway owns the child
lifecycle (lazy spawn + idle reap under `whisper_idle_secs`).

### LiteRT-LM (openaihost)

Text serving for `.litertlm` bundles - Google's LiteRT-LM runtime tuned
for mobile, edge, and NPU execution. Installed as a `litert-lm` pip
environment (~0.4 GiB, any platform) and managed like any other lane:
`engine install --kind openaihost`, `engine update --kind openaihost`,
`doctor versions litert`. A supervised shared child serves every
`.litertlm` model on the lane.
