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

## Install verification

Every lane records a digest at install time, and any recorded digest can be
re-checked later with `blazar engine verify <tag>`:

- **GitHub lanes** (llama.cpp, whisper, kokoro model assets, mistral.rs) are
  digest-verified as the archive downloads - size-truncation guard plus
  sha256 match against the release manifest. `engine verify` re-hashes the
  cached download archive; when the cache has been pruned the command prints
  an honest skip (exit 0): the digest passed install-time verification and
  nothing on disk contradicts it.
- **pip lanes** (SGLang, vLLM, MLX, LiteRT, kokoro runtime) verify the venv
  after install by re-hashing every file its `*.dist-info/RECORD` manifests
  list (RFC 4180 quoted paths included) and stamping the row with
  `pip:<sha256>` - a digest over the normalized, sorted manifest. Any file
  that drifted, went missing, or tries to escape the venv fails the install
  with the file named. `engine verify` re-walks the RECORDs and compares
  against the row; a venv that changed after install (a JIT runtime
  rewriting a file, a hand-edit) fails loud with a reinstall hint.
- **Honest gaps.** Lanes installed before digest recording report
  `unverified` and say so; verification never invents a pass it cannot
  prove.

## Host memory ledger

Host RAM gets the same treatment as VRAM: a reservation ledger, per-resident
accounting, and a pressure watcher - not just a spawn-time floor.

- **One floor, two lanes.** The spawn floor (`model/2 + 512 MiB + CPU offload`,
  `blazar-core` `host_floor_mib`) is engine-agnostic. What differs by lane is
  what a *resident* charges against the ledger: mmap-streaming lanes
  (llama.cpp & friends) hold file-backed pages the kernel would happily reclaim,
  so they charge `min(measured RSS, weights)`; python-runtime lanes (sglang,
  vLLM) hold anonymous pages already excluded from MemAvailable by the kernel,
  so they charge zero - subtracting their RSS too would double-count and
  over-refuse.
- **Atomic window.** A spawn reserves its floor from guard to insertion
  (file-backed lanes release at insert; python lanes hold until settle-measured
  RSS lands), so concurrent launches cannot oversubscribe one host budget.
- **Measured, then reconciled.** At settle, the child's process-group RSS is
  walked from `/proc` (`smaps_rollup`) and logged against the prediction -
  receipts that refine the model, not silent drift.
- **Pressure watcher.** The reaper classifies MemAvailable every tick:
  `normal` > 3 GiB effective, `degraded` <= 2 GiB (new spawns taught the ledger
  numbers), `pressure` <= 1 GiB (evict coldest zero-in-flight resident, one per
  tick, hysteresis on relief). Thresholds are documented constants, not knobs.
- **Surfaced everywhere.** `GET /api/capacity` host block, `blazar explain`
  host admission card, `blazar plan` HOST section, `GET /api/ps`
  `host_rss_mib`/`host_charge_bytes` per resident.

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

### Capability cards

`blazar engine card [tag|--kind K]` renders one lane's derived capability
card: the kind-level contract (serving class, modalities, formats the
routing table lets this lane decode, slot-bank support, serving tier) plus
the installed row's probed facts (build identity, spec/cache types, flag
vocabulary, GPU census, parallelism flags) and the KV connector caps under
the live `[kv.lmcache]` truth. Every field is DERIVED from the predicates
and routing tables the runtime already enforces - the card cannot claim
something the router or supervisor gates would refuse, and a lane grows a
capability by updating the predicate, never by editing the card. An
uninstalled `--kind` still renders the lane contract with build/probe
sections marked absent. `GET /api/engines` serves the same cards as JSON,
one object per installed row.

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
(NVIDIA megakernel / AMD monokernel), and MoE paged experts
(`--enable-paged-experts` + a derived `--paged-experts-num-resident K`):
when a routed-MoE checkpoint (GPTQ, AWQ, FP8, INT8, MXFP4 and the FP4
variants; the triton and Marlin runners) is too large for the GPU, the
compiler keeps K of E experts resident and pages the rest on demand from
a page-locked host store - compute stays on the GPU, unlike a CPU
offload. The ladder engages it automatically on a single-GPU host with
no `cpu_offload_gb` pin and no speculative draft, or pins it explicitly
with `paged_experts = true` (and optionally `paged_experts_num_resident`;
`false` rolls back to the classic offload tiers). The host must be able
to pin ALL routed experts in RAM - the spawn memory guard refuses early
with both numbers when it cannot - and upstream rejects the pairing with
tensor parallelism or speculative decoding.

Live receipt (OLMoE-1B-7B-0924-GPTQ, 64 experts, one RTX 4070): steady decode 44.5 tok/s at
K=8, 63.9 at K=16, 124.5 at K=32 and 254.0 unpaged, with TTFT moving the other way (1382 to
238 ms) and KV token capacity growing 11927 to 33783 (2.8x) as experts leave the card - and
the unpaged baseline cannot even start at the auto memory fraction while the paged build
loads (bench-artifacts/2026-11-23-sota-roadmap/sglang-paged-experts-receipt.md).

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

Draft-free speculative decoding ships behind `[vllm] spec_method`: `suffix`
(SuffixDecoding - variable-length longest-suffix match plus a
device-resident cross-request token corpus, TP-safe by design) or `ngram`
need no draft model, so the pair is a plain spawn flag set. The profile
compiler emits `--spec-method` with `--spec-tokens` (`spec_num_tokens`,
defaulting shallow to 4 - the engine's own unset default is
`min(8, tree_depth)`, deep enough to cost throughput at high concurrency)
and refuses a window that would not stay strictly below the compiled
`--max-model-len` before the model load, where the engine would refuse it
anyway. Manual spec flags riding `extra_args` are refused while the knob
is set (upstream treats the group flags and `--speculative-config` as
mutually exclusive); draft-model families (`eagle` and friends) stay in
`extra_args` for hand-assembled plans. Request-level `n > 1` rides the
byte-faithful OpenAI chat and completions passthrough (vLLM itself refuses
it with greedy sampling - pair `n` with a temperature above zero).

### MLX

Apple-Silicon-native (and CUDA-backed) serving for mlx-community quantized
directories. Installed as a pip environment (`mlx-lm`) and managed like any
other lane. An MLX-format directory has this lane as its only decoder: once
the lane is installed the router picks it with no model pin, and without it
the request fails with a teaching error that names the install command.

Install requirements, checked before any wheel is downloaded: Linux needs
an NVIDIA GPU visible through NVML or `nvidia-smi` (CUDA backend, glibc
2.35 or newer, x86_64 or aarch64); macOS needs Apple Silicon (arm64,
macOS 14 or newer - mlx publishes no Intel-Mac wheel, use the llama.cpp
lane there); Windows is refused because the lane's `mlx-server` shim is a
POSIX script (use WSL2 or another engine kind).

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

Known upstream issue: mlx-lm releases through 0.32.2 (the newest
release at the time of writing) can silently drop a tool call - the
response reports `finish_reason: "tool_calls"` while `message.tool_calls`
is missing and the generated text is lost (the model emitted call
markers but the parser rejected the text between them; most visible with
small quantized models). Reproduced on 0.32.2: `qwen2.5-0.5b-instruct-4bit`
loses the call, `qwen3-1.7b-4bit` calls correctly. The upstream fix
merged on 2026-10-20
([ml-explore/mlx-lm#1973](https://github.com/ml-explore/mlx-lm/pull/1973))
and is in no release yet, so `/v1/chat/completions` tool traffic on this
lane is served through Blazar's recipe lane instead (gateway-rendered
prompt, raw completion, call parsed by the gateway). A model too small to
follow that prompt echoes the tool block instead of calling: the 0.5B
model above does, so tool calls on this lane need a larger model. A
generation cut by `max_tokens` reports `finish_reason: "length"` on both
paths, streaming included.

**Hardware status** (see [Hardware receipts](#hardware-receipts) below):
the CUDA backend has a live receipt on an RTX 4070. Apple Silicon has
none: the install plan and the macOS GPU census are verified against
published `system_profiler` captures and compile-checked, but no Metal run
has been recorded, so the lane is experimental on macOS and `blazar doctor`
flags it there. Parity note: ollama runs models on MLX on Apple Silicon by
default since v0.40.0; Blazar serves the same `mlx-community` directories
through this lane.

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

Receipted on the CPU runtime (smollm2-360m-instruct, 2026-11-25): the
runtime's own benchmark reports 180.33 tok/s prefill and 24.21 tok/s
decode on that host; through Blazar a 200-token generation ran at a
17.0 tok/s median against 19.5 tok/s direct, streaming at 14.1 frames/s
(bench-artifacts/2026-11-23-sota-roadmap/litert-lm-receipt.md). The GPU
and NPU backends are not receipted: `serve` has no backend switch, and
the bundle in the receipt declares no function-call support, so tool
requests come back without calls.

## Hardware receipts

Blazar installs CUDA, ROCm, SYCL, OpenVINO, Vulkan and Metal builds of its
engines, but a build that installs is not a build that was measured. A lane
counts as production-ready on a hardware class only after a live run on that
class left an artifact behind. Every other class below is **experimental**
there: compile-checked and fixture-tested, never run.

| Lane | Backend | GPU vendor | Live receipt |
|---|---|---|---|
| llamacpp | cuda | nvidia | `bench-artifacts/20261005-kv-quality/benchmark.md` |
| llamacpp | rocm | amd | none (experimental) |
| llamacpp | sycl | intel | none (experimental) |
| llamacpp | vulkan | amd | none (experimental) |
| llamacpp | vulkan | intel | none (experimental) |
| llamacpp | vulkan | nvidia | `bench-artifacts/2026-11-23-sota-roadmap/llamacpp-vulkan-receipt.md` |
| llamacpp | metal | apple | none (experimental) |
| llamacpp | openvino | intel | none (experimental) |
| mistralrs | cuda | nvidia | none (experimental) |
| mistralrs | metal | apple | none (experimental) |
| mlx | cuda | nvidia | `bench-artifacts/20261010-engine-utilization/mlx_kv_bits_ab.md` |
| mlx | metal | apple | none (experimental) |
| sdcpp | vulkan | nvidia | `bench-artifacts/20261103-media-lanes/REPORT.md` |

- llamacpp (rocm, amd): no host with an AMD GPU was at hand; the current release ships ubuntu-rocm-10.0-x64, which carries its runtime, installs on any host and lists no device without an AMD GPU

- llamacpp (sycl, intel): no host with the Intel oneAPI runtime was at hand; the SYCL prebuilt needs twelve oneAPI libraries it does not ship and does not start without them

- llamacpp (vulkan, intel): runs on an Intel RPL-S iGPU with correct output at 9 to 14 tok/s, against 25 to 44 tok/s on the CPU backend of the same build (bench-artifacts/2026-11-23-sota-roadmap/llamacpp-vulkan-receipt.md); not a performance receipt

- llamacpp (openvino, intel): the build served through its CPU device on a host without an NPU or an Intel GPU at 41.0 tok/s median, against 52.2 tok/s on the CPU backend of the CUDA build; the GPU and NPU devices never ran (bench-artifacts/2026-11-23-sota-roadmap/llamacpp-openvino-receipt.md); not an accelerator receipt

- mistralrs (cuda, nvidia): a live spawn proof exists (bench-artifacts/20261010-engine-utilization/live_spawn_proofs.md) but names no backend, so it is not counted

A receipt is cited only when the artifact names the GPU and the backend the
run exercised. A class missing from the table has no receipt either.
`blazar doctor` prints the same verdict for the lanes installed on the host
(row `hardware receipts`, group `ENGINES`), so an experimental install is
flagged on the machine that runs it. To close a gap, run the lane on that
hardware, keep the report under `bench-artifacts/` with the GPU and the
backend named in it, and cite its path in
`crates/blazar-core/src/hardware_receipts.rs` (a test fails when the code
and this table drift apart).

### Which llama.cpp build installs where

`blazar engine update` reads the vendor runtime that is installed, not only
the GPU: the `ROCm` build when `/opt/rocm` or `/dev/kfd` exists, the `SYCL`
build when `/opt/intel/oneapi` exists. A Linux x86_64 host with neither gets
the `Vulkan` build when Vulkan lists a discrete GPU, and the CPU build
otherwise; NVIDIA hosts take the CUDA build (setup guide). An
integrated-only host keeps the CPU build: on the measured laptop the Vulkan
build on the Intel iGPU served at 9 to 14 tok/s against 25 to 44 tok/s on
the CPU backend of the same binary. A prebuilt that cannot start names the
runtime it needs: the `SYCL` build exits before its version banner when the
oneAPI runtime is missing, and Blazar refuses the install instead of
registering an engine that never starts. The OpenVINO build is never
picked automatically - it is opt-in through `engine_asset` (NPUs below).

### NPUs and the OpenVINO build

Intel NPUs, and Intel GPUs without a Vulkan driver, are served by the
prebuilt llama.cpp OpenVINO build - opt-in and experimental. Pin the asset
(`engine_asset = "ubuntu-openvino-2026.4.1-x64"`, the suffix changes with
every release), set the device in the engine environment
(`GGML_OPENVINO_DEVICE = "NPU"`, a device the host lacks is a hard error),
and pin models to that engine with one slot and a bounded context: the NPU
backend is stateless, single-slot, and defaults its context to the model's
training size. Blazar never passes `--device` for that build: its listing
rows are the menu for the environment variable, and the build aborts a
spawn whose `--device` names another row. AMD XDNA2 and Snapdragon NPU
runtimes are vendor servers and attach through `[[remotes]]`. The
measurements and the reasoning behind all of this:
`docs/research/2026-11-25-npu-decision.md`.
