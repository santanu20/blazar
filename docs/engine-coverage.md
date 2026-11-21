# Engine capability coverage

Disposition of every engine-advertised flag against Blazar's
emission-truth registry.

- fixtures: `scripts/manifest_flag_fixtures.json` (engine manifest
  snapshots, refreshed from a live store via `--refresh`)
- registry: `scripts/first_class_flags.json` (live emission;
  regenerate via `cargo run -p blazar-runtime --example
  knob_registry`)
- capabilities: the registry's `capabilities` section - the per-lane
  capability matrix (family dispositions, evidence status, receipts)
  generated from crates/blazar-core/src/capability_matrix.rs
- receipts: `scripts/certification_receipts.json` (live-engine runs
  that upgrade first-class flags to certified)
- gate: `python3 scripts/engine_coverage.py --check` fails CI when
  the doc drifts from any input. No timestamps - the bytes are
  the contract.
- census: the same gate also fails when a user-facing inventory
  surface (README, engine/usage docs, system overview, validate
  harness, CLI) stops mentioning any of the 10
  census lanes - exempt files and their reasons live in
  CENSUS_EXEMPT.

| Disposition | Meaning |
|---|---|
| certified | first-class + live-engine receipt (behavior proven) |
| first-class | Blazar pins it from a config knob or derivation |
| passthrough | reachable via `extra_args`, no first-class knob |
| intentionally-unsupported | gateway-owned; `extra_args` refuses it |
| not-applicable | not a capability Blazar could integrate |
| unverified | no verified reach through Blazar on this lane |

Capability matrix evidence: `live-certified` (receipt-backed),
`first-class` (typed knob), `probed` (manifest truth only),
`passthrough` (reachable, unmanaged), `unsupported` (refused or
engine-internal).

Policy: promotion from passthrough to a first-class knob requires
workload evidence - a receipt linking the flag to a live-engine
run. Emission tests gate the *how*; receipts gate the *why*.

## 2023.11.14-2 (piper) - 16 flags probed

**certified: 0 | first-class: 9 | passthrough: 0 | intentionally-unsupported: 0 | not-applicable: 7 | unverified: 0**

Registry lane `piper` pins 10 first-class flags (compile + spawn-time emission).

### Capability matrix

| Family | Evidence | Flags | Notes |
|---|---|---|---|
| batching_scheduling | unsupported | - | one-shot synthesis spawns; concurrency is gateway-side |
| kv_cache_quantization | unsupported | - | no KV cache on the TTS lane |
| prefix_caching | unsupported | - | no decode cache on the TTS lane |
| speculative_decoding | unsupported | - | not applicable to synthesis |
| multi_gpu_parallelism | unsupported | - | CPU synthesis lane |
| memory_offload | unsupported | - | voice models are device-memory-sized by design |
| structured_output | unsupported | - | audio out; no grammar surface |
| lora | unsupported | - | no adapter surface |
| multimodal | unsupported | - | single-modality audio-out lane (see speech_synthesis) |
| cancellation_streaming | first-class | - | gateway-owned job cancel |
| speech_synthesis | live-certified | `--model`, `--output_raw`, `--length_scale`, `--speaker` | piper-TTS wav synthesized and round-trip transcribed by whisper beam-5 child ([proof](bench-artifacts/20261010-engine-utilization/live_spawn_proofs.md)) |

### Not applicable (deliberate, per-flag reasons)

- --config: config-file input competes with Blazar's composed argv
- --debug: interactive CLI ergonomics, not a synthesis capability
- --help: interactive CLI ergonomics, not a synthesis capability
- --json-input: stdin JSON request surface; Blazar feeds plain text lines
- --output_dir: directory batch output; Blazar always names explicit files (--output_file / --output_raw)
- --quiet: long-form alias of the first-class -q the lane emits
- --tashkeel_model: Arabic diacritization; needs a separate model asset and an Arabic voice, and the lane has no extra_args surface

## b12578-cuda (llamacpp) - 340 flags probed

**certified: 2 | first-class: 89 | passthrough: 249 | intentionally-unsupported: 0 | not-applicable: 0 | unverified: 0**

Registry lane `llamacpp` pins 94 first-class flags (compile + spawn-time emission).

### Capability matrix

| Family | Evidence | Flags | Notes |
|---|---|---|---|
| batching_scheduling | first-class | `-np`, `--batch-size`, `--ubatch-size`, `--no-cont-batching` | slots=0 capacity-aware sizing, adaptive_slots governor and the decode gate manage batching |
| kv_cache_quantization | live-certified | `--cache-type-k`, `--cache-type-v` | q4_0 KV quality-free at 2048/32k ctx; f8_e4m3 pin-only above 16k - ladder policy is receipt-driven ([proof](bench-artifacts/20261111-fp8-kv-quality/README.md)) |
| prefix_caching | first-class | `--no-cache-prompt` | prompt cache default-on; cache_hit_rate hint feeds the spec/cache governors |
| speculative_decoding | first-class | `--spec-default`, `--spec-draft-n-min`, `--spec-draft-p-min` | spec=auto family selection, ngram fallback and the adaptive spec governor |
| multi_gpu_parallelism | passthrough | - | rpc/multi-node surface is extra_args-only; single-box layer splits auto-derive |
| memory_offload | first-class | `--no-kv-offload`, `--op-offload`, `--cache-ram` | KV/operator offload and cache-ram budget are config knobs |
| structured_output | first-class | `--grammar`, `--json-schema`, `--json-schema-file` | GGUF grammars and JSON-schema pins ride the launch line |
| lora | first-class | `--lora-scaled`, `--lora-init-without-apply` | adapter scaling and no-apply load are first-class knobs |
| multimodal | first-class | `--media-path`, `--no-mmproj-auto`, `--no-mmproj-offload` | mmproj auto-detect and offload knobs |
| cancellation_streaming | first-class | - | gateway-owned SSE stream and cancel path |

### Certified against a live engine

- `--cache-type-k` [`bench-artifacts/20261111-fp8-kv-quality/README.md`](bench-artifacts/20261111-fp8-kv-quality/README.md) - q4_0 KV quality-free at 2048 and 32k ctx (20261005-kv-quality); f8_e4m3 parity at 2048 but +0.141 ppl at 16384 (~2x err bars) - pin-only, ladder never auto-selects
- `--cache-type-v` [`bench-artifacts/20261111-fp8-kv-quality/README.md`](bench-artifacts/20261111-fp8-kv-quality/README.md) - q4_0 KV quality-free at 2048 and 32k ctx (20261005-kv-quality); f8_e4m3 parity at 2048 but +0.141 ppl at 16384 (~2x err bars) - pin-only, ladder never auto-selects

### Passthrough families

- **serving detail** (68): --cache-idle-slots, --cache-list, --cache-prompt, --cache-reuse, --check-tensors, --completion-bash (+62 more)
- **draft/speculative detail** (60): --cache-type-k-draft, --cache-type-v-draft, --cpu-mask-batch-draft, --cpu-mask-draft, --cpu-moe-draft, --cpu-range-draft (+54 more)
- **uncategorized long-tail** (55): --adaptive-decay, --adaptive-target, --agent, --allow-unauthenticated-tools, --connect, --connect-code (+49 more)
- **per-request sampling** (21): --backend-sampling, --chat-template-file, --dry-sequence-breaker, --dynatemp-exp, --dynatemp-range, --image-min-tokens (+15 more)
- **gateway-owned transport** (17): --api-key, --api-key-file, --api-prefix, --cors-credentials, --cors-headers, --cors-methods (+11 more)
- **observability** (13): --log-colors, --log-disable, --log-file, --log-jsonl, --log-prefix, --log-prompts-dir (+7 more)
- **multimodal** (10): --image-max-tokens, --mmproj, --mmproj-auto, --mmproj-device, --mmproj-offload, --mmproj-url (+4 more)
- **CPU affinity** (3): --cpu-range, --threads-batch, --threads-http
- **LoRA advanced** (1): --lora
- **rpc/multi-node** (1): --rpc

## b5907 (whisper) - 54 flags probed

**certified: 1 | first-class: 14 | passthrough: 39 | intentionally-unsupported: 0 | not-applicable: 0 | unverified: 0**

Registry lane `whisper` pins 15 first-class flags (compile + spawn-time emission).

### Capability matrix

| Family | Evidence | Flags | Notes |
|---|---|---|---|
| batching_scheduling | unsupported | - | single-utterance transcription; no batch surface |
| kv_cache_quantization | unsupported | - | no persistent KV on the audio lane |
| prefix_caching | unsupported | - | no decode cache on the audio lane |
| speculative_decoding | unsupported | - | encoder-decoder; no speculation surface |
| multi_gpu_parallelism | unsupported | - | single-device audio lane |
| memory_offload | unsupported | - | model fits device memory by admission design |
| structured_output | unsupported | - | plain text/json output; no grammar surface |
| lora | unsupported | - | no adapter surface advertised |
| multimodal | first-class | `--diarize`, `--tinydiarize`, `--vad`, `--vad-model` | audio-in lane; diarization and VAD are first-class knobs |
| cancellation_streaming | first-class | - | gateway-owned job cancel |

### Certified against a live engine

- `--beam-size` [`bench-artifacts/20261010-engine-utilization/live_spawn_proofs.md`](bench-artifacts/20261010-engine-utilization/live_spawn_proofs.md) - beam-5 child; piper-TTS wav transcribed exactly

### Passthrough families

- **uncategorized long-tail** (24): --audio-ctx, --convert, --debug-mode, --device, --dtw, --flash-attn (+18 more)
- **segmentation/timing** (8): --duration, --max-context, --max-len, --no-timestamps, --offset-n, --offset-t (+2 more)
- **transcription/translation** (6): --carry-initial-prompt, --detect-language, --language, --no-language-probabilities, --prompt, --translate
- **decoding quality** (1): --no-fallback

## fork-AtomicBot-ai_atomic-llama-cpp-turboquant-8d935078-cuda (llamacpp) - 349 flags probed

**certified: 2 | first-class: 88 | passthrough: 259 | intentionally-unsupported: 0 | not-applicable: 0 | unverified: 0**

Registry lane `llamacpp` pins 94 first-class flags (compile + spawn-time emission).

### Capability matrix

| Family | Evidence | Flags | Notes |
|---|---|---|---|
| batching_scheduling | first-class | `-np`, `--batch-size`, `--ubatch-size`, `--no-cont-batching` | slots=0 capacity-aware sizing, adaptive_slots governor and the decode gate manage batching |
| kv_cache_quantization | live-certified | `--cache-type-k`, `--cache-type-v` | q4_0 KV quality-free at 2048/32k ctx; f8_e4m3 pin-only above 16k - ladder policy is receipt-driven ([proof](bench-artifacts/20261111-fp8-kv-quality/README.md)) |
| prefix_caching | first-class | `--no-cache-prompt` | prompt cache default-on; cache_hit_rate hint feeds the spec/cache governors |
| speculative_decoding | first-class | `--spec-default`, `--spec-draft-n-min`, `--spec-draft-p-min` | spec=auto family selection, ngram fallback and the adaptive spec governor |
| multi_gpu_parallelism | passthrough | - | rpc/multi-node surface is extra_args-only; single-box layer splits auto-derive |
| memory_offload | first-class | `--no-kv-offload`, `--op-offload`, `--cache-ram` | KV/operator offload and cache-ram budget are config knobs |
| structured_output | first-class | `--grammar`, `--json-schema`, `--json-schema-file` | GGUF grammars and JSON-schema pins ride the launch line |
| lora | first-class | `--lora-scaled`, `--lora-init-without-apply` | adapter scaling and no-apply load are first-class knobs |
| multimodal | first-class | `--media-path`, `--no-mmproj-auto`, `--no-mmproj-offload` | mmproj auto-detect and offload knobs |
| cancellation_streaming | first-class | - | gateway-owned SSE stream and cancel path |

### Certified against a live engine

- `--cache-type-k` [`bench-artifacts/20261111-fp8-kv-quality/README.md`](bench-artifacts/20261111-fp8-kv-quality/README.md) - q4_0 KV quality-free at 2048 and 32k ctx (20261005-kv-quality); f8_e4m3 parity at 2048 but +0.141 ppl at 16384 (~2x err bars) - pin-only, ladder never auto-selects
- `--cache-type-v` [`bench-artifacts/20261111-fp8-kv-quality/README.md`](bench-artifacts/20261111-fp8-kv-quality/README.md) - q4_0 KV quality-free at 2048 and 32k ctx (20261005-kv-quality); f8_e4m3 parity at 2048 but +0.141 ppl at 16384 (~2x err bars) - pin-only, ladder never auto-selects

### Passthrough families

- **serving detail** (87): --cache-idle-slots, --cache-list, --cache-prompt, --cache-reuse, --check-tensors, --completion-bash (+81 more)
- **draft/speculative detail** (60): --cache-type-k-draft, --cache-type-v-draft, --cpu-mask-batch-draft, --cpu-mask-draft, --cpu-moe-draft, --cpu-range-draft (+54 more)
- **uncategorized long-tail** (50): --adaptive-decay, --adaptive-target, --agent, --cpu-moe, --cpu-strict, --decision (+44 more)
- **per-request sampling** (22): --backend-sampling, --chat-template-file, --dry-sequence-breaker, --dynatemp-exp, --dynatemp-range, --image-min-tokens (+16 more)
- **gateway-owned transport** (17): --api-key, --api-key-file, --api-prefix, --cors-credentials, --cors-headers, --cors-methods (+11 more)
- **observability** (12): --decision-debug, --log-colors, --log-disable, --log-file, --log-prefix, --log-prompts-dir (+6 more)
- **multimodal** (6): --image-max-tokens, --mmproj, --mmproj-auto, --mmproj-offload, --mmproj-url, --no-mmproj
- **CPU affinity** (3): --cpu-range, --threads-batch, --threads-http
- **LoRA advanced** (1): --lora
- **rpc/multi-node** (1): --rpc

## fork-GuideboardLabs_llama.cpp-sensenova-loop-a7f77a78-cuda (llamacpp) - 334 flags probed

**certified: 2 | first-class: 89 | passthrough: 243 | intentionally-unsupported: 0 | not-applicable: 0 | unverified: 0**

Registry lane `llamacpp` pins 94 first-class flags (compile + spawn-time emission).

### Capability matrix

| Family | Evidence | Flags | Notes |
|---|---|---|---|
| batching_scheduling | first-class | `-np`, `--batch-size`, `--ubatch-size`, `--no-cont-batching` | slots=0 capacity-aware sizing, adaptive_slots governor and the decode gate manage batching |
| kv_cache_quantization | live-certified | `--cache-type-k`, `--cache-type-v` | q4_0 KV quality-free at 2048/32k ctx; f8_e4m3 pin-only above 16k - ladder policy is receipt-driven ([proof](bench-artifacts/20261111-fp8-kv-quality/README.md)) |
| prefix_caching | first-class | `--no-cache-prompt` | prompt cache default-on; cache_hit_rate hint feeds the spec/cache governors |
| speculative_decoding | first-class | `--spec-default`, `--spec-draft-n-min`, `--spec-draft-p-min` | spec=auto family selection, ngram fallback and the adaptive spec governor |
| multi_gpu_parallelism | passthrough | - | rpc/multi-node surface is extra_args-only; single-box layer splits auto-derive |
| memory_offload | first-class | `--no-kv-offload`, `--op-offload`, `--cache-ram` | KV/operator offload and cache-ram budget are config knobs |
| structured_output | first-class | `--grammar`, `--json-schema`, `--json-schema-file` | GGUF grammars and JSON-schema pins ride the launch line |
| lora | first-class | `--lora-scaled`, `--lora-init-without-apply` | adapter scaling and no-apply load are first-class knobs |
| multimodal | first-class | `--media-path`, `--no-mmproj-auto`, `--no-mmproj-offload` | mmproj auto-detect and offload knobs |
| cancellation_streaming | first-class | - | gateway-owned SSE stream and cancel path |

### Certified against a live engine

- `--cache-type-k` [`bench-artifacts/20261111-fp8-kv-quality/README.md`](bench-artifacts/20261111-fp8-kv-quality/README.md) - q4_0 KV quality-free at 2048 and 32k ctx (20261005-kv-quality); f8_e4m3 parity at 2048 but +0.141 ppl at 16384 (~2x err bars) - pin-only, ladder never auto-selects
- `--cache-type-v` [`bench-artifacts/20261111-fp8-kv-quality/README.md`](bench-artifacts/20261111-fp8-kv-quality/README.md) - q4_0 KV quality-free at 2048 and 32k ctx (20261005-kv-quality); f8_e4m3 parity at 2048 but +0.141 ppl at 16384 (~2x err bars) - pin-only, ladder never auto-selects

### Passthrough families

- **serving detail** (68): --cache-idle-slots, --cache-list, --cache-prompt, --cache-reuse, --check-tensors, --completion-bash (+62 more)
- **draft/speculative detail** (60): --cache-type-k-draft, --cache-type-v-draft, --cpu-mask-batch-draft, --cpu-mask-draft, --cpu-moe-draft, --cpu-range-draft (+54 more)
- **uncategorized long-tail** (49): --adaptive-decay, --adaptive-target, --agent, --connect, --connect-code, --cpu-moe (+43 more)
- **per-request sampling** (21): --backend-sampling, --chat-template-file, --dry-sequence-breaker, --dynatemp-exp, --dynatemp-range, --image-min-tokens (+15 more)
- **gateway-owned transport** (17): --api-key, --api-key-file, --api-prefix, --cors-credentials, --cors-headers, --cors-methods (+11 more)
- **observability** (13): --log-colors, --log-disable, --log-file, --log-jsonl, --log-prefix, --log-prompts-dir (+7 more)
- **multimodal** (10): --image-max-tokens, --mmproj, --mmproj-auto, --mmproj-device, --mmproj-offload, --mmproj-url (+4 more)
- **CPU affinity** (3): --cpu-range, --threads-batch, --threads-http
- **LoRA advanced** (1): --lora
- **rpc/multi-node** (1): --rpc

## kokoro-1.0 (kokoro) - 0 flags probed

**certified: 0 | first-class: 0 | passthrough: 0 | intentionally-unsupported: 0 | not-applicable: 0 | unverified: 0**

### Capability matrix

| Family | Evidence | Flags | Notes |
|---|---|---|---|
| batching_scheduling | unsupported | - | single /tts shim child; concurrency is gateway-side |
| kv_cache_quantization | unsupported | - | no KV cache on the TTS lane |
| prefix_caching | unsupported | - | no decode cache on the TTS lane |
| speculative_decoding | unsupported | - | not applicable to synthesis |
| multi_gpu_parallelism | unsupported | - | CPU ONNX lane |
| memory_offload | unsupported | - | 82M-param model is device-memory-sized by design |
| structured_output | unsupported | - | audio out; no grammar surface |
| lora | unsupported | - | no adapter surface |
| multimodal | unsupported | - | single-modality audio-out lane (see speech_synthesis) |
| cancellation_streaming | first-class | - | gateway-owned job cancel |
| speech_synthesis | live-certified | - | zero-CLI-flag python shim: Blazar composes the whole serving path (venv + ONNX model + /tts endpoint); af_heart 24 kHz wav certified live 2026-11-20 ([proof](bench-artifacts/20261120-kokoro-vllm-cert/README.md)) |

Unverified: this lane has no emission-truth registry entry yet - every flag above is listed as unverified until it grows one (see knob_registry.rs).

## litert-0.19.2 (openaihost) - 6 flags probed

**certified: 2 | first-class: 0 | passthrough: 0 | intentionally-unsupported: 0 | not-applicable: 4 | unverified: 0**

Registry lane `openaihost` pins 2 first-class flags (compile + spawn-time emission).

### Capability matrix

| Family | Evidence | Flags | Notes |
|---|---|---|---|
| batching_scheduling | unsupported | - | child owns its serving config; Blazar spawns one shared child |
| kv_cache_quantization | unsupported | - | child-internal, not argv-configurable |
| prefix_caching | unsupported | - | child-internal |
| speculative_decoding | unsupported | - | child-internal |
| multi_gpu_parallelism | unsupported | - | child-internal |
| memory_offload | unsupported | - | child-internal |
| structured_output | probed | - | request-side via the OpenAI surface |
| lora | unsupported | - | child-internal model registry |
| multimodal | probed | - | depends on the child's imported models |
| cancellation_streaming | first-class | - | gateway-owned stream and cancel over the child's OpenAI surface |

### Certified against a live engine

- `--host` [`docs/research/2026-11-07-openai-host-lane-phase-a.md`](docs/research/2026-11-07-openai-host-lane-phase-a.md) - live litert-lm serve lane probe (127.0.0.1:9379) with gateway chat round-trips; phase-A chunked-TE A/B ran against this server
- `--port` [`docs/research/2026-11-07-openai-host-lane-phase-a.md`](docs/research/2026-11-07-openai-host-lane-phase-a.md) - live litert-lm serve lane probe (127.0.0.1:9379) with gateway chat round-trips; phase-A chunked-TE A/B ran against this server

### Not applicable (deliberate, per-flag reasons)

- --config: config-file input competes with Blazar's composed argv and the child's own model registry
- --cors-origin: child binds loopback behind the gateway; the gateway is the public CORS face
- --help: interactive CLI ergonomics, not a serving capability
- --verbose: child-side log verbosity; Blazar owns child log capture

## master-1043-3be2378 (sdcpp) - 165 flags probed

**certified: 2 | first-class: 39 | passthrough: 119 | intentionally-unsupported: 5 | not-applicable: 0 | unverified: 0**

Registry lane `sdcpp` pins 41 first-class flags (compile + spawn-time emission).

### Capability matrix

| Family | Evidence | Flags | Notes |
|---|---|---|---|
| batching_scheduling | unsupported | - | one-shot diffusion generation; no batch admission surface |
| kv_cache_quantization | unsupported | - | no KV cache on a diffusion lane |
| prefix_caching | unsupported | - | no decode cache on a diffusion lane |
| speculative_decoding | unsupported | - | not applicable to diffusion |
| multi_gpu_parallelism | passthrough | `--rpc-servers` | rpc server list is extra_args-reachable; not auto-planned |
| memory_offload | first-class | `--offload-to-cpu`, `--max-vram` | CPU offload and VRAM ceiling knobs |
| structured_output | unsupported | - | image bytes, not constrained text |
| lora | first-class | `--lora-model-dir` | LoRA directory knob for diffusion adapters |
| multimodal | first-class | `--llm`, `--clip_vision`, `--qwen2vl_vision` | the lane IS multimodal generation; component sets are first-class |
| cancellation_streaming | first-class | - | job ledger + gateway cancel own the lifecycle |

### Certified against a live engine

- `--hires-upscalers-dir` [`bench-artifacts/20261010-engine-utilization/live_spawn_proofs.md`](bench-artifacts/20261010-engine-utilization/live_spawn_proofs.md) - RealESRGAN_x4plus end-to-end: 512x512 generation upscaled to 2048x2048
- `--upscale-model` [`bench-artifacts/20261010-engine-utilization/live_spawn_proofs.md`](bench-artifacts/20261010-engine-utilization/live_spawn_proofs.md) - RealESRGAN_x4plus end-to-end: 512x512 generation upscaled to 2048x2048

### Passthrough families

- **uncategorized long-tail** (59): --ad-model, --attn-scale, --audio, --auto-fit, --circular, --circularx (+53 more)
- **sampling/generation** (25): --batch-count, --batched-cfg, --control-strength, --end-img, --extra-sample-args, --guidance (+19 more)
- **input/output files** (18): --ad-negative-prompt, --ad-prompt, --control-image, --control-video, --disable-prefetch, --increase-ref-index (+12 more)
- **component set (blazar-owned)** (13): --audio-vae, --clip-on-cpu, --clip-skip, --force-sdxl-vae-conv-scale, --llm., --llm_vision. (+7 more)
- **memory/backend placement** (4): --control-net-cpu, --diffusion-fa, --flash-attn, --list-devices

Intentionally-unsupported (gateway-owned, refused in `extra_args`): --clip_g, --llm_vision, --model, --qwen2vl_vision, --threads

Scope disposition: knob emission is pinned by the registry; per-component behavior against real diffusion models is certification-pending (tracked by the benchmark harness, not this audit).

## mlx-0.32.2 (mlx) - 27 flags probed

**certified: 1 | first-class: 17 | passthrough: 9 | intentionally-unsupported: 0 | not-applicable: 0 | unverified: 0**

Registry lane `mlx` pins 18 first-class flags (compile + spawn-time emission).

### Capability matrix

| Family | Evidence | Flags | Notes |
|---|---|---|---|
| batching_scheduling | first-class | `--decode-concurrency`, `--prompt-concurrency` | decode/prompt concurrency knobs |
| kv_cache_quantization | live-certified | `--kv-bits`, `--kv-group-size`, `--quantized-kv-start` | A/B: 1.9x slower decode, 108MiB saved - memory lever, opt-in ([proof](bench-artifacts/20261010-engine-utilization/mlx_kv_bits_ab.md)) |
| prefix_caching | first-class | `--prompt-cache-size`, `--prompt-cache-bytes` | prompt cache budget knobs |
| speculative_decoding | first-class | `--draft-model`, `--num-draft-tokens` | draft model + token count knobs |
| multi_gpu_parallelism | unsupported | - | single-device MLX runtime |
| memory_offload | unsupported | - | mlx manages memory internally; kv-bits is the memory lever |
| structured_output | probed | - | request-side only on this lane |
| lora | first-class | `--adapter-path` | adapter path knob |
| multimodal | probed | - | mlx-vl exists upstream; no tower routing wired |
| cancellation_streaming | first-class | - | gateway-owned stream and cancel |

### Certified against a live engine

- `--kv-bits` [`bench-artifacts/20261010-engine-utilization/mlx_kv_bits_ab.md`](bench-artifacts/20261010-engine-utilization/mlx_kv_bits_ab.md) - A/B on qwen2.5-0.5b: 1.9x slower decode, 108MiB saved - memory lever, opt-in

### Passthrough families

- **uncategorized long-tail** (9): --allowed-origins, --help, --log-level, --max-tokens, --min-p, --pipeline (+3 more)

## sglang-0.5.23 (sglang) - 604 flags probed

**certified: 3 | first-class: 69 | passthrough: 531 | intentionally-unsupported: 1 | not-applicable: 0 | unverified: 0**

Registry lane `sglang` pins 72 first-class flags (compile + spawn-time emission).

### Capability matrix

| Family | Evidence | Flags | Notes |
|---|---|---|---|
| batching_scheduling | first-class | `--schedule-policy`, `--schedule-conservativeness`, `--max-total-tokens`, `--max-prefill-tokens` | scheduler policy and token budget knobs |
| kv_cache_quantization | first-class | `--kv-cache-dtype` | KV dtype knob on the paged cache |
| prefix_caching | first-class | `--enable-hierarchical-cache`, `--hicache-ratio`, `--hicache-size`, `--enable-session-radix-cache` | radix + hierarchical cache knobs; hicache L3 codec translates across engines |
| speculative_decoding | live-certified | `--speculative-algorithm`, `--speculative-num-draft-tokens` | live spec battery: +7% on repetitive traffic, negative on diverse - kept opt-in ([proof](bench-artifacts/20261009-sglang-spec/spec_race_lane.md)) |
| multi_gpu_parallelism | first-class | `--tp-size`, `--dp-size`, `--pp-size`, `--ep-size` | all four parallel axes are sized by the multi-GPU planner |
| memory_offload | first-class | `--cpu-offload-gb`, `--mem-fraction-static` | CPU offload and static memory fraction knobs |
| structured_output | probed | - | xgrammar rides requests; the launch-side grammar backend is not pinned |
| lora | live-certified | `--enable-lora`, `--lora-paths` | one-child colon routing + hot-attach on the same pid ([proof](docs/research/2026-10-09-sglang-lora-one-child.md)) |
| multimodal | first-class | `--mm-attention-backend` | multimodal attention backend knob |
| cancellation_streaming | first-class | - | gateway-owned stream and cancel |

### Certified against a live engine

- `--enable-lora` [`docs/research/2026-10-09-sglang-lora-one-child.md`](docs/research/2026-10-09-sglang-lora-one-child.md) - one-child colon routing + hot-attach via /api/lora/sync on the same pid
- `--lora-paths` [`docs/research/2026-10-09-sglang-lora-one-child.md`](docs/research/2026-10-09-sglang-lora-one-child.md) - one-child colon routing + hot-attach via /api/lora/sync on the same pid
- `--speculative-algorithm` [`bench-artifacts/20261009-sglang-spec/spec_race_lane.md`](bench-artifacts/20261009-sglang-spec/spec_race_lane.md) - live sglang-0.5.21 spec battery: ngram exercised via --speculative-algorithm; +7% on repetitive search prompts, negative on diverse traffic - gain is workload-dependent, kept opt-in (raw cells: live_battery.json, bench-archive release)

### Passthrough families

- **uncategorized long-tail** (253): --abort-on-priority-when-disabled, --api-, --bf16-gemm-backend, --boundary-reduction, --c128-page-size, --chat-template (+247 more)
- **scheduler/memory detail** (129): --attention-, --attention-context-parallel-size, --attention-data-parallel-size, --cuda-graph-, --cuda-graph-backend-decode, --cuda-graph-config (+123 more)
- **per-request/gateway-owned** (28): --admin-api-key, --allow-auto-truncate, --constrained-json-disable-any-whitespace, --constrained-json-max-whitespace-cnt, --constrained-json-whitespace-pattern, --enable-http-worker-reuseport (+22 more)
- **observability** (27): --bucket-e2e-request-latency, --bucket-inter-token-latency, --bucket-time-to-first-token, --decode-log-interval, --enable-forward-pass-metrics, --enable-metrics-for-all-schedulers (+21 more)
- **tokenizer/sampler internals** (17): --disable-tokenizer-batch-decode, --enable-tokenizer-batch-encode, --generation-tokens-buckets, --kt-max-deferred-experts-per-token, --mlx-enable-sampling, --num-reserved-decode-tokens (+11 more)
- **multi-node/parallelism** (16): --attn-cp-size, --attn-dp-size, --base-gpu-id, --disable-custom-all-reduce, --dist-init-addr, --dwdp-size (+10 more)
- **disaggregated/PD serving** (15): --disaggregation-bootstrap-port, --disaggregation-decode-allocation-policy, --disaggregation-decode-enable-offload-kvcache, --disaggregation-decode-enable-radix-cache, --disaggregation-decode-extra-slots, --disaggregation-decode-host-receive-threshold (+9 more)
- **MoE backends** (14): --deepep-config, --deepep-dispatcher-output-dtype, --deepep-mode, --deepep-v2-mode, --disable-flashinfer-cutlass-moe-fp4-allgather, --enable-fused-moe-sum-all-reduce (+8 more)
- **RL/rollout tooling** (12): --disable-overlap-schedule, --enable-lora-overlap-loading, --enable-single-batch-overlap, --encoder-register-urls, --encoder-urls, --hicache-storage-prefetch-policy (+6 more)
- **LoRA advanced** (12): --experts-shared-outer-loras, --lora-drain-wait-threshold, --lora-paths., --lora-strict-loading, --lora-target-modules, --lora-use-virtual-experts (+6 more)
- **multimodal/ASR** (8): --allowed-media-domains, --asr-max-buffer-seconds, --asr-max-concurrent-sessions, --disable-fast-image-processor, --enable-multimodal, --image-processor-backend (+2 more)

Intentionally-unsupported (gateway-owned, refused in `extra_args`): --api-key

Scope disposition: Blazar integrates the SGLang server launch path, not a separate router / disaggregated-PD tier. Router and disagg flags the installed server advertises are passthrough-reachable through extra_args where the manifest gate allows; a dedicated router tier is a documented scope decision, not a wiring gap.

## v0.9.5 (mistralrs) - 91 flags probed

**certified: 1 | first-class: 32 | passthrough: 58 | intentionally-unsupported: 0 | not-applicable: 0 | unverified: 0**

Registry lane `mistralrs` pins 39 first-class flags (compile + spawn-time emission).

### Capability matrix

| Family | Evidence | Flags | Notes |
|---|---|---|---|
| batching_scheduling | first-class | `-np`, `--max-batch-size`, `--max-seqs`, `--max-num-batched-tokens` | paged-attention admission knobs derived from capacity |
| kv_cache_quantization | first-class | `--pa-cache-type` | paged-attn cache dtype knob |
| prefix_caching | first-class | `--prefix-cache-n` | prefix retention count knob |
| speculative_decoding | first-class | `--mtp`, `--mtp-model`, `--mtp-n-predict` | MTP draft surface is wired through the tuning knobs |
| multi_gpu_parallelism | unsupported | - | v0.9.5 advertises no TP/PP surface; single-device lane |
| memory_offload | first-class | `--device-layers`, `--pa-memory-mb`, `--encoder-cache-memory-mb` | layer split and memory fraction knobs |
| structured_output | probed | - | request-side grammar only; no launch flag on this lane |
| lora | live-certified | `--enable-lora` | live child --isq/--enable-lora qwen3-1.7b run; adapter caps are knobs ([proof](bench-artifacts/20261010-engine-utilization/live_spawn_proofs.md)) |
| multimodal | first-class | `--mmproj`, `--max-image-length` | vision projector routing is first-class |
| cancellation_streaming | first-class | - | gateway-owned stream and cancel |

### Certified against a live engine

- `--isq` [`bench-artifacts/20261010-engine-utilization/live_spawn_proofs.md`](bench-artifacts/20261010-engine-utilization/live_spawn_proofs.md) - live child --isq q8_0 (qwen3-1.7b temp-0 200 OK); admission stays checkpoint-based

### Passthrough families

- **agent family (their product)** (19): --agent, --agent-permission, --code-exec-python, --code-exec-timeout, --code-exec-workdir, --enable-code-execution (+13 more)
- **uncategorized long-tail** (17): --arch, --cpu, --disable-request-id-header, --dtype, --format, --gqa (+11 more)
- **serving detail** (12): --hf-cache, --kv-cache, --legacy-lora, --legacy-lora-order, --lora, --max-edge (+6 more)
- **observability** (5): --access-log-format, --access-log-health, --log, --topology, --verbose...
- **ISQ/quantize tooling** (3): --from-uqff, --quant, --quantized-file
- **gateway-owned** (2): --token-source, --tokenizer

## vllm-0.34.0 (vllm) - 428 flags probed

**certified: 4 | first-class: 6 | passthrough: 418 | intentionally-unsupported: 0 | not-applicable: 0 | unverified: 0**

Registry lane `vllm` pins 10 first-class flags (compile + spawn-time emission).

### Capability matrix

| Family | Evidence | Flags | Notes |
|---|---|---|---|
| batching_scheduling | first-class | `--max-num-seqs` | slots pin caps admissions; absent rides vllm's continuous batching |
| kv_cache_quantization | probed | - | --kv-cache-dtype is in the supported grammar; the compiler does not pin it yet |
| prefix_caching | probed | - | vllm enables prefix caching by default upstream; no explicit pin is emitted |
| speculative_decoding | unsupported | - | spec decode surface not wired on this lane yet |
| multi_gpu_parallelism | probed | - | upstream tensor parallelism exists; no first-class knob on this lane yet |
| memory_offload | first-class | `--gpu-memory-utilization`, `--kv-transfer-config` | memory fraction always emitted; persistent KV offload is opt-in with a tiered store |
| structured_output | probed | - | guided decoding is request-side on this lane |
| lora | probed | - | upstream LoRA surface not wired on this lane yet |
| multimodal | probed | - | HF vision towers route upstream; no launch knob is pinned |
| cancellation_streaming | first-class | - | gateway-owned stream and cancel; /health readiness probe |

### Certified against a live engine

- `--enable-prompt-tokens-details` [`bench-artifacts/20261120-kokoro-vllm-cert/README.md`](bench-artifacts/20261120-kokoro-vllm-cert/README.md) - E2E lane cert 2026-11-20 (isolated debug daemon, qwen2.5-coder-0.5b BF16, explain source=pin): usage.prompt_tokens_details.cached_tokens present in gateway response
- `--gpu-memory-utilization` [`bench-artifacts/20261120-kokoro-vllm-cert/README.md`](bench-artifacts/20261120-kokoro-vllm-cert/README.md) - E2E lane cert 2026-11-20 (isolated debug daemon, qwen2.5-coder-0.5b BF16, explain source=pin): default 0.85 in child argv; 8 GB lane served + idle reaped to 16 MiB
- `--max-model-len` [`bench-artifacts/20261120-kokoro-vllm-cert/README.md`](bench-artifacts/20261120-kokoro-vllm-cert/README.md) - E2E lane cert 2026-11-20 (isolated debug daemon, qwen2.5-coder-0.5b BF16, explain source=pin): clamped 32768->16384 per ctx policy, child argv observed
- `--served-model-name` [`bench-artifacts/20261120-kokoro-vllm-cert/README.md`](bench-artifacts/20261120-kokoro-vllm-cert/README.md) - E2E lane cert 2026-11-20 (isolated debug daemon, qwen2.5-coder-0.5b BF16, explain source=pin): child argv observed with sanitized name; request model string resolved

### Passthrough families

- **uncategorized long-tail** (418): --additional-config, --aggregate-engine-logging, --all2all-backend, --allow-credentials, --allow-deprecated-quantization, --allowed-headers (+412 more)

## Cross-engine KV interoperability

Pair-wise certification verdicts for KV cache interchange (source -> target). `live-certified` cites a receipt from real engines proving token reuse; `codec-shipped` means the codec and its byte-identity unit proofs landed but the live pair certificate is pending; `unavailable` pairs have no codec lane. Cross-engine admission stays gated on `[kv.translation] enabled` (default off) - the matrix tracks promotion evidence, it never presumes compatibility.

| Pair | Verdict | Notes |
|---|---|---|
| llama.cpp -> llama.cpp | exact | slot-save bank round trip - the native restore path |
| vllm -> vllm | exact | fs offload tier round trip - content-addressed reload |
| sglang -> sglang | exact | hicache L3 file tier round trip |
| mlx -> mlx | exact | mlx-lm prompt-cache file round trip |
| llama.cpp -> vllm | live-certified | gateway chat through the imported prefix reported cached_tokens > 0 on live engines ([proof](docs/research/2026-11-19-llamacpp-vllm-kv-import-data-plane.md)) |
| vllm -> llama.cpp | live-certified | tier -> bank -> native slot restore proven live (qwen2.5-coder-0.5b pair: 24 blocks x 64 tokens imported; slot restore of 1536 tokens logged before first use) ([proof](bench-artifacts/2026-11-21-kv-codec-live-certs/README.md)) |
| mlx -> vllm | codec-shipped | mlx-lm export codec landed; live pair certificate needs a Metal host for the mlx lane |
| vllm -> mlx | codec-shipped | staged preload landed; live pair certificate needs a Metal host for the mlx lane |
| sglang -> vllm | codec-shipped | hicache L3 page codec landed; live pair certificate (both engines, proven reuse) pending - the hicache host-tier guard needs a RAM class with ~15 GiB headroom (guard refusal verified live on the 13.7 GiB cert box) |
| vllm -> sglang | codec-shipped | hicache L3 page codec landed; live pair certificate (both engines, proven reuse) pending - the hicache host-tier guard needs a RAM class with ~15 GiB headroom (guard refusal verified live on the 13.7 GiB cert box) |
| llama.cpp -> mlx | unavailable | no codec lane exists for this pair today |
| mlx -> llama.cpp | unavailable | no codec lane exists for this pair today |
| llama.cpp -> sglang | unavailable | no codec lane exists for this pair today |
| sglang -> llama.cpp | unavailable | no codec lane exists for this pair today |
| sglang -> mlx | unavailable | no codec lane exists for this pair today |
| mlx -> sglang | unavailable | no codec lane exists for this pair today |

Fixtures refreshed from a live store via `--refresh`; they carry tag/kind/flags only (no machine paths). Dispositions are derived, never stored.
