# Engine capability coverage

Disposition of every engine-advertised flag against Blazar's
emission-truth registry.

- fixtures: `scripts/manifest_flag_fixtures.json` (engine manifest
  snapshots, refreshed from a live store via `--refresh`)
- registry: `scripts/first_class_flags.json` (live emission;
  regenerate via `cargo run -p blazar-runtime --example
  knob_registry`)
- receipts: `scripts/certification_receipts.json` (live-engine runs
  that upgrade first-class flags to certified)
- gate: `python3 scripts/engine_coverage.py --check` fails CI when
  the doc drifts from any input. No timestamps — the bytes are
  the contract.

| Disposition | Meaning |
|---|---|
| certified | first-class + live-engine receipt (behavior proven) |
| first-class | Blazar pins it from a config knob or derivation |
| passthrough | reachable via `extra_args`, no first-class knob |
| intentionally-unsupported | gateway-owned; `extra_args` refuses it |
| not-applicable | not a capability Blazar could integrate |
| unverified | no verified reach through Blazar on this lane |

Policy: promotion from passthrough to a first-class knob requires
workload evidence — a receipt linking the flag to a live-engine
run. Emission tests gate the *how*; receipts gate the *why*.

## 2023.11.14-2 (piper) — 16 flags probed

**certified: 0 · first-class: 9 · passthrough: 0 · intentionally-unsupported: 0 · not-applicable: 7 · unverified: 0**

Registry lane `piper` pins 10 first-class flags (compile + spawn-time emission).

### Not applicable (deliberate, per-flag reasons)

- --config: config-file input competes with Blazar's composed argv
- --debug: interactive CLI ergonomics, not a synthesis capability
- --help: interactive CLI ergonomics, not a synthesis capability
- --json-input: stdin JSON request surface; Blazar feeds plain text lines
- --output_dir: directory batch output; Blazar always names explicit files (--output_file / --output_raw)
- --quiet: long-form alias of the first-class -q the lane emits
- --tashkeel_model: Arabic diacritization; needs a separate model asset and an Arabic voice, and the lane has no extra_args surface

## b11539-cuda (llamacpp) — 329 flags probed

**certified: 2 · first-class: 69 · passthrough: 258 · intentionally-unsupported: 0 · not-applicable: 0 · unverified: 0**

Registry lane `llamacpp` pins 75 first-class flags (compile + spawn-time emission).

### Certified against a live engine

- `--cache-type-k` [`bench-artifacts/20261005-kv-quality/benchmark.md`](bench-artifacts/20261005-kv-quality/benchmark.md) — q4_0 KV quality-free at 2048 and 32k ctx (perplexity parity in bars)
- `--cache-type-v` [`bench-artifacts/20261005-kv-quality/benchmark.md`](bench-artifacts/20261005-kv-quality/benchmark.md) — q4_0 KV quality-free at 2048 and 32k ctx (perplexity parity in bars)

### Passthrough families

- **serving detail** (72): --cache-idle-slots, --cache-list, --cache-prompt, --cache-reuse, --check-tensors, --completion-bash (+66 more)
- **draft/speculative detail** (64): --cache-type-k-draft, --cache-type-v-draft, --cpu-mask-batch-draft, --cpu-mask-draft, --cpu-moe-draft, --cpu-range-draft (+58 more)
- **uncategorized long-tail** (49): --adaptive-decay, --adaptive-target, --agent, --cpu-moe, --cpu-strict, --docker-repo (+43 more)
- **per-request sampling** (24): --backend-sampling, --chat-template-file, --dry-sequence-breaker, --dynatemp-exp, --dynatemp-range, --grammar (+18 more)
- **gateway-owned transport** (18): --api-key, --api-key-file, --api-prefix, --cors-credentials, --cors-headers, --cors-methods (+12 more)
- **observability** (13): --log-colors, --log-disable, --log-file, --log-jsonl, --log-prefix, --log-prompts-dir (+7 more)
- **multimodal** (10): --image-max-tokens, --mmproj, --mmproj-auto, --mmproj-device, --mmproj-offload, --mmproj-url (+4 more)
- **CPU affinity** (6): --cpu-mask, --cpu-mask-batch, --cpu-range, --cpu-range-batch, --threads-batch, --threads-http
- **LoRA advanced** (1): --lora
- **rpc/multi-node** (1): --rpc

## b5454 (whisper) — 54 flags probed

**certified: 1 · first-class: 14 · passthrough: 39 · intentionally-unsupported: 0 · not-applicable: 0 · unverified: 0**

Registry lane `whisper` pins 15 first-class flags (compile + spawn-time emission).

### Certified against a live engine

- `--beam-size` [`bench-artifacts/20261010-engine-utilization/live_spawn_proofs.md`](bench-artifacts/20261010-engine-utilization/live_spawn_proofs.md) — beam-5 child; piper-TTS wav transcribed exactly

### Passthrough families

- **uncategorized long-tail** (24): --audio-ctx, --convert, --debug-mode, --device, --dtw, --flash-attn (+18 more)
- **segmentation/timing** (8): --duration, --max-context, --max-len, --no-timestamps, --offset-n, --offset-t (+2 more)
- **transcription/translation** (6): --carry-initial-prompt, --detect-language, --language, --no-language-probabilities, --prompt, --translate
- **decoding quality** (1): --no-fallback

## master-951-f89d9b1 (sdcpp) — 158 flags probed

**certified: 2 · first-class: 39 · passthrough: 112 · intentionally-unsupported: 5 · not-applicable: 0 · unverified: 0**

Registry lane `sdcpp` pins 41 first-class flags (compile + spawn-time emission).

### Certified against a live engine

- `--hires-upscalers-dir` [`bench-artifacts/20261010-engine-utilization/live_spawn_proofs.md`](bench-artifacts/20261010-engine-utilization/live_spawn_proofs.md) — RealESRGAN_x4plus end-to-end: 512x512 generation upscaled to 2048x2048
- `--upscale-model` [`bench-artifacts/20261010-engine-utilization/live_spawn_proofs.md`](bench-artifacts/20261010-engine-utilization/live_spawn_proofs.md) — RealESRGAN_x4plus end-to-end: 512x512 generation upscaled to 2048x2048

### Passthrough families

- **uncategorized long-tail** (56): --ad-model, --attn-scale, --audio, --auto-fit, --circular, --circularx (+50 more)
- **sampling/generation** (24): --batch-count, --control-strength, --end-img, --extra-sample-args, --guidance, --high-noise-cfg-scale (+18 more)
- **input/output files** (18): --ad-negative-prompt, --ad-prompt, --control-image, --control-video, --disable-prefetch, --increase-ref-index (+12 more)
- **component set (blazar-owned)** (11): --audio-vae, --clip-on-cpu, --clip-skip, --force-sdxl-vae-conv-scale, --qwen2vl, --vae-conv-direct (+5 more)
- **memory/backend placement** (3): --control-net-cpu, --diffusion-fa, --list-devices

Intentionally-unsupported (gateway-owned, refused in `extra_args`): --clip_g, --llm_vision, --model, --qwen2vl_vision, --threads

Scope disposition: knob emission is pinned by the registry; per-component behavior against real diffusion models is certification-pending (tracked by the benchmark harness, not this audit).

## mlx-0.32.0 (mlx) — 27 flags probed

**certified: 1 · first-class: 17 · passthrough: 9 · intentionally-unsupported: 0 · not-applicable: 0 · unverified: 0**

Registry lane `mlx` pins 18 first-class flags (compile + spawn-time emission).

### Certified against a live engine

- `--kv-bits` [`bench-artifacts/20261010-engine-utilization/mlx_kv_bits_ab.md`](bench-artifacts/20261010-engine-utilization/mlx_kv_bits_ab.md) — A/B on qwen2.5-0.5b: 1.9x slower decode, 108MiB saved - memory lever, opt-in

### Passthrough families

- **uncategorized long-tail** (9): --allowed-origins, --help, --log-level, --max-tokens, --min-p, --pipeline (+3 more)

## sglang-0.5.21 (sglang) — 556 flags probed

**certified: 3 · first-class: 69 · passthrough: 483 · intentionally-unsupported: 1 · not-applicable: 0 · unverified: 0**

Registry lane `sglang` pins 72 first-class flags (compile + spawn-time emission).

### Certified against a live engine

- `--enable-lora` [`docs/research/2026-10-09-sglang-lora-one-child.md`](docs/research/2026-10-09-sglang-lora-one-child.md) — one-child colon routing + hot-attach via /api/lora/sync on the same pid
- `--lora-paths` [`docs/research/2026-10-09-sglang-lora-one-child.md`](docs/research/2026-10-09-sglang-lora-one-child.md) — one-child colon routing + hot-attach via /api/lora/sync on the same pid
- `--speculative-algorithm` [`bench-artifacts/20261009-sglang-spec/spec_race_lane.md`](bench-artifacts/20261009-sglang-spec/spec_race_lane.md) — live sglang-0.5.21 spec battery: ngram exercised via --speculative-algorithm; +7% on repetitive search prompts, negative on diverse traffic - gain is workload-dependent, kept opt-in (raw cells: live_battery.json, bench-archive release)

### Passthrough families

- **uncategorized long-tail** (224): --abort-on-priority-when-disabled, --api-, --bf16-gemm-backend, --boundary-reduction, --c128-page-size, --chat-template (+218 more)
- **scheduler/memory detail** (119): --attention-, --attention-context-parallel-size, --cuda-graph-, --cuda-graph-backend-decode, --cuda-graph-config, --cuda-graph-max-bs-prefill (+113 more)
- **per-request/gateway-owned** (26): --admin-api-key, --allow-auto-truncate, --constrained-json-disable-any-whitespace, --constrained-json-max-whitespace-cnt, --constrained-json-whitespace-pattern, --enable-ssl-refresh (+20 more)
- **observability** (25): --bucket-e2e-request-latency, --bucket-inter-token-latency, --bucket-time-to-first-token, --decode-log-interval, --enable-forward-pass-metrics, --enable-metrics-for-all-schedulers (+19 more)
- **tokenizer/sampler internals** (17): --disable-tokenizer-batch-decode, --enable-tokenizer-batch-encode, --generation-tokens-buckets, --kt-max-deferred-experts-per-token, --mlx-enable-sampling, --num-reserved-decode-tokens (+11 more)
- **multi-node/parallelism** (15): --attn-cp-size, --base-gpu-id, --disable-custom-all-reduce, --dist-init-addr, --dwdp-size, --enable-nccl-nvls (+9 more)
- **MoE backends** (13): --deepep-config, --deepep-dispatcher-output-dtype, --deepep-mode, --deepep-v2-mode, --disable-flashinfer-cutlass-moe-fp4-allgather, --enable-fused-moe-sum-all-reduce (+7 more)
- **disaggregated/PD serving** (13): --disaggregation-bootstrap-port, --disaggregation-decode-enable-offload-kvcache, --disaggregation-decode-enable-radix-cache, --disaggregation-decode-extra-slots, --disaggregation-decode-host-receive-threshold, --disaggregation-decode-polling-interval (+7 more)
- **RL/rollout tooling** (12): --disable-overlap-schedule, --enable-lora-overlap-loading, --enable-single-batch-overlap, --encoder-register-urls, --encoder-urls, --hicache-storage-prefetch-policy (+6 more)
- **LoRA advanced** (11): --experts-shared-outer-loras, --lora-drain-wait-threshold, --lora-strict-loading, --lora-target-modules, --lora-use-virtual-experts, --max-loaded-loras (+5 more)
- **multimodal/ASR** (8): --allowed-media-domains, --asr-max-buffer-seconds, --asr-max-concurrent-sessions, --disable-fast-image-processor, --enable-multimodal, --image-processor-backend (+2 more)

Intentionally-unsupported (gateway-owned, refused in `extra_args`): --api-key

Scope disposition: Blazar integrates the SGLang server launch path, not a separate router / disaggregated-PD tier. Router and disagg flags the installed server advertises are passthrough-reachable through extra_args where the manifest gate allows; a dedicated router tier is a documented scope decision, not a wiring gap.

## v0.9.4 (mistralrs) — 93 flags probed

**certified: 1 · first-class: 34 · passthrough: 57 · intentionally-unsupported: 1 · not-applicable: 0 · unverified: 0**

Registry lane `mistralrs` pins 39 first-class flags (compile + spawn-time emission).

### Certified against a live engine

- `--isq` [`bench-artifacts/20261010-engine-utilization/live_spawn_proofs.md`](bench-artifacts/20261010-engine-utilization/live_spawn_proofs.md) — live child --isq q8_0 (qwen3-1.7b temp-0 200 OK); admission stays checkpoint-based

### Passthrough families

- **agent family (their product)** (19): --agent, --agent-permission, --code-exec-python, --code-exec-timeout, --code-exec-workdir, --enable-code-execution (+13 more)
- **uncategorized long-tail** (17): --arch, --cpu, --disable-request-id-header, --dtype, --format, --gqa (+11 more)
- **serving detail** (12): --hf-cache, --legacy-lora, --legacy-lora-order, --lora, --max-edge, --max-seq-len (+6 more)
- **observability** (4): --access-log-format, --access-log-health, --log, --topology
- **ISQ/quantize tooling** (3): --from-uqff, --quant, --quantized-file
- **gateway-owned** (2): --token-source, --tokenizer

Intentionally-unsupported (gateway-owned, refused in `extra_args`): --pa-memory-fraction

Fixtures refreshed from a live store via `--refresh`; they carry tag/kind/flags only (no machine paths). Dispositions are derived, never stored.
