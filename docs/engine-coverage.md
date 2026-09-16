# Engine flag coverage — how much of each backend Pallama wires

Reproducible audit: `python3 scripts/engine_coverage.py` (reads engine
manifests from the store, greps the profile compiler + argv translators,
buckets the rest by keyword family). Numbers below from the 2026-09-16
run (llama b10980-cuda, sglang 0.5.19, mistral.rs v0.9.3).

## The two-layer model

Pallama does not chase every CLI flag with a config knob — it splits
each engine's surface into two layers with different guarantees:

1. **First-class knobs** — validated at config-parse time (choices,
   ranges, footguns), emitted flag-gated against the installed engine's
   manifest (old engines warn-and-skip instead of crash), documented in
   `config defaults` / `config edit <model>` hints, and pinned by unit
   tests that assert the exact child argv.
2. **Universal passthrough** — every other flag in each engine's
   manifest stays reachable through per-model `extra_args`:
   llamacpp passes tokens through verbatim (the child validates at
   boot); mistralrs is strict-manifest-gated like sglang (a passthrough
   flag must exist in the probed manifest — typo-proof — and must not
   collide with the twelve flags the supervisor/VRAM ladder own
   (`--host`, `--port`, `-m`, `-f`, `--mmproj`, `--max-model-len`,
   `-np`, `--max-seqs`, `--max-num-batched-tokens`, `--paged-attn`,
   `--pa-memory-fraction`, `--no-ui`); both violations are hard errors
   with manifest teaching); sglang same contract with its reserved
   seven (`--host`, `--port`, `--model-path`, `--served-model-name`,
   `--context-length`, `--mem-fraction-static`, `--cpu-offload-gb`),
   which is a hard error with a pointer to the config knob that owns
   the value.
   Live caveat: mistral.rs v0.9.3's `--isq` in-situ quantization passes
   through cleanly (argv verified, ~6 GiB conversion working set) but
   the engine answers `model_error` on every request for both GGUF and
   safetensors sources and every ISQ level tried (`q4k`, `q8_0`) on the
   cuda130-sm89 build — upstream status, not a routing gap.

Nothing is unreachable: a flag is either a knob or a passthrough; the
only blocked flags are the reserved seven, whose values Pallama
computes (loopback bind, auth, VRAM fit ladder).

## Numbers (flags probed from live manifests)

| engine | flags probed | wired first-class | passthrough long-tail |
|---|---|---|---|
| llama-server b10980 | 328 | 174 (53%) | 154 |
| sglang 0.5.19 | 541 | 74 (14%) | 467 |
| mistral.rs v0.9.3 | 93 | 41 (44%) | 52 |

The sglang percentage is low because its manifest is huge and aimed at
fleet serving; the long-tail buckets (from the audit script) are:

- **llama-server**: draft/speculative detail 37 (EAGLE3 pair IS wired;
  the rest are per-draft-model cache/affinity tuning), per-draft CPU
  affinity 3, per-request sampling 14 (gateway sends these per request,
  not at boot), observability 14, transport/CORS/SSL 17 (the gateway
  owns the public surface), multimodal extras 3, misc serving 37.
- **sglang**: scheduler/memory detail 125 (defaults + VRAM ladder cover
  the single-GPU path; fleet knobs like PD-disaggregation 10,
  multi-node 14, MoE/deepep 13, RL/rollout 13 are multi-box products),
  observability 26, tokenizer/sampler internals 20, LoRA advanced 10
  (the serving switch + capacity + adapter lane ARE wired), multimodal/
  ASR 8.
- **mistral.rs**: agent family 18 (search/shell/code-exec — their
  agent harness, not a model-server surface; Pallama's gateway owns
  agentic behavior), ISQ/quantize tooling 5 (offline CLI operations),
  serving detail 12, observability 4.

## What gets promoted to a first-class knob

A flag becomes a knob when it (a) matters on one box with one or two
GPUs (the Pallama product), (b) benefits from validation beyond what
the child checks at boot (choice sets, footguns like mistral.rs's
LoRA-limits-need-serving boot error), or (c) interacts with Pallama's
own machinery (VRAM ladder, determinism pinning, warm-peg, LoRA
registry). Everything else stays a passthrough by design — a knob per
flag would be a config surface nobody can read.

## Proof artifacts

- argv pins: `unit__sglang__*`, `unit__mistralrs_knobs__*`,
  `unit__llama_server_knobs__*` (profile.rs), mistralrs translator
  tests (`tests/mistralrs.rs`), and the bidirectional lockstep test
  `unit__mistralrs_tuning_flags__lockstep_with_compile_emission`.
- live e2e: all three engines spawned with full knob sets, child
  `/proc/<pid>/cmdline` asserted flag-by-flag (2026-09-16 wave, logged
  in MEMORY.md).
- benchmarks: geokit `bench_matrix_compare` artifacts (quality +
  throughput per engine, saved 2026-09-16 run #5).
