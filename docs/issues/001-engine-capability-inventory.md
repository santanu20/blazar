Title: Complete engine capability inventory for all engine kinds

Priority: P0 | Wave: 1 | Backlog: B01 | Effort: M | Depends: -

## Problem

Blazar ships ten engine kinds (EngineKind in crates/blazar-core/src/engine_kind.rs:
LlamaCpp, MistralRs, Sglang, Vllm, SdCpp, Mlx, Whisper, Kokoro, Piper, OpenAiHost).
Capability truth is currently split across four surfaces:

- scripts/first_class_flags.json: emission truth per lane (what a spawned engine actually accepted)
- scripts/certification_receipts.json: live certification evidence
- crates/blazar-core/src/knob_registry.rs and crates/blazar-runtime/src/knob_registry.rs: typed knobs
- docs/engine-coverage.md: rendered per-lane report (probed / first-class / passthrough / unsupported counts)

Two lanes have no registry entry at all: scripts/engine_coverage.py keeps
LANES_WITHOUT_REGISTRY = {"kokoro", "vllm"}. There is no single per-engine
record that states, per capability, whether it is probed, first-class,
live-certified, passthrough, or unsupported.

## Approach

Extend the existing registry rather than adding a parallel structure:

1. Add a per-capability evidence status to the registry data model:
   Probed | FirstClass | LiveCertified | Passthrough | Unsupported.
2. Classify upstream features per lane into: core serving (first-class
   config), performance (typed knobs + bench validation), experimental
   (explicit status), debugging (operator controls), engine-internal
   (passthrough where safe), incompatible with Blazar architecture
   (explicitly unsupported).
3. Build full records first for the auto-manageable capability families:
   batching/scheduling, KV-cache quantization, prefix caching, speculative
   decoding, multi-GPU parallelism, memory offload, structured output, LoRA,
   multimodal processing, cancellation/streaming.
4. CI gate in scripts/engine_coverage.py: every capability carries an explicit
   disposition; every LiveCertified claim has a receipt id in
   scripts/certification_receipts.json; an engine version change produces a
   capability-difference report.

## Acceptance criteria

- [ ] All ten EngineKind lanes have complete registry records with evidence statuses
- [ ] kokoro and vllm removed from LANES_WITHOUT_REGISTRY
- [ ] No capability is represented LiveCertified without a receipt id
- [ ] docs/engine-coverage.md has a section per lane, generated from the registry
- [ ] CI fails on a missing disposition or an unreceipted certification
- [ ] Engine version bump produces a capability-difference report
- [ ] Missing or broken capabilities surface actionable diagnostics

## Files

scripts/first_class_flags.json, scripts/engine_coverage.py,
scripts/certification_receipts.json, crates/blazar-core/src/knob_registry.rs,
crates/blazar-runtime/src/knob_registry.rs, docs/engine-coverage.md

## Out of scope

Benchmark lane execution (003), the execution planner (005).
