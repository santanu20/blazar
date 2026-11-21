//! Per-engine capability dispositions: one evidence-backed record for
//! every capability family Blazar could auto-manage on every lane.
//!
//! The flag registry ([`crate::knob_registry`]) answers "which flags does
//! Blazar emit first-class"; this module answers the coarser question the
//! execution planner and the coverage gate need: per capability family
//! (batching, KV quantization, prefix caching, speculation, parallelism,
//! offload, structured output, `LoRA`, multimodal, cancellation/streaming)
//! - what is the strongest VERIFIED statement about the lane, and what
//!   evidence backs it. A family is a unit of scheduling/product decisions;
//!   flags are units of argv. Both views are needed and they deliberately
//!   cross-check: every flag a disposition cites must be in the registry
//!   lane (the Python coverage gate enforces it against the generated
//!   JSON, so a cited-but-unemitted flag fails CI).
//!
//! Evidence ladder (monotone - each rung implies the ones below):
//! - `Probed`:        the engine advertises the capability in its
//!   manifest/help surface; Blazar does not pin it.
//! - `FirstClass`:    Blazar emits the controlling flags from config
//!   knobs or derivations (registry-backed).
//! - `LiveCertified`: a live-engine run (receipt artifact in-repo)
//!   exercised the capability end to end. Requires a receipt path; the
//!   gate fails without the artifact.
//! - `Passthrough`:   reachable only via `extra_args`, no knob. Honest
//!   middle ground between Probed and Unsupported.
//! - `Unsupported`:   not applicable to the lane, or incompatible with
//!   Blazar's architecture on it (the note says which).
//!
//! The committed `scripts/first_class_flags.json` "capabilities" section
//! is generated from this module by the `knob_registry` example; the
//! auditor (`scripts/engine_coverage.py --check`) fails CI when a lane
//! is missing a canonical family, a `LiveCertified` row lacks its
//! receipt artifact, or a cited flag is not in the registry lane.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::engine_kind::EngineKind;

/// Evidence status of one capability on one lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityEvidence {
    /// Advertised by the engine manifest; no Blazar knob yet.
    Probed,
    /// Blazar pins the controlling flags (emission-truth registry).
    FirstClass,
    /// Exercised end to end against a live engine; receipt required.
    LiveCertified,
    /// Reachable only via `extra_args`; no first-class knob.
    Passthrough,
    /// Not applicable, or incompatible with Blazar on this lane.
    Unsupported,
}

/// One capability family's disposition on one lane.
#[derive(Debug, Clone, Serialize)]
pub struct CapabilityDisposition {
    /// Canonical family (or a lane-specific extra, e.g. `speech_synthesis`
    /// on the TTS lanes) this row describes.
    pub family: &'static str,
    /// Strongest verified evidence for the capability on this lane.
    pub evidence: CapabilityEvidence,
    /// Registry flags that control this family (must be emitted by the
    /// lane; empty when the capability is gateway-owned or N/A).
    pub flags: &'static [&'static str],
    /// Repo-relative artifact proving a `LiveCertified` row; `None`
    /// otherwise. Validated to exist by the coverage gate.
    pub receipt: Option<&'static str>,
    /// Why this disposition holds - the one-line audit trail.
    pub note: &'static str,
}

/// The ten auto-manageable capability families every lane must dispose
/// of (TTS/audio lanes may append lane-specific extras after these).
pub const CANONICAL_FAMILIES: &[&str] = &[
    "batching_scheduling",
    "kv_cache_quantization",
    "prefix_caching",
    "speculative_decoding",
    "multi_gpu_parallelism",
    "memory_offload",
    "structured_output",
    "lora",
    "multimodal",
    "cancellation_streaming",
];

const fn row(
    family: &'static str,
    evidence: CapabilityEvidence,
    flags: &'static [&'static str],
    receipt: Option<&'static str>,
    note: &'static str,
) -> CapabilityDisposition {
    CapabilityDisposition {
        family,
        evidence,
        flags,
        receipt,
        note,
    }
}

use CapabilityEvidence::{FirstClass, LiveCertified, Passthrough, Probed, Unsupported};

const LLAMACPP: &[CapabilityDisposition] = &[
    row(
        "batching_scheduling",
        FirstClass,
        &["-np", "--batch-size", "--ubatch-size", "--no-cont-batching"],
        None,
        "slots=0 capacity-aware sizing, adaptive_slots governor and the decode gate manage batching",
    ),
    row(
        "kv_cache_quantization",
        LiveCertified,
        &["--cache-type-k", "--cache-type-v"],
        Some("bench-artifacts/20261111-fp8-kv-quality/README.md"),
        "q4_0 KV quality-free at 2048/32k ctx; f8_e4m3 pin-only above 16k - ladder policy is receipt-driven",
    ),
    row(
        "prefix_caching",
        FirstClass,
        &["--no-cache-prompt"],
        None,
        "prompt cache default-on; cache_hit_rate hint feeds the spec/cache governors",
    ),
    row(
        "speculative_decoding",
        FirstClass,
        &["--spec-default", "--spec-draft-n-min", "--spec-draft-p-min"],
        None,
        "spec=auto family selection, ngram fallback and the adaptive spec governor",
    ),
    row(
        "multi_gpu_parallelism",
        Passthrough,
        &[],
        None,
        "rpc/multi-node surface is extra_args-only; single-box layer splits auto-derive",
    ),
    row(
        "memory_offload",
        FirstClass,
        &["--no-kv-offload", "--op-offload", "--cache-ram"],
        None,
        "KV/operator offload and cache-ram budget are config knobs",
    ),
    row(
        "structured_output",
        FirstClass,
        &["--grammar", "--json-schema", "--json-schema-file"],
        None,
        "GGUF grammars and JSON-schema pins ride the launch line",
    ),
    row(
        "lora",
        FirstClass,
        &["--lora-scaled", "--lora-init-without-apply"],
        None,
        "adapter scaling and no-apply load are first-class knobs",
    ),
    row(
        "multimodal",
        FirstClass,
        &["--media-path", "--no-mmproj-auto", "--no-mmproj-offload"],
        None,
        "mmproj auto-detect and offload knobs",
    ),
    row(
        "cancellation_streaming",
        FirstClass,
        &[],
        None,
        "gateway-owned SSE stream and cancel path",
    ),
];

const MISTRALRS: &[CapabilityDisposition] = &[
    row(
        "batching_scheduling",
        FirstClass,
        &[
            "-np",
            "--max-batch-size",
            "--max-seqs",
            "--max-num-batched-tokens",
        ],
        None,
        "paged-attention admission knobs derived from capacity",
    ),
    row(
        "kv_cache_quantization",
        FirstClass,
        &["--pa-cache-type"],
        None,
        "paged-attn cache dtype knob",
    ),
    row(
        "prefix_caching",
        FirstClass,
        &["--prefix-cache-n"],
        None,
        "prefix retention count knob",
    ),
    row(
        "speculative_decoding",
        FirstClass,
        &["--mtp", "--mtp-model", "--mtp-n-predict"],
        None,
        "MTP draft surface is wired through the tuning knobs",
    ),
    row(
        "multi_gpu_parallelism",
        Unsupported,
        &[],
        None,
        "v0.9.5 advertises no TP/PP surface; single-device lane",
    ),
    row(
        "memory_offload",
        FirstClass,
        &[
            "--device-layers",
            "--pa-memory-mb",
            "--encoder-cache-memory-mb",
        ],
        None,
        "layer split and memory fraction knobs",
    ),
    row(
        "structured_output",
        Probed,
        &[],
        None,
        "request-side grammar only; no launch flag on this lane",
    ),
    row(
        "lora",
        LiveCertified,
        &["--enable-lora"],
        Some("bench-artifacts/20261010-engine-utilization/live_spawn_proofs.md"),
        "live child --isq/--enable-lora qwen3-1.7b run; adapter caps are knobs",
    ),
    row(
        "multimodal",
        FirstClass,
        &["--mmproj", "--max-image-length"],
        None,
        "vision projector routing is first-class",
    ),
    row(
        "cancellation_streaming",
        FirstClass,
        &[],
        None,
        "gateway-owned stream and cancel",
    ),
];

const SGLANG: &[CapabilityDisposition] = &[
    row(
        "batching_scheduling",
        FirstClass,
        &[
            "--schedule-policy",
            "--schedule-conservativeness",
            "--max-total-tokens",
            "--max-prefill-tokens",
        ],
        None,
        "scheduler policy and token budget knobs",
    ),
    row(
        "kv_cache_quantization",
        FirstClass,
        &["--kv-cache-dtype"],
        None,
        "KV dtype knob on the paged cache",
    ),
    row(
        "prefix_caching",
        FirstClass,
        &[
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            "--hicache-size",
            "--enable-session-radix-cache",
        ],
        None,
        "radix + hierarchical cache knobs; hicache L3 codec translates across engines",
    ),
    row(
        "speculative_decoding",
        LiveCertified,
        &["--speculative-algorithm", "--speculative-num-draft-tokens"],
        Some("bench-artifacts/20261009-sglang-spec/spec_race_lane.md"),
        "live spec battery: +7% on repetitive traffic, negative on diverse - kept opt-in",
    ),
    row(
        "multi_gpu_parallelism",
        FirstClass,
        &["--tp-size", "--dp-size", "--pp-size", "--ep-size"],
        None,
        "all four parallel axes are sized by the multi-GPU planner",
    ),
    row(
        "memory_offload",
        FirstClass,
        &["--cpu-offload-gb", "--mem-fraction-static"],
        None,
        "CPU offload and static memory fraction knobs",
    ),
    row(
        "structured_output",
        Probed,
        &[],
        None,
        "xgrammar rides requests; the launch-side grammar backend is not pinned",
    ),
    row(
        "lora",
        LiveCertified,
        &["--enable-lora", "--lora-paths"],
        Some("docs/research/2026-10-09-sglang-lora-one-child.md"),
        "one-child colon routing + hot-attach on the same pid",
    ),
    row(
        "multimodal",
        FirstClass,
        &["--mm-attention-backend"],
        None,
        "multimodal attention backend knob",
    ),
    row(
        "cancellation_streaming",
        FirstClass,
        &[],
        None,
        "gateway-owned stream and cancel",
    ),
];

const VLLM: &[CapabilityDisposition] = &[
    row(
        "batching_scheduling",
        FirstClass,
        &["--max-num-seqs"],
        None,
        "slots pin caps admissions; absent rides vllm's continuous batching",
    ),
    row(
        "kv_cache_quantization",
        Probed,
        &[],
        None,
        "--kv-cache-dtype is in the supported grammar; the compiler does not pin it yet",
    ),
    row(
        "prefix_caching",
        Probed,
        &[],
        None,
        "vllm enables prefix caching by default upstream; no explicit pin is emitted",
    ),
    row(
        "speculative_decoding",
        Unsupported,
        &[],
        None,
        "spec decode surface not wired on this lane yet",
    ),
    row(
        "multi_gpu_parallelism",
        Probed,
        &[],
        None,
        "upstream tensor parallelism exists; no first-class knob on this lane yet",
    ),
    row(
        "memory_offload",
        FirstClass,
        &["--gpu-memory-utilization", "--kv-transfer-config"],
        None,
        "memory fraction always emitted; persistent KV offload is opt-in with a tiered store",
    ),
    row(
        "structured_output",
        Probed,
        &[],
        None,
        "guided decoding is request-side on this lane",
    ),
    row(
        "lora",
        Probed,
        &[],
        None,
        "upstream LoRA surface not wired on this lane yet",
    ),
    row(
        "multimodal",
        Probed,
        &[],
        None,
        "HF vision towers route upstream; no launch knob is pinned",
    ),
    row(
        "cancellation_streaming",
        FirstClass,
        &[],
        None,
        "gateway-owned stream and cancel; /health readiness probe",
    ),
];

const SDCPP: &[CapabilityDisposition] = &[
    row(
        "batching_scheduling",
        Unsupported,
        &[],
        None,
        "one-shot diffusion generation; no batch admission surface",
    ),
    row(
        "kv_cache_quantization",
        Unsupported,
        &[],
        None,
        "no KV cache on a diffusion lane",
    ),
    row(
        "prefix_caching",
        Unsupported,
        &[],
        None,
        "no decode cache on a diffusion lane",
    ),
    row(
        "speculative_decoding",
        Unsupported,
        &[],
        None,
        "not applicable to diffusion",
    ),
    row(
        "multi_gpu_parallelism",
        Passthrough,
        &["--rpc-servers"],
        None,
        "rpc server list is extra_args-reachable; not auto-planned",
    ),
    row(
        "memory_offload",
        FirstClass,
        &["--offload-to-cpu", "--max-vram"],
        None,
        "CPU offload and VRAM ceiling knobs",
    ),
    row(
        "structured_output",
        Unsupported,
        &[],
        None,
        "image bytes, not constrained text",
    ),
    row(
        "lora",
        FirstClass,
        &["--lora-model-dir"],
        None,
        "LoRA directory knob for diffusion adapters",
    ),
    row(
        "multimodal",
        FirstClass,
        &["--llm", "--clip_vision", "--qwen2vl_vision"],
        None,
        "the lane IS multimodal generation; component sets are first-class",
    ),
    row(
        "cancellation_streaming",
        FirstClass,
        &[],
        None,
        "job ledger + gateway cancel own the lifecycle",
    ),
];

const WHISPER: &[CapabilityDisposition] = &[
    row(
        "batching_scheduling",
        Unsupported,
        &[],
        None,
        "single-utterance transcription; no batch surface",
    ),
    row(
        "kv_cache_quantization",
        Unsupported,
        &[],
        None,
        "no persistent KV on the audio lane",
    ),
    row(
        "prefix_caching",
        Unsupported,
        &[],
        None,
        "no decode cache on the audio lane",
    ),
    row(
        "speculative_decoding",
        Unsupported,
        &[],
        None,
        "encoder-decoder; no speculation surface",
    ),
    row(
        "multi_gpu_parallelism",
        Unsupported,
        &[],
        None,
        "single-device audio lane",
    ),
    row(
        "memory_offload",
        Unsupported,
        &[],
        None,
        "model fits device memory by admission design",
    ),
    row(
        "structured_output",
        Unsupported,
        &[],
        None,
        "plain text/json output; no grammar surface",
    ),
    row(
        "lora",
        Unsupported,
        &[],
        None,
        "no adapter surface advertised",
    ),
    row(
        "multimodal",
        FirstClass,
        &["--diarize", "--tinydiarize", "--vad", "--vad-model"],
        None,
        "audio-in lane; diarization and VAD are first-class knobs",
    ),
    row(
        "cancellation_streaming",
        FirstClass,
        &[],
        None,
        "gateway-owned job cancel",
    ),
];

const PIPER: &[CapabilityDisposition] = &[
    row(
        "batching_scheduling",
        Unsupported,
        &[],
        None,
        "one-shot synthesis spawns; concurrency is gateway-side",
    ),
    row(
        "kv_cache_quantization",
        Unsupported,
        &[],
        None,
        "no KV cache on the TTS lane",
    ),
    row(
        "prefix_caching",
        Unsupported,
        &[],
        None,
        "no decode cache on the TTS lane",
    ),
    row(
        "speculative_decoding",
        Unsupported,
        &[],
        None,
        "not applicable to synthesis",
    ),
    row(
        "multi_gpu_parallelism",
        Unsupported,
        &[],
        None,
        "CPU synthesis lane",
    ),
    row(
        "memory_offload",
        Unsupported,
        &[],
        None,
        "voice models are device-memory-sized by design",
    ),
    row(
        "structured_output",
        Unsupported,
        &[],
        None,
        "audio out; no grammar surface",
    ),
    row("lora", Unsupported, &[], None, "no adapter surface"),
    row(
        "multimodal",
        Unsupported,
        &[],
        None,
        "single-modality audio-out lane (see speech_synthesis)",
    ),
    row(
        "cancellation_streaming",
        FirstClass,
        &[],
        None,
        "gateway-owned job cancel",
    ),
    row(
        "speech_synthesis",
        LiveCertified,
        &["--model", "--output_raw", "--length_scale", "--speaker"],
        Some("bench-artifacts/20261010-engine-utilization/live_spawn_proofs.md"),
        "piper-TTS wav synthesized and round-trip transcribed by whisper beam-5 child",
    ),
];

const KOKORO: &[CapabilityDisposition] = &[
    row(
        "batching_scheduling",
        Unsupported,
        &[],
        None,
        "single /tts shim child; concurrency is gateway-side",
    ),
    row(
        "kv_cache_quantization",
        Unsupported,
        &[],
        None,
        "no KV cache on the TTS lane",
    ),
    row(
        "prefix_caching",
        Unsupported,
        &[],
        None,
        "no decode cache on the TTS lane",
    ),
    row(
        "speculative_decoding",
        Unsupported,
        &[],
        None,
        "not applicable to synthesis",
    ),
    row(
        "multi_gpu_parallelism",
        Unsupported,
        &[],
        None,
        "CPU ONNX lane",
    ),
    row(
        "memory_offload",
        Unsupported,
        &[],
        None,
        "82M-param model is device-memory-sized by design",
    ),
    row(
        "structured_output",
        Unsupported,
        &[],
        None,
        "audio out; no grammar surface",
    ),
    row("lora", Unsupported, &[], None, "no adapter surface"),
    row(
        "multimodal",
        Unsupported,
        &[],
        None,
        "single-modality audio-out lane (see speech_synthesis)",
    ),
    row(
        "cancellation_streaming",
        FirstClass,
        &[],
        None,
        "gateway-owned job cancel",
    ),
    row(
        "speech_synthesis",
        LiveCertified,
        &[],
        Some("bench-artifacts/20261120-kokoro-vllm-cert/README.md"),
        "zero-CLI-flag python shim: Blazar composes the whole serving path (venv + ONNX model + /tts endpoint); af_heart 24 kHz wav certified live 2026-11-20",
    ),
];

const MLX: &[CapabilityDisposition] = &[
    row(
        "batching_scheduling",
        FirstClass,
        &["--decode-concurrency", "--prompt-concurrency"],
        None,
        "decode/prompt concurrency knobs",
    ),
    row(
        "kv_cache_quantization",
        LiveCertified,
        &["--kv-bits", "--kv-group-size", "--quantized-kv-start"],
        Some("bench-artifacts/20261010-engine-utilization/mlx_kv_bits_ab.md"),
        "A/B: 1.9x slower decode, 108MiB saved - memory lever, opt-in",
    ),
    row(
        "prefix_caching",
        FirstClass,
        &["--prompt-cache-size", "--prompt-cache-bytes"],
        None,
        "prompt cache budget knobs",
    ),
    row(
        "speculative_decoding",
        FirstClass,
        &["--draft-model", "--num-draft-tokens"],
        None,
        "draft model + token count knobs",
    ),
    row(
        "multi_gpu_parallelism",
        Unsupported,
        &[],
        None,
        "single-device MLX runtime",
    ),
    row(
        "memory_offload",
        Unsupported,
        &[],
        None,
        "mlx manages memory internally; kv-bits is the memory lever",
    ),
    row(
        "structured_output",
        Probed,
        &[],
        None,
        "request-side only on this lane",
    ),
    row(
        "lora",
        FirstClass,
        &["--adapter-path"],
        None,
        "adapter path knob",
    ),
    row(
        "multimodal",
        Probed,
        &[],
        None,
        "mlx-vl exists upstream; no tower routing wired",
    ),
    row(
        "cancellation_streaming",
        FirstClass,
        &[],
        None,
        "gateway-owned stream and cancel",
    ),
];

const OPENAIHOST: &[CapabilityDisposition] = &[
    row(
        "batching_scheduling",
        Unsupported,
        &[],
        None,
        "child owns its serving config; Blazar spawns one shared child",
    ),
    row(
        "kv_cache_quantization",
        Unsupported,
        &[],
        None,
        "child-internal, not argv-configurable",
    ),
    row("prefix_caching", Unsupported, &[], None, "child-internal"),
    row(
        "speculative_decoding",
        Unsupported,
        &[],
        None,
        "child-internal",
    ),
    row(
        "multi_gpu_parallelism",
        Unsupported,
        &[],
        None,
        "child-internal",
    ),
    row("memory_offload", Unsupported, &[], None, "child-internal"),
    row(
        "structured_output",
        Probed,
        &[],
        None,
        "request-side via the OpenAI surface",
    ),
    row(
        "lora",
        Unsupported,
        &[],
        None,
        "child-internal model registry",
    ),
    row(
        "multimodal",
        Probed,
        &[],
        None,
        "depends on the child's imported models",
    ),
    row(
        "cancellation_streaming",
        FirstClass,
        &[],
        None,
        "gateway-owned stream and cancel over the child's OpenAI surface",
    ),
];

/// Dispositions for a lane, canonical families first.
#[must_use]
pub fn lane_capabilities(kind: EngineKind) -> &'static [CapabilityDisposition] {
    match kind {
        EngineKind::LlamaCpp => LLAMACPP,
        EngineKind::MistralRs => MISTRALRS,
        EngineKind::Sglang => SGLANG,
        EngineKind::Vllm => VLLM,
        EngineKind::SdCpp => SDCPP,
        EngineKind::Whisper => WHISPER,
        EngineKind::Piper => PIPER,
        EngineKind::Kokoro => KOKORO,
        EngineKind::Mlx => MLX,
        EngineKind::OpenAiHost => OPENAIHOST,
    }
}

/// Serializable capabilities document keyed by lane name - the shape the
/// `knob_registry` example embeds as the JSON "capabilities" section.
#[must_use]
pub fn capabilities_doc() -> BTreeMap<&'static str, Vec<CapabilityDisposition>> {
    EngineKind::ALL
        .iter()
        .map(|k| (k.as_str(), lane_capabilities(*k).to_vec()))
        .collect()
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    /// Every lane disposes of every canonical family exactly once, in
    /// canonical order first - the invariant the coverage gate mirrors.
    #[test]
    fn unit__capability_matrix__canonical_families_complete_and_ordered() {
        for kind in EngineKind::ALL {
            let rows = lane_capabilities(kind);
            for (i, family) in CANONICAL_FAMILIES.iter().enumerate() {
                assert!(
                    rows.get(i).is_some_and(|r| r.family == *family),
                    "{}: canonical family {family} missing or out of order: {:?}",
                    kind,
                    rows.iter().map(|r| r.family).collect::<Vec<_>>()
                );
            }
            let mut seen = std::collections::BTreeSet::new();
            for r in rows {
                assert!(
                    seen.insert(r.family),
                    "{}: duplicate family {}",
                    kind,
                    r.family
                );
            }
        }
    }

    /// A `LiveCertified` claim without a receipt path is exactly the
    /// fabrication this module exists to prevent.
    #[test]
    fn unit__capability_matrix__live_certified_requires_receipt() {
        for kind in EngineKind::ALL {
            for r in lane_capabilities(kind) {
                assert!(
                    (r.evidence == LiveCertified) == r.receipt.is_some(),
                    "{} {}: LiveCertified and receipt must agree ({:?})",
                    kind,
                    r.family,
                    r.receipt
                );
            }
        }
    }

    /// Rows must be non-empty or explicitly Unsupported - a family with
    /// no flags and no evidence text is an unfilled template.
    #[test]
    fn unit__capability_matrix__every_row_carries_a_note() {
        for kind in EngineKind::ALL {
            for r in lane_capabilities(kind) {
                assert!(
                    !r.note.trim().is_empty(),
                    "{} {}: empty note",
                    kind,
                    r.family
                );
            }
        }
    }
}
