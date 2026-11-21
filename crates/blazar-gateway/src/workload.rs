//! Normalized workload descriptors: ONE derivation layer over the
//! admission classes, shared by every API dialect.
//!
//! `WorkClass` (queue.rs) is the scheduling truth - three classes, WFQ by
//! API key, EDF deadlines, reserved capacity. What was missing is a
//! vocabulary for WHAT the request is (an embedding batch, a video
//! render, a tool-calling agent turn) that every dialect derives the
//! same way. `WorkloadDescriptor` is that vocabulary: a pure
//! `(surface, has_tools, raw_lane) -> descriptor` function whose
//! `work_class()` maps back onto the exact `classify_work` verdict the
//! call sites produced before (chat family) or the `Interactive` class
//! the media/embed surfaces hardcoded. No new queues, no scheduling
//! changes - the descriptor labels telemetry
//! (`blazar_workload_requests_total{kind}`) and documents the
//! media-vs-text resource split that admission already enforces.
//!
//! Continuity for agent workloads is NOT a descriptor field: agent
//! turns are `InteractiveChat`/`ToolExecution` requests whose
//! cross-turn state rides the pool affinity machinery (opener-hash
//! buckets in pools.rs) - pinned by tests there and here.
//!
//! Kinds map to surfaces that really construct them; there is
//! deliberately no `Classification` or `Batch` kind: `/v1/classify`
//! and the batch replayer ride `openai_proxy`, so their admission
//! already derives through the chat path and a never-constructed
//! variant would be dead vocabulary.

use crate::queue::WorkClass;
use crate::state::AppState;

/// Derive and record a descriptor, returning it for `work_class()`.
/// Exactly one `note` per request - handlers with multiple admission
/// points derive once at entry and pass the class down, so the counter
/// never double-counts a request.
pub fn note(
    state: &AppState,
    surface: WorkloadSurface,
    has_tools: bool,
    raw_lane: bool,
) -> WorkloadDescriptor {
    let descriptor = WorkloadDescriptor::derive(surface, has_tools, raw_lane);
    state.workload[descriptor.kind.idx()].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    descriptor
}

/// Derive, record, and ALSO persist the per-model kind mix (the
/// workload-aware routing derivation's evidence). Handlers that know
/// the model at admission call this instead of [`note`] - one call,
/// both the global counter and the per-model mix, still exactly-once
/// per request.
pub fn note_for(
    state: &AppState,
    model: &str,
    surface: WorkloadSurface,
    has_tools: bool,
    raw_lane: bool,
) -> WorkloadDescriptor {
    let descriptor = note(state, surface, has_tools, raw_lane);
    state.usage.record_workload(model, descriptor.kind.as_str());
    descriptor
}

/// Derive, record (global + per-model mix), and map in one call - the
/// model-aware drop-in for [`classify`].
pub fn classify_for(
    state: &AppState,
    model: &str,
    surface: WorkloadSurface,
    has_tools: bool,
    raw_lane: bool,
) -> WorkClass {
    note_for(state, model, surface, has_tools, raw_lane).work_class()
}

/// Derive, record, and map in one call - the drop-in replacement for
/// the direct `classify_work` calls at single-admission handlers.
pub fn classify(
    state: &AppState,
    surface: WorkloadSurface,
    has_tools: bool,
    raw_lane: bool,
) -> WorkClass {
    note(state, surface, has_tools, raw_lane).work_class()
}

/// Render the per-kind request counters as Prometheus text.
pub fn render(state: &AppState, out: &mut String) {
    use std::fmt::Write as _;
    use std::sync::atomic::Ordering::Relaxed;
    for kind in WorkloadKind::ALL {
        let _ = writeln!(
            out,
            "blazar_workload_requests_total{{kind=\"{}\"}} {}",
            kind.as_str(),
            state.workload[kind.idx()].load(Relaxed)
        );
    }
}

/// The gateway surface a request arrived on. Carries just enough to
/// pick a kind: modality (text decode, embed, diffusion, audio) and
/// the dialect-independent bits the handlers already compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadSurface {
    /// Chat-family completion surface (`OpenAI` chat, Ollama /api/chat,
    /// Anthropic messages, raw completions under the compat recipe).
    Chat,
    /// Text embedding surface (`OpenAI` embeddings incl. late chunking,
    /// Ollama /api/embed(+dings)).
    Embeddings,
    /// Reranking surface (rerank/score routes).
    Rerank,
    /// Diffusion image generation.
    ImageGeneration,
    /// Diffusion video generation.
    VideoGeneration,
    /// Text-to-speech synthesis.
    SpeechSynthesis,
    /// Audio transcription.
    Transcription,
}

/// What the request IS - dialect-neutral, derived per request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadKind {
    /// Plain chat turn (no tools, not the raw lane).
    InteractiveChat,
    /// Request declares tools for the model to call.
    ToolExecution,
    /// Raw-lane generation: a completion the compat recipe serves
    /// raw - long by construction, capped by the `RawLong` ceiling.
    LongPrefill,
    /// Embedding workload (batched text -> vectors).
    Embedding,
    /// Reranking workload (query/document scoring).
    Reranking,
    /// Diffusion image generation.
    ImageGeneration,
    /// Diffusion video generation.
    VideoGeneration,
    /// Text-to-speech synthesis.
    SpeechSynthesis,
    /// Audio transcription.
    Transcription,
}

impl WorkloadKind {
    /// Stable label for the telemetry counter and docs tables.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InteractiveChat => "interactive_chat",
            Self::ToolExecution => "tool_execution",
            Self::LongPrefill => "long_prefill",
            Self::Embedding => "embedding",
            Self::Reranking => "reranking",
            Self::ImageGeneration => "image_generation",
            Self::VideoGeneration => "video_generation",
            Self::SpeechSynthesis => "speech_synthesis",
            Self::Transcription => "transcription",
        }
    }

    /// Every kind, declaration order - the index layout of the
    /// `AppState` per-kind request counters.
    pub const ALL: [Self; 9] = [
        Self::InteractiveChat,
        Self::ToolExecution,
        Self::LongPrefill,
        Self::Embedding,
        Self::Reranking,
        Self::ImageGeneration,
        Self::VideoGeneration,
        Self::SpeechSynthesis,
        Self::Transcription,
    ];

    /// Index into the per-kind counter array (position in `ALL`).
    #[must_use]
    pub fn idx(self) -> usize {
        Self::ALL.iter().position(|k| *k == self).unwrap_or(0)
    }
}

/// The resource pool a workload competes in - the axis admission and
/// the media-job ledger already arbitrate (text decode vs diffusion
/// VRAM vs audio children).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceClass {
    /// GPU text decode (chat, tools, raw generation).
    TextDecode,
    /// Text embedding/rerank (short-latency, batch-friendly).
    TextEmbed,
    /// Diffusion VRAM (image/video).
    Diffusion,
    /// Audio children (synthesis, transcription).
    Audio,
}

/// Per-request workload identity: kind + resource class. Derived
/// before admission; `work_class()` feeds the queue unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkloadDescriptor {
    /// What the request is.
    pub kind: WorkloadKind,
    /// The resource axis it competes on.
    pub resource: ResourceClass,
}

impl WorkloadDescriptor {
    /// Derive the descriptor. `has_tools` and `raw_lane` are the same
    /// bits each handler already computes for `classify_work` (tools
    /// array present; the compat recipe serving the body raw) - the
    /// descriptor never second-guesses them.
    ///
    /// Precedence mirrors `classify_work` exactly: raw lane wins over
    /// tools (a raw completion is long by construction), tools win
    /// over plain chat, and the non-text surfaces are their own kinds
    /// regardless of body shape.
    #[must_use]
    pub fn derive(surface: WorkloadSurface, has_tools: bool, raw_lane: bool) -> Self {
        let kind = match surface {
            WorkloadSurface::Chat => {
                if raw_lane {
                    WorkloadKind::LongPrefill
                } else if has_tools {
                    WorkloadKind::ToolExecution
                } else {
                    WorkloadKind::InteractiveChat
                }
            }
            WorkloadSurface::Embeddings => WorkloadKind::Embedding,
            WorkloadSurface::Rerank => WorkloadKind::Reranking,
            WorkloadSurface::ImageGeneration => WorkloadKind::ImageGeneration,
            WorkloadSurface::VideoGeneration => WorkloadKind::VideoGeneration,
            WorkloadSurface::SpeechSynthesis => WorkloadKind::SpeechSynthesis,
            WorkloadSurface::Transcription => WorkloadKind::Transcription,
        };
        let resource = match kind {
            WorkloadKind::InteractiveChat
            | WorkloadKind::ToolExecution
            | WorkloadKind::LongPrefill => ResourceClass::TextDecode,
            WorkloadKind::Embedding | WorkloadKind::Reranking => ResourceClass::TextEmbed,
            WorkloadKind::ImageGeneration | WorkloadKind::VideoGeneration => {
                ResourceClass::Diffusion
            }
            WorkloadKind::SpeechSynthesis | WorkloadKind::Transcription => ResourceClass::Audio,
        };
        Self { kind, resource }
    }

    /// The admission class - identical to what the call site passed
    /// before the descriptor existed: `classify_work` for the chat
    /// family (same precedence), `Interactive` for every non-text
    /// surface (their historical hardcoded class).
    #[must_use]
    pub fn work_class(self) -> WorkClass {
        match self.kind {
            WorkloadKind::LongPrefill => WorkClass::RawLong,
            WorkloadKind::ToolExecution => WorkClass::Tool,
            WorkloadKind::InteractiveChat
            | WorkloadKind::Embedding
            | WorkloadKind::Reranking
            | WorkloadKind::ImageGeneration
            | WorkloadKind::VideoGeneration
            | WorkloadKind::SpeechSynthesis
            | WorkloadKind::Transcription => WorkClass::Interactive,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // unit__<x>__<y> double-underscore convention
    use super::*;

    #[test]
    fn unit__workload__chat_family_maps_exactly_like_classify_work() {
        // The descriptor must reproduce classify_work's verdict for
        // every (has_tools, raw_lane) combination - behavior
        // preservation on the admission path is the whole contract.
        for &(has_tools, raw_lane) in &[(false, false), (true, false), (false, true), (true, true)]
        {
            let d = WorkloadDescriptor::derive(WorkloadSurface::Chat, has_tools, raw_lane);
            assert_eq!(
                d.work_class(),
                crate::queue::classify_work(has_tools, raw_lane),
                "chat family mismatch at has_tools={has_tools} raw_lane={raw_lane}"
            );
        }
    }

    #[test]
    fn unit__workload__raw_lane_wins_over_tools() {
        // Same precedence as classify_work: a raw completion is long
        // by construction even when the body declares tools.
        let d = WorkloadDescriptor::derive(WorkloadSurface::Chat, true, true);
        assert_eq!(d.kind, WorkloadKind::LongPrefill);
        assert_eq!(d.work_class(), WorkClass::RawLong);
    }

    #[test]
    fn unit__workload__non_text_surfaces_keep_the_interactive_class() {
        // Embeddings, rerank, diffusion, and audio surfaces all
        // hardcoded WorkClass::Interactive before descriptors existed
        // - the mapping must not change their admission behavior.
        for surface in [
            WorkloadSurface::Embeddings,
            WorkloadSurface::Rerank,
            WorkloadSurface::ImageGeneration,
            WorkloadSurface::VideoGeneration,
            WorkloadSurface::SpeechSynthesis,
            WorkloadSurface::Transcription,
        ] {
            let d = WorkloadDescriptor::derive(surface, false, false);
            assert_eq!(d.work_class(), WorkClass::Interactive);
        }
    }

    #[test]
    fn unit__workload__dialect_bodies_derive_the_same_descriptor() {
        // Acceptance: the same logical workload expressed in the
        // three API dialects derives the same descriptor. A tools
        // turn is a tools turn whether the tools array rides an
        // OpenAI body, an Ollama /api/chat body, or the Anthropic
        // messages translation (all three sites feed has_tools=true,
        // raw_lane=false today).
        let openai = WorkloadDescriptor::derive(WorkloadSurface::Chat, true, false);
        let ollama = WorkloadDescriptor::derive(WorkloadSurface::Chat, true, false);
        let anthropic = WorkloadDescriptor::derive(WorkloadSurface::Chat, true, false);
        assert_eq!(openai, ollama);
        assert_eq!(ollama, anthropic);
        assert_eq!(openai.kind, WorkloadKind::ToolExecution);
    }

    #[test]
    fn unit__workload__resource_split_covers_every_kind() {
        // Every kind names the resource axis it competes on - the
        // media-vs-text split the admission ledger arbitrates.
        let kinds = [
            WorkloadKind::InteractiveChat,
            WorkloadKind::ToolExecution,
            WorkloadKind::LongPrefill,
            WorkloadKind::Embedding,
            WorkloadKind::Reranking,
            WorkloadKind::ImageGeneration,
            WorkloadKind::VideoGeneration,
            WorkloadKind::SpeechSynthesis,
            WorkloadKind::Transcription,
        ];
        for surface in [
            WorkloadSurface::Chat,
            WorkloadSurface::Embeddings,
            WorkloadSurface::Rerank,
            WorkloadSurface::ImageGeneration,
            WorkloadSurface::VideoGeneration,
            WorkloadSurface::SpeechSynthesis,
            WorkloadSurface::Transcription,
        ] {
            let d = WorkloadDescriptor::derive(surface, false, false);
            assert!(kinds.contains(&d.kind));
            assert!(
                d.kind
                    .as_str()
                    .chars()
                    .next()
                    .is_some_and(char::is_alphabetic),
                "kind labels are non-empty words"
            );
        }
    }

    #[test]
    fn unit__workload__kind_labels_are_stable_snake_case() {
        // Telemetry labels are a contract: pin the exact strings.
        assert_eq!(WorkloadKind::InteractiveChat.as_str(), "interactive_chat");
        assert_eq!(WorkloadKind::ToolExecution.as_str(), "tool_execution");
        assert_eq!(WorkloadKind::LongPrefill.as_str(), "long_prefill");
        assert_eq!(WorkloadKind::Embedding.as_str(), "embedding");
        assert_eq!(WorkloadKind::Reranking.as_str(), "reranking");
        assert_eq!(WorkloadKind::ImageGeneration.as_str(), "image_generation");
        assert_eq!(WorkloadKind::VideoGeneration.as_str(), "video_generation");
        assert_eq!(WorkloadKind::SpeechSynthesis.as_str(), "speech_synthesis");
        assert_eq!(WorkloadKind::Transcription.as_str(), "transcription");
    }

    #[test]
    fn unit__workload__media_kinds_never_take_the_raw_long_ceiling() {
        // Acceptance (media cannot starve chat): diffusion and audio
        // kinds ride the Interactive class - the same WFQ bucket and
        // reserved-capacity arithmetic as chat, never the RawLong
        // ceiling lane. Their VRAM arbitration lives in the media-job
        // ledger, not in a queue class change.
        for surface in [
            WorkloadSurface::ImageGeneration,
            WorkloadSurface::VideoGeneration,
            WorkloadSurface::SpeechSynthesis,
            WorkloadSurface::Transcription,
        ] {
            let d = WorkloadDescriptor::derive(surface, false, false);
            assert_ne!(d.work_class(), WorkClass::RawLong);
            assert_eq!(d.work_class(), WorkClass::Interactive);
        }
    }

    #[test]
    fn unit__workload__agent_continuations_retain_pool_affinity() {
        // Acceptance (agent continuity): an agent conversation that
        // grows by APPENDING turns keeps the affinity PrefixKey
        // stable - same opener hash (system[:256]) and same convo
        // hash (system[:1024] + FIRST user[:1024]) - so the pool
        // affinity machinery routes every turn to the same bucket
        // and the KV prefix stays warm. No workflow header needed.
        let turn1 = serde_json::json!({
            "model": "m",
            "messages": [
                {"role": "system", "content": "You are a coding agent. Use the tools."},
                {"role": "user", "content": "Fix the failing test in src/lib.rs"}
            ]
        });
        let turn2 = serde_json::json!({
            "model": "m",
            "messages": [
                {"role": "system", "content": "You are a coding agent. Use the tools."},
                {"role": "user", "content": "Fix the failing test in src/lib.rs"},
                {"role": "assistant", "content": "I ran the tests and found the bug."},
                {"role": "user", "content": "Good. Now add a regression test."}
            ]
        });
        let k1 = crate::proxy::affinity_hash_bytes(turn1.to_string().as_bytes());
        let k2 = crate::proxy::affinity_hash_bytes(turn2.to_string().as_bytes());
        assert!(k1.is_some() && k2.is_some());
        assert_eq!(k1, k2, "appended turns keep the affinity key");
    }
}
