//! Engine kind: which upstream project an installed engine row belongs
//! to. Drives repo/asset resolution, manifest probing, profile argv
//! compilation, and gateway feature gating. Stored in the `engines.kind`
//! column (string-backed, default `llamacpp`) — the single source of
//! truth; the manifest JSON never duplicates it.

use std::fmt;
use std::str::FromStr;

use rusqlite::types::{FromSql, FromSqlError, ToSql, ToSqlOutput, ValueRef};

/// The upstream inference server an engine row wraps.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum EngineKind {
    /// ggml-org/llama.cpp `llama-server` (prebuilt assets + source build).
    #[default]
    LlamaCpp,
    /// EricLBuehler/mistral.rs `mistralrs serve` (prebuilt assets).
    MistralRs,
    /// sgl-project/sglang `python -m sglang.launch_server` (pip venv;
    /// HF safetensors models, Linux CUDA/ROCm upstream).
    Sglang,
}

impl EngineKind {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            EngineKind::LlamaCpp => "llamacpp",
            EngineKind::MistralRs => "mistralrs",
            EngineKind::Sglang => "sglang",
        }
    }

    /// Format+policy-driven route for `[engine_routing] mode = "auto"`:
    /// pick the serving engine for one model from the installed kinds.
    ///
    /// GGUF runs everywhere but llamacpp's quant kernels are the
    /// reference (mistral.rs is the overlap fallback) — for every
    /// policy, until a quant-matched mistral.rs GGUF row lands in the
    /// engine matrix (that experiment is pending).
    ///
    /// safetensors routes by policy, on measured evidence: quality and
    /// throughput prefer sglang (0.612-vs-0.575 quality and
    /// 752-vs-509 tok/s conc4, engine matrix 2026-09-17); latency
    /// prefers mistral.rs (26-ms-vs-73-ms warm TTFT and 8-s-vs-92-s
    /// cold boot — the F16 matrix run). Fallbacks keep the format
    /// served when the preferred lane is not installed. `None` =
    /// nothing installed can serve the format (caller teaches; e.g.
    /// safetensors with only llamacpp installed).
    ///
    /// `quantized` (AWQ/GPTQ/FP8 safetensors checkpoints) collapses the
    /// candidate list to sglang: mistral.rs prebuilt builds dequant
    /// these checkpoints broken (live-proven v0.9.3: dtype-mismatch
    /// empty 200s on default dtype; f16 workaround emits garbage
    /// tokens) while the same lanes serve BF16 dirs perfectly — so a
    /// quantized dir routes sglang-only, and an absent sglang teaches
    /// instead of routing into a known-broken lane.
    #[must_use]
    pub fn route_format(
        safetensors: bool,
        quantized: bool,
        installed: &[EngineKind],
        policy: crate::config::RoutingPolicy,
    ) -> Option<EngineKind> {
        let prefer = |primary: EngineKind, fallback: EngineKind| {
            installed
                .iter()
                .copied()
                .find(|k| *k == primary)
                .or_else(|| installed.iter().copied().find(|k| *k == fallback))
        };
        if safetensors && quantized {
            return installed.iter().copied().find(|k| *k == EngineKind::Sglang);
        }
        let [primary, fallback] = Self::format_preference_order(safetensors, policy);
        prefer(primary, fallback)
    }

    /// The policy-ordered candidate kinds for a format, best first —
    /// the order [`route_format`](Self::route_format) tries and the
    /// order the CLI's just-in-time install offer recommends them in.
    #[must_use]
    pub fn format_preference_order(
        safetensors: bool,
        policy: crate::config::RoutingPolicy,
    ) -> [EngineKind; 2] {
        use crate::config::RoutingPolicy;
        if safetensors {
            match policy {
                RoutingPolicy::Latency => [EngineKind::MistralRs, EngineKind::Sglang],
                RoutingPolicy::Quality | RoutingPolicy::Throughput => {
                    [EngineKind::Sglang, EngineKind::MistralRs]
                }
            }
        } else {
            [EngineKind::LlamaCpp, EngineKind::MistralRs]
        }
    }
}

/// Typed [`serving_lane`] failure for callers that act on the reason
/// (the CLI's just-in-time engine install offer) instead of only
/// teaching. The rendered text is byte-identical to the historical
/// string errors — it flows into API bodies and validate expectations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaneError {
    /// The model's engine pin names a kind with no installed row.
    PinKindMissing { kind: EngineKind, roster: String },
    /// The model's engine pin matches no installed tag or kind.
    PinUnknown { pin: String, roster: String },
    /// No installed engine serves the model's format. `quantized` =
    /// AWQ/GPTQ/FP8 safetensors checkpoint — the teaching narrows to
    /// sglang, the only lane that serves those.
    FormatUnserved {
        safetensors: bool,
        quantized: bool,
        roster: String,
    },
}

impl LaneError {
    /// Kinds whose installation would unblock this model, best first.
    /// Empty = installing nothing helps (unknown pin) — caller teaches.
    #[must_use]
    pub fn missing_kinds(&self, policy: crate::config::RoutingPolicy) -> Vec<EngineKind> {
        match self {
            Self::PinKindMissing { kind, .. } => vec![*kind],
            Self::PinUnknown { .. } => Vec::new(),
            Self::FormatUnserved {
                safetensors,
                quantized,
                ..
            } => {
                if *safetensors && *quantized {
                    vec![EngineKind::Sglang]
                } else {
                    EngineKind::format_preference_order(*safetensors, policy).to_vec()
                }
            }
        }
    }
}

impl fmt::Display for LaneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PinKindMissing { kind, roster } => write!(
                f,
                "no {kind} engine installed — pallama engine install --kind {kind} (installed: {roster})"
            ),
            Self::PinUnknown { pin, roster } => write!(
                f,
                "model engine pin \"{pin}\" matches no installed tag or kind — installed: {roster}"
            ),
            Self::FormatUnserved {
                safetensors,
                quantized,
                roster,
            } if *safetensors && *quantized => write!(
                f,
                "no installed engine serves quantized safetensors (AWQ/GPTQ/FP8) — sglang is the lane for those; pallama engine install --kind sglang (installed: {roster})"
            ),
            Self::FormatUnserved {
                safetensors, roster, ..
            } => write!(
                f,
                "no installed engine serves the {} format — install one (sglang|mistralrs for safetensors, llamacpp for GGUF); installed: {roster}",
                if *safetensors { "safetensors" } else { "GGUF" }
            ),
        }
    }
}

/// The single routing decision both the supervisor (adapter pick at
/// spawn) and the gateway (per-child protocol quirks, e.g. mistral.rs's
/// `default` model id) consult — one source so they can never disagree.
///
/// `pin` = the per-model `engine = "…"` override (exact tag first, then
/// kind), `global` = the daemon's active engine kind, `installed` =
/// (tag, kind) rows newest-first. `Ok(None)` = serve on the global lane,
/// Provenance class of an installed lane, used as the tiebreak when
/// several lanes of the same [`EngineKind`] could serve a model: a
/// `Fork` lane (llama.cpp fork built for an architecture mainline
/// lacks) is a capability shim, never the default for architectures a
/// `Mainstream` lane already serves.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LaneClass {
    #[default]
    Mainstream,
    Fork,
}

/// no routed adapter needed; `Ok(Some((tag, kind)))` = route this spawn
/// to a local adapter of that row; `Err(teaching)` = nothing installed
/// can serve the model (or the pin names something absent).
pub fn serving_lane(
    mode: crate::config::RoutingMode,
    policy: crate::config::RoutingPolicy,
    pin: Option<&str>,
    safetensors: bool,
    quantized: bool,
    global: EngineKind,
    installed: &[(String, EngineKind, LaneClass)],
) -> Result<Option<(String, EngineKind)>, String> {
    serving_lane_typed(mode, policy, pin, safetensors, quantized, global, installed)
        .map_err(|e| e.to_string())
}

/// [`serving_lane`] with the failure reason typed — callers that act on
/// the reason (CLI just-in-time install offer) read the variant; every
/// other caller keeps the string form.
pub fn serving_lane_typed(
    mode: crate::config::RoutingMode,
    policy: crate::config::RoutingPolicy,
    pin: Option<&str>,
    safetensors: bool,
    quantized: bool,
    global: EngineKind,
    installed: &[(String, EngineKind, LaneClass)],
) -> Result<Option<(String, EngineKind)>, LaneError> {
    use crate::config::RoutingMode;
    let roster = || {
        installed
            .iter()
            .map(|(t, k, _)| format!("{t} ({k})"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    if let Some(pin) = pin {
        if let Some(row) = installed.iter().find(|(t, _, _)| t == pin) {
            return Ok(Some((row.0.clone(), row.1)));
        }
        if let Ok(kind) = pin.parse::<EngineKind>() {
            if let Some((tag, _, _)) = installed.iter().find(|(_, k, _)| *k == kind) {
                return Ok(Some((tag.clone(), kind)));
            }
            return Err(LaneError::PinKindMissing {
                kind,
                roster: roster(),
            });
        }
        return Err(LaneError::PinUnknown {
            pin: pin.to_string(),
            roster: roster(),
        });
    }
    if mode == RoutingMode::Manual {
        return Ok(None);
    }
    let kinds: Vec<EngineKind> = installed.iter().map(|(_, k, _)| *k).collect();
    match EngineKind::route_format(safetensors, quantized, &kinds, policy) {
        Some(kind) if kind == global => Ok(None),
        // `kinds` is built from `installed`, so a matching lane always
        // exists; the None arm is pure type-shape. Among same-kind
        // lanes a Mainstream build wins over a Fork shim — a fork
        // exists to serve architectures mainline lacks, so overlaps
        // (same arch in both) belong on the maintained build. Stable
        // min_by_key keeps the caller's order (newest first) inside
        // each class.
        Some(kind) => Ok(installed
            .iter()
            .filter(|(_, k, _)| *k == kind)
            .min_by_key(|(_, _, class)| u8::from(*class == LaneClass::Fork))
            .map(|(tag, _, _)| (tag.clone(), kind))),
        None => Err(LaneError::FormatUnserved {
            safetensors,
            quantized,
            roster: roster(),
        }),
    }
}

impl fmt::Display for EngineKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for EngineKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "llamacpp" => Ok(EngineKind::LlamaCpp),
            "mistralrs" => Ok(EngineKind::MistralRs),
            "sglang" => Ok(EngineKind::Sglang),
            other => Err(format!(
                "unknown engine kind {other:?} (supported: llamacpp, mistralrs, sglang)"
            )),
        }
    }
}

impl ToSql for EngineKind {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::Borrowed(ValueRef::Text(
            self.as_str().as_bytes(),
        )))
    }
}

impl FromSql for EngineKind {
    fn column_result(value: ValueRef<'_>) -> Result<Self, FromSqlError> {
        let text = String::column_result(value)?;
        EngineKind::from_str(&text).map_err(|e| FromSqlError::Other(e.into()))
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn unit__engine_kind_roundtrip__db_and_serde() {
        #[derive(serde::Deserialize)]
        struct Old {
            #[serde(default)]
            kind: EngineKind,
        }
        for k in [
            EngineKind::LlamaCpp,
            EngineKind::MistralRs,
            EngineKind::Sglang,
        ] {
            assert_eq!(EngineKind::from_str(k.as_str()), Ok(k));
            let json = serde_json::to_string(&k).unwrap();
            assert_eq!(serde_json::from_str::<EngineKind>(&json).unwrap(), k);
        }
        assert_eq!(
            EngineKind::from_str("vllm").unwrap_err(),
            "unknown engine kind \"vllm\" (supported: llamacpp, mistralrs, sglang)"
        );
        // serde default on missing field = llamacpp (old rows).
        assert_eq!(
            serde_json::from_str::<Old>("{}").unwrap().kind,
            EngineKind::LlamaCpp
        );
    }

    /// The `[engine_routing]` format+policy table: GGUF prefers llamacpp
    /// with mistral.rs as the overlap fallback (every policy); safetensors
    /// is sglang under quality/throughput (0.612-vs-0.575, 752-vs-509
    /// conc4) and mistral.rs under latency (26-vs-73 ms warm TTFT, 8-vs-
    /// 92 s cold); nothing installed that can serve = None (caller
    /// teaches).
    #[test]
    fn unit__route_format__prefers_native_and_falls_back() {
        use crate::config::RoutingPolicy::{Latency, Quality, Throughput};
        use EngineKind::{LlamaCpp, MistralRs, Sglang};

        let all = [LlamaCpp, MistralRs, Sglang];
        assert_eq!(
            EngineKind::route_format(false, false, &all, Quality),
            Some(LlamaCpp)
        );
        assert_eq!(
            EngineKind::route_format(true, false, &all, Quality),
            Some(Sglang)
        );
        assert_eq!(
            EngineKind::route_format(true, false, &all, Throughput),
            Some(Sglang)
        );
        // Latency flips safetensors to mistral.rs on TTFT/cold evidence;
        // GGUF stays on llamacpp quant kernels regardless.
        assert_eq!(
            EngineKind::route_format(true, false, &all, Latency),
            Some(MistralRs)
        );
        assert_eq!(
            EngineKind::route_format(false, false, &all, Latency),
            Some(LlamaCpp)
        );

        // Quantized safetensors (AWQ/GPTQ/FP8): sglang-only, every
        // policy — mistral.rs dequant is broken on these checkpoints
        // (live-proven v0.9.3), so it must never be the fallback, and
        // a latency policy must not flip a quantized dir onto it.
        assert_eq!(
            EngineKind::route_format(true, true, &all, Quality),
            Some(Sglang)
        );
        assert_eq!(
            EngineKind::route_format(true, true, &all, Latency),
            Some(Sglang)
        );
        assert_eq!(
            EngineKind::route_format(true, true, &[LlamaCpp, MistralRs], Quality),
            None,
            "quantized dir with no sglang = unservable (teach), never mistral.rs"
        );
        // GGUF ignores the quantized flag — GGUF quants are their own
        // well-served lane.
        assert_eq!(
            EngineKind::route_format(false, true, &all, Quality),
            Some(LlamaCpp)
        );

        // Overlap fallbacks: GGUF without llamacpp, safetensors without
        // sglang — both land on mistral.rs.
        assert_eq!(
            EngineKind::route_format(false, false, &[MistralRs, Sglang], Quality),
            Some(MistralRs)
        );
        assert_eq!(
            EngineKind::route_format(true, false, &[LlamaCpp, MistralRs], Quality),
            Some(MistralRs)
        );
        // Latency without mistral.rs falls back to sglang.
        assert_eq!(
            EngineKind::route_format(true, false, &[LlamaCpp, Sglang], Latency),
            Some(Sglang)
        );

        // Unserved formats teach instead of guessing.
        assert_eq!(
            EngineKind::route_format(true, false, &[LlamaCpp], Quality),
            None
        );
        assert_eq!(
            EngineKind::route_format(false, false, &[Sglang], Quality),
            None
        );
    }

    /// The JIT-install offer derives its menu from the typed failure:
    /// pin-to-missing-kind offers exactly that kind, format-missing
    /// offers the policy-ordered candidates, unknown pins offer nothing.
    #[test]
    fn unit__lane_error__missing_kinds_per_class() {
        use crate::config::RoutingPolicy::{Latency, Quality};
        use EngineKind::{LlamaCpp, MistralRs, Sglang};

        let roster = "b1 (llamacpp)".to_string();
        assert_eq!(
            LaneError::PinKindMissing {
                kind: Sglang,
                roster: roster.clone()
            }
            .missing_kinds(Quality),
            vec![Sglang]
        );
        assert_eq!(
            LaneError::PinUnknown {
                pin: "bogus".into(),
                roster: roster.clone()
            }
            .missing_kinds(Quality),
            Vec::<EngineKind>::new()
        );
        // Policy orders the format candidates: quality wants sglang
        // first, latency wants mistral.rs first; GGUF is fixed.
        assert_eq!(
            LaneError::FormatUnserved {
                safetensors: true,
                quantized: false,
                roster: roster.clone()
            }
            .missing_kinds(Quality),
            vec![Sglang, MistralRs]
        );
        assert_eq!(
            LaneError::FormatUnserved {
                safetensors: true,
                quantized: false,
                roster: roster.clone()
            }
            .missing_kinds(Latency),
            vec![MistralRs, Sglang]
        );
        assert_eq!(
            LaneError::FormatUnserved {
                safetensors: false,
                quantized: false,
                roster
            }
            .missing_kinds(Quality),
            vec![LlamaCpp, MistralRs]
        );
        // Quantized safetensors offer narrows to sglang alone — the
        // JIT install menu must not recommend a known-broken lane.
        assert_eq!(
            LaneError::FormatUnserved {
                safetensors: true,
                quantized: true,
                roster: "b1 (llamacpp)".to_string()
            }
            .missing_kinds(Latency),
            vec![Sglang]
        );
        assert_eq!(
            LaneError::FormatUnserved {
                safetensors: true,
                quantized: true,
                roster: "b1 (llamacpp)".to_string()
            }
            .to_string(),
            "no installed engine serves quantized safetensors (AWQ/GPTQ/FP8) — sglang is the lane for those; pallama engine install --kind sglang (installed: b1 (llamacpp))"
        );
    }

    /// The typed errors must render BYTE-IDENTICAL to the historical
    /// string errors — the text flows into API bodies and validate
    /// expectations; parity is the contract.
    #[test]
    fn unit__serving_lane_error_text__byte_identical_to_historical() {
        use crate::config::{RoutingMode, RoutingPolicy};
        use EngineKind::LlamaCpp;

        let installed = vec![("b1".to_string(), LlamaCpp, LaneClass::Mainstream)];
        let cases = [("sglang", true), ("bogus-tag", true), (":irrelevant", true)];
        for (pin, safetensors) in cases {
            let typed = serving_lane_typed(
                RoutingMode::Auto,
                RoutingPolicy::Quality,
                Some(pin),
                safetensors,
                false,
                LlamaCpp,
                &installed,
            )
            .unwrap_err()
            .to_string();
            let historical = serving_lane(
                RoutingMode::Auto,
                RoutingPolicy::Quality,
                Some(pin),
                safetensors,
                false,
                LlamaCpp,
                &installed,
            )
            .unwrap_err();
            assert_eq!(typed, historical, "pin {pin}");
        }
        // Format-missing class: same parity, plus the exact wording.
        let fmt_typed = serving_lane_typed(
            RoutingMode::Auto,
            RoutingPolicy::Quality,
            None,
            true,
            false,
            LlamaCpp,
            &installed,
        )
        .unwrap_err()
        .to_string();
        assert_eq!(
            fmt_typed,
            "no installed engine serves the safetensors format — install one (sglang|mistralrs for safetensors, llamacpp for GGUF); installed: b1 (llamacpp)"
        );
    }

    /// A fork lane never shadows a mainstream lane of the same kind for
    /// overlapping formats, even when the fork is newer (installed lists
    /// arrive newest-first); a fork still serves when it is the only
    /// lane of the kind, and a pin forces it explicitly.
    #[test]
    fn unit__serving_lane__mainstream_beats_newer_fork_same_kind() {
        use crate::config::RoutingMode::Auto;
        use crate::config::RoutingPolicy::Quality;
        use EngineKind::LlamaCpp;
        // Newest-first, exactly as `list_engines` yields rows: the fork
        // was installed after the mainstream build.
        let installed = vec![
            (
                "fork-acme_x-7c81a9f0-cuda".to_string(),
                LlamaCpp,
                LaneClass::Fork,
            ),
            ("b11026-cuda".to_string(), LlamaCpp, LaneClass::Mainstream),
        ];
        // GGUF model with a non-llamacpp global engine: mainstream wins.
        let pick = serving_lane(
            Auto,
            Quality,
            None,
            false,
            false,
            EngineKind::Sglang,
            &installed,
        )
        .unwrap();
        assert_eq!(pick.map(|(t, _)| t), Some("b11026-cuda".to_string()));
        // Pin still forces the fork when the user asks for it.
        let pinned = serving_lane(
            Auto,
            Quality,
            Some("fork-acme_x-7c81a9f0-cuda"),
            false,
            false,
            EngineKind::Sglang,
            &installed,
        )
        .unwrap();
        assert_eq!(
            pinned.map(|(t, _)| t),
            Some("fork-acme_x-7c81a9f0-cuda".to_string())
        );
        // Fork-only roster: the fork serves (it is the shim of last
        // resort, not a dead lane).
        let fork_only = vec![(
            "fork-acme_x-7c81a9f0-cuda".to_string(),
            LlamaCpp,
            LaneClass::Fork,
        )];
        let pick = serving_lane(
            Auto,
            Quality,
            None,
            false,
            false,
            EngineKind::Sglang,
            &fork_only,
        )
        .unwrap();
        assert_eq!(
            pick.map(|(t, _)| t),
            Some("fork-acme_x-7c81a9f0-cuda".to_string())
        );
        // Newest mainstream still wins over an OLDER mainstream (the
        // pre-fork behavior is untouched on mainstream-only boxes).
        let two_mainstream = vec![
            ("b11100-cuda".to_string(), LlamaCpp, LaneClass::Mainstream),
            ("b11026-cuda".to_string(), LlamaCpp, LaneClass::Mainstream),
        ];
        let pick = serving_lane(
            Auto,
            Quality,
            None,
            false,
            false,
            EngineKind::Sglang,
            &two_mainstream,
        )
        .unwrap();
        assert_eq!(pick.map(|(t, _)| t), Some("b11100-cuda".to_string()));
    }
}
