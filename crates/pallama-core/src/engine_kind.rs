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
    /// engine matrix (that experiment is pending; see
    /// docs/engine-coverage.md).
    ///
    /// safetensors routes by policy, on measured evidence: quality and
    /// throughput prefer sglang (0.612-vs-0.575 quality and
    /// 752-vs-509 tok/s conc4, engine matrix 2026-09-17); latency
    /// prefers mistral.rs (26-ms-vs-73-ms warm TTFT and 8-s-vs-92-s
    /// cold boot — the F16 matrix run). Fallbacks keep the format
    /// served when the preferred lane is not installed. `None` =
    /// nothing installed can serve the format (caller teaches; e.g.
    /// safetensors with only llamacpp installed).
    #[must_use]
    pub fn route_format(
        safetensors: bool,
        installed: &[EngineKind],
        policy: crate::config::RoutingPolicy,
    ) -> Option<EngineKind> {
        use crate::config::RoutingPolicy;
        let prefer = |primary: EngineKind, fallback: EngineKind| {
            installed
                .iter()
                .copied()
                .find(|k| *k == primary)
                .or_else(|| installed.iter().copied().find(|k| *k == fallback))
        };
        if safetensors {
            match policy {
                RoutingPolicy::Latency => prefer(EngineKind::MistralRs, EngineKind::Sglang),
                RoutingPolicy::Quality | RoutingPolicy::Throughput => {
                    prefer(EngineKind::Sglang, EngineKind::MistralRs)
                }
            }
        } else {
            prefer(EngineKind::LlamaCpp, EngineKind::MistralRs)
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
/// no routed adapter needed; `Ok(Some((tag, kind)))` = route this spawn
/// to a local adapter of that row; `Err(teaching)` = nothing installed
/// can serve the model (or the pin names something absent).
pub fn serving_lane(
    mode: crate::config::RoutingMode,
    policy: crate::config::RoutingPolicy,
    pin: Option<&str>,
    safetensors: bool,
    global: EngineKind,
    installed: &[(String, EngineKind)],
) -> Result<Option<(String, EngineKind)>, String> {
    use crate::config::RoutingMode;
    let roster = || {
        installed
            .iter()
            .map(|(t, k)| format!("{t} ({k})"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    if let Some(pin) = pin {
        if let Some(row) = installed.iter().find(|(t, _)| t == pin) {
            return Ok(Some(row.clone()));
        }
        if let Ok(kind) = pin.parse::<EngineKind>() {
            if let Some(row) = installed.iter().find(|(_, k)| *k == kind) {
                return Ok(Some(row.clone()));
            }
            return Err(format!(
                "no {kind} engine installed — pallama engine install --kind {kind} (installed: {})",
                roster()
            ));
        }
        return Err(format!(
            "model engine pin \"{pin}\" matches no installed tag or kind — installed: {}",
            roster()
        ));
    }
    if mode == RoutingMode::Manual {
        return Ok(None);
    }
    let kinds: Vec<EngineKind> = installed.iter().map(|(_, k)| *k).collect();
    match EngineKind::route_format(safetensors, &kinds, policy) {
        Some(kind) if kind == global => Ok(None),
        // `kinds` is built from `installed`, so the find always matches;
        // the None arm is pure type-shape.
        Some(kind) => Ok(installed
            .iter()
            .find(|(_, k)| *k == kind)
            .map(|row| row.clone())),
        None => Err(format!(
            "no installed engine serves the {} format — install one (sglang|mistralrs for safetensors, llamacpp for GGUF); installed: {}",
            if safetensors { "safetensors" } else { "GGUF" },
            roster()
        )),
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
            EngineKind::route_format(false, &all, Quality),
            Some(LlamaCpp)
        );
        assert_eq!(EngineKind::route_format(true, &all, Quality), Some(Sglang));
        assert_eq!(
            EngineKind::route_format(true, &all, Throughput),
            Some(Sglang)
        );
        // Latency flips safetensors to mistral.rs on TTFT/cold evidence;
        // GGUF stays on llamacpp quant kernels regardless.
        assert_eq!(
            EngineKind::route_format(true, &all, Latency),
            Some(MistralRs)
        );
        assert_eq!(
            EngineKind::route_format(false, &all, Latency),
            Some(LlamaCpp)
        );

        // Overlap fallbacks: GGUF without llamacpp, safetensors without
        // sglang — both land on mistral.rs.
        assert_eq!(
            EngineKind::route_format(false, &[MistralRs, Sglang], Quality),
            Some(MistralRs)
        );
        assert_eq!(
            EngineKind::route_format(true, &[LlamaCpp, MistralRs], Quality),
            Some(MistralRs)
        );
        // Latency without mistral.rs falls back to sglang.
        assert_eq!(
            EngineKind::route_format(true, &[LlamaCpp, Sglang], Latency),
            Some(Sglang)
        );

        // Unserved formats teach instead of guessing.
        assert_eq!(EngineKind::route_format(true, &[LlamaCpp], Quality), None);
        assert_eq!(EngineKind::route_format(false, &[Sglang], Quality), None);
    }
}
