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

    /// Format-driven route for `[engine_routing] mode = "auto"`: pick the
    /// serving engine for one model from the installed kinds. GGUF runs
    /// everywhere but llamacpp's quant kernels are the reference
    /// (mistral.rs is the overlap fallback); safetensors is sglang's
    /// native lane (mistral.rs fallback — llamacpp has no safetensors
    /// lane at all). `None` = nothing installed can serve the format
    /// (caller teaches; e.g. safetensors with only llamacpp installed).
    #[must_use]
    pub fn route_format(safetensors: bool, installed: &[EngineKind]) -> Option<EngineKind> {
        if safetensors {
            installed
                .iter()
                .find(|k| matches!(k, EngineKind::Sglang))
                .or_else(|| {
                    installed
                        .iter()
                        .find(|k| matches!(k, EngineKind::MistralRs))
                })
                .copied()
        } else {
            installed
                .iter()
                .find(|k| matches!(k, EngineKind::LlamaCpp))
                .or_else(|| {
                    installed
                        .iter()
                        .find(|k| matches!(k, EngineKind::MistralRs))
                })
                .copied()
        }
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

    /// The `[engine_routing]` format table: GGUF prefers llamacpp with
    /// mistral.rs as the overlap fallback; safetensors is sglang-native
    /// (mistral.rs fallback, llamacpp has no lane at all); nothing
    /// installed that can serve = None (caller teaches).
    #[test]
    fn unit__route_format__prefers_native_and_falls_back() {
        use EngineKind::{LlamaCpp, MistralRs, Sglang};

        let all = [LlamaCpp, MistralRs, Sglang];
        assert_eq!(EngineKind::route_format(false, &all), Some(LlamaCpp));
        assert_eq!(EngineKind::route_format(true, &all), Some(Sglang));

        // Overlap fallbacks: GGUF without llamacpp, safetensors without
        // sglang — both land on mistral.rs.
        assert_eq!(
            EngineKind::route_format(false, &[MistralRs, Sglang]),
            Some(MistralRs)
        );
        assert_eq!(
            EngineKind::route_format(true, &[LlamaCpp, MistralRs]),
            Some(MistralRs)
        );

        // Unserved formats teach instead of guessing.
        assert_eq!(EngineKind::route_format(true, &[LlamaCpp]), None);
        assert_eq!(EngineKind::route_format(false, &[Sglang]), None);
    }
}
