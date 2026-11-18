//! Connector registry: engine kind -> connector, with capability
//! honesty. Engines without a slot-bank protocol get `Unsupported`
//! carrying the reason; nothing is ever simulated.

use blazar_core::engine_kind::EngineKind;

use super::{KvConnector, LlamaCppSlotIo};

/// Build the connector for a serving instance.
///
/// `router_model` is `Some(name)` when the child runs in router mode
/// (one engine process serving several models - the bank POST must
/// name which model the state belongs to).
#[must_use]
pub fn connector_for(
    kind: EngineKind,
    auth: Option<String>,
    router_model: Option<String>,
) -> KvConnector {
    if kind.supports_slot_banks() {
        KvConnector::LlamaCpp(LlamaCppSlotIo::new(auth, router_model))
    } else {
        KvConnector::Unsupported(
            "engine kind speaks no slot-bank protocol (llama.cpp slot banks only today)",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    // suite convention: unit__scenario__expected (6b)
    #[allow(non_snake_case)]
    async fn unit__registry__unsupported_kind_reports_not_simulates() {
        let conn = connector_for(EngineKind::Sglang, None, None);
        assert!(!conn.capabilities().export);
        let err = conn
            .export_prefix(crate::kv::KvExportRequest {
                slot: 0,
                filename: "x",
                client: crate::supervisor::test_local_http(),
                base: "http://127.0.0.1:1",
            })
            .await
            .unwrap_err();
        assert!(matches!(err, blazar_core::kv::KvError::Unsupported(_)));
    }

    #[tokio::test]
    // suite convention: unit__scenario__expected (6b)
    #[allow(non_snake_case)]
    async fn unit__registry__slot_bank_kind_gains_llama_connector() {
        let conn = connector_for(EngineKind::LlamaCpp, None, None);
        assert!(conn.capabilities().export);
        assert!(conn.capabilities().import);
    }
}
