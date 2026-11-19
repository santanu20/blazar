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
///
/// `sglang_lmcache` is the config-level shared-daemon truth
/// (`[kv.lmcache] enabled` AND `sglang.lmcache.enabled`) - a global
/// capability fact, deliberately blind to per-model overlays: bank
/// paths gate on `supports_slot_banks()` before reaching here, and
/// doctor/route-scoring ask at fleet scope.
#[must_use]
pub fn connector_for(
    kind: EngineKind,
    auth: Option<String>,
    router_model: Option<String>,
    sglang_lmcache: bool,
) -> KvConnector {
    // The vllm offload store is engine-managed, not gateway-commanded:
    // this arm exists for capability inquiries (doctor, route
    // scoring). The supervisor's bank_save/bank_restore paths gate on
    // supports_slot_banks() before they ever reach here, so a vllm
    // instance never rides the slot-bank protocol by accident.
    if kind == EngineKind::Vllm {
        return KvConnector::Vllm(super::VllmOffloadIo);
    }
    if kind == EngineKind::Sglang && sglang_lmcache {
        // Honest caps only when the lane actually dials the shared
        // daemon; without it sglang speaks no KV protocol at all.
        return KvConnector::SglangLmcache(super::SglangLmcacheIo);
    }
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
        let conn = connector_for(EngineKind::Sglang, None, None, false);
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
        let conn = connector_for(EngineKind::LlamaCpp, None, None, false);
        assert!(conn.capabilities().export);
        assert!(conn.capabilities().import);
    }

    #[tokio::test]
    // suite convention: unit__scenario__expected (6b)
    #[allow(non_snake_case)]
    async fn unit__registry__sglang_lmcache_lane_reports_daemon_store_only_when_enabled() {
        // Daemon off: sglang carries no KV protocol at all.
        let off = connector_for(EngineKind::Sglang, None, None, false);
        assert!(!off.capabilities().persistent);
        // Daemon + lane opt-in: the shared store is real but
        // daemon-mediated - never gateway-commanded.
        let on = connector_for(EngineKind::Sglang, None, None, true);
        let caps = on.capabilities();
        assert!(!caps.export);
        assert!(!caps.import);
        assert!(caps.persistent);
        assert!(caps.prefix_blocks);
        assert!(caps.cross_process);
        assert_eq!(
            caps.supported_formats,
            vec!["lmcache-mp ([kv.lmcache] + sglang.lmcache)"]
        );
        let err = on
            .export_prefix(crate::kv::KvExportRequest {
                slot: 0,
                filename: "x",
                client: crate::supervisor::test_local_http(),
                base: "http://127.0.0.1:1",
            })
            .await
            .unwrap_err();
        match err {
            blazar_core::kv::KvError::Unsupported(why) => {
                assert!(why.contains("daemon-mediated"));
            }
            other => panic!("expected Unsupported teaching, got {other:?}"),
        }
    }

    #[tokio::test]
    // suite convention: unit__scenario__expected (6b)
    #[allow(non_snake_case)]
    async fn unit__registry__vllm_reports_engine_managed_store_and_refuses_transfers() {
        let conn = connector_for(EngineKind::Vllm, None, None, false);
        let caps = conn.capabilities();
        // The fs tier persists prefixes across restarts, but the
        // gateway cannot command individual transfers.
        assert!(!caps.export);
        assert!(!caps.import);
        assert!(caps.persistent);
        assert!(caps.prefix_blocks);
        assert!(caps.cross_process);
        assert_eq!(
            caps.supported_formats,
            vec!["vllm-offload-fs ([vllm.kv_offload])"]
        );
        let err = conn
            .import_prefix(crate::kv::KvImportRequest {
                slot: 0,
                filename: "x",
                client: crate::supervisor::test_local_http(),
                base: "http://127.0.0.1:1",
            })
            .await
            .unwrap_err();
        match err {
            blazar_core::kv::KvError::Unsupported(why) => {
                assert!(why.contains("engine-managed"));
            }
            other => panic!("expected Unsupported teaching, got {other:?}"),
        }
    }
}
