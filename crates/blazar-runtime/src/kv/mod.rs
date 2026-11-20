//! Engine-neutral KV cache connector layer.
//!
//! The enum (not `dyn`) is the dispatch: one real connector today
//! (llama.cpp slot banks, the protocol `bank_save`/`bank_restore_sync`
//! have always spoken) and an explicit `Unsupported` arm for every
//! other engine kind. Unsupported is a *capability statement*: callers
//! fall back to cold prefill, never to a simulated transfer.

pub mod compatibility;
pub mod interchange;
pub mod inventory;
pub mod mlx_preload;
pub mod registry;
pub mod repack;
pub mod translation;

use blazar_core::kv::{KvCapabilities, KvError};

/// Request/response shapes for prefix export/import. v1 carries the
/// slot-bank semantics (engine-local file, slot id); richer artifact
/// flows (cross-process transfer) extend this in later waves without
/// changing the connector surface.
#[derive(Debug, Clone)]
pub struct KvExportRequest<'a> {
    pub slot: u32,
    pub filename: &'a str,
    pub client: &'a reqwest::Client,
    pub base: &'a str,
}

#[derive(Debug, Clone)]
pub struct KvImportRequest<'a> {
    pub slot: u32,
    pub filename: &'a str,
    pub client: &'a reqwest::Client,
    pub base: &'a str,
}

/// Token count llama-server reports in slot save/restore responses
/// (`n_saved` / `n_restored`). `Some(0)` is a real answer - HTTP 200
/// with zero tokens is the silent re-prefill bug class, not a warm
/// start. Missing or non-numeric fields are `None`: unverifiable.
pub(crate) fn slot_reported_tokens(body: &serde_json::Value, field: &str) -> Option<u64> {
    body.get(field).and_then(serde_json::Value::as_u64)
}

/// Receipt for one completed slot operation: how many tokens the
/// engine actually saved/restored, as reported by the engine itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvSlotReceipt {
    pub tokens: Option<u64>,
}

/// The llama.cpp slot-bank protocol: `POST /slots/{id}?action=save|restore`.
///
/// Holds per-instance request context; the HTTP client and base URL
/// are loaned per call so the supervisor can dial once and reuse the
/// connection across slots exactly as before.
pub struct LlamaCppSlotIo {
    auth: Option<String>,
    router_model: Option<String>,
}

impl LlamaCppSlotIo {
    #[must_use]
    pub fn new(auth: Option<String>, router_model: Option<String>) -> Self {
        Self { auth, router_model }
    }

    /// Ask the engine to persist one slot's KV state. `None` tokens =
    /// engine reported nothing saved (empty slot); errors are failures.
    pub async fn export_prefix(&self, req: KvExportRequest<'_>) -> Result<KvSlotReceipt, KvError> {
        let url = format!(
            "{}/slots/{}?action=save&filename={}",
            req.base, req.slot, req.filename
        );
        let mut body = serde_json::json!({ "filename": req.filename });
        if let Some(model) = &self.router_model {
            // Router mode: the child serves many models and needs the
            // bank tagged with which one produced it.
            body["model"] = serde_json::Value::String(model.clone());
        }
        let mut http = req
            .client
            .post(&url)
            .json(&body)
            .timeout(std::time::Duration::from_secs(2));
        if let Some(secret) = &self.auth {
            http = http.bearer_auth(secret);
        }
        let resp = http
            .send()
            .await
            .map_err(|e| KvError::Transport(e.to_string()))?;
        let ok = resp.status().is_success();
        let v: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| KvError::Transport(e.to_string()))?;
        if !ok {
            return Err(KvError::Engine(format!(
                "slot save rejected: {}",
                v["error"]["message"].as_str().unwrap_or("unknown")
            )));
        }
        Ok(KvSlotReceipt {
            tokens: slot_reported_tokens(&v, "n_saved"),
        })
    }

    /// Ask the engine to reload one slot's KV state. The response's
    /// `n_restored` is the truth: HTTP 200 alone can hide a re-prefill
    /// (the original bug class this protocol wraps).
    pub async fn import_prefix(&self, req: KvImportRequest<'_>) -> Result<KvSlotReceipt, KvError> {
        let url = format!(
            "{}/slots/{}?action=restore&filename={}",
            req.base, req.slot, req.filename
        );
        let mut body = serde_json::json!({ "filename": req.filename });
        if let Some(model) = &self.router_model {
            body["model"] = serde_json::Value::String(model.clone());
        }
        let mut http = req
            .client
            .post(&url)
            .json(&body)
            .timeout(std::time::Duration::from_secs(5));
        if let Some(secret) = &self.auth {
            http = http.bearer_auth(secret);
        }
        let resp = http
            .send()
            .await
            .map_err(|e| KvError::Transport(e.to_string()))?;
        let ok = resp.status().is_success();
        let v: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| KvError::Transport(e.to_string()))?;
        if !ok {
            return Err(KvError::Engine(format!(
                "slot restore rejected: {}",
                v["error"]["message"].as_str().unwrap_or("unknown")
            )));
        }
        Ok(KvSlotReceipt {
            tokens: slot_reported_tokens(&v, "n_restored"),
        })
    }
}

/// The vllm offload store: a filesystem tier the ENGINE manages for
/// itself (`[vllm.kv_offload]` spawns the child with the built-in
/// `OffloadingConnector`). There is no gateway-commanded transfer
/// protocol - the connector exists so capability inquiries (doctor,
/// route scoring) report the persistent prefix store honestly instead
/// of "unsupported protocol", and transfer attempts fail with
/// teaching rather than simulation.
pub struct VllmOffloadIo;

impl VllmOffloadIo {
    const TRANSFER_TEACHING: &str = concat!(
        "the vllm offload store is engine-managed ([vllm.kv_offload]): ",
        "the child tiers and reloads blocks itself; the gateway observes ",
        "reuse via cached_tokens, it does not command individual transfers"
    );
}

/// Engine-dispatched connector. Construct through
/// [`registry::connector_for`]; never matched by callers.
pub enum KvConnector {
    LlamaCpp(LlamaCppSlotIo),
    Vllm(VllmOffloadIo),
    /// sglang lane opted into the shared `[kv.lmcache]` daemon: the
    /// store is real (prefixes restore across engine instances of the
    /// same lane - verified by `cached_tokens` on a fresh child) but
    /// daemon-mediated - no gateway-commanded transfer exists.
    SglangLmcache(SglangLmcacheIo),
    Unsupported(&'static str),
}

/// Marker IO for the sglang `LMCache` lane: every method teaches - the
/// daemon owns the protocol, the gateway only owns its lifecycle.
pub struct SglangLmcacheIo;

impl SglangLmcacheIo {
    pub const TRANSFER_TEACHING: &'static str = "sglang LMCache transfers are daemon-mediated ([kv.lmcache]) - the \
         gateway commands none; prefix reuse happens across instances of the \
         same engine lane through the shared daemon";
}

impl KvConnector {
    #[must_use]
    pub fn capabilities(&self) -> KvCapabilities {
        match self {
            Self::LlamaCpp(_) => KvCapabilities {
                export: true,
                import: true,
                persistent: true,
                prefix_blocks: false,
                cross_process: false,
                supported_formats: vec!["llama-slot-bank".into()],
            },
            Self::Vllm(_) => KvCapabilities {
                // Gateway-commanded transfers do not exist for this
                // store - but the engine-side persistence it provides
                // (prefix reuse across restarts, content-addressed by
                // run-config digest) is exactly what the persistent /
                // prefix_blocks / cross_process flags describe.
                export: false,
                import: false,
                persistent: true,
                prefix_blocks: true,
                cross_process: true,
                supported_formats: vec!["vllm-offload-fs ([vllm.kv_offload])".into()],
            },
            Self::SglangLmcache(_) => KvCapabilities {
                // The shared daemon store: prefixes persist and cross
                // engine instances of the lane (the entire point of
                // the lane, verified via cached_tokens on a fresh
                // child) - but the gateway cannot command individual
                // transfers, only the daemon lifecycle.
                export: false,
                import: false,
                persistent: true,
                prefix_blocks: true,
                cross_process: true,
                supported_formats: vec!["lmcache-mp ([kv.lmcache] + sglang.lmcache)".into()],
            },
            Self::Unsupported(_) => KvCapabilities {
                export: false,
                import: false,
                persistent: false,
                prefix_blocks: false,
                cross_process: false,
                supported_formats: Vec::new(),
            },
        }
    }

    pub async fn export_prefix(&self, req: KvExportRequest<'_>) -> Result<KvSlotReceipt, KvError> {
        match self {
            Self::LlamaCpp(io) => io.export_prefix(req).await,
            Self::Vllm(_) => Err(KvError::Unsupported(
                VllmOffloadIo::TRANSFER_TEACHING.into(),
            )),
            Self::SglangLmcache(_) => Err(KvError::Unsupported(
                SglangLmcacheIo::TRANSFER_TEACHING.into(),
            )),
            Self::Unsupported(why) => Err(KvError::Unsupported((*why).to_string())),
        }
    }

    pub async fn import_prefix(&self, req: KvImportRequest<'_>) -> Result<KvSlotReceipt, KvError> {
        match self {
            Self::LlamaCpp(io) => io.import_prefix(req).await,
            Self::Vllm(_) => Err(KvError::Unsupported(
                VllmOffloadIo::TRANSFER_TEACHING.into(),
            )),
            Self::SglangLmcache(_) => Err(KvError::Unsupported(
                SglangLmcacheIo::TRANSFER_TEACHING.into(),
            )),
            Self::Unsupported(why) => Err(KvError::Unsupported((*why).to_string())),
        }
    }
}
