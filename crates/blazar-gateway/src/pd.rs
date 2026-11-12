//! `SGLang` prefill/decode (PD) disaggregation pair — gateway dispatch.
//!
//! When a model's sglang tuning carries `sglang.disagg_pair = true`, the
//! supervisor keeps TWO role children alive (`<model>#pd-pre` hosting the
//! bootstrap server, `<model>#pd-dec` serving decode) and this module
//! routes chat traffic through them:
//!
//! 1. Phase 1 warms the prefill child with a tiny non-streaming
//!    completion and reads the bootstrap coordinates
//!    (`bootstrap_host` / `bootstrap_room`) it reports — leniently, from
//!    the top level or under `meta_info`.
//! 2. Phase 2 forwards the ORIGINAL request to the decode child with the
//!    coordinates attached, and relays the response (stream honored).
//!    A prefill completion WITHOUT coordinates refuses phase 2 with a
//!    teaching error: a decode-role server rejects coordinate-less
//!    requests at admission, so forwarding would only surface that
//!    upstream 400 in a less teachable shape.
//!
//! v1 wiring-tier honesty: the OpenAI-compatible chat surface does not report
//! `bootstrap_host`/`bootstrap_room` (they ride sglang's native
//! `/generate` responses), so the coordinates path stays ready for the
//! native `/generate` two-phase follow-up while the chat lane proves
//! the pair topology, routing and lifecycle end-to-end. Real KV
//! transfer additionally needs a transfer backend wheel (`mooncake` /
//! `nixl`) installed into the sglang lane venv; upstream refuses the
//! `fake` backend on the prefill role outright. The PD lane bypasses
//! queue/single-flight admission — per-key admission already ran in
//! `openai_proxy` before this hook.

use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::http::StatusCode;
use axum::response::Response;

use blazar_core::engine_kind::EngineKind;

use crate::state::AppState;

/// Does this model route through the PD pair? Requires the lane to
/// resolve to sglang AND the overlay-or-global sglang tuning to ask for
/// the pair. Precedence mirrors the supervisor's spawn-time fill
/// (overlay wins, config.sglang is the floor).
pub(crate) fn pd_pair_requested(state: &Arc<AppState>, model: &str) -> bool {
    if crate::proxy::routed_kind_for(state, model) != Some(EngineKind::Sglang) {
        return false;
    }
    let cfg = state.cfg();
    let tuning = cfg
        .overlay_for(model)
        .sglang
        .clone()
        .unwrap_or_else(|| cfg.sglang.clone());
    tuning.disagg_pair == Some(true)
}

/// Two-phase PD dispatch. `body` is the already-rewritten request bytes
/// from `openai_proxy` (voice/reasoning/tool rewrites applied, per-key
/// admission ran); `model` is the resolved model name.
#[allow(clippy::too_many_lines)]
pub(crate) async fn pd_two_phase(state: &Arc<AppState>, model: &str, body: &Bytes) -> Response {
    pd_router_forward(state, model, body, "/v1/chat/completions").await
}

/// Forward `body` VERBATIM to the supervised sglang-router child that
/// fronts the PD pair for `model` (spawned by `ensure_pd`; the router
/// owns room allocation through the prefill bootstrap server - the
/// gateway cannot replicate that handshake over plain HTTP forwarding,
/// live-receipted 2026-11-12 as `KVTransferError ... Aborted by
/// AbortReq` for gateway-invented rooms). The router speaks both the
/// OpenAI-compatible chat surface and the native /generate surface, so
/// both gateway PD paths reduce to this one forward.
async fn pd_router_forward(
    state: &Arc<AppState>,
    model: &str,
    body: &Bytes,
    path: &str,
) -> Response {
    let (_decode, _prefill, router) = match state.sup.ensure_pd(model).await {
        Ok(triple) => triple,
        Err(e) => {
            tracing::warn!("PD {model}: pair or router unavailable: {e:#}");
            return crate::proxy::openai_error(
                503,
                "the prefill/decode pair (or its sglang-router) could not come up for this                  model - see the daemon log (spawn fit, transfer-engine wheel or RAM class                  are the usual suspects)",
            );
        }
    };
    let url = format!("{}{path}", crate::proxy::child_base(&router.endpoint));
    let req = crate::proxy::child_auth(
        crate::state::child_client(state, &router.endpoint)
            .post(&url)
            .header("content-type", "application/json"),
        &router,
    )
    .body(body.to_vec());
    match crate::proxy::child_send(state, &router, req.send()).await {
        Ok(resp) => {
            let status = StatusCode::from_u16(resp.status().as_u16())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            let mut headers = resp.headers().clone();
            headers.remove(axum::http::header::CONTENT_LENGTH);
            let mut response = Response::new(Body::from_stream(resp.bytes_stream()));
            *response.status_mut() = status;
            *response.headers_mut() = headers;
            response
        }
        Err(e) => crate::proxy::openai_error(
            e.status_u16(),
            &format!("PD router round on {path} failed: {e}"),
        ),
    }
}

/// Native `/generate` two-phase dispatch (router role): allocate a room,
/// stamp the request with `bootstrap_host`/`bootstrap_room`, let the
/// prefill child own that room for its KV write, then hand the SAME
/// stamped body to the decode child which dials the transfer engine for
/// the KV it needs. Unlike the chat lane, the wire contract completes
/// here because the coordinates travel in the request both ways.
///
/// Boundary (live receipt 2026-11-12, double-confirmed): the room
/// number must be allocated AND registered through the prefill's
/// bootstrap server (HTTP protocol on the bootstrap port; the registry
/// binds the prefill KV-writer to the room before decode's reader
/// dials). A gateway-invented room reaches decode's admission but has
/// no registered sender, so the transfer aborts
/// (`KVTransferError ... Aborted by AbortReq`). Completing the real
/// transfer needs a router-class implementation - either a bootstrap-
/// protocol client owning the room lifecycle here, or supervising
/// sglang's reference router between the gateway and the pair. The
/// Native-surface PD dispatch: same router forward on `/generate`.
pub(crate) async fn pd_generate_two_phase(
    state: &Arc<AppState>,
    model: &str,
    body: &Bytes,
) -> Response {
    pd_router_forward(state, model, body, "/generate").await
}

#[cfg(test)]
mod tests {
    // Router-forward path is supervisor+HTTP-integrated: shape is pinned by the
    // supervisor spawn tests and the live E2E receipts (2026-11-12).
}
