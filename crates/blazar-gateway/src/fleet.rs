//! Automatic fleet membership routes (`/api/fleet/*`).
//!
//! `blazar rpc worker --join` calls these from every worker machine:
//! join announces endpoints, heartbeat refreshes liveness (the worker
//! self-probes and reports its devices), leave is the graceful exit.
//! The supervisor merges announced endpoints into spawn planning - see
//! `blazar_runtime::fleet`.
//!
//! Trust domain: loopback peers (the common one-box-fleet case, and
//! ssh-tunnelled remote workers) join freely unless a `fleet_token` is
//! configured; non-loopback peers must present the token or hit an
//! explicitly `fleet_open` gateway. The token compare is
//! constant-time; the registry itself is capacity-capped and validated
//! field by field (see `FleetRegistry::join`).

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use axum::Json;
use axum::extract::{ConnectInfo, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;

use crate::error_codes::BlazarCode;
use crate::proxy::openai_error_code;
use crate::state::AppState;
use blazar_runtime::rpc_fleet::RpcDeviceMemory;

/// Header the worker presents its token in (same value as the
/// `--fleet-token` flag / `BLAZAR_FLEET_TOKEN` env).
const FLEET_TOKEN_HEADER: &str = "x-blazar-fleet-token";

#[derive(Deserialize)]
pub struct JoinBody {
    worker_id: Option<String>,
    /// `host:port` RPC endpoints the worker serves (1..=4).
    endpoints: Vec<String>,
    hostname: Option<String>,
    /// Startup self-probe, when the worker already ran it.
    #[serde(default)]
    devices: Vec<DeviceBody>,
    /// Token may ride the body instead of the header (curl-friendly).
    #[serde(default)]
    token: Option<String>,
}

#[derive(Deserialize)]
pub struct HeartbeatBody {
    worker_id: String,
    #[serde(default)]
    devices: Vec<DeviceBody>,
    #[serde(default)]
    token: Option<String>,
}

#[derive(Deserialize)]
pub struct LeaveBody {
    worker_id: String,
    #[serde(default)]
    token: Option<String>,
}

#[derive(Deserialize)]
pub struct InviteBody {
    /// Lifetime of the minted code (default 10 min, capped at 1 h).
    #[serde(default)]
    ttl_secs: Option<u64>,
    /// Token may ride the body instead of the header (curl-friendly).
    #[serde(default)]
    token: Option<String>,
}

/// Default lifetime of an invite code: long enough for a human to walk
/// to the donor box, short enough that a leaked code dies quickly.
const INVITE_DEFAULT_TTL_SECS: u64 = 600;
const INVITE_MAX_TTL_SECS: u64 = 3600;

/// One-use, expiring join codes for guided onboarding (`blazar adopt` /
/// `blazar join`): a code authorizes exactly ONE join from ANY peer,
/// then dies. The master `fleet_token` never travels in an invite.
#[derive(Default)]
pub struct InviteRegistry {
    codes: std::sync::Mutex<std::collections::HashMap<String, std::time::Instant>>,
}

impl InviteRegistry {
    /// Mint a fresh 32-hex code valid for `ttl`.
    pub fn mint(&self, ttl: std::time::Duration) -> String {
        let mut bytes = [0u8; 16];
        getrandom::fill(&mut bytes).expect("OS entropy source");
        let code: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        let expires = std::time::Instant::now() + ttl;
        self.codes
            .lock()
            .expect("invite registry lock")
            .insert(code.clone(), expires);
        code
    }

    /// Redeem a code: true (and consume) exactly once while live.
    /// Expired entries are pruned inline; a second redeem of the same
    /// code fails.
    pub fn redeem(&self, presented: &str) -> bool {
        let mut codes = self.codes.lock().expect("invite registry lock");
        codes.retain(|_, expires| *expires > std::time::Instant::now());
        codes.remove(presented).is_some()
    }
}

#[derive(Deserialize, Clone)]
pub struct DeviceBody {
    pub index: u32,
    pub free_bytes: u64,
    pub total_bytes: u64,
}

impl From<DeviceBody> for RpcDeviceMemory {
    fn from(d: DeviceBody) -> Self {
        Self {
            index: d.index,
            free_bytes: d.free_bytes,
            total_bytes: d.total_bytes,
        }
    }
}

/// XOR-fold equality over the token bytes: same length walk regardless
/// of where the first mismatch sits, so a remote timing probe learns
/// nothing about how many leading bytes matched.
fn token_matches(expected: &str, presented: &str) -> bool {
    if expected.len() != presented.len() {
        return false;
    }
    let mut acc = 0u8;
    for (a, b) in expected.bytes().zip(presented.bytes()) {
        acc |= a ^ b;
    }
    acc == 0
}

fn presented_token(headers: &HeaderMap, body_token: &Option<String>) -> String {
    headers
        .get(FLEET_TOKEN_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .or_else(|| body_token.clone())
        .unwrap_or_default()
}

fn is_loopback(peer: SocketAddr) -> bool {
    match peer.ip() {
        IpAddr::V4(v4) => v4.is_loopback(),
        IpAddr::V6(v6) => v6.is_loopback(),
    }
}

/// 401 teaching response for unauthenticated joins.
fn token_required() -> Response {
    openai_error_code(
        BlazarCode::FleetTokenRequired,
        "fleet join refused: this gateway requires the shared fleet token - pass it with `--fleet-token` (it rides the x-blazar-fleet-token header)",
    )
}

fn gate(
    state: &AppState,
    peer: SocketAddr,
    headers: &HeaderMap,
    body_token: &Option<String>,
) -> Option<Response> {
    let cfg = state.cfg();
    let presented = presented_token(headers, body_token);
    if !cfg.fleet_token.is_empty() {
        if token_matches(&cfg.fleet_token, &presented) {
            return None;
        }
        return Some(token_required());
    }
    // No token configured: loopback trusts the box (worker and gateway
    // share an owner), remote needs an explicitly open gateway.
    if is_loopback(peer) || cfg.fleet_open {
        return None;
    }
    Some(token_required())
}

/// Join gate: the master fleet_token, a live one-use invite code, or
/// the no-token trust domain (loopback / fleet_open). An invite code
/// authorizes exactly one join from ANY peer - the kubeadm-style
/// bootstrap that makes remote `blazar adopt` possible on gateways
/// that never expose the long-lived token.
fn join_gate(
    state: &AppState,
    peer: SocketAddr,
    headers: &HeaderMap,
    body_token: &Option<String>,
) -> Option<Response> {
    let cfg = state.cfg();
    let presented = presented_token(headers, body_token);
    if !cfg.fleet_token.is_empty() {
        if token_matches(&cfg.fleet_token, &presented) || state.invites.redeem(&presented) {
            return None;
        }
        return Some(openai_error_code(
            BlazarCode::FleetTokenRequired,
            "fleet join refused: this gateway requires the shared fleet token or a live invite code - pass `--fleet-token`, or ask the gateway owner to send an invite (`blazar adopt`)",
        ));
    }
    if is_loopback(peer) || cfg.fleet_open || state.invites.redeem(&presented) {
        return None;
    }
    Some(token_required())
}

/// Session gate for heartbeat/leave: the master fleet_token, or the
/// per-worker token minted at THIS worker's join (revoked with its
/// registry record).
fn session_gate(
    state: &AppState,
    peer: SocketAddr,
    headers: &HeaderMap,
    body_token: &Option<String>,
    worker_id: &str,
) -> Option<Response> {
    if gate(state, peer, headers, body_token).is_none() {
        return None;
    }
    let presented = presented_token(headers, body_token);
    if state.sup.fleet_worker_token_matches(worker_id, &presented) {
        return None;
    }
    Some(token_required())
}

/// POST /api/fleet/join
pub async fn join(
    State(state): State<Arc<AppState>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Json(body): Json<JoinBody>,
) -> Response {
    if !state.cfg().fleet_enabled {
        return (
            StatusCode::NOT_FOUND,
            "fleet membership disabled (fleet_enabled = false) - set rpc_servers statically instead",
        )
            .into_response();
    }
    if let Some(denied) = join_gate(&state, peer, &headers, &body.token) {
        return denied;
    }
    let devices = body.devices.into_iter().map(Into::into).collect();
    match state
        .sup
        .fleet_join(body.worker_id, body.endpoints, body.hostname, devices)
    {
        Ok((id, worker_token)) => Json(serde_json::json!({
            "worker_id": id,
            // Session credential for heartbeat/leave: presented instead of
            // the master token by workers that joined via invite code.
            "worker_token": worker_token,
            "heartbeat_interval_secs": state.cfg().fleet_heartbeat_secs,
            "ttl_secs": state.cfg().fleet_ttl_secs,
        }))
        .into_response(),
        Err(teaching) => (StatusCode::UNPROCESSABLE_ENTITY, teaching).into_response(),
    }
}

/// POST /api/fleet/invite - mint a one-use join code (guided onboarding:
/// `blazar adopt` / `blazar join`). Same trust domain as the join route
/// itself: only the box (or token holders) may mint codes.
pub async fn invite(
    State(state): State<Arc<AppState>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    body: Option<Json<InviteBody>>,
) -> Response {
    if !state.cfg().fleet_enabled {
        return (
            StatusCode::NOT_FOUND,
            "fleet membership disabled (fleet_enabled = false) - set rpc_servers statically instead",
        )
            .into_response();
    }
    let body = body.map(|Json(b)| b);
    let body_token = body.as_ref().and_then(|b| b.token.clone());
    if let Some(denied) = gate(&state, peer, &headers, &body_token) {
        return denied;
    }
    let asked = body
        .and_then(|b| b.ttl_secs)
        .unwrap_or(INVITE_DEFAULT_TTL_SECS)
        .min(INVITE_MAX_TTL_SECS);
    let code = state.invites.mint(std::time::Duration::from_secs(asked));
    Json(serde_json::json!({
        "invite_code": code,
        "ttl_secs": asked,
        "note": "one-use: the first join that presents this code consumes it",
    }))
    .into_response()
}

/// POST /api/fleet/heartbeat
pub async fn heartbeat(
    State(state): State<Arc<AppState>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Json(body): Json<HeartbeatBody>,
) -> Response {
    if !state.cfg().fleet_enabled {
        return (
            StatusCode::NOT_FOUND,
            "fleet membership disabled (fleet_enabled = false)",
        )
            .into_response();
    }
    if let Some(denied) = session_gate(&state, peer, &headers, &body.token, &body.worker_id) {
        return denied;
    }
    let devices = body.devices.into_iter().map(Into::into).collect();
    match state.sup.fleet_heartbeat(&body.worker_id, devices) {
        Ok(()) => Json(serde_json::json!({ "ok": true })).into_response(),
        Err(teaching) => (StatusCode::NOT_FOUND, teaching).into_response(),
    }
}

/// POST /api/fleet/leave
pub async fn leave(
    State(state): State<Arc<AppState>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Json(body): Json<LeaveBody>,
) -> Response {
    if !state.cfg().fleet_enabled {
        return (
            StatusCode::NOT_FOUND,
            "fleet membership disabled (fleet_enabled = false)",
        )
            .into_response();
    }
    if let Some(denied) = session_gate(&state, peer, &headers, &body.token, &body.worker_id) {
        return denied;
    }
    // Unknown id = already gone: an idempotent 200 beats a 404 for a
    // shutdown path that races its own TTL expiry.
    let _ = state.sup.fleet_leave(&body.worker_id);
    Json(serde_json::json!({ "ok": true })).into_response()
}

/// GET /api/fleet/workers - pruned registry snapshot.
pub async fn workers(State(state): State<Arc<AppState>>) -> Response {
    Json(serde_json::json!({
        "workers": state.sup.fleet_workers(),
        "heartbeat_interval_secs": state.cfg().fleet_heartbeat_secs,
        "ttl_secs": state.cfg().fleet_ttl_secs,
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit__token_matches__full_length_compare() {
        assert!(token_matches("s3cret", "s3cret"));
        assert!(!token_matches("s3cret", "s3creT"));
        assert!(!token_matches("s3cret", "s3cre"));
        assert!(!token_matches("", "x"));
        assert!(token_matches("", ""));
    }

    #[test]
    fn unit__loopback_gate__loopback_free_remote_needs_token() {
        let lo = "127.0.0.1:50000".parse::<SocketAddr>().unwrap();
        let remote = "10.0.0.9:50000".parse::<SocketAddr>().unwrap();
        assert!(is_loopback(lo));
        assert!(!is_loopback(remote));
        assert!(is_loopback("[::1]:50000".parse::<SocketAddr>().unwrap()));
    }

    #[test]
    fn unit__presented_token__header_wins_then_body_then_empty() {
        let mut headers = HeaderMap::new();
        assert_eq!(presented_token(&headers, &None), "");
        assert_eq!(
            presented_token(&headers, &Some("bodytok".into())),
            "bodytok"
        );
        headers.insert(
            "x-blazar-fleet-token",
            axum::http::HeaderValue::from_str("headtok").unwrap(),
        );
        assert_eq!(
            presented_token(&headers, &Some("bodytok".into())),
            "headtok"
        );
        assert_eq!(presented_token(&headers, &None), "headtok");
    }

    #[test]
    fn unit__device_body__maps_to_wire_type_losslessly() {
        let d = DeviceBody {
            index: 1,
            free_bytes: 7 << 20,
            total_bytes: 8 << 20,
        };
        let wire = RpcDeviceMemory::from(d);
        assert_eq!(
            (wire.index, wire.free_bytes, wire.total_bytes),
            (1, 7 << 20, 8 << 20)
        );
    }

    #[test]
    fn unit__invite_registry__mint_redeem_exactly_once() {
        let reg = InviteRegistry::default();
        let code = reg.mint(std::time::Duration::from_secs(300));
        assert_eq!(code.len(), 32);
        assert!(reg.redeem(&code), "live code redeems");
        assert!(!reg.redeem(&code), "same code is dead (one-use)");
        assert!(
            !reg.redeem("00000000000000000000000000000000"),
            "unknown code fails"
        );
    }

    #[test]
    fn unit__invite_registry__expired_code_refused() {
        let reg = InviteRegistry::default();
        let code = reg.mint(std::time::Duration::from_millis(1));
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(!reg.redeem(&code), "expired code must not authorize a join");
    }
}
