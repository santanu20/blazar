//! Remote engine instances: `[[remotes]]` entries name external
//! OpenAI-compatible servers (another blazar, vLLM, an MLX server,
//! llamactl — anything speaking `/v1/*`). Requests for
//! `<remote-name>:<model>` route there instead of spawning a local
//! child; nothing local is loaded, quotas still apply at the gateway.

use axum::body::Body;
use axum::response::{IntoResponse, Response};

use blazar_core::{Config, Remote};

use crate::state::AppState;

/// Split `"<remote>:<model>"` when the prefix names a configured
/// remote. Returns (remote, stripped-model).
#[must_use]
pub fn split_remote<'a>(model: &'a str, cfg: &'a Config) -> Option<(&'a Remote, &'a str)> {
    let (name, rest) = model.split_once(':')?;
    let remote = cfg.remotes.iter().find(|r| r.name == name)?;
    Some((remote, rest))
}

/// Consecutive failures before a remote is marked down (circuit open).
const REMOTE_MARK_DOWN_FAILS: u32 = 3;
/// How long a marked-down remote is skipped before a half-open probe.
const REMOTE_COOLDOWN: std::time::Duration = std::time::Duration::from_secs(30);

/// Circuit + load state for one remote pool member (C2/C3), plus the
/// v0.16 capacity-aware scheduling hints.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct RemoteHealth {
    pub consec_failures: u32,
    /// `Some(t)` = marked down until `t` (requests skip it, then one
    /// half-open probe passes through).
    pub down_until: Option<std::time::Instant>,
    pub in_flight: u32,
    /// EWMA of observed time-to-response-head in milliseconds (ok
    /// requests only). `None` until the first success — the scheduler
    /// then falls back to a neutral default rather than guessing.
    pub ttft_ewma_ms: Option<f64>,
}

pub(crate) fn health_key(remote: &Remote) -> String {
    format!("{}|{}", remote.name, remote.url)
}

/// Bound on the C4 prefix→remote stickiness table: one entry per
/// distinct conversation prefix; arbitrary eviction keeps it bounded.
const REMOTE_AFFINITY_CAP: usize = 4096;

/// C4: fold pool name + conversation prefix into one affinity key.
fn affinity_key(pool_name: &str, prefix: &blazar_runtime::PrefixKey) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in pool_name
        .as_bytes()
        .iter()
        .chain(&prefix.sys.to_le_bytes())
        .chain(&prefix.convo.to_le_bytes())
    {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Remember which remote served this prefix (called on success).
fn bind_remote(
    map: &std::sync::Mutex<std::collections::HashMap<u64, String>>,
    key: u64,
    hkey: &str,
) {
    let mut m = map.lock().expect("remote_affinity lock poisoned");
    if m.len() >= REMOTE_AFFINITY_CAP
        && !m.contains_key(&key)
        && let Some(evict) = m.keys().next().copied()
    {
        m.remove(&evict);
    }
    m.insert(key, hkey.to_string());
}

/// Forget a prefix binding when it points at the failed remote.
fn unbind_remote(
    map: &std::sync::Mutex<std::collections::HashMap<u64, String>>,
    key: u64,
    hkey: &str,
) {
    let mut m = map.lock().expect("remote_affinity lock poisoned");
    if m.get(&key).is_some_and(|k| k == hkey) {
        m.remove(&key);
    }
}

/// RAII in-flight lease: bumped by `select_remote`, released on drop.
/// Released when the caller's response HEADERS are ready (forward
/// functions return at header time); body streaming continues after —
/// header-time is the meaningful queue signal for load balancing.
pub struct RemoteLease {
    map: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, RemoteHealth>>>,
    key: String,
}

impl Drop for RemoteLease {
    fn drop(&mut self) {
        if let Ok(mut m) = self.map.lock()
            && let Some(h) = m.get_mut(&self.key)
        {
            h.in_flight = h.in_flight.saturating_sub(1);
        }
    }
}

/// Select the remote for a `<remote>:<model>` request: filter the
/// name-pool down to live members (marked-down members are skipped —
/// all-down returns a 503 teaching error with the retry window), then
/// pick the least-busy (C3). With a conversation `prefix` hint (C4),
/// prefer the live pool member that last served this prefix — its KV
/// cache holds the conversation — falling back to least-busy. Returns
/// the remote, the stripped model, an in-flight lease and the affinity
/// key to bind/unbind on the result.
#[allow(clippy::type_complexity, clippy::result_large_err)] // gateway error currency
pub fn select_remote<'a>(
    state: &'a AppState,
    model: &'a str,
    prefix: Option<&blazar_runtime::PrefixKey>,
) -> Result<(&'a Remote, &'a str, RemoteLease, Option<u64>), Response> {
    let Some((name, rest)) = model.split_once(':') else {
        return Err(crate::proxy::openai_error(
            400,
            "model has no remote prefix",
        ));
    };
    let now = std::time::Instant::now();
    // R4: capacity snapshots are plain clones taken before the health
    // lock — the scorer never nests locks.
    let caps: std::collections::HashMap<String, PeerCapacity> = state
        .remote_capacity
        .lock()
        .expect("remote_capacity lock poisoned")
        .clone();
    let mut map = state
        .remote_health
        .lock()
        .expect("remote_health lock poisoned");
    let pool: Vec<&Remote> = state
        .config
        .remotes
        .iter()
        .filter(|r| r.name == name)
        .collect();
    if pool.is_empty() {
        return Err(crate::proxy::openai_error(
            400,
            &format!("unknown remote {name:?}; check [[remotes]] in config"),
        ));
    }
    let live: Vec<(&Remote, RemoteHealth)> = pool
        .iter()
        .filter_map(|r| {
            let h = map.get(&health_key(r)).copied().unwrap_or_default();
            let down = h.down_until.is_some_and(|t| t > now);
            (!down).then_some((*r, h))
        })
        .collect();
    if live.is_empty() {
        let soonest = pool
            .iter()
            .filter_map(|r| map.get(&health_key(r)).and_then(|h| h.down_until))
            .min()
            .unwrap_or(now);
        let secs = soonest.saturating_duration_since(now).as_secs().max(1);
        return Err(crate::proxy::openai_error(
            503,
            &format!(
                "remote {name:?} marked down (circuit open); retry in ~{secs}s or check the remote"
            ),
        ));
    }
    // C4: sticky prefix affinity — the live member holding this
    // conversation's KV wins; without a hint (or after eviction/circuit)
    // it is plain least-busy.
    let akey = prefix.map(|p| affinity_key(name, p));
    let sticky = akey.and_then(|k| {
        state
            .remote_affinity
            .lock()
            .expect("remote_affinity lock poisoned")
            .get(&k)
            .cloned()
    });
    let chosen = sticky.as_ref().and_then(|want| {
        live.iter()
            .find(|(r, _)| health_key(r) == *want)
            .map(|(r, _)| *r)
    });
    // No sticky hit: capacity-aware ranking (warm tier beats catalog,
    // queue wait beats idle-cold, then most free VRAM). `rest` is what
    // the peer actually serves, so the resident lookup keys on it.
    let remote = chosen.unwrap_or_else(|| {
        live.iter()
            .min_by_key(|(r, h)| score_peer(rest, h, caps.get(&health_key(r))))
            .map(|(r, _)| *r)
            .expect("live pool non-empty")
    });
    let key = health_key(remote);
    map.entry(key.clone()).or_default().in_flight += 1;
    let lease = RemoteLease {
        map: std::sync::Arc::clone(&state.remote_health),
        key,
    };
    Ok((remote, rest, lease, akey))
}

/// Record a forward result: success resets the circuit; a failure
/// (connect error or 5xx) increments, and `REMOTE_MARK_DOWN_FAILS` in a
/// row marks the remote down for the cooldown window.
pub fn note_remote_result(state: &AppState, remote: &Remote, ok: bool) {
    let key = health_key(remote);
    let mut map = state
        .remote_health
        .lock()
        .expect("remote_health lock poisoned");
    let h = map.entry(key).or_default();
    if ok {
        if h.consec_failures > 0 || h.down_until.is_some() {
            tracing::info!(target: "blazar::remotes", remote = %remote.name, "remote recovered");
        }
        h.consec_failures = 0;
        h.down_until = None;
        return;
    }
    h.consec_failures += 1;
    if h.consec_failures >= REMOTE_MARK_DOWN_FAILS {
        h.down_until = Some(std::time::Instant::now() + REMOTE_COOLDOWN);
        tracing::warn!(
            target: "blazar::remotes",
            remote = %remote.name,
            url = %remote.url,
            "remote marked down for {}s after {fails} consecutive failures",
            REMOTE_COOLDOWN.as_secs(),
            fails = h.consec_failures
        );
    }
}

/// Fold one observed time-to-response-head (ms) into the remote's EWMA.
/// Ok requests only — failures measure the wrong thing (connect timeouts
/// would inflate decode cost). Takes the map directly (`bind_remote`
/// precedent) so it stays unit-pinnable without an `AppState`.
#[allow(clippy::implicit_hasher)] // one concrete hasher app-wide, bind_remote precedent
pub fn note_remote_latency(
    map: &std::sync::Mutex<std::collections::HashMap<String, RemoteHealth>>,
    remote: &Remote,
    ms: f64,
) {
    let mut m = map.lock().expect("remote_health lock poisoned");
    let h = m.entry(health_key(remote)).or_default();
    h.ttft_ewma_ms = Some(crate::hint_ewma(h.ttft_ewma_ms, ms));
}

/// Result bookkeeping shared by every remote lane: circuit note, C4
/// affinity bind/unbind, and the `x-blazar-remote` observability
/// header naming the member that served the request.
pub fn tag_remote_result(
    state: &AppState,
    remote: &Remote,
    akey: Option<u64>,
    mut resp: Response,
) -> Response {
    let hkey = health_key(remote);
    let ok = resp.status().as_u16() < 500;
    note_remote_result(state, remote, ok);
    if let Some(k) = akey {
        if ok {
            bind_remote(&state.remote_affinity, k, &hkey);
        } else {
            unbind_remote(&state.remote_affinity, k, &hkey);
        }
    }
    if let Ok(v) = axum::http::HeaderValue::from_str(&hkey) {
        resp.headers_mut().insert("x-blazar-remote", v);
    }
    resp
}

/// `forward_openai` + circuit bookkeeping + C4 affinity: select
/// (prefix-sticky LB + mark-down filter), forward, note the result by
/// response class (<500 = ok). Success binds the conversation prefix to
/// the remote that served it; failure unbinds.
pub async fn forward_with_health(
    state: &AppState,
    model: &str,
    method: &axum::http::Method,
    path_query: &str,
    headers: &axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    // Affinity hash over the ORIGINAL body (forward_openai rewrites the
    // model field — the prefix identity lives in the prompt fields).
    let prefix = crate::proxy::affinity_hash_bytes(&body);
    let (remote, remote_model, _lease, akey) = match select_remote(state, model, prefix.as_ref()) {
        Ok(x) => x,
        Err(resp) => return resp,
    };
    // Time-to-response-head: forward_openai returns when the upstream
    // HEADERS are ready, so this is the queue+prefill signal the
    // capacity-aware scheduler wants (buffered dialects blend completion
    // time — an acceptable smear for a hint, documented not filtered).
    let t0 = std::time::Instant::now();
    let resp = forward_openai(
        state,
        remote,
        remote_model,
        method,
        path_query,
        headers,
        body,
    )
    .await;
    let head_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if resp.status().as_u16() < 500 {
        note_remote_latency(&state.remote_health, remote, head_ms);
    }
    tag_remote_result(state, remote, akey, resp)
}

/// Forward an OpenAI-shaped request to a remote: rewrite `model` to the
/// stripped name, attach the remote's bearer key, stream the response
/// back byte-for-byte (the zero-tax contract holds — the remote is
/// someone else's engine).
pub async fn forward_openai(
    state: &AppState,
    remote: &Remote,
    remote_model: &str,
    method: &axum::http::Method,
    path_query: &str,
    headers: &axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    // Rewrite the model field (JSON bodies on the /v1 lane).
    let body = rewrite_model(body, remote_model);
    let url = format!("{}{}", remote.url.trim_end_matches('/'), path_query);
    let mut req = state.http.request(method.clone(), &url);
    for (name, value) in headers {
        if !crate::proxy::STRIP_REQUEST.contains(&name.as_str()) {
            req = req.header(name, value);
        }
    }
    if !remote.key.is_empty() {
        req = req.bearer_auth(&remote.key);
    }
    let resp = match req
        .body(reqwest::Body::wrap_stream(futures::stream::once(
            async move { Ok::<_, std::io::Error>(body) },
        )))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(target: "blazar::remotes", remote = %remote.name, "forward failed: {e:#}");
            return crate::proxy::openai_error(
                502,
                &format!("remote {:?} unreachable: {e}", remote.name),
            );
        }
    };
    let status = axum::http::StatusCode::from_u16(resp.status().as_u16())
        .unwrap_or(axum::http::StatusCode::BAD_GATEWAY);
    let mut builder = Response::builder().status(status);
    for (name, value) in resp.headers() {
        if !crate::proxy::STRIP_RESPONSE.contains(&name.as_str()) {
            builder = builder.header(name, value);
        }
    }
    let stream = futures::StreamExt::map(resp.bytes_stream(), |r| {
        r.map_err(|e| std::io::Error::other(e.to_string()))
    });
    builder.body(Body::from_stream(stream)).unwrap_or_else(|e| {
        crate::proxy::openai_error(500, &format!("remote body: {e}")).into_response()
    })
}

/// Best-effort model rewrite on a JSON body; non-JSON (multipart)
/// passes through untouched (remote gets the prefixed name — its owner
/// can name the model that way if they want).
/// Rewrite the `model` field of a JSON request body (used by the
/// remotes lane and failover chains). Non-JSON bodies pass through
/// unchanged (multipart lanes never carry a model rewrite).
pub(crate) fn rewrite_model(body: axum::body::Bytes, remote_model: &str) -> axum::body::Bytes {
    match serde_json::from_slice::<serde_json::Value>(&body) {
        Ok(mut v) => {
            if let Some(obj) = v.as_object_mut() {
                obj.insert("model".into(), serde_json::json!(remote_model));
            }
            serde_json::to_vec(&v).map_or(body, axum::body::Bytes::from)
        }
        Err(_) => body,
    }
}

/// Health probe for `blazar ps`: one GET /v1/models with a short
/// timeout; returns (ok, model-count-or-error).
pub async fn probe(state: &AppState, remote: &Remote) -> (bool, String) {
    let url = format!("{}/v1/models", remote.url.trim_end_matches('/'));
    let mut req = state.http.get(&url);
    if !remote.key.is_empty() {
        req = req.bearer_auth(&remote.key);
    }
    match req.timeout(std::time::Duration::from_secs(3)).send().await {
        Ok(r) if r.status().is_success() => match r.json::<serde_json::Value>().await {
            Ok(v) => {
                let n = v["data"].as_array().map_or(0, std::vec::Vec::len);
                (true, format!("{n} models"))
            }
            Err(_) => (true, "ok (unparsable /v1/models)".to_string()),
        },
        Ok(r) => (false, format!("HTTP {}", r.status())),
        Err(e) => (false, e.to_string()),
    }
}

/// How long a peer's `/v1/models` listing stays trusted before the
/// next fallback miss re-probes it.
const REMOTE_PRESENCE_TTL: std::time::Duration = std::time::Duration::from_secs(60);

/// A peer that just gained a model inside the TTL window would be
/// invisible until the cache lapses; one forced re-probe per remote
/// per this window closes that gap without per-request GET hammering
/// on genuinely unknown model names.
const REMOTE_PRESENCE_FORCE_THROTTLE: std::time::Duration = std::time::Duration::from_secs(10);

/// Cached `/v1/models` listing for one remote (federation presence).
#[derive(Debug, Clone)]
pub struct PeerPresence {
    pub entries: Vec<String>,
    pub fetched: std::time::Instant,
    pub last_forced: Option<std::time::Instant>,
}

impl PeerPresence {
    fn fresh(&self) -> bool {
        self.fetched.elapsed() < REMOTE_PRESENCE_TTL
    }
}

/// One GPU as a Blazar peer reports it (`GET /api/capacity`). Only the
/// scheduling-relevant fields are kept; names are display sugar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerDevice {
    pub name: String,
    pub total_vram_bytes: u64,
    pub free_vram_bytes: u64,
}

/// One resident model on a Blazar peer: the queue-depth signal the
/// gateway cannot see from outside (`in_flight` there, not ours).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerResident {
    pub model: String,
    pub engine: String,
    pub state: String,
    pub slots: Option<u32>,
    pub slots_configured: Option<u32>,
    pub in_flight: i64,
}

/// Cached `GET /api/capacity` for one remote (v0.16 federation). A
/// non-Blazar remote simply never populates this — absence is the
/// degrade signal, never an error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerCapacity {
    pub fetched: std::time::Instant,
    pub devices: Vec<PeerDevice>,
    pub residents: Vec<PeerResident>,
}

/// Bound on the per-peer capacity cache: one entry per configured
/// remote; arbitrary eviction keeps it bounded under config churn.
const REMOTE_CAPACITY_CAP: usize = 64;

/// Parse a peer's `/api/capacity` body. Pure so the shape contract is
/// unit-pinned without a live peer; malformed rows drop, a body that
/// is not the Blazar shape yields `None` (silent-absent for
/// non-Blazar remotes).
fn parse_capacity(v: &serde_json::Value) -> Option<PeerCapacity> {
    if v.get("object").and_then(|o| o.as_str()) != Some("blazar.capacity") {
        return None;
    }
    let devices: Vec<PeerDevice> = v
        .get("devices")?
        .as_array()?
        .iter()
        .filter_map(|d| {
            Some(PeerDevice {
                name: d.get("name").and_then(|n| n.as_str())?.to_string(),
                total_vram_bytes: d
                    .get("total_vram_bytes")
                    .and_then(serde_json::Value::as_u64)?,
                free_vram_bytes: d
                    .get("free_vram_bytes")
                    .and_then(serde_json::Value::as_u64)?,
            })
        })
        .collect();
    let residents: Vec<PeerResident> = v
        .get("residents")?
        .as_array()?
        .iter()
        .filter_map(|r| {
            Some(PeerResident {
                model: r.get("model").and_then(|n| n.as_str())?.to_string(),
                engine: r.get("engine").and_then(|n| n.as_str())?.to_string(),
                state: r.get("state").and_then(|n| n.as_str())?.to_string(),
                slots: r
                    .get("slots")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|n| u32::try_from(n).ok()),
                slots_configured: r
                    .get("slots_configured")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|n| u32::try_from(n).ok()),
                in_flight: r.get("in_flight").and_then(serde_json::Value::as_i64)?,
            })
        })
        .collect();
    Some(PeerCapacity {
        fetched: std::time::Instant::now(),
        devices,
        residents,
    })
}

/// One `GET /api/capacity` against a remote. Any failure (non-Blazar
/// peer, timeout, 5xx, unparseable body) is `None` — the peer stays a
/// plain OpenAI-shaped remote with no capacity signals.
async fn fetch_capacity(state: &AppState, remote: &Remote) -> Option<PeerCapacity> {
    let url = format!("{}/api/capacity", remote.url.trim_end_matches('/'));
    let mut req = state.http.get(&url);
    if !remote.key.is_empty() {
        req = req.bearer_auth(&remote.key);
    }
    let v = req
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .await
        .ok()?
        .error_for_status()
        .ok()?
        .json::<serde_json::Value>()
        .await
        .ok()?;
    parse_capacity(&v)
}

/// Fetch one remote's capacity and cache it. Observability surfaces
/// (`/api/ps`) call this alongside their presence probe so a fresh
/// gateway shows capacity signals without waiting for routing traffic;
/// routing paths get the same effect via `refresh_presence`.
pub(crate) async fn refresh_capacity_cache(state: &AppState, remote: &Remote) {
    let hkey = health_key(remote);
    if let Some(c) = fetch_capacity(state, remote).await {
        insert_capacity(&state.remote_capacity, &hkey, c);
    }
}

/// Store a fetched capacity snapshot (bounded, arbitrary eviction).
/// Takes the map directly so the bound is unit-pinnable without an
/// `AppState` (same shape as `bind_remote`).
fn insert_capacity(
    map: &std::sync::Mutex<std::collections::HashMap<String, PeerCapacity>>,
    hkey: &str,
    cap: PeerCapacity,
) {
    let mut m = map.lock().expect("remote_capacity lock poisoned");
    if m.len() >= REMOTE_CAPACITY_CAP
        && !m.contains_key(hkey)
        && let Some(evict) = m.keys().next().cloned()
    {
        m.remove(&evict);
    }
    m.insert(hkey.to_string(), cap);
}

/// Latest cached capacity for a remote (display/read path; freshness
/// is owned by the presence refresh lane that populates it).
pub fn cached_capacity(state: &AppState, remote: &Remote) -> Option<PeerCapacity> {
    state
        .remote_capacity
        .lock()
        .ok()?
        .get(&health_key(remote))
        .cloned()
}

/// Neutral time-to-head guess (ms) before the first success teaches a
/// real EWMA. Deliberately modest: it only orders peers we know nothing
/// about, and capacity signals dominate the tier long before it matters.
const REMOTE_TTFT_DEFAULT_MS: f64 = 750.0;

/// Scheduling tier: warm peers always outrank catalog-cold peers, which
/// outrank unknown-capacity peers — a warm peer's queue wait is measured
/// while a cold peer's real cost is the model load we cannot see from
/// here. Numbers, not stringly tiers, so `Ord` derives.
const REMOTE_TIER_WARM: u8 = 0;
const REMOTE_TIER_UNKNOWN: u8 = 1;
const REMOTE_TIER_COLD: u8 = 2;

/// Capacity-aware peer ranking (v0.16). Field order IS the priority:
/// tier, then estimated wait, then most free VRAM. Derived `Ord` — no
/// hand-written comparator to get wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PeerScore {
    pub(crate) tier: u8,
    pub(crate) est_wait_ms: u64,
    pub(crate) free_vram: std::cmp::Reverse<u64>,
}

/// Score one peer for serving `model`. Pure over its signal snapshots
/// so every ordering rule is unit-pinnable without a live peer.
///
/// Estimated wait = queue depth x observed time-to-head: the peer's own
/// `in_flight` for the model (from its capacity residents — R2: the
/// gateway-side lease alone misses the peer's local load) plus our
/// leases on it, divided by its slot count, times the EWMA (or the
/// neutral default before the first success).
#[must_use]
// wave counts (< 2^52 by construction) and ceiling results are exact in f64;
// the ranking math reads clearest in float waves x ttft
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)] // in_flight clamped >= 0; wave counts exact in f64
pub fn score_peer(model: &str, h: &RemoteHealth, cap: Option<&PeerCapacity>) -> PeerScore {
    let ttft_ms = h.ttft_ewma_ms.unwrap_or(REMOTE_TTFT_DEFAULT_MS).max(1.0);
    let (tier, slots, peer_in_flight, free_vram) = match cap {
        None => (REMOTE_TIER_UNKNOWN, 1u64, 0u64, 0u64),
        Some(c) => {
            let free = c
                .devices
                .iter()
                .map(|d| d.free_vram_bytes)
                .max()
                .unwrap_or(0);
            match c.residents.iter().find(|r| r.model == model) {
                Some(r) => {
                    let slots = u64::from(r.slots_configured.or(r.slots).unwrap_or(1)).max(1);
                    let busy = r.in_flight.max(0) as u64;
                    (REMOTE_TIER_WARM, slots, busy, free)
                }
                None => (REMOTE_TIER_COLD, 1, 0, free),
            }
        }
    };
    let queue = u64::from(h.in_flight) + peer_in_flight;
    let waves = queue.div_ceil(slots);
    PeerScore {
        tier,
        est_wait_ms: (waves as f64 * ttft_ms).ceil() as u64,
        free_vram: std::cmp::Reverse(free_vram),
    }
}

/// Pure gate for the forced re-probe: due when never forced or the
/// last forced probe is past the throttle window.
fn force_refresh_due(presence: Option<&PeerPresence>) -> bool {
    presence.is_none_or(|p| {
        p.last_forced
            .is_none_or(|t| t.elapsed() >= REMOTE_PRESENCE_FORCE_THROTTLE)
    })
}

/// Federation is armed: at least one remote configured and the
/// `remote_fallback` kill switch is on. One definition shared by the
/// request hook and the `/v1/models` merge — flipping the config flag
/// rolls the whole feature back to pure-prefix routing.
#[must_use]
pub fn fallback_enabled(cfg: &Config) -> bool {
    !cfg.remotes.is_empty() && cfg.remote_fallback
}

/// Does the peer catalog id `id` satisfy the locally-requested
/// `requested` name? Peer ids carry a quant tag (`qwen3:q4`), the
/// caller may request bare names or a `+adapter` variant — compare
/// exact, then each side's name-part (before `:`), then the
/// lora-stripped base on both sides. Pure.
fn peer_id_matches(id: &str, requested: &str) -> bool {
    if id == requested {
        return true;
    }
    let id_name = id.split(':').next().unwrap_or(id);
    let (req_base, _) = blazar_core::catalog::split_lora_suffix(requested);
    let req_name = req_base.split(':').next().unwrap_or(req_base);
    id_name == req_name || id == req_name
}

/// One GET `/v1/models` against a remote, parsed to model ids. Probe
/// failures return `None` — an unreachable or unparseable peer is
/// simply not a fallback candidate, never a request error.
async fn fetch_presence(state: &AppState, remote: &Remote) -> Option<Vec<String>> {
    let url = format!("{}/v1/models", remote.url.trim_end_matches('/'));
    let mut req = state.http.get(&url);
    if !remote.key.is_empty() {
        req = req.bearer_auth(&remote.key);
    }
    let entries = match req.timeout(std::time::Duration::from_secs(3)).send().await {
        Ok(r) if r.status().is_success() => match r.json::<serde_json::Value>().await {
            Ok(v) => v["data"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|m| m["id"].as_str().map(str::to_string))
                        .collect::<Vec<String>>()
                })
                .unwrap_or_default(),
            Err(_) => return None,
        },
        Ok(_) => return None,
        Err(e) => {
            tracing::warn!(target: "blazar::remotes", remote = %remote.name, "presence probe failed: {e:#}");
            return None;
        }
    };
    Some(entries)
}

/// Fetch (or reuse within TTL) one remote's model listing. Natural
/// refreshes preserve the forced-probe throttle state.
async fn refresh_presence(state: &AppState, remote: &Remote) -> Option<Vec<String>> {
    let hkey = health_key(remote);
    if let Some(p) = state.remote_presence.lock().ok()?.get(&hkey)
        && p.fresh()
    {
        return Some(p.entries.clone());
    }
    // Presence + capacity ride together: a slow-but-alive peer pays
    // max(3s, 3s) wall, never the sum (F15). Capacity failure alone
    // never fails presence — a non-Blazar peer stays routable.
    let (entries, capacity) =
        tokio::join!(fetch_presence(state, remote), fetch_capacity(state, remote));
    if let Some(c) = capacity {
        insert_capacity(&state.remote_capacity, &hkey, c);
    }
    let entries = entries?;
    if let Ok(mut m) = state.remote_presence.lock() {
        let last_forced = m.get(&hkey).and_then(|p| p.last_forced);
        m.insert(
            hkey,
            PeerPresence {
                entries: entries.clone(),
                fetched: std::time::Instant::now(),
                last_forced,
            },
        );
    }
    Some(entries)
}

/// Forced presence re-probe (bypasses the TTL cache), throttled per
/// remote to [`REMOTE_PRESENCE_FORCE_THROTTLE`]. Returns the cached
/// view when the throttle window has not elapsed.
async fn force_refresh_presence(state: &AppState, remote: &Remote) -> Option<Vec<String>> {
    let hkey = health_key(remote);
    let due = state
        .remote_presence
        .lock()
        .ok()
        .and_then(|m| force_refresh_due(m.get(&hkey)).then_some(()))
        .is_some();
    if !due {
        return state
            .remote_presence
            .lock()
            .ok()
            .and_then(|m| m.get(&hkey).map(|p| p.entries.clone()));
    }
    // Same joined refresh as the natural path: the forced probe is the
    // one that runs on a request, so concurrency matters most here.
    let (entries, capacity) =
        tokio::join!(fetch_presence(state, remote), fetch_capacity(state, remote));
    if let Some(c) = capacity {
        insert_capacity(&state.remote_capacity, &hkey, c);
    }
    let entries = entries?;
    if let Ok(mut m) = state.remote_presence.lock() {
        m.insert(
            hkey,
            PeerPresence {
                entries: entries.clone(),
                fetched: std::time::Instant::now(),
                last_forced: Some(std::time::Instant::now()),
            },
        );
    }
    Some(entries)
}

/// Remotes whose cached-or-fresh catalog lists `model`, live-filtered
/// (circuit-open peers skip) and ordered least-in-flight first, config
/// order as the stable tiebreak. First entry is the fallback target.
pub async fn peers_serving<'a>(state: &'a AppState, model: &str) -> Vec<&'a Remote> {
    let mut candidates: Vec<(&Remote, u32)> = Vec::new();
    for remote in &state.config.remotes {
        // Circuit-open peers are excluded by the same health map the
        // explicit-prefix lane uses — fallback never routes into a
        // remote the breaker already marked down.
        let live = state
            .remote_health
            .lock()
            .ok()
            .and_then(|m| {
                m.get(&health_key(remote))
                    .map(|h| h.down_until.is_none_or(|t| std::time::Instant::now() >= t))
            })
            .unwrap_or(true);
        if !live {
            continue;
        }
        if let Some(entries) = refresh_presence(state, remote).await
            && entries.iter().any(|id| peer_id_matches(id, model))
        {
            let in_flight = state
                .remote_health
                .lock()
                .ok()
                .and_then(|m| m.get(&health_key(remote)).map(|h| h.in_flight))
                .unwrap_or(0);
            candidates.push((remote, in_flight));
        }
    }
    candidates.sort_by_key(|(_, f)| *f);
    candidates.into_iter().map(|(r, _)| r).collect()
}

/// [`peers_serving`] plus one throttled forced re-probe per remote when
/// NO cached catalog claims the model: a peer that started serving the
/// model inside the presence TTL is still found (the first fallback
/// miss for that model pays the probe, unknown-name traffic never
/// hammers peers).
pub async fn peers_serving_with_refresh<'a>(state: &'a AppState, model: &str) -> Vec<&'a Remote> {
    let cached = peers_serving(state, model).await;
    if !cached.is_empty() {
        return cached;
    }
    let mut candidates: Vec<(&Remote, u32)> = Vec::new();
    for remote in &state.config.remotes {
        let live = state
            .remote_health
            .lock()
            .ok()
            .and_then(|m| {
                m.get(&health_key(remote))
                    .map(|h| h.down_until.is_none_or(|t| std::time::Instant::now() >= t))
            })
            .unwrap_or(true);
        if !live {
            continue;
        }
        if let Some(entries) = force_refresh_presence(state, remote).await
            && entries.iter().any(|id| peer_id_matches(id, model))
        {
            let in_flight = state
                .remote_health
                .lock()
                .ok()
                .and_then(|m| m.get(&health_key(remote)).map(|h| h.in_flight))
                .unwrap_or(0);
            candidates.push((remote, in_flight));
        }
    }
    candidates.sort_by_key(|(_, f)| *f);
    candidates.into_iter().map(|(r, _)| r).collect()
}

/// Lane selection for [`try_fallback_forward`] — every lane hook shares
/// the same miss-detection and peer pick; only the forward call shape
/// differs per surface.
pub enum FallbackLane<'a> {
    /// OpenAI-shape byte forward (`/v1/*` surfaces): method, path with
    /// query, headers, and the caller-original body ride
    /// [`forward_with_health`].
    OpenAi {
        method: &'a axum::http::Method,
        path: &'a str,
        headers: &'a axum::http::HeaderMap,
        body: axum::body::Bytes,
    },
    /// Ollama `/api/chat`: the parsed ollama request (plus its prefix
    /// affinity hint) rides [`ollama_chat_remote`], which owns the
    /// ollama-to-OpenAI translation and the response shaping.
    OllamaChat {
        req: &'a serde_json::Value,
        prefix: Option<blazar_runtime::PrefixKey>,
    },
}

/// Rank already-filtered peers by capacity score. Health and capacity
/// snapshots are cloned before any comparison (R4: no lock held during
/// scoring); the stable sort preserves config order as tiebreak.
pub(crate) fn rank_peers_by_capacity<'a>(
    state: &'a AppState,
    model: &str,
    peers: &[&'a Remote],
) -> Vec<&'a Remote> {
    let caps = state
        .remote_capacity
        .lock()
        .expect("remote_capacity lock poisoned")
        .clone();
    let healths = state
        .remote_health
        .lock()
        .expect("remote_health lock poisoned")
        .clone();
    let mut ranked = peers.to_vec();
    ranked.sort_by_key(|r| {
        let h = healths.get(&health_key(r)).copied().unwrap_or_default();
        score_peer(model, &h, caps.get(&health_key(r)))
    });
    ranked
}

/// Federation fallback, one shared path for every lane hook: a bare
/// model name the local store does not own (PURE not-found — an
/// ambiguous local match stays local, that is a naming problem) routes
/// to the best-ranked live peer whose catalog lists it (capacity-aware
/// since v0.16). Returns `None` when federation is off, the model
/// resolves locally, or no peer claims it — the caller's existing
/// local error path stays untouched. Key admission (scope + request
/// count) runs at the call site before this, exactly like
/// explicit-prefix routing.
pub async fn try_fallback_forward(
    state: &AppState,
    model: &str,
    lane: FallbackLane<'_>,
) -> Option<axum::response::Response> {
    if !fallback_enabled(&state.config) {
        return None;
    }
    let local_miss = state
        .with_store(|s| crate::proxy::resolve_model(s, model).err())
        .flatten()
        .is_some_and(|e| e.contains("not found") && !e.contains("ambiguous"));
    if !local_miss {
        return None;
    }
    let peers = peers_serving_with_refresh(state, model).await;
    // Capacity-aware pick among serving peers: warm/queue/VRAM ranking
    // (v0.16) replaces blind `first()`; stable sort keeps config order
    // as the tiebreak, and capacity-less peers degrade to neutral tier.
    let peer = rank_peers_by_capacity(state, model, &peers)
        .into_iter()
        .next()?;
    tracing::info!(target: "blazar::remotes", peer = %peer.name, model = %model, "fallback: routing bare model to peer");
    let prefixed = format!("{}:{}", peer.name, model);
    match lane {
        FallbackLane::OpenAi {
            method,
            path,
            headers,
            body,
        } => Some(forward_with_health(state, &prefixed, method, path, headers, body).await),
        FallbackLane::OllamaChat { req, prefix } => {
            let (remote, remote_model, _lease, akey) =
                select_remote(state, &prefixed, prefix.as_ref()).ok()?;
            let resp = ollama_chat_remote(state, remote, remote_model, req).await;
            Some(tag_remote_result(state, remote, akey, resp))
        }
    }
}

/// `/api/chat` against a remote: ollama body -> `OpenAI` -> remote ->
/// ollama shape back (non-stream JSON; stream = SSE->NDJSON with the
/// same translate helpers the local path uses).
#[allow(clippy::too_many_lines, clippy::items_after_statements)] // mirrors the local translate path 1:1
pub async fn ollama_chat_remote(
    state: &AppState,
    remote: &Remote,
    remote_model: &str,
    req: &serde_json::Value,
) -> Response {
    let (mut openai_req, _num_ctx) = match crate::translate::chat_to_openai(req) {
        Ok(r) => r,
        Err(e) => return crate::proxy::openai_error(400, &e),
    };
    if let Some(obj) = openai_req.as_object_mut() {
        obj.insert("model".into(), serde_json::json!(remote_model));
    }
    let stream = req
        .get("stream")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true);
    if stream && let Some(obj) = openai_req.as_object_mut() {
        obj.insert(
            "stream_options".into(),
            serde_json::json!({"include_usage": true}),
        );
    }
    let url = format!("{}/v1/chat/completions", remote.url.trim_end_matches('/'));
    let mut http = state.http.post(&url).json(&openai_req);
    if !remote.key.is_empty() {
        http = http.bearer_auth(&remote.key);
    }
    let resp = match http.send().await {
        Ok(r) => r,
        Err(e) => {
            return crate::proxy::openai_error(
                502,
                &format!("remote {:?} unreachable: {e}", remote.name),
            );
        }
    };
    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        return crate::proxy::openai_error(status, &format!("remote error: {text}"));
    }
    if !stream {
        let openai: serde_json::Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => return crate::proxy::openai_error(502, &format!("bad remote response: {e}")),
        };
        return axum::Json(crate::translate::openai_chat_to_ollama(
            remote_model,
            &openai,
        ))
        .into_response();
    }
    // SSE -> ollama NDJSON (same incremental translate as the local path).
    use futures::StreamExt as _;
    // Shared clock: stamp first/last-byte offsets like the local lane so the
    // final NDJSON line carries gateway-measured timings (remotes report
    // none of their own).
    let t0 = std::time::Instant::now();
    let clock = std::sync::Arc::new(std::sync::Mutex::new((None::<u64>, 0u64)));
    let clock_map = std::sync::Arc::clone(&clock);
    let upstream = resp
        .bytes_stream()
        .map(move |chunk| {
            let elapsed = u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX);
            {
                let mut c = clock_map.lock().unwrap();
                if c.0.is_none() {
                    c.0 = Some(elapsed);
                }
                c.1 = elapsed;
            }
            chunk
        })
        .boxed();
    let model_c = remote_model.to_string();
    // F70: boundary-safe decode — chunk-split UTF-8 chars stay raw until
    // their final byte arrives.
    let ndjson = futures::stream::unfold(
        (
            upstream,
            String::new(),
            crate::translate::LineBuffer::new(),
            model_c,
            false,
            None::<serde_json::Value>,
            None::<String>,
            false,
            std::sync::Arc::clone(&clock),
            crate::translate::ToolCallAccum::default(),
        ),
        |(
            mut stream,
            mut buf,
            mut lines,
            model,
            mut done,
            mut usage,
            mut finish,
            mut usage_sent,
            clock,
            mut tool_accum,
        )| async move {
            loop {
                if done && !usage_sent {
                    // Decode window = last - first byte; total = full wall.
                    let (eval_ns, total_ns) = {
                        let c = clock.lock().unwrap();
                        match c.0 {
                            Some(f) => (Some(c.1.saturating_sub(f)), Some(c.1)),
                            None => (None, (c.1 > 0).then_some(c.1)),
                        }
                    };
                    let final_chunk = crate::translate::ollama_final_chunk(
                        &model,
                        usage.as_ref(),
                        finish.as_deref(),
                        eval_ns,
                        total_ns,
                    );
                    usage_sent = true;
                    let flush_prefix = tool_accum.flush_line(&model).unwrap_or_default();
                    return Some((
                        Ok(axum::body::Bytes::from(format!(
                            "{flush_prefix}{final_chunk}\n"
                        ))),
                        (
                            stream, buf, lines, model, done, usage, finish, usage_sent, clock,
                            tool_accum,
                        ),
                    ));
                }
                match stream.next().await {
                    Some(Ok(bytes)) => {
                        buf.push_str(&lines.feed(&bytes));
                        let (events, saw_done, consumed) = crate::translate::parse_sse(&buf);
                        buf.drain(..consumed);
                        for ev in &events {
                            if let Some(u) = ev.get("usage").filter(|u| !u.is_null()) {
                                usage = Some(u.clone());
                            }
                            if let Some(fr) = ev["choices"][0]["finish_reason"].as_str() {
                                finish = Some(fr.to_string());
                            }
                        }
                        if saw_done {
                            done = true;
                        }
                        let ndjson_lines: Vec<String> = events
                            .iter()
                            .flat_map(|ev| {
                                crate::translate::openai_chunk_to_ollama(
                                    &mut tool_accum,
                                    &model,
                                    ev,
                                )
                            })
                            .map(|v| format!("{v}\n"))
                            .collect();
                        if ndjson_lines.is_empty() {
                            continue;
                        }
                        return Some((
                            Ok(axum::body::Bytes::from(ndjson_lines.join(""))),
                            (
                                stream, buf, lines, model, done, usage, finish, usage_sent, clock,
                                tool_accum,
                            ),
                        ));
                    }
                    Some(Err(e)) => {
                        return Some((
                            Err(std::io::Error::other(e.to_string())),
                            (
                                stream, buf, lines, model, done, usage, finish, usage_sent, clock,
                                tool_accum,
                            ),
                        ));
                    }
                    None => {
                        done = true;
                        if usage_sent {
                            return None;
                        }
                    }
                }
            }
        },
    );
    Response::builder()
        .status(200)
        .header("content-type", "application/x-ndjson")
        .body(Body::from_stream(ndjson))
        .unwrap_or_else(|e| {
            crate::proxy::openai_error(500, &format!("remote stream: {e}")).into_response()
        })
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn unit__remote_lease__drop_releases_in_flight_slot() {
        let map: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, RemoteHealth>>> =
            std::sync::Arc::default();
        {
            let lease = RemoteLease {
                map: std::sync::Arc::clone(&map),
                key: "r|http://10.0.0.4:8000".into(),
            };
            // select_remote bumps in_flight when it hands out the lease.
            map.lock()
                .unwrap()
                .entry(lease.key.clone())
                .or_default()
                .in_flight += 1;
            assert_eq!(map.lock().unwrap()["r|http://10.0.0.4:8000"].in_flight, 1);
        }
        // Lease dropped at header time: the LB slot is back.
        assert_eq!(map.lock().unwrap()["r|http://10.0.0.4:8000"].in_flight, 0);
    }

    #[test]
    fn unit__split_remote__prefix_must_name_a_remote() {
        let cfg = Config {
            remotes: vec![Remote {
                name: "vllm".into(),
                url: "http://10.0.0.4:8000".into(),
                key: String::new(),
            }],
            ..Config::default()
        };
        let (r, m) = split_remote("vllm:qwen3-72b", &cfg).unwrap();
        assert_eq!(r.name, "vllm");
        assert_eq!(m, "qwen3-72b");
        assert!(split_remote("localmodel", &cfg).is_none());
        assert!(split_remote("nope:x", &cfg).is_none());
        // A model whose name contains a colon but matches nothing local
        // must not silently route — split_remote only fires on exact
        // remote-name prefixes.
        assert!(split_remote("localhost:11434", &cfg).is_none());
    }

    #[test]
    fn unit__rewrite_model__json_only() {
        let b = rewrite_model(
            axum::body::Bytes::from(r#"{"model":"vllm:x","messages":[]}"#),
            "x",
        );
        assert!(std::str::from_utf8(&b).unwrap().contains(r#""model":"x""#));
        let raw = rewrite_model(axum::body::Bytes::from_static(b"not-json"), "x");
        assert_eq!(&*raw, b"not-json");
    }

    #[test]
    fn unit__remote_affinity__key_discriminates_pool_and_convo() {
        let p = blazar_runtime::PrefixKey { sys: 1, convo: 2 };
        let a = affinity_key("vllm", &p);
        assert_eq!(a, affinity_key("vllm", &p), "stable for same inputs");
        assert_ne!(
            a,
            affinity_key("vllm", &blazar_runtime::PrefixKey { sys: 9, convo: 2 }),
            "different system prompt = different key"
        );
        assert_ne!(a, affinity_key("mlx", &p), "different pool = different key");
    }

    #[test]
    fn unit__remote_affinity__bind_sticks_unbind_only_owns() {
        let map = std::sync::Mutex::new(std::collections::HashMap::new());
        bind_remote(&map, 7, "vllm|http://a:1");
        assert_eq!(
            map.lock().unwrap().get(&7).map(String::as_str),
            Some("vllm|http://a:1")
        );
        // Unbind with a DIFFERENT remote's key must not steal the entry.
        unbind_remote(&map, 7, "vllm|http://b:2");
        assert!(
            map.lock().unwrap().contains_key(&7),
            "foreign unbind no-ops"
        );
        // Own unbind removes it.
        unbind_remote(&map, 7, "vllm|http://a:1");
        assert!(!map.lock().unwrap().contains_key(&7));
    }

    #[test]
    fn unit__remote_affinity__bounded_at_cap() {
        let map = std::sync::Mutex::new(std::collections::HashMap::new());
        for i in 0..=REMOTE_AFFINITY_CAP {
            bind_remote(&map, i as u64, "r|u");
        }
        assert!(
            map.lock().unwrap().len() <= REMOTE_AFFINITY_CAP,
            "affinity table stays bounded: {}",
            map.lock().unwrap().len()
        );
    }

    #[test]
    fn unit__peer_id_matches__exact_name_part_and_lora_stripped() {
        // Exact id match.
        assert!(peer_id_matches("qwen3:q4", "qwen3:q4"));
        // Bare request matches the peer id's quant-tagged name-part.
        assert!(peer_id_matches("qwen3:q4_0", "qwen3"));
        // Request with a quant tag matches a bare peer id.
        assert!(peer_id_matches("qwen3", "qwen3:q4_0"));
        // +adapter variant resolves to the peer's base name-part.
        assert!(peer_id_matches("qwen3:q4", "qwen3+my-lora"));
        assert!(peer_id_matches("qwen3:q4", "qwen3:q4+my-lora"));
        // Different model: no match in any form.
        assert!(!peer_id_matches("qwen3:q4", "qwen8"));
        assert!(!peer_id_matches("qwen3:q4", "qwen3-x"));
        // A colon-bearing id's name-part must not match a DIFFERENT
        // request's name-part.
        assert!(!peer_id_matches("qwen3:q4", "qwen8:q4"));
    }

    #[test]
    fn unit__peer_presence__ttl_window_boundary() {
        let fresh = PeerPresence {
            entries: vec!["m".into()],
            fetched: std::time::Instant::now(),
            last_forced: None,
        };
        assert!(fresh.fresh(), "just-fetched presence is fresh");
        let stale = PeerPresence {
            entries: vec!["m".into()],
            fetched: std::time::Instant::now()
                .checked_sub(REMOTE_PRESENCE_TTL + std::time::Duration::from_secs(1))
                .unwrap_or(std::time::Instant::now()),
            last_forced: None,
        };
        assert!(!stale.fresh(), "presence past the TTL must re-probe");
    }

    #[test]
    fn unit__parse_capacity__blazar_shape_devices_and_residents() {
        let v = serde_json::json!({
            "object": "blazar.capacity",
            "devices": [
                {"id": "cuda:0", "name": "RTX 4070", "total_vram_bytes": 8_589_934_592u64, "free_vram_bytes": 2_147_483_648u64},
                // Malformed row (missing free) drops, never panics.
                {"id": "cuda:1", "name": "half-there", "total_vram_bytes": 1u64}
            ],
            "residents": [
                {"model": "qwen3-8b", "engine": "llamacpp", "state": "ready",
                 "slots": 2, "slots_configured": 4, "in_flight": 3, "pid": 123},
                // String slots (non-Blazar drift) drops the row.
                {"model": "bad", "engine": "x", "state": "ready", "in_flight": "many"}
            ],
            "external": [], "notes": []
        });
        let c = parse_capacity(&v).expect("blazar shape parses");
        assert_eq!(c.devices.len(), 1);
        assert_eq!(c.devices[0].free_vram_bytes, 2_147_483_648);
        assert_eq!(c.residents.len(), 1);
        assert_eq!(c.residents[0].model, "qwen3-8b");
        assert_eq!(c.residents[0].slots, Some(2));
        assert_eq!(c.residents[0].slots_configured, Some(4));
        assert_eq!(c.residents[0].in_flight, 3);
    }

    #[test]
    fn unit__parse_capacity__foreign_body_is_none_not_error() {
        // A vLLM-style /v1/models body or HTML error page: absent, not
        // an error — the peer stays a plain remote.
        assert!(parse_capacity(&serde_json::json!({"data": [{"id": "m"}]})).is_none());
        assert!(parse_capacity(&serde_json::json!({"object": "other"})).is_none());
        // Missing arrays entirely: no capacity contract.
        assert!(parse_capacity(&serde_json::json!({"object": "blazar.capacity"})).is_none());
    }

    #[test]
    fn unit__insert_capacity__bounded_with_arbitrary_eviction() {
        let map: std::sync::Mutex<std::collections::HashMap<String, PeerCapacity>> =
            std::sync::Mutex::new(std::collections::HashMap::new());
        let cap = || PeerCapacity {
            fetched: std::time::Instant::now(),
            devices: vec![],
            residents: vec![],
        };
        for i in 0..REMOTE_CAPACITY_CAP {
            insert_capacity(&map, &format!("r{i}"), cap());
        }
        assert_eq!(map.lock().unwrap().len(), REMOTE_CAPACITY_CAP);
        // Existing keys still refresh at cap.
        insert_capacity(&map, "r0", cap());
        assert_eq!(map.lock().unwrap().len(), REMOTE_CAPACITY_CAP);
        // One new peer beyond the cap evicts some earlier entry.
        insert_capacity(&map, "r-new", cap());
        assert_eq!(map.lock().unwrap().len(), REMOTE_CAPACITY_CAP);
        assert!(map.lock().unwrap().contains_key("r-new"));
    }

    #[test]
    fn unit__force_refresh_due__throttled_per_remote() {
        // No presence yet: the first forced probe is due.
        assert!(force_refresh_due(None));
        // Never forced: due.
        assert!(force_refresh_due(Some(&PeerPresence {
            entries: vec![],
            fetched: std::time::Instant::now(),
            last_forced: None,
        })));
        // Forced just now: throttled.
        assert!(!force_refresh_due(Some(&PeerPresence {
            entries: vec![],
            fetched: std::time::Instant::now(),
            last_forced: Some(std::time::Instant::now()),
        })));
        // Forced past the throttle window: due again.
        assert!(force_refresh_due(Some(&PeerPresence {
            entries: vec![],
            fetched: std::time::Instant::now(),
            last_forced: std::time::Instant::now()
                .checked_sub(REMOTE_PRESENCE_FORCE_THROTTLE + std::time::Duration::from_secs(1)),
        })));
    }

    #[test]
    fn unit__fallback_enabled__gate_requires_remotes_and_flag() {
        let mut cfg = blazar_core::Config::default();
        assert!(!fallback_enabled(&cfg), "no remotes -> inert");
        cfg.remotes = vec![Remote {
            name: "peer".into(),
            url: "http://127.0.0.1:1".into(),
            key: String::new(),
        }];
        assert!(fallback_enabled(&cfg), "armed by default with a remote");
        cfg.remote_fallback = false;
        assert!(
            !fallback_enabled(&cfg),
            "kill switch rolls the feature back"
        );
    }

    #[test]
    fn unit__note_remote_latency__ewma_converges_toward_new_observations() {
        let map: std::sync::Mutex<std::collections::HashMap<String, RemoteHealth>> =
            std::sync::Mutex::new(std::collections::HashMap::new());
        let r = Remote {
            name: "peer".into(),
            url: "http://127.0.0.1:1".into(),
            key: String::new(),
        };
        note_remote_latency(&map, &r, 1000.0);
        assert_eq!(
            map.lock()
                .unwrap()
                .get(&health_key(&r))
                .unwrap()
                .ttft_ewma_ms,
            Some(1000.0),
            "first observation seeds the EWMA"
        );
        note_remote_latency(&map, &r, 500.0);
        let e = map
            .lock()
            .unwrap()
            .get(&health_key(&r))
            .unwrap()
            .ttft_ewma_ms;
        assert!((e.unwrap() - (0.7 * 1000.0 + 0.3 * 500.0)).abs() < 1e-9);
    }

    fn cap_with(residents: Vec<PeerResident>, free: u64) -> PeerCapacity {
        PeerCapacity {
            fetched: std::time::Instant::now(),
            devices: vec![PeerDevice {
                name: "gpu".into(),
                total_vram_bytes: free * 2,
                free_vram_bytes: free,
            }],
            residents,
        }
    }

    fn resident_row(
        model: &str,
        slots: Option<u32>,
        configured: Option<u32>,
        in_flight: i64,
    ) -> PeerResident {
        PeerResident {
            model: model.into(),
            engine: "llamacpp".into(),
            state: "ready".into(),
            slots,
            slots_configured: configured,
            in_flight,
        }
    }

    #[test]
    fn unit__score_peer__tier_dominates_wait_numbers() {
        let idle = RemoteHealth::default();
        // A warm peer buried in queue work still outranks an idle cold
        // peer and an unknown-capacity peer: the cold peer's real cost
        // is a model load nobody measured from here.
        let warm_busy = cap_with(vec![resident_row("m", Some(4), Some(4), 64)], 1_000_000_000);
        let cold_idle = cap_with(vec![], 20_000_000_000);
        let s_warm = score_peer("m", &idle, Some(&warm_busy));
        let s_cold = score_peer("m", &idle, Some(&cold_idle));
        let s_unknown = score_peer("m", &idle, None);
        assert!(s_warm < s_cold, "warm beats cold regardless of wait");
        assert!(
            s_unknown < s_cold,
            "unknown-capacity outranks known-cold: unknown may be warm, cold needs a load"
        );
    }

    #[test]
    fn unit__score_peer__queue_wait_math_and_ttft_sources() {
        // slots=2, peer in_flight=3, our leases=2 -> queue 5 -> 3 slot
        // waves -> wait = 3 x ttft.
        let mut h = RemoteHealth {
            in_flight: 2,
            ..Default::default()
        };
        let cap = cap_with(vec![resident_row("m", Some(2), Some(2), 3)], 0);
        let s = score_peer("m", &h, Some(&cap));
        assert_eq!(s.est_wait_ms, 3 * 750, "default ttft fills in before EWMA");
        h.ttft_ewma_ms = Some(100.0);
        let s = score_peer("m", &h, Some(&cap));
        assert_eq!(s.est_wait_ms, 300, "observed EWMA drives the estimate");
        // No resident row: cold tier, lease-only queue — two leases at
        // slots=1 are two waves of the observed ttft.
        let s_cold = score_peer("m", &h, Some(&cap_with(vec![], 0)));
        assert_eq!(s_cold.est_wait_ms, 200);
    }

    #[test]
    fn unit__score_peer__free_vram_breaks_wait_ties() {
        let h = RemoteHealth::default();
        let small = cap_with(vec![], 1_000_000_000);
        let big = cap_with(vec![], 9_000_000_000);
        assert!(
            score_peer("m", &h, Some(&big)) < score_peer("m", &h, Some(&small)),
            "equal cold waits rank by most free VRAM"
        );
    }
}
