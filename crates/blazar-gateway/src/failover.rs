//! Ordered failover chains: a virtual model alias served by the first
//! healthy target, escalating on failure with anti-flap benching.
//!
//! Composition over duplication: each target is either a local model or
//! a `[[remotes]]` entry, so chains reuse the existing serving and
//! forwarding lanes verbatim (the chain resolves to a plain model string
//! — `model` for local hops, `remote:model` for remote hops — and the
//! lanes route it exactly as a direct request would). No new forwarding
//! code, one escalation loop per lane.
//!
//! Semantics:
//! * Resolution is sticky-but-ordered: the first non-benched target
//!   from index 0 wins; `sticky` remembers the last target that served.
//! * A failing target is benched for `min_residence_secs` (anti-flap);
//!   escalation tries the next eligible target. All targets benched is
//!   a full-outage posture, not a flap — the chain serves the sticky
//!   target anyway (bench expiry is a flap guard, not an availability
//!   guard).
//! * `pin` forces a target regardless of bench state; `unpin` clears.
//! * Switches publish `FailoverSwitched` on the event bus and land in
//!   the chain's bounded history (`/api/failover/<name>`).

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use axum::response::IntoResponse;
use blazar_core::config::{FailoverChain, FailoverTarget};
use serde_json::json;

/// Internal recursion marker: the attempt index currently in flight.
/// Stripped before any child/remote forward (proxy `STRIP_REQUEST`).
pub const ATTEMPT_HEADER: &str = "x-blazar-failover-attempt";

/// Bounded per-chain switch history (newest last).
const HISTORY_CAP: usize = 32;

/// One resolved chain attempt: which target serves this request and
/// what model string the lane should route.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attempt {
    pub chain: String,
    pub index: usize,
    /// Model string for the serving lane: bare model (local hop) or
    /// `remote:<name>:<model>` (remote hop — the existing prefix the
    /// remotes lane already routes).
    pub serve_model: String,
    /// Human-readable target label for headers/status (`local:<model>`
    /// or `remote:<name>/<model>`).
    pub label: String,
}

#[derive(Debug, Default)]
struct ChainState {
    /// Index of the last target that successfully served.
    sticky: usize,
    /// Forced target index (pin); `None` = automatic resolution.
    pinned: Option<usize>,
    /// Per-target bench expiry (`min_residence_secs` after a failure).
    benched_until: Vec<Option<Instant>>,
    switches: u64,
    history: VecDeque<serde_json::Value>,
}

impl ChainState {
    fn new(target_count: usize) -> Self {
        Self {
            sticky: 0,
            pinned: None,
            benched_until: vec![None; target_count],
            switches: 0,
            history: VecDeque::new(),
        }
    }
}

/// Runtime state of every configured chain, keyed by alias.
pub struct Registry {
    chains: HashMap<String, FailoverChain>,
    state: Mutex<HashMap<String, ChainState>>,
}

/// Outcome of recording a failure: either the next attempt to try or
/// `None` when no other eligible target exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Switch {
    pub attempt: Attempt,
    pub benched_index: usize,
}

impl Registry {
    /// Build from the parsed `[[failover]]` config. Config validation
    /// (unique names/aliases, non-empty targets, known remotes) happens
    /// at load; the registry trusts it.
    #[must_use]
    pub fn from_config(chains: &[FailoverChain]) -> Self {
        let state = chains
            .iter()
            .map(|c| (c.alias.clone(), ChainState::new(c.targets.len())))
            .collect();
        Self {
            chains: chains
                .iter()
                .map(|c| (c.alias.clone(), c.clone()))
                .collect(),
            state: Mutex::new(state),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.chains.is_empty()
    }

    /// Resolve a chain alias to the serving attempt. Non-aliases and
    /// unknown aliases return `None` (the lane serves normally).
    pub fn plan(&self, alias: &str) -> Option<Attempt> {
        let chain = self.chains.get(alias)?;
        let guard = self.state.lock().ok()?;
        let st = guard.get(alias)?;
        Some(Self::plan_locked(chain, st))
    }

    fn plan_locked(chain: &FailoverChain, st: &ChainState) -> Attempt {
        let index =
            st.pinned
                .unwrap_or_else(|| match st.benched_until.iter().position(Option::is_none) {
                    Some(i) => i,
                    // Every target benched: serve the sticky target — a
                    // fully-benched chain must stay available, the bench
                    // is a flap guard rather than an availability guard.
                    None => st.sticky,
                });
        attempt_for(chain, index)
    }

    /// Bench `index` for the chain's `min_residence_secs` and plan the
    /// next eligible attempt. Returns `None` when the failed target
    /// was the only eligible one (the caller returns the failure as-is).
    #[must_use]
    pub fn record_failure(&self, alias: &str, index: usize) -> Option<Switch> {
        let chain = self.chains.get(alias)?;
        let mut states = self.state.lock().ok()?;
        let st = states.get_mut(alias)?;
        let bench = Duration::from_secs(chain.min_residence_secs);
        if st.benched_until.len() > index {
            st.benched_until[index] = Some(Instant::now() + bench);
        }
        if let Some(p) = st.pinned
            && p == index
        {
            // A pinned target that fails keeps serving (explicit user
            // choice outranks availability) — no switch to record.
            return None;
        }
        let next = st
            .benched_until
            .iter()
            .position(Option::is_none)
            .map(|i| attempt_for(chain, i))?;
        Some(Switch {
            attempt: next,
            benched_index: index,
        })
    }

    /// Mark `index` as the serving target; the first serve on a new
    /// index is a switch (recorded in history, published by the caller).
    /// Returns the switch record when one happened.
    #[must_use]
    pub fn record_success(&self, alias: &str, index: usize) -> Option<(usize, usize)> {
        let mut guard = self.state.lock().ok()?;
        let st = guard.get_mut(alias)?;
        let from = st.sticky;
        if from == index {
            return None;
        }
        st.sticky = index;
        st.switches += 1;
        st.history.push_back(json!({
            "from": from,
            "to": index,
            "at": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
        }));
        if st.history.len() > HISTORY_CAP {
            st.history.pop_front();
        }
        Some((from, index))
    }

    /// Pin a chain to a target index (fails when out of range).
    pub fn pin(&self, alias: &str, index: usize) -> Result<(), String> {
        let mut states = self.state.lock().map_err(|_| "registry poisoned")?;
        let st = states
            .get_mut(alias)
            .ok_or_else(|| format!("unknown failover chain {alias:?}"))?;
        if index >= st.benched_until.len() {
            return Err(format!(
                "target index {index} out of range (chain has {} targets)",
                st.benched_until.len()
            ));
        }
        st.pinned = Some(index);
        Ok(())
    }

    pub fn unpin(&self, alias: &str) -> Result<(), String> {
        let mut states = self.state.lock().map_err(|_| "registry poisoned")?;
        states
            .get_mut(alias)
            .ok_or_else(|| format!("unknown failover chain {alias:?}"))?
            .pinned = None;
        Ok(())
    }

    /// Full status of every chain (for `GET /api/failover`).
    pub fn status_json(&self) -> serde_json::Value {
        let chains: Vec<serde_json::Value> = self
            .chains
            .keys()
            .filter_map(|alias| self.chain_json(alias))
            .collect();
        json!({ "chains": chains })
    }

    /// One chain's status (for `GET /api/failover/<name>`); accepts
    /// either the chain name or its alias.
    pub fn chain_json(&self, key: &str) -> Option<serde_json::Value> {
        let alias = self
            .chains
            .values()
            .find(|c| c.name == key)
            .map(|c| c.alias.clone())
            .or_else(|| self.chains.contains_key(key).then(|| key.to_string()))?;
        let chain = &self.chains[&alias];
        let st = self.state.lock().ok()?;
        let st = st.get(&alias)?;
        let now = Instant::now();
        let targets: Vec<serde_json::Value> = chain
            .targets
            .iter()
            .enumerate()
            .map(|(i, t)| {
                json!({
                    "index": i,
                    "label": target_label(t),
                    "benched_for_secs": st.benched_until.get(i)
                        .and_then(|b| b.map(|until| until.saturating_duration_since(now).as_secs()))
                        .unwrap_or(0),
                })
            })
            .collect();
        Some(json!({
            "name": chain.name,
            "alias": chain.alias,
            "min_residence_secs": chain.min_residence_secs,
            "targets": targets,
            "sticky": st.sticky,
            "pinned": st.pinned,
            "switches": st.switches,
            "history": st.history.iter().cloned().collect::<Vec<_>>(),
        }))
    }
}

fn attempt_for(chain: &FailoverChain, index: usize) -> Attempt {
    let t = &chain.targets[index];
    Attempt {
        chain: chain.name.clone(),
        index,
        serve_model: serve_model(t),
        label: target_label(t),
    }
}

/// The model string the serving lane routes: bare model for local
/// hops, `remote:<name>:<model>` for remote hops (the existing prefix
/// convention the remotes lane resolves).
fn serve_model(t: &FailoverTarget) -> String {
    match &t.remote {
        Some(r) => format!("{r}:{}", t.model),
        None => t.model.clone(),
    }
}

fn target_label(t: &FailoverTarget) -> String {
    match &t.remote {
        Some(r) => format!("remote:{r}/{}", t.model),
        None => format!("local:{}", t.model),
    }
}

// ---------- admin plane ----------

/// GET /api/failover — every chain's live status (targets, bench
/// state, sticky/pinned, switch history).
pub async fn list(
    axum::extract::State(state): axum::extract::State<std::sync::Arc<crate::state::AppState>>,
) -> axum::response::Response {
    axum::Json(state.failover.status_json()).into_response()
}

/// GET /api/failover/{chain} — one chain by name OR alias.
pub async fn detail(
    axum::extract::State(state): axum::extract::State<std::sync::Arc<crate::state::AppState>>,
    axum::extract::Path(chain): axum::extract::Path<String>,
) -> axum::response::Response {
    match state.failover.chain_json(&chain) {
        Some(v) => axum::Json(v).into_response(),
        None => crate::proxy::openai_error(
            404,
            &format!(
                "unknown failover chain {chain:?} — GET /api/failover lists every configured chain"
            ),
        ),
    }
}

/// POST /api/failover/{chain}/pin {"target": <index>} — force a target
/// (immune to benching; even a failing pinned target keeps serving).
pub async fn pin(
    axum::extract::State(state): axum::extract::State<std::sync::Arc<crate::state::AppState>>,
    axum::extract::Path(chain): axum::extract::Path<String>,
    body: axum::body::Bytes,
) -> axum::response::Response {
    let target = match serde_json::from_slice::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| v.get("target").and_then(serde_json::Value::as_u64))
    {
        Some(t) => usize::try_from(t).unwrap_or(usize::MAX),
        None => {
            return crate::proxy::openai_error(
                400,
                "body must be {\"target\": <index>} with the target's index from /api/failover",
            );
        }
    };
    // Accept name or alias for the chain.
    let alias = match state.failover.chain_json(&chain) {
        Some(v) => v["alias"].as_str().unwrap_or_default().to_string(),
        None => {
            return crate::proxy::openai_error(404, &format!("unknown failover chain {chain:?}"));
        }
    };
    match state.failover.pin(&alias, target) {
        Ok(()) => axum::Json(serde_json::json!({"pinned": target, "chain": alias})).into_response(),
        Err(e) => crate::proxy::openai_error(400, &e),
    }
}

/// POST /api/failover/{chain}/unpin — return the chain to automatic
/// resolution.
pub async fn unpin(
    axum::extract::State(state): axum::extract::State<std::sync::Arc<crate::state::AppState>>,
    axum::extract::Path(chain): axum::extract::Path<String>,
) -> axum::response::Response {
    let alias = match state.failover.chain_json(&chain) {
        Some(v) => v["alias"].as_str().unwrap_or_default().to_string(),
        None => {
            return crate::proxy::openai_error(404, &format!("unknown failover chain {chain:?}"));
        }
    };
    match state.failover.unpin(&alias) {
        Ok(()) => axum::Json(serde_json::json!({"pinned": null, "chain": alias})).into_response(),
        Err(e) => crate::proxy::openai_error(400, &e),
    }
}

/// Stamp chain provenance on a served response: which chain, which
/// target index, which concrete model. Headers ride every dialect.
pub fn stamp(resp: &mut axum::response::Response, attempt: &Attempt) {
    if let Ok(v) = axum::http::HeaderValue::from_str(&attempt.chain) {
        resp.headers_mut().insert("x-blazar-failover", v);
    }
    if let Ok(v) = axum::http::HeaderValue::from_str(&attempt.label) {
        resp.headers_mut().insert("x-blazar-served-model", v);
    }
}

/// Can this response trigger escalation? Success and caller errors
/// that any target would repeat pass through; gateway infrastructure
/// failures escalate: 5xx, unknown-model 404, and gate-class 400s
/// (fit refusal, spawn errors — our error bodies carry the
/// `blazar_error` type). Gate-class detection buffers the (small)
/// error body; anything larger than the cap passes through unbuffered.
/// Bodies that already streamed cannot be retried — the sentinel and
/// evict lanes own mid-stream failures.
pub async fn escalate_response(resp: &mut axum::response::Response) -> bool {
    let status = resp.status().as_u16();
    if status >= 500 || status == 404 {
        return true;
    }
    if status != 400 {
        return false;
    }
    let taken = std::mem::take(resp);
    let (parts, body) = taken.into_parts();
    if let Ok(bytes) = axum::body::to_bytes(body, 8 * 1024).await {
        let esc = bytes
            .windows(b"blazar_error".len())
            .any(|w| w == b"blazar_error");
        *resp = axum::response::Response::from_parts(parts, axum::body::Body::from(bytes));
        esc
    } else {
        *resp = axum::response::Response::from_parts(parts, axum::body::Body::empty());
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blazar_core::config::{FailoverChain, FailoverTarget};

    fn chain() -> FailoverChain {
        FailoverChain {
            name: "c1".into(),
            alias: "assistant".into(),
            min_residence_secs: 30,
            targets: vec![
                FailoverTarget {
                    remote: None,
                    model: "big:q4".into(),
                },
                FailoverTarget {
                    remote: None,
                    model: "small:q4".into(),
                },
            ],
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__registry__plan_resolves_first_target_and_ignores_unknowns() {
        let reg = Registry::from_config(&[chain()]);
        let a = reg.plan("assistant").unwrap();
        assert_eq!(a.index, 0);
        assert_eq!(a.serve_model, "big:q4");
        assert_eq!(a.label, "local:big:q4");
        assert!(reg.plan("not-a-chain").is_none());
        assert!(reg.plan("big:q4").is_none());
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__registry__failure_benches_and_advances_then_expires() {
        let reg = Registry::from_config(&[chain()]);
        let sw = reg.record_failure("assistant", 0).unwrap();
        assert_eq!(sw.benched_index, 0);
        assert_eq!(sw.attempt.index, 1);
        // Benched target 0 stays out of rotation.
        assert_eq!(reg.plan("assistant").unwrap().index, 1);
        // Expiry is time-based; simulate by rebuilding without the bench.
        let reg2 = Registry::from_config(&[chain()]);
        assert_eq!(reg2.plan("assistant").unwrap().index, 0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__registry__last_failure_has_no_switch() {
        let reg = Registry::from_config(&[chain()]);
        assert!(reg.record_failure("assistant", 0).is_some());
        assert!(reg.record_failure("assistant", 1).is_none());
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__registry__all_benched_serves_sticky() {
        let reg = Registry::from_config(&[chain()]);
        let _ = reg.record_failure("assistant", 0).unwrap();
        let _ = reg.record_success("assistant", 1);
        let _ = reg.record_failure("assistant", 1);
        // Both benched: sticky (1) keeps serving — availability wins.
        assert_eq!(reg.plan("assistant").unwrap().index, 1);
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__registry__success_records_switch_once() {
        let reg = Registry::from_config(&[chain()]);
        assert_eq!(reg.record_success("assistant", 1), Some((0, 1)));
        assert_eq!(reg.record_success("assistant", 1), None);
        let cj = reg.chain_json("c1").unwrap();
        assert_eq!(cj["sticky"], 1);
        assert_eq!(cj["switches"], 1);
        assert_eq!(cj["history"].as_array().unwrap().len(), 1);
        // Name AND alias both resolve.
        assert!(reg.chain_json("assistant").is_some());
        assert!(reg.chain_json("nope").is_none());
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__registry__pin_overrides_bench_and_failure() {
        let reg = Registry::from_config(&[chain()]);
        reg.pin("assistant", 1).unwrap();
        assert_eq!(reg.plan("assistant").unwrap().index, 1);
        // A failing PINNED target stays pinned — no switch recorded.
        assert!(reg.record_failure("assistant", 1).is_none());
        assert!(reg.pin("assistant", 9).is_err());
        reg.unpin("assistant").unwrap();
        assert_eq!(reg.plan("assistant").unwrap().index, 0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__registry__remote_target_composes_remotes_prefix() {
        let c = FailoverChain {
            name: "r".into(),
            alias: "via-remote".into(),
            min_residence_secs: 30,
            targets: vec![FailoverTarget {
                remote: Some("workstation".into()),
                model: "qwen:q4".into(),
            }],
        };
        let reg = Registry::from_config(&[c]);
        let a = reg.plan("via-remote").unwrap();
        assert_eq!(a.serve_model, "workstation:qwen:q4");
        assert_eq!(a.label, "remote:workstation/qwen:q4");
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__escalate_response__gate_400s_5xx_and_404_only() {
        // Status-only fast paths.
        let mut r = axum::response::Response::builder()
            .status(502)
            .body(axum::body::Body::empty())
            .unwrap();
        assert!(futures::executor::block_on(escalate_response(&mut r)));
        let mut r = axum::response::Response::builder()
            .status(404)
            .body(axum::body::Body::empty())
            .unwrap();
        assert!(futures::executor::block_on(escalate_response(&mut r)));
        let mut r = axum::response::Response::builder()
            .status(429)
            .body(axum::body::Body::empty())
            .unwrap();
        assert!(!futures::executor::block_on(escalate_response(&mut r)));
        // Gate-class 400 (fit refusal shape) escalates; caller 400 does not.
        let mut r = axum::response::Response::builder()
            .status(400)
            .body(axum::body::Body::from(
                r#"{"error":{"code":400,"message":"profile: pinned num_ctx cannot fit","type":"blazar_error"}}"#,
            ))
            .unwrap();
        assert!(futures::executor::block_on(escalate_response(&mut r)));
        // Body must survive the peek intact.
        let bytes = futures::executor::block_on(async {
            axum::body::to_bytes(std::mem::take(r.body_mut()), 64 * 1024).await
        })
        .unwrap();
        assert!(bytes.windows(6).any(|w| w == b"blazar"));
        let mut r = axum::response::Response::builder()
            .status(400)
            .body(axum::body::Body::from(
                r#"{"error":{"message":"bad metadata"}}"#,
            ))
            .unwrap();
        assert!(!futures::executor::block_on(escalate_response(&mut r)));
    }
}
