//! Per-key gateway security tier: model scoping, rate limits
//! (rpm/tpm), daily token budgets, concurrency caps, and usage
//! accounting.
//!
//! Zero-tax contract: with no `[[keys]]` configured, nothing here runs.
//! With keys but no token budgets, only exact request counting applies
//! (no body bytes are read). Token counting activates a bounded
//! tail-sniffer on the proxied byte stream - constructed per request
//! ONLY when the requesting key carries `tpm`/`daily_tokens` limits,
//! never for unauthenticated or unlimited keys.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use blazar_core::store::{KeyUsageRow, Store};

/// Request-scoped identity inserted by the auth middleware.
#[derive(Clone)]
pub struct KeyCtx {
    pub name: String,
    /// B5 scheduling weight (>= 1): the admission queue interleaves
    /// contending keys proportionally to this. Authless requests share
    /// the anonymous weight-1 bucket.
    pub weight: u32,
}

/// BLZ-2: control-plane authorization. Fleet-level and mutating
/// operations (failover pinning, replication fan-out, live-adapter
/// surgery, adapter listing) change what serves traffic or touch every
/// peer, so they demand an unscoped (admin) key whenever keys exist.
/// On the open gateway (no keys) they stay open - the auth
/// middleware's documented trust boundary applies unchanged.
pub fn control_plane_authorized(
    keys: &KeysLimiter,
    key: Option<&KeyCtx>,
) -> Result<(), (u16, String)> {
    if keys.is_empty() {
        return Ok(());
    }
    let Some(k) = key else {
        return Err((401, "missing bearer token".into()));
    };
    let scoped = keys
        .entries()
        .into_iter()
        .find(|e| e.name == k.name)
        .is_some_and(|e| !e.models.is_empty());
    if scoped {
        Err((
            403,
            format!(
                "key {:?} is model-scoped; control-plane operations require an unscoped (admin) key",
                k.name
            ),
        ))
    } else {
        Ok(())
    }
}

/// BLZ-2: pure model-scope predicate for model-specific control
/// actions (warm). Authorization only - no quota accounting: warm is
/// an operator nudge, not an inference request.
pub fn scope_permits(
    keys: &KeysLimiter,
    key: Option<&KeyCtx>,
    model: &str,
) -> Result<(), (u16, String)> {
    if keys.is_empty() {
        return Ok(());
    }
    let Some(k) = key else {
        return Err((401, "missing bearer token".into()));
    };
    let Some(entry) = keys.entries().into_iter().find(|e| e.name == k.name) else {
        return Err((401, "unknown key".into()));
    };
    if entry.models.is_empty() || entry.models.iter().any(|m| scope_matches(m, model)) {
        Ok(())
    } else {
        Err((
            403,
            format!("key {:?} is not scoped for model {:?}", k.name, model),
        ))
    }
}

/// MCP server-scope predicate: may `key` drive tools of `server`?
/// Mirrors [`scope_permits`] semantics - an empty `mcp_servers` list
/// admits every configured server (admin posture), a non-empty list is
/// an exact/trailing-`*` allowlist. Enforcement happens at both MCP
/// choke points: tool-catalog injection (unlisted servers' tools never
/// enter the prompt) and pre-execution (a model-emitted call naming an
/// unlisted server is refused before transport).
pub fn mcp_server_permits(
    keys: &KeysLimiter,
    key: Option<&KeyCtx>,
    server: &str,
) -> Result<(), (u16, String)> {
    if keys.is_empty() {
        return Ok(());
    }
    let Some(k) = key else {
        return Err((401, "missing bearer token".into()));
    };
    let Some(entry) = keys.entries().into_iter().find(|e| e.name == k.name) else {
        return Err((401, "unknown key".into()));
    };
    if entry.mcp_servers.is_empty() || entry.mcp_servers.iter().any(|s| scope_matches(s, server)) {
        Ok(())
    } else {
        Err((
            403,
            format!("key {:?} is not scoped for mcp server {:?}", k.name, server),
        ))
    }
}

/// MCP tool-scope predicate: may `key` drive `server`'s tool `tool`?
/// Runs AFTER [`mcp_server_permits`] (an unlisted server is refused
/// regardless of this list). Entries are `<server>:<tool>` selectors;
/// either side matches exactly or by trailing-`*`. An empty `mcp_tools`
/// list admits every tool of the permitted servers. Enforcement mirrors
/// the server gate: filtered at catalog injection and re-checked before
/// execution (a model-emitted call naming an unlisted tool is refused
/// before transport).
pub fn mcp_tool_permits(
    keys: &KeysLimiter,
    key: Option<&KeyCtx>,
    server: &str,
    tool: &str,
) -> Result<(), (u16, String)> {
    if keys.is_empty() {
        return Ok(());
    }
    let Some(k) = key else {
        return Err((401, "missing bearer token".into()));
    };
    let Some(entry) = keys.entries().into_iter().find(|e| e.name == k.name) else {
        return Err((401, "unknown key".into()));
    };
    if entry.mcp_tools.is_empty() {
        return Ok(());
    }
    let admitted = entry.mcp_tools.iter().any(|sel| {
        match sel.split_once(':') {
            // A bare colon-less entry names every tool of a server.
            Some((srv, tl)) => {
                scope_matches(srv, server) && (tl.is_empty() || scope_matches(tl, tool))
            }
            None => scope_matches(sel, server),
        }
    });
    if admitted {
        Ok(())
    } else {
        Err((
            403,
            format!(
                "key {:?} is not scoped for mcp tool {:?} on server {:?}",
                k.name, tool, server
            ),
        ))
    }
}

/// BLZ-1 object-level authorization: may `key` see a durable row created
/// by `owner`? `legacy_open_records` (config, default `true`) decides the
/// tenancy posture for rows with no owner stamp: `true` keeps the
/// documented legacy-sharing posture (pre-scope and open-gateway rows are
/// visible to every caller); `false` quarantines them from every keyed
/// caller - they answer as unknown until an operator re-surfaces them by
/// temporarily flipping the flag. A keyless request implies keys are not
/// configured at all (the auth middleware inserts `KeyCtx` whenever they
/// are), which is the same open-gateway posture either way.
#[must_use]
pub fn owner_allows(owner: Option<&str>, key: Option<&KeyCtx>, legacy_open_records: bool) -> bool {
    match (owner, key) {
        (Some(o), Some(k)) => o == k.name,
        // Unstamped row meeting a keyed caller: shared legacy artifact
        // or quarantined pre-tenant history, per the config posture.
        (None, Some(_)) => legacy_open_records,
        // Open-gateway rows and keyless callers both admit; only a
        // keyed caller meeting a foreign owner is refused.
        _ => true,
    }
}

/// Why a request was rejected (mapped to a response by the caller).
#[derive(Debug)]
pub enum Rejection {
    Scope {
        name: String,
        model: String,
    },
    Rate {
        kind: &'static str,
        retry_after_secs: u64,
        name: String,
    },
}

impl Rejection {
    #[must_use]
    pub fn to_response(&self) -> axum::response::Response {
        use axum::http::StatusCode;
        use axum::response::IntoResponse;
        match self {
            Self::Scope { name, model } => (
                StatusCode::FORBIDDEN,
                axum::Json(serde_json::json!({
                    "error": {"type": "blazar_key_scope", "code": 403, "blazar_code": "KEY_SCOPE_FORBIDDEN",
                        "message": format!("key {name:?} is not scoped for model {model:?}"),
                        "key": name}
                })),
            )
                .into_response(),
            Self::Rate {
                kind,
                retry_after_secs,
                name,
            } => (
                StatusCode::TOO_MANY_REQUESTS,
                [
                    ("retry-after", retry_after_secs.to_string()),
                    ("x-blazar-key", name.clone()),
                ],
                axum::Json(serde_json::json!({
                    "error": {"type": "blazar_key_rate", "code": 429, "blazar_code": "RATE_LIMITED",
                        "message": format!("key {name:?} exceeded its {kind} budget"),
                        "retry_after_secs": retry_after_secs}
                })),
            )
                .into_response(),
        }
    }
}

/// UTC day stamp for the usage table (YYYY-MM-DD).
/// Constant-time equality for secret material. `str ==` short-circuits on
/// length and on the first differing byte, letting request timing disclose
/// how much of a presented secret matched a stored one. This folds every
/// byte plus both lengths into one accumulator with no early exit.
/// Defense-in-depth only: the primary barrier is key entropy (plm_ keys
/// are random 256-bit), not comparison time.
#[must_use]
fn ct_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    // Loop count depends only on max length (length itself is not treated
    // as secret); content never influences control flow.
    let mut diff: usize = a.len() ^ b.len();
    for i in 0..a.len().max(b.len()) {
        let x = if i < a.len() { a[i] } else { 0 };
        let y = if i < b.len() { b[i] } else { 0 };
        diff |= usize::from(x ^ y);
    }
    diff == 0
}

#[must_use]
pub fn utc_day(ts: SystemTime) -> String {
    let secs = ts.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    // Civil-from-days (Howard Hinnant's algorithm), no chrono dep.
    let days = i64::try_from(secs / 86_400).unwrap_or(0);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

/// Seconds until the next UTC midnight (`1..=86_400`) - the true reset
/// point for daily token budgets (F56: was hardcoded 3600).
fn secs_until_utc_midnight() -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let tomorrow = (now.div_euclid(86_400) + 1) * 86_400;
    (tomorrow - now).clamp(1, 86_400)
}

/// Front-drain expired entries from a time-ordered window buffer. The
/// vecs only ever grow via `push(Instant::now())`, so they stay sorted
/// ascending and `partition_point` finds the expired prefix in O(log n).
/// Without this the rpm/tpm vecs grow unboundedly for the daemon's
/// lifetime and every rate check scans the full history (F54).
fn prune_window(win: &mut Vec<Instant>, now: Instant) {
    let cut = win.partition_point(|t| now.duration_since(*t) >= WINDOW);
    win.drain(..cut);
}

#[derive(Default)]
struct KeyState {
    /// Last 60s of admitted request starts (rpm window).
    rpm: Vec<Instant>,
    /// (timestamp, tokens) in the last 60s (tpm window).
    tpm: Vec<(Instant, u64)>,
    /// UTC day stamp the in-memory counters belong to.
    day: String,
    requests: u64,
    tokens: u64,
    /// Live leases (concurrency cap); instantaneous state - never
    /// rolled by `roll_day` nor persisted.
    in_flight: u32,
    /// Outstanding strict-lane token reservations (`reserve_budgets`):
    /// completion ceilings held by in-flight requests, not yet settled
    /// to measured usage. Instantaneous state like `in_flight` - never
    /// rolled nor persisted.
    reserved: u64,
    dirty: bool,
}

impl KeyState {
    fn roll_day(&mut self, today: &str) {
        if self.day != *today {
            self.day = today.to_string();
            self.requests = 0;
            self.tokens = 0;
            self.dirty = false; // fresh day: nothing to persist yet
        }
    }
}

/// Shared limiter + accounting state, owned by `AppState`.
pub struct KeysLimiter {
    /// Live key entries (seeded from `config.keys` at boot; mutated by
    /// `/api/keys` without a daemon restart). Auth resolves here.
    entries: std::sync::RwLock<Vec<blazar_core::ApiKey>>,
    states: Mutex<HashMap<String, KeyState>>,
    /// Loaded from the store at startup so daily budgets survive restarts.
    #[allow(dead_code)] // documents provenance of pre-loaded counters
    loaded_day: String,
}

impl KeysLimiter {
    /// Build with today's persisted usage pre-loaded (best effort - a
    /// store failure degrades to zero-based counters, never blocks boot).
    #[must_use]
    pub fn loaded(store: Option<&Store>, entries: Vec<blazar_core::ApiKey>) -> Self {
        let today = utc_day(SystemTime::now());
        let mut pre: HashMap<String, KeyState> = HashMap::new();
        if let Some(s) = store
            && let Ok(rows) = s.key_usage(&today)
        {
            for KeyUsageRow {
                name,
                requests,
                tokens,
                ..
            } in rows
            {
                pre.insert(
                    name,
                    KeyState {
                        day: today.clone(),
                        requests: u64::try_from(requests).unwrap_or(0),
                        tokens: u64::try_from(tokens).unwrap_or(0),
                        dirty: false,
                        ..KeyState::default()
                    },
                );
            }
        }
        Self {
            entries: std::sync::RwLock::new(entries),
            states: Mutex::new(pre),
            loaded_day: today,
        }
    }

    /// Bearer secret -> key entry (None = unknown / no keys configured).
    /// Secret comparison is constant-time (see [`ct_eq`]). `.find()`
    /// short-circuits, but a later position requires presenting a valid
    /// secret for some earlier entry - position leaks nothing without
    /// the key, and per-compare `ct_eq` hides content differences.
    ///
    /// Lock-poisoning policy (gateway-wide, incl. the supervisor /
    /// requests / remotes / whisper ledgers): `.lock().expect("...
    /// poisoned")` is a deliberate fail-fast. A panic under a lock means
    /// memory or an invariant already broke mid-mutation; poisoning
    /// every later request into the same loud crash is preferable to
    /// serving from state of unknown integrity. The daemon supervisor
    /// restarts the process and the on-disk store is the recovery point.
    #[must_use]
    pub fn resolve(&self, presented: &str) -> Option<blazar_core::ApiKey> {
        self.entries
            .read()
            .expect("keys entries poisoned")
            .iter()
            .find(|k| ct_eq(&k.key, presented))
            .cloned()
    }

    #[must_use]
    pub fn entries(&self) -> Vec<blazar_core::ApiKey> {
        self.entries.read().expect("keys entries poisoned").clone()
    }

    /// Look up a key entry by NAME (the authenticated identity), for
    /// budget checks at the proxy (not a secret comparison).
    #[must_use]
    pub fn entry(&self, name: &str) -> Option<blazar_core::ApiKey> {
        self.entries
            .read()
            .expect("keys entries poisoned")
            .iter()
            .find(|k| k.name == name)
            .cloned()
    }

    /// True when no keys exist (the gateway is authless).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries
            .read()
            .expect("keys entries poisoned")
            .is_empty()
    }

    /// Insert or replace a key by name. The caller persists the config.
    pub fn upsert(&self, key: blazar_core::ApiKey) {
        let mut entries = self.entries.write().expect("keys entries poisoned");
        match entries.iter().position(|k| k.name == key.name) {
            Some(i) => entries[i] = key,
            None => entries.push(key),
        }
    }

    /// Remove a key by name. The caller persists the config.
    pub fn remove(&self, name: &str) -> bool {
        let mut entries = self.entries.write().expect("keys entries poisoned");
        let before = entries.len();
        entries.retain(|k| k.name != name);
        before != entries.len()
    }

    /// Hot-reload swap: make `entries` the complete live key set. Keys
    /// absent from the new list are dropped, survivors are replaced
    /// in place, and new ones are appended - the same semantics the
    /// management API applies one key at a time (`upsert`/`remove`),
    /// applied to the whole list at once. Per-name usage/rate state is
    /// keyed by name, so surviving keys keep their counters. The
    /// caller owns persistence (on reload the config file already IS
    /// the new truth).
    pub fn replace_all(&self, entries: Vec<blazar_core::ApiKey>) {
        let mut cur = self.entries.write().expect("keys entries poisoned");
        cur.retain(|k| entries.iter().any(|n| n.name == k.name));
        for key in entries {
            match cur.iter().position(|e| e.name == key.name) {
                Some(i) => cur[i] = key,
                None => cur.push(key),
            }
        }
    }

    /// Atomic admission (BLZ-3): model scope, then the rpm/tpm/daily
    /// gates, then the request charge - all under ONE lock acquisition.
    /// The old read-then-charge pair let N concurrent requests all pass
    /// the same pre-charge window and overshoot the rpm ceiling; here a
    /// request is only admitted when its own charge lands in the same
    /// critical section, so the window count the gates evaluate is
    /// exact at every decision.
    ///
    /// Token budgets stay measured-usage accounting (charged at
    /// completion via `charge_tokens`): they throttle with completion
    /// lag rather than reserving upfront. Two strict in-flight bounds
    /// exist for hard spending control: the `max_concurrent` cap leased
    /// at auth time, and the opt-in `reserve_budgets` lane
    /// ([`KeysLimiter::reserve`]) which holds each request's completion
    /// ceiling against tpm/daily headroom until it settles.
    pub fn admit(&self, key: &blazar_core::ApiKey, model: &str) -> Result<(), Rejection> {
        if !key.models.is_empty() && !key.models.iter().any(|m| scope_matches(m, model)) {
            return Err(Rejection::Scope {
                name: key.name.clone(),
                model: model.to_string(),
            });
        }
        let now = Instant::now();
        let today = utc_day(SystemTime::now());
        let mut states = self.states.lock().expect("keys state poisoned");
        let st = states.entry(key.name.clone()).or_default();
        st.roll_day(&today);
        prune_window(&mut st.rpm, now);
        if key.rpm > 0 && st.rpm.len() >= usize::try_from(key.rpm).unwrap_or(usize::MAX) {
            // Window is pruned, so the first entry IS the oldest
            // in-window charge; the caller waits for it to age out.
            let retry_after_secs = st.rpm.first().map_or(1, |t| {
                WINDOW
                    .checked_sub(now.duration_since(*t))
                    .unwrap_or(WINDOW)
                    .as_secs()
                    .max(1)
            });
            return Err(Rejection::Rate {
                kind: "requests-per-minute",
                retry_after_secs,
                name: key.name.clone(),
            });
        }
        if key.tpm > 0 {
            let mut oldest_in_window = None;
            let mut spent: u64 = 0;
            for (t, n) in &st.tpm {
                if now.duration_since(*t) < WINDOW {
                    spent += n;
                    oldest_in_window = Some(oldest_in_window.unwrap_or(*t).min(*t));
                }
            }
            if spent >= key.tpm {
                let retry = oldest_in_window.map_or(1, |t| {
                    WINDOW
                        .checked_sub(now.duration_since(t))
                        .unwrap_or(WINDOW)
                        .as_secs()
                        .max(1)
                });
                return Err(Rejection::Rate {
                    kind: "tokens-per-minute",
                    retry_after_secs: retry,
                    name: key.name.clone(),
                });
            }
        }
        if key.daily_tokens > 0 && st.tokens >= key.daily_tokens {
            return Err(Rejection::Rate {
                kind: "daily token budget",
                retry_after_secs: secs_until_utc_midnight(),
                name: key.name.clone(),
            });
        }
        // Granted: the charge lands inside the same critical section
        // the gates just evaluated, closing the check/charge race.
        st.rpm.push(now);
        st.requests += 1;
        st.dirty = true;
        Ok(())
    }

    /// Add measured tokens for a finished chat-family request (exact
    /// values from the response-usage tail, zero when the sniffer found
    /// none - e.g. streaming without a usage chunk).
    pub fn charge_tokens(&self, name: &str, tokens: u64) {
        if tokens == 0 {
            return;
        }
        let now = Instant::now();
        let today = utc_day(SystemTime::now());
        let mut states = self.states.lock().expect("keys state poisoned");
        let st = states.entry(name.to_string()).or_default();
        st.roll_day(&today);
        let cut = st
            .tpm
            .partition_point(|(t, _)| now.duration_since(*t) >= WINDOW);
        st.tpm.drain(..cut);
        st.tpm.push((now, tokens));
        st.tokens += tokens;
        st.dirty = true;
    }

    /// Lease an in-flight slot at auth time. `Ok(None)` = no cap
    /// configured (nothing tracked, zero cost). Checked BEFORE model
    /// scope (cheapest gate first - a capped key cannot even probe
    /// scopes). Rejected requests never consume a slot.
    pub fn acquire(
        self: &Arc<Self>,
        name: &str,
        max_concurrent: u32,
    ) -> Result<Option<SlotLease>, Rejection> {
        if max_concurrent == 0 {
            return Ok(None);
        }
        let mut states = self.states.lock().expect("keys state poisoned");
        let st = states.entry(name.to_string()).or_default();
        if st.in_flight >= max_concurrent {
            return Err(Rejection::Rate {
                kind: "concurrent requests",
                retry_after_secs: 2,
                name: name.to_string(),
            });
        }
        st.in_flight += 1;
        Ok(Some(SlotLease {
            limiter: Arc::clone(self),
            name: name.to_string(),
        }))
    }

    /// Release a lease (called by `SlotLease::drop`). A key removed
    /// mid-flight keeps its state entry - `get_mut` only decrements and
    /// never inserts, so nothing is re-created here.
    fn release(&self, name: &str) {
        let mut states = self.states.lock().expect("keys state poisoned");
        if let Some(st) = states.get_mut(name) {
            st.in_flight = st.in_flight.saturating_sub(1);
        }
    }

    /// Strict-lane admission (`reserve_budgets = true`): hold `estimate`
    /// completion tokens against the key's tpm/daily budgets before any
    /// generation starts. `Ok(None)` = the key carries no token budgets
    /// (or the estimate is zero) - nothing tracked, zero cost. A refusal
    /// is a plain [`Rejection::Rate`] (429 + retry-after) naming the
    /// reservation lane, so a burst cannot slip past the budget between
    /// the last settled charge and the next admission check. Atomic with
    /// the reservation bookkeeping under one lock, mirroring `admit`.
    pub fn reserve(
        self: &std::sync::Arc<Self>,
        entry: &blazar_core::ApiKey,
        estimate: u64,
    ) -> Result<Option<BudgetLease>, Rejection> {
        if (entry.tpm == 0 && entry.daily_tokens == 0) || estimate == 0 {
            return Ok(None);
        }
        let now = Instant::now();
        let today = utc_day(SystemTime::now());
        let mut states = self.states.lock().expect("keys state poisoned");
        let st = states.entry(entry.name.clone()).or_default();
        st.roll_day(&today);
        let cut = st
            .tpm
            .partition_point(|(t, _)| now.duration_since(*t) >= WINDOW);
        st.tpm.drain(..cut);
        if entry.tpm > 0 {
            let mut oldest_in_window = None;
            let mut spent: u64 = 0;
            for (t, n) in &st.tpm {
                spent += n;
                oldest_in_window = Some(oldest_in_window.unwrap_or(*t).min(*t));
            }
            if spent.saturating_add(st.reserved).saturating_add(estimate) > entry.tpm {
                let retry_after_secs = oldest_in_window.map_or(1, |t| {
                    WINDOW
                        .checked_sub(now.duration_since(t))
                        .unwrap_or(WINDOW)
                        .as_secs()
                        .max(1)
                });
                return Err(Rejection::Rate {
                    kind: "token budget reservation (tokens-per-minute)",
                    retry_after_secs,
                    name: entry.name.clone(),
                });
            }
        }
        if entry.daily_tokens > 0
            && st
                .tokens
                .saturating_add(st.reserved)
                .saturating_add(estimate)
                > entry.daily_tokens
        {
            return Err(Rejection::Rate {
                kind: "token budget reservation (daily token budget)",
                retry_after_secs: secs_until_utc_midnight(),
                name: entry.name.clone(),
            });
        }
        st.reserved += estimate;
        Ok(Some(BudgetLease {
            limiter: std::sync::Arc::clone(self),
            name: entry.name.clone(),
            held: estimate,
        }))
    }

    /// Release an unsettled (or partially settled) reservation amount.
    /// `saturating_sub` mirrors `release`: a key removed mid-flight
    /// keeps its state entry, and a stale lease can never underflow.
    fn release_reservation(&self, name: &str, tokens: u64) {
        if tokens == 0 {
            return;
        }
        let mut states = self.states.lock().expect("keys state poisoned");
        if let Some(st) = states.get_mut(name) {
            st.reserved = st.reserved.saturating_sub(tokens);
        }
    }

    /// Snapshot for `/api/keys` (never secrets).
    #[must_use]
    pub fn usage_snapshot(&self) -> Vec<(String, String, u64, u64)> {
        let today = utc_day(SystemTime::now());
        let states = self.states.lock().expect("keys state poisoned");
        let mut out: Vec<(String, String, u64, u64)> = states
            .iter()
            .map(|(name, st)| {
                let mut st = clone_state(st);
                st.roll_day(&today);
                (name.clone(), st.day.clone(), st.requests, st.tokens)
            })
            .collect();
        out.sort();
        out
    }

    /// Persist dirty counters (write-behind; called by the flusher task
    /// and at daemon shutdown). Absolute day-state per key, one tx per
    /// day stamp. F55: entries dirty under a PRIOR day stamp flush under
    /// that day - `roll_day` resets counters on first charge of the new
    /// day, so filtering on `today` here silently dropped the last
    /// <flush-interval of pre-midnight usage every night.
    pub fn flush(&self, dirs: &blazar_core::BlazarDirs) {
        let drained: Vec<(String, String, u64, u64)> = {
            let mut states = self.states.lock().expect("keys state poisoned");
            states
                .iter_mut()
                .filter(|(_, st)| st.dirty)
                .map(|(name, st)| {
                    st.dirty = false;
                    (st.day.clone(), name.clone(), st.requests, st.tokens)
                })
                .collect()
        };
        if drained.is_empty() {
            return;
        }
        let mark_dirty = |rows: &[(&String, &String, &u64, &u64)]| {
            let mut states = self.states.lock().expect("keys state poisoned");
            for (_, name, _, _) in rows {
                if let Some(st) = states.get_mut(name.as_str()) {
                    st.dirty = true;
                }
            }
        };
        let Ok(store) = Store::open(dirs) else {
            tracing::warn!(target: "blazar::keys", "usage flush: store open failed");
            let rows: Vec<_> = drained.iter().map(|(d, n, r, t)| (d, n, r, t)).collect();
            mark_dirty(&rows);
            return;
        };
        let mut by_day: std::collections::BTreeMap<String, Vec<(String, u64, u64)>> =
            std::collections::BTreeMap::new();
        for (day, name, requests, tokens) in &drained {
            by_day
                .entry(day.clone())
                .or_default()
                .push((name.clone(), *requests, *tokens));
        }
        for (day, rows) in &by_day {
            if let Err(e) = store.set_key_usage_day(day, rows) {
                tracing::warn!(target: "blazar::keys", day, "usage flush failed: {e}");
                let failed: Vec<_> = drained
                    .iter()
                    .filter(|(d, _, _, _)| d == day)
                    .map(|(d, n, r, t)| (d, n, r, t))
                    .collect();
                mark_dirty(&failed);
            }
        }
    }
}

fn clone_state(st: &KeyState) -> KeyState {
    KeyState {
        rpm: Vec::new(),
        tpm: Vec::new(),
        day: st.day.clone(),
        requests: st.requests,
        tokens: st.tokens,
        in_flight: 0,
        reserved: 0,
        dirty: st.dirty,
    }
}

/// RAII in-flight lease: increments on acquire, releases when dropped.
/// The guard lives inside the response body, so the slot is held for
/// exactly as long as the response streams (dropped on completion OR
/// client disconnect - hyper drops bodies either way).
pub struct SlotLease {
    limiter: std::sync::Arc<KeysLimiter>,
    name: String,
}

impl std::fmt::Debug for SlotLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Tests unwrap on this type; the limiter itself is not inspectable.
        f.debug_struct("SlotLease")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl Drop for SlotLease {
    fn drop(&mut self) {
        self.limiter.release(&self.name);
    }
}

/// RAII token-budget reservation (strict lane, `reserve_budgets`).
/// Created by [`KeysLimiter::reserve`] at admission time; settles to
/// measured usage via [`BudgetLease::settle`] when the response's
/// stream finishes, and releases any unsettled remainder on Drop - an
/// aborted request never strands budget headroom.
pub struct BudgetLease {
    limiter: std::sync::Arc<KeysLimiter>,
    name: String,
    /// Tokens still held; `settle` zeroes this before the terminal
    /// Drop so the release runs exactly once.
    held: u64,
}

// Limiter internals are irrelevant; name + held size say everything.
#[allow(clippy::missing_fields_in_debug)] // Arc<KeysLimiter> has no Debug by design
impl std::fmt::Debug for BudgetLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BudgetLease")
            .field("key", &self.name)
            .field("held", &self.held)
            .finish()
    }
}

impl BudgetLease {
    /// Replace the provisional reservation with measured usage:
    /// releases everything still held and charges `actual` through the
    /// ordinary accounting path (0 = the sniffer found no usage - the
    /// reservation is still released). Consumes `self`, so the terminal
    /// Drop is a no-op.
    pub fn settle(mut self, actual: u64) {
        self.limiter.release_reservation(&self.name, self.held);
        self.held = 0;
        if actual > 0 {
            self.limiter.charge_tokens(&self.name, actual);
        }
    }
}

impl Drop for BudgetLease {
    fn drop(&mut self) {
        self.limiter.release_reservation(&self.name, self.held);
    }
}

/// Response body that carries a `SlotLease`; releasing it when the body
/// is consumed, errored, or dropped.
pub struct GuardedBody {
    inner: axum::body::Body,
    /// Held purely for its `Drop` (slot release), never read.
    #[allow(dead_code)]
    lease: Option<SlotLease>,
}

impl axum::body::HttpBody for GuardedBody {
    type Data = axum::body::Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        Pin::new(&mut self.inner).poll_frame(cx)
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

/// Attach a lease to a response: the in-flight slot frees when the
/// client has drained (or abandoned) the body.
#[must_use]
pub fn guard_response(
    resp: axum::response::Response,
    lease: Option<SlotLease>,
) -> axum::response::Response {
    match lease {
        None => resp,
        Some(l) => resp.map(|body| {
            axum::body::Body::new(GuardedBody {
                inner: body,
                lease: Some(l),
            })
        }),
    }
}

const WINDOW: Duration = Duration::from_mins(1);

/// Scope matching: exact, or trailing-`*` prefix ("qwen3*" matches
/// "qwen3.5-9b"; "vllm:*" scopes a whole remote lane).
#[must_use]
pub fn scope_matches(pattern: &str, model: &str) -> bool {
    match pattern.strip_suffix('*') {
        Some(prefix) => model.starts_with(prefix),
        None => pattern == model,
    }
}

/// Bounded tail-sniffer: keeps the last `TAIL` bytes of a response body
/// and extracts usage counts at stream end. Understands the three usage
/// dialects blazar serves (`OpenAI` `prompt_tokens`/`completion_tokens`,
/// Responses `input_tokens`/`output_tokens`, ollama NDJSON
/// `prompt_eval_count`/`eval_count`). Constructed only for keys with
/// token budgets - everyone else pays nothing. Streaming lanes enable
/// [`with_frame_counting`] so a body that never carries a usage payload
/// still settles its budget from the observed token-frame count.
pub struct UsageSniffer {
    tail: Vec<u8>,
    key: String,
    // F7 (OpenAI `n`, multi-choice): llama-server reports completion
    // tokens PER CHOICE (live probe: n=2 with ~30 tokens per choice
    // reported completion_tokens=30), so the completion side is
    // multiplied by n for llama.cpp children. sglang sums natively;
    // mistral.rs is unprobed and stays at honest 1x.
    completion_multiplier: u64,
    // Lower-bound completion estimate for usage-less streams (see
    // `usage::FrameCounter`). Inert unless the wrapping lane knows the
    // body is a token-per-line stream.
    frames: crate::usage::FrameCounter,
    count_frames: bool,
}

const TAIL: usize = 8 * 1024;

impl UsageSniffer {
    #[must_use]
    pub fn new(key: &str) -> Self {
        Self {
            tail: Vec::with_capacity(TAIL),
            key: key.to_string(),
            completion_multiplier: 1,
            frames: crate::usage::FrameCounter::default(),
            count_frames: false,
        }
    }

    /// Charge the completion side `mult`x (see field note). Default 1.
    #[must_use]
    pub fn with_completion_multiplier(mut self, mult: u64) -> Self {
        self.completion_multiplier = mult;
        self
    }

    /// Enable the token-frame fallback for a streaming body (SSE or
    /// NDJSON - one line per generated token). When the finished body
    /// carries NO usage payload in any dialect, the observed frame
    /// count stands in as a lower-bound completion charge instead of
    /// zero. An exact usage payload always wins.
    #[must_use]
    pub fn with_frame_counting(mut self, on: bool) -> Self {
        self.count_frames = on;
        self
    }

    /// Key name this sniffer charges (settlement helpers read it).
    #[must_use]
    pub fn key(&self) -> &str {
        &self.key
    }

    pub fn push(&mut self, bytes: &[u8]) {
        if self.count_frames {
            self.frames.push(bytes);
        }
        if self.tail.len() >= TAIL {
            let overflow = self.tail.len() + bytes.len() - TAIL;
            let keep_from = overflow.min(self.tail.len());
            self.tail.drain(..keep_from);
        }
        self.tail.extend_from_slice(bytes);
        if self.tail.len() > TAIL {
            let cut = self.tail.len() - TAIL;
            self.tail.drain(..cut);
        }
    }

    /// Parse the tail and return the scaled token total WITHOUT
    /// charging - the reservation-settle path pairs this with
    /// `BudgetLease::settle`, which releases + charges in one step.
    /// No usage payload in any dialect + frame counting on: the
    /// observed token-frame count stands in (lower-bound completion;
    /// prompt side has no estimator at this layer). Frames are NOT
    /// completion-multiplied - they are directly observed records
    /// across every choice, unlike a per-choice usage report.
    pub fn sniff(&mut self) -> u64 {
        let (prompt, completion) = sniff_usage_parts(&std::mem::take(&mut self.tail));
        if prompt.is_none() && completion.is_none() && self.count_frames {
            return self.frames.count();
        }
        scaled_total(prompt, completion, self.completion_multiplier)
    }

    /// Parse the tail + charge the key. Returns the token total found.
    pub fn finish(mut self, limiter: &KeysLimiter) -> u64 {
        let total = self.sniff();
        limiter.charge_tokens(&self.key, total);
        total
    }
}

/// Extract (prompt + completion) tokens from a response tail. Pure.
#[must_use]
pub fn sniff_usage(tail: &[u8]) -> u64 {
    let (prompt, completion) = sniff_usage_parts(tail);
    prompt.unwrap_or(0).saturating_add(completion.unwrap_or(0))
}

/// (prompt, completion) split of the usage tail - each side uses the
/// first dialect that appears (`OpenAI`, Responses, ollama NDJSON).
/// Pure; unit-pinned alongside `scaled_total`.
#[must_use]
pub fn sniff_usage_parts(tail: &[u8]) -> (Option<u64>, Option<u64>) {
    let prompt = last_int_after(tail, "\"prompt_tokens\"")
        .or_else(|| last_int_after(tail, "\"input_tokens\""))
        .or_else(|| last_int_after(tail, "\"prompt_eval_count\""));
    let completion = last_int_after(tail, "\"completion_tokens\"")
        .or_else(|| last_int_after(tail, "\"output_tokens\""))
        .or_else(|| last_int_after(tail, "\"eval_count\""));
    (prompt, completion)
}

/// Multi-choice-aware total: prompt once + completion x mult, clamped
/// at u64 so an oversized lie cannot wrap around into a small charge.
#[must_use]
pub fn scaled_total(prompt: Option<u64>, completion: Option<u64>, mult: u64) -> u64 {
    prompt
        .unwrap_or(0)
        .saturating_add(completion.unwrap_or(0).saturating_mul(mult.max(1)))
}

/// Shared substring scan (gateway-internal): cheap gate before any JSON
/// parse on streamed bodies.
pub(crate) fn find_sub(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Value of the LAST `<key> <int>` occurrence in a streamed tail - the
/// usage payload is always the final event, so a literal appearing in
/// generated prose earlier must not win. If the last occurrence fails
/// to parse (prose), the previous parse wins.
pub(crate) fn last_int_after(tail: &[u8], key: &str) -> Option<u64> {
    let mut best: Option<u64> = None;
    let mut hay = tail;
    while let Some(pos) = find_sub(hay, key.as_bytes()) {
        let after = &hay[pos + key.len()..];
        best = read_leading_int(after).or(best);
        hay = after;
    }
    best
}

/// Wrap a finished handler `Response` so its body charges the key's
/// token counters as it streams out (usage lives in the final NDJSON
/// line / final JSON object of ollama-API responses). Abort-safe via
/// `Drop` - a client disconnect still charges what was generated.
/// `lease` (strict `reserve_budgets` lane) settles the provisional
/// reservation from the same sniffed total; `None` keeps plain
/// charge-on-measure.
#[must_use]
pub fn charge_outgoing(
    resp: axum::response::Response,
    name: &str,
    limiter: std::sync::Arc<KeysLimiter>,
    lease: Option<BudgetLease>,
    service: Option<LocalServiceStamp>,
) -> axum::response::Response {
    use futures::StreamExt as _;
    /// Abort-safe finisher: `Drop` runs on clean drain AND client abort.
    struct Finish(
        Option<UsageSniffer>,
        std::sync::Arc<KeysLimiter>,
        Option<BudgetLease>,
        // Carried, never read: dropping this field stamps the EWMA.
        #[allow(dead_code)] Option<LocalServiceStamp>,
    );
    impl Drop for Finish {
        fn drop(&mut self) {
            let lease = self.2.take();
            match self.0.take() {
                Some(s) => {
                    let mut s = s;
                    let total = s.sniff();
                    match lease {
                        Some(l) => l.settle(total),
                        None => {
                            if total > 0 {
                                self.1.charge_tokens(&s.key, total);
                            }
                        }
                    }
                }
                None => {
                    // Snifferless wrap with a lease (shouldn't happen -
                    // lease implies budgets implies sniffer): release
                    // the reservation against a zero charge so headroom
                    // is never stranded.
                    if let Some(l) = lease {
                        l.settle(0);
                    }
                }
            }
        }
    }
    let (parts, body) = resp.into_parts();
    // Token-frame fallback arms only for line-framed streams (SSE /
    // NDJSON): a body that never carries a usage payload then settles
    // from the observed frame count instead of zero. JSON bodies keep
    // tail-only sniffing - their usage rides the final object.
    let streaming_ct = parts
        .headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| {
            ct.starts_with("text/event-stream") || ct.starts_with("application/x-ndjson")
        });
    let finish = std::sync::Arc::new(std::sync::Mutex::new(Finish(
        Some(UsageSniffer::new(name).with_frame_counting(streaming_ct)),
        limiter,
        lease,
        service,
    )));
    let mapped = body.into_data_stream().map(move |r| {
        if let Ok(bytes) = r.as_ref()
            && let Some(f) = finish.lock().expect("usage finisher").0.as_mut()
        {
            f.push(bytes);
        }
        r.map_err(|e| std::io::Error::other(e.to_string()))
    });
    let stream = mapped.chain(futures::stream::unfold((), |()| async { None }));
    axum::http::Response::from_parts(parts, axum::body::Body::from_stream(stream))
}

/// Local-lane service-time probe: constructed when a chat request's
/// serving model is known, dropped when the response stream completes
/// (or the scope of a buffered build ends). `Drop` folds the observed
/// request-to-completion milliseconds into a per-model EWMA that pool
/// `adaptive` scoring consumes - the local twin of the remote lane's
/// per-peer `service_ewma_ms`.
///
/// Semantics: stamps chat routes on successful child responses; client
/// aborts mid-stream under-measure (the guard drops early) and blend
/// out via the EWMA. Poisoned or missing map entries are skipped -
/// the probe is advisory telemetry, never a request-path failure.
pub struct LocalServiceStamp {
    map: std::sync::Arc<
        std::sync::Mutex<std::collections::HashMap<String, (f64, std::time::Instant)>>,
    >,
    model: String,
    started: std::time::Instant,
    /// Set by [`LocalServiceStamp::cancel`]: failed forwards and
    /// non-success responses must never sample the EWMA.
    cancelled: bool,
}

impl LocalServiceStamp {
    /// Start the clock for one served model. Callers gate construction
    /// (chat routes, success status, non-empty model) - this is just
    /// the captured context.
    pub fn new(
        map: std::sync::Arc<
            std::sync::Mutex<std::collections::HashMap<String, (f64, std::time::Instant)>>,
        >,
        model: &str,
    ) -> Self {
        Self {
            map,
            model: model.to_string(),
            started: std::time::Instant::now(),
            cancelled: false,
        }
    }

    /// Abort the probe without sampling: failed forwards and error
    /// responses drop out of the EWMA entirely.
    pub fn cancel(mut self) {
        self.cancelled = true;
    }

    /// Buffered-lane variant: the caller already holds the request
    /// start (`t0`), so the probe measures the full round trip, not
    /// just the post-response build tail.
    pub fn from_started(
        map: std::sync::Arc<
            std::sync::Mutex<std::collections::HashMap<String, (f64, std::time::Instant)>>,
        >,
        model: &str,
        started: std::time::Instant,
    ) -> Self {
        Self {
            map,
            model: model.to_string(),
            started,
            cancelled: false,
        }
    }
}

impl Drop for LocalServiceStamp {
    fn drop(&mut self) {
        if self.cancelled {
            return;
        }
        let ms = self.started.elapsed().as_secs_f64() * 1000.0;
        let now = std::time::Instant::now();
        if let Ok(mut map) = self.map.lock() {
            match map.get_mut(&self.model) {
                Some((ewma, at)) => {
                    *ewma = crate::hint_ewma(Some(*ewma), ms);
                    *at = now;
                }
                None => {
                    map.insert(self.model.clone(), (ms, now));
                }
            }
        }
    }
}

/// Body wrapper that carries ONLY a [`LocalServiceStamp`] to stream
/// end - open-gateway lanes with no budget key still feed the local
/// service EWMA (mirrors [`charge_outgoing`] for the stamp-only case).
#[must_use]
pub fn stamp_service_body(
    resp: axum::response::Response,
    service: LocalServiceStamp,
) -> axum::response::Response {
    use futures::StreamExt as _;
    struct StampOnly(
        // Carried, never read: dropping this field stamps the EWMA.
        #[allow(dead_code)] LocalServiceStamp,
    );
    let (parts, body) = resp.into_parts();
    let finish = std::sync::Arc::new(StampOnly(service));
    let mapped = body.into_data_stream().map(move |r| {
        let _hold = std::sync::Arc::clone(&finish);
        r.map_err(|e| std::io::Error::other(e.to_string()))
    });
    let stream = mapped.chain(futures::stream::unfold((), |()| async { None }));
    axum::http::Response::from_parts(parts, axum::body::Body::from_stream(stream))
}

/// Completion-token ceiling for the strict reservation lane: the
/// request's explicit generation cap in any of the dialects blazar
/// serves (ollama `options.num_predict` / top-level `num_predict`,
/// `OpenAI` `max_tokens` / `max_completion_tokens`, Responses
/// `max_output_tokens`), else `default_tokens`. Non-positive values
/// (ollama's -1/-2 "infinite"/"fill context" arrive as non-u64 and
/// fall through; an explicit 0 is treated as "no ceiling stated")
/// resolve to the default too. Prompt-side tokens are deliberately
/// NOT estimated - there is no tokenizer at this layer - and settle
/// from the measured usage tail.
#[must_use]
pub fn completion_ceiling(req: &serde_json::Value, default_tokens: u64) -> u64 {
    let explicit = req
        .get("options")
        .and_then(|o| o.get("num_predict"))
        .or_else(|| req.get("num_predict"))
        .or_else(|| req.get("max_tokens"))
        .or_else(|| req.get("max_completion_tokens"))
        .or_else(|| req.get("max_output_tokens"))
        .and_then(serde_json::Value::as_u64)
        .filter(|n| *n > 0);
    explicit.unwrap_or(default_tokens)
}

/// Usage-tapped engine stream for lanes that TRANSLATE child bytes
/// before the client sees them (the anthropic SSE translator): feeds
/// the tail-sniffer with the raw child chunks and settles/charges at
/// stream end. Clean drain and client abort both finish (`Drop`),
/// matching [`charge_outgoing`] semantics for non-axum streams.
pub fn tap_usage<S, E>(
    stream: S,
    sniffer: Option<UsageSniffer>,
    limiter: std::sync::Arc<KeysLimiter>,
    lease: Option<BudgetLease>,
    service: Option<LocalServiceStamp>,
) -> impl futures::Stream<Item = Result<axum::body::Bytes, E>>
where
    S: futures::Stream<Item = Result<axum::body::Bytes, E>>,
{
    use futures::StreamExt as _;
    struct Tap(
        Option<UsageSniffer>,
        std::sync::Arc<KeysLimiter>,
        Option<BudgetLease>,
        // Carried, never read: dropping this field stamps the EWMA.
        #[allow(dead_code)] Option<LocalServiceStamp>,
    );
    impl Drop for Tap {
        fn drop(&mut self) {
            let lease = self.2.take();
            match self.0.take() {
                Some(s) => {
                    let mut s = s;
                    let total = s.sniff();
                    match lease {
                        Some(l) => l.settle(total),
                        None => {
                            if total > 0 {
                                self.1.charge_tokens(&s.key, total);
                            }
                        }
                    }
                }
                None => {
                    if let Some(l) = lease {
                        l.settle(0);
                    }
                }
            }
        }
    }
    let tap = std::sync::Arc::new(std::sync::Mutex::new(Tap(sniffer, limiter, lease, service)));
    let mapped = stream.map(move |r| {
        if let Ok(bytes) = r.as_ref()
            && let Some(t) = tap.lock().expect("usage tap").0.as_mut()
        {
            t.push(bytes);
        }
        r
    });
    mapped.chain(futures::stream::unfold((), |()| async { None }))
}

pub(crate) fn read_leading_int(bytes: &[u8]) -> Option<u64> {
    let mut i = 0;
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b':' || bytes[i] == b'"') {
        i += 1;
    }
    let start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == start {
        return None;
    }
    std::str::from_utf8(&bytes[start..i]).ok()?.parse().ok()
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use blazar_core::ApiKey;

    /// BLZ-2 truth table: control-plane routes demand an unscoped key
    /// the moment any keys exist; the open gateway stays open.
    #[test]
    fn unit__control_plane__unscoped_key_required_when_keys_exist() {
        let ctx = |name: &str| KeyCtx {
            name: name.into(),
            weight: 1,
        };
        // Open gateway: no keys configured, everything passes.
        let open = KeysLimiter::loaded(None, Vec::new());
        assert!(super::control_plane_authorized(&open, None).is_ok());
        assert!(super::control_plane_authorized(&open, Some(&ctx("anyone"))).is_ok());
        // Keyed gateway: unscoped admin passes, scoped refused, missing 401.
        let keyed = KeysLimiter::loaded(
            None,
            vec![key("admin", 0, 0, 0, &[]), key("scoped", 0, 0, 0, &["m1"])],
        );
        assert!(super::control_plane_authorized(&keyed, Some(&ctx("admin"))).is_ok());
        let (code, msg) = super::control_plane_authorized(&keyed, Some(&ctx("scoped")))
            .expect_err("scoped key must be refused");
        assert_eq!(code, 403);
        assert!(msg.contains("unscoped (admin)"), "{msg}");
        let (code, _) = super::control_plane_authorized(&keyed, None).expect_err("no key");
        assert_eq!(code, 401);
    }

    /// BLZ-2 truth table: warm-style model-specific actions pass on a
    /// pure scope check - allowlist match or admin, never quota.
    #[test]
    fn unit__scope_permits__model_specific_control_actions() {
        let ctx = |name: &str| KeyCtx {
            name: name.into(),
            weight: 1,
        };
        let open = KeysLimiter::loaded(None, Vec::new());
        assert!(super::scope_permits(&open, None, "anything").is_ok());
        let keyed = KeysLimiter::loaded(
            None,
            vec![
                key("admin", 0, 0, 0, &[]),
                key("scoped", 0, 0, 0, &["m1", "fam:*"]),
            ],
        );
        assert!(super::scope_permits(&keyed, Some(&ctx("admin")), "m2").is_ok());
        assert!(super::scope_permits(&keyed, Some(&ctx("scoped")), "m1").is_ok());
        assert!(super::scope_permits(&keyed, Some(&ctx("scoped")), "fam:x").is_ok());
        let (code, msg) = super::scope_permits(&keyed, Some(&ctx("scoped")), "m2")
            .expect_err("out-of-scope model");
        assert_eq!(code, 403);
        assert!(msg.contains("not scoped"), "{msg}");
        let (code, _) = super::scope_permits(&keyed, None, "m1").expect_err("no key");
        assert_eq!(code, 401);
        let (code, _) = super::scope_permits(&keyed, Some(&ctx("ghost")), "m1")
            .expect_err("unknown key fails closed");
        assert_eq!(code, 401);
    }

    /// Per-key MCP server allowlist: empty list = every server (admin
    /// posture), non-empty = exact/trailing-`*` allowlist; the open
    /// gateway and missing/unknown keys mirror `scope_permits`.
    #[test]
    fn unit__mcp_server_permits__per_key_allowlist_truth_table() {
        let ctx = |name: &str| KeyCtx {
            name: name.into(),
            weight: 1,
        };
        let open = KeysLimiter::loaded(None, Vec::new());
        assert!(super::mcp_server_permits(&open, None, "any").is_ok());
        let mut tooler = key("tooler", 0, 0, 0, &[]);
        tooler.mcp_servers = vec!["fs".into(), "web*".into()];
        let keyed = KeysLimiter::loaded(None, vec![key("admin", 0, 0, 0, &[]), tooler]);
        // Empty allowlist admits every configured server.
        assert!(super::mcp_server_permits(&keyed, Some(&ctx("admin")), "anything").is_ok());
        // Exact and trailing-`*` wildcard matches pass.
        assert!(super::mcp_server_permits(&keyed, Some(&ctx("tooler")), "fs").is_ok());
        assert!(super::mcp_server_permits(&keyed, Some(&ctx("tooler")), "web-fetch").is_ok());
        // Unlisted server: 403 with teaching text.
        let (code, msg) = super::mcp_server_permits(&keyed, Some(&ctx("tooler")), "shell")
            .expect_err("unlisted server must be refused");
        assert_eq!(code, 403);
        assert!(msg.contains("not scoped"), "{msg}");
        // Missing and unknown keys fail closed.
        let (code, _) = super::mcp_server_permits(&keyed, None, "fs").expect_err("no key");
        assert_eq!(code, 401);
        let (code, _) = super::mcp_server_permits(&keyed, Some(&ctx("ghost")), "fs")
            .expect_err("unknown key fails closed");
        assert_eq!(code, 401);
    }

    /// Tool-level truth table: `<server>:<tool>` selectors with
    /// trailing-`*` on either side, bare server entries naming every
    /// tool, and the empty-list-admits-all default. The server gate is
    /// a SEPARATE predicate that runs first - this one only narrows
    /// within (or regardless of) it.
    #[test]
    fn unit__mcp_tool_permits__selector_truth_table() {
        let ctx = |name: &str| KeyCtx {
            name: name.into(),
            weight: 1,
        };
        let open = KeysLimiter::loaded(None, Vec::new());
        assert!(super::mcp_tool_permits(&open, None, "any", "thing").is_ok());
        let mut tooler = key("tooler", 0, 0, 0, &[]);
        tooler.mcp_tools = vec![
            "fs:read*".into(), // read-prefixed tools of fs only
            "web*:*".into(),   // every tool of web-prefixed servers
            "shell".into(),    // bare name: every tool of that server
        ];
        let keyed = KeysLimiter::loaded(None, vec![key("admin", 0, 0, 0, &[]), tooler]);
        // Empty list admits every tool (admin posture).
        assert!(super::mcp_tool_permits(&keyed, Some(&ctx("admin")), "fs", "write_file").is_ok());
        // Exact + wildcard selectors.
        assert!(super::mcp_tool_permits(&keyed, Some(&ctx("tooler")), "fs", "read_file").is_ok());
        assert!(super::mcp_tool_permits(&keyed, Some(&ctx("tooler")), "fs", "read_dir").is_ok());
        assert!(
            super::mcp_tool_permits(&keyed, Some(&ctx("tooler")), "web-fetch", "search").is_ok()
        );
        assert!(super::mcp_tool_permits(&keyed, Some(&ctx("tooler")), "shell", "exec").is_ok());
        // Refusals: wrong tool on a listed server, unlisted server.
        let (code, msg) = super::mcp_tool_permits(&keyed, Some(&ctx("tooler")), "fs", "write_file")
            .expect_err("fs:write_file must be refused");
        assert_eq!(code, 403);
        assert!(msg.contains("not scoped"), "{msg}");
        assert!(
            super::mcp_tool_permits(&keyed, Some(&ctx("tooler")), "git", "status").is_err(),
            "server with no selector entry is refused"
        );
        // Missing and unknown keys fail closed (mirrors the server gate).
        let (code, _) =
            super::mcp_tool_permits(&keyed, None, "fs", "read_file").expect_err("no key");
        assert_eq!(code, 401);
        let (code, _) = super::mcp_tool_permits(&keyed, Some(&ctx("ghost")), "fs", "read_file")
            .expect_err("unknown key fails closed");
        assert_eq!(code, 401);
    }

    /// The two gates compose the way operators read the config: a tool
    /// selector can only NARROW within the server allowlist, never grant
    /// access the server gate refuses.
    #[test]
    fn unit__mcp_gates__tool_selector_cannot_widen_server_allowlist() {
        let ctx = |name: &str| KeyCtx {
            name: name.into(),
            weight: 1,
        };
        let mut narrow = key("narrow", 0, 0, 0, &[]);
        narrow.mcp_servers = vec!["fs".into()];
        narrow.mcp_tools = vec!["shell:*".into()]; // server NOT permitted
        let keyed = KeysLimiter::loaded(None, vec![narrow]);
        assert!(super::mcp_server_permits(&keyed, Some(&ctx("narrow")), "shell").is_err());
        // The tool gate alone would admit it - the server gate refuses,
        // and mcp.rs always runs server first.
        assert!(super::mcp_tool_permits(&keyed, Some(&ctx("narrow")), "shell", "exec").is_ok());
    }

    /// BLZ-1 truth table for the two tenancy postures. Owner match is
    /// the only always-refused pair; the flag decides whether unstamped
    /// (legacy/open-gateway) rows stay shared with keyed callers or
    /// answer as unknown. Keyless callers only exist when no keys are
    /// configured, so they admit either way.
    #[test]
    fn unit__owner_allows__legacy_open_shares_unstamped_rows() {
        let ctx = |name: &str| KeyCtx {
            name: name.into(),
            weight: 1,
        };
        let owner = Some("alice");
        let alice = ctx("alice");
        let bob = ctx("bob");
        // Foreign owner is refused regardless of posture.
        assert!(!owner_allows(owner, Some(&bob), true));
        assert!(!owner_allows(owner, Some(&bob), false));
        // Owner match admits.
        assert!(owner_allows(owner, Some(&alice), true));
        assert!(owner_allows(owner, Some(&alice), false));
        // Legacy posture: unstamped rows stay visible to every caller.
        assert!(owner_allows(None, Some(&bob), true));
        assert!(owner_allows(None, None, true));
        assert!(owner_allows(owner, None, true));
    }

    /// BLZ-1 strict mode: unstamped rows quarantine from keyed callers
    /// (answer-as-unknown), keyless/open-gateway posture is unchanged.
    #[test]
    fn unit__owner_allows__strict_mode_quarantines_unstamped_rows() {
        let ctx = |name: &str| KeyCtx {
            name: name.into(),
            weight: 1,
        };
        let bob = ctx("bob");
        assert!(!owner_allows(None, Some(&bob), false));
        // Keyless callers imply no keys are configured: still open.
        assert!(owner_allows(None, None, false));
        assert!(owner_allows(Some("alice"), None, false));
    }

    fn key(name: &str, rpm: u32, tpm: u64, daily: u64, models: &[&str]) -> ApiKey {
        ApiKey {
            name: name.into(),
            key: format!("plm_{name}"),
            models: models.iter().map(|s| String::from(*s)).collect(),
            mcp_servers: Vec::new(),
            mcp_tools: Vec::new(),
            rpm,
            tpm,
            daily_tokens: daily,
            max_concurrent: 0,
            weight: 1,
        }
    }

    #[test]
    fn unit__sniff_usage_parts__splits_prompt_and_completion_dialects() {
        let openai = br#"{"usage":{"prompt_tokens":11,"completion_tokens":7}}"#;
        assert_eq!(
            sniff_usage_parts(openai),
            (Some(11), Some(7)),
            "OpenAI dialect splits"
        );
        let responses = br#"{"usage":{"input_tokens":5,"output_tokens":9}}"#;
        assert_eq!(
            sniff_usage_parts(responses),
            (Some(5), Some(9)),
            "Responses dialect splits"
        );
        let ndjson = br#"{"prompt_eval_count":3,"eval_count":4}"#;
        assert_eq!(
            sniff_usage_parts(ndjson),
            (Some(3), Some(4)),
            "ollama NDJSON dialect splits"
        );
        let bare = br#"{"no_usage_here":true}"#;
        assert_eq!(sniff_usage_parts(bare), (None, None));
        assert_eq!(sniff_usage(&openai[..]), 18, "legacy total unchanged");
    }

    #[test]
    fn unit__scaled_total__multiplies_completion_only_and_saturates() {
        assert_eq!(scaled_total(Some(10), Some(20), 1), 30);
        assert_eq!(
            scaled_total(Some(10), Some(20), 3),
            70,
            "prompt once + n*completion"
        );
        assert_eq!(scaled_total(None, None, 3), 0);
        assert_eq!(
            scaled_total(None, Some(20), 0),
            20,
            "mult clamps to 1, not 0"
        );
        assert_eq!(
            scaled_total(Some(u64::MAX), Some(u64::MAX), 8),
            u64::MAX,
            "saturating, never wraps into a small charge"
        );
    }

    #[test]
    fn unit__utc_day__known_dates() {
        let day = |secs: u64| utc_day(UNIX_EPOCH + Duration::from_secs(secs));
        assert_eq!(day(0), "1970-01-01");
        assert_eq!(day(1_757_137_200), "2025-09-06"); // 2025-09-06T00:00Z
        assert_eq!(day(1_757_222_600), "2025-09-07"); // next day 23:23:20Z
    }

    #[test]
    fn unit__prune_window__drops_expired_prefix_keeps_live_tail() {
        // F54: charge-time front-drain keeps the rpm/tpm vecs bounded -
        // without it every rate check scanned the full request history.
        let now = Instant::now();
        let old = now
            .checked_sub(WINDOW + Duration::from_secs(1))
            .expect("clock goes back far enough for the fixture");
        let edge = now
            .checked_sub(
                WINDOW
                    .checked_sub(Duration::from_secs(1))
                    .expect("WINDOW > 1s"),
            )
            .expect("clock");
        let mut win = vec![old, edge, now];
        prune_window(&mut win, now);
        assert_eq!(win.len(), 2, "expired prefix dropped, live tail kept");
        assert!(win[0] >= edge, "boundary entry (just inside) survives");
        assert_eq!(win[1], now);
    }

    #[test]
    fn unit__secs_until_utc_midnight__bounds() {
        let s = secs_until_utc_midnight();
        assert!((1..=86_400).contains(&s), "midnight distance in range: {s}");
    }

    #[test]
    fn unit__replace_all__swaps_set_and_preserves_survivors_by_name() {
        let limiter = KeysLimiter::loaded(
            None,
            vec![
                blazar_core::ApiKey {
                    name: "admin".into(),
                    key: "plm_admin".into(),
                    ..Default::default()
                },
                blazar_core::ApiKey {
                    name: "scoped".into(),
                    key: "plm_scoped".into(),
                    models: vec!["m".into()],
                    ..Default::default()
                },
            ],
        );
        // New generation: scoped (scope widened) + brand-new key; admin
        // is gone. Order follows the file, not the old list.
        limiter.replace_all(vec![
            blazar_core::ApiKey {
                name: "scoped".into(),
                key: "plm_scoped".into(),
                ..Default::default()
            },
            blazar_core::ApiKey {
                name: "fresh".into(),
                key: "plm_fresh".into(),
                ..Default::default()
            },
        ]);
        let names: Vec<String> = limiter.entries().iter().map(|k| k.name.clone()).collect();
        assert_eq!(names, vec!["scoped".to_string(), "fresh".to_string()]);
        assert!(
            limiter.entry("admin").is_none(),
            "absent key dropped from the live set"
        );
        assert!(
            limiter.resolve("plm_scoped").is_some(),
            "survivor still authenticates"
        );
        // Back to empty = authless: the reverse reload direction.
        limiter.replace_all(Vec::new());
        assert!(limiter.is_empty(), "empty file set -> authless");
    }

    #[test]
    fn unit__ct_eq__equal_and_prefix_cases() {
        assert!(ct_eq("plm_alpha", "plm_alpha"), "identical secrets equal");
        assert!(!ct_eq("plm_alpha", "plm_beta"), "same-length mismatch");
        // Common-prefix mismatch must NOT early-exit into a false equal.
        assert!(!ct_eq("plm_aaaa", "plm_aaab"), "last-byte difference found");
        assert!(
            !ct_eq("plm_aaaa", "plm_aaaa2"),
            "length difference folds in"
        );
        assert!(!ct_eq("", "x"), "empty vs non-empty");
        assert!(ct_eq("", ""), "two empties equal");
    }

    #[test]
    fn unit__resolve__ct_scan_finds_exact_secret_only() {
        let lim = KeysLimiter::loaded(None, vec![key("ci", 0, 0, 0, &["m1"])]);
        let hit = lim.resolve("plm_ci").expect("exact secret resolves");
        assert_eq!(hit.name, "ci");
        assert!(
            lim.resolve("plm_cj").is_none(),
            "same-length wrong secret rejected"
        );
        assert!(
            lim.resolve("plm_c").is_none(),
            "prefix-only secret rejected"
        );
        assert!(lim.resolve("").is_none(), "empty presentation rejected");
    }

    #[test]
    fn unit__scope__wildcard_prefix_matching() {
        let lim = KeysLimiter::loaded(None, Vec::new());
        let k = key("ci", 0, 0, 0, &["qwen3*", "vllm:*"]);
        assert!(lim.admit(&k, "qwen3.5-9b").is_ok());
        assert!(lim.admit(&k, "vllm:whatever").is_ok());
        assert!(lim.admit(&k, "qwen2.5-0.5b-instruct").is_err());
        // Exact patterns still work; '*' alone = everything.
        let k = key("ci", 0, 0, 0, &["m1"]);
        assert!(lim.admit(&k, "m1").is_ok());
        assert!(lim.admit(&k, "m1x").is_err());
        let k = key("ci", 0, 0, 0, &["*"]);
        assert!(lim.admit(&k, "anything").is_ok());
    }

    #[test]
    fn unit__scope__rejects_unlisted_model() {
        let lim = KeysLimiter::loaded(None, Vec::new());
        let k = key("ci", 0, 0, 0, &["a", "b"]);
        assert!(lim.admit(&k, "a").is_ok());
        let err = lim.admit(&k, "c").unwrap_err();
        assert!(matches!(err, Rejection::Scope { .. }));
        let resp = err.to_response();
        assert_eq!(resp.status(), 403);
    }

    #[test]
    fn unit__rpm__rejects_at_window_cap() {
        let lim = KeysLimiter::loaded(None, Vec::new());
        let k = key("ci", 2, 0, 0, &[]);
        assert!(lim.admit(&k, "m").is_ok());
        assert!(lim.admit(&k, "m").is_ok());
        let err = lim.admit(&k, "m").unwrap_err();
        assert!(matches!(
            err,
            Rejection::Rate {
                kind: "requests-per-minute",
                ..
            }
        ));
        assert_eq!(err.to_response().status(), 429);
    }

    #[test]
    fn unit__rpm__admission_is_atomic_under_a_concurrent_burst() {
        // BLZ-3 pin: gates and the charge land in one lock step, so a
        // burst of threads racing the same rpm-limited key admits
        // EXACTLY rpm requests - never more. The old check/charge pair
        // let every racer pass the same pre-charge window.
        let rpm: u32 = 5;
        let racers = 4 * rpm as usize;
        let lim = KeysLimiter::loaded(None, Vec::new());
        let k = key("ci", rpm, 0, 0, &[]);
        let barrier = std::sync::Barrier::new(racers);
        let admitted = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..racers {
                s.spawn(|| {
                    barrier.wait();
                    if lim.admit(&k, "m").is_ok() {
                        admitted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    }
                });
            }
        });
        assert_eq!(
            admitted.load(std::sync::atomic::Ordering::SeqCst),
            rpm as usize,
            "exactly the cap is admitted, the rest are rejected"
        );
    }

    #[test]
    fn unit__daily_budget__blocks_after_quota() {
        let lim = KeysLimiter::loaded(None, Vec::new());
        let k = key("ci", 0, 0, 100, &[]);
        lim.charge_tokens("ci", 60);
        assert!(lim.admit(&k, "m").is_ok()); // 60 < 100
        lim.charge_tokens("ci", 40);
        let err = lim.admit(&k, "m").unwrap_err();
        assert!(matches!(
            err,
            Rejection::Rate {
                kind: "daily token budget",
                ..
            }
        ));
    }

    #[test]
    fn unit__reserve__holds_tpm_headroom_and_releases_on_drop() {
        let lim = std::sync::Arc::new(KeysLimiter::loaded(None, Vec::new()));
        let k = key("ci", 0, 100, 0, &[]);
        // Hold 60 of the 100 tpm: an overlapping ask of 41 no longer fits
        // (60 reserved + 41 > 100) even though nothing is charged yet.
        let lease = lim.reserve(&k, 60).unwrap().expect("lease granted");
        let err = lim.reserve(&k, 41).unwrap_err();
        assert!(matches!(
            err,
            Rejection::Rate {
                kind: "token budget reservation (tokens-per-minute)",
                ..
            }
        ));
        assert_eq!(err.to_response().status(), 429);
        // An aborted request drops the lease: headroom returns untouched
        // (nothing was charged, so the full 100 is bookable again).
        drop(lease);
        assert!(lim.reserve(&k, 41).unwrap().is_some());
    }

    #[test]
    fn unit__reserve__settle_charges_actual_and_frees_remainder() {
        let lim = std::sync::Arc::new(KeysLimiter::loaded(None, Vec::new()));
        let k = key("ci", 0, 0, 100, &[]);
        // Daily lane: reserve 60, generation only measured 20 tokens.
        // settle must release the 40 held and charge the 20 spent, so a
        // follow-up reservation sees 80 headroom, not 40.
        let lease = lim.reserve(&k, 60).unwrap().expect("lease granted");
        let err = lim.reserve(&k, 41).unwrap_err();
        assert!(matches!(
            err,
            Rejection::Rate {
                kind: "token budget reservation (daily token budget)",
                ..
            }
        ));
        lease.settle(20);
        let held = lim.reserve(&k, 80).unwrap().expect("80 bookable"); // 20+80=100
        // With those 80 still held, one more token exceeds the daily cap.
        assert_eq!(
            lim.reserve(&k, 1)
                .unwrap_err()
                .to_response()
                .status()
                .as_u16(),
            429
        );
        drop(held);
    }

    #[test]
    fn unit__reserve__no_budget_or_zero_estimate_is_none() {
        let lim = std::sync::Arc::new(KeysLimiter::loaded(None, Vec::new()));
        // No tpm/daily caps: reservation is pointless, admission says None
        // so callers skip lease plumbing entirely.
        let k = key("ci", 0, 0, 0, &[]);
        assert!(lim.reserve(&k, 500).unwrap().is_none());
        // Capped key but a zero estimate (empty ceiling): same skip.
        let k = key("ci", 0, 100, 100, &[]);
        assert!(lim.reserve(&k, 0).unwrap().is_none());
    }

    #[test]
    fn unit__completion_ceiling__reads_all_dialects_and_falls_back() {
        use serde_json::json;
        let d = 1024_u64;
        // One completion-cap knob per API dialect.
        assert_eq!(
            completion_ceiling(&json!({"options": {"num_predict": 7}}), d),
            7
        );
        assert_eq!(completion_ceiling(&json!({"num_predict": 11}), d), 11);
        assert_eq!(completion_ceiling(&json!({"max_tokens": 13}), d), 13);
        assert_eq!(
            completion_ceiling(&json!({"max_completion_tokens": 17}), d),
            17
        );
        assert_eq!(completion_ceiling(&json!({"max_output_tokens": 19}), d), 19);
        // Absent / zero / negative knobs all fall back to the configured
        // default rather than admitting "unlimited".
        assert_eq!(completion_ceiling(&json!({}), d), d);
        assert_eq!(completion_ceiling(&json!({"max_tokens": 0}), d), d);
        assert_eq!(completion_ceiling(&json!({"max_tokens": -5}), d), d);
    }

    #[test]
    fn unit__concurrency__cap_blocks_and_releases_on_drop() {
        let lim = std::sync::Arc::new(KeysLimiter::loaded(None, Vec::new()));
        // cap=1: first acquire leases, second rejects while held.
        let lease = lim.acquire("ci", 1).unwrap();
        assert!(lease.is_some());
        let err = lim.acquire("ci", 1).unwrap_err();
        assert!(matches!(
            err,
            Rejection::Rate {
                kind: "concurrent requests",
                ..
            }
        ));
        assert_eq!(err.to_response().status(), 429);
        drop(lease);
        // Slot freed: next acquire succeeds again.
        assert!(lim.acquire("ci", 1).unwrap().is_some());
    }

    #[test]
    fn unit__concurrency__zero_cap_is_unlimited_and_untracked() {
        let lim = std::sync::Arc::new(KeysLimiter::loaded(None, Vec::new()));
        for _ in 0..5 {
            assert!(lim.acquire("ci", 0).unwrap().is_none());
        }
    }

    #[test]
    fn unit__concurrency__leases_do_not_touch_daily_counters() {
        let lim = std::sync::Arc::new(KeysLimiter::loaded(None, Vec::new()));
        let k = key("ci", 0, 0, 10, &[]);
        let _lease = lim.acquire("ci", 2).unwrap();
        let _lease2 = lim.acquire("ci", 2).unwrap();
        lim.charge_tokens("ci", 10);
        // Leases are in-flight, not tokens: the daily budget check is
        // unaffected by held slots.
        assert!(matches!(
            lim.admit(&k, "m"),
            Err(Rejection::Rate {
                kind: "daily token budget",
                ..
            })
        ));
        // Third concurrent slot still rejects at cap 2.
        assert!(matches!(
            lim.acquire("ci", 2),
            Err(Rejection::Rate {
                kind: "concurrent requests",
                ..
            })
        ));
    }

    #[test]
    fn unit__usage_sniffer__three_dialects() {
        let openai =
            br#"..."finish_reason":"stop"}],"usage":{"prompt_tokens":120,"completion_tokens":45}}"#;
        assert_eq!(sniff_usage(openai), 165);
        let responses = br#"..."type":"response.completed","response":{"usage":{"input_tokens":80,"output_tokens":20}}}"#;
        assert_eq!(sniff_usage(responses), 100);
        let ndjson = br#"..."done":true,"prompt_eval_count":31,"eval_count":9}"#;
        assert_eq!(sniff_usage(ndjson), 40);
        assert_eq!(sniff_usage(b"no usage here"), 0);
    }

    #[test]
    fn unit__sniffer_tail__bounded_and_charged() {
        let lim = KeysLimiter::loaded(None, Vec::new());
        let mut s = UsageSniffer::new("ci");
        let filler = vec![b'x'; TAIL]; // pushes past the cap
        s.push(&filler);
        s.push(br#"{"usage":{"prompt_tokens":7,"completion_tokens":3}}"#);
        let total = s.finish(&lim);
        assert_eq!(total, 10);
        let snap = lim.usage_snapshot();
        assert!(
            snap.iter()
                .any(|(n, _, req, tok)| n == "ci" && *tok == 10 && *req == 0)
        );
    }

    #[test]
    fn unit__sniffer_frames__stand_in_when_no_usage_payload() {
        // Usage-less SSE stream + frame counting on: the observed
        // token frames charge instead of zero (lower-bound completion).
        let mut s = UsageSniffer::new("ci").with_frame_counting(true);
        s.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n");
        s.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n");
        s.push(b"data: [DONE]\n\n");
        assert_eq!(s.sniff(), 2);
        // Frame counting off (default): same body stays uncounted -
        // the fallback is opt-in per streaming lane.
        let mut s = UsageSniffer::new("ci");
        s.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n");
        assert_eq!(s.sniff(), 0);
    }

    #[test]
    fn unit__sniffer_frames__exact_usage_beats_frame_count() {
        // The stream carries BOTH token frames and a final usage
        // chunk: the exact payload wins, and the completion multiplier
        // still applies to the reported side (llama.cpp per-choice
        // reporting) - frames are never mixed into an exact reading.
        let mut s = UsageSniffer::new("ci")
            .with_frame_counting(true)
            .with_completion_multiplier(2);
        s.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n");
        s.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n");
        s.push(
            b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":3}}\n\n",
        );
        assert_eq!(s.sniff(), 9 + 3 * 2);
    }

    /// `LocalServiceStamp` folds request-to-completion milliseconds
    /// into a per-model EWMA on drop; successive observations blend at
    /// the hint rate (0.3), and `from_started` measures from the
    /// caller's `t0` (buffered round trips).
    #[test]
    fn unit__local_service_stamp__drop_stamps_per_model_ewma() {
        let map: std::sync::Arc<
            std::sync::Mutex<std::collections::HashMap<String, (f64, std::time::Instant)>>,
        > = std::sync::Arc::default();
        {
            let s = LocalServiceStamp::new(std::sync::Arc::clone(&map), "m1");
            std::thread::sleep(std::time::Duration::from_millis(30));
            drop(s);
        }
        let first = map.lock().unwrap().get("m1").expect("first stamp landed").0;
        assert!(first >= 30.0, "elapsed ms observed, got {first}");
        {
            // Second observation blends: 0.7 * first + 0.3 * 100.
            let started = std::time::Instant::now()
                .checked_sub(std::time::Duration::from_millis(100))
                .expect("100 ms is always in the past on test hosts");
            let s = LocalServiceStamp::from_started(std::sync::Arc::clone(&map), "m1", started);
            drop(s);
        }
        let blended = map.lock().unwrap().get("m1").unwrap().0;
        // 0.7*first + 0.3*100; assert the blend moved toward 100 from
        // first without pinning wall-clock exactness.
        assert!(
            blended > first * 0.69 && blended < first + 31.0,
            "EWMA blend 0.3 between {first} and 100, got {blended}"
        );
        // Timestamps ride along for the pools' staleness decay.
        assert!(
            map.lock().unwrap().get("m1").unwrap().1.elapsed() < std::time::Duration::from_secs(1)
        );
        // A different model keeps its own entry.
        assert!(!map.lock().unwrap().contains_key("m2"));
    }
}
