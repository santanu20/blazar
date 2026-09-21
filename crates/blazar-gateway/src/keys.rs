//! Per-key gateway security tier: model scoping, rate limits
//! (rpm/tpm), daily token budgets, concurrency caps, and usage
//! accounting.
//!
//! Zero-tax contract: with no `[[keys]]` configured, nothing here runs.
//! With keys but no token budgets, only exact request counting applies
//! (no body bytes are read). Token counting activates a bounded
//! tail-sniffer on the proxied byte stream — constructed per request
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
                    "error": {"type": "blazar_key_scope", "code": 403,
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
                    "error": {"type": "blazar_key_rate", "code": 429,
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

/// Seconds until the next UTC midnight (`1..=86_400`) — the true reset
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
    /// Live leases (concurrency cap); instantaneous state — never
    /// rolled by `roll_day` nor persisted.
    in_flight: u32,
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
    /// Build with today's persisted usage pre-loaded (best effort — a
    /// store failure degrades to zero-based counters, never blocks boot).
    #[must_use]
    pub fn loaded(store: Option<&Store>, entries: Vec<blazar_core::ApiKey>) -> Self {
        let today = utc_day(SystemTime::now());
        let mut pre: HashMap<String, KeyState> = HashMap::new();
        if let Some(s) = store {
            if let Ok(rows) = s.key_usage(&today) {
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
    /// secret for some earlier entry — position leaks nothing without
    /// the key, and per-compare `ct_eq` hides content differences.
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

    /// Admission check: model scope, then rpm, then tpm, then the daily
    /// budget. Pure read (the rpm/tpm windows are advanced at charge time,
    /// so a checked-but-aborted request never consumed budget).
    pub fn check(&self, key: &blazar_core::ApiKey, model: &str) -> Result<(), Rejection> {
        if !key.models.is_empty() && !key.models.iter().any(|m| scope_matches(m, model)) {
            return Err(Rejection::Scope {
                name: key.name.clone(),
                model: model.to_string(),
            });
        }
        if key.rpm == 0 && key.tpm == 0 && key.daily_tokens == 0 {
            return Ok(());
        }
        let now = Instant::now();
        let today = utc_day(SystemTime::now());
        let mut states = self.states.lock().expect("keys state poisoned");
        let st = states.entry(key.name.clone()).or_default();
        st.roll_day(&today);
        if key.rpm > 0
            && st
                .rpm
                .iter()
                .filter(|t| now.duration_since(**t) < WINDOW)
                .count()
                >= usize::try_from(key.rpm).unwrap_or(usize::MAX)
        {
            let oldest = st
                .rpm
                .iter()
                .find(|t| now.duration_since(**t) < WINDOW)
                .map_or(1, |t| {
                    WINDOW
                        .checked_sub(now.duration_since(*t))
                        .unwrap_or(WINDOW)
                        .as_secs()
                        .max(1)
                });
            return Err(Rejection::Rate {
                kind: "requests-per-minute",
                retry_after_secs: oldest,
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
        Ok(())
    }

    /// Count one admitted request. Called once per request after the
    /// scope/rate check passes — exact for every route, body-free.
    pub fn charge_request(&self, name: &str) {
        let now = Instant::now();
        let today = utc_day(SystemTime::now());
        let mut states = self.states.lock().expect("keys state poisoned");
        let st = states.entry(name.to_string()).or_default();
        st.roll_day(&today);
        prune_window(&mut st.rpm, now);
        st.rpm.push(now);
        st.requests += 1;
        st.dirty = true;
    }

    /// Add measured tokens for a finished chat-family request (exact
    /// values from the response-usage tail, zero when the sniffer found
    /// none — e.g. streaming without a usage chunk).
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
    /// scope (cheapest gate first — a capped key cannot even probe
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
    /// mid-flight keeps its state entry — `get_mut` only decrements and
    /// never inserts, so nothing is re-created here.
    fn release(&self, name: &str) {
        let mut states = self.states.lock().expect("keys state poisoned");
        if let Some(st) = states.get_mut(name) {
            st.in_flight = st.in_flight.saturating_sub(1);
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
    /// that day — `roll_day` resets counters on first charge of the new
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
        dirty: st.dirty,
    }
}

/// RAII in-flight lease: increments on acquire, releases when dropped.
/// The guard lives inside the response body, so the slot is held for
/// exactly as long as the response streams (dropped on completion OR
/// client disconnect — hyper drops bodies either way).
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
/// token budgets — everyone else pays nothing.
pub struct UsageSniffer {
    tail: Vec<u8>,
    key: String,
}

const TAIL: usize = 8 * 1024;

impl UsageSniffer {
    #[must_use]
    pub fn new(key: &str) -> Self {
        Self {
            tail: Vec::with_capacity(TAIL),
            key: key.to_string(),
        }
    }

    pub fn push(&mut self, bytes: &[u8]) {
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

    /// Parse the tail + charge the key. Returns the token total found.
    pub fn finish(self, limiter: &KeysLimiter) -> u64 {
        let total = sniff_usage(&self.tail);
        limiter.charge_tokens(&self.key, total);
        total
    }
}

/// Extract (prompt + completion) tokens from a response tail. Pure.
#[must_use]
pub fn sniff_usage(tail: &[u8]) -> u64 {
    let prompt = last_int_after(tail, "\"prompt_tokens\"")
        .or_else(|| last_int_after(tail, "\"input_tokens\""))
        .or_else(|| last_int_after(tail, "\"prompt_eval_count\""));
    let completion = last_int_after(tail, "\"completion_tokens\"")
        .or_else(|| last_int_after(tail, "\"output_tokens\""))
        .or_else(|| last_int_after(tail, "\"eval_count\""));
    prompt.unwrap_or(0).saturating_add(completion.unwrap_or(0))
}

/// Shared substring scan (gateway-internal): cheap gate before any JSON
/// parse on streamed bodies.
pub(crate) fn find_sub(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Value of the LAST `<key> <int>` occurrence in a streamed tail — the
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
/// `Drop` — a client disconnect still charges what was generated.
#[must_use]
pub fn charge_outgoing(
    resp: axum::response::Response,
    name: &str,
    limiter: std::sync::Arc<KeysLimiter>,
) -> axum::response::Response {
    use futures::StreamExt as _;
    /// Abort-safe finisher: `Drop` runs on clean drain AND client abort.
    struct Finish(Option<UsageSniffer>, std::sync::Arc<KeysLimiter>);
    impl Drop for Finish {
        fn drop(&mut self) {
            if let Some(s) = self.0.take() {
                s.finish(&self.1);
            }
        }
    }
    let (parts, body) = resp.into_parts();
    let finish = std::sync::Arc::new(std::sync::Mutex::new(Finish(
        Some(UsageSniffer::new(name)),
        limiter,
    )));
    let mapped = body.into_data_stream().map(move |r| {
        if let Ok(bytes) = r.as_ref() {
            if let Some(f) = finish.lock().expect("usage finisher").0.as_mut() {
                f.push(bytes);
            }
        }
        r.map_err(|e| std::io::Error::other(e.to_string()))
    });
    let stream = mapped.chain(futures::stream::unfold((), |()| async { None }));
    axum::http::Response::from_parts(parts, axum::body::Body::from_stream(stream))
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

    fn key(name: &str, rpm: u32, tpm: u64, daily: u64, models: &[&str]) -> ApiKey {
        ApiKey {
            name: name.into(),
            key: format!("plm_{name}"),
            models: models.iter().map(|s| String::from(*s)).collect(),
            rpm,
            tpm,
            daily_tokens: daily,
            max_concurrent: 0,
            weight: 1,
        }
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
        // F54: charge-time front-drain keeps the rpm/tpm vecs bounded —
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
        assert!(lim.check(&k, "qwen3.5-9b").is_ok());
        assert!(lim.check(&k, "vllm:whatever").is_ok());
        assert!(lim.check(&k, "qwen2.5-0.5b-instruct").is_err());
        // Exact patterns still work; '*' alone = everything.
        let k = key("ci", 0, 0, 0, &["m1"]);
        assert!(lim.check(&k, "m1").is_ok());
        assert!(lim.check(&k, "m1x").is_err());
        let k = key("ci", 0, 0, 0, &["*"]);
        assert!(lim.check(&k, "anything").is_ok());
    }

    #[test]
    fn unit__scope__rejects_unlisted_model() {
        let lim = KeysLimiter::loaded(None, Vec::new());
        let k = key("ci", 0, 0, 0, &["a", "b"]);
        assert!(lim.check(&k, "a").is_ok());
        let err = lim.check(&k, "c").unwrap_err();
        assert!(matches!(err, Rejection::Scope { .. }));
        let resp = err.to_response();
        assert_eq!(resp.status(), 403);
    }

    #[test]
    fn unit__rpm__rejects_at_window_cap() {
        let lim = KeysLimiter::loaded(None, Vec::new());
        let k = key("ci", 2, 0, 0, &[]);
        assert!(lim.check(&k, "m").is_ok());
        lim.charge_request("ci");
        assert!(lim.check(&k, "m").is_ok());
        lim.charge_request("ci");
        let err = lim.check(&k, "m").unwrap_err();
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
    fn unit__daily_budget__blocks_after_quota() {
        let lim = KeysLimiter::loaded(None, Vec::new());
        let k = key("ci", 0, 0, 100, &[]);
        lim.charge_tokens("ci", 60);
        assert!(lim.check(&k, "m").is_ok()); // 60 < 100
        lim.charge_tokens("ci", 40);
        let err = lim.check(&k, "m").unwrap_err();
        assert!(matches!(
            err,
            Rejection::Rate {
                kind: "daily token budget",
                ..
            }
        ));
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
            lim.check(&k, "m"),
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
        assert!(snap
            .iter()
            .any(|(n, _, req, tok)| n == "ci" && *tok == 10 && *req == 0));
    }
}
