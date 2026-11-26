//! Cancel-and-replay preemption (opt-in: `preemption = "replay"`).
//!
//! A request parked in the admission queue behind long generations can
//! take a slot from a running request of strictly lower urgency on the
//! same model. llama.cpp cannot resume an interrupted generation: a
//! saved slot restores the prompt cache, never the decode (measured on
//! b12606: a cancel frees the slot in about 1 ms, a restore reuses 73 of
//! 74 prompt tokens and then starts a new answer). So the running
//! request is cancelled and replayed in place: it re-enters the queue
//! with its own priority and its client only waits longer.
//!
//! Two invariants keep that invisible to the cancelled request's client:
//! the victim has not sent one body byte yet, and every request is
//! replayed at most once, and its in-flight lease goes straight to the
//! request that cancelled it, so the replay queues again behind it.
//! Thrash is bounded three ways: a victim must
//! have run for [`MIN_VICTIM_AGE`], a model grants [`BUDGET_PER_MINUTE`]
//! preemptions per rolling minute, and equal urgencies never preempt
//! each other (FIFO between peers).

use std::collections::{HashMap, VecDeque};
use std::fmt::Write as _;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use axum::body::Bytes;
use futures::{Stream, StreamExt};
use tokio::sync::watch;

use crate::proxy::{ChildSendError, InFlightGuard};
use crate::queue::{Priority, WorkClass};
use crate::state::AppState;

/// A request younger than this keeps its slot: its prefill may still be
/// running, and cancelling it would only move the wait around.
pub const MIN_VICTIM_AGE: Duration = Duration::from_millis(250);
/// Preemptions a model grants per rolling minute. Long generations
/// arrive in bursts; past this the queue falls back to FIFO ordering.
pub const BUDGET_PER_MINUTE: usize = 6;
const BUDGET_WINDOW: Duration = Duration::from_secs(60);

const OUTCOMES: [&str; 4] = ["cancelled", "replayed", "replay_failed", "too_late"];
const REFUSALS: [&str; 2] = ["no_candidate", "budget"];

type ByteStream = Pin<Box<dyn Stream<Item = std::io::Result<Bytes>> + Send>>;
type SendFuture<'a> =
    Pin<Box<dyn Future<Output = Result<reqwest::Response, ChildSendError>> + Send + 'a>>;

#[must_use]
pub fn enabled(mode: &str) -> bool {
    mode == "replay"
}

/// (priority, work class) as one comparable value. A request yields to
/// every larger value and to no smaller or equal one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Urgency(u8, u8);

impl Urgency {
    #[must_use]
    pub fn new(priority: Priority, class: WorkClass) -> Self {
        Self(priority as u8, class_rank(class))
    }

    #[must_use]
    pub fn priority(self) -> Priority {
        match self.0 {
            0 => Priority::Low,
            1 => Priority::Normal,
            _ => Priority::High,
        }
    }

    #[must_use]
    pub fn class(self) -> WorkClass {
        match self.1 {
            2 => WorkClass::Tool,
            1 => WorkClass::Interactive,
            _ => WorkClass::RawLong,
        }
    }
}

fn class_rank(class: WorkClass) -> u8 {
    match class {
        WorkClass::Tool => 2,
        WorkClass::Interactive => 1,
        WorkClass::RawLong => 0,
    }
}

/// Where the in-flight lease of a request lives. A replay hands its
/// lease back before it re-queues and takes a new one when it is
/// admitted again; the response body's drop chain releases whichever
/// lease is current when the stream ends.
pub enum GuardSlot {
    /// Preemption off: the lease never moves.
    Fixed(Option<InFlightGuard>),
    /// Preemption on: the current lease, swapped by a replay.
    Shared(Arc<Mutex<Option<InFlightGuard>>>),
}

impl GuardSlot {
    #[must_use]
    pub fn shared(guard: Option<InFlightGuard>) -> (Self, Arc<Mutex<Option<InFlightGuard>>>) {
        let cell = Arc::new(Mutex::new(guard));
        (Self::Shared(Arc::clone(&cell)), cell)
    }

    fn take(cell: &Arc<Mutex<Option<InFlightGuard>>>) -> Option<InFlightGuard> {
        cell.lock().unwrap_or_else(PoisonError::into_inner).take()
    }

    /// Remove the current lease without ending it: a cancelled request
    /// hands it to the request that cancelled it. Dropping the returned
    /// guard would release it into the pool instead.
    fn take_lease(&self) -> Option<InFlightGuard> {
        match self {
            Self::Shared(cell) => Self::take(cell),
            Self::Fixed(_) => None,
        }
    }

    /// Park a fresh lease (taken after a replay was admitted again).
    pub fn hold(&self, guard: Option<InFlightGuard>) {
        if let Self::Shared(cell) = self
            && let Some(guard) = guard
        {
            *cell.lock().unwrap_or_else(PoisonError::into_inner) = Some(guard);
        }
    }

    /// Release the current lease at stream end.
    pub fn finish(self) {
        match self {
            Self::Fixed(guard) => drop(guard),
            Self::Shared(cell) => drop(Self::take(&cell)),
        }
    }
}

struct Victim {
    model: String,
    urgency: Urgency,
    started: Instant,
    forwarded: Arc<AtomicU64>,
    replays: Arc<AtomicU8>,
    cancel: watch::Sender<bool>,
    /// Set by the preemptor right before the cancel: where the victim
    /// sends its lease once its child request is dropped.
    handover: Option<tokio::sync::oneshot::Sender<InFlightGuard>>,
}

#[derive(Default)]
struct Inner {
    next_id: u64,
    victims: HashMap<u64, Victim>,
    budget: HashMap<String, VecDeque<Instant>>,
}

/// Running requests that may still be cancelled, per model, plus the
/// preemption budget and the outcome counters.
#[derive(Default)]
pub struct Registry {
    inner: Mutex<Inner>,
    outcomes: [AtomicU64; OUTCOMES.len()],
    refusals: [AtomicU64; REFUSALS.len()],
}

fn lock(inner: &Mutex<Inner>) -> std::sync::MutexGuard<'_, Inner> {
    inner.lock().unwrap_or_else(PoisonError::into_inner)
}

/// What a waiting request found when its preemptor fired.
enum Outcome {
    /// A victim was cancelled. The receiver yields the victim's lease
    /// once it dropped its child request, or closes when it never does.
    Cancelled(tokio::sync::oneshot::Receiver<InFlightGuard>),
    NoCandidate,
    BudgetExhausted,
}

/// The candidate to cancel: same model, strictly less urgent than the
/// waiter, not yet replayed, no body byte sent, past the minimum age.
/// The youngest matching request wins (least work thrown away).
fn choose<'a>(
    victims: impl Iterator<Item = (u64, &'a Victim)>,
    model: &str,
    waiter: Urgency,
    now: Instant,
) -> Option<u64> {
    victims
        .filter(|(_, victim)| {
            victim.model == model
                && victim.urgency < waiter
                && victim.replays.load(Ordering::Relaxed) == 0
                && victim.forwarded.load(Ordering::Relaxed) == 0
                && now.duration_since(victim.started) >= MIN_VICTIM_AGE
        })
        .max_by_key(|(_, victim)| victim.started)
        .map(|(id, _)| id)
}

impl Registry {
    /// Count a running request as a possible victim until the returned
    /// ticket is dropped (the response stream ended or was aborted).
    pub fn register(self: &Arc<Self>, model: &str, urgency: Urgency) -> Ticket {
        let (cancel, cancelled) = watch::channel(false);
        let forwarded = Arc::new(AtomicU64::new(0));
        let replays = Arc::new(AtomicU8::new(0));
        let id = {
            let mut inner = lock(&self.inner);
            inner.next_id += 1;
            let id = inner.next_id;
            inner.victims.insert(
                id,
                Victim {
                    model: model.to_string(),
                    urgency,
                    started: Instant::now(),
                    forwarded: Arc::clone(&forwarded),
                    replays: Arc::clone(&replays),
                    cancel,
                    handover: None,
                },
            );
            id
        };
        Ticket {
            registry: Arc::clone(self),
            id,
            forwarded,
            cancelled,
        }
    }

    fn unregister(&self, id: u64) {
        lock(&self.inner).victims.remove(&id);
    }

    fn try_preempt(&self, model: &str, waiter: Urgency, now: Instant) -> Outcome {
        let mut inner = lock(&self.inner);
        let over_budget = {
            let recent = inner.budget.entry(model.to_string()).or_default();
            while recent
                .front()
                .is_some_and(|at| now.duration_since(*at) > BUDGET_WINDOW)
            {
                recent.pop_front();
            }
            recent.len() >= BUDGET_PER_MINUTE
        };
        if over_budget {
            self.refusals[1].fetch_add(1, Ordering::Relaxed);
            return Outcome::BudgetExhausted;
        }
        let pick = choose(
            inner.victims.iter().map(|(id, victim)| (*id, victim)),
            model,
            waiter,
            now,
        );
        let Some(id) = pick else {
            self.refusals[0].fetch_add(1, Ordering::Relaxed);
            return Outcome::NoCandidate;
        };
        let victim = inner
            .victims
            .get_mut(&id)
            .expect("a chosen victim is registered");
        victim.replays.store(1, Ordering::Relaxed);
        let (handover, lease) = tokio::sync::oneshot::channel();
        victim.handover = Some(handover);
        let _ = victim.cancel.send(true);
        inner
            .budget
            .get_mut(model)
            .expect("the budget row exists")
            .push_back(now);
        Outcome::Cancelled(lease)
    }

    /// The channel a cancelled request hands its lease over on. Taken
    /// once: a second cancel attempt finds nothing to deliver to.
    fn handover_of(&self, id: u64) -> Option<tokio::sync::oneshot::Sender<InFlightGuard>> {
        lock(&self.inner).victims.get_mut(&id)?.handover.take()
    }

    /// The engine's own per-request usage, exposed on `/metrics`.
    #[must_use]
    pub fn metrics(&self) -> String {
        let mut out = String::new();
        out.push_str(
            "# HELP blazar_preemptions_total Running requests cancelled for a waiting request of higher urgency, by what happened to the cancelled request\n# TYPE blazar_preemptions_total counter\n",
        );
        for (label, count) in OUTCOMES.iter().zip(&self.outcomes) {
            let _ = writeln!(
                out,
                "blazar_preemptions_total{{outcome=\"{label}\"}} {}\n",
                count.load(Ordering::Relaxed)
            );
        }
        out.push_str(
            "# HELP blazar_preemption_refusals_total Waiting requests that found nothing they may cancel, by reason\n# TYPE blazar_preemption_refusals_total counter\n",
        );
        for (label, count) in REFUSALS.iter().zip(&self.refusals) {
            let _ = writeln!(
                out,
                "blazar_preemption_refusals_total{{reason=\"{label}\"}} {}\n",
                count.load(Ordering::Relaxed)
            );
        }
        out
    }

    fn count_outcome(&self, index: usize) {
        self.outcomes[index].fetch_add(1, Ordering::Relaxed);
    }
}

/// A registered running request: counts the bytes it has sent to its
/// client and receives the cancel signal.
pub struct Ticket {
    registry: Arc<Registry>,
    id: u64,
    forwarded: Arc<AtomicU64>,
    cancelled: watch::Receiver<bool>,
}

impl Ticket {
    /// Completes once this request has been chosen as a victim.
    pub async fn cancel_changed(&mut self) {
        let _ = self.cancelled.changed().await;
    }

    /// True while a replay is still allowed: the first body byte sent
    /// to the client closes that window for good.
    fn still_virgin(&self) -> bool {
        self.forwarded.load(Ordering::Relaxed) == 0
    }

    /// A cancel that arrived after the first body byte: nothing is
    /// cancelled, the request finishes on its own. A request that has
    /// streamed is never chosen again, so its immunity is untouched.
    fn too_late(&self) {
        self.registry.count_outcome(3);
    }

    /// Give the in-flight lease to the request that cancelled this one
    /// instead of back to the pool: the slot changes hands exactly once
    /// and no other waiter can take it in between. Called after the
    /// child request was dropped, so the engine slot is free by the time
    /// the new holder sends. False when nobody waits for the lease (the
    /// cancel arrived too late): it is then released like any other.
    fn hand_over(&self, lease: Option<InFlightGuard>) -> bool {
        let handover = self.registry.handover_of(self.id);
        match (handover, lease) {
            (Some(handover), Some(lease)) => {
                let _ = handover.send(lease);
                self.registry.count_outcome(0);
                true
            }
            (handover, lease) => {
                drop((handover, lease));
                false
            }
        }
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        self.registry.unregister(self.id);
    }
}

/// Everything a replay needs: the child request exactly as first sent
/// (post gateway mutations) and the queue class to re-enter with.
pub struct ReplayPlan {
    state: Arc<AppState>,
    model: String,
    method: axum::http::Method,
    fwd_path: String,
    headers: axum::http::HeaderMap,
    body: Bytes,
    urgency: Urgency,
}

impl ReplayPlan {
    /// `Some` only where a replay is well defined: a chat route, the
    /// feature on, a request that holds an in-flight lease, and no
    /// pinned-recipe lane (that lane rewrites the forward path and its
    /// adapter state is not re-creatable here).
    #[allow(clippy::too_many_arguments)] // flat call params mirroring the proxy forward seam
    #[allow(clippy::fn_params_excessive_bools)]
    pub fn new(
        state: &Arc<AppState>,
        model: &str,
        path_query: &str,
        method: &axum::http::Method,
        fwd_path: &str,
        headers: &axum::http::HeaderMap,
        original_body: &[u8],
        accounted: bool,
        plain_lane: bool,
    ) -> Option<Self> {
        if !enabled(&state.cfg().preemption)
            || !accounted
            || !plain_lane
            || !crate::proxy::is_chat_route(path_query)
        {
            return None;
        }
        let urgency = Urgency::new(
            Priority::from_header(
                headers
                    .get("x-blazar-priority")
                    .and_then(|value| value.to_str().ok()),
            ),
            crate::workload::classify_for(
                state,
                model,
                crate::workload::WorkloadSurface::Chat,
                crate::openai::body_has_tools(original_body),
                false,
            ),
        );
        Some(Self {
            state: Arc::clone(state),
            model: model.to_string(),
            method: method.clone(),
            fwd_path: fwd_path.to_string(),
            headers: headers.clone(),
            body: Bytes::copy_from_slice(original_body),
            urgency,
        })
    }

    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    #[must_use]
    pub fn urgency(&self) -> Urgency {
        self.urgency
    }

    /// The same child request again, on whichever engine serves now.
    fn send(&self, engine: blazar_runtime::EngineRef) -> SendFuture<'_> {
        Box::pin(async move {
            let url = format!(
                "{}{}",
                crate::proxy::child_base(&engine.endpoint),
                self.fwd_path
            );
            crate::proxy::forward_for_replay(
                &self.state,
                &engine,
                &self.method,
                &url,
                &self.headers,
                self.body.clone(),
            )
            .await
        })
    }

    /// Queue again like every other request: the public admission
    /// gate at this request's own priority and class (never a preemptor,
    /// it already cost one cancel), then the lane. The lease it held
    /// went to the request that cancelled it, so the replay waits behind
    /// it like any other queued request.
    async fn readmit(&self) -> Result<(blazar_runtime::EngineRef, InFlightGuard), String> {
        let lease = crate::proxy::admission_gate_replay(
            &self.state,
            &self.model,
            self.urgency.priority(),
            self.urgency.class(),
        )
        .await?;
        let engine = crate::proxy::ensure_replay_admission(
            &self.state,
            &self.model,
            self.urgency.priority(),
            self.urgency.class(),
        )
        .await?;
        Ok((engine, lease))
    }
}

fn refused(reason: &str) -> reqwest::Response {
    let body = serde_json::json!({
        "error": {
            "code": "preempt_replay_refused",
            "message": reason,
        }
    });
    let response = axum::http::Response::builder()
        .status(axum::http::StatusCode::SERVICE_UNAVAILABLE)
        .header("content-type", "application/json")
        .body(axum::body::Bytes::from(body.to_string()))
        .expect("static response parts");
    reqwest::Response::from(response)
}

/// Send until the child's response headers exist, while the request can
/// still be cancelled: before the headers exist nothing has reached the
/// client, so a cancelled attempt costs nothing but time.
pub(crate) async fn send_preemptable(
    plan: &ReplayPlan,
    slot: &GuardSlot,
    engine: &blazar_runtime::EngineRef,
    mut ticket: Ticket,
) -> (Result<reqwest::Response, ChildSendError>, Ticket) {
    enum Step {
        Sent(Result<reqwest::Response, ChildSendError>),
        Cancelled,
    }
    let mut attempt = plan.send(engine.clone());
    loop {
        let step = tokio::select! {
            result = attempt => Step::Sent(result),
            () = ticket.cancel_changed() => Step::Cancelled,
        };
        match step {
            Step::Sent(result) => return (result, ticket),
            Step::Cancelled => {}
        }
        // `select!` dropped the send future before returning, which
        // cancels the child request. The freed slot goes to the request
        // that cancelled this one; this request queues again.
        ticket.hand_over(slot.take_lease());
        match plan.readmit().await {
            Ok((engine, lease)) => {
                slot.hold(Some(lease));
                plan.state.preempt.count_outcome(1);
                tracing::info!(
                    target: "blazar::preempt",
                    model = plan.model.as_str(),
                    phase = "headers",
                    "preempted before its first response byte - replayed in place",
                );
                attempt = plan.send(engine);
            }
            Err(reason) => {
                plan.state.preempt.count_outcome(2);
                tracing::warn!(
                    target: "blazar::preempt",
                    model = plan.model.as_str(),
                    "preempted request could not be queued again: {reason}",
                );
                return (Ok(refused(&reason)), ticket);
            }
        }
    }
}

/// The in-flight replay: re-admit, re-send, hand back the new body stream.
type ReplayFuture = Pin<Box<dyn Future<Output = Result<ByteStream, std::io::Error>> + Send>>;

/// Response-body phase of a preemptable request. Bytes pass straight
/// through; while none has passed yet a cancel swaps in the replay.
pub struct Preemptable {
    upstream: ByteStream,
    replaying: Option<ReplayFuture>,
    plan: Arc<ReplayPlan>,
    slot: Arc<Mutex<Option<InFlightGuard>>>,
    ticket: Ticket,
    failed: bool,
}

impl Preemptable {
    #[must_use]
    pub fn new(
        plan: Arc<ReplayPlan>,
        ticket: Ticket,
        slot: Arc<Mutex<Option<InFlightGuard>>>,
        upstream: ByteStream,
    ) -> Self {
        Self {
            upstream,
            replaying: None,
            plan,
            slot,
            ticket,
            failed: false,
        }
    }

    fn start_replay(&mut self) {
        // Dropping the child stream cancels the engine task. The freed
        // slot goes to the request that cancelled this one: the lease is
        // handed over, not released. This request queues again.
        self.upstream = Box::pin(futures::stream::empty());
        self.ticket
            .hand_over(GuardSlot::Shared(Arc::clone(&self.slot)).take_lease());
        let plan = Arc::clone(&self.plan);
        let slot = Arc::clone(&self.slot);
        self.replaying = Some(Box::pin(async move {
            let (engine, lease) = plan.readmit().await.map_err(std::io::Error::other)?;
            GuardSlot::Shared(slot).hold(Some(lease));
            let response = plan
                .send(engine)
                .await
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            if !response.status().is_success() {
                return Err(std::io::Error::other(format!(
                    "preempted request was replayed and the engine answered HTTP {}",
                    response.status().as_u16()
                )));
            }
            plan.state.preempt.count_outcome(1);
            tracing::info!(
                target: "blazar::preempt",
                model = plan.model.as_str(),
                phase = "body",
                "preempted before its first response byte - replayed in place",
            );
            Ok(Box::pin(
                response
                    .bytes_stream()
                    .map(|r| r.map_err(std::io::Error::other)),
            ) as ByteStream)
        }));
    }
}

impl Stream for Preemptable {
    type Item = std::io::Result<Bytes>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.failed {
            return Poll::Ready(None);
        }
        loop {
            if let Some(replay) = self.replaying.as_mut() {
                match replay.as_mut().poll(cx) {
                    Poll::Ready(Ok(upstream)) => {
                        self.upstream = upstream;
                        self.replaying = None;
                    }
                    Poll::Ready(Err(e)) => {
                        self.replaying = None;
                        self.failed = true;
                        self.plan.state.preempt.count_outcome(2);
                        return Poll::Ready(Some(Err(e)));
                    }
                    Poll::Pending => return Poll::Pending,
                }
                continue;
            }
            match self.upstream.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    self.ticket
                        .forwarded
                        .fetch_add(bytes.len() as u64, Ordering::Relaxed);
                    return Poll::Ready(Some(Ok(bytes)));
                }
                Poll::Ready(rest) => return Poll::Ready(rest),
                Poll::Pending => {}
            }
            // The only value ever sent on this channel is the cancel.
            let cancelled = {
                let mut changed = std::pin::pin!(self.ticket.cancel_changed());
                changed.as_mut().poll(cx).is_ready()
            };
            if cancelled {
                if self.ticket.still_virgin() {
                    self.start_replay();
                } else {
                    self.ticket.too_late();
                }
                continue;
            }
            return Poll::Pending;
        }
    }
}

/// The waiting request's side of a preemption attempt. Dropping it
/// stops the attempt; [`Preemptor::wait_grant`] yields the lease a
/// cancelled request handed over.
pub struct Preemptor {
    lease: Arc<Mutex<Option<InFlightGuard>>>,
    ready: watch::Receiver<bool>,
    task: tokio::task::JoinHandle<()>,
}

impl Preemptor {
    /// The lease of the request cancelled for this one, or `None` when
    /// the attempt ended without one: nothing eligible, budget spent, or
    /// the victim had already started answering.
    pub async fn wait_grant(self: Arc<Self>) -> Option<InFlightGuard> {
        let mut ready = self.ready.clone();
        loop {
            if ready.changed().await.is_err() {
                return None;
            }
            let lease = self
                .lease
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take();
            if lease.is_some() {
                return lease;
            }
        }
    }
}

impl Drop for Preemptor {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// Arm the preemptor of a request about to wait in the admission queue.
/// It fires once, `preempt_after_ms` later; the caller stops it by
/// dropping the returned handle when the wait ends any other way.
pub fn arm(
    state: &Arc<AppState>,
    model: &str,
    waiter: Urgency,
    allowed: bool,
) -> Option<Arc<Preemptor>> {
    let after = Duration::from_millis(state.cfg().preempt_after_ms);
    if !allowed || !enabled(&state.cfg().preemption) || after.is_zero() {
        return None;
    }
    let state = Arc::clone(state);
    let model = model.to_string();
    let lease = Arc::new(Mutex::new(None));
    let (ready, granted) = watch::channel(false);
    let task = tokio::spawn({
        let lease = Arc::clone(&lease);
        async move {
            tokio::time::sleep(after).await;
            // The waiter itself is in this count: zero means it was
            // admitted meanwhile and there is nobody left to help.
            if state.queue.depth() == 0 {
                return;
            }
            if let Outcome::Cancelled(handover) =
                state.preempt.try_preempt(&model, waiter, Instant::now())
            {
                tracing::info!(
                    target: "blazar::preempt",
                    model = model.as_str(),
                    waited_ms = u64::try_from(after.as_millis()).unwrap_or(u64::MAX),
                    waiter_priority = ?waiter.priority(),
                    "waiting request cancelled a less urgent running one to take its slot",
                );
                // The victim delivers its lease once its child request
                // is dropped; a channel that closes means it never does.
                if let Ok(guard) = handover.await {
                    *lease.lock().unwrap_or_else(PoisonError::into_inner) = Some(guard);
                    let _ = ready.send(true);
                }
            }
        }
    });
    Some(Arc::new(Preemptor {
        lease,
        ready: granted,
        task,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake {
        registry: Arc<Registry>,
    }

    fn fake() -> Fake {
        Fake {
            registry: Arc::new(Registry::default()),
        }
    }

    /// Register, then age the entry past the minimum by moving its start
    /// time back: the selection logic is under test, not the clock.
    fn aged(fake: &Fake, model: &str, urgency: Urgency, age: Duration) -> Ticket {
        let ticket = fake.registry.register(model, urgency);
        lock(&fake.registry.inner)
            .victims
            .get_mut(&ticket.id)
            .expect("registered")
            .started = Instant::now()
            .checked_sub(age)
            .expect("age within the clock");
        ticket
    }

    const NORMAL: Urgency = Urgency(1, 1);
    const HIGH: Urgency = Urgency(2, 1);
    const TOOL: Urgency = Urgency(1, 2);

    #[test]
    #[allow(non_snake_case)]
    fn unit__urgency__orders_like_the_queue_priority_first_then_class() {
        assert_eq!(Urgency::new(Priority::Low, WorkClass::Tool), Urgency(0, 2));
        assert_eq!(
            Urgency::new(Priority::High, WorkClass::RawLong),
            Urgency(2, 0)
        );
        assert!(NORMAL < HIGH);
        assert!(NORMAL < TOOL);
        assert_eq!(Urgency(2, 0).priority(), Priority::High);
        assert_eq!(Urgency(2, 0).class(), WorkClass::RawLong);
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__choose__only_a_less_urgent_request_on_the_same_model_yields() {
        let fake = fake();
        let foreign = aged(&fake, "other", Urgency(0, 0), MIN_VICTIM_AGE);
        let equal = aged(&fake, "m", NORMAL, MIN_VICTIM_AGE);
        let stronger = aged(&fake, "m", HIGH, MIN_VICTIM_AGE);
        let weaker = aged(&fake, "m", Urgency(0, 0), MIN_VICTIM_AGE);
        let now = Instant::now();
        let pick = |waiter: Urgency| {
            choose(
                lock(&fake.registry.inner)
                    .victims
                    .iter()
                    .map(|(id, v)| (*id, v)),
                "m",
                waiter,
                now,
            )
        };
        assert_eq!(
            pick(NORMAL),
            Some(weaker.id),
            "equals and stronger requests keep their slot"
        );
        assert_eq!(
            pick(Urgency(0, 0)),
            None,
            "nothing is less urgent than the lowest class"
        );
        drop((foreign, equal, stronger, weaker));
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__choose__skips_the_young_the_streamed_and_the_already_replayed() {
        let fake = fake();
        let young = aged(
            &fake,
            "m",
            Urgency(0, 0),
            MIN_VICTIM_AGE
                .checked_sub(Duration::from_millis(1))
                .expect("one millisecond under the minimum"),
        );
        let streamed = aged(&fake, "m", Urgency(0, 0), MIN_VICTIM_AGE);
        streamed.forwarded.store(1, Ordering::Relaxed);
        let replayed = aged(&fake, "m", Urgency(0, 0), MIN_VICTIM_AGE);
        lock(&fake.registry.inner).victims[&replayed.id]
            .replays
            .store(1, Ordering::Relaxed);
        let old = aged(&fake, "m", Urgency(0, 0), Duration::from_secs(30));
        let fresh = aged(&fake, "m", Urgency(0, 0), Duration::from_secs(5));
        let now = Instant::now();
        let pick = |waiter: Urgency| {
            choose(
                lock(&fake.registry.inner)
                    .victims
                    .iter()
                    .map(|(id, v)| (*id, v)),
                "m",
                waiter,
                now,
            )
        };
        assert_eq!(
            pick(NORMAL),
            Some(fresh.id),
            "the youngest eligible request loses the least work"
        );
        drop((young, streamed, replayed, old, fresh));
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__try_preempt__signals_the_victim_once_and_respects_the_budget() {
        let fake = fake();
        let waiter = Urgency(2, 2);
        for _ in 0..BUDGET_PER_MINUTE {
            let victim = aged(&fake, "m", Urgency(0, 0), MIN_VICTIM_AGE);
            let Outcome::Cancelled(mut handover) =
                fake.registry.try_preempt("m", waiter, Instant::now())
            else {
                panic!("an aged, less urgent victim is cancelled");
            };
            assert!(victim.cancelled.has_changed().expect("a value was sent"));
            // The victim has not dropped its child request yet: the
            // lease is still its own and nothing counts as cancelled.
            assert!(matches!(
                handover.try_recv(),
                Err(tokio::sync::oneshot::error::TryRecvError::Empty)
            ));
            drop(victim);
        }
        assert!(matches!(
            fake.registry.try_preempt("m", waiter, Instant::now()),
            Outcome::BudgetExhausted
        ));
        assert!(matches!(
            fake.registry.try_preempt("nobody", waiter, Instant::now()),
            Outcome::NoCandidate
        ));
        let metrics = fake.registry.metrics();
        assert!(metrics.contains("blazar_preemptions_total{outcome=\"cancelled\"} 0"));
        assert!(metrics.contains("blazar_preemption_refusals_total{reason=\"budget\"} 1"));
        assert!(metrics.contains("blazar_preemption_refusals_total{reason=\"no_candidate\"} 1"));
    }

    #[test]
    #[allow(non_snake_case)]
    fn unit__ticket__leaves_the_registry_when_the_response_ends() {
        let fake = fake();
        let victim = aged(&fake, "m", Urgency(0, 0), MIN_VICTIM_AGE);
        let id = victim.id;
        assert!(lock(&fake.registry.inner).victims.contains_key(&id));
        drop(victim);
        assert!(!lock(&fake.registry.inner).victims.contains_key(&id));
    }
}
