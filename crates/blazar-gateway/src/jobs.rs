//! Durable background jobs: one ledger under every async lane.
//!
//! Blazar has two job ownership models, and this runtime owns neither:
//! - audio jobs are gateway-owned tokio tasks (`whisper::AudioJobs`),
//! - image/video jobs are child-owned sd-server state the gateway polls.
//!
//! What dies with a process is the LIVE handle, not the record. `JobRuntime`
//! is the SQLite write-through ledger behind both lanes plus the unified
//! read/cancel/events plane at `/v1/jobs`: a completed job survives a
//! gateway restart with its result, an in-flight one is swept to an honest
//! `abandoned` state at boot (renders cannot resume mid-frame — the durable
//! contract is records + artifacts + idempotent resubmit, never silent
//! continuation).
//!
//! Laws carried over from the lanes: no await under a lock (all store access
//! is a short synchronous `with_store` closure), and a store write failure
//! logs loudly but never masks the lane's own answer — the live registry
//! stays authoritative while it lives.

use crate::state::AppState;
use axum::extract::{Path, Query, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use blazar_core::store::JobRow;
use std::sync::Arc;

/// Input artifacts larger than this are skipped (the job row still lands;
/// resubmit needs the original file anyway, and unbounded disk use from
/// repeated audio uploads would be a quiet footgun).
const INPUT_ARTIFACT_CAP_BYTES: usize = 64 * 1024 * 1024;
/// Results up to this size ride inline in the job row (transcription
/// bodies, single images); larger ones become artifact files.
const INLINE_RESULT_CAP_BYTES: usize = 1024 * 1024;
/// Terminal jobs older than this are pruned at boot. Completed artifacts
/// on disk are kept — only the ledger rows go.
const TERMINAL_PRUNE_SECS: i64 = 7 * 24 * 3600;
/// A row must be this stale before the boot sweep will abandon it. Two
/// daemons sharing one data dir is unsupported (single port, single WAL
/// writer); the window only guards the stop→start handoff race where the
/// old process wrote seconds before the new one booted.
const BOOT_GRACE_SECS: i64 = 5;

pub struct JobRuntime {
    /// Artifact root (`<data>/jobs/<id>/…`), owned so recordings never
    /// borrow from `AppState`.
    jobs_dir: std::path::PathBuf,
}

#[derive(serde::Deserialize)]
pub struct ListQuery {
    state: Option<String>,
    kind: Option<String>,
    limit: Option<u64>,
}

impl JobRuntime {
    #[must_use]
    pub fn new(data_dir: &std::path::Path) -> Self {
        Self {
            jobs_dir: data_dir.join("jobs"),
        }
    }

    /// Boot hygiene: abandon stale in-flight rows, prune ancient terminal
    /// ones. Runs detached from `serve()` — sweep latency is not boot
    /// latency, and the store may still be settling on first open.
    pub fn boot_sweep(&self, state: &Arc<AppState>) {
        let swept = state
            .with_store(|s| {
                let abandoned = s.boot_sweep_jobs(BOOT_GRACE_SECS).unwrap_or_default();
                let pruned = s.prune_jobs(TERMINAL_PRUNE_SECS).unwrap_or(0);
                (abandoned, pruned)
            })
            .unwrap_or((Vec::new(), 0));
        if !swept.0.is_empty() {
            tracing::info!(
                target: "blazar::jobs",
                abandoned = swept.0.len(),
                "boot sweep: in-flight jobs from a previous gateway marked abandoned (resubmit to rerun)"
            );
        }
        if swept.1 > 0 {
            tracing::info!(
                target: "blazar::jobs",
                pruned = swept.1,
                "boot sweep: pruned terminal job rows older than 7 days"
            );
        }
    }

    /// Insert the job row at submit time. `request_json` must carry
    /// everything a cancel needs to redispatch (engine name for
    /// child-owned lanes) and everything a resubmit needs to teach.
    pub fn record_created(
        &self,
        state: &AppState,
        id: &str,
        kind: &str,
        model: Option<&str>,
        request_json: serde_json::Value,
    ) {
        let now = blazar_core::store::unix_now();
        let row = JobRow {
            id: id.to_string(),
            kind: kind.to_string(),
            model: model.map(str::to_string),
            state: "queued".into(),
            request_json: serde_json::to_string(&request_json).unwrap_or_else(|_| "{}".into()),
            result_json: None,
            error: None,
            artifact_path: None,
            created_at: now,
            updated_at: now,
        };
        if let Some(err) = state.with_store(|s| s.insert_job(&row).err()).flatten() {
            tracing::warn!(target: "blazar::jobs", job = %id, %err, "job ledger insert failed — live lane continues, record not durable");
        }
    }

    /// Persist an optional input artifact next to the row (audio uploads;
    /// image prompts are JSON and already live in `request_json`).
    pub fn record_input_artifact(&self, id: &str, filename: Option<&str>, data: &[u8]) {
        if data.is_empty() || data.len() > INPUT_ARTIFACT_CAP_BYTES {
            return;
        }
        // Extension-only sanitization: the name never round-trips into a
        // shell or a route, but a crafted filename must not escape the
        // job dir either (`../../x` has no safe extension → none kept).
        let ext = filename
            .and_then(|f| f.rsplit_once('.'))
            .map(|(_, e)| e.to_ascii_lowercase())
            .filter(|e| !e.is_empty() && e.chars().all(|c| c.is_ascii_alphanumeric()))
            .map_or(String::new(), |e| format!(".{e}"));
        let dir = self.jobs_dir.join(sanitize_id(id));
        let path = dir.join(format!("input{ext}"));
        if let Err(err) = std::fs::create_dir_all(&dir).and_then(|_| std::fs::write(&path, data)) {
            tracing::warn!(target: "blazar::jobs", job = %id, %err, "input artifact write failed — job continues without it");
        }
    }

    pub fn record_running(&self, state: &AppState, id: &str) {
        self.transition(state, id, "running", None, None, "running");
    }

    /// Close a job with its result. Bodies ≤1 MiB ride inline (parsed as
    /// JSON when possible, lossy string otherwise — the same vocabulary
    /// `AudioJobs::payload` serves); larger ones become artifact files
    /// with a stub envelope so the row stays small.
    pub fn record_completed(&self, state: &AppState, id: &str, body: &[u8], content_type: &str) {
        let inline = serde_json::from_slice::<serde_json::Value>(body).unwrap_or_else(|_| {
            serde_json::Value::String(String::from_utf8_lossy(body).into_owned())
        });
        let serialized = serde_json::to_string(&inline).unwrap_or_default();
        let (result_json, artifact_path) = if serialized.len() <= INLINE_RESULT_CAP_BYTES {
            (serialized, None)
        } else {
            let dir = self.jobs_dir.join(sanitize_id(id));
            let path = dir.join("result");
            match std::fs::create_dir_all(&dir).and_then(|_| std::fs::write(&path, body)) {
                Ok(()) => (
                    serde_json::to_string(&serde_json::json!({
                        "artifact": true,
                        "content_type": content_type,
                        "bytes": body.len(),
                    }))
                    .unwrap_or_default(),
                    Some(path.to_string_lossy().into_owned()),
                ),
                Err(err) => {
                    tracing::warn!(target: "blazar::jobs", job = %id, %err, "result artifact write failed — recording stub only");
                    (
                        serde_json::to_string(&serde_json::json!({
                            "artifact": "dropped",
                            "content_type": content_type,
                            "bytes": body.len(),
                        }))
                        .unwrap_or_default(),
                        None,
                    )
                }
            }
        };
        self.transition(
            state,
            id,
            "completed",
            Some(&result_json),
            None,
            "completed",
        );
        // Spillover lands in the dedicated column: `set_job_state` only
        // touches state/result/error.
        if let Some(path) = artifact_path.as_deref() {
            if let Some(err) = state
                .with_store(|s| s.set_job_artifact(id, path).err())
                .flatten()
            {
                tracing::warn!(target: "blazar::jobs", job = %id, %err, "artifact path recording failed");
            }
        }
    }

    pub fn record_failed(&self, state: &AppState, id: &str, message: &str) {
        self.transition(state, id, "failed", None, Some(message), "failed");
    }

    pub fn record_cancelled(&self, state: &AppState, id: &str) {
        self.transition(state, id, "cancelled", None, None, "cancelled");
    }

    /// Append a progress event without a state transition — doctor probe
    /// verdicts today, lane milestones later. The row must already exist;
    /// a missing row is a debug-level note (the lane stays authoritative).
    pub fn record_event(&self, state: &AppState, id: &str, kind: &str, data: serde_json::Value) {
        let data = serde_json::to_string(&data).ok();
        let outcome = state.with_store(|s| s.append_job_event(id, kind, data.as_deref()));
        match outcome {
            Some(Ok(())) => {}
            Some(Err(err)) => {
                tracing::warn!(target: "blazar::jobs", job = %id, %err, "job event append failed");
            }
            None => {
                tracing::warn!(target: "blazar::jobs", job = %id, "job ledger unavailable; event not recorded");
            }
        }
    }

    fn transition(
        &self,
        state: &AppState,
        id: &str,
        to: &str,
        result_json: Option<&str>,
        error: Option<&str>,
        event_kind: &str,
    ) {
        let outcome = state.with_store(|s| {
            let flipped = s.set_job_state(id, to, result_json, error).unwrap_or(false);
            if flipped {
                if let Err(err) = s.append_job_event(id, event_kind, None) {
                    tracing::warn!(target: "blazar::jobs", job = %id, %err, "job event append failed");
                }
            }
            flipped
        });
        match outcome {
            Some(true) => {}
            // Unknown row (lane created it before this release, or the
            // insert failed loudly earlier) — the live answer still wins.
            Some(false) => {
                tracing::debug!(target: "blazar::jobs", job = %id, to, "job row absent; transition not recorded")
            }
            None => {
                tracing::warn!(target: "blazar::jobs", job = %id, to, "job ledger unavailable; transition not recorded")
            }
        }
    }
}

/// Job ids are validated `[A-Za-z0-9_-]` at every route; defense in depth
/// for the artifact path join.
fn sanitize_id(id: &str) -> String {
    id.chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect()
}

/// Unified payload from a row — the `/v1/jobs` wire shape. Superset of the
/// lane-native shapes: `id`, `status`, `created_at` keep their existing
/// vocabulary; `kind`, `updated_at`, `result`, `error`, `artifact` are the
/// durable extras.
pub fn row_payload(row: &JobRow) -> serde_json::Value {
    let mut payload = serde_json::json!({
        "id": row.id,
        "object": "blazar.job",
        "kind": row.kind,
        "model": row.model,
        "status": row.state,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    });
    if let Some(result) = row
        .result_json
        .as_deref()
        .and_then(|v| serde_json::from_str::<serde_json::Value>(v).ok())
    {
        payload["result"] = result;
    }
    if let Some(error) = &row.error {
        payload["error"] = serde_json::json!(error);
    }
    if let Some(path) = &row.artifact_path {
        payload["artifact"] = serde_json::json!({
            "path": path,
            "url": format!("/v1/jobs/{}/artifact", row.id),
        });
    }
    payload
}

fn not_found(id: &str) -> Response {
    crate::proxy::openai_error(
        404,
        &format!(
            "job {id} not found — live handles die with their owner (gateway or engine child) \
             and completed records live 7 days; submit a new request"
        ),
    )
}

/// GET /v1/jobs?state=&kind=&limit= — the durable ledger, newest first.
#[allow(clippy::unused_async)] // axum's Handler trait requires async fns
pub async fn jobs_list(State(state): State<Arc<AppState>>, Query(q): Query<ListQuery>) -> Response {
    let rows = state
        .with_store(|s| {
            s.list_jobs(q.state.as_deref(), q.kind.as_deref(), q.limit.unwrap_or(50))
                .unwrap_or_default()
        })
        .unwrap_or_default();
    let data: Vec<serde_json::Value> = rows.iter().map(row_payload).collect();
    axum::Json(serde_json::json!({
        "object": "blazar.job.list",
        "count": data.len(),
        "data": data,
    }))
    .into_response()
}

/// GET /v1/jobs/{id} — terminal rows answer from the ledger; live ones
/// dispatch to their lane for fresh state (audio registry, live child
/// poll for image/video), mirroring whatever terminal state comes back.
pub async fn jobs_get(State(state): State<Arc<AppState>>, Path(job_id): Path<String>) -> Response {
    if !job_id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return crate::proxy::openai_error(400, "invalid job id");
    }
    let Some(row) = state
        .with_store(|s| s.get_job(&job_id).ok().flatten())
        .flatten()
    else {
        return not_found(&job_id);
    };
    if !matches!(row.state.as_str(), "queued" | "running") {
        return axum::Json(row_payload(&row)).into_response();
    }
    match row.kind.as_str() {
        // Audio: the in-memory registry is fresher while the gateway that
        // spawned the task lives; the row is the after-restart truth.
        "audio" => {
            if let Some(live) = state.audio_jobs.payload(&job_id) {
                return axum::Json(live).into_response();
            }
            axum::Json(row_payload(&row)).into_response()
        }
        // Doctor: a gateway-owned probe task writes its progress through
        // this ledger itself — the row IS the live truth (events carry
        // per-probe verdicts as they land).
        "doctor" => axum::Json(row_payload(&row)).into_response(),
        // Image/video: the owning child is the only live truth. A poll
        // that finds a terminal state mirrors it into the ledger; a child
        // that is gone closes the row honestly instead of 404-ing.
        "image" | "video" => match crate::images::poll_live_child(&state, None, &job_id).await {
            Some(child_job) => {
                mirror_child_terminal(&state, &job_id, &child_job);
                let mut payload = child_job;
                if let Some(obj) = payload.as_object_mut() {
                    obj.insert("kind".into(), serde_json::json!(row.kind));
                }
                axum::Json(payload).into_response()
            }
            None => {
                let msg = "job's engine child is gone (eviction, crash or restart) — \
                           resubmit the generation";
                state.jobs.record_failed(&state, &job_id, msg);
                let closed = state
                    .with_store(|s| s.get_job(&job_id).ok().flatten())
                    .flatten();
                match closed {
                    Some(row) => axum::Json(row_payload(&row)).into_response(),
                    None => not_found(&job_id),
                }
            }
        },
        other => crate::proxy::openai_error(
            500,
            &format!("job {job_id} has unknown kind {other:?} — ledger corruption?"),
        ),
    }
}

/// Write a child-reported terminal state into the ledger (the child is
/// the authority; the row catches up on first observation).
pub(crate) fn mirror_child_terminal(state: &AppState, id: &str, child_job: &serde_json::Value) {
    let Some(status) = child_job.get("status").and_then(serde_json::Value::as_str) else {
        return;
    };
    match status {
        "completed" => {
            let body = serde_json::to_vec(child_job).unwrap_or_default();
            state
                .jobs
                .record_completed(state, id, &body, "application/json");
        }
        "failed" => {
            let msg = child_job
                .pointer("/error/message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("generation failed");
            state.jobs.record_failed(state, id, msg);
        }
        "cancelled" => state.jobs.record_cancelled(state, id),
        _ => {}
    }
}

/// POST /v1/jobs/{id}/cancel — idempotent; dispatches by kind: audio
/// aborts the task, image/video best-effort cancels on the live child.
#[allow(clippy::unused_async)] // axum's Handler trait requires async fns
pub async fn jobs_cancel(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> Response {
    if !job_id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return crate::proxy::openai_error(400, "invalid job id");
    }
    let Some(row) = state
        .with_store(|s| s.get_job(&job_id).ok().flatten())
        .flatten()
    else {
        return not_found(&job_id);
    };
    if !matches!(row.state.as_str(), "queued" | "running") {
        return axum::Json(row_payload(&row)).into_response();
    }
    match row.kind.as_str() {
        "audio" => {
            let _ = state.audio_jobs.cancel(&job_id);
            state.jobs.record_cancelled(&state, &job_id);
        }
        // Doctor probes check the row between probes and stop once it is
        // terminal; the one-way close guarantees this flip is final.
        "doctor" => {
            state.jobs.record_cancelled(&state, &job_id);
        }
        "image" | "video" => {
            let engine = row
                .request_json
                .parse::<serde_json::Value>()
                .ok()
                .and_then(|r| {
                    r.get("engine")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_string)
                });
            crate::images::cancel_child_best_effort(&state, engine.as_deref(), &job_id).await;
            state.jobs.record_cancelled(&state, &job_id);
        }
        other => {
            return crate::proxy::openai_error(
                500,
                &format!("job {job_id} has unknown kind {other:?} — ledger corruption?"),
            )
        }
    }
    let closed = state
        .with_store(|s| s.get_job(&job_id).ok().flatten())
        .flatten();
    match closed {
        Some(row) => axum::Json(row_payload(&row)).into_response(),
        None => not_found(&job_id),
    }
}

/// GET /v1/jobs/{id}/events — the append-only story (created → running →
/// terminal, boot-sweep and artifact notes included).
#[allow(clippy::unused_async)] // axum's Handler trait requires async fns
pub async fn jobs_events(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> Response {
    if !job_id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return crate::proxy::openai_error(400, "invalid job id");
    }
    let events = state
        .with_store(|s| s.job_events(&job_id, 1000).ok().unwrap_or_default())
        .unwrap_or_default();
    if events.is_empty() {
        // Absent events and absent row are the same story to a client.
        let known = state
            .with_store(|s| s.get_job(&job_id).ok().flatten())
            .flatten();
        if known.is_none() {
            return not_found(&job_id);
        }
    }
    let data: Vec<serde_json::Value> = events
        .iter()
        .map(|e| {
            let mut v = serde_json::json!({
                "seq": e.seq,
                "ts": e.ts,
                "kind": e.kind,
            });
            if let Some(d) = e
                .data_json
                .as_deref()
                .and_then(|v| serde_json::from_str::<serde_json::Value>(v).ok())
            {
                v["data"] = d;
            }
            v
        })
        .collect();
    axum::Json(serde_json::json!({
        "object": "blazar.job.events",
        "job_id": job_id,
        "data": data,
    }))
    .into_response()
}

/// GET /v1/jobs/{id}/artifact — serve a spilled result artifact with its
/// recorded content type. Path comes from the row, never the request.
#[allow(clippy::unused_async)] // axum's Handler trait requires async fns
pub async fn jobs_artifact(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> Response {
    if !job_id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return crate::proxy::openai_error(400, "invalid job id");
    }
    let Some(row) = state
        .with_store(|s| s.get_job(&job_id).ok().flatten())
        .flatten()
    else {
        return not_found(&job_id);
    };
    let Some(path) = row.artifact_path else {
        return crate::proxy::openai_error(
            404,
            &format!("job {job_id} has no result artifact (inline result or none)"),
        );
    };
    let content_type = row
        .result_json
        .as_deref()
        .and_then(|v| serde_json::from_str::<serde_json::Value>(v).ok())
        .and_then(|r| {
            r.get("content_type")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| "application/octet-stream".into());
    match std::fs::read(&path) {
        Ok(bytes) => Response::builder()
            .status(200)
            .header(header::CONTENT_TYPE, content_type)
            .body(axum::body::Body::from(bytes))
            .unwrap_or_else(|e| {
                crate::proxy::openai_error(500, &format!("response build: {e}")).into_response()
            }),
        Err(err) => crate::proxy::openai_error(
            410,
            &format!("job {job_id} artifact is no longer on disk ({err}) — pruned or cleaned"),
        ),
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    fn row(id: &str, kind: &str, state: &str) -> JobRow {
        JobRow {
            id: id.into(),
            kind: kind.into(),
            model: Some("m".into()),
            state: state.into(),
            request_json: "{}".into(),
            result_json: None,
            error: None,
            artifact_path: None,
            created_at: 100,
            updated_at: 200,
        }
    }

    #[test]
    fn unit__row_payload__shape_and_stubs() {
        let p = row_payload(&row("j1", "audio", "running"));
        assert_eq!(p["object"], serde_json::json!("blazar.job"));
        assert_eq!(p["kind"], serde_json::json!("audio"));
        assert_eq!(p["status"], serde_json::json!("running"));
        assert!(p.get("result").is_none());
        assert!(p.get("error").is_none());
        assert!(p.get("artifact").is_none());
    }

    #[test]
    fn unit__row_payload__result_error_artifact_surface() {
        let mut r = row("j2", "image", "completed");
        r.result_json = Some("{\"done\": true}".into());
        r.error = None;
        r.artifact_path = Some("/tmp/x".into());
        let p = row_payload(&r);
        assert_eq!(p["result"]["done"], serde_json::json!(true));
        assert_eq!(
            p["artifact"]["url"],
            serde_json::json!("/v1/jobs/j2/artifact")
        );
    }

    #[test]
    fn unit__sanitize_id__strips_path_characters() {
        assert_eq!(sanitize_id("aj1-2_3"), "aj1-2_3");
        assert_eq!(sanitize_id("../../etc/passwd"), "etcpasswd");
    }

    #[test]
    fn unit__input_artifact_cap__oversize_skipped_under_written() {
        let tmp = tempfile::tempdir().expect("tmp");
        let rt = JobRuntime::new(tmp.path());
        rt.record_input_artifact("safe_1", Some("talk.wav"), b"tiny");
        let written = tmp.path().join("jobs").join("safe_1").join("input.wav");
        assert!(written.is_file());
        rt.record_input_artifact(
            "safe_2",
            Some("big.bin"),
            &vec![0u8; INPUT_ARTIFACT_CAP_BYTES + 1],
        );
        assert!(!tmp.path().join("jobs").join("safe_2").exists());
    }
}
