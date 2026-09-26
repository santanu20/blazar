//! `/v1/audio/transcriptions` + `/v1/audio/translations`: local
//! whisper.cpp lane (H8).
//!
//! Decision order:
//! (1) explicit remote intent (`name:model`) always forwards — the user
//!     named a remote;
//! (2) local lane when whisper-server + at least one ggml model are
//!     installed (lazy child, hot model swap);
//! (3) otherwise a teaching error (llama.cpp children do not transcribe).
//!
//! Multipart is parsed binary-safe: parts split ONLY on the exact random
//! boundary — audio bytes cannot forge one.
//!
//! The local lane REBUILDS the multipart from a verified field
//! whitelist (upstream has no OpenAI-compatible audio route — `/inference`
//! is the whole surface), so transcription/translation ride the same
//! lazy child. `/v1/audio/translations` forces `translate=true` on the
//! rebuilt request; upstream reports `"task": "translate"` back.
//!
//! Async: upstream `/inference` is sync-only (no job surface to relay,
//! unlike the sd-server image lane), so `"async": "true"` as a form
//! field runs the same forward inside a spawned task and hands back a
//! gateway-owned job handle — `/v1/audio/jobs/{id}` polls it,
//! `/v1/audio/jobs/{id}/cancel` aborts the wait. Jobs die with the
//! gateway process, the same lifetime truth the image lane attaches to
//! its children.

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::Extension;
use blazar_runtime::whisper;

use crate::proxy::openai_error;
use crate::remotes::split_remote;
use crate::state::AppState;

/// One parsed multipart part (cloned into async job tasks, which
/// outlive the request that received the bytes).
#[derive(Clone)]
pub(crate) struct Part {
    pub(crate) name: String,
    pub(crate) filename: Option<String>,
    pub(crate) content_type: Option<String>,
    pub(crate) data: Vec<u8>,
}

/// Boundary-aware multipart split (RFC 2046 essentials: `--boundary`
/// delimiters, CRLF-separated part headers, `--boundary--` terminator).
/// Returns None on structural garbage (caller 400s).
pub(crate) fn parse_multipart(body: &[u8], content_type: &str) -> Option<Vec<Part>> {
    let boundary = content_type
        .split(';')
        .map(str::trim)
        .find_map(|p| p.strip_prefix("boundary="))?
        .trim_matches('"');
    let delim = format!("--{boundary}");
    let mut parts = Vec::new();
    let mut pos = 0usize;
    while let Some(rel) = find_sub(&body[pos..], delim.as_bytes()) {
        let mut seg_start = pos + rel + delim.len();
        // Terminator?
        if body[seg_start..].starts_with(b"--") {
            break;
        }
        // Each delimiter is followed by CRLF before part headers.
        if body[seg_start..].starts_with(b"\r\n") {
            seg_start += 2;
        } else if body[seg_start..].starts_with(b"\n") {
            seg_start += 1;
        }
        let Some(hdr_end_rel) = find_sub(&body[seg_start..], b"\r\n\r\n")
            .or_else(|| find_sub(&body[seg_start..], b"\n\n"))
        else {
            break;
        };
        let sep_len = if body[seg_start + hdr_end_rel..].starts_with(b"\r\n\r\n") {
            4
        } else {
            2
        };
        let headers_raw = &body[seg_start..seg_start + hdr_end_rel];
        let data_start = seg_start + hdr_end_rel + sep_len;
        // Data runs to the NEXT delimiter that starts a line (preceded
        // by CRLF/LF) — binary audio containing the boundary bytes
        // mid-line is data, not a separator.
        let Some(next) = find_line_delim(body, data_start, delim.as_bytes()) else {
            break;
        };
        let mut data_end = next;
        if data_end >= 2 && &body[data_end - 2..data_end] == b"\r\n" {
            data_end -= 2;
        } else if data_end >= 1 && body[data_end - 1] == b'\n' {
            data_end -= 1;
        }
        parts.push(parse_part(headers_raw, &body[data_start..data_end])?);
        pos = next;
    }
    Some(parts)
}

/// Next delimiter occurrence that begins a line: preceded by CRLF or LF,
/// or at `from` itself (start of body). Mid-line matches are data.
fn find_line_delim(hay: &[u8], from: usize, delim: &[u8]) -> Option<usize> {
    let mut pos = from;
    while let Some(rel) = find_sub(&hay[pos..], delim) {
        let at = pos + rel;
        let line_start = at == from
            || (at >= 2 && &hay[at - 2..at] == b"\r\n")
            || (at >= 1 && hay[at - 1] == b'\n');
        if line_start {
            return Some(at);
        }
        pos = at + 1;
    }
    None
}

fn parse_part(headers_raw: &[u8], data: &[u8]) -> Option<Part> {
    let headers = String::from_utf8_lossy(headers_raw);
    let mut name = None;
    let mut filename = None;
    let mut content_type = None;
    for line in headers.lines() {
        let lower = line.to_ascii_lowercase();
        if lower.starts_with("content-disposition:") {
            for piece in line.split(';').map(str::trim) {
                if let Some(v) = piece.strip_prefix("name=") {
                    name = Some(v.trim_matches('"').to_string());
                } else if let Some(v) = piece.strip_prefix("filename=") {
                    filename = Some(v.trim_matches('"').to_string());
                }
            }
        } else if let Some(v) = lower.strip_prefix("content-type:") {
            content_type = Some(v.trim().to_string());
        }
    }
    Some(Part {
        name: name?,
        filename,
        content_type,
        data: data.to_vec(),
    })
}

fn find_sub(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

fn field(parts: &[Part], key: &str) -> Option<String> {
    parts
        .iter()
        .find(|p| p.name == key && p.filename.is_none())
        .map(|p| String::from_utf8_lossy(&p.data).into_owned())
}

/// whisper-server `/inference` fields we forward, live-verified on
/// master b5130 (probe + binary strings). Everything else a client
/// sends (OpenAI-only concepts like `timestamp_granularities` included)
/// is dropped rather than relayed into an upstream 400.
const FORWARDED_FIELDS: &[&str] = &[
    "response_format",
    "language",
    "temperature",
    "prompt",
    "beam_size",
    "no_timestamps",
    "temperature_inc",
    "best_of",
    "offset_t",
    "offset_n",
    "duration",
    "max_context",
    "max_len",
    "split_on_word",
    "word_thold",
    "entropy_thold",
    "logprob_thold",
    "no_fallback",
    "carry_initial_prompt",
    "detect_language",
    "audio_ctx",
    // `translate` is handled by the caller: the transcriptions route
    // relays it as sent; the translations route owns it (always true).
];

/// Rebuilt `/inference` form fields (pure): whitelisted client fields
/// verbatim, `translate` per the route (`force_translate` overrides any
/// client value — a translations request always translates), and the
/// `response_format=json` default that keeps non-verbose clients on the
/// machine-readable shape (observed live: no default upstream).
fn forwarded_fields(parts: &[Part], force_translate: bool) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for key in FORWARDED_FIELDS {
        if let Some(v) = field(parts, key) {
            out.push(((*key).to_string(), v));
        }
    }
    let translate = if force_translate {
        "true".to_string()
    } else {
        field(parts, "translate").unwrap_or_else(|| "false".to_string())
    };
    out.push(("translate".to_string(), translate));
    if field(parts, "response_format").is_none() {
        out.push(("response_format".to_string(), "json".to_string()));
    }
    out
}

/// Local lane transport: ensure the lazy child (hot model swap on size
/// change), forward the multipart fields whisper-server understands,
/// and return the raw upstream triple. Shared by the sync path and the
/// async job task — the Err codes/messages are the sync route's exact
/// error surface, so both lanes fail identically.
async fn forward_local_raw(
    state: &Arc<AppState>,
    parts: &[Part],
    file: &Part,
    size: &str,
    bin: &std::path::Path,
    lib_dir: &std::path::Path,
    force_translate: bool,
) -> Result<(StatusCode, String, Bytes), (u16, String)> {
    let Some(model_path) = whisper::model_file(&state.dirs, size) else {
        return Err((500, format!("whisper model ggml-{size}.bin vanished")));
    };
    let port = match state
        .whisper
        .ensure(
            size,
            &model_path,
            bin,
            lib_dir,
            std::time::Duration::from_mins(2),
        )
        .await
    {
        Ok(p) => p,
        Err(e) => return Err((502, format!("whisper-server: {e:#}"))),
    };
    let mime = file
        .content_type
        .clone()
        .unwrap_or_else(|| "application/octet-stream".to_string());
    let mut form = reqwest::multipart::Form::new().part(
        "file",
        reqwest::multipart::Part::bytes(file.data.clone())
            .file_name(file.filename.clone().unwrap_or_else(|| "audio".to_string()))
            .mime_str(&mime)
            .unwrap_or_else(|_| reqwest::multipart::Part::bytes(file.data.clone())),
    );
    for (key, v) in forwarded_fields(parts, force_translate) {
        form = form.text(key, v);
    }
    let url = format!("http://127.0.0.1:{port}/inference");
    let resp = match state.http.post(&url).multipart(form).send().await {
        Ok(r) => r,
        Err(e) => return Err((502, format!("whisper inference: {e:#}"))),
    };
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    let bytes = resp.bytes().await.unwrap_or_default();
    Ok((status, ct, bytes))
}

/// Sync local lane: same contract as before the async split — the raw
/// triple rendered as a verbatim passthrough, errors as `openai_error`
/// with the historical codes and messages.
async fn forward_local(
    state: &Arc<AppState>,
    parts: &[Part],
    file: &Part,
    size: &str,
    bin: &std::path::Path,
    lib_dir: &std::path::Path,
    force_translate: bool,
) -> Response {
    match forward_local_raw(state, parts, file, size, bin, lib_dir, force_translate).await {
        Ok((status, ct, bytes)) => Response::builder()
            .status(status)
            .header(axum::http::header::CONTENT_TYPE, ct)
            .body(axum::body::Body::from(bytes))
            .unwrap_or_else(|_| openai_error(500, "response build").into_response()),
        Err((code, msg)) => openai_error(code, &msg),
    }
}

/// Registry ceiling: abandoned handles must not grow the map forever.
/// At the cap the oldest terminal job is evicted first (nobody polls
/// those anymore); if all are live, the oldest overall goes — its task
/// is aborted, the same as an explicit cancel.
const MAX_AUDIO_JOBS: usize = 256;

/// One job's lifecycle. `Queued` exists only between handle reservation
/// and the task's first instruction; `Completed` carries the upstream
/// triple verbatim so the poll response is the sync response, replayed.
#[derive(Clone)]
enum AudioJobState {
    Queued,
    Running,
    Completed {
        status: u16,
        content_type: String,
        body: Arc<Vec<u8>>,
    },
    Failed(String),
    Cancelled,
}

impl AudioJobState {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Completed { .. } => "completed",
            Self::Failed(_) => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    fn terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed { .. } | Self::Failed(_) | Self::Cancelled
        )
    }
}

struct AudioJobEntry {
    state: AudioJobState,
    handle: Option<tokio::task::JoinHandle<()>>,
    seq: u64,
    created_at: u64,
}

/// Gateway-owned async audio jobs. Every method takes the inner lock
/// briefly and never across an await — job tasks re-enter the registry
/// to update their own state, so a held lock would deadlock the lane.
#[derive(Default)]
pub struct AudioJobs {
    inner: std::sync::Mutex<AudioJobStore>,
}

#[derive(Default)]
struct AudioJobStore {
    jobs: HashMap<String, AudioJobEntry>,
    seq: u64,
    boot_nanos: u64,
}

impl AudioJobs {
    /// Jobs not yet terminal (queued or running) — the `/metrics`
    /// activity gauge (audit MM9).
    #[must_use]
    pub fn active_count(&self) -> usize {
        let store = self.inner.lock().expect("audio jobs lock");
        store
            .jobs
            .values()
            .filter(|j| matches!(j.state, AudioJobState::Queued | AudioJobState::Running))
            .count()
    }

    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: std::sync::Mutex::new(AudioJobStore {
                boot_nanos: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(0)),
                ..AudioJobStore::default()
            }),
        }
    }

    /// Reserve a job id and Queued slot. Called just before the task
    /// spawns; `set_handle` attaches the `JoinHandle` right after.
    fn reserve(&self) -> String {
        let mut store = self.inner.lock().expect("audio jobs lock");
        store.seq += 1;
        let id = format!("aj{}-{}", store.boot_nanos, store.seq);
        let seq = store.seq;
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        if store.jobs.len() >= MAX_AUDIO_JOBS {
            evict_one(&mut store);
        }
        store.jobs.insert(
            id.clone(),
            AudioJobEntry {
                state: AudioJobState::Queued,
                handle: None,
                seq,
                created_at,
            },
        );
        id
    }

    fn set_handle(&self, id: &str, handle: tokio::task::JoinHandle<()>) {
        let mut store = self.inner.lock().expect("audio jobs lock");
        if let Some(entry) = store.jobs.get_mut(id) {
            // A cancel that fired before the handle landed already
            // marked the job terminal — keep the handle anyway so a
            // later drop can still abort the straggler.
            entry.handle = Some(handle);
        }
    }

    fn mark_running(&self, id: &str) {
        let mut store = self.inner.lock().expect("audio jobs lock");
        if let Some(entry) = store.jobs.get_mut(id) {
            if matches!(entry.state, AudioJobState::Queued) {
                entry.state = AudioJobState::Running;
            }
        }
    }

    fn finish(&self, id: &str, status: u16, content_type: &str, body: Vec<u8>) {
        let mut store = self.inner.lock().expect("audio jobs lock");
        if let Some(entry) = store.jobs.get_mut(id) {
            if matches!(entry.state, AudioJobState::Queued | AudioJobState::Running) {
                entry.state = AudioJobState::Completed {
                    status,
                    content_type: content_type.to_string(),
                    body: Arc::new(body),
                };
            }
        }
    }

    fn fail(&self, id: &str, message: String) {
        let mut store = self.inner.lock().expect("audio jobs lock");
        if let Some(entry) = store.jobs.get_mut(id) {
            if matches!(entry.state, AudioJobState::Queued | AudioJobState::Running) {
                entry.state = AudioJobState::Failed(message);
            }
        }
    }

    /// Cancel a queued/running job: abort its task, mark Cancelled.
    /// `None` = unknown id; `Some(false)` = already terminal (idempotent
    /// poll-friendly no-op).
    fn cancel(&self, id: &str) -> Option<bool> {
        let handle = {
            let mut store = self.inner.lock().expect("audio jobs lock");
            let entry = store.jobs.get_mut(id)?;
            if entry.state.terminal() {
                return Some(false);
            }
            entry.state = AudioJobState::Cancelled;
            entry.handle.take()
        };
        // Abort outside the lock: abort schedules onto the runtime and
        // can run arbitrary Drop code in the aborted future.
        if let Some(handle) = handle {
            handle.abort();
        }
        Some(true)
    }

    /// Poll payload for `/v1/audio/jobs/{id}`: status plus the upstream
    /// triple on completion (body parsed as JSON when possible, raw
    /// text otherwise — transcription bodies are JSON in practice).
    fn payload(&self, id: &str) -> Option<serde_json::Value> {
        let store = self.inner.lock().expect("audio jobs lock");
        let entry = store.jobs.get(id)?;
        let mut payload = serde_json::json!({
            "id": id,
            "object": "whisper.job",
            "status": entry.state.as_str(),
            "created_at": entry.created_at,
        });
        match &entry.state {
            AudioJobState::Completed {
                status,
                content_type,
                body,
            } => {
                payload["status_code"] = serde_json::json!(status);
                payload["content_type"] = serde_json::json!(content_type);
                payload["result"] = serde_json::from_slice::<serde_json::Value>(body)
                    .unwrap_or_else(|_| {
                        serde_json::Value::String(String::from_utf8_lossy(body).into_owned())
                    });
            }
            AudioJobState::Failed(message) => {
                payload["error"] = serde_json::json!(message);
            }
            _ => {}
        }
        Some(payload)
    }
}

/// Registry eviction: oldest terminal job first; if everything is live,
/// the oldest overall (aborted — an implicit cancel).
fn evict_one(store: &mut AudioJobStore) {
    let victim = store
        .jobs
        .iter()
        .min_by_key(|(_, e)| (u64::from(!e.state.terminal()), e.seq))
        .map(|(id, _)| id.clone());
    if let Some(id) = victim {
        if let Some(entry) = store.jobs.remove(&id) {
            if let Some(handle) = entry.handle {
                handle.abort();
            }
        }
    }
}

/// The gateway-only `async` knob: a plain form field (the audio routes
/// are multipart; images uses the same knob as a JSON field). Honored
/// on the local lane only — a remote forward streams the client's body
/// untouched, remotes keep their own sync contracts.
fn wants_async(parts: &[Part]) -> bool {
    field(parts, "async").is_some_and(|v| {
        let v = v.trim();
        v.eq_ignore_ascii_case("true") || v == "1"
    })
}

/// Submit an async job: reserve the handle, spawn the forward task,
/// answer with the gateway-owned job envelope (same shape vocabulary as
/// the images lane's `async` handles). No awaits — the spawned task owns
/// all the waiting.
fn submit_async(
    state: &Arc<AppState>,
    parts: Vec<Part>,
    file: Part,
    size: String,
    bin: &std::path::Path,
    lib_dir: &std::path::Path,
    force_translate: bool,
) -> Response {
    let id = state.audio_jobs.reserve();
    let task_id = id.clone();
    let task_state = Arc::clone(state);
    let bin = bin.to_path_buf();
    let lib_dir = lib_dir.to_path_buf();
    let task = tokio::spawn(async move {
        task_state.audio_jobs.mark_running(&task_id);
        match forward_local_raw(
            &task_state,
            &parts,
            &file,
            &size,
            &bin,
            &lib_dir,
            force_translate,
        )
        .await
        {
            Ok((status, ct, bytes)) => {
                task_state
                    .audio_jobs
                    .finish(&task_id, status.as_u16(), &ct, bytes.to_vec());
            }
            Err((_code, msg)) => task_state.audio_jobs.fail(&task_id, msg),
        }
    });
    state.audio_jobs.set_handle(&id, task);
    let created_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let body = serde_json::json!({
        "id": id,
        "object": "whisper.job",
        "status": "queued",
        "created_at": created_at,
        "poll_url": format!("/v1/audio/jobs/{id}"),
        "cancel_url": format!("/v1/audio/jobs/{id}/cancel"),
    });
    Response::builder()
        .status(200)
        .header(axum::http::header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(
            serde_json::to_vec(&body).unwrap_or_default(),
        ))
        .unwrap_or_else(|_| openai_error(500, "response build").into_response())
}

/// GET /v1/audio/jobs/{id} — poll a gateway-owned async audio job.
#[allow(clippy::unused_async)] // axum's Handler trait requires async fns
pub async fn audio_jobs_get(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(job_id): axum::extract::Path<String>,
) -> Response {
    if !job_id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return openai_error(400, "invalid job id");
    }
    match state.audio_jobs.payload(&job_id) {
        Some(payload) => Response::builder()
            .status(200)
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(
                serde_json::to_vec(&payload).unwrap_or_default(),
            ))
            .unwrap_or_else(|_| openai_error(500, "response build").into_response()),
        None => openai_error(
            404,
            "job not found — audio jobs are gateway-owned and die with the gateway process",
        ),
    }
}

/// POST /v1/audio/jobs/{id}/cancel — abort a queued/running job. The
/// wait stops immediately; an in-flight upstream inference runs to
/// completion inside the child (whisper-server has no cancel surface —
/// the result is simply discarded). Terminal jobs answer with their
/// final state (idempotent).
#[allow(clippy::unused_async)] // axum's Handler trait requires async fns
pub async fn audio_jobs_cancel(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(job_id): axum::extract::Path<String>,
) -> Response {
    if !job_id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return openai_error(400, "invalid job id");
    }
    // cancel() performs the abort side effect (a terminal/unknown job is
    // a no-op); either way the answer is the job's current state — 200
    // with the payload, 404 when no such job exists.
    let _ = state.audio_jobs.cancel(&job_id);
    match state.audio_jobs.payload(&job_id) {
        Some(payload) => Response::builder()
            .status(200)
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(
                serde_json::to_vec(&payload).unwrap_or_default(),
            ))
            .unwrap_or_else(|_| openai_error(500, "response build").into_response()),
        None => openai_error(
            404,
            "job not found — audio jobs are gateway-owned and die with the gateway process",
        ),
    }
}

/// GET /v1/audio/capabilities — the audio lane's gateway-derived menu.
/// Unlike `/v1/images/capabilities` (a relay of the child's sampler
/// menu), whisper-server has no capabilities route, so this reads local
/// truth only: engine presence, pulled models, live child state, and
/// the async/idle knobs. Boots nothing.
pub async fn audio_capabilities(State(state): State<Arc<AppState>>) -> Response {
    let installed = whisper::server_bin(&state.dirs).is_some();
    let models = whisper::list_models(&state.dirs);
    let child = state
        .whisper
        .status()
        .await
        .map(|(_port, loaded)| serde_json::json!({ "alive": true, "loaded": loaded }));
    let body = serde_json::json!({
        "object": "whisper.capabilities",
        "engine": { "kind": "whisper", "installed": installed },
        "models": models,
        "child": child,
        "endpoints": [
            "/v1/audio/transcriptions",
            "/v1/audio/translations",
            "/v1/audio/jobs/{id}",
            "/v1/audio/jobs/{id}/cancel"
        ],
        "async": { "field": "async", "values": ["true", "1"], "local_lane_only": true },
        "idle_timeout_secs": state.config.whisper_idle_secs,
    });
    Response::builder()
        .status(200)
        .header(axum::http::header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(
            serde_json::to_vec(&body).unwrap_or_default(),
        ))
        .unwrap_or_else(|_| openai_error(500, "response build").into_response())
}

pub async fn audio_transcriptions(
    State(state): State<Arc<AppState>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    uri: Uri,
    method: Method,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let content_type = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let Some(parts) = parse_multipart(&body, content_type) else {
        return openai_error(400, "malformed multipart body (no parseable boundary)");
    };
    let Some(file) = parts
        .iter()
        .find(|p| p.name == "file" && p.filename.is_some())
    else {
        return openai_error(400, "missing file part in multipart body");
    };
    let model = field(&parts, "model");

    // (1) Explicit remote intent wins: `name:model` never goes local.
    // F11: remote audio lanes pay key admission too (scope + rate +
    // request count) — every other remote lane has since the audit; the
    // token sniffer has nothing to read in whisper JSON (no usage
    // field), so request counts are the charge unit here.
    if let Some(model) = model.as_deref() {
        if split_remote(model, &state.config).is_some() {
            if let Some(Extension(k)) = &key_ext {
                if let Some(entry) = state.keys.entry(&k.name) {
                    if let Err(rej) = state.keys.check(&entry, model) {
                        return rej.to_response();
                    }
                    state.keys.charge_request(&k.name);
                }
            }
            return crate::remotes::forward_with_health(
                &state,
                model,
                &method,
                uri.path_and_query().map_or(
                    "/v1/audio/transcriptions",
                    axum::http::uri::PathAndQuery::as_str,
                ),
                &headers,
                body,
            )
            .await;
        }
    }

    // (2) Local lane: installed binary + pulled ggml model. Name the exact
    // missing half — "lane not installed" when the server binary is the
    // gap, "no model pulled" when the server is fine but transcription
    // has nothing to load (both halves observed live in validation).
    // The lane pays the same admission the remote branch just paid
    // (scope on the whisper size name + rpm/tpm/daily + request charge):
    // local compute is not a free lane on an authed gateway (audit MM1).
    if let Err(resp) =
        state.admit_or_respond(key_ext.as_ref(), model.as_deref().unwrap_or("whisper-1"))
    {
        return *resp;
    }
    let available = whisper::list_models(&state.dirs);
    let requested = model.as_deref().or(Some("whisper-1"));
    let server = whisper::server_bin(&state.dirs);
    let size = server
        .as_ref()
        .and_then(|_| whisper::resolve_model(requested, &available));
    if let (Some((bin, lib_dir)), Some(size)) = (&server, size) {
        if wants_async(&parts) {
            let file_part = file.clone();
            return submit_async(&state, parts, file_part, size.clone(), bin, lib_dir, false);
        }
        return forward_local(&state, &parts, file, &size, bin, lib_dir, false).await;
    }

    // (3) Teaching error: no local lane and no remote intent.
    openai_error(
        501,
        &teaching_detail(
            server.is_some(),
            requested.unwrap_or("whisper-1"),
            &available,
        ),
    )
}

/// `/v1/audio/translations`: same decision order as transcriptions,
/// same lazy child — the only difference is `translate=true` forced on
/// the rebuilt `/inference` request (upstream has no separate route).
pub async fn audio_translations(
    State(state): State<Arc<AppState>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    uri: Uri,
    method: Method,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let content_type = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let Some(parts) = parse_multipart(&body, content_type) else {
        return openai_error(400, "malformed multipart body (no parseable boundary)");
    };
    let Some(file) = parts
        .iter()
        .find(|p| p.name == "file" && p.filename.is_some())
    else {
        return openai_error(400, "missing file part in multipart body");
    };
    let model = field(&parts, "model");

    // (1) Explicit remote intent wins (F11 admission, same as
    // transcriptions — request counts are the charge unit here).
    if let Some(model) = model.as_deref() {
        if split_remote(model, &state.config).is_some() {
            if let Some(Extension(k)) = &key_ext {
                if let Some(entry) = state.keys.entry(&k.name) {
                    if let Err(rej) = state.keys.check(&entry, model) {
                        return rej.to_response();
                    }
                    state.keys.charge_request(&k.name);
                }
            }
            return crate::remotes::forward_with_health(
                &state,
                model,
                &method,
                uri.path_and_query().map_or(
                    "/v1/audio/translations",
                    axum::http::uri::PathAndQuery::as_str,
                ),
                &headers,
                body,
            )
            .await;
        }
    }

    // (2) Local lane, forced translation. Same admission as the remote
    // branch and every other gated lane (audit MM1).
    if let Err(resp) =
        state.admit_or_respond(key_ext.as_ref(), model.as_deref().unwrap_or("whisper-1"))
    {
        return *resp;
    }
    let available = whisper::list_models(&state.dirs);
    let requested = model.as_deref().or(Some("whisper-1"));
    let server = whisper::server_bin(&state.dirs);
    let size = server
        .as_ref()
        .and_then(|_| whisper::resolve_model(requested, &available));
    if let (Some((bin, lib_dir)), Some(size)) = (&server, size) {
        if wants_async(&parts) {
            let file_part = file.clone();
            return submit_async(&state, parts, file_part, size.clone(), bin, lib_dir, true);
        }
        return forward_local(&state, &parts, file, &size, bin, lib_dir, true).await;
    }

    // (3) Teaching error: no local lane and no remote intent.
    openai_error(
        501,
        &teaching_detail(
            server.is_some(),
            requested.unwrap_or("whisper-1"),
            &available,
        ),
    )
}

/// 501 body for a local-lane miss, naming the EXACT missing half (pure,
/// unit-tested): the server binary vs the model payload.
#[must_use]
fn teaching_detail(has_server: bool, requested: &str, available: &[String]) -> String {
    let avail = if available.is_empty() {
        "<none pulled>".to_string()
    } else {
        available.join(", ")
    };
    if has_server {
        format!(
            "whisper server installed but no usable model for {requested:?} \
             — pull one: `blazar whisper --pull base`. \
             Available local models: {avail}."
        )
    } else {
        format!(
            "no transcription backend: whisper server not installed \
             (`blazar engine install --kind whisper`; legacy: \
             `blazar whisper --install`) and the request does not name a \
             remote (`whisper:<model>` with a [[remotes]] entry). \
             Available local models: {avail}."
        )
    }
}

#[cfg(test)]
#[allow(non_snake_case)] // suite convention: unit__scenario__expected (§6b)
mod tests {
    use super::*;

    fn body(boundary: &str, parts: &[(&str, &str, &[u8])]) -> (Vec<u8>, String) {
        // (name, filename, data) — data is embedded verbatim between the
        // delimiters, exactly like a real audio part would be.
        let mut b = Vec::new();
        for (name, filename, data) in parts {
            b.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            b.extend_from_slice(
                format!(
                    "Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n\
                     Content-Type: application/octet-stream\r\n\r\n"
                )
                .as_bytes(),
            );
            b.extend_from_slice(data);
            b.extend_from_slice(b"\r\n");
        }
        b.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        let ct = format!("multipart/form-data; boundary={boundary}");
        (b, ct)
    }

    #[test]
    fn unit__parse_multipart__two_parts_with_binary_data() {
        let audio: Vec<u8> = (0..=255u8).cycle().take(1024).collect();
        let (b, ct) = body(
            "XbOuNdArY",
            &[
                ("model", "req.bin", b"whisper-1"),
                ("file", "audio.wav", &audio),
            ],
        );
        let parts = parse_multipart(&b, &ct).expect("parses");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].name, "model");
        assert_eq!(parts[0].data, b"whisper-1");
        assert_eq!(parts[1].name, "file");
        assert_eq!(parts[1].filename.as_deref(), Some("audio.wav"));
        assert_eq!(
            parts[1].content_type.as_deref(),
            Some("application/octet-stream")
        );
        assert_eq!(parts[1].data, audio);
    }

    #[test]
    fn unit__parse_multipart__fake_boundary_inside_audio_ignored() {
        // Audio bytes that CONTAIN the delimiter string must not split the
        // part: only delimiters at true boundaries count. We craft data
        // with an embedded "--XbOuNdArY" and verify it stays in the data.
        let mut audio = b"RIFFxxxx".to_vec();
        audio.extend_from_slice(b"--XbOuNdArY\r\nContent-Disposition: forged");
        let (b, ct) = body("XbOuNdArY", &[("file", "a.wav", &audio)]);
        let parts = parse_multipart(&b, &ct).expect("parses");
        assert_eq!(parts.len(), 1, "forged boundary must not split");
        assert_eq!(parts[0].data, audio);
    }

    #[test]
    fn unit__parse_multipart__missing_boundary_param_none() {
        let (b, _) = body("XbOuNdArY", &[("file", "a.wav", b"abc")]);
        assert!(parse_multipart(&b, "multipart/form-data").is_none());
        assert!(parse_multipart(&b, "text/plain").is_none());
    }

    #[test]
    fn unit__parse_multipart__empty_file_part_is_data() {
        let (b, ct) = body("XbOuNdArY", &[("file", "a.wav", b"")]);
        let parts = parse_multipart(&b, &ct).expect("parses");
        assert_eq!(parts.len(), 1);
        assert!(parts[0].data.is_empty());
    }

    #[test]
    fn unit__teaching_detail__server_missing_names_install() {
        let d = teaching_detail(false, "whisper-1", &[]);
        assert!(d.contains("whisper server not installed"), "got: {d}");
        assert!(d.contains("`blazar whisper --install`"), "got: {d}");
        assert!(d.contains("<none pulled>"), "got: {d}");
    }

    #[test]
    fn unit__teaching_detail__model_missing_names_pull() {
        let d = teaching_detail(true, "whisper-1", &[]);
        assert!(
            d.contains("server installed but no usable model"),
            "got: {d}"
        );
        assert!(d.contains("`blazar whisper --pull base`"), "got: {d}");
        assert!(
            !d.contains("not installed"),
            "must not claim server missing: {d}"
        );
    }

    #[test]
    fn unit__teaching_detail__with_models_lists_them_and_request() {
        let avail = vec!["base".to_string(), "small".to_string()];
        let d = teaching_detail(true, "large-v3", &avail);
        assert!(d.contains("\"large-v3\""), "got: {d}");
        assert!(d.contains("base, small"), "got: {d}");
    }

    /// Form body where `fields` are plain value parts (no filename —
    /// what `field()` matches) plus one file part.
    fn mixed_body(
        boundary: &str,
        fields: &[(&str, &[u8])],
        file: (&str, &[u8]),
    ) -> (Vec<u8>, String) {
        let mut b = Vec::new();
        for (name, data) in fields {
            b.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            b.extend_from_slice(
                format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
            );
            b.extend_from_slice(data);
            b.extend_from_slice(b"\r\n");
        }
        b.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        b.extend_from_slice(
            format!(
                "Content-Disposition: form-data; name=\"file\"; \
                 filename=\"{}\"\r\n\r\n",
                file.0
            )
            .as_bytes(),
        );
        b.extend_from_slice(file.1);
        b.extend_from_slice(b"\r\n");
        b.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        let ct = format!("multipart/form-data; boundary={boundary}");
        (b, ct)
    }

    #[test]
    fn unit__forwarded_fields__whitelist_forwards_verified_and_drops_unknowns() {
        // Verified fields ride the rebuilt form; OpenAI-only concepts and
        // the model knob (resolved earlier, upstream has no such field)
        // are dropped; response_format defaults to json; translate
        // defaults to false on the transcriptions route.
        let (b, ct) = mixed_body(
            "XbOuNdArY",
            &[
                ("model", b"whisper-1"),
                ("language", b"en"),
                ("beam_size", b"2"),
                ("timestamp_granularities[]", b"word"),
            ],
            ("audio.wav", b"RIFF"),
        );
        let parts = parse_multipart(&b, &ct).expect("parses");
        let fields = forwarded_fields(&parts, false);
        let get = |k: &str| fields.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str());
        assert_eq!(get("language"), Some("en"));
        assert_eq!(get("beam_size"), Some("2"));
        assert_eq!(get("response_format"), Some("json"), "default injected");
        assert_eq!(get("translate"), Some("false"), "default injected");
        assert!(
            fields
                .iter()
                .all(|(n, _)| n != "timestamp_granularities[]" && n != "model"),
            "unknowns dropped: {fields:?}"
        );
    }

    #[test]
    fn unit__forwarded_fields__force_translate_overrides_client_value() {
        // The translations route OWNS translate — even an explicit
        // client "false" must not disable it, and a client "true" must
        // not duplicate the part.
        let (b, ct) = mixed_body("XbOuNdArY", &[("translate", b"false")], ("a.wav", b"RIFF"));
        let parts = parse_multipart(&b, &ct).expect("parses");
        let fields = forwarded_fields(&parts, true);
        let translates: Vec<&str> = fields
            .iter()
            .filter(|(n, _)| n == "translate")
            .map(|(_, v)| v.as_str())
            .collect();
        assert_eq!(translates, vec!["true"], "forced + no duplicate");
    }

    #[test]
    fn unit__forwarded_fields__explicit_response_format_wins_over_default() {
        let (b, ct) = mixed_body(
            "XbOuNdArY",
            &[("response_format", b"verbose_json")],
            ("a.wav", b"RIFF"),
        );
        let parts = parse_multipart(&b, &ct).expect("parses");
        let fields = forwarded_fields(&parts, false);
        let n = fields
            .iter()
            .filter(|(name, _)| name == "response_format")
            .count();
        assert_eq!(n, 1, "exactly one response_format: {fields:?}");
        assert!(
            fields.contains(&("response_format".to_string(), "verbose_json".to_string())),
            "client value kept: {fields:?}"
        );
    }

    #[test]
    fn unit__wants_async__accepts_true_and_1_rejects_rest() {
        let mk = |v: &str| mixed_body("XbOuNdArY", &[("async", v.as_bytes())], ("a.wav", b"RIFF"));
        for yes in ["true", "TRUE", "1", " true "] {
            let (b, ct) = mk(yes);
            let parts = parse_multipart(&b, &ct).expect("parses");
            assert!(wants_async(&parts), "{yes:?} must request async");
        }
        for no in ["false", "0", "yes", ""] {
            let (b, ct) = mk(no);
            let parts = parse_multipart(&b, &ct).expect("parses");
            assert!(!wants_async(&parts), "{no:?} must stay sync");
        }
        // Absent field = sync, the default every existing client gets.
        let (b, ct) = mixed_body("XbOuNdArY", &[], ("a.wav", b"RIFF"));
        let parts = parse_multipart(&b, &ct).expect("parses");
        assert!(!wants_async(&parts));
    }

    /// Registry lifecycle: reserve → running → completed carries the
    /// upstream triple into the poll payload; fail and cancel are
    /// terminal and idempotent; unknown ids 404 (payload None).
    #[test]
    fn unit__audio_jobs__lifecycle_and_payload_shapes() {
        let jobs = AudioJobs::new();
        let id = jobs.reserve();
        assert_eq!(
            jobs.payload(&id).unwrap()["status"],
            "queued",
            "fresh reservation is queued"
        );
        jobs.mark_running(&id);
        assert_eq!(jobs.payload(&id).unwrap()["status"], "running");
        jobs.finish(&id, 200, "application/json", br#"{"text":"hi"}"#.to_vec());
        let done = jobs.payload(&id).unwrap();
        assert_eq!(done["status"], "completed");
        assert_eq!(done["status_code"], 200);
        assert_eq!(done["result"]["text"], "hi", "body parsed as JSON");
        // Terminal stays terminal: late finish/cancel never overwrite.
        jobs.fail(&id, "late failure".into());
        jobs.cancel(&id);
        assert_eq!(jobs.payload(&id).unwrap()["status"], "completed");

        // Non-JSON body replays as a raw string under "result".
        let id2 = jobs.reserve();
        jobs.mark_running(&id2);
        jobs.finish(&id2, 200, "text/plain", b"plain text".to_vec());
        assert_eq!(
            jobs.payload(&id2).unwrap()["result"],
            serde_json::json!("plain text")
        );

        // Failed jobs surface the sync lane's error message verbatim.
        let id3 = jobs.reserve();
        jobs.mark_running(&id3);
        jobs.fail(&id3, "whisper-server: boom".into());
        let failed = jobs.payload(&id3).unwrap();
        assert_eq!(failed["status"], "failed");
        assert_eq!(failed["error"], "whisper-server: boom");

        // Cancel of a live job flips it to cancelled, second cancel is
        // a no-op, and unknown ids have no payload.
        let id4 = jobs.reserve();
        jobs.mark_running(&id4);
        assert_eq!(jobs.cancel(&id4), Some(true));
        assert_eq!(jobs.cancel(&id4), Some(false), "terminal cancel idempotent");
        assert_eq!(jobs.payload(&id4).unwrap()["status"], "cancelled");
        assert!(jobs.payload("aj-nonexistent").is_none());
        assert_eq!(jobs.cancel("aj-nonexistent"), None);
    }

    /// At the cap the registry evicts the oldest TERMINAL job first and
    /// keeps live ones; ids stay unique across evictions.
    #[test]
    fn unit__audio_jobs__eviction_oldest_terminal_first() {
        let jobs = AudioJobs::new();
        // One live job reserved early, then a wave of terminal jobs past
        // the cap — the early live one must survive every eviction.
        let live = jobs.reserve();
        jobs.mark_running(&live);
        let mut first_terminal = String::new();
        for i in 0..=MAX_AUDIO_JOBS {
            let id = jobs.reserve();
            if i == 0 {
                first_terminal = id.clone();
            }
            jobs.finish(&id, 200, "application/json", Vec::new());
        }
        assert!(
            jobs.payload(&live).is_some(),
            "live job must not be evicted while terminal victims exist"
        );
        assert!(
            jobs.payload(&first_terminal).is_none(),
            "oldest terminal job is the eviction victim"
        );
    }

    /// Audit MM9: `active_count` is the /metrics footprint of the local
    /// audio lane — queued+running only; every terminal state (or
    /// eviction) stops counting.
    #[test]
    fn unit__audio_jobs__active_count_tracks_live_only() {
        let jobs = AudioJobs::new();
        assert_eq!(jobs.active_count(), 0, "empty registry");
        let a = jobs.reserve();
        assert_eq!(jobs.active_count(), 1, "queued counts as active");
        jobs.mark_running(&a);
        assert_eq!(jobs.active_count(), 1, "running still one active");
        let b = jobs.reserve();
        jobs.mark_running(&b);
        assert_eq!(jobs.active_count(), 2, "second live job counted");
        jobs.finish(&a, 200, "application/json", Vec::new());
        assert_eq!(jobs.active_count(), 1, "completed drops out");
        jobs.fail(&b, "boom".into());
        assert_eq!(jobs.active_count(), 0, "failed drops out");
        // Cancelled path too.
        let c = jobs.reserve();
        jobs.cancel(&c);
        assert_eq!(jobs.active_count(), 0, "cancelled drops out");
    }
}
