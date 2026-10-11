//! OpenAI-compatible Batch API (D4): JSONL in, background out.
//!
//! `/v1/files` stores an uploaded JSONL request file under
//! `data/blazar/batch/files/`; `/v1/batches` validates it and spawns a
//! background worker that replays every line through the gateway's own
//! loopback `/v1/chat/completions` (original auth header forwarded, so
//! per-key scopes/quotas apply to batch traffic too). Results are written
//! as an output JSONL file (`response.status_code` + full body per line)
//! and the batch object tracks counts. State is plain files — restart
//! safety is honest: an in-flight batch stops where it was (status stays
//! `in_progress`; re-submit to rerun). Sequential one-line-at-a-time
//! execution by design: batch is a throughput lane, not a latency one.

use crate::state::AppState;
use axum::body::Bytes;
use axum::extract::{Extension, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use blazar_core::BlazarDirs;
use blazar_core::config::BatchLimits;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Upload size cap: kept just under the gateway's 50 MiB default body
/// limit so the route-level check fires before axum's 413.
const MAX_UPLOAD_BYTES: usize = 48 * 1024 * 1024;
/// Only the chat-completions endpoint is batched (honest scope; embeddings
/// lane is a different surface).
const SUPPORTED_ENDPOINT: &str = "/v1/chat/completions";
/// Ids are short random hex so they sort/echo like upstream (`file-…`,
/// `batch-…`). BLZ-6: 48 bits from the OS CSPRNG — the previous
/// nanos-XOR-counter scheme was predictable, and ids must stay
/// unguessable even with object authz in place.
pub(crate) fn short_id(prefix: &str) -> String {
    use std::fmt::Write as _;
    let mut bytes = [0u8; 6];
    getrandom::fill(&mut bytes).expect("OS entropy source");
    let mut hex = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(hex, "{b:02x}");
    }
    format!("{prefix}-{hex}")
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn root(dirs: &BlazarDirs) -> PathBuf {
    dirs.data_dir.join("batch")
}
fn files_dir(dirs: &BlazarDirs) -> PathBuf {
    root(dirs).join("files")
}
fn batches_dir(dirs: &BlazarDirs) -> PathBuf {
    root(dirs).join("batches")
}
fn batch_meta(dirs: &BlazarDirs, id: &str) -> PathBuf {
    batches_dir(dirs).join(format!("{id}.json"))
}

fn read_json(path: &PathBuf) -> Option<Value> {
    std::fs::read(path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
}
fn write_json(path: &PathBuf, v: &Value) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(
        path,
        serde_json::to_vec_pretty(v).map_err(std::io::Error::other)?,
    )
}

fn err(status: StatusCode, message: &str) -> Response {
    (
        status,
        axum::Json(json!({"error": {"message": message, "type": "invalid_request_error"}})),
    )
        .into_response()
}

/// BLZ-1: the creating key's name, stamped into meta JSON as `_owner`
/// (underscore = internal, never part of a public wire shape). `None`
/// on the open gateway — such objects stay visible to every caller.
fn owner_name(key_ext: Option<&Extension<crate::keys::KeyCtx>>) -> Option<String> {
    key_ext.map(|k| k.name.clone())
}

/// BLZ-1: does `key` see meta stamped with `_owner`? Missing/null
/// `_owner` (legacy or open-gateway object) is visible to all.
fn meta_allows(m: &Value, key: Option<&crate::keys::KeyCtx>) -> bool {
    crate::keys::owner_allows(m["_owner"].as_str(), key)
}

/// Drop the internal `_owner` stamp before a meta object reaches the
/// wire — public shapes stay byte-identical to pre-scope responses.
fn strip_owner(mut v: Value) -> Value {
    if let Some(o) = v.as_object_mut() {
        o.remove("_owner");
    }
    v
}

// ---------- BLZ-6: storage quotas + retention ----------

/// OpenAI-dialect terminal statuses — a batch in any of these states has
/// no worker left and can never make progress again. The superset covers
/// statuses upstream clients echo back in listings.
fn openai_terminal(status: &str) -> bool {
    matches!(
        status,
        "completed" | "failed" | "cancelled" | "canceled" | "expired" | "succeeded" | "errored"
    )
}

/// Is this batch meta (either dialect) finished with its work?
fn batch_terminal(m: &Value) -> bool {
    if m["dialect"] == "anthropic" {
        m["internal_status"] == "ended"
    } else {
        m["status"].as_str().is_some_and(openai_terminal)
    }
}

/// Current on-disk footprint of the batch store: total bytes plus a
/// per-owner breakdown (uploads without an `_owner` stamp — open-gateway
/// or legacy — count toward the global total only).
fn stored_bytes(dirs: &BlazarDirs) -> (u64, HashMap<String, u64>) {
    let mut total = 0u64;
    let mut per_owner: HashMap<String, u64> = HashMap::new();
    let Ok(entries) = std::fs::read_dir(files_dir(dirs)) else {
        return (0, per_owner);
    };
    for e in entries.flatten() {
        let Some(name) = e.file_name().to_str().map(String::from) else {
            continue;
        };
        if !name.ends_with(".meta.json") {
            continue;
        }
        let Some(m) = read_json(&e.path()) else {
            continue;
        };
        // Meta records the upload size; fall back to the data file when
        // an older meta lacks the field.
        let bytes = m["bytes"].as_u64().unwrap_or_else(|| {
            let id = name.trim_end_matches(".meta.json");
            std::fs::metadata(files_dir(dirs).join(format!("{id}.jsonl"))).map_or(0, |md| md.len())
        });
        total = total.saturating_add(bytes);
        if let Some(owner) = m["_owner"].as_str() {
            *per_owner.entry(owner.to_string()).or_insert(0) += bytes;
        }
    }
    (total, per_owner)
}

/// Would `incoming` more bytes for `owner` exceed a configured store
/// quota? `None` = allowed. Zero quotas (the defaults) mean unlimited,
/// preserving pre-BLZ-6 behavior exactly.
fn store_quota_refusal(
    dirs: &BlazarDirs,
    limits: &BatchLimits,
    owner: Option<&str>,
    incoming: u64,
) -> Option<(StatusCode, String)> {
    if limits.max_store_bytes_total == 0 && limits.max_store_bytes_per_key == 0 {
        return None;
    }
    let (total, per_owner) = stored_bytes(dirs);
    if limits.max_store_bytes_total > 0
        && total.saturating_add(incoming) > limits.max_store_bytes_total
    {
        return Some((
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "upload exceeds the global batch store quota: {total} bytes stored + {incoming} incoming exceeds the {}-byte cap",
                limits.max_store_bytes_total
            ),
        ));
    }
    if limits.max_store_bytes_per_key > 0
        && let Some(owner) = owner
    {
        let used = per_owner.get(owner).copied().unwrap_or(0);
        if used.saturating_add(incoming) > limits.max_store_bytes_per_key {
            return Some((
                StatusCode::PAYLOAD_TOO_LARGE,
                format!(
                    "upload exceeds the per-key batch store quota: {used} bytes stored + {incoming} incoming exceeds the {}-byte cap for key '{owner}'",
                    limits.max_store_bytes_per_key
                ),
            ));
        }
    }
    None
}

/// Batches still running for `owner`, across both dialects.
fn active_batch_count(dirs: &BlazarDirs, owner: &str) -> u64 {
    let Ok(entries) = std::fs::read_dir(batches_dir(dirs)) else {
        return 0;
    };
    entries
        .flatten()
        .filter_map(|e| read_json(&e.path()))
        .filter(|m| m["_owner"].as_str() == Some(owner) && !batch_terminal(m))
        .count()
        .try_into()
        .unwrap_or(u64::MAX)
}

/// Retention sweep — BLZ-6. Runs inline at upload/batch-create time (no
/// timer thread to leak or race a daemon stop). Conservative by contract:
/// a batch is removed only when its status is TERMINAL and its finalize
/// (or creation) stamp is older than the retention window — in-flight
/// work is never touched. Output files ride along with their batch; an
/// input/output file is removed only when NO surviving batch references
/// it and its own `created_at` ages out. Every removal is logged;
/// per-entry errors are logged and skipped so one bad row cannot wedge
/// the lane. `retention_secs == 0` disables the sweep (retain forever).
fn sweep_expired(dirs: &BlazarDirs, retention_secs: u64, now: u64) {
    if retention_secs == 0 {
        return;
    }
    let cutoff = now.saturating_sub(retention_secs);

    // Pass 1: classify every batch meta and collect the file ids that
    // SURVIVING batches still reference (those files stay no matter
    // their age).
    let mut referenced: HashSet<String> = HashSet::new();
    let mut expired: Vec<(String, Option<String>)> = Vec::new();
    let Ok(entries) = std::fs::read_dir(batches_dir(dirs)) else {
        return;
    };
    for e in entries.flatten() {
        let Some(m) = read_json(&e.path()) else {
            continue;
        };
        let terminal = batch_terminal(&m);
        let stamp = if m["dialect"] == "anthropic" {
            m["ended_at_secs"]
                .as_u64()
                .or_else(|| m["created_at_secs"].as_u64())
        } else {
            m["finalized_at"]
                .as_u64()
                .or_else(|| m["created_at"].as_u64())
        };
        if terminal && stamp.is_some_and(|s| s < cutoff) {
            expired.push((
                m["id"].as_str().unwrap_or_default().to_string(),
                m["output_file_id"].as_str().map(String::from),
            ));
        } else {
            if let Some(i) = m["input_file_id"].as_str() {
                referenced.insert(i.to_string());
            }
            if let Some(o) = m["output_file_id"].as_str() {
                referenced.insert(o.to_string());
            }
        }
    }

    // Pass 2: remove expired batch metas and their output files. The
    // output file is uniquely owned by its batch, so it rides along.
    for (id, output_id) in expired {
        if let Err(e) = std::fs::remove_file(batch_meta(dirs, &id)) {
            tracing::warn!(target: "blazar::batch", batch = %id, error = %e, "retention sweep: batch meta removal failed");
            continue;
        }
        tracing::info!(target: "blazar::batch", batch = %id, "retention sweep removed expired batch");
        if let Some(output_id) = output_id {
            remove_store_file(dirs, &output_id);
        }
    }

    // Pass 3: orphaned files — no surviving batch references them and
    // their own created_at has aged out.
    let Ok(files) = std::fs::read_dir(files_dir(dirs)) else {
        return;
    };
    for f in files.flatten() {
        let Some(name) = f.file_name().to_str().map(String::from) else {
            continue;
        };
        if !name.ends_with(".meta.json") {
            continue;
        }
        let Some(m) = read_json(&f.path()) else {
            continue;
        };
        let Some(id) = m["id"].as_str() else {
            continue;
        };
        if referenced.contains(id) {
            continue;
        }
        if m["created_at"].as_u64().is_some_and(|t| t < cutoff) {
            remove_store_file(dirs, id);
        }
    }
}

/// Remove one file's data + meta pair, logging the removal (BLZ-6 —
/// deletions must be observable, and so must failed deletions: the
/// "removed" line only fires when something actually went away).
fn remove_store_file(dirs: &BlazarDirs, id: &str) {
    let mut removed_any = false;
    for suffix in [".jsonl", ".meta.json"] {
        let p = files_dir(dirs).join(format!("{id}{suffix}"));
        match std::fs::remove_file(&p) {
            Ok(()) => removed_any = true,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                tracing::warn!(target: "blazar::batch", file = %id, error = %e, "retention sweep: file removal failed");
            }
        }
    }
    if removed_any {
        tracing::info!(target: "blazar::batch", file = %id, "retention sweep removed expired file");
    }
}

/// `POST /v1/files` — multipart upload with a `file` part (and optional
/// `purpose` form field, accepted and recorded). Returns an OpenAI-style
/// file object.
pub async fn upload_file(
    State(state): State<std::sync::Arc<AppState>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let Some(content_type) = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(String::from)
    else {
        return err(StatusCode::BAD_REQUEST, "missing Content-Type");
    };
    let Some(parts) = crate::whisper::parse_multipart(&body, &content_type) else {
        return err(StatusCode::BAD_REQUEST, "malformed multipart body");
    };
    let file = parts.iter().find(|p| p.name == "file");
    let Some(file) = file else {
        return err(
            StatusCode::BAD_REQUEST,
            "multipart body needs a 'file' part",
        );
    };
    if file.data.is_empty() {
        return err(StatusCode::BAD_REQUEST, "file part is empty");
    }
    if file.data.len() > MAX_UPLOAD_BYTES {
        return err(
            StatusCode::PAYLOAD_TOO_LARGE,
            "file exceeds 48 MiB batch upload cap",
        );
    }
    let purpose = parts
        .iter()
        .find(|p| p.name == "purpose")
        .map(|p| String::from_utf8_lossy(&p.data).trim().to_string());
    if let Some(p) = &purpose
        && p != "batch"
    {
        return err(StatusCode::BAD_REQUEST, "only purpose=batch is supported");
    }
    // BLZ-6: refresh retention first so the quota check counts live data,
    // then enforce the configured store quotas on the incoming bytes.
    sweep_expired(&state.dirs, state.config.batch.retention_secs, now_secs());
    let owner = owner_name(key_ext.as_ref());
    if let Some((status, message)) = store_quota_refusal(
        &state.dirs,
        &state.config.batch,
        owner.as_deref(),
        file.data.len() as u64,
    ) {
        return err(status, &message);
    }
    let id = short_id("file");
    let dir = files_dir(&state.dirs);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("create batch dir: {e}"),
        );
    }
    let path = dir.join(format!("{id}.jsonl"));
    if let Err(e) = std::fs::write(&path, &file.data) {
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("write file: {e}"),
        );
    }
    let filename = file
        .filename
        .clone()
        .unwrap_or_else(|| "batch.jsonl".into());
    let obj = json!({
        "id": id,
        "object": "file",
        "bytes": file.data.len(),
        "created_at": now_secs(),
        "filename": filename,
        "purpose": purpose.unwrap_or_else(|| "batch".into()),
        "_owner": owner_name(key_ext.as_ref()),
    });
    if let Err(e) = write_json(&dir.join(format!("{id}.meta.json")), &obj) {
        tracing::error!(target: "blazar::batch", file = %id, error = %e, "file meta write failed");
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("write file meta: {e}"),
        );
    }
    (StatusCode::OK, axum::Json(strip_owner(obj))).into_response()
}

fn file_meta(dirs: &BlazarDirs, id: &str) -> Option<Value> {
    read_json(&files_dir(dirs).join(format!("{id}.meta.json")))
}

/// `GET /v1/files/{id}` — file metadata.
pub async fn get_file(
    State(state): State<std::sync::Arc<AppState>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return err(StatusCode::BAD_REQUEST, "invalid file id");
    }
    match file_meta(&state.dirs, &id) {
        // BLZ-1: a foreign file reads exactly like an unknown one.
        Some(v) if meta_allows(&v, key_ext.as_deref()) => {
            (StatusCode::OK, axum::Json(strip_owner(v))).into_response()
        }
        _ => err(StatusCode::NOT_FOUND, "unknown file id"),
    }
}

/// `GET /v1/files/{id}/content` — raw stored JSONL.
pub async fn get_file_content(
    State(state): State<std::sync::Arc<AppState>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return err(StatusCode::BAD_REQUEST, "invalid file id");
    }
    // BLZ-1: gate on the meta before serving the bytes — the content
    // route and the meta route must agree on visibility.
    match file_meta(&state.dirs, &id) {
        Some(v) if meta_allows(&v, key_ext.as_deref()) => {}
        _ => return err(StatusCode::NOT_FOUND, "unknown file id"),
    }
    match std::fs::read(files_dir(&state.dirs).join(format!("{id}.jsonl"))) {
        Ok(bytes) => (
            StatusCode::OK,
            [("content-type", "application/jsonl")],
            bytes,
        )
            .into_response(),
        Err(_) => err(StatusCode::NOT_FOUND, "unknown file id"),
    }
}

/// `POST /v1/batches` — validate the input file and spawn the worker.
pub async fn create_batch(
    State(state): State<std::sync::Arc<AppState>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let Ok(req) = serde_json::from_slice::<Value>(&body) else {
        return err(StatusCode::BAD_REQUEST, "body must be JSON");
    };
    let Some(input_file_id) = req["input_file_id"].as_str() else {
        return err(StatusCode::BAD_REQUEST, "input_file_id is required");
    };
    let endpoint = req["endpoint"].as_str().unwrap_or(SUPPORTED_ENDPOINT);
    if endpoint != SUPPORTED_ENDPOINT {
        return err(
            StatusCode::BAD_REQUEST,
            &format!("only endpoint='{SUPPORTED_ENDPOINT}' is supported"),
        );
    }
    let Some(meta) = file_meta(&state.dirs, input_file_id) else {
        return err(StatusCode::NOT_FOUND, "unknown input_file_id");
    };
    // BLZ-1: a batch may only consume an input file its key can read.
    if !meta_allows(&meta, key_ext.as_deref()) {
        return err(StatusCode::NOT_FOUND, "unknown input_file_id");
    }
    let input_path = files_dir(&state.dirs).join(format!("{input_file_id}.jsonl"));
    let Ok(lines) = std::fs::read_to_string(&input_path) else {
        return err(StatusCode::NOT_FOUND, "input file content missing");
    };
    let total = lines.lines().filter(|l| !l.trim().is_empty()).count();
    if total == 0 {
        return err(StatusCode::BAD_REQUEST, "input file has no request lines");
    }
    // BLZ-6: refresh retention, then enforce the per-key concurrent-batch
    // bound before spawning another worker.
    sweep_expired(&state.dirs, state.config.batch.retention_secs, now_secs());
    if let Some(owner) = owner_name(key_ext.as_ref())
        && state.config.batch.max_active_batches_per_key > 0
        && active_batch_count(&state.dirs, &owner)
            >= u64::from(state.config.batch.max_active_batches_per_key)
    {
        return err(
            StatusCode::BAD_REQUEST,
            &format!(
                "key '{owner}' is already running the configured maximum of {} concurrent batches",
                state.config.batch.max_active_batches_per_key
            ),
        );
    }

    let id = short_id("batch");
    let output_id = short_id("file");
    let created = now_secs();
    let batch = json!({
        "id": id,
        "object": "batch",
        "endpoint": endpoint,
        "status": "in_progress",
        "input_file_id": input_file_id,
        "input_file": strip_owner(meta),
        "_owner": owner_name(key_ext.as_ref()),
        "output_file_id": Value::Null,
        "created_at": created,
        "metadata": req.get("metadata").cloned().unwrap_or(json!({})),
        "request_counts": {"total": total, "completed": 0, "failed": 0},
    });
    if let Err(e) = write_json(&batch_meta(&state.dirs, &id), &batch) {
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("persist batch: {e}"),
        );
    }

    // Worker replays lines through the gateway's own loopback listener so
    // auth/quotas/sentinel all apply. Prefer the ACTUAL bound address
    // (recorded at serve time — config port may be 0 = ephemeral); fold
    // wildcard hosts to loopback-reachable.
    let (host, port) = state.http_addr.get().cloned().unwrap_or_else(|| {
        let h = if state.config.host == "0.0.0.0" || state.config.host == "::" {
            "127.0.0.1".to_string()
        } else {
            state.config.host.clone()
        };
        (h, state.config.port)
    });
    let auth_headers: Vec<(String, String)> = ["authorization", "x-api-key"]
        .iter()
        .filter_map(|name| {
            headers
                .get(*name)
                .and_then(|v| v.to_str().ok())
                .map(|v| ((*name).to_string(), v.to_string()))
        })
        .collect();

    let dirs = state.dirs.clone();
    let job = BatchJob {
        id,
        output_id,
        host,
        port,
        auth_headers,
        max_output_bytes: state.config.batch.max_output_bytes_per_batch,
    };
    tokio::spawn(async move { run_batch(dirs, job, lines).await });

    (StatusCode::OK, axum::Json(strip_owner(batch))).into_response()
}

/// POST one request body through the gateway's own listener, forwarding
/// the caller's auth headers so key scopes/quotas apply.
async fn replay_line(
    client: &reqwest::Client,
    base: &str,
    auth_headers: &[(String, String)],
    body: &Value,
) -> Result<(u16, String), reqwest::Error> {
    let mut rb = client
        .post(format!("{base}/v1/chat/completions"))
        .json(body);
    for (name, value) in auth_headers {
        rb = rb.header(name, value);
    }
    let r = rb.send().await?;
    let code = r.status().as_u16();
    let text = r.text().await?;
    Ok((code, text))
}

/// Build one failed-request row for the output JSONL.
fn error_line(custom_id: &str, message: &str) -> String {
    serde_json::to_string(&json!({
        "id": short_id("breq"),
        "custom_id": custom_id,
        "response": Value::Null,
        "error": {"message": message},
    }))
    .unwrap_or_default()
}

/// Everything the background worker needs to replay one batch through
/// the gateway's own listener.
struct BatchJob {
    id: String,
    output_id: String,
    host: String,
    port: u16,
    auth_headers: Vec<(String, String)>,
    /// BLZ-6: cumulative output cap (0 = unlimited).
    max_output_bytes: u64,
}

#[allow(clippy::too_many_lines)] // one cohesive replay loop
async fn run_batch(dirs: BlazarDirs, job: BatchJob, input: String) {
    let BatchJob {
        id,
        output_id,
        host,
        port,
        auth_headers,
        max_output_bytes,
    } = job;
    let job_ref = BatchJob {
        id: id.clone(),
        output_id: output_id.clone(),
        host: host.clone(),
        port,
        auth_headers: Vec::new(),
        max_output_bytes,
    };
    // Output lands in files/ as {output_id}.jsonl — same lookup path the
    // GET /v1/files/{id}/content route serves uploads from.
    let out_path = files_dir(&dirs).join(format!("{output_id}.jsonl"));
    if let Some(d) = out_path.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let client = crate::http_pool::tuned(reqwest::Client::builder())
        .timeout(std::time::Duration::from_mins(10))
        .build()
        .unwrap_or_default();
    let mut completed = 0u64;
    let mut failed = 0u64;
    // BLZ-6: cumulative output size, enforced inside write_row against
    // the per-batch cap.
    let mut written = 0u64;
    // F74: stream rows straight to disk instead of accumulating the whole
    // output in memory (a 48 MiB input can inflate far beyond that with
    // response bodies; a crash also loses everything buffered so far).
    let mut writer = match std::fs::File::create(&out_path) {
        Ok(f) => std::io::BufWriter::new(f),
        Err(e) => {
            tracing::error!(target: "blazar::batch", batch = %id, error = %e, "batch output file create failed");
            fail_batch(&dirs, &job_ref, &format!("output file create: {e}"));
            return;
        }
    };
    let base = format!("http://{host}:{port}");

    for line in input.lines().filter(|l| !l.trim().is_empty()) {
        // Cancel is signalled through the meta file; check between items.
        if let Some(m) = read_json(&batch_meta(&dirs, &id))
            && m["status"] == "cancelling"
        {
            finish_batch(
                &dirs,
                &job_ref,
                "cancelled",
                (completed, failed),
                &mut writer,
            );
            return;
        }
        let (custom_id, body) = match serde_json::from_str::<Value>(line) {
            Ok(v) => (
                v["custom_id"].as_str().unwrap_or("request").to_string(),
                v["body"].clone(),
            ),
            Err(e) => {
                failed += 1;
                if write_row(
                    &mut writer,
                    &error_line("request", &format!("unparseable input line: {e}")),
                    &dirs,
                    &job_ref,
                    completed,
                    failed,
                    &mut written,
                ) {
                    continue;
                }
                return;
            }
        };
        if body["model"].as_str().is_none() {
            failed += 1;
            if write_row(
                &mut writer,
                &error_line(&custom_id, "request body has no model"),
                &dirs,
                &job_ref,
                completed,
                failed,
                &mut written,
            ) {
                continue;
            }
            return;
        }
        let (ok, row) = replay_to_row(&client, &base, &auth_headers, &custom_id, &body).await;
        if ok {
            completed += 1;
        } else {
            failed += 1;
        }
        if !write_row(
            &mut writer,
            &row,
            &dirs,
            &job_ref,
            completed,
            failed,
            &mut written,
        ) {
            return;
        }
    }

    finish_batch(
        &dirs,
        &job_ref,
        "completed",
        (completed, failed),
        &mut writer,
    );
}

/// Replay one parsed request body through the loopback listener and build
/// its output JSONL row. Returns (`status_was_200`, row).
async fn replay_to_row(
    client: &reqwest::Client,
    base: &str,
    auth_headers: &[(String, String)],
    custom_id: &str,
    body: &Value,
) -> (bool, String) {
    let (status_code, resp_body) = match replay_line(client, base, auth_headers, body).await {
        Ok(r) => r,
        Err(e) => (
            0u16,
            serde_json::to_string(&json!({"error": {"message": format!("loopback call: {e}")}}))
                .unwrap_or_default(),
        ),
    };
    let ok = status_code == 200;
    let resp_json: Value = serde_json::from_str(&resp_body).unwrap_or(json!({"raw": resp_body}));
    let row = serde_json::to_string(&json!({
        "id": short_id("breq"),
        "custom_id": custom_id,
        "response": {"status_code": status_code, "body": resp_json},
        "error": Value::Null,
    }))
    .unwrap_or_default();
    (ok, row)
}

/// Append one JSONL row to the output writer; false = abort the batch
/// (output write failed — F76: disk errors must not masquerade as
/// success; or the row pushed cumulative output past the BLZ-6
/// per-batch byte cap).
fn write_row(
    writer: &mut std::io::BufWriter<std::fs::File>,
    row: &str,
    dirs: &BlazarDirs,
    job: &BatchJob,
    completed: u64,
    failed: u64,
    written: &mut u64,
) -> bool {
    if let Err(e) = writeln!(writer, "{row}") {
        tracing::error!(target: "blazar::batch", batch = %job.id, error = %e, "batch output write failed");
        fail_batch(dirs, job, &format!("output write: {e}"));
        return false;
    }
    *written = written.saturating_add(row.len() as u64 + 1);
    if job.max_output_bytes > 0 && *written > job.max_output_bytes {
        tracing::warn!(target: "blazar::batch", batch = %job.id, written, cap = job.max_output_bytes, "batch output exceeded the per-batch byte cap");
        fail_batch(
            dirs,
            job,
            "output exceeded the configured max_output_bytes_per_batch cap",
        );
        return false;
    }
    update_progress(dirs, &job.id, completed, failed);
    true
}

/// Mark a batch failed with the reason (output-file or flush errors).
fn fail_batch(dirs: &BlazarDirs, job: &BatchJob, why: &str) {
    let path = batch_meta(dirs, &job.id);
    if let Some(mut m) = read_json(&path) {
        m["status"] = json!("failed");
        m["error"] = json!({"message": why});
        m["request_counts"]["failed"] =
            json!(m["request_counts"]["failed"].as_u64().unwrap_or(0) + 1);
        m["finalized_at"] = json!(now_secs());
        if let Err(e) = write_json(&path, &m) {
            tracing::error!(target: "blazar::batch", batch = %job.id, error = %e, "failed-batch meta write failed");
        }
    }
}
fn update_progress(dirs: &BlazarDirs, id: &str, completed: u64, failed: u64) {
    let path = batch_meta(dirs, id);
    if let Some(mut m) = read_json(&path) {
        // F75: never clobber a cancel signal that landed between our read
        // and write — the worker observes it on its next loop iteration.
        if m["status"] == "cancelling" {
            return;
        }
        m["request_counts"]["completed"] = json!(completed);
        m["request_counts"]["failed"] = json!(failed);
        if let Err(e) = write_json(&path, &m) {
            tracing::warn!(target: "blazar::batch", batch = %id, error = %e, "progress meta write failed");
        }
    }
}

fn finish_batch(
    dirs: &BlazarDirs,
    job: &BatchJob,
    status: &str,
    counts: (u64, u64),
    writer: &mut std::io::BufWriter<std::fs::File>,
) {
    let (id, output_id) = (&job.id, &job.output_id);
    let (completed, failed) = counts;
    let out_path = files_dir(dirs).join(format!("{output_id}.jsonl"));
    // F76: a failed flush must not report success — downgrade the status
    // and surface the reason instead of swallowing it.
    let flush = writer.flush();
    let status = if flush.is_err() { "failed" } else { status };
    let bytes = std::fs::metadata(&out_path).map_or(0, |m| m.len());
    if let Err(e) = flush {
        tracing::error!(target: "blazar::batch", batch = %id, error = %e, "batch output flush failed");
    }
    // The output file gets a file-object meta entry too, so
    // GET /v1/files/{output_file_id} describes it like an upload.
    // (meta + content both live in files/ under {output_id}.)
    let out_meta = json!({
        "id": output_id,
        "object": "file",
        "bytes": bytes,
        "created_at": now_secs(),
        "filename": format!("{id}-output.jsonl"),
        "purpose": "batch_output",
    });
    if let Err(e) = write_json(
        &files_dir(dirs).join(format!("{output_id}.meta.json")),
        &out_meta,
    ) {
        tracing::warn!(target: "blazar::batch", batch = %id, error = %e, "output file meta write failed");
    }
    write_final_meta(dirs, id, output_id, status, completed, failed);
}

fn write_final_meta(
    dirs: &BlazarDirs,
    id: &str,
    output_id: &str,
    status: &str,
    completed: u64,
    failed: u64,
) {
    let path = batch_meta(dirs, id);
    if let Some(mut m) = read_json(&path) {
        m["status"] = json!(status);
        if status == "failed" {
            m["error"] = json!({"message": "batch output write failed"});
        }
        m["output_file_id"] = json!(output_id);
        m["request_counts"]["completed"] = json!(completed);
        m["request_counts"]["failed"] = json!(failed);
        m["finalized_at"] = json!(now_secs());
        if let Err(e) = write_json(&path, &m) {
            tracing::error!(target: "blazar::batch", batch = %id, error = %e, "final batch meta write failed");
        }
    }
    tracing::info!(target: "blazar::batch", batch = %id, status, completed, failed, "batch finished");
}

/// `GET /v1/batches/{id}` — live batch object.
pub async fn get_batch(
    State(state): State<std::sync::Arc<AppState>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return err(StatusCode::BAD_REQUEST, "invalid batch id");
    }
    match read_json(&batch_meta(&state.dirs, &id)) {
        // BLZ-1: a foreign batch reads exactly like an unknown one.
        Some(v) if meta_allows(&v, key_ext.as_deref()) => {
            (StatusCode::OK, axum::Json(strip_owner(v))).into_response()
        }
        _ => err(StatusCode::NOT_FOUND, "unknown batch id"),
    }
}

/// `GET /v1/batches` — list all batches (newest first).
pub async fn list_batches(
    State(state): State<std::sync::Arc<AppState>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
) -> Response {
    let mut rows: Vec<Value> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(batches_dir(&state.dirs)) {
        for e in entries.flatten() {
            let name = e.file_name();
            if let Some(name) = name.to_str()
                && std::path::Path::new(name)
                    .extension()
                    .is_some_and(|e| e == "json")
                && let Some(v) = read_json(&e.path())
                // BLZ-1: data-layer filter — only this key's batches (and
                // legacy null-owner rows) reach the listing.
                && meta_allows(&v, key_ext.as_deref())
            {
                rows.push(strip_owner(v));
            }
        }
    }
    rows.sort_by_key(|v| std::cmp::Reverse(v["created_at"].as_u64().unwrap_or(0)));
    (
        StatusCode::OK,
        axum::Json(json!({"object": "list", "data": rows})),
    )
        .into_response()
}

/// `POST /v1/batches/{id}/cancel` — flags the worker to stop; final state
/// lands asynchronously (`cancelled` once the current item finishes).
pub async fn cancel_batch(
    State(state): State<std::sync::Arc<AppState>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    // F77: same charset contract as get_batch — an unvalidated id would
    // traverse (`../`) into arbitrary JSON files for read+rewrite.
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return err(StatusCode::BAD_REQUEST, "invalid batch id");
    }
    let path = batch_meta(&state.dirs, &id);
    match read_json(&path) {
        // BLZ-1: cancelling another key's batch reads as unknown.
        Some(mut m) if meta_allows(&m, key_ext.as_deref()) => {
            let status = m["status"].as_str().unwrap_or("").to_string();
            if matches!(status.as_str(), "completed" | "cancelled" | "failed") {
                // "failed" included: the worker is gone, so a cancel flag
                // would wedge the status at "cancelling" forever.
                return err(StatusCode::BAD_REQUEST, "batch already finished");
            }
            m["status"] = json!("cancelling");
            if let Err(e) = write_json(&path, &m) {
                return err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("persist cancel: {e}"),
                );
            }
            (StatusCode::OK, axum::Json(strip_owner(m))).into_response()
        }
        _ => err(StatusCode::NOT_FOUND, "unknown batch id"),
    }
}

// ---------- Anthropic Message Batches ----------
//
// Anthropic's batch dialect embeds the requests inline (`requests:
// [{custom_id, params}]`) instead of referencing an uploaded file, reports
// five-way request counts, stamps RFC3339 times, and streams results as a
// `.jsonl` of `{custom_id, result}` rows. Same execution model as the
// OpenAI lane: a worker replays each request through the gateway's own
// loopback `/v1/messages` listener so auth/quotas/sentinel all apply.

/// Anthropic caps a batch at 100k requests (their documented maxItems).
const ANTHROPIC_MAX_REQUESTS: usize = 100_000;
/// Results are retained server-side for 24h upstream; the same retention
/// window is advertised here.
const ANTHROPIC_RETENTION_SECS: u64 = 24 * 60 * 60;

/// `custom_id` charset per upstream: 1..=64 chars of `[A-Za-z0-9_-]`.
fn valid_anthropic_custom_id(id: &str) -> bool {
    (1..=64).contains(&id.len())
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Internal meta → the public `MessageBatch` object. Internal fields carry
/// `_secs` unix ints; the wire shape is ISO-3339 strings.
fn anthropic_public_shape(m: &Value) -> Value {
    let created = m["created_at_secs"].as_u64().unwrap_or(0);
    let ended = m["ended_at_secs"].as_u64();
    let cancel_requested = m["cancel_requested"].as_bool().unwrap_or(false);
    let processing_status = if m["internal_status"] == "ended" {
        "ended"
    } else if cancel_requested {
        "canceling"
    } else {
        "in_progress"
    };
    json!({
        "id": m["id"],
        "type": "message_batch",
        "archived_at": m["archived_at_secs"].as_u64().map_or(Value::Null, |s| Value::String(crate::ollama::iso(i64::try_from(s).unwrap_or(0)))),
        "cancel_initiated_at": m["cancel_initiated_at_secs"].as_u64().map_or(Value::Null, |s| Value::String(crate::ollama::iso(i64::try_from(s).unwrap_or(0)))),
        "created_at": Value::String(crate::ollama::iso(i64::try_from(created).unwrap_or(0))),
        "ended_at": ended.map_or(Value::Null, |s| Value::String(crate::ollama::iso(i64::try_from(s).unwrap_or(0)))),
        "expires_at": Value::String(crate::ollama::iso(i64::try_from(created + ANTHROPIC_RETENTION_SECS).unwrap_or(0))),
        "processing_status": processing_status,
        "request_counts": m["request_counts"],
        "results_url": format!("/v1/messages/batches/{}/results", m["id"].as_str().unwrap_or_default()),
    })
}

/// One output row: a successful request carries the translated
/// `OpenAI`-shape message body of the `/v1/messages` response.
fn anthropic_row_success(custom_id: &str, message: &Value) -> String {
    serde_json::to_string(&json!({
        "custom_id": custom_id,
        "result": {"type": "message", "message": message},
    }))
    .unwrap_or_default()
}

/// One output row: a failed request carries the lane's error shape.
fn anthropic_row_errored(custom_id: &str, error_type: &str, message: &str) -> String {
    serde_json::to_string(&json!({
        "custom_id": custom_id,
        "result": {"type": "errored", "error": {"type": error_type, "message": message}},
    }))
    .unwrap_or_default()
}

/// One output row: cancellation reached this request before it ran.
fn anthropic_row_canceled(custom_id: &str) -> String {
    serde_json::to_string(&json!({
        "custom_id": custom_id,
        "result": {"type": "canceled"},
    }))
    .unwrap_or_default()
}

/// POST one `params` body through the gateway's own `/v1/messages`.
async fn anthropic_replay_line(
    client: &reqwest::Client,
    base: &str,
    auth_headers: &[(String, String)],
    body: &Value,
) -> Result<(u16, String), reqwest::Error> {
    let mut rb = client.post(format!("{base}/v1/messages")).json(body);
    for (name, value) in auth_headers {
        rb = rb.header(name, value);
    }
    let r = rb.send().await?;
    let code = r.status().as_u16();
    let text = r.text().await?;
    Ok((code, text))
}

/// Shared worker state for the anthropic lane (mirrors `BatchJob`).
struct AnthropicBatchJob {
    id: String,
    output_id: String,
    host: String,
    port: u16,
    auth_headers: Vec<(String, String)>,
    /// BLZ-6: cumulative output cap (0 = unlimited).
    max_output_bytes: u64,
}

/// Progress update: mutates only the counts so a concurrent cancel flag
/// survives (same `F75` discipline as the `OpenAI` lane).
fn update_anthropic_progress(dirs: &BlazarDirs, id: &str, counts: &Value) {
    let path = batch_meta(dirs, id);
    if let Some(mut m) = read_json(&path) {
        m["request_counts"] = counts.clone();
        if let Err(e) = write_json(&path, &m) {
            tracing::warn!(target: "blazar::batch", batch = %id, error = %e, "anthropic progress meta write failed");
        }
    }
}

/// Terminal write: flush output, record the output-file meta, and stamp
/// `ended` (with `cancel_initiated_at` when the run was cancelled).
fn finish_anthropic_batch(
    dirs: &BlazarDirs,
    job: &AnthropicBatchJob,
    counts: &Value,
    writer: &mut std::io::BufWriter<std::fs::File>,
    was_cancelled: bool,
) {
    let out_path = files_dir(dirs).join(format!("{}.jsonl", job.output_id));
    // F76 discipline: a failed flush must not masquerade as success.
    let flush = writer.flush();
    let bytes = std::fs::metadata(&out_path).map_or(0, |md| md.len());
    if let Err(e) = &flush {
        tracing::error!(target: "blazar::batch", batch = %job.id, error = %e, "anthropic batch output flush failed");
    }
    let out_meta = json!({
        "id": job.output_id,
        "object": "file",
        "bytes": bytes,
        "created_at": now_secs(),
        "filename": format!("{}-results.jsonl", job.id),
        "purpose": "batch_output",
    });
    if let Err(e) = write_json(
        &files_dir(dirs).join(format!("{}.meta.json", job.output_id)),
        &out_meta,
    ) {
        tracing::warn!(target: "blazar::batch", batch = %job.id, error = %e, "anthropic output file meta write failed");
    }
    let path = batch_meta(dirs, &job.id);
    if let Some(mut m) = read_json(&path) {
        m["internal_status"] = json!("ended");
        m["ended_at_secs"] = json!(now_secs());
        m["output_file_id"] = json!(job.output_id);
        m["request_counts"] = counts.clone();
        if was_cancelled && m["cancel_initiated_at_secs"].is_null() {
            m["cancel_initiated_at_secs"] = json!(now_secs());
        }
        if let Err(e) = write_json(&path, &m) {
            tracing::error!(target: "blazar::batch", batch = %job.id, error = %e, "anthropic final meta write failed");
        }
    }
    tracing::info!(target: "blazar::batch", batch = %job.id, cancelled = was_cancelled, "anthropic batch finished");
}

#[allow(clippy::too_many_lines)] // one cohesive replay loop
async fn run_anthropic_batch(dirs: BlazarDirs, job: AnthropicBatchJob, input: String) {
    let out_path = files_dir(&dirs).join(format!("{}.jsonl", job.output_id));
    if let Some(d) = out_path.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let client = crate::http_pool::tuned(reqwest::Client::builder())
        .timeout(std::time::Duration::from_mins(10))
        .build()
        .unwrap_or_default();
    let mut writer = match std::fs::File::create(&out_path) {
        Ok(f) => std::io::BufWriter::new(f),
        Err(e) => {
            tracing::error!(target: "blazar::batch", batch = %job.id, error = %e, "anthropic output file create failed");
            return;
        }
    };
    let base = format!("http://{}:{}", job.host, job.port);
    let mut succeeded = 0u64;
    let mut errored = 0u64;
    let mut canceled = 0u64;
    // BLZ-6: cumulative output size against the per-batch cap.
    let mut written = 0u64;
    let mut remaining: u64 = input
        .lines()
        .filter(|l| !l.trim().is_empty())
        .count()
        .try_into()
        .unwrap_or(0);

    let finalize = |succeeded: u64,
                    errored: u64,
                    canceled: u64,
                    was_cancelled: bool,
                    writer: &mut std::io::BufWriter<std::fs::File>| {
        let counts = json!({
            "processing": 0, "succeeded": succeeded, "errored": errored,
            "canceled": canceled, "expired": 0,
        });
        finish_anthropic_batch(&dirs, &job, &counts, writer, was_cancelled);
    };

    for line in input.lines().filter(|l| !l.trim().is_empty()) {
        remaining = remaining.saturating_sub(1);
        // Cancel is signalled through the meta file; check between items.
        if let Some(m) = read_json(&batch_meta(&dirs, &job.id))
            && m["cancel_requested"].as_bool().unwrap_or(false)
        {
            let row = anthropic_row_canceled(&serde_json::from_str::<Value>(line).map_or_else(
                |_| "request".into(),
                |v| v["custom_id"].as_str().unwrap_or("request").to_string(),
            ));
            canceled += 1;
            written = written.saturating_add(row.len() as u64 + 1);
            if writeln!(writer, "{row}").is_err() {
                break;
            }
            continue;
        }
        let (custom_id, params) = match serde_json::from_str::<Value>(line) {
            Ok(v) => (
                v["custom_id"].as_str().unwrap_or("request").to_string(),
                v["params"].clone(),
            ),
            Err(_) => ("request".to_string(), Value::Null),
        };
        let row = match anthropic_replay_line(&client, &base, &job.auth_headers, &params).await {
            Ok((200, body)) => {
                succeeded += 1;
                let msg = serde_json::from_str(&body).unwrap_or(json!({"raw": body}));
                anthropic_row_success(&custom_id, &msg)
            }
            Ok((_, body)) => {
                errored += 1;
                let (etype, emsg) = serde_json::from_str::<Value>(&body).map_or_else(
                    |_| ("api_error".into(), body.trim().to_string()),
                    |v| {
                        (
                            v.pointer("/error/type")
                                .and_then(Value::as_str)
                                .unwrap_or("api_error")
                                .to_string(),
                            v.pointer("/error/message")
                                .and_then(Value::as_str)
                                .unwrap_or(body.trim())
                                .to_string(),
                        )
                    },
                );
                anthropic_row_errored(&custom_id, &etype, &emsg)
            }
            Err(e) => {
                errored += 1;
                anthropic_row_errored(&custom_id, "api_error", &format!("loopback call: {e}"))
            }
        };
        written = written.saturating_add(row.len() as u64 + 1);
        if job.max_output_bytes > 0 && written > job.max_output_bytes {
            tracing::warn!(target: "blazar::batch", batch = %job.id, written, cap = job.max_output_bytes, "anthropic batch output exceeded the per-batch byte cap");
            let truncation = anthropic_row_errored(
                "batch",
                "api_error",
                "batch output exceeded the configured max_output_bytes_per_batch cap; results truncated",
            );
            let _ = writeln!(writer, "{truncation}");
            errored += 1;
            break;
        }
        if writeln!(writer, "{row}").is_err() {
            tracing::error!(target: "blazar::batch", batch = %job.id, "anthropic output write failed");
            break;
        }
        update_anthropic_progress(
            &dirs,
            &job.id,
            &json!({
                "processing": remaining, "succeeded": succeeded,
                "errored": errored, "canceled": canceled, "expired": 0,
            }),
        );
    }
    let was_cancelled = read_json(&batch_meta(&dirs, &job.id))
        .is_some_and(|m| m["cancel_requested"].as_bool().unwrap_or(false));
    finalize(succeeded, errored, canceled, was_cancelled, &mut writer);
}

/// `POST /v1/messages/batches` — validate inline requests, persist them
/// as an internal JSONL, spawn the worker.
#[allow(clippy::too_many_lines)] // one cohesive validation + spawn path
pub async fn anthropic_batches_create(
    State(state): State<std::sync::Arc<AppState>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let Ok(req) = serde_json::from_slice::<Value>(&body) else {
        return err(StatusCode::BAD_REQUEST, "body must be JSON");
    };
    let Some(requests) = req["requests"].as_array() else {
        return err(
            StatusCode::BAD_REQUEST,
            "requests must be an array of {custom_id, params} objects",
        );
    };
    if requests.is_empty() {
        return err(StatusCode::BAD_REQUEST, "requests must not be empty");
    }
    if requests.len() > ANTHROPIC_MAX_REQUESTS {
        return err(
            StatusCode::BAD_REQUEST,
            &format!("requests exceed the {ANTHROPIC_MAX_REQUESTS}-item batch cap"),
        );
    }
    let mut seen = std::collections::HashSet::new();
    for r in requests {
        let Some(custom_id) = r["custom_id"].as_str() else {
            return err(
                StatusCode::BAD_REQUEST,
                "every request needs a custom_id string",
            );
        };
        if !valid_anthropic_custom_id(custom_id) {
            return err(
                StatusCode::BAD_REQUEST,
                "custom_id must be 1..=64 chars of [a-zA-Z0-9_-]",
            );
        }
        if !seen.insert(custom_id.to_string()) {
            return err(
                StatusCode::BAD_REQUEST,
                &format!("custom_id '{custom_id}' appears more than once (must be unique)"),
            );
        }
        let params = &r["params"];
        if !params.is_object() {
            return err(
                StatusCode::BAD_REQUEST,
                "every request needs a params object",
            );
        }
        if params["model"].as_str().is_none() {
            return err(
                StatusCode::BAD_REQUEST,
                &format!("params for '{custom_id}' is missing model"),
            );
        }
        if !params["messages"].is_array() {
            return err(
                StatusCode::BAD_REQUEST,
                &format!("params for '{custom_id}' is missing messages"),
            );
        }
    }

    // Internal JSONL: one {custom_id, params} line per request, stored in
    // the same files/ pool the OpenAI lane uses. BLZ-6: retention refresh
    // + store quotas + the per-key concurrent-batch bound apply to this
    // dialect too — it writes the same disk.
    sweep_expired(&state.dirs, state.config.batch.retention_secs, now_secs());
    let owner = owner_name(key_ext.as_ref());
    if let Some(owner) = &owner
        && state.config.batch.max_active_batches_per_key > 0
        && active_batch_count(&state.dirs, owner)
            >= u64::from(state.config.batch.max_active_batches_per_key)
    {
        return err(
            StatusCode::BAD_REQUEST,
            &format!(
                "key '{owner}' is already running the configured maximum of {} concurrent batches",
                state.config.batch.max_active_batches_per_key
            ),
        );
    }
    let lines: Vec<String> = requests
        .iter()
        .map(|r| serde_json::to_string(r).unwrap_or_default())
        .collect();
    let input_bytes: u64 = lines.iter().map(|l| l.len() as u64 + 1).sum();
    if let Some((status, message)) = store_quota_refusal(
        &state.dirs,
        &state.config.batch,
        owner.as_deref(),
        input_bytes,
    ) {
        return err(status, &message);
    }
    let input_id = short_id("file");
    let input_path = files_dir(&state.dirs).join(format!("{input_id}.jsonl"));
    if let Err(e) = std::fs::create_dir_all(files_dir(&state.dirs)) {
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("create batch dir: {e}"),
        );
    }
    if let Err(e) = std::fs::write(&input_path, lines.join("\n") + "\n") {
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("write requests file: {e}"),
        );
    }

    let id = short_id("msgbatch");
    let output_id = short_id("file");
    let total = requests.len();
    let meta = json!({
        "dialect": "anthropic",
        "id": id,
        "internal_status": "in_progress",
        "cancel_requested": false,
        "_owner": owner_name(key_ext.as_ref()),
        "created_at_secs": now_secs(),
        "ended_at_secs": Value::Null,
        "cancel_initiated_at_secs": Value::Null,
        "archived_at_secs": Value::Null,
        "input_file_id": input_id,
        "output_file_id": Value::Null,
        "request_counts": {
            "processing": total, "succeeded": 0, "errored": 0,
            "canceled": 0, "expired": 0,
        },
    });
    if let Err(e) = write_json(&batch_meta(&state.dirs, &id), &meta) {
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("persist batch: {e}"),
        );
    }

    let (host, port) = state.http_addr.get().cloned().unwrap_or_else(|| {
        let h = if state.config.host == "0.0.0.0" || state.config.host == "::" {
            "127.0.0.1".to_string()
        } else {
            state.config.host.clone()
        };
        (h, state.config.port)
    });
    let auth_headers: Vec<(String, String)> = ["authorization", "x-api-key"]
        .iter()
        .filter_map(|name| {
            headers
                .get(*name)
                .and_then(|v| v.to_str().ok())
                .map(|v| ((*name).to_string(), v.to_string()))
        })
        .collect();
    let dirs = state.dirs.clone();
    let job = AnthropicBatchJob {
        id: id.clone(),
        output_id,
        host,
        port,
        auth_headers,
        max_output_bytes: state.config.batch.max_output_bytes_per_batch,
    };
    tokio::spawn(async move { run_anthropic_batch(dirs, job, lines.join("\n")).await });

    (
        StatusCode::OK,
        axum::Json(anthropic_public_shape(
            &read_json(&batch_meta(&state.dirs, &id)).unwrap_or(meta),
        )),
    )
        .into_response()
}

/// Load an anthropic-dialect batch meta or produce the error response.
#[allow(clippy::result_large_err)] // Response is the handlers' currency here
fn anthropic_meta(
    state: &std::sync::Arc<AppState>,
    id: &str,
    key: Option<&crate::keys::KeyCtx>,
) -> Result<Value, Response> {
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(err(StatusCode::BAD_REQUEST, "invalid message batch id"));
    }
    match read_json(&batch_meta(&state.dirs, id)) {
        // BLZ-1: every anthropic route funnels through here — gate once,
        // and a foreign batch stays indistinguishable from an unknown id
        // (an OpenAI-dialect batch under these routes reads the same way).
        Some(m) if m["dialect"] == "anthropic" && meta_allows(&m, key) => Ok(m),
        _ => Err(err(
            StatusCode::NOT_FOUND,
            "unknown message batch id (message batches use the /v1/messages/batches routes)",
        )),
    }
}

/// `GET /v1/messages/batches/{id}` — idempotent poll target.
pub async fn anthropic_batches_get(
    State(state): State<std::sync::Arc<AppState>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    match anthropic_meta(&state, &id, key_ext.as_deref()) {
        Ok(m) => (StatusCode::OK, axum::Json(anthropic_public_shape(&m))).into_response(),
        Err(resp) => resp,
    }
}

/// `GET /v1/messages/batches` — newest first.
pub async fn anthropic_batches_list(
    State(state): State<std::sync::Arc<AppState>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
) -> Response {
    let mut rows: Vec<Value> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(batches_dir(&state.dirs)) {
        for e in entries.flatten() {
            if let Some(name) = e.file_name().to_str()
                && std::path::Path::new(name)
                    .extension()
                    .is_some_and(|x| x == "json")
                && let Some(v) = read_json(&e.path())
                && v["dialect"] == "anthropic"
                // BLZ-1: data-layer filter, same visibility law as the
                // OpenAI listing.
                && meta_allows(&v, key_ext.as_deref())
            {
                rows.push(anthropic_public_shape(&v));
            }
        }
    }
    // UTC ISO-8601 stamps sort correctly as plain strings — newest first.
    rows.sort_by(|a, b| b["created_at"].as_str().cmp(&a["created_at"].as_str()));
    let first_id = rows
        .first()
        .and_then(|v| v["id"].as_str())
        .map(String::from);
    let last_id = rows.last().and_then(|v| v["id"].as_str()).map(String::from);
    (
        StatusCode::OK,
        axum::Json(json!({
            "data": rows,
            "has_more": false,
            "first_id": first_id,
            "last_id": last_id,
        })),
    )
        .into_response()
}

/// `POST /v1/messages/batches/{id}/cancel` — flags the worker; final state
/// lands asynchronously (`ended` once the in-flight request finishes).
pub async fn anthropic_batches_cancel(
    State(state): State<std::sync::Arc<AppState>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    let mut m = match anthropic_meta(&state, &id, key_ext.as_deref()) {
        Ok(m) => m,
        Err(resp) => return resp,
    };
    if m["internal_status"] == "ended" {
        return err(StatusCode::BAD_REQUEST, "batch already ended");
    }
    if m["cancel_requested"].as_bool().unwrap_or(false) {
        return err(StatusCode::BAD_REQUEST, "batch cancel already in progress");
    }
    m["cancel_requested"] = json!(true);
    m["cancel_initiated_at_secs"] = json!(now_secs());
    let path = batch_meta(&state.dirs, &id);
    if let Err(e) = write_json(&path, &m) {
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("persist cancel: {e}"),
        );
    }
    (StatusCode::OK, axum::Json(anthropic_public_shape(&m))).into_response()
}

/// `DELETE /v1/messages/batches/{id}` — archives the batch (results stay
/// readable through `expires_at`; upstream semantics, documented honestly).
pub async fn anthropic_batches_delete(
    State(state): State<std::sync::Arc<AppState>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    let mut m = match anthropic_meta(&state, &id, key_ext.as_deref()) {
        Ok(m) => m,
        Err(resp) => return resp,
    };
    m["archived_at_secs"] = json!(now_secs());
    if let Err(e) = write_json(&batch_meta(&state.dirs, &id), &m) {
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("persist archive: {e}"),
        );
    }
    (StatusCode::OK, axum::Json(anthropic_public_shape(&m))).into_response()
}

/// `GET /v1/messages/batches/{id}/results` — streams the `.jsonl` result
/// file once the batch has ended.
pub async fn anthropic_batches_results(
    State(state): State<std::sync::Arc<AppState>>,
    key_ext: Option<Extension<crate::keys::KeyCtx>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    let m = match anthropic_meta(&state, &id, key_ext.as_deref()) {
        Ok(m) => m,
        Err(resp) => return resp,
    };
    if m["internal_status"] != "ended" {
        return err(
            StatusCode::BAD_REQUEST,
            "results are streamed only after the batch ends — poll \
             GET /v1/messages/batches/{id} until processing_status is \"ended\"",
        );
    }
    let Some(output_id) = m["output_file_id"].as_str() else {
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "ended batch has no output file (ledger corruption?)",
        );
    };
    match std::fs::read(files_dir(&state.dirs).join(format!("{output_id}.jsonl"))) {
        Ok(bytes) => (
            StatusCode::OK,
            [("content-type", "application/jsonl")],
            bytes,
        )
            .into_response(),
        Err(_) => err(StatusCode::NOT_FOUND, "results file missing"),
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn unit__short_id__prefix_and_uniqueness() {
        let a = short_id("file");
        let b = short_id("batch");
        assert!(a.starts_with("file-"));
        assert!(b.starts_with("batch-"));
        assert_ne!(a, short_id("file"));
        // BLZ-6: ids come from the OS CSPRNG — lowercase hex, and a run of
        // a thousand must never collide (predictable ids are not secrets).
        for id in [&a, &b] {
            let tail = &id[id.find('-').unwrap() + 1..];
            assert_eq!(tail.len(), 12, "id: {id}");
            assert!(
                tail.bytes()
                    .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
            );
        }
        let mut seen = std::collections::HashSet::new();
        for _ in 0..1000 {
            assert!(seen.insert(short_id("file")));
        }
    }

    #[test]
    fn unit__batch_paths__nested_under_data() {
        let dirs = BlazarDirs {
            config_dir: PathBuf::from("/tmp/c"),
            data_dir: PathBuf::from("/tmp/d"),
        };
        assert_eq!(files_dir(&dirs), PathBuf::from("/tmp/d/batch/files"));
        assert_eq!(
            batch_meta(&dirs, "batch-1"),
            PathBuf::from("/tmp/d/batch/batches/batch-1.json")
        );
    }

    #[test]
    fn unit__anthropic_custom_id__charset_pattern() {
        assert!(valid_anthropic_custom_id("a"));
        assert!(valid_anthropic_custom_id("task-1"));
        assert!(valid_anthropic_custom_id("Job_42"));
        assert!(valid_anthropic_custom_id(&"x".repeat(64)));
        assert!(!valid_anthropic_custom_id(""), "empty");
        assert!(!valid_anthropic_custom_id(&"x".repeat(65)), "65 chars");
        assert!(!valid_anthropic_custom_id("no spaces"), "space");
        assert!(!valid_anthropic_custom_id("dot.id"), "dot");
        assert!(!valid_anthropic_custom_id("id:1"), "colon");
    }

    #[test]
    fn unit__anthropic_public_shape__statuses_and_counts_keys() {
        let base = json!({
            "dialect": "anthropic",
            "id": "msgbatch_t",
            "internal_status": "in_progress",
            "cancel_requested": false,
            "created_at_secs": 1_729_032_000_u64,
            "ended_at_secs": Value::Null,
            "cancel_initiated_at_secs": Value::Null,
            "archived_at_secs": Value::Null,
            "request_counts": {
                "processing": 2, "succeeded": 0, "errored": 0,
                "canceled": 0, "expired": 0,
            },
        });
        let v = anthropic_public_shape(&base);
        assert_eq!(v["type"], "message_batch");
        assert_eq!(v["processing_status"], "in_progress");
        assert_eq!(v["created_at"], "2024-10-15T22:40:00Z");
        assert_eq!(v["expires_at"], "2024-10-16T22:40:00Z", "24h retention");
        assert_eq!(v["results_url"], "/v1/messages/batches/msgbatch_t/results");
        assert!(v["ended_at"].is_null());
        assert!(v["archived_at"].is_null());
        let mut keys: Vec<&str> = v["request_counts"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec!["canceled", "errored", "expired", "processing", "succeeded"]
        );

        let mut canceling = base.clone();
        canceling["cancel_requested"] = json!(true);
        assert_eq!(
            anthropic_public_shape(&canceling)["processing_status"],
            "canceling"
        );

        let mut ended = base.clone();
        ended["internal_status"] = json!("ended");
        ended["ended_at_secs"] = json!(1_729_032_100_u64);
        assert_eq!(anthropic_public_shape(&ended)["processing_status"], "ended");
        assert_eq!(
            anthropic_public_shape(&ended)["ended_at"],
            "2024-10-15T22:41:40Z"
        );
    }

    #[test]
    fn unit__anthropic_rows__success_errored_canceled() {
        let ok = anthropic_row_success("a-1", &json!({"id": "msg_1", "role": "assistant"}));
        let v: Value = serde_json::from_str(&ok).unwrap();
        assert_eq!(v["custom_id"], "a-1");
        assert_eq!(v["result"]["type"], "message");
        assert_eq!(v["result"]["message"]["role"], "assistant");

        let bad = anthropic_row_errored("a-2", "invalid_request_error", "max_tokens is required");
        let v: Value = serde_json::from_str(&bad).unwrap();
        assert_eq!(v["result"]["type"], "errored");
        assert_eq!(v["result"]["error"]["type"], "invalid_request_error");
        assert_eq!(v["result"]["error"]["message"], "max_tokens is required");

        let cxl = anthropic_row_canceled("a-3");
        let v: Value = serde_json::from_str(&cxl).unwrap();
        assert_eq!(v["result"]["type"], "canceled");
    }

    /// Worker-level cancel pin: a cancel flag in the meta file turns every
    /// remaining request into a `canceled` row — deterministic, zero network
    /// (the worker checks the flag BEFORE its first replay).
    #[tokio::test]
    async fn unit__anthropic_worker__cancel_flag_cancels_remaining_rows() {
        let tmp = std::env::temp_dir().join(format!("blazar-abatch-{}", std::process::id()));
        let dirs = BlazarDirs {
            config_dir: tmp.clone(),
            data_dir: tmp.clone(),
        };
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(files_dir(&dirs)).unwrap();
        std::fs::create_dir_all(batches_dir(&dirs)).unwrap();

        let input = "{\"custom_id\":\"r-1\",\"params\":{\"model\":\"m\"}}\n\
                     {\"custom_id\":\"r-2\",\"params\":{\"model\":\"m\"}}\n\
                     {\"custom_id\":\"r-3\",\"params\":{\"model\":\"m\"}}\n";
        let meta = json!({
            "dialect": "anthropic",
            "id": "msgbatch_c",
            "internal_status": "in_progress",
            "cancel_requested": true,
            "created_at_secs": 1_u64,
            "ended_at_secs": Value::Null,
            "cancel_initiated_at_secs": 1_u64,
            "archived_at_secs": Value::Null,
            "input_file_id": "f",
            "output_file_id": Value::Null,
            "request_counts": {
                "processing": 3, "succeeded": 0, "errored": 0,
                "canceled": 0, "expired": 0,
            },
        });
        let meta_path = batch_meta(&dirs, "msgbatch_c");
        std::fs::write(&meta_path, serde_json::to_vec_pretty(&meta).unwrap()).unwrap();

        let job = AnthropicBatchJob {
            id: "msgbatch_c".into(),
            output_id: "out".into(),
            host: "127.0.0.1".into(),
            // Never contacted: the cancel flag precedes the first replay.
            port: 1,
            auth_headers: Vec::new(),
            max_output_bytes: 0,
        };
        run_anthropic_batch(dirs.clone(), job, input.to_string()).await;

        let m: Value = serde_json::from_slice(&std::fs::read(&meta_path).unwrap()).unwrap();
        assert_eq!(m["internal_status"], "ended");
        assert_eq!(m["request_counts"]["canceled"], 3, "meta: {m:?}");
        assert_eq!(m["request_counts"]["processing"], 0);
        assert!(m["ended_at_secs"].as_u64().is_some());
        assert!(m["cancel_initiated_at_secs"].as_u64().is_some());
        assert_eq!(m["output_file_id"], "out");
        let out = std::fs::read_to_string(files_dir(&dirs).join("out.jsonl")).unwrap();
        let rows: Vec<Value> = out
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(rows.len(), 3, "all rows written: {out}");
        for (i, r) in rows.iter().enumerate() {
            assert_eq!(r["custom_id"], format!("r-{}", i + 1), "row: {r:?}");
            assert_eq!(r["result"]["type"], "canceled", "row: {r:?}");
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    // ---------- BLZ-6 test fixtures ----------

    /// Scratch store with both directories created; caller cleans up.
    fn scratch_dirs(tag: &str) -> BlazarDirs {
        let tmp = std::env::temp_dir().join(format!("blazar-batch6-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let dirs = BlazarDirs {
            config_dir: tmp.clone(),
            data_dir: tmp,
        };
        std::fs::create_dir_all(files_dir(&dirs)).unwrap();
        std::fs::create_dir_all(batches_dir(&dirs)).unwrap();
        dirs
    }

    fn write_store_file(
        dirs: &BlazarDirs,
        id: &str,
        owner: Option<&str>,
        bytes: u64,
        created_at: u64,
    ) {
        std::fs::write(
            files_dir(dirs).join(format!("{id}.jsonl")),
            vec![b'x'; usize::try_from(bytes).unwrap_or(usize::MAX)],
        )
        .unwrap();
        let meta = json!({
            "id": id, "object": "file", "bytes": bytes, "created_at": created_at,
            "filename": format!("{id}.jsonl"), "purpose": "batch", "_owner": owner,
        });
        std::fs::write(
            files_dir(dirs).join(format!("{id}.meta.json")),
            serde_json::to_vec_pretty(&meta).unwrap(),
        )
        .unwrap();
    }

    #[allow(clippy::too_many_arguments)] // flat fixture: every knob explicit
    fn write_openai_batch(
        dirs: &BlazarDirs,
        id: &str,
        status: &str,
        input: &str,
        output: Option<&str>,
        owner: &str,
        created_at: u64,
        finalized_at: Option<u64>,
    ) {
        let meta = json!({
            "id": id, "status": status, "input_file_id": input,
            "output_file_id": output, "created_at": created_at,
            "finalized_at": finalized_at, "_owner": owner,
        });
        std::fs::write(
            batch_meta(dirs, id),
            serde_json::to_vec_pretty(&meta).unwrap(),
        )
        .unwrap();
    }

    /// Retention sweep pins: removal requires TERMINAL status AND an aged
    /// stamp; in-flight batches and their referenced files survive; orphaned
    /// files age out on their own `created_at`; output files ride along with
    /// their expired batch.
    #[test]
    fn unit__sweep_expired__terminal_and_age_gated() {
        let dirs = scratch_dirs("sweep");
        // now = 10_000, retention = 1_000 → cutoff = 9_000.
        let now = 10_000_u64;

        // Terminal + old (finalized 8_000): swept, output rides along.
        write_store_file(&dirs, "file-gone", Some("alpha"), 10, 7_000);
        write_store_file(&dirs, "file-out", None, 10, 7_000);
        write_openai_batch(
            &dirs,
            "batch-old",
            "completed",
            "file-gone",
            Some("file-out"),
            "alpha",
            7_000,
            Some(8_000),
        );

        // In-flight + old: NEVER touched (binding contract).
        write_store_file(&dirs, "file-live", Some("alpha"), 10, 1_000);
        write_openai_batch(
            &dirs,
            "batch-live",
            "in_progress",
            "file-live",
            None,
            "alpha",
            1_000,
            None,
        );

        // Terminal but FRESH (finalized 9_500): kept.
        write_openai_batch(
            &dirs,
            "batch-fresh",
            "completed",
            "file-live2",
            None,
            "alpha",
            9_000,
            Some(9_500),
        );

        // Orphaned old file (no batch references it): aged out alone.
        write_store_file(&dirs, "file-orphan", None, 10, 8_000);
        // Orphaned FRESH file: kept.
        write_store_file(&dirs, "file-fresh", None, 10, 9_500);

        sweep_expired(&dirs, 1_000, now);

        let batches: Vec<String> = std::fs::read_dir(batches_dir(&dirs))
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            !batches.iter().any(|b| b.starts_with("batch-old")),
            "{batches:?}"
        );
        assert!(
            batches.iter().any(|b| b.starts_with("batch-live")),
            "{batches:?}"
        );
        assert!(
            batches.iter().any(|b| b.starts_with("batch-fresh")),
            "{batches:?}"
        );

        let files: Vec<String> = std::fs::read_dir(files_dir(&dirs))
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            !files.iter().any(|f| f.starts_with("file-out.")),
            "{files:?}"
        );
        assert!(
            !files.iter().any(|f| f.starts_with("file-orphan.")),
            "{files:?}"
        );
        assert!(
            files.iter().any(|f| f.starts_with("file-live.")),
            "{files:?}"
        );
        assert!(
            files.iter().any(|f| f.starts_with("file-fresh.")),
            "{files:?}"
        );
        // The expired batch's INPUT file was released by its removal and
        // its own created_at (7_000) is past cutoff → orphan pass removes it.
        assert!(
            !files.iter().any(|f| f.starts_with("file-gone.")),
            "{files:?}"
        );

        // retention 0 = disabled: nothing is ever removed.
        write_openai_batch(
            &dirs,
            "batch-old2",
            "expired",
            "file-fresh",
            None,
            "alpha",
            1_000,
            Some(1_000),
        );
        sweep_expired(&dirs, 0, now);
        assert!(batch_meta(&dirs, "batch-old2").exists());

        let _ = std::fs::remove_dir_all(files_dir(&dirs).parent().unwrap());
    }

    /// Quota pins: per-key caps key-scoped owners, the global cap catches
    /// everyone (including open-gateway uploads), zero quotas = unlimited.
    #[test]
    fn unit__store_quota_refusal__per_key_and_total() {
        let dirs = scratch_dirs("quota");
        write_store_file(&dirs, "f-alpha", Some("alpha"), 100, 1);
        write_store_file(&dirs, "f-legacy", None, 50, 1);

        let unlimited = BatchLimits::default();
        assert!(store_quota_refusal(&dirs, &unlimited, Some("alpha"), 1_000_000).is_none());

        let per_key = BatchLimits {
            max_store_bytes_per_key: 150,
            ..BatchLimits::default()
        };
        assert!(
            store_quota_refusal(&dirs, &per_key, Some("alpha"), 50).is_none(),
            "150 == cap, at-cap is allowed"
        );
        let refused = store_quota_refusal(&dirs, &per_key, Some("alpha"), 51).unwrap();
        assert_eq!(refused.0, StatusCode::PAYLOAD_TOO_LARGE);
        assert!(refused.1.contains("per-key"), "{}", refused.1);
        // Another key is unaffected by alpha's usage.
        assert!(store_quota_refusal(&dirs, &per_key, Some("beta"), 150).is_none());
        // Open-gateway uploads skip the per-key leg entirely.
        assert!(store_quota_refusal(&dirs, &per_key, None, 1_000).is_none());

        let total = BatchLimits {
            max_store_bytes_total: 150,
            ..BatchLimits::default()
        };
        let refused = store_quota_refusal(&dirs, &total, None, 1).unwrap();
        assert_eq!(refused.0, StatusCode::PAYLOAD_TOO_LARGE);
        assert!(refused.1.contains("global"), "{}", refused.1);

        let _ = std::fs::remove_dir_all(files_dir(&dirs).parent().unwrap());
    }

    /// Active-batch pins: only NON-terminal batches of the SAME owner
    /// count, across both dialects.
    #[test]
    fn unit__active_batch_count__owner_scoped_non_terminal() {
        let dirs = scratch_dirs("active");
        write_openai_batch(&dirs, "b-run", "in_progress", "f", None, "alpha", 1, None);
        write_openai_batch(&dirs, "b-done", "completed", "f", None, "alpha", 1, Some(2));
        // Anthropic dialect: ended = terminal, in_progress = active.
        for (id, status, owner) in [
            ("msg_run", "in_progress", "alpha"),
            ("msg_done", "ended", "beta"),
            ("msg_beta", "in_progress", "beta"),
        ] {
            let meta = json!({
                "dialect": "anthropic", "id": id, "internal_status": status,
                "created_at_secs": 1_u64, "ended_at_secs": Value::Null, "_owner": owner,
            });
            std::fs::write(
                batch_meta(&dirs, id),
                serde_json::to_vec_pretty(&meta).unwrap(),
            )
            .unwrap();
        }
        assert_eq!(active_batch_count(&dirs, "alpha"), 2);
        assert_eq!(active_batch_count(&dirs, "beta"), 1);
        assert_eq!(active_batch_count(&dirs, "nobody"), 0);

        let _ = std::fs::remove_dir_all(files_dir(&dirs).parent().unwrap());
    }
}
