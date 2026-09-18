//! OpenAI-compatible Batch API (D4): JSONL in, background out.
//!
//! `/v1/files` stores an uploaded JSONL request file under
//! `data/pallama/batch/files/`; `/v1/batches` validates it and spawns a
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
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use pallama_core::PallamaDirs;
use serde_json::{json, Value};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Upload size cap: kept just under the gateway's 50 MiB default body
/// limit so the route-level check fires before axum's 413.
const MAX_UPLOAD_BYTES: usize = 48 * 1024 * 1024;
/// Only the chat-completions endpoint is batched (honest scope; embeddings
/// lane is a different surface).
const SUPPORTED_ENDPOINT: &str = "/v1/chat/completions";
/// Ids are short random hex so they sort/echo like upstream (`file-…`,
/// `batch-…`).
fn short_id(prefix: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos()),
    )
    // u128 nanos fit u64 until year 2554; the xor only needs entropy.
    .unwrap_or(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{:012x}", nanos ^ n.rotate_left(17))
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn root(dirs: &PallamaDirs) -> PathBuf {
    dirs.data_dir.join("batch")
}
fn files_dir(dirs: &PallamaDirs) -> PathBuf {
    root(dirs).join("files")
}
fn batches_dir(dirs: &PallamaDirs) -> PathBuf {
    root(dirs).join("batches")
}
fn batch_meta(dirs: &PallamaDirs, id: &str) -> PathBuf {
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

/// `POST /v1/files` — multipart upload with a `file` part (and optional
/// `purpose` form field, accepted and recorded). Returns an OpenAI-style
/// file object.
pub async fn upload_file(
    State(state): State<std::sync::Arc<AppState>>,
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
    if let Some(p) = &purpose {
        if p != "batch" {
            return err(StatusCode::BAD_REQUEST, "only purpose=batch is supported");
        }
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
    });
    if let Err(e) = write_json(&dir.join(format!("{id}.meta.json")), &obj) {
        tracing::error!(target: "pallama::batch", file = %id, error = %e, "file meta write failed");
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("write file meta: {e}"),
        );
    }
    (StatusCode::OK, axum::Json(obj)).into_response()
}

fn file_meta(dirs: &PallamaDirs, id: &str) -> Option<Value> {
    read_json(&files_dir(dirs).join(format!("{id}.meta.json")))
}

/// `GET /v1/files/{id}` — file metadata.
pub async fn get_file(
    State(state): State<std::sync::Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return err(StatusCode::BAD_REQUEST, "invalid file id");
    }
    match file_meta(&state.dirs, &id) {
        Some(v) => (StatusCode::OK, axum::Json(v)).into_response(),
        None => err(StatusCode::NOT_FOUND, "unknown file id"),
    }
}

/// `GET /v1/files/{id}/content` — raw stored JSONL.
pub async fn get_file_content(
    State(state): State<std::sync::Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return err(StatusCode::BAD_REQUEST, "invalid file id");
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
    let input_path = files_dir(&state.dirs).join(format!("{input_file_id}.jsonl"));
    let Ok(lines) = std::fs::read_to_string(&input_path) else {
        return err(StatusCode::NOT_FOUND, "input file content missing");
    };
    let total = lines.lines().filter(|l| !l.trim().is_empty()).count();
    if total == 0 {
        return err(StatusCode::BAD_REQUEST, "input file has no request lines");
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
        "input_file": meta,
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
    };
    tokio::spawn(async move { run_batch(dirs, job, lines).await });

    (StatusCode::OK, axum::Json(batch)).into_response()
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
}

async fn run_batch(dirs: PallamaDirs, job: BatchJob, input: String) {
    let BatchJob {
        id,
        output_id,
        host,
        port,
        auth_headers,
    } = job;
    let job_ref = BatchJob {
        id: id.clone(),
        output_id: output_id.clone(),
        host: host.clone(),
        port,
        auth_headers: Vec::new(),
    };
    // Output lands in files/ as {output_id}.jsonl — same lookup path the
    // GET /v1/files/{id}/content route serves uploads from.
    let out_path = files_dir(&dirs).join(format!("{output_id}.jsonl"));
    if let Some(d) = out_path.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_mins(10))
        .build()
        .unwrap_or_default();
    let mut completed = 0u64;
    let mut failed = 0u64;
    // F74: stream rows straight to disk instead of accumulating the whole
    // output in memory (a 48 MiB input can inflate far beyond that with
    // response bodies; a crash also loses everything buffered so far).
    let mut writer = match std::fs::File::create(&out_path) {
        Ok(f) => std::io::BufWriter::new(f),
        Err(e) => {
            tracing::error!(target: "pallama::batch", batch = %id, error = %e, "batch output file create failed");
            fail_batch(&dirs, &job_ref, &format!("output file create: {e}"));
            return;
        }
    };
    let base = format!("http://{host}:{port}");

    for line in input.lines().filter(|l| !l.trim().is_empty()) {
        // Cancel is signalled through the meta file; check between items.
        if let Some(m) = read_json(&batch_meta(&dirs, &id)) {
            if m["status"] == "cancelling" {
                finish_batch(
                    &dirs,
                    &job_ref,
                    "cancelled",
                    (completed, failed),
                    &mut writer,
                );
                return;
            }
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
        if !write_row(&mut writer, &row, &dirs, &job_ref, completed, failed) {
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
/// (output write failed — F76: disk errors must not masquerade as success).
fn write_row(
    writer: &mut std::io::BufWriter<std::fs::File>,
    row: &str,
    dirs: &PallamaDirs,
    job: &BatchJob,
    completed: u64,
    failed: u64,
) -> bool {
    if let Err(e) = writeln!(writer, "{row}") {
        tracing::error!(target: "pallama::batch", batch = %job.id, error = %e, "batch output write failed");
        fail_batch(dirs, job, &format!("output write: {e}"));
        return false;
    }
    update_progress(dirs, &job.id, completed, failed);
    true
}

/// Mark a batch failed with the reason (output-file or flush errors).
fn fail_batch(dirs: &PallamaDirs, job: &BatchJob, why: &str) {
    let path = batch_meta(dirs, &job.id);
    if let Some(mut m) = read_json(&path) {
        m["status"] = json!("failed");
        m["error"] = json!({"message": why});
        m["request_counts"]["failed"] =
            json!(m["request_counts"]["failed"].as_u64().unwrap_or(0) + 1);
        m["finalized_at"] = json!(now_secs());
        if let Err(e) = write_json(&path, &m) {
            tracing::error!(target: "pallama::batch", batch = %job.id, error = %e, "failed-batch meta write failed");
        }
    }
}
fn update_progress(dirs: &PallamaDirs, id: &str, completed: u64, failed: u64) {
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
            tracing::warn!(target: "pallama::batch", batch = %id, error = %e, "progress meta write failed");
        }
    }
}

fn finish_batch(
    dirs: &PallamaDirs,
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
        tracing::error!(target: "pallama::batch", batch = %id, error = %e, "batch output flush failed");
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
        tracing::warn!(target: "pallama::batch", batch = %id, error = %e, "output file meta write failed");
    }
    write_final_meta(dirs, id, output_id, status, completed, failed);
}

fn write_final_meta(
    dirs: &PallamaDirs,
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
            tracing::error!(target: "pallama::batch", batch = %id, error = %e, "final batch meta write failed");
        }
    }
    tracing::info!(target: "pallama::batch", batch = %id, status, completed, failed, "batch finished");
}

/// `GET /v1/batches/{id}` — live batch object.
pub async fn get_batch(
    State(state): State<std::sync::Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return err(StatusCode::BAD_REQUEST, "invalid batch id");
    }
    match read_json(&batch_meta(&state.dirs, &id)) {
        Some(v) => (StatusCode::OK, axum::Json(v)).into_response(),
        None => err(StatusCode::NOT_FOUND, "unknown batch id"),
    }
}

/// `GET /v1/batches` — list all batches (newest first).
pub async fn list_batches(State(state): State<std::sync::Arc<AppState>>) -> Response {
    let mut rows: Vec<Value> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(batches_dir(&state.dirs)) {
        for e in entries.flatten() {
            let name = e.file_name();
            if let Some(name) = name.to_str() {
                if std::path::Path::new(name)
                    .extension()
                    .is_some_and(|e| e == "json")
                {
                    if let Some(v) = read_json(&e.path()) {
                        rows.push(v);
                    }
                }
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
        Some(mut m) => {
            let status = m["status"].as_str().unwrap_or("").to_string();
            if status == "completed" || status == "cancelled" {
                return err(StatusCode::BAD_REQUEST, "batch already finished");
            }
            m["status"] = json!("cancelling");
            if let Err(e) = write_json(&path, &m) {
                return err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("persist cancel: {e}"),
                );
            }
            (StatusCode::OK, axum::Json(m)).into_response()
        }
        None => err(StatusCode::NOT_FOUND, "unknown batch id"),
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
    }

    #[test]
    fn unit__batch_paths__nested_under_data() {
        let dirs = PallamaDirs {
            config_dir: PathBuf::from("/tmp/c"),
            data_dir: PathBuf::from("/tmp/d"),
        };
        assert_eq!(files_dir(&dirs), PathBuf::from("/tmp/d/batch/files"));
        assert_eq!(
            batch_meta(&dirs, "batch-1"),
            PathBuf::from("/tmp/d/batch/batches/batch-1.json")
        );
    }
}
