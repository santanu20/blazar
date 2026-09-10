//! E4 audit log: one JSON line per GENERATION request (chat lanes only)
//! appended to `<data>/log/audit.jsonl` when `audit_log = true`.
//!
//! Compliance-shaped (who/what/how-long), not a prompt dumper: the line
//! records identity and outcome — key name, model, status, latency,
//! priority, queue depth — never request or response content. Writing
//! is fully off the request path (P14.4): the middleware `try_send`s
//! into a bounded channel; a dedicated task owns the file. A full
//! channel DROPS the line and counts it (an audit trail must never
//! stall generations), surfacing via `/metrics` as
//! `pallama_audit_dropped_total`.

use std::path::Path;
use std::path::PathBuf;

use serde::Serialize;

/// Generation lanes audited. Management, health, key and metric routes
/// are deliberately excluded — they carry no tokens.
pub const AUDITED_PATHS: [&str; 6] = [
    "/api/chat",
    "/api/generate",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/messages",
    "/v1/responses",
];

/// Bodies above this size skip the `model` extraction buffer (the line
/// is still written, with `model: null`). Aligned with the gateway's
/// 50 MiB `DefaultBodyLimit` — audit must never 413 a body the gateway
/// itself would accept.
pub const AUDIT_BODY_SNIFF_LIMIT: usize = 50 * 1024 * 1024;

/// Rotate the file when it passes this size; keep the newest tail.
const AUDIT_ROTATE_BYTES: u64 = 16 * 1024 * 1024;
/// Lines retained across a rotation.
const AUDIT_ROTATE_KEEP: usize = 2000;
/// Pending-lines channel bound; beyond this lines drop (counted).
pub const AUDIT_CHANNEL_CAP: usize = 4096;

/// One audit record. Field order is the wire order (serde keeps struct
/// order) — stable for log parsers.
#[derive(Debug, Clone, Serialize)]
pub struct AuditLine {
    /// Unix seconds at request START.
    pub ts: u64,
    pub trace: String,
    /// Authenticated API key name; `None` on an open gateway.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    pub method: String,
    pub path: String,
    /// Model field from the request body, when extractable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub status: u16,
    pub ms: u128,
    pub priority: String,
    pub queue_depth: usize,
}

/// Extract the `model` string field from a JSON body (best effort —
/// non-JSON or missing field returns None). Never fails, never reads
/// beyond the sniffed bytes.
#[must_use]
pub fn extract_model(body: &[u8]) -> Option<String> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    v.get("model")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

/// Whether this request should buffer its body for model extraction.
/// Only POSTs on audited lanes, only when the size is sane.
#[must_use]
pub fn should_sniff(method: &axum::http::Method, path: &str, content_len: Option<u64>) -> bool {
    method == axum::http::Method::POST
        && AUDITED_PATHS.contains(&path)
        && content_len.is_none_or(|n| n <= AUDIT_BODY_SNIFF_LIMIT as u64)
}

/// Background writer: owns the file, drains the channel, rotates at the
/// size cap keeping the newest `AUDIT_ROTATE_KEEP` lines. IO failures
/// are logged and retried on the next line — never fatal to the daemon.
pub async fn writer_task(
    mut rx: tokio::sync::mpsc::Receiver<AuditLine>,
    dir: PathBuf,
    dropped: std::sync::Arc<std::sync::atomic::AtomicU64>,
) {
    let path = dir.join("audit.jsonl");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::warn!(target: "pallama::audit", "audit dir {}: {e}", dir.display());
        // Drain-and-count so senders never block.
        while rx.recv().await.is_some() {
            dropped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        return;
    }
    let mut written = std::fs::metadata(&path).map_or(0, |m| m.len());
    while let Some(line) = rx.recv().await {
        let bytes = match serde_json::to_vec(&line) {
            Ok(mut b) => {
                b.push(b'\n');
                b
            }
            Err(_) => continue,
        };
        if written >= AUDIT_ROTATE_BYTES {
            if let Err(e) = rotate(&path) {
                tracing::warn!(target: "pallama::audit", "audit rotate: {e}");
            } else {
                tracing::info!(
                    target: "pallama::audit",
                    "audit log rotated at {AUDIT_ROTATE_BYTES} bytes (kept {AUDIT_ROTATE_KEEP} lines)"
                );
            }
            // F79: re-stat instead of blindly zeroing — on a failed
            // rotate the file is still the old size, and resetting the
            // counter here would let it grow unbounded with a rotate
            // attempt on every line forever.
            written = std::fs::metadata(&path).map_or(0, |m| m.len());
        }
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut f| {
                use std::io::Write;
                f.write_all(&bytes)?;
                f.flush()
            }) {
            Ok(()) => written += bytes.len() as u64,
            Err(e) => {
                dropped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                tracing::warn!(target: "pallama::audit", "audit append: {e}");
            }
        }
    }
}

/// Keep the newest `AUDIT_ROTATE_KEEP` lines (same file, rewrite).
fn rotate(path: &Path) -> std::io::Result<()> {
    let raw = std::fs::read_to_string(path)?;
    let kept: Vec<&str> = raw
        .lines()
        .filter(|l| !l.trim().is_empty())
        .rev()
        .take(AUDIT_ROTATE_KEEP)
        .collect();
    let mut out = String::new();
    for line in kept.iter().rev() {
        out.push_str(line);
        out.push('\n');
    }
    std::fs::write(path, out)
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn unit__audit__extract_model_from_json_bodies() {
        assert_eq!(
            extract_model(br#"{"model":"qwen3-8b","messages":[]}"#),
            Some("qwen3-8b".into())
        );
        assert_eq!(extract_model(br#"{"messages":[]}"#), None);
        assert_eq!(extract_model(b"not json"), None);
        // Non-string model field: null, not a crash.
        assert_eq!(extract_model(br#"{"model":42}"#), None);
    }

    #[test]
    fn unit__audit__sniff_gate_by_method_path_and_size() {
        let post = axum::http::Method::POST;
        let get = axum::http::Method::GET;
        assert!(should_sniff(&post, "/api/chat", Some(128)));
        assert!(!should_sniff(&get, "/api/chat", Some(128)), "GETs skip");
        assert!(!should_sniff(&post, "/api/tags", Some(128)));
        assert!(!should_sniff(
            &post,
            "/api/chat",
            Some(AUDIT_BODY_SNIFF_LIMIT as u64 + 1)
        ));
        assert!(should_sniff(&post, "/v1/messages", None)); // chunked: unknown
    }

    #[test]
    fn unit__audit__rotate_keeps_newest_tail_only() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("audit.jsonl");
        let total = AUDIT_ROTATE_KEEP + 500;
        let mut seeded = String::new();
        for i in 0..total {
            seeded.push_str(&i.to_string());
            seeded.push('\n');
        }
        std::fs::write(&path, seeded).unwrap();
        rotate(&path).expect("rotate");
        let after = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = after.lines().collect();
        assert_eq!(lines.len(), AUDIT_ROTATE_KEEP, "tail kept, head dropped");
        // The NEWEST AUDIT_ROTATE_KEEP survive: first kept = total - KEEP.
        assert_eq!(
            lines.first(),
            Some(&format!("{}", total - AUDIT_ROTATE_KEEP).as_str())
        );
        assert_eq!(lines.last(), Some(&(total - 1).to_string().as_str()));
    }

    #[tokio::test]
    async fn integration__audit__writer_writes_and_rotates_tail() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (tx, rx) = tokio::sync::mpsc::channel(AUDIT_CHANNEL_CAP);
        let dropped = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let task = tokio::spawn(writer_task(
            rx,
            dir.path().to_path_buf(),
            std::sync::Arc::clone(&dropped),
        ));
        for i in 0u64..3 {
            let _ = tx
                .send(AuditLine {
                    ts: 1_700_000_000 + i,
                    trace: format!("plm-{i}"),
                    key: Some("ci".into()),
                    method: "POST".into(),
                    path: "/api/chat".into(),
                    model: Some("m".into()),
                    status: 200,
                    ms: 42,
                    priority: "normal".into(),
                    queue_depth: 0,
                })
                .await;
        }
        drop(tx);
        task.await.expect("writer exits with channel");
        let raw = std::fs::read_to_string(dir.path().join("audit.jsonl")).unwrap();
        let lines: Vec<&str> = raw.lines().collect();
        assert_eq!(lines.len(), 3, "{raw}");
        assert!(lines[0].contains("\"trace\":\"plm-0\""));
        assert!(lines[0].contains("\"key\":\"ci\""));
        assert!(lines[2].contains("\"status\":200"));
        assert_eq!(dropped.load(std::sync::atomic::Ordering::Relaxed), 0);
    }
}
